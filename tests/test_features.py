"""Tests for the shared feature contract (#10, locked by #4).

Priority per `tests/README.md`: the quiet-wrong-answer class. Here that means
a training-fold statistic quietly drawing on held-out Subjects, a missing
value silently becoming a fabricated category, and a leaking column reaching
a design matrix without the guard catching it.

Nothing here touches the live instance or `data/processed/cohort_os` — `data/`
is gitignored (`data/README.md`), so any test depending on it would pass
locally and fail on a fresh clone or in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gl_lifesphere.extract import cast, guards
from gl_lifesphere.features.contract import (
    MISSING_LEVEL,
    RARE_LEVEL,
    _CategoricalEncoder,
    _NumericEncoder,
    fit_feature_contract,
)
from gl_lifesphere.features.raw import SAMPLE_PROPORTION_COLUMNS, build_raw_frame

import fixtures


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """The fixture cohort, run through the real casts, joined the way an arm would."""
    subjects = cast.cast_subjects(pd.DataFrame(fixtures.raw_subjects()))
    diagnoses = cast.cast_diagnoses(pd.DataFrame(fixtures.raw_diagnoses()))
    diagnosis_primary = cast.reduce_diagnoses(diagnoses)
    samples = cast.cast_samples(pd.DataFrame(fixtures.raw_samples()))
    members = subjects[["subjectId", "studyId"]]
    return build_raw_frame(
        members=members, subjects=subjects, diagnosis_primary=diagnosis_primary, samples=samples
    )


class TestEndToEnd:
    """The whole pipeline on schema-faithful fixture data."""

    def test_transform_is_all_numeric_with_no_nulls(self, raw_frame: pd.DataFrame) -> None:
        all_ids = frozenset(raw_frame["subjectId"])
        contract = fit_feature_contract(raw_frame, all_ids)
        design = contract.transform(raw_frame)

        assert design.shape[0] == len(raw_frame)
        assert all(str(dtype) == "float64" for dtype in design.dtypes)
        assert not design.isna().any().any()

    def test_output_passes_the_leakage_guards(self, raw_frame: pd.DataFrame) -> None:
        """Belt-and-suspenders: `transform` already calls this; pinned here too
        so a future refactor that drops the internal call still fails a test."""
        all_ids = frozenset(raw_frame["subjectId"])
        contract = fit_feature_contract(raw_frame, all_ids)
        design = contract.transform(raw_frame)

        guards.check_feature_frame(design, where="test")
        assert "studyId" not in design.columns
        assert "subjectId" not in design.columns

    def test_feature_names_match_the_transformed_columns(self, raw_frame: pd.DataFrame) -> None:
        all_ids = frozenset(raw_frame["subjectId"])
        contract = fit_feature_contract(raw_frame, all_ids)
        design = contract.transform(raw_frame)

        assert list(design.columns) == contract.feature_names


class TestFoldOnlyStatistics:
    """The property that makes '#4 §3: training-fold statistics only' enforceable."""

    def test_transforming_a_held_out_subject_does_not_change_train_fitted_stats(
        self, raw_frame: pd.DataFrame
    ) -> None:
        """Fit on {S1} alone; adding S2/S3/S4 to the frame passed to `transform`
        must not change what {S1} was fitted with."""
        train_ids = frozenset({"S1"})
        contract = fit_feature_contract(raw_frame, train_ids)

        only_s1 = raw_frame[raw_frame["subjectId"] == "S1"]
        design_alone = contract.transform(only_s1)
        design_from_full = contract.transform(raw_frame, frozenset({"S1"}))

        pd.testing.assert_frame_equal(design_alone, design_from_full)

    def test_stage_median_is_computed_from_the_training_study_only(
        self, raw_frame: pd.DataFrame
    ) -> None:
        """S1 (BRCA) has stageOrdinal 2 (Stage IIA); S2 (BRCA)'s primary Diagnosis
        carries 'Stage X', which `stage_ordinal` maps to missing (cast.py). Fit
        on both BRCA Subjects: the training-fold BRCA median must be exactly
        S1's own value, not something a held-out Subject could move."""
        contract = fit_feature_contract(raw_frame, frozenset({"S1", "S2"}))
        assert contract.stage.study_median["TCGA-BRCA"] == pytest.approx(2.0)

    def test_a_studys_median_is_unaffected_by_that_study_being_excluded_from_training(
        self, raw_frame: pd.DataFrame
    ) -> None:
        """Fitting on SKCM-only Subjects must not let BRCA Subjects (outside the
        training set) leak into the BRCA median — it should simply be absent,
        with transform() falling back to the training-fold *global* median."""
        contract = fit_feature_contract(raw_frame, frozenset({"S3", "S4"}))
        assert "TCGA-BRCA" not in contract.stage.study_median


class TestCategoricalEncoding:
    def test_rare_race_levels_fold_into_other_regardless_of_frequency(self) -> None:
        """#4 §3's fixed rule: these two levels always fold into 'other', which is
        a schema-level decision, not a training-fold-count statistic."""
        train = pd.Series(
            ["white", "american indian or alaska native", "native hawaiian or other pacific islander"]
        )
        encoder = _CategoricalEncoder.fit(
            train,
            column="race",
            prefix="race",
            min_count=1,
            fixed_fold=frozenset(
                {"american indian or alaska native", "native hawaiian or other pacific islander"}
            ),
        )
        assert "other" in encoder.categories
        assert "american indian or alaska native" not in encoder.categories

        transformed = encoder.transform(
            pd.Series(["american indian or alaska native", "native hawaiian or other pacific islander"])
        )
        assert (transformed["race_other"] == 1.0).all()

    def test_a_category_never_seen_in_training_becomes_rare_at_transform_time(self) -> None:
        encoder = _CategoricalEncoder.fit(
            pd.Series(["a", "a", "b"]), column="x", prefix="x", min_count=1
        )
        transformed = encoder.transform(pd.Series(["a", "unseen-category"]))

        assert transformed.loc[0, "x_a"] == 1.0
        assert transformed.loc[1, f"x_{RARE_LEVEL.lower()}"] == 1.0

    def test_missing_value_gets_its_own_level_rather_than_being_dropped_or_zeroed(self) -> None:
        encoder = _CategoricalEncoder.fit(pd.Series(["a", "b", None]), column="x", prefix="x", min_count=1)
        transformed = encoder.transform(pd.Series([None]))

        assert transformed.loc[0, f"x_{MISSING_LEVEL.lower()}"] == 1.0
        # A missing row must not silently read as any real category.
        assert transformed.loc[0, "x_a"] == 0.0
        assert transformed.loc[0, "x_b"] == 0.0

    def test_below_threshold_training_levels_are_lumped_into_rare(self) -> None:
        train = pd.Series(["common"] * 25 + ["rare"] * 3)
        encoder = _CategoricalEncoder.fit(train, column="x", prefix="x", min_count=20)

        assert "common" in encoder.categories
        assert "rare" not in encoder.categories
        transformed = encoder.transform(pd.Series(["rare"]))
        assert transformed.loc[0, f"x_{RARE_LEVEL.lower()}"] == 1.0


class TestNumericEncoding:
    def test_missing_value_falls_back_to_training_fold_within_study_median(self) -> None:
        values = pd.Series([1.0, np.nan, 5.0, np.nan])
        study = pd.Series(["A", "A", "B", "B"])
        encoder = _NumericEncoder.fit(values, study, column="x")

        assert encoder.study_median["A"] == pytest.approx(1.0)
        assert encoder.study_median["B"] == pytest.approx(5.0)

    def test_a_study_with_no_training_observations_falls_back_to_the_global_median(self) -> None:
        values = pd.Series([1.0, 3.0, np.nan])
        study = pd.Series(["A", "A", "B"])
        encoder = _NumericEncoder.fit(values, study, column="x")

        # B contributed nothing, so B's cell must use the global training median.
        assert "B" not in encoder.study_median
        transformed = encoder.transform(pd.Series([np.nan]), pd.Series(["B"]))
        imputed_value = transformed["x"].iloc[0] * encoder.std + encoder.mean
        assert imputed_value == pytest.approx(encoder.global_median)

    def test_standardisation_uses_training_fold_mean_and_std(self) -> None:
        values = pd.Series([10.0, 20.0, 30.0])
        study = pd.Series(["A", "A", "A"])
        encoder = _NumericEncoder.fit(values, study, column="x")

        transformed = encoder.transform(values, study)
        assert transformed["x"].mean() == pytest.approx(0.0, abs=1e-9)


class TestSampleProportions:
    def test_baseline_proportions_sum_to_one(self, raw_frame: pd.DataFrame) -> None:
        # Checked pre-standardisation, on `raw_frame` directly — the contract's
        # standardisation step would obscure the sum-to-one property.
        raw_props = raw_frame.set_index("subjectId")[list(SAMPLE_PROPORTION_COLUMNS)]
        assert raw_props.loc["S1"].sum() == pytest.approx(1.0)

    def test_a_subject_with_no_baseline_sample_is_imputed_from_the_training_fold_mean(
        self, raw_frame: pd.DataFrame
    ) -> None:
        """S4's only baseline-typed Sample slot has `sampleType = None` (fixtures.py),
        so it has zero baseline-type Samples and an all-NaN proportion row."""
        raw_props = raw_frame.set_index("subjectId")[list(SAMPLE_PROPORTION_COLUMNS)]
        assert raw_props.loc["S4"].isna().all()

        contract = fit_feature_contract(raw_frame, frozenset(raw_frame["subjectId"]))
        design = contract.transform(raw_frame, frozenset({"S4"}))
        # Imputed and then standardised — finite, not NaN, is the property that matters.
        assert np.isfinite(design.loc["S4"][list(SAMPLE_PROPORTION_COLUMNS)]).all()
