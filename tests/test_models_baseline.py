"""Tests for arm 1's orchestration (#9): `run_fold` / `run_all_folds` end to end.

A synthetic cohort, not the real 6,811-Subject one — `tests/README.md` asks
for tests that do not need the live instance, and the real cohort is only
available locally via the (gitignored) extract artefacts. Priority is the
class of bug that would corrupt the comparison silently: a leaked val/test
statistic in the feature contract, a decoder scored on the wrong Subject set,
or a sanity check that double-counts a Subject across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gl_lifesphere.evaluation.splits import assign_folds, fold_split
from gl_lifesphere.features.raw import SAMPLE_PROPORTION_COLUMNS
from gl_lifesphere.models.baseline.sanity_check import (
    SANITY_BANDS,
    pooled_out_of_sample_predictions,
    run_sanity_check,
)
from gl_lifesphere.models.baseline.train import BaselineArmConfig, run_all_folds, run_fold
from gl_lifesphere.survival.targets import SurvivalTarget

N_STUDIES = 6
N_PER_STUDY = 40


def _synthetic_raw_and_targets(seed: int = 0) -> tuple[pd.DataFrame, SurvivalTarget, pd.DataFrame]:
    """A `raw_frame` (features.contract's input shape) plus its `SurvivalTarget`
    and fold assignment, for `N_STUDIES` Studies of `N_PER_STUDY` Subjects each
    — enough for 5-fold Study-stratified CV to reach every fold in every Study."""
    rng = np.random.default_rng(seed)
    n = N_STUDIES * N_PER_STUDY
    subject_ids = [f"SUBJ{i:04d}" for i in range(n)]
    study = np.array([f"STUDY-{i % N_STUDIES}" for i in range(n)])

    stage = rng.integers(1, 5, size=n).astype(float)
    age = rng.normal(60, 10, size=n)
    sex = rng.choice(["male", "female"], size=n)
    # Race carries real missingness (~10%, matching #4 §3's cohort-wide rate)
    # rather than being fully observed — a fully-observed synthetic fixture
    # left `race`'s `__MISSING__` dummy constant-zero in every training fold,
    # which turned out to reproduce a real numerical trap in
    # `features.contract._CategoricalEncoder`: it always emits `__RARE__` and
    # `__MISSING__` columns even when a fold's training data has zero rows in
    # either bucket, and two constant-zero columns are exactly collinear,
    # which singularises the Cox fit. The real 6,811-Subject cohort always has
    # both buckets populated in every fold, so this is a fixture-realism fix
    # rather than a change to arm 1's own code.
    race = rng.choice(
        ["white", "asian", "black or african american", np.nan], size=n, p=[0.55, 0.2, 0.15, 0.1]
    )
    # Mostly 3 common subtypes, plus a long thin tail of near-singleton values
    # so the training-fold `__RARE__` bucket (n < 20, #4 §3) is populated too.
    common_subtype = rng.choice(["SUBTYPE-A", "SUBTYPE-B", "SUBTYPE-C"], size=n, p=[0.45, 0.35, 0.2])
    rare_subtype = np.array([f"SUBTYPE-RARE-{i}" for i in range(n)])
    subtype = np.where(rng.random(n) < 0.1, rare_subtype, common_subtype)
    condition = rng.choice([f"COND-{i}" for i in range(3)], size=n)

    # A planted effect on stage/age so the fitted decoder actually discriminates.
    linear = 0.5 * stage + 0.03 * age
    latent = rng.exponential(scale=np.exp(-linear / 10.0))
    censor = rng.exponential(scale=1.5)
    # Scaled so every fold's test-set follow-up comfortably exceeds
    # `metrics.DEFAULT_HORIZONS_DAYS`'s 1095-day horizon — `sksurv` requires
    # every horizon strictly inside the *test* fold's own follow-up range.
    duration = np.ceil(np.minimum(latent, censor) * 3000.0) + 1.0
    event = latent <= censor

    raw_frame = pd.DataFrame(
        {
            "subjectId": subject_ids,
            "studyId": study,
            "sexAtBirth": sex,
            "race": race,
            "ageAtIndexYears": age,
            "stageOrdinal": stage,
            "ageAtDiagnosisYears": age,
            "conditionSubtype": subtype,
            "conditionName": condition,
        }
    )
    for i, column in enumerate(SAMPLE_PROPORTION_COLUMNS):
        raw_frame[column] = rng.random(n) if i == 0 else 0.0

    labels = pd.DataFrame(
        {"subjectId": subject_ids, "studyId": study, "durationDays": duration, "event": event}
    )
    targets = SurvivalTarget(
        subject_id=labels["subjectId"].to_numpy(),
        study=labels["studyId"].to_numpy(),
        time=labels["durationDays"].to_numpy(dtype="float64"),
        event=labels["event"].to_numpy(dtype=bool),
    )
    assignment = assign_folds(labels)
    return raw_frame, targets, assignment


@pytest.fixture(scope="module")
def cohort() -> tuple[pd.DataFrame, SurvivalTarget, pd.DataFrame]:
    return _synthetic_raw_and_targets()


class TestRunFold:
    def test_test_predictions_cover_exactly_the_outer_test_fold(
        self, cohort: tuple[pd.DataFrame, SurvivalTarget, pd.DataFrame]
    ) -> None:
        raw_frame, targets, assignment = cohort
        split = fold_split(assignment, 0)
        result = run_fold(
            0, raw=raw_frame, targets=targets, split=split, config=BaselineArmConfig()
        )
        assert set(result.test_predictions.index) == split.test

    def test_covariates_exclude_condition_and_sample_columns(
        self, cohort: tuple[pd.DataFrame, SurvivalTarget, pd.DataFrame]
    ) -> None:
        raw_frame, targets, assignment = cohort
        split = fold_split(assignment, 0)
        result = run_fold(
            0, raw=raw_frame, targets=targets, split=split, config=BaselineArmConfig()
        )
        assert not any(c.startswith("condition_") for c in result.covariates)
        assert not any(c in SAMPLE_PROPORTION_COLUMNS for c in result.covariates)

    def test_metrics_are_populated(
        self, cohort: tuple[pd.DataFrame, SurvivalTarget, pd.DataFrame]
    ) -> None:
        raw_frame, targets, assignment = cohort
        split = fold_split(assignment, 0)
        result = run_fold(
            0, raw=raw_frame, targets=targets, split=split, config=BaselineArmConfig()
        )
        payload = result.fold_metrics.to_dict()
        pooled_c = payload["pooled_harrell_c"]
        within_study_c = payload["within_study_harrell_c"]
        assert isinstance(pooled_c, float) and 0.0 <= pooled_c <= 1.0
        assert isinstance(within_study_c, float) and 0.0 <= within_study_c <= 1.0


class TestRunAllFolds:
    def test_produces_five_disjoint_test_sets_covering_the_whole_cohort(
        self, cohort: tuple[pd.DataFrame, SurvivalTarget, pd.DataFrame]
    ) -> None:
        raw_frame, targets, assignment = cohort

        # `run_all_folds` reads folds via `evaluation.splits.load_folds`, which
        # this synthetic fixture does not persist to disk, so its 5-fold loop
        # is reproduced directly against the in-memory assignment instead.
        results = [
            run_fold(i, raw=raw_frame, targets=targets, split=fold_split(assignment, i), config=BaselineArmConfig())
            for i in range(5)
        ]
        test_sets = [set(r.test_predictions.index) for r in results]
        union = set().union(*test_sets)
        assert union == set(raw_frame["subjectId"])
        for a in range(5):
            for b in range(a + 1, 5):
                assert test_sets[a] & test_sets[b] == set()


class TestSanityCheck:
    def test_band_is_checked_against_unos_c_not_harrells_c(self) -> None:
        """The fix this pins: an earlier version compared Harrell's C to these
        bands directly and got a false alarm on the real BRCA result (0.7304,
        above its 0.60-0.67 band) that vanished under Uno's C (0.6375, inside
        it) -- Harrell reads optimistic under censoring (#2 §7), and the bands
        are calibrated against Herrmann et al.'s own Uno's C (#2 §5). Reuses
        `tests/test_survival_metrics.py`'s stratified-cohort shape (heavy,
        integer-day censoring) collapsed onto one Study, since that is where
        the Harrell/Uno gap is known to be measurable and non-flaky."""
        rng = np.random.default_rng(0)
        beta = np.array([0.8, -0.5, 0.2])
        n, n_strata = 400, 5
        x = rng.normal(size=(n, len(beta)))
        stratum = rng.integers(0, n_strata, size=n)
        latent = rng.exponential(scale=np.exp(-(x @ beta)) * (1.0 + stratum))
        censor = rng.exponential(scale=2.0 * (1.0 + stratum), size=n)
        time = np.ceil(np.minimum(latent, censor) * 60.0)
        event = latent <= censor
        risk = x @ beta

        frame = pd.DataFrame(
            {"studyId": ["TCGA-BRCA"] * n, "risk": risk, "durationDays": time, "event": event},
            index=pd.Index([f"S{i}" for i in range(n)], name="subjectId"),
        )
        result = next(r for r in run_sanity_check(frame) if r.study == "TCGA-BRCA")
        assert result.harrell_c is not None and result.observed_c is not None
        assert result.harrell_c > result.observed_c
        assert result.harrell_c == pytest.approx(result.to_dict()["harrell_c"])
        assert result.observed_c == pytest.approx(result.to_dict()["observed_c"])

    def test_pooling_raises_on_a_duplicated_subject(self) -> None:
        one = pd.DataFrame(
            {"studyId": ["TCGA-BRCA"], "risk": [0.1], "durationDays": [10.0], "event": [True]},
            index=pd.Index(["X1"], name="subjectId"),
        )
        with pytest.raises(ValueError, match="more than one"):
            pooled_out_of_sample_predictions([one, one])

    def test_reports_one_result_per_band_and_flags_out_of_band(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        # A near-chance risk score against BRCA -> expected out of its 0.60-0.67 band.
        rows = []
        for study in SANITY_BANDS:
            risk = rng.normal(size=n)
            time = rng.exponential(size=n)  # independent of risk: chance-level
            event = rng.random(n) < 0.5
            rows.append(
                pd.DataFrame(
                    {"studyId": [study] * n, "risk": risk, "durationDays": time, "event": event},
                    index=pd.Index([f"{study}-{i}" for i in range(n)], name="subjectId"),
                )
            )
        pooled = pooled_out_of_sample_predictions(rows)
        results = run_sanity_check(pooled)
        assert {r.study for r in results} == set(SANITY_BANDS)
        # A chance-level score (~0.5) should not land inside any of these bands.
        assert all(r.in_band is False for r in results)

    def test_a_study_absent_from_predictions_is_reported_as_unscoreable(self) -> None:
        empty = pd.DataFrame(
            {"studyId": pd.Series(dtype="object"), "risk": pd.Series(dtype="float64"),
             "durationDays": pd.Series(dtype="float64"), "event": pd.Series(dtype="bool")}
        )
        results = run_sanity_check(empty)
        assert all(r.observed_c is None and r.in_band is None for r in results)
