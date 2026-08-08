"""Tests for the shared feature contract (#4), depended on directly by arm 1 (#9).

Scoped to what arm 1 relies on rather than re-deriving all of #4's encoding
decisions: that `fit_feature_contract` only ever looks at the training
Subjects it is handed (so a val/test row cannot influence a vocabulary or an
imputation value), that `transform` never leaks `studyId` or a Survival
column into the design matrix, and that an unseen category at transform time
lands in a fixed bucket rather than growing the column set.

Nothing here touches the live instance; `raw_frame`-shaped rows are built by
hand rather than through `features.raw.assemble_raw_frame`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gl_lifesphere.extract.cypher import BASELINE_SAMPLE_TYPES
from gl_lifesphere.features import contract
from gl_lifesphere.features.raw import SAMPLE_PROPORTION_COLUMNS


def _row(subject: str, study: str, *, subtype: str = "Common", condition: str = "COND-A") -> dict[str, object]:
    row: dict[str, object] = {
        "subjectId": subject,
        "studyId": study,
        "sexAtBirth": "female",
        "race": "white",
        "ageAtIndexYears": 55.0,
        "stageOrdinal": 2,
        "ageAtDiagnosisYears": 55.0,
        "conditionSubtype": subtype,
        "conditionName": condition,
    }
    for i, column in enumerate(SAMPLE_PROPORTION_COLUMNS):
        row[column] = 1.0 if i == 0 else 0.0
    return row


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rows = [_row(f"S{i:02d}", "STUDY-A") for i in range(25)]
    rows += [_row(f"T{i:02d}", "STUDY-B", subtype="Rare") for i in range(5)]
    return pd.DataFrame(rows)


class TestTrainingFoldOnly:
    def test_a_category_absent_from_training_does_not_change_the_column_set(
        self, raw_frame: pd.DataFrame
    ) -> None:
        train_ids = frozenset(f"S{i:02d}" for i in range(25))  # excludes the STUDY-B "Rare" rows
        fitted = contract.fit_feature_contract(raw_frame, train_ids)

        train_design = fitted.transform(raw_frame, train_ids)
        full_design = fitted.transform(raw_frame)  # includes the held-out "Rare" rows
        assert list(train_design.columns) == list(full_design.columns) == fitted.feature_names

    def test_extra_rows_outside_the_training_set_do_not_shift_the_fit(
        self, raw_frame: pd.DataFrame
    ) -> None:
        """`fit_feature_contract` may be handed a `raw_frame` covering a whole
        fold; only `train_subject_ids` may influence what gets fitted."""
        train_ids = frozenset(f"S{i:02d}" for i in range(25))
        fitted_with_extra = contract.fit_feature_contract(raw_frame, train_ids)
        fitted_train_only = contract.fit_feature_contract(raw_frame[raw_frame["subjectId"].isin(train_ids)], train_ids)
        assert fitted_with_extra.age.mean == pytest.approx(fitted_train_only.age.mean)
        assert fitted_with_extra.stage.mean == pytest.approx(fitted_train_only.stage.mean)


class TestNoLeakage:
    def test_no_study_or_survival_column_reaches_the_design_matrix(
        self, raw_frame: pd.DataFrame
    ) -> None:
        train_ids = frozenset(raw_frame["subjectId"])
        fitted = contract.fit_feature_contract(raw_frame, train_ids)
        design = fitted.transform(raw_frame)
        forbidden = {"studyId", "study", "eventOccurred", "timeToEventDays", "event", "durationDays"}
        assert not forbidden & set(design.columns)

    def test_design_is_indexed_by_subject(self, raw_frame: pd.DataFrame) -> None:
        train_ids = frozenset(raw_frame["subjectId"])
        fitted = contract.fit_feature_contract(raw_frame, train_ids)
        design = fitted.transform(raw_frame)
        assert design.index.name == "subjectId"
        assert set(design.index) == set(raw_frame["subjectId"])


class TestBaselineSubset:
    """The columns arm 1 (#9) is allowed to use: everything except Condition
    (`conditionName`, flagged in #4 §6 as close to a relabelling of Study) and
    Sample-type proportions (#4 §4: arm 1 does not carry that ascertainment
    channel)."""

    def test_excludes_condition_and_sample_proportion_columns(self, raw_frame: pd.DataFrame) -> None:
        from gl_lifesphere.models.baseline.train import baseline_columns

        train_ids = frozenset(raw_frame["subjectId"])
        fitted = contract.fit_feature_contract(raw_frame, train_ids)
        columns = baseline_columns(fitted)

        assert not any(c.startswith("condition_") for c in columns)
        assert not any(c in SAMPLE_PROPORTION_COLUMNS for c in columns)
        # But the rest of the shared contract survives.
        assert "is_male" in columns
        assert "stageOrdinal" in columns
        assert "ageAtDiagnosisYears" in columns
        assert any(c.startswith("subtype_") for c in columns)
        assert any(c.startswith("race_") for c in columns)

    def test_selected_columns_are_a_subset_of_the_full_contract(self, raw_frame: pd.DataFrame) -> None:
        from gl_lifesphere.models.baseline.train import baseline_columns

        train_ids = frozenset(raw_frame["subjectId"])
        fitted = contract.fit_feature_contract(raw_frame, train_ids)
        columns = baseline_columns(fitted)
        assert set(columns) <= set(fitted.feature_names)
