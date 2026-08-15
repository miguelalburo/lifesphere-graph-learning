"""Tests for the 5-fold Study-stratified split protocol (#7).

Priority is the leakage class of bug: a Subject reachable from two folds, or
from both sides of the nested train/val boundary. `tests/fixtures.py`'s tiny
extracts are too small to stratify by Study at all (some Studies would have
fewer Subjects than folds), so this file builds its own synthetic label
frames instead — shaped like the real `labels` table, sized to exercise
5-fold stratification.

Nothing here touches the live instance or the frozen cohort artefact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gl_lifesphere.evaluation import splits


def _labels(rows: list[tuple[str, str, int, bool]]) -> pd.DataFrame:
    """Build a labels-shaped frame from (subjectId, studyId, durationDays, event) tuples."""
    frame = pd.DataFrame(rows, columns=["subjectId", "studyId", "durationDays", "event"])
    return frame.astype(
        {
            "subjectId": "string",
            "studyId": "string",
            "durationDays": "Int64",
            "event": "boolean",
        }
    )


@pytest.fixture
def labels() -> pd.DataFrame:
    """20 Subjects in a big Study, 10 in a mid Study, 10 in a sparse-events Study.

    `STUDY-SPARSE` carries exactly one event among its 10 Subjects: with 5
    folds, at most one fold can hold it, so 4 of 5 folds are guaranteed to
    show zero `STUDY-SPARSE` events regardless of the shuffle seed. That
    determinism is what makes the sparse-Study assertions non-flaky.
    """
    rows: list[tuple[str, str, int, bool]] = []
    for i in range(20):
        rows.append((f"BIG-{i:02d}", "STUDY-BIG", 100 + i, i % 2 == 0))
    for i in range(10):
        rows.append((f"MID-{i:02d}", "STUDY-MID", 200 + i, i % 3 == 0))
    for i in range(10):
        rows.append((f"SPARSE-{i:02d}", "STUDY-SPARSE", 300 + i, i == 0))
    return _labels(rows)


@pytest.fixture
def assignment(labels: pd.DataFrame) -> pd.DataFrame:
    return splits.assign_folds(labels)


class TestAssignFolds:
    """Fold assignment stratified by Study, at Subject level."""

    def test_every_subject_gets_a_fold(self, labels: pd.DataFrame, assignment: pd.DataFrame) -> None:
        assert set(assignment["subjectId"]) == set(labels["subjectId"])
        assert assignment["fold"].notna().all()

    def test_fold_values_are_in_range(self, assignment: pd.DataFrame) -> None:
        assert set(assignment["fold"].unique()) <= set(range(splits.N_SPLITS))

    def test_no_subject_appears_in_two_folds(self, assignment: pd.DataFrame) -> None:
        """The ticket's first leakage assertion."""
        splits.assert_one_fold_per_subject(assignment)

    def test_a_duplicated_subject_with_conflicting_folds_is_caught(
        self, assignment: pd.DataFrame
    ) -> None:
        conflict = pd.concat(
            [assignment, assignment.iloc[[0]].assign(fold=(assignment.iloc[0]["fold"] + 1) % 5)],
            ignore_index=True,
        )
        with pytest.raises(AssertionError, match="more than one fold"):
            splits.assert_one_fold_per_subject(conflict)

    def test_deterministic_for_a_fixed_seed(self, labels: pd.DataFrame) -> None:
        first = splits.assign_folds(labels, seed=0)
        second = splits.assign_folds(labels, seed=0)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_can_disagree(self, labels: pd.DataFrame) -> None:
        """Not a required property, but pins that `seed` is actually threaded through."""
        first = splits.assign_folds(labels, seed=0)
        second = splits.assign_folds(labels, seed=1)
        assert not first["fold"].equals(second["fold"])

    def test_each_study_is_spread_across_every_fold(
        self, labels: pd.DataFrame, assignment: pd.DataFrame
    ) -> None:
        """Study-stratified: a Study with enough Subjects should reach every fold."""
        joined = assignment.merge(labels[["subjectId"]], on="subjectId")
        for study in ["STUDY-BIG", "STUDY-MID", "STUDY-SPARSE"]:
            folds_seen = set(joined.loc[assignment["studyId"] == study, "fold"])
            assert folds_seen == set(range(splits.N_SPLITS)), study


class TestFoldSplit:
    """The nested train/val/test partition, derived from one stratified assignment."""

    def test_train_val_test_partition_every_subject_exactly_once(
        self, labels: pd.DataFrame, assignment: pd.DataFrame
    ) -> None:
        """The ticket's second leakage assertion, restated as a set property.

        No Subject's rows can straddle the train/val boundary if train, val,
        and test are pairwise disjoint and their union is everyone.
        """
        all_subjects = set(labels["subjectId"])
        for outer_fold in range(splits.N_SPLITS):
            split = splits.fold_split(assignment, outer_fold)
            assert split.train & split.val == set()
            assert split.train & split.test == set()
            assert split.val & split.test == set()
            assert split.train | split.val | split.test == all_subjects

    def test_val_is_stratified_the_same_way_as_the_outer_folds(
        self, assignment: pd.DataFrame
    ) -> None:
        """The nested val slice is a rotated fold from the same partition.

        It is stratified identically to the outer folds by construction: it
        *is* one of the outer folds, not a second independent split.
        """
        for outer_fold in range(splits.N_SPLITS):
            val_fold = (outer_fold + 1) % splits.N_SPLITS
            expected_val = set(assignment.loc[assignment["fold"] == val_fold, "subjectId"])
            split = splits.fold_split(assignment, outer_fold)
            assert split.val == expected_val

    def test_test_fold_matches_the_named_fold(self, assignment: pd.DataFrame) -> None:
        for outer_fold in range(splits.N_SPLITS):
            expected_test = set(assignment.loc[assignment["fold"] == outer_fold, "subjectId"])
            assert splits.fold_split(assignment, outer_fold).test == expected_test

    def test_out_of_range_outer_fold_raises(self, assignment: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="outer_fold"):
            splits.fold_split(assignment, splits.N_SPLITS)
        with pytest.raises(ValueError, match="outer_fold"):
            splits.fold_split(assignment, -1)


class TestSparseStudies:
    """Deciding what per-Study metrics mean for a Study a fold barely samples."""

    def test_the_single_event_sparse_study_is_flagged(
        self, labels: pd.DataFrame, assignment: pd.DataFrame
    ) -> None:
        assert "STUDY-SPARSE" in splits.sparse_studies(labels, assignment)

    def test_a_study_with_events_in_every_fold_is_not_flagged(
        self, labels: pd.DataFrame, assignment: pd.DataFrame
    ) -> None:
        assert "STUDY-BIG" not in splits.sparse_studies(labels, assignment)

    def test_event_counts_are_reported_per_study_per_fold(
        self, labels: pd.DataFrame, assignment: pd.DataFrame
    ) -> None:
        counts = splits.study_fold_event_counts(labels, assignment)
        assert counts.loc["STUDY-SPARSE"].sum() == 1
        assert counts.loc["STUDY-BIG"].sum() == 10


class TestPersistence:
    """The assignment is written once and read back identically (#7's third bullet)."""

    def test_freeze_then_load_round_trips(self, labels: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
        written = splits.freeze(labels=labels, destination=tmp_path)
        back = splits.load_folds(directory=tmp_path)
        pd.testing.assert_frame_equal(written, back)

    def test_freeze_is_reproducible_across_reruns(self, labels: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A fresh `freeze()` call must reproduce the same folds, not just the same seed.

        Guards against a library version bump changing `StratifiedKFold`'s
        internal tie-breaking: the persisted file, not the seed alone, is
        what every model and every rerun must agree on.
        """
        first = splits.freeze(labels=labels, destination=tmp_path / "a")
        second = splits.freeze(labels=labels, destination=tmp_path / "b")
        pd.testing.assert_frame_equal(first, second)
