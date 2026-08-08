"""The 5-fold Study-stratified split protocol, split at Subject level (#7).

Settled on the map (#1): 5-fold Study-stratified CV, split at Subject level,
with a nested validation slice for early stopping on the neural arms, at a
fixed seed. All three arms consume the identical fold assignment, which is
persisted rather than re-derived — a seed alone is not enough once library
versions move (`TestPersistence` in `tests/test_evaluation.py` pins that a
fresh `freeze()` reproduces the same folds a stale `StratifiedKFold` update
could otherwise change silently).

**Stratification is by Study only, not by Study and event jointly.** TCGA-TGCT
carries 4 events across 133 Subjects — fewer events than folds — so stratifying
jointly is not achievable for every Study and would raise inside
`StratifiedKFold` for the smallest one. Study-only stratification is what the
map actually asks for ("each fold preserves the cancer-type mix"); event
coverage per fold is a consequence of that, not a second constraint layered on
top.

**The nested validation slice is a rotated fold, not a second random split.**
For outer test fold `i`, the validation slice is fold `(i + 1) % 5` and the
remaining three folds are train. Reusing the same stratified partition means
the val slice is stratified identically to the outer folds by construction,
and "no Subject in two folds" holds for the whole train/val/test partition
without a second leakage check.

**Small Studies do not get events in every fold, and that is not a bug.**
`sparse_studies` flags Studies where some fold holds too few events for a
per-fold metric on that Study to mean anything (TCGA-TGCT, at 4 events across
133 Subjects, cannot reach every one of 5 folds no matter how it is split).
The decision this module makes, for `compare` to honour once it exists: such
Studies are reported **pooled** into the cross-Study figure rather than
per-fold or silently dropped — dropping would hide that the model saw no
positive examples of that Study in that fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..extract.store import PROCESSED_DIR, read_table, write_table

N_SPLITS = 5
SEED = 0

FOLDS_DIR = PROCESSED_DIR / "cohort_os"


@dataclass(frozen=True)
class FoldSplit:
    """The three disjoint Subject sets for one outer fold."""

    train: frozenset[str]
    val: frozenset[str]
    test: frozenset[str]


def assign_folds(labels: pd.DataFrame, *, seed: int = SEED) -> pd.DataFrame:
    """Assign every Subject in `labels` to one of `N_SPLITS` folds, stratified by Study.

    `labels` is one row per Subject (the frozen cohort's invariant), so
    stratifying by `studyId` here is stratifying at the Subject level.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold = np.empty(len(labels), dtype="int64")
    dummy_x = np.zeros(len(labels))
    for fold_id, (_, test_idx) in enumerate(skf.split(dummy_x, labels["studyId"])):
        fold[test_idx] = fold_id
    return pd.DataFrame(
        {
            "subjectId": labels["subjectId"],
            "studyId": labels["studyId"],
            "fold": pd.array(fold, dtype="Int64"),
        }
    ).reset_index(drop=True)


def assert_one_fold_per_subject(assignment: pd.DataFrame) -> None:
    """Raise unless every Subject in `assignment` maps to exactly one fold."""
    counts = assignment.groupby("subjectId")["fold"].nunique()
    offenders = counts[counts > 1]
    if len(offenders):
        raise AssertionError(
            f"{len(offenders)} Subject(s) map to more than one fold: {offenders.to_dict()}"
        )


def fold_split(assignment: pd.DataFrame, outer_fold: int) -> FoldSplit:
    """The train/val/test Subject sets for one outer fold.

    `val` is the fold immediately after `outer_fold` in rotation, so it is
    drawn from the same stratified partition rather than a fresh split.
    """
    if not 0 <= outer_fold < N_SPLITS:
        raise ValueError(f"outer_fold must be in [0, {N_SPLITS}), got {outer_fold}")
    val_fold = (outer_fold + 1) % N_SPLITS
    test = frozenset(assignment.loc[assignment["fold"] == outer_fold, "subjectId"])
    val = frozenset(assignment.loc[assignment["fold"] == val_fold, "subjectId"])
    train = frozenset(
        assignment.loc[~assignment["fold"].isin([outer_fold, val_fold]), "subjectId"]
    )
    return FoldSplit(train=train, val=val, test=test)


def study_fold_event_counts(labels: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Event count per (Study, fold) cell, for deciding which Studies are sparse."""
    joined = labels.merge(assignment[["subjectId", "fold"]], on="subjectId", how="left")
    return (
        joined.groupby(["studyId", "fold"])["event"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=range(N_SPLITS), fill_value=0)
    )


def sparse_studies(
    labels: pd.DataFrame, assignment: pd.DataFrame, *, min_events: int = 1
) -> list[str]:
    """Studies where at least one fold holds fewer than `min_events` events.

    Per this module's decision (see the module docstring), such Studies are
    pooled into the cross-Study figure rather than reported fold-by-fold.
    """
    counts = study_fold_event_counts(labels, assignment)
    return sorted(counts.index[(counts < min_events).any(axis=1)])


def freeze(
    *,
    labels: pd.DataFrame,
    destination: Path | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compute the fold assignment and persist it to `destination`.

    Written beside the cohort it splits (`data/processed/cohort_os/` by
    default) so every arm reads one file rather than recomputing `assign_folds`
    with the risk of a library version drifting the result.
    """
    target = destination if destination is not None else FOLDS_DIR
    assignment = assign_folds(labels, seed=seed)
    assert_one_fold_per_subject(assignment)
    meta = {
        "n_splits": N_SPLITS,
        "seed": seed,
        "sparse_studies": sparse_studies(labels, assignment),
    }
    write_table("folds", assignment, directory=target, meta=meta)
    return assignment


def load_folds(*, directory: Path | None = None) -> pd.DataFrame:
    """Read the persisted fold assignment back."""
    return read_table("folds", directory=directory if directory is not None else FOLDS_DIR)
