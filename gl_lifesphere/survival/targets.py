"""The Cox target every arm trains and is scored against (#3 §1, §2).

One entry point, `load_survival_frame`, so no arm re-derives the label from
`data/processed/cohort_os/` by path. `SurvivalTarget` is the plain-array form
every downstream loss/metric/decoder call takes — `study` is carried as a
string array here and coded to small integers only where a specific library
needs that (`losses.py`), so this module stays free of any library convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation.splits import load_folds
from ..extract.pipeline import load_cohort_labels


@dataclass(frozen=True)
class SurvivalTarget:
    """`time`/`event`/`study` in one fixed Subject order, plus that order's ids."""

    subject_id: np.ndarray
    study: np.ndarray
    time: np.ndarray
    event: np.ndarray

    def __len__(self) -> int:
        return len(self.subject_id)

    def subset(self, subject_ids: frozenset[str]) -> "SurvivalTarget":
        mask = np.isin(self.subject_id, list(subject_ids))
        return SurvivalTarget(
            subject_id=self.subject_id[mask],
            study=self.study[mask],
            time=self.time[mask],
            event=self.event[mask],
        )

    def reorder(self, subject_ids: "pd.Index | list[str]") -> "SurvivalTarget":
        """Re-index to exactly `subject_ids`, in that order.

        A design matrix (`FeatureContract.transform`) and a `SurvivalTarget`
        are built independently and must be zipped by `subjectId`, not by
        position — this is the one place that alignment happens, so an arm
        never re-derives its own subject-order bookkeeping.
        """
        position = {subject: i for i, subject in enumerate(self.subject_id)}
        order = np.array([position[str(subject)] for subject in subject_ids])
        return SurvivalTarget(
            subject_id=self.subject_id[order],
            study=self.study[order],
            time=self.time[order],
            event=self.event[order],
        )


def load_survival_frame() -> pd.DataFrame:
    """The frozen labels joined to their fold assignment, one row per cohort Subject."""
    labels = load_cohort_labels()
    folds = load_folds()
    frame = labels.merge(folds[["subjectId", "fold"]], on="subjectId", how="left")
    if frame["fold"].isna().any():
        missing = int(frame["fold"].isna().sum())
        raise ValueError(f"{missing} cohort Subject(s) have no fold assignment")
    return frame.sort_values("subjectId").reset_index(drop=True)


def load_targets() -> SurvivalTarget:
    """`SurvivalTarget` for the whole frozen cohort, sorted by `subjectId`."""
    frame = load_survival_frame()
    return SurvivalTarget(
        subject_id=frame["subjectId"].to_numpy(),
        study=frame["studyId"].astype(str).to_numpy(),
        time=frame["durationDays"].astype("float64").to_numpy(),
        event=frame["event"].astype("bool").to_numpy(),
    )
