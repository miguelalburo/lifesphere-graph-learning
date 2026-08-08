"""The OS cohort, materialised as an artefact rather than re-derived per experiment.

The eligibility rule lives here exactly once, in Python, where it is testable
without the live instance. It is deliberately *not* pushed into Cypher: doing so
would split the rule across two languages and put its numeric predicate
(`timeToEventDays > 0`) on the far side of a cast this package exists to make
explicit.

Freezing matters because all three arms must train on an identical Subject set
for any difference between them to be attributable to the representation. A
cohort re-queried per experiment is a cohort that can drift between arms.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import guards
from .cypher import COHORT_STUDIES, OS_ENDPOINT
from .store import INTERIM_DIR, PROCESSED_DIR, read_table, write_table

# The locked figures, from #3 §2 as amended and reproduced live on 2026-08-08.
# A mismatch is a bug to investigate, not a number to update.
EXPECTED_SUBJECTS = 6_811
EXPECTED_EVENTS = 2_091
EXPECTED_STUDIES = 20

# Before the `timeToEventDays > 0` rule, for the reconciliation in `freeze`.
EXPECTED_SUBJECTS_BEFORE_LANDMARK = 6_884
EXPECTED_EVENTS_BEFORE_LANDMARK = 2_097


class CohortMismatchError(AssertionError):
    """Raised when the frozen cohort does not match the locked figures."""


@dataclass(frozen=True)
class CohortCounts:
    """What an eligibility pass produced, for comparison against the locked figures."""

    n_subjects: int
    n_events: int
    n_studies: int

    @property
    def censoring_rate(self) -> float:
        return 1.0 - (self.n_events / self.n_subjects) if self.n_subjects else 0.0

    def describe(self) -> str:
        return (
            f"{self.n_subjects} Subjects / {self.n_events} events "
            f"({self.censoring_rate:.1%} censored) / {self.n_studies} Studies"
        )


def select_os_cohort(survival: pd.DataFrame) -> pd.DataFrame:
    """Apply the eligibility rule to a typed Survival table.

    Three conditions, in the order they were settled:

    1. **Study is one of the pinned 20** — the Studies where AJCC pathologic
       staging is present. Admitting the other 13 would hand every arm a
       near-perfect cancer-type proxy through "stage unknown" (#1).
    2. **The record is the OS endpoint.** Exactly one OS record exists per
       Subject, so no de-duplication rule is needed (#3 §2).
    3. **`timeToEventDays > 0`**, per TCGA-CDR's documented rule. A Subject
       censored at t=0 sits in `R(0)` and nowhere else and contributes zero
       comparable pairs to the C-index, so they inflate the reported cohort size
       without entering any fit; the 6 day-0 deaths would enter the outer sum
       against the full risk set, giving an implausible record maximal leverage.
       Removes 73 records — 67 censored, 6 events, none negative.
    """
    eligible = survival[
        survival["studyId"].isin(COHORT_STUDIES)
        & (survival["survivalType"] == OS_ENDPOINT)
        & (survival["timeToEventDays"] > 0)
    ]
    return eligible.sort_values("subjectId").reset_index(drop=True)


def count(cohort: pd.DataFrame) -> CohortCounts:
    """Summarise a cohort frame."""
    return CohortCounts(
        n_subjects=int(cohort["subjectId"].nunique()),
        n_events=int(cohort["eventOccurred"].sum()),
        n_studies=int(cohort["studyId"].nunique()),
    )


def assert_locked(counts: CohortCounts) -> None:
    """Raise unless the cohort matches the figures locked by #3.

    The issue's own instruction: treat a mismatch as a bug to investigate, not a
    number to update. The likely causes, in order — a cast that dropped records,
    a Study list recomputed from the `>= 0.86` threshold (which yields 19 and
    silently drops TCGA-HNSC), or a graph reload.
    """
    expected = CohortCounts(EXPECTED_SUBJECTS, EXPECTED_EVENTS, EXPECTED_STUDIES)
    if counts != expected:
        raise CohortMismatchError(
            f"Frozen cohort is {counts.describe()}; expected {expected.describe()}. "
            "Investigate rather than updating the constant — check the cast, then "
            "check whether the Study list was recomputed from a threshold (#4)."
        )


def freeze(
    *,
    interim: Path | None = None,
    destination: Path | None = None,
) -> tuple[pd.DataFrame, CohortCounts]:
    """Materialise the frozen cohort to `data/processed/cohort_os/`.

    Writes two tables, separated on purpose:

    - `labels` — Subject, Study, duration, event. The Cox target and the split
      key, and the only place `:Survival` is permitted to appear.
    - `members` — Subject and Study alone, the roster every arm joins against.

    Nothing feature-shaped is written here. Keeping the label table physically
    apart from anything an arm builds a design matrix from is what makes
    `guards.check_no_survival_columns` an enforceable rule rather than a
    convention.
    """
    source = interim if interim is not None else INTERIM_DIR
    target = destination if destination is not None else PROCESSED_DIR / "cohort_os"

    survival = read_table("survival", directory=source)
    cohort = select_os_cohort(survival)
    counts = count(cohort)
    assert_locked(counts)
    guards.check_cohort_studies(cohort)

    labels = cohort[["subjectId", "studyId", "timeToEventDays", "eventOccurred"]].rename(
        columns={"timeToEventDays": "durationDays", "eventOccurred": "event"}
    )
    members = labels[["subjectId", "studyId"]]

    per_study = (
        labels.groupby("studyId")
        .agg(n_subjects=("subjectId", "nunique"), n_events=("event", "sum"))
        .sort_index()
    )

    meta = {
        "endpoint": OS_ENDPOINT,
        "frozen_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "rule": (
            "studyId in the pinned 20 Studies; survivalType == 'OS'; timeToEventDays > 0"
        ),
        "n_subjects": counts.n_subjects,
        "n_events": counts.n_events,
        "n_studies": counts.n_studies,
        "censoring_rate": round(counts.censoring_rate, 4),
        "studies": list(COHORT_STUDIES),
        "per_study": {
            str(study): {"n_subjects": int(row.n_subjects), "n_events": int(row.n_events)}
            for study, row in per_study.iterrows()
        },
        "excluded_day_zero": int(
            (
                (survival["studyId"].isin(COHORT_STUDIES))
                & (survival["survivalType"] == OS_ENDPOINT)
                & (survival["timeToEventDays"] <= 0)
            ).sum()
        ),
    }

    write_table("labels", labels, directory=target, meta=meta)
    write_table("members", members, directory=target, meta=meta)
    return labels, counts
