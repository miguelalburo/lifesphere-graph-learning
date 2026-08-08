"""Checks that fail loudly on the bugs this layer exists to prevent.

These are library functions rather than test bodies so that the pipeline runs
them on the real extract, not only on fixtures. `tests/test_extract.py` calls
the same functions against recorded fixtures, so a rule cannot pass in tests and
be absent in the pull.

Everything here targets the quiet-wrong-answer class `tests/README.md` names:
failures that produce a plausible number rather than an exception.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

import pandas as pd

from .cast import MISSING_TOKENS
from .cypher import COHORT_STUDIES, POST_BASELINE_SAMPLE_TYPES

# Accepts a frame, its `.columns` index, or a plain list of names, so a caller
# can check a design matrix it has not built yet.
Columns: TypeAlias = "pd.DataFrame | pd.Index[str] | Iterable[str]"

# Columns derived from the `:Survival` node. These are the label. Encoder doc §0
# flags this as obvious-when-stated and very easy to lose, because "pull the
# Subject's whole neighbourhood" is the natural Cypher to write and it sweeps
# the label in with the features.
SURVIVAL_DERIVED: frozenset[str] = frozenset(
    {
        "survivalId",
        "survivalType",
        "timeToEventDays",
        "eventOccurred",
        "durationDays",
        "event",
        "OS",
        "OS.time",
    }
)

# Study is the split key and the Cox stratum. It is never a covariate: it is the
# dominant pan-cancer survival signal, and #2 measured that ~0.18 of a pooled
# pan-cancer C-index is cancer-type separation rather than prognostication.
SPLIT_KEY_ONLY: frozenset[str] = frozenset({"studyId", "study", "studyName", "studyAbbreviation"})

# Columns that encode how long a Subject was observed, in the sense of #4 §4.
IMMORTAL_TIME_DERIVED: frozenset[str] = frozenset(
    {"daysToCollection", "startDay", "nInterventions", "interventionType", "monthsObserved"}
)


class LeakageError(AssertionError):
    """Raised when a forbidden column reaches a feature matrix."""


def check_no_survival_columns(columns: Columns, *, where: str) -> None:
    """Assert no `:Survival`-derived column is present.

    Matching is case-insensitive and covers the endpoint-prefixed spellings a
    later PFI/DSS ticket would produce (`PFI.time`), so the check does not
    quietly stop applying when the endpoint changes.
    """
    present = sorted(
        column
        for column in _as_columns(columns)
        if _matches(column, SURVIVAL_DERIVED) or _bare(column).endswith(".time")
    )
    if present:
        raise LeakageError(
            f"{where}: Survival-derived column(s) reached the feature side: {present}. "
            "Survival is the label; see encoder doc §0."
        )


def check_study_is_split_key_only(columns: Columns, *, where: str) -> None:
    """Assert Study appears nowhere in a feature matrix."""
    present = sorted(
        column for column in _as_columns(columns) if _matches(column, SPLIT_KEY_ONLY)
    )
    if present:
        raise LeakageError(
            f"{where}: Study reached the feature side: {present}. "
            "Study is the split key and the Cox stratum, never a covariate (#3 §1)."
        )


def check_no_leaking_columns(columns: Columns, *, where: str) -> None:
    """Assert no immortal-time or ascertainment column reaches the feature side.

    #4 §4 inverted the map's original rule by measurement: the Sample *count* is
    harmless (HR 1.002, p = 0.56) while `daysToCollection` correlates with
    follow-up at r = 0.711, and bare Intervention presence is itself the leak
    (HR 0.716, p = 8.5e-05).
    """
    present = sorted(
        column for column in _as_columns(columns) if _matches(column, IMMORTAL_TIME_DERIVED)
    )
    if present:
        raise LeakageError(f"{where}: immortal-time column(s) reached the feature side: {present}")


def check_feature_frame(frame: pd.DataFrame, *, where: str) -> None:
    """Run every feature-side guard over one frame."""
    check_no_survival_columns(frame.columns, where=where)
    check_study_is_split_key_only(frame.columns, where=where)
    check_no_leaking_columns(frame.columns, where=where)


# ---------------------------------------------------------------------------
# Schema-drift checks (#4 §9)
# ---------------------------------------------------------------------------


def check_age_units(subjects: pd.DataFrame, diagnoses: pd.DataFrame) -> None:
    """Assert the two age properties still carry the units their casts assume.

    The single most likely quiet-wrong-answer bug in the feature contract:
    `Subject.ageAtIndexDays` holds **years** despite its name, and
    `Diagnosis.ageAtDiagnosisDays` holds **days**. Both are checked *undivided*
    so a schema fix that swapped either unit fails here rather than producing a
    cohort with a median age of 0.16.
    """
    index_age = subjects["ageAtIndexYears"].dropna()
    if len(index_age) and not index_age.between(0, 120).all():
        raise AssertionError(
            "Subject.ageAtIndexDays no longer holds years: observed range "
            f"[{index_age.min()}, {index_age.max()}], expected [0, 120]."
        )

    diagnosis_age = diagnoses["ageAtDiagnosisYears"].dropna()
    if len(diagnosis_age) and not diagnosis_age.between(0, 120).all():
        raise AssertionError(
            "Diagnosis.ageAtDiagnosisDays no longer holds days: converted range "
            f"[{diagnosis_age.min():.1f}, {diagnosis_age.max():.1f}] years, expected [0, 120]."
        )


def check_missing_tokens(frame: pd.DataFrame, *, where: str) -> None:
    """Assert no known missing token survived the cast as a category level.

    Both representations occur for the same property — the live instance stores
    `race` absence as a true null today, and #4 §3 recorded the literal string
    `"None"` for the same field on the same cohort. A load that normalises one
    into the other must not silently create a category called `"None"`.
    """
    offenders: dict[str, list[str]] = {}
    for column in frame.columns:
        if not isinstance(frame[column].dtype, pd.StringDtype):
            continue
        found = sorted(set(frame[column].dropna().unique()) & MISSING_TOKENS)
        if found:
            offenders[str(column)] = found
    if offenders:
        raise AssertionError(f"{where}: missing tokens survived as category levels: {offenders}")


def check_one_primary_diagnosis(diagnoses: pd.DataFrame) -> pd.Series:
    """Return the count of primary Diagnoses per Subject, asserting it never exceeds one.

    Zero is permitted and returned for inspection — 3 cohort Subjects have no
    primary Diagnosis (#4 §8) and the fallback covers them. More than one would
    make the reduction order-dependent, which is the failure worth catching.
    """
    counts = (
        diagnoses.assign(primary=diagnoses["isPrimaryDiagnosis"].fillna(False))
        .groupby("subjectId")["primary"]
        .sum()
    )
    over = counts[counts > 1]
    if len(over):
        raise AssertionError(
            f"{len(over)} Subject(s) carry more than one primary Diagnosis, "
            f"e.g. {over.head().to_dict()}. The reduction in cast.reduce_diagnoses "
            "assumes at most one."
        )
    return counts


def check_sample_class_is_redundant(samples: pd.DataFrame) -> None:
    """Assert `sampleClass` is still a deterministic function of `sampleType`.

    It is dropped as carrying exactly zero information (#4 §4). If the mapping
    ever stops being one-to-one the field has started carrying something, and
    dropping it stops being free.
    """
    pairs = samples.dropna(subset=["sampleType", "sampleClass"])
    ambiguous = pairs.groupby("sampleType")["sampleClass"].nunique()
    offenders = ambiguous[ambiguous > 1]
    if len(offenders):
        raise AssertionError(
            f"sampleClass is no longer determined by sampleType: {offenders.to_dict()}"
        )


def check_no_post_baseline_samples(samples: pd.DataFrame, *, where: str) -> None:
    """Assert no post-baseline Sample type survives into a feature-side table."""
    present = sorted(set(samples["sampleType"].dropna().unique()) & POST_BASELINE_SAMPLE_TYPES)
    if present:
        raise LeakageError(
            f"{where}: post-baseline sample type(s) present: {present}. "
            "These exist only because the Subject lived to be sampled again (#4 §4)."
        )


def check_cohort_studies(frame: pd.DataFrame, *, column: str = "studyId") -> None:
    """Assert a frame's Studies are exactly the pinned 20."""
    found = set(frame[column].dropna().unique())
    expected = set(COHORT_STUDIES)
    if found != expected:
        raise AssertionError(
            f"Study set differs from the pinned cohort: "
            f"unexpected={sorted(found - expected)}, missing={sorted(expected - found)}"
        )


def _as_columns(columns: Columns) -> list[str]:
    if isinstance(columns, pd.DataFrame):
        return [str(column) for column in columns.columns]
    return [str(column) for column in columns]


def _bare(column: str) -> str:
    """Normalise a column name for comparison against the forbidden lists."""
    return column.strip().lower()


def _matches(column: str, forbidden: frozenset[str]) -> bool:
    """True if `column` is a forbidden name, or a one-hot expansion of one.

    One-hot encoding turns `eventOccurred` into `eventOccurred_1`, which an
    exact-match check would wave through — so a trailing `_<level>` is stripped
    before comparing.
    """
    bare = _bare(column)
    lowered = {name.lower() for name in forbidden}
    return bare in lowered or any(bare.startswith(f"{name}_") for name in lowered)
