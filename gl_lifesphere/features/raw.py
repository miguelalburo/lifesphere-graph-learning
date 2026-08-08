"""Assemble the one-row-per-Subject frame the feature contract fits and transforms.

Deliberately unencoded: every column here is still a raw string, ordinal, or
proportion, with no imputation, standardisation or one-hot expansion applied.
Those are all training-fold statistics (#4 §3) and therefore belong to
`contract.py`, which is refit per fold. This module's output is fold-independent
and safe to build once and reuse across all five folds.

Joins **from the cohort roster** (`members`), never inner-joined onto
`diagnosis_primary` — one cohort Subject has no Diagnosis at all, and an
inner join would silently drop them and desynchronise the Subject set every
arm is required to share (`cast.reduce_diagnoses`'s docstring).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..extract.cypher import BASELINE_SAMPLE_TYPES
from ..extract.store import INTERIM_DIR, PROCESSED_DIR, read_table

# Columns pulled from each interim table. Everything else — `ethnicity`,
# `diagnosisMethod`, `sampleClass`, `preservationMethod`, `daysToCollection`,
# `nDiagnoses`, `hasPrimaryDiagnosis`, `primaryDiagnosisId` — is excluded here
# rather than dropped downstream, so a leaking column can never reach the
# contract by accident (#4 §2, §3, §4).
_SUBJECT_COLUMNS = ["subjectId", "sexAtBirth", "race", "ageAtIndexYears"]
_DIAGNOSIS_COLUMNS = [
    "subjectId",
    "stageOrdinal",
    "ageAtDiagnosisYears",
    "conditionSubtype",
    "conditionName",
]

SAMPLE_PROPORTION_COLUMNS: tuple[str, ...] = tuple(
    f"p_{sample_type.lower().replace(' ', '_')}" for sample_type in BASELINE_SAMPLE_TYPES
)


def _sample_proportions(samples: pd.DataFrame) -> pd.DataFrame:
    """Per-Subject proportion vector over the three baseline Sample types.

    Post-baseline types are excluded from both the numerator and the
    denominator — the proportion is taken over a Subject's baseline-type
    Samples alone, not over all their Samples (#4 §4). A Subject with no
    baseline-type Sample gets an all-NaN row, left for the contract's
    fold-mean imputation rather than assumed here.
    """
    baseline = samples[samples["sampleType"].isin(BASELINE_SAMPLE_TYPES)]
    counts = (
        baseline.groupby(["subjectId", "sampleType"])
        .size()
        .unstack("sampleType", fill_value=0)
        .reindex(columns=list(BASELINE_SAMPLE_TYPES), fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions.columns = list(SAMPLE_PROPORTION_COLUMNS)
    return proportions.reset_index()


def build_raw_frame(
    *,
    members: pd.DataFrame,
    subjects: pd.DataFrame,
    diagnosis_primary: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """One row per `members` Subject, joined from already-loaded interim tables.

    `studyId` is carried through for the Cox stratum and for training-fold
    within-Study statistics — never as a covariate (`guards.check_study_is_split_key_only`
    enforces this at the contract's output, not here). Pure function, so it is
    exercised directly against `tests/fixtures.py` without touching disk.
    """
    frame = members.merge(subjects[_SUBJECT_COLUMNS], on="subjectId", how="left")
    frame = frame.merge(diagnosis_primary[_DIAGNOSIS_COLUMNS], on="subjectId", how="left")
    frame = frame.merge(_sample_proportions(samples), on="subjectId", how="left")
    return frame.sort_values("subjectId").reset_index(drop=True)


def assemble_raw_frame(
    *,
    interim: Path | None = None,
    cohort: Path | None = None,
) -> pd.DataFrame:
    """`build_raw_frame`, reading its inputs from `data/interim/` and `data/processed/`."""
    interim_dir = interim if interim is not None else INTERIM_DIR
    cohort_dir = cohort if cohort is not None else PROCESSED_DIR / "cohort_os"

    return build_raw_frame(
        members=read_table("members", directory=cohort_dir),
        subjects=read_table("subjects", directory=interim_dir),
        diagnosis_primary=read_table("diagnosis_primary", directory=interim_dir),
        samples=read_table("samples", directory=interim_dir),
    )
