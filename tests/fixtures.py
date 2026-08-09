"""Small schema-faithful extracts for the tests, built rather than recorded.

`tests/README.md` asks for fixtures recorded from small extracts, and these are
shaped exactly like one — same column names, same string representations, same
missingness tokens, same edge-property spellings — but the values are invented.

That is a deliberate substitution. `data/README.md` states the repo does not
redistribute LifeSphere's underlying data, and `tests/` *is* version-controlled,
so a literal recording would commit patient survival times to git. Fidelity is
kept where it does the work: every row here reproduces a trap that was verified
live on 2026-08-08, and `test_extract.py` documents which trap each row carries.

The counts are not the cohort's. Tests that pin the locked 6,811 / 2,091 figures
assert against `cohort.EXPECTED_*` and the live artefact, not against these.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Imported for annotations only: this module is loaded by every test file,
    # and the arm packages pull in torch/lifelines at import time.
    from gl_lifesphere.constructions.cache import SubjectRecords
    from gl_lifesphere.evaluation.splits import FoldSplit
    from gl_lifesphere.survival.targets import SurvivalTarget


def raw_subjects() -> list[dict[str, Any]]:
    """Subjects as the driver returns them: every value a string or None.

    Carries three traps. `ageAtIndexYears` is the `ageAtIndexDays` property,
    which holds **years** — the values are 0–89-scale, so a reader that divides
    by 365 produces infants. `-1` is the missing sentinel. And `race` /
    `ethnicity` absence appears in *both* representations: a true `None` and the
    literal string `"None"`, which is what the live instance held before and
    after a load normalisation between #4 and now.
    """
    return [
        {
            "subjectId": "S1",
            "studyId": "TCGA-BRCA",
            "sexAtBirth": "female",
            "race": "white",
            "ethnicity": "not hispanic or latino",
            "ageAtIndexYears": "61",
        },
        {
            "subjectId": "S2",
            "studyId": "TCGA-BRCA",
            "sexAtBirth": "female",
            "race": None,  # true null
            "ethnicity": "None",  # the string form of the same absence
            "ageAtIndexYears": "48",
        },
        {
            "subjectId": "S3",
            "studyId": "TCGA-SKCM",
            "sexAtBirth": "male",
            "race": "black or african american",
            "ethnicity": None,
            "ageAtIndexYears": "-1",  # missing sentinel, not a 1-year-old
        },
        {
            "subjectId": "S4",
            "studyId": "TCGA-SKCM",
            "sexAtBirth": "male",
            "race": "asian",
            "ethnicity": "hispanic or latino",
            "ageAtIndexYears": "89",
        },
    ]


def raw_survival() -> list[dict[str, Any]]:
    """Survival records, all four endpoints, every property a string.

    `timeToEventDays` includes `"9"` and `"1000"` so a lexicographic sort is
    distinguishable from a numeric one, and a day-0 record so the `> 0`
    eligibility rule has something to exclude.
    """
    return [
        {
            "survivalId": "V1",
            "subjectId": "S1",
            "studyId": "TCGA-BRCA",
            "survivalType": "OS",
            "timeToEventDays": "1000",
            "eventOccurred": "0",
        },
        {
            "survivalId": "V2",
            "subjectId": "S2",
            "studyId": "TCGA-BRCA",
            "survivalType": "OS",
            "timeToEventDays": "9",
            "eventOccurred": "1",
        },
        {
            "survivalId": "V3",
            "subjectId": "S3",
            "studyId": "TCGA-SKCM",
            "survivalType": "OS",
            "timeToEventDays": "0",  # excluded by the landmark rule
            "eventOccurred": "0",
        },
        {
            "survivalId": "V4",
            "subjectId": "S4",
            "studyId": "TCGA-SKCM",
            "survivalType": "OS",
            "timeToEventDays": "365",
            "eventOccurred": "1",
        },
        {
            "survivalId": "V5",
            "subjectId": "S1",
            "studyId": "TCGA-BRCA",
            "survivalType": "PFI",  # a non-OS endpoint the cohort rule must drop
            "timeToEventDays": "500",
            "eventOccurred": "1",
        },
    ]


def raw_diagnoses() -> list[dict[str, Any]]:
    """Diagnoses with their edge flag inline.

    `isPrimaryDiagnosis` holds the **strings** `'True'`/`'False'`, which is the
    schema's only edge property and the reason a Cypher `= true` comparison
    matches nothing.

    S3 reproduces the SKCM pattern that makes the per-field fallback necessary:
    its primary Diagnosis has no stage, while a secondary one does. S4 has no
    primary Diagnosis at all (null flag), one of the three such cohort Subjects.
    """
    return [
        {
            "diagnosisId": "D1",
            "subjectId": "S1",
            "isPrimaryDiagnosis": "True",
            "pathologicStage": "Stage IIA",
            "ageAtDiagnosisDays": "22280",
            "conditionSubtype": "Infiltrating Ductal Carcinoma",
            "diagnosisMethod": None,
            "conditionId": "C1",
            "conditionName": "Breast Invasive Carcinoma",
        },
        {
            "diagnosisId": "D2",
            "subjectId": "S2",
            "isPrimaryDiagnosis": "True",
            "pathologicStage": "Stage X",  # cannot be assessed -> missing, not a rank
            "ageAtDiagnosisDays": "17520",
            "conditionSubtype": "Lobular Carcinoma",
            "diagnosisMethod": None,
            "conditionId": "C1",
            "conditionName": "Breast Invasive Carcinoma",
        },
        {
            "diagnosisId": "D3",
            "subjectId": "S3",
            "isPrimaryDiagnosis": "True",
            "pathologicStage": None,  # stage recorded on the secondary instead
            "ageAtDiagnosisDays": "20000",
            "conditionSubtype": "Melanoma",
            "diagnosisMethod": None,
            "conditionId": "C2",
            "conditionName": "Skin Cutaneous Melanoma",
        },
        {
            "diagnosisId": "D4",
            "subjectId": "S3",
            "isPrimaryDiagnosis": "False",
            "pathologicStage": "Stage IIIB",
            "ageAtDiagnosisDays": None,
            "conditionSubtype": None,
            "diagnosisMethod": None,
            "conditionId": None,  # secondary diagnoses carry no Condition
            "conditionName": None,
        },
        {
            "diagnosisId": "D5",
            "subjectId": "S4",
            "isPrimaryDiagnosis": None,  # no primary flag at all
            "pathologicStage": "Stage IS",
            "ageAtDiagnosisDays": "12000",
            "conditionSubtype": "Melanoma",
            "diagnosisMethod": None,
            "conditionId": "C2",
            "conditionName": "Skin Cutaneous Melanoma",
        },
    ]


def raw_samples() -> list[dict[str, Any]]:
    """Samples spanning baseline and post-baseline types.

    `Metastatic` and `Recurrent Tumor` are the immortal-time types: a Subject
    must live long enough to be sampled again. `sampleClass` is the redundant
    deterministic function of `sampleType`.
    """
    return [
        {
            "sampleId": "M1",
            "subjectId": "S1",
            "sampleType": "Primary Tumor",
            "sampleClass": "Tumor",
            "preservationMethod": "FFPE",
            "daysToCollection": "12",
        },
        {
            "sampleId": "M2",
            "subjectId": "S1",
            "sampleType": "Blood Derived Normal",
            "sampleClass": "Normal",
            "preservationMethod": None,
            "daysToCollection": "14",
        },
        {
            "sampleId": "M3",
            "subjectId": "S2",
            "sampleType": "Solid Tissue Normal",
            "sampleClass": "Normal",
            "preservationMethod": None,
            "daysToCollection": None,
        },
        {
            "sampleId": "M4",
            "subjectId": "S3",
            "sampleType": "Metastatic",
            "sampleClass": "Tumor",
            "preservationMethod": None,
            "daysToCollection": "800",
        },
        {
            "sampleId": "M5",
            "subjectId": "S4",
            "sampleType": "Recurrent Tumor",
            "sampleClass": "Tumor",
            "preservationMethod": None,
            "daysToCollection": "1200",
        },
        {
            "sampleId": "M6",
            "subjectId": "S4",
            "sampleType": None,  # 2 such Sample nodes exist graph-wide
            "sampleClass": None,
            "preservationMethod": None,
            "daysToCollection": None,
        },
        {
            # A second Primary Tumor, so `sampleType` -> `sampleClass` has a
            # repeated key. Without one, the redundancy check is vacuously true.
            "sampleId": "M7",
            "subjectId": "S2",
            "sampleType": "Primary Tumor",
            "sampleClass": "Tumor",
            "preservationMethod": "Frozen",
            "daysToCollection": "31",
        },
    ]


def raw_pathology_details() -> list[dict[str, Any]]:
    """PathologyDetails. `necrosisPercent` is empty on every cohort row.

    Two rows hang off D1, because `HAS_PATHOLOGY` is **not** 1:1 with Diagnosis
    — 1,734 cohort Diagnoses reach 2–6 details, correcting the encoder doc.
    """
    return [
        {
            "pathologyDetailId": "P1",
            "diagnosisId": "D1",
            "subjectId": "S1",
            "necrosisPercent": None,
        },
        {
            "pathologyDetailId": "P2",
            "diagnosisId": "D1",
            "subjectId": "S1",
            "necrosisPercent": None,
        },
        {
            "pathologyDetailId": "P3",
            "diagnosisId": "D3",
            "subjectId": "S3",
            "necrosisPercent": None,
        },
    ]


RAW_PULLS = {
    "subjects": raw_subjects,
    "survival": raw_survival,
    "diagnoses": raw_diagnoses,
    "samples": raw_samples,
    "pathology_details": raw_pathology_details,
}


# ---------------------------------------------------------------------------
# A synthetic cohort, shaped like `data/interim/` (#12)
# ---------------------------------------------------------------------------
#
# The raw pulls above are four Subjects chosen to carry specific traps, which is
# right for the extract layer and far too small for anything that fits a feature
# contract or trains an encoder. This builds a cohort of the same *shape* — the
# interim tables, already cast — big enough that vocabularies, medians and
# standardisation constants are meaningful, while staying small enough to run in
# a test.
#
# It carries the structural shapes #11 asks the construction to handle: a
# Subject with no Diagnosis at all, one whose Diagnosis has no `OF_CONDITION`,
# one with several Diagnoses, one with no PathologyDetail, and post-baseline
# `sampleType` rows that #4 §4 excludes as nodes. It also reproduces the
# `conditionName` collapse: distinct `conditionId` values sharing one name.

SYNTHETIC_STUDIES = ("STUDY-A", "STUDY-B", "STUDY-C")
SYNTHETIC_BASELINE_TYPES = ("Primary Tumor", "Blood Derived Normal", "Solid Tissue Normal")

NO_DIAGNOSIS = "SUBJ-000"
NO_CONDITION = "SUBJ-001"
MANY_DIAGNOSES = "SUBJ-002"
NO_PATHOLOGY = "SUBJ-003"


def synthetic_interim(n_per_study: int = 20) -> dict[str, pd.DataFrame]:
    """`members` / `subjects` / `diagnoses` / `samples` / `pathology_details`."""

    subject_ids = [f"SUBJ-{i:03d}" for i in range(n_per_study * len(SYNTHETIC_STUDIES))]
    studies = [SYNTHETIC_STUDIES[i // n_per_study] for i in range(len(subject_ids))]

    subjects = pd.DataFrame(
        {
            "subjectId": subject_ids,
            "studyId": studies,
            "sexAtBirth": ["male" if i % 2 else "female" for i in range(len(subject_ids))],
            "race": ["white" if i % 3 else "asian" for i in range(len(subject_ids))],
            "ethnicity": [None] * len(subject_ids),
            "ageAtIndexYears": [50.0 + (i % 30) for i in range(len(subject_ids))],
        }
    )

    diagnosis_rows: list[dict[str, Any]] = []
    pathology_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for i, subject in enumerate(subject_ids):
        if subject != NO_DIAGNOSIS:
            n_diagnoses = 3 if subject == MANY_DIAGNOSES else 1 + (i % 2)
            for k in range(n_diagnoses):
                diagnosis_id = f"DIAG-{i:03d}-{k}"
                primary = k == 0
                has_condition = primary and subject != NO_CONDITION
                diagnosis_rows.append(
                    {
                        "diagnosisId": diagnosis_id,
                        "subjectId": subject,
                        "isPrimaryDiagnosis": primary,
                        "pathologicStageRaw": f"Stage {'I' * (1 + i % 4)}",
                        # A secondary Diagnosis usually records no stage (#11).
                        "stageOrdinal": float(1 + i % 4) if primary else None,
                        "ageAtDiagnosisYears": 50.0 + (i % 30),
                        "conditionSubtype": f"subtype-{i % 4}",
                        "diagnosisMethod": None,
                        # `OF_CONDITION` exists only on primary Diagnoses (#4 §1).
                        "conditionId": f"ICD10:C{i % 5:02d}.9" if has_condition else None,
                        # The live graph's collapse: distinct ids, one name.
                        "conditionName": "Malignant melanoma, NOS" if has_condition else None,
                    }
                )
                if subject != NO_PATHOLOGY:
                    for p in range(1 + (i % 2)):
                        pathology_rows.append(
                            {
                                "pathologyDetailId": f"PATH-{i:03d}-{k}-{p}",
                                "diagnosisId": diagnosis_id,
                                "subjectId": subject,
                                "necrosisPercent": None,
                            }
                        )

        for s in range(3 + (i % 4)):
            sample_rows.append(
                {
                    "sampleId": f"SAMP-{i:03d}-{s}",
                    "subjectId": subject,
                    "sampleType": (
                        "Metastatic"
                        if s == 0 and i % 4 == 0
                        else SYNTHETIC_BASELINE_TYPES[s % len(SYNTHETIC_BASELINE_TYPES)]
                    ),
                    "sampleClass": "Tumor",
                    "preservationMethod": None,
                    "daysToCollection": None,
                }
            )

    return {
        "members": pd.DataFrame({"subjectId": subject_ids, "studyId": studies}),
        "subjects": subjects,
        "diagnoses": pd.DataFrame(diagnosis_rows),
        "samples": pd.DataFrame(sample_rows),
        "pathology_details": pd.DataFrame(pathology_rows),
    }


def synthetic_survival(interim: dict[str, pd.DataFrame], *, seed: int = 0) -> pd.DataFrame:
    """Labels for `synthetic_interim`, with a genuine (if weak) stage signal.

    Higher stage -> shorter time, so a working pipeline scores above 0.5 and a
    sign-inverted one is visible rather than merely noisy. Scaled to thousands
    of days like the real cohort's `timeToEventDays`, so the locked 1/2/3-year
    horizons fall inside this cohort's follow-up range.
    """
    rng = np.random.default_rng(seed)
    members = interim["members"]
    primary = interim["diagnoses"][interim["diagnoses"]["isPrimaryDiagnosis"].fillna(False)]
    stage = (
        members.merge(primary[["subjectId", "stageOrdinal"]], on="subjectId", how="left")[
            "stageOrdinal"
        ]
        .fillna(2.0)
        .to_numpy(dtype="float64")
    )

    risk = 0.6 * stage
    latent = rng.exponential(scale=np.exp(-risk) * 8000.0)
    censor = rng.exponential(scale=4000.0, size=len(members))
    return pd.DataFrame(
        {
            "subjectId": members["subjectId"],
            "studyId": members["studyId"],
            "durationDays": np.ceil(np.minimum(latent, censor)) + 1.0,
            "event": latent <= censor,
        }
    )


def synthetic_arm_inputs(
    n_per_study: int = 20,
) -> "tuple[SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit]":
    """`(records, raw, targets, split)` — everything an arm's `run_fold` needs.

    Shared by `test_graph.py` and `test_diagnostics.py` rather than built twice.
    #13's controls are only meaningful if they run against the same inputs the
    arm does, and two fixtures that were supposed to agree and drifted would
    make a diagnostic pass or fail for reasons that have nothing to do with the
    thing it tests.

    The split is a contiguous 60/20/20, stratified by construction: each Study
    occupies a fixed contiguous block of `n_per_study` Subjects.
    """
    from gl_lifesphere.constructions import cache
    from gl_lifesphere.evaluation.splits import FoldSplit
    from gl_lifesphere.extract.cast import reduce_diagnoses
    from gl_lifesphere.features.raw import build_raw_frame
    from gl_lifesphere.survival.targets import SurvivalTarget

    interim = synthetic_interim(n_per_study)
    labels = synthetic_survival(interim)

    records = cache.build_subject_records(
        members=interim["members"],
        subjects=interim["subjects"],
        diagnoses=interim["diagnoses"],
        samples=interim["samples"],
        pathology=interim["pathology_details"],
    )
    raw = build_raw_frame(
        members=interim["members"],
        subjects=interim["subjects"],
        diagnosis_primary=reduce_diagnoses(interim["diagnoses"]),
        samples=interim["samples"],
    )
    targets = SurvivalTarget(
        subject_id=labels["subjectId"].to_numpy(),
        study=labels["studyId"].to_numpy(),
        time=labels["durationDays"].to_numpy(dtype="float64"),
        event=labels["event"].to_numpy(dtype="bool"),
    )

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()
    subject_ids = list(interim["members"]["subjectId"])
    for s in range(len(SYNTHETIC_STUDIES)):
        block = subject_ids[s * n_per_study : (s + 1) * n_per_study]
        train_ids.update(block[:12])
        val_ids.update(block[12:16])
        test_ids.update(block[16:])

    split = FoldSplit(
        train=frozenset(train_ids), val=frozenset(val_ids), test=frozenset(test_ids)
    )
    return records, raw, targets, split
