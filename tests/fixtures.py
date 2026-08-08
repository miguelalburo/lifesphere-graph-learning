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

from typing import Any


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
