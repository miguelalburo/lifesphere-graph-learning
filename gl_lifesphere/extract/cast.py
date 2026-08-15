"""Typed casts from `data/raw/` to `data/interim/`.

Every property on the live graph is `STRING NOT NULL` — including
`timeToEventDays`, `eventOccurred`, `ageAtIndexDays`, `pathologicStage` and the
`isPrimaryDiagnosis` edge flag. So every cast in the project happens here, once,
under test.

The casts are strict on purpose. `pd.to_numeric(..., errors="coerce")` and
`astype(bool)` are the two calls that turn a schema change into a plausible
wrong answer rather than a crash: `astype(bool)` maps the string `"0"` to
`True`, and `errors="coerce"` maps an unrecognised token to `NaN` that reads as
"not recorded". Both are avoided; unparseable input raises, and the missing
tokens are enumerated rather than inferred.
"""

from __future__ import annotations

import re

import pandas as pd

# Values that mean "not recorded" rather than naming a category.
#
# Both representations genuinely occur for the same property, so a reader that
# handles one is wrong about the other. The live instance currently stores
# `race` / `ethnicity` absence as **true nulls** (720 and 1,368 cohort
# Subjects); #4 §3 recorded the literal string `"None"` for the same fields on
# the same cohort, which means the load has been normalised between those two
# measurements. Handling both is what makes this layer survive the next such
# change — see `guards.check_missing_tokens`.
MISSING_TOKENS: frozenset[str] = frozenset(
    {"None", "none", "", "NA", "N/A", "null", "NULL", "nan", "NaN", "unknown", "Unknown"}
)

# `Subject.ageAtIndexDays` uses -1 for "not recorded" (graph-wide; it does not
# occur inside the cohort). It is a sentinel, not an age.
AGE_MISSING_SENTINEL = -1

# `eventOccurred` is the string "0" or "1", with no nulls, across all 10,805 OS
# records. Mapped explicitly because `astype(bool)` maps *both* to True.
EVENT_VALUES: dict[str, bool] = {"0": False, "1": True}

# `isPrimaryDiagnosis` is the schema's only edge property and holds the strings
# "True" / "False" — 11,293 / 7,542 / 8 null graph-wide. A Cypher comparison
# against the boolean `true` matches zero rows and raises nothing, so the
# comparison is done here where an unexpected value fails loudly.
PRIMARY_VALUES: dict[str, bool] = {"True": True, "False": False}

DAYS_PER_YEAR = 365.25

# "Stage IIIC" -> ("III", "C"). Ordered longest-first so IV is not read as I.
_STAGE_PATTERN = re.compile(r"^\s*(?:Stage\s+)?(0is|0a|0|IV|III|II|I)([A-Za-z0-9]*)\s*$")
_ROMAN_TO_ORDINAL: dict[str, int] = {"0": 0, "0a": 0, "0is": 0, "I": 1, "II": 2, "III": 3, "IV": 4}


def normalise_missing(series: pd.Series) -> pd.Series:
    """Map every representation of "not recorded" to `pd.NA`.

    Applied before any other cast, so the downstream casts only ever see a real
    value or `pd.NA`. Values that are already null are left alone; `isin` does
    not match them, and masking is idempotent.
    """
    return series.mask(series.isin(MISSING_TOKENS))


def to_int(series: pd.Series, *, name: str) -> pd.Series:
    """Cast a string column to nullable `Int64`, raising on anything unparseable.

    Strict rather than coercing: `timeToEventDays` sorting lexicographically
    (`"1000" < "999"`) is silent, and so is a coerced `NaN` that later reads as
    an unrecorded follow-up.
    """
    cleaned = normalise_missing(series)
    present = cleaned[cleaned.notna()].astype(str)
    bad = present[~present.str.fullmatch(r"-?\d+")]
    if len(bad):
        raise ValueError(
            f"{name}: {len(bad)} value(s) are not integers, e.g. {sorted(set(bad))[:5]}"
        )
    return cleaned.astype("Float64").astype("Int64")


def to_float(series: pd.Series, *, name: str) -> pd.Series:
    """Cast a string column to nullable `Float64`, raising on anything unparseable."""
    cleaned = normalise_missing(series)
    present = cleaned[cleaned.notna()].astype(str)
    bad = present[~present.str.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")]
    if len(bad):
        raise ValueError(f"{name}: {len(bad)} value(s) are not numeric, e.g. {sorted(set(bad))[:5]}")
    return cleaned.astype("Float64")


def to_bool(series: pd.Series, mapping: dict[str, bool], *, name: str) -> pd.Series:
    """Cast a string column to nullable `boolean` through an explicit mapping.

    Anything outside `mapping` raises. This is the cast that most needs to fail
    loudly: a censoring indicator that is wrong in one direction produces a
    C-index of `1 - C`, which reads as a bad model rather than as a defect.
    """
    cleaned = normalise_missing(series)
    present = cleaned[cleaned.notna()].astype(str)
    unknown = sorted(set(present) - set(mapping))
    if unknown:
        raise ValueError(f"{name}: unexpected value(s) {unknown}; expected {sorted(mapping)}")
    return cleaned.map(mapping).astype("boolean")


def stage_ordinal(series: pd.Series) -> pd.Series:
    """Collapse `pathologicStage` to the ordinal scale 0–4.

    Settled by measurement in #4 §3: against a fold sd of ~0.016, ordinal I–IV
    (0.6666), ordinal plus fractional sub-stage (0.6682) and a 17-level one-hot
    (0.6659) span 0.0023 — indistinguishable under #3's ~0.02 resolution limit.
    The ordinal wins on parameter count, not on fit: it does not ask a
    ~65-events-per-parameter budget to fit levels with n = 1, 5 and 7.

    Sub-stage letters are stripped by pattern rather than by a hand-listed
    vocabulary, so a value the cohort does not currently carry (the GDC
    dictionary permits `IB1`, `IIIC2`, …) collapses correctly instead of
    silently becoming missing.

    Two judgement calls, both recorded because #4 fixes only the first:
    `Stage X` means "cannot be assessed" and becomes missing rather than
    acquiring a rank; `Stage IS` (46 Subjects, testicular in situ) strips to
    `I` under the same rule as every other suffix. The raw string is kept in a
    neighbouring column, so revisiting either is a one-line change.
    """
    cleaned = normalise_missing(series)

    def parse(value: object) -> object:
        if value is pd.NA or value is None:
            return pd.NA
        match = _STAGE_PATTERN.match(str(value))
        if match is None:
            # `Stage X` and anything else unrecognised: no defensible rank.
            return pd.NA
        return _ROMAN_TO_ORDINAL[match.group(1)]

    return cleaned.map(parse).astype("Int64")


# ---------------------------------------------------------------------------
# Per-table casts
# ---------------------------------------------------------------------------


def cast_subjects(raw: pd.DataFrame) -> pd.DataFrame:
    """Type the Subject table.

    `ageAtIndexYears` arrives already renamed by the pull, because the property
    is called `ageAtIndexDays` and holds **years** (#3, #4 §3). The `-1` missing
    sentinel is removed here rather than surviving as a 1-year-old.
    """
    age = to_int(raw["ageAtIndexYears"], name="ageAtIndexYears")
    return pd.DataFrame(
        {
            "subjectId": raw["subjectId"].astype("string"),
            "studyId": raw["studyId"].astype("string"),
            "sexAtBirth": normalise_missing(raw["sexAtBirth"]).astype("string"),
            "race": normalise_missing(raw["race"]).astype("string"),
            "ethnicity": normalise_missing(raw["ethnicity"]).astype("string"),
            "ageAtIndexYears": age.where(age != AGE_MISSING_SENTINEL).astype("Float64"),
        }
    )


def cast_survival(raw: pd.DataFrame) -> pd.DataFrame:
    """Type the Survival table. All four endpoints; selection happens in `cohort`."""
    return pd.DataFrame(
        {
            "survivalId": raw["survivalId"].astype("string"),
            "subjectId": raw["subjectId"].astype("string"),
            "studyId": raw["studyId"].astype("string"),
            "survivalType": raw["survivalType"].astype("string"),
            "timeToEventDays": to_int(raw["timeToEventDays"], name="timeToEventDays"),
            "eventOccurred": to_bool(raw["eventOccurred"], EVENT_VALUES, name="eventOccurred"),
        }
    )


def cast_diagnoses(raw: pd.DataFrame) -> pd.DataFrame:
    """Type the Diagnosis table, adding the collapsed stage and age in years.

    `ageAtDiagnosisDays` genuinely is days (cohort range 4,385–32,872), unlike
    its Subject-level sibling. The division is done once, here.
    """
    days = to_float(raw["ageAtDiagnosisDays"], name="ageAtDiagnosisDays")
    stage_raw = normalise_missing(raw["pathologicStage"]).astype("string")
    return pd.DataFrame(
        {
            "diagnosisId": raw["diagnosisId"].astype("string"),
            "subjectId": raw["subjectId"].astype("string"),
            "isPrimaryDiagnosis": to_bool(
                raw["isPrimaryDiagnosis"], PRIMARY_VALUES, name="isPrimaryDiagnosis"
            ),
            "pathologicStageRaw": stage_raw,
            "stageOrdinal": stage_ordinal(stage_raw),
            "ageAtDiagnosisYears": days / DAYS_PER_YEAR,
            "conditionSubtype": normalise_missing(raw["conditionSubtype"]).astype("string"),
            "diagnosisMethod": normalise_missing(raw["diagnosisMethod"]).astype("string"),
            "conditionId": normalise_missing(raw["conditionId"]).astype("string"),
            "conditionName": normalise_missing(raw["conditionName"]).astype("string"),
        }
    )


def cast_samples(raw: pd.DataFrame) -> pd.DataFrame:
    """Type the Sample table.

    `daysToCollection` is cast but carries a warning in its name's docs: it
    correlates with follow-up time at r = 0.711 and must not reach a design
    matrix (#4 §4). `guards.check_no_leaking_columns` enforces that.
    """
    return pd.DataFrame(
        {
            "sampleId": raw["sampleId"].astype("string"),
            "subjectId": raw["subjectId"].astype("string"),
            "sampleType": normalise_missing(raw["sampleType"]).astype("string"),
            "sampleClass": normalise_missing(raw["sampleClass"]).astype("string"),
            "preservationMethod": normalise_missing(raw["preservationMethod"]).astype("string"),
            "daysToCollection": to_float(raw["daysToCollection"], name="daysToCollection"),
        }
    )


def cast_pathology_details(raw: pd.DataFrame) -> pd.DataFrame:
    """Type the PathologyDetail table. `necrosisPercent` is empty on the cohort."""
    return pd.DataFrame(
        {
            "pathologyDetailId": raw["pathologyDetailId"].astype("string"),
            "diagnosisId": raw["diagnosisId"].astype("string"),
            "subjectId": raw["subjectId"].astype("string"),
            "necrosisPercent": to_float(raw["necrosisPercent"], name="necrosisPercent"),
        }
    )


CASTS = {
    "subjects": cast_subjects,
    "survival": cast_survival,
    "diagnoses": cast_diagnoses,
    "samples": cast_samples,
    "pathology_details": cast_pathology_details,
}


# ---------------------------------------------------------------------------
# The Diagnosis reduction (#4 §1)
# ---------------------------------------------------------------------------


def reduce_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Reduce many Diagnoses per Subject to one row: primary, with per-field fallback.

    Take the Diagnosis on the `isPrimaryDiagnosis` edge; where one of its fields
    is null, coalesce from that Subject's other Diagnoses.

    The fallback is not a nicety. Under primary-*strict*, stage is missing for
    561 cohort Subjects and **337 of TCGA-SKCM's 426 (79%)**, because melanoma
    stage is routinely recorded on a non-primary Diagnosis. A stage column that
    is 79% missing in exactly one Study is the map's bimodality trap reappearing
    *inside* the cohort built to prevent it. With the fallback, 264 Subjects
    lack stage, spread across 17 Studies at a maximum of 14% — measured in #4 §3
    to be neither a cancer-type channel (R² = 0.008) nor prognostic (p = 0.40).

    Three cohort Subjects have no primary Diagnosis (#4 §8). Two have a null
    flag and still get a row, filled entirely from the fallback. The third has
    **no Diagnosis at all**, so it has no row here — this table has 6,810 rows
    against the cohort's 6,811.

    That asymmetry is deliberate rather than papered over with an empty row: a
    Subject with no Diagnosis is a different thing from one whose fields are all
    null, and only the join makes it visible. **Models must left-join from the
    cohort roster (`members`), never inner-join onto this table**, or they
    silently drop a Subject and no longer train on an identical set. Imputing
    the resulting nulls is the feature layer's job, since the rule is
    training-fold-dependent.
    """
    ordered = diagnoses.sort_values(
        ["subjectId", "isPrimaryDiagnosis", "diagnosisId"],
        ascending=[True, False, True],
        na_position="last",
    )

    fields = [
        "pathologicStageRaw",
        "stageOrdinal",
        "ageAtDiagnosisYears",
        "conditionSubtype",
        "diagnosisMethod",
        "conditionId",
        "conditionName",
    ]

    grouped = ordered.groupby("subjectId", sort=True)
    # `first()` skips NA by default, which is exactly the per-field coalesce:
    # rows are ordered primary-first, so a present primary value wins and a null
    # one falls through to the next Diagnosis.
    reduced = grouped[fields].first()

    # Provenance, so a later ticket can tell a fallback value from a primary one
    # without re-deriving the rule.
    primary_id = (
        ordered[ordered["isPrimaryDiagnosis"].fillna(False)]
        .groupby("subjectId")["diagnosisId"]
        .first()
    )
    reduced["primaryDiagnosisId"] = primary_id
    reduced["nDiagnoses"] = grouped.size()
    reduced["hasPrimaryDiagnosis"] = reduced["primaryDiagnosisId"].notna()

    return reduced.reset_index()
