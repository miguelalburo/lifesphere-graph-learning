"""Tests for the extract layer (#6).

Priority is `tests/README.md`'s quiet-wrong-answer class: failures that produce
a plausible number rather than an exception. Every trap pinned here was
verified against the live instance on 2026-08-08; the fixtures reproduce its
string representations without carrying its data (see `tests/fixtures.py`).

Nothing here touches the live instance.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from gl_lifesphere.extract import cast, cohort, guards, store
from gl_lifesphere.extract.cypher import (
    BASELINE_SAMPLE_TYPES,
    COHORT_STUDIES,
    DROPPED_NODE_TYPES,
    OS_ENDPOINT,
    POST_BASELINE_SAMPLE_TYPES,
    PULLS,
)

import fixtures


@pytest.fixture
def raw() -> dict[str, pd.DataFrame]:
    """Every raw pull as a frame of strings and nulls, as `read_raw` returns it."""
    return {name: pd.DataFrame(builder()) for name, builder in fixtures.RAW_PULLS.items()}


@pytest.fixture
def typed(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Every raw pull after its cast."""
    return {name: cast.CASTS[name](frame) for name, frame in raw.items()}


class TestSurvivalCasts:
    """The censoring bookkeeping. A wrong answer here is silent in both directions."""

    def test_event_occurred_string_zero_is_false(self, typed: dict[str, pd.DataFrame]) -> None:
        """`astype(bool)` maps the string "0" to True. The explicit mapping must not."""
        events = typed["survival"].set_index("survivalId")["eventOccurred"]
        assert events["V1"] is not True and not bool(events["V1"])
        assert bool(events["V2"])
        # The bug this guards against, stated as the property that must not hold.
        assert bool(pd.Series(["0"]).astype(bool).iloc[0]) is True

    def test_event_counts_survive_the_cast(self, typed: dict[str, pd.DataFrame]) -> None:
        os_records = typed["survival"][typed["survival"]["survivalType"] == OS_ENDPOINT]
        assert int(os_records["eventOccurred"].sum()) == 2

    def test_time_sorts_numerically_not_lexicographically(
        self, typed: dict[str, pd.DataFrame]
    ) -> None:
        """String `"9"` sorts before `"1000"`; integer 9 does not."""
        times = typed["survival"].set_index("survivalId")["timeToEventDays"]
        assert times["V2"] < times["V1"]
        assert sorted(["1000", "9"]) == ["1000", "9"]  # the string ordering being avoided

    def test_survival_properties_are_typed(self, typed: dict[str, pd.DataFrame]) -> None:
        frame = typed["survival"]
        assert frame["timeToEventDays"].dtype == "Int64"
        assert frame["eventOccurred"].dtype == "boolean"

    def test_unexpected_event_value_raises(self) -> None:
        """A third value must fail loudly rather than become NA or True."""
        rows = fixtures.raw_survival()
        rows[0]["eventOccurred"] = "Yes"
        with pytest.raises(ValueError, match="unexpected value"):
            cast.cast_survival(pd.DataFrame(rows))

    def test_unparseable_time_raises_rather_than_coercing(self) -> None:
        rows = fixtures.raw_survival()
        rows[0]["timeToEventDays"] = "unknown-duration"
        with pytest.raises(ValueError, match="not integers"):
            cast.cast_survival(pd.DataFrame(rows))


class TestMissingness:
    """Both representations of "not recorded" occur for the same property."""

    def test_both_null_and_string_none_become_na(self, typed: dict[str, pd.DataFrame]) -> None:
        """S2 carries a true null race and the literal string "None" as ethnicity.

        The live instance stored these fields as `"None"` when #4 was written
        and as true nulls now, so a reader handling one is wrong about the other.
        """
        subjects = typed["subjects"].set_index("subjectId")
        assert pd.isna(subjects.loc["S2", "race"])
        assert pd.isna(subjects.loc["S2", "ethnicity"])
        assert pd.isna(subjects.loc["S3", "ethnicity"])

    def test_missing_tokens_do_not_survive_as_categories(
        self, typed: dict[str, pd.DataFrame]
    ) -> None:
        for name, frame in typed.items():
            guards.check_missing_tokens(frame, where=name)

    def test_guard_catches_a_surviving_token(self) -> None:
        frame = pd.DataFrame({"race": pd.Series(["white", "None"], dtype="string")})
        with pytest.raises(AssertionError, match="missing tokens survived"):
            guards.check_missing_tokens(frame, where="synthetic")

    def test_age_sentinel_is_not_an_age(self, typed: dict[str, pd.DataFrame]) -> None:
        """`-1` means not recorded, not a newborn."""
        subjects = typed["subjects"].set_index("subjectId")
        assert pd.isna(subjects.loc["S3", "ageAtIndexYears"])


class TestAgeUnits:
    """The single most likely quiet-wrong-answer bug in the feature contract (#4 §9)."""

    def test_both_properties_are_in_range(self, typed: dict[str, pd.DataFrame]) -> None:
        guards.check_age_units(typed["subjects"], typed["diagnoses"])

    def test_index_age_is_years_undivided(self, typed: dict[str, pd.DataFrame]) -> None:
        """`ageAtIndexDays` holds years. Dividing by 365 is the bug."""
        ages = typed["subjects"]["ageAtIndexYears"].dropna()
        assert 14 <= float(ages.max()) <= 120

    def test_diagnosis_age_is_days_and_gets_divided(self, typed: dict[str, pd.DataFrame]) -> None:
        """`ageAtDiagnosisDays` genuinely is days, so 22,280 becomes ~61 years."""
        ages = typed["diagnoses"].set_index("diagnosisId")["ageAtDiagnosisYears"].astype(float)
        row = ages.loc["D1"]
        assert row == pytest.approx(22280 / 365.25, rel=1e-6)
        assert 60 < row < 62

    def test_swapped_units_fail_loudly(self, typed: dict[str, pd.DataFrame]) -> None:
        """A schema fix that renamed the property into being correct must fail here."""
        swapped = typed["subjects"].copy()
        swapped["ageAtIndexYears"] = swapped["ageAtIndexYears"] * 365.25
        with pytest.raises(AssertionError, match="no longer holds years"):
            guards.check_age_units(swapped, typed["diagnoses"])


class TestPrimaryDiagnosis:
    """`isPrimaryDiagnosis` is the schema's only edge property, and it is a string."""

    def test_string_true_casts_to_boolean(self, typed: dict[str, pd.DataFrame]) -> None:
        flags = typed["diagnoses"].set_index("diagnosisId")["isPrimaryDiagnosis"]
        assert flags["D1"] is True or bool(flags["D1"])
        assert not bool(flags["D4"])
        assert pd.isna(flags["D5"])

    def test_boolean_comparison_would_match_nothing(self) -> None:
        """The Cypher trap, restated as the property that makes it silent.

        `edge.isPrimaryDiagnosis = true` matched zero rows on the live graph —
        verified — and raised nothing, so it reads as "no primary diagnoses
        exist" rather than as an error.
        """
        assert "True" != True  # noqa: E712 — the point is that these are different values
        assert cast.PRIMARY_VALUES["True"] is True

    def test_at_most_one_primary_per_subject(self, typed: dict[str, pd.DataFrame]) -> None:
        counts = guards.check_one_primary_diagnosis(typed["diagnoses"])
        assert counts.max() <= 1
        assert counts["S4"] == 0  # no primary flag at all; the fallback covers it

    def test_two_primaries_fail_loudly(self, typed: dict[str, pd.DataFrame]) -> None:
        """More than one primary makes the reduction order-dependent."""
        doubled = typed["diagnoses"].copy()
        doubled.loc[doubled["diagnosisId"] == "D4", "isPrimaryDiagnosis"] = True
        with pytest.raises(AssertionError, match="more than one primary"):
            guards.check_one_primary_diagnosis(doubled)


class TestDiagnosisReduction:
    """The primary-with-fallback rule (#4 §1)."""

    def test_fallback_recovers_stage_from_a_secondary(
        self, typed: dict[str, pd.DataFrame]
    ) -> None:
        """S3 reproduces the SKCM pattern: primary has no stage, a secondary does.

        Under primary-*strict* this Subject loses stage, and on the live cohort
        that costs 337 of TCGA-SKCM's 426 Subjects (79%) — the map's bimodality
        trap reappearing inside the cohort built to prevent it.
        """
        reduced = cast.reduce_diagnoses(typed["diagnoses"]).set_index("subjectId")
        assert reduced.loc["S3", "pathologicStageRaw"] == "Stage IIIB"
        assert reduced.loc["S3", "stageOrdinal"] == 3

    def test_primary_value_wins_over_secondary(self, typed: dict[str, pd.DataFrame]) -> None:
        reduced = cast.reduce_diagnoses(typed["diagnoses"]).set_index("subjectId")
        assert reduced.loc["S3", "conditionSubtype"] == "Melanoma"  # from the primary D3

    def test_one_row_per_subject(self, typed: dict[str, pd.DataFrame]) -> None:
        reduced = cast.reduce_diagnoses(typed["diagnoses"])
        assert len(reduced) == reduced["subjectId"].nunique()

    def test_subject_without_a_primary_still_gets_a_row(
        self, typed: dict[str, pd.DataFrame]
    ) -> None:
        """3 cohort Subjects have no primary Diagnosis (#4 §8)."""
        reduced = cast.reduce_diagnoses(typed["diagnoses"]).set_index("subjectId")
        assert "S4" in reduced.index
        assert not reduced.loc["S4", "hasPrimaryDiagnosis"]
        assert reduced.loc["S4", "stageOrdinal"] == 1  # still filled from its own Diagnosis

    def test_a_subject_with_no_diagnosis_has_no_row(
        self, typed: dict[str, pd.DataFrame]
    ) -> None:
        """One cohort Subject has no Diagnosis at all, so the table is one row short.

        Pinned because it dictates the join direction: an inner join onto this
        table silently drops that Subject, and the three models would then no
        longer train on an identical set — which is the one property the whole
        comparison rests on.
        """
        without = typed["diagnoses"][typed["diagnoses"]["subjectId"] != "S1"]
        reduced = cast.reduce_diagnoses(without)
        assert "S1" not in set(reduced["subjectId"])

        roster = pd.DataFrame({"subjectId": ["S1", "S2", "S3", "S4"]})
        joined = roster.merge(reduced, on="subjectId", how="left")
        assert len(joined) == 4
        assert pd.isna(joined.set_index("subjectId").loc["S1", "stageOrdinal"])

    def test_secondary_diagnoses_carry_no_condition(self, typed: dict[str, pd.DataFrame]) -> None:
        """The "40% have no OF_CONDITION" figure is entirely secondary diagnoses."""
        frame = typed["diagnoses"]
        secondary = frame[frame["isPrimaryDiagnosis"] == False]  # noqa: E712
        assert secondary["conditionId"].isna().all()
        primary = frame[frame["isPrimaryDiagnosis"] == True]  # noqa: E712
        assert primary["conditionId"].notna().all()


class TestStageEncoding:
    """Ordinal I–IV, settled by measurement in #4 §3."""

    @pytest.mark.parametrize(
        ("raw_stage", "expected"),
        [
            ("Stage I", 1),
            ("Stage IA", 1),
            ("Stage IB", 1),
            ("Stage IS", 1),
            ("Stage II", 2),
            ("Stage IIA", 2),
            ("Stage IIC", 2),
            ("Stage III", 3),
            ("Stage IIIC", 3),
            ("Stage IV", 4),
            ("Stage IVC", 4),
            ("Stage 0", 0),
            ("Stage 0a", 0),
            ("Stage 0is", 0),
        ],
    )
    def test_substages_collapse_to_the_roman_numeral(self, raw_stage: str, expected: int) -> None:
        assert cast.stage_ordinal(pd.Series([raw_stage])).iloc[0] == expected

    def test_stage_x_becomes_missing(self) -> None:
        """"Cannot be assessed" has no defensible rank."""
        assert pd.isna(cast.stage_ordinal(pd.Series(["Stage X"])).iloc[0])

    def test_iv_is_not_read_as_i(self) -> None:
        """The roman numerals must be matched longest-first."""
        assert cast.stage_ordinal(pd.Series(["Stage IV"])).iloc[0] == 4
        assert cast.stage_ordinal(pd.Series(["Stage III"])).iloc[0] == 3

    def test_unseen_substage_collapses_rather_than_vanishing(self) -> None:
        """The GDC dictionary permits levels the cohort does not currently carry."""
        for unseen, expected in [("Stage IB1", 1), ("Stage IIIC2", 3), ("Stage IIA1", 2)]:
            assert cast.stage_ordinal(pd.Series([unseen])).iloc[0] == expected

    def test_raw_string_is_kept_beside_the_ordinal(self, typed: dict[str, pd.DataFrame]) -> None:
        """So revisiting the collapse is a one-line change, not a re-extract."""
        frame = typed["diagnoses"].set_index("diagnosisId")
        assert frame.loc["D1", "pathologicStageRaw"] == "Stage IIA"
        assert frame.loc["D1", "stageOrdinal"] == 2

    def test_ordinal_sorts_clinically_not_alphabetically(self) -> None:
        stages = cast.stage_ordinal(pd.Series(["Stage IV", "Stage I", "Stage II"]))
        assert list(stages.sort_values()) == [1, 2, 4]
        assert sorted(["Stage IV", "Stage I"]) == ["Stage I", "Stage IV"]


class TestLeakageGuards:
    """The two assertions the ticket names, as tests rather than comments."""

    def test_survival_never_reaches_the_feature_side(self) -> None:
        for column in ["timeToEventDays", "eventOccurred", "survivalType", "durationDays", "event"]:
            with pytest.raises(guards.LeakageError, match="Survival-derived"):
                guards.check_no_survival_columns([column, "age"], where="synthetic")

    def test_survival_guard_catches_one_hot_expansions(self) -> None:
        with pytest.raises(guards.LeakageError):
            guards.check_no_survival_columns(["eventOccurred_1", "age"], where="synthetic")

    def test_survival_guard_catches_a_later_endpoint(self) -> None:
        """A PFI/DSS ticket produces `PFI.time`; the guard must not stop applying."""
        with pytest.raises(guards.LeakageError):
            guards.check_no_survival_columns(["PFI.time"], where="synthetic")

    def test_study_is_a_split_key_only(self) -> None:
        for column in ["studyId", "study", "studyAbbreviation"]:
            with pytest.raises(guards.LeakageError, match="Study reached"):
                guards.check_study_is_split_key_only([column, "age"], where="synthetic")

    def test_study_guard_catches_one_hot_expansions(self) -> None:
        with pytest.raises(guards.LeakageError):
            guards.check_study_is_split_key_only(["studyId_TCGA-BRCA"], where="synthetic")

    def test_immortal_time_columns_are_refused(self) -> None:
        """#4 §4: `daysToCollection` correlates with follow-up at r = 0.711."""
        for column in ["daysToCollection", "startDay", "nInterventions"]:
            with pytest.raises(guards.LeakageError, match="immortal-time"):
                guards.check_no_leaking_columns([column, "age"], where="synthetic")

    def test_a_clean_feature_frame_passes(self) -> None:
        clean = pd.DataFrame(
            {"subjectId": ["S1"], "stageOrdinal": [2], "ageAtDiagnosisYears": [61.0]}
        )
        guards.check_feature_frame(clean, where="synthetic")


class TestSampleRules:
    """Post-baseline types and the redundant class field (#4 §4)."""

    def test_post_baseline_types_are_refused_on_the_feature_side(
        self, typed: dict[str, pd.DataFrame]
    ) -> None:
        with pytest.raises(guards.LeakageError, match="post-baseline"):
            guards.check_no_post_baseline_samples(typed["samples"], where="synthetic")

    def test_baseline_only_passes(self, typed: dict[str, pd.DataFrame]) -> None:
        baseline = typed["samples"][typed["samples"]["sampleType"].isin(BASELINE_SAMPLE_TYPES)]
        guards.check_no_post_baseline_samples(baseline, where="synthetic")

    def test_the_two_type_sets_are_disjoint(self) -> None:
        assert not set(BASELINE_SAMPLE_TYPES) & POST_BASELINE_SAMPLE_TYPES

    def test_sample_class_is_a_function_of_sample_type(
        self, typed: dict[str, pd.DataFrame]
    ) -> None:
        guards.check_sample_class_is_redundant(typed["samples"])

    def test_sample_class_carrying_information_fails(
        self, typed: dict[str, pd.DataFrame]
    ) -> None:
        """If the redundancy stops holding, dropping the field stops being free."""
        drifted = typed["samples"].copy()
        # M1 and M7 are both Primary Tumor; splitting their class breaks the map.
        drifted.loc[drifted["sampleId"] == "M7", "sampleClass"] = "Normal"
        with pytest.raises(AssertionError, match="no longer determined"):
            guards.check_sample_class_is_redundant(drifted)


class TestCohortRule:
    """The eligibility rule, in Python so it is testable without the instance."""

    def test_endpoint_and_landmark_filters(self, typed: dict[str, pd.DataFrame]) -> None:
        selected = cohort.select_os_cohort(typed["survival"])
        assert set(selected["survivalId"]) == {"V1", "V2", "V4"}
        assert "V5" not in set(selected["survivalId"])  # PFI, wrong endpoint
        assert "V3" not in set(selected["survivalId"])  # day 0, excluded

    def test_day_zero_is_excluded(self, typed: dict[str, pd.DataFrame]) -> None:
        """A Subject censored at t=0 contributes zero comparable pairs (#3 §2)."""
        selected = cohort.select_os_cohort(typed["survival"])
        assert (selected["timeToEventDays"] > 0).all()

    def test_studies_outside_the_pinned_list_are_dropped(
        self, typed: dict[str, pd.DataFrame]
    ) -> None:
        extended = typed["survival"].copy()
        extended.loc[extended["survivalId"] == "V1", "studyId"] = "TCGA-GBM"
        selected = cohort.select_os_cohort(extended)
        assert "V1" not in set(selected["survivalId"])

    def test_locked_counts_are_asserted_not_updated(self) -> None:
        wrong = cohort.CohortCounts(n_subjects=6_884, n_events=2_097, n_studies=20)
        with pytest.raises(cohort.CohortMismatchError, match="expected"):
            cohort.assert_locked(wrong)

    def test_the_locked_figures_are_the_amended_ones(self) -> None:
        """#3 §2 amended the map's 6,884 / 2,097 by excluding 73 day-0 records."""
        assert cohort.EXPECTED_SUBJECTS == 6_811
        assert cohort.EXPECTED_EVENTS == 2_091
        assert (
            cohort.EXPECTED_SUBJECTS_BEFORE_LANDMARK - cohort.EXPECTED_SUBJECTS == 73
        )
        assert cohort.EXPECTED_EVENTS_BEFORE_LANDMARK - cohort.EXPECTED_EVENTS == 6


class TestPinnedStudyList:
    """The threshold trap that motivates pinning the members (#4)."""

    def test_twenty_studies(self) -> None:
        assert len(COHORT_STUDIES) == cohort.EXPECTED_STUDIES == 20

    def test_hnsc_is_a_member(self) -> None:
        """Its staging coverage is 0.8598, so a literal `>= 0.86` filter drops it.

        That would silently yield 19 Studies and lose 528 Subjects / 219 events.
        """
        assert "TCGA-HNSC" in COHORT_STUDIES

    def test_unstaged_studies_are_absent(self) -> None:
        """The 13 Studies at 0–2.3% staging coverage; admitting any is a cancer-type proxy."""
        for study in ["TCGA-GBM", "TCGA-OV", "TCGA-LAML", "TCGA-LGG", "TCGA-PRAD", "TCGA-UCEC"]:
            assert study not in COHORT_STUDIES

    def test_no_duplicates(self) -> None:
        assert len(set(COHORT_STUDIES)) == len(COHORT_STUDIES)


class TestQueriesDoNotCast:
    """Casts belong in `cast.py`, where they are tested without the instance."""

    def test_no_pull_casts_in_cypher(self) -> None:
        for name, query in PULLS.items():
            for forbidden in ("toInteger", "toFloat", "toBoolean"):
                assert forbidden not in query, f"{name} casts in Cypher; cast in cast.py instead"

    def test_the_endpoint_property_is_survival_type(self) -> None:
        """`endpoint` does not exist on the node and returns null for all 37,063 records."""
        assert OS_ENDPOINT == "OS"
        assert "survivalType" in PULLS["survival"]
        assert "record.endpoint" not in PULLS["survival"]

    def test_dropped_node_types_are_not_pulled(self) -> None:
        """Intervention and PhenotypeObservation are excluded by decision (#4 §4, §5)."""
        for label in DROPPED_NODE_TYPES:
            for query in PULLS.values():
                assert f":{label}" not in query

    def test_index_age_is_aliased_away_from_its_property_name(self) -> None:
        """The pull renames it so no downstream reader trusts "Days"."""
        assert "AS ageAtIndexYears" in PULLS["subjects"]


class TestStoreRoundTrip:
    """The format exists to stop a round-trip from re-inferring types."""

    def test_raw_round_trips_verbatim(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        rows = fixtures.raw_survival()
        store.write_raw("survival", rows, directory=tmp_path)
        back = store.read_raw("survival", directory=tmp_path)
        assert back["eventOccurred"].tolist() == [row["eventOccurred"] for row in rows]
        assert back["timeToEventDays"].iloc[0] == "1000"  # still a string
        assert isinstance(back["timeToEventDays"].iloc[0], str)

    def test_typed_round_trips_with_dtypes(self, tmp_path, typed) -> None:  # type: ignore[no-untyped-def]
        store.write_table("survival", typed["survival"], directory=tmp_path)
        back = store.read_table("survival", directory=tmp_path)
        assert back["timeToEventDays"].dtype == "Int64"
        assert back["eventOccurred"].dtype == "boolean"
        pd.testing.assert_frame_equal(back, typed["survival"])

    def test_schema_drift_is_caught_at_read_time(self, tmp_path, typed) -> None:  # type: ignore[no-untyped-def]
        store.write_table("survival", typed["survival"], directory=tmp_path)
        schema_path = tmp_path / "survival.schema.json"
        schema = json.loads(schema_path.read_text())
        schema["columns"]["notAColumn"] = "Int64"
        schema_path.write_text(json.dumps(schema))
        with pytest.raises(ValueError, match="absent from the data"):
            store.read_table("survival", directory=tmp_path)

    def test_row_count_drift_is_caught(self, tmp_path, typed) -> None:  # type: ignore[no-untyped-def]
        store.write_table("survival", typed["survival"], directory=tmp_path)
        schema_path = tmp_path / "survival.schema.json"
        schema = json.loads(schema_path.read_text())
        schema["n_rows"] = 999
        schema_path.write_text(json.dumps(schema))
        with pytest.raises(ValueError, match="records 999 rows"):
            store.read_table("survival", directory=tmp_path)
