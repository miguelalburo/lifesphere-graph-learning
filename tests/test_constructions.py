"""The rooted subgraph construction and its cache (#11's assertions, #12's cache).

#12 asks that `check_subgraph` *run here* rather than be reimplemented — it is
already known to fail correctly, and a check that only ever passes is not
evidence of anything. So `TestReverseEdgesAreLoadBearing` runs the encoder doc
§2.1 bug on purpose and pins that the checks catch it.

The fixture is a synthetic cohort shaped like `data/interim/`, pushed through
the *same* `cast.reduce_diagnoses` -> `features.raw.build_raw_frame` ->
`fit_feature_contract` path the flattened arm uses. That matters more than it
looks: the whole three-arm design rests on the two arms sharing one feature
contract, so the graph arm's tests must not fit their features by a private
route.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import fixtures
import pandas as pd
import pytest
from fixtures import MANY_DIAGNOSES, NO_CONDITION, NO_DIAGNOSIS

from gl_lifesphere.constructions import cache
from gl_lifesphere.constructions import subject_subgraph as ss
from gl_lifesphere.extract.cast import reduce_diagnoses
from gl_lifesphere.features.contract import FeatureContract, fit_feature_contract
from gl_lifesphere.features.raw import build_raw_frame


@pytest.fixture
def interim() -> dict[str, pd.DataFrame]:
    """A synthetic `data/interim/`, dtypes and all."""
    return fixtures.synthetic_interim()


@pytest.fixture
def records(interim: dict[str, pd.DataFrame]) -> cache.SubjectRecords:
    return cache.build_subject_records(
        members=interim["members"],
        subjects=interim["subjects"],
        diagnoses=interim["diagnoses"],
        samples=interim["samples"],
        pathology=interim["pathology_details"],
    )


def _contract(interim: dict[str, pd.DataFrame], train: frozenset[str]) -> FeatureContract:
    """Fit through the flattened arm's own path, never a private one."""
    raw = build_raw_frame(
        members=interim["members"],
        subjects=interim["subjects"],
        diagnosis_primary=reduce_diagnoses(interim["diagnoses"]),
        samples=interim["samples"],
    )
    return fit_feature_contract(raw, train)


@pytest.fixture
def contract(interim: dict[str, pd.DataFrame]) -> FeatureContract:
    return _contract(interim, frozenset(interim["members"]["subjectId"]))


class TestSubjectRecords:
    def test_a_subject_with_no_diagnosis_still_gets_a_record(
        self, records: cache.SubjectRecords
    ) -> None:
        """#4 §8: one cohort Subject has no Diagnosis at all. Iterating from the
        Diagnosis table instead of the roster would drop them silently, and the
        arms would no longer train on an identical Subject set."""
        assert NO_DIAGNOSIS in records.subject_ids
        record = records.record(NO_DIAGNOSIS)
        assert len(record.diagnoses) == 0
        # An empty *slice*, so the column set and dtypes still hold.
        assert "isPrimaryDiagnosis" in record.diagnoses.columns

    def test_a_subject_missing_from_the_subjects_table_fails_loudly(
        self, interim: dict[str, pd.DataFrame]
    ) -> None:
        with pytest.raises(ValueError, match="no row in the subjects table"):
            cache.build_subject_records(
                members=interim["members"],
                subjects=interim["subjects"].iloc[1:],
                diagnoses=interim["diagnoses"],
                samples=interim["samples"],
                pathology=interim["pathology_details"],
            )


class TestCheckSubgraph:
    """#11's assertions, re-run here rather than reimplemented."""

    def test_every_check_passes_for_every_subject(
        self, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        failures: dict[str, list[str]] = {}
        for subject_id in records.subject_ids:
            record = records.record(subject_id)
            data = ss.build_subgraph(record, contract)
            failed = [c.name for c in ss.check_subgraph(data, record, contract) if not c.passed]
            if failed:
                failures[subject_id] = failed
        assert not failures

    def test_the_root_is_node_zero_of_subject_and_there_is_exactly_one(
        self, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        """Root readout is `h_dict['Subject']` with no index bookkeeping only
        because Subject has exactly one row per graph (encoder doc §3)."""
        for subject_id in (NO_DIAGNOSIS, MANY_DIAGNOSES):
            data = ss.build_subgraph(records.record(subject_id), contract)
            assert data[ss.SUBJECT].num_nodes == 1
            assert data[ss.SUBJECT].subject_id == [subject_id]

    def test_a_diagnosis_with_no_condition_yields_no_condition_node(
        self, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        data = ss.build_subgraph(records.record(NO_CONDITION), contract)
        assert data[ss.CONDITION].num_nodes == 0
        assert data[ss.OF_CONDITION].edge_index.numel() == 0


class TestReverseEdgesAreLoadBearing:
    """Encoder doc §2.1, run as a negative control rather than asserted.

    Every relation in the schema points away from the Subject, so a subgraph
    built with Neo4j's directions alone leaves the root receiving nothing and
    the encoder silently reduces to two demographic fields. #11 verified this
    fails for all 6,811 Subjects; the point of keeping it is that a check which
    cannot be made to fail is not evidence that it passes.
    """

    def test_dropping_them_strands_every_node_from_the_root(
        self, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        record = records.record(MANY_DIAGNOSES)
        data = ss.build_subgraph(record, contract, add_reverse_edges=False)

        reached = ss.messages_reaching_root(data)
        assert bool(reached[ss.SUBJECT][0])  # the root reaches itself, and nothing else
        assert not any(bool(mask.any()) for t, mask in reached.items() if t != ss.SUBJECT)

    def test_check_subgraph_reports_it(
        self, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        record = records.record(MANY_DIAGNOSES)
        data = ss.build_subgraph(record, contract, add_reverse_edges=False)

        checks = ss.check_subgraph(data, record, contract, add_reverse_edges=False)
        failed = [c.name for c in checks if not c.passed]
        assert [c for c in failed if "reaches the root" in c]


class TestConditionVocabulary:
    """#4 §2's amendment, kept as a live guard rather than a historical note."""

    def test_conditionid_distinguishes_every_condition(
        self, interim: dict[str, pd.DataFrame]
    ) -> None:
        checks = ss.check_condition_vocabulary(interim["diagnoses"])
        assert all(check.passed for check in checks)

    def test_conditionname_does_not_and_the_check_says_so(
        self, interim: dict[str, pd.DataFrame]
    ) -> None:
        """The fixture reproduces the live collapse: distinct `conditionId`,
        one shared `conditionName`. This is the failure that moved the feature
        onto the id, so it must still be reachable."""
        checks = ss.check_condition_vocabulary(
            interim["diagnoses"], feature_key="conditionName"
        )
        assert not any(check.passed for check in checks)
        assert "collapse" in checks[0].detail


class TestCache:
    def test_round_trips_and_reuses_the_entry(
        self, tmp_path: Path, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        first = cache.load_or_build(0, records, contract, directory=tmp_path)
        assert cache.cache_path(0, directory=tmp_path).exists()

        second = cache.load_or_build(0, records, contract, directory=tmp_path)
        assert second.fingerprint == first.fingerprint
        assert second.subject_ids == first.subject_ids
        assert len(second) == len(records.subject_ids)
        for built, read_back in zip(first.graphs, second.graphs):
            assert built[ss.SUBJECT].x.equal(read_back[ss.SUBJECT].x)
            assert built[ss.PROVIDED_SAMPLE].edge_index.equal(read_back[ss.PROVIDED_SAMPLE].edge_index)

    def test_a_contract_fitted_on_different_subjects_misses(
        self, tmp_path: Path, interim: dict[str, pd.DataFrame], records: cache.SubjectRecords
    ) -> None:
        """The stale read that matters: the column names are identical across
        folds, so only the *fitted statistics* distinguish fold 0's features
        from fold 3's. A cache keyed on names alone would train fold 3 on fold
        0's imputation values and standardisation constants."""
        everyone = frozenset(interim["members"]["subjectId"])
        half = frozenset(sorted(everyone)[:30])

        fold_0 = cache.load_or_build(0, records, _contract(interim, everyone), directory=tmp_path)
        fold_1 = cache.load_or_build(0, records, _contract(interim, half), directory=tmp_path)

        assert fold_0.fingerprint != fold_1.fingerprint
        # Compared on a Subject that actually has Diagnosis rows — SUBJ-000 has
        # none, so its feature matrix is empty under either contract.
        row = fold_0.subject_ids.index(MANY_DIAGNOSES)
        assert not fold_0.graphs[row][ss.DIAGNOSIS].x.equal(fold_1.graphs[row][ss.DIAGNOSIS].x)

    def test_changing_a_build_option_misses(
        self, tmp_path: Path, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        split = cache.load_or_build(0, records, contract, directory=tmp_path)
        unsplit = cache.load_or_build(
            0,
            records,
            contract,
            options=cache.BuildOptions(split_primary_relation=False),
            directory=tmp_path,
        )
        assert split.fingerprint != unsplit.fingerprint
        assert ss.HAS_PRIMARY_DIAGNOSIS in split.graphs[0].edge_types
        assert ss.HAS_DIAGNOSIS in unsplit.graphs[0].edge_types

    def test_a_corrupt_entry_is_a_miss_not_a_crash(
        self, tmp_path: Path, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        target = cache.cache_path(0, directory=tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not a torch archive")

        rebuilt = cache.load_or_build(0, records, contract, directory=tmp_path)
        assert len(rebuilt) == len(records.subject_ids)

    def test_select_preserves_order_and_returns_the_ids_alongside(
        self, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        built = cache.build_subgraphs(records, contract)
        wanted = frozenset({MANY_DIAGNOSES, NO_DIAGNOSIS})

        graphs, ids = built.select(wanted)
        assert ids == sorted(wanted)
        assert len(graphs) == 2

    def test_select_fails_on_an_unknown_subject(
        self, records: cache.SubjectRecords, contract: FeatureContract
    ) -> None:
        built = cache.build_subgraphs(records, contract)
        with pytest.raises(KeyError, match="not in this construction"):
            built.select(frozenset({"SUBJ-999"}))


class TestFingerprintIsStableAcrossProcesses:
    """The cache key must not depend on Python's per-process hash randomisation.

    `FeatureContract` carries a `frozenset` (race's fixed fold), so a `repr`-based
    fingerprint changes between runs of identical code on identical data. That is
    the quiet half of a cache bug: nothing raises, every number stays correct, and
    the cache simply never hits — which is invisible unless it is asserted.
    """

    def _fingerprint_under(self, seed: str, tmp_path: Path) -> str:
        script = tmp_path / "fingerprint.py"
        script.write_text(
            "import fixtures\n"
            "from gl_lifesphere.constructions import cache\n"
            "from gl_lifesphere.extract.cast import reduce_diagnoses\n"
            "from gl_lifesphere.features.contract import fit_feature_contract\n"
            "from gl_lifesphere.features.raw import build_raw_frame\n"
            "interim = fixtures.synthetic_interim()\n"
            "records = cache.build_subject_records(\n"
            "    members=interim['members'], subjects=interim['subjects'],\n"
            "    diagnoses=interim['diagnoses'], samples=interim['samples'],\n"
            "    pathology=interim['pathology_details'])\n"
            "raw = build_raw_frame(members=interim['members'], subjects=interim['subjects'],\n"
            "    diagnosis_primary=reduce_diagnoses(interim['diagnoses']), samples=interim['samples'])\n"
            "contract = fit_feature_contract(raw, frozenset(interim['members']['subjectId']))\n"
            "print(cache.fingerprint(records, contract, cache.BuildOptions()))\n"
        )
        # The script runs from tmp_path, so sys.path[0] is not the repo — both
        # the package and the `fixtures` module have to be put on the path.
        tests_dir = Path(__file__).parent
        environment = {
            **os.environ,
            "PYTHONHASHSEED": seed,
            "PYTHONPATH": os.pathsep.join([str(tests_dir.parent), str(tests_dir)]),
        }
        completed = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True,
            env=environment, cwd=Path(__file__).parent.parent, check=True,
        )
        return completed.stdout.strip()

    def test_hash_randomisation_does_not_change_the_key(self, tmp_path: Path) -> None:
        seeds = ["1", "3", "12345"]
        digests = {seed: self._fingerprint_under(seed, tmp_path) for seed in seeds}
        assert len(set(digests.values())) == 1, (
            f"fingerprint depends on PYTHONHASHSEED: {digests}"
        )
