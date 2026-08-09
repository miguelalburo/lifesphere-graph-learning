"""The trust gate's own tests (#13).

The three diagnostics are controls — each builds a model that *should* fail and
checks that it does — so what needs testing here is not "does it produce a
number" but "would it still fail if the thing it guards against were present".
Three properties carry most of that weight:

- the structure ablation differs from the full arm in its **edges and nothing
  else**, including its parameter count, or it measures capacity as well as
  structure;
- the permutation moves `(time, event)` as a pair and leaves every marginal
  intact, or the null it tests is a weaker one a leak could clear;
- the degree probe's own design matrix **fails the project's leakage guard**,
  which is what makes it a probe rather than a fourth arm.

Fast and fully synthetic throughout, like `test_graph.py` and `test_tabular.py`.
Nothing here touches the live instance or the frozen cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import fixtures
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from gl_lifesphere.constructions import cache
from gl_lifesphere.diagnostics import counts, degree_probe, gate, shuffle
from gl_lifesphere.diagnostics.ablation import AblationResult
from gl_lifesphere.evaluation.compare import paired_delta
from gl_lifesphere.evaluation.splits import FoldSplit
from gl_lifesphere.extract import guards
from gl_lifesphere.features.contract import fit_feature_contract
from gl_lifesphere.models.graph import GraphArmConfig, run_fold
from gl_lifesphere.models.graph.network import SubjectSubgraphEncoder
from gl_lifesphere.constructions.subject_subgraph import FeatureBlocks
from gl_lifesphere.survival import metrics
from gl_lifesphere.survival.targets import SurvivalTarget
from gl_lifesphere.survival.two_stage import two_stage_score

N_PER_STUDY = 20

ArmInputs = tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit]


@pytest.fixture
def synthetic_cohort() -> ArmInputs:
    return fixtures.synthetic_arm_inputs(N_PER_STUDY)


@pytest.fixture
def built(synthetic_cohort: ArmInputs) -> tuple[cache.SubgraphSet, FeatureBlocks]:
    records, raw, _, split = synthetic_cohort
    contract = fit_feature_contract(raw, split.train)
    blocks = FeatureBlocks.from_contract(contract)
    return cache.build_subgraphs(records, contract), blocks


def _config(**overrides: object) -> GraphArmConfig:
    defaults: dict[str, object] = {"d": 8, "max_epochs": 15, "patience": 5, "seed": 0}
    return GraphArmConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Diagnostic one: the structure ablation
# ---------------------------------------------------------------------------


class TestWithoutEdges:
    def test_empties_every_edge_index_while_keeping_every_edge_type(
        self, built: tuple[cache.SubgraphSet, FeatureBlocks]
    ) -> None:
        subgraphs, _ = built

        ablated = cache.without_edges(subgraphs)

        assert sum(int(g.num_edges) for g in subgraphs.graphs) > 0
        assert sum(int(g.num_edges) for g in ablated.graphs) == 0
        for before, after in zip(subgraphs.graphs, ablated.graphs, strict=True):
            assert set(after.edge_types) == set(before.edge_types)

    def test_leaves_every_node_feature_untouched(
        self, built: tuple[cache.SubgraphSet, FeatureBlocks]
    ) -> None:
        """Only the edges may differ — a feature that moved would make the
        ablation a second experiment rather than a control."""
        subgraphs, _ = built

        ablated = cache.without_edges(subgraphs)

        for before, after in zip(subgraphs.graphs, ablated.graphs, strict=True):
            assert set(after.node_types) == set(before.node_types)
            for node_type in before.node_types:
                assert torch.equal(after[node_type].x, before[node_type].x)

    def test_does_not_mutate_the_construction_it_was_given(
        self, built: tuple[cache.SubgraphSet, FeatureBlocks]
    ) -> None:
        """The ablated run reads the same cache entry as the real one, so an
        in-place strip would silently ablate the real arm too."""
        subgraphs, _ = built
        before = sum(int(g.num_edges) for g in subgraphs.graphs)

        cache.without_edges(subgraphs)

        assert sum(int(g.num_edges) for g in subgraphs.graphs) == before

    def test_fingerprint_cannot_be_mistaken_for_the_real_construction(
        self, built: tuple[cache.SubgraphSet, FeatureBlocks]
    ) -> None:
        subgraphs, _ = built

        ablated = cache.without_edges(subgraphs)

        assert ablated.fingerprint != subgraphs.fingerprint
        assert subgraphs.fingerprint in ablated.fingerprint
        assert ablated.fingerprint.startswith("edges-ablated:")

    def test_the_encoder_over_it_has_the_identical_parameter_count(
        self, built: tuple[cache.SubgraphSet, FeatureBlocks]
    ) -> None:
        """The whole point of keeping the edge *types*: encoder doc §7 asks for
        "an identical model", and an ablation that also removed each dropped
        relation's `W_r` would be measuring capacity as much as structure."""
        subgraphs, blocks = built
        ablated = cache.without_edges(subgraphs)

        full_encoder = SubjectSubgraphEncoder(
            blocks.widths, subgraphs.graphs[0].edge_types, d=8
        )
        ablated_encoder = SubjectSubgraphEncoder(
            blocks.widths, ablated.graphs[0].edge_types, d=8
        )

        assert sum(p.numel() for p in ablated_encoder.parameters()) == sum(
            p.numel() for p in full_encoder.parameters()
        )

    def test_the_readout_stops_depending_on_the_neighbourhood(
        self, built: tuple[cache.SubgraphSet, FeatureBlocks]
    ) -> None:
        """The behavioural claim §7 actually makes: with every relation-sum
        empty, `z_G` is a function of the root's own features alone. Perturbing
        a non-root node changes the full encoder's readout and must not change
        the ablated one."""
        subgraphs, blocks = built
        encoder = SubjectSubgraphEncoder(blocks.widths, subgraphs.graphs[0].edge_types, d=8).eval()

        original = subgraphs.graphs[0].clone()
        perturbed = subgraphs.graphs[0].clone()
        perturbed["Sample"].x = perturbed["Sample"].x + 7.0

        with torch.no_grad():
            full_before = encoder.embed(Batch.from_data_list([original]))
            full_after = encoder.embed(Batch.from_data_list([perturbed]))
            stripped = cache.without_edges(
                cache.SubgraphSet(
                    subject_ids=("a", "b"),
                    graphs=(original, perturbed),
                    fingerprint="test",
                )
            )
            ablated_before = encoder.embed(Batch.from_data_list([stripped.graphs[0]]))
            ablated_after = encoder.embed(Batch.from_data_list([stripped.graphs[1]]))

        assert not torch.allclose(full_before, full_after)
        assert torch.allclose(ablated_before, ablated_after)


class TestAblatedArmRun:
    def test_records_that_the_edges_were_ablated(self, synthetic_cohort: ArmInputs) -> None:
        """The relation set is unchanged by the ablation, so `relation_types`
        cannot show it — the flag and the edge count are the only evidence a
        reader of a fold file has.

        `expect_discrimination=False` is what the ablation always needs, here
        and on the real cohort. `fixtures.synthetic_survival` builds its entire
        risk from `stageOrdinal`, which lives on the Diagnosis node — and the
        real construction is no kinder, since stage is on Diagnosis and
        histology and cancer type are on Condition, leaving the root with
        demographics and Sample-type proportions alone. Sever the edges and
        none of it is reachable, so the ablated encoder scores at chance
        (measured at 0.469 in-sample on the real fold 0). That is the control
        working, and #3 §6's check would turn it into a crash.
        """
        records, raw, targets, split = synthetic_cohort

        result = run_fold(
            0,
            records=records,
            raw=raw,
            targets=targets,
            split=split,
            config=_config(ablate_edges=True),
            use_cache=False,
            expect_discrimination=False,
        )

        construction = result.to_dict()["construction"]
        assert isinstance(construction, dict)
        assert construction["edges_ablated"] is True
        assert construction["n_edges"] == 0
        assert len(construction["relation_types"]) == 10
        json.dumps(result.to_dict(), default=str)

    def test_the_unablated_run_still_carries_its_edges(
        self, synthetic_cohort: ArmInputs
    ) -> None:
        records, raw, targets, split = synthetic_cohort

        result = run_fold(
            0, records=records, raw=raw, targets=targets, split=split,
            config=_config(), use_cache=False,
        )

        construction = result.to_dict()["construction"]
        assert isinstance(construction, dict)
        assert construction["edges_ablated"] is False
        assert construction["n_edges"] > 0


# ---------------------------------------------------------------------------
# Diagnostic three: the label shuffle
# ---------------------------------------------------------------------------


def _target(n: int = 60, *, seed: int = 0) -> SurvivalTarget:
    rng = np.random.default_rng(seed)
    return SurvivalTarget(
        subject_id=np.array([f"S{i:03d}" for i in range(n)]),
        study=np.array(["A"] * (n // 2) + ["B"] * (n - n // 2)),
        time=rng.uniform(10.0, 4000.0, size=n),
        event=rng.random(n) < 0.4,
    )


class TestPermuteTargets:
    def test_preserves_every_marginal(self) -> None:
        """§7's own justification for the control: permutation destroys the
        association while preserving every marginal, so a model that still
        scores is reading something it should not have."""
        targets = _target()

        permuted = shuffle.permute_targets(targets, seed=0)

        assert sorted(permuted.time) == sorted(targets.time)
        assert permuted.event.sum() == targets.event.sum()

    def test_time_and_event_travel_as_one_pair(self) -> None:
        """Permuting the two independently would also destroy their dependence
        — censoring is not independent of duration — which is a different and
        weaker null than the one §7 specifies."""
        targets = _target()

        permuted = shuffle.permute_targets(targets, seed=0)

        assert sorted(zip(permuted.time, permuted.event)) == sorted(
            zip(targets.time, targets.event)
        )

    def test_leaves_subject_and_study_where_they_were(self) -> None:
        """Only the label moves; the keys every arm zips on must not."""
        targets = _target()

        permuted = shuffle.permute_targets(targets, seed=0)

        assert np.array_equal(permuted.subject_id, targets.subject_id)
        assert np.array_equal(permuted.study, targets.study)

    def test_actually_moves_the_label(self) -> None:
        targets = _target()

        permuted = shuffle.permute_targets(targets, seed=0)

        assert not np.array_equal(permuted.time, targets.time)

    def test_within_study_keeps_every_label_inside_its_own_study(self) -> None:
        """The second scheme's defining property: Study-level survival
        differences survive, so a pooled C above chance under it is the
        Condition~Study channel rather than a leak (#4 §6)."""
        targets = _target()

        permuted = shuffle.permute_targets(targets, seed=0, scheme=shuffle.WITHIN_STUDY)

        for name in np.unique(targets.study):
            mask = targets.study == name
            assert sorted(permuted.time[mask]) == sorted(targets.time[mask])

    def test_global_scheme_moves_labels_between_studies(self) -> None:
        targets = _target()

        permuted = shuffle.permute_targets(targets, seed=0, scheme=shuffle.GLOBAL)

        mask = targets.study == "A"
        assert sorted(permuted.time[mask]) != sorted(targets.time[mask])

    def test_is_deterministic_given_a_seed(self) -> None:
        targets = _target()

        first = shuffle.permute_targets(targets, seed=7)
        second = shuffle.permute_targets(targets, seed=7)

        assert np.array_equal(first.time, second.time)
        assert not np.array_equal(first.time, shuffle.permute_targets(targets, seed=8).time)

    def test_rejects_an_unknown_scheme(self) -> None:
        with pytest.raises(ValueError, match="unknown shuffle scheme"):
            shuffle.permute_targets(_target(), seed=0, scheme="sideways")


class TestSelfCheckUnderAShuffledLabel:
    """#3 §6 has every arm assert its own training-fold risk beats chance,
    because a sign-inverted score silently returns `1 - C`. A permuted label
    inverts that premise, so the shuffle has to be able to switch the check off
    — and nothing else may."""

    @staticmethod
    def _noise(n: int, seed: int) -> tuple[pd.DataFrame, SurvivalTarget]:
        rng = np.random.default_rng(seed)
        target = _target(n, seed=seed)
        z = pd.DataFrame(
            rng.normal(size=(n, 2)), columns=["z0", "z1"], index=list(target.subject_id)
        )
        return z, target

    def test_two_stage_records_the_training_c_it_computed(self) -> None:
        z, target = self._noise(80, seed=1)

        result = two_stage_score(
            z_train=z, train_target=target,
            z_trainval=z, trainval_target=target,
            z_test=z, test_target=target,
            where="test", expect_discrimination=False,
        )

        assert 0.0 <= result.train_harrell_c <= 1.0

    def test_the_self_check_is_opt_out_and_the_default_still_runs_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one thing this flag can break: if the default ever flipped, the
        arms would stop self-checking and a sign-inverted risk would go back to
        silently reporting `1 - C` (#3 §6).

        Asserted on the call rather than on a contrived non-discriminating fit,
        because a penalised Cox scored on its own training fold ranks above 0.5
        for pure noise — there is no honest input that makes the real check
        fire here. `assert_discriminates` raising is already covered directly in
        `test_survival_metrics.py`; what is untested is that `two_stage_score`
        still routes through it.
        """
        z, target = self._noise(80, seed=3)
        calls: list[str] = []
        monkeypatch.setattr(
            "gl_lifesphere.survival.two_stage.metrics.assert_discriminates",
            lambda *a, **k: calls.append(k["where"]) or 0.7,  # type: ignore[func-returns-value]
        )

        def score(expect: bool) -> None:
            two_stage_score(
                z_train=z, train_target=target,
                z_trainval=z, trainval_target=target,
                z_test=z, test_target=target,
                where="test", expect_discrimination=expect,
            )

        score(True)
        assert calls == ["test"]

        score(False)
        assert calls == ["test"]


class TestLabelShuffleRun:
    def test_an_arm_retrains_on_a_permuted_label_without_tripping_the_self_check(
        self, synthetic_cohort: ArmInputs
    ) -> None:
        """The wiring test: the arm's **own** `run_fold` is what runs — a
        shuffle routed through a reimplementation could not detect a leak that
        lived in the arm — and a control that scores at chance completes rather
        than raising."""
        _, raw, targets, _ = synthetic_cohort
        folds = pd.DataFrame(
            {
                "subjectId": targets.subject_id,
                "studyId": targets.study,
                # Round-robin within the Subject order, which is Study-blocked
                # in the synthetic cohort, so every fold sees every Study.
                "fold": np.arange(len(targets)) % 5,
            }
        )

        result = shuffle.run_label_shuffle(
            arms=("arm1_baseline",),
            schemes=(shuffle.GLOBAL,),
            seed=0,
            raw=raw,
            targets=targets,
            folds=folds,
        )

        run = result.run("arm1_baseline", shuffle.GLOBAL)
        assert len(run.fold_metrics) == 5
        assert all(0.0 <= c <= 1.0 for c in run.within_study)
        json.dumps(result.to_dict(), default=str)

    def test_each_scheme_gets_one_run_per_arm(self, synthetic_cohort: ArmInputs) -> None:
        _, raw, targets, _ = synthetic_cohort
        folds = pd.DataFrame(
            {
                "subjectId": targets.subject_id,
                "studyId": targets.study,
                "fold": np.arange(len(targets)) % 5,
            }
        )

        result = shuffle.run_label_shuffle(
            arms=("arm1_baseline",), schemes=shuffle.SCHEMES, seed=0,
            raw=raw, targets=targets, folds=folds,
        )

        assert {(r.arm, r.scheme) for r in result.runs} == {
            ("arm1_baseline", shuffle.GLOBAL),
            ("arm1_baseline", shuffle.WITHIN_STUDY),
        }


# ---------------------------------------------------------------------------
# Diagnostic two: the degree-only probe
# ---------------------------------------------------------------------------


def _intervention_counts(subject_ids: list[str], *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "subjectId": subject_ids,
            "nInterventions": rng.integers(0, 12, size=len(subject_ids)),
        }
    )


class TestAccrualCounts:
    def test_every_roster_subject_gets_a_row_even_with_no_rows_to_count(self) -> None:
        """One cohort Subject has no Diagnosis at all (#4 §8). The probe should
        see `n_diagnoses = 0` for them, not lose the row — `members` is the
        roster and the counts join to it, never the other way round."""
        interim = fixtures.synthetic_interim(N_PER_STUDY)
        roster = [str(s) for s in interim["members"]["subjectId"]]

        accrual = counts.build_accrual_counts(
            members=interim["members"],
            samples=interim["samples"],
            diagnoses=interim["diagnoses"],
            intervention_counts=_intervention_counts(roster),
        )

        assert len(accrual) == len(roster)
        assert sorted(accrual.frame.index) == sorted(roster)
        assert accrual.frame.loc[fixtures.NO_DIAGNOSIS, "nDiagnoses"] == 0

    def test_a_subject_absent_from_the_intervention_pull_counts_zero(self) -> None:
        interim = fixtures.synthetic_interim(N_PER_STUDY)
        roster = [str(s) for s in interim["members"]["subjectId"]]

        accrual = counts.build_accrual_counts(
            members=interim["members"],
            samples=interim["samples"],
            diagnoses=interim["diagnoses"],
            intervention_counts=_intervention_counts(roster[1:]),
        )

        assert accrual.frame.loc[roster[0], "nInterventions"] == 0

    def test_design_is_log1p_of_the_counts(self) -> None:
        interim = fixtures.synthetic_interim(N_PER_STUDY)
        roster = [str(s) for s in interim["members"]["subjectId"]]
        accrual = counts.build_accrual_counts(
            members=interim["members"],
            samples=interim["samples"],
            diagnoses=interim["diagnoses"],
            intervention_counts=_intervention_counts(roster),
        )

        design = accrual.design()

        assert list(design.columns) == list(counts.PROBE_COLUMNS)
        np.testing.assert_allclose(
            design["log1p_nSamples"].to_numpy(),
            np.log1p(accrual.frame["nSamples"].to_numpy(dtype="float64")),
        )

    def test_the_probe_frame_fails_the_projects_own_leakage_guard(self) -> None:
        """This is what makes it a probe and not a fourth arm. #4 §4 excludes
        accrual from every arm, `guards.IMMORTAL_TIME_DERIVED` enforces it, and
        the control is built from exactly the thing the guard forbids — so the
        guard must still recognise these columns."""
        with pytest.raises(guards.LeakageError, match="immortal-time"):
            guards.check_no_leaking_columns(counts.COUNT_COLUMNS, where="degree probe")

    def test_no_arm_carries_the_probes_counts(self, synthetic_cohort: ArmInputs) -> None:
        """The other half of the same claim: the exclusion is real, so the
        shared design matrix every arm reads holds none of these columns."""
        _, raw, _, split = synthetic_cohort
        contract = fit_feature_contract(raw, split.train)

        design = contract.transform(raw, split.train)

        assert not set(design.columns) & set(counts.COUNT_COLUMNS)
        assert not set(design.columns) & set(counts.PROBE_COLUMNS)
        guards.check_feature_frame(design, where="arm design matrix")


class TestDegreeProbe:
    def test_scores_a_fold_and_records_a_hazard_ratio_per_covariate(
        self, synthetic_cohort: ArmInputs
    ) -> None:
        """The coefficients are the diagnosis, not just the C: a protective HR
        on an accrual count is the immortal-time signature even when the
        overall score is weak (#4 §4)."""
        _, raw, targets, split = synthetic_cohort
        roster = [str(s) for s in raw["subjectId"]]
        accrual = counts.build_accrual_counts(
            members=raw[["subjectId", "studyId"]],
            samples=fixtures.synthetic_interim(N_PER_STUDY)["samples"],
            diagnoses=fixtures.synthetic_interim(N_PER_STUDY)["diagnoses"],
            intervention_counts=_intervention_counts(roster),
        )

        result = degree_probe.run_fold(0, counts=accrual, targets=targets, split=split)

        assert 0.0 <= result.fold_metrics.within_study.cindex <= 1.0
        assert 0.0 <= result.train_harrell_c <= 1.0
        assert set(result.hazard_ratios) == set(counts.PROBE_COLUMNS)
        for stats in result.hazard_ratios.values():
            assert stats["hazard_ratio"] > 0.0
            assert 0.0 <= stats["p"] <= 1.0
        json.dumps(result.to_dict(), default=str)

    def test_does_not_raise_when_the_probe_scores_at_chance(
        self, synthetic_cohort: ArmInputs
    ) -> None:
        """A weak probe is a *passing* probe. If this went through
        `assert_discriminates` like an arm does, the expected outcome would
        crash the gate."""
        _, raw, targets, split = synthetic_cohort
        roster = [str(s) for s in raw["subjectId"]]
        rng = np.random.default_rng(0)
        pure_noise = pd.DataFrame(
            {
                "nSamples": rng.integers(0, 5, size=len(roster)),
                "nDiagnoses": rng.integers(0, 3, size=len(roster)),
                "nInterventions": rng.integers(0, 9, size=len(roster)),
            },
            index=pd.Index(roster, name="subjectId"),
        )

        result = degree_probe.run_fold(
            0, counts=counts.AccrualCounts(frame=pure_noise), targets=targets, split=split
        )

        assert result.fold_metrics.within_study.cindex is not None

    def test_the_decomposition_keeps_the_arm3_channel_out_of_the_headline(self) -> None:
        """`has_secondary_diagnosis` is the one accrual channel arm 3 can see,
        and it is fitted so the gate's claim about that exposure is measured
        rather than asserted. It is not one of §7's three covariates, so it must
        never move the headline number."""
        grid = degree_probe.subset_grid()

        assert grid[degree_probe.ALL] == counts.PROBE_COLUMNS
        assert counts.SECONDARY_DIAGNOSIS not in grid[degree_probe.ALL]
        assert grid[f"arm3_visible:{counts.SECONDARY_DIAGNOSIS}"] == (
            counts.SECONDARY_DIAGNOSIS,
        )

    def test_the_arm3_channel_is_the_multiple_diagnosis_bit(self) -> None:
        roster = ["A", "B", "C"]
        frame = pd.DataFrame(
            {"nSamples": [3, 3, 3], "nDiagnoses": [1, 2, 5], "nInterventions": [0, 0, 0]},
            index=pd.Index(roster, name="subjectId"),
        )

        design = counts.AccrualCounts(frame=frame).design(
            columns=(counts.SECONDARY_DIAGNOSIS,)
        )

        assert list(design[counts.SECONDARY_DIAGNOSIS]) == [0.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# The verdicts
# ---------------------------------------------------------------------------


def _fold_metrics(within: float, pooled: float = 0.5) -> metrics.FoldMetrics:
    return metrics.FoldMetrics(
        pooled_harrell_c=pooled,
        within_study=metrics.WithinStudyConcordance(
            cindex=within, excluded_studies=(), per_study={}
        ),
        uno_c=within,
        td_auc={},
        integrated_brier=None,
        n_test=100,
        n_events_test=30,
    )


def _ablation(full: list[float], ablated: list[float]) -> AblationResult:
    return AblationResult(
        config=GraphArmConfig(ablate_edges=True),
        full_per_fold=tuple(full),
        ablated_per_fold=tuple(ablated),
        delta=paired_delta(full, ablated),
    )


def _shuffle_result(means: dict[str, float], scheme: str = shuffle.GLOBAL) -> shuffle.ShuffleResult:
    return shuffle.ShuffleResult(
        runs=tuple(
            shuffle.ArmShuffle(
                arm=arm,
                scheme=scheme,
                seed=0,
                fold_metrics=tuple(_fold_metrics(mean) for _ in range(5)),
            )
            for arm, mean in means.items()
        )
    )


def _probe(mean: float) -> degree_probe.DegreeProbeResult:
    ratios = {
        "log1p_nInterventions": {"coef": -0.3, "hazard_ratio": 0.74, "p": 0.001},
        "log1p_nDiagnoses": {"coef": 1.3, "hazard_ratio": 3.8, "p": 1e-20},
        "log1p_nSamples": {"coef": 0.08, "hazard_ratio": 1.09, "p": 0.4},
    }
    folds = tuple(
        degree_probe.ProbeFold(
            fold=i,
            chosen_penalizer=0.1,
            train_harrell_c=mean,
            hazard_ratios=ratios,
            fold_metrics=_fold_metrics(mean),
        )
        for i in range(5)
    )
    return degree_probe.DegreeProbeResult(by_subset={degree_probe.ALL: folds})


def _inputs(
    *,
    full: list[float] | None = None,
    ablated: list[float] | None = None,
    probe: float = 0.52,
    shuffled: dict[str, float] | None = None,
    scheme: str = shuffle.GLOBAL,
    extra_runs: tuple[shuffle.ArmShuffle, ...] = (),
    arm_means: dict[str, float] | None = None,
) -> gate.GateInputs:
    """A `GateInputs` built by serialising the real result objects.

    Deliberately routed through each result's own `to_dict()` rather than
    hand-written payloads: the gate now reads the recorded artefacts, so the
    coupling worth pinning is that what a diagnostic *writes* is what a verdict
    *reads*. A hand-built dict would keep passing after a key was renamed on
    one side only.
    """
    runs = _shuffle_result(shuffled or {"arm3_graph": 0.50}, scheme=scheme).runs + extra_runs
    return gate.GateInputs(
        ablation=_ablation(full or [0.70] * 5, ablated or [0.60] * 5).to_dict(),
        degree_probe=_probe(probe).to_dict(),
        label_shuffle=shuffle.ShuffleResult(runs=runs).to_dict(),
        arm_means=arm_means or {"arm1_baseline": 0.6556, "arm3_graph": 0.6767},
    )


class TestVerdicts:
    def test_ablation_passes_only_when_the_full_encoder_clears_the_declared_margin(self) -> None:
        config = gate.GateConfig(ablation_margin=0.02)

        clear = gate.assess_ablation(_inputs(full=[0.70] * 5, ablated=[0.60] * 5), config)
        marginal = gate.assess_ablation(_inputs(full=[0.68] * 5, ablated=[0.67] * 5), config)

        assert clear.verdict == gate.PASS
        assert marginal.verdict == gate.FAIL

    def test_a_passing_ablation_says_what_it_does_not_establish(self) -> None:
        """Severing the edges removes the encoder's access to stage and cancer
        type, which live on non-root nodes — so beating the ablation is a much
        weaker claim than beating a flat encoding of the same features. Read as
        "structure beats flattening" it would overturn #12's own conclusion."""
        verdict = gate.assess_ablation(_inputs(full=[0.70] * 5, ablated=[0.50] * 5), gate.GateConfig())

        assert verdict.verdict == gate.PASS
        assert "does not establish that structure beats flattening" in verdict.reading
        assert "arm 2" in verdict.reading

    def test_a_flat_ablation_does_not_blame_the_reverse_edges(self) -> None:
        """§7 names the §2.1 reverse-edge bug as the first suspect, but
        `tests/test_constructions.py` already rules it out live — so the
        failure reading has to point at the data, not send someone hunting a
        bug that is covered."""
        verdict = gate.assess_ablation(
            _inputs(full=[0.68] * 5, ablated=[0.68] * 5), gate.GateConfig()
        )

        assert verdict.verdict == gate.FAIL
        assert "reverse-edge" in verdict.reading
        assert "already ruled out" in verdict.reading

    def test_degree_probe_passes_when_accrual_stays_weak(self) -> None:
        config = gate.GateConfig(degree_probe_ceiling=0.55)

        weak = gate.assess_degree_probe(_inputs(probe=0.52), config)
        strong = gate.assess_degree_probe(_inputs(probe=0.61), config)

        assert weak.verdict == gate.PASS
        assert strong.verdict == gate.FAIL
        assert "immortal-time" in strong.reading

    def test_degree_probe_separates_protective_counts_from_harmful_ones(self) -> None:
        """The two directions have opposite implications: protective is
        immortal time and is a defect, harmful is disease burden and merely
        raises the floor. A verdict that lumped them together would send
        someone hunting a bias that is not there."""
        verdict = gate.assess_degree_probe(_inputs(probe=0.60), gate.GateConfig())

        assert verdict.detail["protective_covariates_p_lt_0.05"] == ["log1p_nInterventions"]
        assert verdict.detail["harmful_covariates_p_lt_0.05"] == ["log1p_nDiagnoses"]

    def test_degree_probe_records_which_arms_clear_its_floor(self) -> None:
        """§7: "this probe's C-index is the floor a structural result has to
        clear to be interesting"."""
        verdict = gate.assess_degree_probe(
            _inputs(probe=0.66, arm_means={"arm1_baseline": 0.6556, "arm3_graph": 0.6767}),
            gate.GateConfig(degree_probe_ceiling=0.70),
        )

        assert verdict.detail["arms_clearing_the_floor"] == ["arm3_graph"]

    def test_shuffle_passes_only_when_every_arm_sits_at_chance(self) -> None:
        config = gate.GateConfig(shuffle_tolerance=0.05)

        at_chance = gate.assess_label_shuffle(
            _inputs(shuffled={"arm1_baseline": 0.501, "arm3_graph": 0.497}), config
        )
        leaking = gate.assess_label_shuffle(
            _inputs(shuffled={"arm1_baseline": 0.502, "arm3_graph": 0.61}), config
        )

        assert at_chance.verdict == gate.PASS
        assert leaking.verdict == gate.FAIL
        assert leaking.detail["worst_arm"] == "arm3_graph"
        assert "SURVIVAL_DERIVED" in leaking.reading

    def test_shuffle_is_judged_only_on_the_gating_scheme(self) -> None:
        """`within_study` is expected to leave a pooled C above chance and is
        recorded rather than gated on — judging it would fail the gate on #4
        §6's immunity argument working correctly."""
        inputs = _inputs(
            shuffled={"arm3_graph": 0.50},
            extra_runs=_shuffle_result(
                {"arm3_graph": 0.62}, scheme=shuffle.WITHIN_STUDY
            ).runs,
        )

        verdict = gate.assess_label_shuffle(
            inputs, gate.GateConfig(shuffle_gate_scheme=shuffle.GLOBAL)
        )

        assert verdict.verdict == gate.PASS

    def test_shuffle_refuses_to_judge_when_the_gating_scheme_was_not_run(self) -> None:
        inputs = _inputs(shuffled={"arm3_graph": 0.50}, scheme=shuffle.WITHIN_STUDY)

        with pytest.raises(ValueError, match="nothing to judge"):
            gate.assess_label_shuffle(
                inputs, gate.GateConfig(shuffle_gate_scheme=shuffle.GLOBAL)
            )

    def test_the_gate_is_the_conjunction_of_its_three_verdicts(self) -> None:
        config = gate.GateConfig()

        report = gate.assess(_inputs(shuffled={"arm3_graph": 0.50}), config)
        assert report.overall == gate.PASS

        failing = gate.assess(_inputs(shuffled={"arm3_graph": 0.62}), config)
        assert failing.overall == gate.FAIL
        json.dumps(failing.to_dict(), default=str)


class TestGateInputsRoundTrip:
    """The gate reads the three artefacts rather than the objects a run is
    holding, so the round trip through disk is part of the contract."""

    def test_judges_a_run_from_its_written_files(self, tmp_path: Path) -> None:
        source = _inputs(full=[0.70] * 5, ablated=[0.50] * 5, probe=0.52)
        for name, payload in (
            ("structure_ablation", source.ablation),
            ("degree_probe", source.degree_probe),
            ("label_shuffle", source.label_shuffle),
        ):
            (tmp_path / f"{name}.json").write_text(json.dumps(payload, default=str))

        report = gate.assess(
            gate.GateInputs.from_files(tmp_path, arm_means=source.arm_means),
            gate.GateConfig(),
        )

        assert report.overall == gate.PASS
        assert [v.verdict for v in report.verdicts] == [gate.PASS] * 3

    def test_says_which_diagnostic_is_missing_and_how_to_produce_it(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="--only ablation"):
            gate.GateInputs.from_files(tmp_path, arm_means={})

    def test_the_commands_it_suggests_are_ones_the_cli_accepts(self) -> None:
        """The `--only` value is spelled out per file rather than derived from
        the filename, because `structure_ablation.json` would otherwise suggest
        `--only structure-ablation`, which argparse rejects — an error message
        whose advice does not run is worse than no advice."""
        from gl_lifesphere.diagnostics.__main__ import ONLY_CHOICES

        assert set(gate.PRODUCED_BY.values()) <= set(ONLY_CHOICES)


class TestGateConfig:
    def test_the_shipped_config_declares_its_thresholds_before_the_run(self) -> None:
        """#13 is "a gate, not a measurement", which only holds if the bar is
        in a file rather than in the code that reads the result."""
        from gl_lifesphere.diagnostics.__main__ import DEFAULT_CONFIG

        payload = json.loads(DEFAULT_CONFIG.read_text())
        config = gate.GateConfig.from_dict(payload)

        assert config.ablation_margin > 0.0
        assert 0.5 < config.degree_probe_ceiling < 1.0
        assert 0.0 < config.shuffle_tolerance < 0.5
        assert config.shuffle_gate_scheme in config.shuffle_schemes
        config.check()

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("endpoint", "PFI", "endpoint"),
            ("split", "10fold_random_seed7", "split"),
            ("shuffle_gate_scheme", "sideways", "nothing would be judged"),
        ],
    )
    def test_a_config_declaring_something_else_fails_loudly(
        self, field: str, value: str, match: str
    ) -> None:
        """The same rule every arm honours (#7, `experiments/README.md`): the
        loaders read the frozen cohort and the persisted folds regardless of
        what a config says, so a gate declaring PFI would pass or fail OS
        results under a PFI label."""
        config = gate.GateConfig(**{field: value})  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=match):
            config.check()

    def test_the_arms_are_built_from_their_own_config_files(self) -> None:
        """The shuffle retrains the arms, so it has to build them exactly as
        their own runs did. Reading `experiments/configs/arm*.json` directly is
        what makes that true by construction — a copy inside the gate's config
        would be a second place to change them, and nothing checks it."""
        from gl_lifesphere.diagnostics.__main__ import DEFAULT_CONFIG

        directory = DEFAULT_CONFIG.parent
        payload = json.loads(DEFAULT_CONFIG.read_text())
        baseline, tabular, graph = gate.GateConfig.from_dict(payload).arm_configs(directory)

        arm1 = json.loads((directory / "arm1_baseline.json").read_text())
        arm2 = json.loads((directory / "arm2_tabular.json").read_text())
        arm3 = json.loads((directory / "arm3_graph.json").read_text())

        assert baseline.seed == arm1["seed"]
        assert tabular.d == arm2["d"]
        assert tabular.hidden_dims == tuple(arm2["hidden_dims"])
        assert graph.d == arm3["d"]
        assert graph.num_layers == arm3["num_layers"]
        assert graph.split_primary_relation == arm3["split_primary_relation"]
        # `run_structure_ablation` is what turns the flag on; the shuffle needs
        # the real arm, so the arm's own config must not pre-ablate.
        assert graph.ablate_edges is False

    def test_the_gate_config_holds_no_second_copy_of_the_arms(self) -> None:
        """The drift this design exists to prevent: a duplicated `arms` block
        would be a second source of truth for hyperparameters the shuffle
        rebuilds each arm from."""
        from gl_lifesphere.diagnostics.__main__ import DEFAULT_CONFIG

        payload = json.loads(DEFAULT_CONFIG.read_text())

        assert "arms" not in payload


class TestPairedDelta:
    def test_pairs_by_fold_and_carries_the_caveat_into_the_record(self) -> None:
        """The interval is anti-conservative by design (overlapping training
        folds), so the warning has to travel with the number rather than live
        in a docstring a reader of `gate.json` never opens."""
        delta = paired_delta([0.70, 0.72, 0.68], [0.60, 0.64, 0.66])

        assert delta.mean == pytest.approx(0.0666667, abs=1e-6)
        assert delta.folds_won == 3
        assert "anti-conservative" in delta.to_dict()["caveat"]  # type: ignore[operator]

    def test_refuses_to_pair_runs_scored_on_different_fold_counts(self) -> None:
        with pytest.raises(ValueError, match="same folds"):
            paired_delta([0.7, 0.6], [0.5])
