"""#15's follow-up probe: is model 3's margin the secondary-diagnosis channel?

#13's degree probe found a channel model 3 uniquely holds. #11 split
`HAS_DIAGNOSIS` on `isPrimaryDiagnosis`, and a bias-free convolution over an
empty relation contributes exactly `0` while a non-empty one does not — so
"this Subject has at least one secondary Diagnosis" is structurally visible to
the encoder, a 1-bit read on `n_diagnoses > 1` worth within-Study C = 0.5588 on
its own. This probe collapses the split and re-runs model 3 against it.

Three properties carry the weight, and two of them are the probe's own honesty
rather than its arithmetic:

- collapsing really does remove the **clean presence/absence bit** — under one
  relation nothing is empty for exactly the single-Diagnosis Subjects;
- and it really does **not** remove multiplicity, which is why a null bounds
  the effect rather than eliminating it. #15 states that caveat in prose; here
  it is a test, so a later change to the construction that quietly made the
  collapse total would fail rather than leave the caveat over-cautious;
- the probe cannot move #13's PASS/FAIL. It is a follow-up the gate raised, not
  a fourth gate item, and a follow-up that could rewrite the verdict it came
  from would be the worst of both.

Fast and fully synthetic, like `test_diagnostics.py`. Nothing here touches the
live instance or the frozen cohort.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import fixtures
import pandas as pd
import pytest

from gl_lifesphere.constructions import cache
from gl_lifesphere.constructions.subject_subgraph import (
    DIAGNOSIS,
    HAS_DIAGNOSIS,
    HAS_PRIMARY_DIAGNOSIS,
    HAS_SECONDARY_DIAGNOSIS,
)
from gl_lifesphere.diagnostics import gate, relation_split
from gl_lifesphere.diagnostics.recorded import RecordedModel, comparable_fields
from gl_lifesphere.evaluation.splits import FoldSplit, N_SPLITS
from gl_lifesphere.features.contract import fit_feature_contract
from gl_lifesphere.models.graph import GraphModelConfig
from gl_lifesphere.survival.targets import SurvivalTarget

ModelInputs = tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit]


@pytest.fixture
def synthetic_cohort() -> ModelInputs:
    return fixtures.synthetic_model_inputs(20)


@pytest.fixture
def both_constructions(
    synthetic_cohort: ModelInputs,
) -> tuple[cache.SubgraphSet, cache.SubgraphSet]:
    """The same Subjects and the same fitted contract, split and collapsed.

    Built from one contract rather than two fits, because the probe's whole
    claim is that the relation split is the only difference between the two
    runs — a fixture that refitted would let a vocabulary drift in here too.
    """
    records, raw, _, split = synthetic_cohort
    contract = fit_feature_contract(raw, split.train)
    return (
        cache.build_subgraphs(
            records, contract, options=cache.BuildOptions(split_primary_relation=True)
        ),
        cache.build_subgraphs(
            records, contract, options=cache.BuildOptions(split_primary_relation=False)
        ),
    )


def _by_subject(subgraphs: cache.SubgraphSet, edge_type: tuple[str, str, str]) -> dict[str, int]:
    """Each Subject's edge count on one relation."""
    return {
        subject: int(graph[edge_type].edge_index.shape[1])
        for graph, subject in zip(subgraphs.graphs, subgraphs.subject_ids)
    }


# ---------------------------------------------------------------------------
# What the collapse does and does not remove
# ---------------------------------------------------------------------------


class TestWhatCollapsingChanges:
    def test_the_split_makes_multi_diagnosis_readable_as_an_empty_relation(
        self, both_constructions: tuple[cache.SubgraphSet, cache.SubgraphSet]
    ) -> None:
        """The channel #13 identified, demonstrated rather than assumed.

        Among the Subjects that have a Diagnosis at all,
        `HAS_SECONDARY_DIAGNOSIS` is empty for exactly the single-Diagnosis
        ones — so its emptiness *is* `n_diagnoses > 1`, and the convolutions
        being bias-free, an empty relation contributes exactly 0 where a
        non-empty one does not.
        """
        split, unsplit = both_constructions

        secondary = _by_subject(split, HAS_SECONDARY_DIAGNOSIS)
        total = _by_subject(unsplit, HAS_DIAGNOSIS)

        empty_secondary = {s for s, n in secondary.items() if n == 0}
        with_a_diagnosis = {s for s, n in total.items() if n >= 1}
        single_diagnosis = {s for s, n in total.items() if n == 1}

        assert empty_secondary & with_a_diagnosis == single_diagnosis
        # Both classes must be present or the assertion above is vacuous.
        assert 0 < len(single_diagnosis) < len(with_a_diagnosis)

    def test_collapsing_leaves_emptiness_reading_only_no_diagnosis_at_all(
        self, both_constructions: tuple[cache.SubgraphSet, cache.SubgraphSet]
    ) -> None:
        """Under one relation, emptiness no longer separates single- from
        multi-Diagnosis Subjects.

        What survives is a strictly smaller bit: a Subject with **no** Diagnosis
        still has an empty relation. That is not the channel #13 measured — the
        frozen cohort holds exactly one such Subject in 6,811 (#4 §8, which is
        why every model left-joins from the `members` roster), and one Subject
        carries no prognostic weight. Pinned here so the distinction is a test
        rather than an assumption.
        """
        _, unsplit = both_constructions

        assert HAS_PRIMARY_DIAGNOSIS not in unsplit.graphs[0].edge_types
        assert HAS_SECONDARY_DIAGNOSIS not in unsplit.graphs[0].edge_types

        total = _by_subject(unsplit, HAS_DIAGNOSIS)
        assert {s for s, n in total.items() if n == 0} == {fixtures.NO_DIAGNOSIS}

    def test_collapsing_does_not_remove_multiplicity(
        self, both_constructions: tuple[cache.SubgraphSet, cache.SubgraphSet]
    ) -> None:
        """#15's caveat, as a test.

        "A Subject with 3 Diagnoses still mean-aggregates to a different value
        than one with 1." If that stopped being true the probe would become a
        clean elimination rather than a bound, and its reading would have to
        change — so the caveat is pinned here rather than only asserted in the
        result file.
        """
        split, unsplit = both_constructions

        counts = _by_subject(unsplit, HAS_DIAGNOSIS)
        assert len(set(counts.values())) > 1

        # And the aggregate the root sees is genuinely moved by the extra rows:
        # the mean over a multi-Diagnosis Subject's Diagnosis block differs from
        # its primary Diagnosis alone, which is the residual channel the collapse
        # leaves behind.
        multi = next(s for s, n in counts.items() if n > 1)
        index = unsplit.subject_ids.index(multi)
        rows = unsplit.graphs[index][DIAGNOSIS].x
        primary_row = split.graphs[split.subject_ids.index(multi)][DIAGNOSIS].x[
            int(split.graphs[split.subject_ids.index(multi)][HAS_PRIMARY_DIAGNOSIS].edge_index[1, 0])
        ]
        assert not bool((rows.mean(dim=0) == primary_row).all())


# ---------------------------------------------------------------------------
# The guard: one field varies, and nothing else
# ---------------------------------------------------------------------------


def _recorded(model: str, per_fold: list[float], config: dict[str, object]) -> RecordedModel:
    return RecordedModel(
        model=model,
        folds=tuple(
            {"fold": i, "metrics": {relation_split.METRIC: value}}
            for i, value in enumerate(per_fold)
        ),
        config=config,
    )


class TestComparabilityGuard:
    def test_the_probe_config_is_model_3_with_only_the_split_flag_flipped(self) -> None:
        graph_config = GraphModelConfig(d=8)

        probe = relation_split.probe_config(graph_config)

        assert probe.split_primary_relation is False
        assert graph_config.split_primary_relation is True
        varied = {
            name
            for name, value in comparable_fields(probe).items()
            if comparable_fields(graph_config)[name] != value
        }
        assert varied == {"split_primary_relation"}

    def test_a_recorded_model_3_trained_at_a_different_width_is_refused(self) -> None:
        """A silent second difference is the failure this guard exists for: the
        probe would produce a plausible number attributing a width change to the
        relation split."""
        graph_config = GraphModelConfig(d=8)
        model3 = _recorded("model3_graph", [0.68] * N_SPLITS, {**comparable_fields(graph_config), "d": 32})

        with pytest.raises(ValueError, match="split_primary_relation"):
            relation_split.check_probe_comparable(model3, graph_config)

    def test_a_probe_config_declaring_a_seed_the_model_does_not_supply_is_refused(self) -> None:
        """`seed` and `construction` are declared in the probe's config but
        *supplied* by model 3's, so a probe file naming seed 1 against a model run
        at seed 0 would file one run's numbers under another's label —
        `tests/README.md`'s quiet-wrong-answer class."""
        config = relation_split.RelationSplitConfig(seed=1)

        with pytest.raises(ValueError, match="seed"):
            config.check_against(GraphModelConfig(seed=0))

    def test_a_probe_config_declaring_the_wrong_model_is_refused(self) -> None:
        """Delegated to `training.check_run_identity` rather than re-implemented,
        so there is one place this rule lives rather than three."""
        config = relation_split.RelationSplitConfig(model="model2_tabular")

        with pytest.raises(ValueError, match="model"):
            config.check_against(GraphModelConfig())

    def test_the_comparison_model_must_be_scored_on_the_same_endpoint_and_split(self) -> None:
        """Model 2 and model 3 are different model classes, so field-by-field
        comparison is meaningless — but pairing their folds is only meaningful
        on one endpoint and one split, and that much is checkable."""
        graph_config = GraphModelConfig(d=8)
        model2 = _recorded(
            "model2_tabular",
            [0.65] * N_SPLITS,
            {"model": "model2_tabular", "endpoint": "PFI", "split": graph_config.split},
        )

        with pytest.raises(ValueError, match="endpoint"):
            relation_split.check_comparison_model(model2, graph_config)


# ---------------------------------------------------------------------------
# The attribution arithmetic and the reading it produces
# ---------------------------------------------------------------------------


def _assess(
    split: list[float], unsplit: list[float], comparison: list[float]
) -> relation_split.Attribution:
    return relation_split.assess_attribution(
        split_per_fold=split,
        unsplit_per_fold=unsplit,
        comparison_per_fold=comparison,
        config=relation_split.RelationSplitConfig(),
    )


class TestAttribution:
    def test_the_margin_decomposes_exactly_fold_by_fold(self) -> None:
        """`(split - reference) - (unsplit - reference) == split - unsplit`,
        which is what makes `attributed_share` a share of something rather than
        a ratio of two independently estimated numbers."""
        split = [0.68, 0.71, 0.66, 0.69, 0.67]
        unsplit = [0.66, 0.70, 0.65, 0.68, 0.67]
        comparison = [0.65, 0.69, 0.63, 0.64, 0.68]

        attribution = _assess(split, unsplit, comparison)

        for cost, wide, narrow in zip(
            attribution.cost_of_collapsing.per_fold,
            attribution.split_margin.per_fold,
            attribution.unsplit_margin.per_fold,
        ):
            assert cost == pytest.approx(wide - narrow)

    def test_a_consistent_cost_attributes_the_margin_to_the_channel(self) -> None:
        split = [0.68, 0.71, 0.66, 0.69, 0.67]
        unsplit = [0.66, 0.69, 0.64, 0.67, 0.65]
        comparison = [0.65, 0.69, 0.63, 0.64, 0.68]

        attribution = _assess(split, unsplit, comparison)

        assert attribution.verdict == relation_split.ATTRIBUTED
        assert attribution.attributed_share is not None
        assert attribution.attributed_share > 0.0
        assert attribution.bound is None

    def test_a_null_reports_a_bound_rather_than_claiming_elimination(self) -> None:
        """#15: "A null result therefore bounds the effect rather than
        eliminating it." The bound is the upper end of the paired interval, and
        it has to be in the artefact — a reader of the result file must not have
        to remember the ticket to know a null is not a proof."""
        split = [0.68, 0.71, 0.66, 0.69, 0.67]
        unsplit = [0.69, 0.70, 0.665, 0.68, 0.675]
        comparison = [0.65, 0.69, 0.63, 0.64, 0.68]

        attribution = _assess(split, unsplit, comparison)

        assert attribution.verdict == relation_split.NOT_ATTRIBUTED
        assert attribution.bound == pytest.approx(attribution.cost_of_collapsing.ci95[1])
        assert "bound" in attribution.reading.lower()

    def test_a_collapse_that_helps_is_reported_as_its_own_outcome(self) -> None:
        """Not folded into the null. Collapsing helping would contradict #11's
        own measurement, so it needs a reading that says so rather than a
        "not attributed" that reads as reassurance."""
        split = [0.66, 0.69, 0.64, 0.67, 0.65]
        unsplit = [0.68, 0.71, 0.66, 0.69, 0.67]
        comparison = [0.65, 0.69, 0.63, 0.64, 0.68]

        attribution = _assess(split, unsplit, comparison)

        assert attribution.verdict == relation_split.REVERSED

    def test_a_margin_below_the_resolution_limit_is_flagged_in_the_share(self) -> None:
        """The quantity being decomposed is +0.0180 against #3 §5's ~0.02, so
        the share is a share of something the design cannot resolve. Reporting
        the number without that caveat would be the more misleading half."""
        split = [0.68, 0.71, 0.66, 0.69, 0.67]
        unsplit = [0.66, 0.69, 0.64, 0.67, 0.65]
        comparison = [x - 0.01 for x in split]

        attribution = _assess(split, unsplit, comparison)

        assert attribution.margin_below_resolution_limit is True
        assert "resolution" in json.dumps(attribution.to_dict()).lower()

    def test_no_share_is_reported_when_there_was_no_margin_to_explain(self) -> None:
        """Dividing by a margin at or below zero would produce a percentage that
        reads as meaningful and is not."""
        split = [0.65, 0.68, 0.63, 0.66, 0.64]
        unsplit = [0.64, 0.67, 0.62, 0.65, 0.63]
        comparison = [0.68, 0.71, 0.66, 0.69, 0.67]

        attribution = _assess(split, unsplit, comparison)

        assert attribution.attributed_share is None

    def test_folds_must_line_up_on_all_three_runs(self) -> None:
        with pytest.raises(ValueError):
            _assess([0.68] * 5, [0.66] * 4, [0.65] * 5)


class TestResultFile:
    def test_carries_the_caveat_and_the_asymmetry_note_for_14(self) -> None:
        """#15: "#14 must state the asymmetry either way." A note that lives only
        in a ticket comment is not carried by the artefact #14 reads."""
        attribution = _assess([0.68] * 5, [0.66, 0.67, 0.65, 0.66, 0.67], [0.65] * 5)

        payload = attribution.to_dict()

        assert relation_split.COLLAPSE_CAVEAT in json.dumps(payload)
        assert relation_split.ASYMMETRY_FOR_14 in json.dumps(payload)

    def test_the_capacity_confound_is_recorded_as_numbers_not_only_prose(self) -> None:
        """Collapsing two relations into one merges their `W_r`, so unlike #13's
        structure ablation this comparison does not hold capacity fixed. Both
        differences push the unsplit run down, so the cost is an upper bound —
        and a reader has to be able to see how large the second difference is.
        """
        result = relation_split.RelationSplitResult(
            attribution=_assess([0.68] * 5, [0.66] * 5, [0.65] * 5),
            unsplit_config=relation_split.probe_config(GraphModelConfig(d=8)),
            split_parameters=(25984,) * 5,
            unsplit_parameters=(21888,) * 5,
        )

        payload = result.to_dict()

        assert payload["capacity_is_not_held_fixed"] == relation_split.CAPACITY_CAVEAT
        assert payload["parameters_per_fold"] == {
            "split": [25984] * 5,
            "unsplit": [21888] * 5,
            "removed": [4096] * 5,
        }

    def test_the_decision_rule_travels_with_the_verdict(self) -> None:
        """`gate.Verdict` carries the rule that produced it; so must this. A
        verdict whose rule lives only in code is one a reader of the artefact
        cannot check — and this rule rests on an interval the project itself
        labels anti-conservative, which the artefact has to say."""
        payload = _assess([0.68] * 5, [0.66] * 5, [0.65] * 5).to_dict()

        rule = payload["rule"]
        assert isinstance(rule, str)
        assert relation_split.ATTRIBUTED in rule
        assert "anti-conservative" in rule

    def test_the_share_is_never_offered_without_its_uncertainty(self) -> None:
        """A bare 37.9% is the number #14 would otherwise quote."""
        payload = _assess([0.68] * 5, [0.66, 0.67, 0.65, 0.66, 0.67], [0.65] * 5).to_dict()

        guidance = payload["how_to_read_the_share"]
        assert isinstance(guidance, str)
        assert "POINT ESTIMATE" in guidance
        assert "cost_of_collapsing" in guidance

    def test_records_the_metric_and_all_three_score_vectors(self) -> None:
        attribution = _assess([0.68] * 5, [0.66] * 5, [0.65] * 5)

        payload = attribution.to_dict()

        assert payload["metric"] == relation_split.METRIC
        assert payload["split_per_fold"] == [0.68] * 5
        assert payload["unsplit_per_fold"] == [0.66] * 5
        assert payload["comparison_per_fold"] == [0.65] * 5


# ---------------------------------------------------------------------------
# The probe is a follow-up, not a fourth gate item
# ---------------------------------------------------------------------------


class TestNotAGateItem:
    def test_the_gate_has_no_input_path_for_this_probe(self, tmp_path: Path) -> None:
        """#13's verdict is over the three §7 diagnostics and nothing else. A
        follow-up able to rewrite the gate it came from would let a later,
        cheaper run overturn a declared FAIL."""
        assert relation_split.RESULT_FILE not in gate.PRODUCED_BY
        (tmp_path / relation_split.RESULT_FILE).write_text("{}")

        # Present in the directory and still not a substitute for anything the
        # gate needs: assembling it demands the three diagnostics as before.
        with pytest.raises(FileNotFoundError, match="structure_ablation.json"):
            gate.GateInputs.from_files(tmp_path, model_means={})

    def test_its_verdict_vocabulary_is_not_the_gates(self) -> None:
        """PASS/FAIL would invite a reader to add this to the gate's tally. This
        probe answers an attribution question; it does not gate anything."""
        vocabulary = {
            relation_split.ATTRIBUTED,
            relation_split.NOT_ATTRIBUTED,
            relation_split.REVERSED,
        }
        assert vocabulary.isdisjoint({gate.PASS, gate.FAIL})
