"""End-to-end mechanics test for model 3 (#12).

Not a result test — a small, fast, fully synthetic cohort exercising the whole
two-stage pipeline (`fit_feature_contract` -> construction cache -> collation ->
encoder training with early stopping -> penalty selection -> shared decoder ->
`score_fold`) so a wiring bug between any two of those pieces fails here rather
than an hour into a real 5-fold run. `tests/test_tabular.py` is the same test
for model 2, deliberately: the two models must break in the same places.

The encoder's own properties get their own class, because three of them are
architectural commitments rather than incidental — `L` equals the subgraph
diameter, the residual is a bare identity, and every relation's convolution is
bias-free so an empty relation-sum contributes exactly zero.
"""

from __future__ import annotations

import json

import fixtures
import pandas as pd
import pytest
import torch
from fixtures import MANY_DIAGNOSES
from torch_geometric.data import Batch

from gl_lifesphere.constructions import cache
from gl_lifesphere.constructions import subject_subgraph as ss
from gl_lifesphere.evaluation.splits import FoldSplit
from gl_lifesphere.features.contract import fit_feature_contract
from gl_lifesphere.models import training
from gl_lifesphere.models.graph import (
    GraphModelConfig,
    RelationalLayer,
    SubjectSubgraphEncoder,
    run_fold,
)
from gl_lifesphere.survival.targets import SurvivalTarget

N_PER_STUDY = 20


@pytest.fixture
def synthetic_cohort() -> tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit]:
    return fixtures.synthetic_model_inputs(N_PER_STUDY)


def _config(**overrides: object) -> GraphModelConfig:
    defaults: dict[str, object] = {"d": 8, "max_epochs": 15, "patience": 5, "seed": 0}
    return GraphModelConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestRunFold:
    def test_runs_end_to_end_and_returns_populated_metrics(
        self,
        synthetic_cohort: tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit],
    ) -> None:
        records, raw, targets, split = synthetic_cohort

        result = run_fold(
            0, records=records, raw=raw, targets=targets, split=split, config=_config(),
            use_cache=False,
        )

        assert result.fold_metrics.n_test == len(split.test)
        assert 0.0 <= result.fold_metrics.pooled_harrell_c <= 1.0
        assert 0.0 <= result.fold_metrics.within_study.cindex <= 1.0
        assert result.fold_metrics.integrated_brier is not None
        assert result.penalty_selection.chosen_penalizer in _config().penalty_grid

    def test_runs_the_two_locked_diagnostics_alongside_the_primary_result(
        self,
        synthetic_cohort: tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit],
    ) -> None:
        """#3 §3 pre-declares two controls as "not as options": a frozen
        randomly-initialised encoder, and the end-to-end score that is a
        by-product of stage one. Both must run on every fold, in every model."""
        records, raw, targets, split = synthetic_cohort

        result = run_fold(
            0, records=records, raw=raw, targets=targets, split=split, config=_config(),
            use_cache=False,
        )

        assert 0.0 <= result.random_init.fold_metrics.pooled_harrell_c <= 1.0
        assert result.random_init.penalty_selection.chosen_penalizer in _config().penalty_grid
        # The end-to-end score bypasses the shared decoder's Breslow baseline
        # hazard entirely, so it never carries a Brier score.
        assert 0.0 <= result.end_to_end.pooled_harrell_c <= 1.0
        assert result.end_to_end.integrated_brier is None

    def test_is_deterministic_given_a_fixed_seed(
        self,
        synthetic_cohort: tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit],
    ) -> None:
        records, raw, targets, split = synthetic_cohort

        first = run_fold(
            0, records=records, raw=raw, targets=targets, split=split, config=_config(),
            use_cache=False,
        )
        second = run_fold(
            0, records=records, raw=raw, targets=targets, split=split, config=_config(),
            use_cache=False,
        )

        assert first.fold_metrics.pooled_harrell_c == pytest.approx(
            second.fold_metrics.pooled_harrell_c
        )
        assert first.random_init.fold_metrics.pooled_harrell_c == pytest.approx(
            second.random_init.fold_metrics.pooled_harrell_c
        )

    def test_result_serialises_to_json_safe_dict(
        self,
        synthetic_cohort: tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit],
    ) -> None:
        records, raw, targets, split = synthetic_cohort

        result = run_fold(
            0, records=records, raw=raw, targets=targets, split=split, config=_config(),
            use_cache=False,
        )
        json.dumps(result.to_dict(), default=str)

    def test_records_the_construction_that_produced_the_metric(
        self,
        synthetic_cohort: tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit],
    ) -> None:
        """The construction's shape has moved three times across #1, #4 and #11,
        so a metric with no record of which shape produced it is not
        reproducible."""
        records, raw, targets, split = synthetic_cohort

        result = run_fold(
            0, records=records, raw=raw, targets=targets, split=split, config=_config(),
            use_cache=False,
        )
        construction = result.to_dict()["construction"]
        assert isinstance(construction, dict)
        assert sorted(construction["node_types"]) == sorted(ss.NODE_TYPES)
        # 5 forward relations and their 5 reverses.
        assert len(construction["relation_types"]) == 10
        assert construction["n_subjects"] == len(records.subject_ids)


class TestRunIdentityIsHonoured:
    """`experiments/README.md` requires a config to name the model, construction,
    endpoint, split and seed, "so a run is only useful if it is pinned down
    enough to line up against the others". Declaring is not honouring: the
    loaders read the frozen cohort and the persisted folds regardless, so
    without a check a config reading `"endpoint": "PFI"` would produce OS
    numbers filed under a PFI label."""

    def test_the_shipped_config_declares_the_run_it_actually_gets(self) -> None:
        from gl_lifesphere.models.graph.__main__ import DEFAULT_CONFIG

        config = GraphModelConfig.from_dict(json.loads(DEFAULT_CONFIG.read_text()))
        training.check_run_identity(
            model=config.model, expected_model="model3_graph",
            endpoint=config.endpoint, split=config.split,
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [("endpoint", "PFI"), ("split", "10fold_random_seed7"), ("model", "model2_tabular")],
    )
    def test_a_config_declaring_something_else_fails_loudly(
        self, field: str, value: str
    ) -> None:
        config = _config(**{field: value})
        with pytest.raises(ValueError, match="declares a run it will not get"):
            training.check_run_identity(
                model=config.model, expected_model="model3_graph",
                endpoint=config.endpoint, split=config.split,
            )


class TestEncoder:
    """The three architectural commitments, asserted rather than assumed."""

    @pytest.fixture
    def batch_and_blocks(
        self,
        synthetic_cohort: tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit],
    ) -> tuple[Batch, ss.FeatureBlocks]:
        records, raw, _, split = synthetic_cohort
        contract = fit_feature_contract(raw, split.train)
        built = cache.build_subgraphs(records, contract)
        return Batch.from_data_list(list(built.graphs)), ss.FeatureBlocks.from_contract(contract)

    def test_readout_is_one_row_per_graph_in_batch_order(
        self, batch_and_blocks: tuple[Batch, ss.FeatureBlocks]
    ) -> None:
        """Root readout needs no index bookkeeping *because* Subject has exactly
        one row per graph after batching (encoder doc §3)."""
        batch, blocks = batch_and_blocks
        encoder = SubjectSubgraphEncoder(blocks.widths, batch.edge_types, d=8, dropout=0.0)

        z = encoder.embed(batch)
        assert z.shape == (batch[ss.SUBJECT].num_nodes, 8)
        assert bool(torch.isfinite(z).all())

    def test_every_relation_convolution_is_bias_free(
        self, batch_and_blocks: tuple[Batch, ss.FeatureBlocks]
    ) -> None:
        """An empty relation-sum must contribute exactly 0. With a bias it would
        contribute a learned constant, making "this Subject has no Condition"
        and "this Subject's Condition embeds to the bias" the same vector
        (encoder doc §2.4)."""
        batch, blocks = batch_and_blocks
        encoder = SubjectSubgraphEncoder(blocks.widths, batch.edge_types, d=8)

        for layer in encoder.layers:
            assert isinstance(layer, RelationalLayer)
            for convolution in layer.convolution.convs.values():
                assert convolution.lin_l.bias is None
                assert not convolution.root_weight

    def test_depth_defaults_to_the_subgraph_diameter(
        self, batch_and_blocks: tuple[Batch, ss.FeatureBlocks]
    ) -> None:
        """`L = 2` is the diameter of the graph the schema produces, not a knob
        — at `L = 1` the root never sees Condition or PathologyDetail."""
        batch, blocks = batch_and_blocks
        encoder = SubjectSubgraphEncoder(blocks.widths, batch.edge_types, d=8)
        assert len(encoder.layers) == ss.SUBGRAPH_RADIUS == 2

    def test_the_residual_carries_the_roots_own_features_to_the_readout(
        self,
        synthetic_cohort: tuple[cache.SubjectRecords, pd.DataFrame, SurvivalTarget, FoldSplit],
    ) -> None:
        """Encoder doc §2.4: without the residual, a node's own features reach
        the readout only via a round trip through its neighbours. Pinned by
        emptying every edge — with the residual the root still produces a
        feature-dependent embedding, and two Subjects differing only in their
        own demographics must not collapse onto the same `z`."""
        records, raw, _, split = synthetic_cohort
        contract = fit_feature_contract(raw, split.train)
        built = cache.build_subgraphs(records, contract)
        blocks = ss.FeatureBlocks.from_contract(contract)

        graphs = [built.graphs[0], built.graphs[built.subject_ids.index(MANY_DIAGNOSES)]]
        batch = Batch.from_data_list(graphs)
        for edge_type in batch.edge_types:
            batch[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)

        encoder = SubjectSubgraphEncoder(blocks.widths, batch.edge_types, d=8, dropout=0.0).eval()
        z = encoder.embed(batch)

        assert bool(torch.isfinite(z).all())
        assert not bool(torch.allclose(z[0], z[1]))
