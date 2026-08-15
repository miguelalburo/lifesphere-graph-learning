"""Rung 3 of the encoder doc §4 ladder: the full uncollapsed R-GCN, root readout.

Three maps, in the encoder doc's own order:

1. **Type-aware input embedding** (§1) — `h_v^(0) = f_τ(v)(x_v)`, one `Linear`
   per node type, because a Sample's `sampleType` and a Diagnosis's stage are
   not the same column and there is no shared space they arrive in. The shared
   output width `d` is forced rather than stylistic: a Subject sums messages
   from Sample *and* Diagnosis neighbours and adds its own residual, so all
   three must live in one space to be added.
2. **Relation-aware message passing** (§2) —
   `h_v^(l+1) = σ(h_v^(l) + Σ_r mean_{u ∈ N_r(v)} W_r^(l) h_u^(l))`, per-type
   LayerNorm and dropout after each layer, `L = 2`.
3. **Root readout** (§3) — `z_G = h_{v_0}^(L)`. The graph is rooted, so reading
   the root's vector is permutation-invariant by role rather than by pooling,
   and after PyG batches `HeteroData` the `Subject` store holds exactly one row
   per graph in batch order: the readout is an indexing-free `h['Subject']`.

Four choices here are load-bearing and none of them is a default.

**`L = 2` is the diameter, not a hyperparameter.** Every node in a per-Subject
subgraph is within 2 hops of the root, so `L = 1` never reaches Condition or
PathologyDetail and `L ≥ 3` adds no new nodes while adding over-smoothing.
Tune `d` instead (§2.2).

**The residual is a bare identity, not a learned self-loop.** Schlichtkrull's
R-GCN writes `W_0 h_v`; the identity is the stricter choice, and it is chosen
because it *guarantees* the root's own demographics survive to the readout,
where a learned `W_0` could zero itself. At ~2,091 events, prefer the guarantee
(§2.4).

**Every relation's convolution is bias-free.** An empty relation-sum must
contribute exactly `0` — with a bias it would contribute a learned constant, so
"this Subject has no Condition" and "this Subject's Condition embeds to the
bias" would be the same vector (§2.4).

**Aggregation is mean, and no degree feature is offered.** Mean makes Sample
count invisible, which is the intended protection: accrual is partly a marker
of having survived long enough to accrue, and the locked scope for #1 takes the
no-degree-features option. Sum aggregation would smuggle the count back in
implicitly and make #13's ablation impossible to run cleanly (§2.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from ...constructions.subject_subgraph import SUBGRAPH_RADIUS, SUBJECT

EdgeType = tuple[str, str, str]


class RelationalLayer(nn.Module):
    """One application of `h_v <- dropout(LayerNorm(σ(h_v + Σ_r mean_r W_r h_u)))`.

    The whole update rule lives in one module so the encoder is a list of these
    rather than three parallel lists that have to be zipped correctly.
    """

    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[EdgeType],
        *,
        d: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # `aggr="sum"` across relations is the `Σ_r` of the update rule;
        # `aggr="mean"` inside each `SAGEConv` is the `1/|N_r(v)|`. Separate
        # `W_r` per relation — including each reverse relation, since "this
        # Sample belongs to that Subject" and "this Subject has that Sample"
        # are genuinely different messages (§2.1).
        self.convolution = HeteroConv(
            {
                edge_type: SAGEConv(d, d, aggr="mean", root_weight=False, bias=False)
                for edge_type in edge_types
            },
            aggr="sum",
        )
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(d) for node_type in node_types})
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        messages = self.convolution(h, edge_index_dict)
        updated: dict[str, torch.Tensor] = {}
        for node_type, current in h.items():
            # A node type that received nothing contributes exactly zero, which
            # is what makes structural absence and a zero-valued message
            # indistinguishable — stated here because it is a property of the
            # encoder, not an oversight (§2.4).
            incoming = messages.get(node_type)
            total = current if incoming is None else current + incoming
            updated[node_type] = self.dropout(self.norms[node_type](torch.relu(total)))
        return updated


class SubjectSubgraphEncoder(nn.Module):
    """`HeteroData` (batched) -> `z ∈ R^d` per Subject, plus the training head.

    Mirrors `models.tabular.network.TabularEncoder`'s contract exactly —
    `embed()` returns the representation the shared decoder is refit on once
    frozen, `forward()` returns the scalar risk that trains it — so model 2 and
    model 3 differ in their representation and in nothing else.

    `feature_widths` and `edge_types` come from the construction rather than
    from a config: the per-type input widths are the fitted contract's blocks
    (`FeatureBlocks.widths`), which vary by fold as vocabularies do, and the
    relation set depends on whether `HAS_DIAGNOSIS` was split.
    """

    def __init__(
        self,
        feature_widths: Mapping[str, int],
        edge_types: Sequence[EdgeType],
        *,
        d: int = 32,
        num_layers: int = SUBGRAPH_RADIUS,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.d = d
        self.node_types = tuple(sorted(feature_widths))
        self.edge_types = tuple(edge_types)

        self.input_projections = nn.ModuleDict(
            {node_type: nn.Linear(width, d) for node_type, width in sorted(feature_widths.items())}
        )

        self.layers = nn.ModuleList(
            RelationalLayer(self.node_types, self.edge_types, d=d, dropout=dropout)
            for _ in range(num_layers)
        )

        # No bias: an intercept is unidentifiable under the Cox partial
        # likelihood, so a trainable one would drift under weight decay alone
        # and mean nothing when read (decoder doc §4.2).
        self.head = nn.Linear(d, 1, bias=False)

    def embed(self, batch: HeteroData) -> torch.Tensor:
        """`z_G` for every graph in `batch`, as a `B x d` matrix in batch order."""
        h = {
            node_type: self.input_projections[node_type](batch[node_type].x)
            for node_type in self.node_types
        }

        for layer in self.layers:
            h = layer(h, batch.edge_index_dict)

        # Rootedness is why this needs no index bookkeeping: after PyG batches
        # `HeteroData`, the `Subject` store holds exactly one row per graph, in
        # batch order (encoder doc §3).
        return h[SUBJECT]

    def forward(self, batch: HeteroData) -> torch.Tensor:
        """`r`, the scalar risk score used only to train the encoder (stage one)."""
        return self.head(self.embed(batch)).squeeze(-1)
