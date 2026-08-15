"""Model 3 — GL/GNN models over the graph constructions.

Rung 3 of encoder doc §4's ablation ladder: the full uncollapsed R-GCN over the
rooted per-Subject subgraph, type-aware input embedding into a shared `d`,
relation-aware message passing at `L = 2`, and root readout `z_G = h_{v_0}^(L)`.
The risk head and survival loss sit downstream in `gl_lifesphere.survival`, and
stage two is literally model 2's code (`survival.two_stage`), so the models differ
in their representation and nothing else.

**The construction is not schema-complete any more, and the shape is the
result.** #1 planned 7 node types and 6 relations; #4 §4–§5 removed
`Intervention` (immortal-time leak in its bare presence) and
`PhenotypeObservation` (featureless cancer-type channel), leaving 5 and 4; #11
split `HAS_DIAGNOSIS` on `isPrimaryDiagnosis`, giving 5 relations before reverse
edges. `Condition` survives, re-keyed on `conditionId` (#4 §2's amendment,
resolved on #12) and flagged as the project's most severe cancer-type channel.

Which encoder is valid depends on the construction's task family — graph-level
readout for per-Subject subgraphs (here), node-level for a patient similarity
network, attention (HGT) at rung 4.
"""

from __future__ import annotations

from .network import RelationalLayer, SubjectSubgraphEncoder
from .train import (
    FoldResult,
    GraphModelConfig,
    SplitBatch,
    collate,
    iter_folds,
    run_fold,
    train_encoder,
)

__all__ = [
    "FoldResult",
    "GraphModelConfig",
    "RelationalLayer",
    "SplitBatch",
    "SubjectSubgraphEncoder",
    "collate",
    "iter_folds",
    "run_fold",
    "train_encoder",
]
