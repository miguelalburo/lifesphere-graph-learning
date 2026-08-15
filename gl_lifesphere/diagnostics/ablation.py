"""Diagnostic one: retrain model 3 with every `edge_index` emptied (encoder doc §7).

"Train an identical model with all `edge_index` tensors emptied, so every
relation-sum is zero and the root sees only its own demographics. If the full
encoder does not clearly beat this, message passing is contributing nothing."

The ablated model keeps every relation type and therefore every `W_r`, so its
parameter count is identical to the full encoder's (`cache.without_edges`
explains why that matters). What changes is that each `W_r` now aggregates over
an empty neighbourhood and, the convolutions being bias-free, contributes
exactly `0`. The bare-identity residual is the only surviving path to the
readout, which makes this rung **−1** of encoder doc §4's ladder: below even
model 2's flattened MLP, since the root's own feature block is all that reaches
`z_G`.

**What this test does and does not settle.** It is the §7 control, and it
answers "does message passing contribute beyond the root's own features". It is
*not* the test of whether structure beats flattening, and the difference matters
because #12 left that second question open at +0.0180 with an interval crossing
zero. On this construction stage sits on Diagnosis and cancer type on Condition,
so severing the edges takes away the encoder's access to the features
themselves rather than only to their arrangement — the ablated model is rung −1,
below model 2, and beating it is a much weaker claim than beating a flat encoding
of the same data. **Model 2 remains the control for the structural question.**

**Reading a failure.** If the full encoder does not clear the ablation, §7 names
the §2.1 reverse-edge bug as the first suspect — but that suspect is already
ruled out here, because `tests/test_constructions.py` runs the reverse-edge
negative control live (drop them and all 6,811 Subjects strand from the root at
exactly zero). So a flat ablation on this construction is not a wiring bug; it
is `message-passing.md` §9.3's pre-registered prior about the clinical-only
schema, and the honest reading is that there is no structure here to learn.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..constructions import cache
from ..evaluation.compare import PairedDelta, paired_delta
from ..evaluation.splits import N_SPLITS, fold_split, load_folds
from ..features import assemble_raw_frame
from ..models.graph.train import MODEL, GraphModelConfig, FoldResult, run_fold
from ..survival import metrics
from ..survival.targets import SurvivalTarget, load_targets
from . import recorded

METRIC = metrics.HEADLINE


@dataclass(frozen=True)
class AblationResult:
    """The ablated model-3 run, paired fold-for-fold against the recorded full model 3."""

    config: GraphModelConfig
    full_per_fold: tuple[float, ...]
    ablated_per_fold: tuple[float, ...]
    delta: PairedDelta
    # The full fold results behind `ablated_per_fold`, carried for the record
    # file. Empty is legitimate — `gate.assess_ablation` needs the two score
    # vectors and nothing else, so a verdict can be checked without training.
    ablated_folds: tuple[FoldResult, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "diagnostic": "structure_ablation",
            "description": (
                "Model 3 retrained with every edge_index emptied (encoder doc §7). Identical "
                "features, identical relation set and parameter count, identical seed and cached "
                "construction -- the edges are the only difference."
            ),
            "what_this_does_not_test": (
                "Whether structure beats flattening. Stage lives on Diagnosis and cancer type "
                "on Condition, so removing the edges removes the encoder's access to those "
                "features rather than only to their arrangement: the ablated model is rung -1 "
                "of encoder doc §4's ladder, below model 2. Model 2 vs model 3 is the structural "
                "comparison, and it is a different and much closer number."
            ),
            # `experiments/README.md`: a run is only useful if it is pinned down
            # enough to line up against the others, and the model this is paired
            # against is recorded under a different config file.
            "config": recorded.comparable_fields(self.config),
            "metric": METRIC,
            "full_per_fold": list(self.full_per_fold),
            "ablated_per_fold": list(self.ablated_per_fold),
            "full_mean": float(np.mean(self.full_per_fold)),
            "ablated_mean": float(np.mean(self.ablated_per_fold)),
            # full - ablated, so a positive delta means message passing helped.
            "delta_full_minus_ablated": self.delta.to_dict(),
            "reverse_edge_control": (
                "Already green and not re-run here: tests/test_constructions.py asserts that "
                "dropping the reverse edges strands every Subject from the root at exactly zero "
                "(encoder doc §2.1). A flat ablation therefore cannot be attributed to that bug."
            ),
            "folds": [fold.to_dict() for fold in self.ablated_folds],
        }


def run_structure_ablation(
    config: GraphModelConfig,
    *,
    records: cache.SubjectRecords | None = None,
    raw: pd.DataFrame | None = None,
    targets: SurvivalTarget | None = None,
    full_model: recorded.RecordedModel | None = None,
    use_cache: bool = True,
) -> AblationResult:
    """Train the ablated model across all 5 folds and pair it against the recorded full model.

    `config` is model 3's own config with `ablate_edges` forced on, so the two
    sides of the comparison cannot drift: everything the encoder and the
    construction read comes from the same dataclass, and `check_comparable`
    refuses to proceed if the recorded model was run under different settings.
    """
    ablated_config = replace(config, ablate_edges=True)
    model = full_model if full_model is not None else recorded.load_model(MODEL)
    recorded.check_comparable(
        model, recorded.comparable_fields(ablated_config), varies=("ablate_edges",)
    )

    subject_records = records if records is not None else cache.load_subject_records()
    raw_frame = raw if raw is not None else assemble_raw_frame()
    survival_targets = targets if targets is not None else load_targets()
    folds = load_folds()

    ablated = tuple(
        run_fold(
            outer_fold,
            records=subject_records,
            raw=raw_frame,
            targets=survival_targets,
            split=fold_split(folds, outer_fold),
            config=ablated_config,
            use_cache=use_cache,
            # The #3 §6 self-check has to come off, and the reason is the
            # ablation's whole point rather than a concession to it. On this
            # construction the root node carries demographics and Sample-type
            # proportions only: stage sits on Diagnosis, histology and cancer
            # type on Condition. Sever the edges and none of that is reachable,
            # so the ablated encoder scoring at chance is the expected reading
            # of the control -- measured at 0.469 in-sample on fold 0 -- and a
            # check that raises on it would make the diagnostic unrunnable
            # exactly when it has something to report. (This is not the reading
            # for the *full* model, which keeps the check.)
            expect_discrimination=False,
        )
        for outer_fold in range(N_SPLITS)
    )

    full_per_fold = model.metric(METRIC)
    ablated_per_fold = [float(f.fold_metrics.within_study.cindex) for f in ablated]
    return AblationResult(
        config=ablated_config,
        full_per_fold=tuple(full_per_fold),
        ablated_per_fold=tuple(ablated_per_fold),
        delta=paired_delta(full_per_fold, ablated_per_fold),
        ablated_folds=ablated,
    )
