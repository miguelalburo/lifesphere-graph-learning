"""Two-stage training for arm 3, one outer fold at a time (#12, #3 §3).

Stage one trains `SubjectSubgraphEncoder` end-to-end on the stratified Efron Cox
loss, full-batch, with early stopping on the nested validation slice. Stage two
is `survival.two_stage` — the same code arm 2 calls, not merely the same
protocol — so the only thing that differs between the two arms is the
representation stage one learned from.

**Full-batch, and that is a correctness requirement rather than a convenience.**
The Cox partial likelihood is not a sum over independent rows: each event's term
is normalised over the risk set of everyone still under observation. Mini-batching
would evaluate it against a random subset of that risk set, which is a different
objective, and arm 2 trains on the whole training fold at once. The subgraphs
are small enough to make this free — a training fold is ~4,100 graphs of ~12
nodes, so one `Batch` is ~50k nodes at `d = 32`.

**The construction is cached, not rebuilt per epoch.** `constructions.cache`
builds one `HeteroData` per Subject under this fold's fitted contract; every
epoch then reuses the same three collated batches.

**Honest prior (#12, `message-passing.md` §10).** This arm is expected to be
thin on the clinical-only schema: the Intervention layer that supplied most of
the fan-out was removed by #4 §4 as an immortal-time leak, and what remains is
one set (Samples, spanning ~2 distinct types per Subject) plus a near-1:1
Diagnosis chain. A null result here is a pre-registered outcome, not a failure —
which is why #13's trust gate, not this module, decides whether any number is
believable.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Batch, HeteroData

from ...constructions import cache
from ...constructions.subject_subgraph import FeatureBlocks
from ...evaluation.splits import N_SPLITS, FoldSplit, fold_split, load_folds
from ...features import FeatureContract, assemble_raw_frame, fit_feature_contract
from ...survival import decoder, losses, metrics
from ...survival.targets import SurvivalTarget, load_targets
from ...survival.two_stage import TwoStageResult, two_stage_score
from .network import SubjectSubgraphEncoder


@dataclass(frozen=True)
class GraphArmConfig:
    """Every hyperparameter this module needs; everything else is fixed by #3 and #4.

    `num_layers` is exposed but should stay at 2: it is the diameter of the
    subgraph the schema produces, not a tuning knob (encoder doc §2.2). It is a
    field rather than a constant only so a config file records the value that
    was actually run.

    `split_primary_relation` is the one construction switch worth keeping
    reachable. #11 adopted the split on measurement — under a single relation
    the root mean-aggregates the primary Diagnosis together with 0–4 secondaries
    that carry the training-fold within-Study median stage, shifting the
    Diagnosis-set mean 0.35 sd off the primary's own stage for the 37.3% of
    Subjects with several. If arm 3 underperforms, #11 names this the second
    thing to check after the reverse edges, and checking it means running it.
    """

    d: int = 32
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 500
    patience: int = 30
    seed: int = 0
    split_primary_relation: bool = True
    add_reverse_edges: bool = True
    penalty_grid: tuple[float, ...] = decoder.PENALTY_GRID

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GraphArmConfig":
        """Ignores keys the dataclass does not declare (e.g. a `_comment`), so a
        documented config file need not be stripped before loading."""
        known = {f.name for f in fields(cls)}
        kwargs = {key: value for key, value in payload.items() if key in known}
        if "penalty_grid" in kwargs:
            kwargs["penalty_grid"] = tuple(kwargs["penalty_grid"])  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]

    @property
    def build_options(self) -> cache.BuildOptions:
        return cache.BuildOptions(
            split_primary_relation=self.split_primary_relation,
            add_reverse_edges=self.add_reverse_edges,
        )


@dataclass(frozen=True)
class SplitBatch:
    """One collated `Batch` plus the Subject order it is in, and that order's target.

    The ids travel with the batch because graphs and labels are zipped by
    `subjectId`, never by position (`SurvivalTarget.reorder`). Keeping them in
    one value makes it impossible to collate one split's graphs against
    another's targets.
    """

    batch: Batch
    subject_ids: tuple[str, ...]
    target: SurvivalTarget

    def __len__(self) -> int:
        return len(self.subject_ids)


@dataclass(frozen=True)
class EncoderFit:
    model: SubjectSubgraphEncoder
    train_loss_curve: tuple[float, ...]
    val_loss_curve: tuple[float, ...]
    best_epoch: int


def collate(
    subgraphs: cache.SubgraphSet, subject_ids: frozenset[str], targets: SurvivalTarget
) -> SplitBatch:
    """The graphs for one split, collated once, with their targets re-indexed to match."""
    graphs, ids = subgraphs.select(subject_ids)
    batch = Batch.from_data_list(list(graphs))
    return SplitBatch(batch=batch, subject_ids=tuple(ids), target=targets.reorder(list(ids)))


def _new_encoder(
    reference: HeteroData, blocks: FeatureBlocks, *, config: GraphArmConfig
) -> SubjectSubgraphEncoder:
    """A freshly initialised encoder sized to this fold's construction.

    Both the per-type input widths and the relation set are read off the built
    graphs rather than configured: widths are the fitted contract's blocks and
    move with a fold's vocabulary, and the relation set depends on whether
    `HAS_DIAGNOSIS` was split. Configuring either would let the model and the
    construction disagree silently.
    """
    torch.manual_seed(config.seed)
    return SubjectSubgraphEncoder(
        blocks.widths,
        reference.edge_types,
        d=config.d,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def train_encoder(
    train: SplitBatch,
    val: SplitBatch,
    blocks: FeatureBlocks,
    *,
    config: GraphArmConfig,
) -> EncoderFit:
    """Full-batch Adam on the stratified Efron Cox loss, early-stopped on val loss."""
    model = _new_encoder(train.batch, blocks, config=config)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    train_curve: list[float] = []
    val_curve: list[float] = []

    for epoch in range(config.max_epochs):
        model.train()
        optimiser.zero_grad()
        train_risk = model(train.batch)
        train_loss = losses.stratified_efron_cox_loss(
            train_risk, train.target.time, train.target.event, train.target.study
        )
        train_loss.backward()
        optimiser.step()
        train_curve.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_risk = model(val.batch)
            val_loss = losses.stratified_efron_cox_loss(
                val_risk, val.target.time, val.target.event, val.target.study
            )
        val_curve.append(float(val_loss))

        if val_curve[-1] < best_val:
            best_val = val_curve[-1]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return EncoderFit(
        model=model,
        train_loss_curve=tuple(train_curve),
        val_loss_curve=tuple(val_curve),
        best_epoch=best_epoch,
    )


def _embed(model: SubjectSubgraphEncoder, split: SplitBatch) -> pd.DataFrame:
    """Frozen `z`, as a DataFrame so it plugs straight into `survival.two_stage`."""
    model.eval()
    with torch.no_grad():
        z = model.embed(split.batch).numpy()
    return pd.DataFrame(
        z, columns=[f"z{i}" for i in range(z.shape[1])], index=list(split.subject_ids)
    )


def _end_to_end_score(
    model: SubjectSubgraphEncoder,
    *,
    trainval: SplitBatch,
    test: SplitBatch,
    where: str,
) -> metrics.FoldMetrics:
    """The by-product diagnostic (#3 §3): score with the trained head directly,
    bypassing the shared decoder entirely. No survival function exists here —
    a raw NN head produces a ranking, not a Breslow baseline hazard — so this
    never carries an integrated Brier score."""
    model.eval()
    with torch.no_grad():
        trainval_risk = model(trainval.batch).numpy()
        test_risk = model(test.batch).numpy()

    metrics.assert_discriminates(
        trainval.target.event, trainval.target.time, trainval_risk, where=where
    )

    return metrics.score_fold(
        train_time=trainval.target.time,
        train_event=trainval.target.event,
        test_time=test.target.time,
        test_event=test.target.event,
        test_study=test.target.study,
        risk=test_risk,
    )


@dataclass(frozen=True)
class FoldResult:
    fold: int
    config: GraphArmConfig
    contract: FeatureContract = field(repr=False)
    encoder_fit: EncoderFit = field(repr=False)
    construction: dict[str, object] = field(repr=False)
    primary: TwoStageResult
    # #3 §3's two pre-declared diagnostics — "not as options" — run alongside
    # the primary two-stage result on every fold, at near-zero marginal cost.
    # The third (#13's structure ablation, degree probe and label shuffle) is
    # the trust gate's, not this ticket's.
    random_init: TwoStageResult
    end_to_end: metrics.FoldMetrics

    @property
    def penalty_selection(self) -> decoder.PenaltySelection:
        return self.primary.penalty_selection

    @property
    def fold_metrics(self) -> metrics.FoldMetrics:
        return self.primary.fold_metrics

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "chosen_penalizer": self.primary.penalty_selection.chosen_penalizer,
            "penalty_scores": {str(k): v for k, v in self.primary.penalty_selection.scores.items()},
            "n_epochs_trained": len(self.encoder_fit.train_loss_curve),
            "best_epoch": self.encoder_fit.best_epoch,
            "final_train_loss": self.encoder_fit.train_loss_curve[-1],
            "final_val_loss": self.encoder_fit.val_loss_curve[-1],
            "construction": self.construction,
            "metrics": self.primary.fold_metrics.to_dict(),
            "diagnostics": {
                "random_init_encoder": {
                    "description": (
                        "Frozen, untrained encoder (#3 §3): if this scores near "
                        "the primary result, message passing/learning is not "
                        "doing the work."
                    ),
                    "chosen_penalizer": self.random_init.penalty_selection.chosen_penalizer,
                    "metrics": self.random_init.fold_metrics.to_dict(),
                },
                "end_to_end": {
                    "description": (
                        "By-product of stage one (#3 §3): the trained encoder's "
                        "own head, scored directly with no shared decoder. "
                        "'Best-case arm 3' — never the primary number."
                    ),
                    "metrics": self.end_to_end.to_dict(),
                },
            },
        }


def _describe(subgraphs: cache.SubgraphSet, blocks: FeatureBlocks, model: SubjectSubgraphEncoder) -> dict[str, object]:
    """What was actually built and trained, recorded beside the metrics.

    The shape of the raw construction has moved twice already — 7 node types
    and 6 relations on #1, 5/4 after #4, and the relation split on #11 — so a
    metric with no record of which shape produced it is not reproducible.
    """
    reference = subgraphs.graphs[0]
    return {
        "n_subjects": len(subgraphs),
        "node_types": sorted(reference.node_types),
        "relation_types": sorted("__".join(edge) for edge in reference.edge_types),
        "feature_widths": dict(sorted(blocks.widths.items())),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "fingerprint": subgraphs.fingerprint,
    }


def run_fold(
    outer_fold: int,
    *,
    records: cache.SubjectRecords,
    raw: pd.DataFrame,
    targets: SurvivalTarget,
    split: FoldSplit,
    config: GraphArmConfig,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> FoldResult:
    """One outer fold, end to end: fit contract -> build -> train -> select lambda -> score.

    The feature contract is fitted on the *inner* training set (`split.train`,
    excluding `split.val`), identically to arm 2 — the nested validation slice
    must see only train-fitted statistics, or its role as an early-stopping and
    lambda-selection check on held-out data is compromised. The construction is
    then built under that contract, which is why the cache is keyed on it.
    """
    contract = fit_feature_contract(raw, split.train)
    blocks = FeatureBlocks.from_contract(contract)
    blocks.check_partitions_contract(contract)

    subgraphs = cache.load_or_build(
        outer_fold,
        records,
        contract,
        options=config.build_options,
        directory=cache_dir,
        use_cache=use_cache,
    )

    train = collate(subgraphs, split.train, targets)
    val = collate(subgraphs, split.val, targets)
    test = collate(subgraphs, split.test, targets)
    trainval = collate(subgraphs, split.train | split.val, targets)

    encoder_fit = train_encoder(train, val, blocks, config=config)

    primary = two_stage_score(
        z_train=_embed(encoder_fit.model, train),
        train_target=train.target,
        z_trainval=_embed(encoder_fit.model, trainval),
        trainval_target=trainval.target,
        z_test=_embed(encoder_fit.model, test),
        test_target=test.target,
        penalty_grid=config.penalty_grid,
        where=f"arm3 fold {outer_fold} (train+val)",
    )

    random_model = _new_encoder(train.batch, blocks, config=config).eval()
    random_init = two_stage_score(
        z_train=_embed(random_model, train),
        train_target=train.target,
        z_trainval=_embed(random_model, trainval),
        trainval_target=trainval.target,
        z_test=_embed(random_model, test),
        test_target=test.target,
        penalty_grid=config.penalty_grid,
        where=f"arm3 fold {outer_fold} (random-init control)",
    )

    end_to_end = _end_to_end_score(
        encoder_fit.model,
        trainval=trainval,
        test=test,
        where=f"arm3 fold {outer_fold} (end-to-end diagnostic)",
    )

    return FoldResult(
        fold=outer_fold,
        config=config,
        contract=contract,
        encoder_fit=encoder_fit,
        construction=_describe(subgraphs, blocks, encoder_fit.model),
        primary=primary,
        random_init=random_init,
        end_to_end=end_to_end,
    )


def iter_folds(
    config: GraphArmConfig,
    *,
    records: cache.SubjectRecords | None = None,
    raw: pd.DataFrame | None = None,
    targets: SurvivalTarget | None = None,
    use_cache: bool = True,
) -> Iterator[FoldResult]:
    """The 5 outer folds of the locked Study-stratified split (#7), one at a time.

    A generator rather than a list because a fold here costs a construction
    build plus a training run, so a caller should be able to persist fold 3
    before fold 4 starts. Losing four completed folds to an interruption in the
    fifth is avoidable, and the caller is the only one that knows where they go.
    """
    subject_records = records if records is not None else cache.load_subject_records()
    raw_frame = raw if raw is not None else assemble_raw_frame()
    survival_targets = targets if targets is not None else load_targets()
    folds = load_folds()

    for outer_fold in range(N_SPLITS):
        yield run_fold(
            outer_fold,
            records=subject_records,
            raw=raw_frame,
            targets=survival_targets,
            split=fold_split(folds, outer_fold),
            config=config,
            use_cache=use_cache,
        )


def run_all_folds(
    config: GraphArmConfig,
    *,
    records: cache.SubjectRecords | None = None,
    raw: pd.DataFrame | None = None,
    targets: SurvivalTarget | None = None,
    use_cache: bool = True,
) -> list[FoldResult]:
    """All 5 outer folds, collected. `iter_folds` for anything that persists as it goes."""
    return list(
        iter_folds(config, records=records, raw=raw, targets=targets, use_cache=use_cache)
    )
