"""Two-stage training for arm 2, one outer fold at a time (#10, #3 §3, §8).

Stage one trains `TabularEncoder` end-to-end on the stratified Efron Cox loss,
full-batch, with early stopping on the nested validation fold. Stage two
freezes the encoder, extracts `z` for train+val and test, and refits the
identical `lifelines.CoxPHFitter(strata=['study'])` decoder every arm passes
through — with `lambda` selected on the same nested slice, per #3 §3.

This module owns the training *mechanics*; `experiments/configs/` owns the
hyperparameters, so a config change never requires touching this file.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd
import torch

from ...evaluation.splits import FoldSplit, fold_split, load_folds
from ...features import FeatureContract, assemble_raw_frame, fit_feature_contract
from ...survival import decoder, losses, metrics
from ...survival.targets import SurvivalTarget, load_targets
from .network import TabularEncoder


@dataclass(frozen=True)
class TabularArmConfig:
    """Every hyperparameter this module needs; everything else is fixed by #3."""

    hidden_dims: tuple[int, ...] = (32,)
    d: int = 16
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 500
    patience: int = 30
    seed: int = 0
    penalty_grid: tuple[float, ...] = decoder.PENALTY_GRID

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TabularArmConfig":
        """Ignores keys the dataclass does not declare (e.g. a `_comment`), so a
        documented config file need not be stripped before loading."""
        known = {f.name for f in fields(cls)}
        kwargs = {key: value for key, value in payload.items() if key in known}
        if "hidden_dims" in kwargs:
            kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])  # type: ignore[arg-type]
        if "penalty_grid" in kwargs:
            kwargs["penalty_grid"] = tuple(kwargs["penalty_grid"])  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class EncoderFit:
    model: TabularEncoder
    train_loss_curve: tuple[float, ...]
    val_loss_curve: tuple[float, ...]
    best_epoch: int


def _to_tensor(frame: pd.DataFrame) -> torch.Tensor:
    return torch.as_tensor(frame.to_numpy(dtype="float32"))


def train_encoder(
    x_train: pd.DataFrame,
    train_target: SurvivalTarget,
    x_val: pd.DataFrame,
    val_target: SurvivalTarget,
    *,
    config: TabularArmConfig,
) -> EncoderFit:
    """Full-batch Adam on the stratified Efron Cox loss, early-stopped on val loss."""
    torch.manual_seed(config.seed)
    model = TabularEncoder(
        input_dim=x_train.shape[1], hidden_dims=config.hidden_dims, d=config.d, dropout=config.dropout
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_x = _to_tensor(x_train)
    val_x = _to_tensor(x_val)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    train_curve: list[float] = []
    val_curve: list[float] = []

    for epoch in range(config.max_epochs):
        model.train()
        optimiser.zero_grad()
        train_risk = model(train_x)
        train_loss = losses.stratified_efron_cox_loss(
            train_risk, train_target.time, train_target.event, train_target.study
        )
        train_loss.backward()
        optimiser.step()
        train_curve.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_risk = model(val_x)
            val_loss = losses.stratified_efron_cox_loss(
                val_risk, val_target.time, val_target.event, val_target.study
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


def _embed(model: TabularEncoder, x: pd.DataFrame) -> pd.DataFrame:
    """Frozen `z`, as a DataFrame so it plugs straight into `survival.decoder`."""
    with torch.no_grad():
        z = model.embed(_to_tensor(x)).numpy()
    return pd.DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])], index=x.index)


@dataclass(frozen=True)
class TwoStageResult:
    """One pass of stage two: select `lambda`, fit the shared decoder, score it."""

    penalty_selection: decoder.PenaltySelection
    fold_metrics: metrics.FoldMetrics


@dataclass(frozen=True)
class FoldResult:
    fold: int
    config: TabularArmConfig
    contract: FeatureContract = field(repr=False)
    encoder_fit: EncoderFit = field(repr=False)
    primary: TwoStageResult
    # #3 §3's two pre-declared diagnostics — "not as options" — run alongside
    # the primary two-stage result on every fold, at near-zero marginal cost.
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
                        "'Best-case arm 2' — never the primary number."
                    ),
                    "metrics": self.end_to_end.to_dict(),
                },
            },
        }


def _design_and_target(
    contract: FeatureContract, raw: pd.DataFrame, subject_ids: frozenset[str], targets: SurvivalTarget
) -> tuple[pd.DataFrame, SurvivalTarget]:
    design = contract.transform(raw, subject_ids)
    return design, targets.reorder(design.index)


def _two_stage_score(
    *,
    z_train: pd.DataFrame,
    train_target: SurvivalTarget,
    z_trainval: pd.DataFrame,
    trainval_target: SurvivalTarget,
    z_test: pd.DataFrame,
    test_target: SurvivalTarget,
    penalty_grid: tuple[float, ...],
    where: str,
) -> TwoStageResult:
    """Stage two, given a (frozen, already-embedded) `z`: select `lambda`, fit, score.

    Shared by the primary trained-encoder pass and the random-init-encoder
    diagnostic (#3 §3) — both are "extract `z`, then pass it through the
    identical shared decoder", differing only in where `z` came from.
    """
    selection = decoder.select_penalty(
        z_train,
        train_study=train_target.study,
        train_time=train_target.time,
        train_event=train_target.event,
        trainval=z_trainval,
        trainval_study=trainval_target.study,
        trainval_time=trainval_target.time,
        trainval_event=trainval_target.event,
        grid=penalty_grid,
    )

    fit = decoder.fit_decoder(
        z_trainval,
        study=trainval_target.study,
        time=trainval_target.time,
        event=trainval_target.event,
        penalizer=selection.chosen_penalizer,
    )

    # #3 §6: every arm must self-check its training-fold score before trusting
    # the held-out one — a sign-inverted risk silently produces `1 - C`.
    train_risk = decoder.risk_scores(fit, z_trainval, study=trainval_target.study)
    metrics.assert_discriminates(trainval_target.event, trainval_target.time, train_risk, where=where)

    test_risk = decoder.risk_scores(fit, z_test, study=test_target.study)
    horizons = metrics.DEFAULT_HORIZONS_DAYS
    test_survival = decoder.survival_function(
        fit, z_test, study=test_target.study, times=np.asarray(horizons)
    )
    fold_metrics = metrics.score_fold(
        train_time=trainval_target.time,
        train_event=trainval_target.event,
        test_time=test_target.time,
        test_event=test_target.event,
        test_study=test_target.study,
        risk=test_risk,
        survival_probabilities=test_survival,
        horizons=horizons,
    )
    return TwoStageResult(penalty_selection=selection, fold_metrics=fold_metrics)


def _end_to_end_score(
    model: TabularEncoder,
    *,
    x_trainval: pd.DataFrame,
    trainval_target: SurvivalTarget,
    x_test: pd.DataFrame,
    test_target: SurvivalTarget,
    where: str,
) -> metrics.FoldMetrics:
    """The by-product diagnostic (#3 §3): score with the trained head directly,
    bypassing the shared decoder entirely. No survival function exists here —
    a raw NN head produces a ranking, not a Breslow baseline hazard — so this
    never carries an integrated Brier score."""
    with torch.no_grad():
        trainval_risk = model(_to_tensor(x_trainval)).numpy()
        test_risk = model(_to_tensor(x_test)).numpy()

    metrics.assert_discriminates(trainval_target.event, trainval_target.time, trainval_risk, where=where)

    return metrics.score_fold(
        train_time=trainval_target.time,
        train_event=trainval_target.event,
        test_time=test_target.time,
        test_event=test_target.event,
        test_study=test_target.study,
        risk=test_risk,
    )


def run_fold(
    outer_fold: int,
    *,
    raw: pd.DataFrame,
    targets: SurvivalTarget,
    split: FoldSplit,
    config: TabularArmConfig,
) -> FoldResult:
    """One outer fold, end to end: fit contract -> train encoder -> select lambda -> score.

    The feature contract is fit on the *inner* training set (`split.train`,
    excluding `split.val`) — the nested validation slice must see only
    train-fitted statistics, or its role as an early-stopping/lambda-selection
    check on held-out data is compromised. Also runs #3 §3's two pre-declared
    diagnostics (random-init encoder, end-to-end score) alongside the primary
    two-stage result.
    """
    contract = fit_feature_contract(raw, split.train)

    x_train, train_target = _design_and_target(contract, raw, split.train, targets)
    x_val, val_target = _design_and_target(contract, raw, split.val, targets)
    x_test, test_target = _design_and_target(contract, raw, split.test, targets)

    x_trainval = pd.concat([x_train, x_val])
    trainval_target = targets.reorder(x_trainval.index)

    encoder_fit = train_encoder(x_train, train_target, x_val, val_target, config=config)

    z_train = _embed(encoder_fit.model, x_train)
    z_trainval = _embed(encoder_fit.model, x_trainval)
    z_test = _embed(encoder_fit.model, x_test)

    primary = _two_stage_score(
        z_train=z_train,
        train_target=train_target,
        z_trainval=z_trainval,
        trainval_target=trainval_target,
        z_test=z_test,
        test_target=test_target,
        penalty_grid=config.penalty_grid,
        where=f"arm2 fold {outer_fold} (train+val)",
    )

    torch.manual_seed(config.seed)
    random_model = TabularEncoder(
        input_dim=x_train.shape[1], hidden_dims=config.hidden_dims, d=config.d, dropout=config.dropout
    ).eval()
    random_init = _two_stage_score(
        z_train=_embed(random_model, x_train),
        train_target=train_target,
        z_trainval=_embed(random_model, x_trainval),
        trainval_target=trainval_target,
        z_test=_embed(random_model, x_test),
        test_target=test_target,
        penalty_grid=config.penalty_grid,
        where=f"arm2 fold {outer_fold} (random-init control)",
    )

    end_to_end = _end_to_end_score(
        encoder_fit.model,
        x_trainval=x_trainval,
        trainval_target=trainval_target,
        x_test=x_test,
        test_target=test_target,
        where=f"arm2 fold {outer_fold} (end-to-end diagnostic)",
    )

    return FoldResult(
        fold=outer_fold,
        config=config,
        contract=contract,
        encoder_fit=encoder_fit,
        primary=primary,
        random_init=random_init,
        end_to_end=end_to_end,
    )


def run_all_folds(
    config: TabularArmConfig,
    *,
    raw: pd.DataFrame | None = None,
    targets: SurvivalTarget | None = None,
) -> list[FoldResult]:
    """All 5 outer folds of the locked Study-stratified split (#7)."""
    raw_frame = raw if raw is not None else assemble_raw_frame()
    survival_targets = targets if targets is not None else load_targets()
    folds = load_folds()

    results = []
    for outer_fold in range(5):
        split = fold_split(folds, outer_fold)
        results.append(
            run_fold(outer_fold, raw=raw_frame, targets=survival_targets, split=split, config=config)
        )
    return results
