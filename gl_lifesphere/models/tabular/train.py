"""Two-stage training for model 2, one outer fold at a time (#10, #3 §3, §8).

Stage one trains `TabularEncoder` end-to-end on the stratified Efron Cox loss,
full-batch, with early stopping on the nested validation fold. Stage two
freezes the encoder, extracts `z` for train+val and test, and refits the
identical `lifelines.CoxPHFitter(strata=['study'])` decoder every model passes
through — with `lambda` selected on the same nested slice, per #3 §3.

This module owns the training *mechanics*; `experiments/configs/` owns the
hyperparameters, so a config change never requires touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch

from ...evaluation.splits import FoldSplit, fold_split, load_folds
from ...features import FeatureContract, assemble_raw_frame, fit_feature_contract
from ...survival import decoder, metrics
from ...survival.targets import SurvivalTarget, load_targets
from ...survival.two_stage import TwoStageResult, two_stage_score
from .. import training
from ..training import EncoderFit
from .network import TabularEncoder

MODEL = "model2_tabular"


@dataclass(frozen=True)
class TabularModelConfig:
    """Every hyperparameter this module needs; everything else is fixed by #3."""

    model: str = MODEL
    endpoint: str = training.LOCKED_ENDPOINT
    split: str = field(default_factory=training.locked_split_name)
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
    def from_dict(cls, payload: dict[str, object]) -> "TabularModelConfig":
        """Ignores keys the dataclass does not declare (e.g. a `_comment`), so a
        documented config file need not be stripped before loading."""
        kwargs = training.config_from_dict(cls, payload)
        if "hidden_dims" in kwargs:
            kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])  # type: ignore[arg-type]
        if "penalty_grid" in kwargs:
            kwargs["penalty_grid"] = tuple(kwargs["penalty_grid"])  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]


def _to_tensor(frame: pd.DataFrame) -> torch.Tensor:
    return torch.as_tensor(frame.to_numpy(dtype="float32"))


def train_encoder(
    x_train: pd.DataFrame,
    train_target: SurvivalTarget,
    x_val: pd.DataFrame,
    val_target: SurvivalTarget,
    *,
    config: TabularModelConfig,
) -> EncoderFit[TabularEncoder]:
    """Full-batch Adam on the stratified Efron Cox loss, early-stopped on val loss."""
    return training.train_with_early_stopping(
        _new_encoder(x_train.shape[1], config=config),
        train_input=_to_tensor(x_train),
        train_target=train_target,
        val_input=_to_tensor(x_val),
        val_target=val_target,
        lr=config.lr,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        patience=config.patience,
    )


def _new_encoder(input_dim: int, *, config: TabularModelConfig) -> TabularEncoder:
    """A freshly initialised encoder. Seeding lives here because #3 §3's
    random-init control requires exactly these pre-training weights."""
    torch.manual_seed(config.seed)
    return TabularEncoder(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        d=config.d,
        dropout=config.dropout,
    )


def _embed(model: TabularEncoder, x: pd.DataFrame) -> pd.DataFrame:
    """Frozen `z`, as a DataFrame so it plugs straight into `survival.decoder`."""
    with torch.no_grad():
        z = model.embed(_to_tensor(x)).numpy()
    return pd.DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])], index=x.index)


@dataclass(frozen=True)
class FoldResult:
    fold: int
    config: TabularModelConfig
    contract: FeatureContract = field(repr=False)
    encoder_fit: EncoderFit[TabularEncoder] = field(repr=False)
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
            "diagnostics": training.diagnostics_to_dict(
                model_label="model 2",
                random_init_penalizer=self.random_init.penalty_selection.chosen_penalizer,
                random_init=self.random_init.fold_metrics,
                end_to_end=self.end_to_end,
            ),
        }


def _design_and_target(
    contract: FeatureContract, raw: pd.DataFrame, subject_ids: frozenset[str], targets: SurvivalTarget
) -> tuple[pd.DataFrame, SurvivalTarget]:
    design = contract.transform(raw, subject_ids)
    return design, targets.reorder(design.index)


def run_fold(
    outer_fold: int,
    *,
    raw: pd.DataFrame,
    targets: SurvivalTarget,
    split: FoldSplit,
    config: TabularModelConfig,
    expect_discrimination: bool = True,
) -> FoldResult:
    """One outer fold, end to end: fit contract -> train encoder -> select lambda -> score.

    The feature contract is fit on the *inner* training set (`split.train`,
    excluding `split.val`) — the nested validation slice must see only
    train-fitted statistics, or its role as an early-stopping/lambda-selection
    check on held-out data is compromised. Also runs #3 §3's two pre-declared
    diagnostics (random-init encoder, end-to-end score) alongside the primary
    two-stage result.

    `expect_discrimination=False` belongs to #13's label shuffle alone; see
    `survival.two_stage.two_stage_score`.
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

    primary = two_stage_score(
        z_train=z_train,
        train_target=train_target,
        z_trainval=z_trainval,
        trainval_target=trainval_target,
        z_test=z_test,
        test_target=test_target,
        penalty_grid=config.penalty_grid,
        where=f"model2 fold {outer_fold} (train+val)",
        expect_discrimination=expect_discrimination,
    )

    random_model = _new_encoder(x_train.shape[1], config=config).eval()
    random_init = two_stage_score(
        z_train=_embed(random_model, x_train),
        train_target=train_target,
        z_trainval=_embed(random_model, x_trainval),
        trainval_target=trainval_target,
        z_test=_embed(random_model, x_test),
        test_target=test_target,
        penalty_grid=config.penalty_grid,
        where=f"model2 fold {outer_fold} (random-init control)",
        expect_discrimination=expect_discrimination,
    )

    end_to_end = training.end_to_end_score(
        encoder_fit.model,
        trainval_input=_to_tensor(x_trainval),
        trainval_target=trainval_target,
        test_input=_to_tensor(x_test),
        test_target=test_target,
        where=f"model2 fold {outer_fold} (end-to-end diagnostic)",
        expect_discrimination=expect_discrimination,
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
    config: TabularModelConfig,
    *,
    raw: pd.DataFrame | None = None,
    targets: SurvivalTarget | None = None,
) -> list[FoldResult]:
    """All 5 outer folds of the locked Study-stratified split (#7)."""
    training.check_run_identity(
        model=config.model, expected_model=MODEL, endpoint=config.endpoint, split=config.split
    )
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
