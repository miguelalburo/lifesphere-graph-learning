"""Stage-one mechanics every neural arm shares, and the run identity they declare.

`survival.two_stage` already made stage two one implementation rather than a
convention two arms are each expected to honour. This is the same argument
applied to stage one: an arm's *representation* is what the study varies, so
the encoder differs — but the early-stopping loop, the by-product diagnostic,
and the shape of a recorded result must not, or a divergence between them lands
in the comparison looking like a structural finding.

What is genuinely arm-specific stays with the arm: constructing the encoder
(including its seeding, which fixes the random-init control of #3 §3) and
choosing what to feed it. Everything here is generic in that input type — arm 2
passes a design-matrix tensor, arm 3 a collated `Batch`, and both encoders map
their input to a risk score by the same `model(x)` contract.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from torch import nn

from ..evaluation.splits import N_SPLITS, SEED
from ..survival import losses, metrics
from ..survival.targets import SurvivalTarget

ModelT = TypeVar("ModelT", bound=nn.Module)
InputT = TypeVar("InputT")

# The one endpoint implemented. #3 locked OS for the first comparison; PFI/DSS/DFI
# are a later ticket, and `targets.load_targets` reads the frozen OS cohort
# unconditionally, so a config naming another endpoint would be silently ignored.
LOCKED_ENDPOINT = "OS"


def locked_split_name() -> str:
    """The split every arm consumes, spelled the way a config declares it (#7)."""
    return f"{N_SPLITS}fold_study_stratified_seed{SEED}"


def check_run_identity(*, arm: str, expected_arm: str, endpoint: str, split: str) -> None:
    """Raise unless a config's declared identity matches the run it will actually get.

    `experiments/README.md` requires a config to name "the arm, the construction
    ..., the survival endpoint, the split, and the seed", because "a run is only
    useful if it is pinned down enough to line up against the others". Declaring
    those fields is not the same as honouring them: the loaders here read the
    frozen cohort and the persisted fold assignment regardless of what a config
    says, so without this check a config reading `"endpoint": "PFI"` would
    produce OS numbers filed under a PFI label — a quiet wrong answer of exactly
    the class `tests/README.md` puts first.
    """
    expected_split = locked_split_name()
    mismatches = {
        name: (declared, expected)
        for name, declared, expected in (
            ("arm", arm, expected_arm),
            ("endpoint", endpoint, LOCKED_ENDPOINT),
            ("split", split, expected_split),
        )
        if declared != expected
    }
    if mismatches:
        detail = ", ".join(
            f"{name}: config says {declared!r}, this run is {expected!r}"
            for name, (declared, expected) in sorted(mismatches.items())
        )
        raise ValueError(
            f"config declares a run it will not get -- {detail}. "
            "Change the config to match, or implement what it asks for."
        )


def config_from_dict(cls: type[Any], payload: dict[str, object]) -> dict[str, object]:
    """The subset of `payload` a config dataclass declares.

    Every config file carries `_comment` and similar documentation keys, so a
    config is never passed to its dataclass verbatim. Returning the filtered
    kwargs rather than the constructed object leaves each arm free to coerce its
    own tuple-valued fields.
    """
    known = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in known}


@dataclass(frozen=True)
class EncoderFit(Generic[ModelT]):
    """A trained encoder plus the curves that show how it got there."""

    model: ModelT
    train_loss_curve: tuple[float, ...]
    val_loss_curve: tuple[float, ...]
    best_epoch: int


def train_with_early_stopping(
    model: ModelT,
    *,
    train_input: InputT,
    train_target: SurvivalTarget,
    val_input: InputT,
    val_target: SurvivalTarget,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> EncoderFit[ModelT]:
    """Full-batch Adam on the stratified Efron Cox loss, early-stopped on val loss.

    Full-batch is a correctness requirement rather than a convenience: the Cox
    partial likelihood is not a sum over independent rows, since each event's
    term is normalised over the risk set of everyone still under observation.
    Mini-batching would evaluate a different objective.

    `model` arrives already constructed, because construction is where the seed
    is set and #3 §3's random-init control requires "exactly the pre-training
    weights" — moving it in here would put the arm's seeding at one remove from
    the arm.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    train_curve: list[float] = []
    val_curve: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        optimiser.zero_grad()
        train_loss = losses.stratified_efron_cox_loss(
            model(train_input), train_target.time, train_target.event, train_target.study
        )
        train_loss.backward()
        optimiser.step()
        train_curve.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = losses.stratified_efron_cox_loss(
                model(val_input), val_target.time, val_target.event, val_target.study
            )
        val_curve.append(float(val_loss))

        if val_curve[-1] < best_val:
            best_val = val_curve[-1]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return EncoderFit(
        model=model,
        train_loss_curve=tuple(train_curve),
        val_loss_curve=tuple(val_curve),
        best_epoch=best_epoch,
    )


def end_to_end_score(
    model: nn.Module,
    *,
    trainval_input: Any,
    trainval_target: SurvivalTarget,
    test_input: Any,
    test_target: SurvivalTarget,
    where: str,
    expect_discrimination: bool = True,
) -> metrics.FoldMetrics:
    """#3 §3's by-product diagnostic: score with the trained head directly.

    Bypasses the shared decoder entirely. No survival function exists here — a
    raw NN head produces a ranking, not a Breslow baseline hazard — so this
    never carries an integrated Brier score.

    `expect_discrimination=False` is #13's label-shuffle control and nothing
    else; see `survival.two_stage.two_stage_score` for why the self-check has
    to come off there.
    """
    model.eval()
    with torch.no_grad():
        trainval_risk = model(trainval_input).numpy()
        test_risk = model(test_input).numpy()

    # #3 §6: self-check the training-fold score before trusting the held-out
    # one — a sign-inverted risk silently produces `1 - C`.
    if expect_discrimination:
        metrics.assert_discriminates(
            trainval_target.event, trainval_target.time, trainval_risk, where=where
        )

    return metrics.score_fold(
        train_time=trainval_target.time,
        train_event=trainval_target.event,
        test_time=test_target.time,
        test_event=test_target.event,
        test_study=test_target.study,
        risk=test_risk,
    )


def diagnostics_to_dict(
    *,
    arm_label: str,
    random_init_penalizer: float,
    random_init: metrics.FoldMetrics,
    end_to_end: metrics.FoldMetrics,
) -> dict[str, object]:
    """#3 §3's two pre-declared controls, recorded in one shape across arms.

    They are pre-declared "not as options", so they run on every fold of every
    neural arm — and a reader comparing two arms' fold files should not have to
    check whether the two recorded them the same way.
    """
    return {
        "random_init_encoder": {
            "description": (
                "Frozen, untrained encoder (#3 §3): if this scores near the "
                "primary result, message passing/learning is not doing the work."
            ),
            "chosen_penalizer": random_init_penalizer,
            "metrics": random_init.to_dict(),
        },
        "end_to_end": {
            "description": (
                "By-product of stage one (#3 §3): the trained encoder's own head, "
                f"scored directly with no shared decoder. 'Best-case {arm_label}' "
                "— never the primary number."
            ),
            "metrics": end_to_end.to_dict(),
        },
    }


SUMMARY_KEYS: tuple[str, ...] = (
    "pooled_harrell_c",
    "within_study_harrell_c",
    "uno_c_ipcw",
    "integrated_brier_score",
)


def summarise_folds(fold_metrics: list[metrics.FoldMetrics]) -> dict[str, object]:
    """Mean/std across folds for the headline metrics, plus the raw per-fold values."""
    summary: dict[str, object] = {}
    for key in SUMMARY_KEYS:
        values = [m.to_dict()[key] for m in fold_metrics]
        present = [float(v) for v in values if isinstance(v, (int, float))]
        summary[key] = {
            "mean": float(np.mean(present)) if present else None,
            "std": float(np.std(present)) if present else None,
            "per_fold": values,
        }
    return summary
