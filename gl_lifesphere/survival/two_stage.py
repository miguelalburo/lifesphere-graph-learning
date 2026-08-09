"""Stage two: freeze the encoder's `z`, select `lambda`, fit the shared decoder, score.

#3 §3 makes the two-stage wiring the primary result for every neural arm — an
encoder trained on the stratified Efron Cox loss, then frozen, then a
`lifelines.CoxPHFitter(strata=['study'])` refit on its `z`. Stage *one* differs
between the arms, because that is the representation under test; stage two must
not, because a difference there would land in the comparison as if it were a
structural finding.

This module is that shared half, extracted so it is one implementation rather
than a convention two `train.py` files are each expected to honour. It takes
`z` as a plain frame and knows nothing about where it came from — an MLP over a
design matrix (#10) or an R-GCN over a rooted subgraph (#12).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import decoder, metrics
from .targets import SurvivalTarget


@dataclass(frozen=True)
class TwoStageResult:
    """One pass of stage two: select `lambda`, fit the shared decoder, score it."""

    penalty_selection: decoder.PenaltySelection
    fold_metrics: metrics.FoldMetrics


def two_stage_score(
    *,
    z_train: pd.DataFrame,
    train_target: SurvivalTarget,
    z_trainval: pd.DataFrame,
    trainval_target: SurvivalTarget,
    z_test: pd.DataFrame,
    test_target: SurvivalTarget,
    penalty_grid: tuple[float, ...] = decoder.PENALTY_GRID,
    where: str,
) -> TwoStageResult:
    """Stage two, given a (frozen, already-embedded) `z`.

    Shared by each arm's primary trained-encoder pass and by #3 §3's
    random-init-encoder diagnostic — both are "extract `z`, then pass it through
    the identical shared decoder", differing only in where `z` came from.
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
