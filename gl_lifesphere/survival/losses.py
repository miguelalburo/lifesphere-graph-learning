"""The differentiable stratified Efron Cox loss (#3 §1, §3).

`torchsurv` is the encoder-training half of the two-library split — it trains
stage one (here) and `decoder.py` is the identical `lifelines` fit every model's
stage two passes through. Efron ties, Study stratification, full-batch: all
three are settled, not configurable, because `tests/test_stack.py`'s
`TestObjectiveAgreement` pins that the two libraries' objectives agree only
under exactly this combination.
"""

from __future__ import annotations

import numpy as np
import torch
from torchsurv.loss.cox import neg_partial_log_likelihood

TIES_METHOD = "efron"


def _strata_codes(study: np.ndarray) -> torch.Tensor:
    """Map Study labels to small integer codes, stable within one call only.

    `torchsurv` only needs strata to partition Subjects consistently within a
    single loss evaluation — the codes never need to mean anything across
    calls, so no fitted vocabulary is required here (contrast `features/`,
    where a category *does* need to be stable across train/val/test).
    """
    _, codes = np.unique(study, return_inverse=True)
    return torch.as_tensor(codes, dtype=torch.long)


def stratified_efron_cox_loss(
    risk: torch.Tensor, time: np.ndarray, event: np.ndarray, study: np.ndarray
) -> torch.Tensor:
    """Mean negative stratified Efron partial log-likelihood, for backprop into an encoder.

    `risk` is the only argument carrying gradient; `time`/`event`/`study` are
    plain arrays the loss is evaluated against.
    """
    return neg_partial_log_likelihood(
        risk,
        torch.as_tensor(event, dtype=torch.bool),
        torch.as_tensor(time, dtype=torch.float64),
        ties_method=TIES_METHOD,
        reduction="mean",
        strata=_strata_codes(study),
    )


def stratified_partial_log_likelihood(
    risk: np.ndarray, time: np.ndarray, event: np.ndarray, study: np.ndarray
) -> float:
    """The bare (unpenalised) stratified Efron partial log-likelihood, summed over events.

    This is the statistic `decoder.select_penalty` compares across the penalty
    grid — never `CoxPHFitter.log_likelihood_`, which nets off the penalty once
    `penalizer > 0` and would bias selection toward small `lambda`
    (`tests/test_stack.py::test_penalised_log_likelihood_is_not_the_bare_partial_likelihood`).
    """
    with torch.no_grad():
        negative = neg_partial_log_likelihood(
            torch.as_tensor(risk, dtype=torch.float64),
            torch.as_tensor(event, dtype=torch.bool),
            torch.as_tensor(time, dtype=torch.float64),
            ties_method=TIES_METHOD,
            reduction="sum",
            strata=_strata_codes(study),
        )
    return -float(negative)
