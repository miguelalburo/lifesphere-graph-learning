"""Censored survival targets, losses, and metrics — shared by every arm.

This is what makes the three-way comparison apples-to-apples: each arm produces
a risk score per Subject, and the same target formulation, loss, and metric sit
downstream of it regardless of whether that score came from a GNN, an MLP over
flattened features, or the clinical baseline.

- targets  Endpoint selection (OS/PFI/DSS/DFI), fixed-horizon labels, and the
           censoring bookkeeping they require.
- losses   Cox partial likelihood and discrete-time (Logistic-Hazard/PMF)
           formulations. Plain regression on `timeToEventDays` is not an option
           here — it cannot use censored records, which are ~70% of the data.
- metrics  C-index, time-dependent AUC, Brier score.

Locked by #3: Cox partial likelihood (continuous time), Efron ties, Study
stratification, `Linear(d, 1, bias=False)` head, two-stage wiring
(`torchsurv` trains the encoder, `lifelines.CoxPHFitter(strata=['study'])` is
the one shared decoder every arm is refit and scored through).
"""

from __future__ import annotations

from . import decoder, losses, metrics
from .targets import SurvivalTarget, load_survival_frame, load_targets
from .two_stage import TwoStageResult, two_stage_score

__all__ = [
    "SurvivalTarget",
    "TwoStageResult",
    "decoder",
    "load_survival_frame",
    "load_targets",
    "losses",
    "metrics",
    "two_stage_score",
]
