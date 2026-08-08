"""Arm 1 -- the clinical baseline (#9).

The standard-of-care survival model using staging and pathology features only,
no molecular data. This is the benchmark every other arm is measured against,
so it should be built to be genuinely competitive rather than a strawman.

Settled by #8: a **penalised, Study-stratified Cox model**,
`lifelines.CoxPHFitter(penalizer=lambda, l1_ratio=0, strata=['study'])` -- the
same decoder every arm is scored through (#3 §3), fit directly on clinical
covariates rather than on a learned representation. A Random Survival Forest
was considered and ruled out as *the* baseline: it does not pass through that
decoder, so choosing it would reopen the "same decoder, different
representation" confound the two-stage design exists to close.
"""

from __future__ import annotations

from .sanity_check import SANITY_BANDS, SanityCheckResult, run_sanity_check
from .train import BaselineArmConfig, FoldResult, baseline_columns, run_all_folds, run_fold

__all__ = [
    "SANITY_BANDS",
    "BaselineArmConfig",
    "FoldResult",
    "SanityCheckResult",
    "baseline_columns",
    "run_all_folds",
    "run_fold",
    "run_sanity_check",
]
