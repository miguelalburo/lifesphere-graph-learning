"""The single scoring entry point every arm is required to call through (#3 §5, §6).

Arm code must never call `sksurv` or `lifelines` metrics directly — the sign
convention (higher risk = higher hazard) and the pair-pooling rule for
within-Study concordance are both quiet-wrong-answer traps
(`lifelines.utils.concordance_index` silently returns `1 - C`; averaging
per-Study C-indices instead of pooling their pair counts silently gives
TGCT's handful of pairs the same weight as BRCA's thousands), and pinning them
in one place is what keeps every arm from having to get both right
independently.

Convention throughout: **higher risk score `r` = higher hazard = shorter
survival** (Cox log-partial-hazard). Every ranking metric here is invariant to
the scale of `r` and any monotone transform of it, and reflects to `1 - C` if
its sign is flipped — which does not raise, hence `assert_discriminates`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score as _sksurv_integrated_brier_score,
)
from sksurv.util import Surv

# 1, 2 and 3 years in days, and the Brier integration window (#3 §5) — chosen
# from what the follow-up actually supports, not by convention: beyond 3 years
# a per-Study metric is fiction (decoder doc §5's at-risk table).
DEFAULT_HORIZONS_DAYS: tuple[float, ...] = (365.0, 730.0, 1095.0)


@dataclass(frozen=True)
class PairCounts:
    concordant: int
    discordant: int
    tied_risk: int

    @property
    def comparable(self) -> int:
        return self.concordant + self.discordant + self.tied_risk

    @property
    def cindex(self) -> float | None:
        if self.comparable == 0:
            return None
        return (self.concordant + 0.5 * self.tied_risk) / self.comparable

    def __add__(self, other: "PairCounts") -> "PairCounts":
        return PairCounts(
            self.concordant + other.concordant,
            self.discordant + other.discordant,
            self.tied_risk + other.tied_risk,
        )


_EMPTY = PairCounts(0, 0, 0)


def _pair_counts(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> PairCounts | None:
    """`None` for a stratum with no comparable pairs, rather than a fabricated value.

    A stratum with zero events (all censored) is the same "nothing to score"
    case as zero comparable pairs, but `sksurv` raises a bare `ValueError`
    ("All samples are censored") for it rather than `NoComparablePairException`
    — checked explicitly up front rather than added to the `except` clause, so
    a sparse Study/fold cell is excluded rather than crashing the whole fold.
    """
    if not event.any():
        return None
    try:
        _, concordant, discordant, tied_risk, _ = concordance_index_censored(event, time, risk)
    except NoComparablePairException:
        return None
    return PairCounts(int(concordant), int(discordant), int(tied_risk))


def harrell_c(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    """Pooled Harrell C-index over the whole array — pan-cancer, not per-Study."""
    counts = _pair_counts(event, time, risk)
    if counts is None or counts.cindex is None:
        raise NoComparablePairException("No comparable pairs in the supplied set.")
    return counts.cindex


@dataclass(frozen=True)
class WithinStudyConcordance:
    """Pair-pooled within-Study Harrell C (#3 §5): sum pair counts, then divide once."""

    cindex: float
    excluded_studies: tuple[str, ...]
    per_study: dict[str, float | None]


def within_study_harrell_c(
    event: np.ndarray, time: np.ndarray, risk: np.ndarray, study: np.ndarray
) -> WithinStudyConcordance:
    """Cross-Study pairs are never counted: concordance is computed inside each
    Study slice before any pooling happens."""
    total = _EMPTY
    excluded: list[str] = []
    per_study: dict[str, float | None] = {}
    for name in sorted(set(study)):
        mask = study == name
        counts = _pair_counts(event[mask], time[mask], risk[mask])
        if counts is None:
            excluded.append(str(name))
            per_study[str(name)] = None
            continue
        per_study[str(name)] = counts.cindex
        total = total + counts

    if total.cindex is None:
        raise NoComparablePairException("No Study contributed a comparable pair.")
    return WithinStudyConcordance(
        cindex=total.cindex, excluded_studies=tuple(excluded), per_study=per_study
    )


def uno_c(
    *,
    train_event: np.ndarray,
    train_time: np.ndarray,
    test_event: np.ndarray,
    test_time: np.ndarray,
    risk: np.ndarray,
    tau: float | None = None,
) -> float:
    """Uno's IPCW-corrected C-index, censoring distribution estimated on the training fold.

    `tau` defaults to the 3-year horizon — the same window the Brier score is
    integrated over — rather than the full follow-up tail. IPCW weights scale
    with `1/G(t)^2` where `G` is the censoring-survival estimate, and near the
    end of follow-up `G` is estimated from a handful of Subjects and can be
    close to zero; measured on this cohort, extending `tau` to the observed
    maximum inflates a *pure-noise* risk score's Uno C from ~0.5 to 0.4-0.6
    swings across reruns, which is the estimator's tail variance rather than
    signal. Capping at 3 years is also required for `tau` to stay within
    `train_time`'s range regardless of which fold is being scored.
    """
    survival_train = Surv.from_arrays(event=train_event, time=train_time)
    survival_test = Surv.from_arrays(event=test_event, time=test_time)
    if tau is None:
        tau = min(DEFAULT_HORIZONS_DAYS[-1], float(train_time.max()), float(test_time.max()))
    return float(concordance_index_ipcw(survival_train, survival_test, risk, tau=tau)[0])


def time_dependent_auc(
    *,
    train_event: np.ndarray,
    train_time: np.ndarray,
    test_event: np.ndarray,
    test_time: np.ndarray,
    risk: np.ndarray,
    horizons: tuple[float, ...] = DEFAULT_HORIZONS_DAYS,
) -> dict[float, float]:
    """AUC(t) at each horizon, for a time-invariant risk score."""
    survival_train = Surv.from_arrays(event=train_event, time=train_time)
    survival_test = Surv.from_arrays(event=test_event, time=test_time)
    per_horizon, _ = cumulative_dynamic_auc(survival_train, survival_test, risk, list(horizons))
    return {horizon: float(value) for horizon, value in zip(horizons, per_horizon, strict=True)}


def integrated_brier(
    *,
    train_event: np.ndarray,
    train_time: np.ndarray,
    test_event: np.ndarray,
    test_time: np.ndarray,
    survival_probabilities: np.ndarray,
    horizons: tuple[float, ...] = DEFAULT_HORIZONS_DAYS,
) -> float:
    """Integrated Brier score over `horizons`, from a (n_subjects, n_horizons) S(t|x) matrix."""
    survival_train = Surv.from_arrays(event=train_event, time=train_time)
    survival_test = Surv.from_arrays(event=test_event, time=test_time)
    return float(
        _sksurv_integrated_brier_score(
            survival_train, survival_test, survival_probabilities, list(horizons)
        )
    )


def assert_discriminates(
    event: np.ndarray, time: np.ndarray, risk: np.ndarray, *, where: str, min_c: float = 0.5
) -> None:
    """Raise unless `risk` discriminates better than chance (#3 §6).

    Every arm should call this on its own training-fold score. A sign error in
    the risk convention does not crash — it silently produces `C = 1 - C`,
    which reads as "a bad model" rather than as the defect it is.
    """
    c = harrell_c(event, time, risk)
    if c <= min_c:
        raise ValueError(
            f"{where}: Harrell C = {c:.4f} does not exceed {min_c}; the risk score may be "
            "inverted (a flipped sign silently produces 1 - C) or the model may not have trained."
        )


@dataclass(frozen=True)
class FoldMetrics:
    """Everything #3 §5 asks for from one fold's held-out test set."""

    pooled_harrell_c: float
    within_study: WithinStudyConcordance
    uno_c: float
    td_auc: dict[float, float]
    integrated_brier: float | None
    n_test: int
    n_events_test: int

    def to_dict(self) -> dict[str, object]:
        return {
            "pooled_harrell_c": self.pooled_harrell_c,
            "within_study_harrell_c": self.within_study.cindex,
            "within_study_excluded_studies": list(self.within_study.excluded_studies),
            "per_study_harrell_c": self.within_study.per_study,
            "uno_c_ipcw": self.uno_c,
            "time_dependent_auc": {str(k): v for k, v in self.td_auc.items()},
            "integrated_brier_score": self.integrated_brier,
            "n_test": self.n_test,
            "n_events_test": self.n_events_test,
        }


def score_fold(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    test_study: np.ndarray,
    risk: np.ndarray,
    survival_probabilities: np.ndarray | None = None,
    horizons: tuple[float, ...] = DEFAULT_HORIZONS_DAYS,
) -> FoldMetrics:
    """The one call an arm's fold-evaluation step makes.

    `survival_probabilities` (n_test, len(horizons)) is optional because it
    requires the decoder's Breslow baseline hazard, which not every caller has
    computed — when omitted, `integrated_brier_score` is `None` and every
    ranking metric is still returned.
    """
    brier = (
        integrated_brier(
            train_event=train_event,
            train_time=train_time,
            test_event=test_event,
            test_time=test_time,
            survival_probabilities=survival_probabilities,
            horizons=horizons,
        )
        if survival_probabilities is not None
        else None
    )
    return FoldMetrics(
        pooled_harrell_c=harrell_c(test_event, test_time, risk),
        within_study=within_study_harrell_c(test_event, test_time, risk, test_study),
        uno_c=uno_c(
            train_event=train_event,
            train_time=train_time,
            test_event=test_event,
            test_time=test_time,
            risk=risk,
        ),
        td_auc=time_dependent_auc(
            train_event=train_event,
            train_time=train_time,
            test_event=test_event,
            test_time=test_time,
            risk=risk,
            horizons=horizons,
        ),
        integrated_brier=brier,
        n_test=int(len(test_time)),
        n_events_test=int(test_event.sum()),
    )
