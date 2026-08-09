"""Paired across-fold comparison of two runs, with the caveat attached to it.

Every question this study asks is a difference between two runs scored on the
same 5 folds — arm against arm, full encoder against #13's ablation, real label
against a permuted one. Pairing by fold is the right estimator for that, and it
lives here rather than in any one arm because #13 and #14 both need the same
arithmetic and must not each write their own.

**The interval this produces is anti-conservative, and that is a property of the
design rather than of the code.** Each fold trains on 3/5 of the cohort, so any
two folds share about half their training Subjects; a paired-t interval over
k-fold CV assumes an independence the folds do not have, and there is no
unbiased estimator of that variance (Bengio & Grandvalet 2004). `PairedDelta`
carries the caveat as a field so it travels into every result file that records
one, rather than living in a docstring a reader of `gate.json` never sees.

#3 §5 puts the resolution limit of this design at ~0.02 C. A difference smaller
than that is not resolvable by 5 folds however tight the interval looks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Recorded on every comparison rather than stated once, per the module docstring.
PAIRING_CAVEAT = (
    "Paired across 5 folds with overlapping training sets (any two folds share ~half their "
    "training Subjects), so this interval assumes an independence the folds do not have and is "
    "anti-conservative. #3 §5 puts the resolution limit of this design at ~0.02 C."
)


@dataclass(frozen=True)
class PairedDelta:
    """`treatment - reference`, paired by fold."""

    mean: float
    se: float
    ci95: tuple[float, float]
    per_fold: tuple[float, ...]
    folds_won: int
    n_folds: int
    caveat: str = PAIRING_CAVEAT

    @property
    def excludes_zero(self) -> bool:
        low, high = self.ci95
        return low > 0.0 or high < 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "se": self.se,
            "ci95": list(self.ci95),
            "per_fold": list(self.per_fold),
            "folds_won": f"{self.folds_won}/{self.n_folds}",
            "excludes_zero": self.excludes_zero,
            "caveat": self.caveat,
        }


def paired_delta(treatment: list[float], reference: list[float]) -> PairedDelta:
    """Per-fold differences, their mean, and a paired-t 95% interval.

    Both arguments must be the same folds in the same order — this pairs by
    position, and the only thing that makes that safe is that every run in this
    project is scored on the identical persisted fold assignment (#7).
    """
    if len(treatment) != len(reference):
        raise ValueError(
            f"paired comparison needs the same folds on both sides, got "
            f"{len(treatment)} and {len(reference)}"
        )
    if len(treatment) < 2:
        raise ValueError("a paired interval needs at least 2 folds")

    differences = np.asarray(treatment, dtype="float64") - np.asarray(reference, dtype="float64")
    n = len(differences)
    mean = float(differences.mean())
    # ddof=1: the sample standard deviation, since the folds are the sample.
    se = float(differences.std(ddof=1) / np.sqrt(n))
    half_width = float(stats.t.ppf(0.975, df=n - 1)) * se
    return PairedDelta(
        mean=mean,
        se=se,
        ci95=(mean - half_width, mean + half_width),
        per_fold=tuple(float(d) for d in differences),
        folds_won=int((differences > 0).sum()),
        n_folds=n,
    )
