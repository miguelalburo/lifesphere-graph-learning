"""The literature sanity check #8 asks #9 to run before the baseline number is trusted.

#8's resolution: "does the fitted baseline land in the published range on the
cancer types where staging-only numbers are well documented (BRCA, KIRC,
LUAD)? Scoring far below is a pipeline bug, not a result." The bands below are
recorded on #8/#2 from the literature review (Herrmann et al.'s per-type Uno's
C, widened to a band rather than read as a point target — #2 §7 warns their
intervals "are not valid confidence intervals").

Scored against the **out-of-sample** prediction for every cohort Subject in
these three Studies, pooled across all 5 outer folds. #7's folds are a
partition of the cohort, so each Subject contributes exactly one held-out
prediction; concatenating every fold's `test_predictions` therefore covers
every BRCA/KIRC/LUAD Subject once, out of sample, with no re-use of a
training-fold prediction — the same failure mode #3 §5's within-Study metric
guards against, one fold at a time, is the reason this file is a study-level
pool across folds instead of a report from a single fold's held-out slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sksurv.exceptions import NoComparablePairException

from ...survival import metrics

# study -> (low, high). #8, 2026-08-08: "Per-cancer-type bands to check
# against (KIRC 0.72-0.79, LUAD 0.63-0.70, BRCA 0.60-0.67)".
SANITY_BANDS: dict[str, tuple[float, float]] = {
    "TCGA-BRCA": (0.60, 0.67),
    "TCGA-KIRC": (0.72, 0.79),
    "TCGA-LUAD": (0.63, 0.70),
}


@dataclass(frozen=True)
class SanityCheckResult:
    study: str
    band: tuple[float, float]
    observed_c: float | None
    n_subjects: int
    n_events: int

    @property
    def in_band(self) -> bool | None:
        if self.observed_c is None:
            return None
        low, high = self.band
        return low <= self.observed_c <= high

    def to_dict(self) -> dict[str, object]:
        return {
            "study": self.study,
            "band": list(self.band),
            "observed_c": self.observed_c,
            "in_band": self.in_band,
            "n_subjects": self.n_subjects,
            "n_events": self.n_events,
        }


def pooled_out_of_sample_predictions(fold_predictions: list[pd.DataFrame]) -> pd.DataFrame:
    """Every cohort Subject's held-out prediction, from every fold's test slice.

    Raises if a Subject is missing or duplicated — #7's folds are a partition
    by construction (`evaluation.splits.assert_one_fold_per_subject`), so a
    violation here means a fold-boundary bug upstream, not something to
    average over silently.
    """
    pooled = pd.concat(fold_predictions)
    duplicates = pooled.index[pooled.index.duplicated()]
    if len(duplicates):
        raise ValueError(
            f"{len(duplicates)} Subject(s) have more than one out-of-sample prediction: "
            f"{sorted(set(duplicates))[:5]}"
        )
    return pooled


def run_sanity_check(oos_predictions: pd.DataFrame) -> list[SanityCheckResult]:
    """One `SanityCheckResult` per Study in `SANITY_BANDS`.

    `oos_predictions` must carry `studyId`/`risk`/`durationDays`/`event`, one
    row per Subject — `pooled_out_of_sample_predictions`'s shape.
    """
    results = []
    for study, band in SANITY_BANDS.items():
        rows = oos_predictions[oos_predictions["studyId"] == study]
        n_events = int(rows["event"].sum())
        c: float | None
        try:
            c = metrics.harrell_c(
                rows["event"].to_numpy(), rows["durationDays"].to_numpy(), rows["risk"].to_numpy()
            )
        except NoComparablePairException:
            # No Study rows at all, or none with a comparable pair -- reported
            # as unscoreable (`observed_c=None`), not silently averaged in.
            c = None
        results.append(
            SanityCheckResult(
                study=study, band=band, observed_c=c, n_subjects=len(rows), n_events=n_events
            )
        )
    return results
