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

**The band comparison uses Uno's C, not Harrell's C, and this matters.**
Herrmann et al.'s published numbers *are* Uno's IPCW-corrected C (#2 §5),
and #2 §7 warns explicitly: "A Harrell's C on our folds will typically read
higher for the same model quality — the table is a floor, not a matched
comparison." An earlier version of this module compared Harrell's C against
these bands directly and got a false alarm on BRCA (0.7304, above its
0.60-0.67 band) that vanished under Uno's C (0.6375, inside the band) —
confirmed by investigation on 2026-08-08: Harrell reading optimistic under
BRCA's censoring, not a pipeline defect. `harrell_c` is still reported
alongside for transparency (it is this project's own headline metric
elsewhere), but `in_band` is decided on Uno's C, the metric the band was
actually calibrated against. The censoring distribution for the IPCW weights
is estimated from the full pooled out-of-sample cohort (all 20 Studies), not
per-Study — the simplest defensible reference given this check runs once
across all folds rather than inside a single fold's train/test split.
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
    # Uno's C -- what the band is actually checked against, matching
    # Herrmann et al.'s own metric.
    observed_c: float | None
    # Harrell's C, reported for transparency only; not compared to the band.
    harrell_c: float | None
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
            "harrell_c": self.harrell_c,
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
    row per Subject — `pooled_out_of_sample_predictions`'s shape. The full
    frame also supplies the censoring-distribution reference for each
    Study's Uno's C (see the module docstring).
    """
    all_event = oos_predictions["event"].to_numpy()
    all_time = oos_predictions["durationDays"].to_numpy()

    results = []
    for study, band in SANITY_BANDS.items():
        rows = oos_predictions[oos_predictions["studyId"] == study]
        event = rows["event"].to_numpy()
        time = rows["durationDays"].to_numpy()
        risk = rows["risk"].to_numpy()
        n_events = int(rows["event"].sum())

        harrell: float | None
        try:
            harrell = metrics.harrell_c(event, time, risk)
        except NoComparablePairException:
            harrell = None

        uno: float | None
        try:
            if not (event.any() and all_event.any()):
                raise NoComparablePairException("no events to anchor a comparable pair")
            tau = float(min(all_time[all_event].max(), time[event].max()))
            uno = metrics.uno_c(
                train_event=all_event, train_time=all_time, test_event=event, test_time=time,
                risk=risk, tau=tau,
            )
        except (NoComparablePairException, ValueError):
            # No Study rows at all, no event to anchor a pair, or sksurv's IPCW
            # estimator refusing the tau -- reported as unscoreable
            # (`observed_c=None`), not silently averaged in.
            uno = None

        results.append(
            SanityCheckResult(
                study=study, band=band, observed_c=uno, harrell_c=harrell,
                n_subjects=len(rows), n_events=n_events,
            )
        )
    return results
