"""Reading an arm's already-written per-fold metrics back out of `results/metrics/`.

The gate compares against the arms rather than re-running them. That is safe
because an arm's fold is deterministic given its config and the persisted
split — `tests/test_graph.py::test_is_deterministic_given_a_fixed_seed` pins it,
and re-running arm 3's fold 0 reproduces its recorded within-Study C to every
digit — so re-training 5 folds to obtain numbers that are already on disk would
buy nothing but an hour.

What it does buy is a failure mode, so the reader is strict about it: the
recorded config must match the run the gate thinks it is comparing against,
field by field, with the fields the gate deliberately varies named explicitly.
A structure ablation silently compared against an arm 3 trained at a different
`d` would produce a plausible number and a wrong verdict, which is the exact
class `tests/README.md` puts first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..evaluation.splits import N_SPLITS
from ..extract.connection import REPO_ROOT

METRICS_DIR = REPO_ROOT / "results" / "metrics"


@dataclass(frozen=True)
class RecordedArm:
    """One arm's per-fold result files, in fold order."""

    arm: str
    folds: tuple[dict[str, object], ...]
    config: dict[str, object]

    def metric(self, key: str) -> list[float]:
        """One metric across the folds, e.g. `within_study_harrell_c`."""
        values = []
        for fold in self.folds:
            metrics = fold["metrics"]
            assert isinstance(metrics, dict)
            values.append(float(metrics[key]))
        return values


def load_arm(arm: str, *, directory: Path | None = None) -> RecordedArm:
    """Read `results/metrics/<arm>/fold_*.json` plus the summary's config block."""
    base = (directory if directory is not None else METRICS_DIR) / arm
    folds = []
    for fold in range(N_SPLITS):
        path = base / f"fold_{fold}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing, so the trust gate has nothing to compare against. "
                f"Run `python -m gl_lifesphere.models.{_module(arm)}` first."
            )
        folds.append(json.loads(path.read_text()))

    summary_path = base / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    config = summary.get("config", {})
    return RecordedArm(arm=arm, folds=tuple(folds), config=config)


def check_comparable(
    recorded: RecordedArm, config: dict[str, object], *, varies: tuple[str, ...]
) -> None:
    """Raise unless the recorded run and `config` agree on everything except `varies`.

    `varies` is the point of the control — `ablate_edges` for the structure
    ablation — and naming it here is what keeps the comparison honest: any
    *other* field that has drifted between the recorded arm and the gate's run
    is an uncontrolled second difference, and the verdict would be measuring it
    as much as the thing under test.

    Keys absent from the recorded config are skipped rather than treated as
    mismatches. `summary.json` carries the config file verbatim, and a field
    that was defaulted rather than written down is not evidence of a difference.
    An *entirely* empty recorded config is a different matter and raises: a
    guard that silently checks nothing is worse than no guard, because it reads
    as reassurance.
    """
    if not recorded.config:
        raise ValueError(
            f"{recorded.arm}'s summary.json carries no config block, so there is nothing to "
            "check this run against. Re-run the arm so its config is recorded, rather than "
            "comparing against a run whose settings are unknown."
        )
    mismatches = {
        key: (recorded.config[key], value)
        for key, value in config.items()
        if not key.startswith("_")
        and key not in varies
        and key in recorded.config
        and recorded.config[key] != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}: recorded {was!r}, this run {now!r}" for key, (was, now) in sorted(mismatches.items())
        )
        raise ValueError(
            f"{recorded.arm}'s recorded run differs from this one in more than {list(varies)} -- "
            f"{detail}. Re-run the arm, or the comparison is measuring that difference too."
        )


def _module(arm: str) -> str:
    return {"arm1_baseline": "baseline", "arm2_tabular": "tabular", "arm3_graph": "graph"}.get(
        arm, arm
    )
