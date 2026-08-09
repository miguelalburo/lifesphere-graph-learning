"""Run the trust gate and write its three results plus the verdict.

    python -m gl_lifesphere.diagnostics [--config PATH] [--out DIR]
                                        [--only ablation|degree-probe|shuffle|gate]
                                        [--pull-counts] [--no-cache]

`--pull-counts` is the one step that needs the live Neo4j instance: the degree
probe's `n_interventions` is not in the frozen extract, because `Intervention`
is a dropped node type (#4 §4). Run it once; the counts land in
`data/interim/diagnostics/` and every later run reads them from there.

Each diagnostic is written as it finishes, and `--only` runs one of them, because
the shuffle alone is 30 arm-3 training runs and losing a completed ablation to an
interruption in it is avoidable. `gate.json` is then assembled from those three
files — never from the objects a run happens to be holding — so `--only gate`
re-judges a finished run in a second, and what the gate judges is by
construction exactly what was recorded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..constructions import cache
from ..extract.connection import REPO_ROOT
from ..features import assemble_raw_frame
from ..models.baseline.train import ARM as BASELINE_ARM
from ..models.graph.train import ARM as GRAPH_ARM
from ..models.tabular.train import ARM as TABULAR_ARM
from ..survival import metrics
from ..survival.targets import load_targets
from . import recorded
from .ablation import run_structure_ablation
from .counts import load_accrual_counts, pull_intervention_counts
from .degree_probe import run_degree_probe
from .gate import GateConfig, GateInputs, assess
from .shuffle import run_label_shuffle

DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "trust_gate.json"
DEFAULT_OUT = REPO_ROOT / "results" / "metrics" / "trust_gate"

ARMS = (BASELINE_ARM, TABULAR_ARM, GRAPH_ARM)
METRIC = metrics.HEADLINE

# What `--only` accepts. A module constant because `gate.PRODUCED_BY` suggests
# these values in its "run this first" errors, and `tests/test_diagnostics.py`
# pins that the advice is a command the CLI actually takes.
ONLY_CHOICES: tuple[str, ...] = ("ablation", "degree-probe", "shuffle", "gate")


@dataclass(frozen=True)
class SharedLoads:
    """The two expensive loads the ablation and the shuffle both need."""

    records: cache.SubjectRecords
    raw: pd.DataFrame


def main(
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT,
    *,
    only: str | None = None,
    pull_counts: bool = False,
    use_cache: bool = True,
) -> dict[str, object]:
    payload = json.loads(config_path.read_text())
    config = GateConfig.from_dict(payload)
    config.check()
    baseline_config, tabular_config, graph_config = config.arm_configs(config_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pull_counts:
        pull_intervention_counts()

    targets = load_targets()
    # Loaded once and shared, because the shuffle retrains all three arms across
    # two schemes and re-reading the interim tables per run would dominate its
    # cost — but loaded lazily, because the degree probe needs neither and
    # `--only degree-probe` should not pay for a construction it never opens.
    inputs: SharedLoads | None = None

    def shared_loads() -> SharedLoads:
        nonlocal inputs
        if inputs is None:
            inputs = SharedLoads(cache.load_subject_records(), assemble_raw_frame())
        return inputs

    if only in (None, "ablation"):
        shared = shared_loads()
        ablation = run_structure_ablation(
            graph_config,
            records=shared.records,
            raw=shared.raw,
            targets=targets,
            use_cache=use_cache,
        )
        _write(out_dir / "structure_ablation.json", ablation.to_dict())

    if only in (None, "degree-probe"):
        probe = run_degree_probe(counts=load_accrual_counts(), targets=targets)
        _write(out_dir / "degree_probe.json", probe.to_dict())

    if only in (None, "shuffle"):
        shared = shared_loads()
        shuffled = run_label_shuffle(
            schemes=config.shuffle_schemes,
            seed=config.seed,
            baseline_config=baseline_config,
            tabular_config=tabular_config,
            graph_config=graph_config,
            records=shared.records,
            raw=shared.raw,
            targets=targets,
            use_cache=use_cache,
        )
        _write(out_dir / "label_shuffle.json", shuffled.to_dict())

    if only is not None and only != "gate":
        return {"wrote": only}

    # Assembled from the files rather than from the objects just built, so the
    # gate always judges exactly what was recorded — and so `--only gate`
    # re-judges a previous run without retraining anything.
    report = assess(
        GateInputs.from_files(
            out_dir,
            arm_means={
                arm: float(np.mean(recorded.load_arm(arm).metric(METRIC))) for arm in ARMS
            },
        ),
        config,
    )
    gate = report.to_dict()
    _write(out_dir / "gate.json", gate)
    return gate


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run #13's trust gate: the three §7 diagnostics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        choices=ONLY_CHOICES,
        default=None,
        help="Run one diagnostic; 'gate' re-judges the three already on disk.",
    )
    parser.add_argument(
        "--pull-counts",
        action="store_true",
        help="Pull per-Subject Intervention counts from the live instance (needed once).",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    result = main(
        args.config,
        args.out,
        only=args.only,
        pull_counts=args.pull_counts,
        use_cache=not args.no_cache,
    )
    print(json.dumps(result.get("overall", result), indent=2))
