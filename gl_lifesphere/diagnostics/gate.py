"""The three verdicts, and the thresholds they are judged against.

#13 is explicit that this is "a gate, not a measurement", which imposes one
discipline the rest of the project does not need: **the bar is declared before
the run**. The thresholds live in `experiments/configs/trust_gate.json` and are
read from there, so a diagnostic that lands just the wrong side of one cannot be
rescued by moving it. A gate whose bar moves to accommodate its result is not a
gate.

Each threshold is a number the project already owns rather than a fresh
judgement call:

- **Ablation margin, 0.02.** #3 §5's resolution limit for this design. §7 asks
  the full encoder to "clearly beat" the ablation, and 5 correlated folds cannot
  resolve a difference below that, so "clearly" cannot mean less.
- **Degree-probe ceiling, 0.55.** #13 says the counts are excluded from every
  model "precisely so this probe should come out weak", so the bar is the top of
  the chance band and nothing else: 0.5 plus three standard errors of a C-index
  on ~1,360 held-out Subjects (~0.017 each). Deliberately *not* derived from
  how the models scored — a threshold read off the numbers it is meant to gate is
  not a declared threshold.
- **Shuffle tolerance, 0.05.** The same three standard errors, for the same
  reason: §7 asks for "~0.5", and this is wide enough not to fail on a single
  permutation draw, narrow enough that a real leak cannot hide inside it.

A FAIL is not a caveat to attach to the results table. #13: "If any fails, the
resolution is the diagnosis and the fix." Each verdict therefore carries the
reading its own failure implies, so the next step is in the artefact rather than
in someone's memory of this ticket.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from ..models import training
from ..models.baseline.train import BaselineModelConfig
from ..models.graph.train import MODEL as GRAPH_MODEL
from ..models.graph.train import GraphModelConfig
from ..models.tabular.train import TabularModelConfig
from ..survival import metrics
from .shuffle import GLOBAL, SCHEMES

PASS = "PASS"
FAIL = "FAIL"

METRIC = metrics.HEADLINE


@dataclass(frozen=True)
class GateConfig:
    """Declared before the run; see the module docstring for where each number comes from."""

    endpoint: str = training.LOCKED_ENDPOINT
    split: str = field(default_factory=training.locked_split_name)
    seed: int = 0
    ablation_margin: float = 0.02
    degree_probe_ceiling: float = 0.55
    shuffle_tolerance: float = 0.05
    shuffle_schemes: tuple[str, ...] = SCHEMES
    shuffle_gate_scheme: str = GLOBAL

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GateConfig":
        """Ignores keys the dataclass does not declare (e.g. a `_comment`)."""
        kwargs = {f.name: payload[f.name] for f in fields(cls) if f.name in payload}
        if "shuffle_schemes" in kwargs:
            kwargs["shuffle_schemes"] = tuple(kwargs["shuffle_schemes"])  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]

    def check(self) -> None:
        """Raise unless this config declares the run it will actually get.

        The same rule `training.check_run_identity` applies to every model, and
        the gate needs it for the same reason: the loaders read the frozen
        cohort and the persisted fold assignment regardless of what a config
        says, so a gate declaring `"endpoint": "PFI"` would pass or fail OS
        results under a PFI label. Spelled out here rather than delegated
        because the gate has no single model to name.
        """
        expected_split = training.locked_split_name()
        problems = [
            f"endpoint: config says {self.endpoint!r}, this run is {training.LOCKED_ENDPOINT!r}"
            if self.endpoint != training.LOCKED_ENDPOINT
            else "",
            f"split: config says {self.split!r}, this run is {expected_split!r}"
            if self.split != expected_split
            else "",
            f"shuffle_gate_scheme {self.shuffle_gate_scheme!r} is not among "
            f"shuffle_schemes {list(self.shuffle_schemes)}, so nothing would be judged"
            if self.shuffle_gate_scheme not in self.shuffle_schemes
            else "",
        ]
        detail = ", ".join(p for p in problems if p)
        if detail:
            raise ValueError(f"trust gate declares a run it will not get -- {detail}.")

    def model_configs(
        self, directory: Path
    ) -> tuple[BaselineModelConfig, TabularModelConfig, GraphModelConfig]:
        """The three models' configs, read from the models' **own** config files.

        The gate retrains each model under a permuted label, so it has to build
        them exactly as their own runs did. That means reading
        `experiments/configs/model{1,2,3}_*.json` directly: a copy of those
        hyperparameters inside the gate's config would be a second place to
        change them, and a second place that nothing checks. It would also drift
        silently — the shuffle has no `check_comparable` equivalent, so a model
        rebuilt from stale numbers would produce a control for a model nobody
        ran.
        """
        return (
            BaselineModelConfig.from_dict(_read(directory / "model1_baseline.json")),
            TabularModelConfig.from_dict(_read(directory / "model2_tabular.json")),
            GraphModelConfig.from_dict(_read(directory / f"{GRAPH_MODEL}.json")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "split": self.split,
            "seed": self.seed,
            "ablation_margin": self.ablation_margin,
            "degree_probe_ceiling": self.degree_probe_ceiling,
            "shuffle_tolerance": self.shuffle_tolerance,
            "shuffle_schemes": list(self.shuffle_schemes),
            "shuffle_gate_scheme": self.shuffle_gate_scheme,
        }


@dataclass(frozen=True)
class GateInputs:
    """The three diagnostics' recorded results, plus the models they are judged beside.

    **Held as the serialised dicts, not as the result objects, and that is the
    point.** What the gate judges is then literally what was written to
    `results/metrics/trust_gate/`, and judging a finished run and re-judging one
    from disk are the same code path rather than two that are supposed to agree.

    It also makes the diagnostics independently re-runnable. Each is expensive
    and they are written as they finish (`__main__`'s `--only`), so a gate that
    could only be assembled from three live objects would force a full retrain
    — an hour — to re-read a verdict after fixing one of them. There is
    deliberately no `from_results` alternative: one path, so the two can never
    disagree.
    """

    ablation: dict[str, object]
    degree_probe: dict[str, object]
    label_shuffle: dict[str, object]
    model_means: dict[str, float]

    @classmethod
    def from_files(cls, directory: Path, *, model_means: dict[str, float]) -> "GateInputs":
        """Re-read the three diagnostics from a previous run's output directory."""
        return cls(
            ablation=_read(directory / "structure_ablation.json"),
            degree_probe=_read(directory / "degree_probe.json"),
            label_shuffle=_read(directory / "label_shuffle.json"),
            model_means=model_means,
        )


# Each diagnostic's file, and the `--only` value that produces it. Spelled out
# rather than derived from the filename, so the error tells a reader a command
# that actually exists.
PRODUCED_BY: dict[str, str] = {
    "structure_ablation.json": "ablation",
    "degree_probe.json": "degree-probe",
    "label_shuffle.json": "shuffle",
}


def _read(path: Path) -> dict[str, object]:
    if not path.exists():
        hint = (
            f" Run `python -m gl_lifesphere.diagnostics --only {PRODUCED_BY[path.name]}` first."
            if path.name in PRODUCED_BY
            else ""
        )
        raise FileNotFoundError(f"{path} is missing, so the gate cannot be assembled.{hint}")
    payload: dict[str, object] = json.loads(path.read_text())
    return payload


def _floats(payload: dict[str, object], key: str) -> list[float]:
    values = payload[key]
    assert isinstance(values, list)
    return [float(v) for v in values]


@dataclass(frozen=True)
class Verdict:
    """One diagnostic's outcome, with the rule that produced it and the reading it implies."""

    diagnostic: str
    verdict: str
    statistic: float
    threshold: float
    rule: str
    reading: str
    detail: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == PASS

    def to_dict(self) -> dict[str, object]:
        return {
            "diagnostic": self.diagnostic,
            "verdict": self.verdict,
            "statistic": self.statistic,
            "threshold": self.threshold,
            "rule": self.rule,
            "reading": self.reading,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GateReport:
    config: GateConfig
    verdicts: tuple[Verdict, ...]

    @property
    def overall(self) -> str:
        return PASS if all(v.passed for v in self.verdicts) else FAIL

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": "trust_gate",
            "issue": "#13",
            "metric": METRIC,
            "overall": self.overall,
            "config": self.config.to_dict(),
            "verdicts": [v.to_dict() for v in self.verdicts],
            "what_a_verdict_is": (
                "A gate, not a measurement (#13). PASS means the control failed the way a "
                "control is supposed to, so a result from this pipeline is worth reading. It "
                "says nothing about whether that result is good."
            ),
        }


def assess_ablation(inputs: GateInputs, config: GateConfig) -> Verdict:
    """PASS when the full encoder clears the edge-ablated one by the declared margin."""
    full_per_fold = _floats(inputs.ablation, "full_per_fold")
    ablated_per_fold = _floats(inputs.ablation, "ablated_per_fold")
    full = float(np.mean(full_per_fold))
    ablated = float(np.mean(ablated_per_fold))
    delta = full - ablated
    passed = delta > config.ablation_margin
    return Verdict(
        diagnostic="structure_ablation",
        verdict=PASS if passed else FAIL,
        statistic=delta,
        threshold=config.ablation_margin,
        rule=f"PASS if mean(full) - mean(edges ablated) > {config.ablation_margin} in {METRIC}",
        reading=(
            (
                "Message passing contributes beyond the root's own features. "
                "**This does not establish that structure beats flattening, and it must not be "
                "quoted as if it did.** The ablated encoder is rung -1 of encoder doc §4's "
                "ladder -- below model 2 -- because stage lives on Diagnosis and cancer type on "
                "Condition, so severing the edges removes the encoder's access to the features "
                "themselves, not merely to their arrangement. What it proves is that the edges "
                "are load-bearing for reaching the data at all. The question of whether passing "
                "messages over those features beats flattening them is model 2 vs model 3, which is "
                "a different comparison and is still inside the noise floor (+0.0180, CI crosses "
                "zero, #12)."
            )
            if passed
            else (
                "Message passing contributes nothing measurable. Encoder doc §7 names the §2.1 "
                "reverse-edge bug as the first suspect, but that is already ruled out -- "
                "tests/test_constructions.py runs the reverse-edge negative control live. So "
                "this is message-passing.md §9.3's pre-registered prior rather than a defect: "
                "on the clinical-only schema there is no structure to learn, and model 3's margin "
                "over model 2 is capacity rather than structure."
            )
        ),
        detail={
            "full_mean": full,
            "ablated_mean": ablated,
            "delta_full_minus_ablated": inputs.ablation["delta_full_minus_ablated"],
        },
    )


def _consistent_covariates(degree_probe: dict[str, object], *, direction: int) -> list[str]:
    """Covariates significant in `direction` on **every** fold, not just one.

    All five folds are fitted and recorded, so classifying a covariate as
    immortal-time or disease-burden off a single fold would let one fold's
    p-value drive the verdict's diagnosis. Requiring agreement across folds
    costs nothing here — the hazard ratios are extremely stable — and refuses to
    name a channel the data does not consistently support.
    """
    folds = degree_probe["folds"]
    assert isinstance(folds, list)
    if not folds:
        return []

    def qualifies(stats: dict[str, float]) -> bool:
        beyond_one = (
            stats["hazard_ratio"] < 1.0 if direction < 0 else stats["hazard_ratio"] > 1.0
        )
        return beyond_one and stats["p"] < 0.05

    per_fold = [
        {covariate for covariate, stats in fold["hazard_ratios"].items() if qualifies(stats)}
        for fold in folds
    ]
    return sorted(set.intersection(*per_fold))


def assess_degree_probe(inputs: GateInputs, config: GateConfig) -> Verdict:
    """PASS when the accrual counts alone stay weak — the exclusion rule held."""
    model_means = inputs.model_means
    probe = float(np.mean(_floats(inputs.degree_probe, "per_fold")))
    passed = probe < config.degree_probe_ceiling
    # Split by direction, because the two directions have opposite meanings.
    # Protective is encoder doc §0's immortal-time bias and is a defect;
    # harmful is disease burden and merely raises the floor.
    protective = _consistent_covariates(inputs.degree_probe, direction=-1)
    harmful = _consistent_covariates(inputs.degree_probe, direction=+1)
    return Verdict(
        diagnostic="degree_probe",
        verdict=PASS if passed else FAIL,
        statistic=probe,
        threshold=config.degree_probe_ceiling,
        rule=f"PASS if the probe's mean {METRIC} < {config.degree_probe_ceiling}",
        reading=(
            "Accrual alone is weak, so #4 §4's exclusion of the counts held and no model is "
            "materially reading follow-up duration through them."
            if passed
            else (
                "Accrual alone is strongly prognostic, so the floor a structural result has to "
                "clear is high -- read `decomposition` and the hazard-ratio directions before "
                "concluding anything about the models. A PROTECTIVE count is encoder doc §0's "
                "immortal-time bias and is a defect wherever a model can see it; a HARMFUL count "
                "is disease burden and is ordinary prognostic signal that raises the floor "
                "without implying a leak. The models' own exposure is the separate question: "
                "tests/test_diagnostics.py pins that none of these columns is in the shared "
                "design matrix, so a strong probe is only a leak where a model can reconstruct "
                "the count some other way."
            )
        ),
        detail={
            "floor_for_a_structural_result": probe,
            "model_means": model_means,
            "models_clearing_the_floor": sorted(
                model for model, mean in model_means.items() if mean > probe
            ),
            "margin_over_the_floor": {
                model: mean - probe for model, mean in sorted(model_means.items())
            },
            "decomposition": inputs.degree_probe.get("decomposition", {}),
            "protective_covariates_p_lt_0.05": protective,
            "harmful_covariates_p_lt_0.05": harmful,
            # A strong probe only implicates a model that can actually see the
            # count. None of these columns is in the shared design matrix
            # (pinned in tests/test_diagnostics.py), so the question is whether
            # a model can reconstruct one some other way.
            "model_exposure_to_the_counts": {
                "model1_baseline": "None. Its covariate set is staging and pathology only (#8).",
                "model2_tabular": (
                    "None directly. It carries Sample-type *proportions* (p_*), which are "
                    "ratios over a Subject's baseline Samples and do not recover n_samples."
                ),
                "model3_graph": (
                    "One bit, and it is worth stating precisely. Mean aggregation makes "
                    "degree invisible by design (encoder doc §2.3), so neither n_samples nor "
                    "n_diagnoses reaches the encoder as a number. But #11 split HAS_DIAGNOSIS "
                    "on isPrimaryDiagnosis, and an empty relation contributes exactly 0 while "
                    "a non-empty one does not -- so 'this Subject has at least one secondary "
                    "Diagnosis' (37.3% of Subjects) is structurally visible. That is a 1-bit "
                    "read on n_diagnoses > 1. Intervention is absent from every model entirely "
                    "(#4 §4), so the protective immortal-time term reaches nothing."
                ),
            },
            "note": (
                "A protective hazard ratio on an accrual count is the immortal-time signature "
                "even when the overall C is weak: more of the thing, lower hazard, because "
                "accruing it required surviving to accrue it (#4 §4 measured bare Intervention "
                "presence at HR 0.716, p = 8.5e-05)."
            ),
        },
    )


def assess_label_shuffle(inputs: GateInputs, config: GateConfig) -> Verdict:
    """PASS when every model sits at chance under the gating permutation scheme."""
    runs = inputs.label_shuffle["runs"]
    assert isinstance(runs, list)
    deviations = {
        str(run["model"]): abs(float(run[METRIC]["mean"]) - 0.5)
        for run in runs
        if run["scheme"] == config.shuffle_gate_scheme
    }
    if not deviations:
        raise ValueError(
            f"no shuffle run under the gating scheme {config.shuffle_gate_scheme!r}; "
            "the gate has nothing to judge"
        )
    worst_model = max(deviations, key=lambda model: deviations[model])
    worst = deviations[worst_model]
    passed = worst < config.shuffle_tolerance
    return Verdict(
        diagnostic="label_shuffle",
        verdict=PASS if passed else FAIL,
        statistic=worst,
        threshold=config.shuffle_tolerance,
        rule=(
            f"PASS if every model's mean {METRIC} is within {config.shuffle_tolerance} of 0.5 "
            f"under the {config.shuffle_gate_scheme!r} permutation"
        ),
        reading=(
            "No model can predict a permuted label, so nothing Survival-derived reached the "
            "feature side."
            if passed
            else (
                f"{worst_model} predicts a permuted label, which it cannot do without leakage. "
                "Encoder doc §0 names the likely cause: a :Survival property that survived into "
                "the feature export. Check extract.guards.SURVIVAL_DERIVED against the model's "
                "design matrix before reading any number from this pipeline."
            )
        ),
        detail={
            "deviation_from_chance_by_model": deviations,
            "worst_model": worst_model,
            "per_scheme": {
                f"{run['model']}/{run['scheme']}": {
                    METRIC: float(run[METRIC]["mean"]),
                    "pooled_harrell_c": float(run["pooled_harrell_c"]["mean"]),
                }
                for run in runs
            },
            "reading_the_within_study_scheme": (
                "Under 'within_study' a pooled C above 0.5 is expected and is not a leak: "
                "labels stay inside their Study, so Study-level survival differences survive "
                "and models 2/3 can still read them off Condition (R²_study = 0.948). The "
                "within-Study metric falling to chance under the same permutation is #4 §6's "
                "immunity argument demonstrated rather than asserted."
            ),
        },
    )


def assess(inputs: GateInputs, config: GateConfig) -> GateReport:
    """The three verdicts, in encoder doc §7's order."""
    return GateReport(
        config=config,
        verdicts=(
            assess_ablation(inputs, config),
            assess_degree_probe(inputs, config),
            assess_label_shuffle(inputs, config),
        ),
    )
