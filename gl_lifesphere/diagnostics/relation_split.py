"""#15: is model 3's margin over model 2 the secondary-diagnosis channel, or message passing?

#12 records model 3 beating model 2 by **+0.0180** within-Study Harrell C, and
~85% of model 3's advantage over model 1 as a structural term whose interval
crosses zero. #13's degree probe then found a specific channel that could
account for the whole of it.

**The channel.** Mean aggregation makes degree invisible by design (encoder doc
§2.3), so neither `n_samples` nor `n_diagnoses` reaches the encoder as a number.
But #11 split `HAS_DIAGNOSIS` on `isPrimaryDiagnosis`, and a bias-free
convolution over an **empty** relation contributes exactly `0` while a non-empty
one does not — so "this Subject has at least one secondary Diagnosis" is
structurally visible to the encoder. That is a 1-bit read on `n_diagnoses > 1`,
true of 37.3% of Subjects, and fitted alone it measures within-Study C =
**0.5588**, higher than the full diagnosis count (0.5569). Models 1 and 2 cannot
see it: `features/raw.py` joins from `diagnosis_primary`, one row per Subject,
and excludes `nDiagnoses` explicitly.

So model 3 uniquely holds a prognostic channel worth 0.5588 on its own and beats
model 2 by +0.0180. This module tests the alternative explanation — an extra
feature rather than message passing — by re-running model 3 with
`split_primary_relation=False` and pairing it fold-for-fold against the recorded
split run and against model 2.

**This is not a proposal to change the model.** #11 adopted the split on
measurement: unsplit, the root mean-aggregates the primary Diagnosis together
with 0–4 secondaries carrying the training-fold within-Study median stage, and
the Diagnosis-set mean sits 0.35 sd off the primary's own stage for the 37.3% of
Subjects with several. The model stays as it is whatever this returns; what
changes is what #14 may claim about the margin.

**And it is not a gate item.** #13's verdict is over encoder doc §7's three
diagnostics and nothing else. This is a follow-up the gate raised, so it uses
its own verdict vocabulary and the gate has no input path to it — a cheaper
later run must not be able to overturn a declared FAIL.

**The comparison is not perfectly clean, and the artefact says so.** Collapsing
to a single `HAS_DIAGNOSIS` relation does not remove the multiplicity signal
outright: a Subject with 3 Diagnoses still mean-aggregates to a different value
than one with 1. It removes the *clean* presence/absence indicator, not the
channel. A null therefore **bounds** the effect rather than eliminating it, and
a positive result is the stronger reading. `tests/test_relation_split.py` pins
the residual channel rather than trusting the prose. (One strictly smaller bit
also survives the collapse: a Subject with *no* Diagnosis still has an empty
relation. The frozen cohort holds exactly one such Subject in 6,811 — #4 §8 —
so it is a curiosity rather than a channel.)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..constructions import cache
from ..evaluation.compare import PairedDelta, paired_delta
from ..evaluation.splits import N_SPLITS, fold_split, load_folds
from ..extract.connection import REPO_ROOT
from ..extract.store import PROCESSED_DIR
from ..features import assemble_raw_frame
from ..models import training
from ..models.graph.train import MODEL as GRAPH_MODEL
from ..models.graph.train import CONSTRUCTION as GRAPH_CONSTRUCTION
from ..models.graph.train import FoldResult, GraphModelConfig, run_fold
from ..models.tabular.train import MODEL as TABULAR_MODEL
from ..survival import metrics
from ..survival.targets import SurvivalTarget, load_targets
from . import recorded

PROBE = "relation_split"
RESULT_FILE = "relation_split.json"
METRIC = metrics.HEADLINE

# The one field that varies between the recorded model 3 and this probe.
VARIES = "split_primary_relation"

# The collapsed construction is cached beside model 3's rather than over it.
# `cache.cache_path` keys the file on the fold alone, so the two builds would
# otherwise evict each other every run — correctly, since the fingerprint
# carries the build options and a mismatch rebuilds, but at the cost of a full
# rebuild for whichever ran second. A probe should not leave the model it is
# comparing against slower than it found it.
UNSPLIT_CACHE_DIR = PROCESSED_DIR / "subject_subgraph_unsplit"

# Deliberately not PASS/FAIL. #13's gate owns that vocabulary, and a follow-up
# sharing it would invite a reader to add this to the gate's tally — which is
# exactly what a probe that cannot gate anything must not do.
ATTRIBUTED = "ATTRIBUTED"
NOT_ATTRIBUTED = "NOT ATTRIBUTED"
REVERSED = "REVERSED"

COLLAPSE_CAVEAT = (
    "Collapsing to a single HAS_DIAGNOSIS relation does not remove the multiplicity signal "
    "outright -- a Subject with 3 Diagnoses still mean-aggregates to a different value than one "
    "with 1. It removes the clean presence/absence indicator, not the channel. So a null result "
    "BOUNDS the effect rather than eliminating it, and a positive result is the stronger reading."
)

CAPACITY_CAVEAT = (
    "Capacity is NOT held fixed, unlike #13's structure ablation. Collapsing HAS_PRIMARY/"
    "HAS_SECONDARY_DIAGNOSIS into one relation merges their weight matrices, so the unsplit "
    "encoder is smaller -- the ablation could keep every W_r because it emptied edge contents "
    "rather than removing a relation, and this manipulation cannot. Both differences push the "
    "unsplit run down, so a measured cost is an UPPER bound on the presence bit's contribution "
    "and the residual multiplicity channel makes it a partial removal besides. Read "
    "parameters_per_fold for the size of the capacity term."
)

ASYMMETRY_FOR_14 = (
    "State this wherever model3 - model2 is reported (#15): model 3 holds a prognostic channel models 1 "
    "and 2 do not. #11's relation split makes 'has at least one secondary Diagnosis' structurally "
    "visible to the encoder -- a 1-bit read on n_diagnoses > 1 (37.3% of Subjects) worth "
    "within-Study C = 0.5588 fitted alone (#13). Models 1 and 2 join from diagnosis_primary, one "
    "row per Subject, and exclude nDiagnoses explicitly."
)


@dataclass(frozen=True)
class RelationSplitConfig:
    """Declared before the run, like the gate's thresholds and for the same reason.

    `resolution_limit` is not a bar this probe passes or fails — it is #3 §5's
    ~0.02, the smallest difference 5 correlated folds can resolve, and it is
    here because the quantity being decomposed (+0.0180) already sits below it.
    A share of an unresolvable margin is still a share of an unresolvable
    margin, and the result file has to say so next to the number rather than
    leave it to a reader who remembers #3.
    """

    model: str = GRAPH_MODEL
    construction: str = GRAPH_CONSTRUCTION
    endpoint: str = training.LOCKED_ENDPOINT
    split: str = field(default_factory=training.locked_split_name)
    seed: int = 0
    resolution_limit: float = 0.02
    comparison_model: str = TABULAR_MODEL

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RelationSplitConfig":
        """Ignores keys the dataclass does not declare (e.g. a `_comment`)."""
        kwargs = {f.name: payload[f.name] for f in fields(cls) if f.name in payload}
        return cls(**kwargs)  # type: ignore[arg-type]

    def check_against(self, graph_config: GraphModelConfig) -> None:
        """Raise unless this config declares the run it will actually get.

        Model, endpoint and split go through `training.check_run_identity` — the
        one place that rule lives, rather than a third hand-rolled copy of it
        beside `gate.GateConfig.check`'s. What this adds is the pair that check
        cannot see: `construction` and `seed` are declared here but *supplied*
        by model 3's own config, so a probe file naming a different seed would
        otherwise record a number produced under another one. That is the same
        quiet-wrong-answer class `tests/README.md` puts first.
        """
        training.check_run_identity(
            model=self.model,
            expected_model=GRAPH_MODEL,
            endpoint=self.endpoint,
            split=self.split,
        )
        mismatches = {
            name: (declared, supplied)
            for name, declared, supplied in (
                ("construction", self.construction, graph_config.construction),
                ("seed", self.seed, graph_config.seed),
            )
            if declared != supplied
        }
        if mismatches:
            detail = ", ".join(
                f"{name}: {PROBE} says {declared!r}, model 3's config supplies {supplied!r}"
                for name, (declared, supplied) in sorted(mismatches.items())
            )
            raise ValueError(
                f"{PROBE} declares a run it will not get -- {detail}. The probe takes these "
                "from experiments/configs/model3_graph.json; change the config to match."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "construction": self.construction,
            "endpoint": self.endpoint,
            "split": self.split,
            "seed": self.seed,
            "resolution_limit": self.resolution_limit,
            "comparison_model": self.comparison_model,
        }


def probe_config(graph_config: GraphModelConfig) -> GraphModelConfig:
    """Model 3's own config with the relation split collapsed, and nothing else touched.

    Derived rather than read from a second config file, for the reason
    `run_structure_ablation` derives its own: everything the encoder and the
    construction see then comes from one dataclass, so the two sides of the
    comparison cannot drift apart between edits. #15 describes this as "one
    config edit in `experiments/configs/model3_graph.json`" — editing that file in
    place would instead change the recorded model's declared config and break the
    comparability check against the run already on disk.
    """
    # Spelled out rather than splatted through `VARIES`, so the type checker
    # sees a real field name. The two cannot drift unnoticed:
    # `check_probe_comparable` names `VARIES` as the one permitted difference,
    # so any disagreement makes the comparability guard raise rather than pass.
    return replace(graph_config, split_primary_relation=False)


def check_probe_comparable(model: recorded.RecordedModel, graph_config: GraphModelConfig) -> None:
    """Refuse to pair unless the recorded model 3 differs from the probe in `VARIES` alone."""
    recorded.check_comparable(
        model, recorded.comparable_fields(probe_config(graph_config)), varies=(VARIES,)
    )


def check_comparison_model(
    comparison: recorded.RecordedModel, graph_config: GraphModelConfig
) -> None:
    """Refuse to pair against the comparison model unless it is the same experiment.

    Model 2 and model 3 are different model classes with different config
    dataclasses, so `check_comparable`'s field-by-field rule cannot apply — its
    whole premise is that the two runs are the same model. What *must* agree is
    what makes a fold-paired comparison meaningful at all: one endpoint and one
    persisted split. Pairing fold 2 of an OS run against fold 2 of a PFI run
    would produce a number rather than an error.
    """
    if not comparison.config:
        raise ValueError(
            f"{comparison.model}'s summary.json carries no config block, so there is nothing to "
            "check this run against. Re-run the model so its config is recorded."
        )
    mismatches = {
        key: (comparison.config[key], value)
        for key, value in (("endpoint", graph_config.endpoint), ("split", graph_config.split))
        if key in comparison.config and comparison.config[key] != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}: {comparison.model} recorded {was!r}, model 3 is {now!r}"
            for key, (was, now) in sorted(mismatches.items())
        )
        raise ValueError(
            f"{comparison.model} was not scored on the same experiment as model 3 -- {detail}. "
            "Pairing their folds would compare two different runs."
        )


@dataclass(frozen=True)
class Attribution:
    """How much of model 3's margin over the comparison model the collapsed relation costs.

    The three deltas are all paired across the same 5 folds, which makes the
    decomposition exact rather than approximate: fold by fold,
    `(split - comparison) - (unsplit - comparison) == split - unsplit`, so
    `attributed_share` is a share of the margin rather than a ratio of two
    independently estimated quantities.
    """

    config: RelationSplitConfig
    split_per_fold: tuple[float, ...]
    unsplit_per_fold: tuple[float, ...]
    comparison_per_fold: tuple[float, ...]
    # split - unsplit: what removing the clean presence bit costs model 3.
    cost_of_collapsing: PairedDelta
    # split - comparison: #12's +0.0180, recomputed from the recorded folds.
    split_margin: PairedDelta
    # unsplit - comparison: the margin that survives without the clean bit.
    unsplit_margin: PairedDelta
    verdict: str
    reading: str
    # Set only on a null: the upper end of the paired interval on the cost, i.e.
    # the most the channel could be worth and still have produced this result.
    bound: float | None
    attributed_share: float | None
    margin_below_resolution_limit: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "probe": PROBE,
            "issue": "#15",
            "question": (
                "Is model 3's margin over model 2 the secondary-diagnosis channel rather than "
                "message passing?"
            ),
            "metric": METRIC,
            "verdict": self.verdict,
            "rule": (
                f"ATTRIBUTED if the paired 95% interval on (split - unsplit) in {METRIC} "
                "excludes zero and its mean is positive; REVERSED if it excludes zero and the "
                "mean is negative; NOT ATTRIBUTED otherwise, reporting the interval's upper end "
                "as a bound. The interval is the one evaluation.compare labels anti-conservative "
                "-- see the caveat carried on every delta below -- so ATTRIBUTED is the verdict "
                "that side errs toward, and it is the one this run did not reach."
            ),
            "reading": self.reading,
            "caveat": COLLAPSE_CAVEAT,
            "for_14": ASYMMETRY_FOR_14,
            "config": self.config.to_dict(),
            "comparison_model": self.config.comparison_model,
            "split_per_fold": list(self.split_per_fold),
            "unsplit_per_fold": list(self.unsplit_per_fold),
            "comparison_per_fold": list(self.comparison_per_fold),
            "split_mean": float(np.mean(self.split_per_fold)),
            "unsplit_mean": float(np.mean(self.unsplit_per_fold)),
            "comparison_mean": float(np.mean(self.comparison_per_fold)),
            "cost_of_collapsing": self.cost_of_collapsing.to_dict(),
            "split_margin_over_comparison": self.split_margin.to_dict(),
            "unsplit_margin_over_comparison": self.unsplit_margin.to_dict(),
            "attributed_share": self.attributed_share,
            "bound_on_the_channel": self.bound,
            "margin_below_resolution_limit": self.margin_below_resolution_limit,
            "how_to_read_the_share": (
                "attributed_share is (split - unsplit) / (split - comparison), both paired over "
                "the same 5 folds, so it is an exact decomposition of the margin rather than a "
                "ratio of two independent estimates. It is a POINT ESTIMATE and inherits the "
                "uncertainty of cost_of_collapsing: where the verdict is NOT ATTRIBUTED that "
                "interval spans zero, so the share must never be quoted without it. It is null "
                "when the margin is at or below zero -- there is then nothing to explain. "
                "margin_below_resolution_limit flags "
                f"that the margin itself is under #3 §5's ~{self.config.resolution_limit} "
                "resolution limit, in which case a share of it inherits that unresolvability."
            ),
        }


def _reading(verdict: str, bound: float | None) -> str:
    """The reading each outcome implies, written where the rule is rather than at the call site."""
    if verdict == ATTRIBUTED:
        return (
            "Collapsing the relation split costs model 3 a difference the folds resolve, so at "
            "least part of model 3's margin over the comparison model is the secondary-diagnosis "
            "channel rather than message passing. #12's structural term is weakened by the "
            "attributed share, and #14 must report the margin with that subtraction stated. "
            "Read the size with both confounds in mind, because they push in opposite "
            "directions: the collapse removes only the clean presence bit and not multiplicity, "
            "which understates the channel -- but it also merges two relations into one and so "
            "removes their weight matrices, which inflates the cost with a capacity term. See "
            "capacity_is_not_held_fixed."
        )
    if verdict == REVERSED:
        return (
            "Collapsing the relation split *improves* model 3, which contradicts #11's own "
            "measurement (unsplit, the Diagnosis-set mean sits 0.35 sd off the primary's stage "
            "for the 37.3% of Subjects with several, and stage is the strongest prognostic "
            "feature in the study). Treat this as a finding about the model rather than about the "
            "channel, and re-open #11's decision before #14 quotes either number."
        )
    assert bound is not None, "a null verdict always carries the interval that produced it"
    return (
        "The folds do not resolve a cost to collapsing the relation split, so the "
        "secondary-diagnosis channel is not established as the explanation for model 3's margin. "
        f"This BOUNDS the channel rather than eliminating it: at most {bound:.4f} C on the upper "
        "end of the paired interval, and the collapse removes only the clean presence/absence "
        "indicator -- multiplicity still reaches the root through mean aggregation. #14 should "
        "report model 3's margin as surviving this control while stating the asymmetry, not as "
        "cleared of it."
    )


def assess_attribution(
    *,
    split_per_fold: list[float] | tuple[float, ...],
    unsplit_per_fold: list[float] | tuple[float, ...],
    comparison_per_fold: list[float] | tuple[float, ...],
    config: RelationSplitConfig,
) -> Attribution:
    """Turn three fold-aligned score vectors into the attribution and its reading.

    Pure, and separated from the training for the reason `gate.assess` is: a
    verdict that can only be reached by retraining 5 folds is a verdict nobody
    can re-check.
    """
    split = list(split_per_fold)
    unsplit = list(unsplit_per_fold)
    comparison = list(comparison_per_fold)

    cost = paired_delta(split, unsplit)
    split_margin = paired_delta(split, comparison)
    unsplit_margin = paired_delta(unsplit, comparison)

    if not cost.excludes_zero:
        verdict = NOT_ATTRIBUTED
    elif cost.mean > 0.0:
        verdict = ATTRIBUTED
    else:
        verdict = REVERSED

    bound = cost.ci95[1] if verdict == NOT_ATTRIBUTED else None
    # A margin at or below zero leaves nothing to apportion, and a share of it
    # would read as meaningful while being an artefact of the sign.
    share = cost.mean / split_margin.mean if split_margin.mean > 0.0 else None

    return Attribution(
        config=config,
        split_per_fold=tuple(split),
        unsplit_per_fold=tuple(unsplit),
        comparison_per_fold=tuple(comparison),
        cost_of_collapsing=cost,
        split_margin=split_margin,
        unsplit_margin=unsplit_margin,
        verdict=verdict,
        reading=_reading(verdict, bound),
        bound=bound,
        attributed_share=share,
        margin_below_resolution_limit=abs(split_margin.mean) < config.resolution_limit,
    )


@dataclass(frozen=True)
class RelationSplitResult:
    """The attribution, plus the unsplit run that produced half of it."""

    attribution: Attribution
    unsplit_config: GraphModelConfig
    # Encoder sizes on both sides, fold by fold. Recorded because collapsing a
    # relation removes its `W_r`, so this comparison -- unlike the structure
    # ablation's -- does not hold capacity fixed, and a reader must be able to
    # see how large that second difference is rather than take the prose for it.
    split_parameters: tuple[int, ...] = ()
    unsplit_parameters: tuple[int, ...] = ()
    # Empty is legitimate: `assess_attribution` needs the three score vectors
    # and nothing else, so a reading can be re-checked without training.
    unsplit_folds: tuple[FoldResult, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            **self.attribution.to_dict(),
            "description": (
                "Model 3 retrained with split_primary_relation=False and paired fold-for-fold "
                "against the recorded split run and against the comparison model. Identical "
                "features, seed, architecture and folds -- the relation split is the only "
                "difference (#15)."
            ),
            "unsplit_config": recorded.comparable_fields(self.unsplit_config),
            "capacity_is_not_held_fixed": CAPACITY_CAVEAT,
            "parameters_per_fold": {
                "split": list(self.split_parameters),
                "unsplit": list(self.unsplit_parameters),
                "removed": [
                    before - after
                    for before, after in zip(self.split_parameters, self.unsplit_parameters)
                ],
            },
            "folds": [fold.to_dict() for fold in self.unsplit_folds],
        }


def run_relation_split(
    graph_config: GraphModelConfig,
    config: RelationSplitConfig | None = None,
    *,
    records: cache.SubjectRecords | None = None,
    raw: pd.DataFrame | None = None,
    targets: SurvivalTarget | None = None,
    graph_model: recorded.RecordedModel | None = None,
    comparison: recorded.RecordedModel | None = None,
    cache_dir: Path | None = None,
    on_fold: Callable[[FoldResult], None] | None = None,
    use_cache: bool = True,
) -> RelationSplitResult:
    """Train the collapsed model across all 5 folds and attribute model 3's margin.

    The split side and the comparison side are both read from `results/metrics/`
    rather than retrained: a model's fold is deterministic given its config and
    the persisted split (`recorded`'s docstring), so retraining them would buy
    an hour and a chance to diverge.

    `on_fold` fires as each fold finishes, so the caller can persist it before
    the next one starts. The first run of this probe builds all 5 constructions
    from scratch into a cache directory of its own, which is exactly the cost
    `models/graph/__main__` streams its fold files to avoid.
    """
    probe = config if config is not None else RelationSplitConfig()
    probe.check_against(graph_config)

    model3 = graph_model if graph_model is not None else recorded.load_model(GRAPH_MODEL)
    comparison_model = (
        comparison if comparison is not None else recorded.load_model(probe.comparison_model)
    )
    check_probe_comparable(model3, graph_config)
    check_comparison_model(comparison_model, graph_config)

    unsplit_config = probe_config(graph_config)
    subject_records = records if records is not None else cache.load_subject_records()
    raw_frame = raw if raw is not None else assemble_raw_frame()
    survival_targets = targets if targets is not None else load_targets()
    folds = load_folds()

    unsplit: list[FoldResult] = []
    for outer_fold in range(N_SPLITS):
        result = run_fold(
            outer_fold,
            records=subject_records,
            raw=raw_frame,
            targets=survival_targets,
            split=fold_split(folds, outer_fold),
            config=unsplit_config,
            cache_dir=cache_dir if cache_dir is not None else UNSPLIT_CACHE_DIR,
            use_cache=use_cache,
            # Kept on, unlike the structure ablation's. This is model 3 with one
            # relation collapsed, not a control designed to score at chance:
            # every feature is still reachable from the root, so a training-fold
            # risk that fails to discriminate is a sign inversion and should
            # raise exactly as it would for the model itself (#3 §6).
            expect_discrimination=True,
        )
        unsplit.append(result)
        if on_fold is not None:
            on_fold(result)

    return RelationSplitResult(
        attribution=assess_attribution(
            split_per_fold=model3.metric(METRIC),
            unsplit_per_fold=[float(f.fold_metrics.within_study.cindex) for f in unsplit],
            comparison_per_fold=comparison_model.metric(METRIC),
            config=probe,
        ),
        unsplit_config=unsplit_config,
        split_parameters=_recorded_parameter_counts(model3),
        unsplit_parameters=tuple(_n_parameters(f.construction) for f in unsplit),
        unsplit_folds=tuple(unsplit),
    )


def _recorded_parameter_counts(model: recorded.RecordedModel) -> tuple[int, ...]:
    """Each recorded fold's encoder size, or empty if the model did not record one.

    Tolerant rather than strict: `construction.n_parameters` arrived with #12,
    and a run recorded before it is still perfectly usable for the score vectors
    this probe actually pairs on. An absent capacity figure should leave the
    field empty, not refuse the comparison.
    """
    sizes = []
    for fold in model.folds:
        construction = fold.get("construction")
        if not isinstance(construction, dict) or "n_parameters" not in construction:
            return ()
        sizes.append(_n_parameters(construction))
    return tuple(sizes)


def _n_parameters(construction: dict[str, object]) -> int:
    value = construction["n_parameters"]
    assert isinstance(value, (int, float, str))
    return int(value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "relation_split.json"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "experiments" / "configs" / f"{GRAPH_MODEL}.json"
DEFAULT_OUT = REPO_ROOT / "results" / "metrics" / PROBE


def main(
    config_path: Path = DEFAULT_CONFIG,
    model_config_path: Path = DEFAULT_MODEL_CONFIG,
    out_dir: Path = DEFAULT_OUT,
    *,
    use_cache: bool = True,
) -> dict[str, object]:
    """Run the probe and write `relation_split.json`.

    Its own command and its own output directory rather than a fourth `--only`
    on `python -m gl_lifesphere.diagnostics`: that CLI's no-argument path runs
    the three §7 diagnostics and then assembles `gate.json` from exactly what it
    wrote, and a follow-up sitting inside that flow would be one edit away from
    becoming a gate input.
    """
    config = RelationSplitConfig.from_dict(json.loads(config_path.read_text()))
    # Read from model 3's own config file, never from a copy here — the same rule
    # `gate.GateConfig.model_configs` follows, and for the same reason: a second
    # place to change a hyperparameter is a second place to forget.
    graph_config = GraphModelConfig.from_dict(json.loads(model_config_path.read_text()))

    out_dir.mkdir(parents=True, exist_ok=True)

    def persist(fold: FoldResult) -> None:
        (out_dir / f"fold_{fold.fold}.json").write_text(
            json.dumps(fold.to_dict(), indent=2, default=str) + "\n"
        )

    result = run_relation_split(graph_config, config, on_fold=persist, use_cache=use_cache)
    payload = result.to_dict()
    (out_dir / RESULT_FILE).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="#15: re-run model 3 with the HAS_DIAGNOSIS relation split collapsed."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    written = main(args.config, args.model_config, args.out, use_cache=not args.no_cache)
    headline = ("verdict", "rule", "reading", "cost_of_collapsing", "attributed_share")
    print(json.dumps({key: written[key] for key in headline}, indent=2))
