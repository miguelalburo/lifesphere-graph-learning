"""Fig. 5 — #13's three diagnostics against the thresholds declared before the run.

The panels are drawn on one shared x-axis (within-Study Harrell's C, the metric
all three diagnostics are judged in) so the reader can carry a distance from one
panel to the next. Each panel names its own verdict, and the verdict wording is
the gate's: **PASS means the control failed the way a control is supposed to.**
It is not a claim that a model is good, and the panel titles are worded to make
that hard to misread.

Two things this figure deliberately does not do. It does not draw the ablated
encoder as a rival to model 2 — severing the edges removes the encoder's access to
stage and cancer type themselves, so it sits *below* flattened on encoder doc
§4's ladder, and panel (a)'s subtitle says so. And it draws no bars from a
truncated baseline: every panel is points with their five folds, because the gap
between 0.50 and 0.60 in this metric is the whole subject of the figure and a
bar clipped at 0.45 would triple it by accident.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from . import inputs
from .style import MODEL_STYLE, CONTROL, CONTROL_LIGHT, GRID, RULE, row_points, save, subtitle

# The decomposition rows worth a panel, in the order they are read: the probe
# itself, then the single covariate that carries it, then the one channel model 3
# can see that models 1 and 2 cannot (#15).
DECOMPOSITION_ROWS = (
    ("all", "All three counts\n(the probe)"),
    ("only:log1p_nDiagnoses", "n diagnoses alone"),
    ("only:log1p_nSamples", "n samples alone"),
    ("only:log1p_nInterventions", "n interventions alone"),
    ("model3_visible:has_secondary_diagnosis", "has secondary diagnosis\n(model 3 sees this)"),
)


def figure_5(*, directory: Path | None = None) -> list[Path]:
    """The structure ablation, the degree probe and the label shuffle, with verdicts."""
    ablation = inputs.read_json("trust_gate", "structure_ablation.json")
    probe = inputs.read_json("trust_gate", "degree_probe.json")
    shuffle = inputs.read_json("trust_gate", "label_shuffle.json")
    verdicts = {
        entry["diagnostic"]: entry
        for entry in inputs.read_json("trust_gate", "gate.json")["verdicts"]
    }

    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(7.2, 8.6), height_ratios=[1.0, 1.9, 2.1], sharex=True
    )
    graph = MODEL_STYLE["model3_graph"]
    for axis in (top, middle, bottom):
        axis.set_title("", pad=20)
        axis.xaxis.grid(True, zorder=0)
        axis.set_axisbelow(True)
    bottom.set_xlim(0.44, 0.79)

    # ---- (a) structure ablation ---------------------------------------------
    row_points(top, 1, ablation["full_per_fold"], color=graph.color, marker=graph.marker)
    row_points(top, 0, ablation["ablated_per_fold"], color=CONTROL, marker="X")
    delta = ablation["delta_full_minus_ablated"]
    top.set_yticks([1, 0])
    top.set_yticklabels(["Model 3, full", "Model 3, edges ablated"], fontsize=8)
    top.set_ylim(-0.6, 1.6)
    top.set_title(
        f"a  Structure ablation — {verdicts['structure_ablation']['verdict']} "
        f"(Δ = {delta['mean']:+.3f}, threshold > {verdicts['structure_ablation']['threshold']})",
        pad=20,
    )
    subtitle(
        top,
        "Edges are load-bearing for reaching the features at all. The ablated encoder sits below "
        "model 2, so this is not\nevidence that structure beats flattening — that comparison is Fig. 3b.",
    )

    # ---- (b) degree probe ----------------------------------------------------
    ceiling = float(verdicts["degree_probe"]["threshold"])
    rows = [row for row in DECOMPOSITION_ROWS if row[0] in probe["decomposition"]]
    for position, (key, label) in zip(range(len(rows))[::-1], rows):
        entry = probe["decomposition"][key]
        color = graph.color if key.startswith("model3_visible") else CONTROL
        row_points(middle, position, entry["per_fold"], color=color, marker="D")
    middle.axvline(ceiling, color=RULE, linestyle="--", linewidth=0.9, zorder=1)
    middle.annotate(
        f"gate ceiling {ceiling:.2f}",
        xy=(ceiling, len(rows) - 0.55),
        ha="center",
        fontsize=7,
        color=RULE,
    )
    # Models 1 and 2 sit 0.003 C apart, so their labels are stacked rather than
    # placed at a shared height where they would print on top of each other.
    model_means = verdicts["degree_probe"]["detail"]["model_means"]
    for index, (model_key, mean) in enumerate(sorted(model_means.items(), key=lambda item: item[1])):
        model = MODEL_STYLE[model_key]
        middle.axvline(mean, color=model.color, linestyle=":", linewidth=1.0, zorder=1)
        middle.annotate(
            f"{model_key.split('_')[0]} {mean:.3f}",
            xy=(mean, len(rows) - 0.55 - 0.42 * index),
            ha="center",
            va="center",
            fontsize=6.5,
            color=model.color,
        )
    middle.set_yticks(range(len(rows))[::-1])
    middle.set_yticklabels([label for _, label in rows], fontsize=7.5)
    middle.set_ylim(-0.6, len(rows) - 0.15)
    middle.set_title(
        f"b  Degree probe — {verdicts['degree_probe']['verdict']} "
        f"(probe = {probe['mean']:.3f}, ceiling {ceiling:.2f})",
        pad=20,
    )
    subtitle(
        middle,
        "Accrual alone is prognostic, so the floor a model must clear is high — not that a model "
        "leaked: none of these\ncounts is in the shared design matrix (#4 §4). Dotted verticals mark "
        "the three models' own scores.",
    )

    # ---- (c) label shuffle ---------------------------------------------------
    tolerance = float(verdicts["label_shuffle"]["threshold"])
    bottom.axvspan(0.5 - tolerance, 0.5 + tolerance, color=GRID, alpha=0.5, zorder=0, linewidth=0)
    bottom.axvline(0.5, color=CONTROL, linewidth=0.9, zorder=1)
    schemes = ("global", "within_study")
    runs = {(run["model"], run["scheme"]): run for run in shuffle["runs"]}
    labels: list[str] = []
    position = len(schemes) * len(MODEL_STYLE) - 1
    for scheme in schemes:
        for model_key in ("model1_baseline", "model2_tabular", "model3_graph"):
            model = MODEL_STYLE[model_key]
            run = runs[(model_key, scheme)]
            row_points(
                bottom,
                position,
                run["within_study_harrell_c"]["per_fold"],
                color=model.color if scheme == "global" else CONTROL_LIGHT,
                marker=model.marker,
            )
            labels.append(f"{model_key.split('_')[0]}, {scheme.replace('_', '-')}")
            position -= 1
    bottom.set_yticks(range(len(labels))[::-1])
    bottom.set_yticklabels(labels, fontsize=7.5)
    bottom.set_ylim(-0.6, len(labels) - 0.15)
    bottom.set_title(
        f"c  Label shuffle — {verdicts['label_shuffle']['verdict']} "
        f"(worst model {verdicts['label_shuffle']['statistic']:.3f} from chance, "
        f"tolerance {tolerance})",
        pad=20,
    )
    subtitle(
        bottom,
        "Shaded: 0.5 ± 0.05. 'Global' is the scheme the gate judges; 'within-study' destroys what "
        "this metric measures\nand is drawn greyed, as context rather than as a second verdict.",
    )
    bottom.set_xlabel("Within-Study Harrell's C")

    fig.suptitle(
        "A PASS means the control failed the way a control is supposed to — it says nothing about "
        "whether a result is good.",
        fontsize=7.5,
        color=RULE,
        x=0.012,
        ha="left",
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.982), h_pad=2.2)
    return save(fig, "fig5_trust_gate", directory=directory)
