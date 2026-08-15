"""Fig. 3 and Fig. 4 — the three-model comparison, pooled and per Study.

Fig. 3 is the study's headline display item, and its two panels answer two
different questions that are easy to conflate. Panel (a) is *where each model
lands*, with all five folds shown rather than a mean alone. Panel (b) is *what
separates them*, which is a paired quantity and not the difference of the two
means in panel (a): every model is scored on the identical persisted folds (#7),
so the fold-by-fold difference is the estimate with the fold-to-fold variance
divided out. Panel (b) carries the ±0.02 resolution band across it, because the
comparison the study exists to run (model 3 - model 2) is inside that band and a
reader must see that before reading the point estimate.

Fig. 4 is the same headline metric taken apart by Study, ordered by event count
so the reader sees immediately that the Studies with the widest model-to-model gaps
are not always the ones with the most events to estimate them from.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import inputs
from .style import MODELS, CONTROL, GRID, RULE, row_points, save


def figure_3(*, directory: Path | None = None) -> list[Path]:
    """Within-Study Harrell C by model, and the paired differences between them."""
    scores = {model.key: inputs.load_model_scores(model.key) for model in MODELS}
    # #13's degree probe: accrual counts alone reach this within-Study C, so it
    # is the floor a model clears before any of its margin is its own.
    floor = float(inputs.read_json("trust_gate", "degree_probe.json")["mean"])

    deltas = [
        ("Model 3 − Model 1\n(graph vs baseline)", inputs.delta("model3_graph", "model1_baseline")),
        ("Model 3 − Model 2\n(graph vs flattened)", inputs.delta("model3_graph", "model2_tabular")),
        ("Model 2 − Model 1\n(flattened vs baseline)", inputs.delta("model2_tabular", "model1_baseline")),
    ]

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.2), width_ratios=[1.0, 1.05])

    # ---- (a) where each model lands -------------------------------------------
    positions = list(range(len(MODELS)))[::-1]  # model 1 at the top
    left.set_xlim(0.48, 0.79)
    for position, model in zip(positions, MODELS):
        score = scores[model.key]
        row_points(
            left,
            position,
            score.per_fold,
            color=model.color,
            marker=model.marker,
            spread=0.20,
            value_format=f"{{mean:.3f}} ± {score.std:.3f}",
        )
        left.plot(
            [score.mean - score.std, score.mean + score.std],
            [position, position],
            color=model.color,
            linewidth=1.2,
            zorder=2,
        )

    left.axvline(0.5, color=RULE, linestyle=":", linewidth=0.9, zorder=1)
    left.axvline(floor, color=RULE, linestyle="--", linewidth=0.9, zorder=1)
    left.annotate("chance", xy=(0.5, 2.42), ha="center", fontsize=7, color=RULE)
    left.annotate(
        f"degree-probe\nfloor {floor:.3f}", xy=(floor, 2.30), ha="center", fontsize=7, color=RULE
    )

    left.set_yticks(positions)
    left.set_yticklabels([model.label.split(" — ")[0] for model in MODELS])
    left.set_ylim(-0.5, 3.0)
    left.set_xlabel("Within-Study Harrell's C")
    left.set_title("a  Model scores (mean ± SD over 5 folds)")
    left.xaxis.grid(True, zorder=0)
    left.set_axisbelow(True)

    # ---- (b) what separates them --------------------------------------------
    delta_positions = list(range(len(deltas)))[::-1]
    right.set_xlim(-0.032, 0.058)
    right.axvspan(
        -inputs.RESOLUTION_LIMIT,
        inputs.RESOLUTION_LIMIT,
        color=GRID,
        alpha=0.45,
        zorder=0,
        linewidth=0,
    )
    right.axvline(0.0, color=CONTROL, linewidth=0.9, zorder=1)

    for position, (label, paired) in zip(delta_positions, deltas):
        low, high = paired.ci95
        resolved = paired.excludes_zero and abs(paired.mean) > inputs.RESOLUTION_LIMIT
        right.errorbar(
            paired.mean,
            position,
            xerr=[[paired.mean - low], [high - paired.mean]],
            fmt="D",
            color="black" if resolved else CONTROL,
            markersize=5,
            capsize=3,
            elinewidth=1.2,
            zorder=3,
        )
        right.annotate(
            f"{paired.mean:+.4f}  [{low:+.4f}, {high:+.4f}]   {paired.folds_won}/"
            f"{paired.n_folds} folds",
            xy=(-0.031, position + 0.32),
            fontsize=7,
            color="black" if resolved else CONTROL,
        )

    right.set_yticks(delta_positions)
    right.set_yticklabels([label for label, _ in deltas], fontsize=7.5)
    right.set_ylim(-0.5, 3.0)
    right.set_xlabel(
        "Paired difference in within-Study Harrell's C\n"
        "shaded: ±0.02, the resolution limit of a 5-fold design (#3 §5)"
    )
    right.set_title("b  Paired differences (95% CI)")
    right.xaxis.grid(True, zorder=0)
    right.set_axisbelow(True)

    fig.tight_layout()
    return save(fig, "fig3_model_comparison", directory=directory)


def figure_4(*, directory: Path | None = None) -> list[Path]:
    """Per-Study Harrell C for the three models, ordered by event count."""
    per_model = {model.key: inputs.load_per_study(model.key) for model in MODELS}
    sizes = inputs.study_sizes()
    n_folds = {model.key: inputs.folds_scoring_study(model.key) for model in MODELS}
    partial = {
        study
        for study in sizes
        if any(n_folds[model.key].get(study, 0) < 5 for model in MODELS)
    }

    scored = sorted(
        (study for study in sizes if any(study in per_model[model.key] for model in MODELS)),
        key=lambda study: sizes[study][1],
    )
    excluded = sorted(study for study in sizes if study not in scored)

    fig, axis = plt.subplots(figsize=(6.4, 7.2))
    for position, study in enumerate(scored):
        values = [per_model[model.key].get(study) for model in MODELS]
        present = [value for value in values if value is not None]
        if len(present) > 1:
            axis.plot(
                [min(present), max(present)],
                [position, position],
                color=GRID,
                linewidth=2.4,
                solid_capstyle="round",
                zorder=1,
            )
        for model, value in zip(MODELS, values):
            if value is None:
                continue
            axis.scatter(
                value,
                position,
                marker=model.marker,
                s=26,
                color=model.color,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                label=model.label if position == 0 else None,
            )

    axis.axvline(0.5, color=RULE, linestyle=":", linewidth=0.9, zorder=0)
    axis.annotate("chance", xy=(0.5, len(scored) - 0.4), fontsize=7, color=RULE, ha="center")

    axis.set_yticks(range(len(scored)))
    axis.set_yticklabels(
        [
            f"{study.replace('TCGA-', '')}{'*' if study in partial else ''}"
            f"  ({sizes[study][1]}/{sizes[study][0]})"
            for study in scored
        ],
        fontsize=7.5,
    )
    axis.set_ylim(-0.8, len(scored) - 0.2)
    axis.set_xlim(0.3, 1.0)
    axis.set_xlabel("Within-Study Harrell's C (mean over the 5 folds)")
    axis.set_ylabel("Study  (events / Subjects), ordered by events")
    axis.set_title("Per-Study discrimination, by model")
    axis.xaxis.grid(True, zorder=0)
    axis.set_axisbelow(True)
    axis.legend(loc="lower left", bbox_to_anchor=(0.0, -0.115), ncols=3, fontsize=7.5)

    notes = []
    if partial:
        notes.append(
            "* scored in fewer than 5 folds — too few events for a comparable pair in the rest"
        )
    if excluded:
        notes.append(
            "excluded from the within-Study metric in every fold: "
            + ", ".join(study.replace("TCGA-", "") for study in excluded)
        )
    if notes:
        axis.annotate(
            "\n".join(notes),
            xy=(0.0, -0.155),
            xycoords="axes fraction",
            ha="left",
            fontsize=7,
            color=RULE,
        )

    fig.tight_layout()
    return save(fig, "fig4_per_study", directory=directory)
