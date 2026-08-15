"""Fig. S1 — #15's probe: is model 3's margin the secondary-diagnosis channel?

Kept out of Fig. 5 for the same reason `relation_split.json` is kept out of
`metrics/trust_gate/`: this is a follow-up the gate raised, not a fourth verdict,
and its vocabulary is ATTRIBUTED / NOT ATTRIBUTED / REVERSED rather than
PASS / FAIL. Drawing it beside the three diagnostics would let a reader carry the
gate's wording onto it.

The panel that matters is the right one, and it is a bound rather than a null:
collapsing HAS_PRIMARY/SECONDARY_DIAGNOSIS back into one relation removes the
clean presence/absence indicator but not multiplicity, which still reaches the
root through mean aggregation. So the interval's upper end is drawn and labelled
as the bound on the channel, and the verdict string is taken from the file rather
than re-derived here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from . import inputs
from .style import MODEL_STYLE, CONTROL, GRID, RULE, row_points, save


def figure_s1(*, directory: Path | None = None) -> list[Path]:
    """Split vs unsplit relations, against model 2, paired over the same folds."""
    probe = inputs.read_json("relation_split", "relation_split.json")
    graph = MODEL_STYLE["model3_graph"]
    tabular = MODEL_STYLE[probe["comparison_model"]]

    rows = (
        ("Model 3, relations split\n(#11 default)", probe["split_per_fold"], graph.color, graph.marker),
        ("Model 3, relations collapsed\n(single HAS_DIAGNOSIS)", probe["unsplit_per_fold"], CONTROL, "D"),
        ("Model 2, flattened", probe["comparison_per_fold"], tabular.color, tabular.marker),
    )
    deltas = (
        ("Split − collapsed\n(the channel)", probe["cost_of_collapsing"]),
        ("Split − model 2", probe["split_margin_over_comparison"]),
        ("Collapsed − model 2", probe["unsplit_margin_over_comparison"]),
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.0))

    left.set_xlim(0.62, 0.745)
    for position, (label, per_fold, color, marker) in zip(range(len(rows))[::-1], rows):
        row_points(left, position, per_fold, color=color, marker=marker)

    left.set_yticks(range(len(rows))[::-1])
    left.set_yticklabels([label for label, *_ in rows], fontsize=7.5)
    left.set_ylim(-0.6, len(rows) - 0.35)
    left.set_xlabel("Within-Study Harrell's C")
    left.set_title("a  Run scores (5 folds)")

    right.set_xlim(-0.033, 0.052)
    right.axvspan(
        -inputs.RESOLUTION_LIMIT,
        inputs.RESOLUTION_LIMIT,
        color=GRID,
        alpha=0.45,
        zorder=0,
        linewidth=0,
    )
    right.axvline(0.0, color=CONTROL, linewidth=0.9, zorder=1)
    for position, (label, paired) in zip(range(len(deltas))[::-1], deltas):
        low, high = paired["ci95"]
        right.errorbar(
            paired["mean"],
            position,
            xerr=[[paired["mean"] - low], [high - paired["mean"]]],
            fmt="D",
            color=CONTROL if not paired["excludes_zero"] else "black",
            markersize=5,
            capsize=3,
            elinewidth=1.2,
            zorder=3,
        )
        right.annotate(
            f"{paired['mean']:+.4f}  [{low:+.4f}, {high:+.4f}]",
            xy=(-0.032, position + 0.30),
            fontsize=7,
            color=RULE,
        )

    bound = float(probe["bound_on_the_channel"])
    right.annotate(
        f"bound on the channel: ≤ {bound:.4f} C",
        xy=(bound, len(deltas) - 1),
        xytext=(bound + 0.008, len(deltas) - 1 - 0.30),
        fontsize=7,
        color=RULE,
        arrowprops={"arrowstyle": "->", "color": RULE, "linewidth": 0.7},
    )
    right.set_yticks(range(len(deltas))[::-1])
    right.set_yticklabels([label for label, _ in deltas], fontsize=7.5)
    right.set_ylim(-0.6, len(deltas) - 0.35)
    right.set_xlabel(
        "Paired difference in within-Study Harrell's C\nshaded: ±0.02, the resolution limit (#3 §5)"
    )
    right.set_title(f"b  Paired differences (95% CI) — {probe['verdict']}")

    for axis in (left, right):
        axis.xaxis.grid(True, zorder=0)
        axis.set_axisbelow(True)

    fig.suptitle(
        "Collapsing the relation split removes the presence/absence indicator, not multiplicity — "
        "a null BOUNDS the channel rather than eliminating it.",
        fontsize=7,
        color=RULE,
        x=0.01,
        ha="left",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, "figS1_relation_split", directory=directory)
