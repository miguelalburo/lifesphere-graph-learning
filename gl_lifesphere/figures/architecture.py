"""Fig. 2 — the three model architectures, from the rooted patient subgraph to a risk score.

Drawn rather than measured: this is the only figure in the package that reads no
file from `results/metrics/`. It is here anyway so that the architecture diagram
is regenerated from the same style module as every other display item, and so a
change to the encoder is a diff rather than a redraw in an external editor.

**The figure's argument is what the columns share, not what the lanes contain.**
Input, feature contract and decoder are drawn as single objects spanning all
three lanes, because they *are* single objects: one cohort, one fitted contract,
one `lifelines.CoxPHFitter(strata=['study'])`. Only the encoder column differs
per model, which is the study's whole design (#3 §3) and the reason a difference
in the output column is attributable to representation.

Two details carry methodological weight and are drawn deliberately:

* **Model 1 has no encoder.** Its box is dashed and empty-by-design rather than
  omitted, so the lane reads as "covariates enter the decoder directly" instead
  of as a missing panel.
* **The secondary-Diagnosis edge is dashed** in the input schematic. #11's
  relation split makes its presence structurally visible to model 3 alone
  (#15), and that one bit is the live alternative explanation for model 3's
  margin — so it is drawn, not smoothed away.

Nodes are drawn with `scatter` rather than `Circle`. These axes are not
equal-aspect, so a patch with a radius renders as an ellipse whose eccentricity
follows the figure size; a marker stays round whatever the figure is resized to.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import CONTROL, GRID, INK, MODEL_STYLE, RULE, save

# Column left edge and width in axes fraction, so the layout is editable in one
# place rather than by hunting literals through the drawing calls.
COLUMNS: dict[str, tuple[float, float]] = {
    "input": (0.005, 0.180),
    "representation": (0.215, 0.145),
    "encoder": (0.380, 0.195),
    "embedding": (0.595, 0.068),
    "decoder": (0.695, 0.170),
    "output": (0.885, 0.100),
}

# Lane centres, top to bottom in the order the report argues the ladder.
LANES: dict[str, float] = {
    "model1_baseline": 0.760,
    "model2_tabular": 0.480,
    "model3_graph": 0.185,
}

LANE_HEIGHT = 0.150
DECODER_TOP, DECODER_BOTTOM = 0.635, 0.365


def _box(
    axis: plt.Axes,
    column: str,
    y: float,
    text: str,
    *,
    height: float = LANE_HEIGHT,
    color: str = INK,
    dashed: bool = False,
    fontsize: float = 6.8,
) -> None:
    """One rounded box, centred vertically on `y` and filling its column."""
    x, width = COLUMNS[column]
    axis.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=0.9,
            edgecolor=color,
            facecolor="white",
            linestyle=(0, (3, 2)) if dashed else "-",
            zorder=2,
        )
    )
    axis.text(
        x + width / 2,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.5,
        zorder=3,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.9,
            color=RULE,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def _right(column: str) -> float:
    x, width = COLUMNS[column]
    return x + width


def _centre(column: str) -> float:
    x, width = COLUMNS[column]
    return x + width / 2


# (x, y, label, dx, dy, ha, va) — label offsets are hand-placed per node so no
# label lands on an edge or on a neighbour's label.
_NODES: dict[str, tuple[float, float, str, float, float, str, str]] = {
    "S": (0.062, 0.500, "Subject\n(root)", -0.015, 0.0, "right", "center"),
    "D1": (0.112, 0.615, "Diagnosis", -0.014, 0.0, "right", "center"),
    "C": (0.155, 0.675, "Condition", 0.0, 0.020, "center", "bottom"),
    "P": (0.155, 0.545, "Pathology", 0.0, -0.020, "center", "top"),
    "D2": (0.112, 0.400, "Diagnosis", -0.014, 0.0, "right", "center"),
    "Sm": (0.062, 0.330, "Sample", 0.0, -0.022, "center", "top"),
}

# (source, target, dashed) — dashed marks the secondary-Diagnosis relation.
_EDGES: tuple[tuple[str, str, bool], ...] = (
    ("S", "D1", False),
    ("D1", "C", False),
    ("D1", "P", False),
    ("S", "D2", True),
    ("S", "Sm", False),
)


def _subgraph(axis: plt.Axes) -> None:
    """The rooted per-patient subgraph, drawn inside the input column."""
    for source, target, dashed in _EDGES:
        x0, y0 = _NODES[source][:2]
        x1, y1 = _NODES[target][:2]
        axis.plot(
            [x0, x1],
            [y0, y1],
            color=RULE,
            linewidth=0.8,
            linestyle=(0, (2.5, 2)) if dashed else "-",
            zorder=2,
        )
    for key, (x, y, label, dx, dy, ha, va) in _NODES.items():
        root = key == "S"
        axis.scatter(
            [x],
            [y],
            s=52 if root else 34,
            facecolor=MODEL_STYLE["model3_graph"].color if root else "white",
            edgecolor=INK if root else RULE,
            linewidth=0.9,
            zorder=3,
        )
        axis.text(
            x + dx,
            y + dy,
            label,
            ha=ha,
            va=va,
            fontsize=5.6,
            color=RULE,
            zorder=3,
        )


LANES_CONTENT: dict[str, dict[str, object]] = {
    "model1_baseline": {
        "representation": "one row per patient\nstaging + pathology\ncovariate subset",
        "encoder": "none — covariates enter\nthe decoder directly",
        "embedding": "$\\mathbf{x}_i$",
        "dashed_encoder": True,
        "encoder_height": LANE_HEIGHT,
    },
    "model2_tabular": {
        "representation": "one row per patient\n167 columns",
        "encoder": "Linear(167 $\\rightarrow$ 32) + ReLU\ndropout 0.2\nLinear(32 $\\rightarrow$ 16)",
        "embedding": "$\\mathbf{z}_i \\in \\mathbb{R}^{16}$",
        "dashed_encoder": False,
        "encoder_height": LANE_HEIGHT,
    },
    "model3_graph": {
        "representation": "subgraph kept\nper-node-type\nfeature blocks",
        "encoder": (
            "type-aware Linear($w_t \\rightarrow$ 32)\n"
            "2 $\\times$ relation-aware message\n"
            "passing — mean agg, identity\n"
            "residual, per-type LayerNorm\n"
            "root readout"
        ),
        "embedding": "$\\mathbf{z}_i \\in \\mathbb{R}^{32}$",
        "dashed_encoder": False,
        "encoder_height": 0.185,
    },
}

# Where each lane's embedding arrow meets the shared decoder's left edge. Three
# arrows landing on one point reads as a collision rather than as convergence.
_DECODER_ENTRY: dict[str, float] = {
    "model1_baseline": 0.575,
    "model2_tabular": 0.500,
    "model3_graph": 0.425,
}


def figure_2(*, directory: Path | None = None) -> list[Path]:
    """The three architectures side by side, sharing their input and their decoder."""
    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    for column, label in (
        ("input", "Input"),
        ("representation", "Representation"),
        ("encoder", "Encoder\n(stage 1 · differs)"),
        ("embedding", "Embedding"),
        ("decoder", "Decoder\n(stage 2 · shared)"),
        ("output", "Output"),
    ):
        axis.text(
            _centre(column),
            0.955,
            label,
            ha="center",
            va="bottom",
            fontsize=7.2,
            weight="bold",
            color=INK,
            linespacing=1.4,
        )
    axis.plot([0.0, 1.0], [0.945, 0.945], color=GRID, linewidth=0.8, zorder=0)

    # --- shared input, spanning every lane -------------------------------
    axis.add_patch(
        FancyBboxPatch(
            (COLUMNS["input"][0], 0.255),
            COLUMNS["input"][1],
            0.490,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=0.9,
            edgecolor=INK,
            facecolor="white",
            zorder=2,
        )
    )
    _subgraph(axis)
    axis.text(
        _centre("input"),
        0.235,
        "6,811 patients · 5 node types\n5 relations + 5 reverses",
        ha="center",
        va="top",
        fontsize=6.0,
        color=RULE,
        linespacing=1.4,
    )

    # --- per-lane boxes ---------------------------------------------------
    for key, lane in LANES_CONTENT.items():
        style = MODEL_STYLE[key]
        y = LANES[key]

        axis.text(
            COLUMNS["representation"][0],
            y + LANE_HEIGHT / 2 + 0.022,
            style.label,
            ha="left",
            va="bottom",
            fontsize=7.4,
            weight="bold",
            color=style.color,
        )

        _arrow(
            axis,
            (_right("input"), 0.500),
            (COLUMNS["representation"][0], y),
            rad=0.0 if key == "model2_tabular" else (-0.14 if key == "model1_baseline" else 0.14),
        )
        _box(axis, "representation", y, str(lane["representation"]), color=style.color)
        _arrow(axis, (_right("representation"), y), (COLUMNS["encoder"][0], y))
        _box(
            axis,
            "encoder",
            y,
            str(lane["encoder"]),
            color=CONTROL if lane["dashed_encoder"] else style.color,
            dashed=bool(lane["dashed_encoder"]),
            fontsize=6.4,
            height=float(lane["encoder_height"]),
        )
        _arrow(axis, (_right("encoder"), y), (COLUMNS["embedding"][0], y))
        _box(
            axis,
            "embedding",
            y,
            str(lane["embedding"]),
            color=style.color,
            height=0.080,
            fontsize=7.6,
        )
        _arrow(
            axis,
            (_right("embedding"), y),
            (COLUMNS["decoder"][0], _DECODER_ENTRY[key]),
            rad=0.0 if key == "model2_tabular" else (0.12 if key == "model1_baseline" else -0.12),
        )

    # --- shared decoder and output ---------------------------------------
    axis.add_patch(
        FancyBboxPatch(
            (COLUMNS["decoder"][0], DECODER_BOTTOM),
            COLUMNS["decoder"][1],
            DECODER_TOP - DECODER_BOTTOM,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=0.9,
            edgecolor=INK,
            facecolor="white",
            zorder=2,
        )
    )
    axis.text(
        _centre("decoder"),
        0.500,
        "encoder frozen\n\nstratified Cox\nEfron ties, 20 strata\npenalty per fold",
        ha="center",
        va="center",
        fontsize=6.4,
        color=INK,
        linespacing=1.5,
        zorder=3,
    )
    _arrow(axis, (_right("decoder"), 0.500), (COLUMNS["output"][0], 0.500))
    _box(axis, "output", 0.500, "risk score\n$\\eta_i$", height=0.110, fontsize=7.0)
    axis.text(
        _centre("output"),
        0.428,
        "scored by\nwithin-Study\nHarrell's C",
        ha="center",
        va="top",
        fontsize=6.0,
        color=RULE,
        linespacing=1.4,
    )

    # --- the footing that makes the comparison legible --------------------
    axis.plot([0.0, 1.0], [0.085, 0.085], color=GRID, linewidth=0.8, zorder=0)
    axis.text(
        0.0,
        0.052,
        "Shared by construction: cohort, 5-fold Study-stratified splits, fitted feature contract, Cox objective, decoder — so only the encoder column\n"
        "differs, and a difference in the output is attributable to the representation. Flattening (models 1 and 2) reduces each patient to the primary\n"
        "Diagnosis with per-field fallback; the dashed secondary-Diagnosis edge is structurally visible to model 3 alone.",
        ha="left",
        va="top",
        fontsize=6.1,
        color=RULE,
        linespacing=1.55,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return save(fig, "fig2_architectures", directory=directory)
