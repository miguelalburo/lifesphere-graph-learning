"""One place the figures agree on ink, so a reader can compare them by eye.

The rules here are the ones that make a difference legible in a journal column
and in greyscale print:

* **A model keeps its colour and its marker everywhere.** Model 1 is blue/circle,
  model 2 orange/square, model 3 green/triangle, in every panel of every figure.
  Colour alone is never the encoding — the marker carries the same identity for
  a colourblind or a photocopied reader. The three hues are Okabe-Ito steps and
  clear the CVD separation check (worst adjacent pair ΔE 11.4 protan).
* **No truncated bars.** Every quantity plotted here is a C-index, whose
  meaningful floor is 0.5 rather than 0, so a bar from a zero baseline wastes
  the panel and a bar from a 0.45 baseline overstates the difference. Points
  with their per-fold spread are used instead throughout, which is also what
  makes the 5 folds visible rather than hidden inside a mean.
* **The reference lines are the thresholds this project declared in advance**
  (#3 §5's ~0.02 resolution limit, #13's gate ceilings), drawn recessively in
  grey so they frame the data without competing with it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from ..extract.connection import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "results" / "figures"

# Vector for the manuscript, raster for reading the file in a browser or a slide.
FORMATS = ("pdf", "png")
DPI = 300

GRID = "#d9d9d9"
RULE = "#7a7a7a"
INK = "#222222"


@dataclass(frozen=True)
class ModelStyle:
    """How one model is drawn, and what it is called in a legend."""

    key: str
    label: str
    color: str
    marker: str


# Ordered baseline -> flattened -> graph: the ladder the study climbs (encoder
# doc §4), so a legend read top to bottom is read in the order the report argues.
MODELS: tuple[ModelStyle, ...] = (
    ModelStyle("model1_baseline", "Model 1 — clinical baseline", "#0072B2", "o"),
    ModelStyle("model2_tabular", "Model 2 — flattened/tabular", "#E69F00", "s"),
    ModelStyle("model3_graph", "Model 3 — graph (R-GCN)", "#009E73", "^"),
)

MODEL_STYLE = {model.key: model for model in MODELS}

# A control is not a model and must not be mistaken for one in a legend, so the
# diagnostics draw in neutral greys rather than borrowing a model's hue.
CONTROL = "#6a6a6a"
CONTROL_LIGHT = "#b0b0b0"


def use_house_style() -> None:
    """Set the rcParams every figure in this package is drawn under."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.2,
        }
    )


def fold_offsets(n_folds: int, spread: float = 0.26) -> list[float]:
    """Fixed vertical offsets for per-fold points — jitter that is not random.

    A random jitter would move the points between two runs of this script and
    make two printed copies of the same figure disagree, so the offsets are
    evenly spaced and deterministic instead.
    """
    if n_folds == 1:
        return [0.0]
    step = (2 * spread) / (n_folds - 1)
    return [-spread + i * step for i in range(n_folds)]


def row_points(
    axis: plt.Axes,
    position: float,
    per_fold: Iterable[float],
    *,
    color: str,
    marker: str,
    spread: float = 0.18,
    value_format: str = "{mean:.3f}",
    label: str | None = None,
) -> float:
    """Draw one row — five open folds behind a filled mean — and label its value.

    The value label goes to the *right of the row's rightmost fold*, never above
    the mean. Above the mean is where the folds are, and every collision in the
    first draft of these figures was a mean label landing on one.
    """
    values = list(per_fold)
    offsets = fold_offsets(len(values), spread=spread)
    axis.scatter(
        values,
        [position + offset for offset in offsets],
        s=14,
        facecolors="none",
        edgecolors=color,
        linewidths=0.8,
        zorder=3,
    )
    mean = sum(values) / len(values)
    axis.scatter(
        mean,
        position,
        marker=marker,
        s=40,
        color=color,
        zorder=4,
        label=label,
        edgecolors="white",
        linewidths=0.5,
    )
    # The gap is in points rather than data units: a gap in C would be a
    # different visual distance in every panel, and what has to clear the last
    # fold marker is a fixed number of pixels.
    axis.annotate(
        value_format.format(mean=mean),
        xy=(max(values), position),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=7.5,
        color=color,
    )
    return mean


def subtitle(axis: plt.Axes, text: str) -> None:
    """A recessive line between a panel's title and its frame.

    Every panel in Fig. 5 needs a caveat attached to it rather than left to the
    figure legend, and a caveat drawn inside the axes lands on the data.
    """
    axis.annotate(
        text,
        xy=(0.0, 1.015),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=7,
        color=RULE,
    )


def save(fig: plt.Figure, name: str, *, directory: Path | None = None) -> list[Path]:
    """Write one figure in every format, and return what was written."""
    target = directory if directory is not None else FIGURES_DIR
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in FORMATS:
        path = target / f"{name}.{suffix}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written
