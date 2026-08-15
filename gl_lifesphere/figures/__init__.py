"""The report's display items, drawn from `results/metrics/` and nothing else.

One figure per question the report asks, matching the display-item table in
`docs/report_draft.md`:

| Figure | Panel(s)                                              | Source                            |
| ------ | ----------------------------------------------------- | --------------------------------- |
| Fig. 2 | the three architectures, input to risk score          | *drawn, reads no metrics*         |
| Fig. 3 | model scores; paired differences                        | `metrics/model{1,2,3}_*/`           |
| Fig. 4 | per-Study C for the three models, ordered by events     | the same, `per_study_harrell_c`   |
| Fig. 5 | the trust gate's three diagnostics vs. its thresholds | `metrics/trust_gate/`             |
| Fig. S1| #15's relation-split probe                            | `metrics/relation_split/`         |

Redrawing is free and never retrains anything, so a figure is always the current
contents of `results/metrics/`. Fig. 1 and Fig. 5's subgraph schematic are drawn
by hand and are not produced here.

Fig. 2 is the one exception to the "drawn from `results/metrics/` and nothing
else" rule above: it is a schematic with no measurement in it. It lives here
anyway so that an architecture change is a diff against this package rather than
a redraw in an external editor, and so it inherits `style.py`'s palette and
markers along with every other display item.
"""

from __future__ import annotations

from .architecture import figure_2
from .models import figure_3, figure_4
from .gate import figure_5
from .relation_split import figure_s1
from .style import use_house_style

__all__ = ["figure_2", "figure_3", "figure_4", "figure_5", "figure_s1", "use_house_style"]
