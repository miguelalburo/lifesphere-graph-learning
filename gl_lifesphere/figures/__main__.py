"""Draw the report's figures from the recorded metrics.

    python -m gl_lifesphere.figures [--out DIR] [--only fig3|fig4|fig5|figS1]

Every figure is a pure function of files already in `results/metrics/`, so this
is safe to re-run at any time and takes about a second. `--only` exists because
a figure whose source file is missing raises, and one missing follow-up should
not stop the three that are ready.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import figure_2, figure_3, figure_4, figure_5, figure_s1, use_house_style
from .style import FIGURES_DIR

BUILDERS = {
    "fig2": figure_2,
    "fig3": figure_3,
    "fig4": figure_4,
    "fig5": figure_5,
    "figS1": figure_s1,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gl_lifesphere.figures")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR, help="where to write")
    parser.add_argument("--only", choices=sorted(BUILDERS), help="draw one figure")
    arguments = parser.parse_args(argv)

    use_house_style()
    chosen = [arguments.only] if arguments.only else list(BUILDERS)
    for name in chosen:
        for path in BUILDERS[name](directory=arguments.out):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
