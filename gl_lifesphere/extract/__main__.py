"""Run the extract from the command line.

    python -m gl_lifesphere.extract              # pull, cast, freeze
    python -m gl_lifesphere.extract --skip-pull  # replay from data/raw/
"""

from __future__ import annotations

import argparse
import sys

from .connection import settings_from_env
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gl_lifesphere.extract", description=__doc__)
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Reuse the existing data/raw/ instead of querying the live instance.",
    )
    args = parser.parse_args(argv)

    if not args.skip_pull:
        print(f"source: {settings_from_env().redacted()}")

    result = run(skip_pull=args.skip_pull)

    print("\nraw:")
    for name, n in result.raw_counts.items():
        print(f"  {name:<20} {n:>8,}")
    print("interim:")
    for name, n in result.interim_counts.items():
        print(f"  {name:<20} {n:>8,}")
    print(f"\ncohort: {result.cohort.describe()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
