"""Reading and writing the `data/` tables, with dtypes recorded rather than inferred.

Format is JSON Lines plus a `.schema.json` sidecar. That combination is chosen
against this project's specific failure mode rather than for convenience.

CSV was rejected because reading one back re-infers types, which is the exact
bug the extract layer exists to prevent: `eventOccurred` round-trips through CSV
as int64 `0`/`1` and `subjectId` values that look numeric become floats. Parquet
would round-trip correctly but needs `pyarrow`, and the stack is version-locked
under #5 — adding a dependency to store five tables is not that ticket's call to
make.

JSON already distinguishes the three things that matter here: the string `"0"`
from the number `0`, and both from `null`. So `data/raw/` round-trips verbatim
with no reader flags to remember, and `data/interim/` carries an explicit dtype
manifest that `read_table` *enforces* — a column whose type drifts fails loudly
instead of arriving as `object`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .connection import REPO_ROOT

DATA_ROOT = REPO_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"


def write_raw(name: str, rows: list[dict[str, Any]], *, directory: Path | None = None) -> Path:
    """Write a verbatim pull to `data/raw/<name>.jsonl`.

    No dtype manifest: raw tables are strings and nulls by construction, and
    writing a schema beside them would imply a cast has happened.
    """
    target = (directory if directory is not None else RAW_DIR) / f"{name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def read_raw(name: str, *, directory: Path | None = None) -> pd.DataFrame:
    """Read a verbatim pull back, with every column left as `object`.

    Parsed line by line with `json.loads` rather than through `pd.read_json`,
    which applies its own type inference: it turns `"0"`/`"1"` into integers and
    `"00123"` into `123`, reintroducing the very cast this layer exists to make
    explicit. `json.loads` returns a `str` for a JSON string and nothing else,
    so the round-trip is exact by construction rather than by passing the right
    flags.
    """
    source = (directory if directory is not None else RAW_DIR) / f"{name}.jsonl"
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return pd.DataFrame(rows, dtype=object)


def write_table(
    name: str,
    frame: pd.DataFrame,
    *,
    directory: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a typed frame plus the sidecar recording its dtypes.

    `meta` is stored alongside the schema for anything the reader should not
    have to recompute — row counts, the rule that produced the table, the
    instance it came from.
    """
    base = directory if directory is not None else INTERIM_DIR
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{name}.jsonl"

    frame.to_json(target, orient="records", lines=True, date_format="iso")
    schema = {
        "columns": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        # Recorded separately because the schema is written with sorted keys, and
        # a design matrix whose columns silently reorder between writes is a
        # cross-model comparability bug rather than a cosmetic one.
        "order": [str(column) for column in frame.columns],
        "n_rows": int(len(frame)),
        "meta": meta or {},
    }
    (base / f"{name}.schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def read_table(name: str, *, directory: Path | None = None) -> pd.DataFrame:
    """Read a typed frame back and re-apply its recorded dtypes.

    Raises if the sidecar names a column the file does not have, so a schema
    that has drifted away from its data is caught at read time rather than
    surfacing as an all-`NaN` column in a design matrix.
    """
    base = directory if directory is not None else INTERIM_DIR
    frame = pd.read_json(base / f"{name}.jsonl", lines=True, convert_dates=False)
    schema = json.loads((base / f"{name}.schema.json").read_text(encoding="utf-8"))

    recorded: dict[str, str] = schema["columns"]
    missing = sorted(set(recorded) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: schema names columns absent from the data: {missing}")

    if len(frame) != schema["n_rows"]:
        raise ValueError(
            f"{name}: schema records {schema['n_rows']} rows, file holds {len(frame)}"
        )

    order: list[str] = schema.get("order") or sorted(recorded)
    return frame.astype(recorded)[order]


def read_meta(name: str, *, directory: Path | None = None) -> dict[str, Any]:
    """Return the `meta` block written beside `name`."""
    base = directory if directory is not None else INTERIM_DIR
    schema = json.loads((base / f"{name}.schema.json").read_text(encoding="utf-8"))
    meta: dict[str, Any] = schema["meta"]
    return meta
