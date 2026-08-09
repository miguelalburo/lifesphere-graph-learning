"""PROTOTYPE — throwaway inspector for the rooted subgraph construction (#11).

    .venv/bin/python -m gl_lifesphere.constructions.prototype_subgraph_tui

**This file is not production code and is not meant to become production code.**
It is the hand-driven shell around `subject_subgraph.py`, which is the part
worth keeping. No tests, no error handling beyond what makes it runnable, and
nothing here is cached to disk — #11's `data/processed/subject_subgraph/` cache
belongs to the arm-3 ticket (#12), not to the question of whether the
construction is right.

**The question.** #11 asks whether the schema-faithful rooted subgraph, once
built, actually looks like what we think it looks like. So the shell walks a
deliberately varied Subject sample — one Diagnosis and several, few Samples and
near the 113 maximum, a Diagnosis with no `OF_CONDITION`, no Samples at all, no
Diagnosis at all — and renders the whole subgraph plus every assertion after
each step. Three keys exist purely to break it on purpose: `[r]` drops the
reverse edges (encoder doc §2.1's bug, in its exact form), `[p]` splits
`HAS_DIAGNOSIS` on `isPrimaryDiagnosis`, and `[e]` runs a real two-layer
`HeteroConv` over whatever is on screen to answer "can the encoder consume it".

`[a]` sweeps the whole cohort and reports every check that ever fails — that is
the "assert it, do not eyeball it" bullet; the rest of the screen is for the
part that has to be looked at.
"""

from __future__ import annotations

import sys
import termios
import tty
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from ..evaluation.splits import fold_split, load_folds
from ..extract.store import INTERIM_DIR, PROCESSED_DIR, read_table
from ..features import assemble_raw_frame, fit_feature_contract
from ..features.contract import FeatureContract
from . import subject_subgraph as ss

BOLD, DIM, RED, GREEN, YELLOW, CYAN, RESET = (
    "\x1b[1m", "\x1b[2m", "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[36m", "\x1b[0m"
)


# ---------------------------------------------------------------------------
# Loading — done once at startup, then everything is in memory
# ---------------------------------------------------------------------------


@dataclass
class Cohort:
    members: pd.DataFrame
    all_diagnoses: pd.DataFrame
    subjects: dict[str, pd.Series]
    diagnoses: dict[str, pd.DataFrame]
    samples: dict[str, pd.DataFrame]
    pathology: dict[str, pd.DataFrame]
    contract: FeatureContract
    blocks: ss.FeatureBlocks

    def record(self, subject_id: str) -> ss.SubjectRecord:
        empty_d = self.diagnoses[next(iter(self.diagnoses))].iloc[:0]
        empty_s = self.samples[next(iter(self.samples))].iloc[:0]
        empty_p = self.pathology[next(iter(self.pathology))].iloc[:0]
        subject = self.subjects[subject_id]
        return ss.SubjectRecord(
            subject_id=subject_id,
            study_id=str(subject["studyId"]),
            subject=subject,
            diagnoses=self.diagnoses.get(subject_id, empty_d),
            samples=self.samples.get(subject_id, empty_s),
            pathology=self.pathology.get(subject_id, empty_p),
        )


def _by_subject(frame: pd.DataFrame, in_cohort: set[str]) -> dict[str, pd.DataFrame]:
    """Group once at load, so building 6,811 subgraphs is 6,811 dict lookups."""
    restricted = frame[frame["subjectId"].isin(in_cohort)]
    return {str(key): group for key, group in restricted.groupby("subjectId")}


def load_cohort() -> Cohort:
    cohort_dir = PROCESSED_DIR / "cohort_os"
    members = read_table("members", directory=cohort_dir)
    subjects = read_table("subjects", directory=INTERIM_DIR)
    diagnoses = read_table("diagnoses", directory=INTERIM_DIR)
    samples = read_table("samples", directory=INTERIM_DIR)
    pathology = read_table("pathology_details", directory=INTERIM_DIR)

    # Fit on outer fold 0's training Subjects, the same way arm 2 does — a
    # contract fitted on the whole cohort would leak, and this prototype is
    # checking arm-3 features against arm-2 features, so both must be fold 0's.
    folds = load_folds()
    train = fold_split(folds, 0).train
    contract = fit_feature_contract(assemble_raw_frame(), frozenset(train))

    in_cohort = set(members["subjectId"])
    return Cohort(
        members=members,
        all_diagnoses=diagnoses[diagnoses["subjectId"].isin(in_cohort)],
        subjects={
            r["subjectId"]: r
            for _, r in subjects[subjects["subjectId"].isin(in_cohort)].iterrows()
        },
        diagnoses=_by_subject(diagnoses, in_cohort),
        samples=_by_subject(samples, in_cohort),
        pathology=_by_subject(pathology, in_cohort),
        contract=contract,
        blocks=ss.FeatureBlocks.from_contract(contract),
    )


# ---------------------------------------------------------------------------
# Case selection — the "deliberately varied sample" #11 asks for
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    label: str
    subject_id: str
    why: str


def pick_cases(cohort: Cohort) -> tuple[list[Case], list[str]]:
    """Pick one Subject per structurally interesting shape, by measurement.

    Every case is *found* rather than hard-coded, so a re-extract that changes
    the cohort re-picks rather than silently inspecting a Subject that no longer
    has the property the label claims. Shapes with no Subject come back in the
    second return value: a shape #11 asks about that turns out not to occur is
    an answer, and silently dropping it from the list would hide it.
    """
    ids = sorted(cohort.subjects)
    stats = pd.DataFrame(
        {
            "subjectId": ids,
            "n_diagnoses": [len(cohort.diagnoses.get(i, [])) for i in ids],
            "n_samples": [
                int(cohort.samples[i]["sampleType"].isin(ss.BASELINE_SAMPLE_TYPES).sum())
                if i in cohort.samples else 0
                for i in ids
            ],
            "n_pathology": [len(cohort.pathology.get(i, [])) for i in ids],
            "n_conditions": [
                cohort.diagnoses[i]["conditionId"].notna().sum() if i in cohort.diagnoses else 0
                for i in ids
            ],
            "n_primary": [
                int((cohort.diagnoses[i]["isPrimaryDiagnosis"] == True).sum())  # noqa: E712
                if i in cohort.diagnoses else 0
                for i in ids
            ],
        }
    ).set_index("subjectId")

    used: set[str] = set()

    def first(mask: pd.Series, order: str | None = None, ascending: bool = True) -> str | None:
        rows = stats[mask]
        if rows.empty:
            return None
        if order:
            rows = rows.sort_values(order, ascending=ascending)
        # Prefer a Subject no earlier case already claimed, so ten cases cover
        # ten Subjects rather than collapsing onto whichever one is extreme in
        # several ways at once.
        fresh = [i for i in rows.index if i not in used]
        chosen = str(fresh[0] if fresh else rows.index[0])
        used.add(chosen)
        return chosen

    peak = int(stats.n_samples.max())
    wanted: list[tuple[str, str | None, str]] = [
        ("one Diagnosis, typical",
         first((stats.n_diagnoses == 1) & (stats.n_samples.between(11, 15)) & (stats.n_pathology > 0)),
         "the median shape — one Diagnosis, ~13 Samples, some pathology"),
        ("most Diagnoses in the cohort",
         first(stats.n_diagnoses == stats.n_diagnoses.max(), "n_diagnoses", False),
         "the top of the Diagnosis fan-out, where every secondary Diagnosis has no Condition"),
        ("fewest Samples in the cohort",
         first(stats.n_samples == stats[stats.n_samples > 0].n_samples.min(), "n_samples"),
         "the smallest Sample set the encoder ever has to aggregate"),
        (f"most Samples ({peak}, not 113)",
         first(stats.n_samples == stats.n_samples.max(), "n_samples", False),
         "encoder doc §3's bottleneck case — but #11's 113 counted post-baseline types too"),
        ("no Sample at all",
         first(stats.n_samples == 0),
         "the empty set: PROVIDED_SAMPLE has no edges and the graph mean is 0"),
        ("a Diagnosis with no OF_CONDITION",
         first((stats.n_diagnoses > stats.n_conditions) & (stats.n_conditions > 0)),
         "#4 §1: 100% of secondary Diagnoses lack a Condition, 100% of primaries have one"),
        ("no PathologyDetail",
         first((stats.n_pathology == 0) & (stats.n_diagnoses > 0)),
         "HAS_PATHOLOGY empty — the featureless structural node is absent entirely"),
        ("no Diagnosis at all",
         first(stats.n_diagnoses == 0),
         "#4 §8's edge case: one cohort Subject has no Diagnosis, so no Condition and no pathology"),
        ("no primary Diagnosis",
         first((stats.n_diagnoses > 0) & (stats.n_primary == 0)),
         "#4 §8: 2 Subjects carry a null isPrimaryDiagnosis — watch [p] put them all in SECONDARY"),
        ("two Conditions on one Subject",
         first(stats.n_conditions > 1),
         "would be the only case where deduplicating the shared vocabulary node matters"),
    ]
    cases = [Case(label, sid, why) for label, sid, why in wanted if sid is not None]
    absent = [label for label, sid, _ in wanted if sid is None]
    return cases, absent


# ---------------------------------------------------------------------------
# The encoder probe — "can the encoder consume it?"
# ---------------------------------------------------------------------------


def encoder_probe(data: HeteroData, *, d: int = 8) -> tuple[str, torch.Tensor | None]:
    """Two `HeteroConv` layers over the real graph, returning the root embedding.

    Deliberately the crudest possible stand-in for the arm-3 encoder — untrained,
    `SAGEConv`, `d = 8`. It is not measuring anything; it answers one question
    the shape checks cannot, which is whether PyG will actually *run* on an
    empty node store, a zero-edge relation, or a one-column feature matrix.
    """
    torch.manual_seed(0)
    in_dims = {t: int(data[t].x.size(1)) for t in data.node_types}
    notes: list[str] = []
    layers = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for layer in range(2):
            dims = in_dims if layer == 0 else {t: d for t in in_dims}
            convs = {
                edge_type: SAGEConv((dims[edge_type[0]], dims[edge_type[2]]), d)
                for edge_type in data.edge_types
            }
            layers.append(HeteroConv(convs, aggr="sum"))
        notes = [str(w.message).split(".")[0] for w in caught]

    x_dict = dict(data.x_dict)
    try:
        for layer in layers:
            out = layer(x_dict, data.edge_index_dict)
            # A node type that is never a destination gets no output at all —
            # PyG warns about it, and zeroing is the honest depiction of what
            # the readout then has: nothing.
            x_dict = {
                t: torch.relu(out[t]) if t in out else torch.zeros((data[t].num_nodes, d))
                for t in in_dims
            }
        return "; ".join(notes) or "ok", x_dict[ss.SUBJECT][0]
    except Exception as error:  # the whole point is to find out which ones raise
        return f"{type(error).__name__}: {error}", None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass
class State:
    cursor: int = 0
    split_primary: bool = False
    reverse_edges: bool = True
    dedupe_conditions: bool = True
    view: str = "graph"  # graph | features | contract
    message: str = ""


def _matrix(x: torch.Tensor, columns: tuple[str, ...], *, max_cols: int = 8, max_rows: int = 6) -> list[str]:
    if x.numel() == 0:
        return [f"  {DIM}(no rows){RESET}"]
    names = list(columns) or [f"c{i}" for i in range(x.size(1))]
    keep = list(range(min(len(names), max_cols)))
    head = "  " + DIM + " ".join(f"{names[i][:9]:>9}" for i in keep)
    if len(names) > max_cols:
        head += f"  … +{len(names) - max_cols}"
    head += RESET
    rows = [head]
    for r in range(min(x.size(0), max_rows)):
        rows.append("  " + " ".join(f"{x[r, i].item():>9.3f}" for i in keep))
    if x.size(0) > max_rows:
        rows.append(f"  {DIM}… +{x.size(0) - max_rows} more rows{RESET}")
    return rows


def render(cohort: Cohort, cases: list[Case], absent: list[str], state: State) -> str:
    case = cases[state.cursor]
    record = cohort.record(case.subject_id)
    data = ss.build_subgraph(
        record, cohort.contract, blocks=cohort.blocks,
        split_primary_relation=state.split_primary,
        add_reverse_edges=state.reverse_edges,
        dedupe_conditions=state.dedupe_conditions,
    )
    checks = ss.check_subgraph(
        data, record, cohort.contract, blocks=cohort.blocks,
        split_primary_relation=state.split_primary,
        add_reverse_edges=state.reverse_edges,
    )
    summary = ss.summarise(data)

    out: list[str] = []
    out.append(f"{BOLD}PROTOTYPE #11 — schema-faithful rooted subgraph{RESET}  "
               f"{DIM}case {state.cursor + 1}/{len(cases)}{RESET}")
    out.append(f"{BOLD}{CYAN}{case.label}{RESET}  {DIM}{record.subject_id[:8]}… · {record.study_id}{RESET}")
    out.append(f"{DIM}{case.why}{RESET}")
    out.append("")

    flags = [
        f"reverse edges {'ON ' if state.reverse_edges else RED + 'OFF' + RESET}",
        f"primary split {'ON ' if state.split_primary else 'off'}",
        f"dedupe Conditions {'ON ' if state.dedupe_conditions else 'off'}",
    ]
    out.append(f"{DIM}flags:{RESET} " + f"{DIM} · {RESET}".join(flags))
    out.append("")

    if state.view == "graph":
        out.append(f"{BOLD}nodes{RESET}")
        for t in ss.NODE_TYPES:
            n = summary.nodes.get(t, 0)
            width = cohort.blocks.widths[t]
            bar = "█" * min(n, 40)
            out.append(f"  {t:<16} {n:>4}  {DIM}x width {width:<4}{RESET} {bar}")
        out.append("")
        out.append(f"{BOLD}edges{RESET}")
        for (src, rel, dst), n in sorted(summary.edges.items()):
            arrow = f"{src} -{rel}-> {dst}"
            colour = DIM if rel.startswith("rev_") else ""
            out.append(f"  {colour}{arrow:<52}{RESET} {n:>4}")
        if summary.isolated:
            out.append(f"  {YELLOW}isolated nodes:{RESET} {summary.isolated}")
        out.append("")
        reached = ss.messages_reaching_root(data)
        parts = []
        for t in ss.NODE_TYPES:
            total = int(data[t].num_nodes)
            hit = int(reached[t].sum()) if total else 0
            mark = GREEN + "✓" + RESET if hit == total else RED + "✗" + RESET
            parts.append(f"{mark} {t} {hit}/{total}")
        out.append(f"{BOLD}reaching the root in ≤2 hops{RESET}  " + "  ".join(parts))
        if absent:
            out.append("")
            out.append(f"{YELLOW}shapes #11 asks about that no cohort Subject has:{RESET} "
                       + f"{DIM}; {RESET}".join(absent))

    elif state.view == "features":
        for t in ss.NODE_TYPES:
            out.append(f"{BOLD}{t}{RESET} {DIM}({int(data[t].num_nodes)} nodes){RESET}")
            out.extend(_matrix(data[t].x, cohort.blocks.columns(t)))
        out.append("")

    else:  # contract
        out.append(f"{BOLD}the shared feature contract, partitioned across node types{RESET}")
        total = 0
        for t in ss.NODE_TYPES:
            cols = cohort.blocks.columns(t)
            total += len(cols)
            shown = ", ".join(cols[:6]) + (f", … +{len(cols) - 6}" if len(cols) > 6 else "")
            out.append(f"  {t:<16} {len(cols):>3}  {DIM}{shown or '(featureless — 1 constant column)'}{RESET}")
        out.append(f"  {DIM}{'assigned':<16} {total:>3}   contract declares {len(cohort.contract.feature_names)}{RESET}")
        out.append("")
        out.append(f"{BOLD}what the flattened arm gets for this Subject{RESET}")
        row = cohort.contract.transform(
            assemble_raw_frame(), frozenset({record.subject_id})
        )
        sample_cols = list(cohort.blocks.sample)
        out.append(f"  {DIM}Sample block, arm 2:{RESET} " +
                   "  ".join(f"{c}={row[c].iloc[0]:.3f}" for c in sample_cols))
        graph_mean = data[ss.SAMPLE].x.mean(dim=0) if data[ss.SAMPLE].num_nodes else torch.zeros(len(sample_cols))
        out.append(f"  {DIM}Sample block, arm 3 (mean over nodes):{RESET} " +
                   "  ".join(f"{c}={v:.3f}" for c, v in zip(sample_cols, graph_mean.tolist())))
        out.append("")
        out.append(f"{BOLD}the Condition node's vocabulary, cohort-wide{RESET}")
        for check in ss.check_condition_vocabulary(cohort.all_diagnoses):
            mark = f"{GREEN}✓{RESET}" if check.passed else f"{RED}✗{RESET}"
            out.append(f"  {mark} {check.name}")
            out.append(f"      {DIM}{check.detail}{RESET}")

    out.append("")
    out.append(f"{BOLD}checks{RESET}")
    for check in checks:
        mark = f"{GREEN}✓{RESET}" if check.passed else f"{RED}✗{RESET}"
        out.append(f"  {mark} {check.name}")
        if check.detail:
            out.append(f"      {DIM}{check.detail}{RESET}")

    if state.message:
        out.append("")
        out.append(state.message)

    out.append("")
    out.append(
        f"{DIM}[j/k] next/prev case  [v] view: {state.view}  [r] reverse edges  "
        f"[p] primary split  [d] dedupe  [e] encoder probe  [a] sweep cohort  [q] quit{RESET}"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The whole-cohort sweep
# ---------------------------------------------------------------------------


def sweep(cohort: Cohort, state: State, *, limit: int | None = None) -> str:
    ids = sorted(cohort.subjects)
    if limit:
        ids = ids[:limit]
    failures: dict[str, list[str]] = {}
    errors: dict[str, int] = {}
    print(f"\n{DIM}sweeping {len(ids)} Subjects…{RESET}", flush=True)
    for n, subject_id in enumerate(ids):
        if n % 500 == 0:
            print(f"{DIM}  {n}/{len(ids)}{RESET}", flush=True)
        record = cohort.record(subject_id)
        try:
            data = ss.build_subgraph(
                record, cohort.contract, blocks=cohort.blocks,
                split_primary_relation=state.split_primary,
                add_reverse_edges=state.reverse_edges,
                dedupe_conditions=state.dedupe_conditions,
            )
            for check in ss.check_subgraph(
                data, record, cohort.contract, blocks=cohort.blocks,
                split_primary_relation=state.split_primary,
                add_reverse_edges=state.reverse_edges,
            ):
                if not check.passed:
                    failures.setdefault(check.name, []).append(f"{subject_id[:8]} {check.detail}")
        except Exception as error:
            errors[f"{type(error).__name__}: {error}"] = errors.get(f"{type(error).__name__}: {error}", 0) + 1

    lines = [f"{BOLD}sweep over {len(ids)} Subjects{RESET}"]
    if not failures and not errors:
        lines.append(f"  {GREEN}every check passed on every Subject{RESET}")
    for name, hits in sorted(failures.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {RED}✗{RESET} {name}  {BOLD}{len(hits)}{RESET} Subjects")
        for hit in hits[:3]:
            lines.append(f"      {DIM}{hit}{RESET}")
    for message, count in errors.items():
        lines.append(f"  {RED}raised{RESET} {message}  ×{count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    print("loading cohort and fitting the fold-0 feature contract…")
    cohort = load_cohort()
    cases, absent = pick_cases(cohort)
    state = State()
    views = ["graph", "features", "contract"]

    while True:
        print("\x1b[2J\x1b[H" + render(cohort, cases, absent, state))
        state.message = ""
        key = _read_key()
        if key in ("q", "\x03"):
            print()
            return
        if key == "j":
            state.cursor = (state.cursor + 1) % len(cases)
        elif key == "k":
            state.cursor = (state.cursor - 1) % len(cases)
        elif key == "v":
            state.view = views[(views.index(state.view) + 1) % len(views)]
        elif key == "r":
            state.reverse_edges = not state.reverse_edges
        elif key == "p":
            state.split_primary = not state.split_primary
        elif key == "d":
            state.dedupe_conditions = not state.dedupe_conditions
        elif key == "e":
            record = cohort.record(cases[state.cursor].subject_id)
            data = ss.build_subgraph(
                record, cohort.contract, blocks=cohort.blocks,
                split_primary_relation=state.split_primary,
                add_reverse_edges=state.reverse_edges,
                dedupe_conditions=state.dedupe_conditions,
            )
            status, z = encoder_probe(data)
            if z is None:
                state.message = f"{BOLD}encoder probe{RESET}  {RED}{status}{RESET}"
            else:
                vec = " ".join(f"{v:+.3f}" for v in z.tolist())
                nonzero = int((z.abs() > 1e-9).sum())
                state.message = (
                    f"{BOLD}encoder probe{RESET}  root z (d=8): {vec}\n"
                    f"  {DIM}{nonzero}/8 non-zero — all-zero means the root received nothing{RESET}"
                )
        elif key == "a":
            print("\x1b[2J\x1b[H")
            report = sweep(cohort, state)
            print("\n" + report + f"\n\n{DIM}[any key] back{RESET}")
            _read_key()


if __name__ == "__main__":
    main()
