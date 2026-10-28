"""Stage 4 of the rover safe-haven chain: chain the seeds over the stage-3 edges (private).

An edge ``A -> B`` of stage 3 says a rover that wakes at seed ``A`` (its ``after_step``) can be on the pixel of
seed ``B`` at ``B``'s ``before_step``: it hibernates through ``B``'s outage and wakes at ``B``'s ``after_step``.
Following edges gives a chain of seeds, each reached from the one before, and a mission that lasts from the
first seed's wake-up to the end of the last seed's lit endurance.

Every seed is given one label, the chain that reaches it: ``(root, root_after_step)``, the first seed of the chain
and its operations start. A seed with no incoming edge is its own root. Otherwise it takes the label of the
incoming edge whose root starts earliest, ties broken by the lowest root id; among the edges with that same
root the chain that is longer is kept, then the lowest predecessor id, so the result does not depend on the
order of the tables. The labels are made in time order of ``after_step`` (an edge always leads to a later
``after_step``), so one pass over the edges gives every seed's label, and every seed gets exactly one.

For each seed the output gives ``root_id``, ``predecessor_id`` (0 for a root), ``chain_length`` (the seeds in the
chain, the seed itself included), the root's operations start and ``mission_hours``: the hours from the root's
``after_step`` to the end of this seed's lit endurance (a value whose last seed is at the 28-day flood cap is a
lower bound). The stage-2 and stage-3 columns are carried along, so the table is the final per-seed table;
its ``x_m`` and ``y_m`` are the seed pixel's centre in the DEM's CRS.
"""

from __future__ import annotations

import csv
import json
import os
import time as clock
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .pixel_coordinates import format_coordinate, pixel_centres
from .safe_haven_floods import FLOOD_COLUMNS, read_edge_table, read_flood_table
from .safe_haven_plan import SEED_COLUMNS, read_seed_table
from .safe_haven_windows import FloodGrid

CHAIN_COLUMNS = (
    *SEED_COLUMNS, *FLOOD_COLUMNS[1:],
    "root_id", "predecessor_id", "chain_length", "root_start_utc", "mission_hours", "previous_seeds", "next_seeds",
)
TOP_CHAINS = 5


@dataclass
class ChainReport:
    seeds: int
    edges: int
    roots: int
    seeds_in_chains: int            # a seed with an incoming or an outgoing edge
    longest_chain: int
    chain_length_counts: dict       # chain_length -> seeds with it
    longest_mission_hours: float
    mission_hours_percentiles: dict
    top_chains: list = field(default_factory=list)      # the seeds with the longest chains: ids from root to seed
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def chain_labels(
    after_step: npt.NDArray[np.int64], sources: npt.NDArray[np.int64], targets: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(root, predecessor, length)`` per seed, arrays indexed by ``seed_id - 1``; ``sources[i]`` and
    ``targets[i]`` are the 1-based ids of edge ``i``. ``predecessor`` is 0 for a root."""
    count = len(after_step)
    ids = np.arange(1, count + 1, dtype=np.int64)
    root, predecessor, length = ids.copy(), np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)
    if not len(sources):
        return root, predecessor, length
    order = np.lexsort((sources, targets))
    sources, targets = sources[order], targets[order]
    starts = np.searchsorted(targets, np.arange(1, count + 2))
    receiving = np.unique(targets)
    for target in receiving[np.lexsort((receiving, after_step[receiving - 1]))]:
        best_key, best = None, 0
        for source in sources[starts[target - 1]:starts[target]]:
            key = (after_step[root[source - 1] - 1], root[source - 1], -length[source - 1], source)
            if best_key is None or key < best_key:
                best_key, best = key, int(source)
        root[target - 1] = root[best - 1]
        predecessor[target - 1] = best
        length[target - 1] = length[best - 1] + 1
    return root, predecessor, length


def _format(value) -> str:
    if value is None:
        return ""
    return f"{value:g}" if isinstance(value, float) else str(value)


def _write_rows(path: Path, columns, rows) -> None:
    staging = path.with_name(path.name + ".partial")
    try:
        with staging.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            for row in rows:
                writer.writerow([_format(row[name]) for name in columns])
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def run_chains(
    output_path: str | Path, *, seeds_path: str | Path, floods_path: str | Path, edges_path: str | Path,
    geo_transform, step_minutes: int = 120, overwrite: bool = False,
) -> ChainReport:
    """Write the final per-seed table (``CHAIN_COLUMNS``) and ``<output>.report.json``. ``geo_transform`` is the
    DEM's GDAL transform: the pixel-centre coordinates ``x_m`` and ``y_m`` of each seed are made from it (a
    seed table that already has them is overwritten)."""
    started = clock.monotonic()
    output_path = Path(output_path)
    report_path = output_path.with_name(output_path.name + ".report.json")
    for path in (output_path, report_path):
        if path.exists() and not overwrite:
            raise InputError(f"{path} exists; pass overwrite=True to replace it", code="chains_output_exists",
                             details={"path": str(path)})
    seeds, floods, edges = read_seed_table(seeds_path), read_flood_table(floods_path), read_edge_table(edges_path)
    count = len(seeds)
    if [s["seed_id"] for s in seeds] != list(range(1, count + 1)) or [f["seed_id"] for f in floods] != list(range(1, count + 1)):
        raise InputError("the seed table and the flood table do not hold the same seeds 1..N in order",
                         code="chains_inputs_mismatch", details={"seeds": count, "floods": len(floods)})
    after = np.array([s["after_step"] for s in seeds], dtype=np.int64)
    sources = np.array([e[0] for e in edges], dtype=np.int64)
    targets = np.array([e[1] for e in edges], dtype=np.int64)
    if len(edges) and (min(sources.min(), targets.min()) < 1 or max(sources.max(), targets.max()) > count):
        raise InputError("an edge names a seed that is not in the seed table", code="chains_inputs_mismatch")
    if len(edges) and (after[targets - 1] <= after[sources - 1]).any():
        bad = int(np.flatnonzero(after[targets - 1] <= after[sources - 1])[0])
        raise InputError("an edge does not lead to a later operations start", code="chains_edge_order",
                         details={"source": int(sources[bad]), "target": int(targets[bad])})

    root, predecessor, length = chain_labels(after, sources, targets)
    previous = np.bincount(targets, minlength=count + 1)[1:]
    following = np.bincount(sources, minlength=count + 1)[1:]
    step_hours = step_minutes / 60.0
    endurance = np.array([f["lit_endurance_hours"] for f in floods], dtype=float)
    mission = (after - after[root - 1]) * step_hours + endurance
    root_start = np.datetime_as_string(FloodGrid(step_minutes).times(after[root - 1]), unit="us")

    x, y = pixel_centres(geo_transform, [s["row"] for s in seeds], [s["column"] for s in seeds])
    rows = []
    for index, (seed, flood) in enumerate(zip(seeds, floods)):
        row = dict(seed)
        row.update(x_m=format_coordinate(float(x[index])), y_m=format_coordinate(float(y[index])))
        row.update({name: flood[name] for name in FLOOD_COLUMNS[1:]})
        row.update(root_id=int(root[index]), predecessor_id=int(predecessor[index]), chain_length=int(length[index]),
                   root_start_utc=str(root_start[index]) + "Z", mission_hours=float(mission[index]),
                   previous_seeds=int(previous[index]), next_seeds=int(following[index]))
        rows.append(row)
    _write_rows(output_path, CHAIN_COLUMNS, rows)

    top = []
    for end in np.lexsort((np.arange(count), -mission, -length))[:TOP_CHAINS]:
        path, node = [], int(end) + 1
        while node:
            path.append(node)
            node = int(predecessor[node - 1])
        top.append({"seed_id": int(end) + 1, "chain_length": int(length[end]), "mission_hours": float(mission[end]),
                    "chain": path[::-1]})
    lengths, counts = np.unique(length, return_counts=True)
    report = ChainReport(
        seeds=count, edges=len(edges), roots=int((predecessor == 0).sum()),
        seeds_in_chains=int(((previous > 0) | (following > 0)).sum()), longest_chain=int(length.max()) if count else 0,
        chain_length_counts={int(a): int(b) for a, b in zip(lengths, counts)},
        longest_mission_hours=float(mission.max()) if count else 0.0,
        mission_hours_percentiles={f"p{p}": float(np.percentile(mission, p)) for p in (10, 50, 90, 99)} if count else {},
        top_chains=top, seconds=clock.monotonic() - started)
    report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return report


def read_chain_table(path: str | Path) -> list[dict]:
    """The final table written by :func:`run_chains`; numbers are ints or floats, blanks are ``None``."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CHAIN_COLUMNS:
            raise InputError(f"{path} is not a chain table", code="chains_table_unreadable")
        rows = []
        for record in reader:
            row = {}
            for name, value in record.items():
                if value == "":
                    row[name] = None
                elif name in ("month", "outage_kind", "outage_start_utc", "outage_end_utc", "root_start_utc"):
                    row[name] = value
                elif name in ("x_m", "y_m"):
                    row[name] = float(value)
                else:
                    row[name] = float(value) if "hours" in name else int(value)
            rows.append(row)
    return rows
