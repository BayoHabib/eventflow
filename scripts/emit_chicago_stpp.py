#!/usr/bin/env python3
"""Emit STPP exact-event files for Chicago crime (Q1 2024).

Reads the raw Q1 2024 CSV through :func:`load_chicago_crime` (which
cleans it: 60,672 raw -> 60,235 cleaned), extracts exact events
(projected kilometres, EPSG:26971), splits chronologically with a
14-day auxiliary period and 2-day embargos, and writes the three
emitter formats plus ``manifest.json`` into the output directory:

- ``{aux,train,dev,test}.pkl`` (+ ``.json``) — Gatech / THP / EasyTPP
- ``{split}_catalog.csv`` + ``{split}_txy.npy`` — catalog + (t, x, y)
- ``{split}_tick.pkl`` — tick per-node timestamp lists
- ``manifest.json``

Splits (decided before implementation; see
``docs/design-notes/stpp-emitters.md``):

- aux   [2024-01-01, 2024-01-15)
- train [2024-01-15, 2024-03-01)
- embargo [2024-03-01, 2024-03-03)
- dev   [2024-03-03, 2024-03-15)
- embargo [2024-03-15, 2024-03-17)
- test  [2024-03-17, 2024-04-01)

Reconciliation (printed, and enforced): aux + train + dev + test +
both embargos == 60,235 (the cleaned total).

Emitted data files are NOT committed to the repository; they live
under ``~/workspace/eventflow-eda/stpp-emitters/chicago/`` by default.

Usage:
    PYTHONPATH=<work>/src python scripts/emit_chicago_stpp.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from shikong.datasets.chicago_crime.mapping import load_chicago_crime
from shikong.stpp.emitters import (
    SplitBoundary,
    build_manifest,
    chronological_event_splits,
    emit_catalog,
    emit_gatech,
    emit_tick,
    extract_exact_events,
    write_manifest,
)

RAW_CSV = (
    Path.home()
    / "workspace"
    / "eventflow-eda"
    / "data"
    / "raw"
    / "chicago_crime"
    / "crimes_q1_2024.csv"
)
DEFAULT_OUTPUT_DIR = Path.home() / "workspace" / "eventflow-eda" / "stpp-emitters" / "chicago"

TARGET_CRS = "EPSG:26971"
TIME_ZONE = "America/Chicago"

BOUNDARIES = [
    SplitBoundary("aux", datetime(2024, 1, 15)),
    SplitBoundary("train", datetime(2024, 3, 1), timedelta(days=2)),
    SplitBoundary("dev", datetime(2024, 3, 15), timedelta(days=2)),
    SplitBoundary("test", datetime(2024, 4, 1)),
]
WINDOW_STARTS = {
    "aux": datetime(2024, 1, 1),
    "train": datetime(2024, 1, 15),
    "dev": datetime(2024, 3, 3),
    "test": datetime(2024, 3, 17),
}
SPLIT_ENDS = {b.name: b.end for b in BOUNDARIES}


def _iso(moment: datetime) -> str:
    # Microsecond precision: GeoLife split ends are thirds of a 48:37 span.
    return moment.isoformat(timespec="microseconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-csv", type=Path, default=RAW_CSV)
    args = parser.parse_args(argv)

    n_raw = pl.scan_csv(args.raw_csv).select(pl.len()).collect().item()
    frame = load_chicago_crime(str(args.raw_csv))
    events = extract_exact_events(frame, mark_col="primary_type", target_crs=TARGET_CRS)
    n_clean = events.height
    print(f"raw rows: {n_raw:,}; cleaned events: {n_clean:,}; removed: {n_raw - n_clean:,}")

    result = chronological_event_splits(events, boundaries=BOUNDARIES)
    splits = result.splits
    codebook = sorted(set(events["mark_label"].to_list()))
    print(f"codebook ({len(codebook)} marks): {codebook}")

    # Reconciliation: splits + embargos must equal the cleaned total.
    total = sum(f.height for f in splits.values()) + sum(result.embargo_counts.values())
    print("counts per split:")
    for name in ("aux", "train", "dev", "test"):
        f = splits[name]
        first = f["timestamp"].min()
        last = f["timestamp"].max()
        print(f"  {name}: {f.height:,} events, [{first}, {last}]")
    for label, count in result.embargo_counts.items():
        start, end = result.embargo_intervals[label]
        print(f"  {label}: {count:,} events, [{start}, {end})")
    print(f"reconciliation: splits + embargos = {total:,} (cleaned total {n_clean:,})")
    if total != n_clean:
        print(f"ERROR: reconciliation failed: {total:,} != {n_clean:,}", file=sys.stderr)
        return 1

    written: list[Path] = []
    written += emit_gatech(splits, output_dir=args.output_dir, dim_process=len(codebook))
    written += emit_catalog(splits, output_dir=args.output_dir)
    written += emit_tick(
        splits, output_dir=args.output_dir, split_ends=SPLIT_ENDS, codebook=codebook
    )
    manifest = build_manifest(
        dataset="chicago_crime",
        n_events=n_clean,
        n_raw=n_raw,
        n_removed_by_cleaning=n_raw - n_clean,
        source_crs="EPSG:4326",
        target_crs=TARGET_CRS,
        time_zone=TIME_ZONE,
        codebook=codebook,
        splits={
            name: {
                "start": _iso(WINDOW_STARTS[name]),
                "end": _iso(SPLIT_ENDS[name]),
                "n_events": splits[name].height,
                "first_event": _iso(splits[name]["timestamp"].min()),
                "last_event": _iso(splits[name]["timestamp"].max()),
            }
            for name in ("aux", "train", "dev", "test")
        },
        embargoes={
            label: {
                "start": _iso(start),
                "end": _iso(end),
                "n_events": result.embargo_counts[label],
            }
            for label, (start, end) in result.embargo_intervals.items()
        },
        files=[p.name for p in written],
        notes=(
            "Splits: auxiliary [2024-01-01, 2024-01-15), train [2024-01-15, 2024-03-01), "
            "dev [2024-03-03, 2024-03-15), test [2024-03-17, 2024-04-01); 2-day embargos "
            "[2024-03-01, 2024-03-03) and [2024-03-15, 2024-03-17). One sequence per split. "
            "Catalog has no magnitude column (crime has none) — deliberate deviation from "
            "EarthquakeNPP, see docs/design-notes/stpp-emitters.md."
        ),
    )
    manifest_path = write_manifest(args.output_dir, manifest)
    print(f"wrote {len(written)} data files + {manifest_path.name} into {args.output_dir}")
    for path in sorted(written):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
