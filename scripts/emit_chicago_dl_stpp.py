#!/usr/bin/env python3
"""Emit deep-learning STPP files for Chicago crime (Q1 2024).

Reads the raw Q1 2024 CSV through :func:`load_chicago_crime` (which
cleans it: 60,672 raw -> 60,235 cleaned), extracts exact events
(projected kilometres, EPSG:26971), splits chronologically with a
14-day auxiliary period and 2-day embargos (same splits as
``emit_chicago_stpp.py``), and writes the deep-learning formats plus
``manifest.json`` into the output directory:

- ``chicago_crime.npz`` — DeepSTPP/AutoSTPP shared wrapper contract
  (keys ``train``/``val``/``test``, lists of ``[t, x, y]`` arrays)
- ``chicago_crime_nstpp.npz`` — NSTPP windowed format (keys
  ``YYYYMMDD``, 30-day windows / 7-day step, ``[t_days, lon, lat]``)
- ``dataset/chicago_crime/data_{train,val,test}.pkl`` — DSTPP
  positional ``[t, lng, lat]`` pickles
- ``dataset/chicago_crime/data_{train,val,test}.pkl`` — SMASH
  positional ``[t, mark, lng, lat]`` pickles (same directory layout as
  DSTPP's, different files — see the design note)
- ``{aux,train,dev,test}_deepstpp.csv`` — DeepSTPP ``[x, y, t]`` file shape

Note: DSTPP and SMASH pickles share the ``dataset/<name>/`` directory
but are different files — this script writes them into separate
subdirectories (``dataset/chicago_crime_dstpp/`` and
``dataset/chicago_crime_smash/``) so the two same-named files do not
collide. The loader path each model expects is ``dataset/<name>/``;
rename the subdirectory to the dataset name the loader is run with.

Splits (same as ``emit_chicago_stpp.py``):

- aux   [2024-01-01, 2024-01-15)
- train [2024-01-15, 2024-03-01)
- embargo [2024-03-01, 2024-03-03)
- dev   [2024-03-03, 2024-03-15)
- embargo [2024-03-15, 2024-03-17)
- test  [2024-03-17, 2024-04-01)

Reconciliation (printed, and enforced): aux + train + dev + test +
both embargos == 60,235 (the cleaned total). The NSTPP windows cover
the whole cleaned period (embargo events included — the file has no
splits); window event counts overlap and do NOT sum to the total.

Emitted data files are NOT committed to the repository; they live
under ``~/workspace/eventflow-eda/stpp-dl-data-prep/chicago/`` by
default.

Usage:
    PYTHONPATH=<work>/src python scripts/emit_chicago_dl_stpp.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from shikong.datasets.chicago_crime.mapping import load_chicago_crime
from shikong.stpp.dl_formats import (
    emit_deepstpp_csv,
    emit_dstpp_pickles,
    emit_nstpp_npz,
    emit_smash_pickles,
    emit_wrapper_npz,
    window_events_nstpp,
)
from shikong.stpp.emitters import (
    SplitBoundary,
    build_manifest,
    chronological_event_splits,
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
DEFAULT_OUTPUT_DIR = Path.home() / "workspace" / "eventflow-eda" / "stpp-dl-data-prep" / "chicago"

TARGET_CRS = "EPSG:26971"
TIME_ZONE = "America/Chicago"
DATASET_NAME = "chicago_crime"

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

    total = sum(f.height for f in splits.values()) + sum(result.embargo_counts.values())
    print("counts per split:")
    for name in ("aux", "train", "dev", "test"):
        f = splits[name]
        print(f"  {name}: {f.height:,} events, [{f['timestamp'].min()}, {f['timestamp'].max()}]")
    for label, count in result.embargo_counts.items():
        start, end = result.embargo_intervals[label]
        print(f"  {label}: {count:,} events, [{start}, {end})")
    print(f"reconciliation: splits + embargos = {total:,} (cleaned total {n_clean:,})")
    if total != n_clean:
        print(f"ERROR: reconciliation failed: {total:,} != {n_clean:,}", file=sys.stderr)
        return 1

    written: list[Path] = []
    written += emit_wrapper_npz(splits, output_dir=args.output_dir, name=DATASET_NAME)
    written += emit_nstpp_npz(events, output_dir=args.output_dir, name=DATASET_NAME)
    written += emit_dstpp_pickles(
        splits, output_dir=args.output_dir, name=f"{DATASET_NAME}_dstpp"
    )
    written += emit_smash_pickles(
        splits, output_dir=args.output_dir, name=f"{DATASET_NAME}_smash"
    )
    written += emit_deepstpp_csv(splits, output_dir=args.output_dir)

    windows = window_events_nstpp(events)
    n_windows = len(windows)
    n_empty = sum(1 for arr in windows.values() if arr.shape[0] == 0)
    n_in_windows = sum(arr.shape[0] for arr in windows.values())
    print(f"NSTPP windows: {n_windows} (30 d / 7 d step), {n_empty} empty, "
          f"{n_in_windows:,} event occurrences (overlapping — do not sum to the total)")
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
        files=[str(p.relative_to(args.output_dir)) for p in written],
        notes=(
            "Deep-learning STPP formats (see docs/design-notes/stpp-dl-data-prep.md). "
            "Wrapper .npz (DeepSTPP/AutoSTPP): keys train/val/test, dev maps to val, "
            "aux excluded (the loaders have no aux key). NSTPP .npz: 30-day windows / "
            "7-day step over ALL cleaned events (embargoes included, no splits in the "
            "file), t in days since window start, lon/lat in degrees; window counts "
            "overlap and do not sum to the total. DSTPP pickles: positional [t, lng, lat], "
            "t in seconds since split start — the reference loader's assert Min[1] > 0 "
            "FAILS on Chicago (same-minute ties + first gap 0.0); format-level "
            "compatibility only, see the design note. SMASH pickles: positional "
            "[t, mark, lng, lat], mark = primary_type codebook index (0-based). "
            "DSTPP and SMASH pickles are written under separate subdirectories "
            "(chicago_crime_dstpp / chicago_crime_smash) because both loaders expect "
            "dataset/<name>/data_*.pkl with the same filenames but different schemas. "
            "DeepSTPP CSVs: [x, y, t], %f, one per split (aux included). "
            "No normalisation, no pre-windowing, no tie-breaking by Shikong."
        ),
    )
    manifest_path = write_manifest(args.output_dir, manifest)
    print(f"wrote {len(written)} data files + {manifest_path.name} into {args.output_dir}")
    for path in sorted(written):
        print(f"  {path.relative_to(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
