#!/usr/bin/env python3
"""Emit STPP exact-event files for the GeoLife Phase 3 subset.

Reads the two ``.plt`` files of the Phase 3 subset (1,471 fixes),
extracts exact events (projected kilometres, EPSG:32650; single mark
``"fix"`` — GeoLife carries no categorical column), splits
chronologically, and writes the three emitter formats plus
``manifest.json`` into the output directory (same layout as
``emit_chicago_stpp.py``).

Boundaries are computed from the data, because the subset is two
trajectories 643 days apart, not a continuous quarter:

- aux   [t_min, t_min + 14 days) — the whole first trajectory
  (2007-08-04, 1,116 fixes);
- the remainder is the second trajectory (2009-05-08, 355 fixes over
  ~49 minutes), split into train/dev/test by equal thirds of its time
  span. The thirds are uneven in count (train 180, dev 33, test 142)
  because the trajectory has a ~16-minute gap (10:23–10:39) falling in
  the middle third — equal thirds of time, not of count.

Honest limitation (also recorded in the manifest): the 2-day embargo
used for Chicago is infeasible here — the entire remainder spans
~49 minutes, so a 2-day embargo would swallow it whole. The GeoLife
embargo is therefore 0 and the three splits are consecutive thirds of
the one remaining trajectory. The splits are thin (dev: 33 fixes)
each); they exist so the three formats have the same shape as
Chicago's, not because a 49-minute trajectory supports a serious
held-out evaluation.

Emitted data files are NOT committed to the repository; they live
under ``~/workspace/eventflow-eda/stpp-emitters/geolife/`` by default.

Usage:
    PYTHONPATH=<work>/src python scripts/emit_geolife_stpp.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from shikong.datasets.geolife.loader import load_geolife
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

SUBSET_DIR = Path.home() / "workspace" / "eventflow-eda" / "phase3" / "data" / "geolife" / "subset"
DEFAULT_OUTPUT_DIR = Path.home() / "workspace" / "eventflow-eda" / "stpp-emitters" / "geolife"

TARGET_CRS = "EPSG:32650"
TIME_ZONE = "Asia/Shanghai"
AUX_DAYS = 14


def _iso(moment: datetime) -> str:
    # Microsecond precision: GeoLife split ends are thirds of a 48:37 span.
    return moment.isoformat(timespec="microseconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-dir", type=Path, default=SUBSET_DIR)
    args = parser.parse_args(argv)

    plt_files = sorted(args.subset_dir.rglob("*.plt"))
    if not plt_files:
        print(f"ERROR: no .plt files under {args.subset_dir}", file=sys.stderr)
        return 1
    print(f"found {len(plt_files)} .plt file(s)")
    frame = load_geolife([str(p) for p in plt_files])
    events = extract_exact_events(frame, mark_col=None, target_crs=TARGET_CRS)
    n_events = events.height
    print(f"fixes: {n_events:,}")

    t_min = events["timestamp"].min()
    t_max = events["timestamp"].max()
    aux_end = t_min + timedelta(days=AUX_DAYS)
    remainder = events.filter(events["timestamp"] >= aux_end)
    if remainder.height == 0:
        print("ERROR: no events after the 14-day auxiliary period", file=sys.stderr)
        return 1
    rem_first = remainder["timestamp"].min()
    rem_last = remainder["timestamp"].max()
    span = rem_last - rem_first
    if span.total_seconds() <= 0:
        print("ERROR: remainder spans zero time; cannot split it", file=sys.stderr)
        return 1
    third = span / 3
    train_end = rem_first + third
    dev_end = rem_first + 2 * third
    test_end = rem_last + timedelta(seconds=1)
    print(f"data range: [{t_min}, {t_max}]")
    print(f"auxiliary period: [{t_min}, {aux_end}) — {AUX_DAYS} days")
    print(f"remainder: {remainder.height:,} fixes in [{rem_first}, {rem_last}] (span {span})")
    print(f"remainder thirds: train end {train_end}, dev end {dev_end}, test end {test_end}")
    print("embargo: 0 (a 2-day embargo would swallow the 49-minute remainder; see manifest)")

    boundaries = [
        SplitBoundary("aux", aux_end),
        SplitBoundary("train", train_end),
        SplitBoundary("dev", dev_end),
        SplitBoundary("test", test_end),
    ]
    result = chronological_event_splits(events, boundaries=boundaries)
    splits = result.splits
    codebook = ["fix"]

    total = sum(f.height for f in splits.values()) + sum(result.embargo_counts.values())
    print("counts per split:")
    for name in ("aux", "train", "dev", "test"):
        f = splits[name]
        print(f"  {name}: {f.height:,} fixes, [{f['timestamp'].min()}, {f['timestamp'].max()}]")
    print(f"reconciliation: splits + embargos = {total:,} (subset total {n_events:,})")
    if total != n_events:
        print(f"ERROR: reconciliation failed: {total:,} != {n_events:,}", file=sys.stderr)
        return 1

    window_starts = {
        "aux": t_min,
        "train": aux_end,
        "dev": train_end,
        "test": dev_end,
    }
    split_ends = {b.name: b.end for b in boundaries}
    written: list[Path] = []
    written += emit_gatech(splits, output_dir=args.output_dir, dim_process=len(codebook))
    written += emit_catalog(splits, output_dir=args.output_dir)
    written += emit_tick(
        splits, output_dir=args.output_dir, split_ends=split_ends, codebook=codebook
    )
    manifest = build_manifest(
        dataset="geolife",
        n_events=n_events,
        n_raw=n_events,
        n_removed_by_cleaning=0,
        source_crs="EPSG:4326",
        target_crs=TARGET_CRS,
        time_zone=TIME_ZONE,
        codebook=codebook,
        splits={
            name: {
                "start": _iso(window_starts[name]),
                "end": _iso(split_ends[name]),
                "n_events": splits[name].height,
                "first_event": _iso(splits[name]["timestamp"].min()),
                "last_event": _iso(splits[name]["timestamp"].max()),
            }
            for name in ("aux", "train", "dev", "test")
        },
        embargoes={},
        files=[p.name for p in written],
        notes=(
            "Phase 3 subset: 2 trajectories, 1,471 fixes — 2007-08-04 (1,116 fixes, ~44 min) "
            "and 2009-05-08 (355 fixes, ~49 min), 643 days apart. Auxiliary = first 14 days "
            "[t_min, t_min+14d); the remainder (second trajectory only) is split into "
            "train/dev/test by equal thirds of its time span. The thirds are uneven in count "
            "(train 180, dev 33, test 142) because the trajectory has a ~16-minute gap "
            "(10:23-10:39) falling in the middle third — equal thirds of time, not of count. "
            "Embargo is 0: a 2-day embargo "
            "would swallow the entire ~49-minute remainder. Single mark 'fix' (no categorical "
            "column in .plt files). The splits are thin (dev: 33 fixes) and exist for format "
            "parity with Chicago, not as a serious held-out evaluation. Time zone "
            "Asia/Shanghai is the loader's assumption (see geolife schema), not a declared "
            "property of the .plt files."
        ),
    )
    manifest_path = write_manifest(args.output_dir, manifest)
    print(f"wrote {len(written)} data files + {manifest_path.name} into {args.output_dir}")
    for path in sorted(written):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
