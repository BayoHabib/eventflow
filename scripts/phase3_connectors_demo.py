#!/usr/bin/env python
"""External-connectors demonstration (phase 3, criterion 5).

Joins Beijing hourly meteorology (UCI 501, already on disk) onto
GeoLife GPS fixes via the generic ``TabularFileSource`` connector and
the existing ``TemporalContextJoinStep`` (asof-backward, 1 h tolerance).

The honest finding first: the two real ranges do NOT overlap.
GeoLife subset fixes run 2007-08-04 -> 2009-05-08; the Beijing data
start 2013-03-01. Part A runs that join anyway and records that the
tolerance correctly rejects every point (0 matched / 1,471
unmatched) instead of silently matching across a ~4-year gap.

Part B is the positive control: the same real Beijing table joined
onto synthetic event timestamps placed inside the Beijing range
(2013-03-01, realistic GPS-fix cadence). The joined values are
hand-checked against an independent read of the raw PRSA CSV for the
same station — the weather values are real, only the event times are
synthetic, and the script says so. Part C repeats the control with
the "exact" strategy.

Value subset: TEMP, PRES, DEWP, WSPM. There is no HUMI (humidity)
column in UCI 501 — the task's suggested subset was adjusted to what
the data actually carry; see the design note.

Outputs:
- audit/phase-3/connectors_demo.json (this work copy)

Usage:
    PYTHONPATH=src python scripts/phase3_connectors_demo.py \
        [--beijing-parquet .../phase3/data/beijing/beijing_raw.parquet] \
        [--beijing-csv-dir .../phase3/data/beijing/PRSA_Data_20130301-20170228] \
        [--geolife-dir .../phase3/data/geolife/subset] \
        [--json audit/phase-3/connectors_demo.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import polars as pl

from shikong.core.context import TabularFileSource
from shikong.core.event_frame import EventFrame
from shikong.core.schema import EventMetadata, EventSchema
from shikong.core.steps.context import TemporalContextJoinStep
from shikong.datasets.beijing_air_quality.schema import STATION_COORDS
from shikong.datasets.geolife.loader import load_geolife

VALUE_COLS = ["TEMP", "PRES", "DEWP", "WSPM"]
TOLERANCE = "1h"

# Synthetic event times for the positive control (part B): 2013-03-01,
# realistic GPS-fix cadence, plus one event before the first context
# row (2013-02-28 23:30) to prove the backward asof stays null there.
CONTROL_TIMES = [
    datetime(2013, 3, 1, 0, 7, 11),
    datetime(2013, 3, 1, 0, 59, 59),
    datetime(2013, 3, 1, 1, 0, 0),
    datetime(2013, 3, 1, 1, 47, 22),
    datetime(2013, 3, 1, 3, 0, 0),
    datetime(2013, 3, 1, 5, 59, 59),
    datetime(2013, 3, 1, 12, 30, 0),
    datetime(2013, 3, 1, 23, 45, 0),
    datetime(2013, 2, 28, 23, 30, 0),  # before the first context row
]
CONTROL_LAT, CONTROL_LON = 40.0005, 116.3275

_CSV_FLOAT_OVERRIDES = dict.fromkeys(
    ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"], pl.Float64
)


def _nearest_station(lat: float, lon: float) -> tuple[str, float]:
    """Nearest UCI station to a point (equirectangular approx, km)."""
    best, best_km = None, math.inf
    for station, (slon, slat) in STATION_COORDS.items():
        km = math.hypot((slat - lat) * 111.0, (slon - lon) * 85.0)
        if km < best_km:
            best, best_km = station, km
    assert best is not None
    return best, round(best_km, 1)


def _read_raw_station_csv(csv_dir: Path, station: str) -> pl.DataFrame:
    """Independent read of one raw PRSA CSV (no shared loader code)."""
    path = csv_dir / f"PRSA_Data_{station}_20130301-20170228.csv"
    return pl.read_csv(path, null_values="NA", schema_overrides=_CSV_FLOAT_OVERRIDES)


def _hand_check(
    joined: pl.DataFrame, raw_csv: pl.DataFrame, event_times: list[datetime]
) -> list[dict]:
    """Re-derive each joined row from the raw CSV for its expected hour."""
    checks = []
    by_time = {r["timestamp"]: r for r in joined.iter_rows(named=True)}
    for event_time in event_times:
        expected_hour = event_time.replace(minute=0, second=0, microsecond=0)
        raw = raw_csv.filter(
            (pl.col("year") == expected_hour.year)
            & (pl.col("month") == expected_hour.month)
            & (pl.col("day") == expected_hour.day)
            & (pl.col("hour") == expected_hour.hour)
        )
        assert raw.height == 1, f"raw CSV has {raw.height} rows for {expected_hour}"
        raw_row = raw.row(0, named=True)
        got = by_time[event_time]
        values = {}
        all_match = True
        for col in VALUE_COLS:
            expected = raw_row[col]
            actual = got[col]
            match = (expected is None and actual is None) or expected == actual
            all_match = all_match and bool(match)
            values[col] = {"expected_from_raw_csv": expected, "joined": actual, "match": match}
        checks.append(
            {
                "event_timestamp": event_time.isoformat(),
                "expected_context_hour": expected_hour.isoformat(),
                "values": values,
                "match": all_match,
            }
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beijing-parquet",
        default=str(
            Path.home() / "workspace/eventflow-eda/phase3/data/beijing/beijing_raw.parquet"
        ),
    )
    parser.add_argument(
        "--beijing-csv-dir",
        default=str(
            Path.home() / "workspace/eventflow-eda/phase3/data/beijing/PRSA_Data_20130301-20170228"
        ),
    )
    parser.add_argument(
        "--geolife-dir",
        default=str(Path.home() / "workspace/eventflow-eda/phase3/data/geolife/subset"),
    )
    parser.add_argument(
        "--json",
        default=str(Path(__file__).resolve().parent.parent / "audit/phase-3/connectors_demo.json"),
    )
    args = parser.parse_args()
    started = time.time()

    # -- Connector declaration (the same class any dataset would use). --
    source = TabularFileSource(
        args.beijing_parquet,
        timestamp_col="timestamp",
        value_cols=VALUE_COLS,
        key_cols=["station"],
    )
    context_all = source.load()
    ctx_range = context_all.select(
        pl.col("timestamp").min().alias("t_min"), pl.col("timestamp").max().alias("t_max")
    ).collect()
    beijing_min = ctx_range["t_min"][0]
    beijing_max = ctx_range["t_max"][0]

    # -- GeoLife subset (real events). -----------------------------------
    plt_paths = sorted(Path(args.geolife_dir).glob("*/Trajectory/*.plt"))
    assert len(plt_paths) == 2, f"expected 2 subset .plt files, found {len(plt_paths)}"
    events = load_geolife([str(p) for p in plt_paths])
    fixes = events.collect()
    geolife_min = fixes["timestamp"].min()
    geolife_max = fixes["timestamp"].max()
    station, station_km = _nearest_station(
        float(fixes["latitude"].mean()), float(fixes["longitude"].mean())
    )
    context = context_all.filter(pl.col("station") == station)

    # -- Part A: real GeoLife x real Beijing, asof-backward, 1 h. --------
    joined_real = (
        TemporalContextJoinStep(context, "timestamp", VALUE_COLS, tolerance=TOLERANCE)
        .run(events)
        .collect()
    )
    matched_a = int(joined_real["TEMP"].drop_nulls().len())
    part_a = {
        "strategy": "asof_backward",
        "tolerance": TOLERANCE,
        "station": station,
        "events": joined_real.height,
        "matched": matched_a,
        "unmatched": joined_real.height - matched_a,
        "finding": (
            "No overlap: GeoLife subset ends 2009-05-08, Beijing data start "
            "2013-03-01. The 1 h tolerance rejects every point (0 matched); "
            "widening it to force a match would join 2009 fixes to 2013 "
            "weather and was not done."
        ),
    }
    assert matched_a == 0, f"expected 0 matches across the 4-year gap, got {matched_a}"
    gap_days = (beijing_min - geolife_max).total_seconds() / 86400.0

    # -- Part B: positive control on synthetic event times. --------------
    control = EventFrame(
        pl.DataFrame(
            {
                "timestamp": CONTROL_TIMES,
                "latitude": [CONTROL_LAT] * len(CONTROL_TIMES),
                "longitude": [CONTROL_LON] * len(CONTROL_TIMES),
            }
        ).lazy(),
        EventSchema(timestamp_col="timestamp", lat_col="latitude", lon_col="longitude"),
        EventMetadata(
            dataset_name="connectors-demo-synthetic-events",
            crs="EPSG:4326",
            time_zone="Asia/Shanghai",
        ),
    )
    joined_control = (
        TemporalContextJoinStep(context, "timestamp", VALUE_COLS, tolerance=TOLERANCE)
        .run(control)
        .collect()
        .sort("timestamp")
    )
    matched_b = int(joined_control["TEMP"].drop_nulls().len())
    assert (
        matched_b == len(CONTROL_TIMES) - 1
    ), f"expected {len(CONTROL_TIMES) - 1} matched control events, got {matched_b}"
    pre_range = joined_control.filter(pl.col("timestamp") == CONTROL_TIMES[-1])
    assert pre_range.height == 1 and pre_range["TEMP"][0] is None
    raw_csv = _read_raw_station_csv(Path(args.beijing_csv_dir), station)
    hand_checks_b = _hand_check(joined_control, raw_csv, CONTROL_TIMES[:-1])
    assert all(c["match"] for c in hand_checks_b), "hand-check mismatch in part B"

    # -- Part C: exact strategy on the same control events. -------------
    joined_exact = (
        TemporalContextJoinStep(context, "timestamp", VALUE_COLS, strategy="exact")
        .run(control)
        .collect()
        .sort("timestamp")
    )
    matched_c = int(joined_exact["TEMP"].drop_nulls().len())
    on_the_hour = [t for t in CONTROL_TIMES if t.minute == 0 and t.second == 0]
    assert matched_c == len(
        on_the_hour
    ), f"expected {len(on_the_hour)} exact matches, got {matched_c}"
    hand_checks_c = _hand_check(joined_exact, raw_csv, on_the_hour)
    assert all(c["match"] for c in hand_checks_c), "hand-check mismatch in part C"

    report = {
        "criterion": "phase 3, exit criterion 5 (external connectors)",
        "connector": {
            "class": "shikong.core.context.TabularFileSource",
            "source_file": args.beijing_parquet,
            "timestamp_col": "timestamp",
            "key_cols": ["station"],
            "value_cols": VALUE_COLS,
            "value_subset_note": (
                "No HUMI column exists in UCI 501; the sensible meteorological "
                "subset is TEMP, PRES, DEWP, WSPM (RAIN and wd not joined)."
            ),
        },
        "ranges": {
            "geolife_subset": {
                "n_points": fixes.height,
                "t_min": geolife_min.isoformat(),
                "t_max": geolife_max.isoformat(),
            },
            "beijing_context": {
                "t_min": beijing_min.isoformat(),
                "t_max": beijing_max.isoformat(),
                "station": station,
                "station_km_from_subset_centroid": station_km,
                "station_note": "Nearest UCI station to the subset centroid; "
                "the temporal join does not use location, one station is "
                "used so (station, hour) keys stay unique.",
            },
            "gap_days_geolife_max_to_beijing_min": round(gap_days, 1),
            "overlap": False,
        },
        "part_a_real_geolife_x_real_beijing": part_a,
        "part_b_positive_control": {
            "strategy": "asof_backward",
            "tolerance": TOLERANCE,
            "station": station,
            "events": len(CONTROL_TIMES),
            "matched": matched_b,
            "unmatched": len(CONTROL_TIMES) - matched_b,
            "unmatched_note": "The 2013-02-28 23:30 event predates the first "
            "context row (2013-03-01 00:00) and correctly stays null.",
            "event_times_are_synthetic": True,
            "weather_values_are_real": True,
            "hand_checks_vs_raw_prsa_csv": hand_checks_b,
        },
        "part_c_exact_strategy_control": {
            "strategy": "exact",
            "events": len(CONTROL_TIMES),
            "matched": matched_c,
            "unmatched": len(CONTROL_TIMES) - matched_c,
            "hand_checks_vs_raw_prsa_csv": hand_checks_c,
        },
        "verdict": (
            "Connector works end-to-end on real data: the join is correct "
            "where the ranges overlap (parts B/C, all hand-checks match the "
            "raw CSV) and correctly refuses to match where they do not "
            "(part A, 0/1471). A production GeoLife enrichment needs an "
            "overlapping weather source for 2007-2009 (e.g. a reanalysis or "
            "a weather API written to a local file and loaded the same way); "
            "no such source is on disk, and none was fabricated."
        ),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"part A (real GeoLife x real Beijing, asof 1h): "
        f"{part_a['matched']} matched / {part_a['unmatched']} unmatched"
    )
    print(
        f"part B (control, asof 1h): {matched_b} matched, "
        f"{len(hand_checks_b)} hand-checks all match: "
        f"{all(c['match'] for c in hand_checks_b)}"
    )
    print(
        f"part C (control, exact): {matched_c} matched, "
        f"{len(hand_checks_c)} hand-checks all match: "
        f"{all(c['match'] for c in hand_checks_c)}"
    )
    print(f"demo JSON: {out_path}")


if __name__ == "__main__":
    main()
