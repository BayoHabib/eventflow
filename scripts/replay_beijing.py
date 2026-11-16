#!/usr/bin/env python
"""Replay the Beijing v1 recipe on the UCI Multi-Site Air Quality data.

Two stages, as required by the phase-3 exit criteria:

1. One station x one year first (Dongsi, 2016 — a leap year, 8,784 hours).
2. All 12 stations, full range (2013-03-01 -> 2017-02-28).

For each stage the script records rows/columns in and out, the
observation-count sum, NA counts per measurement column before and
after panel completion, and per-station coverage. Hand-checks
re-derive a few station-hours from the raw CSV files with an
independent read (eager ``pl.read_csv`` on the single-station file,
not the loader) and compare against the pipeline output.

Outputs:
- ``~/workspace/eventflow-eda/phase3/beijing_features.parquet``
  (full-run feature frame; large files are never committed)
- ``<work-copy>/audit/phase-3/beijing_replay.json``
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from shikong.core.schema import RecipeConfig
from shikong.core.steps.panel import CompletePanelStep
from shikong.core.steps.temporal import (
    CalendarEncodingStep,
    ExtractTemporalComponentsStep,
    MovingAverageStep,
    TemporalLagStep,
    TimeBinsStep,
)
from shikong.datasets.beijing_air_quality.mapping import load_beijing_air_quality
from shikong.datasets.beijing_air_quality.recipes.beijing_air_quality_v1 import (
    LAG_HOURS,
    ROLLING_WINDOW_HOURS,
    BeijingAirQualityV1Recipe,
)
from shikong.datasets.beijing_air_quality.schema import MEASUREMENT_COLS, POLLUTANT_COLS

WORK = Path(__file__).resolve().parent.parent
PHASE3 = Path.home() / "workspace" / "eventflow-eda" / "phase3"
DATA_DIR = PHASE3 / "data" / "beijing" / "PRSA_Data_20130301-20170228"
PARQUET_OUT = PHASE3 / "beijing_features.parquet"
JSON_OUT = WORK / "audit" / "phase-3" / "beijing_replay.json"

RAW_CSV_SCHEMA = {
    "No": pl.Int64,
    "year": pl.Int64,
    "month": pl.Int64,
    "day": pl.Int64,
    "hour": pl.Int64,
    "PM2.5": pl.Float64,
    "PM10": pl.Float64,
    "SO2": pl.Float64,
    "NO2": pl.Float64,
    "CO": pl.Float64,
    "O3": pl.Float64,
    "TEMP": pl.Float64,
    "PRES": pl.Float64,
    "DEWP": pl.Float64,
    "RAIN": pl.Float64,
    "wd": pl.String,
    "WSPM": pl.Float64,
    "station": pl.String,
}


def _na_counts(df: pl.DataFrame) -> dict[str, int]:
    return {c: df[c].null_count() for c in MEASUREMENT_COLS}


def _run_recipe_staged(frame, label: str) -> dict:
    """Run the recipe in two stages to measure NA before/after completion."""
    started = time.monotonic()
    before = frame.collect()
    na_before = _na_counts(before)

    # Stage 1: bin + complete (the completion boundary).
    completed = CompletePanelStep(
        unit_col="station",
        value_cols=["count"],
        measure_cols=MEASUREMENT_COLS,
        min_periods=1,
    ).run(TimeBinsStep(bin_size="1h").run(frame))
    completed_df = completed.collect()
    na_after = _na_counts(completed_df)
    n_generated = int((completed_df["count"] == 0).sum())

    # Stage 2: the remaining steps, on the completed panel.
    result = completed
    for step in (
        ExtractTemporalComponentsStep(
            components=["hour_of_day", "day_of_week", "day_of_month", "month", "is_weekend"]
        ),
        CalendarEncodingStep(encoding_type="cyclical", components=["hour", "day_of_week", "month"]),
        TemporalLagStep(
            value_cols=POLLUTANT_COLS,
            lag_periods=LAG_HOURS,
            time_unit="1h",
            group_cols=["station"],
        ),
        MovingAverageStep(
            value_cols=POLLUTANT_COLS,
            windows=ROLLING_WINDOW_HOURS,
            time_unit="1h",
            group_cols=["station"],
            closed="left",
        ),
    ):
        result = step.run(result)
    out = result.collect()
    elapsed = time.monotonic() - started

    # The staged run must agree with the packaged recipe run.
    packaged = (
        BeijingAirQualityV1Recipe(
            RecipeConfig(
                dataset="beijing_air_quality",
                recipe="beijing_air_quality_v1",
                temporal={"time_bin": "1h"},
            )
        )
        .run(frame)
        .collect()
    )
    assert packaged.equals(out), f"{label}: staged run disagrees with the packaged recipe"

    return {
        "label": label,
        "input_rows": before.height,
        "input_cols": len(before.columns),
        "completed_rows": completed_df.height,
        "output_rows": out.height,
        "output_cols": len(out.columns),
        "output_columns": out.columns,
        "count_sum": int(out["count"].sum()),
        "generated_pairs": n_generated,
        "na_before_completion": na_before,
        "na_after_completion": na_after,
        "na_after_full_pipeline": _na_counts(out),
        "seconds": round(elapsed, 2),
    }


def _read_raw_station(station: str) -> pl.DataFrame:
    """Independent raw read of one station CSV (not via the loader)."""
    path = DATA_DIR / f"PRSA_Data_{station}_20130301-20170228.csv"
    df = pl.read_csv(path, null_values=["NA"], schema_overrides=RAW_CSV_SCHEMA)
    return df.with_columns(pl.datetime("year", "month", "day", "hour").alias("timestamp"))


def _hand_check(out: pl.DataFrame, station: str, ts: datetime, checks: dict[str, object]) -> dict:
    """Compare pipeline output cells against hand-derived expectations."""
    row = out.filter((pl.col("station") == station) & (pl.col("timestamp") == ts))
    assert row.height == 1, f"no output row for {station} @ {ts}"
    detail: dict[str, object] = {}
    ok = True
    for col, expected in checks.items():
        got = row[col].item()
        match = (got is None and expected is None) or (
            got is not None and expected is not None and abs(got - expected) < 1e-9
        )
        detail[col] = {"expected": expected, "got": got, "match": bool(match)}
        ok = ok and bool(match)
    return {"station": station, "timestamp": ts.isoformat(), "checks": detail, "passed": ok}


def main() -> None:
    print(f"Loading Beijing data from {DATA_DIR}")
    events = load_beijing_air_quality(str(DATA_DIR))
    raw = events.collect()
    print(f"Loaded {raw.height} rows")

    per_station = {}
    for station in sorted(raw["station"].unique().to_list()):
        sub = raw.filter(pl.col("station") == station)
        per_station[station] = {
            "rows": sub.height,
            "timestamp_min": sub["timestamp"].min().isoformat(),
            "timestamp_max": sub["timestamp"].max().isoformat(),
            "pm25_nulls": sub["PM2.5"].null_count(),
            "pm25_null_rate": round(sub["PM2.5"].null_count() / sub.height, 4),
        }

    # Stage 1: 1 station x 1 year (Dongsi, 2016 — leap year, 8,784 hours).
    dongsi_2016 = events.lazy_frame.filter(
        (pl.col("station") == "Dongsi")
        & (pl.col("timestamp") >= datetime(2016, 1, 1))
        & (pl.col("timestamp") < datetime(2017, 1, 1))
    )
    stage1 = _run_recipe_staged(events.with_lazy_frame(dongsi_2016), "dongsi_2016_1_station_1_year")
    print(
        f"Stage 1: {stage1['output_rows']} rows x {stage1['output_cols']} cols "
        f"in {stage1['seconds']}s"
    )

    # Stage 2: all 12 stations, full range.
    stage2 = _run_recipe_staged(events, "all_12_stations_full_range")
    print(
        f"Stage 2: {stage2['output_rows']} rows x {stage2['output_cols']} cols "
        f"in {stage2['seconds']}s"
    )

    # Re-run the full pipeline once more for the parquet output (the
    # staged run already validated it equals the packaged recipe).
    full = (
        BeijingAirQualityV1Recipe(
            RecipeConfig(
                dataset="beijing_air_quality",
                recipe="beijing_air_quality_v1",
                temporal={"time_bin": "1h"},
            )
        )
        .run(events)
        .collect()
    )
    full.write_parquet(PARQUET_OUT)
    print(f"Wrote {PARQUET_OUT} ({full.height} rows)")

    # Hand-checks, re-derived from the raw CSVs (independent read).
    hand_checks = []
    dongsi_raw = _read_raw_station("Dongsi")
    ts1 = datetime(2016, 6, 15, 12)
    raw1 = dongsi_raw.filter(pl.col("timestamp") == ts1).row(0, named=True)
    prev24 = dongsi_raw.filter(
        (pl.col("timestamp") >= datetime(2016, 6, 14, 12))
        & (pl.col("timestamp") < datetime(2016, 6, 15, 12))
    )["PM2.5"].drop_nulls()
    hand_checks.append(
        _hand_check(
            full,
            "Dongsi",
            ts1,
            {
                "PM2.5": raw1["PM2.5"],
                "TEMP": raw1["TEMP"],
                "PM2.5_lag_24": dongsi_raw.filter(pl.col("timestamp") == datetime(2016, 6, 14, 12))[
                    "PM2.5"
                ].item(),
                "PM2.5_ma_24": prev24.mean(),
            },
        )
    )
    # First Dongsi hour in 2016 with a missing PM2.5: the reading stays
    # null, and the 1h lag at the next hour is null too.
    null_rows = (
        dongsi_raw.filter((pl.col("timestamp") >= datetime(2016, 1, 1)) & pl.col("PM2.5").is_null())
        .select("timestamp")
        .head(1)
    )
    ts_null = null_rows["timestamp"].item()
    hand_checks.append(_hand_check(full, "Dongsi", ts_null, {"PM2.5": None}))
    hand_checks.append(
        _hand_check(
            full,
            "Dongsi",
            ts_null + timedelta(hours=1),
            {"PM2.5_lag_1": None},
        )
    )
    # A different station and pollutant, 168h lag (one week).
    shunyi_raw = _read_raw_station("Shunyi")
    ts3 = datetime(2015, 3, 10, 8)
    raw3 = shunyi_raw.filter(pl.col("timestamp") == ts3).row(0, named=True)
    hand_checks.append(
        _hand_check(
            full,
            "Shunyi",
            ts3,
            {
                "O3": raw3["O3"],
                "PM10": raw3["PM10"],
                "PM10_lag_168": shunyi_raw.filter(pl.col("timestamp") == datetime(2015, 3, 3, 8))[
                    "PM10"
                ].item(),
            },
        )
    )
    for hc in hand_checks:
        print(
            f"Hand-check {hc['station']} @ {hc['timestamp']}: "
            f"{'PASS' if hc['passed'] else 'FAIL'}"
        )
        for col, d in hc["checks"].items():
            print(f"  {col}: expected={d['expected']} got={d['got']} match={d['match']}")
    assert all(hc["passed"] for hc in hand_checks), "a hand-check failed"

    report = {
        "dataset": "Beijing Multi-Site Air Quality (UCI 501, DOI 10.24432/C5RK5G)",
        "source_zip": "PRSA2017_Data_20130301-20170228.zip",
        "stations": 12,
        "time_range": ["2013-03-01T00:00:00", "2017-02-28T23:00:00"],
        "hourly_axis_hours": 35064,
        "raw_rows": raw.height,
        "per_station_coverage": per_station,
        "na_counts_all_rows_raw": _na_counts(raw),
        "stage1_one_station_one_year": stage1,
        "stage2_all_stations": stage2,
        "parquet_output": str(PARQUET_OUT),
        "hand_checks": hand_checks,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {JSON_OUT}")


if __name__ == "__main__":
    main()
