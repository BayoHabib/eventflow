"""Tests for the generic external-table connector (phase 3, criterion 5).

Covers ``TabularFileSource`` declaration/data validation, loading, and
the end-to-end joins it feeds: ``TemporalContextJoinStep`` (asof /
exact / nearest, tolerance, fill), ``SpatioTemporalContextJoinStep``
(group isolation), ``SpatialContextJoinStep``, and ``EnricherStep``.

All fixtures are synthetic files under ``tmp_path`` — no test depends
on the uncommitted phase-3 data directories. The real-data
demonstration lives in ``scripts/phase3_connectors_demo.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from shapely.geometry import box

from shikong.core import EventFrame, EventMetadata, EventSchema
from shikong.core.context import EnricherStep, TabularFileSource, TemporalJoin
from shikong.core.steps.context import (
    SpatialContextJoinStep,
    SpatioTemporalContextJoinStep,
    TemporalContextJoinStep,
)

T0 = datetime(2024, 1, 1)


def frame(data: dict) -> EventFrame:
    """Minimal EventFrame with timestamp col ``time`` and lat/lon."""
    n = len(next(iter(data.values())))
    return EventFrame(
        pl.DataFrame(
            {
                "time": [T0] * n,
                "x": [0.5] * n,
                "y": [0.5] * n,
                **data,
            }
        ).lazy(),
        EventSchema(timestamp_col="time", lon_col="x", lat_col="y"),
        EventMetadata(dataset_name="generic-observations", crs="EPSG:3857"),
    )


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_parquet(path: Path, data: dict) -> Path:
    pl.DataFrame(data).write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Declaration validation (construction time)
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError, match="not found"):
        TabularFileSource(missing, timestamp_col="ts", value_cols=["v"])


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.txt", "ts,v\n2024-01-01 00:00:00,1\n")
    with pytest.raises(ValueError, match="Unsupported"):
        TabularFileSource(path, timestamp_col="ts", value_cols=["v"])


def test_missing_timestamp_column_raises(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.csv", "other,v\n2024-01-01 00:00:00,1\n")
    with pytest.raises(ValueError, match="missing from the file"):
        TabularFileSource(path, timestamp_col="ts", value_cols=["v"])


def test_unknown_value_column_raises(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.csv", "ts,v\n2024-01-01 00:00:00,1\n")
    with pytest.raises(ValueError, match=r"\['nope'\]"):
        TabularFileSource(path, timestamp_col="ts", value_cols=["v", "nope"])


def test_unknown_key_column_raises(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.csv", "ts,site,v\n2024-01-01 00:00:00,A,1\n")
    with pytest.raises(ValueError, match=r"\['zone'\]"):
        TabularFileSource(path, timestamp_col="ts", value_cols=["v"], key_cols=["zone"])


def test_empty_value_cols_raises(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.csv", "ts,v\n2024-01-01 00:00:00,1\n")
    with pytest.raises(ValueError, match="at least one column"):
        TabularFileSource(path, timestamp_col="ts", value_cols=[])


def test_overlapping_declarations_raise(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.csv", "ts,v\n2024-01-01 00:00:00,1\n")
    with pytest.raises(ValueError, match="must be distinct"):
        TabularFileSource(path, timestamp_col="ts", value_cols=["ts"])
    with pytest.raises(ValueError, match="must be distinct"):
        TabularFileSource(path, timestamp_col="ts", value_cols=["v", "v"])


def test_non_temporal_timestamp_dtype_raises(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "data.parquet", {"ts": [1, 2], "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="expected Datetime, Date"):
        TabularFileSource(path, timestamp_col="ts", value_cols=["v"])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_csv_string_timestamps_parsed_without_format(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "data.csv",
        "ts,v\n2024-01-01 02:00:00,2\n2024-01-01 00:00:00,0\n2024-01-01 01:00:00,1\n",
    )
    source = TabularFileSource(path, timestamp_col="ts", value_cols=["v"])
    loaded = source.load().collect()
    assert loaded["ts"].dtype == pl.Datetime
    # Sorted by timestamp, undeclared columns absent.
    assert loaded["ts"].to_list() == [T0, T0 + timedelta(hours=1), T0 + timedelta(hours=2)]
    assert loaded["v"].to_list() == [0, 1, 2]


def test_csv_string_timestamps_parsed_with_explicit_format(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.csv", "ts,v\n01/01/2024 00:00,0\n01/01/2024 01:00,1\n")
    source = TabularFileSource(
        path, timestamp_col="ts", value_cols=["v"], timestamp_format="%d/%m/%Y %H:%M"
    )
    loaded = source.load().collect()
    assert loaded["ts"].to_list() == [T0, T0 + timedelta(hours=1)]


def test_unparseable_timestamp_raises(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.csv", "ts,v\nnot-a-date,1\n")
    source = TabularFileSource(path, timestamp_col="ts", value_cols=["v"])
    with pytest.raises(ValueError, match="could not be parsed"):
        source.load()


def test_null_timestamp_raises(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.csv", "ts,v\n2024-01-01 00:00:00,1\n,2\n")
    source = TabularFileSource(path, timestamp_col="ts", value_cols=["v"])
    with pytest.raises(ValueError, match="null value"):
        source.load()


def test_duplicate_key_timestamp_raises(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "data.parquet",
        {
            "ts": [T0, T0, T0 + timedelta(hours=1)],
            "site": ["A", "A", "A"],
            "v": [1.0, 1.0, 2.0],
        },
    )
    source = TabularFileSource(path, timestamp_col="ts", value_cols=["v"], key_cols=["site"])
    with pytest.raises(ValueError, match="duplicate"):
        source.load()


def test_duplicate_timestamp_without_key_raises(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "data.parquet", {"ts": [T0, T0], "v": [1.0, 2.0]})
    source = TabularFileSource(path, timestamp_col="ts", value_cols=["v"])
    with pytest.raises(ValueError, match="duplicate"):
        source.load()


def test_date_timestamp_cast_to_datetime(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "data.parquet",
        {"ts": [T0.date(), (T0 + timedelta(days=1)).date()], "v": [1.0, 2.0]},
    )
    loaded = TabularFileSource(path, timestamp_col="ts", value_cols=["v"]).load().collect()
    assert loaded["ts"].dtype == pl.Datetime
    assert loaded["ts"].to_list() == [T0, T0 + timedelta(days=1)]


def test_load_keeps_only_declared_columns(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "data.parquet",
        {"ts": [T0], "v": [1.0], "extra": ["drop-me"], "station": ["A"]},
    )
    loaded = (
        TabularFileSource(path, timestamp_col="ts", value_cols=["v"], key_cols=["station"])
        .load()
        .collect()
    )
    assert loaded.columns == ["station", "ts", "v"]


def test_schema_property(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "data.parquet", {"ts": [T0], "site": ["A"], "v": [1.0]})
    source = TabularFileSource(path, timestamp_col="ts", value_cols=["v"], key_cols=["site"])
    assert source.schema.timestamp_col == "ts"
    assert source.schema.spatial_col == "site"
    assert source.schema.attribute_cols == ["v"]
    assert source.schema.has_temporal() and source.schema.has_spatial()
    no_key = TabularFileSource(path, timestamp_col="ts", value_cols=["v"])
    assert no_key.schema.spatial_col is None
    assert not no_key.schema.has_spatial()


# ---------------------------------------------------------------------------
# Joins fed by the connector
# ---------------------------------------------------------------------------


def _context(tmp_path: Path, rows: list[tuple[datetime, float]]) -> pl.LazyFrame:
    path = write_parquet(
        tmp_path / "ctx.parquet",
        {"context_time": [r[0] for r in rows], "weather": [r[1] for r in rows]},
    )
    return TabularFileSource(path, timestamp_col="context_time", value_cols=["weather"]).load()


def test_asof_backward_with_tolerance(tmp_path: Path) -> None:
    ef = frame({"time": [T0 + timedelta(hours=1), T0 + timedelta(hours=3)]})
    ctx = _context(tmp_path, [(T0, 10.0)])
    result = (
        TemporalContextJoinStep(ctx, "context_time", ["weather"], tolerance="1h").run(ef).collect()
    )
    # Sorted by time: 1 h away matches, 3 h away is beyond tolerance.
    assert result["weather"].to_list() == [10.0, None]


def test_asof_backward_without_tolerance_matches_any_past_row(tmp_path: Path) -> None:
    ef = frame({"time": [T0 + timedelta(days=30)]})
    ctx = _context(tmp_path, [(T0, 10.0)])
    result = TemporalContextJoinStep(ctx, "context_time", ["weather"]).run(ef).collect()
    assert result["weather"].to_list() == [10.0]


def test_exact_strategy_matches_only_exact_hours(tmp_path: Path) -> None:
    ef = frame({"time": [T0, T0 + timedelta(minutes=30)]})
    ctx = _context(tmp_path, [(T0, 10.0)])
    result = (
        TemporalContextJoinStep(ctx, "context_time", ["weather"], strategy="exact")
        .run(ef)
        .collect()
        .sort("time")
    )
    assert result["weather"].to_list() == [10.0, None]


def test_nearest_strategy_picks_closest_hour(tmp_path: Path) -> None:
    ef = frame({"time": [T0 + timedelta(minutes=20), T0 + timedelta(minutes=50)]})
    ctx = _context(tmp_path, [(T0, 10.0), (T0 + timedelta(hours=1), 20.0)])
    result = (
        TemporalContextJoinStep(
            ctx, "context_time", ["weather"], strategy="nearest", tolerance="1h"
        )
        .run(ef)
        .collect()
    )
    assert result["weather"].to_list() == [10.0, 20.0]


def test_unmatched_stay_null_unless_fill_value_given(tmp_path: Path) -> None:
    ef = frame({"time": [T0 + timedelta(hours=3)]})
    ctx = _context(tmp_path, [(T0, 10.0)])
    plain = TemporalContextJoinStep(ctx, "context_time", ["weather"], tolerance="1h").run(ef)
    assert plain.collect()["weather"].to_list() == [None]
    filled = TemporalContextJoinStep(
        ctx, "context_time", ["weather"], tolerance="1h", fill_value=-999.0
    ).run(ef)
    assert filled.collect()["weather"].to_list() == [-999.0]


def test_spatiotemporal_join_isolates_groups(tmp_path: Path) -> None:
    ef = frame({"time": [T0 + timedelta(hours=2)] * 2, "site": ["A", "B"]})
    path = write_parquet(
        tmp_path / "ctx.parquet",
        {
            "ct": [T0, T0 + timedelta(hours=1)],
            "site": ["A", "B"],
            "weather": [1.0, 2.0],
        },
    )
    ctx = TabularFileSource(
        path, timestamp_col="ct", value_cols=["weather"], key_cols=["site"]
    ).load()
    result = (
        SpatioTemporalContextJoinStep(ctx, "site", "ct", ["weather"], tolerance="1h")
        .run(ef)
        .collect()
    )
    # Site A is 2 h past its row (beyond tolerance); site B is 1 h past.
    # Neither event may borrow the other site's weather.
    assert result["weather"].to_list() == [None, 2.0]


def test_spatial_join_from_loaded_table(tmp_path: Path) -> None:
    ef = frame({"x": [0.5, 2.0], "y": [0.5, 0.5]})
    path = write_parquet(
        tmp_path / "ctx.parquet",
        {"ts": [T0], "geometry": [box(0, 0, 1, 1).wkt], "label": ["inside"]},
    )
    ctx = TabularFileSource(path, timestamp_col="ts", value_cols=["geometry", "label"]).load()
    result = SpatialContextJoinStep(ctx.collect(), "geometry", ["label"]).run(ef).collect()
    assert result["label"].to_list() == ["inside", None]


def test_enricher_step_accepts_tabular_source(tmp_path: Path) -> None:
    ef = frame({"time": [T0 + timedelta(minutes=30), T0 + timedelta(hours=5)]})
    path = write_parquet(tmp_path / "ctx.parquet", {"context_time": [T0], "weather": [10.0]})
    source = TabularFileSource(path, timestamp_col="context_time", value_cols=["weather"])
    result = EnricherStep(source, TemporalJoin(strategy="backward", tolerance="1h")).run(ef)
    assert result.collect()["weather"].to_list() == [10.0, None]
