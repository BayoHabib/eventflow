"""Equivalence, reference, leakage and validation tests for the vectorized
spatial lag (Phase 2, Decisions 1 + 2).

The old (Phase 1) implementation of ``SpatialLagStep.run()`` is kept here as
``_old_spatial_lag`` — copied verbatim from
``src/eventflow/core/steps/spatial.py`` at main @ ``36aa7ec``, minus the
provenance registration (which does not affect values) — and used as the
oracle for the vectorized rewrite.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.spatial import (
    MultiScaleSpatialLagConfig,
    MultiScaleSpatialLagStep,
    SpatialLagStep,
    _spatial_result,
    _spatial_snapshots,
)
from eventflow.core.weights import SpatialWeights

# -----------------------------------------------------------------------------
# Old implementation (oracle), copied verbatim from Phase 1
# -----------------------------------------------------------------------------


def _old_spatial_lag(
    event_frame: EventFrame,
    value_cols: list[str],
    *,
    neighbor_col: str = "grid_id",
    weights: str = "queen",
    k: int = 4,
    distance_threshold: float | None = None,
    group_cols: list[str] | None = None,
    spatial_units: pl.DataFrame | None = None,
    spatial_weights: SpatialWeights | None = None,
) -> EventFrame:
    """Pre-rewrite SpatialLagStep.run() (Phase 1), kept as the equivalence oracle."""
    outputs = []
    for part, stats, rows in _spatial_snapshots(
        event_frame,
        neighbor_col,
        value_cols,
        list(group_cols or []),
        weights,
        k,
        distance_threshold,
        spatial_units,
        spatial_weights,
    ):
        features = []
        for col in value_cols:
            values = stats[col].to_numpy()
            lag = [float(values[list(row)].mean()) if row else None for row in rows]
            features.append(pl.Series(f"{col}_spatial_lag", lag, dtype=pl.Float64))
        lookup = stats.select(neighbor_col).with_columns(features)
        outputs.append(
            part.join(lookup, on=neighbor_col, how="left", validate="m:1", maintain_order="left")
        )
    return _spatial_result(
        event_frame, outputs, {f"{col}_spatial_lag": pl.Float64 for col in value_cols}
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _bin(day: int, hour: int = 0) -> datetime:
    return datetime(2024, 1, day, hour)


def _make_panel(
    units: list[tuple[int, int, int]],
    bins: list[datetime],
    values: dict[str, Callable[[int, int, int, int], float | int | str]],
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> EventFrame:
    """One row per (unit, bin); ``values[col]`` is fn(grid_id, row, col, bin_idx)."""
    data: dict[str, list] = {
        "grid_id": [],
        "grid_row": [],
        "grid_col": [],
        "time_bin": [],
        lat_col: [],
        lon_col: [],
    }
    for col in values:
        data[col] = []
    for bin_idx, time_bin in enumerate(bins):
        for grid_id, row, col in units:
            data["grid_id"].append(grid_id)
            data["grid_row"].append(row)
            data["grid_col"].append(col)
            data["time_bin"].append(time_bin)
            data[lat_col].append(41.8781)
            data[lon_col].append(-87.6298)
            for name, fn in values.items():
                data[name].append(fn(grid_id, row, col, bin_idx))
    lf = pl.LazyFrame(data)
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col=lat_col,
        lon_col=lon_col,
        numeric_cols=list(values),
        categorical_cols=[],
    )
    metadata = EventMetadata(dataset_name="test-lag", crs="EPSG:26971", time_bin="1d")
    return EventFrame(lf, schema, metadata)


def _grid_units(n_rows: int, n_cols: int) -> list[tuple[int, int, int]]:
    """grid_id = row * n_cols + col + 1, rows/cols 0-based."""
    return [(r * n_cols + c + 1, r, c) for r in range(n_rows) for c in range(n_cols)]


def _assert_lag_equal(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    lag_cols: list[str],
    key_cols: list[str],
    *,
    tol: float = 1e-9,
) -> None:
    """Same rows, same lag values to ``tol`` (null == null), after key sort."""
    assert actual.height == expected.height
    assert actual.columns == expected.columns
    a = actual.sort(key_cols)
    e = expected.sort(key_cols)
    non_lag = [c for c in actual.columns if c not in lag_cols]
    assert_frame_equal(a.select(non_lag), e.select(non_lag))
    for col in lag_cols:
        assert a[col].is_null().to_list() == e[col].is_null().to_list(), col
        diff = (a[col].fill_null(0.0) - e[col].fill_null(0.0)).abs()
        assert (diff <= tol).all(), col


def _brute_force_ring_means(
    values: dict[tuple[int, int], float], rings: list[int]
) -> dict[tuple[int, int, int], float | None]:
    """Independent oracle: mean of values at Chebyshev distance exactly r."""
    out: dict[tuple[int, int, int], float | None] = {}
    for (row, col), _ in values.items():
        for ring in rings:
            neighbours = [
                values[(r, c)] for (r, c) in values if max(abs(r - row), abs(c - col)) == ring
            ]
            out[(row, col, ring)] = sum(neighbours) / len(neighbours) if neighbours else None
    return out


# -----------------------------------------------------------------------------
# Equivalence: new SpatialLagStep vs the Phase 1 oracle
# -----------------------------------------------------------------------------


class TestVectorizedEquivalence:
    """New == old to 1e-9 on synthetic panels and on a Chicago slice."""

    @pytest.mark.parametrize("weights", ["queen", "rook"])
    def test_queen_grid_with_island(self, weights: str) -> None:
        units = _grid_units(3, 3) + [(99, 10, 10)]  # unit 99 is an island
        bins = [_bin(1), _bin(2), _bin(3)]
        frame = _make_panel(
            units, bins, {"count": lambda gid, r, c, bi: float((r * 3 + c + 1) * (bi + 1))}
        )
        step = SpatialLagStep(
            ["count"], neighbor_col="grid_id", weights=weights, group_cols=["time_bin"]
        )
        actual = step.run(frame).collect()
        expected = _old_spatial_lag(
            frame, ["count"], weights=weights, group_cols=["time_bin"]
        ).collect()
        _assert_lag_equal(actual, expected, ["count_spatial_lag"], ["grid_id", "time_bin"])
        # The island has no neighbours in any bin.
        assert actual.filter(pl.col("grid_id") == 99)["count_spatial_lag"].is_null().all()

    def test_multiple_value_cols(self) -> None:
        units = _grid_units(4, 4)
        bins = [_bin(1), _bin(2)]
        frame = _make_panel(
            units,
            bins,
            {
                "count": lambda gid, r, c, bi: float(r * 4 + c + bi),
                "count2": lambda gid, r, c, bi: float((r * 4 + c) * 2 + bi + 1),
            },
        )
        step = SpatialLagStep(["count", "count2"], group_cols=["time_bin"])
        actual = step.run(frame).collect()
        expected = _old_spatial_lag(frame, ["count", "count2"], group_cols=["time_bin"]).collect()
        _assert_lag_equal(
            actual,
            expected,
            ["count_spatial_lag", "count2_spatial_lag"],
            ["grid_id", "time_bin"],
        )

    def test_two_group_cols(self) -> None:
        # sector is a static per-unit attribute; snapshots are (time_bin, sector).
        units = _grid_units(3, 3)
        bins = [_bin(1), _bin(2), _bin(3)]
        frame = _make_panel(
            units,
            bins,
            {
                "count": lambda gid, r, c, bi: float(gid + bi),
                "sector": lambda gid, r, c, bi: "north" if r < 2 else "south",
            },
        )
        step = SpatialLagStep(["count"], group_cols=["time_bin", "sector"])
        actual = step.run(frame).collect()
        expected = _old_spatial_lag(frame, ["count"], group_cols=["time_bin", "sector"]).collect()
        _assert_lag_equal(actual, expected, ["count_spatial_lag"], ["grid_id", "time_bin"])

    def test_no_group_cols_single_snapshot(self) -> None:
        # Without group_cols the oracle aggregates each unit's mean over all bins.
        units = _grid_units(3, 4)
        bins = [_bin(1), _bin(2), _bin(3)]
        frame = _make_panel(units, bins, {"count": lambda gid, r, c, bi: float(gid * (bi + 1))})
        step = SpatialLagStep(["count"])
        actual = step.run(frame).collect()
        expected = _old_spatial_lag(frame, ["count"]).collect()
        _assert_lag_equal(actual, expected, ["count_spatial_lag"], ["grid_id", "time_bin"])

    def test_supplied_spatial_weights(self) -> None:
        units = _grid_units(4, 4)
        bins = [_bin(1), _bin(2)]
        frame = _make_panel(units, bins, {"count": lambda gid, r, c, bi: float((gid + bi) % 7)})
        units_df = pl.DataFrame(
            {
                "grid_id": [u[0] for u in units],
                "grid_row": [u[1] for u in units],
                "grid_col": [u[2] for u in units],
            }
        )
        weights = SpatialWeights.from_units(units_df, "grid_id", method="queen")
        step = SpatialLagStep(["count"], group_cols=["time_bin"], spatial_weights=weights)
        actual = step.run(frame).collect()
        expected = _old_spatial_lag(
            frame, ["count"], group_cols=["time_bin"], spatial_weights=weights
        ).collect()
        _assert_lag_equal(actual, expected, ["count_spatial_lag"], ["grid_id", "time_bin"])

    def test_knn_weights_from_coordinates(self) -> None:
        # No grid_row/grid_col: the reference falls back to lon/lat (metric x/y).
        data: dict[str, list] = {
            "grid_id": [],
            "time_bin": [],
            "count": [],
            "latitude": [],
            "longitude": [],
        }
        bins = [_bin(1), _bin(2)]
        for bin_idx, time_bin in enumerate(bins):
            for gid in range(1, 10):
                row, col = divmod(gid - 1, 3)
                data["grid_id"].append(gid)
                data["time_bin"].append(time_bin)
                data["count"].append(float(gid + bin_idx))
                data["latitude"].append(float(row * 500))
                data["longitude"].append(float(col * 500))
        schema = EventSchema(
            timestamp_col="time_bin",
            lat_col="latitude",
            lon_col="longitude",
            numeric_cols=["count"],
        )
        frame = EventFrame(
            pl.LazyFrame(data),
            schema,
            EventMetadata(dataset_name="test-lag", crs="EPSG:26971", time_bin="1d"),
        )
        step = SpatialLagStep(["count"], weights="knn", k=3, group_cols=["time_bin"])
        actual = step.run(frame).collect()
        expected = _old_spatial_lag(
            frame, ["count"], weights="knn", k=3, group_cols=["time_bin"]
        ).collect()
        _assert_lag_equal(actual, expected, ["count_spatial_lag"], ["grid_id", "time_bin"])

    def test_large_synthetic_panel(self) -> None:
        units = _grid_units(20, 20)
        bins = [_bin(d) for d in range(1, 6)]
        frame = _make_panel(
            units,
            bins,
            {"count": lambda gid, r, c, bi: float((r * 7 + c * 13 + bi * 3) % 11)},
        )
        step = SpatialLagStep(["count"], group_cols=["time_bin"])
        actual = step.run(frame).collect()
        expected = _old_spatial_lag(frame, ["count"], group_cols=["time_bin"]).collect()
        _assert_lag_equal(actual, expected, ["count_spatial_lag"], ["grid_id", "time_bin"])

    def test_row_order_preserved(self) -> None:
        # The lag join must not reorder rows: shuffle the panel and check the
        # output row order (grid_id, time_bin) matches the input order.
        units = _grid_units(3, 3)
        bins = [_bin(1), _bin(2)]
        frame = _make_panel(units, bins, {"count": lambda gid, r, c, bi: float(gid + bi)})
        shuffled = frame.with_lazy_frame(frame.lazy_frame.reverse())
        result = SpatialLagStep(["count"], group_cols=["time_bin"]).run(shuffled).collect()
        expect = shuffled.collect().select(["grid_id", "time_bin"])
        assert_frame_equal(result.select(["grid_id", "time_bin"]), expect)
        multi = (
            MultiScaleSpatialLagStep(["count"], rings=(1, 2), group_cols=["time_bin"])
            .run(shuffled)
            .collect()
        )
        assert_frame_equal(multi.select(["grid_id", "time_bin"]), expect)


# -----------------------------------------------------------------------------
# Chicago slice (real data; skipped when the parquet is not in this checkout)
# -----------------------------------------------------------------------------


def _chicago_slice() -> EventFrame | None:
    """Completed panel, 5x5 cells around grid (1930, 1195) x 19 bins of 6 h.

    Built from the processed Chicago features parquet (sparse observed counts):
    units are the cells present in the parquet inside the window, bins run
    2024-02-13 00:00 .. 2024-02-17 12:00, missing pairs are explicit zeros,
    and ``count_lag_1`` is the count 1 day (4 bins) earlier, nulls filled
    with 0 — the recipe's boundary policy for the lag feeding the spatial lag.
    """
    candidates = [
        Path(os.environ.get("CHICAGO_FEATURES_PARQUET", "")),
        Path(__file__).resolve().parents[4] / "data" / "processed" / "chicago_features.parquet",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None
    raw = pl.read_parquet(path)
    window = raw.filter(
        pl.col("grid_row").is_between(1928, 1932) & pl.col("grid_col").is_between(1193, 1197)
    )
    units = window.select("grid_id", "grid_row", "grid_col").unique().sort("grid_id")
    centres = window.select("grid_id", "longitude_proj", "latitude_proj").unique()
    units = units.join(centres, on="grid_id", how="left")
    bins = [datetime(2024, 2, 13) + timedelta(hours=6 * i) for i in range(19)]
    panel = units.join(pl.DataFrame({"time_bin": bins}), how="cross")
    counts = window.filter(pl.col("time_bin").is_between(bins[0], bins[-1])).select(
        "grid_id", "time_bin", "count"
    )
    panel = (
        panel.join(counts, on=["grid_id", "time_bin"], how="left")
        .with_columns(pl.col("count").fill_null(0))
        .sort(["grid_id", "time_bin"])
        .with_columns(pl.col("count").shift(4).over("grid_id").alias("count_lag_1"))
        .with_columns(pl.col("count_lag_1").fill_null(0))
        .sort(["time_bin", "grid_id"])
    )
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col="latitude_proj",
        lon_col="longitude_proj",
        numeric_cols=["count", "count_lag_1"],
    )
    metadata = EventMetadata(dataset_name="test-chicago-slice", crs="EPSG:26971", time_bin="6h")
    return EventFrame(panel.lazy(), schema, metadata)


class TestChicagoSlice:
    """Equivalence on real Chicago counts + the design-note pin value 0.375."""

    def test_equivalence_on_chicago_slice(self) -> None:
        frame = _chicago_slice()
        if frame is None:
            pytest.skip("Chicago features parquet not available in this checkout")
        step = SpatialLagStep(["count_lag_1"], group_cols=["time_bin"])
        actual = step.run(frame).collect()
        expected = _old_spatial_lag(frame, ["count_lag_1"], group_cols=["time_bin"]).collect()
        _assert_lag_equal(actual, expected, ["count_lag_1_spatial_lag"], ["grid_id", "time_bin"])

    def test_busiest_cell_pin_value(self) -> None:
        frame = _chicago_slice()
        if frame is None:
            pytest.skip("Chicago features parquet not available in this checkout")
        result = SpatialLagStep(["count_lag_1"], group_cols=["time_bin"]).run(frame).collect()
        row = result.filter(
            (pl.col("grid_row") == 1930)
            & (pl.col("grid_col") == 1195)
            & (pl.col("time_bin") == datetime(2024, 2, 17, 12))
        )
        assert row.height == 1
        # Neighbours' counts on 2024-02-16 12:00 sum to 3 over 8 queen neighbours.
        assert row["count_lag_1_spatial_lag"][0] == pytest.approx(0.375)

    def test_multiscale_ring1_matches_single_scale_on_chicago_slice(self) -> None:
        frame = _chicago_slice()
        if frame is None:
            pytest.skip("Chicago features parquet not available in this checkout")
        single = SpatialLagStep(["count_lag_1"], group_cols=["time_bin"]).run(frame).collect()
        multi = (
            MultiScaleSpatialLagStep(["count_lag_1"], rings=(1, 2, 3), group_cols=["time_bin"])
            .run(frame)
            .collect()
        )
        keys = ["grid_id", "time_bin"]
        a = single.sort(keys)["count_lag_1_spatial_lag"]
        b = multi.sort(keys)["count_lag_1_spatial_lag_r1"]
        assert a.is_null().to_list() == b.is_null().to_list()
        assert ((a.fill_null(0.0) - b.fill_null(0.0)).abs() <= 1e-9).all()


# -----------------------------------------------------------------------------
# Multi-scale reference tests (hand-computed values)
# -----------------------------------------------------------------------------

# 3x3 grid, values 1..9 row-major:
#   1 2 3
#   4 5 6
#   7 8 9
_HAND_3X3: dict[tuple[int, int], tuple[float | None, float | None, float | None]] = {
    # (row, col): (ring1, ring2, ring3)
    (0, 0): (11 / 3, 33 / 5, None),
    (0, 1): (19 / 5, 24 / 3, None),
    (0, 2): (13 / 3, 29 / 5, None),
    (1, 0): (23 / 5, 18 / 3, None),
    (1, 1): (40 / 8, None, None),
    (1, 2): (27 / 5, 12 / 3, None),
    (2, 0): (17 / 3, 21 / 5, None),
    (2, 1): (31 / 5, 6 / 3, None),
    (2, 2): (19 / 3, 17 / 5, None),
}


class TestMultiScaleReference:
    """Hand-computed ring means on a 3x3 grid; brute-force oracle on 5x5."""

    def test_hand_computed_3x3(self) -> None:
        units = _grid_units(3, 3)
        frame = _make_panel(units, [_bin(1)], {"count": lambda gid, r, c, bi: float(r * 3 + c + 1)})
        result = MultiScaleSpatialLagStep(["count"], rings=(1, 2, 3)).run(frame).collect()
        assert result.height == 9
        for (row, col), (r1, r2, r3) in _HAND_3X3.items():
            cell = result.filter((pl.col("grid_row") == row) & (pl.col("grid_col") == col))
            assert cell.height == 1
            for out_col, expected in [
                ("count_spatial_lag_r1", r1),
                ("count_spatial_lag_r2", r2),
                ("count_spatial_lag_r3", r3),
            ]:
                got = cell[out_col][0]
                if expected is None:
                    assert got is None, (row, col, out_col)
                else:
                    assert got == pytest.approx(expected), (row, col, out_col)

    def test_ring1_reproduces_single_scale(self) -> None:
        units = _grid_units(4, 5)
        bins = [_bin(1), _bin(2)]
        frame = _make_panel(
            units, bins, {"count": lambda gid, r, c, bi: float((gid * (bi + 2)) % 9)}
        )
        single = SpatialLagStep(["count"], group_cols=["time_bin"]).run(frame).collect()
        multi = (
            MultiScaleSpatialLagStep(["count"], rings=(1, 2, 3), group_cols=["time_bin"])
            .run(frame)
            .collect()
        )
        keys = ["grid_id", "time_bin"]
        a = single.sort(keys)["count_spatial_lag"]
        b = multi.sort(keys)["count_spatial_lag_r1"]
        assert a.is_null().to_list() == b.is_null().to_list()
        assert ((a.fill_null(0.0) - b.fill_null(0.0)).abs() <= 1e-9).all()

    def test_brute_force_5x5_all_rings(self) -> None:
        units = _grid_units(5, 5)
        bins = [_bin(1), _bin(2)]
        values = {(r, c): float((r * 7 + c * 11 + 3) % 13) for r in range(5) for c in range(5)}
        frame = _make_panel(
            units,
            bins,
            {
                "count": lambda gid, r, c, bi: values[(r, c)] + 10 * bi,
                "count2": lambda gid, r, c, bi: values[(r, c)] * 2 - bi,
            },
        )
        result = (
            MultiScaleSpatialLagStep(["count", "count2"], rings=(1, 2, 3), group_cols=["time_bin"])
            .run(frame)
            .collect()
        )
        for bin_idx, time_bin in enumerate(bins):
            shifted = {k: v + 10 * bin_idx for k, v in values.items()}
            shifted2 = {k: v * 2 - bin_idx for k, v in values.items()}
            expected = _brute_force_ring_means(shifted, [1, 2, 3])
            expected2 = _brute_force_ring_means(shifted2, [1, 2, 3])
            for (row, col, ring), want in expected.items():
                cell = result.filter(
                    (pl.col("grid_row") == row)
                    & (pl.col("grid_col") == col)
                    & (pl.col("time_bin") == time_bin)
                )
                assert cell.height == 1
                got = cell[f"count_spatial_lag_r{ring}"][0]
                if want is None:
                    assert got is None
                else:
                    assert got == pytest.approx(want)
            for (row, col, ring), want in expected2.items():
                cell = result.filter(
                    (pl.col("grid_row") == row)
                    & (pl.col("grid_col") == col)
                    & (pl.col("time_bin") == time_bin)
                )
                got = cell[f"count2_spatial_lag_r{ring}"][0]
                if want is None:
                    assert got is None
                else:
                    assert got == pytest.approx(want)

    def test_rings_are_disjoint_not_cumulative(self) -> None:
        # Corner (0,0) of the 3x3: ring 2 (6.6) excludes the ring-1 cells
        # (2, 4, 5) — a cumulative ring 2 would give (2+4+5+3+6+7+8+9)/8 = 5.5.
        units = _grid_units(3, 3)
        frame = _make_panel(units, [_bin(1)], {"count": lambda gid, r, c, bi: float(r * 3 + c + 1)})
        result = MultiScaleSpatialLagStep(["count"], rings=(2,)).run(frame).collect()
        assert result.columns.count("count_spatial_lag_r2") == 1
        assert "count_spatial_lag_r1" not in result.columns
        corner = result.filter((pl.col("grid_row") == 0) & (pl.col("grid_col") == 0))
        assert corner["count_spatial_lag_r2"][0] == pytest.approx(6.6)

    def test_spatial_units_reference(self) -> None:
        # Panel without grid_row/grid_col; the ring geometry comes from spatial_units.
        units = _grid_units(3, 3)
        frame = _make_panel(units, [_bin(1)], {"count": lambda gid, r, c, bi: float(gid)})
        bare = frame.with_lazy_frame(frame.lazy_frame.drop(["grid_row", "grid_col"]))
        units_df = pl.DataFrame(
            {
                "grid_id": [u[0] for u in units],
                "grid_row": [u[1] for u in units],
                "grid_col": [u[2] for u in units],
            }
        )
        result = (
            MultiScaleSpatialLagStep(["count"], rings=(1,), spatial_units=units_df)
            .run(bare)
            .collect()
        )
        centre = result.filter(pl.col("grid_id") == 5)
        assert centre["count_spatial_lag_r1"][0] == pytest.approx(5.0)

    def test_registers_provenance(self) -> None:
        units = _grid_units(3, 3)
        frame = _make_panel(units, [_bin(1)], {"count": lambda gid, r, c, bi: float(gid)})
        result = MultiScaleSpatialLagStep(["count"], rings=(1, 2)).run(frame)
        provenance = result.metadata.feature_provenance["count_spatial_lag_r2"]
        assert provenance.produced_by == "MultiScaleSpatialLagStep"
        assert provenance.metadata["rings"] == [1, 2]
        assert provenance.metadata["ring"] == 2
        assert provenance.metadata["normalization"] == "row"
        assert provenance.metadata["islands"] == "null"


# -----------------------------------------------------------------------------
# Leakage (cutoff) tests
# -----------------------------------------------------------------------------


class TestLeakage:
    """Changing bins > t must not change outputs at bins <= t."""

    def _leakage_panel(self) -> tuple[EventFrame, list[datetime]]:
        units = _grid_units(4, 4)
        bins = [_bin(d) for d in range(1, 5)]
        frame = _make_panel(
            units,
            bins,
            {"count": lambda gid, r, c, bi: float((r * 4 + c + 1) * (bi + 1))},
        )
        return frame, bins

    @staticmethod
    def _mutate_after(frame: EventFrame, cutoff: datetime) -> EventFrame:
        df = frame.collect().with_columns(
            pl.when(pl.col("time_bin") > cutoff)
            .then(pl.col("count") * 100.0)
            .otherwise(pl.col("count"))
            .alias("count")
        )
        return frame.with_lazy_frame(df.lazy())

    def test_spatial_lag_cutoff(self) -> None:
        frame, bins = self._leakage_panel()
        cutoff = bins[1]
        base = SpatialLagStep(["count"], group_cols=["time_bin"]).run(frame).collect()
        mutated = (
            SpatialLagStep(["count"], group_cols=["time_bin"])
            .run(self._mutate_after(frame, cutoff))
            .collect()
        )

        def early(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("time_bin") <= cutoff).sort(["grid_id", "time_bin"])

        a, b = early(base), early(mutated)
        assert a["count_spatial_lag"].to_list() == b["count_spatial_lag"].to_list()

        # Control: the step does read same-bin values — later bins do change.
        def late(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("time_bin") > cutoff).sort(["grid_id", "time_bin"])

        assert (
            late(base)["count_spatial_lag"].to_list()
            != late(mutated)["count_spatial_lag"].to_list()
        )

    def test_multiscale_lag_cutoff(self) -> None:
        frame, bins = self._leakage_panel()
        cutoff = bins[1]
        step = MultiScaleSpatialLagStep(["count"], rings=(1, 2, 3), group_cols=["time_bin"])
        base = step.run(frame).collect()
        mutated = step.run(self._mutate_after(frame, cutoff)).collect()
        lag_cols = [f"count_spatial_lag_r{r}" for r in (1, 2, 3)]

        def early(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(pl.col("time_bin") <= cutoff).sort(["grid_id", "time_bin"])

        a, b = early(base), early(mutated)
        for col in lag_cols:
            assert a[col].to_list() == b[col].to_list(), col

    def test_same_bin_change_moves_neighbours_lag(self) -> None:
        # Sanity control: changing one cell's value at bin t changes its
        # neighbours' lags at bin t (the step reads same-bin values).
        frame, bins = self._leakage_panel()
        step = SpatialLagStep(["count"], group_cols=["time_bin"])
        base = step.run(frame).collect()
        df = frame.collect().with_columns(
            pl.when((pl.col("time_bin") == bins[0]) & (pl.col("grid_id") == 1))
            .then(pl.col("count") + 1000.0)
            .otherwise(pl.col("count"))
            .alias("count")
        )
        changed = step.run(frame.with_lazy_frame(df.lazy())).collect()

        def cell(d: pl.DataFrame) -> float | None:
            return d.filter((pl.col("time_bin") == bins[0]) & (pl.col("grid_id") == 2))[
                "count_spatial_lag"
            ][0]

        assert cell(changed) != cell(base)


# -----------------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------------


class TestValidation:
    """Blocking errors for bad values, bad rings and missing references."""

    def test_null_values_raise(self) -> None:
        units = _grid_units(3, 3)
        frame = _make_panel(
            units,
            [_bin(1)],
            {"count": lambda gid, r, c, bi: None if gid == 5 else float(gid)},
        )
        with pytest.raises(ValueError, match="finite"):
            SpatialLagStep(["count"]).run(frame)
        with pytest.raises(ValueError, match="finite"):
            MultiScaleSpatialLagStep(["count"]).run(frame)

    def test_nonfinite_values_raise(self) -> None:
        units = _grid_units(3, 3)
        frame = _make_panel(
            units,
            [_bin(1)],
            {"count": lambda gid, r, c, bi: float("inf") if gid == 5 else float(gid)},
        )
        with pytest.raises(ValueError, match="finite"):
            SpatialLagStep(["count"]).run(frame)

    def test_missing_value_column_raises(self) -> None:
        units = _grid_units(3, 3)
        frame = _make_panel(units, [_bin(1)], {"count": lambda gid, r, c, bi: float(gid)})
        with pytest.raises(ValueError, match="Missing required column"):
            SpatialLagStep(["nope"]).run(frame)

    @pytest.mark.parametrize("rings", [[], [0], [-1], [1, 0], [1.5], [True], [1, 1], "12", 3])
    def test_bad_rings_raise(self, rings: object) -> None:
        with pytest.raises(ValueError):
            MultiScaleSpatialLagStep(["count"], rings=rings)  # type: ignore[arg-type]

    def test_rings_tuple_accepted(self) -> None:
        step = MultiScaleSpatialLagStep(["count"], rings=(2, 3))
        assert step.rings == [2, 3]

    def test_config_model_validates_rings(self) -> None:
        assert MultiScaleSpatialLagConfig(value_cols=["count"]).rings == [1, 2, 3]
        with pytest.raises(ValidationError):
            MultiScaleSpatialLagConfig(value_cols=["count"], rings=[0])
        with pytest.raises(ValidationError):
            MultiScaleSpatialLagConfig(value_cols=["count"], rings=[])

    def test_supplied_weights_rejected_for_multiscale(self) -> None:
        units = _grid_units(3, 3)
        units_df = pl.DataFrame(
            {
                "grid_id": [u[0] for u in units],
                "grid_row": [u[1] for u in units],
                "grid_col": [u[2] for u in units],
            }
        )
        weights = SpatialWeights.from_units(units_df, "grid_id", method="queen")
        with pytest.raises(ValueError, match="cannot derive queen rings"):
            MultiScaleSpatialLagStep(["count"], spatial_weights=weights)

    def test_multiscale_without_grid_reference_raises(self) -> None:
        units = _grid_units(3, 3)
        frame = _make_panel(units, [_bin(1)], {"count": lambda gid, r, c, bi: float(gid)})
        bare = frame.with_lazy_frame(frame.lazy_frame.drop(["grid_row", "grid_col"]))
        with pytest.raises(ValueError, match="grid_row/grid_col"):
            MultiScaleSpatialLagStep(["count"]).run(bare)

    def test_empty_panel_gains_null_lag_columns(self) -> None:
        schema = {
            "grid_id": pl.Int64,
            "grid_row": pl.Int64,
            "grid_col": pl.Int64,
            "time_bin": pl.Datetime,
            "count": pl.Float64,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
        }
        lf = pl.LazyFrame({name: [] for name in schema}, schema=schema)
        frame = EventFrame(
            lf,
            EventSchema(
                timestamp_col="time_bin",
                lat_col="latitude",
                lon_col="longitude",
                numeric_cols=["count"],
            ),
            EventMetadata(dataset_name="test-empty", crs="EPSG:26971", time_bin="1d"),
        )
        single = SpatialLagStep(["count"], group_cols=["time_bin"]).run(frame).collect()
        assert single.height == 0
        assert single["count_spatial_lag"].dtype == pl.Float64
        multi = MultiScaleSpatialLagStep(["count"], rings=(1, 2)).run(frame).collect()
        assert multi.height == 0
        assert multi["count_spatial_lag_r1"].dtype == pl.Float64
        assert multi["count_spatial_lag_r2"].dtype == pl.Float64
