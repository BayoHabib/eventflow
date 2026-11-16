"""Tests for density steps (Gaussian KDE and space-time KDE).

All reference values are computed by hand or by an independent
brute-force loop in the test (plain Python, never the step itself).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from shikong.core.event_frame import EventFrame
from shikong.core.schema import EventMetadata, EventSchema
from shikong.core.steps.density import (
    KernelDensityConfig,
    KernelDensityStep,
    SpaceTimeKernelDensityConfig,
    SpaceTimeKernelDensityStep,
)

# 3x3 grid, 300 m spacing; centres at (300*r, 300*c) so that the
# centre of cell (r, c) is exactly 300 m from its orthogonal
# neighbours and 300*sqrt(2) m from its diagonal neighbours.
SPACING_M = 300.0
CELLS_3X3 = {
    grid_id: (SPACING_M * (grid_id % 3), SPACING_M * (grid_id // 3)) for grid_id in range(9)
}
BIN0 = datetime(2024, 1, 1)
BIN1 = datetime(2024, 1, 2)
BIN2 = datetime(2024, 1, 3)


def make_panel(
    cells: dict[int, tuple[float, float]],
    counts_by_bin: Sequence[Mapping[int, float | None]],
    *,
    crs: str = "EPSG:26971",
    time_bin: str = "1d",
    lon_col: str = "x",
    lat_col: str = "y",
) -> EventFrame:
    """Completed-panel fixture: every (cell, bin) present, static centres."""
    bins = [BIN0 + timedelta(days=i) for i in range(len(counts_by_bin))]
    rows: dict[str, list[Any]] = {
        "grid_id": [],
        "time_bin": [],
        "grid_row": [],
        "grid_col": [],
        lon_col: [],
        lat_col: [],
        "count": [],
    }
    for bin_idx, counts in enumerate(counts_by_bin):
        for grid_id, (x, y) in cells.items():
            rows["grid_id"].append(grid_id)
            rows["time_bin"].append(bins[bin_idx])
            rows["grid_row"].append(grid_id // 3)
            rows["grid_col"].append(grid_id % 3)
            rows[lon_col].append(x)
            rows[lat_col].append(y)
            rows["count"].append(counts.get(grid_id, 0.0))
    lf = pl.LazyFrame(rows).sort(["grid_id", "time_bin"], maintain_order=True)
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col=lat_col,
        lon_col=lon_col,
        numeric_cols=["count"],
        categorical_cols=["grid_id"],
    )
    metadata = EventMetadata(dataset_name="test-density", crs=crs, time_bin=time_bin)
    return EventFrame(lf, schema, metadata)


def brute_force_kde(
    centres: dict[int, tuple[float, float]],
    values: dict[int, float],
    bandwidth_m: float,
) -> dict[int, float]:
    """Independent Gaussian KDE: sum over neighbours within 3h, by hand."""
    norm = 2.0 * math.pi * bandwidth_m**2
    out: dict[int, float] = {}
    for i, (xi, yi) in centres.items():
        total = 0.0
        for j, (xj, yj) in centres.items():
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            if d2 <= (3.0 * bandwidth_m) ** 2:
                total += values.get(j, 0.0) * math.exp(-d2 / (2.0 * bandwidth_m**2)) / norm
        out[i] = total
    return out


def brute_force_stkde(
    centres: dict[int, tuple[float, float]],
    values_by_bin: list[dict[int, float]],
    bandwidth_m: float,
    halflife_bins: float,
) -> list[dict[int, float]]:
    """Independent STKDE: per-bin KDE, then exact exponential decay sum."""
    kdes = [brute_force_kde(centres, values, bandwidth_m) for values in values_by_bin]
    out: list[dict[int, float]] = []
    for t in range(len(values_by_bin)):
        row = {}
        for cell in centres:
            row[cell] = sum(kdes[s][cell] * 0.5 ** ((t - s) / halflife_bins) for s in range(t + 1))
        out.append(row)
    return out


def density_for(df: pl.DataFrame, grid_id: int, col: str, time_bin: datetime) -> float:
    """Density value of one (cell, bin) row."""
    rows = df.filter((pl.col("grid_id") == grid_id) & (pl.col("time_bin") == time_bin))[
        col
    ].to_list()
    assert len(rows) == 1
    return float(rows[0])


class TestKernelDensityStep:
    """Tests for KernelDensityStep."""

    def test_single_isolated_cell(self) -> None:
        """One cell: density is its own value / (2*pi*h^2)."""
        frame = make_panel({0: (0.0, 0.0)}, [{0: 5.0}])
        result = KernelDensityStep(value_col="count", bandwidth_m=600.0).run(frame)

        df = result.collect()
        assert "count_kde" in df.columns
        assert df["count_kde"].to_list() == pytest.approx([5.0 / (2.0 * math.pi * 600.0**2)])

    def test_two_cells_known_distance(self) -> None:
        """Two cells 300 m apart: hand-computed cross-contribution."""
        frame = make_panel({0: (0.0, 0.0), 1: (300.0, 0.0)}, [{0: 2.0, 1: 4.0}])
        result = KernelDensityStep(value_col="count", bandwidth_m=600.0).run(frame)

        df = result.collect()
        norm = 2.0 * math.pi * 600.0**2
        cross = math.exp(-(300.0**2) / (2.0 * 600.0**2))
        assert density_for(df, 0, "count_kde", BIN0) == pytest.approx((2.0 + 4.0 * cross) / norm)
        assert density_for(df, 1, "count_kde", BIN0) == pytest.approx((4.0 + 2.0 * cross) / norm)

    def test_far_cell_beyond_3h_is_ignored(self) -> None:
        """A cell 2000 m away (> 3h for h=600) contributes nothing."""
        frame = make_panel({0: (0.0, 0.0), 1: (2000.0, 0.0)}, [{0: 1.0, 1: 100.0}])
        result = KernelDensityStep(value_col="count", bandwidth_m=600.0).run(frame)

        df = result.collect()
        assert density_for(df, 0, "count_kde", BIN0) == pytest.approx(
            1.0 / (2.0 * math.pi * 600.0**2)
        )
        assert density_for(df, 1, "count_kde", BIN0) == pytest.approx(
            100.0 / (2.0 * math.pi * 600.0**2)
        )

    def test_3x3_grid_matches_brute_force_two_bins(self) -> None:
        """3x3 grid, two bins: every cell matches the brute-force KDE."""
        bin0_counts = {i: float(i) for i in range(9)}
        bin1_counts = {i: float(8 - i) for i in range(9)}
        frame = make_panel(CELLS_3X3, [bin0_counts, bin1_counts])
        result = KernelDensityStep(
            value_col="count", bandwidth_m=600.0, group_cols=["time_bin"]
        ).run(frame)

        df = result.collect()
        expected0 = brute_force_kde(CELLS_3X3, bin0_counts, 600.0)
        expected1 = brute_force_kde(CELLS_3X3, bin1_counts, 600.0)
        for grid_id in range(9):
            assert density_for(df, grid_id, "count_kde", BIN0) == pytest.approx(expected0[grid_id])
            assert density_for(df, grid_id, "count_kde", BIN1) == pytest.approx(expected1[grid_id])

    def test_density_never_null_for_zero_counts(self) -> None:
        """All-zero counts still give 0.0 (self-contribution), never null."""
        frame = make_panel(CELLS_3X3, [dict.fromkeys(range(9), 0.0)])
        result = KernelDensityStep(
            value_col="count", bandwidth_m=600.0, group_cols=["time_bin"]
        ).run(frame)

        df = result.collect()
        assert df["count_kde"].null_count() == 0
        assert df["count_kde"].to_list() == [0.0] * 9

    def test_grouping_does_not_pool_across_bins(self) -> None:
        """Bin 1 all 10s, bin 2 all 0s: bin 2 densities are 0, bin 1 are not."""
        frame = make_panel(
            CELLS_3X3,
            [dict.fromkeys(range(9), 10.0), dict.fromkeys(range(9), 0.0)],
        )
        result = KernelDensityStep(
            value_col="count", bandwidth_m=600.0, group_cols=["time_bin"]
        ).run(frame)

        df = result.collect()
        expected1 = brute_force_kde(CELLS_3X3, dict.fromkeys(range(9), 10.0), 600.0)
        for grid_id in range(9):
            assert density_for(df, grid_id, "count_kde", BIN0) == pytest.approx(expected1[grid_id])
            assert density_for(df, grid_id, "count_kde", BIN1) == 0.0

    def test_leakage_later_bins_do_not_change_earlier_bins(self) -> None:
        """Changing bin 2 counts leaves bin 1 densities bit-identical."""
        counts_a = [{i: float(i) for i in range(9)}, dict.fromkeys(range(9), 1.0)]
        counts_b = [{i: float(i) for i in range(9)}, dict.fromkeys(range(9), 99.0)]
        step = KernelDensityStep(value_col="count", bandwidth_m=600.0, group_cols=["time_bin"])
        df_a = step.run(make_panel(CELLS_3X3, counts_a)).collect()
        df_b = step.run(make_panel(CELLS_3X3, counts_b)).collect()

        bin1_a = df_a.filter(pl.col("time_bin") == BIN0).sort("grid_id")["count_kde"].to_list()
        bin1_b = df_b.filter(pl.col("time_bin") == BIN0).sort("grid_id")["count_kde"].to_list()
        # Parallel group_by summation order is not deterministic to the
        # last ulp, so compare to 1e-12 relative, not bit-for-bit.
        assert bin1_a == pytest.approx(bin1_b, rel=1e-12)

    def test_bandwidth_validation(self) -> None:
        """h <= 0 or non-finite must fail at construction."""
        for bad_h in (0.0, -600.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="bandwidth_m"):
                KernelDensityStep(value_col="count", bandwidth_m=bad_h)

    def test_null_value_raises(self) -> None:
        """A null value is a blocking error, never silently treated as zero."""
        frame = make_panel({0: (0.0, 0.0), 1: (300.0, 0.0)}, [{0: 1.0, 1: None}])
        step = KernelDensityStep(value_col="count", bandwidth_m=600.0)
        with pytest.raises(ValueError, match="null"):
            step.run(frame)

    def test_null_group_col_raises(self) -> None:
        """A null group column is a blocking error (Phase 2 Chicago replay).

        Grouping by a column that is null on generated panel rows
        silently merges them all into one group. The step must refuse
        a null group column.
        """
        frame = make_panel({0: (0.0, 0.0)}, [{0: 1.0}, {0: 2.0}])
        df = frame.collect().with_columns(
            pl.when(pl.col("time_bin") == BIN0 + timedelta(days=1))
            .then(None)
            .otherwise(pl.col("time_bin"))
            .alias("time_bin")
        )
        bad = EventFrame(df.lazy(), frame.schema, frame.metadata)
        step = KernelDensityStep(value_col="count", bandwidth_m=600.0, group_cols=["time_bin"])
        with pytest.raises(ValueError, match="non-null"):
            step.run(bad)

    def test_nonfinite_value_raises(self) -> None:
        """An infinite value is a blocking error."""
        frame = make_panel({0: (0.0, 0.0)}, [{0: float("inf")}])
        step = KernelDensityStep(value_col="count", bandwidth_m=600.0)
        with pytest.raises(ValueError, match="non-finite"):
            step.run(frame)

    def test_non_metric_crs_raises(self) -> None:
        """Geographic coordinates (degrees) cannot carry a metre bandwidth."""
        frame = make_panel({0: (0.0, 0.0)}, [{0: 1.0}], crs="EPSG:4326")
        step = KernelDensityStep(value_col="count", bandwidth_m=600.0)
        with pytest.raises(ValueError, match="projected CRS"):
            step.run(frame)

    def test_missing_centre_columns_raises(self) -> None:
        """No schema lon/lat columns means no cell centres to smooth over."""
        frame = make_panel({0: (0.0, 0.0)}, [{0: 1.0}])
        frame = frame.with_schema(lon_col=None, lat_col=None)
        step = KernelDensityStep(value_col="count", bandwidth_m=600.0)
        with pytest.raises(ValueError, match="lon/lat"):
            step.run(frame)

    def test_output_column_name_uses_value_col(self) -> None:
        """A lagged value column yields e.g. count_lag_1_kde."""
        rows = {
            "grid_id": [0],
            "time_bin": [BIN0],
            "grid_row": [0],
            "grid_col": [0],
            "x": [0.0],
            "y": [0.0],
            "count_lag_1": [3.0],
        }
        lf = pl.LazyFrame(rows)
        schema = EventSchema(
            timestamp_col="time_bin",
            lat_col="y",
            lon_col="x",
            numeric_cols=["count_lag_1"],
            categorical_cols=["grid_id"],
        )
        metadata = EventMetadata(dataset_name="test-density", crs="EPSG:26971", time_bin="1d")
        frame = EventFrame(lf, schema, metadata)
        result = KernelDensityStep(value_col="count_lag_1", bandwidth_m=600.0).run(frame)

        df = result.collect()
        assert "count_lag_1_kde" in df.columns
        assert df["count_lag_1_kde"].to_list() == pytest.approx([3.0 / (2.0 * math.pi * 600.0**2)])

    def test_registers_provenance(self) -> None:
        """KDE columns carry provenance with bandwidth, kernel and causality."""
        frame = make_panel({0: (0.0, 0.0)}, [{0: 1.0}])
        result = KernelDensityStep(value_col="count", bandwidth_m=600.0).run(frame)

        provenance = result.metadata.feature_provenance["count_kde"]
        assert provenance.produced_by == "KernelDensityStep"
        assert "spatial" in provenance.tags
        assert provenance.metadata["bandwidth_m"] == 600.0
        assert provenance.metadata["kernel"] == "gaussian"
        assert provenance.metadata["truncation_bandwidths"] == 3.0
        assert provenance.metadata["causality"] == "same-bin"

    def test_config_model(self) -> None:
        """The pydantic config round-trips the step parameters."""
        config = KernelDensityConfig(
            value_col="count", bandwidth_m=600.0, unit_col="grid_id", group_cols=["time_bin"]
        )
        step = KernelDensityStep(
            value_col=config.value_col,
            bandwidth_m=config.bandwidth_m,
            unit_col=config.unit_col,
            group_cols=config.group_cols,
        )
        assert step.bandwidth_m == 600.0
        assert step.group_cols == ["time_bin"]
        with pytest.raises(ValidationError):
            KernelDensityConfig(bandwidth_m=0.0)


class TestSpaceTimeKernelDensityStep:
    """Tests for SpaceTimeKernelDensityStep."""

    def test_single_cell_matches_brute_force(self) -> None:
        """1 cell x 5 bins: exact exponential decay, hand-computed."""
        counts = [{0: 3.0}, {0: 0.0}, {0: 2.0}, {0: 0.0}, {0: 1.0}]
        frame = make_panel({0: (0.0, 0.0)}, counts)
        result = SpaceTimeKernelDensityStep(
            value_col="count", bandwidth_m=600.0, halflife_days=7.0
        ).run(frame)

        df = result.collect()
        assert "count_stkde" in df.columns
        expected = brute_force_stkde({0: (0.0, 0.0)}, counts, 600.0, 7.0)
        got = df.sort("time_bin")["count_stkde"].to_list()
        assert got == pytest.approx([row[0] for row in expected])
        # Hand spot-checks: bin 0 is the KDE alone; bin 1 is half of it
        # (one half-life is 7 bins, so lag-1 weight is 0.5 ** (1/7)).
        norm = 2.0 * math.pi * 600.0**2
        assert got[0] == pytest.approx(3.0 / norm)
        assert got[1] == pytest.approx(3.0 / norm * 0.5 ** (1.0 / 7.0))

    def test_first_bin_equals_spatial_kde(self) -> None:
        """At the first bin only lag 0 contributes, with weight 1."""
        frame = make_panel(CELLS_3X3, [{i: float(i + 1) for i in range(9)}])
        kde = KernelDensityStep(value_col="count", bandwidth_m=600.0, group_cols=["time_bin"]).run(
            frame
        )
        stkde = SpaceTimeKernelDensityStep(
            value_col="count", bandwidth_m=600.0, halflife_days=7.0
        ).run(frame)

        kde_df = kde.collect().sort("grid_id")
        stkde_df = stkde.collect().sort("grid_id")
        assert stkde_df["count_stkde"].to_list() == pytest.approx(kde_df["count_kde"].to_list())

    def test_multi_cell_matches_brute_force(self) -> None:
        """3x3 grid x 3 bins: space and time kernels jointly, brute force."""
        counts = [
            {i: float(i) for i in range(9)},
            {i: float(8 - i) for i in range(9)},
            dict.fromkeys(range(9), 1.0),
        ]
        frame = make_panel(CELLS_3X3, counts)
        result = SpaceTimeKernelDensityStep(
            value_col="count", bandwidth_m=600.0, halflife_days=7.0
        ).run(frame)

        df = result.collect()
        expected = brute_force_stkde(CELLS_3X3, counts, 600.0, 7.0)
        bins = [BIN0, BIN1, BIN2]
        for t, time_bin in enumerate(bins):
            for grid_id in range(9):
                assert density_for(df, grid_id, "count_stkde", time_bin) == pytest.approx(
                    expected[t][grid_id]
                )

    def test_leakage_later_bins_do_not_change_earlier_bins(self) -> None:
        """Changing bin 3 counts leaves bins 1-2 STKDE bit-identical."""
        counts_a = [{0: 3.0}, {0: 0.0}, {0: 2.0}]
        counts_b = [{0: 3.0}, {0: 0.0}, {0: 99.0}]
        step = SpaceTimeKernelDensityStep(value_col="count", bandwidth_m=600.0, halflife_days=7.0)
        df_a = step.run(make_panel({0: (0.0, 0.0)}, counts_a)).collect()
        df_b = step.run(make_panel({0: (0.0, 0.0)}, counts_b)).collect()

        # Parallel group_by summation order is not deterministic to the
        # last ulp, so compare to 1e-12 relative, not bit-for-bit.
        assert df_a.sort("time_bin")["count_stkde"].to_list()[:2] == pytest.approx(
            df_b.sort("time_bin")["count_stkde"].to_list()[:2], rel=1e-12
        )

    def test_uses_time_bin_metadata(self) -> None:
        """A 6h panel converts a 1d half-life to 4 bins."""
        counts = [{0: 4.0}, {0: 0.0}, {0: 0.0}, {0: 0.0}, {0: 0.0}]
        frame = make_panel({0: (0.0, 0.0)}, counts, time_bin="6h")
        result = SpaceTimeKernelDensityStep(
            value_col="count", bandwidth_m=600.0, halflife_days=1.0
        ).run(frame)

        df = result.collect()
        expected = brute_force_stkde({0: (0.0, 0.0)}, counts, 600.0, 4.0)
        assert df.sort("time_bin")["count_stkde"].to_list() == pytest.approx(
            [row[0] for row in expected]
        )

    def test_requires_time_bin_metadata(self) -> None:
        """Without metadata.time_bin the half-life cannot convert to bins."""
        frame = make_panel({0: (0.0, 0.0)}, [{0: 1.0}]).with_metadata(time_bin=None)
        step = SpaceTimeKernelDensityStep(value_col="count", bandwidth_m=600.0, halflife_days=7.0)
        with pytest.raises(ValueError, match="time_bin"):
            step.run(frame)

    def test_halflife_validation(self) -> None:
        """halflife_days <= 0 or non-finite must fail at construction."""
        for bad_ht in (0.0, -7.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="halflife_days"):
                SpaceTimeKernelDensityStep(halflife_days=bad_ht)

    def test_bandwidth_validation(self) -> None:
        """bandwidth_m <= 0 must fail at construction."""
        with pytest.raises(ValueError, match="bandwidth_m"):
            SpaceTimeKernelDensityStep(bandwidth_m=0.0)

    def test_null_value_raises(self) -> None:
        """A null value is a blocking error."""
        frame = make_panel({0: (0.0, 0.0)}, [{0: 1.0}, {0: None}])
        step = SpaceTimeKernelDensityStep(value_col="count", bandwidth_m=600.0, halflife_days=7.0)
        with pytest.raises(ValueError, match="null"):
            step.run(frame)

    def test_null_time_raises(self) -> None:
        """A null time is a blocking error (Phase 2 Chicago replay).

        On the completed Chicago panel the pre-completion "time_bin"
        column is null on every generated row; grouping/sorting by it
        silently merges all generated rows into one time group and
        corrupts the time smoothing. The step must refuse a null time
        column and point at the completed panel's timestamp column.
        """
        frame = make_panel({0: (0.0, 0.0)}, [{0: 1.0}, {0: 2.0}])
        df = frame.collect().with_columns(
            pl.when(pl.col("time_bin") == BIN0 + timedelta(days=1))
            .then(None)
            .otherwise(pl.col("time_bin"))
            .alias("time_bin")
        )
        bad = EventFrame(df.lazy(), frame.schema, frame.metadata)
        step = SpaceTimeKernelDensityStep(value_col="count", bandwidth_m=600.0, halflife_days=7.0)
        with pytest.raises(ValueError, match="non-null"):
            step.run(bad)

    def test_intermediate_kde_column_is_dropped(self) -> None:
        """Only count_stkde is added; no stray intermediate column remains."""
        frame = make_panel({0: (0.0, 0.0)}, [{0: 1.0}, {0: 2.0}])
        result = SpaceTimeKernelDensityStep(
            value_col="count", bandwidth_m=600.0, halflife_days=7.0
        ).run(frame)

        df = result.collect()
        assert "count_stkde" in df.columns
        assert "count_kde" not in df.columns
        assert "__count_kde_for_stkde" not in df.columns

    def test_registers_provenance(self) -> None:
        """STKDE columns carry provenance with both bandwidths."""
        frame = make_panel({0: (0.0, 0.0)}, [{0: 1.0}])
        result = SpaceTimeKernelDensityStep(
            value_col="count", bandwidth_m=600.0, halflife_days=7.0
        ).run(frame)

        provenance = result.metadata.feature_provenance["count_stkde"]
        assert provenance.produced_by == "SpaceTimeKernelDensityStep"
        assert "spatial" in provenance.tags
        assert "temporal" in provenance.tags
        assert provenance.metadata["bandwidth_m"] == 600.0
        assert provenance.metadata["halflife_days"] == 7.0
        assert provenance.metadata["halflife_bins"] == 7.0
        assert provenance.metadata["causality"] == "backward-looking, includes current bin"

    def test_config_model(self) -> None:
        """The pydantic config round-trips the step parameters."""
        config = SpaceTimeKernelDensityConfig(
            value_col="count", bandwidth_m=600.0, halflife_days=7.0
        )
        step = SpaceTimeKernelDensityStep(
            value_col=config.value_col,
            bandwidth_m=config.bandwidth_m,
            halflife_days=config.halflife_days,
            unit_col=config.unit_col,
            time_col=config.time_col,
        )
        assert step.halflife_days == 7.0
        assert step.time_col == "time_bin"
        with pytest.raises(ValidationError):
            SpaceTimeKernelDensityConfig(halflife_days=0.0)
