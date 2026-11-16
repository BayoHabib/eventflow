"""Synthetic ground-truth recovery tests (Phase 4, criterion 7).

The real library Steps run on panels produced by
``tests.evaluation.synthetic.generate_synthetic_panel`` and must recover
the planted parameters within the tolerances stated below. The
tolerances are part of the deliverable: they were fixed from the design
of the generator (sampling sizes and the known Poisson attenuation of
the latent AR(1) signal) BEFORE the recovery runs, and are justified in
``docs/design-notes/phase4-synthetic-ground-truth.md``. They were not
adjusted to make the tests pass.

Stated tolerances (planted values in parentheses):

- Lag-1: pooled correlation between ``count`` and the
  ``TemporalLagStep`` output ``count_lag_1`` lies in [0.25, 0.75]
  (planted latent AR(1) phi = 0.6). The lower bound excludes 0 by
  >10 pooled standard errors (SE ~ 0.02 over ~2,000 pairs); the upper
  bound allows the attenuation Poisson noise imposes on the latent
  correlation, which can only pull the estimate below phi.
- Seasonality: weekend/weekday mean-count ratio is 1.6 +/- 0.25
  (planted weekend multiplier 1.6). The band is ~5x the ratio's
  sampling error (~0.05).
- Hotspot: the 4 units with the highest mean count are exactly the
  planted hotspot units (3x rate over 56 bins: the rate gap is ~12x
  the standard error of a unit mean).
- Spatial lag: the mean ring-1 ``MultiScaleSpatialLagStep`` lag of
  ``count_lag_1`` for hotspot-adjacent units exceeds that for units
  far from the hotspot by >= 0.5 counts (the planted gap is ~1.0).
- Moving average: correlation between ``count`` and the trailing
  7-day ``MovingAverageStep`` output exceeds 0.25 (the trailing mean
  tracks the persistent latent level).
- Categories: ``CategoryCountsStep`` with top_n=2 keeps exactly
  {THEFT, BATTERY}; the global share of THEFT is 0.40 +/- 0.05 and of
  ``other`` is 0.35 +/- 0.05 (sampling SE ~ 0.007 over ~5,000 events).
- Calendar: ``CalendarEncodingStep`` cyclical day-of-week sin/cos
  match 2*pi*weekday/7 exactly (tolerance 1e-9; this is a deterministic
  formula, not a statistical estimate).
- Panel completion: ``CompletePanelStep`` restores every dropped
  (unit, bin) pair as an explicit zero and leaves observed counts
  byte-identical.

Null control (``signal=False``): with no signal planted, the same
steps must recover ~no signal — |lag-1 correlation| < 0.10,
|weekend ratio - 1| < 0.15, |MA-7 correlation| < 0.10 (each ~4x the
pooled sampling error, so a real signal cannot hide inside the band
and pure noise essentially never escapes it).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from eventflow.core.steps.categorical import CategoryCountsStep
from eventflow.core.steps.panel import CompletePanelStep
from eventflow.core.steps.spatial import MultiScaleSpatialLagStep
from eventflow.core.steps.temporal import (
    CalendarEncodingStep,
    FillNullStep,
    MovingAverageStep,
    TemporalLagStep,
)

from .synthetic import (
    SyntheticData,
    events_event_frame,
    generate_synthetic_panel,
    make_sparse_panel,
    panel_event_frame,
)

# --- Stated tolerances (see module docstring) ---------------------------------

LAG1_CORR_LO = 0.25
LAG1_CORR_HI = 0.75
WEEKEND_RATIO_TOL = 0.25
SPATIAL_LAG_GAP_MIN = 0.5
MA7_CORR_MIN = 0.25
CATEGORY_SHARE_TOL = 0.05
NOISE_LAG1_CORR_MAX = 0.10
NOISE_WEEKEND_RATIO_TOL = 0.15
NOISE_MA7_CORR_MAX = 0.10


# --- Helpers -------------------------------------------------------------------


def _pooled_lag1_corr(df: pl.DataFrame) -> float:
    """Pearson correlation between count and its lag-1 column."""
    sub = df.drop_nulls(["count", "count_lag_1"])
    return float(np.corrcoef(sub["count"].to_numpy(), sub["count_lag_1"].to_numpy())[0, 1])


def _weekend_ratio(df: pl.DataFrame) -> float:
    """Mean weekend count / mean weekday count (Polars weekday: Mon=1..Sun=7)."""
    weekend = df.filter(pl.col("time_bin").dt.weekday() >= 6)["count"].mean()
    weekday = df.filter(pl.col("time_bin").dt.weekday() < 6)["count"].mean()
    assert weekend is not None and weekday is not None
    return float(weekend / weekday)


def _lagged_panel(data: SyntheticData) -> pl.DataFrame:
    """Panel with TemporalLagStep count_lag_1 attached."""
    frame = panel_event_frame(data.panel)
    result = TemporalLagStep(
        value_cols=["count"],
        lag_periods=[1],
        time_unit="1d",
        group_cols=["grid_id"],
        mode="duration",
    ).run(frame)
    return result.collect()


def _hotspot_adjacent_and_far(data: SyntheticData) -> tuple[set[int], set[int]]:
    """Units adjacent (Chebyshev distance 1) to the hotspot, and units far from it."""
    n_cols = data.params.n_grid_cols
    hot_ids = set(data.params.hotspot_units)
    hot_cells = {((u - 1) // n_cols, (u - 1) % n_cols) for u in hot_ids}
    adjacent: set[int] = set()
    for r, c in hot_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < data.params.n_grid_rows and 0 <= cc < data.params.n_grid_cols:
                    unit = rr * n_cols + cc + 1
                    if unit not in hot_ids:
                        adjacent.add(unit)
    far = set(range(1, data.params.n_units + 1)) - hot_ids - adjacent
    return adjacent, far


# --- Generator sanity ----------------------------------------------------------


class TestGenerator:
    """The generator itself: determinism and panel/event consistency."""

    def test_same_seed_reproduces_panel(self) -> None:
        a = generate_synthetic_panel(seed=42)
        b = generate_synthetic_panel(seed=42)
        assert a.panel.equals(b.panel)
        assert a.events.equals(b.events)
        assert a.params == b.params

    def test_different_seed_changes_panel(self) -> None:
        a = generate_synthetic_panel(seed=42)
        b = generate_synthetic_panel(seed=7)
        assert not a.panel.equals(b.panel)

    def test_panel_shape_and_event_consistency(self) -> None:
        data = generate_synthetic_panel(seed=42)
        assert data.panel.height == 36 * 56
        assert data.panel["grid_id"].n_unique() == 36
        assert data.panel["time_bin"].n_unique() == 56
        # Event rows sum back to the panel counts, per (unit, bin).
        event_counts = data.events.group_by(["grid_id", "time_bin"]).len()
        joined = data.panel.join(event_counts, on=["grid_id", "time_bin"], how="left").with_columns(
            pl.col("len").fill_null(0)
        )
        assert (joined["count"] == joined["len"]).all()


# --- Recovery: temporal structure ----------------------------------------------


class TestTemporalRecovery:
    """TemporalLagStep / MovingAverageStep recover the planted AR(1)."""

    @pytest.mark.parametrize("seed", [42, 7])
    def test_lag1_correlation_recovered(self, seed: int) -> None:
        data = generate_synthetic_panel(seed=seed)
        df = _lagged_panel(data)
        corr = _pooled_lag1_corr(df)
        assert LAG1_CORR_LO <= corr <= LAG1_CORR_HI, (
            f"lag-1 corr {corr:.3f} outside [{LAG1_CORR_LO}, {LAG1_CORR_HI}] "
            f"for planted phi={data.params.lag1_phi}"
        )

    @pytest.mark.parametrize("seed", [42, 7])
    def test_weekend_seasonality_recovered(self, seed: int) -> None:
        data = generate_synthetic_panel(seed=seed)
        ratio = _weekend_ratio(data.panel)
        planted = data.params.weekend_multiplier
        assert (
            abs(ratio - planted) <= WEEKEND_RATIO_TOL
        ), f"weekend ratio {ratio:.3f} vs planted {planted}"

    def test_trailing_moving_average_tracks_level(self) -> None:
        data = generate_synthetic_panel(seed=42)
        frame = panel_event_frame(data.panel)
        result = MovingAverageStep(
            value_cols=["count"],
            windows=[7],
            time_unit="1d",
            group_cols=["grid_id"],
            mode="duration",
        ).run(frame)
        df = result.collect()
        sub = df.drop_nulls(["count", "count_ma_7"])
        corr = float(np.corrcoef(sub["count"].to_numpy(), sub["count_ma_7"].to_numpy())[0, 1])
        assert corr > MA7_CORR_MIN, f"MA-7 corr {corr:.3f} <= {MA7_CORR_MIN}"

    def test_calendar_encoding_matches_weekday_formula(self) -> None:
        data = generate_synthetic_panel(seed=42)
        frame = panel_event_frame(data.panel)
        result = CalendarEncodingStep(
            encoding_type="cyclical", components=["hour", "day_of_week"]
        ).run(frame)
        df = result.collect()
        # Polars weekday: Monday=1 .. Sunday=7, matching the step's formula.
        weekday = df["time_bin"].dt.weekday().to_numpy()
        expected_sin = np.sin(2 * math.pi * weekday / 7)
        expected_cos = np.cos(2 * math.pi * weekday / 7)
        assert np.allclose(df["dow_sin"].to_numpy(), expected_sin, atol=1e-9)
        assert np.allclose(df["dow_cos"].to_numpy(), expected_cos, atol=1e-9)
        # Every bin is midnight, so the hour encoding is exactly (sin 0, cos 1).
        assert np.allclose(df["hour_sin"].to_numpy(), 0.0, atol=1e-9)
        assert np.allclose(df["hour_cos"].to_numpy(), 1.0, atol=1e-9)


# --- Recovery: spatial structure -----------------------------------------------


class TestSpatialRecovery:
    """Hotspot location and spatial lag recover the planted hotspot."""

    @pytest.mark.parametrize("seed", [42, 7])
    def test_hotspot_units_ranked_top_by_mean_count(self, seed: int) -> None:
        data = generate_synthetic_panel(seed=seed)
        ranked = (
            data.panel.group_by("grid_id")
            .agg(pl.col("count").mean().alias("mean_count"))
            .sort("mean_count", descending=True)
        )
        top4 = set(ranked.head(4)["grid_id"].to_list())
        assert top4 == set(data.params.hotspot_units), (
            f"top-4 units {sorted(top4)} != planted hotspot " f"{sorted(data.params.hotspot_units)}"
        )

    def test_spatial_lag_elevated_next_to_hotspot(self) -> None:
        data = generate_synthetic_panel(seed=42)
        df = _lagged_panel(data)
        # Lag-1 counts are null before the panel start; the Chicago recipe
        # policy applies here too: a missing pre-window count is 0.
        frame = panel_event_frame(df)
        filled = FillNullStep(value_cols=["count_lag_1"], fill_value=0).run(frame)
        lagged = MultiScaleSpatialLagStep(
            value_cols=["count_lag_1"], rings=(1,), group_cols=["time_bin"]
        ).run(filled)
        out = lagged.collect()
        assert "count_lag_1_spatial_lag_r1" in out.columns
        per_unit = out.group_by("grid_id").agg(
            pl.col("count_lag_1_spatial_lag_r1").mean().alias("mean_spatial_lag")
        )
        adjacent, far = _hotspot_adjacent_and_far(data)
        mean_adj = per_unit.filter(pl.col("grid_id").is_in(sorted(adjacent)))[
            "mean_spatial_lag"
        ].mean()
        mean_far = per_unit.filter(pl.col("grid_id").is_in(sorted(far)))["mean_spatial_lag"].mean()
        assert mean_adj is not None and mean_far is not None
        assert float(mean_adj) - float(mean_far) >= SPATIAL_LAG_GAP_MIN, (
            f"spatial-lag gap {float(mean_adj) - float(mean_far):.3f} " f"< {SPATIAL_LAG_GAP_MIN}"
        )


# --- Recovery: panel completion and categories ---------------------------------


class TestPanelAndCategoryRecovery:
    """CompletePanelStep restores dropped pairs; CategoryCountsStep recovers top-N."""

    def test_complete_panel_restores_dropped_pairs_as_zeros(self) -> None:
        data = generate_synthetic_panel(seed=42)
        sparse = make_sparse_panel(data, drop_fraction=0.10, seed=99)
        assert sparse.panel.height < data.panel.height
        assert len(sparse.dropped_pairs) > 0
        frame = panel_event_frame(sparse.panel)
        result = CompletePanelStep().run(frame).collect()
        assert result.height == data.panel.height
        # Dropped pairs come back as explicit zeros.
        for grid_id, time_bin in sparse.dropped_pairs:
            row = result.filter((pl.col("grid_id") == grid_id) & (pl.col("time_bin") == time_bin))
            assert row.height == 1
            assert row["count"].item() == 0
        # Observed pairs are unchanged (compare against the sparse panel,
        # not the full one: dropped pairs legitimately come back as 0).
        joined = result.join(sparse.panel, on=["grid_id", "time_bin"], suffix="_orig").sort(
            ["grid_id", "time_bin"]
        )
        assert joined.height == sparse.panel.height
        assert (joined["count"] == joined["count_orig"]).all()

    def test_category_counts_recover_planted_top_n(self) -> None:
        data = generate_synthetic_panel(seed=42)
        frame = events_event_frame(data.events)
        result = CategoryCountsStep(category_cols=["primary_type"], top_n=2).run(frame)
        df = result.collect()
        assert "count_primary_type_THEFT" in df.columns
        assert "count_primary_type_BATTERY" in df.columns
        # ASSAULT (0.20) and BURGLARY (0.15) fall into "other", not top-N.
        assert "count_primary_type_ASSAULT" not in df.columns
        assert "count_primary_type_BURGLARY" not in df.columns
        # Global shares from the per-group counts (one row per group).
        groups = df.unique(subset=["grid_id", "time_bin"], keep="first")
        total = data.events.height
        share_theft = groups["count_primary_type_THEFT"].sum() / total
        share_other = groups["count_primary_type_other"].sum() / total
        assert abs(share_theft - 0.40) <= CATEGORY_SHARE_TOL, f"THEFT share {share_theft:.3f}"
        assert abs(share_other - 0.35) <= CATEGORY_SHARE_TOL, f"other share {share_other:.3f}"
        # Shares sum to 1 within every group.
        share_cols = [c for c in df.columns if c.startswith("share_primary_type_")]
        row_sums = df.select(pl.sum_horizontal(share_cols).alias("s"))["s"]
        assert np.allclose(row_sums.to_numpy(), 1.0, atol=1e-9)


# --- Null control: no signal planted -------------------------------------------


class TestNoiseControl:
    """With signal=False the same steps must recover ~no signal."""

    def test_noise_panel_recovers_no_temporal_signal(self) -> None:
        data = generate_synthetic_panel(seed=42, signal=False)
        assert data.params.lag1_phi == 0.0
        assert data.params.hotspot_multiplier == 1.0
        assert data.params.weekend_multiplier == 1.0
        df = _lagged_panel(data)
        assert abs(_pooled_lag1_corr(df)) < NOISE_LAG1_CORR_MAX
        assert abs(_weekend_ratio(data.panel) - 1.0) < NOISE_WEEKEND_RATIO_TOL

    def test_noise_panel_moving_average_is_uncorrelated(self) -> None:
        data = generate_synthetic_panel(seed=42, signal=False)
        frame = panel_event_frame(data.panel)
        result = MovingAverageStep(
            value_cols=["count"],
            windows=[7],
            time_unit="1d",
            group_cols=["grid_id"],
            mode="duration",
        ).run(frame)
        df = result.collect()
        sub = df.drop_nulls(["count", "count_ma_7"])
        corr = float(np.corrcoef(sub["count"].to_numpy(), sub["count_ma_7"].to_numpy())[0, 1])
        assert abs(corr) < NOISE_MA7_CORR_MAX, f"noise MA-7 corr {corr:.3f}"
