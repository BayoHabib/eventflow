"""Tests for dynamics pipeline steps (EWMA, trend, recency, zero features)."""

from __future__ import annotations

import math

import polars as pl
import pytest
from pydantic import ValidationError

from shikong.core.event_frame import EventFrame
from shikong.core.schema import EventMetadata, EventSchema
from shikong.core.steps.dynamics import (
    BurstinessConfig,
    BurstinessStep,
    EWMAMeanConfig,
    EWMAMeanStep,
    TimeSinceLastEventConfig,
    TimeSinceLastEventStep,
    TrendConfig,
    TrendStep,
    ZeroFeaturesConfig,
    ZeroFeaturesStep,
)
from shikong.core.steps.temporal import MovingAverageStep

# Hand-chosen counts including zeros, 2 units x 10 daily bins.
UNIT_A_COUNTS = [0, 2, 0, 4, 1, 0, 0, 3, 0, 2]
UNIT_B_COUNTS = [5, 0, 0, 0, 1, 2, 0, 0, 4, 0]
DAYS = [f"2024-01-{day:02d}T00:00:00" for day in range(1, 11)]


@pytest.fixture
def panel_frame() -> EventFrame:
    """Small synthetic completed panel: 2 units x 10 daily bins, explicit zeros."""
    lf = pl.LazyFrame(
        {
            "timestamp": DAYS * 2,
            "unit": ["A"] * 10 + ["B"] * 10,
            "latitude": [41.8781] * 20,
            "longitude": [-87.6298] * 20,
            "count": UNIT_A_COUNTS + UNIT_B_COUNTS,
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())
    # Reorder so rows are unit-major, time-minor (the order steps sort to).
    lf = lf.sort(["unit", "timestamp"], maintain_order=True)

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["count"],
        categorical_cols=["unit"],
    )
    metadata = EventMetadata(dataset_name="test-dynamics", crs="EPSG:4326", time_bin="1d")
    return EventFrame(lf, schema, metadata)


def ewma_adjust(values: list[float], halflife_bins: float) -> list[float]:
    """Hand-computed EWMA with adjust=True (Polars default).

    alpha = 1 - 0.5 ** (1 / halflife); at index i the value is the
    weighted mean of values[0..i] with weights (1 - alpha) ** (i - j).
    """
    alpha = 1.0 - 0.5 ** (1.0 / halflife_bins)
    out: list[float] = []
    for i, _ in enumerate(values):
        weights = [(1.0 - alpha) ** (i - j) for j in range(i + 1)]
        out.append(sum(w * v for w, v in zip(weights, values[: i + 1], strict=True)) / sum(weights))
    return out


def rows_for(df: pl.DataFrame, unit: str, col: str) -> list[float | None]:
    """Values of *col* for one unit, in timestamp order."""
    return df.filter(pl.col("unit") == unit).sort("timestamp")[col].to_list()


class TestEWMAMeanStep:
    """Tests for EWMAMeanStep."""

    def test_ewma_values_halflife_1d(self, panel_frame: EventFrame) -> None:
        """Hand-computed EWMA (halflife 1 bin, alpha = 0.5) for both units."""
        step = EWMAMeanStep(value_cols=["count"], halflife_days=1.0, group_cols=["unit"])
        result = step.run(panel_frame)

        df = result.collect()
        assert "count_ewma_1d" in df.columns
        expected_a = ewma_adjust([float(v) for v in UNIT_A_COUNTS], 1.0)
        expected_b = ewma_adjust([float(v) for v in UNIT_B_COUNTS], 1.0)
        assert rows_for(df, "A", "count_ewma_1d") == pytest.approx(expected_a)
        assert rows_for(df, "B", "count_ewma_1d") == pytest.approx(expected_b)
        # Spot-check the first values by hand: A = [0, 4/3, 4/7, 2.4].
        assert rows_for(df, "A", "count_ewma_1d")[1] == pytest.approx(4 / 3)
        assert rows_for(df, "A", "count_ewma_1d")[2] == pytest.approx(4 / 7)
        assert rows_for(df, "A", "count_ewma_1d")[3] == pytest.approx(2.4)

    def test_ewma_halflife_7d_column_name_and_values(self, panel_frame: EventFrame) -> None:
        """A 7d half-life on a daily panel is 7 bins; label has no trailing '.0'."""
        step = EWMAMeanStep(value_cols=["count"], halflife_days=7.0, group_cols=["unit"])
        result = step.run(panel_frame)

        df = result.collect()
        assert "count_ewma_7d" in df.columns
        expected_a = ewma_adjust([float(v) for v in UNIT_A_COUNTS], 7.0)
        assert rows_for(df, "A", "count_ewma_7d") == pytest.approx(expected_a)

    def test_ewma_fractional_halflife_label(self, panel_frame: EventFrame) -> None:
        """A 2.5d half-life labels the column 'count_ewma_2.5d'."""
        step = EWMAMeanStep(value_cols=["count"], halflife_days=2.5, group_cols=["unit"])
        result = step.run(panel_frame)

        assert "count_ewma_2.5d" in result.collect().columns

    def test_ewma_min_periods(self, panel_frame: EventFrame) -> None:
        """min_periods=2 leaves the first bin null."""
        step = EWMAMeanStep(
            value_cols=["count"], halflife_days=1.0, group_cols=["unit"], min_periods=2
        )
        result = step.run(panel_frame)

        df = result.collect()
        assert rows_for(df, "A", "count_ewma_1d")[0] is None
        assert rows_for(df, "A", "count_ewma_1d")[1] is not None

    def test_ewma_uses_time_bin_metadata(self, panel_frame: EventFrame) -> None:
        """A 6h panel converts a 1d half-life to 4 bins."""
        frame_6h = panel_frame.with_metadata(time_bin="6h")
        step = EWMAMeanStep(value_cols=["count"], halflife_days=1.0, group_cols=["unit"])
        result = step.run(frame_6h)

        df = result.collect()
        expected_a = ewma_adjust([float(v) for v in UNIT_A_COUNTS], 4.0)
        assert rows_for(df, "A", "count_ewma_1d") == pytest.approx(expected_a)

    def test_ewma_requires_time_bin_metadata(self, panel_frame: EventFrame) -> None:
        """Without metadata.time_bin the halflife cannot be converted to bins."""
        frame = panel_frame.with_metadata(time_bin=None)
        step = EWMAMeanStep(value_cols=["count"], halflife_days=1.0)
        with pytest.raises(ValueError, match="time_bin"):
            step.run(frame)

    def test_ewma_registers_provenance(self, panel_frame: EventFrame) -> None:
        """EWMA columns carry provenance with the halflife used."""
        step = EWMAMeanStep(value_cols=["count"], halflife_days=7.0, group_cols=["unit"])
        result = step.run(panel_frame)

        provenance = result.metadata.feature_provenance["count_ewma_7d"]
        assert provenance.produced_by == "EWMAMeanStep"
        assert "temporal" in provenance.tags
        assert provenance.metadata["halflife_days"] == 7.0

    def test_ewma_rejects_empty_value_cols(self) -> None:
        """An empty value_cols list must fail, not silently no-op."""
        with pytest.raises(ValueError, match="value_cols"):
            EWMAMeanStep(value_cols=[])

    def test_ewma_rejects_unparseable_time_bin(self, panel_frame: EventFrame) -> None:
        """An unparseable metadata.time_bin must fail with a clear error."""
        frame = panel_frame.with_metadata(time_bin="bogus")
        step = EWMAMeanStep(value_cols=["count"], halflife_days=1.0)
        with pytest.raises(ValueError, match="bogus"):
            step.run(frame)

    def test_ewma_rejects_nonpositive_halflife(self) -> None:
        """halflife_days <= 0 must fail at construction."""
        with pytest.raises(ValueError, match="halflife_days"):
            EWMAMeanStep(value_cols=["count"], halflife_days=0)
        with pytest.raises(ValueError, match="halflife_days"):
            EWMAMeanStep(value_cols=["count"], halflife_days=-3.0)

    def test_ewma_rejects_bad_min_periods(self) -> None:
        """min_periods = 0 must fail at construction."""
        with pytest.raises(ValueError, match="min_periods"):
            EWMAMeanStep(value_cols=["count"], min_periods=0)

    def test_ewma_rejects_missing_column(self, panel_frame: EventFrame) -> None:
        """A missing value column must fail, not silently no-op."""
        step = EWMAMeanStep(value_cols=["nope"], halflife_days=1.0)
        with pytest.raises(ValueError, match="nope"):
            step.run(panel_frame)

    def test_config_model_roundtrip(self, panel_frame: EventFrame) -> None:
        """The registry path (config -> model_dump -> step kwargs) works."""
        config = EWMAMeanConfig(value_cols=["count"], halflife_days=7.0, group_cols=["unit"])
        step = EWMAMeanStep(**config.model_dump())
        result = step.run(panel_frame)
        assert "count_ewma_7d" in result.collect().columns

    def test_config_model_rejects_bad_values(self) -> None:
        """Pydantic guards the registry path too."""
        with pytest.raises(ValidationError):
            EWMAMeanConfig(value_cols=["count"], halflife_days=0)
        with pytest.raises(ValidationError):
            EWMAMeanConfig(value_cols=["count"], min_periods=0)


class TestTrendStep:
    """Tests for TrendStep."""

    @pytest.fixture
    def ma_frame(self, panel_frame: EventFrame) -> EventFrame:
        """Panel with closed-left rolling means ma_1 and ma_3."""
        step = MovingAverageStep(
            value_cols=["count"], windows=[1, 3], time_unit="1d", group_cols=["unit"]
        )
        return step.run(panel_frame)

    def test_trend_ratio_values(self, ma_frame: EventFrame) -> None:
        """Ratio recent/long computed by hand from the rolling means.

        Unit A counts [0, 2, 0, 4, 1, ...]: ma_1 is the previous bin's
        count, ma_3 the mean of the previous 3 bins (closed-left).
        """
        step = TrendStep(recent_col="count_ma_1", long_col="count_ma_3")
        result = step.run(ma_frame)

        df = result.collect()
        col = "count_ma_1_vs_count_ma_3"
        assert col in df.columns
        values = rows_for(df, "A", col)
        # t0: both null -> null. t1: ma_1=0, ma_3=0 -> null (zero denominator).
        assert values[0] is None
        assert values[1] is None
        # t2: ma_1=2, ma_3=mean([0, 2])=1 -> 2.0
        assert values[2] == pytest.approx(2.0)
        # t3: ma_1=0, ma_3=mean([0, 2, 0])=2/3 -> 0.0
        assert values[3] == pytest.approx(0.0)
        # t4: ma_1=4, ma_3=mean([2, 0, 4])=2 -> 2.0
        assert values[4] == pytest.approx(2.0)

    def test_trend_zero_denominator_is_null_not_inf(self, ma_frame: EventFrame) -> None:
        """A zero long mean yields null, never inf."""
        step = TrendStep(recent_col="count_ma_1", long_col="count_ma_3")
        result = step.run(ma_frame)

        df = result.collect()
        values = df["count_ma_1_vs_count_ma_3"].to_list()
        assert not any(v == float("inf") or v == float("-inf") for v in values if v is not None)

    def test_trend_custom_output_col(self, ma_frame: EventFrame) -> None:
        """An explicit output_col is honoured."""
        step = TrendStep(recent_col="count_ma_1", long_col="count_ma_3", output_col="trend_1_3")
        result = step.run(ma_frame)
        assert "trend_1_3" in result.collect().columns

    def test_trend_registers_provenance(self, ma_frame: EventFrame) -> None:
        """The trend column carries provenance naming both inputs."""
        step = TrendStep(recent_col="count_ma_1", long_col="count_ma_3")
        result = step.run(ma_frame)

        provenance = result.metadata.feature_provenance["count_ma_1_vs_count_ma_3"]
        assert provenance.produced_by == "TrendStep"
        assert "temporal" in provenance.tags
        assert provenance.inputs == [
            "count_ma_1",
            "count_ma_3",
            "timestamp",
        ]

    def test_trend_rejects_missing_column(self, panel_frame: EventFrame) -> None:
        """A missing input column must fail, not silently no-op."""
        step = TrendStep(recent_col="count_ma_1", long_col="count_ma_99")
        with pytest.raises(ValueError, match="count_ma_99"):
            step.run(panel_frame)

    def test_config_model_roundtrip(self, ma_frame: EventFrame) -> None:
        """The registry path (config -> model_dump -> step kwargs) works."""
        config = TrendConfig(recent_col="count_ma_1", long_col="count_ma_3")
        step = TrendStep(**config.model_dump())
        result = step.run(ma_frame)
        assert "count_ma_1_vs_count_ma_3" in result.collect().columns


class TestTimeSinceLastEventStep:
    """Tests for TimeSinceLastEventStep."""

    def test_time_since_last_values(self, panel_frame: EventFrame) -> None:
        """Days since the last bin with count > 0, computed by hand."""
        step = TimeSinceLastEventStep(group_cols=["unit"])
        result = step.run(panel_frame)

        df = result.collect()
        col = "count_time_since_last_d"
        assert col in df.columns
        # Unit A [0, 2, 0, 4, 1, 0, 0, 3, 0, 2]
        assert rows_for(df, "A", col) == [
            None,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            2.0,
            0.0,
            1.0,
            0.0,
        ]
        # Unit B [5, 0, 0, 0, 1, 2, 0, 0, 4, 0]
        assert rows_for(df, "B", col) == [
            0.0,
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            1.0,
            2.0,
            0.0,
            1.0,
        ]

    def test_time_since_last_current_bin_counts(self, panel_frame: EventFrame) -> None:
        """The current bin counts: 0.0 when the current bin has count > 0."""
        step = TimeSinceLastEventStep(group_cols=["unit"])
        result = step.run(panel_frame)

        df = result.collect()
        # Every bin with count > 0 has time-since-last == 0.0.
        joined = df.filter(pl.col("count") > 0)["count_time_since_last_d"].to_list()
        assert all(v == 0.0 for v in joined)

    def test_time_since_last_no_event_is_null(self) -> None:
        """A unit with no event at all stays null everywhere."""
        lf = pl.LazyFrame(
            {
                "timestamp": DAYS,
                "unit": ["C"] * 10,
                "latitude": [41.8781] * 10,
                "longitude": [-87.6298] * 10,
                "count": [0] * 10,
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())
        schema = EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
            numeric_cols=["count"],
            categorical_cols=["unit"],
        )
        metadata = EventMetadata(dataset_name="test", crs="EPSG:4326", time_bin="1d")
        frame = EventFrame(lf, schema, metadata)

        result = TimeSinceLastEventStep(group_cols=["unit"]).run(frame)
        assert result.collect()["count_time_since_last_d"].to_list() == [None] * 10

    def test_time_since_last_registers_provenance(self, panel_frame: EventFrame) -> None:
        """The output column carries provenance."""
        result = TimeSinceLastEventStep(group_cols=["unit"]).run(panel_frame)
        provenance = result.metadata.feature_provenance["count_time_since_last_d"]
        assert provenance.produced_by == "TimeSinceLastEventStep"
        assert "temporal" in provenance.tags

    def test_rejects_unsupported_output_unit(self) -> None:
        """Only 'd' is supported in v1."""
        with pytest.raises(ValueError, match="output_unit"):
            TimeSinceLastEventStep(output_unit="h")

    def test_rejects_missing_column(self, panel_frame: EventFrame) -> None:
        """A missing value column must fail, not silently no-op."""
        step = TimeSinceLastEventStep(value_col="nope")
        with pytest.raises(ValueError, match="nope"):
            step.run(panel_frame)

    def test_config_model_roundtrip(self, panel_frame: EventFrame) -> None:
        """The registry path (config -> model_dump -> step kwargs) works."""
        config = TimeSinceLastEventConfig(group_cols=["unit"])
        step = TimeSinceLastEventStep(**config.model_dump())
        result = step.run(panel_frame)
        assert "count_time_since_last_d" in result.collect().columns


class TestBurstinessStep:
    """Tests for BurstinessStep."""

    def test_burstiness_values(self, panel_frame: EventFrame) -> None:
        """CV over the trailing 3d window (closed-left), computed by hand.

        Unit A [0, 2, 0, 4, 1, ...]: at t2 the window is [0, 2] (mean 1,
        sample std sqrt(2), CV sqrt(2)); at t3 the window is [0, 2, 0]
        (mean 2/3, std 2/sqrt(3), CV sqrt(3)); at t4 the window is
        [2, 0, 4] (mean 2, std 2, CV 1).
        """
        step = BurstinessStep(window="3d", group_cols=["unit"])
        result = step.run(panel_frame)

        df = result.collect()
        col = "count_burstiness_3d"
        assert col in df.columns
        values = rows_for(df, "A", col)
        assert values[0] is None  # no prior bin
        assert values[1] is None  # window [0]: mean 0 -> null
        assert values[2] == pytest.approx(math.sqrt(2))
        assert values[3] == pytest.approx(math.sqrt(3))
        assert values[4] == pytest.approx(1.0)

    def test_burstiness_zero_mean_is_null_not_inf(self, panel_frame: EventFrame) -> None:
        """A zero trailing mean yields null, never inf."""
        step = BurstinessStep(window="3d", group_cols=["unit"])
        result = step.run(panel_frame)

        df = result.collect()
        values = df["count_burstiness_3d"].to_list()
        assert not any(v == float("inf") or v == float("-inf") for v in values if v is not None)
        # Unit B starts with a 5 then zeros: its window mean hits 0 at t4
        # (window [0, 0, 0]).
        assert rows_for(df, "B", "count_burstiness_3d")[4] is None

    def test_burstiness_registers_provenance(self, panel_frame: EventFrame) -> None:
        """The burstiness column carries provenance with the window used."""
        result = BurstinessStep(window="28d", group_cols=["unit"]).run(panel_frame)
        provenance = result.metadata.feature_provenance["count_burstiness_28d"]
        assert provenance.produced_by == "BurstinessStep"
        assert "temporal" in provenance.tags
        assert provenance.metadata["window"] == "28d"
        assert provenance.metadata["closed"] == "left"

    def test_burstiness_rejects_bad_window(self) -> None:
        """An unparseable window must fail at construction."""
        with pytest.raises(ValueError, match="3x"):
            BurstinessStep(window="3x")

    def test_burstiness_rejects_bad_min_periods(self) -> None:
        """min_periods = 0 must fail at construction."""
        with pytest.raises(ValueError, match="min_periods"):
            BurstinessStep(min_periods=0)

    def test_burstiness_rejects_missing_column(self, panel_frame: EventFrame) -> None:
        """A missing value column must fail, not silently no-op."""
        step = BurstinessStep(value_col="nope")
        with pytest.raises(ValueError, match="nope"):
            step.run(panel_frame)

    def test_config_model_roundtrip(self, panel_frame: EventFrame) -> None:
        """The registry path (config -> model_dump -> step kwargs) works."""
        config = BurstinessConfig(window="7d", group_cols=["unit"])
        step = BurstinessStep(**config.model_dump())
        result = step.run(panel_frame)
        assert "count_burstiness_7d" in result.collect().columns


class TestZeroFeaturesStep:
    """Tests for ZeroFeaturesStep."""

    def test_active_and_share_values(self, panel_frame: EventFrame) -> None:
        """Active flag and share of nonzero bins over a 3d window (closed-right).

        Unit A [0, 2, 0, 4, 1, 0, 0, 3, 0, 2]: the window at bin t is
        bins t-2..t. Active is True unless all three are zero; share is
        the fraction of the three that are nonzero.
        """
        step = ZeroFeaturesStep(active_window="3d", share_window="3d", group_cols=["unit"])
        result = step.run(panel_frame)

        df = result.collect()
        assert "count_active_3d" in df.columns
        assert "count_share_nonzero_3d" in df.columns
        assert rows_for(df, "A", "count_active_3d") == [
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        assert rows_for(df, "A", "count_share_nonzero_3d") == pytest.approx(
            [0.0, 0.5, 1 / 3, 2 / 3, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3, 2 / 3]
        )
        # Unit B [5, 0, 0, 0, 1, 2, 0, 0, 4, 0]: only t3 has three zeros.
        assert rows_for(df, "B", "count_active_3d") == [
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        assert rows_for(df, "B", "count_share_nonzero_3d") == pytest.approx(
            [1.0, 0.5, 1 / 3, 0.0, 1 / 3, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3]
        )

    def test_zero_features_current_bin_included(self, panel_frame: EventFrame) -> None:
        """The current bin is included: a lone event makes the bin active."""
        step = ZeroFeaturesStep(active_window="3d", share_window="3d", group_cols=["unit"])
        result = step.run(panel_frame)

        df = result.collect()
        # Unit B t4 has count 1 after three zeros: active True, share 1/3.
        assert rows_for(df, "B", "count_active_3d")[4] is True
        assert rows_for(df, "B", "count_share_nonzero_3d")[4] == pytest.approx(1 / 3)

    def test_zero_features_min_periods(self, panel_frame: EventFrame) -> None:
        """min_periods=2 leaves the first bin null for both outputs."""
        step = ZeroFeaturesStep(
            active_window="3d", share_window="3d", group_cols=["unit"], min_periods=2
        )
        result = step.run(panel_frame)

        df = result.collect()
        assert rows_for(df, "A", "count_active_3d")[0] is None
        assert rows_for(df, "A", "count_share_nonzero_3d")[0] is None
        assert rows_for(df, "A", "count_active_3d")[1] is not None

    def test_zero_features_registers_provenance(self, panel_frame: EventFrame) -> None:
        """Both output columns carry provenance with the window used."""
        result = ZeroFeaturesStep(group_cols=["unit"]).run(panel_frame)
        for col in ("count_active_7d", "count_share_nonzero_28d"):
            provenance = result.metadata.feature_provenance[col]
            assert provenance.produced_by == "ZeroFeaturesStep"
            assert "temporal" in provenance.tags
            assert provenance.metadata["closed"] == "right"

    def test_zero_features_rejects_bad_window(self) -> None:
        """An unparseable window must fail at construction."""
        with pytest.raises(ValueError, match="bogus"):
            ZeroFeaturesStep(active_window="bogus")
        with pytest.raises(ValueError, match="bogus"):
            ZeroFeaturesStep(share_window="bogus")

    def test_zero_features_rejects_bad_min_periods(self) -> None:
        """min_periods = 0 must fail at construction."""
        with pytest.raises(ValueError, match="min_periods"):
            ZeroFeaturesStep(min_periods=0)

    def test_zero_features_rejects_missing_column(self, panel_frame: EventFrame) -> None:
        """A missing value column must fail, not silently no-op."""
        step = ZeroFeaturesStep(value_col="nope")
        with pytest.raises(ValueError, match="nope"):
            step.run(panel_frame)

    def test_config_model_roundtrip(self, panel_frame: EventFrame) -> None:
        """The registry path (config -> model_dump -> step kwargs) works."""
        config = ZeroFeaturesConfig(group_cols=["unit"])
        step = ZeroFeaturesStep(**config.model_dump())
        result = step.run(panel_frame)
        df = result.collect()
        assert "count_active_7d" in df.columns
        assert "count_share_nonzero_28d" in df.columns


class TestDynamicsLeakage:
    """No feature at bin t may depend on bins after t."""

    def _modified_frame(
        self, panel_frame: EventFrame, unit: str, day: int, count: int
    ) -> EventFrame:
        """Copy of the panel with one bin's count changed."""
        df = panel_frame.collect()
        target = f"2024-01-{day:02d}T00:00:00"
        df = df.with_columns(
            pl.when(
                (pl.col("unit") == unit) & (pl.col("timestamp") == pl.lit(target).str.to_datetime())
            )
            .then(count)
            .otherwise(pl.col("count"))
            .alias("count")
        )
        return EventFrame(df.lazy(), panel_frame.schema, panel_frame.metadata)

    def test_ewma_ignores_future_bins(self, panel_frame: EventFrame) -> None:
        """Changing counts at bins > t leaves the EWMA at bins <= t unchanged."""
        step = EWMAMeanStep(value_cols=["count"], halflife_days=1.0, group_cols=["unit"])
        before = rows_for(step.run(panel_frame).collect(), "A", "count_ewma_1d")
        # Change unit A counts on days 8, 9, 10 (bins after t = day 5).
        changed = panel_frame
        for day, count in [(8, 99), (9, 99), (10, 99)]:
            changed = self._modified_frame(changed, "A", day, count)
        after = rows_for(step.run(changed).collect(), "A", "count_ewma_1d")
        assert before[:5] == pytest.approx(after[:5])
        assert before[7:] != after[7:]  # sanity: later bins really did change

    def test_burstiness_ignores_future_bins(self, panel_frame: EventFrame) -> None:
        """Changing counts at bins > t leaves burstiness at bins <= t unchanged."""
        step = BurstinessStep(window="3d", group_cols=["unit"])
        before = rows_for(step.run(panel_frame).collect(), "A", "count_burstiness_3d")
        changed = panel_frame
        for day, count in [(8, 99), (9, 99), (10, 99)]:
            changed = self._modified_frame(changed, "A", day, count)
        after = rows_for(step.run(changed).collect(), "A", "count_burstiness_3d")
        assert before[:5] == pytest.approx(after[:5])

    def test_burstiness_closed_left_ignores_current_bin(self, panel_frame: EventFrame) -> None:
        """Changing the count AT bin t leaves burstiness at bin t unchanged."""
        step = BurstinessStep(window="3d", group_cols=["unit"])
        before = rows_for(step.run(panel_frame).collect(), "A", "count_burstiness_3d")
        # Change unit A day 5 (index 4) from 1 to 99.
        changed = self._modified_frame(panel_frame, "A", 5, 99)
        after = rows_for(step.run(changed).collect(), "A", "count_burstiness_3d")
        assert before[4] == pytest.approx(after[4])
        # But the NEXT bin's window now includes the changed count.
        assert before[5] != after[5]

    def test_zero_features_ignore_future_bins(self, panel_frame: EventFrame) -> None:
        """Changing counts after t leaves active/share at bins <= t unchanged.

        (ZeroFeaturesStep windows are closed-right, i.e. they include the
        current bin by design, so this test only asserts no future leakage.)
        """
        step = ZeroFeaturesStep(active_window="3d", share_window="3d", group_cols=["unit"])
        before_df = step.run(panel_frame).collect()
        changed = panel_frame
        for day, count in [(8, 99), (9, 99), (10, 99)]:
            changed = self._modified_frame(changed, "A", day, count)
        after_df = step.run(changed).collect()
        for col in ("count_active_3d", "count_share_nonzero_3d"):
            assert rows_for(before_df, "A", col)[:5] == pytest.approx(
                rows_for(after_df, "A", col)[:5]
            )

    def test_time_since_last_ignores_future_bins(self, panel_frame: EventFrame) -> None:
        """Changing counts at bins > t leaves time-since-last at bins <= t unchanged."""
        step = TimeSinceLastEventStep(group_cols=["unit"])
        before = rows_for(step.run(panel_frame).collect(), "A", "count_time_since_last_d")
        changed = panel_frame
        for day, count in [(8, 99), (9, 99), (10, 99)]:
            changed = self._modified_frame(changed, "A", day, count)
        after = rows_for(step.run(changed).collect(), "A", "count_time_since_last_d")
        assert before[:5] == after[:5]
