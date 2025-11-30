"""Tests for temporal pipeline steps."""

from __future__ import annotations

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.temporal import (
    CalendarEncodingStep,
    ExtractTemporalComponentsStep,
    InterArrivalStep,
    MovingAverageStep,
    RecencyWeightStep,
    TemporalLagStep,
    TimeBinsStep,
)


@pytest.fixture
def sample_event_frame() -> EventFrame:
    """Create a sample EventFrame for testing temporal steps."""
    lf = pl.LazyFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00",
                "2024-01-01T01:00:00",
                "2024-01-02T00:00:00",
                "2024-01-03T00:00:00",
                "2024-01-04T00:00:00",
                "2024-01-05T00:00:00",
                "2024-01-06T00:00:00",  # Saturday
                "2024-01-07T00:00:00",  # Sunday
            ],
            "latitude": [41.8781] * 8,
            "longitude": [-87.6298] * 8,
            "value": [10.0, 20.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["value"],
        categorical_cols=["category"],
    )
    metadata = EventMetadata(dataset_name="test-temporal", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


class TestExtractTemporalComponentsStep:
    """Tests for ExtractTemporalComponentsStep."""

    def test_extracts_hour_of_day(self, sample_event_frame: EventFrame) -> None:
        """Test hour extraction."""
        step = ExtractTemporalComponentsStep(components=["hour_of_day"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "hour_of_day" in df.columns
        assert df["hour_of_day"][0] == 0
        assert df["hour_of_day"][1] == 1

    def test_extracts_day_of_week(self, sample_event_frame: EventFrame) -> None:
        """Test day of week extraction."""
        step = ExtractTemporalComponentsStep(components=["day_of_week"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "day_of_week" in df.columns
        # 2024-01-01 is Monday (Polars uses 1=Monday, 7=Sunday)
        assert df["day_of_week"][0] == 1

    def test_extracts_is_weekend(self, sample_event_frame: EventFrame) -> None:
        """Test weekend detection."""
        step = ExtractTemporalComponentsStep(components=["is_weekend"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "is_weekend" in df.columns
        # Saturday and Sunday should be True
        assert df["is_weekend"][6] is True  # Saturday
        assert df["is_weekend"][7] is True  # Sunday
        assert df["is_weekend"][0] is False  # Monday

    def test_extracts_multiple_components(self, sample_event_frame: EventFrame) -> None:
        """Test extracting multiple components at once."""
        step = ExtractTemporalComponentsStep(
            components=["hour_of_day", "day_of_week", "month", "year"]
        )
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "hour_of_day" in df.columns
        assert "day_of_week" in df.columns
        assert "month" in df.columns
        assert "year" in df.columns

    def test_registers_provenance(self, sample_event_frame: EventFrame) -> None:
        """Test that extracted components have provenance."""
        step = ExtractTemporalComponentsStep(components=["hour_of_day"])
        result = step.run(sample_event_frame)

        assert "hour_of_day" in result.metadata.feature_provenance
        provenance = result.metadata.feature_provenance["hour_of_day"]
        assert provenance.produced_by == "ExtractTemporalComponentsStep"
        assert "temporal" in provenance.tags

    def test_builder_api(self, sample_event_frame: EventFrame) -> None:
        """Test builder API for component configuration."""
        ConfiguredStep = ExtractTemporalComponentsStep.with_components(
            ["hour_of_day", "day_of_week"]
        )
        step = ConfiguredStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "hour_of_day" in df.columns
        assert "day_of_week" in df.columns


class TestTimeBinsStep:
    """Tests for TimeBinsStep."""

    def test_creates_hourly_bins(self, sample_event_frame: EventFrame) -> None:
        """Test hourly time binning."""
        step = TimeBinsStep(bin_size="1h", bin_col="time_bin")
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "time_bin" in df.columns

    def test_creates_daily_bins(self, sample_event_frame: EventFrame) -> None:
        """Test daily time binning."""
        step = TimeBinsStep(bin_size="1d", bin_col="day_bin")
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "day_bin" in df.columns

    def test_updates_metadata(self, sample_event_frame: EventFrame) -> None:
        """Test that time_bin is stored in metadata."""
        step = TimeBinsStep(bin_size="6h")
        result = step.run(sample_event_frame)

        assert result.metadata.time_bin == "6h"

    def test_updates_context_requirements(self, sample_event_frame: EventFrame) -> None:
        """Test temporal resolution in context requirements."""
        step = TimeBinsStep(bin_size="1h")
        result = step.run(sample_event_frame)

        assert result.schema.context_requirements.temporal_resolution == "1h"


class TestTemporalLagStep:
    """Tests for TemporalLagStep."""

    def test_computes_single_lag(self, sample_event_frame: EventFrame) -> None:
        """Test single lag computation."""
        step = TemporalLagStep(value_cols=["value"], lag_periods=[1])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "value_lag_1" in df.columns
        # First value should be null
        assert df["value_lag_1"][0] is None
        # Second should be first value
        assert df["value_lag_1"][1] == 10.0

    def test_computes_multiple_lags(self, sample_event_frame: EventFrame) -> None:
        """Test multiple lag periods."""
        step = TemporalLagStep(value_cols=["value"], lag_periods=[1, 2, 3])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "value_lag_1" in df.columns
        assert "value_lag_2" in df.columns
        assert "value_lag_3" in df.columns

    def test_builder_api(self, sample_event_frame: EventFrame) -> None:
        """Test builder API for lag configuration."""
        ConfiguredStep = TemporalLagStep.with_lags([1, 7, 30])
        step = ConfiguredStep(value_cols=["value"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "value_lag_1" in df.columns
        assert "value_lag_7" in df.columns
        assert "value_lag_30" in df.columns

    def test_registers_provenance(self, sample_event_frame: EventFrame) -> None:
        """Test that lag features have provenance."""
        step = TemporalLagStep(value_cols=["value"], lag_periods=[1])
        result = step.run(sample_event_frame)

        assert "value_lag_1" in result.metadata.feature_provenance


class TestMovingAverageStep:
    """Tests for MovingAverageStep."""

    def test_computes_moving_average(self, sample_event_frame: EventFrame) -> None:
        """Test moving average computation."""
        step = MovingAverageStep(value_cols=["value"], windows=[3])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "value_ma_3" in df.columns

    def test_computes_multiple_windows(self, sample_event_frame: EventFrame) -> None:
        """Test multiple window sizes."""
        step = MovingAverageStep(value_cols=["value"], windows=[3, 5, 7])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "value_ma_3" in df.columns
        assert "value_ma_5" in df.columns
        assert "value_ma_7" in df.columns

    def test_builder_api(self, sample_event_frame: EventFrame) -> None:
        """Test builder API for window configuration."""
        ConfiguredStep = MovingAverageStep.with_windows([7, 14, 30])
        step = ConfiguredStep(value_cols=["value"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "value_ma_7" in df.columns
        assert "value_ma_14" in df.columns
        assert "value_ma_30" in df.columns

    def test_moving_average_values(self, sample_event_frame: EventFrame) -> None:
        """Test that moving average values are correct."""
        step = MovingAverageStep(value_cols=["value"], windows=[2])
        result = step.run(sample_event_frame)

        df = result.collect()
        # First value is just itself (min_periods=1)
        assert df["value_ma_2"][0] == 10.0
        # Second value is average of first two
        assert df["value_ma_2"][1] == 15.0  # (10 + 20) / 2


class TestRecencyWeightStep:
    """Tests for RecencyWeightStep."""

    def test_computes_recency_weights(self, sample_event_frame: EventFrame) -> None:
        """Test recency weight computation."""
        step = RecencyWeightStep(half_life=2.0, time_unit="1d")
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "recency_weight" in df.columns

    def test_most_recent_has_highest_weight(self, sample_event_frame: EventFrame) -> None:
        """Test that most recent event has weight of 1.0."""
        step = RecencyWeightStep(half_life=7.0, time_unit="1d")
        result = step.run(sample_event_frame)

        df = result.collect()
        # Last event should have weight of 1.0 (reference point)
        assert df["recency_weight"][-1] == pytest.approx(1.0)

    def test_older_events_lower_weight(self, sample_event_frame: EventFrame) -> None:
        """Test that older events have lower weights."""
        step = RecencyWeightStep(half_life=1.0, time_unit="1d")
        result = step.run(sample_event_frame)

        df = result.collect()
        weights = df["recency_weight"].to_list()
        # Weights should be monotonically increasing
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]

    def test_registers_provenance(self, sample_event_frame: EventFrame) -> None:
        """Test that recency weight has provenance."""
        step = RecencyWeightStep(half_life=7.0)
        result = step.run(sample_event_frame)

        provenance = result.metadata.feature_provenance["recency_weight"]
        assert provenance.metadata.get("half_life") == 7.0


class TestCalendarEncodingStep:
    """Tests for CalendarEncodingStep."""

    def test_cyclical_hour_encoding(self, sample_event_frame: EventFrame) -> None:
        """Test cyclical encoding of hour."""
        step = CalendarEncodingStep(encoding_type="cyclical", components=["hour"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns

    def test_cyclical_day_of_week_encoding(self, sample_event_frame: EventFrame) -> None:
        """Test cyclical encoding of day of week."""
        step = CalendarEncodingStep(encoding_type="cyclical", components=["day_of_week"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "dow_sin" in df.columns
        assert "dow_cos" in df.columns

    def test_cyclical_month_encoding(self, sample_event_frame: EventFrame) -> None:
        """Test cyclical encoding of month."""
        step = CalendarEncodingStep(encoding_type="cyclical", components=["month"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "month_sin" in df.columns
        assert "month_cos" in df.columns

    def test_ordinal_encoding(self, sample_event_frame: EventFrame) -> None:
        """Test ordinal encoding."""
        step = CalendarEncodingStep(encoding_type="ordinal", components=["hour", "day_of_week"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "hour_ordinal" in df.columns
        assert "dow_ordinal" in df.columns

    def test_sin_cos_range(self, sample_event_frame: EventFrame) -> None:
        """Test that sin/cos values are in [-1, 1] range."""
        step = CalendarEncodingStep(encoding_type="cyclical", components=["hour"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["hour_sin"].min() >= -1.0
        assert df["hour_sin"].max() <= 1.0
        assert df["hour_cos"].min() >= -1.0
        assert df["hour_cos"].max() <= 1.0


class TestInterArrivalStep:
    """Tests for InterArrivalStep."""

    def test_computes_inter_arrival_time(self, sample_event_frame: EventFrame) -> None:
        """Test inter-arrival time computation."""
        step = InterArrivalStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "inter_arrival_seconds" in df.columns

    def test_first_event_null(self, sample_event_frame: EventFrame) -> None:
        """Test that first event has null inter-arrival time."""
        step = InterArrivalStep(include_stats=False)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["inter_arrival_seconds"][0] is None

    def test_correct_inter_arrival(self, sample_event_frame: EventFrame) -> None:
        """Test inter-arrival time values are correct."""
        step = InterArrivalStep(include_stats=False)
        result = step.run(sample_event_frame)

        df = result.collect()
        # Second event is 1 hour after first
        assert df["inter_arrival_seconds"][1] == 3600.0  # 1 hour in seconds
        # Third event is 23 hours after second
        assert df["inter_arrival_seconds"][2] == 23 * 3600.0

    def test_includes_rolling_stats(self, sample_event_frame: EventFrame) -> None:
        """Test rolling statistics of inter-arrival times."""
        step = InterArrivalStep(include_stats=True, stat_windows=[3, 5])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "inter_arrival_seconds_ma_3" in df.columns
        assert "inter_arrival_seconds_std_3" in df.columns
        assert "inter_arrival_seconds_ma_5" in df.columns
        assert "inter_arrival_seconds_std_5" in df.columns

    def test_grouped_inter_arrival(self, sample_event_frame: EventFrame) -> None:
        """Test inter-arrival time with grouping."""
        step = InterArrivalStep(group_cols=["category"], include_stats=False)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "inter_arrival_seconds" in df.columns
        # First event of each category should have null
        # (this tests the grouping behavior)
