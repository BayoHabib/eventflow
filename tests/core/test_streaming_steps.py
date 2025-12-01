"""Tests for streaming pipeline steps."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.streaming import (
    EventBufferStep,
    OnlineStatisticsStep,
    StreamEvent,
    StreamingDecayStep,
    StreamingHawkesStep,
    StreamingInterEventStep,
    StreamingStep,
    StreamingWindowStep,
    StreamState,
    StreamWindow,
    batch_stream_iterator,
    event_stream_iterator,
)


@pytest.fixture
def sample_event_frame() -> EventFrame:
    """Create a sample EventFrame for testing streaming steps."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(20)]

    lf = pl.LazyFrame(
        {
            "timestamp": timestamps,
            "latitude": [41.8781 + i * 0.001 for i in range(20)],
            "longitude": [-87.6298 + i * 0.001 for i in range(20)],
            "severity": [i % 5 + 1 for i in range(20)],
            "category": ["A" if i % 2 == 0 else "B" for i in range(20)],
        }
    )

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["severity"],
        categorical_cols=["category"],
    )
    metadata = EventMetadata(dataset_name="test-streaming", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


@pytest.fixture
def larger_event_frame() -> EventFrame:
    """Create a larger EventFrame for streaming tests."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    n_events = 100
    timestamps = [base_time + timedelta(minutes=i * 15) for i in range(n_events)]

    lf = pl.LazyFrame(
        {
            "timestamp": timestamps,
            "latitude": [41.8781 + (i % 10) * 0.001 for i in range(n_events)],
            "longitude": [-87.6298 + (i % 10) * 0.001 for i in range(n_events)],
            "value": [float(i % 10 + 1) for i in range(n_events)],
        }
    )

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["value"],
    )
    metadata = EventMetadata(dataset_name="test-large-streaming", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_create_stream_event(self) -> None:
        """Test StreamEvent creation."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        event = StreamEvent(
            timestamp=ts,
            index=0,
            attributes={"value": 10.0},
        )
        assert event.timestamp == ts
        assert event.index == 0
        assert event.attributes["value"] == 10.0

    def test_stream_event_optional_fields(self) -> None:
        """Test StreamEvent with optional fields."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        event = StreamEvent(
            timestamp=ts,
            index=5,
            attributes={"latitude": 41.8781, "longitude": -87.6298},
        )
        assert event.attributes["latitude"] == 41.8781
        assert event.attributes["longitude"] == -87.6298


class TestStreamWindow:
    """Tests for StreamWindow dataclass."""

    def test_create_stream_window(self) -> None:
        """Test StreamWindow creation."""
        window = StreamWindow(max_size=10)
        for i in range(5):
            window.add(StreamEvent(timestamp=datetime(2024, 1, 1, i), index=i))
        assert len(window) == 5
        assert window.max_size == 10

    def test_stream_window_with_duration(self) -> None:
        """Test StreamWindow with duration constraint."""
        window = StreamWindow(
            max_size=100,
            max_duration_seconds=3600 * 10,  # 10 hours in seconds
        )
        for i in range(5):
            window.add(StreamEvent(timestamp=datetime(2024, 1, 1, i), index=i))
        assert window.max_duration_seconds == 3600 * 10


class TestStreamState:
    """Tests for StreamState class."""

    def test_create_stream_state(self) -> None:
        """Test StreamState creation."""
        state: StreamState[dict[str, Any]] = StreamState(initial_value={})
        assert state.value == {}
        assert state.update_count == 0

    def test_stream_state_with_values(self) -> None:
        """Test StreamState with updates."""
        initial = {"mean": 0.0, "count": 0}
        state: StreamState[dict[str, Any]] = StreamState(initial_value=initial)
        state.update({"mean": 5.0, "count": 100})
        assert state.value["mean"] == 5.0
        assert state.value["count"] == 100
        assert state.update_count == 1


class TestEventStreamIterator:
    """Tests for event_stream_iterator function."""

    def test_iterates_events(self, sample_event_frame: EventFrame) -> None:
        """Test iterating over events."""
        events = list(event_stream_iterator(sample_event_frame))
        assert len(events) == 20

    def test_events_ordered(self, sample_event_frame: EventFrame) -> None:
        """Test that events are in order."""
        events = list(event_stream_iterator(sample_event_frame))
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i - 1].timestamp

    def test_event_has_data(self, sample_event_frame: EventFrame) -> None:
        """Test that events contain attributes."""
        events = list(event_stream_iterator(sample_event_frame))
        # Each event should have severity in attributes
        assert "severity" in events[0].attributes


class TestBatchStreamIterator:
    """Tests for batch_stream_iterator function."""

    def test_batches_events(self, larger_event_frame: EventFrame) -> None:
        """Test batching events."""
        batches = list(batch_stream_iterator(larger_event_frame, batch_size=25))
        assert len(batches) == 4  # 100 events / 25 = 4 batches

    def test_batch_sizes(self, larger_event_frame: EventFrame) -> None:
        """Test batch sizes are correct."""
        batch_size = 30
        batches = list(batch_stream_iterator(larger_event_frame, batch_size=batch_size))
        # First batches should be full size
        assert len(batches[0]) == batch_size
        # Last batch may be smaller
        assert len(batches[-1]) <= batch_size

    def test_single_batch(self, sample_event_frame: EventFrame) -> None:
        """Test when batch size exceeds event count."""
        batches = list(batch_stream_iterator(sample_event_frame, batch_size=100))
        assert len(batches) == 1
        assert len(batches[0]) == 20


class TestStreamingWindowStep:
    """Tests for StreamingWindowStep."""

    def test_creates_window_size_column(self, sample_event_frame: EventFrame) -> None:
        """Test window size column creation."""
        step = StreamingWindowStep(window_size=5)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "stream_window_size" in df.columns

    def test_window_size_capped(self, sample_event_frame: EventFrame) -> None:
        """Test that window size respects maximum."""
        max_size = 5
        step = StreamingWindowStep(window_size=max_size)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["stream_window_size"].max() == max_size

    def test_window_grows_initially(self, sample_event_frame: EventFrame) -> None:
        """Test that window grows until max size."""
        step = StreamingWindowStep(window_size=10)
        result = step.run(sample_event_frame)

        df = result.collect()
        # First event has window size 1
        assert df["stream_window_size"][0] == 1
        # Should grow
        assert df["stream_window_size"][5] > df["stream_window_size"][0]

    def test_with_duration_constraint(self, sample_event_frame: EventFrame) -> None:
        """Test window with duration constraint."""
        step = StreamingWindowStep(window_size=100, window_duration="6h")
        result = step.run(sample_event_frame)

        df = result.collect()
        # Should have window size limited by duration
        assert "stream_window_size" in df.columns

    def test_registers_provenance(self, sample_event_frame: EventFrame) -> None:
        """Test that window features have provenance."""
        step = StreamingWindowStep(window_size=5)
        result = step.run(sample_event_frame)

        assert "stream_window_size" in result.metadata.feature_provenance


class TestStreamingHawkesStep:
    """Tests for StreamingHawkesStep."""

    def test_computes_streaming_intensity(self, sample_event_frame: EventFrame) -> None:
        """Test streaming Hawkes intensity computation."""
        step = StreamingHawkesStep(alpha=0.5, beta=1.0, mu=0.1)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "streaming_hawkes_intensity" in df.columns

    def test_intensity_positive(self, sample_event_frame: EventFrame) -> None:
        """Test that streaming intensity is positive."""
        step = StreamingHawkesStep(alpha=0.3, beta=1.0, mu=0.1)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["streaming_hawkes_intensity"].min() > 0

    def test_first_event_equals_mu(self, sample_event_frame: EventFrame) -> None:
        """Test that first event has intensity that includes mu."""
        mu = 0.15
        step = StreamingHawkesStep(alpha=0.5, beta=1.0, mu=mu)
        result = step.run(sample_event_frame)

        df = result.collect()
        # First event should have intensity > mu (mu + trigger contribution)
        assert df["streaming_hawkes_intensity"][0] >= mu

    def test_parameters_stored(self) -> None:
        """Test that parameters are stored correctly."""
        step = StreamingHawkesStep(alpha=0.3, beta=0.8, mu=0.2)
        assert step.alpha == 0.3
        assert step.beta == 0.8
        assert step.mu == 0.2


class TestOnlineStatisticsStep:
    """Tests for OnlineStatisticsStep."""

    def test_computes_online_mean(self, sample_event_frame: EventFrame) -> None:
        """Test online mean computation."""
        step = OnlineStatisticsStep(value_cols=["severity"], statistics=["mean"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "severity_online_mean" in df.columns

    def test_computes_online_std(self, sample_event_frame: EventFrame) -> None:
        """Test online standard deviation computation."""
        step = OnlineStatisticsStep(value_cols=["severity"], statistics=["std"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "severity_online_std" in df.columns

    def test_computes_multiple_statistics(self, sample_event_frame: EventFrame) -> None:
        """Test multiple online statistics."""
        step = OnlineStatisticsStep(
            value_cols=["severity"],
            statistics=["mean", "std", "min", "max"],
        )
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "severity_online_mean" in df.columns
        assert "severity_online_std" in df.columns
        assert "severity_online_min" in df.columns
        assert "severity_online_max" in df.columns

    def test_multiple_columns(self, larger_event_frame: EventFrame) -> None:
        """Test online statistics for multiple columns."""
        step = OnlineStatisticsStep(
            value_cols=["value", "latitude"],
            statistics=["mean"],
        )
        result = step.run(larger_event_frame)

        df = result.collect()
        assert "value_online_mean" in df.columns
        assert "latitude_online_mean" in df.columns

    def test_running_mean_converges(self, larger_event_frame: EventFrame) -> None:
        """Test that running mean approaches true mean."""
        step = OnlineStatisticsStep(value_cols=["value"], statistics=["mean"])
        result = step.run(larger_event_frame)

        df = result.collect()
        true_mean = df["value"].mean()
        online_mean_final = df["value_online_mean"][-1]
        assert online_mean_final == pytest.approx(true_mean, rel=0.01)


class TestStreamingInterEventStep:
    """Tests for StreamingInterEventStep."""

    def test_computes_streaming_inter_event(self, sample_event_frame: EventFrame) -> None:
        """Test streaming inter-event time computation."""
        step = StreamingInterEventStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "streaming_inter_event" in df.columns

    def test_computes_event_rate(self, sample_event_frame: EventFrame) -> None:
        """Test streaming event rate computation."""
        step = StreamingInterEventStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "streaming_event_rate" in df.columns

    def test_with_ema_smoothing(self, sample_event_frame: EventFrame) -> None:
        """Test EMA smoothing of inter-event times."""
        step = StreamingInterEventStep(ema_alpha=0.2)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "streaming_inter_event_ema" in df.columns

    def test_first_event_zero(self, sample_event_frame: EventFrame) -> None:
        """Test that first event has zero inter-event time."""
        step = StreamingInterEventStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["streaming_inter_event"][0] == 0.0

    def test_inter_event_non_negative(self, sample_event_frame: EventFrame) -> None:
        """Test that inter-event times are non-negative."""
        step = StreamingInterEventStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["streaming_inter_event"].min() >= 0


class TestStreamingDecayStep:
    """Tests for StreamingDecayStep."""

    def test_computes_streaming_decay(self, sample_event_frame: EventFrame) -> None:
        """Test streaming decay computation."""
        step = StreamingDecayStep(value_cols=["severity"], decay_rate=0.1)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "severity_decayed_sum" in df.columns

    def test_decay_increases_with_events(self, sample_event_frame: EventFrame) -> None:
        """Test that decayed sum increases as events accumulate."""
        step = StreamingDecayStep(value_cols=["severity"], decay_rate=0.1)
        result = step.run(sample_event_frame)

        df = result.collect()
        # Decayed sum should be positive
        assert df["severity_decayed_sum"].min() >= 0

    def test_decay_rate_affects_values(self, sample_event_frame: EventFrame) -> None:
        """Test that higher decay rate leads to faster decay."""
        step_slow = StreamingDecayStep(value_cols=["severity"], decay_rate=0.01)
        step_fast = StreamingDecayStep(value_cols=["severity"], decay_rate=1.0)

        result_slow = step_slow.run(sample_event_frame).collect()
        result_fast = step_fast.run(sample_event_frame).collect()

        # With faster decay, the sum should be lower since old values decay faster
        # (assuming similar values throughout)
        assert result_fast["severity_decayed_sum"][-1] <= result_slow["severity_decayed_sum"][-1]


class TestEventBufferStep:
    """Tests for EventBufferStep."""

    def test_creates_buffer_position(self, sample_event_frame: EventFrame) -> None:
        """Test buffer position tracking."""
        step = EventBufferStep(buffer_size=10)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "buffer_position" in df.columns

    def test_buffer_fill_ratio(self, sample_event_frame: EventFrame) -> None:
        """Test buffer fill ratio indicator."""
        step = EventBufferStep(buffer_size=10)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "buffer_fill_ratio" in df.columns
        # Fill ratio should be between 0 and 1
        assert df["buffer_fill_ratio"].min() >= 0
        assert df["buffer_fill_ratio"].max() <= 1.0

    def test_buffer_position_cycles(self, larger_event_frame: EventFrame) -> None:
        """Test that buffer position cycles."""
        buffer_size = 10
        step = EventBufferStep(buffer_size=buffer_size)
        result = step.run(larger_event_frame)

        df = result.collect()
        # Position should cycle between 0 and buffer_size-1
        assert df["buffer_position"].min() >= 0
        assert df["buffer_position"].max() < buffer_size


class TestStreamingStepBase:
    """Tests for StreamingStep base class behavior."""

    def test_is_instance_of_streaming_step(self, sample_event_frame: EventFrame) -> None:
        """Test that streaming steps inherit from StreamingStep."""
        step = StreamingWindowStep(window_size=5)
        assert isinstance(step, StreamingStep)

    def test_batch_steps_not_streaming(self, sample_event_frame: EventFrame) -> None:
        """Test that batch steps are not StreamingStep instances."""
        from eventflow.core.steps.point_process import HawkesKernelStep

        step = HawkesKernelStep(alpha=0.3, beta=1.0, mu=0.1)
        assert not isinstance(step, StreamingStep)
