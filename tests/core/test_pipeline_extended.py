"""Tests for extended pipeline functionality (StreamPipeline, BranchingPipeline)."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import BranchingPipeline, Pipeline, StreamPipeline
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.point_process import HawkesKernelStep
from eventflow.core.steps.streaming import (
    OnlineStatisticsStep,
    StreamingHawkesStep,
    StreamingInterEventStep,
    StreamingWindowStep,
)
from eventflow.core.steps.temporal import ExtractTemporalComponentsStep


@pytest.fixture
def sample_event_frame() -> EventFrame:
    """Create a sample EventFrame for pipeline tests."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    n_events = 50
    timestamps = [base_time + timedelta(hours=i) for i in range(n_events)]

    lf = pl.LazyFrame(
        {
            "timestamp": timestamps,
            "latitude": [41.8781 + (i % 10) * 0.001 for i in range(n_events)],
            "longitude": [-87.6298 + (i % 10) * 0.001 for i in range(n_events)],
            "value": [float(i % 10 + 1) for i in range(n_events)],
            "category": ["A" if i % 2 == 0 else "B" for i in range(n_events)],
        }
    )

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["value"],
        categorical_cols=["category"],
    )
    metadata = EventMetadata(dataset_name="test-pipeline", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


class TestStreamPipeline:
    """Tests for StreamPipeline class."""

    def test_creates_stream_pipeline(self) -> None:
        """Test StreamPipeline creation."""
        steps = [
            StreamingWindowStep(window_size=10),
            StreamingHawkesStep(alpha=0.3, beta=1.0, mu=0.1),
        ]
        pipeline = StreamPipeline(steps=steps, batch_size=20)
        assert pipeline.batch_size == 20
        assert len(pipeline.steps) == 2

    def test_run_returns_result(self, sample_event_frame: EventFrame) -> None:
        """Test that run returns EventFrame."""
        pipeline = StreamPipeline(
            steps=[StreamingWindowStep(window_size=10)],
            batch_size=10,
        )
        result = pipeline.run(sample_event_frame)
        assert isinstance(result, EventFrame)

    def test_run_applies_steps(self, sample_event_frame: EventFrame) -> None:
        """Test that steps are applied."""
        pipeline = StreamPipeline(
            steps=[
                StreamingWindowStep(window_size=10),
                StreamingInterEventStep(),
            ],
            batch_size=10,
        )
        result = pipeline.run(sample_event_frame)
        df = result.collect()

        assert "stream_window_size" in df.columns
        assert "streaming_inter_event" in df.columns

    def test_run_incremental(self, sample_event_frame: EventFrame) -> None:
        """Test incremental batch processing."""
        pipeline = StreamPipeline(
            steps=[StreamingWindowStep(window_size=5)],
            batch_size=10,
        )

        batches = list(pipeline.run_incremental(sample_event_frame))
        assert len(batches) == 5  # 50 events / 10 = 5 batches

        # Each batch should be an EventFrame
        for batch in batches:
            assert isinstance(batch, EventFrame)

    def test_batch_size_respected(self, sample_event_frame: EventFrame) -> None:
        """Test that batch size is respected."""
        batch_size = 15
        pipeline = StreamPipeline(
            steps=[StreamingWindowStep(window_size=5)],
            batch_size=batch_size,
        )

        batches = list(pipeline.run_incremental(sample_event_frame))
        # First batches should have batch_size events
        first_batch_df = batches[0].collect()
        assert len(first_batch_df) == batch_size

    def test_process_single_event(self, sample_event_frame: EventFrame) -> None:
        """Test single event processing."""
        pipeline = StreamPipeline(
            steps=[StreamingWindowStep(window_size=5)],
            batch_size=10,
        )

        # Get a single event
        single_event_lf = sample_event_frame.lazy_frame.head(1)
        single_ef = EventFrame(
            single_event_lf,
            sample_event_frame.schema,
            sample_event_frame.metadata,
        )

        result = pipeline.run(single_ef)
        df = result.collect()
        assert len(df) == 1


class TestBranchingPipeline:
    """Tests for BranchingPipeline class."""

    def test_creates_branching_pipeline(self) -> None:
        """Test BranchingPipeline creation."""
        batch_steps = [
            ExtractTemporalComponentsStep(["hour_of_day"]),
            HawkesKernelStep(alpha=0.3, beta=1.0, mu=0.1),
        ]
        stream_steps = [
            StreamingWindowStep(window_size=10),
            StreamingInterEventStep(),
        ]
        pipeline = BranchingPipeline(
            batch_steps=batch_steps,
            stream_steps=stream_steps,
        )
        assert len(pipeline.batch_steps) == 2
        assert len(pipeline.stream_steps) == 2

    def test_from_steps_auto_categorizes(self) -> None:
        """Test automatic step categorization."""
        all_steps = [
            ExtractTemporalComponentsStep(["hour_of_day"]),  # Batch
            HawkesKernelStep(alpha=0.3, beta=1.0, mu=0.1),  # Batch
            StreamingWindowStep(window_size=10),  # Stream
            StreamingInterEventStep(),  # Stream
        ]
        pipeline = BranchingPipeline.from_steps(all_steps)

        assert len(pipeline.batch_steps) == 2
        assert len(pipeline.stream_steps) == 2

    def test_run_applies_all_steps(self, sample_event_frame: EventFrame) -> None:
        """Test that run applies both batch and stream steps."""
        pipeline = BranchingPipeline(
            batch_steps=[ExtractTemporalComponentsStep(["hour_of_day"])],
            stream_steps=[StreamingWindowStep(window_size=5)],
        )

        result = pipeline.run(sample_event_frame)
        df = result.collect()

        # Both types of features should be present
        assert "hour_of_day" in df.columns
        assert "stream_window_size" in df.columns

    def test_batch_first_then_stream(self, sample_event_frame: EventFrame) -> None:
        """Test that batch steps run before stream steps."""
        pipeline = BranchingPipeline(
            batch_steps=[ExtractTemporalComponentsStep(["hour_of_day"])],
            stream_steps=[StreamingWindowStep(window_size=5)],
        )

        result = pipeline.run(sample_event_frame)
        df = result.collect()

        # Both should be present - batch runs first
        assert "hour_of_day" in df.columns
        assert "stream_window_size" in df.columns

    def test_empty_batch_steps(self, sample_event_frame: EventFrame) -> None:
        """Test with only stream steps."""
        pipeline = BranchingPipeline(
            batch_steps=[],
            stream_steps=[StreamingWindowStep(window_size=5)],
        )

        result = pipeline.run(sample_event_frame)
        df = result.collect()
        assert "stream_window_size" in df.columns

    def test_empty_stream_steps(self, sample_event_frame: EventFrame) -> None:
        """Test with only batch steps."""
        pipeline = BranchingPipeline(
            batch_steps=[ExtractTemporalComponentsStep(["hour_of_day"])],
            stream_steps=[],
        )

        result = pipeline.run(sample_event_frame)
        df = result.collect()
        assert "hour_of_day" in df.columns


class TestPipelineWithPointProcessSteps:
    """Tests for Pipeline with point-process steps."""

    def test_pipeline_with_hawkes(self, sample_event_frame: EventFrame) -> None:
        """Test pipeline with Hawkes step."""
        pipeline = Pipeline([
            HawkesKernelStep(alpha=0.3, beta=1.0, mu=0.1),
        ])

        result = pipeline.run(sample_event_frame)
        df = result.collect()

        assert "hawkes_intensity" in df.columns
        assert "hawkes_trigger" in df.columns
        assert "hawkes_background" in df.columns

    def test_pipeline_with_streaming_steps(self, sample_event_frame: EventFrame) -> None:
        """Test pipeline with streaming steps."""
        pipeline = Pipeline([
            StreamingWindowStep(window_size=10),
            StreamingHawkesStep(alpha=0.3, beta=1.0, mu=0.1),
            OnlineStatisticsStep(value_cols=["value"], statistics=["mean"]),
        ])

        result = pipeline.run(sample_event_frame)
        df = result.collect()

        assert "stream_window_size" in df.columns
        assert "streaming_hawkes_intensity" in df.columns
        assert "value_online_mean" in df.columns

    def test_mixed_pipeline(self, sample_event_frame: EventFrame) -> None:
        """Test pipeline with both batch and streaming steps."""
        pipeline = Pipeline([
            ExtractTemporalComponentsStep(["hour_of_day", "day_of_week"]),
            HawkesKernelStep(alpha=0.3, beta=1.0, mu=0.1),
            StreamingWindowStep(window_size=10),
            StreamingInterEventStep(),
        ])

        result = pipeline.run(sample_event_frame)
        df = result.collect()

        # All features should be present
        assert "hour_of_day" in df.columns
        assert "hawkes_intensity" in df.columns
        assert "stream_window_size" in df.columns
        assert "streaming_inter_event" in df.columns

    def test_pipeline_preserves_original_columns(self, sample_event_frame: EventFrame) -> None:
        """Test that pipeline preserves original columns."""
        pipeline = Pipeline([
            HawkesKernelStep(alpha=0.3, beta=1.0, mu=0.1),
        ])

        result = pipeline.run(sample_event_frame)
        df = result.collect()

        # Original columns should still be present
        assert "timestamp" in df.columns
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        assert "value" in df.columns
        assert "category" in df.columns


class TestPipelineEdgeCases:
    """Tests for pipeline edge cases."""

    def test_empty_event_frame(self) -> None:
        """Test pipeline with empty EventFrame."""
        lf = pl.LazyFrame({
            "timestamp": [],
            "latitude": [],
            "longitude": [],
        }).with_columns([
            pl.col("timestamp").cast(pl.Datetime),
            pl.col("latitude").cast(pl.Float64),
            pl.col("longitude").cast(pl.Float64),
        ])
        schema = EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
        )
        metadata = EventMetadata(dataset_name="empty", crs="EPSG:4326")
        ef = EventFrame(lf, schema, metadata)

        pipeline = Pipeline([
            StreamingWindowStep(window_size=5),
        ])

        result = pipeline.run(ef)
        df = result.collect()
        assert len(df) == 0

    def test_single_event(self) -> None:
        """Test pipeline with single event."""
        lf = pl.LazyFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "latitude": [41.8781],
            "longitude": [-87.6298],
        })
        schema = EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
        )
        metadata = EventMetadata(dataset_name="single", crs="EPSG:4326")
        ef = EventFrame(lf, schema, metadata)

        pipeline = Pipeline([
            StreamingWindowStep(window_size=5),
            StreamingInterEventStep(),
        ])

        result = pipeline.run(ef)
        df = result.collect()
        assert len(df) == 1
        assert df["stream_window_size"][0] == 1
