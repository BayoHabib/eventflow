"""Streaming pipeline steps for continuous event flows.

This module provides steps that operate on ordered events using lazy iterators
to support continuous/streaming data flows. These steps are designed for:
- Real-time event processing
- Memory-efficient processing of large event sequences
- Sliding window computations over event streams
- Online feature updates
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import polars as pl
from pydantic import BaseModel, Field

from eventflow.core.pipeline import Step
from eventflow.core.schema import FeatureProvenance
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)

T = TypeVar("T")


# -----------------------------------------------------------------------------
# Streaming Data Structures
# -----------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """A single event in a stream with timestamp and attributes."""

    timestamp: Any
    attributes: dict[str, Any] = field(default_factory=dict)
    index: int = 0


@dataclass
class StreamWindow:
    """A sliding window of events for streaming computations."""

    events: deque[StreamEvent] = field(default_factory=deque)
    max_size: int = 100
    max_duration_seconds: float | None = None

    def add(self, event: StreamEvent) -> list[StreamEvent]:
        """Add event to window and return expired events."""
        expired: list[StreamEvent] = []

        # Remove events exceeding max size
        while len(self.events) >= self.max_size:
            expired.append(self.events.popleft())

        # Remove events exceeding max duration
        if self.max_duration_seconds and self.events:
            cutoff = event.timestamp
            while self.events:
                oldest = self.events[0]
                dt = (cutoff - oldest.timestamp).total_seconds()
                if dt > self.max_duration_seconds:
                    expired.append(self.events.popleft())
                else:
                    break

        self.events.append(event)
        return expired

    def get_events(self) -> list[StreamEvent]:
        """Get all events in the window."""
        return list(self.events)

    def __len__(self) -> int:
        return len(self.events)


class StreamState(Generic[T]):
    """Mutable state container for streaming computations."""

    def __init__(self, initial_value: T) -> None:
        self._value = initial_value
        self._history: list[T] = []
        self._update_count = 0

    @property
    def value(self) -> T:
        return self._value

    def update(self, new_value: T, keep_history: bool = False) -> None:
        """Update state value."""
        if keep_history:
            self._history.append(self._value)
        self._value = new_value
        self._update_count += 1

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def history(self) -> list[T]:
        return self._history


# -----------------------------------------------------------------------------
# Pydantic Config Models
# -----------------------------------------------------------------------------


class StreamingWindowConfig(BaseModel):
    """Configuration for streaming window step."""

    window_size: int = Field(default=100, ge=1, description="Maximum events in window")
    window_duration: str | None = Field(
        default=None, description="Maximum time span (e.g., '1h', '6h')"
    )
    emit_on_expire: bool = Field(default=False, description="Emit features when events expire")


class OnlineStatisticsConfig(BaseModel):
    """Configuration for online statistics step."""

    statistics: list[str] = Field(
        default=["mean", "std", "min", "max"],
        description="Statistics to compute online",
    )
    value_cols: list[str] = Field(..., description="Columns to compute statistics for")
    decay_factor: float = Field(
        default=0.99, ge=0, le=1, description="Exponential decay for weighted stats"
    )


class StreamingHawkesConfig(BaseModel):
    """Configuration for streaming Hawkes intensity."""

    alpha: float = Field(default=0.5, gt=0, le=1, description="Excitation parameter")
    beta: float = Field(default=1.0, gt=0, description="Decay rate")
    mu: float = Field(default=0.1, gt=0, description="Background intensity")
    time_unit: str = Field(default="1h", description="Time unit")
    update_threshold: float = Field(default=0.001, description="Minimum intensity change to update")


class EventBufferConfig(BaseModel):
    """Configuration for event buffer step."""

    buffer_size: int = Field(default=1000, ge=1, description="Maximum buffer size")
    flush_interval: str | None = Field(default=None, description="Time interval to flush buffer")
    flush_on_full: bool = Field(default=True, description="Flush when buffer is full")


# -----------------------------------------------------------------------------
# Abstract Streaming Step Base
# -----------------------------------------------------------------------------


class StreamingStep(Step, ABC):
    """Abstract base class for streaming pipeline steps.

    Streaming steps maintain state across event arrivals and support
    incremental computation over event streams.
    """

    @abstractmethod
    def initialize_state(self) -> Any:
        """Initialize streaming state before processing."""
        raise NotImplementedError

    @abstractmethod
    def process_event(self, event: StreamEvent, state: Any) -> dict[str, Any]:
        """Process a single event and return computed features."""
        raise NotImplementedError

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute streaming step by iterating over sorted events."""
        timestamp_col = event_frame.schema.timestamp_col
        df = event_frame.lazy_frame.sort(timestamp_col).collect()

        state = self.initialize_state()
        results: list[dict[str, Any]] = []

        for i, row in enumerate(df.iter_rows(named=True)):
            event = StreamEvent(
                timestamp=row[timestamp_col],
                attributes=row,
                index=i,
            )
            features = self.process_event(event, state)
            results.append(features)

        # Add computed features as new columns
        if results:
            for col_name in results[0]:
                df = df.with_columns(pl.Series(col_name, [r.get(col_name) for r in results]))

        return event_frame.with_lazy_frame(df.lazy())


# -----------------------------------------------------------------------------
# Streaming Steps
# -----------------------------------------------------------------------------


class StreamingWindowStep(StreamingStep):
    """Maintain a sliding window over the event stream.

    Computes window-based features as events arrive and optionally
    emits features when events expire from the window.

    Outputs:
        - window_size: Current number of events in window
        - window_duration: Time span of events in window
        - events_expired: Number of events expired in this update
    """

    def __init__(
        self,
        window_size: int = 100,
        window_duration: str | None = None,
        emit_on_expire: bool = False,
    ) -> None:
        self.window_size = window_size
        self.window_duration = window_duration
        self.emit_on_expire = emit_on_expire

        self._duration_seconds: float | None = None
        if window_duration:
            self._duration_seconds = _parse_duration_seconds(window_duration)

    def initialize_state(self) -> StreamWindow:
        """Initialize the sliding window."""
        return StreamWindow(
            max_size=self.window_size,
            max_duration_seconds=self._duration_seconds,
        )

    def process_event(self, event: StreamEvent, state: StreamWindow) -> dict[str, Any]:
        """Process event and update window."""
        expired = state.add(event)
        events = state.get_events()

        duration = 0.0
        if len(events) > 1:
            duration = (events[-1].timestamp - events[0].timestamp).total_seconds()

        return {
            "stream_window_size": len(state),
            "stream_window_duration": duration,
            "stream_events_expired": len(expired),
        }

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute streaming window step."""
        result = super().run(event_frame)

        for col in ["stream_window_size", "stream_window_duration", "stream_events_expired"]:
            result = result.register_feature(
                col,
                {"source_step": "StreamingWindowStep"},
                provenance=FeatureProvenance(
                    produced_by="StreamingWindowStep",
                    inputs=[event_frame.schema.timestamp_col],
                    tags={"continuous", "streaming"},
                    description=f"Streaming window feature: {col}",
                ),
            )

        return result


class OnlineStatisticsStep(StreamingStep):
    """Compute online/streaming statistics using Welford's algorithm.

    Maintains running statistics that update incrementally as events arrive.
    Supports exponentially weighted statistics with decay factor.

    Outputs:
        - {col}_online_mean: Running mean
        - {col}_online_std: Running standard deviation
        - {col}_online_min: Running minimum
        - {col}_online_max: Running maximum
    """

    def __init__(
        self,
        value_cols: Sequence[str],
        statistics: Sequence[str] | None = None,
        decay_factor: float = 0.99,
    ) -> None:
        self.value_cols = list(value_cols)
        self.statistics = list(statistics) if statistics else ["mean", "std", "min", "max"]
        self.decay_factor = decay_factor

    def initialize_state(self) -> dict[str, dict[str, float]]:
        """Initialize running statistics state."""
        return {
            col: {
                "count": 0.0,
                "mean": 0.0,
                "m2": 0.0,  # For Welford's algorithm
                "min": float("inf"),
                "max": float("-inf"),
            }
            for col in self.value_cols
        }

    def process_event(
        self, event: StreamEvent, state: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Update online statistics with new event."""
        result: dict[str, Any] = {}

        for col in self.value_cols:
            value = event.attributes.get(col)
            if value is None:
                continue

            try:
                value = float(value)
            except (ValueError, TypeError):
                continue

            stats = state[col]
            stats["count"] += 1
            n = stats["count"]

            # Welford's online algorithm for mean and variance
            delta = value - stats["mean"]
            stats["mean"] += delta / n
            delta2 = value - stats["mean"]
            stats["m2"] += delta * delta2

            # Min/max
            if value < stats["min"]:
                stats["min"] = value
            if value > stats["max"]:
                stats["max"] = value

            # Compute outputs
            if "mean" in self.statistics:
                result[f"{col}_online_mean"] = stats["mean"]
            if "std" in self.statistics and n > 1:
                variance = stats["m2"] / (n - 1)
                result[f"{col}_online_std"] = variance**0.5
            elif "std" in self.statistics:
                result[f"{col}_online_std"] = 0.0
            if "min" in self.statistics:
                result[f"{col}_online_min"] = stats["min"]
            if "max" in self.statistics:
                result[f"{col}_online_max"] = stats["max"]

        return result

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute online statistics step."""
        result = super().run(event_frame)

        for col in self.value_cols:
            for stat in self.statistics:
                feat_name = f"{col}_online_{stat}"
                result = result.register_feature(
                    feat_name,
                    {"source_step": "OnlineStatisticsStep"},
                    provenance=FeatureProvenance(
                        produced_by="OnlineStatisticsStep",
                        inputs=[col],
                        tags={"continuous", "streaming"},
                        description=f"Online {stat} of {col}",
                    ),
                )

        return result


class StreamingHawkesStep(StreamingStep):
    """Compute Hawkes intensity in streaming fashion.

    Maintains a running estimate of the Hawkes process intensity
    that updates efficiently as events arrive.

    Outputs:
        - streaming_hawkes_intensity: Current intensity estimate
        - streaming_hawkes_trigger: Cumulative trigger contribution
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        mu: float = 0.1,
        time_unit: str = "1h",
        update_threshold: float = 0.001,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.time_unit = time_unit
        self.update_threshold = update_threshold
        self._beta_per_second = beta / _parse_duration_seconds(time_unit)

    def initialize_state(self) -> dict[str, Any]:
        """Initialize Hawkes state."""
        return {
            "last_timestamp": None,
            "trigger_sum": 0.0,
            "event_times": deque(maxlen=100),  # Keep recent event times
        }

    def process_event(self, event: StreamEvent, state: dict[str, Any]) -> dict[str, Any]:
        """Update Hawkes intensity with new event."""
        current_ts = event.timestamp

        # Decay existing trigger contributions
        if state["last_timestamp"] is not None:
            dt = (current_ts - state["last_timestamp"]).total_seconds()
            decay = -self._beta_per_second * dt
            import math

            state["trigger_sum"] *= math.exp(decay)

        # Add new trigger from this event
        state["trigger_sum"] += self.alpha * self.beta
        state["last_timestamp"] = current_ts
        state["event_times"].append(current_ts)

        intensity = self.mu + state["trigger_sum"]

        return {
            "streaming_hawkes_intensity": intensity,
            "streaming_hawkes_trigger": state["trigger_sum"],
        }

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute streaming Hawkes step."""
        result = super().run(event_frame)

        for col in ["streaming_hawkes_intensity", "streaming_hawkes_trigger"]:
            result = result.register_feature(
                col,
                {"source_step": "StreamingHawkesStep"},
                provenance=FeatureProvenance(
                    produced_by="StreamingHawkesStep",
                    inputs=[event_frame.schema.timestamp_col],
                    tags={"continuous", "streaming", "point_process"},
                    description=f"Streaming Hawkes feature: {col}",
                    metadata={
                        "alpha": self.alpha,
                        "beta": self.beta,
                        "mu": self.mu,
                    },
                ),
            )

        return result


class EventBufferStep(StreamingStep):
    """Buffer events for batch processing.

    Accumulates events in a buffer and provides batch-level features.
    Useful for micro-batching in streaming pipelines.

    Outputs:
        - buffer_position: Position within current buffer
        - buffer_fill_ratio: How full the buffer is
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        flush_interval: str | None = None,
        flush_on_full: bool = True,
    ) -> None:
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.flush_on_full = flush_on_full

        self._flush_seconds: float | None = None
        if flush_interval:
            self._flush_seconds = _parse_duration_seconds(flush_interval)

    def initialize_state(self) -> dict[str, Any]:
        """Initialize buffer state."""
        return {
            "buffer": [],
            "last_flush": None,
            "flush_count": 0,
        }

    def process_event(self, event: StreamEvent, state: dict[str, Any]) -> dict[str, Any]:
        """Add event to buffer and check flush conditions."""
        state["buffer"].append(event)
        buffer_len = len(state["buffer"])

        # Check flush conditions
        should_flush = False
        if self.flush_on_full and buffer_len >= self.buffer_size:
            should_flush = True
        elif self._flush_seconds and state["last_flush"]:
            dt = (event.timestamp - state["last_flush"]).total_seconds()
            if dt >= self._flush_seconds:
                should_flush = True

        if should_flush:
            state["buffer"] = []
            state["last_flush"] = event.timestamp
            state["flush_count"] += 1
            buffer_len = 0

        return {
            "buffer_position": buffer_len,
            "buffer_fill_ratio": buffer_len / self.buffer_size,
            "buffer_flush_count": state["flush_count"],
        }

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute buffer step."""
        result = super().run(event_frame)

        for col in ["buffer_position", "buffer_fill_ratio", "buffer_flush_count"]:
            result = result.register_feature(
                col,
                {"source_step": "EventBufferStep"},
                provenance=FeatureProvenance(
                    produced_by="EventBufferStep",
                    inputs=[event_frame.schema.timestamp_col],
                    tags={"streaming"},
                    description=f"Buffer feature: {col}",
                ),
            )

        return result


class StreamingDecayStep(StreamingStep):
    """Apply streaming exponential decay to accumulated values.

    Maintains decayed sums that update as events arrive.

    Outputs:
        - {col}_decayed_sum: Exponentially decayed cumulative sum
    """

    def __init__(
        self,
        value_cols: Sequence[str],
        decay_rate: float = 0.1,
        time_unit: str = "1h",
    ) -> None:
        self.value_cols = list(value_cols)
        self.decay_rate = decay_rate
        self.time_unit = time_unit
        self._decay_per_second = decay_rate / _parse_duration_seconds(time_unit)

    def initialize_state(self) -> dict[str, dict[str, Any]]:
        """Initialize decay state."""
        return {col: {"decayed_sum": 0.0, "last_timestamp": None} for col in self.value_cols}

    def process_event(self, event: StreamEvent, state: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Update decayed sums."""
        import math

        result: dict[str, Any] = {}

        for col in self.value_cols:
            col_state = state[col]
            value = event.attributes.get(col, 0)

            try:
                value = float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                value = 0.0

            # Decay existing sum
            if col_state["last_timestamp"] is not None:
                dt = (event.timestamp - col_state["last_timestamp"]).total_seconds()
                decay = math.exp(-self._decay_per_second * dt)
                col_state["decayed_sum"] *= decay

            # Add new value
            col_state["decayed_sum"] += value
            col_state["last_timestamp"] = event.timestamp

            result[f"{col}_decayed_sum"] = col_state["decayed_sum"]

        return result

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute streaming decay step."""
        result = super().run(event_frame)

        for col in self.value_cols:
            feat_name = f"{col}_decayed_sum"
            result = result.register_feature(
                feat_name,
                {"source_step": "StreamingDecayStep"},
                provenance=FeatureProvenance(
                    produced_by="StreamingDecayStep",
                    inputs=[col, event_frame.schema.timestamp_col],
                    tags={"continuous", "streaming"},
                    description=f"Exponentially decayed sum of {col}",
                    metadata={"decay_rate": self.decay_rate},
                ),
            )

        return result


class StreamingInterEventStep(StreamingStep):
    """Compute inter-event features in streaming fashion.

    Tracks inter-event times and computes running statistics.

    Outputs:
        - streaming_inter_event: Current inter-event time
        - streaming_inter_event_ema: Exponential moving average
        - streaming_event_rate: Events per time unit
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        time_unit: str = "1h",
    ) -> None:
        self.ema_alpha = ema_alpha
        self.time_unit = time_unit
        self._unit_seconds = _parse_duration_seconds(time_unit)

    def initialize_state(self) -> dict[str, Any]:
        """Initialize inter-event state."""
        return {
            "last_timestamp": None,
            "ema_inter_event": None,
            "event_count": 0,
            "first_timestamp": None,
        }

    def process_event(self, event: StreamEvent, state: dict[str, Any]) -> dict[str, Any]:
        """Update inter-event statistics."""
        state["event_count"] += 1

        if state["first_timestamp"] is None:
            state["first_timestamp"] = event.timestamp

        inter_event = 0.0
        if state["last_timestamp"] is not None:
            inter_event = (event.timestamp - state["last_timestamp"]).total_seconds()

            # Update EMA
            if state["ema_inter_event"] is None:
                state["ema_inter_event"] = inter_event
            else:
                state["ema_inter_event"] = (
                    self.ema_alpha * inter_event + (1 - self.ema_alpha) * state["ema_inter_event"]
                )

        state["last_timestamp"] = event.timestamp

        # Compute event rate
        total_time = (event.timestamp - state["first_timestamp"]).total_seconds()
        event_rate = (
            state["event_count"] / (total_time / self._unit_seconds) if total_time > 0 else 0.0
        )

        return {
            "streaming_inter_event": inter_event,
            "streaming_inter_event_ema": state["ema_inter_event"] or 0.0,
            "streaming_event_rate": event_rate,
        }

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute streaming inter-event step."""
        result = super().run(event_frame)

        for col in [
            "streaming_inter_event",
            "streaming_inter_event_ema",
            "streaming_event_rate",
        ]:
            result = result.register_feature(
                col,
                {"source_step": "StreamingInterEventStep"},
                provenance=FeatureProvenance(
                    produced_by="StreamingInterEventStep",
                    inputs=[event_frame.schema.timestamp_col],
                    tags={"continuous", "streaming"},
                    description=f"Streaming inter-event feature: {col}",
                ),
            )

        return result


# -----------------------------------------------------------------------------
# Stream Iterator Support
# -----------------------------------------------------------------------------


def event_stream_iterator(
    event_frame: EventFrame,
    batch_size: int | None = None,
) -> Iterator[StreamEvent]:
    """Create an iterator over events in the EventFrame.

    Args:
        event_frame: Source EventFrame
        batch_size: Optional batch size for chunked iteration

    Yields:
        StreamEvent objects in timestamp order
    """
    timestamp_col = event_frame.schema.timestamp_col
    df = event_frame.lazy_frame.sort(timestamp_col).collect()

    for i, row in enumerate(df.iter_rows(named=True)):
        yield StreamEvent(
            timestamp=row[timestamp_col],
            attributes=row,
            index=i,
        )


def batch_stream_iterator(
    event_frame: EventFrame,
    batch_size: int = 100,
) -> Iterator[list[StreamEvent]]:
    """Create a batched iterator over events.

    Args:
        event_frame: Source EventFrame
        batch_size: Number of events per batch

    Yields:
        Lists of StreamEvent objects
    """
    batch: list[StreamEvent] = []

    for event in event_stream_iterator(event_frame):
        batch.append(event)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _parse_duration_seconds(duration: str) -> float:
    """Parse duration string to seconds."""
    unit_map = {
        "s": 1,
        "m": 60,
        "min": 60,
        "h": 3600,
        "hr": 3600,
        "d": 86400,
        "day": 86400,
        "w": 604800,
        "week": 604800,
        "1s": 1,
        "1m": 60,
        "1h": 3600,
        "1d": 86400,
        "1w": 604800,
    }
    return float(unit_map.get(duration.lower(), 3600))
