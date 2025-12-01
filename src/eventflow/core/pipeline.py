"""Pipeline orchestration for composable transformations."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.utils import get_logger

logger = get_logger(__name__)


class Step(ABC):
    """
    Base class for pipeline steps.

    A step is a stateless transformation that takes an EventFrame
    and returns a new EventFrame.
    """

    @abstractmethod
    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute the step transformation and return the modified frame."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """String representation of the step."""
        return f"{self.__class__.__name__}()"


class Pipeline:
    """
    A pipeline is a sequence of steps applied to an EventFrame.

    Each step transforms the EventFrame and passes it to the next step.
    """

    def __init__(self, steps: list[Step]) -> None:
        """
        Initialize a pipeline.

        Args:
            steps: List of Step instances to apply in sequence
        """
        self.steps = steps
        self._last_schema: EventSchema | None = None
        self._last_metadata: EventMetadata | None = None
        logger.info(
            f"Created pipeline with {len(steps)} steps: {[s.__class__.__name__ for s in steps]}"
        )

    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Run the pipeline on an EventFrame.

        Args:
            event_frame: Input EventFrame

        Returns:
            Transformed EventFrame after all steps
        """
        logger.info(f"Starting pipeline execution with {len(self.steps)} steps")
        current = event_frame
        self._last_schema = event_frame.schema
        self._last_metadata = event_frame.metadata

        for i, step in enumerate(self.steps, 1):
            step_name = step.__class__.__name__
            logger.info(f"Step {i}/{len(self.steps)}: Executing {step_name}")
            try:
                previous_schema = current.schema
                result = step.run(current)
                if not isinstance(result, EventFrame):
                    raise TypeError(
                        f"Step {step_name} returned {type(result).__name__} instead of EventFrame"
                    )

                issues = previous_schema.compatibility_issues(result.schema)
                if issues:
                    issue_summary = "; ".join(issues)
                    logger.error(
                        "Step %s produced an incompatible schema: %s",
                        step_name,
                        issue_summary,
                    )
                    raise ValueError(
                        f"Step {step_name} produced incompatible schema: {issue_summary}"
                    )

                current = result
                self._last_schema = current.schema
                self._last_metadata = current.metadata
                logger.debug(
                    "Step %s completed successfully; schema modalities=%s",
                    step_name,
                    sorted(mod.value for mod in current.schema.output_modalities),
                )
            except Exception as e:
                logger.error(f"Step {i}/{len(self.steps)}: {step_name} failed with error: {e}")
                raise

        logger.info("Pipeline execution completed successfully")
        self._last_schema = current.schema
        self._last_metadata = current.metadata
        return current

    def add_step(self, step: Step) -> "Pipeline":
        """
        Add a step to the pipeline.

        Args:
            step: Step to add

        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"Pipeline({' -> '.join(step_names)})"

    def __len__(self) -> int:
        """Get the number of steps."""
        return len(self.steps)

    @property
    def last_schema(self) -> EventSchema | None:
        """Return the schema produced by the most recent pipeline execution."""

        return self._last_schema

    @property
    def last_metadata(self) -> EventMetadata | None:
        """Return the metadata produced by the most recent pipeline execution."""

        return self._last_metadata


class LambdaStep(Step):
    """
    A step that wraps a function.

    Useful for quick transformations without defining a new class.
    """

    def __init__(self, fn: Callable[[EventFrame], EventFrame], name: str | None = None) -> None:
        """
        Initialize a lambda step.

        Args:
            fn: Function that takes and returns an EventFrame
            name: Optional name for the step
        """
        self.fn = fn
        self.name = name or "LambdaStep"

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute the wrapped function."""
        return self.fn(event_frame)

    def __repr__(self) -> str:
        """String representation."""
        return f"LambdaStep({self.name})"


class ConditionalStep(Step):
    """
    A step that conditionally applies another step.
    """

    def __init__(
        self,
        condition: Callable[[EventFrame], bool],
        if_true: Step,
        if_false: Step | None = None,
    ) -> None:
        """
        Initialize a conditional step.

        Args:
            condition: Function that takes EventFrame and returns bool
            if_true: Step to apply if condition is True
            if_false: Optional step to apply if condition is False
        """
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute conditional logic."""
        if self.condition(event_frame):
            return self.if_true.run(event_frame)
        elif self.if_false is not None:
            return self.if_false.run(event_frame)
        else:
            return event_frame

    def __repr__(self) -> str:
        """String representation."""
        return f"ConditionalStep({self.if_true} | {self.if_false})"


class ParallelSteps(Step):
    """
    Apply multiple steps in parallel and merge results.

    Note: This doesn't actually execute in parallel (Polars handles that),
    but it applies multiple transformations to the same input and merges them.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        """
        Initialize parallel steps.

        Args:
            steps: Steps to apply in parallel
        """
        self.steps = list(steps)

    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Apply each step sequentially using the same input instance.

        Notes:
            This intentionally mirrors the sequential behaviour of
            :class:`Pipeline` while keeping the convenience of grouping a list
            of steps under a single wrapper.
        """
        if not self.steps:
            return event_frame

        result = event_frame
        for step in self.steps:
            result = step.run(result)

        return result

    def __repr__(self) -> str:
        """String representation."""
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"ParallelSteps([{', '.join(step_names)}])"


# -----------------------------------------------------------------------------
# Stream Pipeline Support
# -----------------------------------------------------------------------------


class StreamPipeline:
    """
    A pipeline that operates in streaming mode on ordered events.

    StreamPipeline supports:
    - Incremental processing of events as they arrive
    - State management across event batches
    - Branching into stream mode from regular Pipeline
    - Lazy iteration over processed events
    """

    def __init__(
        self,
        steps: list[Step],
        batch_size: int = 100,
        maintain_state: bool = True,
    ) -> None:
        """
        Initialize a stream pipeline.

        Args:
            steps: List of Step instances to apply
            batch_size: Number of events to process per batch
            maintain_state: Whether to maintain state across batches
        """
        self.steps = steps
        self.batch_size = batch_size
        self.maintain_state = maintain_state
        self._state: dict[str, Any] = {}
        self._processed_count = 0
        logger.info(
            f"Created StreamPipeline with {len(steps)} steps, "
            f"batch_size={batch_size}"
        )

    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Run the stream pipeline on an EventFrame.

        Processes events in batches while maintaining streaming state.

        Args:
            event_frame: Input EventFrame

        Returns:
            Transformed EventFrame after all steps
        """
        logger.info(f"Starting stream pipeline execution with {len(self.steps)} steps")

        # Sort by timestamp for streaming order
        timestamp_col = event_frame.schema.timestamp_col
        sorted_frame = event_frame.sort(timestamp_col)

        # Apply steps sequentially
        current = sorted_frame
        for i, step in enumerate(self.steps, 1):
            step_name = step.__class__.__name__
            logger.info(f"Stream step {i}/{len(self.steps)}: {step_name}")

            # Check if step supports streaming
            if hasattr(step, "initialize_state") and hasattr(step, "process_event"):
                # Streaming step - use its native streaming mode
                current = step.run(current)
            else:
                # Regular step - apply normally
                current = step.run(current)

            self._processed_count += 1

        logger.info(
            f"Stream pipeline completed, processed {current.count()} events"
        )
        return current

    def run_incremental(
        self,
        event_frame: EventFrame,
    ) -> Iterator[EventFrame]:
        """
        Run pipeline incrementally, yielding results per batch.

        Args:
            event_frame: Input EventFrame

        Yields:
            EventFrame for each processed batch
        """
        import polars as pl

        timestamp_col = event_frame.schema.timestamp_col
        df = event_frame.lazy_frame.sort(timestamp_col).collect()

        n_events = len(df)
        n_batches = (n_events + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_events)

            batch_df = df.slice(start_idx, end_idx - start_idx)
            batch_frame = event_frame.with_lazy_frame(batch_df.lazy())

            # Process batch through steps
            current = batch_frame
            for step in self.steps:
                current = step.run(current)

            yield current

    def process_event(
        self,
        event_data: dict[str, Any],
        schema: EventSchema,
        metadata: EventMetadata,
    ) -> dict[str, Any]:
        """
        Process a single event through the pipeline.

        Useful for real-time event processing.

        Args:
            event_data: Dictionary of event attributes
            schema: Event schema
            metadata: Event metadata

        Returns:
            Dictionary of processed event with computed features
        """
        import polars as pl

        # Create single-event frame
        df = pl.DataFrame([event_data])
        event_frame = EventFrame(df.lazy(), schema, metadata)

        # Process through pipeline
        result = self.run(event_frame)

        # Extract result
        result_df = result.lazy_frame.collect()
        if len(result_df) > 0:
            return result_df.row(0, named=True)
        return event_data

    @property
    def state(self) -> dict[str, Any]:
        """Get current pipeline state."""
        return self._state

    def reset_state(self) -> None:
        """Reset pipeline state."""
        self._state = {}
        self._processed_count = 0
        logger.info("Stream pipeline state reset")

    def __repr__(self) -> str:
        """String representation."""
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"StreamPipeline({' -> '.join(step_names)})"


class BranchingPipeline:
    """
    A pipeline that can branch between batch and stream modes.

    Automatically selects the appropriate execution mode based on
    the types of steps in the pipeline.
    """

    def __init__(
        self,
        batch_steps: list[Step] | None = None,
        stream_steps: list[Step] | None = None,
        auto_detect: bool = True,
    ) -> None:
        """
        Initialize a branching pipeline.

        Args:
            batch_steps: Steps to run in batch mode
            stream_steps: Steps to run in stream mode
            auto_detect: Automatically detect streaming steps
        """
        self.batch_steps = batch_steps or []
        self.stream_steps = stream_steps or []
        self.auto_detect = auto_detect

        self._batch_pipeline: Pipeline | None = None
        self._stream_pipeline: StreamPipeline | None = None

        if self.batch_steps:
            self._batch_pipeline = Pipeline(self.batch_steps)
        if self.stream_steps:
            self._stream_pipeline = StreamPipeline(self.stream_steps)

    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Run the branching pipeline.

        Executes batch steps first, then stream steps.

        Args:
            event_frame: Input EventFrame

        Returns:
            Transformed EventFrame
        """
        current = event_frame

        # Run batch steps
        if self._batch_pipeline:
            logger.info("Executing batch phase")
            current = self._batch_pipeline.run(current)

        # Run stream steps
        if self._stream_pipeline:
            logger.info("Branching to stream mode")
            current = self._stream_pipeline.run(current)

        return current

    @classmethod
    def from_steps(
        cls,
        steps: list[Step],
        stream_step_types: tuple[type, ...] | None = None,
    ) -> "BranchingPipeline":
        """
        Create a branching pipeline by automatically categorizing steps.

        Args:
            steps: All pipeline steps
            stream_step_types: Tuple of types considered streaming steps

        Returns:
            BranchingPipeline with steps categorized
        """
        batch_steps = []
        stream_steps = []

        for step in steps:
            # Check if step is a streaming step
            is_streaming = (
                hasattr(step, "initialize_state")
                and hasattr(step, "process_event")
            )

            if stream_step_types:
                is_streaming = is_streaming or isinstance(step, stream_step_types)

            if is_streaming:
                stream_steps.append(step)
            else:
                batch_steps.append(step)

        return cls(batch_steps=batch_steps, stream_steps=stream_steps)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BranchingPipeline(batch={len(self.batch_steps)}, "
            f"stream={len(self.stream_steps)})"
        )
