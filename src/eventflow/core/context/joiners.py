"""Join strategies for context enrichment."""

from typing import Literal

import polars as pl

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import ContextSchema


class TemporalJoin:
    """
    Temporal join strategy for aligning events with temporal context.
    """

    def __init__(
        self,
        strategy: Literal["exact", "nearest", "backward", "forward"] = "nearest",
        window: str | None = None,
        tolerance: str | None = None,
    ) -> None:
        """
        Initialize temporal join.

        Args:
            strategy: Join strategy:
                - "exact": Exact timestamp match
                - "nearest": Nearest timestamp
                - "backward": Most recent before event
                - "forward": Next after event
            window: Optional time window for matching
            tolerance: Maximum time difference for matches
        """
        self.strategy: Literal["exact", "nearest", "backward", "forward"] = strategy
        self.window = window
        self.tolerance = tolerance

    def join(
        self,
        event_frame: EventFrame,
        context_frame: pl.LazyFrame,
        context_schema: ContextSchema,
    ) -> EventFrame:
        """
        Perform temporal join.

        Args:
            event_frame: Event data
            context_frame: Context data
            context_schema: Schema of context data

        Returns:
            EventFrame with context columns joined
        """
        event_ts_col = event_frame.schema.timestamp_col
        context_ts_col = context_schema.timestamp_col

        if context_ts_col is None:
            raise ValueError("Context schema must have timestamp_col for temporal join")

        if self.strategy == "exact":
            lf = event_frame.lazy_frame.join(
                context_frame,
                left_on=event_ts_col,
                right_on=context_ts_col,
                how="left",
            )
        else:
            # Use asof join for temporal alignment
            lf = event_frame.lazy_frame.join_asof(
                context_frame,
                left_on=event_ts_col,
                right_on=context_ts_col,
                strategy=self.strategy,
            )

        return event_frame.with_lazy_frame(lf)

    def __repr__(self) -> str:
        """String representation."""
        return f"TemporalJoin(strategy={self.strategy})"


class SpatialJoin:
    """
    Spatial join strategy for aligning events with spatial context.
    """

    def __init__(
        self,
        join_type: str = "grid",
        spatial_col: str = "grid_id",
    ) -> None:
        """
        Initialize spatial join.

        Args:
            join_type: Type of spatial join:
                - "grid": Join on grid_id
                - "zone": Join on zone_id
                - "nearest": Nearest neighbor
            spatial_col: Name of spatial column to join on
        """
        self.join_type = join_type
        self.spatial_col = spatial_col

    def join(
        self,
        event_frame: EventFrame,
        context_frame: pl.LazyFrame,
        context_schema: ContextSchema,
    ) -> EventFrame:
        """
        Perform spatial join.

        Args:
            event_frame: Event data
            context_frame: Context data
            context_schema: Schema of context data

        Returns:
            EventFrame with context columns joined
        """
        context_spatial_col = context_schema.spatial_col

        if context_spatial_col is None:
            raise ValueError("Context schema must have spatial_col for spatial join")

        if self.join_type in ["grid", "zone"]:
            # Simple join on spatial ID
            lf = event_frame.lazy_frame.join(
                context_frame,
                left_on=self.spatial_col,
                right_on=context_spatial_col,
                how="left",
            )
        else:
            raise NotImplementedError(f"Spatial join type {self.join_type} not yet implemented")

        return event_frame.with_lazy_frame(lf)

    def __repr__(self) -> str:
        """String representation."""
        return f"SpatialJoin(type={self.join_type}, col={self.spatial_col})"


class SpatioTemporalJoin:
    """
    Combined spatio-temporal join strategy.
    """

    def __init__(
        self,
        temporal_join: TemporalJoin,
        spatial_join: SpatialJoin,
    ) -> None:
        """
        Initialize spatio-temporal join.

        Args:
            temporal_join: Temporal join strategy
            spatial_join: Spatial join strategy
        """
        self.temporal_join = temporal_join
        self.spatial_join = spatial_join

    def join(
        self,
        event_frame: EventFrame,
        context_frame: pl.LazyFrame,
        context_schema: ContextSchema,
    ) -> EventFrame:
        """
        Perform spatio-temporal join.

        First applies spatial join, then temporal join.

        Args:
            event_frame: Event data
            context_frame: Context data
            context_schema: Schema of context data

        Returns:
            EventFrame with context columns joined
        """
        # Apply spatial join first
        if context_schema.spatial_col is not None:
            event_frame = self.spatial_join.join(event_frame, context_frame, context_schema)

        # Then apply temporal join
        if context_schema.timestamp_col is not None:
            event_frame = self.temporal_join.join(event_frame, context_frame, context_schema)

        return event_frame

    def __repr__(self) -> str:
        """String representation."""
        return f"SpatioTemporalJoin({self.spatial_join}, {self.temporal_join})"
