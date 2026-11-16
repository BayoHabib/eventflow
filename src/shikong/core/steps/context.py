"""Context join steps for enriching events with external data.

Each step:
- Inherits from Step base class
- Supports CRS transformation for spatial joins
- Handles missing data gracefully
- Registers provenance for joined columns
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
import shapely
from pydantic import BaseModel, Field
from pyproj import Transformer
from shapely import STRtree, geometry

from shikong.core.pipeline import Step
from shikong.core.schema import FeatureProvenance
from shikong.core.spatial import require_metric_crs
from shikong.core.utils import get_logger

if TYPE_CHECKING:
    from shikong.core.event_frame import EventFrame

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Pydantic Config Models
# -----------------------------------------------------------------------------


class SpatialContextJoinConfig(BaseModel):
    """Configuration for spatial context join step."""

    context_crs: str | None = Field(
        default=None, description="CRS of context data (auto-transform if differs from events)"
    )
    join_type: Literal["contains", "intersects", "nearest"] = Field(
        default="contains", description="Spatial predicate for join"
    )
    max_distance: float | None = Field(
        default=None, ge=0, description="Maximum distance for nearest neighbor join"
    )
    fill_value: Any = Field(default=None, description="Value for unmatched events")


class TemporalContextJoinConfig(BaseModel):
    """Configuration for temporal context join step."""

    strategy: Literal["exact", "nearest", "asof_backward", "asof_forward"] = Field(
        default="asof_backward", description="Temporal join strategy"
    )
    tolerance: str | None = Field(
        default=None, description="Maximum time difference for matching (e.g., '1h', '1d')"
    )
    fill_value: Any = Field(default=None, description="Value for unmatched events")


class SpatioTemporalContextJoinConfig(BaseModel):
    """Configuration for combined spatio-temporal context join."""

    spatial_key: str = Field(..., description="Spatial key column in both datasets")
    temporal_key: str = Field(..., description="Temporal key column in both datasets")
    temporal_strategy: Literal["exact", "asof_backward", "asof_forward"] = Field(
        default="asof_backward", description="Temporal join strategy"
    )
    tolerance: str | None = None
    fill_value: Any = Field(default=None, description="Value for unmatched events")


# -----------------------------------------------------------------------------
# Context Join Steps
# -----------------------------------------------------------------------------


class SpatialContextJoinStep(Step):
    """Join events with spatial context data.

    Supports point-in-polygon, nearest neighbor, and intersection joins.
    Automatically handles CRS transformation if source and target differ.

    Inputs:
        - lat_col, lon_col from EventSchema
        - context_data with spatial information

    Outputs:
        - Joined columns from context_data
    """

    def __init__(
        self,
        context_data: pl.DataFrame | pl.LazyFrame,
        context_spatial_col: str,
        context_cols: Sequence[str],
        join_type: str = "contains",
        context_crs: str | None = None,
        max_distance: float | None = None,
        fill_value: Any = None,
    ) -> None:
        self.context_data = context_data
        self.context_spatial_col = context_spatial_col
        self.context_cols = list(context_cols)
        self.join_type = join_type
        self.context_crs = context_crs
        if max_distance is not None and (not np.isfinite(max_distance) or max_distance < 0):
            raise ValueError("max_distance must be finite and non-negative")
        self.max_distance = max_distance
        self.fill_value = fill_value

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute spatial context join."""
        schema = event_frame.schema

        if schema.lat_col is None or schema.lon_col is None:
            raise ValueError("EventFrame must have lat/lon columns for spatial context join")

        lon_col = schema.lon_col
        lat_col = schema.lat_col
        event_crs = event_frame.metadata.crs

        ctx_df = (
            self.context_data.collect()
            if isinstance(self.context_data, pl.LazyFrame)
            else self.context_data
        )
        geoms = shapely.from_wkt(ctx_df[self.context_spatial_col].to_numpy())
        if not np.all(shapely.is_valid(geoms) & ~shapely.is_empty(geoms)):
            raise ValueError("Context geometries must be valid and non-empty")
        df = event_frame.collect()
        coords = df.select(lon_col, lat_col).to_numpy().astype(float)
        target_crs = self.context_crs or event_crs
        valid = np.isfinite(coords).all(axis=1)
        if target_crs != event_crs:
            # Projection and geometry work stay in this worker, including collection.
            transformer = Transformer.from_crs(event_crs, target_crs, always_xy=True)
            coords[valid, 0], coords[valid, 1] = transformer.transform(
                coords[valid, 0], coords[valid, 1]
            )
        valid &= np.isfinite(coords).all(axis=1)
        if self.join_type == "nearest":
            require_metric_crs(target_crs)
        elif self.join_type not in {"contains", "intersects"}:
            raise ValueError(f"Unknown join type: {self.join_type}")
        tree = STRtree(geoms)
        indices: list[int | None] = []
        for i, (x, y) in enumerate(coords):
            if not valid[i] or len(geoms) == 0:
                indices.append(None)
                continue
            point = geometry.Point(x, y)
            if self.join_type == "nearest":
                matches = tree.query_nearest(point, all_matches=True)
                selected = min(int(j) for j in matches)
                if (
                    self.max_distance is not None
                    and point.distance(geoms[selected]) > self.max_distance
                ):
                    indices.append(None)
                else:
                    indices.append(selected)
            else:
                # STRtree applies predicate(query, tree geometry), hence within.
                predicate = "within" if self.join_type == "contains" else "intersects"
                matches = tree.query(point, predicate=predicate)
                indices.append(min(int(j) for j in matches) if len(matches) else None)
        columns = []
        for col in self.context_cols:
            series = ctx_df[col].gather(pl.Series(indices, dtype=pl.UInt32))
            if self.fill_value is not None:
                series = series.fill_null(self.fill_value)
            columns.append(series.alias(col))
        result = event_frame.with_lazy_frame(df.with_columns(columns).lazy())

        # Register provenance for joined columns
        for col in self.context_cols:
            provenance = FeatureProvenance(
                produced_by="SpatialContextJoinStep",
                inputs=[lon_col, lat_col],
                tags={"context", "spatial"},
                description=f"Spatially joined context: {col}",
                metadata={"join_type": self.join_type},
            )
            result = result.register_feature(
                col,
                {"source_step": "SpatialContextJoinStep", "join_type": self.join_type},
                provenance=provenance,
            )

        return result.require_context(context_tags={"spatial_context"})


class TemporalContextJoinStep(Step):
    """Join events with temporal context data.

    Supports exact matching, nearest timestamp, and as-of joins.

    Inputs:
        - timestamp_col from EventSchema
        - context_data with temporal information

    Outputs:
        - Joined columns from context_data
    """

    def __init__(
        self,
        context_data: pl.DataFrame | pl.LazyFrame,
        context_timestamp_col: str,
        context_cols: Sequence[str],
        strategy: str = "asof_backward",
        tolerance: str | None = None,
        fill_value: Any = None,
    ) -> None:
        self.context_data = context_data
        self.context_timestamp_col = context_timestamp_col
        self.context_cols = list(context_cols)
        self.strategy = strategy
        self.tolerance = tolerance
        self.fill_value = fill_value

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute temporal context join."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        # Prepare context data
        ctx_lf = (
            self.context_data
            if isinstance(self.context_data, pl.LazyFrame)
            else self.context_data.lazy()
        )

        # Select only needed columns and ensure proper sorting
        ctx_lf = ctx_lf.select([self.context_timestamp_col] + self.context_cols).sort(
            self.context_timestamp_col
        )

        lf = lf.sort(timestamp_col)

        # Perform join based on strategy
        if self.strategy == "exact":
            result_lf = lf.join(
                ctx_lf,
                left_on=timestamp_col,
                right_on=self.context_timestamp_col,
                how="left",
            )
        elif self.strategy == "nearest":
            result_lf = lf.join_asof(
                ctx_lf,
                left_on=timestamp_col,
                right_on=self.context_timestamp_col,
                strategy="nearest",
                tolerance=self.tolerance,
            )
        elif self.strategy == "asof_backward":
            result_lf = lf.join_asof(
                ctx_lf,
                left_on=timestamp_col,
                right_on=self.context_timestamp_col,
                strategy="backward",
                tolerance=self.tolerance,
            )
        elif self.strategy == "asof_forward":
            result_lf = lf.join_asof(
                ctx_lf,
                left_on=timestamp_col,
                right_on=self.context_timestamp_col,
                strategy="forward",
                tolerance=self.tolerance,
            )
        else:
            raise ValueError(f"Unknown join strategy: {self.strategy}")

        # Fill nulls if fill_value provided
        if self.fill_value is not None:
            for col in self.context_cols:
                result_lf = result_lf.with_columns([pl.col(col).fill_null(self.fill_value)])

        result = event_frame.with_lazy_frame(result_lf)

        # Register provenance
        for col in self.context_cols:
            provenance = FeatureProvenance(
                produced_by="TemporalContextJoinStep",
                inputs=[timestamp_col],
                tags={"context", "temporal"},
                description=f"Temporally joined context: {col}",
                metadata={"strategy": self.strategy},
            )
            result = result.register_feature(
                col,
                {"source_step": "TemporalContextJoinStep", "strategy": self.strategy},
                provenance=provenance,
            )

        return result.require_context(context_tags={"temporal_context"})


class SpatioTemporalContextJoinStep(Step):
    """Join events with spatio-temporal context data.

    Performs a two-stage join: first by spatial key, then by temporal key.

    Inputs:
        - spatial_key column
        - timestamp_col from EventSchema
        - context_data with both dimensions

    Outputs:
        - Joined columns from context_data
    """

    def __init__(
        self,
        context_data: pl.DataFrame | pl.LazyFrame,
        spatial_key: str,
        temporal_key: str,
        context_cols: Sequence[str],
        temporal_strategy: str = "asof_backward",
        fill_value: Any = None,
        tolerance: str | None = None,
    ) -> None:
        self.context_data = context_data
        self.spatial_key = spatial_key
        self.temporal_key = temporal_key
        self.context_cols = list(context_cols)
        self.temporal_strategy = temporal_strategy
        self.fill_value = fill_value
        self.tolerance = tolerance

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute spatio-temporal context join."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        # Prepare context data
        ctx_lf = (
            self.context_data
            if isinstance(self.context_data, pl.LazyFrame)
            else self.context_data.lazy()
        )

        ctx_lf = ctx_lf.select([self.spatial_key, self.temporal_key] + self.context_cols).sort(
            [self.spatial_key, self.temporal_key]
        )

        lf = lf.sort([self.spatial_key, timestamp_col])

        # For spatio-temporal join, we join by spatial key then use asof for temporal
        if self.temporal_strategy == "asof_backward":
            result_lf = lf.join_asof(
                ctx_lf,
                left_on=timestamp_col,
                right_on=self.temporal_key,
                by=self.spatial_key,
                strategy="backward",
                tolerance=self.tolerance,
            )
        elif self.temporal_strategy == "asof_forward":
            result_lf = lf.join_asof(
                ctx_lf,
                left_on=timestamp_col,
                right_on=self.temporal_key,
                by=self.spatial_key,
                strategy="forward",
                tolerance=self.tolerance,
            )
        elif self.temporal_strategy == "exact":
            result_lf = lf.join(
                ctx_lf,
                left_on=[self.spatial_key, timestamp_col],
                right_on=[self.spatial_key, self.temporal_key],
                how="left",
            )
        else:
            raise ValueError(f"Unknown temporal strategy: {self.temporal_strategy}")

        # Fill nulls if fill_value provided
        if self.fill_value is not None:
            for col in self.context_cols:
                result_lf = result_lf.with_columns([pl.col(col).fill_null(self.fill_value)])

        result = event_frame.with_lazy_frame(result_lf)

        # Register provenance
        for col in self.context_cols:
            provenance = FeatureProvenance(
                produced_by="SpatioTemporalContextJoinStep",
                inputs=[self.spatial_key, timestamp_col],
                tags={"context", "spatial", "temporal"},
                description=f"Spatio-temporally joined context: {col}",
                metadata={"temporal_strategy": self.temporal_strategy},
            )
            result = result.register_feature(
                col,
                {"source_step": "SpatioTemporalContextJoinStep"},
                provenance=provenance,
            )

        return result.require_context(context_tags={"spatiotemporal_context"})


class ContextAggregationStep(Step):
    """Aggregate context data to spatial/temporal units before joining.

    Useful for computing neighborhood-level statistics from point context.

    Inputs:
        - context_data with raw observations
        - spatial_unit or temporal_unit for aggregation

    Outputs:
        - Aggregated context columns joined to events
    """

    def __init__(
        self,
        context_data: pl.DataFrame | pl.LazyFrame,
        group_cols: Sequence[str],
        agg_specs: dict[str, str | Sequence[str]],
        join_on: Sequence[str],
        prefix: str = "ctx_",
    ) -> None:
        """
        Initialize context aggregation step.

        Args:
            context_data: Raw context data to aggregate
            group_cols: Columns to group by for aggregation
            agg_specs: Dict mapping column names to aggregation functions
                       e.g., {"temperature": ["mean", "max"], "precipitation": "sum"}
            join_on: Columns to join aggregated context to events
            prefix: Prefix for output column names
        """
        self.context_data = context_data
        self.group_cols = list(group_cols)
        self.agg_specs = agg_specs
        self.join_on = list(join_on)
        self.prefix = prefix

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute context aggregation and join."""
        lf = event_frame.lazy_frame

        # Prepare context data
        ctx_lf = (
            self.context_data
            if isinstance(self.context_data, pl.LazyFrame)
            else self.context_data.lazy()
        )

        # Build aggregation expressions
        agg_exprs: list[pl.Expr] = []
        output_cols: list[str] = []

        for col, funcs in self.agg_specs.items():
            if isinstance(funcs, str):
                funcs = [funcs]
            for func in funcs:
                output_col = f"{self.prefix}{col}_{func}"
                if func == "mean":
                    agg_exprs.append(pl.col(col).mean().alias(output_col))
                elif func == "sum":
                    agg_exprs.append(pl.col(col).sum().alias(output_col))
                elif func == "min":
                    agg_exprs.append(pl.col(col).min().alias(output_col))
                elif func == "max":
                    agg_exprs.append(pl.col(col).max().alias(output_col))
                elif func == "std":
                    agg_exprs.append(pl.col(col).std().alias(output_col))
                elif func == "count":
                    agg_exprs.append(pl.col(col).count().alias(output_col))
                output_cols.append(output_col)

        # Perform aggregation
        agg_lf = ctx_lf.group_by(self.group_cols).agg(agg_exprs)

        # Join to events
        result_lf = lf.join(agg_lf, on=self.join_on, how="left")

        result = event_frame.with_lazy_frame(result_lf)

        # Register provenance
        for col in output_cols:
            provenance = FeatureProvenance(
                produced_by="ContextAggregationStep",
                inputs=list(self.group_cols),
                tags={"context"},
                description=f"Aggregated context: {col}",
            )
            result = result.register_feature(
                col,
                {"source_step": "ContextAggregationStep"},
                provenance=provenance,
            )

        return result.require_context(context_tags={"aggregated_context"})
