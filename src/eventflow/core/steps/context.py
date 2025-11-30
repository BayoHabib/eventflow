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

import polars as pl
from pydantic import BaseModel, Field
from pyproj import Transformer
from shapely import STRtree, geometry
from shapely.wkt import loads as wkt_loads

from eventflow.core.pipeline import Step
from eventflow.core.schema import FeatureProvenance
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

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
        default=None, description="Maximum distance for nearest neighbor join"
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

        # Collect context data if lazy
        ctx_df = (
            self.context_data.collect()
            if isinstance(self.context_data, pl.LazyFrame)
            else self.context_data
        )

        # Handle CRS transformation if needed
        transformer: Transformer | None = None
        if self.context_crs and self.context_crs != event_crs:
            logger.info(f"Transforming coordinates from {event_crs} to {self.context_crs}")
            transformer = Transformer.from_crs(event_crs, self.context_crs, always_xy=True)

        # Build spatial index from context geometries
        geoms = [wkt_loads(wkt) for wkt in ctx_df[self.context_spatial_col]]
        tree = STRtree(geoms)

        # Build lookup dictionary for context attributes
        ctx_records = ctx_df.select([self.context_spatial_col] + self.context_cols).to_dicts()
        geom_to_attrs = {}
        for i, rec in enumerate(ctx_records):
            geom_to_attrs[i] = {k: rec[k] for k in self.context_cols}

        def find_context(lon: float, lat: float) -> dict[str, Any]:
            """Find matching context for a point."""
            # Transform coordinates if needed
            if transformer:
                lon, lat = transformer.transform(lon, lat)

            point = geometry.Point(lon, lat)

            if self.join_type == "contains":
                idx = tree.query(point, predicate="contains")
            elif self.join_type == "intersects":
                idx = tree.query(point, predicate="intersects")
            elif self.join_type == "nearest":
                idx = tree.nearest(point)
                if self.max_distance is not None:
                    nearest_geom = geoms[idx]
                    if point.distance(nearest_geom) > self.max_distance:
                        return dict.fromkeys(self.context_cols, self.fill_value)
                return geom_to_attrs.get(idx, dict.fromkeys(self.context_cols, self.fill_value))
            else:
                raise ValueError(f"Unknown join type: {self.join_type}")

            if len(idx) > 0:
                return geom_to_attrs.get(
                    int(idx[0]), dict.fromkeys(self.context_cols, self.fill_value)
                )
            return dict.fromkeys(self.context_cols, self.fill_value)

        # Apply join
        lf = event_frame.lazy_frame

        for col in self.context_cols:
            col_name = col  # Capture in local variable for lambda

            def get_col_value(coord: dict[str, float], c: str = col_name) -> Any:
                return find_context(coord[lon_col], coord[lat_col]).get(c, self.fill_value)

            lf = lf.with_columns(
                [
                    pl.struct([lon_col, lat_col])
                    .map_elements(
                        get_col_value,
                        return_dtype=pl.Object,
                    )
                    .alias(col)
                ]
            )

        result = event_frame.with_lazy_frame(lf)

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
            )
        elif self.strategy == "asof_backward":
            result_lf = lf.join_asof(
                ctx_lf,
                left_on=timestamp_col,
                right_on=self.context_timestamp_col,
                strategy="backward",
            )
        elif self.strategy == "asof_forward":
            result_lf = lf.join_asof(
                ctx_lf,
                left_on=timestamp_col,
                right_on=self.context_timestamp_col,
                strategy="forward",
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
    ) -> None:
        self.context_data = context_data
        self.spatial_key = spatial_key
        self.temporal_key = temporal_key
        self.context_cols = list(context_cols)
        self.temporal_strategy = temporal_strategy
        self.fill_value = fill_value

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
            )
        elif self.temporal_strategy == "asof_forward":
            result_lf = lf.join_asof(
                ctx_lf,
                left_on=timestamp_col,
                right_on=self.temporal_key,
                by=self.spatial_key,
                strategy="forward",
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
