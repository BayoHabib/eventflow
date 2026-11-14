"""Baseline Chicago Crime recipe (v1)."""

import polars as pl

from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import Pipeline, Step
from eventflow.core.registry import StepRegistry
from eventflow.core.schema import RecipeConfig
from eventflow.core.steps.spatial import AssignToGridStep, TransformCRSStep
from eventflow.core.steps.temporal import ExtractTemporalComponentsStep, TimeBinsStep
from eventflow.recipes.base import BaseRecipe


class ChicagoCountStep(Step):
    """Count events per cell/bin and retain cell centers and bin-start timestamps."""

    def __init__(self, size_m: float, origin: tuple[float, float]) -> None:
        self.size_m = size_m
        self.origin = origin

    def run(self, ef: EventFrame) -> EventFrame:
        """Aggregate events while keeping the output schema usable downstream."""
        lon, lat = ef.schema.lon_col, ef.schema.lat_col
        if lon is None or lat is None:
            raise ValueError("Chicago aggregation requires projected coordinate columns")
        lf = ef.lazy_frame.group_by(["grid_id", "time_bin"]).agg(
            pl.len().alias("count"), pl.col(lon).first(), pl.col(lat).first()
        )
        lf = lf.with_columns(
            (
                ((pl.col(lon) - self.origin[0]) / self.size_m).floor() * self.size_m
                + self.origin[0]
                + self.size_m / 2
            ).alias(lon),
            (
                ((pl.col(lat) - self.origin[1]) / self.size_m).floor() * self.size_m
                + self.origin[1]
                + self.size_m / 2
            ).alias(lat),
            pl.col("time_bin").alias(ef.schema.timestamp_col),
        ).sort(["time_bin", "grid_id"])
        result = ef.with_lazy_frame(lf).with_schema(
            numeric_cols=["count", "grid_id", lon, lat], categorical_cols=[]
        )
        return result.register_feature(
            "count", {"source_step": "ChicagoCountStep", "inputs": ["grid_id", "time_bin"]}
        )


class ChicagoCrimeV1Recipe(BaseRecipe):
    """
    Baseline recipe for Chicago Crime dataset.

    Creates spatial-temporal features without external context:
    - 300m x 300m spatial grid
    - 6-hour time bins
    - Temporal components (hour, day of week, etc.)
    - Count aggregations per grid cell and time bin
    """

    def __init__(
        self,
        config: RecipeConfig,
        *,
        step_registry: StepRegistry | None = None,
    ) -> None:
        """
        Initialize recipe.

        Args:
            config: Recipe configuration
        """
        super().__init__(config, step_registry=step_registry)
        self.name = "chicago_crime_v1"

    def build_pipeline(self) -> Pipeline:
        """Build the transformation pipeline."""
        grid_config = self.config.grid
        temporal_config = self.config.temporal

        size_m = float(grid_config.get("size_m", 300))
        origin_config = grid_config.get("origin", [0.0, 0.0])
        if len(origin_config) != 2:
            raise ValueError("grid.origin must contain x and y in projected meters")
        origin = (float(origin_config[0]), float(origin_config[1]))

        return Pipeline(
            [
                TransformCRSStep(target_crs="EPSG:26971"),
                AssignToGridStep(grid_size_m=size_m, origin=origin),
                TimeBinsStep(bin_size=temporal_config.get("time_bin", "6h")),
                ChicagoCountStep(size_m, origin),
                ExtractTemporalComponentsStep(
                    components=temporal_config.get("components", ["hour_of_day", "day_of_week"])
                ),
            ]
        )
