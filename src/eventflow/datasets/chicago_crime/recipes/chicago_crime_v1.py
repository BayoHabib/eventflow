"""Baseline Chicago Crime recipe (v1)."""

from eventflow.recipes.base import BaseRecipe
from eventflow.core.pipeline import Pipeline, Step
from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import RecipeConfig
from eventflow.core import spatial, temporal, features
import polars as pl


class ChicagoCrimeV1Recipe(BaseRecipe):
    """
    Baseline recipe for Chicago Crime dataset.

    Creates spatial-temporal features without external context:
    - 300m x 300m spatial grid
    - 6-hour time bins
    - Temporal components (hour, day of week, etc.)
    - Count aggregations per grid cell and time bin
    """

    def __init__(self, config: RecipeConfig) -> None:
        """
        Initialize recipe.

        Args:
            config: Recipe configuration
        """
        self.config = config
        self.name = "chicago_crime_v1"

    def build_pipeline(self) -> Pipeline:
        """Build the transformation pipeline."""
        grid_config = self.config.grid
        temporal_config = self.config.temporal
        grid_size = grid_config.get("size_m", 300)
        target_crs = grid_config.get("crs", "EPSG:26971")
        time_bin = temporal_config.get("time_bin", "6h")
        temporal_components = temporal_config.get("components", ["hour_of_day", "day_of_week"])

        class TimestampCastStep(Step):
            """Ensure timestamp column is datetime."""

            def run(self, ef: EventFrame) -> EventFrame:
                ts = ef.schema.timestamp_col
                lf = ef.lazy_frame.with_columns(pl.col(ts).str.to_datetime(strict=False))
                return ef.with_lazy_frame(lf)

        class SpatialGridStep(Step):
            """Add spatial grid."""

            def __init__(self, size_m: float, target_crs: str):
                self.size_m = size_m
                self.target_crs = target_crs

            def run(self, ef: EventFrame) -> EventFrame:
                # Transform to projected CRS and use projected columns for grid assignment
                ef_proj = spatial.transform_crs(ef, self.target_crs)
                lf = ef_proj.lazy_frame
                lon_proj = "longitude_proj"
                lat_proj = "latitude_proj"

                # Compute bounds from projected columns
                stats = lf.select(
                    [
                        pl.col(lon_proj).min().alias("minx"),
                        pl.col(lat_proj).min().alias("miny"),
                        pl.col(lon_proj).max().alias("maxx"),
                        pl.col(lat_proj).max().alias("maxy"),
                    ]
                ).collect().row(0, named=True)
                bounds = (stats["minx"], stats["miny"], stats["maxx"], stats["maxy"])

                grid = spatial.create_grid(bounds=bounds, size_m=self.size_m, crs=self.target_crs)
                return spatial.assign_to_grid(ef_proj, grid, lon_col=lon_proj, lat_col=lat_proj)

        class TemporalBinStep(Step):
            """Add time bins."""

            def __init__(self, bin_size: str):
                self.bin_size = bin_size

            def run(self, ef: EventFrame) -> EventFrame:
                return temporal.create_time_bins(ef, self.bin_size)

        class TemporalComponentsStep(Step):
            """Extract temporal components."""

            def __init__(self, components: list[str]):
                self.components = components

            def run(self, ef: EventFrame) -> EventFrame:
                return temporal.extract_temporal_components(ef, self.components)

        class AggregationStep(Step):
            """Aggregate counts."""

            def run(self, ef: EventFrame) -> EventFrame:
                return features.aggregate_counts(
                    ef,
                    group_by=["grid_id", "time_bin"],
                )

        return Pipeline(
            [
                TimestampCastStep(),
                SpatialGridStep(size_m=grid_size, target_crs=target_crs),
                TemporalBinStep(bin_size=time_bin),
                TemporalComponentsStep(components=temporal_components),
                AggregationStep(),
            ]
        )

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Run the recipe."""
        pipeline = self.build_pipeline()
        return pipeline.run(event_frame)
