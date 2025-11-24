"""Baseline Chicago Crime recipe (v1)."""

from eventflow.recipes.base import BaseRecipe
from eventflow.core.pipeline import Pipeline, Step
from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import RecipeConfig
from eventflow.core import spatial, temporal, features


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
        
        class SpatialGridStep(Step):
            """Add spatial grid."""
            def __init__(self, size_m: float):
                self.size_m = size_m
            
            def run(self, ef: EventFrame) -> EventFrame:
                # Transform to projected CRS
                ef = spatial.transform_crs(ef, "EPSG:26971")
                # Create and assign grid (simplified - real implementation would be more robust)
                return ef.with_columns(grid_id=None)  # Placeholder
        
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
        
        return Pipeline([
            SpatialGridStep(size_m=grid_config.get("size_m", 300)),
            TemporalBinStep(bin_size=temporal_config.get("time_bin", "6h")),
            TemporalComponentsStep(
                components=temporal_config.get("components", ["hour_of_day", "day_of_week"])
            ),
            AggregationStep(),
        ])

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Run the recipe."""
        pipeline = self.build_pipeline()
        return pipeline.run(event_frame)
