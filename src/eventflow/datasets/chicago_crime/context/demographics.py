"""Demographics context source for Chicago."""

import polars as pl

from eventflow.core.context.sources import StaticSpatialSource
from eventflow.core.schema import ContextSchema


class ChicagoDemographicsSource(StaticSpatialSource):
    """
    Chicago census tract demographics.

    Provides demographic information by census tract or community area:
    - Population
    - Income levels
    - Age distribution
    - Education levels
    """

    def __init__(self, data_path: str, spatial_level: str = "community_area") -> None:
        """
        Initialize demographics source.

        Args:
            data_path: Path to demographics data file
            spatial_level: Spatial aggregation level ("census_tract" or "community_area")
        """
        super().__init__(data_path)
        self.spatial_level = spatial_level

    def load(self) -> pl.LazyFrame:
        """Load demographics data."""
        lf = pl.scan_parquet(self.data_path)

        lf = lf.select([
            pl.col(self.spatial_level).alias("zone_id"),
            pl.col("population"),
            pl.col("median_income"),
            pl.col("pct_below_poverty"),
            pl.col("pct_college_educated"),
        ])

        return lf

    @property
    def schema(self) -> ContextSchema:
        """Demographics schema."""
        return ContextSchema(
            spatial_col="zone_id",
            attribute_cols=[
                "population",
                "median_income",
                "pct_below_poverty",
                "pct_college_educated",
            ],
        )
