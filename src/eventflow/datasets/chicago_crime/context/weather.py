"""Weather context source for Chicago."""

import polars as pl

from eventflow.core.context.sources import DynamicTemporalSource
from eventflow.core.schema import ContextSchema


class ChicagoNOAAWeatherSource(DynamicTemporalSource):
    """
    NOAA weather data for Chicago.

    Provides hourly weather observations including:
    - Temperature
    - Precipitation
    - Wind speed
    - Conditions
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize Chicago weather source.

        Args:
            data_path: Path to weather data files
        """
        super().__init__(data_path)

    def load(self) -> pl.LazyFrame:
        """Load weather data."""
        lf = pl.scan_parquet(self.data_path)

        # Standardize column names
        lf = lf.select(
            [
                pl.col("timestamp"),
                pl.col("temperature_f").alias("temperature"),
                pl.col("precipitation_in").alias("precipitation"),
                pl.col("wind_speed_mph").alias("wind_speed"),
                pl.col("conditions").alias("weather_condition"),
            ]
        )

        return lf

    @property
    def schema(self) -> ContextSchema:
        """Weather schema."""
        return ContextSchema(
            timestamp_col="timestamp",
            attribute_cols=[
                "temperature",
                "precipitation",
                "wind_speed",
                "weather_condition",
            ],
        )
