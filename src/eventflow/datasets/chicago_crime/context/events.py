"""Special events context source for Chicago."""

import polars as pl
from eventflow.core.context.sources import StaticTemporalSource
from eventflow.core.schema import ContextSchema


class ChicagoSpecialEventsSource(StaticTemporalSource):
    """
    Chicago special events data.
    
    Provides information about major events that may impact crime patterns:
    - Sports events
    - Festivals
    - Concerts
    - Parades
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize special events source.
        
        Args:
            data_path: Path to events data file
        """
        super().__init__(data_path)

    def load(self) -> pl.LazyFrame:
        """Load special events data."""
        lf = pl.scan_parquet(self.data_path)
        
        lf = lf.select([
            pl.col("event_date").alias("timestamp"),
            pl.col("event_name"),
            pl.col("event_type"),
            pl.col("expected_attendance"),
        ])
        
        return lf

    @property
    def schema(self) -> ContextSchema:
        """Events schema."""
        return ContextSchema(
            timestamp_col="timestamp",
            attribute_cols=[
                "event_name",
                "event_type",
                "expected_attendance",
            ],
        )
