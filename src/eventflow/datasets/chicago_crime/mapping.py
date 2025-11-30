"""Data loading and mapping for Chicago Crime dataset."""

import polars as pl

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import DatasetConfig
from eventflow.core.utils import get_logger
from eventflow.datasets.chicago_crime.schema import CHICAGO_CRIME_SCHEMA, create_chicago_metadata

logger = get_logger(__name__)


def load_raw_chicago(raw_path: str, layout: str = "nested") -> pl.LazyFrame:
    """
    Load raw Chicago crime data from Parquet files.

    Args:
        raw_path: Root path to raw data
        layout: Directory layout ("flat" or "nested")

    Returns:
        LazyFrame with raw data
    """
    if layout == "nested":
        # Assumes year/month partitioned structure
        pattern = f"{raw_path}/**/*.parquet"
    else:
        # Flat structure
        pattern = f"{raw_path}/*.parquet"

    logger.info(f"Loading Chicago crime data from: {pattern}")
    lf = pl.scan_parquet(pattern)
    logger.debug(f"LazyFrame created with columns: {lf.columns}")

    return lf


def clean_chicago_data(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply basic cleaning to Chicago crime data.

    Args:
        lf: Raw LazyFrame

    Returns:
        Cleaned LazyFrame
    """
    logger.info("Applying data cleaning to Chicago crime data")
    # Parse date column
    lf = lf.with_columns([pl.col("date").str.to_datetime("%m/%d/%Y %I:%M:%S %p").alias("date")])

    # Filter out rows with missing coordinates
    lf = lf.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())

    # Filter out invalid coordinates
    lf = lf.filter(
        (pl.col("latitude").is_between(-90, 90)) & (pl.col("longitude").is_between(-180, 180))
    )

    # Convert boolean-like columns
    lf = lf.with_columns(
        [
            (pl.col("arrest") == "true").alias("arrest"),
            (pl.col("domestic") == "true").alias("domestic"),
        ]
    )

    logger.info("Data cleaning completed")
    return lf


def load_chicago_crime(
    raw_path: str,
    config: DatasetConfig | None = None,
    apply_cleaning: bool = True,
) -> EventFrame:
    """
    Load Chicago crime data as an EventFrame.

    Args:
        raw_path: Path to raw data files
        config: Optional dataset configuration
        apply_cleaning: Whether to apply data cleaning

    Returns:
        EventFrame with Chicago crime data
    """
    # Use config if provided
    if config is not None:
        raw_path = config.raw_root
        layout = config.layout
        crs = config.crs
        time_zone = config.time_zone
    else:
        layout = "nested"
        crs = "EPSG:4326"
        time_zone = "America/Chicago"

    # Load raw data
    lf = load_raw_chicago(raw_path, layout=layout)

    # Apply cleaning if requested
    if apply_cleaning:
        lf = clean_chicago_data(lf)

    # Create metadata
    metadata = create_chicago_metadata(
        crs=crs,
        time_zone=time_zone,
    )

    # Create EventFrame
    return EventFrame(lf, CHICAGO_CRIME_SCHEMA, metadata)


def sample_chicago_data(
    event_frame: EventFrame,
    n: int | None = None,
    fraction: float | None = None,
) -> EventFrame:
    """
    Sample from Chicago crime data for testing.

    Args:
        event_frame: Input EventFrame
        n: Number of rows to sample (exact)
        fraction: Fraction of rows to sample (0-1)

    Returns:
        Sampled EventFrame
    """
    df = event_frame.collect()

    if n is not None:
        sampled = df.sample(n=n)
    elif fraction is not None:
        sampled = df.sample(fraction=fraction)
    else:
        raise ValueError("Must specify either n or fraction")

    return event_frame.with_lazy_frame(sampled.lazy())
