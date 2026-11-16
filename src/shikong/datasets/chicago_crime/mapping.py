"""Data loading and mapping for Chicago Crime dataset."""

from pathlib import Path

import polars as pl

from shikong.core.event_frame import EventFrame
from shikong.core.schema import DatasetConfig
from shikong.core.spatial import transform_crs
from shikong.core.utils import get_logger
from shikong.datasets.chicago_crime.schema import CHICAGO_CRIME_SCHEMA, create_chicago_metadata

logger = get_logger(__name__)


def load_raw_chicago(raw_path: str, layout: str = "nested") -> pl.LazyFrame:
    """
    Load one file or a directory of Chicago crime CSV or Parquet chunks.

    Args:
        raw_path: Root path to raw data
        layout: Directory layout ("flat" or "nested")

    Returns:
        LazyFrame with raw data
    """
    if layout not in {"nested", "flat"}:
        raise ValueError("layout must be 'nested' or 'flat'")
    root = Path(raw_path)
    if root.is_file():
        files = [root]
    else:
        candidates = root.rglob("*") if layout == "nested" else root.glob("*")
        files = sorted(
            p for p in candidates if p.is_file() and p.suffix.lower() in {".csv", ".parquet"}
        )
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet data files found under {raw_path}")
    formats = {p.suffix.lower() for p in files}
    if len(formats) != 1 or not formats <= {".csv", ".parquet"}:
        raise ValueError("Select input files containing only CSV or only Parquet chunks")

    logger.info("Loading %d Chicago crime chunk(s) from %s", len(files), raw_path)
    frames = [
        pl.scan_parquet(p) if p.suffix.lower() == ".parquet" else pl.scan_csv(p) for p in files
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def clean_chicago_data(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply basic cleaning to Chicago crime data.

    Args:
        lf: Raw LazyFrame

    Returns:
        Cleaned LazyFrame
    """
    logger.info("Applying data cleaning to Chicago crime data")
    schema = lf.collect_schema()
    required = {"date", "latitude", "longitude"}
    if missing := required - set(schema.names()):
        raise ValueError(f"Missing required Chicago crime columns: {sorted(missing)}")

    if schema["date"] == pl.String:
        date_expr = pl.coalesce(
            pl.col("date").str.to_datetime(fmt, strict=False)
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%.f",
                "%Y-%m-%d %H:%M:%S%.f",
                "%m/%d/%Y %I:%M:%S %p",
            )
        )
    elif schema["date"] == pl.Date or schema["date"] == pl.Datetime:
        date_expr = pl.col("date").cast(pl.Datetime)
    else:
        raise ValueError("Chicago crime dates must be strings, dates, or datetimes")

    lf = lf.with_columns(
        date_expr.alias("date"),
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
    )

    # Filter out rows with missing coordinates
    lf = lf.filter(pl.col("date").is_not_null())

    # Filter out invalid coordinates
    lf = lf.filter(
        pl.col("latitude").is_finite()
        & pl.col("longitude").is_finite()
        & pl.col("latitude").is_between(-90, 90)
        & pl.col("longitude").is_between(-180, 180)
    )

    # Convert boolean-like columns
    for name in ("arrest", "domestic"):
        if name in schema and schema[name] != pl.Boolean:
            text = pl.col(name).cast(pl.String).str.to_lowercase()
            lf = lf.with_columns(
                pl.when(text.is_in(["true", "1"]))
                .then(True)
                .when(text.is_in(["false", "0"]))
                .then(False)
                .otherwise(None)
                .cast(pl.Boolean)
                .alias(name)
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
        crs="EPSG:4326",  # Socrata latitude/longitude are geographic input coordinates.
        time_zone=time_zone,
    )

    # Create EventFrame
    event_frame = EventFrame(lf, CHICAGO_CRIME_SCHEMA, metadata)
    return transform_crs(event_frame, crs)


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
