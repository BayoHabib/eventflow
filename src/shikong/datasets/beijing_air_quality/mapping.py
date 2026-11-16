"""Data loading and mapping for the Beijing Multi-Site Air Quality dataset (UCI 501).

Source: Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G

The download (``PRSA2017_Data_20130301-20170228.zip``) contains one
CSV per station, ``PRSA_Data_<Station>_20130301-20170228.csv``, with
columns ``No, year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3,
TEMP, PRES, DEWP, RAIN, wd, WSPM, station``. Missing values are the
literal string ``NA``. The loader also accepts a single Parquet file
(the concatenated raw frame) for replay convenience.
"""

from pathlib import Path

import polars as pl

from shikong.core.event_frame import EventFrame
from shikong.core.schema import DatasetConfig
from shikong.core.utils import get_logger
from shikong.datasets.beijing_air_quality.schema import (
    BEIJING_AIR_QUALITY_SCHEMA,
    MEASUREMENT_COLS,
    STATION_COORDS,
    create_beijing_metadata,
)

logger = get_logger(__name__)

# Exact dtypes for the per-station CSVs. RAIN must be Float64: the
# first rows are integers, so schema inference alone reads it as Int64
# and then fails on the first "0.4".
_BEIJING_CSV_SCHEMA = {
    "No": pl.Int64,
    "year": pl.Int64,
    "month": pl.Int64,
    "day": pl.Int64,
    "hour": pl.Int64,
    "PM2.5": pl.Float64,
    "PM10": pl.Float64,
    "SO2": pl.Float64,
    "NO2": pl.Float64,
    "CO": pl.Float64,
    "O3": pl.Float64,
    "TEMP": pl.Float64,
    "PRES": pl.Float64,
    "DEWP": pl.Float64,
    "RAIN": pl.Float64,
    "wd": pl.String,
    "WSPM": pl.Float64,
    "station": pl.String,
}


def _station_from_filename(path: Path) -> str:
    """Derive the station name from a PRSA_Data_<Station>_*.csv filename."""
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[0] != "PRSA" or parts[1] != "Data":
        raise ValueError(f"Cannot derive a station name from {path.name!r}")
    return parts[2]


def load_raw_beijing(raw_path: str) -> pl.LazyFrame:
    """Load the Beijing CSVs (or a single concatenated Parquet) lazily.

    Args:
        raw_path: Directory of per-station CSVs, one CSV file, or one
            Parquet file holding the concatenated raw frame.

    Returns:
        LazyFrame with the raw columns (``station`` included).
    """
    root = Path(raw_path)
    if root.is_file():
        if root.suffix.lower() == ".parquet":
            logger.info("Loading Beijing parquet from %s", raw_path)
            return pl.scan_parquet(root)
        if root.suffix.lower() == ".csv":
            files = [root]
        else:
            raise ValueError(f"Unsupported Beijing input file: {root.suffix!r}")
    elif root.is_dir():
        files = sorted(p for p in root.glob("PRSA_Data_*.csv") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No PRSA_Data_*.csv files found under {raw_path}")
    else:
        raise FileNotFoundError(f"Beijing data path does not exist: {raw_path}")

    logger.info("Loading %d Beijing station CSV(s) from %s", len(files), raw_path)
    frames = []
    for path in files:
        station = _station_from_filename(path)
        if station not in STATION_COORDS:
            raise ValueError(f"Unknown Beijing station {station!r} in {path.name!r}")
        frame = pl.scan_csv(path, schema_overrides=_BEIJING_CSV_SCHEMA, null_values=["NA"])
        # The per-station files already carry a station column; keep it
        # and check it agrees with the filename instead of trusting either.
        frames.append(frame.with_columns(pl.lit(station).alias("_station_from_file")))
    combined = pl.concat(frames, how="diagonal_relaxed")
    return combined


def clean_beijing_data(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Clean raw Beijing data into one row per (station, hour).

    - Builds the ``timestamp`` from year/month/day/hour (or keeps an
      existing ``timestamp`` column, e.g. from a Parquet replay input).
    - Attaches each station's published latitude/longitude.
    - Adds ``count`` = 1: one row is one observed hourly station
      record, and the count is the only column the panel completion
      ever zero-fills.
    - Drops rows with a null timestamp or station (unbinnable).

    Measurement nulls are preserved exactly as read: a missing PM2.5
    is a missing reading, not a zero.
    """
    logger.info("Applying data cleaning to Beijing air quality data")
    schema = lf.collect_schema()
    if "station" not in schema:
        raise ValueError("Beijing data is missing the 'station' column")
    if "_station_from_file" in schema:
        mismatch = lf.filter(pl.col("station") != pl.col("_station_from_file"))
        if mismatch.select(pl.len()).collect().item() > 0:
            raise ValueError("Station column disagrees with the source filename")
        lf = lf.drop("_station_from_file")

    if "timestamp" in schema:
        timestamp_expr = pl.col("timestamp").cast(pl.Datetime)
    else:
        required = {"year", "month", "day", "hour"}
        if missing := required - set(schema.names()):
            raise ValueError(f"Missing required Beijing time columns: {sorted(missing)}")
        timestamp_expr = pl.datetime("year", "month", "day", "hour")

    for col in MEASUREMENT_COLS:
        if col not in schema:
            raise ValueError(f"Missing required Beijing measurement column: {col!r}")

    lf = lf.with_columns(
        timestamp_expr.alias("timestamp"),
        pl.lit(1).alias("count"),
    )

    lf = lf.filter(pl.col("timestamp").is_not_null() & pl.col("station").is_not_null())

    # Attach each station's published coordinates (schema.py) and
    # reject an unknown station instead of silently leaving nulls.
    known = lf.select("station").unique().collect()["station"].to_list()
    if unknown := sorted(set(known) - set(STATION_COORDS)):
        raise ValueError(f"Unknown Beijing station(s): {unknown}")
    coords = pl.DataFrame(
        {
            "station": list(STATION_COORDS),
            "longitude": [v[0] for v in STATION_COORDS.values()],
            "latitude": [v[1] for v in STATION_COORDS.values()],
        }
    ).lazy()
    lf = lf.join(coords, on="station", how="left")

    # One row per (station, hour): the UCI files are already deduplicated,
    # so a duplicate here means a corrupt or double-loaded input.
    dupes = (
        lf.group_by(["station", "timestamp"])
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    if dupes:
        raise ValueError(f"Duplicate (station, hour) pairs in Beijing input: {dupes}")

    logger.info("Data cleaning completed")
    return lf


def load_beijing_air_quality(
    raw_path: str,
    config: DatasetConfig | None = None,
    apply_cleaning: bool = True,
) -> EventFrame:
    """Load Beijing air quality data as an EventFrame.

    Args:
        raw_path: Directory of per-station CSVs, one CSV, or one
            Parquet file (see :func:`load_raw_beijing`).
        config: Optional dataset configuration (only ``raw_root`` and
            ``time_zone`` are honoured).
        apply_cleaning: Whether to apply data cleaning.

    Returns:
        EventFrame with the Beijing air quality data.
    """
    if config is not None:
        raw_path = config.raw_root
        time_zone = config.time_zone
    else:
        time_zone = "Asia/Shanghai"

    lf = load_raw_beijing(raw_path)
    if apply_cleaning:
        lf = clean_beijing_data(lf)

    metadata = create_beijing_metadata(
        crs="EPSG:4326",  # Station coordinates are geographic (lon/lat).
        time_zone=time_zone,
    )
    return EventFrame(lf, BEIJING_AIR_QUALITY_SCHEMA, metadata)
