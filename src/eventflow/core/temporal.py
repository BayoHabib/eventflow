"""Temporal operations for event data."""

import polars as pl

from eventflow.core.event_frame import EventFrame
from eventflow.core.utils import get_logger

logger = get_logger(__name__)


def extract_temporal_components(
    event_frame: EventFrame,
    components: list[str],
) -> EventFrame:
    """
    Extract temporal components from timestamp.

    Args:
        event_frame: Input EventFrame
        components: List of components to extract:
            - "hour_of_day": Hour (0-23)
            - "day_of_week": Day of week (0=Monday, 6=Sunday)
            - "day_of_month": Day of month (1-31)
            - "day_of_year": Day of year (1-366)
            - "month": Month (1-12)
            - "year": Year
            - "is_weekend": Boolean for weekend
            - "is_holiday": Boolean for holiday (requires holiday calendar)

    Returns:
        EventFrame with temporal component columns added
    """
    timestamp_col = event_frame.schema.timestamp_col
    lf = event_frame.lazy_frame

    logger.info(f"Extracting temporal components: {components}")
    exprs = []

    if "hour_of_day" in components:
        exprs.append(pl.col(timestamp_col).dt.hour().alias("hour_of_day"))

    if "day_of_week" in components:
        exprs.append(pl.col(timestamp_col).dt.weekday().alias("day_of_week"))

    if "day_of_month" in components:
        exprs.append(pl.col(timestamp_col).dt.day().alias("day_of_month"))

    if "day_of_year" in components:
        exprs.append(pl.col(timestamp_col).dt.ordinal_day().alias("day_of_year"))

    if "month" in components:
        exprs.append(pl.col(timestamp_col).dt.month().alias("month"))

    if "year" in components:
        exprs.append(pl.col(timestamp_col).dt.year().alias("year"))

    if "is_weekend" in components:
        exprs.append(
            (pl.col(timestamp_col).dt.weekday().is_in([5, 6])).alias("is_weekend")
        )

    if exprs:
        lf = lf.with_columns(exprs)

    return event_frame.with_lazy_frame(lf)


def create_time_bins(
    event_frame: EventFrame,
    bin_size: str,
    bin_col: str = "time_bin",
) -> EventFrame:
    """
    Create time bins for temporal aggregation.

    Args:
        event_frame: Input EventFrame
        bin_size: Time bin size (e.g., "1h", "6h", "1d")
        bin_col: Name of the bin column to create

    Returns:
        EventFrame with time bin column added
    """
    timestamp_col = event_frame.schema.timestamp_col

    logger.info(f"Creating time bins with size: {bin_size}")
    lf = event_frame.lazy_frame.with_columns([
        pl.col(timestamp_col).dt.truncate(bin_size).alias(bin_col)
    ])

    return event_frame.with_lazy_frame(lf).with_metadata(time_bin=bin_size)


def align_temporal(
    event_frame: EventFrame,
    context_frame: pl.LazyFrame,
    context_timestamp_col: str,
    strategy: str = "nearest",
    window: str | None = None,
) -> EventFrame:
    """
    Align events with context data temporally.

    Args:
        event_frame: Input EventFrame
        context_frame: Context data LazyFrame
        context_timestamp_col: Timestamp column in context data
        strategy: Alignment strategy:
            - "exact": Exact timestamp match
            - "nearest": Nearest timestamp
            - "before": Most recent timestamp before event
            - "after": Next timestamp after event
        window: Time window for matching (e.g., "1h")

    Returns:
        EventFrame with context columns joined
    """
    event_timestamp_col = event_frame.schema.timestamp_col

    if strategy == "exact":
        # Exact join on timestamp
        lf = event_frame.lazy_frame.join(
            context_frame,
            left_on=event_timestamp_col,
            right_on=context_timestamp_col,
            how="left",
        )

    elif strategy == "nearest":
        # This is a simplified implementation
        # In production, use more efficient temporal join algorithms
        lf = event_frame.lazy_frame.join_asof(
            context_frame,
            left_on=event_timestamp_col,
            right_on=context_timestamp_col,
            strategy="nearest",
        )

    elif strategy == "before":
        lf = event_frame.lazy_frame.join_asof(
            context_frame,
            left_on=event_timestamp_col,
            right_on=context_timestamp_col,
            strategy="backward",
        )

    elif strategy == "after":
        lf = event_frame.lazy_frame.join_asof(
            context_frame,
            left_on=event_timestamp_col,
            right_on=context_timestamp_col,
            strategy="forward",
        )

    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")

    return event_frame.with_lazy_frame(lf)


def compute_time_deltas(
    event_frame: EventFrame,
    reference_timestamps: pl.LazyFrame,
    reference_timestamp_col: str,
    delta_col: str = "time_delta",
) -> EventFrame:
    """
    Compute time deltas between events and reference timestamps.

    Args:
        event_frame: Input EventFrame
        reference_timestamps: Reference timestamp data
        reference_timestamp_col: Timestamp column in reference data
        delta_col: Name of delta column to create

    Returns:
        EventFrame with time delta column
    """
    event_timestamp_col = event_frame.schema.timestamp_col

    lf = event_frame.lazy_frame.join_asof(
        reference_timestamps,
        left_on=event_timestamp_col,
        right_on=reference_timestamp_col,
        strategy="nearest",
    ).with_columns([
        (pl.col(event_timestamp_col) - pl.col(reference_timestamp_col))
        .dt.total_seconds()
        .alias(delta_col)
    ])

    return event_frame.with_lazy_frame(lf)


def create_temporal_windows(
    event_frame: EventFrame,
    window_sizes: list[str],
) -> EventFrame:
    """
    Create temporal window identifiers for aggregation.

    Args:
        event_frame: Input EventFrame
        window_sizes: List of window sizes (e.g., ["1d", "7d", "30d"])

    Returns:
        EventFrame with window columns added
    """
    timestamp_col = event_frame.schema.timestamp_col
    lf = event_frame.lazy_frame

    for window in window_sizes:
        lf = lf.with_columns([
            pl.col(timestamp_col).dt.truncate(window).alias(f"window_{window}")
        ])

    return event_frame.with_lazy_frame(lf)
