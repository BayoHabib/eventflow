"""Feature engineering utilities."""

from typing import Any, cast

import polars as pl

from eventflow.core.event_frame import EventFrame
from eventflow.core.utils import get_logger

logger = get_logger(__name__)


def aggregate_counts(
    event_frame: EventFrame,
    group_by: list[str],
    count_col: str = "count",
) -> EventFrame:
    """
    Compute event counts per group.

    Args:
        event_frame: Input EventFrame
        group_by: Columns to group by (e.g., ["grid_id", "time_bin"])
        count_col: Name of count column

    Returns:
        EventFrame with aggregated counts
    """
    logger.info(f"Aggregating counts by: {group_by}")
    lf = event_frame.lazy_frame.group_by(group_by).agg([pl.len().alias(count_col)])
    logger.debug(f"Created count column: {count_col}")

    return event_frame.with_lazy_frame(lf)


def aggregate_by_category(
    event_frame: EventFrame,
    group_by: list[str],
    category_col: str,
    prefix: str = "count",
) -> EventFrame:
    """
    Compute counts per category within groups.

    Args:
        event_frame: Input EventFrame
        group_by: Columns to group by
        category_col: Category column to count
        prefix: Prefix for output columns

    Returns:
        EventFrame with category counts
    """
    categories_df = event_frame.lazy_frame.select(
        pl.col(category_col).drop_nulls().unique().sort()
    ).collect()
    categories = categories_df[category_col].to_list()

    if not categories:
        logger.info("No categories found for column '%s'", category_col)
        total_counts = event_frame.lazy_frame.group_by(group_by).agg(
            [pl.len().alias(f"{prefix}_total")]
        )
        return event_frame.with_lazy_frame(total_counts)

    indicator_columns: list[str] = []
    exprs = []
    for value in categories:
        col_name = f"{prefix}_{value}"
        indicator_columns.append(col_name)
        exprs.append(
            pl.when(pl.col(category_col) == value)
            .then(1)
            .otherwise(0)
            .cast(pl.Int64)
            .alias(col_name)
        )

    enriched = event_frame.lazy_frame.with_columns(exprs)
    agg_exprs = [pl.col(name).sum().alias(name) for name in indicator_columns]
    lf = enriched.group_by(group_by).agg(agg_exprs)

    return event_frame.with_lazy_frame(lf)


def moving_window_aggregation(
    event_frame: EventFrame,
    window_size: str,
    group_by: list[str],
    agg_col: str,
    agg_fn: str = "mean",
    output_col: str | None = None,
) -> EventFrame:
    """
    Compute moving window aggregations.

    Args:
        event_frame: Input EventFrame
        window_size: Window size (e.g., "7d", "1h")
        group_by: Columns to group by (typically includes time_bin)
        agg_col: Column to aggregate
        agg_fn: Aggregation function ("mean", "sum", "min", "max", "std")
        output_col: Name of output column

    Returns:
        EventFrame with moving window features
    """
    if output_col is None:
        output_col = f"{agg_col}_{agg_fn}_{window_size}"

    timestamp_col = event_frame.schema.timestamp_col

    # Sort by timestamp for window operations
    lf = event_frame.lazy_frame.sort(timestamp_col)

    # Apply rolling aggregation
    # NOTE: Polars supports string durations for rolling windows at runtime, but
    # the current type stubs only accept integers. Cast to Any to avoid false
    # positives while keeping the expressive API.
    column_expr = cast(Any, pl.col(agg_col))

    if agg_fn == "mean":
        agg_expr = column_expr.rolling_mean(window_size)
    elif agg_fn == "sum":
        agg_expr = column_expr.rolling_sum(window_size)
    elif agg_fn == "min":
        agg_expr = column_expr.rolling_min(window_size)
    elif agg_fn == "max":
        agg_expr = column_expr.rolling_max(window_size)
    elif agg_fn == "std":
        agg_expr = column_expr.rolling_std(window_size)
    else:
        raise ValueError(f"Unknown aggregation function: {agg_fn}")

    lf = lf.with_columns([agg_expr.alias(output_col)])

    return event_frame.with_lazy_frame(lf)


def lag_features(
    event_frame: EventFrame,
    lag_col: str,
    lags: list[int],
    group_by: list[str] | None = None,
) -> EventFrame:
    """
    Create lag features.

    Args:
        event_frame: Input EventFrame
        lag_col: Column to create lags for
        lags: List of lag periods (e.g., [1, 7, 30])
        group_by: Optional grouping columns

    Returns:
        EventFrame with lag features
    """
    lf = event_frame.lazy_frame.sort(event_frame.schema.timestamp_col)

    exprs = []
    for lag in lags:
        if group_by:
            expr = pl.col(lag_col).shift(lag).over(group_by).alias(f"{lag_col}_lag_{lag}")
        else:
            expr = pl.col(lag_col).shift(lag).alias(f"{lag_col}_lag_{lag}")
        exprs.append(expr)

    lf = lf.with_columns(exprs)

    return event_frame.with_lazy_frame(lf)


def encode_categorical(
    event_frame: EventFrame,
    col: str,
    method: str = "onehot",
    prefix: str | None = None,
) -> EventFrame:
    """
    Encode categorical variables.

    Args:
        event_frame: Input EventFrame
        col: Column to encode
        method: Encoding method:
            - "onehot": One-hot encoding
            - "ordinal": Ordinal encoding (0, 1, 2, ...)
            - "target": Target encoding (requires target column)
        prefix: Prefix for output columns

    Returns:
        EventFrame with encoded features
    """
    if prefix is None:
        prefix = col

    lf = event_frame.lazy_frame

    logger.info(f"Encoding categorical column '{col}' using method '{method}'")
    if method == "onehot":
        # Get unique values
        unique_vals = lf.select(pl.col(col).unique()).collect()[col].to_list()

        # Create binary columns
        exprs = [
            (pl.col(col) == val).cast(pl.Int32).alias(f"{prefix}_{val}") for val in unique_vals
        ]
        lf = lf.with_columns(exprs)

    elif method == "ordinal":
        # Create ordinal encoding
        unique_vals = lf.select(pl.col(col).unique().sort()).collect()[col].to_list()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}

        lf = lf.with_columns(
            [
                pl.col(col)
                .map_elements(lambda value: mapping.get(value), return_dtype=pl.Int64)
                .alias(f"{col}_encoded")
            ]
        )

    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return event_frame.with_lazy_frame(lf)


def compute_ratios(
    event_frame: EventFrame,
    numerator_col: str,
    denominator_col: str,
    output_col: str | None = None,
) -> EventFrame:
    """
    Compute ratios between columns.

    Args:
        event_frame: Input EventFrame
        numerator_col: Numerator column
        denominator_col: Denominator column
        output_col: Name of output column

    Returns:
        EventFrame with ratio column
    """
    if output_col is None:
        output_col = f"{numerator_col}_per_{denominator_col}"

    lf = event_frame.lazy_frame.with_columns(
        [(pl.col(numerator_col) / pl.col(denominator_col).clip(1e-10)).alias(output_col)]
    )

    return event_frame.with_lazy_frame(lf)


def normalize_features(
    event_frame: EventFrame,
    cols: list[str],
    method: str = "zscore",
) -> EventFrame:
    """
    Normalize numeric features.

    Args:
        event_frame: Input EventFrame
        cols: Columns to normalize
        method: Normalization method:
            - "zscore": Z-score normalization (mean=0, std=1)
            - "minmax": Min-max normalization (0-1)

    Returns:
        EventFrame with normalized features
    """
    lf = event_frame.lazy_frame

    if method == "zscore":
        exprs = [
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(f"{col}_normalized")
            for col in cols
        ]
    elif method == "minmax":
        exprs = [
            ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())).alias(
                f"{col}_normalized"
            )
            for col in cols
        ]
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    lf = lf.with_columns(exprs)

    return event_frame.with_lazy_frame(lf)
