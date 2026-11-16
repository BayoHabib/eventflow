"""Feature engineering utilities.

Phase-0 status (2026-11-15) — this module is PARTLY EXPERIMENTAL:

- SUPPORTED (tested): ``compute_ratios``. One pure function, no tested
  Step equivalent, covered by ``tests/core/test_features_utils.py``.
- EXPERIMENTAL (untested, use at your own risk): ``aggregate_by_category``,
  ``encode_categorical``, ``normalize_features``. They are generic and have
  no exact tested Step equivalent, but each has a known limitation
  documented on the function (eager category collection, no fit/transform
  split, stale output schema — see below). They must not be presented as
  production-ready, and promoting one requires fixing its documented
  limitation first and adding tests.
- REMOVED in phase 0: ``aggregate_counts``, ``moving_window_aggregation``
  and ``lag_features``. All three were superseded by tested Steps
  (``GridAggregationStep`` / ``ChicagoCountStep``,
  ``MovingAverageStep``, ``TemporalLagStep``); ``moving_window_aggregation``
  was additionally broken (it ignored ``group_by`` and raised on duration
  windows) and ``lag_features`` silently allowed negative lags (lookahead
  leakage). Use the Steps instead.

Known limitation shared by every function here: results are returned via
``EventFrame.with_lazy_frame()``, which keeps the input schema — the
returned frame's schema does not describe the new columns. Tested Steps
register provenance instead; prefer them.
"""

import polars as pl

from shikong.core.event_frame import EventFrame
from shikong.core.utils import get_logger

logger = get_logger(__name__)


def aggregate_by_category(
    event_frame: EventFrame,
    group_by: list[str],
    category_col: str,
    prefix: str = "count",
) -> EventFrame:
    """
    Compute counts per category within groups.

    EXPERIMENTAL (phase 0): no tested Step does a per-category pivot, but
    this function eagerly collects the distinct categories (it does not
    stay lazy) and does not escape category values in the generated column
    names. Do not use it in production without fixing those two points.

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


def encode_categorical(
    event_frame: EventFrame,
    col: str,
    method: str = "onehot",
    prefix: str | None = None,
) -> EventFrame:
    """
    Encode categorical variables.

    EXPERIMENTAL (phase 0): one-hot column order is now sorted (fixed in
    phase 0), but ordinal encoding still uses a Python ``map_elements``
    UDF (slow, breaks lazy/streaming) and nulls are not handled
    explicitly. There is no ``"target"`` encoding method — the docstring
    used to promise one, it was removed in phase 0 (target encoding must
    be out-of-fold to avoid leakage, see the literature review).

    Args:
        event_frame: Input EventFrame
        col: Column to encode
        method: Encoding method:
            - "onehot": One-hot encoding
            - "ordinal": Ordinal encoding (0, 1, 2, ...)
        prefix: Prefix for output columns

    Returns:
        EventFrame with encoded features
    """
    if prefix is None:
        prefix = col

    lf = event_frame.lazy_frame

    logger.info(f"Encoding categorical column '{col}' using method '{method}'")
    if method == "onehot":
        # Get unique values (sorted, so column order is deterministic)
        unique_vals = lf.select(pl.col(col).unique().sort()).collect()[col].to_list()

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

    Supported in phase 0 (tested in ``tests/core/test_features_utils.py``).
    A zero denominator yields null, never an explosion: the previous
    implementation clipped the denominator to 1e-10, so a zero denominator
    produced a ratio of ``numerator / 1e-10``.

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
        [
            pl.when(pl.col(denominator_col) == 0)
            .then(None)
            .otherwise(pl.col(numerator_col) / pl.col(denominator_col))
            .alias(output_col)
        ]
    )

    return event_frame.with_lazy_frame(lf)


def normalize_features(
    event_frame: EventFrame,
    cols: list[str],
    method: str = "zscore",
) -> EventFrame:
    """
    Normalize numeric features.

    EXPERIMENTAL (phase 0): statistics (mean/std/min/max) are computed
    over the WHOLE frame, so future and test rows leak into training
    features. There is no fit/transform split. Do not use this on a
    train/test split without fitting the statistics on the training part
    only. Division by a zero standard deviation (or an empty min-max
    range) is also unguarded. Promoting this function requires a fitted
    state object first.

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
