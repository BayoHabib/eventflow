"""Tests for the supported pure functions in core/features.py (phase 0).

Only ``compute_ratios`` is supported in phase 0. ``aggregate_by_category``,
``encode_categorical`` and ``normalize_features`` are EXPERIMENTAL (see the
module docstring) and deliberately have no correctness tests here — testing
them as-is would bless their known limitations.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from shikong.core.event_frame import EventFrame
from shikong.core.features import compute_ratios
from shikong.core.schema import EventMetadata, EventSchema


def _frame(numerator: list[float], denominator: list[float]) -> EventFrame:
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "latitude": [41.88, 41.89],
            "longitude": [-87.62, -87.63],
            "numerator": numerator,
            "denominator": denominator,
        }
    )
    return EventFrame(
        df.lazy(),
        EventSchema(timestamp_col="timestamp", lat_col="latitude", lon_col="longitude"),
        EventMetadata(dataset_name="ratios-test"),
    )


def test_compute_ratios_basic() -> None:
    """Ratios are computed per row with the default output name."""
    out = compute_ratios(_frame([10.0, 6.0], [2.0, 3.0]), "numerator", "denominator").collect()
    assert out["numerator_per_denominator"].to_list() == [5.0, 2.0]


def test_compute_ratios_custom_output_col() -> None:
    """A custom output column name is honoured."""
    out = compute_ratios(
        _frame([10.0, 6.0], [2.0, 3.0]), "numerator", "denominator", output_col="rate"
    ).collect()
    assert out["rate"].to_list() == [5.0, 2.0]


def test_compute_ratios_zero_denominator_is_null() -> None:
    """A zero denominator yields null — never numerator / 1e-10."""
    out = compute_ratios(_frame([10.0, 0.0], [0.0, 5.0]), "numerator", "denominator").collect()
    values = out["numerator_per_denominator"].to_list()
    assert values[0] is None
    assert values[1] == 0.0


def test_removed_functions_are_gone() -> None:
    """aggregate_counts / moving_window_aggregation / lag_features were deleted."""
    import shikong.core.features as features

    for name in ("aggregate_counts", "moving_window_aggregation", "lag_features"):
        assert not hasattr(features, name), f"{name} should have been removed in phase 0"


def test_experimental_functions_are_documented_as_such() -> None:
    """Experimental functions must say so in their docstring."""
    import shikong.core.features as features

    for name in ("aggregate_by_category", "encode_categorical", "normalize_features"):
        assert "EXPERIMENTAL" in getattr(features, name).__doc__
