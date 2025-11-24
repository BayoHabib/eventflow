"""Tests for utility helpers."""

import pytest

from eventflow.core.utils import (
    get_logger,
    is_numeric_col,
    is_categorical_col,
    is_temporal_col,
    parse_time_window,
    validate_bounds,
)
import polars as pl


def test_get_logger_returns_logger() -> None:
    logger = get_logger("eventflow.test")
    assert logger.name == "eventflow.test"


def test_validate_bounds_success_and_failure() -> None:
    assert validate_bounds((0, 0, 1, 1)) == (0, 0, 1, 1)
    with pytest.raises(ValueError):
        validate_bounds((1, 0, 0, 1))
    with pytest.raises(ValueError):
        validate_bounds((0, 2, 1, 1))


def test_parse_time_window_parses_units() -> None:
    assert parse_time_window("7d") == {"days": 7}
    assert parse_time_window("1h") == {"hours": 1}
    with pytest.raises(ValueError):
        parse_time_window("bad")
    with pytest.raises(ValueError):
        parse_time_window("10x")


def test_dtype_helpers() -> None:
    assert is_numeric_col(pl.Int64)
    assert is_categorical_col(pl.Utf8)
    assert is_temporal_col(pl.Datetime)
