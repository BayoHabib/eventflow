"""Common test fixtures and utilities."""


import polars as pl
import pytest


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_events():
    """Create sample event data."""
    data = {
        "timestamp": [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
        ],
        "latitude": [41.8781, 41.8800, 41.8820],
        "longitude": [-87.6298, -87.6300, -87.6310],
        "type": ["A", "B", "A"],
        "value": [1, 2, 3],
    }

    return pl.DataFrame(data).with_columns([
        pl.col("timestamp").str.to_datetime()
    ])
