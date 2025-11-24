"""Integration-style tests for Chicago crime cleaning."""

import polars as pl

from eventflow.datasets.chicago_crime.mapping import clean_chicago_data


def test_clean_chicago_data_casts_and_filters() -> None:
    raw = pl.LazyFrame(
        {
            "date": ["01/01/2024 12:00:00 AM", "01/02/2024 01:30:00 AM"],
            "latitude": ["41.0", None],
            "longitude": ["-87.0", "-181.0"],
            "primary_type": ["THEFT", "BURGLARY"],
            "description": ["desc1", "desc2"],
            "location_description": ["street", "home"],
            "arrest": ["true", "false"],
            "domestic": ["false", "false"],
        }
    )

    cleaned = clean_chicago_data(raw).collect()

    # One row should remain (second has null lat; second has invalid lon)
    assert len(cleaned) == 1
    assert cleaned["latitude"].dtype == pl.Float64
    assert cleaned["longitude"].dtype == pl.Float64
    assert cleaned["arrest"].to_list() == [True]
    assert cleaned["domestic"].to_list() == [False]
    assert cleaned["date"].dtype == pl.Datetime
