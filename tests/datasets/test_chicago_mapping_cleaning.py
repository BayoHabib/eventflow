"""Integration-style tests for Chicago crime cleaning."""

import polars as pl

from eventflow.datasets.chicago_crime.mapping import clean_chicago_data


def test_clean_chicago_data_casts_and_filters() -> None:
    raw = pl.LazyFrame(
        {
            "date": ["01/01/2024 12:00:00 AM", "2024-01-02T01:30:00.000", "01/03/2024 02:00:00 AM"],
            "latitude": ["41.0", None, "95.0"],
            "longitude": ["-87.0", "-181.0", "-87.5"],
            "primary_type": ["THEFT", "BURGLARY", "THEFT"],
            "description": ["desc1", "desc2", "desc3"],
            "location_description": ["street", "home", "park"],
            "arrest": ["true", "false", "false"],
            "domestic": ["false", "false", "true"],
        }
    )

    cleaned = clean_chicago_data(raw).collect()

    # One valid row should remain (others dropped for null/invalid coords)
    assert len(cleaned) == 1
    assert cleaned["latitude"].dtype == pl.Float64
    assert cleaned["longitude"].dtype == pl.Float64
    assert cleaned["arrest"].to_list() == [True]
    assert cleaned["domestic"].to_list() == [False]
    assert cleaned["date"].dtype == pl.Datetime
