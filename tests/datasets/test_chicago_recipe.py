"""Integration tests for Chicago crime recipe."""

import polars as pl

from eventflow.datasets.chicago_crime.recipes.chicago_crime_v1 import ChicagoCrimeV1Recipe
from eventflow.core.schema import RecipeConfig, EventSchema, EventMetadata
from eventflow.core.event_frame import EventFrame


def _sample_event_frame() -> EventFrame:
    data = {
        "date": [
            "2024-01-01T00:10:00",
            "2024-01-01T01:00:00",
            "2024-01-01T06:30:00",
        ],
        "latitude": [41.0, 41.0005, 41.001],
        "longitude": [-87.0, -87.0005, -87.001],
        "primary_type": ["THEFT", "ROBBERY", "THEFT"],
        "description": ["a", "b", "c"],
        "location_description": ["street", "street", "home"],
        "arrest": [True, False, False],
        "domestic": [False, False, False],
        "beat": ["1", "1", "1"],
        "district": ["1", "1", "1"],
        "ward": ["1", "1", "1"],
        "community_area": ["1", "1", "1"],
        "fbi_code": ["06", "03", "06"],
        "x_coordinate": [0.0, 0.0, 0.0],
        "y_coordinate": [0.0, 0.0, 0.0],
    }
    lf = pl.LazyFrame(data)
    schema = EventSchema(
        timestamp_col="date",
        lat_col="latitude",
        lon_col="longitude",
        categorical_cols=["primary_type"],
    )
    metadata = EventMetadata(dataset_name="chicago_crime")
    return EventFrame(lf, schema, metadata)


def test_chicago_recipe_runs_and_adds_grid_counts() -> None:
    recipe = ChicagoCrimeV1Recipe(
        RecipeConfig(dataset="chicago_crime", recipe="chicago_crime_v1", grid={"size_m": 100})
    )
    ef = _sample_event_frame()

    result = recipe.run(ef).collect()

    assert "grid_id" in result.columns
    assert "time_bin" in result.columns
    assert "count" in result.columns
    # Expect 2 rows: theft bucketed in two bins, robbery in its bin
    assert result["count"].sum() == 3
