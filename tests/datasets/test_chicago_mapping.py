"""Tests for Chicago crime dataset mapping utilities."""

import polars as pl

from eventflow.datasets.chicago_crime.mapping import (
    clean_chicago_data,
    load_chicago_crime,
    sample_chicago_data,
)
from eventflow.datasets.chicago_crime.schema import CHICAGO_CRIME_SCHEMA, create_chicago_metadata
from eventflow.core.schema import DatasetConfig
from eventflow.core.event_frame import EventFrame


def _raw_lazyframe() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "date": ["01/01/2024 12:00:00 AM", "01/02/2024 01:30:00 AM"],
            "latitude": [41.0, None],
            "longitude": [-87.0, -87.1],
            "primary_type": ["THEFT", "BURGLARY"],
            "description": ["desc1", "desc2"],
            "location_description": ["street", "home"],
            "arrest": ["true", "false"],
            "domestic": ["false", "false"],
        }
    )


def test_clean_chicago_data_filters_and_coerces() -> None:
    lf = _raw_lazyframe()
    cleaned = clean_chicago_data(lf).collect()

    # Drops row with null lat
    assert len(cleaned) == 1
    assert cleaned["arrest"][0] is True
    assert cleaned["domestic"][0] is False
    # Timestamp parsed
    assert cleaned["date"].dtype == pl.Datetime


def test_load_chicago_crime_with_config(monkeypatch) -> None:
    # patch loader to avoid filesystem access
    monkeypatch.setattr(
        "eventflow.datasets.chicago_crime.mapping.load_raw_chicago",
        lambda raw_path, layout="nested": _raw_lazyframe(),
    )

    cfg = DatasetConfig(
        dataset_name="chicago_crime",
        raw_root="ignored",
        layout="nested",
        crs="EPSG:26971",
        time_zone="America/Chicago",
    )

    ef = load_chicago_crime("ignored", config=cfg, apply_cleaning=True)
    assert ef.schema == CHICAGO_CRIME_SCHEMA
    assert ef.metadata.crs == "EPSG:26971"
    assert ef.metadata.time_zone == "America/Chicago"
    assert ef.count() == 1


def test_sample_chicago_data_fraction() -> None:
    # patch loader to avoid filesystem scan
    lf = _raw_lazyframe()
    ef = EventFrame(lf, CHICAGO_CRIME_SCHEMA, create_chicago_metadata())
    sampled = sample_chicago_data(ef, n=1)
    assert len(sampled.collect()) == 1
