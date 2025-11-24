"""Tests for context joiners and enricher."""

import polars as pl
import pytest

from eventflow.core.context.enricher import EnricherStep
from eventflow.core.context.joiners import SpatialJoin, SpatioTemporalJoin, TemporalJoin
from eventflow.core.context.sources import BaseContextSource
from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import ContextSchema, EventMetadata, EventSchema


class DummyTemporalSource(BaseContextSource):
    """Temporal-only context source."""

    def load(self) -> pl.LazyFrame:
        return pl.LazyFrame({"ts": ["2024-01-01 00:00:00"], "ctx": [1]}).with_columns(
            pl.col("ts").str.to_datetime()
        )

    @property
    def schema(self) -> ContextSchema:
        return ContextSchema(timestamp_col="ts", attribute_cols=["ctx"])


class DummySpatialSource(BaseContextSource):
    """Spatial-only context source."""

    def load(self) -> pl.LazyFrame:
        return pl.LazyFrame({"grid_id": [1], "ctx": ["z"]})

    @property
    def schema(self) -> ContextSchema:
        return ContextSchema(spatial_col="grid_id", attribute_cols=["ctx"])


class DummySpatioTemporalSource(BaseContextSource):
    """Spatio-temporal context source."""

    def load(self) -> pl.LazyFrame:
        return pl.LazyFrame(
            {"grid_id": [1], "ts": ["2024-01-01 00:00:00"], "ctx": [5]}
        ).with_columns(pl.col("ts").str.to_datetime())

    @property
    def schema(self) -> ContextSchema:
        return ContextSchema(timestamp_col="ts", spatial_col="grid_id", attribute_cols=["ctx"])


def _event_frame() -> EventFrame:
    lf = pl.LazyFrame(
        {
            "timestamp": ["2024-01-01 00:00:00"],
            "grid_id": [1],
            "value": [10],
            "latitude": [0.0],
            "longitude": [0.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())
    schema = EventSchema(timestamp_col="timestamp", lat_col="latitude", lon_col="longitude")
    metadata = EventMetadata(dataset_name="test")
    return EventFrame(lf, schema, metadata)


def test_temporal_join_exact() -> None:
    ef = _event_frame()
    source = DummyTemporalSource()
    joiner = TemporalJoin(strategy="exact")
    joined = joiner.join(ef, source.load(), source.schema).collect()
    assert joined["ctx"].to_list() == [1]


def test_spatial_join_on_grid() -> None:
    ef = _event_frame()
    source = DummySpatialSource()
    joiner = SpatialJoin(join_type="grid", spatial_col="grid_id")
    joined = joiner.join(ef, source.load(), source.schema).collect()
    assert joined["ctx"].to_list() == ["z"]


def test_spatio_temporal_join_combines() -> None:
    ef = _event_frame()
    source = DummySpatioTemporalSource()
    joiner = SpatioTemporalJoin(SpatialJoin("grid", "grid_id"), TemporalJoin("exact"))
    joined = joiner.join(ef, source.load(), source.schema).collect()
    assert joined["ctx"].to_list() == [5]


def test_enricher_step_uses_source_and_joiner() -> None:
    ef = _event_frame()
    source = DummyTemporalSource()
    joiner = TemporalJoin(strategy="exact")
    enricher = EnricherStep(source, joiner)

    enriched = enricher.run(ef).collect()
    assert "ctx" in enriched.columns
