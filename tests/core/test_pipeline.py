"""Tests for pipeline metadata compatibility handling."""

from __future__ import annotations

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import Pipeline, Step
from eventflow.core.schema import EventMetadata, EventSchema, FeatureProvenance


@pytest.fixture()
def base_event_frame() -> EventFrame:
    lf = pl.LazyFrame(
        {
            "timestamp": ["2024-01-01T00:00:00"],
            "value": [1],
            "latitude": [41.0],
            "longitude": [-87.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["value"],
    )
    metadata = EventMetadata(dataset_name="pipeline-test")
    return EventFrame(lf, schema, metadata)


def test_pipeline_detects_incompatible_schema(base_event_frame: EventFrame) -> None:
    enriched = base_event_frame.register_feature(
        "value_mean",
        {"description": "Rolling mean", "source_step": "MeanStep"},
    )

    class DropFeatureStep(Step):
        def run(self, event_frame: EventFrame) -> EventFrame:
            mutated_schema = event_frame.schema.model_copy(update={"feature_provenance": {}})
            mutated_metadata = event_frame.metadata.model_copy(update={"feature_provenance": {}})
            return EventFrame(event_frame.lazy_frame, mutated_schema, mutated_metadata)

    pipeline = Pipeline([DropFeatureStep()])

    with pytest.raises(ValueError, match="incompatible schema"):
        pipeline.run(enriched)


def test_pipeline_updates_last_schema(base_event_frame: EventFrame) -> None:
    class AddFeatureStep(Step):
        def run(self, event_frame: EventFrame) -> EventFrame:
            provenance = FeatureProvenance(produced_by="Adder")
            return event_frame.register_feature(
                "value_twice",
                {"description": "Value times two", "source_step": "Adder"},
                provenance=provenance,
            )

    pipeline = Pipeline([AddFeatureStep()])
    result = pipeline.run(base_event_frame)

    assert pipeline.last_schema is result.schema
    assert pipeline.last_metadata is result.metadata
