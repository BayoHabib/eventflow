"""Tests for pipeline orchestration."""

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import ConditionalStep, LambdaStep, Pipeline, Step
from eventflow.core.schema import EventSchema, EventMetadata


@pytest.fixture
def sample_event_frame() -> EventFrame:
    """Sample EventFrame for pipeline tests."""
    data = {
        "timestamp": ["2024-01-01", "2024-01-02"],
        "latitude": [1.0, 1.0],
        "longitude": [1.0, 1.0],
        "value": [1, 2],
    }

    lf = pl.LazyFrame(data).with_columns(pl.col("timestamp").str.to_datetime())
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["value"],
    )
    metadata = EventMetadata(dataset_name="test")
    return EventFrame(lf, schema, metadata)


class AddColumnStep(Step):
    """Step that adds a literal column."""

    def __init__(self, name: str, value: int | str) -> None:
        self.name = name
        self.value = value

    def run(self, event_frame: EventFrame) -> EventFrame:
        return event_frame.with_columns(pl.lit(self.value).alias(self.name))


class DependentStep(Step):
    """Step that builds on a previously added column."""

    def run(self, event_frame: EventFrame) -> EventFrame:
        return event_frame.with_columns((pl.col("first") + 1).alias("second"))


def test_pipeline_runs_steps_in_order(sample_event_frame: EventFrame) -> None:
    """Steps consume outputs from earlier steps."""
    pipeline = Pipeline([AddColumnStep("first", 1), DependentStep()])
    df = pipeline.run(sample_event_frame).collect().sort("timestamp")

    assert df["first"].to_list() == [1, 1]
    assert df["second"].to_list() == [2, 2]


def test_lambda_step_executes(sample_event_frame: EventFrame) -> None:
    """LambdaStep applies inline transformation."""
    lam = LambdaStep(lambda ef: ef.with_columns(pl.lit("ok").alias("lambda_col")), name="lam")
    df = Pipeline([lam]).run(sample_event_frame).collect()

    assert df["lambda_col"].to_list() == ["ok", "ok"]


def test_conditional_step_branching(sample_event_frame: EventFrame) -> None:
    """ConditionalStep chooses the correct branch."""
    cond_step = ConditionalStep(
        lambda ef: ef.count() > 1,
        if_true=AddColumnStep("flag", "many"),
        if_false=AddColumnStep("flag", "few"),
    )

    df = Pipeline([cond_step]).run(sample_event_frame).collect()
    assert df["flag"].to_list() == ["many", "many"]


def test_add_step_appends(sample_event_frame: EventFrame) -> None:
    """Pipeline.add_step adds new stages."""
    pipeline = Pipeline([])
    pipeline.add_step(AddColumnStep("added", 1))

    assert len(pipeline) == 1
    df = pipeline.run(sample_event_frame).collect()
    assert df["added"].to_list() == [1, 1]
