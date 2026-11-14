"""Async execution contracts, cancellation boundaries and synchronous parity."""

from __future__ import annotations

import asyncio
import threading
from contextvars import ContextVar
from datetime import datetime

import polars as pl
import pytest
from shapely.geometry import box

from eventflow.core import EventFrame, EventMetadata, EventSchema
from eventflow.core.domain import SpatialDomain
from eventflow.core.pipeline import (
    AsyncStep,
    ConditionalStep,
    LambdaStep,
    ParallelSteps,
    Pipeline,
    Step,
)
from eventflow.core.registry import FeatureStepRegistry
from eventflow.core.schema import RecipeConfig
from eventflow.core.steps import SpatialDomainMaskStep
from eventflow.recipes.base import BaseRecipe


@pytest.fixture
def frame() -> EventFrame:
    return EventFrame(
        pl.DataFrame(
            {"time": [datetime(2024, 1, 1)] * 2, "x": [1.0, 2.0], "y": [1.0, 2.0], "value": [2, 3]}
        ).lazy(),
        EventSchema(timestamp_col="time", lon_col="x", lat_col="y", numeric_cols=["value"]),
        EventMetadata(dataset_name="async-sensors", crs="EPSG:3857"),
    )


class DoubleAsyncStep(AsyncStep):
    async def run_async(self, event_frame: EventFrame) -> EventFrame:
        await asyncio.sleep(0)
        return event_frame.with_columns((pl.col("value") * 2).alias("value"))


def test_mixed_steps_keep_order_and_materialize_async(frame: EventFrame) -> None:
    async def scenario() -> None:
        pipeline = Pipeline(
            [
                LambdaStep(lambda ef: ef.with_columns((pl.col("value") + 1).alias("value"))),
                DoubleAsyncStep(),
                ParallelSteps(
                    [
                        DoubleAsyncStep(),
                        LambdaStep(
                            lambda ef: ef.with_columns((pl.col("value") - 1).alias("value"))
                        ),
                    ]
                ),
            ]
        )
        result = await pipeline.run_async(frame)
        assert (await result.collect_async())["value"].to_list() == [11, 15]
        assert pipeline.last_schema is result.schema
        assert pipeline.last_metadata is result.metadata
        assert frame.collect()["value"].to_list() == [2, 3]

    asyncio.run(scenario())


def test_existing_steps_match_sync_results(frame: EventFrame) -> None:
    def pipeline() -> Pipeline:
        return Pipeline([SpatialDomainMaskStep(SpatialDomain(box(0, 0, 1.5, 1.5), "EPSG:3857"))])

    async def scenario() -> None:
        expected = pipeline().run(frame).collect()
        result = await pipeline().run_async(frame)
        assert (await result.collect_async()).equals(expected)

    asyncio.run(scenario())


def test_empty_pipeline_returns_input(frame: EventFrame) -> None:
    pipeline = Pipeline([])
    assert asyncio.run(pipeline.run_async(frame)) is frame
    assert pipeline.last_schema is frame.schema


def test_native_async_step_requires_async_entrypoint(frame: EventFrame) -> None:
    pipeline = Pipeline([DoubleAsyncStep()])
    with pytest.raises(TypeError, match="run_async"):
        pipeline.run(frame)
    assert asyncio.run(pipeline.run_async(frame)).collect()["value"].to_list() == [4, 6]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_schema_checks_are_identical(frame: EventFrame, asynchronous: bool) -> None:
    class DropFeature(Step):
        def run(self, ef: EventFrame) -> EventFrame:
            return EventFrame(
                ef.lazy_frame,
                ef.schema.model_copy(update={"feature_provenance": {}}),
                ef.metadata.model_copy(update={"feature_provenance": {}}),
            )

    source = frame.register_feature("value", {"source_step": "source"})
    pipeline = Pipeline([DropFeature()])
    with pytest.raises(ValueError, match="incompatible schema"):
        if asynchronous:
            asyncio.run(pipeline.run_async(source))
        else:
            pipeline.run(source)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wrong_result_type_is_rejected(frame: EventFrame, asynchronous: bool) -> None:
    class WrongResult(Step):
        def run(self, ef: EventFrame) -> EventFrame:
            return None  # type: ignore[return-value]

    pipeline = Pipeline([WrongResult()])
    with pytest.raises(TypeError, match="instead of EventFrame"):
        if asynchronous:
            asyncio.run(pipeline.run_async(frame))
        else:
            pipeline.run(frame)


@pytest.mark.parametrize("native", [False, True])
def test_step_errors_stop_following_steps_and_release_pipeline(
    frame: EventFrame, native: bool
) -> None:
    calls = []

    class FailSync(Step):
        def run(self, ef: EventFrame) -> EventFrame:
            calls.append("first")
            if len(calls) == 1:
                raise ValueError("backend failure")
            return ef

    class FailAsync(AsyncStep):
        async def run_async(self, ef: EventFrame) -> EventFrame:
            return FailSync().run(ef)

    def next_step(ef: EventFrame) -> EventFrame:
        calls.append("next")
        return ef

    async def scenario() -> None:
        pipeline = Pipeline([FailAsync() if native else FailSync(), LambdaStep(next_step)])
        with pytest.raises(ValueError, match="backend failure"):
            await pipeline.run_async(frame)
        assert calls == ["first"]
        await pipeline.run_async(frame)
        assert calls == ["first", "first", "next"]

    asyncio.run(scenario())


def test_sync_work_does_not_block_loop_and_context_is_propagated(frame: EventFrame) -> None:
    request_id: ContextVar[str] = ContextVar("request_id", default="unset")
    started, release = threading.Event(), threading.Event()
    caller_thread = threading.get_ident()
    seen = []

    class BlockingStep(Step):
        def run(self, ef: EventFrame) -> EventFrame:
            seen.append((threading.get_ident(), request_id.get()))
            started.set()
            if not release.wait(5):
                raise TimeoutError("test worker not released")
            return ef

    async def scenario() -> None:
        request_id.set("request-42")
        pipeline = Pipeline([BlockingStep()])
        task = asyncio.create_task(pipeline.run_async(frame))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            # Executed by the event loop while the synchronous step is waiting.
            assert not task.done()
            with pytest.raises(RuntimeError, match="already running"):
                await pipeline.run_async(frame)
            with pytest.raises(RuntimeError, match="already running"):
                pipeline.run(frame)
            with pytest.raises(RuntimeError, match="already running"):
                pipeline.add_step(DoubleAsyncStep())
        finally:
            release.set()
            await task
        pipeline.add_step(DoubleAsyncStep())

    asyncio.run(scenario())
    assert seen == [(seen[0][0], "request-42")]
    assert seen[0][0] != caller_thread


@pytest.mark.parametrize("backend_fails", [False, True])
def test_cancel_drains_sync_worker_and_never_starts_next_step(
    frame: EventFrame,
    backend_fails: bool,
) -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    next_calls = []

    class BlockingStep(Step):
        def run(self, ef: EventFrame) -> EventFrame:
            started.set()
            try:
                if not release.wait(5):
                    raise TimeoutError("test worker not released")
                if backend_fails:
                    raise ValueError("error after cancellation")
                return ef
            finally:
                finished.set()

    def next_step(ef: EventFrame) -> EventFrame:
        next_calls.append(True)
        return ef

    async def scenario() -> None:
        pipeline = Pipeline([BlockingStep(), LambdaStep(next_step)])
        task = asyncio.create_task(pipeline.run_async(frame))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()  # Repeated cancellation must not orphan the worker.
            await asyncio.sleep(0)
            assert not task.done()
            assert not finished.is_set()
            with pytest.raises(RuntimeError, match="already running"):
                await pipeline.run_async(frame)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert finished.is_set()
        assert next_calls == []
        pipeline.add_step(DoubleAsyncStep())  # Ownership was released after cleanup.

    asyncio.run(scenario())


def test_native_async_cancellation_releases_pipeline(frame: EventFrame) -> None:
    async def scenario() -> None:
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        class WaitingStep(AsyncStep):
            async def run_async(self, ef: EventFrame) -> EventFrame:
                started.set()
                await release.wait()
                return ef

        def next_step(ef: EventFrame) -> EventFrame:
            calls.append(True)
            return ef

        pipeline = Pipeline([WaitingStep(), LambdaStep(next_step)])
        task = asyncio.create_task(pipeline.run_async(frame))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == []
        release.set()
        await pipeline.run_async(frame)
        assert calls == [True]

    asyncio.run(scenario())


def test_independent_pipelines_can_overlap(frame: EventFrame) -> None:
    async def scenario() -> None:
        both_started, release = asyncio.Event(), asyncio.Event()
        started_count = 0

        class WaitingStep(AsyncStep):
            async def run_async(self, ef: EventFrame) -> EventFrame:
                nonlocal started_count
                started_count += 1
                if started_count == 2:
                    both_started.set()
                await release.wait()
                return ef

        pipelines = [Pipeline([WaitingStep()]), Pipeline([WaitingStep()])]
        tasks = [asyncio.create_task(p.run_async(frame)) for p in pipelines]
        try:
            await asyncio.wait_for(both_started.wait(), 5)
            assert all(not task.done() for task in tasks)
        finally:
            release.set()
            assert await asyncio.gather(*tasks) == [frame, frame]

    asyncio.run(scenario())


@pytest.mark.parametrize("condition", [True, False])
def test_conditional_wrapper_supports_async_branches(frame: EventFrame, condition: bool) -> None:
    pipeline = Pipeline([ConditionalStep(lambda ef: condition, DoubleAsyncStep())])
    result = asyncio.run(pipeline.run_async(frame))
    assert result.collect()["value"].to_list() == ([4, 6] if condition else [2, 3])


def test_collection_happens_off_loop(frame: EventFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    original = frame.collect
    threads = []

    def collect() -> pl.DataFrame:
        threads.append(threading.get_ident())
        return original()

    monkeypatch.setattr(frame, "collect", collect)
    caller = threading.get_ident()
    assert asyncio.run(frame.collect_async())["value"].to_list() == [2, 3]
    assert threads and threads[0] != caller


def test_recipe_async_respects_configured_step_registry(frame: EventFrame) -> None:
    registry = FeatureStepRegistry()
    registry.register("double_async", DoubleAsyncStep)

    class ConfiguredRecipe(BaseRecipe):
        def build_pipeline(self) -> Pipeline:
            raise AssertionError("Configured registry steps should be used")

    recipe = ConfiguredRecipe(
        RecipeConfig(dataset="sensors", recipe="custom", features={"steps": ["double_async"]}),
        step_registry=registry,
    )
    result = asyncio.run(recipe.run_async(frame))
    assert result.collect()["value"].to_list() == [4, 6]
