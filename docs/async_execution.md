# Asynchronous pipeline execution

Shikong's ordinary `Pipeline` can be awaited inside an asyncio application.
Existing synchronous steps run in worker threads. Native async steps can await
their I/O directly. The synchronous `run()` API remains available.

```python
prepared = await pipeline.run_async(events)
table = await prepared.collect_async()
```

`run_async()` returns an EventFrame, preserving Polars lazy evaluation. The
second await materializes the data off the event loop. As with synchronous
pipelines, validation and computation errors in a lazy expression may only
surface at materialization. Both operations propagate their original errors.

## Existing and native async steps

The default `Step.run_async()` offloads `run()` using the default asyncio thread
executor and propagates context variables, such as application request IDs.
An existing step needs no wrapper to participate in an asynchronous pipeline.

For native asynchronous operations, implement `AsyncStep`:

```python
from shikong.core.pipeline import AsyncStep

class ReferenceEnrichment(AsyncStep):
    def __init__(self, load_reference):
        self.load_reference = load_reference

    async def run_async(self, events):
        reference = await self.load_reference()
        return events.with_lazy_frame(
            events.lazy_frame.join(reference.lazy(), on="site_id", how="left")
        )
```

The caller supplies the async loader and its connection lifecycle/timeouts.
Native async code must avoid blocking operations on the event loop. Calling
the synchronous pipeline entry point with an `AsyncStep` raises a clear error;
the library does not start nested event loops.

Steps are awaited in their declared order. Return-type and schema compatibility
checks are shared by synchronous and asynchronous pipeline execution.
`ConditionalStep` offloads its synchronous predicate and awaits its selected
branch. `ParallelSteps` retains its legacy sequential transformation semantics
with async children as well.

`BaseRecipe.run_async()` builds the configured pipeline off-loop and awaits it,
including recipes assembled from registry step definitions.

## Concurrency and ownership

A Pipeline instance admits one execution at a time, across synchronous and
asynchronous callers. Overlapping calls and `add_step()` during execution raise
`RuntimeError`. Each run uses a snapshot of its step list. `last_schema` and
`last_metadata` track the latest successfully completed step in that run.

Use separate pipeline/step instances for independent executions. For example,
an application can limit concurrency and retain ownership through collection:

```python
import asyncio

limit = asyncio.Semaphore(4)

async def prepare_one(events):
    async with limit:
        pipeline = make_pipeline()  # A fresh pipeline and its own steps.
        prepared = await pipeline.run_async(events)
        return await prepared.collect_async()
```

Callers must avoid mutating frames, steps, or reference data while operations
using them are active. Thread-affine resources need an appropriate step adapter:
create/use synchronous resources in their owning worker or implement native
async I/O. A lazy expression should not capture thread-bound native handles
created on a different thread.

## Cancellation and errors

- Cancelling a native async step follows its coroutine's cancellation behavior.
- A running synchronous worker cannot be safely killed by Python. Shikong
  waits asynchronously for that worker to finish, discards its result, and
  propagates `asyncio.CancelledError`. It never starts the following step.
- Repeated cancellation does not release pipeline ownership prematurely or leave
  an unobserved worker error. If the worker fails during cancellation, cancellation
  takes precedence after cleanup.
- Cancelling `collect_async()` also drains its running worker before propagating.
- Cancellation does not roll back side effects already performed by a custom step.

`asyncio.wait_for()` requests cancellation, so its timeout is **not a hard stop**
for an already executing synchronous backend. Configure appropriate timeouts in
the backend itself. The event loop remains available during worker cleanup.

These APIs provide coroutine integration and explicit lifecycle behavior for
`Pipeline`, its conditional/sequential wrappers, recipes and EventFrame
collection. They do not introduce a DAG scheduler, distributed executor,
persistent job queue, or change the separate StreamPipeline/BranchingPipeline
execution APIs. CPU throughput remains dependent on the backend, the GIL and
native thread pools; applications choose appropriate concurrency limits.
