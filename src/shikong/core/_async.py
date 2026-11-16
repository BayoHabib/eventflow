"""Internal asyncio bridge for non-interruptible synchronous work."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


async def run_blocking(function: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Offload work, draining the worker before propagating cancellation.

    Python cannot safely stop an executing thread. Keeping its awaiter alive
    until completion prevents a cancelled pipeline from releasing state while
    its synchronous step is still running. Repeated cancellation is tolerated.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()  # Retrieve any backend error; cancellation takes precedence.
        raise
