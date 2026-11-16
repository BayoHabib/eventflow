"""Expanding-window walk-forward evaluation over time bins.

The harness refits the pipeline on each fold's train window and scores it
on the following test window. It never chooses a model and never scores by
itself: scoring is a caller-supplied callable, so the harness stays usable
for any downstream task (hotspot ranking, classification, ...).

Fittable steps are detected with ``hasattr(step, "fit")`` and fit on the
train fold only, before the fold runs. The ``FittableStep`` protocol
(``fit(event_frame) -> Self`` / ``is_fitted -> bool``) is defined in
``shikong.core.pipeline`` by the worker owning criterion 5; this module
deliberately does not redefine it and only relies on the ``fit`` method
existing at runtime.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

import polars as pl

from shikong.core.event_frame import EventFrame
from shikong.core.pipeline import Pipeline, Step
from shikong.evaluation.splits import FrameLike, _require_no_nulls, validate_split

T = TypeVar("T", pl.LazyFrame, EventFrame, pl.DataFrame)

TimeBound = dt.datetime | dt.date


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window fold: train window plus the following test window."""

    fold: int
    train_start: TimeBound
    train_end: TimeBound
    test_start: TimeBound
    test_end: TimeBound
    train: FrameLike
    test: FrameLike


@dataclass(frozen=True)
class FoldResult:
    """Per-fold outcome of :func:`evaluate_walk_forward`."""

    fold: int
    train_start: TimeBound
    train_end: TimeBound
    test_start: TimeBound
    test_end: TimeBound
    n_train: int
    n_test: int
    scores: dict[str, float]


def _time_range(lazy: pl.LazyFrame, time_col: str) -> tuple[TimeBound, TimeBound]:
    """Return (min, max) of *time_col* as Python datetimes/dates."""
    row = (
        lazy.select(pl.col(time_col).min().alias("t_min"), pl.col(time_col).max().alias("t_max"))
        .collect()
        .row(0)
    )
    t_min, t_max = row
    if t_min is None or t_max is None:
        raise ValueError(f"walk-forward: time column {time_col!r} has no non-null values")
    return t_min, t_max


def _as_lazy(frame: FrameLike) -> pl.LazyFrame:
    """Return the underlying LazyFrame of *frame*.

    A collected DataFrame is converted with ``.lazy()`` (cheap, no data
    copy); anything else raises a clear TypeError instead of failing
    later with a cryptic AttributeError.
    """
    if isinstance(frame, EventFrame):
        return frame.lazy_frame
    if isinstance(frame, pl.DataFrame):
        return frame.lazy()
    if isinstance(frame, pl.LazyFrame):
        return frame
    raise TypeError("expected a LazyFrame, EventFrame or DataFrame, " f"got {type(frame).__name__}")


def _reattach(frame: T, lazy: pl.LazyFrame) -> T:
    """Wrap *lazy* in the same container type *frame* came in."""
    if isinstance(frame, EventFrame):
        return frame.with_lazy_frame(lazy)  # type: ignore[return-value]
    if isinstance(frame, pl.DataFrame):
        return lazy.collect()  # type: ignore[return-value]
    return lazy  # type: ignore[return-value]


def _validate_window(initial_train: dt.timedelta, step: dt.timedelta, gap: dt.timedelta) -> None:
    """Raise on non-positive windows or a negative embargo gap."""
    for name, value in (("initial_train", initial_train), ("step", step)):
        if not isinstance(value, dt.timedelta):
            raise ValueError(f"walk-forward: {name} must be a datetime.timedelta, got {value!r}")
        if value <= dt.timedelta(0):
            raise ValueError(f"walk-forward: {name} must be positive, got {value!r}")
    if not isinstance(gap, dt.timedelta):
        raise ValueError(f"walk-forward: gap must be a datetime.timedelta, got {gap!r}")
    if gap < dt.timedelta(0):
        raise ValueError(f"walk-forward: gap must not be negative, got {gap!r}")


def walk_forward_splits(
    frame: T,
    time_col: str,
    initial_train: dt.timedelta,
    step: dt.timedelta,
    gap: dt.timedelta | None = None,
    max_folds: int | None = None,
) -> Iterator[WalkForwardFold]:
    """Yield expanding-window (train, test) folds over time.

    The first fold trains on ``[t_min, t_min + initial_train)`` and tests
    on the next *step* after the embargo *gap*; each later fold extends the
    train window by one *step* (expanding window, never sliding) and tests
    on the following *step*. Folds stop when the test window starts after
    the last observed time bin. A fold whose test window holds no rows
    (sparse time column) is skipped, not yielded.

    Every yielded fold is checked with
    :func:`shikong.evaluation.splits.validate_split` on the time column,
    so a fold can never share a time bin between train and test.

    Args:
        frame: Input frame (LazyFrame, EventFrame, or collected
            DataFrame — the folds are returned in the same kind).
        time_col: Temporal column (Datetime or Date).
        initial_train: Duration of the first train window.
        step: Duration each fold advances the test window by.
        gap: Optional embargo between train end and test start.
        max_folds: Optional cap on the number of yielded folds.

    Yields:
        :class:`WalkForwardFold` values in chronological order.

    Raises:
        ValueError: missing/non-temporal time column, a time column
            containing nulls, non-positive ``initial_train``/``step``,
            negative gap, or ``max_folds`` < 1.
    """
    where = "walk_forward_splits"
    lazy = _as_lazy(frame)
    if time_col not in lazy.collect_schema().names():
        raise ValueError(f"{where}: column {time_col!r} is not present in the frame")
    dtype = lazy.collect_schema()[time_col]
    if dtype not in (pl.Datetime, pl.Date):
        raise ValueError(
            f"{where}: time column {time_col!r} has dtype {dtype}; expected Datetime or Date"
        )
    _require_no_nulls(lazy, time_col, where=where)
    embargo = gap if gap is not None else dt.timedelta(0)
    _validate_window(initial_train, step, embargo)
    if max_folds is not None and max_folds < 1:
        raise ValueError(f"{where}: max_folds must be >= 1, got {max_folds!r}")

    t_min, t_max = _time_range(lazy, time_col)
    fold_idx = 0
    while True:
        if max_folds is not None and fold_idx >= max_folds:
            return
        train_end = t_min + initial_train + fold_idx * step
        test_start = train_end + embargo
        if test_start > t_max:
            return
        test_end = test_start + step
        train_lazy = lazy.filter(pl.col(time_col) < train_end)
        test_lazy = lazy.filter((pl.col(time_col) >= test_start) & (pl.col(time_col) < test_end))
        fold_idx += 1
        if test_lazy.select(pl.len()).collect().item() == 0:
            continue
        train = _reattach(frame, train_lazy)
        test = _reattach(frame, test_lazy)
        validate_split(train, test, time_col=time_col)
        yield WalkForwardFold(
            fold=fold_idx - 1,
            train_start=t_min,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train=train,
            test=test,
        )


def _count(frame: FrameLike) -> int:
    """Row count of a LazyFrame, EventFrame, or DataFrame."""
    if isinstance(frame, EventFrame):
        return frame.count()
    return int(_as_lazy(frame).select(pl.len()).collect().item())


def evaluate_walk_forward(
    steps: Sequence[Step],
    frame: EventFrame,
    time_col: str,
    initial_train: dt.timedelta,
    step: dt.timedelta,
    score_fn: Callable[[EventFrame, EventFrame], dict[str, float]],
    gap: dt.timedelta | None = None,
    max_folds: int | None = None,
) -> list[FoldResult]:
    """Run a pipeline per walk-forward fold and score each fold.

    For every fold from :func:`walk_forward_splits`, each step that
    implements ``fit`` is fit on the train fold only (fresh fit per fold:
    fitted statistics from a previous fold are overwritten, never
    carried over), then the pipeline runs on the train fold and on the
    test fold, and ``score_fn(train_out, test_out)`` produces the fold's
    scores. The pipeline instance is fresh per fold.

    Args:
        steps: Pipeline steps, in order. Steps with a ``fit`` method are
            fit on each fold's train frame before the fold runs.
        frame: Input EventFrame (must be an EventFrame: pipeline steps
            operate on EventFrames).
        time_col: Temporal column driving the folds.
        initial_train: Duration of the first train window.
        step: Duration each fold advances the test window by.
        score_fn: Caller-supplied scoring callable taking the pipeline
            outputs ``(train_out, test_out)`` and returning a dict of
            metric name -> finite float.
        gap: Optional embargo between train end and test start.
        max_folds: Optional cap on the number of folds.

    Returns:
        One :class:`FoldResult` per fold, in chronological order.

    Raises:
        TypeError: *frame* is not an EventFrame — pipeline steps operate
            on EventFrames, so a LazyFrame or DataFrame cannot run here
            (use :func:`walk_forward_splits` for frame-level folds).
        ValueError: no fold is produced (e.g. the frame is shorter than
            ``initial_train + step``), or ``score_fn`` returns a malformed
            result.
    """
    if not isinstance(frame, EventFrame):
        raise TypeError(
            "evaluate_walk_forward: frame must be an EventFrame (pipeline steps "
            f"operate on EventFrames), got {type(frame).__name__}"
        )
    results: list[FoldResult] = []
    for fold in walk_forward_splits(
        frame, time_col, initial_train, step, gap=gap, max_folds=max_folds
    ):
        train_frame = fold.train
        assert isinstance(train_frame, EventFrame)
        test_frame = fold.test
        assert isinstance(test_frame, EventFrame)

        for step_obj in steps:
            fit = getattr(step_obj, "fit", None)
            if callable(fit):
                fit(train_frame)

        pipeline = Pipeline(list(steps))
        train_out = pipeline.run(train_frame)
        test_out = pipeline.run(test_frame)

        scores = score_fn(train_out, test_out)
        if not isinstance(scores, dict):
            raise ValueError(
                f"evaluate_walk_forward: score_fn must return a dict, got {type(scores).__name__}"
            )
        clean_scores: dict[str, float] = {}
        for key, value in scores.items():
            if not isinstance(key, str):
                raise ValueError(
                    "evaluate_walk_forward: score_fn dict keys must be str, " f"got {key!r}"
                )
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"evaluate_walk_forward: score {key!r} is not a number: {value!r}"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(f"evaluate_walk_forward: score {key!r} is not finite: {value!r}")
            clean_scores[key] = number

        results.append(
            FoldResult(
                fold=fold.fold,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                n_train=_count(train_frame),
                n_test=_count(test_frame),
                scores=clean_scores,
            )
        )

    if not results:
        raise ValueError(
            "evaluate_walk_forward: no folds produced; the frame may be shorter than "
            "initial_train + step, or every test window may be empty"
        )
    return results


__all__ = [
    "FoldResult",
    "WalkForwardFold",
    "evaluate_walk_forward",
    "walk_forward_splits",
]
