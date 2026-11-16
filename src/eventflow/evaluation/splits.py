"""Blocked train/test splits for spatio-temporal evaluation.

A *blocked* split is one where the train and test sides are separated by a
hard boundary in time, space, or entity identity, so no information can leak
across the boundary through the mechanisms that make random splits invalid
here:

- time: the same time bin (or neighbouring bins, when features use windows)
  on both sides lets a feature computed from a bin predict that same bin;
- space: the same spatial unit on both sides lets a spatial lag or KDE
  centred on a test unit read train events from that same unit, and
  neighbouring units leak through spatial smoothing;
- entity: the same entity (station, vehicle, user) on both sides lets an
  entity-specific pattern learned in train apply verbatim in test.

Every function here validates its inputs and raises instead of silently
producing a degenerate split (empty side, unknown unit, unparseable time
column). ``validate_split`` is the negative control: a deliberately
leaking split MUST be rejected by it.

All functions accept either a ``polars.LazyFrame`` or an ``EventFrame``
and return the same kind they were given.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable
from typing import Any, TypeVar

import polars as pl

from eventflow.core.event_frame import EventFrame

FrameLike = pl.LazyFrame | EventFrame
T = TypeVar("T", pl.LazyFrame, EventFrame)


def _as_lazy(frame: FrameLike) -> pl.LazyFrame:
    """Return the underlying LazyFrame of *frame*."""
    if isinstance(frame, EventFrame):
        return frame.lazy_frame
    return frame


def _reattach(frame: T, lazy: pl.LazyFrame) -> T:
    """Wrap *lazy* in the same container type *frame* came in."""
    if isinstance(frame, EventFrame):
        return frame.with_lazy_frame(lazy)  # type: ignore[return-value]
    return lazy  # type: ignore[return-value]


def _require_column(lazy: pl.LazyFrame, column: str, *, where: str) -> None:
    """Raise if *column* is absent from *lazy*."""
    if column not in lazy.collect_schema().names():
        raise ValueError(f"{where}: column {column!r} is not present in the frame")


def _check_non_empty(lazy: pl.LazyFrame, *, side: str, where: str) -> None:
    """Raise if the lazy frame has no rows."""
    n = lazy.select(pl.len()).collect().item()
    if n == 0:
        raise ValueError(f"{where}: {side} side is empty; refusing a degenerate split")


def _distinct_values(lazy: pl.LazyFrame, column: str) -> set[Any]:
    """Collect the distinct values of *column* as a Python set."""
    return set(lazy.select(pl.col(column).unique()).collect().to_series().to_list())


def _require_temporal_column(lazy: pl.LazyFrame, time_col: str, *, where: str) -> pl.DataType:
    """Return the dtype of *time_col*, raising unless it is temporal."""
    _require_column(lazy, time_col, where=where)
    dtype = lazy.collect_schema()[time_col]
    if dtype not in (pl.Datetime, pl.Date):
        raise ValueError(
            f"{where}: time column {time_col!r} has dtype {dtype}; " "expected Datetime or Date"
        )
    return dtype


def _require_matching_cutoff(
    dtype: pl.DataType, cutoff: dt.datetime | dt.date, *, where: str
) -> dt.datetime | dt.date:
    """Raise unless *cutoff* matches the temporal dtype of the time column."""
    if dtype == pl.Datetime and not isinstance(cutoff, dt.datetime):
        raise ValueError(f"{where}: cutoff {cutoff!r} must be a datetime for a Datetime column")
    if dtype == pl.Date:
        if isinstance(cutoff, dt.datetime) or not isinstance(cutoff, dt.date):
            raise ValueError(f"{where}: cutoff {cutoff!r} must be a date for a Date column")
    return cutoff


def _require_gap(gap: dt.timedelta | None, *, where: str) -> dt.timedelta:
    """Normalise *gap* to a timedelta, raising on a negative value."""
    if gap is None:
        return dt.timedelta(0)
    if not isinstance(gap, dt.timedelta):
        raise ValueError(f"{where}: gap must be a datetime.timedelta, got {gap!r}")
    if gap < dt.timedelta(0):
        raise ValueError(f"{where}: gap must not be negative, got {gap!r}")
    return gap


def temporal_split(
    frame: T,
    time_col: str,
    cutoff: dt.datetime | dt.date,
    gap: dt.timedelta | None = None,
) -> tuple[T, T]:
    """Split a frame into (train, test) at a time boundary.

    Train holds rows with ``time_col < cutoff``; test holds rows with
    ``time_col >= cutoff + gap``. The optional *gap* is an embargo: rows in
    ``[cutoff, cutoff + gap)`` belong to neither side, so features that use
    a look-back window cannot read across the boundary.

    Raises:
        ValueError: missing/non-temporal time column, cutoff of the wrong
            type, negative gap, or an empty train or test side.
    """
    where = "temporal_split"
    lazy = _as_lazy(frame)
    dtype = _require_temporal_column(lazy, time_col, where=where)
    _require_matching_cutoff(dtype, cutoff, where=where)
    embargo = _require_gap(gap, where=where)

    train_lazy = lazy.filter(pl.col(time_col) < cutoff)
    test_lazy = lazy.filter(pl.col(time_col) >= cutoff + embargo)
    _check_non_empty(train_lazy, side="train", where=where)
    _check_non_empty(test_lazy, side="test", where=where)
    return _reattach(frame, train_lazy), _reattach(frame, test_lazy)


def _partition_by_values(
    frame: T,
    column: str,
    test_values: Iterable[Any],
    *,
    where: str,
) -> tuple[T, T]:
    """Split on set membership in *column*; shared by spatial/entity splits."""
    lazy = _as_lazy(frame)
    _require_column(lazy, column, where=where)
    test_set = set(test_values)
    if not test_set:
        raise ValueError(f"{where}: test set must not be empty")
    present = _distinct_values(lazy, column)
    unknown = test_set - present
    if unknown:
        raise ValueError(
            f"{where}: test values not present in the frame: {sorted(map(repr, unknown))[:5]}"
        )
    if test_set >= present:
        raise ValueError(f"{where}: test set covers every {column!r}; train side would be empty")

    train_lazy = lazy.filter(~pl.col(column).is_in(test_set))
    test_lazy = lazy.filter(pl.col(column).is_in(test_set))
    _check_non_empty(train_lazy, side="train", where=where)
    _check_non_empty(test_lazy, side="test", where=where)
    return _reattach(frame, train_lazy), _reattach(frame, test_lazy)


def spatial_split(frame: T, unit_col: str, test_units: Iterable[Any]) -> tuple[T, T]:
    """Split a frame into (train, test) with disjoint spatial units.

    Every row whose *unit_col* is in *test_units* goes to test; all other
    rows go to train. No unit appears on both sides.

    Raises:
        ValueError: missing unit column, empty test set, unknown test unit,
            or a test set covering every unit.
    """
    return _partition_by_values(frame, unit_col, test_units, where="spatial_split")


def entity_split(frame: T, entity_col: str, test_entities: Iterable[Any]) -> tuple[T, T]:
    """Split a frame into (train, test) with disjoint entities.

    Every row whose *entity_col* is in *test_entities* goes to test; all
    other rows go to train. No entity appears on both sides.

    Raises:
        ValueError: missing entity column, empty test set, unknown test
            entity, or a test set covering every entity.
    """
    return _partition_by_values(frame, entity_col, test_entities, where="entity_split")


def partition_units(
    frame: FrameLike,
    unit_col: str,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[Any], list[Any]]:
    """Deterministically partition the distinct units into (train, test).

    Units are ordered by the SHA-256 hash of ``(seed, unit)`` and the first
    ``round(test_fraction * n_units)`` (at least one, never all) become the
    test set. The partition depends only on the unit values and *seed*, so
    it is reproducible across processes and runs — no RNG state involved.

    Returns:
        ``(train_units, test_units)`` as sorted lists of unit values.

    Raises:
        ValueError: missing unit column, ``test_fraction`` outside (0, 1),
            or fewer than two distinct units.
    """
    where = "partition_units"
    lazy = _as_lazy(frame)
    _require_column(lazy, unit_col, where=where)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"{where}: test_fraction must be in (0, 1), got {test_fraction!r}")
    units = sorted(_distinct_values(lazy, unit_col), key=repr)
    if len(units) < 2:
        raise ValueError(f"{where}: need at least 2 distinct units, found {len(units)}")
    n_test = max(1, min(len(units) - 1, round(test_fraction * len(units))))
    hashed = sorted(units, key=lambda u: hashlib.sha256(f"{seed}:{u!r}".encode()).hexdigest())
    test_units = sorted(hashed[:n_test], key=repr)
    train_units = sorted(hashed[n_test:], key=repr)
    return train_units, test_units


def validate_split(
    train: FrameLike,
    test: FrameLike,
    *,
    time_col: str | None = None,
    unit_col: str | None = None,
    entity_col: str | None = None,
) -> None:
    """Reject a train/test pair that leaks across a blocked dimension.

    Checks, for each column supplied, that the two sides share no values:

    - ``time_col``: no shared time bin (temporal leakage);
    - ``unit_col``: no shared spatial unit (spatial leakage);
    - ``entity_col``: no shared entity (entity leakage).

    This is the negative control: a deliberately leaking split MUST raise
    ``ValueError`` here. A clean split returns ``None``.

    Raises:
        ValueError: no blocking column supplied, a column missing from
            either side, or any shared value.
    """
    checks = (
        ("time bin", time_col),
        ("unit", unit_col),
        ("entity", entity_col),
    )
    if all(col is None for _, col in checks):
        raise ValueError("validate_split: supply at least one of time_col, unit_col, entity_col")
    train_lazy, test_lazy = _as_lazy(train), _as_lazy(test)
    for label, col in checks:
        if col is None:
            continue
        _require_column(train_lazy, col, where="validate_split (train)")
        _require_column(test_lazy, col, where="validate_split (test)")
        shared = _distinct_values(train_lazy, col) & _distinct_values(test_lazy, col)
        if shared:
            sample = sorted(map(repr, shared))[:5]
            raise ValueError(
                f"validate_split: train and test share {len(shared)} {label}(s): "
                f"{sample} — blocked split violated"
            )


def blocked_split(
    frame: T,
    time_col: str,
    cutoff: dt.datetime | dt.date,
    *,
    unit_col: str | None = None,
    test_units: Iterable[Any] | None = None,
    entity_col: str | None = None,
    test_entities: Iterable[Any] | None = None,
    gap: dt.timedelta | None = None,
) -> tuple[T, T]:
    """Combined time x space (x entity) blocked split.

    Train holds rows strictly before *cutoff* whose unit/entity is NOT in
    the held-out sets; test holds rows at or after ``cutoff + gap`` whose
    unit/entity IS in the held-out sets. Rows in the two off-diagonal
    quadrants (early but held-out unit, or late but train unit) belong to
    neither side and are dropped — they cannot be assigned without breaking
    one of the blocks. At least one of ``unit_col`` / ``entity_col`` must be
    supplied alongside the time boundary; a time-only split is
    :func:`temporal_split`.

    The result is checked with :func:`validate_split` before returning.

    Raises:
        ValueError: same input errors as the single-dimension splitters,
            no blocking column supplied, or an empty train or test side.
    """
    where = "blocked_split"
    lazy = _as_lazy(frame)
    dtype = _require_temporal_column(lazy, time_col, where=where)
    _require_matching_cutoff(dtype, cutoff, where=where)
    embargo = _require_gap(gap, where=where)

    train_pred = pl.col(time_col) < cutoff
    test_pred = pl.col(time_col) >= cutoff + embargo
    for col, values, label in (
        (unit_col, test_units, "unit"),
        (entity_col, test_entities, "entity"),
    ):
        if col is None and values is not None:
            raise ValueError(f"{where}: test {label}s supplied without a {label} column")
    if unit_col is None and entity_col is None:
        raise ValueError(
            f"{where}: supply unit_col/test_units and/or entity_col/test_entities; "
            "a time-only split is temporal_split"
        )
    for col, values, label in (
        (unit_col, test_units, "unit"),
        (entity_col, test_entities, "entity"),
    ):
        if col is not None:
            if values is None:
                raise ValueError(f"{where}: {col!r} supplied without test {label}s")
            _require_column(lazy, col, where=where)
            value_set = set(values)
            if not value_set:
                raise ValueError(f"{where}: test {label} set must not be empty")
            unknown = value_set - _distinct_values(lazy, col)
            if unknown:
                raise ValueError(
                    f"{where}: test {label}s not present in the frame: "
                    f"{sorted(map(repr, unknown))[:5]}"
                )
            train_pred = train_pred & ~pl.col(col).is_in(value_set)
            test_pred = test_pred & pl.col(col).is_in(value_set)

    train, test = _reattach(frame, lazy.filter(train_pred)), _reattach(
        frame, lazy.filter(test_pred)
    )
    _check_non_empty(_as_lazy(train), side="train", where=where)
    _check_non_empty(_as_lazy(test), side="test", where=where)
    validate_split(train, test, time_col=time_col, unit_col=unit_col, entity_col=entity_col)
    return train, test
