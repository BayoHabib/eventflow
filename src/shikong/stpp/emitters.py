"""Exact-event emitters for spatio-temporal point processes (STPP).

The emitters turn an :class:`~shikong.core.event_frame.EventFrame` of
exact events into the three input shapes the surveyed STPP
implementations consume (see ``docs/design-notes/stpp-emitters.md``):

- **Gatech / THP / EasyTPP** — one ``{split}.pkl`` (+ ``{split}.json``)
  per split, a dict ``{dim_process, <split>: [sequence]}`` whose
  per-event dicts carry exactly ``idx_event, time_since_start,
  time_since_last_event, time_since_last_same_event, type_event``.
- **Catalog CSV + ``(t, x, y)`` array** — EarthquakeNPP-style, one row
  per event (``id, time, longitude, latitude, mark, mark_label, x, y``,
  ``x``/``y`` in projected kilometres), plus a per-split float64
  ``*_txy.npy`` of shape ``[n, 3]`` = ``(t_seconds_since_start, x_km,
  y_km)`` matching the CSV row for row.
- **tick** — one pickle per split holding a list of realizations, each
  a list of ``n_nodes`` 1-D sorted timestamp arrays (seconds since the
  sequence start), with ``end_times``, ``node_labels`` and
  ``time_unit == "s"``.

Design choices (decided before implementation; documented in the
design note, not re-derived here):

- Coordinates are projected kilometres. The emitter projects lon/lat
  (EPSG:4326) to the caller-supplied ``target_crs`` itself (metres /
  1000); loaders are called without a CRS transform so lon/lat stay
  geographic. ``target_crs`` has no default — it must be chosen and
  recorded explicitly.
- Time unit is seconds (float) for every ``time_since_*`` field and
  every tick timestamp. Catalog ``time`` is ``YYYY-MM-DD HH:MM:SS.sss``
  in the dataset's local time; the time zone is recorded in the
  manifest, never converted.
- Marks are an ad-hoc codebook: the sorted unique mark labels
  (deterministic), ``mark`` / ``type_event`` = 0-based index. A frame
  with no mark column gets the single mark 0, label ``"fix"``.
- Sequences: one sequence per split (the full chronological run). No
  daily windowing — any segmentation is arbitrary; models window
  internally.
- Splits are chronological, built by chaining
  :func:`shikong.evaluation.splits.temporal_split` (with a gap =
  embargo at the train/dev and dev/test cutoffs); embargo events are
  counted by direct filters on the embargo intervals, never silently
  dropped from the reconciliation.

Nothing in this module fits a model: there is no intensity, no
likelihood, no kernel, no learned parameter. Every emitter is a pure
function of its inputs (same input bytes in, same output bytes out).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pyproj import Transformer

from shikong import __version__ as _LIBRARY_VERSION
from shikong.core.event_frame import EventFrame
from shikong.evaluation.splits import temporal_split

# Columns of an extracted exact-event frame, in order.
EVENT_COLUMNS = ("timestamp", "longitude", "latitude", "mark", "mark_label", "x", "y")

# Columns of a catalog CSV, in order.
CATALOG_COLUMNS = ("id", "time", "longitude", "latitude", "mark", "mark_label", "x", "y")

# Per-event keys of a Gatech sequence, in order.
GATECH_KEYS = (
    "idx_event",
    "time_since_start",
    "time_since_last_event",
    "time_since_last_same_event",
    "type_event",
)

# Mark label used when the frame carries no mark column.
SINGLE_MARK_LABEL = "fix"

# Statement recorded in every manifest: this increment emits data only.
NO_FITTED_MODEL_STATEMENT = (
    "Exact-event emission only: no fitted model, no likelihood, "
    "no intensity (lambda) is computed or stored."
)


def _require_no_nulls(df: pl.DataFrame, column: str, *, where: str) -> None:
    """Raise if *column* contains nulls, in the spirit of the split guardrail.

    A null timestamp/coordinate/mark cannot be placed in a sequence, so
    it is refused loudly instead of silently dropped.
    """
    n_null = df.select(pl.col(column).is_null().sum()).item()
    if n_null:
        raise ValueError(
            f"{where}: column {column!r} contains {n_null:,} null value(s) out of "
            f"{df.height:,} rows; exact-event emission refuses nulls"
        )


def extract_exact_events(
    frame: EventFrame,
    *,
    mark_col: str | None,
    target_crs: str,
    source_crs: str = "EPSG:4326",
) -> pl.DataFrame:
    """Extract exact, chronologically sorted events from an EventFrame.

    Args:
        frame: EventFrame whose schema names the timestamp, latitude
            and longitude columns. The frame's lon/lat must be
            geographic (``source_crs``); call the dataset loader
            without a CRS transform.
        mark_col: Name of the categorical mark column, or ``None`` for
            a single mark (label ``"fix"``, index 0) — e.g. GeoLife,
            which carries no categorical column.
        target_crs: Projected CRS for the emitted ``x``/``y`` columns
            (e.g. ``"EPSG:26971"`` for Chicago, ``"EPSG:32650"`` for
            Beijing). Explicit, no default: the projection must be
            chosen and recorded, never silently assumed.
        source_crs: CRS of the frame's lon/lat columns (default
            ``"EPSG:4326"``).

    Returns:
        DataFrame with columns ``timestamp, longitude, latitude, mark,
        mark_label, x, y`` (``x``/``y`` in projected kilometres),
        sorted by timestamp (stable: ties keep input order).

    Raises:
        ValueError: missing timestamp/lon/lat/mark column, a null or
            non-finite timestamp/coordinate/mark value, or an empty
            frame.
    """
    where = "extract_exact_events"
    schema = frame.schema
    time_col, lat_col, lon_col = schema.timestamp_col, schema.lat_col, schema.lon_col
    if lat_col is None or lon_col is None:
        raise ValueError(f"{where}: schema has no lat/lon columns")
    df = frame.collect()
    if df.height == 0:
        raise ValueError(f"{where}: refusing to emit from an empty frame")
    for col in (time_col, lat_col, lon_col):
        if col not in df.columns:
            raise ValueError(f"{where}: column {col!r} is not present in the frame")
        _require_no_nulls(df, col, where=where)
    if mark_col is not None:
        if mark_col not in df.columns:
            raise ValueError(f"{where}: mark column {mark_col!r} is not present in the frame")
        _require_no_nulls(df, mark_col, where=where)
    for col in (lat_col, lon_col):
        n_bad = df.filter(~pl.col(col).is_finite()).height
        if n_bad:
            raise ValueError(
                f"{where}: column {col!r} contains {n_bad:,} non-finite value(s); "
                "exact-event emission refuses non-finite coordinates"
            )

    # Codebook: sorted unique mark labels (deterministic). Labels are
    # the string form of the raw mark values, sorted lexicographically.
    # Explicit annotation: np.zeros infers ndarray[tuple[int], ...]
    # while np.array infers ndarray[tuple[int, ...], ...]; without it
    # mypy rejects the second assignment (numpy 2.2 stubs, Python 3.10).
    mark_values: np.ndarray[tuple[int, ...], np.dtype[np.int64]]
    if mark_col is None:
        labels = [SINGLE_MARK_LABEL]
        mark_values = np.zeros(df.height, dtype=np.int64)
    else:
        raw_labels = df.select(pl.col(mark_col).cast(pl.String)).to_series().to_list()
        labels = sorted(set(raw_labels))
        index_of = {label: i for i, label in enumerate(labels)}
        mark_values = np.array([index_of[label] for label in raw_labels], dtype=np.int64)

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lon = df.select(pl.col(lon_col)).to_series().to_numpy()
    lat = df.select(pl.col(lat_col)).to_series().to_numpy()
    x_m, y_m = transformer.transform(lon, lat)
    x_km = np.asarray(x_m, dtype=np.float64) / 1000.0
    y_km = np.asarray(y_m, dtype=np.float64) / 1000.0

    out = pl.DataFrame(
        {
            "timestamp": df.select(pl.col(time_col)).to_series(),
            "longitude": lon,
            "latitude": lat,
            "mark": mark_values,
            "mark_label": (
                np.array([labels[int(m)] for m in mark_values], dtype=object)
                if mark_col is not None
                else np.array([SINGLE_MARK_LABEL] * df.height, dtype=object)
            ),
            "x": x_km,
            "y": y_km,
        },
        schema_overrides={
            "timestamp": pl.Datetime,
            "longitude": pl.Float64,
            "latitude": pl.Float64,
            "mark": pl.Int64,
            "mark_label": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
        },
    )
    return out.sort("timestamp", maintain_order=True)


@dataclass(frozen=True)
class SplitBoundary:
    """End of one chronological split.

    Attributes:
        name: Split name (``"aux"``, ``"train"``, ``"dev"``, ``"test"``).
        end: Exclusive end of the split (the cutoff passed to
            :func:`temporal_split`).
        gap: Embargo after this split: events in ``[end, end + gap)``
            belong to no split. Must be zero on the last boundary —
            there is nothing after the final split to embargo.
    """

    name: str
    end: datetime
    gap: timedelta = timedelta(0)


@dataclass(frozen=True)
class ChronologicalSplits:
    """Result of :func:`chronological_event_splits`.

    Attributes:
        splits: Split name -> extracted events in that split,
            chronologically ordered, non-empty.
        embargo_counts: Embargo-interval label (``"embargo_after_<name>"``)
            -> number of events in that interval, counted by direct
            filter on the source frame.
        embargo_intervals: Embargo-interval label -> ``(start, end)``.
        trailing_count: Events at or after the last boundary's end
            (always 0 — a nonzero value raises instead of returning).
    """

    splits: dict[str, pl.DataFrame]
    embargo_counts: dict[str, int] = field(default_factory=dict)
    embargo_intervals: dict[str, tuple[datetime, datetime]] = field(default_factory=dict)
    trailing_count: int = 0


def chronological_event_splits(
    events: pl.DataFrame,
    *,
    time_col: str = "timestamp",
    boundaries: list[SplitBoundary],
) -> ChronologicalSplits:
    """Split extracted events chronologically by chaining temporal_split.

    Split ``i`` holds events in ``[previous end + previous gap,
    boundary[i].end)``; the first split starts at the earliest event.
    Every intermediate cutoff is applied with
    :func:`shikong.evaluation.splits.temporal_split` (gap = embargo),
    so the same null/empty-side guardrails apply here. Embargo events
    are counted by direct filters on ``[end, end + gap)`` and reported
    in the result — they are never silently dropped from the
    reconciliation ``sum(splits) + sum(embargoes) == total``.

    Args:
        events: Extracted events (as returned by
            :func:`extract_exact_events`), sorted by timestamp.
        time_col: Timestamp column name.
        boundaries: One :class:`SplitBoundary` per split, in
            chronological order, strictly increasing ``end``. The last
            boundary's ``gap`` must be zero.

    Returns:
        :class:`ChronologicalSplits`.

    Raises:
        ValueError: no boundaries, non-increasing ends, nonzero gap on
            the last boundary, an empty split, or events at/after the
            last boundary's end (trailing events are refused, not
            dropped).
    """
    where = "chronological_event_splits"
    if not boundaries:
        raise ValueError(f"{where}: at least one boundary is required")
    if boundaries[-1].gap != timedelta(0):
        raise ValueError(
            f"{where}: gap on the last boundary {boundaries[-1].name!r} must be zero; "
            "there is nothing after the final split to embargo"
        )
    for prev, cur in zip(boundaries, boundaries[1:], strict=False):
        if cur.end <= prev.end:
            raise ValueError(
                f"{where}: boundary ends must be strictly increasing, "
                f"got {prev.name!r} ending {prev.end} then {cur.name!r} ending {cur.end}"
            )
    if events.height == 0:
        raise ValueError(f"{where}: refusing to split an empty frame")
    if time_col not in events.columns:
        raise ValueError(f"{where}: column {time_col!r} is not present in the frame")

    splits: dict[str, pl.DataFrame] = {}
    embargo_counts: dict[str, int] = {}
    embargo_intervals: dict[str, tuple[datetime, datetime]] = {}
    remaining = events.lazy()
    for boundary in boundaries[:-1]:
        head, remaining = temporal_split(remaining, time_col, boundary.end, gap=boundary.gap)
        splits[boundary.name] = head.collect()
        if boundary.gap:
            label = f"embargo_after_{boundary.name}"
            start, end = boundary.end, boundary.end + boundary.gap
            embargo_intervals[label] = (start, end)
            embargo_counts[label] = events.filter(
                (pl.col(time_col) >= start) & (pl.col(time_col) < end)
            ).height
    last = boundaries[-1]
    last_split = remaining.filter(pl.col(time_col) < last.end).collect()
    if last_split.height == 0:
        raise ValueError(f"{where}: split {last.name!r} is empty; refusing a degenerate split")
    trailing = remaining.filter(pl.col(time_col) >= last.end).collect().height
    if trailing:
        raise ValueError(
            f"{where}: {trailing:,} event(s) at or after the last boundary end "
            f"{last.end} would be dropped — extend the last boundary instead"
        )
    splits[last.name] = last_split
    return ChronologicalSplits(
        splits=splits,
        embargo_counts=embargo_counts,
        embargo_intervals=embargo_intervals,
        trailing_count=0,
    )


def build_gatech_sequence(events: pl.DataFrame) -> list[dict[str, float | int]]:
    """Build one Gatech sequence from a split's extracted events.

    Per-event dicts carry exactly ``idx_event, time_since_start,
    time_since_last_event, time_since_last_same_event, type_event``.
    Times are seconds (float) since the sequence's first event; the
    first event of each type has ``time_since_last_same_event == 0.0``.

    Args:
        events: Extracted events of one split, sorted by timestamp.

    Raises:
        ValueError: empty input.
    """
    if events.height == 0:
        raise ValueError("build_gatech_sequence: refusing an empty split")
    timestamps = events.select(pl.col("timestamp")).to_series().to_list()
    marks = events.select(pl.col("mark")).to_series().to_list()
    start = timestamps[0]
    last_of_type: dict[int, datetime] = {}
    sequence: list[dict[str, float | int]] = []
    for idx, (moment, mark) in enumerate(zip(timestamps, marks, strict=True)):
        time_since_start = (moment - start).total_seconds()
        time_since_last = 0.0 if idx == 0 else (moment - timestamps[idx - 1]).total_seconds()
        if mark in last_of_type:
            time_since_last_same = (moment - last_of_type[mark]).total_seconds()
        else:
            time_since_last_same = 0.0
        last_of_type[mark] = moment
        sequence.append(
            {
                "idx_event": idx,
                "time_since_start": time_since_start,
                "time_since_last_event": time_since_last,
                "time_since_last_same_event": time_since_last_same,
                "type_event": int(mark),
            }
        )
    return sequence


def emit_gatech(
    splits: dict[str, pl.DataFrame],
    *,
    output_dir: str | Path,
    dim_process: int,
) -> list[Path]:
    """Write one Gatech ``{split}.pkl`` (+ ``{split}.json``) per split.

    Each file is a dict ``{dim_process, <split>: [sequence]}`` with a
    single sequence (the split's full chronological run). The pickle
    is the primary artefact (THP loads it with
    ``pickle.load(f, encoding='latin-1')``); the JSON holds the same
    content for inspection.

    Args:
        splits: Split name -> extracted events.
        output_dir: Directory to write into (created if needed).
        dim_process: Number of distinct marks (codebook size).

    Returns:
        Sorted list of written file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(splits):
        payload: dict[str, Any] = {
            "dim_process": dim_process,
            name: [build_gatech_sequence(splits[name])],
        }
        pkl_path = out_dir / f"{name}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(payload, f)
        written.append(pkl_path)
        json_path = out_dir / f"{name}.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(json_path)
    return sorted(written)


def events_to_txy(events: pl.DataFrame) -> np.ndarray:
    """Return a split's events as a float64 ``[n, 3]`` ``(t, x, y)`` array.

    ``t`` is seconds since the split's first event; ``x``/``y`` are
    projected kilometres. Row ``i`` matches row ``i`` of the split's
    catalog CSV.
    """
    if events.height == 0:
        raise ValueError("events_to_txy: refusing an empty split")
    timestamps = events.select(pl.col("timestamp")).to_series().to_list()
    start = timestamps[0]
    t = np.array([(m - start).total_seconds() for m in timestamps], dtype=np.float64)
    x = events.select(pl.col("x")).to_series().to_numpy().astype(np.float64)
    y = events.select(pl.col("y")).to_series().to_numpy().astype(np.float64)
    return np.column_stack([t, x, y])


def _catalog_frame(events: pl.DataFrame) -> pl.DataFrame:
    """Build the catalog rows for one split (columns :data:`CATALOG_COLUMNS`)."""
    if events.height == 0:
        raise ValueError("_catalog_frame: refusing an empty split")
    return events.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("id"),
        pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S%.3f").alias("time"),
        pl.col("longitude"),
        pl.col("latitude"),
        pl.col("mark"),
        pl.col("mark_label"),
        pl.col("x"),
        pl.col("y"),
    )


def emit_catalog(
    splits: dict[str, pl.DataFrame],
    *,
    output_dir: str | Path,
) -> list[Path]:
    """Write one ``{split}_catalog.csv`` + ``{split}_txy.npy`` per split.

    The CSV has exactly the columns ``id, time, longitude, latitude,
    mark, mark_label, x, y`` (``time`` = ``YYYY-MM-DD HH:MM:SS.sss`` in
    dataset local time; ``x``/``y`` in projected km). The ``.npy`` is
    float64 ``[n, 3]`` = ``(t_seconds_since_start, x_km, y_km)`` and
    matches the CSV row for row.

    Returns:
        Sorted list of written file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(splits):
        events = splits[name]
        csv_path = out_dir / f"{name}_catalog.csv"
        _catalog_frame(events).write_csv(csv_path)
        written.append(csv_path)
        npy_path = out_dir / f"{name}_txy.npy"
        np.save(npy_path, events_to_txy(events))
        written.append(npy_path)
    return sorted(written)


def build_tick_payload(
    events: pl.DataFrame,
    *,
    split_end: datetime,
    codebook: list[str],
) -> dict[str, Any]:
    """Build one tick payload from a split's extracted events.

    The payload is ``{"events": [[array per node]], "end_times": [T],
    "node_labels": [...], "time_unit": "s"}``: one realization, one
    1-D sorted float array of timestamps (seconds since the split's
    first event) per node, nodes = marks in codebook order. ``T`` is
    the split end (window end, not the last event) in the same seconds
    since the sequence start.

    Args:
        events: Extracted events of one split, sorted by timestamp.
        split_end: Exclusive end of the split window.
        codebook: Mark labels in index order; ``len(codebook)`` is the
            number of nodes.

    Raises:
        ValueError: empty input, or ``split_end`` before the first event.
    """
    if events.height == 0:
        raise ValueError("build_tick_payload: refusing an empty split")
    timestamps = events.select(pl.col("timestamp")).to_series().to_list()
    marks = events.select(pl.col("mark")).to_series().to_list()
    start = timestamps[0]
    if split_end < start:
        raise ValueError(
            f"build_tick_payload: split end {split_end} is before the first event {start}"
        )
    per_node: list[list[float]] = [[] for _ in codebook]
    for moment, mark in zip(timestamps, marks, strict=True):
        per_node[int(mark)].append((moment - start).total_seconds())
    arrays = [np.array(times, dtype=np.float64) for times in per_node]
    return {
        "events": [arrays],
        "end_times": [(split_end - start).total_seconds()],
        "node_labels": list(codebook),
        "time_unit": "s",
    }


def emit_tick(
    splits: dict[str, pl.DataFrame],
    *,
    output_dir: str | Path,
    split_ends: dict[str, datetime],
    codebook: list[str],
) -> list[Path]:
    """Write one ``{split}_tick.pkl`` per split (tick input format).

    Args:
        splits: Split name -> extracted events.
        output_dir: Directory to write into (created if needed).
        split_ends: Split name -> exclusive window end (the tick
            ``end_times`` value).
        codebook: Mark labels in index order.

    Returns:
        Sorted list of written file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(splits):
        if name not in split_ends:
            raise ValueError(f"emit_tick: no split end supplied for split {name!r}")
        payload = build_tick_payload(splits[name], split_end=split_ends[name], codebook=codebook)
        path = out_dir / f"{name}_tick.pkl"
        with path.open("wb") as f:
            pickle.dump(payload, f)
        written.append(path)
    return sorted(written)


def build_manifest(
    *,
    dataset: str,
    n_events: int,
    n_raw: int,
    n_removed_by_cleaning: int,
    source_crs: str,
    target_crs: str,
    time_zone: str,
    codebook: list[str],
    splits: dict[str, dict[str, Any]],
    embargoes: dict[str, dict[str, Any]],
    files: list[str],
    notes: str | None = None,
) -> dict[str, Any]:
    """Build the manifest dict for one emission.

    Args:
        dataset: Dataset name (``"chicago_crime"``, ``"geolife"``).
        n_events: Cleaned event count emitted.
        n_raw: Raw event count before cleaning.
        n_removed_by_cleaning: ``n_raw - n_events``.
        source_crs: CRS of the loader's lon/lat columns.
        target_crs: Projected CRS of the emitted ``x``/``y`` (km).
        time_zone: Time zone of the catalog ``time`` strings.
        codebook: Mark labels in index order.
        splits: Split name -> ``{"start", "end", "n_events",
            "first_event", "last_event"}`` (ISO strings / int).
        embargoes: Embargo label -> ``{"start", "end", "n_events"}``.
        files: Emitted file names (relative, sorted).
        notes: Free-text notes (e.g. why an embargo is zero).

    Returns:
        Manifest dict (JSON-serializable, deterministic under
        ``sort_keys=True``). Always records ``library_version`` and
        the explicit ``"no fitted model"`` statement.
    """
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "n_events": n_events,
        "n_raw": n_raw,
        "n_removed_by_cleaning": n_removed_by_cleaning,
        "source_crs": source_crs,
        "target_crs": target_crs,
        "time_unit": "s",
        "time_zone": time_zone,
        "codebook": list(codebook),
        "dim_process": len(codebook),
        "splits": splits,
        "embargoes": embargoes,
        "files": sorted(files),
        "library": "shikong",
        "library_version": _LIBRARY_VERSION,
        "fitted_model": "none",
        "model": NO_FITTED_MODEL_STATEMENT,
    }
    if notes is not None:
        manifest["notes"] = notes
    return manifest


def write_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest.json`` (sorted keys, deterministic) into *output_dir*."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
