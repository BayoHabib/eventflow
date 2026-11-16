"""Deep-learning STPP data formats (data preparation only).

Extensions of :mod:`shikong.stpp.emitters` for the input formats the
neural STPP implementations consume. Every loader specification below
was read verbatim from the reference repositories in 2026-11-16 (see
``docs/design-notes/stpp-dl-data-prep.md`` for the per-model table, the
sources, and what stays unverified):

- **DeepSTPP / AutoSTPP — shared sliding-window wrapper contract.**
  Both repos ship (near-)identical ``SlidingWindowWrapper`` classes
  (DeepSTPP ``src/data/dataset.py``; AutoSTPP
  ``src/data/lightning/sliding_window.py``): input is a list of
  ``[n, 3]`` arrays, columns ``[t, x, y]``, time absolute from the
  sequence start; the wrapper itself converts column 0 to gaps
  (``np.diff(..., prepend=0)``), min–max scales to [0, 1] (fitted on
  train only, in AutoSTPP's Lightning module), and windows internally
  (``lookback=20, lookahead=1``). AutoSTPP's file layout is one
  ``.npz`` per dataset with keys ``train`` / ``val`` / ``test``
  (``allow_pickle=True``); DeepSTPP has no canonical file format at
  all — the wrapper is in-memory, and the repo's only file format is
  the synthetic CSV ``[x, y, t]`` (``synthetic.py::save``).
- **NSTPP** — ``facebookresearch/neural_stpp``,
  ``data/download_and_preprocess_earthquakes.py`` (read in full): one
  ``.npz`` with one key per 30-day window named ``YYYYMMDD`` (window
  start), value float64 ``[n, 3]`` = ``[t_days_since_window_start,
  longitude, latitude]``; windows stepped weekly, overlapping; no
  splits in the file (splits happen at load in ``datasets.py``, which
  was NOT read — see the design note).
- **DSTPP** — ``tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-
  Processes``, ``app.py`` + ``DSTPP/Dataset.py`` (read in full):
  ``dataset/<name>/data_{train,val,test}.pkl``; each pickle is a list
  of sequences, each sequence a list of positional events
  ``[t, lng, lat]`` (dim=2, unmarked); the loader inserts the gap at
  index 1 and asserts the minimum gap is strictly positive.
- **SMASH** — ``zichongli5/SMASH``, ``app.py`` + ``model/Dataset.py``
  (read in full): ``dataset/<name>/data_{train,val,test}.pkl``; each
  pickle is a list of sequences of positional events ``[t, mark, lng,
  lat]`` (dim=3, marked; mark = integer event type, 0-based in the
  pickle); the loader inserts the gap at index 1 and log-transforms it
  by default (``log(max(gap, 1e-4))``).

Design choices (decided before implementation; documented in the
design note, not re-derived here):

- One sequence per split for the split-based formats (the full
  chronological run) — same as the Gatech/catalog/tick emitters.
  Split mapping is ``train`` → ``train``, ``dev`` → ``val``,
  ``test`` → ``test``. The ``aux`` split is EXCLUDED from the
  fixed-schema containers (the loaders have no aux key); it stays
  available in the Gatech/catalog/tick emissions.
- Time is seconds (float) since the split start for the wrapper,
  DSTPP, SMASH and DeepSTPP-CSV formats — the same origin as
  :func:`shikong.stpp.emitters.events_to_txy`. NSTPP windows use
  DAYS since the window start (the preprocessor's unit).
- Coordinates: projected kilometres (``x``/``y``) for the wrapper and
  DeepSTPP CSV (the loaders treat them opaquely and min–max scale);
  raw longitude/latitude DEGREES for NSTPP, DSTPP and SMASH (their
  field names are ``lng``/``lat``).
- Shikong ships UN-normalised absolute values and does NO pre-windowing
  and NO tie-breaking: normalisation is the loader's job (fitted on
  train only — the pre-split scaler fit in DeepSTPP's ``DataAdapter``
  and ``synthetic.py`` leaks by construction and is deliberately NOT
  replicated); windowing is the loader's job (lookback 20); breaking
  same-minute ties would fabricate sub-minute precision. See the
  design note for the DSTPP consequence: the reference loader's
  ``assert Min[1] > 0`` fails on Chicago's 1-minute-resolution ties —
  documented, not hidden.

Nothing in this module fits a model: there is no intensity, no
likelihood, no kernel, no learned parameter. Every emitter is a pure
function of its inputs (same input bytes in, same output bytes out).
"""

from __future__ import annotations

import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from shikong.stpp.emitters import events_to_txy

# Split names the fixed-schema containers understand (dev maps to "val").
WRAPPER_SPLITS = ("train", "val", "test")
_SPLIT_TO_WRAPPER_KEY = {"train": "train", "dev": "val", "test": "test"}

# NSTPP window defaults, matching the earthquakes preprocessor
# (30-day windows, 7-day step).
NSTPP_WINDOW_DAYS = 30
NSTPP_STEP_DAYS = 7


def _require_splits(splits: dict[str, pl.DataFrame], *, where: str) -> None:
    """Raise unless *splits* holds non-empty train/dev/test splits."""
    for name in ("train", "dev", "test"):
        if name not in splits:
            raise ValueError(f"{where}: missing required split {name!r}")
        if splits[name].height == 0:
            raise ValueError(f"{where}: refusing an empty split {name!r}")


def emit_wrapper_npz(
    splits: dict[str, pl.DataFrame],
    *,
    output_dir: str | Path,
    name: str,
) -> list[Path]:
    """Write one ``<name>.npz`` for the DeepSTPP/AutoSTPP wrapper contract.

    Keys ``train`` / ``val`` / ``test`` (mapping: train→train,
    dev→val, test→test; the ``aux`` split is excluded — see the module
    docstring). Each value is an object array holding ONE float64
    ``[n, 3]`` array, columns ``[t_seconds_since_start, x_km, y_km]``
    (the split's full chronological run, via
    :func:`shikong.stpp.emitters.events_to_txy`). The file loads with
    ``np.load(..., allow_pickle=True)`` — the way AutoSTPP's
    ``SlidingWindowDataModule`` loads it.

    Args:
        splits: Split name -> extracted events.
        output_dir: Directory to write into (created if needed).
        name: Dataset name (the ``.npz`` stem).

    Returns:
        Sorted list of written file paths (one file).
    """
    _require_splits(splits, where="emit_wrapper_npz")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for split_name, key in _SPLIT_TO_WRAPPER_KEY.items():
        # Assign into a preallocated object array: np.array([arr],
        # dtype=object) would recurse and turn the inner float64 array
        # into an object-dtype array of Python floats instead.
        seqs = np.empty(1, dtype=object)
        seqs[0] = events_to_txy(splits[split_name])
        payload[key] = seqs
    path = out_dir / f"{name}.npz"
    np.savez(path, **payload)
    return [path]


def window_events_nstpp(
    events: pl.DataFrame,
    *,
    window_days: int = NSTPP_WINDOW_DAYS,
    step_days: int = NSTPP_STEP_DAYS,
) -> dict[str, np.ndarray]:
    """Window extracted events the NSTPP earthquakes-preprocessor way.

    One 30-day window per 7-day step (defaults), overlapping, keyed by
    the window-start date ``YYYYMMDD`` (dataset local time). Each value
    is a float64 ``[n, 3]`` array, columns ``[t_days_since_window_start,
    longitude, latitude]``. Windows are anchored at the day floor of
    the earliest event and start while the start is before the latest
    event. Empty windows are INCLUDED (the preprocessor emits every
    window step regardless of content). The 2011 / Nov–Dec-2010
    exclusions of the Japan preprocessor are NOT replicated — they are
    dataset-specific quirks, not format.

    Args:
        events: Extracted events (as returned by
            :func:`shikong.stpp.emitters.extract_exact_events`), sorted
            by timestamp.
        window_days: Window length in days (must be positive).
        step_days: Step between window starts in days (must be positive
            and no larger than ``window_days``, so the windows cover the
            period without gaps).

    Returns:
        Window key -> float64 ``[n, 3]`` array (possibly empty).

    Raises:
        ValueError: empty input, or non-positive / gapped window
            parameters.
    """
    where = "window_events_nstpp"
    if events.height == 0:
        raise ValueError(f"{where}: refusing an empty frame")
    if window_days <= 0:
        raise ValueError(f"{where}: window_days must be positive, got {window_days}")
    if step_days <= 0:
        raise ValueError(f"{where}: step_days must be positive, got {step_days}")
    if step_days > window_days:
        raise ValueError(
            f"{where}: step_days ({step_days}) must not exceed window_days "
            f"({window_days}); larger steps would leave events in no window"
        )
    timestamps = events.select(pl.col("timestamp")).to_series().to_list()
    t_min = min(timestamps)
    t_max = max(timestamps)
    anchor = t_min.replace(hour=0, minute=0, second=0, microsecond=0)
    lon = events.select(pl.col("longitude")).to_series().to_numpy().astype(np.float64)
    lat = events.select(pl.col("latitude")).to_series().to_numpy().astype(np.float64)
    t_days_all = np.array(
        [(m - t_min).total_seconds() / 86400.0 for m in timestamps], dtype=np.float64
    )
    anchor_offset = (t_min - anchor).total_seconds() / 86400.0

    windows: dict[str, np.ndarray] = {}
    start = anchor
    while start < t_max:
        key = start.strftime("%Y%m%d")
        win_start_days = (start - anchor).total_seconds() / 86400.0
        win_end_days = win_start_days + window_days
        # Event times relative to the anchor, then to the window start.
        rel = t_days_all + anchor_offset
        mask = (rel >= win_start_days) & (rel < win_end_days)
        t_win = rel[mask] - win_start_days
        windows[key] = np.column_stack([t_win, lon[mask], lat[mask]]).astype(np.float64)
        start += timedelta(days=step_days)
    return windows


def emit_nstpp_npz(
    events: pl.DataFrame,
    *,
    output_dir: str | Path,
    name: str,
    window_days: int = NSTPP_WINDOW_DAYS,
    step_days: int = NSTPP_STEP_DAYS,
) -> list[Path]:
    """Write one ``<name>_nstpp.npz`` for the NSTPP earthquakes format.

    Keys are window-start dates ``YYYYMMDD``; values are the float64
    ``[n, 3]`` arrays from :func:`window_events_nstpp`. Loads with
    plain ``np.load(...)`` (no pickled objects).

    Args:
        events: Extracted events (all cleaned events — the file has no
            splits; windows span the whole period).
        output_dir: Directory to write into (created if needed).
        name: Dataset name (the ``.npz`` stem gets a ``_nstpp`` suffix).
        window_days: Window length in days.
        step_days: Step between window starts in days.

    Returns:
        Sorted list of written file paths (one file).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = window_events_nstpp(events, window_days=window_days, step_days=step_days)
    path = out_dir / f"{name}_nstpp.npz"
    # Any-typed so mypy accepts the ** unpacking against numpy's stubs
    # (which also declare allow_pickle: bool in the signature).
    payload: dict[str, Any] = dict(windows)
    np.savez(path, **payload)
    return [path]


def _positional_events(
    events: pl.DataFrame,
    *,
    marked: bool,
) -> list[list[list[float | int]]]:
    """One sequence of positional events for DSTPP/SMASH pickles.

    The sequence is the split's full chronological run. Unmarked:
    ``[t, lng, lat]`` (Python floats; ``t`` = seconds since the split's
    first event). Marked: ``[t, mark, lng, lat]`` (``mark`` = 0-based
    integer codebook index). The loaders insert the gap at index 1
    themselves — Shikong ships raw absolute times, never gaps.
    """
    txy = events_to_txy(events)
    lon = events.select(pl.col("longitude")).to_series().to_list()
    lat = events.select(pl.col("latitude")).to_series().to_list()
    if marked:
        marks = events.select(pl.col("mark")).to_series().to_list()
        sequence = [
            [float(t), int(mark), float(lng), float(la)]
            for (t, _x, _y), mark, lng, la in zip(txy, marks, lon, lat, strict=True)
        ]
    else:
        sequence = [
            [float(t), float(lng), float(la)]
            for (t, _x, _y), lng, la in zip(txy, lon, lat, strict=True)
        ]
    return [sequence]


def emit_dstpp_pickles(
    splits: dict[str, pl.DataFrame],
    *,
    output_dir: str | Path,
    name: str,
) -> list[Path]:
    """Write DSTPP pre-split pickles ``dataset/<name>/data_*.pkl``.

    Files ``data_train.pkl`` / ``data_val.pkl`` / ``data_test.pkl``
    (mapping: train→train, dev→val, test→test; the ``aux`` split is
    excluded). Each pickle is a list with ONE sequence (the split's
    full chronological run) of positional ``[t, lng, lat]`` events
    (``t`` = seconds since the split start; lng/lat in degrees) — the
    raw shape DSTPP's ``app.py`` ``data_loader`` reads, which inserts
    the gap at index 1 itself. Directory layout follows the LOADER's
    expected path (``dataset/``, singular).

    Args:
        splits: Split name -> extracted events.
        output_dir: Directory to write into (created if needed).
        name: Dataset name (subdirectory under ``dataset/``).

    Returns:
        Sorted list of written file paths.
    """
    _require_splits(splits, where="emit_dstpp_pickles")
    out_dir = Path(output_dir) / "dataset" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for split_name, key in _SPLIT_TO_WRAPPER_KEY.items():
        payload = _positional_events(splits[split_name], marked=False)
        path = out_dir / f"data_{key}.pkl"
        with path.open("wb") as f:
            pickle.dump(payload, f)
        written.append(path)
    return sorted(written)


def emit_smash_pickles(
    splits: dict[str, pl.DataFrame],
    *,
    output_dir: str | Path,
    name: str,
) -> list[Path]:
    """Write SMASH pre-split pickles ``dataset/<name>/data_*.pkl``.

    Files ``data_train.pkl`` / ``data_val.pkl`` / ``data_test.pkl``
    (mapping: train→train, dev→val, test→test; the ``aux`` split is
    excluded). Each pickle is a list with ONE sequence (the split's
    full chronological run) of positional ``[t, mark, lng, lat]``
    events (``t`` = seconds since the split start; ``mark`` = 0-based
    integer codebook index; lng/lat in degrees) — the raw dim=3 shape
    SMASH's ``app.py`` ``data_loader`` reads (``--dim 3``, as in
    ``scripts/train_earthquake.sh``), which inserts the gap at index 1
    and log-transforms it by default. Directory layout follows the
    LOADER's expected path (``dataset/``, singular) — the repo ships
    ``datasets/`` (plural), which the loader cannot open; see the
    design note.

    Args:
        splits: Split name -> extracted events.
        output_dir: Directory to write into (created if needed).
        name: Dataset name (subdirectory under ``dataset/``).

    Returns:
        Sorted list of written file paths.
    """
    _require_splits(splits, where="emit_smash_pickles")
    out_dir = Path(output_dir) / "dataset" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for split_name, key in _SPLIT_TO_WRAPPER_KEY.items():
        payload = _positional_events(splits[split_name], marked=True)
        path = out_dir / f"data_{key}.pkl"
        with path.open("wb") as f:
            pickle.dump(payload, f)
        written.append(path)
    return sorted(written)


def emit_deepstpp_csv(
    splits: dict[str, pl.DataFrame],
    *,
    output_dir: str | Path,
) -> list[Path]:
    """Write one ``{split}_deepstpp.csv`` per split (DeepSTPP file shape).

    Columns ``[x, y, t]`` (``x_km, y_km, t_seconds_since_start``),
    comma-separated, ``%f`` — the shape of
    ``synthetic.py::save`` in ``Rose-STL-Lab/DeepSTPP``, the repo's
    only file-backed pipeline. All four splits are written (aux
    included — this is a per-split export like the catalog CSV, not a
    fixed-schema container). The wrapper ``.npz`` (see
    :func:`emit_wrapper_npz`) is the model-facing contract; this CSV is
    the secondary file-shape export.

    Args:
        splits: Split name -> extracted events.
        output_dir: Directory to write into (created if needed).

    Returns:
        Sorted list of written file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(splits):
        txy = events_to_txy(splits[name])
        path = out_dir / f"{name}_deepstpp.csv"
        np.savetxt(path, txy[:, [1, 2, 0]], delimiter=",", fmt="%f")
        written.append(path)
    return sorted(written)
