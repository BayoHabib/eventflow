"""Exact-event emitters for spatio-temporal point processes (STPP).

This package prepares exact ``(time, x, y, mark)`` event sequences for
STPP models. It emits data only — no fitted model, no likelihood, no
intensity (``lambda``) is computed or stored anywhere here.

Formats emitted (see ``docs/design-notes/stpp-emitters.md`` for the
full design note and the sources each format was read from):

- Gatech / THP / EasyTPP: one ``{split}.pkl`` (+ ``{split}.json``) per
  split, ``{dim_process, <split>: [sequence]}``.
- EarthquakeNPP-style catalog CSV (one row per event, projected ``x``,
  ``y`` in km) plus a per-split ``(t, x, y)`` NumPy array.
- tick-style pickle: per-node timestamp lists, one realization per split.
- Deep-learning STPP formats (see ``docs/design-notes/stpp-dl-data-prep.md``):
  the DeepSTPP/AutoSTPP shared sliding-window ``.npz`` (keys
  ``train``/``val``/``test``, lists of ``[t, x, y]`` arrays), the NSTPP
  windowed ``.npz`` (keys ``YYYYMMDD``, ``[t_days, lon, lat]``), the
  DSTPP/SMASH pre-split positional pickles, and the DeepSTPP CSV
  ``[x, y, t]`` file shape.
"""

from shikong.stpp.dl_formats import (
    NSTPP_STEP_DAYS,
    NSTPP_WINDOW_DAYS,
    WRAPPER_SPLITS,
    emit_deepstpp_csv,
    emit_dstpp_pickles,
    emit_nstpp_npz,
    emit_smash_pickles,
    emit_wrapper_npz,
    window_events_nstpp,
)
from shikong.stpp.emitters import (
    ChronologicalSplits,
    SplitBoundary,
    build_gatech_sequence,
    build_manifest,
    build_tick_payload,
    chronological_event_splits,
    emit_catalog,
    emit_gatech,
    emit_tick,
    events_to_txy,
    extract_exact_events,
    write_manifest,
)

__all__ = [
    "ChronologicalSplits",
    "NSTPP_STEP_DAYS",
    "NSTPP_WINDOW_DAYS",
    "SplitBoundary",
    "WRAPPER_SPLITS",
    "build_gatech_sequence",
    "build_manifest",
    "build_tick_payload",
    "chronological_event_splits",
    "emit_catalog",
    "emit_deepstpp_csv",
    "emit_dstpp_pickles",
    "emit_gatech",
    "emit_nstpp_npz",
    "emit_smash_pickles",
    "emit_tick",
    "emit_wrapper_npz",
    "events_to_txy",
    "extract_exact_events",
    "window_events_nstpp",
    "write_manifest",
]
