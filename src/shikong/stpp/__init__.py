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
"""

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
    "SplitBoundary",
    "build_gatech_sequence",
    "build_manifest",
    "build_tick_payload",
    "chronological_event_splits",
    "emit_catalog",
    "emit_gatech",
    "emit_tick",
    "events_to_txy",
    "extract_exact_events",
    "write_manifest",
]
