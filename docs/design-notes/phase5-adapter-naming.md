# Phase 5 — Adapter output naming decisions

Date: 2026-11-15. Scope: naming-only change to the five adapter outputs
(`TableOutput`, `SequenceOutput`, `RasterOutput`, `GraphOutput`,
`StreamOutput`). No adapter computation was changed; no existing field was
renamed or removed. Implementation: `src/eventflow/core/adapters/`
(`sequence.py`, `stream.py`, `raster.py`, `graph.py`, `table.py`).
Tests: `tests/core/test_adapter_aliases.py` (21 tests).

## Decision 1 — Aliases are read-only properties, old names keep working

| Old name | New alias | Rationale |
|---|---|---|
| `SequenceOutput.masks` | `attention_mask` | HuggingFace/PyTorch convention; the name downstream users expect. |
| `StreamOutput.inter_times` | `inter_event_times` | Full, unambiguous name. |
| `StreamOutput.states` | `state_features` | Consistent `_features` suffix with `node_features`, `feature_names`. |

Implementation: plain read-only `@property` on the (non-frozen) dataclasses,
returning the underlying field — the alias is the *identical array object*
(`alias is field`), not a copy. Verified by experiment before implementing:

- Properties do not collide with dataclass fields (the new names are not
  fields), do not appear in `dataclasses.fields()` / `asdict()` / `repr()`,
  and do not affect `__eq__` (which only compares fields).
- Pickling and `copy.deepcopy` work unchanged; assigning to an alias raises
  `AttributeError` (read-only).
- The same pattern also works on frozen dataclasses, should any output
  become frozen later.

Note on `__eq__`: comparing two outputs with `==` raises `ValueError`
("truth value of an array is ambiguous") because dataclass equality compares
ndarray fields element-wise. That is **pre-existing** behaviour, unrelated to
the properties, and was not changed. Tests compare arrays with
`np.testing.assert_array_equal` instead.

## Decision 2 — Serialization keys: new names added *alongside*, never replacing

Everywhere the old name appears as a key, the new key is written next to it:

- `to_torch()` dicts (note: the method is `to_torch()`, there is no
  `to_pytorch()`): `"masks"` + `"attention_mask"`,
  `"states"` + `"state_features"`, `"inter_times"` + `"inter_event_times"`.
- `AdapterMetadata.shapes` and `dtypes` (via `get_metadata()`): same
  alongside-keys policy. `dtypes` gains `"attention_mask": "bool"` and
  `"state_features": <dtype>`; `inter_times` was never in `dtypes`, so no
  key was added for it there.
- NumPy (`.npz`) and JSON files: both keys stored (duplicate arrays in the
  file; the cost is one extra array per alias and is accepted for
  self-describing files).
- PyTorch (`.pt`) files: both keys, via `to_torch()`.

Deserializers accept **either** key and prefer the new one, so files written
before this change (old keys only) still load and the aliases work on the
loaded object. Covered by tests that hand-build pre-alias files.

Deliberately **not** touched (storage-internal names, not attribute names):

- Stream Parquet columns (`"inter_time"` column) and the Sequence Arrow
  flat columns (`"valid"` for masks): renaming a stored column would break
  existing files for no user-facing benefit, and there is no Parquet/Arrow
  deserialize path for these adapters anyway. Documented here, not changed.
- `GraphOutput.to_torch_geometric()` keys (`"x"`, `"edge_index"`,
  `"edge_attr"`): PyTorch Geometric's own convention; not ours to rename.

## Decision 3 — Uniform `.data` accessor: added as read-only property

Each output now exposes its main payload as `.data`, alongside its
descriptive per-adapter name (which is kept — it is used in tests and docs
and is the clearer name in context):

| Output | Descriptive name (kept) | `.data` points at |
|---|---|---|
| `TableOutput` | `data` (already the field) | — (no change) |
| `SequenceOutput` | `sequences` | `sequences` |
| `RasterOutput` | `raster` | `raster` |
| `GraphOutput` | `node_features` | `node_features` |
| `StreamOutput` | `states` | `states` |

This was clean to implement: the outputs are non-frozen dataclasses, the
properties return the identical object, and (per the experiment in
Decision 1) `__eq__`, pickling, `asdict()` and `repr()` are unaffected.
`.data` is read-only on the four outputs where it is a property;
`TableOutput.data` remains a plain (assignable) field, which is the one
asymmetry — accepted, since making the field read-only would be a breaking
change to `TableOutput` construction.

## Decision 4 — Naming convention: `feature_names` is canonical; the other two stay

Survey of actual usage in `src/`:

- `AdapterMetadata.feature_names` — the cross-adapter metadata field; every
  adapter's `get_metadata()` fills it (Raster and Stream map their local
  names into it).
- `TableOutput` / `SequenceOutput` / `GraphOutput` use `feature_names`
  (Graph additionally has `edge_feature_names`).
- `StreamOutput.state_names` — names of the state-vector dimensions.
- `RasterOutput.channel_names` — names of the raster channels.

Settled convention: **`feature_names` is the canonical, cross-adapter name**
for "the names of the feature dimensions of the main payload". `state_names`
and `channel_names` are **not** renamed: they are the domain-accurate
spelling of the same concept in their modality (a continuous-time state
vector; CNN channels), and both are already surfaced as `feature_names` in
`AdapterMetadata`, so generic code never needs to know the local spelling.
Renaming them would churn public names for zero gain and would need a
deprecation path; the convention is therefore documented here rather than
enforced by renaming.

## Decision 5 — Config flags left as-is (with one real inconsistency found)

`SequenceAdapterConfig.return_masks` and
`StreamAdapterConfig.return_inter_times` stay exactly as-is: the flags are
named after the *old* attribute names, but they are config flags, not output
attributes, and renaming them would break existing configs for no benefit.

Real inconsistency found while checking them: **neither flag is actually
consulted by `convert()`** — masks and inter-event times are always computed
and returned regardless of the flag value. This is pre-existing behaviour
and out of scope for a naming-only change (honouring the flags would change
behaviour: the output fields are required dataclass fields, so
`return_masks=False` has no clean "no masks" representation). Left open:
either wire the flags through (needs an `Optional` masks field) or remove
them — a follow-up decision, not taken here.

## What was deliberately NOT renamed

- No existing field, config flag, serialization key, or method was renamed
  or removed. Everything above is purely additive.
- `state_names` / `channel_names`: kept (Decision 4).
- `return_masks` / `return_inter_times`: kept (Decision 5).
- Parquet/Arrow storage column names and PyTorch Geometric keys: kept
  (Decision 2).
- `GraphOutput.node_features` was not renamed to `data`, and no adapter's
  descriptive payload name was replaced by `data` — `.data` is an alias,
  not a replacement (Decision 3).

## Verification

- New tests: `tests/core/test_adapter_aliases.py`, 21 passed, 2 skipped
  (the two `to_torch()` key tests skip because `torch` is not installed in
  the dev venv; they run wherever torch is present).
- Full suite after the change: 1,012 tests collected (991 baseline + 21 new),
  all passing, coverage ≥ 86% — see the Phase 5 exit-criteria note for the
  final numbers.
- `ruff check`, `black --check` (line-length 100), and
  `mypy src/eventflow/core/adapters` are clean on all touched files.
