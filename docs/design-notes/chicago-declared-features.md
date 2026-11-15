# Declared Chicago features — removed in phase 0, planned for phase 1

**Date:** 2026-11-15 (phase 0)
**Status:** design note — nothing here is implemented yet.

## What was removed, and why

Up to commit `9903a6c`, `configs/recipes/chicago_crime_v1.yaml` declared three
features that `build_pipeline()` never read:

| Declared key | What a user would expect | What the recipe actually produced |
|---|---|---|
| `features.categorical_features: [primary_type, location_description]` | Per-category counts per cell/bin, e.g. `count_THEFT`, `count_BATTERY` | Nothing — 12 output columns, no per-category column |
| `features.aggregation_windows: ["1d", "7d", "30d"]` | Rolling mean/sum of `count` over 1/7/30 days per cell | Nothing — no rolling column at all |
| `output.partition_by: [year, month]` | Output split into one file per year/month | Nothing — the CLI writes a single file |
| `grid.buffer_m: 0` | A buffer around the grid | Nothing — no reader anywhere in the code |

The audit of 2026-11-15 (objective 27, **non atteint**) confirmed the silence:
`RecipeConfig.features` was an untyped `dict[str, Any]` (`core/schema.py:242`)
and `build_pipeline()` only read `grid` and `temporal`. Phase 0 removed the
four keys from the YAML instead of letting the recipe keep promising them.

## How phase 0 prevents the silence from coming back

- `features` is now a typed, strict Pydantic model (`FeatureConfig`,
  `core/schema.py`, `extra="forbid"` + a validator that lists the supported
  keys). An unknown key raises a `ValidationError` at load time.
- `categorical_features` and `aggregation_windows` are **declared** in
  `FeatureConfig` (so the intent stays typed) but **not implemented**: any
  recipe that reads them must call `FeatureConfig.require_implemented()`,
  which raises a blocking `ValueError` when the lists are non-empty.
  `chicago_crime_v1.build_pipeline()` calls it.
- `grid.buffer_m != 0` and any `context.*.enabled: true` raise a blocking
  error in `build_pipeline()`; a non-empty `output.partition_by` raises a
  blocking error in the CLI `run` command. (The `validate` command only
  checks the schema — it does not run these recipe-level checks.)
- Tests: `tests/core/test_recipe_features_config.py` proves each of these
  rejections, and that the shipped YAML loads and runs.

## What phase 1 must implement (generic, not Chicago-specific)

All three features are generic concepts — any `EventFrame` with a timestamp,
a grid/cell id and an optional category column. They must land in the core
(`core/steps/` + registry), never as Chicago-only code:

1. **Counts by category** (`categorical_features`): one column per distinct
   category value, per cell/bin (e.g. `count_THEFT`). Guardrails needed
   before implementing: top-N / minimum-frequency threshold, because
   cardinality explodes for columns like `location_description`; column
   names must be escaped; the output schema/provenance must be updated
   (the old `core/features.py::aggregate_by_category` did none of this —
   see the phase-0 triage in `docs/audit/known-limitations.md`).
2. **Rolling windows** (`aggregation_windows`): rolling mean/sum of `count`
   over 1d/7d/30d per cell. With 6 h bins, 1d = 4 bins, 7d = 28, 30d = 120.
   **Prerequisite, to decide first:** the cell×bin panel must be completed
   with observed zeros before any rolling window is computed — on the
   Chicago output only 2.96 % of the 1,908,088 possible pairs are observed,
   so rolling over the sparse output would be misleading. Window alignment
   (trailing/causal only, never centred) and minimum-periods policy must
   be explicit.
3. **Partitioned output** (`partition_by`): write one file per partition
   value (year/month) instead of a single file. Smaller scope, CLI-level.

## Other keys that are still silently ignored (known, not fixed in phase 0)

- `DatasetConfig.quality.*` (`min_year`, `max_missing_coords`,
  `outlier_detection` in `configs/datasets/chicago_crime_example.yaml`):
  never read by `mapping.py` — cleaning is fixed in `clean_chicago_data`.
  Recorded in `docs/audit/known-limitations.md`; typing `DatasetConfig`
  is a later phase-0 follow-up, not done here.
- `grid` / `temporal` / `context` / `output` sections are still
  `dict[str, Any]` in `RecipeConfig` (only `features` was typed in
  phase 0). Typos inside those sections can still fall back to defaults
  silently. Typing them is the natural next honesty step.
- `TransformCRSStep(target_crs="EPSG:26971")` is hardcoded in the recipe,
  not taken from `DatasetConfig.crs`.
