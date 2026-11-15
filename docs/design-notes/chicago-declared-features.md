# Declared Chicago features — removed in phase 0, planned for phase 1

**Date:** 2026-11-15 (phase 0)
**Status:** design note — items 1–2 implemented in phase 1 (see below);
item 3 (`partition_by`) still open.

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
  `FeatureConfig` (so the intent stays typed). Phase 1 **implements** them
  (`CategoryCountsStep`, `TemporalLagStep`, `MovingAverageStep` plus the
  dependent groups — see "What phase 1 implemented" below), and
  `FeatureConfig.require_implemented()` now enforces the cross-key rules
  (spatial lag / trend need windows; recency / zero features / EWMA need
  a completed panel) instead of rejecting the keys outright.
  `chicago_crime_v1.build_pipeline()` calls it.
- `grid.buffer_m != 0` and any `context.*.enabled: true` raise a blocking
  error in `build_pipeline()`; a non-empty `output.partition_by` raises a
  blocking error in the CLI `run` command. (The `validate` command only
  checks the schema — it does not run these recipe-level checks.)
- Tests: `tests/core/test_recipe_features_config.py` proves each of these
  rejections, and that the shipped YAML loads and runs.

## What phase 1 implemented (generic, not Chicago-specific)

Items 1–2 below landed in phase 1 as generic core steps
(`core/steps/` + registry), wired into `chicago_crime_v1` via the typed
`FeatureConfig` and the shipped YAML. Item 3 is still open.

1. **Counts by category** (`categorical_features`): **implemented** —
   `CategoryCountsStep` (`core/steps/categorical.py`) produces
   `count_<col>_<value>`, `share_<col>_<value>` and `<col>_entropy` per
   (unit, time), with a top-N guardrail (`categorical_top_n`, default 10)
   plus an `"other"` bucket; category values are sanitized to valid column
   names (runs of non-alphanumerics → `_`); nulls fall into `"other"`.
   Chicago: `primary_type` (30 values) and `location_description`
   (113 values) → 11 count columns each at top_n=10; the `"other"` bucket
   is genuinely used (12,149 rows for `location_description`). Shares are
   null when the bin total is 0; entropy is Shannon (`-sum(p·ln p)`), null
   when the total is 0. `ChicagoCountStep` carries the columns through the
   aggregation with `first()` (they are already per cell/bin), and
   `CompletePanelStep` zero-fills the `count_*` columns for generated
   pairs (`value_prefixes=["count"]`) while shares/entropy stay null there.
2. **Rolling windows** (`aggregation_windows`): **implemented** — the
   panel is completed with explicit zeros first (`CompletePanelStep`,
   opt-in per recipe, `min_periods`), then `TemporalLagStep` (exact
   elapsed lags, duration mode) and `MovingAverageStep` (trailing,
   `closed="left"`) run over the declared windows (Chicago: 1d/7d/30d →
   `count_lag_1/7/30`, `count_ma_1/7/30`). With 6 h bins, 1d = 4 bins,
   7d = 28, 30d = 120. Phase 1 also adds the dependent groups on top of
   the same windows: EWMA (`EWMAMeanStep`, `count_ewma_7d`), trend
   (`TrendStep`, `count_ma_1_vs_count_ma_30`), queen spatial lag of the
   shortest lag (`SpatialLagStep`, `count_lag_1_spatial_lag`), time since
   last event + burstiness (`count_time_since_last_d`,
   `count_burstiness_28d`), zero features (`count_active_7d`,
   `count_share_nonzero_28d`), and cyclical calendar encodings
   (`CalendarEncodingStep`, `hour_sin/cos`, `dow_sin/cos`,
   `month_sin/cos`). Causality convention and window policies:
   `docs/design-notes/phase1-causality.md`. Replay proof (Q1 2024):
   1,908,088 rows, sum `count` = 60,235, hand-verified lag / rolling
   mean / spatial lag / EWMA / trend / entropy (see `make audit`,
   `audit/phase-1/replay_chicago.json`).
3. **Partitioned output** (`partition_by`): **still open** — write one
   file per partition value (year/month) instead of a single file.
   Smaller scope, CLI-level. Still a blocking error in the CLI.

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
