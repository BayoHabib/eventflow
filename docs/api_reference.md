# Eventflow API Reference

This document provides detailed API documentation for the Eventflow library.

> **Executed examples.** Every code example in this file is executed by
> `tests/test_docs_examples.py` (phase 5 rule: no example is claimed from
> reading). The examples share one small fixture, a 3×3 completed panel
> defined once below; each section states the values its test asserts.
> Numbers from the real-data replays (Chicago, Beijing, GeoLife) are
> labelled as replay measurements and point at the design note or audit
> evidence that produced them.

## Causality convention (all feature steps)

A feature row at bin `t` is for **predicting the count in bin `t + 1`**.
No feature may use data from bins after `t`. Within that bound, each
step documents its exact window policy in its docstring and in its
provenance metadata; the full table is in
[phase1-causality.md](design-notes/phase1-causality.md). Leakage tests pin
the guarantee: changing data at bins `> t` must not change any feature
at bins `<= t`.

## Shared fixture for the examples

```python
import polars as pl
from datetime import datetime, timedelta

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema

BIN0 = datetime(2024, 1, 1)

def make_panel(counts_by_unit):
    """3x3 completed panel, 300 m spacing, 4 daily bins from 2024-01-01.

    counts_by_unit maps grid_id (row * 3 + col) to one count per bin.
    Schema lon/lat columns are the projected-metre centres x/y
    (EPSG:26971), so the KDE steps can use them directly.
    """
    rows = {
        "grid_id": [], "time_bin": [], "grid_row": [], "grid_col": [],
        "x": [], "y": [], "count": [],
    }
    bins = [BIN0 + timedelta(days=i) for i in range(4)]
    for bin_idx, bin_start in enumerate(bins):
        for grid_id in range(9):
            rows["grid_id"].append(grid_id)
            rows["time_bin"].append(bin_start)
            rows["grid_row"].append(grid_id // 3)
            rows["grid_col"].append(grid_id % 3)
            rows["x"].append(300.0 * (grid_id % 3))
            rows["y"].append(300.0 * (grid_id // 3))
            rows["count"].append(counts_by_unit[grid_id][bin_idx])
    schema = EventSchema(
        timestamp_col="time_bin", lat_col="y", lon_col="x",
        numeric_cols=["count"], categorical_cols=["grid_id"],
    )
    metadata = EventMetadata(dataset_name="docs-panel", crs="EPSG:26971", time_bin="1d")
    return EventFrame(pl.LazyFrame(rows), schema, metadata)
```

## Core Module

### eventflow.core.schema

#### EventSchema

Describes the structure of event data. Either (`lat_col` and `lon_col`)
or `geometry_col` must be provided (validated at construction).

```python
class EventSchema(BaseModel):
    timestamp_col: str
    lat_col: str | None = None
    lon_col: str | None = None
    geometry_col: str | None = None
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []
```

#### ContextSchema

Describes the structure of context data sources.

```python
class ContextSchema(BaseModel):
    timestamp_col: str | None = None
    spatial_col: str | None = None
    attribute_cols: list[str]
```

### eventflow.core.event_frame

#### EventFrame

Central abstraction for event data: a Polars `LazyFrame` plus schema,
metadata and feature provenance.

```python
class EventFrame:
    def __init__(
        self,
        lazy_frame: pl.LazyFrame,
        schema: EventSchema,
        metadata: EventMetadata,
    ): ...

    def with_lazy_frame(self, lazy_frame: pl.LazyFrame) -> EventFrame:
        """Return a copy sharing schema/metadata with a new lazy frame."""
        ...

    def with_metadata(self, **updates) -> EventFrame: ...
    def with_schema(self, **updates) -> EventFrame: ...

    def register_feature(
        self, name: str, info: dict, provenance: FeatureProvenance | None = None
    ) -> EventFrame:
        """Record a produced feature (and its provenance) on the frame."""
        ...

    def collect(self) -> pl.DataFrame:
        """Materialize the lazy frame."""
        ...

    async def collect_async(self) -> pl.DataFrame: ...
    def head(self, n: int = 5) -> pl.DataFrame: ...
    def describe(self) -> pl.DataFrame: ...
    def select(self, *exprs) -> EventFrame: ...
    def filter(self, *predicates) -> EventFrame: ...
    def with_columns(self, *exprs, **named_exprs) -> EventFrame: ...
    def sort(self, *args, **kwargs) -> EventFrame: ...
    def count(self) -> int: ...
```

### eventflow.core.registry

#### FeatureStepRegistry

Registry for discovering and instantiating reusable pipeline steps.
`get_default_registry()` (in `eventflow.core.steps.registration`)
returns a registry with all built-in steps; the names are listed in
[Step registry names](#step-registry-names) below.

```python
from eventflow.core.steps.registration import get_default_registry

registry = get_default_registry()

# Instantiate directly (params validated through the step's config model)
step = registry.create("complete_panel", params={})

# Or build a pipeline from config-style definitions
pipeline = registry.build_pipeline([
    {"name": "complete_panel", "params": {}},
    {"name": "fill_null", "params": {"value_cols": ["count_lag_1"]}},
])
```

Each registry entry is represented by a `StepSpec` that tracks tags,
documentation, and optional configuration schema metadata.

### eventflow.core.spatial

Functions for spatial operations:

- `transform_crs(event_frame, target_crs)`: Transform coordinates to a
  target CRS (updates the schema's active coordinate columns).
- `create_grid(bounds, size_m, crs="EPSG:4326", *, origin=None)`:
  Generate canonical half-open metric cells covering `bounds`.
- `assign_to_grid(event_frame, grid)`: Assign events to cells of a
  validated square-cell reference (rows outside it get null `grid_id`).
- `assign_to_zones(event_frame, zones)`: Assign events to zones.
- `compute_distances(event_frame, points)`: Distances to points of
  interest.

### eventflow.core.temporal

Functions for temporal operations:

- `extract_temporal_components(event_frame, components)`: Extract hour,
  day of week, month, etc. from the timestamp.
- `create_time_bins(event_frame, bin_size, bin_col="time_bin")`: Create
  time bins.
- `align_temporal(event_frame, context_frame, context_timestamp_col,
  strategy="nearest", window=None)`: Align events with context data.
- `compute_time_deltas(event_frame, ...)`: Time deltas between events.
- `create_temporal_windows(event_frame, ...)`: Temporal windows.

### eventflow.core.features

Free functions for feature engineering. Most of this module is
**experimental**; recipes use the tested Steps instead:

- `compute_ratios(event_frame, numerator_col, denominator_col)`: Ratios
  (supported, tested; zero denominator yields null).
- `aggregate_by_category(event_frame, group_by, category_col)`: Per-category
  counts — EXPERIMENTAL, untested, superseded for recipes by the tested
  `CategoryCountsStep` (top-N + `"other"` bucket).
- `encode_categorical(event_frame, col, method)`: Categorical encoding
  (onehot/ordinal only) — EXPERIMENTAL, untested, and it learns its
  vocabulary over the whole frame, so it must not be used across
  train/test (see [phase4-fitted-statistics.md](design-notes/phase4-fitted-statistics.md)).
- `normalize_features(event_frame, cols, method)`: Z-score / min-max —
  EXPERIMENTAL, untested (whole-frame statistics: leakage risk, see the
  module docstring).

Removed in phase 0 (use the tested Steps instead): `aggregate_counts` →
`GridAggregationStep`, `moving_window_aggregation` → `MovingAverageStep`,
`lag_features` → `TemporalLagStep`.

## Pipeline Steps

All steps inherit from `eventflow.core.pipeline.Step`, validate their
parameters at construction (a bad parameter fails there, never silently
at `run()` time), and register every produced column with provenance.
Steps that learn statistics from data implement the `FittableStep`
protocol (`fit(event_frame) -> Self`, `is_fitted -> bool`) — see
[Fit / freeze](#fit--freeze-fittablestep).

### eventflow.core.steps.temporal

#### TimeBinsStep

```python
class TimeBinsStep(Step):
    def __init__(self, bin_size: str = "1h", bin_col: str = "time_bin") -> None: ...
```

Bins the schema timestamp column into `bin_col` and records
`metadata.time_bin`, which the duration-based steps use to convert
days into bins.

#### ExtractTemporalComponentsStep

```python
class ExtractTemporalComponentsStep(Step):
    def __init__(self, components: Sequence[str] | None = None) -> None: ...
```

Extracts calendar components (default set includes hour, day of week,
month, …) from the timestamp into new columns. Unknown component names
raise `ValueError` at construction.

#### TemporalLagStep

```python
class TemporalLagStep(Step):
    def __init__(
        self,
        value_cols: Sequence[str],
        lag_periods: Sequence[int] | None = None,
        time_unit: str = "1d",
        group_cols: Sequence[str] | None = None,
        mode: Literal["duration", "rows"] = "duration",
    ) -> None: ...
```

Produces `{value_col}_lag_{period}`. In duration mode the lag is an
exact elapsed time (`t - lag`); a missing value there stays null. In
rows mode the lag is a row shift within each group. A lag before the
panel start is null by construction — see `FillNullStep` for the
boundary policy.

#### MovingAverageStep

```python
class MovingAverageStep(Step):
    def __init__(
        self,
        value_cols: Sequence[str],
        windows: Sequence[int] | None = None,
        time_unit: str = "1d",
        group_cols: Sequence[str] | None = None,
        center: bool = False,
        mode: Literal["duration", "rows"] = "duration",
        closed: Literal["left", "right", "both", "none"] = "left",
    ) -> None: ...
```

Produces `{value_col}_ma_{window}`. Duration mode is a trailing mean
(`closed="left"` by default: strictly before `t`). In rows mode
`closed` is ignored and the window includes the current row.

#### CalendarEncodingStep

```python
class CalendarEncodingStep(Step):
    def __init__(
        self,
        encoding_type: str = "cyclical",
        components: Sequence[str] | None = None,
    ) -> None: ...
```

Cyclical (`sin`/`cos` per component) or ordinal encoding of calendar
components. Unknown *components* raise `ValueError` at construction.

> **Known limitation (issue #41, still open):** `encoding_type="onehot"`
> — and any other unknown encoding type — is accepted by the
> constructor but produces **no columns at all**, silently. Only
> `"cyclical"` and `"ordinal"` produce output. Do not use `"onehot"`.

#### FillNullStep

```python
class FillNullStep(Step):
    def __init__(self, value_cols: Sequence[str], fill_value: float | int = 0) -> None: ...
```

Fills nulls in named numeric columns with a constant. This is a
boundary policy made explicit, not a silent imputation: use it only
where a null means exactly "absent, counts as the fill value". The
phase-1 use case is a lag before the panel start (no data exists
before the observation window, so the lagged count is treated as 0).

```python
from eventflow.core.steps import FillNullStep
from eventflow.core.steps.temporal import TemporalLagStep

frame = make_panel({u: [1.0, 2.0, 3.0, 4.0] for u in range(9)})
frame = TemporalLagStep(
    value_cols=["count"], lag_periods=[1], time_unit="1d", group_cols=["grid_id"]
).run(frame)
filled = FillNullStep(value_cols=["count_lag_1"], fill_value=0).run(frame).collect()
# count_lag_1 is 0 on the first bin of every unit (was null), never null after.
assert filled["count_lag_1"].null_count() == 0
```

#### InterArrivalStep

```python
class InterArrivalStep(Step):
    def __init__(
        self,
        group_cols: Sequence[str] | None = None,
        output_col: str = "inter_arrival_seconds",
        include_stats: bool = True,
        stat_windows: Sequence[int] | None = None,
    ) -> None: ...
```

Inter-arrival times between consecutive events, with optional rolling
statistics.

#### RecencyWeightStep

```python
class RecencyWeightStep(Step):
    def __init__(self, half_life: float = 7.0, time_unit: str = "1d", ...) -> None: ...
```

Exponential recency weights decaying from a reference time.

### eventflow.core.steps.panel

#### CompletePanelStep

Completes a sparse unit × time panel with explicit zeros, inside the
observation support only: the units kept are those observed in at
least `min_periods` distinct time bins, and the time axis runs from
the earliest to the latest observed bin. Pairs outside that support
are not rows — they are never silently invented. The step adds no
information: it only makes absence explicit, so downstream windows
and lags see a zero where nothing was recorded instead of a gap.

```python
class CompletePanelStep(Step):
    def __init__(
        self,
        unit_col: str = "grid_id",
        time_col: str | None = None,
        bin_size: str | None = None,
        value_cols: Sequence[str] | None = None,
        measure_cols: Sequence[str] | None = None,
        value_prefixes: Sequence[str] | None = None,
        min_periods: int = 1,
        status_col: str = "observation_status",
    ) -> None: ...
```

- `value_cols` (default `["count"]`): numeric columns zero-filled for
  generated pairs. `value_prefixes` zero-fills further columns matched
  by name prefix (e.g. `"count_"` for per-category count columns).
- `measure_cols`: numeric measurement columns carried through the
  (unit, time) join and **never zero-filled** — a missing reading is
  not a zero reading. Generated pairs get null and an observed null
  stays null, so the NA count per measurement column is
  `NA_before + n_generated_pairs`. Declaring a measurement in
  `value_cols` (or matching it with a `value_prefixes` entry) is a
  hard error.
- Every output row carries `observation_status = "observed"`:
  generated pairs are in-support by construction.
- Columns other than unit/time/value/measure are classified per unit:
  constant within a unit means a static attribute, carried to every
  pair; varying within a unit (e.g. a share) is joined on (unit,
  time), so observed pairs keep their value and generated pairs get
  null — never silently collapsed with `first()`.
- Requires an already-binned time column (e.g. the `time_bin` from
  `TimeBinsStep`) and `bin_size` (explicit, or `metadata.time_bin`).
  Duplicate (unit, time) pairs are rejected: aggregate first.

```python
from eventflow.core.steps import CompletePanelStep

# Sparse panel: 7 observed rows, 3 units x 4 daily bins; pm25 has 1 null.
rows = [
    (1, BIN0, 5, 20.0),
    (1, BIN0 + timedelta(days=2), 7, None),
    (2, BIN0 + timedelta(days=1), 3, 15.0),
    (3, BIN0, 1, 12.0),
    (3, BIN0 + timedelta(days=1), 2, 11.0),
    (3, BIN0 + timedelta(days=2), 3, 10.0),
    (3, BIN0 + timedelta(days=3), 4, 9.0),
]
sparse = EventFrame(
    pl.LazyFrame({
        "grid_id": [r[0] for r in rows],
        "time_bin": [r[1] for r in rows],
        "count": [r[2] for r in rows],
        "pm25": [r[3] for r in rows],
        "latitude": [41.8781] * len(rows),
        "longitude": [-87.6298] * len(rows),
    }),
    EventSchema(
        timestamp_col="time_bin", lat_col="latitude", lon_col="longitude",
        numeric_cols=["count", "pm25"],
    ),
    EventMetadata(dataset_name="docs-sparse", crs="EPSG:4326", time_bin="1d"),
)
result = CompletePanelStep().run(sparse).collect()
# 12 rows (3 units x 4 bins); 5 generated pairs with count == 0.
assert result.height == 12
assert result.filter(pl.col("count") == 0).height == 5
# Measurements are never zero-filled: 1 observed null + 5 generated = 6 nulls.
assert result["pm25"].null_count() == 6
```

On the Chicago replay (Q1 2024, `configs/recipes/chicago_crime_v1.yaml`)
the step completes 5,242 cells × 364 bins of 6 h = 1,908,088 rows with
1,851,597 explicit-zero rows and 0 null counts; the sum of `count`
stays 60,235 (see [phase2-spacetime-v2.md](design-notes/phase2-spacetime-v2.md)).

### eventflow.core.steps.dynamics

Dynamics steps operate on the completed panel (one row per unit × time
bin, explicit zeros, never null counts). Window policies per step:

| Step | Window policy |
|---|---|
| `EWMAMeanStep` | Includes the current bin (standard EWMA, `adjust=True`). |
| `TrendStep` | Ratio of two existing rolling-mean columns; no new look-ahead. |
| `TimeSinceLastEventStep` | Look-back includes the current bin (0.0 when it has an event). |
| `BurstinessStep` | Closed-left (strictly before `t`). |
| `ZeroFeaturesStep` | Closed-right (includes the current bin). |

All are allowed under the causality convention because the feature row
at `t` predicts the count in bin `t + 1`.

#### EWMAMeanStep

```python
class EWMAMeanStep(Step):
    def __init__(
        self,
        value_cols: Sequence[str],
        halflife_days: float = 7.0,
        group_cols: Sequence[str] | None = None,
        min_periods: int = 1,
    ) -> None: ...
```

Produces `{value_col}_ewma_{halflife}d`. The half-life in days is
converted to bins using `metadata.time_bin` (so 7 d on a daily panel
is 7 bins, on a 6-hour panel 28 bins).

```python
from eventflow.core.steps import EWMAMeanStep

frame = make_panel({u: [1.0, 2.0, 3.0, 4.0] for u in range(9)})
result = EWMAMeanStep(
    value_cols=["count"], halflife_days=7.0, group_cols=["grid_id"]
).run(frame).collect()
# The first bin's EWMA equals the count itself (adjust=True, min_samples=1).
assert result.filter(pl.col("time_bin") == BIN0)["count_ewma_7d"].to_list() == [1.0] * 9
```

#### TrendStep

```python
class TrendStep(Step):
    def __init__(
        self,
        recent_col: str,
        long_col: str,
        output_col: str | None = None,
    ) -> None: ...
```

Ratio of two existing rolling-mean columns (default output
`{recent_col}_vs_{long_col}`). Null when the long (denominator)
column is null or 0 — a zero denominator never produces `inf`.

```python
from eventflow.core.steps import TrendStep
from eventflow.core.steps.temporal import MovingAverageStep

frame = make_panel({u: [1.0, 2.0, 3.0, 4.0] for u in range(9)})
frame = MovingAverageStep(
    value_cols=["count"], windows=[1, 2], time_unit="1d",
    group_cols=["grid_id"], closed="left",
).run(frame)
result = TrendStep(recent_col="count_ma_1", long_col="count_ma_2").run(frame).collect()
# Bin 3: ma_1 = 3.0 (count at bin 2), ma_2 = mean(2, 3) = 2.5 -> ratio 1.2.
row = result.filter(
    (pl.col("grid_id") == 0) & (pl.col("time_bin") == BIN0 + timedelta(days=3))
)
assert row["count_ma_1_vs_count_ma_2"].item() == 1.2
```

#### TimeSinceLastEventStep

```python
class TimeSinceLastEventStep(Step):
    def __init__(
        self,
        value_col: str = "count",
        group_cols: Sequence[str] | None = None,
        output_col: str = "count_time_since_last_d",
        output_unit: str = "d",
    ) -> None: ...
```

Days since the most recent bin with `value_col > 0` (Float64).
Null when no prior bin (including the current one) had an event.
Only `"d"` is supported as `output_unit` in v1.

```python
from eventflow.core.steps import TimeSinceLastEventStep

counts = {u: [1.0, 1.0, 1.0, 1.0] for u in range(9)}
counts[4] = [0.0, 0.0, 5.0, 0.0]
frame = make_panel(counts)
result = TimeSinceLastEventStep(value_col="count", group_cols=["grid_id"]).run(frame).collect()
unit4 = result.filter(pl.col("grid_id") == 4).sort("time_bin")
assert unit4["count_time_since_last_d"].to_list() == [None, None, 0.0, 1.0]
```

#### BurstinessStep

```python
class BurstinessStep(Step):
    def __init__(
        self,
        value_col: str = "count",
        window: str = "28d",
        group_cols: Sequence[str] | None = None,
        min_periods: int = 1,
    ) -> None: ...
```

Rolling coefficient of variation (std / mean) of counts over a
trailing closed-left window: `{value_col}_burstiness_{window}`. This
is a burstiness *proxy*, not the Goh-Barabasi inter-event-time `B`.
Null when the trailing mean is null or 0 (never `inf`).

#### ZeroFeaturesStep

```python
class ZeroFeaturesStep(Step):
    def __init__(
        self,
        value_col: str = "count",
        active_window: str = "7d",
        share_window: str = "28d",
        group_cols: Sequence[str] | None = None,
        min_periods: int = 1,
    ) -> None: ...
```

Features of absence, both with windows closed on the right (including
the current bin):

- `{value_col}_active_{active_window}` (Boolean): any value > 0 in the
  trailing window.
- `{value_col}_share_nonzero_{share_window}` (Float64): fraction of
  bins with value > 0 in the trailing window.

```python
from eventflow.core.steps import ZeroFeaturesStep

counts = {u: [1.0, 1.0, 1.0, 1.0] for u in range(9)}
counts[4] = [0.0, 0.0, 5.0, 0.0]
frame = make_panel(counts)
result = ZeroFeaturesStep(value_col="count", group_cols=["grid_id"]).run(frame).collect()
unit4 = result.filter(pl.col("grid_id") == 4).sort("time_bin")
assert unit4["count_active_7d"].to_list() == [False, False, True, True]
assert unit4["count_share_nonzero_28d"].to_list()[-1] == 0.25  # 1 of 4 bins
```

### eventflow.core.steps.categorical

#### CategoryCountsStep

Per-category counts, shares and Shannon entropy per (unit, time)
group. Replaces the declared-but-ignored `categorical_features` of
the early Chicago recipe and the experimental
`core/features.py::aggregate_by_category`.

```python
class CategoryCountsStep(Step):
    def __init__(
        self,
        category_cols: Sequence[str],
        unit_col: str = "grid_id",
        time_col: str = "time_bin",
        top_n: int = 10,
    ) -> None: ...

    def fit(self, event_frame: EventFrame) -> CategoryCountsStep: ...
    @property
    def is_fitted(self) -> bool: ...
    def fitted_values(self, cat_col: str) -> list[tuple[Any, str]]: ...
```

- Keeps only the top-N categories per column by total frequency
  (ties broken by sorted value, for determinism) and folds every
  other value into an `other` bucket, so a high-cardinality column
  cannot explode the output schema. Null category values are counted
  in `other` and never occupy a top-N slot.
- Category values are sanitized before use in column names; a value
  that sanitizes to empty, to `"other"`, or to the same name as
  another kept value raises `ValueError` instead of colliding.
- Outputs per category column `<col>`, joined back onto every event
  row on `(unit_col, time_col)`: `count_<col>_<value>` and
  `count_<col>_other`, `share_<col>_<value>` and
  `share_<col>_other` (null when the group total is 0), and
  `<col>_entropy` (Shannon entropy, natural log, over the top-N +
  `other` buckets; null when the total is 0).

```python
from eventflow.core.steps import CategoryCountsStep

# Event rows: (grid_id, time_bin, primary_type); THEFT x3, BATTERY x3, ASSAULT x1.
event_rows = [
    (0, BIN0, "THEFT"), (0, BIN0, "THEFT"), (0, BIN0, "BATTERY"),
    (0, BIN0 + timedelta(days=1), "THEFT"),
    (0, BIN0 + timedelta(days=1), "ASSAULT"),
    (1, BIN0, "BATTERY"), (1, BIN0, "BATTERY"),
    (1, BIN0 + timedelta(days=1), None),
]
events_frame = EventFrame(
    pl.LazyFrame({
        "grid_id": [r[0] for r in event_rows],
        "time_bin": [r[1] for r in event_rows],
        "primary_type": [r[2] for r in event_rows],
        "latitude": [41.8781] * len(event_rows),
        "longitude": [-87.6298] * len(event_rows),
    }),
    EventSchema(
        timestamp_col="time_bin", lat_col="latitude", lon_col="longitude",
        numeric_cols=[], categorical_cols=["primary_type"],
    ),
    EventMetadata(dataset_name="docs-events", crs="EPSG:4326", time_bin="1d"),
)
result = CategoryCountsStep(
    category_cols=["primary_type"], top_n=2
).run(events_frame).collect()
# Top 2 are BATTERY and THEFT; ASSAULT lands in "other".
assert "count_primary_type_THEFT" in result.columns
assert "count_primary_type_other" in result.columns
assert "count_primary_type_ASSAULT" not in result.columns
```

#### Fit / freeze (FittableStep)

`CategoryCountsStep` is the library's only fittable step (phase 4).
`fit(event_frame)` learns the top-N vocabulary from that frame only
and freezes it; `run` after `fit` uses the frozen vocabulary and
never re-learns from the run frame, so a category frequent in a
later slice but absent (or rare) in the fit frame lands in `other`.
This is the anti-leakage contract for train/test evaluation: fit on
the train fold, run on the test fold. Calling `fit` again replaces
the frozen vocabulary cleanly. Without `fit`, `run` keeps the
original behaviour of learning the top-N from the run frame
(single-run recipes). `evaluate_walk_forward` fits fittable steps on
each fold's train frame only (see
[Evaluation](#evaluation-module)).

```python
step = CategoryCountsStep(category_cols=["primary_type"], top_n=2).fit(events_frame)
assert step.is_fitted
assert [v for v, _ in step.fitted_values("primary_type")] == ["BATTERY", "THEFT"]
# A later frame where ROBBERY dominates: it lands in "other", never in a top slot.
test_frame = EventFrame(
    pl.LazyFrame({
        "grid_id": [0] * 6,
        "time_bin": [BIN0] * 6,
        "primary_type": ["ROBBERY"] * 5 + ["THEFT"],
        "latitude": [41.8781] * 6,
        "longitude": [-87.6298] * 6,
    }),
    EventSchema(
        timestamp_col="time_bin", lat_col="latitude", lon_col="longitude",
        numeric_cols=[], categorical_cols=["primary_type"],
    ),
    EventMetadata(dataset_name="docs-events", crs="EPSG:4326", time_bin="1d"),
)
result = step.run(test_frame).collect()
assert "count_primary_type_ROBBERY" not in result.columns
assert result["count_primary_type_other"].unique().to_list() == [5]
```

### eventflow.core.steps.spatial

#### SpatialLagStep

Row-standardized binary spatial lag, computed with one edge join
(vectorized in phase 2; the single-scale lag on the full Chicago
panel runs in ~6 s, down from 51.4 s — see the phase-2 note for the
honest timing caveat: the original ≤ 2 s criterion was not met).

```python
class SpatialLagStep(Step):
    def __init__(
        self,
        value_cols: Sequence[str],
        neighbor_col: str = "grid_id",
        weights: str = "queen",
        k: int = 4,
        distance_threshold: float | None = None,
        group_cols: Sequence[str] | None = None,
        spatial_units: pl.DataFrame | None = None,
        spatial_weights: SpatialWeights | None = None,
    ) -> None: ...
```

Produces `{value_col}_spatial_lag` (Float64): the mean of neighbours'
values in the same snapshot (`group_cols`, e.g. `time_bin`, are never
pooled). Islands (units with no neighbours) yield null. Values must
be finite and non-null — null/non-finite values are a blocking error,
never silently treated as zero. The recipe feeds it lagged counts, so
the feature at `t` describes neighbours' counts strictly before `t`.

#### MultiScaleSpatialLagStep

Row-standardized binary spatial lag over disjoint queen rings.
Ring `r` is the set of cells at Chebyshev distance exactly `r`, so
rings are disjoint and ring 1 reproduces `SpatialLagStep` exactly.

```python
from eventflow.core.steps.spatial import MultiScaleSpatialLagStep, SpatialLagStep

class MultiScaleSpatialLagStep(Step):
    def __init__(
        self,
        value_cols: Sequence[str],
        neighbor_col: str = "grid_id",
        rings: Sequence[int] = (1, 2, 3),
        group_cols: Sequence[str] | None = None,
        spatial_units: pl.DataFrame | None = None,
        spatial_weights: SpatialWeights | None = None,
    ) -> None: ...
```

Produces `{value_col}_spatial_lag_r{r}` per value and ring. A supplied
`SpatialWeights` object cannot define rings 2+ and is rejected at
construction. Rings are computed one at a time so the edges ×
snapshots intermediate stays bounded.

```python
# 3x3 panel, centre cell (grid_id 4) has count 10 in bin 0, 1 elsewhere.
counts_centre = {u: [1.0, 1.0, 1.0, 1.0] for u in range(9)}
counts_centre[4] = [10.0, 1.0, 1.0, 1.0]
frame = make_panel(counts_centre)
multi = MultiScaleSpatialLagStep(
    value_cols=["count"], rings=(1, 2), group_cols=["time_bin"]
).run(frame).collect()
single = SpatialLagStep(value_cols=["count"], group_cols=["time_bin"]).run(frame).collect()
assert multi["count_spatial_lag_r1"].to_list() == single["count_spatial_lag"].to_list()
centre_bin0 = multi.filter(
    (pl.col("grid_id") == 4) & (pl.col("time_bin") == BIN0)
).row(0, named=True)
assert centre_bin0["count_spatial_lag_r1"] == 1.0   # 8 neighbours at 1
assert centre_bin0["count_spatial_lag_r2"] is None  # no cell at distance 2: island
```

#### KDEStep

```python
class KDEStep(Step):
    def __init__(
        self,
        bandwidth: float = 500.0,
        kernel: str = "gaussian",
        output_col: str = "kde_density",
    ) -> None: ...
```

> **Approximation, not a KDE.** `KDEStep` counts neighbours inside
> the bandwidth and divides by the buffer area; no kernel function is
> applied. It is kept registered for backwards compatibility. For a
> proper Gaussian KDE use `KernelDensityStep`; for a space-time KDE
> use `SpaceTimeKernelDensityStep`.

#### LocalMoranStep / GetisOrdStep

Anselin local Moran's I and Getis-Ord Gi\* hotspot statistics. Both
still use the per-snapshot helper (they were not part of the phase-2
vectorization). See their class docstrings for the exact statistics.

### eventflow.core.steps.density

#### KernelDensityStep

Gaussian kernel density of a value column at each cell centre
(density in events per m²). The bandwidth `h` is declared explicitly
by the caller — no automatic estimation (estimating one on the full
panel would be a fitted statistic and a leakage surface).

```python
from eventflow.core.steps.density import KernelDensityStep

class KernelDensityStep(Step):
    def __init__(
        self,
        value_col: str = "count",
        bandwidth_m: float = 600.0,
        unit_col: str = "grid_id",
        group_cols: Sequence[str] | None = None,
    ) -> None: ...
```

Produces `{value_col}_kde`. The kernel is truncated at `3h`; every
cell is its own neighbour at distance 0, so the density is never
null for an observed cell. No edge correction is applied: density
near the boundary of the observed support is underestimated.
Causality: a **same-bin smoother** — the density at bin `t` only uses
values from bin `t`. Fed lagged counts it is a predictor feature; fed
current-bin counts it is a same-bin descriptor, not a predictor.
Requires cell-centre coordinates in the schema lon/lat columns
(projected metres; a metric CRS is required). Eager: the result is
collected before returning, so the edge join runs on its own instead
of together with the other giant joins at the final collect.

```python
counts_centre = {u: [1.0, 1.0, 1.0, 1.0] for u in range(9)}
counts_centre[4] = [10.0, 1.0, 1.0, 1.0]
frame = make_panel(counts_centre)
result = KernelDensityStep(
    value_col="count", bandwidth_m=600.0, group_cols=["time_bin"]
).run(frame).collect()
bin0 = result.filter(pl.col("time_bin") == BIN0)
assert bin0["count_kde"].null_count() == 0
# The centre cell (count 10 in bin 0) has the highest density.
centre = bin0.filter(pl.col("grid_id") == 4)["count_kde"].item()
corner = bin0.filter(pl.col("grid_id") == 0)["count_kde"].item()
assert centre > corner
```

#### SpaceTimeKernelDensityStep

Separable product kernel: Gaussian in space, exponential in time.

```python
from eventflow.core.steps.density import SpaceTimeKernelDensityStep

class SpaceTimeKernelDensityStep(Step):
    def __init__(
        self,
        value_col: str = "count",
        bandwidth_m: float = 600.0,
        halflife_days: float = 7.0,
        unit_col: str = "grid_id",
        time_col: str = "time_bin",
    ) -> None: ...
```

Produces `{value_col}_stkde` (density, events per m²):

`stkde(i, t) = sum_{s <= t} kde(i, s) * 0.5 ** ((t - s) / ht_bins)`

The temporal kernel only looks backward, including the current bin
with weight 1 (same "includes current bin" precedent as
`EWMAMeanStep`). The half-life in days is converted to bins using
`metadata.time_bin`. The smoothing assumes a regular completed panel
(one row per unit per bin, no gaps): the per-unit bin index is the
elapsed time in bins. Eager, as `KernelDensityStep`.

> **Time-column trap (phase 2, fixed and guarded).** On the completed
> panel the pre-completion `time_bin` column is null on every
> generated row — only the schema timestamp column holds the bin
> datetime on all rows. Pass the timestamp column as `time_col`
> (the Chicago recipe passes `time_col="date"`), never the
> pre-completion `time_bin`. The step rejects a null time column
> with a blocking `ValueError`.

### eventflow.core.steps.near_repeat

#### NearRepeatStep

Near-repeat counts: for each (cell, bin `t`) and each configured
queen ring `r`, the number of events in the ring over the trailing
7 d and over the trailing 7–14 d, both closed-left (strictly before
`t`), plus a binary flag that is 1 when any event occurred in ring 1
in the trailing 7 d.

```python
from eventflow.core.steps.near_repeat import NearRepeatStep

class NearRepeatStep(Step):
    def __init__(
        self,
        value_col: str = "count",
        rings: Sequence[int] = (1, 2, 3),
        unit_col: str = "grid_id",
        time_col: str = "time_bin",
    ) -> None: ...
```

Produces `{value_col}_nr_r{r}_7d`, `{value_col}_nr_r{r}_7_14d`
(counts) and `{value_col}_nr_event_7d` (Int64 0/1). A ring with no
neighbours yields 0, not null. Early bins sum over the bins that
exist (0 when the panel starts). Same time-column trap as STKDE:
pass the schema timestamp column as `time_col`. Eager: each ring's
edge join runs on its own and the result is collected before
returning.

```python
counts_centre = {u: [1.0, 1.0, 1.0, 1.0] for u in range(9)}
counts_centre[4] = [10.0, 1.0, 1.0, 1.0]
frame = make_panel(counts_centre)
result = NearRepeatStep(value_col="count", rings=(1,)).run(frame).collect()
# Bin 0: strictly-past windows are empty -> 0, flag 0.
first = result.filter(pl.col("time_bin") == BIN0)
assert first["count_nr_r1_7d"].to_list() == [0] * 9
# Centre cell, bin 1: ring 1 held 8 neighbours x 1 event in bin 0.
row = result.filter(
    (pl.col("grid_id") == 4) & (pl.col("time_bin") == BIN0 + timedelta(days=1))
).row(0, named=True)
assert row["count_nr_r1_7d"] == 8
assert row["count_nr_event_7d"] == 1
```

### eventflow.core.steps.hawkes

#### HawkesExcitationStep

Fixed-kernel Hawkes excitation of a panel value column — a feature,
not a fitted Hawkes process: no parameter estimation, no
background/excitation separation, no branching ratio.

```python
from eventflow.core.steps.hawkes import HawkesExcitationStep

class HawkesExcitationStep(Step):
    def __init__(
        self,
        value_col: str = "count",
        halflife_days: float = 7.0,
        group_cols: Sequence[str] | None = None,
        value_cols: Sequence[str] | None = None,
    ) -> None: ...
```

Per unit, `E(t) = sum_{s<t} count(s) * exp(-(t-s)/tau)` with
`tau = halflife_days / ln 2`. Strictly past bins only (`s < t`); the
first bin of each unit is 0. Produces
`{value_col}_hawkes_{halflife}d` in units of decayed event count.
This differs from the event-level `HawkesKernelStep` (alpha/beta/mu
on continuous event times) by operating on the completed panel with
a single caller-declared half-life.

```python
counts = {u: [1.0, 1.0, 1.0, 1.0] for u in range(9)}
counts[4] = [0.0, 0.0, 5.0, 0.0]
frame = make_panel(counts)
result = HawkesExcitationStep(
    value_col="count", halflife_days=7.0, group_cols=["grid_id"]
).run(frame).collect()
first = result.filter(pl.col("time_bin") == BIN0)
assert first["count_hawkes_7d"].to_list() == [0.0] * 9
# The centre unit's excitation is 0 until after its bin-2 event.
unit4 = result.filter(pl.col("grid_id") == 4).sort("time_bin")
assert unit4["count_hawkes_7d"].to_list()[:3] == [0.0, 0.0, 0.0]
assert unit4["count_hawkes_7d"].to_list()[3] > 0
```

### Step registry names

`get_default_registry()` registers (among others):

| Name | Step |
|---|---|
| `complete_panel` | `CompletePanelStep` |
| `category_counts` | `CategoryCountsStep` |
| `observation_panel` | `ObservationPanelStep` |
| `transform_crs`, `assign_to_grid`, `assign_to_zones`, `compute_distances`, `grid_aggregation` | spatial steps |
| `spatial_lag` | `SpatialLagStep` |
| `multiscale_spatial_lag` | `MultiScaleSpatialLagStep` |
| `kde` | `KDEStep` (approximation, see above) |
| `kernel_density` | `KernelDensityStep` |
| `space_time_kde` | `SpaceTimeKernelDensityStep` |
| `near_repeat` | `NearRepeatStep` |
| `hawkes_excitation` | `HawkesExcitationStep` |
| `local_moran`, `getis_ord` | `LocalMoranStep`, `GetisOrdStep` |
| `extract_temporal_components`, `time_bins`, `temporal_lag`, `moving_average`, `recency_weight`, `calendar_encoding`, `fill_null`, `inter_arrival` | temporal steps |
| `ewma_mean`, `trend`, `time_since_last_event`, `burstiness`, `zero_features` | dynamics steps |
| `temporal_context_join`, `spatial_context_join`, `spatio_temporal_context_join`, `context_aggregation` | context join steps |
| `spatial_domain_mask`, `grid_domain_coverage` | domain steps |

> **Import paths.** The phase-2 steps (`MultiScaleSpatialLagStep`,
> `KernelDensityStep`, `SpaceTimeKernelDensityStep`, `NearRepeatStep`,
> `HawkesExcitationStep`) are registered but **not** re-exported from
> `eventflow.core.steps` — import them from their own modules as shown
> in the sections above. Everything else in this section is importable
> from `eventflow.core.steps`.

### eventflow.core.steps.observation

#### ObservationPanelStep

Counts raw point-event rows on explicit observation support while
keeping observed zeros, missing totals and outside-domain rows
distinct (see [observation_panel.md](observation_panel.md)).

```python
class ObservationPanelStep(Step):
    def __init__(
        self,
        panel: ObservationPanel,
        count_col: str = "event_count",
        recorded_col: str = "recorded_count",
        rate_col: str = "event_rate",
        offset_col: str = "log_exposure",
        events_outside: Literal["error", "ignore"] = "error",
        event_id_col: str | None = None,
        cadence_rule: CadenceRule | dict[str, Any] | None = None,
        cadence_prefix: str = "cadence",
        include_missing_times: bool = False,
        on_gap: Literal["report", "raise"] = "report",
    ) -> None: ...
```

## Evaluation Module

### eventflow.evaluation

Blocked splits, walk-forward evaluation, and downstream metrics.
`eventflow.evaluation` exposes: `temporal_split`, `spatial_split`,
`entity_split`, `partition_units`, `blocked_split`,
`validate_split`, `walk_forward_splits`, `evaluate_walk_forward`,
`pai_score`, `pr_auc_score`, `WalkForwardFold`, `FoldResult`.
All split functions accept a `LazyFrame` or an `EventFrame` and
return the same kind.

#### Why random splits are invalid here

A random train/test split of panel rows leaks through three
mechanisms: **time** (lag/window features read several bins back, so
a shared bin scores a model on data its features were built from),
**space** (spatial lags, KDE and near-repeat read neighbouring
units), and **entity** (the same station/vehicle/user on both sides).
"Blocked" means: temporal = disjoint time intervals (plus an optional
embargo gap); spatial = disjoint unit sets; entity = disjoint entity
sets. See [phase4-evaluation.md](design-notes/phase4-evaluation.md).

```python
from eventflow.evaluation import (
    blocked_split, temporal_split, validate_split,
)
from datetime import timedelta

# Shared frame for the split examples: 20 daily bins x 3 units.
rows = {"grid_id": [], "time_bin": [], "count": []}
for day in range(20):
    for unit in range(3):
        rows["grid_id"].append(unit)
        rows["time_bin"].append(BIN0 + timedelta(days=day))
        rows["count"].append((day + unit) % 3)
frame = pl.LazyFrame(rows)

# Temporal split with a 1-day embargo: rows in [cutoff, cutoff + gap)
# belong to neither side.
train, test = temporal_split(
    frame, "time_bin", BIN0 + timedelta(days=10), gap=timedelta(days=1)
)
validate_split(train, test, time_col="time_bin")  # raises on any shared bin
```

```python
# Combined time x space blocking. Rows in the two off-diagonal
# quadrants (early but held-out unit, or late but train unit) are
# dropped — they cannot be assigned without breaking one of the blocks.
train, test = blocked_split(
    frame, "time_bin", BIN0 + timedelta(days=10),
    unit_col="grid_id", test_units=[2],
)
validate_split(train, test, time_col="time_bin", unit_col="grid_id")
```

`validate_split` is the negative control: a deliberately leaking
split **must** raise `ValueError` there. `partition_units(frame,
unit_col, test_fraction=0.2, seed=0)` deterministically partitions
the distinct units into `(train_units, test_units)` (SHA-256 hash of
`(seed, unit)`, reproducible across runs). `spatial_split(frame,
unit_col, test_units)` and `entity_split(frame, entity_col,
test_entities)` are the single-dimension unit/entity splitters.

#### Walk-forward

```python
from eventflow.evaluation import walk_forward_splits, evaluate_walk_forward

def walk_forward_splits(
    frame: FrameLike,
    time_col: str,
    initial_train: dt.timedelta,
    step: dt.timedelta,
    gap: dt.timedelta | None = None,
    max_folds: int | None = None,
) -> Iterator[WalkForwardFold]: ...
```

Expanding-window folds over time: fold 0 trains on
`[t_min, t_min + initial_train)` and tests on the next `step` after
the embargo `gap`; each later fold extends the train window by one
`step` (expanding, never sliding) and tests on the following `step`.
Folds stop when the test window starts after the last observed time
bin; a fold whose test window holds no rows is skipped. Every yielded
fold is checked with `validate_split` on the time column.

```python
def evaluate_walk_forward(
    steps: Sequence[Step],
    frame: EventFrame,
    time_col: str,
    initial_train: dt.timedelta,
    step: dt.timedelta,
    score_fn: Callable[[EventFrame, EventFrame], dict[str, float]],
    gap: dt.timedelta | None = None,
    max_folds: int | None = None,
) -> list[FoldResult]: ...
```

Runs a fresh `Pipeline` per fold. Each step that implements `fit`
is fit on the train fold only (fresh fit per fold: fitted statistics
from a previous fold are overwritten, never carried over), then the
pipeline runs on the train fold and on the test fold, and
`score_fn(train_out, test_out)` produces the fold's scores. One
`FoldResult` per fold, in chronological order (fold index, window
bounds, row counts, score dict).

```python
from eventflow.evaluation import pai_score
from tests.evaluation.synthetic import generate_synthetic_panel, panel_event_frame

# Synthetic 2x2-unit x 20-bin panel, as an EventFrame.
panel = panel_event_frame(
    generate_synthetic_panel(n_grid_rows=2, n_grid_cols=2, n_bins=20).panel
)

def score_fn(train_out, test_out):
    test = test_out.collect()
    y_true = (test["count"] > 0).cast(pl.Int64).to_numpy()
    y_score = test["count_lag_1"].fill_null(0).to_numpy()
    return {"pai_20pct": pai_score(y_true, y_score, area_fraction=0.2)}

results = evaluate_walk_forward(
    [TemporalLagStep(value_cols=["count"], lag_periods=[1], time_unit="1d",
                     group_cols=["grid_id"])],
    panel, "time_bin",
    initial_train=timedelta(days=10), step=timedelta(days=5),
    score_fn=score_fn,
)
# 2 folds on a 20-bin panel; every score is a finite float.
assert len(results) == 2
```

#### Metrics

```python
def pai_score(y_true: ArrayLike, y_score: ArrayLike, area_fraction: float) -> float: ...
def pr_auc_score(y_true: ArrayLike, y_score: ArrayLike) -> float: ...
```

- **PAI** (Prediction Accuracy Index, hotspot ranking): rank all rows
  by score descending, take the top `area_fraction` as the predicted
  hotspot area, then `PAI = (events captured / total events) /
  realised_area_fraction`. PAI = 1 is chance level; the maximum is
  `1 / a`. Ties are resolved by input order (stable sort).
- **PR-AUC**: area under the precision-recall curve, evaluated after
  each distinct score value (tied rows consumed as one group) and
  integrated with the trapezoid rule. Chance level is the positive
  rate, not 0.5.

Both are pure functions on numpy arrays (no sklearn/scipy) and raise
`ValueError` on empty input, mismatched lengths, non-binary labels,
non-finite scores, or no positive events.

```python
assert pai_score([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3], area_fraction=0.25) == 4.0
assert pr_auc_score([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1]) == 1.0
```

Both metrics are worked examples, not acceptance criteria: no
threshold on them accepts or rejects the preprocessing. On the
Chicago replay the worked examples give PAI 7.96 at 1% and PR-AUC
0.0684 for `count_ma_7` (see
[phase4-synthetic-ground-truth.md](design-notes/phase4-synthetic-ground-truth.md)).

#### Synthetic ground truth (test-only helper)

`tests/evaluation/synthetic.py::generate_synthetic_panel` builds a
synthetic unit × time-bin panel with planted structure (lag-1 φ =
0.6, hotspot ×3, weekend ×1.6; defaults: seed 42, 6×6 units, 56
daily bins from 2024-01-01). It is test infrastructure, not library
API — the evaluation section documents it because the recovery
numbers below are the library's check that its own steps recover
planted structure.

```python
from tests.evaluation.synthetic import generate_synthetic_panel, panel_event_frame

data = generate_synthetic_panel()
frame = panel_event_frame(data.panel)
lagged = TemporalLagStep(
    value_cols=["count"], lag_periods=[1], time_unit="1d", group_cols=["grid_id"]
).run(frame).collect()
paired = lagged.drop_nulls(["count_lag_1"])
corr = paired.select(pl.corr("count", "count_lag_1")).item()
# Planted phi = 0.6; Poisson noise attenuates it. Measured 0.529 on the
# default panel; the documented tolerance is [0.25, 0.75].
assert 0.25 <= corr <= 0.75
# The top-4 units by mean count are exactly the planted hotspot units.
top4 = (
    lagged.group_by("grid_id").agg(pl.col("count").mean().alias("mean_count"))
    .sort("mean_count", descending=True).head(4)["grid_id"].to_list()
)
assert set(top4) == set(data.params.hotspot_units)
```

The same helper drives the planted-leak demonstration: a column
computed from future information scores PR-AUC 1.0000 on a random
70/30 row split but 0.4031 on the blocked causal split (chance =
prevalence 0.111). The harness rejects leaking *splits*, not leaking
*columns* — a future-peeking column must not be built in the first
place (see
[phase4-synthetic-ground-truth.md](design-notes/phase4-synthetic-ground-truth.md)).

## Context Module

### eventflow.core.context.tabular

#### TabularFileSource

A generic external-table connector: one local CSV or Parquet file
under a declared schema, feeding the existing context join steps
(`TemporalContextJoinStep`, `SpatialContextJoinStep`,
`SpatioTemporalContextJoinStep`, `EnricherStep`). It replaces the
pattern of hand-rolling one loader per dataset.

```python
from eventflow.core.context import TabularFileSource

class TabularFileSource(DynamicTemporalSource):
    def __init__(
        self,
        path: str | Path,
        *,
        timestamp_col: str,
        value_cols: Sequence[str],
        key_cols: Sequence[str] = (),
        timestamp_format: str | None = None,
    ) -> None: ...

    def load(self) -> pl.LazyFrame: ...
```

- `timestamp_col`: Datetime/Date column, or a string column parsed
  strictly (optionally with an explicit `strptime`
  `timestamp_format`); unparseable values fail the load instead of
  becoming null.
- `value_cols`: columns to join onto events (at least one).
- `key_cols`: optional join keys (e.g. a station id) used by
  spatio-temporal joins to keep groups isolated.
- The declaration is validated at construction (missing file, unknown
  suffix, empty/duplicate/overlapping declarations, missing declared
  columns, non-temporal timestamp dtype). The data are validated at
  `load()` (unparseable or null timestamps, duplicate
  `(key, timestamp)` pairs). `load()` returns exactly the declared
  columns, sorted by `key_cols + [timestamp_col]`. Unmatched events
  stay null after a join; the connector never invents a value.
- Local files only, by design: no live APIs, no network fetching, no
  credentials. A future provider plugs in by subclassing
  `BaseContextSource` (or by writing its payload to a local file and
  reusing this class) — see
  [phase3-connectors.md](design-notes/phase3-connectors.md).

```python
from eventflow.core.context import EnricherStep, TabularFileSource, TemporalJoin

# weather.csv: ts,temp_c / 2024-01-01 00:00:00,10.0 / 2024-01-01 01:00:00,11.0
source = TabularFileSource("weather.csv", timestamp_col="ts", value_cols=["temp_c"])
loaded = source.load().collect()
assert loaded.columns == ["ts", "temp_c"]
assert loaded["temp_c"].to_list() == [10.0, 11.0]

events = EventFrame(
    pl.LazyFrame({
        "time": [datetime(2024, 1, 1, 0, 30), datetime(2024, 1, 1, 1, 30)],
        "x": [0.5, 0.5],
        "y": [0.5, 0.5],
    }),
    EventSchema(timestamp_col="time", lon_col="x", lat_col="y"),
    EventMetadata(dataset_name="docs-context", crs="EPSG:3857"),
)
enriched = EnricherStep(source, TemporalJoin(strategy="backward")).run(events).collect()
# Events at 00:30 and 01:30 join the most recent prior row: [10.0, 11.0].
assert enriched["temp_c"].to_list() == [10.0, 11.0]
```

The real-data demonstration (`scripts/phase3_connectors_demo.py`) is
an honest negative result: the GeoLife subset (2007–2009) and the
Beijing meteorology (from 2013-03-01) do not overlap in time, so the
asof-backward join with a 1 h tolerance correctly matches **0 /
1,471** points. The positive control (synthetic event times inside
the Beijing range, real weather values) matches 8/9 with asof-backward
1 h and 2/9 with `"exact"`. No overlapping data was fabricated.

## Dataset Module

### eventflow.datasets.chicago_crime

Chicago crime dataset adapter (City of Chicago Data Portal,
Socrata). The loader normalizes ISO/legacy timestamps, numeric
coordinate strings and booleans, removes records without valid
timestamps or coordinates, and projects longitude/latitude from
EPSG:4326 to EPSG:26971 before assigning meter-based cells.

#### load_chicago_crime

```python
def load_chicago_crime(
    raw_path: str,
    config: DatasetConfig | None = None,
    apply_cleaning: bool = True,
) -> EventFrame:
    """Load Chicago crime data as EventFrame."""
    ...
```

`--input` accepts a file or a directory of one format (CSV or
Parquet). The bundled recipe (`chicago_crime_v1`, see
[Recipes](#recipe-module)) keeps **300-meter cells and six-hour
bins** on a fixed grid origin `(0, 0)` in projected meters, counts
observed cell/bin pairs, and — with the shipped YAML — completes
the panel with explicit zeros and builds the phase-1 and phase-2
feature families (per-category counts, lags, rolling means, EWMA,
cyclical encodings, multi-scale spatial lag, KDE, STKDE,
near-repeat, Hawkes, trend, recency, zero features). Replay on real
Q1 2024 data: 1,908,088 rows × 90 columns, sum `count` = 60,235
(`make audit`, see `docs/audit/`).

Context sources (weather, special events, demographics) exist under
`datasets/chicago_crime/context/` but are unwired and untested
(experimental); enabling any `context.<source>.enabled` in the
Chicago recipe raises a blocking error.

### eventflow.datasets.beijing_air_quality

Beijing Multi-Site Air Quality (UCI 501, Chen 2017): 12 stations ×
hourly, 2013-03-01 → 2017-02-28 (35,064 hours each), 6 pollutants +
6 meteorological variables; missing values are the literal string
`NA`. Total 420,768 rows. Station coordinates come from the
literature (Ma et al. 2023 Table 1 for 11 stations, Zhang et al.
2020 Table S1 for Shunyi) — the UCI files carry none — and no step
in the v1 recipe consumes coordinates.

#### load_beijing_air_quality

```python
def load_beijing_air_quality(
    raw_path: str,
    config: DatasetConfig | None = None,
    apply_cleaning: bool = True,
) -> EventFrame:
    """Load Beijing air quality data as an EventFrame.

    raw_path: directory of per-station CSVs, one CSV, or one Parquet
    file. Only config.raw_root and config.time_zone are honoured.
    """
    ...
```

The recipe (`beijing_air_quality_v1`) is the second-domain proof:
the same generic steps build a sensor panel with no Chicago-specific
assumption — unit = station, bin = hour, only temporal steps apply
(lags 1 h / 24 h / 168 h, rolling means 24 h / 168 h, calendar
encoding). `CompletePanelStep` runs with
`measure_cols=[11 measurements]`: the shipped panel is already
complete (0 generated pairs), so the NA counts are exactly identical
before and after completion (PM2.5 8,739 → 8,739). Replay: 420,768
rows × 64 columns, count sum 420,768. Honest limit: the proof covers
gaps / missing-stays-missing and per-station separation, **not**
irregular sampling (regular hourly timestamps) — see
[phase3-beijing.md](design-notes/phase3-beijing.md).

### eventflow.datasets.geolife

GeoLife GPS trajectories (Microsoft Research): one GPS fix is one
event; entity = user (`user_id`), a trajectory (`trajectory_id`,
the `.plt` file stem) groups the fixes of one continuous log.
Coordinates are WGS 84, projected to UTM 50N (EPSG:32650, metres)
for distance work.

#### load_geolife

```python
def load_geolife(
    paths: list[str | Path],
    *,
    user_ids: list[str] | None = None,
    metadata: EventMetadata | None = None,
) -> EventFrame:
    """Load .plt files into an EventFrame (one row per GPS fix, EPSG:4326).

    paths: .plt files to load, one trajectory each. user_ids defaults
    to the parent folder name (the Data/<user>/ folder).
    """
    ...
```

The recipe (`geolife_v1`) ships only the steps that apply to
trajectories: CRS transform → `TrajectoryKinematicsStep`
(`group_cols`, `max_gap_s=300.0`: distance, elapsed time and speed;
speed withheld across gaps and on `dt <= 0`) → temporal components
→ cyclical calendar → `InterArrivalStep` → rows-mode lag / rolling
mean of speed → 5-min bins → 100 m grid → fix counts per
(cell, bin). Replay on the documented subset (users `010` and `000`,
1,471 fixes): point-level output 1,471 rows × 26 columns; panel
112 rows × 8 columns, count sum 1,471. The panel-family steps
(spatial lag, KDE, STKDE, near-repeat, Hawkes, panel completion,
category counts, burstiness) were evaluated on the subset and
**do not apply** as-is — the verdict table with the evidence is in
[phase3-geolife-boundary.md](design-notes/phase3-geolife-boundary.md).
Timezone is an assumption (`.plt` times labelled `Asia/Shanghai`,
no conversion).

## Recipe Module

### eventflow.recipes.base

#### BaseRecipe

```python
class BaseRecipe(ABC):
    def __init__(
        self,
        config: RecipeConfig,
        *,
        step_registry: StepRegistry | None = None,
    ) -> None: ...

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """Build the transformation pipeline used when no registry config is supplied."""
        ...

    def get_pipeline(self) -> Pipeline:
        """Return a pipeline, favoring `config.features["steps"]` when provided."""
        ...

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Resolve the pipeline and execute it against an EventFrame."""
        ...
```

`FeatureConfig` (`eventflow.core.schema`) declares the recipe's
feature families; unknown or unimplemented keys are a blocking
error, never silently ignored:

| Key | Step(s) |
|---|---|
| `count_features: true` | per (unit, time) counts |
| `categorical_features: [...]`, `categorical_top_n: 10` | `CategoryCountsStep` |
| `aggregation_windows: ["1d", "7d", "30d"]` | `TemporalLagStep`, `MovingAverageStep` |
| `complete_panel: true`, `panel_min_periods: 1` | `CompletePanelStep` |
| `ewma_halflife_days: 7.0` | `EWMAMeanStep` |
| `cyclical_encoding: true` | `CalendarEncodingStep` (cyclical) |
| `spatial_lag: true`, `spatial_lag_rings: [1, 2, 3]` | `MultiScaleSpatialLagStep` |
| `kde_bandwidth_m: 600.0` | `KernelDensityStep` |
| `stkde_bandwidth_m: 600.0`, `stkde_halflife_days: 7.0` | `SpaceTimeKernelDensityStep` |
| `near_repeat: true` | `NearRepeatStep` |
| `hawkes_halflife_days: 7.0` | `HawkesExcitationStep` |
| `trend: true` | `TrendStep` |
| `recency: true` | `TimeSinceLastEventStep`, `BurstinessStep` |
| `zero_features: true` | `ZeroFeaturesStep` |

`measure_cols` is **not** exposed in `FeatureConfig` (known
limitation): the Beijing recipe constructs `CompletePanelStep`
directly. A recipe that wants YAML-declared `measure_cols` needs a
`FeatureConfig` extension.

### eventflow.recipes.registry

Recipe registration and discovery:

- `register_recipe(dataset, name, recipe)`: Register a recipe
- `get_recipe(dataset, name)`: Get registered recipe
- `list_recipes(dataset)`: List available recipes
- `list_datasets()`: List registered datasets

Registered recipes: `chicago_crime / chicago_crime_v1`,
`beijing_air_quality / beijing_air_quality_v1`, `geolife / geolife_v1`.

## Tracking Module

### eventflow.tracking.protocol

#### TrackerProtocol

```python
class TrackerProtocol(Protocol):
    def log_param(self, key: str, value: Any) -> None: ...
    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None: ...
    def set_tags(self, tags: dict[str, str]) -> None: ...
```

### eventflow.tracking.mlflow_tracker

MLflow implementation of TrackerProtocol (no test against a real
MLflow backend).

## CLI Module

### eventflow.cli.main

Command-line interface:

```bash
eventflow run --dataset DATASET --recipe RECIPE --config CONFIG
eventflow list-datasets
eventflow list-recipes --dataset DATASET
eventflow validate --config CONFIG
eventflow version
```

`eventflow run` also accepts `--input` (file or directory of one
format) and `--dataset-config`; `--input` overrides the config's
`raw_root` when both are supplied. `output.partition_by` is not
implemented and is rejected with a blocking error.

## Output Adapters

Adapters convert an `EventFrame` into the shape a model expects:
`TableOutput` (flat rows), `SequenceOutput` (padded sequences),
`RasterOutput` (gridded tensors), `GraphOutput` (node features +
edges), `StreamOutput` (continuous-time event streams).

### Naming (Phase 5)

Aliases are read-only properties returning the identical array object
(`alias is field`); the old names keep working. Serialization carries
both keys (`to_torch()` dicts, metadata shapes/dtypes, NumPy `.npz`
and JSON files); deserializers accept either key, preferring the new
one, so files written before Phase 5 still load. Parquet/Arrow column
names and PyTorch Geometric keys are unchanged.

| Output | Old name | Alias | `.data` points at |
|---|---|---|---|
| `SequenceOutput` | `masks` | `attention_mask` | `sequences` |
| `StreamOutput` | `inter_times` | `inter_event_times` | `states` |
| `StreamOutput` | `states` | `state_features` | `states` |
| `RasterOutput` | `raster` | — | `raster` |
| `GraphOutput` | `node_features` | — | `node_features` |
| `TableOutput` | `data` | — | `data` (plain field) |

`feature_names` is the canonical name vocabulary (`AdapterMetadata`
field, used by 3 of 5 outputs); `state_names` and `channel_names`
remain as domain-accurate spellings, already surfaced as
`feature_names` in metadata. Known limitation: the config flags
`return_masks` / `return_inter_times` are not consulted by
`convert()` — masks and inter-event times are always returned.

```python
from eventflow.core.adapters import (
    SequenceAdapter,
    SequenceAdapterConfig,
    StreamAdapter,
    StreamAdapterConfig,
)

# 2 units x 3 daily bins, one numeric column.
frame = EventFrame(
    pl.LazyFrame({
        "time": [datetime(2024, 1, 1) + timedelta(days=d) for d in range(3)] * 2,
        "grid_id": ["0_0"] * 3 + ["0_1"] * 3,
        "x": [0.0] * 6,
        "y": [0.0] * 6,
        "count": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }),
    EventSchema(timestamp_col="time", lon_col="x", lat_col="y"),
    EventMetadata(dataset_name="docs-adapters", crs="EPSG:3857"),
)

seq = SequenceAdapter(
    SequenceAdapterConfig(spatial_col="grid_id", feature_cols=["count"])
).convert(frame)
# Aliases are the identical arrays: seq.attention_mask is seq.masks.
assert seq.attention_mask is seq.masks
assert seq.data is seq.sequences

stream = StreamAdapter(StreamAdapterConfig(state_cols=["count"])).convert(frame)
assert stream.inter_event_times is stream.inter_times
assert stream.state_features is stream.states
assert stream.data is stream.states
```

## Examples

### Basic Usage

```python
from eventflow.core.pipeline import Pipeline
from eventflow.core.steps import (
    EWMAMeanStep, FillNullStep, TimeSinceLastEventStep, ZeroFeaturesStep,
)
from eventflow.core.steps.temporal import TemporalLagStep

frame = make_panel({u: [1.0, 2.0, 3.0, 4.0] for u in range(9)})
pipeline = Pipeline([
    TemporalLagStep(
        value_cols=["count"], lag_periods=[1], time_unit="1d", group_cols=["grid_id"]
    ),
    FillNullStep(value_cols=["count_lag_1"], fill_value=0),
    EWMAMeanStep(value_cols=["count"], halflife_days=7.0, group_cols=["grid_id"]),
    TimeSinceLastEventStep(value_col="count", group_cols=["grid_id"]),
    ZeroFeaturesStep(value_col="count", group_cols=["grid_id"]),
])
result = pipeline.run(frame).collect()
# count_lag_1, count_ewma_7d, count_time_since_last_d,
# count_active_7d, count_share_nonzero_28d
```

### With Context Enrichment

```python
from eventflow.core.context import EnricherStep, TabularFileSource, TemporalJoin

# weather.csv: ts,temp_c / 2024-01-01 00:00:00,10.0 / 2024-01-01 01:00:00,11.0
weather = TabularFileSource("weather.csv", timestamp_col="ts", value_cols=["temp_c"])
events = EventFrame(
    pl.LazyFrame({
        "time": [datetime(2024, 1, 1, 0, 30), datetime(2024, 1, 1, 1, 30)],
        "x": [0.5, 0.5],
        "y": [0.5, 0.5],
    }),
    EventSchema(timestamp_col="time", lon_col="x", lat_col="y"),
    EventMetadata(dataset_name="docs-context", crs="EPSG:3857"),
)
pipeline = Pipeline([
    EnricherStep(weather, TemporalJoin(strategy="backward")),
])
result = pipeline.run(events).collect()
# Events at 00:30 and 01:30 join the most recent prior row: [10.0, 11.0].
assert result["temp_c"].to_list() == [10.0, 11.0]
```

> The Chicago recipe does **not** wire context enrichment: the
> `ChicagoWeatherSource` / `ChicagoEventsSource` /
> `ChicagoDemographicsSource` classes exist but are unwired and
> untested, and enabling any `context.<source>.enabled` in
> `chicago_crime_v1` raises a blocking error. Use
> `TabularFileSource` for new datasets instead of hand-rolling a
> loader (see [phase3-connectors.md](design-notes/phase3-connectors.md)).

## See Also

- [Architecture Documentation](architecture.md)
- [Dataset Documentation](datasets.md)
- [Observation Panels](observation_panel.md)
- [Observation Quality](observation_quality.md)
- [Spatial Domains](spatial_domains.md)
- [Preprocessing Contracts](preprocessing_contracts.md)
- [Design Notes](design-notes/phase1-causality.md)
- [Known Limitations](audit/known-limitations.md)
- [GitHub Repository](https://github.com/BayoHabib/eventflow)
