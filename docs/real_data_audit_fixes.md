# Real-data audit fixes and migration

The November 2026 audit of commit `e88d54cec3cc2556a4ab31b17ee9152eea3f37a4`
found two integration gaps and a local H-transform inconsistency. This change
fixes the compositions and defines an explicit estimator contract. These are
changes to generic preprocessing; no Chicago-specific geometry or time interval
is added to the core.

## Shared, versioned grid identity

`create_grid`, `assign_to_grid` and `AssignToGridStep` now use the same
`GridSpec(size_m, origin, crs)` and `signed_cantor_v1` cell identifiers. Indices
are signed, cells are half-open, and no fixed column stride or batch width is
used. The pairing is:

```text
a = 2*row if row >= 0 else -2*row-1
b = 2*column if column >= 0 else -2*column-1
grid_id = (a+b)*(a+b+1)//2 + b
```

Indices are limited to +/-1,000,000,000 on each axis so every supported pair and
intermediate operation is safe in Int64. Excessive indices raise; they never
wrap or collide. Numeric proximity/order of IDs has no spatial interpretation.

```python
from eventflow.core import GridSpec
from eventflow.core.spatial import create_grid, assign_to_grid
from eventflow.core.steps.spatial import AssignToGridStep

# projected_events already uses this metric CRS.
origin = (350000.0, 570000.0)
grid = create_grid(
    (350000.0, 570000.0, 352000.0, 572000.0), 500,
    crs="EPSG:26971", origin=origin,
)
a = assign_to_grid(projected_events, grid)
b = AssignToGridStep(500, origin=origin).run(projected_events)
# IDs/indices agree for points inside grid. a is bounded by that reference;
# b assigns the entire anchored lattice, including points beyond those bounds.
```

A `grid_id` is scoped to its lattice. Compare the versioned `grid_spec`
fingerprint in `EventMetadata.custom` when persisting or joining different
products. Identical size, origin and CRS produce the same identity independent
of batch width; different origins or CRSs are distinct lattices. Supplying a
fixed origin is essential for multiple batches. Omitting it still uses each
batch's minimum coordinates and does not promise cross-batch stability.

Core grid references now retain origin/CRS and grid_row/grid_col columns.
Shuffled and sparse references work. Core assignment checks canonical identity,
square alignment and CRS; points outside the supplied reference keep null IDs
rather than aliasing another reference cell. Pipeline grid assignment remains
unbounded, subject to the explicit Int64 index range.

### Migrating persisted IDs

**Both old encodings change**: core row-major numbering and the pipeline's
100000-stride numbering are not `signed_cantor_v1`. Existing tables are never
silently assumed to use the new convention. `assign_to_grid` rejects legacy IDs
and asks for migration. Rebuild references from their original geometry and
known grid parameters. Keep an old-to-new mapping if external records use the
old numbers; attach attributes by geometry/indices, not numeric ID similarity.

For a table with reliable cell centers and known projection, origin and size:

```python
import polars as pl
from eventflow.core import GridSpec

spec = GridSpec(300, (0.0, 0.0), "EPSG:26971")
migrated = spec.assign(
    previous.lazy().with_columns(pl.col("grid_id").alias("legacy_grid_id")),
    "longitude_proj", "latitude_proj",
).collect()
# Persist spec.metadata() with the migrated table.
```

Do not infer old grid parameters from the old ID alone. Without the original
origin/resolution/CRS or geometry, an unambiguous migration is not established.

## Aggregation preserves spatial support

`ChicagoCountStep` now retains `grid_row` and `grid_col`; it also preserves the
shared lattice metadata. The default recipe output can therefore be passed to
`SpatialLagStep(["count"], group_cols=["time_bin"])` without reconstructing an
external grid reference. Queen contiguity follows the retained indices, never
ID ordering. Counts, cell centers and time-bin totals are unchanged.

This recipe returns occupied bins only. Its spatial lag describes present
neighbors in each snapshot. If the analysis requires a full support, construct
an explicit observation panel with the appropriate quality declarations. Missing
records must not be silently replaced by observed zeros. Domain barriers still
require caller-defined polygons or connectivity; preserving indices does not
infer geophysical constraints.

## Local K/L/H with declared observation support

The old step counted neighbors without an observation-window normalization,
accepted correction options it did not apply, and computed H using r/2 with a
special zero for isolated points. The step now requires a metric `SpatialDomain`
window, implements the local K normalization and applies H=L-r for every defined
value. The default edge correction is translation; `none` is an explicitly
uncorrected alternative.

```python
from shapely.geometry import box
from eventflow.core import SpatialDomain
from eventflow.core.steps.point_process import KFunctionStep

window = SpatialDomain(box(350000, 570000, 360000, 580000), "EPSG:26971")
result = KFunctionStep(
    max_distance=1000, window=window,
    edge_correction="translation", transform="H",
).run(events_in_this_window)
```

Configured pipelines use the existing serializable domain representation:

```python
from eventflow.core.steps import get_default_registry
step = get_default_registry().create("k_function", params={
    "max_distance": 1000,
    "window": {"allowed_wkt": [window.geometry.wkt], "crs": window.crs},
    "edge_correction": "translation", "transform": "H",
})
```

Compatibility changes are intentional: running without a window raises; the
old unsupported `ripley`/`border` options and unused `n_bins` raise explicitly.
A geographic window must be converted by the caller to an appropriate metric
CRS. The event coordinates can use another CRS and are transformed internally.
All points must be finite and covered by the declared window. Apply any filtering
policy explicitly before the calculation. The library never derives observation
coverage/completeness from the event bounding box.

`k_function_local` is always included; L/H selection adds its corresponding
column. K is in m2 and L/H in m, with window fingerprint, normalization and
correction recorded in feature provenance. n < 2 gives null values. A point with
no neighbors in an otherwise estimable pattern has K=0, L=0 and H=-r. See
[point-process contracts](point_process_features.md#local-klh-contract) for
stationarity, coincident records, edge bias and local/global interpretation.

## Validation and real-data replay

Analytic regression tests use a 10-by-10 meter window and three points. At a
3-meter radius, two points are neighbors and the third is isolated:

| Quantity | No edge correction | Translation correction |
|---|---:|---:|
| Window area | 100 m2 | 100 m2 |
| Pair displacement | (2, 0) m | (2, 0) m |
| Overlap area | not used | 80 m2 |
| Pair weight | 1 | 1.25 |
| Local K for each neighboring point | 50 m2 | 62.5 m2 |
| Isolated point H | -3 m | -3 m |

Tests also cover unit rescaling, projection consistency, holes/outside points,
undefined n<2 cases, unsupported corrections, coincident records, zero overlap,
registry construction and async parity. Grid regressions exercise signed and
large indices, the previous stride collision, reference coverage joins, bounds,
legacy-ID rejection, batch stability and the recipe-to-lag composition.

The frozen inputs are in the separately delivered
`EventFlow_Real_Data_Validation_Bundle.zip`. Extract it and run:

```bash
python scripts/check_real_data_audit.py --audit-root /path/to/extracted/bundle
# Optional: --output /path/to/new-results.json
```

The script verifies the nine original input/support SHA-256 hashes. It does not
change that bundle or its original report. The follow-up checks include all
19,509 retained incidents, core/pipeline identity, domain coverage, every one of
the 18,261 default-recipe spatial lags against an independent eight-offset
reference, async parity, K normalization and H/L consistency on 100 real points,
and preservation of the NOAA archive's 38 missing hours and one null temperature.
The real K calculation is an arithmetic check on a selected pattern, not an
inferential claim about Chicago crime. Full result details and source hashes
are recorded in [validation evidence](validation/real_data_audit_fixes.json).

The original runner intentionally pins the original source snapshot. It should
continue to reproduce the original failures; it is not repurposed to hide them.

## Statistical references

- [spatstat localK/localL](https://search.r-project.org/CRAN/refmans/spatstat.explore/html/localK.html): local normalization and local/global interpretation.
- [spatstat Kest](https://search.r-project.org/CRAN/refmans/spatstat.explore/html/Kest.html): intensity, window, translation correction and stationary-process assumptions.
- [Centered L](https://search.r-project.org/CRAN/refmans/onpoint/html/center_l_function.html): centering by L(r)-r.
