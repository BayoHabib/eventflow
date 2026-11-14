# Generic spatial-domain preprocessing

EventFlow prepares spatio-temporal data for downstream analyses. Its generic core
provides composable operations; datasets and recipes supply source data, domains,
parameters, and domain-specific interpretations.

A spatial domain describes a **caller-defined, static 2D footprint**. It may
represent a survey area, sensor coverage, a service territory, or any polygonal
study boundary. The caller supplies exclusions when they apply to the phenomenon.
No city, land/water classification, grid size, or time interval is built into
these operations.

## Define a footprint

```python
from eventflow.core import SpatialDomain

domain = SpatialDomain.from_wkt(
    allowed_wkt=["POLYGON ((0 0, 4 0, 4 2, 0 2, 0 0))"],
    excluded_wkt=["POLYGON ((2 0, 3 0, 3 2, 2 2, 2 0))"],
    crs="EPSG:3857",
    name="survey-footprint-v1",
)
```

Allowed polygons are unioned, so overlapping areas are counted once. Exclusions
are subtracted. Holes, multiple disconnected parts, and a completely excluded
domain are supported. Geometries must be valid Polygon/MultiPolygon inputs.
Coordinates are evaluated in 2D. `SpatialDomain(geometry, crs, name)` also accepts
an existing Shapely polygon.

The domain records its name, CRS, and a SHA-256 of its geometry representation
and CRS. This identifies the supplied representation, not topological equivalence
between differently encoded polygons.

## Annotate or explicitly filter event locations

```python
from eventflow.core.pipeline import Pipeline
from eventflow.core.steps import SpatialDomainMaskStep

pipeline = Pipeline([
    SpatialDomainMaskStep(domain),  # Annotation is the default.
])
prepared = pipeline.run(events)  # events is an EventFrame with any dataset/schema.
```

The step resolves coordinates through `EventSchema`; arbitrary column names are
supported. It reprojects the reference domain into `EventMetadata.crs` without
changing event coordinates, timestamps, or dataset identity. Event evaluation
remains lazy and runs in vectorized batches. Reference polygons are validated
eagerly as auxiliary data.

| `inside_domain` | Meaning |
|---|---|
| `true` | The location is in the caller-defined domain. |
| `false` | The location is known and outside the domain or in an exclusion. |
| `null` | The location is missing or its point coordinates are not finite. |

`mode="filter"` explicitly retains only `true` rows. Boundary points are included
by default; `include_boundary=False` requires strict interior containment. For
geometry-only `EventSchema.geometry_col` inputs, the **whole geometry** must be
covered, not just its centroid or one intersecting point. Coordinate columns take
precedence when a schema supplies both representations; the caller selects the
active representation through the schema. Invalid WKT/geometry is an error.

Domain identity, boundary policy, and operation mode are recorded in feature
provenance and context notes. An existing output column is never overwritten.

## Measure cell coverage

```python
import polars as pl
from eventflow.core import annotate_grid_coverage

cells = pl.DataFrame({
    "cell": [10, 20, 30],
    "footprint": [
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
        "POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))",
        "POLYGON ((4 0, 6 0, 6 2, 4 2, 4 0))",
    ],
})
coverage = annotate_grid_coverage(
    cells, domain, crs="EPSG:3857", geometry_col="footprint",
)
# domain_area_m2: [4, 2, 0]
# domain_fraction: [1.0, 0.5, 0.0]
# domain_covered: [True, True, False]
```

This supports any caller-supplied polygon cells, including irregular zones.
Original cells and zero-coverage cells remain present. The caller can select an
appropriate fraction threshold or construct a grid using existing helpers.

The measurements are **planar geometric areas in the chosen projected CRS**.
A metric CRS is required; geographic and foot-based grids are rejected. Choose
a projection appropriate to the study area and required area accuracy. A geometric
fraction is separate from population, traffic, observation effort, sampling
probability, or travel accessibility. Those can be supplied through context data.

`GridDomainCoverageStep(domain, cells, crs, grid_id_col="cell",
geometry_col="footprint")` joins these columns onto an EventFrame using explicit
cell identifiers. Supplied IDs must be unique and match the event grid. Unknown
IDs yield `null` coverage, distinct from a known cell with zero intersection.
The operation retains the event time schema and records domain provenance.

## Configured pipelines

The default registry exposes `spatial_domain_mask` with a validated YAML config:

```yaml
- name: spatial_domain_mask
  params:
    mode: annotate
    include_boundary: true
    domain:
      name: survey-footprint-v1
      crs: EPSG:3857
      allowed_wkt:
        - "POLYGON ((0 0, 4 0, 4 2, 0 2, 0 0))"
      excluded_wkt:
        - "POLYGON ((2 0, 3 0, 3 2, 2 2, 2 0))"
```

`get_default_registry().build_pipeline(yaml.safe_load(text))` assembles this
operation alongside existing spatial/temporal steps. `grid_domain_coverage` can
be constructed through the registry with Python parameters including its grid
DataFrame; external grid loading remains the caller's responsibility.

These operators prepare data. They do not infer whether an absent event is a
zero, missing observation, or unobserved area. Observation calendars and panel
completion are separate choices. Time-varying domains, terrain costs, and road
network routing require additional reference data and operations.
