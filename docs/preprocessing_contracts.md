# Spatio-temporal preprocessing contracts

This document separates implemented numerical guarantees from the remaining
work needed for a broader generic preparation library. None of the defaults is
a domain-specific choice such as Chicago, six-hour bins, a city CRS, or a
particular physical barrier.

## Elapsed time and row counts

`TemporalLagStep` now defaults to `mode="duration"`. A lag of 1 with
`time_unit="1d"` looks for the same entity exactly 24 elapsed hours earlier.
Missing observations yield null, never the previous available row or zero.
Use `group_cols` for entity identity. Duplicate entity/timestamp keys and null
keys are rejected; aggregate or disambiguate observations before this operation.

```python
TemporalLagStep(["temperature"], [1, 7], time_unit="1d", group_cols=["station"])
MovingAverageStep(["temperature"], [6], time_unit="1h", group_cols=["station"])
```

Duration moving means use `[t - window, t)` by default (`closed="left"`): they
exclude the current measurement and all simultaneous measurements. Other
boundary policies are explicit. Values inside a window receive equal weight;
this is an observation mean, not a duration-weighted integral of a signal.
Duration windows reject duplicate keys rather than inventing tie semantics.

Fixed-duration parsing recognizes positive multipliers such as `6h`, `15min`
and `2d`; days are 24 hours and weeks are seven such days, including across DST.
Months and years are not fixed elapsed durations. Unsupported units raise an
error instead of silently falling back to a day or an hour. Calendar-day bins
remain a separate Polars operation. Time-zone metadata does not localize naive
timestamps: callers must supply correctly localized/converted data.

The supported Polars floor is 1.30.0. CRS projection materializes its input and
creates/uses PROJ state in the owning Python thread, outside Polars expression
callbacks. This avoids a native crash reproduced with older query engines
re-evaluating projection callbacks on worker threads. The async wrapper offloads
the whole step. NumPy is declared as a direct runtime dependency.

Migration: `mode="rows"` retains the previous row-based lag/rolling behavior.
Row means include the current row and may be centered with `center=True` for
retrospective use. `closed` applies only to duration windows. A centered row
window may include future information. Both modes retain chronological ordering
within each entity. These changes can intentionally change numerical outputs.

## Spatial units and neighborhoods

Spatial lag, local Moran and Gi* no longer use sorted IDs as a neighborhood.
`AssignToGridStep` retains `grid_row` and `grid_col`. Queen neighbors share an
edge or corner; rook neighbors share an edge. Arbitrary polygon units can be
supplied in a reference table using WKT `geometry` instead.

`SpatialWeights.from_units` also builds directed KNN or binary inclusive
distance bands from reference `x`/`y` coordinates in the EventFrame's metric
CRS. KNN ties follow reference-table order. Thresholds are in meters. If the
events contain multiple positions per unit, supply one representative location
per unit explicitly; EventFlow does not choose an event centroid implicitly.

```python
from eventflow.core.weights import SpatialWeights

# Caller-defined connectivity can exclude edges crossing a river, or follow a
# supplied transport/network graph. Include reverse edges if appropriate.
weights = SpatialWeights.from_edges(
    ["upstream", "downstream", "other_basin"],
    [("upstream", "downstream"), ("downstream", "upstream")],
)
step = SpatialLagStep(["value"], neighbor_col="site", spatial_weights=weights)
```

These weights are binary. Spatial lag takes a row-standardized mean, excludes
self, and returns null for islands. The graph can be directed. A geometric
domain mask does not automatically modify connectivity or infer physical
barriers; supply the relevant graph. No transport or geophysical model is
assumed.

Use `group_cols=["time_bin"]` (or other snapshot keys) to avoid pooling across
time. Without `group_cols`, the input is one spatial snapshot. Lag and local
Moran aggregate each unit by mean; Gi* aggregates by sum. Non-finite values
require a caller-selected missing-data policy before these statistics.

Reference graphs must contain the same units as each snapshot. Missing units
raise an error: a missing observation is not implicitly zero. Automatic graphs
use the units actually present in a snapshot and do not create a full panel.
Adjacency lists avoid a dense matrix; metric distance/KNN construction still
does quadratic distance work on the reference units and is intended for
bounded snapshots, not an unbounded event stream.

## Statistical definitions and validation

- Local Moran uses `I_i = (x_i - mean(x)) * lag(x - mean(x)) / m2`, where
  `m2 = sum((x - mean(x))**2) / n`, and row-standardized, self-excluding weights.
  This is the population-second-moment convention used by `spdep` with
  `mlvar=True`. PySAL's sample convention differs by the factor `n/(n-1)`.
- Conditional randomization holds the focal observation fixed and samples the
  other values without replacement for its neighbors. Two-sided Monte Carlo
  p-values compare absolute deviations from the conditional expectation
  `-((x_i - mean(x))**2) / ((n-1)*m2)` and use a +1 correction. `seed` makes
  simulation reproducible. Cluster labels require `p <= significance`; with
  zero permutations no significance is inferred. No multiple-testing adjustment
  is performed. Islands or zero variance yield null statistics/p-values and NS.
- Gi* uses binary neighbors plus self. Its normal z-score denominator is
  `S * sqrt((n*sum(w**2) - sum(w)**2)/(n-1))`, where `S` is the population
  standard deviation. Zero variance or a degenerate neighborhood yields null.
  Hot/cold labels use unadjusted normal cutoffs +/-1.96; small-sample validity
  and multiplicity remain the analyst's responsibility.
- Hawkes features use independent histories for `group_cols`. Only strictly
  earlier timestamps contribute; simultaneous events do not excite each other.
  `max_history` limits prior events within each group. Intensities are per
  configured time unit. Parameters are supplied, not fitted by this step; the
  retained history is an explicit approximation to the infinite-history model.

Hand-calculated fixtures and counterexamples run in the standard test suite.
The optional `scripts/check_spatial_reference.py` compares local I and Gi* with
PySAL, accounting for the different variance convention. It was checked with
esda 2.10.0 and libpysal 4.15.1. Neither is a runtime dependency. The Monte Carlo
p-value convention is specified above; parity with PySAL's permutation p-values
is not claimed.

## Context joins

Point/polygon `contains` tests the point within the polygon (strict interior).
`intersects` includes boundaries. If multiple geometries match, the first row
of the context reference wins deterministically; callers should check overlaps
or aggregate explicitly. Nearest joins require a metric CRS, apply
`max_distance`, and resolve ties by context row order. Unknown or non-finite
locations stay unmatched. Joined columns retain their source data types.

Context projection and spatial lookup complete in the executing worker, so a
lazy expression does not capture a native projection handle from another thread.
This join materializes its event input; its async step wrapper keeps that work
off the event loop but does not make it out-of-core.

Temporal `tolerance`/`window` parameters are now applied in the step, strategy
and functional APIs; as-of inputs are sorted. Backward matching selects past
timestamps; forward and nearest matching remain valid retrospective options
but can use future data. Tolerances use Polars duration syntax.

## Remaining foundations

The numerical contracts and [observation panels](observation_panel.md) cover
bounded batch preparation. Explicit intervals, observed/missing/outside status,
and optional exposure/rates are implemented for point-event counts. These
foundations still require work:

1. [Independent quality declarations and explicit cadence checks](observation_quality.md)
   are implemented. Automatic coverage inference, reference-file loading, and
   continuous-measure support-aware resampling remain separate work. Geometric coverage is not population or observation
   exposure; completeness still needs a caller-supplied declaration.
2. Separate phenomenon time and data-availability time, bitemporal joins, and
   fit/transform boundaries for learned preprocessing. A backward join alone
   does not guarantee that a value was available at prediction time.
3. A generic raw-data quality stage, measurement units/uncertainty contracts,
   and explicit policies for geometry repair, duplicate records and imputation.
4. Change-of-support rules: additive totals, intensive measurements,
   duration/area-weighted aggregation, and multi-resolution sensitivity checks.
5. Native trajectory and field/cube preparation through appropriate adapters.
6. True out-of-order streaming semantics (event time, lateness, updates and
   recovery). Awaitable batch execution does not supply those semantics.

[Local K/L/H](real_data_audit_fixes.md#local-klh-with-declared-observation-support)
now require an explicit metric window and implement window normalization and
translation correction, with separate analytic tests. Pair correlation, KDE
and survival-related features are not certified by those checks; they still
require their own estimator and boundary audits.

## References

- Pebesma (2012), [spacetime: Spatio-Temporal Data in R](https://doi.org/10.18637/jss.v051.i07).
- OGC (2023), [Observations, measurements and samples](https://docs.ogc.org/as/20-082r4/20-082r4.html).
- Anselin (1995), [Local Indicators of Spatial Association—LISA](https://doi.org/10.1111/j.1538-4632.1995.tb00338.x);
  [spdep implementation conventions](https://r-spatial.github.io/spdep/reference/localmoran.html).
- [PySAL local G reference](https://pysal.org/esda/stable/generated/esda.G_Local.html).
- Cheng and Adepeju (2014), [Modifiable Temporal Unit Problem](https://doi.org/10.1371/journal.pone.0100465).
- Roberts et al. (2017), [Cross-validation strategies for structured data](https://doi.org/10.1111/ecog.02881).
- Akidau et al. (2015), [The Dataflow Model](https://research.google.com/pubs/archive/43864.pdf).
