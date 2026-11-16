# Observation panels and exposure

Shikong can count raw point-event records on **caller-declared observation
support**. The panel lists the intended spatial units/strata, observation
intervals and collection status. It does not infer the observed domain from the
locations or timestamps at which events happened.

`ObservationPanel` is an immutable, validated reference. `ObservationPanelStep`
integrates it into ordinary synchronous/asynchronous pipelines and the built-in
step registry under `observation_panel`.

## Independent quality declarations

For new panels, configure optional `domain_col`, `collection_col` and
`completeness_col` separately. Collection activity and batch completeness are
never inferred from each other or from record presence. Unspecified quality
stays unknown; a panel with no quality information retains received-record
counts while leaving interpretable totals unknown. The reference rows themselves
define intended domain membership unless a nullable domain flag is supplied.

Optional [cadence checks](observation_quality.md) report scheduled gaps,
duplicates and off-schedule records. They do not fill measurements, synthesize
rows or certify collection/batch completeness. The same document provides the
recommended API and migration details.

## Legacy consolidated status: complete, incomplete and excluded observations

The original `observation_status` remains supported for compatibility. The
examples and status table below use that consolidated legacy declaration;
metadata marks it as legacy, and it is not expanded into invented independent
quality fields. Mixing legacy and independent declarations is rejected.


Intervals are half-open `[start, end)`. Adjacent intervals are allowed;
overlapping intervals for the same key combination are rejected. Interval
lengths may differ. Additional keys can represent an event type, sensor class,
or other stratum; each combination has its own independent support.

| Declared status | `recorded_count` | `event_count` | Interpretation |
|---|---|---|---|
| `observed` | Accepted input rows | Same count, including zero | Collection is declared complete for this support |
| `missing` | Accepted input rows, possibly zero | null | Collection is unknown or incomplete; a reliable total is unavailable |
| `outside` | zero | null | Support is excluded from the observation domain |

The completeness declaration is supplied by the caller. Shikong cannot prove
that a sensor was operational or that an external source reported all events.
Each run treats its input as the full batch for that support, not an incremental
update. Declare `observed` only when the supplied batch covers the corresponding
collection; running a partial batch against fully observed support would assert
zeros for records absent from that batch.
An explicit panel may be sparse; no Cartesian product, domain extension or
additional spatial/temporal rows are invented.

`recorded_count` counts records, including repeated records. Supply
`event_id_col` to require globally unique, non-null identifiers in the input
batch. Otherwise, deduplicate upstream if the domain requires it. Counting
records does not sum rain, average temperature, or reconstruct missing signals.

## Exposure and normalization

Exposure is optional. When supplied, its numerical column and unit are explicit:
sensor-hours, person-years, surveyed area-time, population, or another relevant
denominator. Shikong never substitutes geometric cell coverage or elapsed
interval length for exposure.

- A known exposure must be finite, non-negative and representable in Float64;
  null means unknown. The reference preserves its numeric type (including
  Decimal); rates and log offsets use Float64.
- Outside-domain rows can have zero or unknown exposure, not positive exposure.
- Recorded events with known zero exposure are rejected as inconsistent.
- `event_rate = event_count / exposure` and `log_exposure = ln(exposure)` are
  produced only for `observed` support with known positive exposure.
- An observed zero with positive exposure has rate zero and a finite offset.
- An observed zero with zero/unknown exposure keeps its count zero, but its
  rate and offset remain null. Missing/outside rows never get a rate or offset.
- Without exposure, count fields are still produced; rate/offset fields are not.

Rates are per supplied exposure unit. There is no implicit multiplier such as
"per 100,000 people". Log offsets are natural logs in that chosen numerical
unit. A downstream model must select suitable observation/exposure rows; adapters
do not automatically remove unknown totals or provide a missing-data model.

## Example

```python
from datetime import datetime, timedelta
import polars as pl
from shikong.core import EventFrame, EventMetadata, EventSchema, ObservationPanel
from shikong.core.pipeline import Pipeline
from shikong.core.steps import ObservationPanelStep

t = datetime(2024, 1, 1)
support = pl.DataFrame({
    "site": ["A", "B", "C", "D"],
    "start": [t] * 4,
    "end": [t + timedelta(hours=1)] * 4,
    "observation_status": ["observed", "observed", "missing", "outside"],
    "effort": [2.0, 4.0, 1.0, 0.0],
    "x": [0.5, 1.5, 2.5, 3.5], "y": [0.5] * 4,
})
panel = ObservationPanel(
    support, crs="EPSG:3857", key_cols=["site"],
    exposure_col="effort", exposure_unit="sensor_hours",
)
raw = pl.DataFrame({
    "time": [t + timedelta(minutes=m) for m in [10, 30, 20]],
    "site": ["A", "A", "C"],
    "event_x": [0.1, 0.2, 2.1], "event_y": [0.1] * 3,
})
events = EventFrame(
    raw.lazy(),
    EventSchema(timestamp_col="time", lon_col="event_x", lat_col="event_y"),
    EventMetadata(dataset_name="sensors", crs="EPSG:3857"),
)
pipeline = Pipeline([ObservationPanelStep(panel)])
result = pipeline.run(events)
print(result.collect().select("site", "recorded_count", "event_count", "event_rate"))

# Inside an asyncio application:
# result = await pipeline.run_async(events)
# table = await result.collect_async()
```

| site | recorded_count | event_count | event_rate |
|---|---:|---:|---:|
| A | 2 | 2 | 1.0 |
| B | 0 | 0 | 0.0 |
| C | 1 | null | null |
| D | 0 | null | null |

The sample unit is illustrative; the library imposes no city, bin duration,
grid resolution, or exposure unit. Prepare an appropriate CRS for real data.

## Validation and integration

- Key names and dtypes must agree between events and panel. Null keys/times and
  non-finite numeric identifiers are rejected. The example uses one key; pass
  multiple `key_cols` for independent strata.
- Bounds require identical Polars Datetime precision/time zone. Input events
  must have the same Datetime precision as the panel; normalize explicitly to
  avoid silently losing sub-second boundaries. Different aware time zones are
  converted to the panel zone. Naive/aware mixtures require explicit localization.
- Assign unit/stratum keys upstream. The count operation joins those keys and
  timestamps; it does not use event coordinates to assign spatial membership.
- The reference supplies finite `x`/`y` support coordinates by default. For WKT
  geometry support, set `x_col=None, y_col=None, geometry_col="geometry"`.
  The step requires compatible event/reference CRS and does not transform or
  invent support coordinates.
- Records with no matching interval/key, or in outside-domain support, raise by
  default. `events_outside="ignore"` explicitly excludes them and records their
  count in diagnostics. Invalid/null keys are errors even with this option.
- The step expects unbinned point-event records. It rejects a declared temporal
  bin/resolution and repeated counting of panel output. Create counts before
  downstream bin annotations, temporal lags, or spatial statistics.
- Output reference order is preserved. The EventFrame timestamp names the
  interval start; both interval bounds remain available. Coordinates describe
  the support, not the original event location. Active schema attributes are
  updated and source measurements are not carried over as fabricated values.
- Existing feature provenance remains as source lineage. New count/rate fields
  have their own provenance. `metadata.custom["observation_panel"]` and context
  notes record interval conventions, exposure unit, support fingerprint, source
  attributes, and input/accepted/excluded record counts. The source EventFrame
  is unchanged. Legacy point bounds are cleared and date range covers support.
- Reference tables are cloned; configuration is immutable. Collection is a
  materialization boundary: sorting/interval matching uses the selected input
  key/time/ID columns in memory. The async wrapper offloads this bounded work.

For registry composition, supply the panel as a runtime object:

```python
from shikong.core.steps import get_default_registry
step = get_default_registry().create("observation_panel", params={"panel": panel})
```

A YAML reference-file loader, automatic sensor-uptime estimation, availability
time/bitemporal matching, and support-aware resampling of continuous measurements
remain separate work. Scheduled-gap diagnostics are available when an explicit
cadence rule is supplied; they do not establish actual uptime. This is not an implementation of the complete OGC
observation model. See [preprocessing contracts](preprocessing_contracts.md) and
the [OGC observation semantics](https://docs.ogc.org/as/20-082r4/20-082r4.html).
