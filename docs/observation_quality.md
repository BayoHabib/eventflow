# Independent observation quality and cadence checks

Shikong keeps four different questions separate:

| Information | Meaning | What it does not establish |
|---|---|---|
| Study domain | Intended spatial units and time intervals | Actual operation of a source |
| Collection activity | Declared active, inactive or unknown collection | Receipt of every expected record |
| Batch completeness | Declared complete, incomplete or unknown input batch | Physical detection of all events |
| Exposure | An explicitly supplied denominator and unit | Collection activity or batch completeness |

The quality fields are optional. Their absence, or an explicit null/`unknown`,
does not mean active collection or complete data. Rows in the supplied reference
define the intended study domain unless an explicit nullable Boolean domain
column is configured. Null membership remains unknown; false membership means
outside. The library does not discover the study domain from event locations.

## Independent declarations

```python
from datetime import datetime, timedelta
import polars as pl
from shikong.core import ObservationPanel

t = datetime(2024, 1, 1)
reference = pl.DataFrame({
    "site": ["A", "B", "C", "D"],
    "start": [t] * 4, "end": [t + timedelta(minutes=40)] * 4,
    "x": [0.0, 1.0, 2.0, 3.0], "y": [0.0] * 4,
    "in_study": [True, True, True, False],
    "collection": ["active", "inactive", "unknown", "active"],
    "batch": ["complete", "complete", "unknown", "complete"],
})
panel = ObservationPanel(
    reference, "EPSG:3857", key_cols=["site"],
    domain_col="in_study", collection_col="collection",
    completeness_col="batch",
)
```

With independent declarations, an interpretable `event_count` requires known
in-domain support, active collection, and a complete input batch. Otherwise it
stays null, even when some records have arrived. `recorded_count` still reports
the accepted records actually present. Thus an inactive source with a completely
received empty batch is different from an active source with no detected event.
Exposure/rate/offset rules remain unchanged; no exposure is inferred from quality.

Without any quality fields, counts remain unknown. The reference still defines
the intended support; raw received-record counts remain available. True physical
event totals are not certified by any of these declarations.

## Opt-in cadence checks

A `CadenceRule` describes expected record timestamps. It is appropriate for
scheduled measurements or heartbeats, not automatically for an irregular event
stream such as incidents, purchases or animal sightings.

```python
from shikong.core import CadenceRule

records = pl.DataFrame({
    "site": ["A"] * 4,
    "time": [t + timedelta(minutes=m) for m in [0, 10, 10, 30]],
}).lazy()
report, diagnostics = panel.check_cadence(
    records, "time", CadenceRule(every="10m"),
    include_missing_times=True,
)
print(report.select("site", "cadence_expected", "cadence_present",
                    "cadence_missing", "cadence_duplicates", "cadence_status"))
```

Site A has four expected slots, three represented slots, one missing slot at
00:20, and one duplicate. Having four received rows does not conceal that gap.
Neither site B's collection status nor site C's batch completeness is changed
by the diagnostic. Site D is outside and the check is not applicable there.

Each interval uses `[start, end)`. The schedule is anchored at its start by
default. Supply `origin=...` for a global phase: it is never estimated from the
first received record. A global origin must be an explicit datetime or ISO string;
numeric epoch units are not guessed. Python datetime origins have microsecond
precision: sub-microsecond global origins are rejected instead of truncated.
Use native interval starts for nanosecond phases. Periods are fixed elapsed durations, including through
DST changes, not calendar-day schedules. Period/tolerance/origin must be exactly
representable at the panel timestamp precision; timestamps are checked as
integers to preserve nanosecond identities.

Optional `tolerance="30s"` allows a bounded distance from a scheduled timestamp.
It must be less than half the period, so one record cannot fill two slots.
Matching is confined to the record's own panel interval; a neighboring interval
cannot borrow a record across a boundary. Distinct-slot counts, duplicate rows,
and off-schedule rows are reported separately.

| Cadence status | Meaning |
|---|---|
| `unknown` | No rule, or unknown domain applicability |
| `not_applicable` | Outside the declared study domain |
| `gaps` | At least one expected slot has no matching record |
| `satisfied` | Every expected slot has a matching record; duplicates/extras may still exist |

The diagnostic reports `cadence_expected`, `cadence_present`, `cadence_missing`,
`cadence_duplicates`, `cadence_off_schedule`, and `cadence_status` per reference
row. `include_missing_times=True` additionally reports typed timestamp lists.
Without a rule, numerical diagnostics are null, not zero. A short interval with
no scheduled slot under an explicit global origin has zero expected slots; that
does not certify batch completeness.

**A satisfied cadence does not certify sensor operation or complete receipt of
all possible records.** Received scheduled timestamps say nothing about missed
physical events or additional event types. Gaps do not rewrite caller-supplied
declarations either. Default `on_gap="report"` preserves them for inspection;
`on_gap="raise"` stops processing when the configured schedule has gaps.

## Pipeline integration

```python
from shikong.core.steps import ObservationPanelStep

step = ObservationPanelStep(
    panel,
    cadence_rule={"every": "10m", "tolerance": "30s", "max_slots": 100_000},
    include_missing_times=True,
    on_gap="raise",  # An explicit validation policy for this pipeline.
)
```

The step can count and check the same matched input in one run, including through
`await pipeline.run_async(...)`. It also works through the existing
`observation_panel` registry entry. Without a rule, the count output schema is
unchanged and metadata records that the cadence is unknown/not evaluated.

Checks never add synthetic measurement rows, interpolate values or replace
missing values by zero. They materialize selected input columns and scan records
in Python, retaining distinct slot indices only. `max_slots` bounds the total
evaluated schedule across all applicable reference rows; exceeding the budget
raises before building missing-slot lists. This is bounded batch checking, not
an out-of-order streaming/lateness policy. A partially delivered live batch may
show provisional gaps; data availability still needs explicit handling upstream.

## Compatibility and migration

The original `observation_status` values `observed`, `missing`, and `outside`
remain supported. If that column is present and no independent field is
configured, it selects legacy mode. A custom `status_col` explicitly selects
that consolidated legacy declaration. Its numerical behavior is preserved, and
metadata marks `quality_mode="legacy"`; `observed` is not expanded into invented
collection/completeness fields.

To use the independent model, remove the legacy column and configure the desired
quality columns. Mixing both models is rejected. A panel with neither a legacy
status nor independent declarations uses the safe unknown-quality mode.
Fingerprints include the versioned quality contract and selected field roles.
Cadence rule parameters and the gap policy are recorded separately in output
metadata/provenance. See [observation panels](observation_panel.md) for exposure,
record identity, and outside-support handling.
