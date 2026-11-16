"""Observation support is supplied, never inferred from the presence of events."""

from __future__ import annotations

import asyncio
import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest
from shapely.geometry import box

from shikong.core import EventFrame, EventMetadata, EventSchema, ObservationPanel
from shikong.core.pipeline import Pipeline
from shikong.core.steps import ObservationPanelStep, get_default_registry
from shikong.core.steps.temporal import TemporalLagStep

T = datetime(2024, 1, 1)


def cells() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unit": ["A", "A", "B", "C", "D", "E"],
            "start": [T, T + timedelta(hours=1), T, T, T, T],
            "end": [T + timedelta(hours=1), T + timedelta(hours=2), *[T + timedelta(hours=1)] * 4],
            "observation_status": [
                "observed",
                "missing",
                "observed",
                "outside",
                "observed",
                "observed",
            ],
            "effort": [2.0, 1.0, 4.0, 0.0, 0.0, None],
            "x": [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0] * 6,
        }
    )


def panel(data: pl.DataFrame | None = None, **kwargs) -> ObservationPanel:
    return ObservationPanel(
        cells() if data is None else data,
        "EPSG:3857",
        key_cols=["unit"],
        exposure_col="effort",
        exposure_unit="sensor_hours",
        **kwargs,
    )


def events(**overrides) -> EventFrame:
    data = {
        "time": [T + timedelta(minutes=10), T + timedelta(minutes=30), T + timedelta(minutes=70)],
        "unit": ["A"] * 3,
        "event_id": [1, 2, 3],
        "event_x": [9.0] * 3,
        "event_y": [8.0] * 3,
        "measurement": [10.0, 100.0, 1000.0],
    }
    data.update(overrides)
    return EventFrame(
        pl.DataFrame(data).lazy(),
        EventSchema(
            timestamp_col="time", lon_col="event_x", lat_col="event_y", numeric_cols=["measurement"]
        ),
        EventMetadata(dataset_name="sensors", crs="EPSG:3857"),
    )


def test_observed_zero_missing_and_outside_are_distinct() -> None:
    output = ObservationPanelStep(panel()).run(events()).collect()
    assert output["event_count"].to_list() == [2, None, 0, None, 0, 0]
    assert output["recorded_count"].to_list() == [2, 1, 0, 0, 0, 0]
    assert output["event_rate"].to_list() == [1.0, None, 0.0, None, None, None]
    assert output["log_exposure"].to_list()[:3] == [math.log(2), None, math.log(4)]
    assert output["log_exposure"].to_list()[3:] == [None] * 3
    assert output.height == cells().height
    assert output["event_count"].dtype == pl.Int64
    assert output["x"].to_list() == cells()["x"].to_list()
    assert "measurement" not in output.columns  # Counting records does not sum a measurement.


def test_schema_and_lineage_describe_interval_support() -> None:
    source = events().register_feature("measurement", {"source_step": "source"})
    result = Pipeline([ObservationPanelStep(panel())]).run(source)
    data = result.collect()
    assert result.schema.lon_col == "x" and result.schema.lat_col == "y"
    assert "measurement" not in result.schema.numeric_cols
    assert set(result.schema.numeric_cols) <= set(data.columns)
    assert result.metadata.feature_provenance["event_count"].produced_by == "ObservationPanelStep"
    assert "measurement" in result.metadata.feature_provenance  # Retained as source lineage.
    note = result.metadata.custom["observation_panel"]
    assert note["bounds"] == "[start, end)" and note["timestamp_role"] == "interval start"
    assert note["accepted_records"] == 3 and note["excluded_records"] == 0
    assert note["exposure_unit"] == "sensor_hours"
    assert result.metadata.bounds is None
    assert source.collect()["measurement"].to_list() == [10.0, 100.0, 1000.0]


def test_half_open_intervals_assign_boundary_to_next_interval() -> None:
    data = events(time=[T, T + timedelta(hours=1), T + timedelta(hours=1)])
    result = ObservationPanelStep(panel()).run(data).collect()
    assert result["recorded_count"].to_list()[:2] == [1, 2]


@pytest.mark.parametrize(
    "change", ["unknown_unit", "before", "at_final_end", "outside_unit", "gap"]
)
def test_outside_support_errors_by_default(change: str) -> None:
    source = events()
    p = panel()
    if change == "unknown_unit":
        source = source.with_columns(pl.lit("Z").alias("unit"))
    elif change == "outside_unit":
        source = source.with_columns(pl.lit("C").alias("unit"))
    elif change == "before":
        source = source.with_columns(pl.lit(T - timedelta(seconds=1)).alias("time"))
    elif change == "at_final_end":
        source = source.with_columns(pl.lit(T + timedelta(hours=2)).alias("time"))
    else:
        p = panel(
            cells().with_columns(
                pl.when(pl.col("start") == T + timedelta(hours=1))
                .then(pl.lit(T + timedelta(minutes=90)))
                .otherwise(pl.col("start"))
                .alias("start")
            )
        )
    with pytest.raises(ValueError, match="outside the declared observation support"):
        ObservationPanelStep(p).run(source)


def test_ignoring_outside_records_is_explicit_and_audited() -> None:
    source = events(unit=["A", "C", "Z"])
    result = ObservationPanelStep(panel(), events_outside="ignore").run(source)
    assert result.collect()["recorded_count"].to_list() == [1, 0, 0, 0, 0, 0]
    audit = result.metadata.custom["observation_panel"]
    assert audit["input_records"] == 3 and audit["excluded_records"] == 2
    assert audit["events_outside"] == "ignore"


@pytest.mark.parametrize("status", ["unknown", "Observed", None])
def test_invalid_or_unknown_status_is_rejected(status) -> None:
    with pytest.raises(ValueError):
        panel(cells().with_columns(pl.lit(status, dtype=pl.String).alias("observation_status")))


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_invalid_exposure_never_normalizes(value: float) -> None:
    with pytest.raises(ValueError, match="Exposure must be finite"):
        panel(cells().with_columns(pl.lit(value).alias("effort")))


def test_outside_rows_cannot_have_positive_exposure() -> None:
    with pytest.raises(ValueError, match="Outside-domain"):
        panel(cells().with_columns(pl.lit(1.0).alias("effort")))


def test_recorded_event_with_known_zero_exposure_is_a_contradiction() -> None:
    with pytest.raises(ValueError, match="zero exposure"):
        ObservationPanelStep(panel()).run(events(unit=["D"] * 3).filter(pl.col("event_id") < 3))


def test_exposure_unknown_keeps_counts_known_and_rate_unknown() -> None:
    source = events(unit=["E"] * 3).filter(pl.col("event_id") < 3)
    output = ObservationPanelStep(panel()).run(source).collect()
    assert output["event_count"][-1] == 2
    assert output["event_rate"][-1] is None and output["log_exposure"][-1] is None


def test_exposure_is_optional_and_never_inferred_from_coordinates() -> None:
    p = ObservationPanel(cells().drop("effort"), "EPSG:3857", key_cols=["unit"])
    output = ObservationPanelStep(p).run(events()).collect()
    assert "event_rate" not in output.columns and "log_exposure" not in output.columns
    assert output["event_count"].to_list() == [2, None, 0, None, 0, 0]


def test_exposure_requires_units() -> None:
    with pytest.raises(ValueError, match="explicit non-empty unit"):
        ObservationPanel(cells(), "EPSG:3857", key_cols=["unit"], exposure_col="effort")


def test_overlapping_intervals_rejected_but_distinct_strata_allowed() -> None:
    with pytest.raises(ValueError, match="overlap"):
        panel(cells().with_columns(pl.lit(T).alias("start")))
    data = cells().head(1)
    stratified = pl.concat(
        [
            data.with_columns(pl.lit("a").alias("category")),
            data.with_columns(pl.lit("b").alias("category")),
        ]
    )
    p = ObservationPanel(stratified, "EPSG:3857", key_cols=["unit", "category"])
    source = events().with_columns(pl.lit("b").alias("category")).filter(pl.col("event_id") < 3)
    result = ObservationPanelStep(p).run(source).collect()
    assert result["event_count"].to_list() == [0, 2]


@pytest.mark.parametrize(
    "change", ["null_key", "zero_interval", "negative_interval", "missing_column"]
)
def test_malformed_support_is_rejected(change: str) -> None:
    data = cells()
    if change == "null_key":
        data = data.with_columns(pl.lit(None, dtype=pl.String).alias("unit"))
    elif change == "zero_interval":
        data = data.with_columns(pl.col("start").alias("end"))
    elif change == "negative_interval":
        data = data.with_columns((pl.col("start") - pl.duration(hours=1)).alias("end"))
    else:
        data = data.drop("end")
    with pytest.raises(ValueError):
        panel(data)


def test_unknown_or_duplicate_record_keys_do_not_silently_disappear() -> None:
    for source in [
        events().with_columns(pl.lit(None, dtype=pl.String).alias("unit")),
        events(event_id=[1, 1, 3]),
    ]:
        with pytest.raises(ValueError):
            ObservationPanelStep(panel(), event_id_col="event_id").run(source)


def test_duplicates_count_as_records_unless_identifiers_are_declared() -> None:
    result = ObservationPanelStep(panel()).run(events(event_id=[1, 1, 3])).collect()
    assert result["event_count"][0] == 2


def test_empty_input_retains_zeros_only_for_observed_support() -> None:
    result = ObservationPanelStep(panel()).run(events().filter(pl.lit(False))).collect()
    assert result["event_count"].to_list() == [0, None, 0, None, 0, 0]
    assert result["recorded_count"].to_list() == [0] * 6


def test_empty_panel_preserves_typed_columns_and_rejects_unmatched_records() -> None:
    p = panel(cells().clear())
    source = events().filter(pl.lit(False))
    result = ObservationPanelStep(p).run(source)
    assert result.collect().is_empty() and result.collect()["event_count"].dtype == pl.Int64
    assert result.metadata.date_range is None
    with pytest.raises(ValueError, match="outside"):
        ObservationPanelStep(p).run(events())


def test_panel_snapshot_and_fingerprint_do_not_follow_external_mutation() -> None:
    data = cells()
    p = panel(data)
    fingerprint = p.fingerprint
    data.replace_column(data.get_column_index("effort"), pl.Series("effort", [100.0] * 6))
    assert p.cells["effort"][0] == 2.0 and p.fingerprint == fingerprint
    p.cells.replace_column(0, pl.Series("unit", ["mutated"] * 6))
    assert p.cells["unit"][0] == "A"
    assert panel().fingerprint == fingerprint


def test_geometry_support_does_not_reuse_event_point_coordinates() -> None:
    data = (
        cells()
        .with_columns(pl.Series("geometry", [box(i, 0, i + 1, 1).wkt for i in range(6)]))
        .drop("x", "y")
    )
    p = panel(data, x_col=None, y_col=None, geometry_col="geometry")
    result = ObservationPanelStep(p).run(events())
    assert result.schema.geometry_col == "geometry" and result.schema.lon_col is None
    assert "event_x" not in result.collect().columns


def test_panel_timestamp_precision_is_not_silently_truncated() -> None:
    source = events().with_columns(pl.col("time").dt.cast_time_unit("ns"))
    with pytest.raises(ValueError, match="same Datetime precision"):
        ObservationPanelStep(panel()).run(source)


def test_aware_zones_are_aligned_but_naive_aware_mix_is_rejected() -> None:
    aware_cells = cells().with_columns(pl.col("start", "end").dt.replace_time_zone("UTC"))
    p = panel(aware_cells)
    with pytest.raises(ValueError, match="naive and time-zone-aware"):
        ObservationPanelStep(p).run(events())
    source = events().with_columns(
        pl.col("time").dt.replace_time_zone("UTC").dt.convert_time_zone("Europe/Paris")
    )
    result = ObservationPanelStep(p).run(source)
    assert result.collect()["event_count"][0] == 2 and result.metadata.time_zone == "UTC"


def test_panel_is_registry_composable_and_async_parity_is_preserved() -> None:
    step = get_default_registry().create("observation_panel", params={"panel": panel()})
    pipeline = Pipeline(
        [step, TemporalLagStep(["event_count"], [1], time_unit="1h", group_cols=["unit"])]
    )
    expected = pipeline.run(events()).collect()

    async def run():
        return await (await pipeline.run_async(events())).collect_async()

    result = asyncio.run(run())
    assert result.equals(expected)
    assert result.filter(pl.col("unit") == "A")["event_count_lag_1"].to_list() == [None, 2]


def test_counting_already_aggregated_panels_is_rejected() -> None:
    result = ObservationPanelStep(panel()).run(events())
    with pytest.raises(ValueError, match="raw point-event records"):
        ObservationPanelStep(panel()).run(result)


def test_output_names_cannot_silently_overwrite_reference_columns() -> None:
    with pytest.raises(ValueError, match="output names"):
        ObservationPanelStep(panel(), count_col="effort").run(events())


def test_events_and_reference_need_explicitly_consistent_crs() -> None:
    with pytest.raises(ValueError, match="CRS must agree"):
        ObservationPanelStep(panel()).run(events().with_metadata(crs="EPSG:4326"))


def test_panel_configuration_is_immutable_and_part_of_its_fingerprint() -> None:
    p = panel()
    with pytest.raises(FrozenInstanceError):
        p.exposure_unit = "other"  # type: ignore[misc]
    changed = ObservationPanel(
        cells(), "EPSG:3857", key_cols=["unit"], exposure_col="effort", exposure_unit="person_years"
    )
    assert changed.fingerprint != p.fingerprint


def test_duplicate_intervals_are_not_counted_twice() -> None:
    with pytest.raises(ValueError, match="overlap"):
        panel(pl.concat([cells(), cells().head(1)]))


def test_internal_work_columns_cannot_clobber_user_keys() -> None:
    p = ObservationPanel(
        cells().rename({"unit": "_panel_row"}), "EPSG:3857", key_cols=["_panel_row"]
    )
    result = (
        ObservationPanelStep(p)
        .run(events().with_lazy_frame(events().lazy_frame.rename({"unit": "_panel_row"})))
        .collect()
    )
    assert result["event_count"].to_list() == [2, None, 0, None, 0, 0]


def test_unbinned_point_record_contract_rejects_a_time_resolution_label() -> None:
    with pytest.raises(ValueError, match="unbinned point-event"):
        ObservationPanelStep(panel()).run(events().with_metadata(time_bin="6h"))


def test_irregular_interval_lengths_are_preserved() -> None:
    data = cells().head(1)
    second = data.with_columns(
        pl.lit(T + timedelta(hours=1)).alias("start"), pl.lit(T + timedelta(hours=4)).alias("end")
    )
    result = ObservationPanelStep(panel(pl.concat([data, second]))).run(events()).collect()
    assert result["event_count"].to_list() == [2, 1]
    assert result["end"].to_list() == [T + timedelta(hours=1), T + timedelta(hours=4)]


def test_panel_reference_order_is_stable_even_with_unsorted_events() -> None:
    reversed_panel = panel(cells().reverse())
    result = (
        ObservationPanelStep(reversed_panel)
        .run(events().with_lazy_frame(events().lazy_frame.reverse()))
        .collect()
    )
    assert result["unit"].to_list() == cells().reverse()["unit"].to_list()
    assert result["event_count"].to_list() == [0, 0, None, 0, None, 2]


def test_numeric_keys_cannot_be_nonfinite() -> None:
    with pytest.raises(ValueError, match="numeric keys must be finite"):
        panel(cells().with_columns(pl.lit(float("nan")).alias("unit")))


def test_unknown_exposure_is_not_zero_and_known_zero_is_not_log_zero() -> None:
    output = ObservationPanelStep(panel()).run(events()).collect()
    assert output["effort"][-2] == 0.0 and output["effort"][-1] is None
    assert output["log_exposure"][-2] is None and output["log_exposure"][-1] is None


def test_decimal_exposure_retains_reference_type_and_normalizes_as_float() -> None:
    data = cells().head(1).with_columns(pl.Series("effort", [Decimal("2.0")]))
    output = (
        ObservationPanelStep(panel(data)).run(events().filter(pl.col("event_id") < 3)).collect()
    )
    assert output["effort"][0] == Decimal("2.0")
    assert output["event_rate"][0] == 1.0 and output["event_rate"].dtype == pl.Float64
    assert output["log_exposure"][0] == math.log(2.0)
