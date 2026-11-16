"""Independent quality declarations and explicit scheduled-slot diagnostics."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from shikong.core import CadenceRule, EventFrame, EventMetadata, EventSchema, ObservationPanel
from shikong.core.pipeline import Pipeline
from shikong.core.steps import get_default_registry

T = datetime(2024, 1, 1)


def support(**columns) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unit": ["A"],
            "start": [T],
            "end": [T + timedelta(minutes=40)],
            "x": [0.0],
            "y": [0.0],
            **columns,
        }
    )


def panel(data: pl.DataFrame | None = None, **kwargs) -> ObservationPanel:
    return ObservationPanel(
        support() if data is None else data, "EPSG:3857", key_cols=["unit"], **kwargs
    )


def samples(minutes=(0, 10, 10, 30)) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "unit": pl.Series(["A"] * len(minutes), dtype=pl.String),
            "t": pl.Series([T + timedelta(minutes=m) for m in minutes], dtype=pl.Datetime("us")),
        }
    ).lazy()


def event_frame(data: pl.LazyFrame | None = None) -> EventFrame:
    data = samples() if data is None else data
    return EventFrame(
        data.with_columns(pl.lit(0.0).alias("x"), pl.lit(0.0).alias("y")),
        EventSchema(timestamp_col="t", lon_col="x", lat_col="y"),
        EventMetadata(dataset_name="scheduled-sensors", crs="EPSG:3857"),
    )


@pytest.mark.parametrize("collection", ["active", "inactive", "unknown", None])
@pytest.mark.parametrize("completeness", ["complete", "incomplete", "unknown", None])
def test_only_independent_active_and_complete_declarations_allow_zero(
    collection, completeness
) -> None:
    p = panel(
        support(collect=[collection], batch=[completeness]),
        collection_col="collect",
        completeness_col="batch",
    )
    output, _ = p.count_events(samples(()), "t")
    assert output["event_count"][0] == (
        0 if collection == "active" and completeness == "complete" else None
    )
    assert output["collect"][0] == collection and output["batch"][0] == completeness
    assert p.metadata()["quality_mode"] == "independent"


def test_no_quality_information_preserves_unknown_without_implicit_status() -> None:
    p = panel()
    output, _ = p.count_events(samples(), "t")
    assert output["recorded_count"][0] == 4 and output["event_count"][0] is None
    assert "observation_status" not in output.columns
    assert p.metadata()["unspecified_quality"] == "unknown"


@pytest.mark.parametrize("membership", [False, None])
def test_domain_membership_is_independent_of_source_activity(membership) -> None:
    p = panel(
        support(
            domain=pl.Series([membership], dtype=pl.Boolean), collect=["active"], batch=["complete"]
        ),
        domain_col="domain",
        collection_col="collect",
        completeness_col="batch",
    )
    output, _ = p.count_events(samples(()), "t")
    assert output["event_count"][0] is None
    report, _ = p.check_cadence(samples(()), "t", CadenceRule(every="10m"))
    assert report["cadence_status"][0] == ("not_applicable" if membership is False else "unknown")
    assert report["cadence_expected"][0] is None


def test_complete_batch_does_not_infer_active_collection() -> None:
    p = panel(support(batch=["complete"]), completeness_col="batch")
    output, _ = p.count_events(samples(), "t")
    assert output["event_count"][0] is None
    assert p.metadata()["collection_col"] is None


def test_legacy_status_is_accepted_and_marked_as_legacy() -> None:
    p = panel(support(observation_status=["observed"]))
    assert p.count_events(samples(()), "t")[0]["event_count"][0] == 0
    assert p.metadata()["quality_mode"] == "legacy"
    assert p.metadata()["collection_col"] is None  # Never synthesized from 'observed'.


def test_mixing_legacy_and_independent_quality_is_rejected() -> None:
    with pytest.raises(ValueError, match="Do not mix legacy"):
        panel(
            support(observation_status=["observed"], collect=["active"]), collection_col="collect"
        )


@pytest.mark.parametrize(
    "column,value", [("collect", "complete"), ("batch", "active"), ("domain", "yes")]
)
def test_independent_declarations_have_distinct_vocabularies(column, value) -> None:
    config = {"collect": "collection_col", "batch": "completeness_col", "domain": "domain_col"}
    with pytest.raises(ValueError):
        panel(support(**{column: [value]}), **{config[column]: column})


def test_duplicates_cannot_hide_missing_scheduled_slots() -> None:
    report, diagnostics = panel().check_cadence(
        samples(), "t", CadenceRule(every="10m"), include_missing_times=True
    )
    assert report.select(
        "cadence_expected",
        "cadence_present",
        "cadence_missing",
        "cadence_duplicates",
        "cadence_off_schedule",
    ).row(0) == (4, 3, 1, 1, 0)
    assert report["cadence_status"][0] == "gaps"
    assert report["cadence_missing_times"][0].to_list() == [T + timedelta(minutes=20)]
    assert diagnostics["input_records"] == 4


def test_without_rule_cadence_is_unknown_even_when_records_are_regular() -> None:
    report, _ = panel().check_cadence(samples((0, 10, 20, 30)), "t", include_missing_times=True)
    assert report["cadence_status"][0] == "unknown"
    assert all(
        report[c][0] is None
        for c in [
            "cadence_expected",
            "cadence_missing",
            "cadence_duplicates",
            "cadence_missing_times",
        ]
    )


def test_a_satisfied_schedule_never_certifies_quality_or_creates_counts() -> None:
    p = panel(
        support(collect=["unknown"], batch=["unknown"]),
        collection_col="collect",
        completeness_col="batch",
    )
    output, _ = p.count_events(samples((0, 10, 20, 30)), "t", cadence_rule=CadenceRule(every="10m"))
    assert output["cadence_status"][0] == "satisfied"
    assert output["event_count"][0] is None and output["batch"][0] == "unknown"
    assert output.height == 1 and output["recorded_count"][0] == 4


def test_empty_input_reports_all_expected_gaps_without_filling_values() -> None:
    report, _ = panel().check_cadence(samples(()), "t", CadenceRule(every="10m"))
    assert report["cadence_present"][0] == 0 and report["cadence_missing"][0] == 4
    assert "event_count" not in report.columns


def test_tolerance_reports_records_outside_the_scheduled_neighborhood() -> None:
    report, _ = panel().check_cadence(
        samples((0.5, 10.5, 21.1)), "t", CadenceRule(every="10m", tolerance="1m")
    )
    assert report["cadence_present"][0] == 2 and report["cadence_missing"][0] == 2
    assert report["cadence_off_schedule"][0] == 1


@pytest.mark.parametrize("tolerance", ["5m", "6m"])
def test_ambiguous_tolerance_is_rejected(tolerance: str) -> None:
    with pytest.raises(ValueError, match="less than half"):
        CadenceRule(every="10m", tolerance=tolerance)


def test_global_origin_controls_phase_instead_of_first_received_record() -> None:
    p = panel(support(start=[T + timedelta(minutes=12)], end=[T + timedelta(minutes=37)]))
    report, _ = p.check_cadence(
        samples((15, 35)),
        "t",
        CadenceRule(every="10m", origin=T + timedelta(minutes=5)),
        include_missing_times=True,
    )
    assert report["cadence_expected"][0] == 3 and report["cadence_missing"][0] == 1
    assert report["cadence_missing_times"][0].to_list() == [T + timedelta(minutes=25)]
    default, _ = p.check_cadence(samples((15, 35)), "t", CadenceRule(every="10m"))
    assert default["cadence_present"][0] == 0 and default["cadence_off_schedule"][0] == 2


def test_half_open_windows_do_not_borrow_a_neighboring_interval_slot() -> None:
    data = pl.concat(
        [
            support(end=[T + timedelta(minutes=10)]),
            support(start=[T + timedelta(minutes=10)], end=[T + timedelta(minutes=20)]),
        ]
    )
    report, _ = panel(data).check_cadence(
        samples((9.9, 10)), "t", CadenceRule(every="10m", tolerance="1m")
    )
    assert report["cadence_present"].to_list() == [0, 1]
    assert report["cadence_off_schedule"].to_list() == [1, 0]


def test_budget_is_total_across_keys_and_does_not_allocate_an_unbounded_grid() -> None:
    data = pl.concat([support(), support(unit=["B"])])
    with pytest.raises(ValueError, match="exceed max_slots"):
        panel(data).check_cadence(samples(()), "t", CadenceRule(every="10m", max_slots=6))


@pytest.mark.parametrize("on_gap", ["report", "raise"])
def test_gap_action_is_explicit_and_never_rewrites_declarations(on_gap: str) -> None:
    p = panel(
        support(collect=["active"], batch=["complete"]),
        collection_col="collect",
        completeness_col="batch",
    )
    if on_gap == "raise":
        with pytest.raises(ValueError, match="missing scheduled slots"):
            p.count_events(samples(), "t", cadence_rule=CadenceRule(every="10m"), on_gap="raise")
    else:
        output, _ = p.count_events(samples(), "t", cadence_rule=CadenceRule(every="10m"))
        assert output["batch"][0] == "complete" and output["collect"][0] == "active"
        assert output["cadence_status"][0] == "gaps"


def test_rule_and_origin_must_be_representable_in_timestamp_precision() -> None:
    p = panel(support().with_columns(pl.col("start", "end").dt.cast_time_unit("ms")))
    raw = samples(()).with_columns(pl.col("t").dt.cast_time_unit("ms"))
    with pytest.raises(ValueError, match="not representable"):
        p.check_cadence(raw, "t", CadenceRule(every="1us"))
    with pytest.raises(ValueError, match="origin is not representable"):
        p.check_cadence(raw, "t", CadenceRule(every="10m", origin=T + timedelta(microseconds=1)))


def test_nanosecond_slot_identity_is_not_rounded_through_python_datetimes() -> None:
    data = support().with_columns(
        pl.Series("start", [0]).cast(pl.Datetime("ns")),
        pl.Series("end", [4004]).cast(pl.Datetime("ns")),
    )
    raw = pl.DataFrame(
        {"unit": ["A"] * 3, "t": pl.Series([0, 1001, 3003]).cast(pl.Datetime("ns"))}
    ).lazy()
    report, _ = panel(data).check_cadence(
        raw, "t", CadenceRule(every="1.001us"), include_missing_times=True
    )
    assert report["cadence_present"][0] == 3
    assert report["cadence_missing_times"].explode().cast(pl.Int64).to_list() == [2002]


def test_timezone_aware_schedule_uses_elapsed_time_across_dst() -> None:
    start = datetime(2024, 3, 10, 6, tzinfo=timezone.utc)
    data = support(start=[start], end=[start + timedelta(hours=4)]).with_columns(
        pl.col("start", "end").dt.convert_time_zone("America/Toronto")
    )
    raw = pl.DataFrame(
        {"unit": ["A"] * 3, "t": [start + timedelta(hours=h) for h in (0, 1, 3)]}
    ).lazy()
    report, _ = panel(data).check_cadence(
        raw, "t", CadenceRule(every="1h", origin=start), include_missing_times=True
    )
    assert report["cadence_expected"][0] == 4 and report["cadence_missing"][0] == 1
    missing = report["cadence_missing_times"].explode().dt.convert_time_zone("UTC")
    assert missing.to_list() == [start + timedelta(hours=2)]
    with pytest.raises(ValueError, match="naive/aware"):
        panel(data).check_cadence(raw, "t", CadenceRule(every="1h", origin=T))


def test_outside_records_are_audited_by_the_independent_check() -> None:
    raw = samples().with_columns(pl.lit("Z").alias("unit"))
    with pytest.raises(ValueError, match="outside the declared"):
        panel().check_cadence(raw, "t", CadenceRule(every="10m"))
    report, diagnostics = panel().check_cadence(
        raw, "t", CadenceRule(every="10m"), events_outside="ignore"
    )
    assert diagnostics["excluded_records"] == 4 and report["cadence_missing"][0] == 4


def test_empty_panel_has_typed_reports() -> None:
    report, _ = panel(support().clear()).check_cadence(
        samples(()), "t", CadenceRule(every="10m"), include_missing_times=True
    )
    assert report.is_empty() and report["cadence_expected"].dtype == pl.Int64
    assert report["cadence_missing_times"].dtype == pl.List(pl.Datetime("us"))


def test_cadence_diagnostics_integrate_with_registry_and_async_execution() -> None:
    step = get_default_registry().create(
        "observation_panel",
        params={"panel": panel(), "cadence_rule": {"every": "10m"}, "include_missing_times": True},
    )
    pipeline = Pipeline([step])
    expected = pipeline.run(event_frame()).collect()

    async def scenario():
        return await (await pipeline.run_async(event_frame())).collect_async()

    assert asyncio.run(scenario()).equals(expected)
    meta = pipeline.last_metadata.custom["observation_panel"]
    assert meta["cadence_rule"]["every"] == "10m"
    assert meta["cadence_does_not_certify_collection_or_completeness"] is True


def test_cadence_prefix_collisions_and_unsupported_rules_are_rejected() -> None:
    with pytest.raises(ValueError, match="collide"):
        panel(support(cadence_expected=[0])).check_cadence(samples(), "t", CadenceRule(every="10m"))
    for value in ("0h", "1mo", "nonsense"):
        with pytest.raises(ValueError):
            CadenceRule(every=value)


@pytest.mark.parametrize("origin", [0, 1_700_000_000, True, "2024-01-01T00:00:00.000000001Z"])
def test_origin_never_infers_epoch_units_or_silently_truncates_submicroseconds(origin) -> None:
    with pytest.raises(ValueError):
        CadenceRule(every="10m", origin=origin)


@pytest.mark.parametrize("origin_offset", [None, -3, 3])
@pytest.mark.parametrize("period,tolerance", [(2, 0), (5, 30)])
def test_cadence_matches_brute_force_reference_with_strata_gaps_and_negative_epoch(
    origin_offset,
    period: int,
    tolerance: int,
) -> None:
    base = datetime(1969, 12, 31, 23)
    rows = [
        {
            "unit": unit,
            "start": base + timedelta(minutes=start),
            "end": base + timedelta(minutes=start + 17),
            "x": 0.0,
            "y": 0.0,
            "domain": {"A": True, "B": True, "C": False, "D": None}[unit],
        }
        for unit in ["B", "A", "C", "D"]
        for start in [25, 0]
    ]
    rng = random.Random(12)
    raw_rows = [
        {"unit": unit, "t": base + timedelta(seconds=s)}
        for unit in ["A", "B", "C", "D", "Z"]
        for s in [0, 120, 300, 600, 600, 1500, 1800, *[rng.randrange(-60, 2600) for _ in range(20)]]
    ]
    p = panel(pl.DataFrame(rows), domain_col="domain")
    origin = None if origin_offset is None else base + timedelta(minutes=origin_offset)
    report, diagnostics = p.check_cadence(
        pl.DataFrame(raw_rows).lazy(),
        "t",
        CadenceRule(every=f"{period}m", tolerance=f"{tolerance}s", origin=origin),
        events_outside="ignore",
    )
    assert report["unit"].to_list() == [row["unit"] for row in rows]
    excluded = 0
    for event in raw_rows:
        if not any(
            row["unit"] == event["unit"]
            and row["domain"] is not False
            and row["start"] <= event["t"] < row["end"]
            for row in rows
        ):
            excluded += 1
    assert diagnostics["excluded_records"] == excluded
    for i, row in enumerate(rows):
        if row["domain"] is not True:
            assert report["cadence_expected"][i] is None
            continue
        anchor = row["start"] if origin is None else origin
        expected = [
            anchor + timedelta(minutes=k * period)
            for k in range(-100, 101)
            if row["start"] <= anchor + timedelta(minutes=k * period) < row["end"]
        ]
        times = [
            event["t"]
            for event in raw_rows
            if event["unit"] == row["unit"] and row["start"] <= event["t"] < row["end"]
        ]
        matches = [
            sum(abs((time - slot).total_seconds()) <= tolerance for time in times)
            for slot in expected
        ]
        present = sum(count > 0 for count in matches)
        assert report["cadence_expected"][i] == len(expected)
        assert report["cadence_present"][i] == present
        assert report["cadence_missing"][i] == len(expected) - present
        assert report["cadence_duplicates"][i] == sum(matches) - present
        assert report["cadence_off_schedule"][i] == len(times) - sum(matches)
