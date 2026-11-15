"""Tests for panel completion (CompletePanelStep)."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.panel import CompletePanelConfig, CompletePanelStep


def _bin(day: int) -> datetime:
    return datetime(2024, 1, day)


def make_panel_frame(
    rows: list[tuple[int, datetime, int, str]],
    *,
    time_bin: str | None = "1d",
    extra_cols: dict[str, list] | None = None,
) -> EventFrame:
    """Build a small aggregated panel: one row per observed (unit, bin).

    Each row is (grid_id, time_bin, count, zone); ``zone`` is a static
    per-unit attribute carried through completion.
    """
    data: dict[str, list] = {
        "grid_id": [r[0] for r in rows],
        "time_bin": [r[1] for r in rows],
        "count": [r[2] for r in rows],
        "zone": [r[3] for r in rows],
        "latitude": [41.8781] * len(rows),
        "longitude": [-87.6298] * len(rows),
    }
    if extra_cols:
        data.update(extra_cols)
    lf = pl.LazyFrame(data)
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["count"],
        categorical_cols=["zone"],
    )
    metadata = EventMetadata(dataset_name="test-panel", crs="EPSG:4326", time_bin=time_bin)
    return EventFrame(lf, schema, metadata)


# Observed rows shared by the reference and min_periods tests:
# unit 1 in bins 1 and 3, unit 2 in bin 2 only, unit 3 in all four bins.
OBSERVED_ROWS = [
    (1, _bin(1), 5, "north"),
    (1, _bin(3), 7, "north"),
    (2, _bin(2), 3, "south"),
    (3, _bin(1), 1, "east"),
    (3, _bin(2), 2, "east"),
    (3, _bin(3), 3, "east"),
    (3, _bin(4), 4, "east"),
]


def _as_tuples(df: pl.DataFrame) -> set[tuple]:
    return set(df.select(["grid_id", "time_bin", "count"]).iter_rows())


class TestReferenceOutput:
    """Hand-computed expected output for a 3-unit x 4-bin panel."""

    def test_completes_missing_pairs_with_explicit_zeros(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        result = CompletePanelStep().run(frame).collect()

        assert result.height == 12  # 3 units x 4 bins
        assert _as_tuples(result) == {
            (1, _bin(1), 5),
            (1, _bin(2), 0),  # generated
            (1, _bin(3), 7),
            (1, _bin(4), 0),  # generated
            (2, _bin(1), 0),  # generated
            (2, _bin(2), 3),
            (2, _bin(3), 0),  # generated
            (2, _bin(4), 0),  # generated
            (3, _bin(1), 1),
            (3, _bin(2), 2),
            (3, _bin(3), 3),
            (3, _bin(4), 4),
        }

    def test_static_columns_carried_per_unit(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        result = CompletePanelStep().run(frame).collect()

        zones = result.group_by("grid_id").agg(pl.col("zone").n_unique().alias("n_zones"))
        assert zones["n_zones"].to_list() == [1, 1, 1]
        assert result.filter(pl.col("grid_id") == 1)["zone"].unique().to_list() == ["north"]
        assert result.filter(pl.col("grid_id") == 2)["zone"].unique().to_list() == ["south"]
        assert result.filter(pl.col("grid_id") == 3)["zone"].unique().to_list() == ["east"]

    def test_value_prefixes_zero_fill_matching_columns(self) -> None:
        # Per-category count columns (count_primary_type_THEFT, ...) are
        # zero-filled via prefix, while other columns are not.
        frame = make_panel_frame(
            OBSERVED_ROWS,
            extra_cols={
                "count_primary_type_THEFT": [2, 0, 1, 0, 1, 0, 2],
                "share_primary_type_THEFT": [0.4, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
            },
        )
        result = CompletePanelStep(value_prefixes=["count_"]).run(frame).collect()
        generated = result.filter((pl.col("grid_id") == 1) & (pl.col("time_bin") == _bin(2)))
        assert generated["count"].item() == 0
        assert generated["count_primary_type_THEFT"].item() == 0
        assert generated["share_primary_type_THEFT"].item() is None
        observed = result.filter((pl.col("grid_id") == 1) & (pl.col("time_bin") == _bin(1)))
        assert observed["count_primary_type_THEFT"].item() == 2
        assert observed["share_primary_type_THEFT"].item() == pytest.approx(0.4)

    def test_status_column_is_observed_everywhere(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        result = CompletePanelStep().run(frame).collect()

        assert "observation_status" in result.columns
        assert result["observation_status"].unique().to_list() == ["observed"]

    def test_output_sorted_by_unit_then_time(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        result = CompletePanelStep().run(frame).collect()

        assert result.select(["grid_id", "time_bin"]).equals(
            result.select(["grid_id", "time_bin"]).sort(["grid_id", "time_bin"])
        )

    def test_value_dtype_preserved_for_generated_zeros(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        result = CompletePanelStep().run(frame).collect()

        assert result.schema["count"] == pl.Int64
        generated = result.filter((pl.col("grid_id") == 1) & (pl.col("time_bin") == _bin(2)))
        assert generated["count"].to_list() == [0]


class TestMinPeriods:
    """A unit enters the panel only if observed in enough distinct bins."""

    def test_sparse_unit_dropped_with_min_periods_2(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        result = CompletePanelStep(min_periods=2).run(frame).collect()

        assert result.height == 8  # units 1 and 3 x 4 bins
        assert result["grid_id"].unique().sort().to_list() == [1, 3]

    def test_sparse_unit_kept_with_min_periods_1(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        result = CompletePanelStep(min_periods=1).run(frame).collect()

        assert result.height == 12
        assert result["grid_id"].unique().sort().to_list() == [1, 2, 3]

    def test_no_unit_meets_min_periods_raises(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        with pytest.raises(ValueError, match="at least 5 distinct"):
            CompletePanelStep(min_periods=5).run(frame)


class TestDefaultsAndConfig:
    """time_col/bin_size resolve from the frame; config round-trips."""

    def test_defaults_come_from_schema_and_metadata(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        result = CompletePanelStep().run(frame).collect()
        assert result.height == 12

    def test_explicit_time_col_and_bin_size(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS, time_bin=None)
        result = CompletePanelStep(time_col="time_bin", bin_size="1d").run(frame).collect()
        assert result.height == 12

    def test_config_model_builds_step_like_registry(self) -> None:
        config = CompletePanelConfig(
            unit_col="grid_id", value_cols=["count"], min_periods=2, bin_size="1d"
        )
        step = CompletePanelStep(**config.model_dump())
        frame = make_panel_frame(OBSERVED_ROWS)
        result = step.run(frame).collect()
        assert result.height == 8

    def test_config_defaults(self) -> None:
        config = CompletePanelConfig()
        assert config.unit_col == "grid_id"
        assert config.time_col is None
        assert config.bin_size is None
        assert config.value_cols == ["count"]
        assert config.min_periods == 1
        assert config.status_col == "observation_status"

    def test_provenance_registered(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        result = CompletePanelStep(min_periods=2).run(frame)

        provenance = result.metadata.feature_provenance["observation_status"]
        assert provenance.produced_by == "CompletePanelStep"
        assert provenance.tags == {"temporal", "spatial"}
        assert provenance.metadata["min_periods"] == 2
        assert provenance.metadata["bin_size"] == "1d"
        assert provenance.metadata["n_units"] == 2
        assert provenance.metadata["n_bins"] == 4


class TestErrors:
    """Every failure mode raises ValueError with an explicit message."""

    def test_duplicate_unit_time_raises(self) -> None:
        rows = OBSERVED_ROWS + [(1, _bin(1), 9, "north")]
        frame = make_panel_frame(rows)
        with pytest.raises(ValueError, match="Duplicate \\(unit, time\\)"):
            CompletePanelStep().run(frame)

    def test_empty_frame_raises(self) -> None:
        frame = make_panel_frame([])
        with pytest.raises(ValueError, match="empty frame"):
            CompletePanelStep().run(frame)

    def test_min_periods_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_periods"):
            CompletePanelStep(min_periods=0)

    def test_unresolvable_bin_size_raises(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS, time_bin=None)
        with pytest.raises(ValueError, match="bin_size"):
            CompletePanelStep().run(frame)

    def test_unparseable_bin_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported elapsed duration"):
            CompletePanelStep(bin_size="1month")

    def test_missing_value_col_raises(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        with pytest.raises(ValueError, match="missing from the frame"):
            CompletePanelStep(value_cols=["nope"]).run(frame)

    def test_non_numeric_value_col_raises(self) -> None:
        frame = make_panel_frame(OBSERVED_ROWS)
        with pytest.raises(ValueError, match="must be numeric"):
            CompletePanelStep(value_cols=["zone"]).run(frame)

    def test_null_unit_raises(self) -> None:
        lf = pl.LazyFrame(
            {
                "grid_id": [1, None],
                "time_bin": [_bin(1), _bin(2)],
                "count": [5, 3],
                "zone": ["north", "south"],
                "latitude": [41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298],
            }
        )
        frame = EventFrame(
            lf,
            EventSchema(timestamp_col="time_bin", lat_col="latitude", lon_col="longitude"),
            EventMetadata(dataset_name="test-panel", crs="EPSG:4326", time_bin="1d"),
        )
        with pytest.raises(ValueError, match="cannot contain nulls"):
            CompletePanelStep().run(frame)

    def test_null_time_raises(self) -> None:
        lf = pl.LazyFrame(
            {
                "grid_id": [1, 1],
                "time_bin": [_bin(1), None],
                "count": [5, 3],
                "zone": ["north", "north"],
                "latitude": [41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298],
            }
        )
        frame = EventFrame(
            lf,
            EventSchema(timestamp_col="time_bin", lat_col="latitude", lon_col="longitude"),
            EventMetadata(dataset_name="test-panel", crs="EPSG:4326", time_bin="1d"),
        )
        with pytest.raises(ValueError, match="cannot contain nulls"):
            CompletePanelStep().run(frame)

    def test_misaligned_bins_raise(self) -> None:
        # 2024-01-01T06:00 is not on the 1d grid anchored at 2024-01-01T00:00.
        rows = [
            (1, datetime(2024, 1, 1, 0, 0), 5, "north"),
            (1, datetime(2024, 1, 1, 6, 0), 7, "north"),
        ]
        frame = make_panel_frame(rows)
        with pytest.raises(ValueError, match="not aligned"):
            CompletePanelStep().run(frame)

    def test_varying_column_is_joined_not_collapsed(self) -> None:
        # hour_of_day is derived from the time bin: it varies within a unit
        # and must be joined on (unit, time) — observed pairs keep their
        # value, generated pairs get null — never collapsed with first().
        lf = pl.LazyFrame(
            {
                "grid_id": [r[0] for r in OBSERVED_ROWS],
                "time_bin": [r[1] for r in OBSERVED_ROWS],
                "count": [r[2] for r in OBSERVED_ROWS],
                "zone": [r[3] for r in OBSERVED_ROWS],
                "hour_of_day": [0, 6, 12, 0, 6, 12, 18],
                "latitude": [41.8781] * len(OBSERVED_ROWS),
                "longitude": [-87.6298] * len(OBSERVED_ROWS),
            }
        )
        frame = EventFrame(
            lf,
            EventSchema(timestamp_col="time_bin", lat_col="latitude", lon_col="longitude"),
            EventMetadata(dataset_name="test-panel", crs="EPSG:4326", time_bin="1d"),
        )
        result = CompletePanelStep().run(frame).collect()
        observed = result.filter((pl.col("grid_id") == 1) & (pl.col("time_bin") == _bin(3)))
        assert observed["hour_of_day"].item() == 6
        generated = result.filter((pl.col("grid_id") == 1) & (pl.col("time_bin") == _bin(2)))
        assert generated["hour_of_day"].item() is None
        assert generated["count"].item() == 0
        # Static columns are still carried to generated pairs.
        assert generated["zone"].item() == "north"

    def test_existing_status_col_raises(self) -> None:
        frame = make_panel_frame(
            OBSERVED_ROWS, extra_cols={"observation_status": ["observed"] * len(OBSERVED_ROWS)}
        )
        with pytest.raises(ValueError, match="already exists"):
            CompletePanelStep().run(frame)


class TestNoLeakage:
    """Completion is per-bin: later data never rewrites earlier bins."""

    def test_future_events_do_not_change_overlapping_bins(self) -> None:
        step = CompletePanelStep()
        base = step.run(make_panel_frame(OBSERVED_ROWS)).collect()
        base_overlap = base.filter(pl.col("time_bin") <= _bin(4))

        extended_rows = OBSERVED_ROWS + [
            (1, _bin(5), 9, "north"),
            (1, _bin(6), 2, "north"),
            (3, _bin(5), 8, "east"),
        ]
        extended = step.run(make_panel_frame(extended_rows)).collect()
        extended_overlap = extended.filter(pl.col("time_bin") <= _bin(4))

        # The time axis extends (6 bins now) — that is correct behaviour,
        # not leakage — so compare only the overlapping bins.
        assert extended.height == 18  # 3 units x 6 bins
        assert extended_overlap.equals(base_overlap)

    def test_event_at_bin_t_does_not_change_bins_before_t(self) -> None:
        step = CompletePanelStep()
        base = step.run(make_panel_frame(OBSERVED_ROWS)).collect()
        base_before = base.filter((pl.col("grid_id") == 2) & (pl.col("time_bin") < _bin(3)))

        rows_with_later_event = OBSERVED_ROWS + [(2, _bin(3), 4, "south")]
        later = step.run(make_panel_frame(rows_with_later_event)).collect()
        later_before = later.filter((pl.col("grid_id") == 2) & (pl.col("time_bin") < _bin(3)))

        assert later_before.equals(base_before)
        assert later.filter((pl.col("grid_id") == 2) & (pl.col("time_bin") == _bin(3)))[
            "count"
        ].to_list() == [4]
