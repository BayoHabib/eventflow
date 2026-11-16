"""Tests for the near-repeat step (Phase 2, Decision 4)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import polars as pl
import pytest
from pydantic import ValidationError

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.near_repeat import NearRepeatConfig, NearRepeatStep

START = datetime(2024, 1, 1)


def make_panel(
    events: dict[tuple[int, int, int], int],
    n_rows: int = 3,
    n_cols: int = 3,
    n_days: int = 20,
    time_bin: str = "1d",
    value_col: str = "count",
) -> EventFrame:
    """Completed panel: every cell x bin present, explicit zeros.

    ``events`` maps (grid_row, grid_col, day_index) -> count; every
    other (cell, bin) has count 0. Bins are daily starting 2024-01-01
    (or 6-hourly when ``time_bin == "6h"``, with day_index counting
    6 h bins instead).
    """
    step = timedelta(days=1) if time_bin == "1d" else timedelta(hours=6)
    rows: list[dict[str, object]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            for d in range(n_days):
                rows.append(
                    {
                        "grid_id": r * n_cols + c,
                        "grid_row": r,
                        "grid_col": c,
                        "time_bin": START + d * step,
                        value_col: events.get((r, c, d), 0),
                        "latitude": 41.8781,
                        "longitude": -87.6298,
                    }
                )
    lf = pl.LazyFrame(rows)
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=[value_col],
        categorical_cols=["grid_id"],
    )
    metadata = EventMetadata(dataset_name="test-near-repeat", crs="EPSG:4326", time_bin=time_bin)
    return EventFrame(lf, schema, metadata)


def expected_nr(
    events: dict[tuple[int, int, int], int],
    n_rows: int,
    n_cols: int,
    n_days: int,
    rings: list[int],
    window_bins: int = 7,
) -> dict[tuple[int, int, int, str], int]:
    """Independent brute-force near-repeat counts.

    Key: (grid_id, day_index, ring, window) with window "7d" or
    "7_14d". A neighbour at Chebyshev distance exactly ``ring`` from
    the cell contributes its count at bin ``j`` to the 7 d window when
    ``t - window_bins <= j < t`` and to the 7-14 d window when
    ``t - 2 * window_bins <= j < t - window_bins``.
    """
    out: dict[tuple[int, int, int, str], int] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            cell = r * n_cols + c
            for ring in rings:
                neighbours = [
                    (rr, cc)
                    for rr in range(n_rows)
                    for cc in range(n_cols)
                    if max(abs(rr - r), abs(cc - c)) == ring
                ]
                for t in range(n_days):
                    total_7 = sum(
                        events.get((rr, cc, j), 0)
                        for rr, cc in neighbours
                        for j in range(max(0, t - window_bins), t)
                    )
                    total_7_14 = sum(
                        events.get((rr, cc, j), 0)
                        for rr, cc in neighbours
                        for j in range(max(0, t - 2 * window_bins), max(0, t - window_bins))
                    )
                    out[(cell, t, ring, "7d")] = total_7
                    out[(cell, t, ring, "7_14d")] = total_7_14
    return out


def cell_day_values(df: pl.DataFrame, grid_id: int, col: str) -> list[int]:
    """Values of *col* for one cell, in time-bin order."""
    return (
        df.filter(pl.col("grid_id") == grid_id).sort("time_bin")[col].to_list()  # type: ignore[return-value]
    )


def with_changed_count(frame: EventFrame, grid_id: int, day: int, count: int) -> EventFrame:
    """Copy of the panel with one (cell, bin) count replaced."""
    df = frame.collect()
    target = START + timedelta(days=day)
    df = df.with_columns(
        pl.when((pl.col("grid_id") == grid_id) & (pl.col("time_bin") == target))
        .then(count)
        .otherwise(pl.col("count"))
        .alias("count")
    )
    return EventFrame(df.lazy(), frame.schema, frame.metadata)


# Hand-placed events on a 5x5 grid, 20 daily bins (used by the
# brute-force reference test below).
EVENTS_5X5: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 3,
    (1, 1, 2): 5,
    (4, 4, 5): 2,
    (2, 2, 9): 4,
    (0, 4, 12): 1,
    (3, 1, 15): 6,
    (4, 0, 3): 2,
    (1, 3, 7): 3,
    (2, 4, 18): 7,
    (3, 3, 1): 1,
}


class TestNearRepeatReference:
    """Reference tests vs hand-computed / brute-force values."""

    def test_all_rings_match_brute_force_5x5(self) -> None:
        """Every ring/window/bin on a 5x5 panel matches the brute force."""
        frame = make_panel(EVENTS_5X5, n_rows=5, n_cols=5, n_days=20)
        result = NearRepeatStep().run(frame)
        df = result.collect()

        expected = expected_nr(EVENTS_5X5, 5, 5, 20, [1, 2, 3])
        assert result.collect().height == 5 * 5 * 20  # row count preserved
        for cell in range(25):
            for ring in (1, 2, 3):
                assert cell_day_values(df, cell, f"count_nr_r{ring}_7d") == [
                    expected[(cell, t, ring, "7d")] for t in range(20)
                ]
                assert cell_day_values(df, cell, f"count_nr_r{ring}_7_14d") == [
                    expected[(cell, t, ring, "7_14d")] for t in range(20)
                ]
            # The flag is 1 exactly when the ring-1 7 d count is > 0.
            assert cell_day_values(df, cell, "count_nr_event_7d") == [
                1 if expected[(cell, t, 1, "7d")] > 0 else 0 for t in range(20)
            ]

    def test_single_event_hand_computed_3x3(self) -> None:
        """One event (5 at the top-middle cell, day 3) checked by hand.

        Centre (1,1) and corner (0,0) have that cell in ring 1;
        corner (2,2) has it in ring 2 only; the event is 5 days before
        bin day 8 and 10 days before bin day 13.
        """
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20)
        df = NearRepeatStep().run(frame).collect()

        centre = 1 * 3 + 1
        # Ring 1, 7 d: 5 while the event is within the previous 7 bins
        # (bins day 4..day 10), 0 from day 11 on.
        assert cell_day_values(df, centre, "count_nr_r1_7d") == ([0, 0, 0] + [5] * 7 + [0] * 10)
        # Ring 1, 7-14 d: 5 while the event is 8..14 bins back
        # (bins day 11..day 17), 0 before and after.
        assert cell_day_values(df, centre, "count_nr_r1_7_14d") == ([0] * 10 + [5] * 7 + [0] * 3)
        # Flag: 1 exactly on the bins where the ring-1 7 d count is 5.
        assert cell_day_values(df, centre, "count_nr_event_7d") == ([0, 0, 0] + [1] * 7 + [0] * 10)
        # Corner (0,0) also has the event cell in ring 1: same series.
        assert cell_day_values(df, 0, "count_nr_r1_7d") == cell_day_values(
            df, centre, "count_nr_r1_7d"
        )
        # Corner (2,2) has the event cell in ring 2 only: ring 1 is
        # all zero (and so is the flag), ring 2 mirrors the centre's
        # ring-1 series.
        assert cell_day_values(df, 8, "count_nr_r1_7d") == [0] * 20
        assert cell_day_values(df, 8, "count_nr_event_7d") == [0] * 20
        assert cell_day_values(df, 8, "count_nr_r2_7d") == cell_day_values(
            df, centre, "count_nr_r1_7d"
        )
        # No ring-3 neighbours exist on a 3x3 grid: all zero.
        assert cell_day_values(df, centre, "count_nr_r3_7d") == [0] * 20
        assert cell_day_values(df, centre, "count_nr_r3_7_14d") == [0] * 20

    def test_ring3_counts_on_5x5(self) -> None:
        """Ring 3 is non-empty on a 5x5 grid: hand-checked corner cell.

        Corner (0,0)'s ring 3 is (0,3), (1,3), (2,3), (3,0), (3,1),
        (3,2), (3,3). Put 4 events at (3,3) on day 2 and 2 at (0,3)
        on day 5: at day 6 the trailing 7 d ring-3 count is 6.
        """
        frame = make_panel({(3, 3, 2): 4, (0, 3, 5): 2}, n_rows=5, n_cols=5, n_days=20)
        df = NearRepeatStep().run(frame).collect()
        assert cell_day_values(df, 0, "count_nr_r3_7d")[6] == 6
        assert cell_day_values(df, 0, "count_nr_r3_7d")[3] == 4
        assert cell_day_values(df, 0, "count_nr_r3_7d")[2] == 0
        # Ring 1 of the corner is untouched by these events.
        assert cell_day_values(df, 0, "count_nr_r1_7d") == [0] * 20

    def test_event_5_days_vs_10_days_before(self) -> None:
        """The 7-14 d window excludes the trailing 7 d.

        One event at day 2 (centre's ring-1 neighbour). At day 8 the
        event is 5 days back: in the 7 d window, not in 7-14 d. At
        day 13 it is 10 days back: in the 7-14 d window, not in 7 d.
        """
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20)
        df = NearRepeatStep().run(frame).collect()
        centre = 4
        assert cell_day_values(df, centre, "count_nr_r1_7d")[7] == 5
        assert cell_day_values(df, centre, "count_nr_r1_7_14d")[7] == 0
        assert cell_day_values(df, centre, "count_nr_r1_7d")[12] == 0
        assert cell_day_values(df, centre, "count_nr_r1_7_14d")[12] == 5

    def test_island_cell_gives_zero_not_null(self) -> None:
        """A lone cell has no neighbours: all ring counts and flag are 0."""
        frame = make_panel({(0, 0, 2): 5, (0, 0, 5): 3}, n_rows=1, n_cols=1, n_days=20)
        df = NearRepeatStep().run(frame).collect()
        for ring in (1, 2, 3):
            assert cell_day_values(df, 0, f"count_nr_r{ring}_7d") == [0] * 20
            assert cell_day_values(df, 0, f"count_nr_r{ring}_7_14d") == [0] * 20
        assert cell_day_values(df, 0, "count_nr_event_7d") == [0] * 20
        # Its own events never count as near-repeat (no self-ring).
        assert df["count"].to_list() == [0, 0, 5, 0, 0, 3] + [0] * 14

    def test_early_bins_have_no_history(self) -> None:
        """Bins before any history sum over the bins that exist (0 at t=0)."""
        frame = make_panel({(0, 1, 0): 5}, n_rows=3, n_cols=3, n_days=20)
        df = NearRepeatStep().run(frame).collect()
        # The event is at bin 0: it enters the centre's trailing 7 d
        # window from bin 1, and the 7-14 d window from bin 8.
        assert cell_day_values(df, 4, "count_nr_r1_7d")[0] == 0
        assert cell_day_values(df, 4, "count_nr_r1_7d")[1] == 5
        assert cell_day_values(df, 4, "count_nr_r1_7_14d")[7] == 0
        assert cell_day_values(df, 4, "count_nr_r1_7_14d")[8] == 5

    def test_six_hour_panel_converts_days_to_bins(self) -> None:
        """On a 6 h panel a 7 d window is 28 bins (EWMAMean convention)."""
        # 6 h bins: day_index counts 6 h bins; event at bin 2.
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=120, time_bin="6h")
        df = NearRepeatStep().run(frame).collect()
        # Included for the 28 bins after the event (bins 3..30), then out.
        assert cell_day_values(df, 4, "count_nr_r1_7d")[30] == 5
        assert cell_day_values(df, 4, "count_nr_r1_7d")[31] == 0
        # At bin 31 the event is 29 bins (7 d 6 h) back: in 7-14 d.
        assert cell_day_values(df, 4, "count_nr_r1_7_14d")[31] == 5
        assert cell_day_values(df, 4, "count_nr_r1_7_14d")[30] == 0

    def test_custom_rings_and_flag_without_ring1(self) -> None:
        """rings=(2, 3) emits only r2/r3 columns, but the flag still uses ring 1."""
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20)
        df = NearRepeatStep(rings=(2, 3)).run(frame).collect()
        assert "count_nr_r1_7d" not in df.columns
        assert "count_nr_r2_7d" in df.columns
        assert "count_nr_r3_7_14d" in df.columns
        assert cell_day_values(df, 4, "count_nr_event_7d") == [0, 0, 0] + [1] * 7 + [0] * 10

    def test_custom_value_col(self) -> None:
        """A non-default value_col renames every output column."""
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20, value_col="events")
        df = NearRepeatStep(value_col="events").run(frame).collect()
        assert cell_day_values(df, 4, "events_nr_r1_7d")[3] == 5
        assert "events_nr_event_7d" in df.columns
        assert "count_nr_r1_7d" not in df.columns

    def test_output_dtypes(self) -> None:
        """Ring counts keep the count dtype; the flag is Int64 0/1."""
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20)
        df = NearRepeatStep().run(frame).collect()
        assert df["count_nr_r1_7d"].dtype == pl.Int64
        assert df["count_nr_event_7d"].dtype == pl.Int64
        assert set(df["count_nr_event_7d"].to_list()) <= {0, 1}

    def test_registers_provenance(self) -> None:
        """Output columns carry provenance with rings, windows, closed-left."""
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20)
        result = NearRepeatStep().run(frame)
        provenance = result.metadata.feature_provenance["count_nr_r1_7d"]
        assert provenance.produced_by == "NearRepeatStep"
        assert "spatial" in provenance.tags
        assert "temporal" in provenance.tags
        assert provenance.metadata["rings"] == [1, 2, 3]
        assert provenance.metadata["windows_days"] == [7, 14]
        assert provenance.metadata["closed"] == "left"
        assert provenance.metadata["window_bins_7d"] == 7
        assert provenance.metadata["islands"] == "zero"
        flag_prov = result.metadata.feature_provenance["count_nr_event_7d"]
        assert flag_prov.metadata["ring"] == 1


class TestNearRepeatLeakage:
    """No feature at bin t may depend on bins after t (or on bin t itself)."""

    def test_ignores_future_bins(self) -> None:
        """Changing counts at bins > t leaves outputs at bins <= t unchanged.

        Cell 6 is (1,1); cell 1 is (0,1), one of its ring-1 neighbours.
        Changing cell 1's counts at days 15/17/19 must not move any of
        cell 6's outputs at bins <= 9, must move its ring-1 7 d count
        at bin 16 (window [9, 15] includes day 15), and must leave its
        ring-3 counts untouched everywhere (cell 1 is not a ring-3
        neighbour of cell 6).
        """
        frame = make_panel(EVENTS_5X5, n_rows=5, n_cols=5, n_days=20)
        step = NearRepeatStep()
        before = step.run(frame).collect()
        changed = frame
        for day, count in [(15, 99), (17, 99), (19, 99)]:
            changed = with_changed_count(changed, 1, day, count)
        after = step.run(changed).collect()
        for col in (
            "count_nr_r1_7d",
            "count_nr_r1_7_14d",
            "count_nr_r2_7d",
            "count_nr_r3_7_14d",
            "count_nr_event_7d",
        ):
            assert cell_day_values(before, 6, col)[:10] == cell_day_values(after, 6, col)[:10]
        assert (
            cell_day_values(before, 6, "count_nr_r1_7d")[16]
            != cell_day_values(after, 6, "count_nr_r1_7d")[16]
        )
        assert cell_day_values(before, 6, "count_nr_r3_7d") == cell_day_values(
            after, 6, "count_nr_r3_7d"
        )
        assert cell_day_values(before, 6, "count_nr_r3_7_14d") == cell_day_values(
            after, 6, "count_nr_r3_7_14d"
        )

    def test_closed_left_ignores_current_bin(self) -> None:
        """Changing the count AT bin t leaves the outputs at bin t unchanged."""
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20)
        step = NearRepeatStep()
        before = step.run(frame).collect()
        # Change the event cell's own count at day 5 (index 5).
        changed = with_changed_count(frame, 1, 5, 99)
        after = step.run(changed).collect()
        for col in ("count_nr_r1_7d", "count_nr_r1_7_14d", "count_nr_event_7d"):
            assert cell_day_values(before, 4, col)[5] == cell_day_values(after, 4, col)[5]
        # The NEXT bin's trailing 7 d window now includes the changed count.
        assert (
            cell_day_values(before, 4, "count_nr_r1_7d")[6]
            != cell_day_values(after, 4, "count_nr_r1_7d")[6]
        )
        # The flag stays 1 at bin 6 (it was already 1) but turns 1 at
        # bin 10, where the original window had gone quiet.
        assert (
            cell_day_values(before, 4, "count_nr_event_7d")[6]
            == cell_day_values(after, 4, "count_nr_event_7d")[6]
        )
        assert (
            cell_day_values(before, 4, "count_nr_event_7d")[10]
            != cell_day_values(after, 4, "count_nr_event_7d")[10]
        )
        # The 7-14 d window only reaches day 5 at bin 13.
        assert (
            cell_day_values(before, 4, "count_nr_r1_7_14d")[6]
            == cell_day_values(after, 4, "count_nr_r1_7_14d")[6]
        )
        assert (
            cell_day_values(before, 4, "count_nr_r1_7_14d")[13]
            != cell_day_values(after, 4, "count_nr_r1_7_14d")[13]
        )


class TestNearRepeatValidation:
    """Bad parameters and bad values fail loudly, never silently."""

    def test_null_value_raises(self) -> None:
        """A null count is a blocking error, not a silent zero."""
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20)
        df = frame.collect().with_columns(
            pl.when(pl.col("time_bin") == START + timedelta(days=3))
            .then(None)
            .otherwise(pl.col("count"))
            .alias("count")
        )
        bad = EventFrame(df.lazy(), frame.schema, frame.metadata)
        with pytest.raises(ValueError, match="finite"):
            NearRepeatStep().run(bad)

    def test_nonfinite_value_raises(self) -> None:
        """An infinite count is a blocking error."""
        frame = make_panel({}, n_rows=3, n_cols=3, n_days=20)
        df = frame.collect().with_columns(
            pl.when(pl.col("time_bin") == START + timedelta(days=3))
            .then(float("inf"))
            .otherwise(pl.col("count").cast(pl.Float64))
            .alias("count")
        )
        bad = EventFrame(df.lazy(), frame.schema, frame.metadata)
        with pytest.raises(ValueError, match="finite"):
            NearRepeatStep().run(bad)

    def test_null_time_raises(self) -> None:
        """A null time is a blocking error (Phase 2 Chicago replay).

        On the completed Chicago panel the pre-completion "time_bin"
        column is null on every generated row; running the step on it
        silently computes the trailing windows over the sparse
        observed-bins-only series (wrong values, no error). The step
        must refuse a null time column and point at the completed
        panel's timestamp column.
        """
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20)
        df = frame.collect().with_columns(
            pl.when(pl.col("time_bin") == START + timedelta(days=3))
            .then(None)
            .otherwise(pl.col("time_bin"))
            .alias("time_bin")
        )
        bad = EventFrame(df.lazy(), frame.schema, frame.metadata)
        with pytest.raises(ValueError, match="non-null times"):
            NearRepeatStep().run(bad)

    def test_ring_zero_raises(self) -> None:
        """Ring 0 (the cell itself) is not a ring."""
        with pytest.raises(ValueError, match="rings"):
            NearRepeatStep(rings=(0, 1))

    def test_negative_ring_raises(self) -> None:
        """Negative rings are rejected at construction."""
        with pytest.raises(ValueError, match="rings"):
            NearRepeatStep(rings=(-1, 2))

    def test_empty_rings_raises(self) -> None:
        """An empty rings list must fail, not silently no-op."""
        with pytest.raises(ValueError, match="rings"):
            NearRepeatStep(rings=[])

    def test_duplicate_rings_raises(self) -> None:
        """Duplicate rings are rejected at construction."""
        with pytest.raises(ValueError, match="rings"):
            NearRepeatStep(rings=(1, 1, 2))

    def test_missing_time_bin_metadata_raises(self) -> None:
        """Without metadata.time_bin the day windows cannot be converted."""
        frame = make_panel({}, n_rows=3, n_cols=3, n_days=20).with_metadata(time_bin=None)
        with pytest.raises(ValueError, match="time_bin"):
            NearRepeatStep().run(frame)

    def test_unparseable_time_bin_raises(self) -> None:
        """An unparseable metadata.time_bin must fail with a clear error."""
        frame = make_panel({}, n_rows=3, n_cols=3, n_days=20).with_metadata(time_bin="bogus")
        with pytest.raises(ValueError, match="bogus"):
            NearRepeatStep().run(frame)

    def test_missing_grid_columns_raise(self) -> None:
        """grid_row / grid_col are required to build the edge table."""
        frame = make_panel({}, n_rows=3, n_cols=3, n_days=20)
        df = frame.collect().drop("grid_row")
        bad = EventFrame(df.lazy(), frame.schema, frame.metadata)
        with pytest.raises(ValueError, match="grid_row"):
            NearRepeatStep().run(bad)

    def test_missing_value_column_raises(self) -> None:
        """A missing value column must fail, not silently no-op."""
        frame = make_panel({}, n_rows=3, n_cols=3, n_days=20)
        with pytest.raises(ValueError, match="nope"):
            NearRepeatStep(value_col="nope").run(frame)

    def test_config_model_roundtrip(self) -> None:
        """The registry path (config -> model_dump -> step kwargs) works."""
        frame = make_panel({(0, 1, 2): 5}, n_rows=3, n_cols=3, n_days=20)
        config = NearRepeatConfig(rings=[1, 2])
        step = NearRepeatStep(**config.model_dump())
        df = step.run(frame).collect()
        assert "count_nr_r1_7d" in df.columns
        assert "count_nr_r2_7_14d" in df.columns
        assert "count_nr_r3_7d" not in df.columns

    def test_config_model_rejects_bad_rings(self) -> None:
        """Pydantic guards the registry path too."""
        with pytest.raises(ValidationError):
            NearRepeatConfig(rings=[0])
        with pytest.raises(ValidationError):
            NearRepeatConfig(rings=[])
        with pytest.raises(ValidationError):
            NearRepeatConfig(rings=[-2])


class TestNearRepeatPerformance:
    """Modest performance sanity check (not a benchmark)."""

    def test_1000_cells_x_20_bins_completes(self) -> None:
        """~1,000 cells x 20 bins runs in well under a minute."""
        events = {
            (r, c, d): (r + c + d) % 3 for r in range(25) for c in range(40) for d in (0, 5, 11)
        }
        frame = make_panel(events, n_rows=25, n_cols=40, n_days=20)
        start = time.perf_counter()
        df = NearRepeatStep().run(frame).collect()
        elapsed = time.perf_counter() - start
        assert df.height == 1000 * 20
        assert df["count_nr_r1_7d"].sum() is not None
        assert elapsed < 60
