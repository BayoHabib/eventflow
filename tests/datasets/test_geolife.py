"""GeoLife GPS trajectories: loader, kinematics, recipe, boundary tests.

Boundary convention: a step that only *runs* on trajectories is not
enough — each verdict below is backed by hand-computed values on a
tiny synthetic trajectory/panel, or by a demonstrated failure
(all-null lags, invented panel zeros, ...). The full statement lives
in ``docs/design-notes/phase3-geolife-boundary.md``.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema, RecipeConfig
from eventflow.core.steps.categorical import CategoryCountsStep
from eventflow.core.steps.density import KernelDensityStep
from eventflow.core.steps.dynamics import (
    BurstinessStep,
    EWMAMeanStep,
    TimeSinceLastEventStep,
    TrendStep,
    ZeroFeaturesStep,
)
from eventflow.core.steps.hawkes import HawkesExcitationStep
from eventflow.core.steps.near_repeat import NearRepeatStep
from eventflow.core.steps.panel import CompletePanelStep
from eventflow.core.steps.spatial import MultiScaleSpatialLagStep
from eventflow.core.steps.temporal import (
    CalendarEncodingStep,
    ExtractTemporalComponentsStep,
    FillNullStep,
    InterArrivalStep,
    MovingAverageStep,
    RecencyWeightStep,
    TemporalLagStep,
    TimeBinsStep,
)
from eventflow.datasets.geolife.kinematics import TrajectoryKinematicsStep
from eventflow.datasets.geolife.loader import load_geolife, parse_plt
from eventflow.datasets.geolife.recipes.geolife_v1 import GeoLifeV1Recipe

_EPOCH = date(1899, 12, 30)
_PLT_HEADER = (
    "Geolife trajectory\n"
    "WGS 84\n"
    "Altitude is in Feet\n"
    "Reserved 3\n"
    "0,2,255,My Track,0,0,2,8421376\n"
    "0\n"
)


def _plt_line(moment: datetime, lat: float, lon: float, alt_ft: float = 100.0) -> str:
    days = (moment.date() - _EPOCH).days + (
        moment.hour * 3600 + moment.minute * 60 + moment.second
    ) / 86400
    return f"{lat},{lon},0,{alt_ft},{days:.7f},{moment.date().isoformat()},{moment.time().strftime('%H:%M:%S')}"


def _write_plt(path: Path, moments: list[datetime], lats: list[float], lons: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_plt_line(m, lat, lon) for m, lat, lon in zip(moments, lats, lons, strict=True)]
    path.write_text(_PLT_HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def _point_frame(rows: list[dict]) -> EventFrame:
    """EventFrame of projected-metre points (no CRS transform needed)."""
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["speed_m_s", "speed", "count"],
    )
    metadata = EventMetadata(dataset_name="test-geolife", crs="EPSG:32650")
    return EventFrame(pl.LazyFrame(rows), schema, metadata)


def _panel_frame() -> EventFrame:
    """One person moving A -> B -> C, one cell per 5-minute bin.

    Cells (grid_row, grid_col): A=(0,0), B=(0,1), C=(0,2); centres
    (lon, lat) in projected metres: (50,50), (150,50), (250,50).
    Counts: A=[5,0,0], B=[0,4,0], C=[0,0,3] over bins t0/t1/t2.
    """
    t0 = datetime(2024, 1, 1, 0, 0)
    rows = []
    for grid_id, (row, col, x), counts in (
        (1, (0, 0, 50), [5, 0, 0]),
        (2, (0, 1, 150), [0, 4, 0]),
        (3, (0, 2, 250), [0, 0, 3]),
    ):
        for bin_index, count in enumerate(counts):
            moment = t0 + timedelta(minutes=5 * bin_index)
            rows.append(
                {
                    "grid_id": grid_id,
                    "grid_row": row,
                    "grid_col": col,
                    "timestamp": moment,
                    "time_bin": moment,
                    "count": count,
                    "longitude_proj": float(x),
                    "latitude_proj": 50.0,
                }
            )
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude_proj",
        lon_col="longitude_proj",
        numeric_cols=["count", "grid_id", "grid_row", "grid_col"],
    )
    metadata = EventMetadata(dataset_name="test-geolife-panel", crs="EPSG:32650", time_bin="5m")
    return EventFrame(pl.LazyFrame(rows), schema, metadata)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_parse_plt_rows_and_altitude_conversion(self, tmp_path: Path) -> None:
        path = _write_plt(
            tmp_path / "000" / "Trajectory" / "20070804033032.plt",
            [datetime(2007, 8, 4, 3, 30, 32), datetime(2007, 8, 4, 3, 30, 33)],
            [39.921712, 39.921705],
            [116.472343, 116.472343],
        )
        frame = parse_plt(path, user_id="010")
        assert frame.height == 2
        assert frame["user_id"].to_list() == ["010", "010"]
        assert frame["trajectory_id"].to_list() == ["20070804033032"] * 2
        assert frame["timestamp"].to_list() == [
            datetime(2007, 8, 4, 3, 30, 32),
            datetime(2007, 8, 4, 3, 30, 33),
        ]
        # 100 ft * 0.3048 = 30.48 m.
        assert frame["altitude_m"].to_list() == [30.48, 30.48]
        assert frame["latitude"].to_list() == [39.921712, 39.921705]

    def test_parse_plt_rejects_bad_header(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.plt"
        path.write_text("Not a trajectory\nWGS 84\n", encoding="utf-8")
        with pytest.raises(ValueError, match="header"):
            parse_plt(path, user_id="000")

    def test_parse_plt_rejects_days_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.plt"
        moment = datetime(2007, 8, 4, 3, 30, 32)
        line = _plt_line(moment, 39.9, 116.4).split(",")
        line[4] = "1.0"  # days column disagrees with the date by years
        path.write_text(_PLT_HEADER + ",".join(line) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="days"):
            parse_plt(path, user_id="000")

    def test_parse_plt_rejects_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.plt"
        path.write_text(_PLT_HEADER, encoding="utf-8")
        with pytest.raises(ValueError, match="no GPS fixes"):
            parse_plt(path, user_id="000")

    def test_load_geolife_combines_and_sorts(self, tmp_path: Path) -> None:
        p1 = _write_plt(
            tmp_path / "010" / "Trajectory" / "20070804033032.plt",
            [datetime(2007, 8, 4, 3, 30, 32)],
            [39.921712],
            [116.472343],
        )
        p2 = _write_plt(
            tmp_path / "000" / "Trajectory" / "20090508100431.plt",
            [datetime(2009, 5, 8, 10, 4, 31)],
            [40.000299],
            [116.327056],
        )
        frame = load_geolife([p1, p2]).collect()
        assert frame.height == 2
        assert frame["user_id"].to_list() == ["000", "010"]
        assert frame.schema["timestamp"] == pl.Datetime


# ---------------------------------------------------------------------------
# Kinematics (generic per-point derivation)
# ---------------------------------------------------------------------------


class TestKinematics:
    def test_hand_computed_distances_and_speeds(self) -> None:
        # Points (lon, lat) in metres: (0,0) -> (3,4) -> (3,4) -> (13,4)
        # at t = 0, 10, 10, 30 s. Distances 5, 0, 10 m; dt 10, 0, 20 s.
        rows = [
            {
                "user_id": "u",
                "trajectory_id": "t",
                "timestamp": datetime(2024, 1, 1) + timedelta(seconds=s),
                "latitude": lat,
                "longitude": lon,
            }
            for s, (lon, lat) in zip(
                [0, 10, 10, 30],
                [(0.0, 0.0), (3.0, 4.0), (3.0, 4.0), (13.0, 4.0)],
                strict=True,
            )
        ]
        out = TrajectoryKinematicsStep().run(_point_frame(rows)).collect()
        assert out["step_distance_m"].to_list() == [None, 5.0, 0.0, 10.0]
        assert out["dt_s"].to_list() == [None, 10, 0, 20]
        # Speed: first fix null; dt=0 null (no division by zero);
        # 5/10 = 0.5 and 10/20 = 0.5.
        assert out["speed_m_s"].to_list() == [None, 0.5, None, 0.5]

    def test_long_gap_reports_distance_but_no_speed(self) -> None:
        rows = [
            {
                "user_id": "u",
                "trajectory_id": "t",
                "timestamp": datetime(2024, 1, 1) + timedelta(seconds=s),
                "latitude": 0.0,
                "longitude": lon,
            }
            for s, lon in zip([0, 1000], [0.0, 100.0], strict=True)
        ]
        out = TrajectoryKinematicsStep(max_gap_s=300.0).run(_point_frame(rows)).collect()
        assert out["step_distance_m"].to_list() == [None, 100.0]
        assert out["dt_s"].to_list() == [None, 1000]
        assert out["speed_m_s"].to_list() == [None, None]

    def test_groups_do_not_leak_across_trajectories(self) -> None:
        rows = [
            {
                "user_id": "u1",
                "trajectory_id": "t1",
                "timestamp": datetime(2024, 1, 1),
                "latitude": 0.0,
                "longitude": 0.0,
            },
            {
                "user_id": "u2",
                "trajectory_id": "t2",
                "timestamp": datetime(2024, 1, 1),
                "latitude": 0.0,
                "longitude": 100.0,
            },
        ]
        out = TrajectoryKinematicsStep().run(_point_frame(rows)).collect()
        # Each trajectory's first fix has null kinematics: no cross-user leak.
        assert out["step_distance_m"].to_list() == [None, None]
        assert out["speed_m_s"].to_list() == [None, None]


# ---------------------------------------------------------------------------
# Point-level temporal steps: APPLY (rows mode), NOT duration mode
# ---------------------------------------------------------------------------


class TestPointLevelTemporal:
    def _speed_frame(self) -> EventFrame:
        rows = [
            {
                "user_id": "u",
                "trajectory_id": "t",
                "timestamp": datetime(2024, 1, 1) + timedelta(seconds=10 * i),
                "latitude": 0.0,
                "longitude": float(i),
                "speed": float(v),
                "speed_m_s": float(v),
            }
            for i, v in enumerate([2.0, 4.0, 6.0, 8.0])
        ]
        return _point_frame(rows)

    def test_rows_mode_lag_and_rolling_mean_hand_computed(self) -> None:
        frame = self._speed_frame()
        lagged = TemporalLagStep(
            ["speed"], lag_periods=[1], mode="rows", group_cols=["user_id"]
        ).run(frame)
        assert lagged.collect()["speed_lag_1"].to_list() == [None, 2.0, 4.0, 6.0]
        averaged = MovingAverageStep(
            ["speed"], windows=[5], mode="rows", group_cols=["user_id"]
        ).run(frame)
        # Rows mode includes the current row: [2], [2,4], [2,4,6], [2,4,6,8].
        assert averaged.collect()["speed_ma_5"].to_list() == [2.0, 3.0, 4.0, 5.0]

    def test_duration_mode_lag_is_all_null_on_irregular_sampling(self) -> None:
        # Fixes at 0, 1, 3, 7 s: no fix exists exactly 60 s before any fix,
        # so the exact-elapsed lag is null everywhere. Duration-mode lags
        # do not apply to irregularly sampled trajectories.
        rows = [
            {
                "user_id": "u",
                "trajectory_id": "t",
                "timestamp": datetime(2024, 1, 1) + timedelta(seconds=s),
                "latitude": 0.0,
                "longitude": float(s),
                "speed": float(s + 1),
                "speed_m_s": float(s + 1),
            }
            for s in [0, 1, 3, 7]
        ]
        out = (
            TemporalLagStep(
                ["speed"], lag_periods=[1], time_unit="60s", mode="duration", group_cols=["user_id"]
            )
            .run(_point_frame(rows))
            .collect()
        )
        assert out["speed_lag_1"].null_count() == 4

    def test_inter_arrival_hand_computed(self) -> None:
        rows = [
            {
                "user_id": "u",
                "trajectory_id": "t",
                "timestamp": datetime(2024, 1, 1) + timedelta(seconds=s),
                "latitude": 0.0,
                "longitude": 0.0,
                "speed": 1.0,
                "speed_m_s": 1.0,
            }
            for s in [0, 5, 15, 16]
        ]
        out = (
            InterArrivalStep(group_cols=["user_id", "trajectory_id"])
            .run(_point_frame(rows))
            .collect()
        )
        assert out["inter_arrival_seconds"].to_list() == [None, 5, 10, 1]
        # Rolling mean over 10 rows, nulls ignored: [5], [5,10], [5,10,1].
        assert out["inter_arrival_seconds_ma_10"].to_list() == pytest.approx(
            [None, 5.0, 7.5, 16.0 / 3.0]
        )

    def test_calendar_and_components_hand_computed(self) -> None:
        # 2007-08-04 03:00 was a Saturday (Polars weekday: 6).
        rows = [
            {
                "user_id": "u",
                "trajectory_id": "t",
                "timestamp": datetime(2007, 8, 4, 3, 0, 0),
                "latitude": 0.0,
                "longitude": 0.0,
                "speed": 1.0,
                "speed_m_s": 1.0,
            }
        ]
        frame = _point_frame(rows)
        out = ExtractTemporalComponentsStep(["hour_of_day", "day_of_week"]).run(frame).collect()
        assert out["hour_of_day"].to_list() == [3]
        assert out["day_of_week"].to_list() == [6]
        encoded = CalendarEncodingStep("cyclical", ["hour", "day_of_week"]).run(frame).collect()
        assert encoded["hour_sin"].to_list() == pytest.approx([math.sin(2 * math.pi * 3 / 24)])
        assert encoded["hour_cos"].to_list() == pytest.approx([math.cos(2 * math.pi * 3 / 24)])
        assert encoded["dow_sin"].to_list() == pytest.approx([math.sin(2 * math.pi * 6 / 7)])
        assert encoded["dow_cos"].to_list() == pytest.approx([math.cos(2 * math.pi * 6 / 7)])

    def test_recency_weight_hand_computed(self) -> None:
        # w = 0.5 ** ((t_ref - t) / halflife); halflife 10 s, t_ref = 20 s.
        rows = [
            {
                "user_id": "u",
                "trajectory_id": "t",
                "timestamp": datetime(2024, 1, 1) + timedelta(seconds=s),
                "latitude": 0.0,
                "longitude": 0.0,
                "speed": 1.0,
                "speed_m_s": 1.0,
            }
            for s in [0, 10, 20]
        ]
        out = RecencyWeightStep(half_life=10.0, time_unit="1s").run(_point_frame(rows)).collect()
        assert out["recency_weight"].to_list() == pytest.approx([0.25, 0.5, 1.0])

    def test_trend_ratio_of_speed_means(self) -> None:
        frame = self._speed_frame()
        averaged = MovingAverageStep(
            ["speed"], windows=[2, 4], mode="rows", group_cols=["user_id"]
        ).run(frame)
        out = TrendStep(recent_col="speed_ma_2", long_col="speed_ma_4").run(averaged).collect()
        # ma_2 = [2, 3, 5, 7]; ma_4 = [2, 3, 4, 5]; ratio = [1, 1, 1.25, 1.4].
        assert out["speed_ma_2_vs_speed_ma_4"].to_list() == pytest.approx([1.0, 1.0, 1.25, 1.4])

    def test_ewma_on_point_speed_needs_sub_day_halflife(self) -> None:
        # EWMA is defined over bins: it needs metadata.time_bin, and on a
        # 44-minute trajectory the half-life must be minutes, not days.
        # halflife 5 min on 5-min bins = 1 bin; speeds [1,2,3] with
        # adjust=True: [1, 5/3, 2.428571...].
        rows = [
            {
                "user_id": "u",
                "trajectory_id": "t",
                "timestamp": datetime(2024, 1, 1) + timedelta(minutes=5 * i),
                "latitude": 0.0,
                "longitude": float(i),
                "speed": float(i + 1),
                "speed_m_s": float(i + 1),
            }
            for i in range(3)
        ]
        binned = TimeBinsStep(bin_size="5m").run(_point_frame(rows))
        out = (
            EWMAMeanStep(["speed"], halflife_days=5.0 / 1440.0, group_cols=["user_id"])
            .run(binned)
            .collect()
        )
        col = next(c for c in out.columns if c.startswith("speed_ewma_"))
        assert out[col].to_list() == pytest.approx([1.0, 5.0 / 3.0, 2.4285714285714284])


# ---------------------------------------------------------------------------
# Panel-family steps on the (cell, bin) count panel: they RUN, but the
# semantics collapse to re-describing the one person's path.
# ---------------------------------------------------------------------------


class TestPanelBoundary:
    def test_complete_panel_invents_zeros_where_the_person_was_not(self) -> None:
        out = (
            CompletePanelStep(unit_col="grid_id", value_cols=["count"], bin_size="5m")
            .run(_panel_frame())
            .collect()
        )
        # 3 observed (cell, bin) pairs -> 3 cells x 3 bins = 9 rows.
        assert out.height == 9
        assert out["observation_status"].to_list() == ["observed"] * 9
        by_key = {(r["grid_id"], r["timestamp"]): r["count"] for r in out.iter_rows(named=True)}
        t0, t1 = datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 5)
        assert by_key[(1, t0)] == 5
        # Cell A at t1: the person was in B, not A — the zero means
        # "person elsewhere", not "no event recorded in A".
        assert by_key[(1, t1)] == 0
        assert sum(1 for v in by_key.values() if v == 0) == 6

    def test_spatial_lag_hand_computed_is_the_path_one_bin_ago(self) -> None:
        panel = _panel_frame()
        lagged = TemporalLagStep(
            ["count"], lag_periods=[1], time_unit="5m", mode="duration", group_cols=["grid_id"]
        ).run(panel)
        filled = FillNullStep(value_cols=["count_lag_1"], fill_value=0).run(lagged)
        out = (
            MultiScaleSpatialLagStep(
                value_cols=["count_lag_1"],
                neighbor_col="grid_id",
                rings=(1,),
                group_cols=["timestamp"],
            )
            .run(filled)
            .collect()
        )
        by_key = {
            (r["grid_id"], r["timestamp"]): r["count_lag_1_spatial_lag_r1"]
            for r in out.iter_rows(named=True)
        }
        t0 = datetime(2024, 1, 1)
        t1, t2 = t0 + timedelta(minutes=5), t0 + timedelta(minutes=10)
        # Ring-1 lag of B at t1 = mean of neighbours' lagged counts
        # (A=5, C=0) = 2.5; of A at t2 = B's lagged count = 4.
        assert by_key[(2, t1)] == pytest.approx(2.5)
        assert by_key[(1, t2)] == pytest.approx(4.0)
        assert by_key[(3, t2)] == pytest.approx(4.0)
        # 6 of 9 lag values are exactly 0: the "field" is empty except
        # where the path just was (t1: B=2.5; t2: A=C=4.0).
        values = list(by_key.values())
        assert sum(1 for v in values if v == 0.0) == 6

    def test_kde_smears_the_path_into_empty_cells(self) -> None:
        panel = _panel_frame()
        lagged = TemporalLagStep(
            ["count"], lag_periods=[1], time_unit="5m", mode="duration", group_cols=["grid_id"]
        ).run(panel)
        filled = FillNullStep(value_cols=["count_lag_1"], fill_value=0).run(lagged)
        out = (
            KernelDensityStep(value_col="count_lag_1", bandwidth_m=150.0, group_cols=["timestamp"])
            .run(filled)
            .collect()
        )
        t1 = datetime(2024, 1, 1, 0, 5)
        row = out.filter(pl.col("timestamp") == t1).sort("grid_id")
        kde = row["count_lag_1_kde"].to_list()
        # At t1 the lagged mass is 5 in A only (h=150, centres 100 m apart):
        # kde(A) = 5/(2*pi*h^2); kde(B) = 5*exp(-100^2/2h^2)/(2*pi*h^2);
        # kde(C) = 5*exp(-200^2/2h^2)/(2*pi*h^2) > 0 although C's own
        # count is 0 — the density is the path, smoothed.
        norm = 2 * math.pi * 150.0**2
        assert kde == pytest.approx(
            [
                5.0 / norm,
                5.0 * math.exp(-10000.0 / 45000.0) / norm,
                5.0 * math.exp(-40000.0 / 45000.0) / norm,
            ]
        )
        assert kde[0] > kde[1] > kde[2] > 0

    def test_near_repeat_counts_the_trajectory_itself(self) -> None:
        out = (
            NearRepeatStep(value_col="count", time_col="timestamp", rings=(1,))
            .run(_panel_frame())
            .collect()
        )
        by_key = {
            (r["grid_id"], r["timestamp"]): (r["count_nr_r1_7d"], r["count_nr_event_7d"])
            for r in out.iter_rows(named=True)
        }
        t1 = datetime(2024, 1, 1, 0, 5)
        # B at t1: ring-1 neighbours A and C over the trailing 7 d
        # (here: the whole 5-minute history) hold 5 + 0 = 5 fixes —
        # the "near repeat" is the trajectory's own earlier position.
        assert by_key[(2, t1)] == (5, 1)
        assert by_key[(1, t1)] == (0, 0)
        assert by_key[(3, t1)] == (0, 0)

    def test_hawkes_excitation_is_decayed_self_count(self) -> None:
        out = (
            HawkesExcitationStep(value_col="count", halflife_days=7.0, group_cols=["grid_id"])
            .run(_panel_frame())
            .collect()
        )
        col = "count_hawkes_7d"
        rows = out.filter(pl.col("grid_id") == 1).sort("timestamp")
        decay = 0.5 ** (1.0 / 2016.0)  # halflife 7 d on 5-min bins = 2016 bins
        # Cell A counts [5, 0, 0]: E = [0, 5*d, 5*d^2].
        assert rows[col].to_list() == pytest.approx([0.0, 5 * decay, 5 * decay**2])

    def test_time_since_last_event_is_a_revisit_time(self) -> None:
        out = (
            TimeSinceLastEventStep(value_col="count", group_cols=["grid_id"])
            .run(_panel_frame())
            .collect()
        )
        rows = out.filter(pl.col("grid_id") == 1).sort("timestamp")
        # Cell A: positive at t0, then 5 and 10 minutes of absence,
        # reported in days (the only supported output unit).
        assert rows["count_time_since_last_d"].to_list() == pytest.approx(
            [0.0, 5.0 / 1440.0, 10.0 / 1440.0]
        )
        # Cell B first appears at t1: no prior positive bin -> null.
        rows_b = out.filter(pl.col("grid_id") == 2).sort("timestamp")
        assert rows_b["count_time_since_last_d"].to_list()[0] is None

    def test_burstiness_and_zero_features_on_panel(self) -> None:
        panel = _panel_frame()
        bursty = (
            BurstinessStep(value_col="count", window="28d", group_cols=["grid_id"])
            .run(panel)
            .collect()
        )
        rows = bursty.filter(pl.col("grid_id") == 1).sort("timestamp")
        # Cell A, t2: prior counts [5, 0] -> mean 2.5, sample std 3.5355.
        assert rows["count_burstiness_28d"].to_list()[2] == pytest.approx(3.5355339059327378 / 2.5)
        zeroed = ZeroFeaturesStep(value_col="count", group_cols=["grid_id"]).run(panel).collect()
        rows = zeroed.filter(pl.col("grid_id") == 1).sort("timestamp")
        # Share of nonzero bins over 28 d at t2: 1 of 3.
        assert rows["count_share_nonzero_28d"].to_list()[2] == pytest.approx(1.0 / 3.0)

    def test_category_counts_has_no_column_to_count(self) -> None:
        # .plt files carry no categorical column, so CategoryCountsStep
        # cannot run on trajectory data at all.
        with pytest.raises(ValueError, match="transport_mode"):
            CategoryCountsStep(
                category_cols=["transport_mode"], unit_col="grid_id", time_col="timestamp"
            ).run(_panel_frame())


# ---------------------------------------------------------------------------
# Recipe on synthetic .plt files
# ---------------------------------------------------------------------------


class TestRecipe:
    def test_recipe_counts_every_fix(self, tmp_path: Path) -> None:
        p1 = _write_plt(
            tmp_path / "010" / "Trajectory" / "20070804033032.plt",
            [datetime(2007, 8, 4, 3, 30, 32 + 5 * i) for i in range(3)],
            [39.921712, 39.921800, 39.921900],
            [116.472343, 116.472400, 116.472450],
        )
        p2 = _write_plt(
            tmp_path / "000" / "Trajectory" / "20090508100431.plt",
            [datetime(2009, 5, 8, 10, 4, 31 + 5 * i) for i in range(3)],
            [40.000299, 40.000350, 40.000400],
            [116.327056, 116.327100, 116.327150],
        )
        events = load_geolife([p1, p2])
        config = RecipeConfig(
            dataset="geolife",
            recipe="geolife_v1",
            grid={"size_m": 100.0},
            temporal={"time_bin": "5m"},
        )
        out = GeoLifeV1Recipe(config).run(events).collect()
        # Every fix is counted exactly once, in exactly one (cell, bin).
        assert out["count"].sum() == 6
        assert {"grid_id", "time_bin", "count", "grid_row", "grid_col"} <= set(out.columns)
        assert out["timestamp"].equals(out["time_bin"])
