"""Tests that execute the code examples in docs/api_reference.md (phase 5).

Rule for this phase: every code example in the docs must be executed by
a test or a script, never claimed from reading. The examples below are
the ones printed in ``docs/api_reference.md`` (panel completion, dynamics
steps, categorical counts + fit/freeze, spatial v2, evaluation utilities,
the generic connector, and the registry); each test runs the same code
the doc shows and asserts the values the doc states. If the library
changes, the doc must change with it — or this fails.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from eventflow.core.context import EnricherStep, TabularFileSource, TemporalJoin
from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import Pipeline
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps import (
    BurstinessStep,
    CategoryCountsStep,
    CompletePanelStep,
    EWMAMeanStep,
    FillNullStep,
    TimeSinceLastEventStep,
    TrendStep,
    ZeroFeaturesStep,
)
from eventflow.core.steps.density import KernelDensityStep, SpaceTimeKernelDensityStep
from eventflow.core.steps.hawkes import HawkesExcitationStep
from eventflow.core.steps.near_repeat import NearRepeatStep
from eventflow.core.steps.spatial import MultiScaleSpatialLagStep, SpatialLagStep
from eventflow.core.steps.temporal import MovingAverageStep, TemporalLagStep, TimeBinsStep
from eventflow.evaluation import (
    blocked_split,
    evaluate_walk_forward,
    pai_score,
    pr_auc_score,
    temporal_split,
    validate_split,
    walk_forward_splits,
)
from tests.evaluation.synthetic import generate_synthetic_panel, panel_event_frame

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Shared fixtures: a 3x3 completed panel and a sparse panel
# ---------------------------------------------------------------------------

BIN0 = datetime(2024, 1, 1)
SPACING_M = 300.0


def _bins(n: int = 4) -> list[datetime]:
    return [BIN0 + timedelta(days=i) for i in range(n)]


def make_panel(counts_by_unit: dict[int, list[float]]) -> EventFrame:
    """Completed 3x3 panel: every (cell, bin) present, centres in metres.

    ``counts_by_unit`` maps grid_id (row * 3 + col) to one count per bin.
    Schema lon/lat columns are the projected-metre centres ``x``/``y``
    (EPSG:26971), so the KDE steps can use them directly.
    """
    rows: dict[str, list] = {
        "grid_id": [],
        "time_bin": [],
        "grid_row": [],
        "grid_col": [],
        "x": [],
        "y": [],
        "count": [],
    }
    for bin_idx, bin_start in enumerate(_bins()):
        for grid_id in range(9):
            rows["grid_id"].append(grid_id)
            rows["time_bin"].append(bin_start)
            rows["grid_row"].append(grid_id // 3)
            rows["grid_col"].append(grid_id % 3)
            rows["x"].append(SPACING_M * (grid_id % 3))
            rows["y"].append(SPACING_M * (grid_id // 3))
            rows["count"].append(counts_by_unit[grid_id][bin_idx])
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col="y",
        lon_col="x",
        numeric_cols=["count"],
        categorical_cols=["grid_id"],
    )
    metadata = EventMetadata(dataset_name="docs-panel", crs="EPSG:26971", time_bin="1d")
    return EventFrame(pl.LazyFrame(rows), schema, metadata)


def make_sparse_panel() -> EventFrame:
    """Sparse 3-unit x 4-bin panel for the CompletePanelStep example."""
    rows = [
        (1, BIN0, 5),
        (1, BIN0 + timedelta(days=2), 7),
        (2, BIN0 + timedelta(days=1), 3),
        (3, BIN0, 1),
        (3, BIN0 + timedelta(days=1), 2),
        (3, BIN0 + timedelta(days=2), 3),
        (3, BIN0 + timedelta(days=3), 4),
    ]
    data = {
        "grid_id": [r[0] for r in rows],
        "time_bin": [r[1] for r in rows],
        "count": [r[2] for r in rows],
        # A measurement column: observed nulls must stay null (never 0).
        "pm25": [20.0, None, 15.0, 12.0, 11.0, 10.0, 9.0],
        "latitude": [41.8781] * len(rows),
        "longitude": [-87.6298] * len(rows),
    }
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["count", "pm25"],
    )
    metadata = EventMetadata(dataset_name="docs-sparse", crs="EPSG:4326", time_bin="1d")
    return EventFrame(pl.LazyFrame(data), schema, metadata)


def make_events() -> EventFrame:
    """Raw event rows for the CategoryCountsStep example."""
    rows = [
        (0, BIN0, "THEFT"),
        (0, BIN0, "THEFT"),
        (0, BIN0, "BATTERY"),
        (0, BIN0 + timedelta(days=1), "THEFT"),
        (0, BIN0 + timedelta(days=1), "ASSAULT"),
        (1, BIN0, "BATTERY"),
        (1, BIN0, "BATTERY"),
        (1, BIN0 + timedelta(days=1), None),
    ]
    data = {
        "grid_id": [r[0] for r in rows],
        "time_bin": [r[1] for r in rows],
        "primary_type": [r[2] for r in rows],
        "latitude": [41.8781] * len(rows),
        "longitude": [-87.6298] * len(rows),
    }
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=[],
        categorical_cols=["primary_type"],
    )
    metadata = EventMetadata(dataset_name="docs-events", crs="EPSG:4326", time_bin="1d")
    return EventFrame(pl.LazyFrame(data), schema, metadata)


# ---------------------------------------------------------------------------
# Panel completion
# ---------------------------------------------------------------------------


def test_docs_complete_panel() -> None:
    """api_reference: CompletePanelStep — 7 observed rows -> 12, 5 zeros."""
    frame = make_sparse_panel()
    result = CompletePanelStep().run(frame).collect()

    assert result.height == 12  # 3 units x 4 bins
    assert result["count"].sum() == 25  # observed counts unchanged
    # Every row is in-support, so every row carries "observed".
    assert result["observation_status"].unique().to_list() == ["observed"]
    assert result.filter(pl.col("count") == 0).height == 5
    # Measurements are never zero-filled: 1 observed null + 5 generated
    # pairs = 6 nulls, and the observed null stays null.
    assert result["pm25"].null_count() == 6


def test_docs_complete_panel_measure_cols() -> None:
    """api_reference: measure_cols keep missing-stays-missing."""
    frame = make_sparse_panel()
    result = CompletePanelStep(measure_cols=["pm25"]).run(frame).collect()
    assert result["pm25"].null_count() == 6
    observed = result.filter(
        (pl.col("grid_id") == 1) & (pl.col("time_bin") == BIN0 + timedelta(days=2))
    )
    assert observed["pm25"].item() is None  # observed null stays null
    assert observed["count"].item() == 7


# ---------------------------------------------------------------------------
# Dynamics steps
# ---------------------------------------------------------------------------

# One unit's series, shared by the dynamics examples: [0, 0, 5, 0].
COUNTS_ZERO_FEATURES = {u: [1.0, 1.0, 1.0, 1.0] for u in range(9)}
COUNTS_ZERO_FEATURES[4] = [0.0, 0.0, 5.0, 0.0]


def test_docs_ewma_mean() -> None:
    """api_reference: EWMAMeanStep — first bin equals the count itself."""
    frame = make_panel({u: [1.0, 2.0, 3.0, 4.0] for u in range(9)})
    result = (
        EWMAMeanStep(value_cols=["count"], halflife_days=7.0, group_cols=["grid_id"])
        .run(frame)
        .collect()
    )
    first = result.filter(pl.col("time_bin") == BIN0)
    assert first["count_ewma_7d"].to_list() == [1.0] * 9
    assert result["count_ewma_7d"].null_count() == 0


def test_docs_trend_step() -> None:
    """api_reference: TrendStep — ratio of two rolling means, null on 0."""
    frame = make_panel({u: [1.0, 2.0, 3.0, 4.0] for u in range(9)})
    frame = MovingAverageStep(
        value_cols=["count"],
        windows=[1, 2],
        time_unit="1d",
        group_cols=["grid_id"],
        closed="left",
    ).run(frame)
    result = TrendStep(recent_col="count_ma_1", long_col="count_ma_2").run(frame).collect()
    # Bin 3 (4th): ma_1 = count at bin 2 = 3.0; ma_2 = mean(2, 3) = 2.5.
    row = result.filter((pl.col("grid_id") == 0) & (pl.col("time_bin") == BIN0 + timedelta(days=3)))
    assert row["count_ma_1_vs_count_ma_2"].item() == pytest.approx(3.0 / 2.5)
    # Bin 0: both means are null (no prior bins) -> ratio null, never inf.
    first = result.filter(pl.col("time_bin") == BIN0)
    assert first["count_ma_1_vs_count_ma_2"].null_count() == 9


def test_docs_time_since_last_event() -> None:
    """api_reference: TimeSinceLastEventStep — 0.0 on the event bin itself."""
    frame = make_panel(COUNTS_ZERO_FEATURES)
    result = TimeSinceLastEventStep(value_col="count", group_cols=["grid_id"]).run(frame).collect()
    unit4 = result.filter(pl.col("grid_id") == 4).sort("time_bin")
    assert unit4["count_time_since_last_d"].to_list() == [None, None, 0.0, 1.0]


def test_docs_burstiness() -> None:
    """api_reference: BurstinessStep — null while the trailing mean is 0."""
    frame = make_panel(COUNTS_ZERO_FEATURES)
    result = (
        BurstinessStep(value_col="count", window="28d", group_cols=["grid_id"]).run(frame).collect()
    )
    # First bin: no prior bins -> null. Centre unit after the event bin:
    # trailing mean > 0, so the coefficient of variation is defined.
    first = result.filter(pl.col("time_bin") == BIN0)
    assert first["count_burstiness_28d"].null_count() == 9
    last = result.filter(
        (pl.col("grid_id") == 4) & (pl.col("time_bin") == BIN0 + timedelta(days=3))
    )
    assert last["count_burstiness_28d"].item() is not None


def test_docs_zero_features() -> None:
    """api_reference: ZeroFeaturesStep — active flag + nonzero share."""
    frame = make_panel(COUNTS_ZERO_FEATURES)
    result = ZeroFeaturesStep(value_col="count", group_cols=["grid_id"]).run(frame).collect()
    unit4 = result.filter(pl.col("grid_id") == 4).sort("time_bin")
    # Active flag (7 d, includes the current bin): only the event bin.
    assert unit4["count_active_7d"].to_list() == [False, False, True, True]
    # Share of nonzero bins over 28 d (includes the current bin): 1 of 4
    # bins at the last bin.
    assert unit4["count_share_nonzero_28d"].to_list()[-1] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Categorical counts + fit/freeze
# ---------------------------------------------------------------------------


def test_docs_category_counts() -> None:
    """api_reference: CategoryCountsStep — top-2 + other, shares, entropy."""
    result = (
        CategoryCountsStep(category_cols=["primary_type"], top_n=2).run(make_events()).collect()
    )
    # THEFT (3) and BATTERY (3) are the top 2; ASSAULT + null go to other.
    assert "count_primary_type_THEFT" in result.columns
    assert "count_primary_type_BATTERY" in result.columns
    assert "count_primary_type_other" in result.columns
    assert "share_primary_type_THEFT" in result.columns
    assert "primary_type_entropy" in result.columns
    assert "count_primary_type_ASSAULT" not in result.columns
    row = result.filter((pl.col("grid_id") == 0) & (pl.col("time_bin") == BIN0)).row(0, named=True)
    assert row["count_primary_type_THEFT"] == 2
    assert row["count_primary_type_BATTERY"] == 1
    assert row["count_primary_type_other"] == 0
    assert row["share_primary_type_THEFT"] == pytest.approx(2 / 3)


def test_docs_category_counts_fit_freeze() -> None:
    """api_reference: fit() freezes the vocabulary; new values -> other."""
    train = make_events()
    step = CategoryCountsStep(category_cols=["primary_type"], top_n=2).fit(train)
    assert step.is_fitted
    assert [value for value, _ in step.fitted_values("primary_type")] == [
        "BATTERY",
        "THEFT",
    ]
    # A later frame where ROBBERY dominates: unfitted it would take a
    # top slot; fitted it lands in "other".
    test_rows = {
        "grid_id": [0] * 6,
        "time_bin": [BIN0] * 6,
        "primary_type": ["ROBBERY"] * 5 + ["THEFT"],
        "latitude": [41.8781] * 6,
        "longitude": [-87.6298] * 6,
    }
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=[],
        categorical_cols=["primary_type"],
    )
    metadata = EventMetadata(dataset_name="docs-events", crs="EPSG:4326", time_bin="1d")
    test_frame = EventFrame(pl.LazyFrame(test_rows), schema, metadata)
    result = step.run(test_frame).collect()
    assert "count_primary_type_ROBBERY" not in result.columns
    # Every event row carries its group's aggregates: 5 ROBBERY in other.
    assert result["count_primary_type_other"].unique().to_list() == [5]
    assert result["count_primary_type_THEFT"].unique().to_list() == [1]


# ---------------------------------------------------------------------------
# Spatial v2
# ---------------------------------------------------------------------------

# Counts concentrated on the centre cell (grid_id 4) in bin 0, 1 elsewhere.
COUNTS_CENTRE = {u: [1.0, 1.0, 1.0, 1.0] for u in range(9)}
COUNTS_CENTRE[4] = [10.0, 1.0, 1.0, 1.0]


def test_docs_spatial_lag_multiscale() -> None:
    """api_reference: ring 1 reproduces SpatialLagStep; ring 2 is null here.

    On a 3x3 grid the centre cell has no cell at Chebyshev distance 2,
    so its ring-2 lag is null (island in that ring), not 0.
    """
    frame = make_panel(COUNTS_CENTRE)
    multi = (
        MultiScaleSpatialLagStep(value_cols=["count"], rings=(1, 2), group_cols=["time_bin"])
        .run(frame)
        .collect()
    )
    single = SpatialLagStep(value_cols=["count"], group_cols=["time_bin"]).run(frame).collect()
    assert multi["count_spatial_lag_r1"].to_list() == single["count_spatial_lag"].to_list()
    centre_bin0 = multi.filter((pl.col("grid_id") == 4) & (pl.col("time_bin") == BIN0)).row(
        0, named=True
    )
    assert centre_bin0["count_spatial_lag_r1"] == pytest.approx(1.0)  # 8 neighbours at 1
    assert centre_bin0["count_spatial_lag_r2"] is None


def test_docs_kernel_density() -> None:
    """api_reference: KernelDensityStep — centre density is the maximum."""
    frame = make_panel(COUNTS_CENTRE)
    result = (
        KernelDensityStep(value_col="count", bandwidth_m=600.0, group_cols=["time_bin"])
        .run(frame)
        .collect()
    )
    bin0 = result.filter(pl.col("time_bin") == BIN0)
    assert bin0["count_kde"].null_count() == 0
    assert (bin0["count_kde"] > 0).all()
    centre = bin0.filter(pl.col("grid_id") == 4)["count_kde"].item()
    corner = bin0.filter(pl.col("grid_id") == 0)["count_kde"].item()
    assert centre > corner


def test_docs_space_time_kde() -> None:
    """api_reference: SpaceTimeKernelDensityStep — backward-only smoothing."""
    frame = make_panel(COUNTS_CENTRE)
    result = (
        SpaceTimeKernelDensityStep(value_col="count", bandwidth_m=600.0, halflife_days=7.0)
        .run(frame)
        .collect()
    )
    assert result["count_stkde"].null_count() == 0
    assert (result["count_stkde"] > 0).all()
    # The centre cell's STKDE accumulates the bin-0 spike: bin 1 > bin 0
    # (the discounted past adds to the current bin's own density).
    centre = result.filter(pl.col("grid_id") == 4).sort("time_bin")
    assert centre["count_stkde"][1] > centre["count_stkde"][0]


def test_docs_near_repeat() -> None:
    """api_reference: NearRepeatStep — 0 before any past bin, flag 0/1."""
    frame = make_panel(COUNTS_CENTRE)
    result = NearRepeatStep(value_col="count", rings=(1,)).run(frame).collect()
    assert "count_nr_r1_7d" in result.columns
    assert "count_nr_r1_7_14d" in result.columns
    assert "count_nr_event_7d" in result.columns
    # Bin 0: strictly-past windows are empty -> 0, flag 0.
    first = result.filter(pl.col("time_bin") == BIN0)
    assert first["count_nr_r1_7d"].to_list() == [0] * 9
    assert first["count_nr_event_7d"].to_list() == [0] * 9
    # Centre cell, bin 1: ring 1 held 8 neighbours x 1 event in bin 0.
    row = result.filter(
        (pl.col("grid_id") == 4) & (pl.col("time_bin") == BIN0 + timedelta(days=1))
    ).row(0, named=True)
    assert row["count_nr_r1_7d"] == 8
    assert row["count_nr_event_7d"] == 1


def test_docs_hawkes_excitation() -> None:
    """api_reference: HawkesExcitationStep — first bin is 0, strictly past."""
    frame = make_panel(COUNTS_ZERO_FEATURES)
    result = (
        HawkesExcitationStep(value_col="count", halflife_days=7.0, group_cols=["grid_id"])
        .run(frame)
        .collect()
    )
    first = result.filter(pl.col("time_bin") == BIN0)
    assert first["count_hawkes_7d"].to_list() == [0.0] * 9
    assert (result["count_hawkes_7d"] >= 0).all()
    # The centre unit's excitation is 0 until after its bin-2 event.
    unit4 = result.filter(pl.col("grid_id") == 4).sort("time_bin")
    assert unit4["count_hawkes_7d"].to_list()[:3] == [0.0, 0.0, 0.0]
    assert unit4["count_hawkes_7d"].to_list()[3] > 0


def test_docs_fill_null() -> None:
    """api_reference: FillNullStep — a lag before the panel start is 0."""
    frame = make_panel({u: [1.0, 2.0, 3.0, 4.0] for u in range(9)})
    frame = TemporalLagStep(
        value_cols=["count"], lag_periods=[1], time_unit="1d", group_cols=["grid_id"]
    ).run(frame)
    filled = FillNullStep(value_cols=["count_lag_1"], fill_value=0).run(frame).collect()
    assert filled["count_lag_1"].null_count() == 0
    first = filled.filter(pl.col("time_bin") == BIN0)
    assert first["count_lag_1"].to_list() == [0] * 9


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def _eval_frame() -> pl.LazyFrame:
    """20 daily bins x 3 units, counts 0/1/2 — for the split examples."""
    rows: dict[str, list] = {"grid_id": [], "time_bin": [], "count": []}
    for day in range(20):
        for unit in range(3):
            rows["grid_id"].append(unit)
            rows["time_bin"].append(BIN0 + timedelta(days=day))
            rows["count"].append((day + unit) % 3)
    return pl.LazyFrame(rows)


def test_docs_temporal_split() -> None:
    """api_reference: temporal_split + validate_split; leak is rejected."""
    frame = _eval_frame()
    train, test = temporal_split(
        frame, "time_bin", BIN0 + timedelta(days=10), gap=timedelta(days=1)
    )
    validate_split(train, test, time_col="time_bin")
    assert train.collect().height == 30  # 10 days x 3 units
    assert test.collect().height == 27  # 9 days x 3 units (1 embargo day)
    with pytest.raises(ValueError, match="share"):
        validate_split(train, train, time_col="time_bin")


def test_docs_blocked_split() -> None:
    """api_reference: blocked_split — time x space, off-diagonals dropped."""
    frame = _eval_frame()
    train, test = blocked_split(
        frame,
        "time_bin",
        BIN0 + timedelta(days=10),
        unit_col="grid_id",
        test_units=[2],
    )
    validate_split(train, test, time_col="time_bin", unit_col="grid_id")
    assert test.collect()["grid_id"].unique().to_list() == [2]
    assert train.collect()["grid_id"].unique().to_list() == [0, 1]


def test_docs_walk_forward() -> None:
    """api_reference: walk_forward_splits — expanding window, 2 folds."""
    frame = panel_event_frame(
        generate_synthetic_panel(n_grid_rows=2, n_grid_cols=2, n_bins=20).panel
    )
    folds = list(
        walk_forward_splits(
            frame,
            "time_bin",
            initial_train=timedelta(days=10),
            step=timedelta(days=5),
        )
    )
    assert len(folds) == 2
    assert folds[0].train.collect().height < folds[1].train.collect().height
    assert folds[0].test.collect().height == 20  # 4 units x 5 days
    assert folds[1].test.collect().height == 20


def test_docs_evaluate_walk_forward() -> None:
    """api_reference: evaluate_walk_forward refits per fold and scores."""

    def score_fn(train_out: EventFrame, test_out: EventFrame) -> dict[str, float]:
        test = test_out.collect()
        y_true = (test["count"] > 0).cast(pl.Int64).to_numpy()
        y_score = test["count_lag_1"].fill_null(0).to_numpy()
        return {"pai_20pct": pai_score(y_true, y_score, area_fraction=0.2)}

    data = generate_synthetic_panel(n_grid_rows=2, n_grid_cols=2, n_bins=20)
    frame = panel_event_frame(data.panel)
    results = evaluate_walk_forward(
        [
            TemporalLagStep(
                value_cols=["count"],
                lag_periods=[1],
                time_unit="1d",
                group_cols=["grid_id"],
            )
        ],
        frame,
        "time_bin",
        initial_train=timedelta(days=10),
        step=timedelta(days=5),
        score_fn=score_fn,
    )
    assert len(results) == 2
    assert all(math.isfinite(r.scores["pai_20pct"]) for r in results)
    assert results[0].fold == 0 and results[1].fold == 1


def test_docs_pai_and_pr_auc() -> None:
    """api_reference: PAI = 4.0 on the worked example; perfect PR-AUC = 1."""
    assert pai_score([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3], area_fraction=0.25) == 4.0
    assert pr_auc_score([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no positive"):
        pai_score([0, 0, 0], [0.1, 0.2, 0.3], area_fraction=0.5)


def test_docs_synthetic_ground_truth() -> None:
    """api_reference: planted lag-1 phi = 0.6 is recovered in [0.25, 0.75].

    The design note (docs/design-notes/phase4-synthetic-ground-truth.md)
    records the measured 0.529 on the default 6x6 x 56-bin panel; the
    test pins the documented tolerance, not the point value.
    """
    data = generate_synthetic_panel()
    frame = panel_event_frame(data.panel)
    lagged = (
        TemporalLagStep(
            value_cols=["count"], lag_periods=[1], time_unit="1d", group_cols=["grid_id"]
        )
        .run(frame)
        .collect()
    )
    paired = lagged.drop_nulls(["count_lag_1"])
    corr = paired.select(pl.corr("count", "count_lag_1")).item()
    assert corr == pytest.approx(0.529, abs=0.01)  # measured on the default panel
    assert 0.25 <= corr <= 0.75
    top4 = (
        lagged.group_by("grid_id")
        .agg(pl.col("count").mean().alias("mean_count"))
        .sort("mean_count", descending=True)
        .head(4)["grid_id"]
        .to_list()
    )
    assert set(top4) == set(data.params.hotspot_units)


# ---------------------------------------------------------------------------
# Generic connector
# ---------------------------------------------------------------------------


def test_docs_tabular_file_source(tmp_path: Path) -> None:
    """api_reference: TabularFileSource + EnricherStep, end to end."""
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "ts,temp_c\n" "2024-01-01 00:00:00,10.0\n" "2024-01-01 01:00:00,11.0\n",
        encoding="utf-8",
    )
    source = TabularFileSource(csv_path, timestamp_col="ts", value_cols=["temp_c"])
    loaded = source.load().collect()
    assert loaded.columns == ["ts", "temp_c"]
    assert loaded["temp_c"].to_list() == [10.0, 11.0]

    events = EventFrame(
        pl.DataFrame(
            {
                "time": [datetime(2024, 1, 1, 0, 30), datetime(2024, 1, 1, 1, 30)],
                "x": [0.5, 0.5],
                "y": [0.5, 0.5],
            }
        ).lazy(),
        EventSchema(timestamp_col="time", lon_col="x", lat_col="y"),
        EventMetadata(dataset_name="docs-context", crs="EPSG:3857"),
    )
    enriched = EnricherStep(source, TemporalJoin(strategy="backward")).run(events).collect()
    assert enriched["temp_c"].to_list() == [10.0, 11.0]


# ---------------------------------------------------------------------------
# Registry + doc guards
# ---------------------------------------------------------------------------


def test_docs_registry_names() -> None:
    """api_reference: the documented registry names resolve and build."""
    from eventflow.core.steps.registration import get_default_registry

    registry = get_default_registry()
    for name in (
        "complete_panel",
        "category_counts",
        "ewma_mean",
        "multiscale_spatial_lag",
        "kernel_density",
        "space_time_kde",
        "near_repeat",
        "hawkes_excitation",
        "fill_null",
    ):
        assert registry.get(name) is not None, f"registry name missing: {name}"
    pipeline = registry.build_pipeline(
        [
            {"name": "complete_panel", "params": {}},
            {"name": "fill_null", "params": {"value_cols": ["count_lag_1"]}},
        ]
    )
    assert [type(s).__name__ for s in pipeline.steps] == [
        "CompletePanelStep",
        "FillNullStep",
    ]


def test_docs_phase2_steps_import_paths() -> None:
    """api_reference: phase-2 steps import from their own modules.

    They are registered (see test_docs_registry_names) but NOT re-exported
    from ``eventflow.core.steps`` — the doc shows the real import path.
    """
    import eventflow.core.steps as steps

    for name in (
        "MultiScaleSpatialLagStep",
        "KernelDensityStep",
        "SpaceTimeKernelDensityStep",
        "NearRepeatStep",
        "HawkesExcitationStep",
    ):
        assert not hasattr(steps, name), f"{name} unexpectedly exported"
    # And the documented paths do import.
    assert MultiScaleSpatialLagStep is not None
    assert KernelDensityStep is not None
    assert SpaceTimeKernelDensityStep is not None
    assert NearRepeatStep is not None
    assert HawkesExcitationStep is not None


def test_docs_api_reference_examples_present() -> None:
    """Every example added to api_reference.md must be present in the doc."""
    text = (REPO_ROOT / "docs" / "api_reference.md").read_text(encoding="utf-8")
    for snippet in (
        "CompletePanelStep(",
        "EWMAMeanStep(",
        "TrendStep(",
        "TimeSinceLastEventStep(",
        "BurstinessStep(",
        "ZeroFeaturesStep(",
        "CategoryCountsStep(",
        ".fit(",
        "MultiScaleSpatialLagStep(",
        "KernelDensityStep(",
        "SpaceTimeKernelDensityStep(",
        "NearRepeatStep(",
        "HawkesExcitationStep(",
        "FillNullStep(",
        "temporal_split(",
        "blocked_split(",
        "walk_forward_splits(",
        "evaluate_walk_forward(",
        "pai_score(",
        "pr_auc_score(",
        "TabularFileSource(",
        "generate_synthetic_panel(",
    ):
        assert snippet in text, f"documented example missing from api_reference.md: {snippet}"
    for stale in (
        "SpatialGridStep",
        "TemporalBinStep",
        "with_spatial_grid",
        "with_time_bins",
        "https://github.com/eventflow/eventflow",
    ):
        assert stale not in text, f"stale text still in api_reference.md: {stale}"


def test_docs_pipeline_runs_end_to_end() -> None:
    """api_reference: the worked pipeline composes the documented steps."""
    frame = make_panel(COUNTS_ZERO_FEATURES)
    pipeline = Pipeline(
        [
            TemporalLagStep(
                value_cols=["count"],
                lag_periods=[1],
                time_unit="1d",
                group_cols=["grid_id"],
            ),
            FillNullStep(value_cols=["count_lag_1"], fill_value=0),
            EWMAMeanStep(value_cols=["count"], halflife_days=7.0, group_cols=["grid_id"]),
            TimeSinceLastEventStep(value_col="count", group_cols=["grid_id"]),
            ZeroFeaturesStep(value_col="count", group_cols=["grid_id"]),
        ]
    )
    result = pipeline.run(frame).collect()
    for col in (
        "count_lag_1",
        "count_ewma_7d",
        "count_time_since_last_d",
        "count_active_7d",
        "count_share_nonzero_28d",
    ):
        assert col in result.columns


def test_docs_datasets_template_recipe() -> None:
    """docs/datasets.md §5: the recipe template, executed as written."""
    from eventflow.core.schema import RecipeConfig
    from eventflow.recipes.base import BaseRecipe

    class YourDatasetV1Recipe(BaseRecipe):
        def build_pipeline(self) -> Pipeline:
            return Pipeline(
                [
                    TimeBinsStep(bin_size=self.config.temporal.time_bin),
                    TemporalLagStep(value_cols=["count"], lag_periods=[1], time_unit="1d"),
                ]
            )

    config = RecipeConfig(dataset="your_dataset", recipe="v1")
    recipe = YourDatasetV1Recipe(config)
    frame = EventFrame(
        pl.LazyFrame(
            {
                "timestamp": [BIN0 + timedelta(days=i) for i in range(5)],
                "count": [1, 2, 3, 4, 5],
                "latitude": [41.8781] * 5,
                "longitude": [-87.6298] * 5,
            }
        ),
        EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
            numeric_cols=["count"],
        ),
        EventMetadata(dataset_name="docs-template", crs="EPSG:4326", time_bin="1d"),
    )
    result = recipe.run(frame).collect()
    assert "time_bin" in result.columns
    assert result["count_lag_1"].to_list() == [None, 1, 2, 3, 4]


def test_docs_datasets_no_phantom_recipes() -> None:
    """docs/datasets.md must name only recipes that exist."""
    text = (REPO_ROOT / "docs" / "datasets.md").read_text(encoding="utf-8")
    assert "chicago_crime_with_weather" not in text
    assert "chicago_crime_full" not in text
    assert "SpatialGridStep" not in text
    assert "TemporalBinStep" not in text
    assert "FeatureAggregationStep" not in text
    for name in (
        "beijing_air_quality_v1",
        "geolife_v1",
        "load_beijing_air_quality",
        "load_geolife",
        "TabularFileSource",
    ):
        assert name in text, f"datasets.md missing: {name}"


def test_docs_architecture_pipeline(tmp_path: Path) -> None:
    """docs/architecture.md: the Pipeline example, executed as written."""
    from eventflow.core.steps import AssignToGridStep, GridAggregationStep
    from eventflow.core.steps.temporal import TimeBinsStep

    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "ts,temp_c\n2024-01-01 00:00:00,10.0\n2024-01-01 01:00:00,11.0\n",
        encoding="utf-8",
    )
    weather_source = TabularFileSource(csv_path, timestamp_col="ts", value_cols=["temp_c"])
    event_frame = EventFrame(
        pl.LazyFrame(
            {
                "timestamp": [datetime(2024, 1, 1, h) for h in range(6)],
                "x": [100.0, 400.0, 700.0, 150.0, 450.0, 750.0],
                "y": [100.0, 100.0, 100.0, 400.0, 400.0, 400.0],
            }
        ),
        EventSchema(timestamp_col="timestamp", lon_col="x", lat_col="y"),
        EventMetadata(dataset_name="docs-arch", crs="EPSG:26971"),
    )
    pipeline = Pipeline(
        [
            AssignToGridStep(grid_size_m=300),
            TimeBinsStep(bin_size="6h"),
            GridAggregationStep(grid_col="grid_id"),
            EnricherStep(weather_source, TemporalJoin(strategy="nearest")),
        ]
    )
    result = pipeline.run(event_frame).collect()
    assert "grid_id" in result.columns
    assert "time_bin" in result.columns
    assert "event_count" in result.columns
    assert result["event_count"].sum() == 6
    # Nearest join: 00:00 -> 10.0, 01:00-05:00 -> 11.0 (nearest to 01:00).
    assert result["temp_c"].to_list() == [10.0, 11.0, 11.0, 11.0, 11.0, 11.0]


def test_docs_architecture_no_stale_step_names() -> None:
    """docs/architecture.md must not name steps that do not exist."""
    text = (REPO_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for stale in (
        "SpatialGridStep",
        "TemporalBinStep",
        "FeatureAggregationStep",
        "ContextEnricherStep",
    ):
        assert stale not in text, f"stale step name in architecture.md: {stale}"
    for current in ("AssignToGridStep", "GridAggregationStep", "EnricherStep"):
        assert current in text, f"architecture.md missing: {current}"


def _github_slug(heading: str) -> str:
    """GitHub-style anchor slug for a Markdown heading."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s", "-", slug)  # one hyphen per space, no collapsing


def test_docs_internal_links_resolve() -> None:
    """Every internal link in the touched docs resolves to a real file/section."""
    docs = [
        "docs/api_reference.md",
        "README.md",
        "GETTING_STARTED.md",
        "docs/datasets.md",
        "docs/architecture.md",
        "notebooks/README.md",
    ]
    link_re = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    for rel in docs:
        doc_path = REPO_ROOT / rel
        text = doc_path.read_text(encoding="utf-8")
        for match in link_re.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, anchor = target.partition("#")
            target_path = doc_path.parent / path_part if path_part else doc_path
            assert target_path.is_file(), f"{rel}: broken link to {target}"
            if anchor:
                headings = [
                    _github_slug(line.lstrip("#").strip())
                    for line in target_path.read_text(encoding="utf-8").splitlines()
                    if line.startswith("#")
                ]
                assert anchor in headings, f"{rel}: broken anchor #{anchor} in {target}"


def test_docs_api_reference_adapter_aliases() -> None:
    """api_reference.md adapter example: aliases are the identical arrays.

    Same code as the "Output Adapters" section; also guards the section
    itself (worker A section was lost once in a rewrite — this fails if
    the alias names disappear from the doc again).
    """
    from eventflow.core.adapters import (
        SequenceAdapter,
        SequenceAdapterConfig,
        StreamAdapter,
        StreamAdapterConfig,
    )

    text = (REPO_ROOT / "docs" / "api_reference.md").read_text(encoding="utf-8")
    for name in ("attention_mask", "inter_event_times", "state_features"):
        assert name in text, f"api_reference.md missing adapter alias: {name}"

    frame = EventFrame(
        pl.LazyFrame(
            {
                "time": [datetime(2024, 1, 1) + timedelta(days=d) for d in range(3)] * 2,
                "grid_id": ["0_0"] * 3 + ["0_1"] * 3,
                "x": [0.0] * 6,
                "y": [0.0] * 6,
                "count": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        ),
        EventSchema(timestamp_col="time", lon_col="x", lat_col="y"),
        EventMetadata(dataset_name="docs-adapters", crs="EPSG:3857"),
    )
    seq = SequenceAdapter(
        SequenceAdapterConfig(spatial_col="grid_id", feature_cols=["count"])
    ).convert(frame)
    assert seq.attention_mask is seq.masks
    assert seq.data is seq.sequences
    stream = StreamAdapter(StreamAdapterConfig(state_cols=["count"])).convert(frame)
    assert stream.inter_event_times is stream.inter_times
    assert stream.state_features is stream.states
    assert stream.data is stream.states
