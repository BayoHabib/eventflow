"""Small numerical counterexamples for the spatio-temporal preparation contract."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from shapely.geometry import box

from eventflow.core import EventFrame, EventMetadata, EventSchema
from eventflow.core._time import duration_seconds
from eventflow.core.context.joiners import TemporalJoin
from eventflow.core.pipeline import Pipeline
from eventflow.core.schema import ContextSchema
from eventflow.core.steps.context import (
    SpatialContextJoinStep,
    SpatioTemporalContextJoinStep,
    TemporalContextJoinStep,
)
from eventflow.core.steps.point_process import HawkesKernelStep
from eventflow.core.steps.spatial import (
    GetisOrdStep,
    LocalMoranStep,
    SpatialLagStep,
    TransformCRSStep,
)
from eventflow.core.steps.temporal import MovingAverageStep, TemporalLagStep
from eventflow.core.temporal import align_temporal
from eventflow.core.weights import SpatialWeights


def frame(data: dict, crs: str = "EPSG:3857") -> EventFrame:
    n = len(next(iter(data.values())))
    return EventFrame(
        pl.DataFrame(
            {
                "time": [datetime(2024, 1, 1)] * n,
                "x": [0.5] * n,
                "y": [0.5] * n,
                "value": [1.0] * n,
                **data,
            }
        ).lazy(),
        EventSchema(timestamp_col="time", lon_col="x", lat_col="y", numeric_cols=["value"]),
        EventMetadata(dataset_name="generic-observations", crs=crs),
    )


def test_duration_lag_keeps_gaps_unknown_and_rows_mode_is_explicit() -> None:
    ef = frame({"time": [datetime(2024, 1, 1), datetime(2024, 1, 3)], "value": [10.0, 20.0]})
    assert TemporalLagStep(["value"], [1]).run(ef).collect()["value_lag_1"].to_list() == [
        None,
        None,
    ]
    assert TemporalLagStep(["value"], [2]).run(ef).collect()["value_lag_2"].to_list() == [
        None,
        10.0,
    ]
    assert TemporalLagStep(["value"], [1], mode="rows").run(ef).collect()[
        "value_lag_1"
    ].to_list() == [None, 10.0]


def test_duration_lag_never_crosses_entities() -> None:
    ef = frame(
        {
            "time": [datetime(2024, 1, 2), datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "entity": ["A", "A", "B"],
            "value": [20.0, 10.0, 100.0],
        }
    )
    result = TemporalLagStep(["value"], [1], group_cols=["entity"]).run(ef).collect()
    assert result["value_lag_1"].to_list() == [None, 10.0, None]


@pytest.mark.parametrize("step", [TemporalLagStep(["value"]), MovingAverageStep(["value"])])
def test_duration_features_reject_ambiguous_duplicate_times(step) -> None:
    with pytest.raises(ValueError, match="Duplicate temporal keys"):
        step.run(frame({"value": [1.0, 2.0]}))


def test_fixed_day_is_24_elapsed_hours_across_dst() -> None:
    tz = ZoneInfo("America/Toronto")
    ef = frame(
        {"time": [datetime(2024, 3, 9, 12, tzinfo=tz), datetime(2024, 3, 10, 12, tzinfo=tz)]}
    )
    assert TemporalLagStep(["value"]).run(ef).collect()["value_lag_1"].to_list() == [None, None]


def test_trailing_mean_uses_duration_and_excludes_current_value() -> None:
    ef = frame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 10)],
            "value": [10.0, 20.0, 1000.0],
        }
    )
    result = MovingAverageStep(["value"], [2]).run(ef).collect()
    assert result["value_ma_2"].to_list() == [None, 10.0, None]
    inclusive = MovingAverageStep(["value"], [2], closed="right").run(ef).collect()
    assert inclusive["value_ma_2"].to_list() == [10.0, 15.0, 1000.0]


@pytest.mark.parametrize(
    "unit,expected", [("6h", 21600), ("15min", 900), ("2d", 172800), ("ms", 0.001)]
)
def test_duration_parser_honors_multiplier(unit: str, expected: float) -> None:
    assert duration_seconds(unit) == expected


@pytest.mark.parametrize("unit", ["0h", "unknown", "1mo", "-1d"])
def test_unsupported_time_units_never_silently_fallback(unit: str) -> None:
    with pytest.raises(ValueError):
        duration_seconds(unit)


def test_grouped_hawkes_has_strict_independent_history() -> None:
    t = datetime(2024, 1, 1)
    ef = frame(
        {
            "time": [t, t + timedelta(hours=1), t, t, t + timedelta(hours=1)],
            "entity": ["A", "A", "B", "B", "B"],
        }
    )
    result = HawkesKernelStep(group_cols=["entity"]).run(ef).collect()
    assert result["hawkes_intensity"].to_list() == pytest.approx(
        [0.1, 0.1 + 0.5 / math.e, 0.1, 0.1, 0.1 + 1 / math.e]
    )


def grid_frame() -> EventFrame:
    # ID order deliberately has no relationship to geographic order.
    return frame(
        {
            "grid_id": ["z", "a", "q", "b"],
            "grid_row": [0, 0, 1, 9],
            "grid_col": [0, 1, 1, 9],
            "value": [10.0, 20.0, 30.0, 999.0],
        }
    )


def test_rook_and_queen_use_real_adjacency_without_self_or_isolated_cells() -> None:
    ef = grid_frame()
    rook = SpatialLagStep(["value"], weights="rook").run(ef).collect()
    queen = SpatialLagStep(["value"], weights="queen").run(ef).collect()
    assert rook["value_spatial_lag"].to_list() == [20.0, 20.0, 20.0, None]
    assert queen["value_spatial_lag"].to_list() == [25.0, 20.0, 15.0, None]


def test_id_renaming_cannot_change_spatial_lag() -> None:
    ef = grid_frame()
    renamed = ef.with_columns(pl.Series("grid_id", [101, 700, 3, -99]))
    step = SpatialLagStep(["value"])
    assert (
        step.run(ef).collect()["value_spatial_lag"].to_list()
        == step.run(renamed).collect()["value_spatial_lag"].to_list()
    )


def test_polygon_rook_and_queen_handle_corner_contacts() -> None:
    units = pl.DataFrame(
        {
            "id": ["A", "B", "C"],
            "geometry": [box(0, 0, 1, 1).wkt, box(1, 1, 2, 2).wkt, box(1, 0, 2, 1).wkt],
        }
    )
    assert SpatialWeights.from_units(units, "id", method="queen").neighbors[0] == (1, 2)
    assert SpatialWeights.from_units(units, "id", method="rook").neighbors[0] == (2,)


def test_distance_band_and_knn_honor_their_parameters() -> None:
    units = pl.DataFrame({"id": ["A", "B", "C"], "x": [0.0, 1.0, 10.0], "y": [0.0, 0.0, 0.0]})
    distance = SpatialWeights.from_units(
        units, "id", method="distance", crs="EPSG:3857", distance_threshold=2
    )
    assert distance.neighbors == ((1,), (0,), ())
    assert SpatialWeights.from_units(units, "id", method="knn", crs="EPSG:3857", k=1).neighbors == (
        (1,),
        (0,),
        (1,),
    )
    with pytest.raises(ValueError, match="projected CRS in meters"):
        SpatialWeights.from_units(
            units, "id", method="distance", crs="EPSG:4326", distance_threshold=2
        )


def test_explicit_graph_can_remove_edges_across_a_barrier() -> None:
    ef = grid_frame()
    weights = SpatialWeights.from_edges(["z", "a", "q", "b"], [("z", "a"), ("a", "z")])
    result = SpatialLagStep(["value"], spatial_weights=weights).run(ef).collect()
    assert result["value_spatial_lag"].to_list() == [20.0, 10.0, None, None]


def test_reference_graph_rejects_missing_units_instead_of_fabricating_zeroes() -> None:
    weights = SpatialWeights.from_edges(["A", "B"], [("A", "B")])
    with pytest.raises(ValueError, match="observation panel"):
        weights.aligned(["A"])


def test_spatial_snapshots_do_not_pool_values_across_time() -> None:
    ef = frame(
        {
            "grid_id": ["A", "B", "A", "B"],
            "snapshot": [0, 0, 1, 1],
            "grid_row": [0] * 4,
            "grid_col": [0, 1, 0, 1],
            "value": [1.0, 2.0, 100.0, 200.0],
        }
    )
    result = SpatialLagStep(["value"], group_cols=["snapshot"]).run(ef).collect()
    assert result["value_spatial_lag"].to_list() == [2.0, 1.0, 200.0, 100.0]


def test_local_moran_matches_explicit_population_variance_formula() -> None:
    ef = frame({"grid_id": [0, 1, 2, 3], "value": [1.0, 2.0, 4.0, 8.0]})
    w = SpatialWeights.from_edges([0, 1, 2, 3], [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
    result = LocalMoranStep("value", permutations=0, spatial_weights=w).run(ef).collect()
    # Deviations: -2.75, -1.75, .25, 4.25; population second moment = 7.1875.
    expected = [4.8125 / 7.1875, 2.1875 / 7.1875, 0.3125 / 7.1875, 1.0625 / 7.1875]
    assert result["value_local_moran_i"].to_list() == pytest.approx(expected)
    assert result["value_local_moran_p"].null_count() == 4
    assert result["value_moran_cluster"].to_list() == ["NS"] * 4


def test_conditional_permutations_are_reproducible_and_used() -> None:
    ef = grid_frame()
    step = LocalMoranStep("value", permutations=39, seed=32)
    a, b = step.run(ef).collect(), step.run(ef).collect()
    assert a.equals(b)
    pvalues = a["value_local_moran_p"].drop_nulls().to_list()
    assert len(pvalues) == 3
    assert all(1 / 40 <= p <= 1 for p in pvalues)


def test_gi_star_matches_binary_weight_normalization() -> None:
    ef = frame({"grid_id": [0, 1, 2, 3], "value": [1.0, 2.0, 4.0, 8.0]})
    w = SpatialWeights.from_edges([0, 1, 2, 3], [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
    result = GetisOrdStep("value", spatial_weights=w).run(ef).collect()
    std = math.sqrt(7.1875)
    expected = [
        (3 - 7.5) / (std * math.sqrt(4 / 3)),
        (7 - 11.25) / std,
        (14 - 11.25) / std,
        (12 - 7.5) / (std * math.sqrt(4 / 3)),
    ]
    assert result["value_gi_star"].to_list() == pytest.approx(expected)


@pytest.mark.parametrize(
    "step,col",
    [(LocalMoranStep("value"), "value_local_moran_i"), (GetisOrdStep("value"), "value_gi_star")],
)
def test_zero_variance_never_produces_statistical_evidence(step, col: str) -> None:
    ef = grid_frame().with_columns(pl.lit(1.0).alias("value"))
    assert step.run(ef).collect()[col].null_count() == 4


def test_spatial_context_preserves_native_types_and_missing_locations() -> None:
    ef = frame({"x": [0.5, 1.0, float("nan"), None], "y": [0.5] * 4})
    context = pl.DataFrame({"geometry": [box(0, 0, 1, 1).wkt], "label": [7]})
    step = SpatialContextJoinStep(context, "geometry", ["label"])
    result = step.run(ef).collect()
    assert result["label"].to_list() == [7, None, None, None]
    assert result["label"].dtype == pl.Int64
    intersects = SpatialContextJoinStep(context, "geometry", ["label"], join_type="intersects")
    assert intersects.run(ef).collect()["label"].to_list() == [7, 7, None, None]


def test_nearest_context_respects_metric_distance_cutoff() -> None:
    ef = frame({"x": [0.5, 10.0], "y": [0.5, 0.5]})
    context = pl.DataFrame({"geometry": [box(0, 0, 1, 1).wkt], "label": ["inside"]})
    step = SpatialContextJoinStep(
        context, "geometry", ["label"], join_type="nearest", max_distance=2
    )
    assert step.run(ef).collect()["label"].to_list() == ["inside", None]


@pytest.mark.parametrize("api", ["step", "strategy", "function"])
def test_temporal_join_tolerance_is_effective_in_each_api(api: str) -> None:
    t = datetime(2024, 1, 1)
    ef = frame({"time": [t + timedelta(hours=3), t + timedelta(hours=1)]})
    ctx = pl.DataFrame({"context_time": [t], "weather": [10.0]})
    if api == "step":
        result = TemporalContextJoinStep(ctx, "context_time", ["weather"], tolerance="1h").run(ef)
    elif api == "strategy":
        result = TemporalJoin(strategy="backward", tolerance="1h").join(
            ef, ctx.lazy(), ContextSchema(timestamp_col="context_time")
        )
    else:
        result = align_temporal(ef, ctx.lazy(), "context_time", strategy="before", window="1h")
    assert result.collect()["weather"].to_list() == [10.0, None]


def test_corrected_temporal_steps_preserve_async_parity() -> None:
    ef = frame({"time": [datetime(2024, 1, 1), datetime(2024, 1, 2)], "value": [1.0, 2.0]})
    pipeline = Pipeline([TemporalLagStep(["value"], [1, 2])])

    async def scenario():
        return await (await pipeline.run_async(ef)).collect_async()

    assert pipeline.run(ef).collect().equals(asyncio.run(scenario()))


def test_spatial_lag_missing_values_require_explicit_policy() -> None:
    ef = grid_frame().with_columns(pl.lit(None, dtype=pl.Float64).alias("value"))
    with pytest.raises(ValueError, match="missing-data policy"):
        SpatialLagStep(["value"]).run(ef)


def test_spatiotemporal_tolerance_and_group_isolation() -> None:
    t = datetime(2024, 1, 1)
    ef = frame({"time": [t + timedelta(hours=2)] * 2, "site": ["A", "B"]})
    ctx = pl.DataFrame(
        {"ct": [t, t + timedelta(hours=1)], "site": ["A", "B"], "weather": [1.0, 2.0]}
    )
    result = (
        SpatioTemporalContextJoinStep(ctx, "site", "ct", ["weather"], tolerance="1h")
        .run(ef)
        .collect()
    )
    assert result["weather"].to_list() == [None, 2.0]


@pytest.mark.parametrize(
    "step,column",
    [
        (TemporalLagStep(["value"]), "value_lag_1"),
        (MovingAverageStep(["value"]), "value_ma_7"),
        (SpatialLagStep(["value"]), "value_spatial_lag"),
        (LocalMoranStep("value"), "value_local_moran_i"),
        (GetisOrdStep("value"), "value_gi_star"),
        (HawkesKernelStep(group_cols=["grid_id"]), "hawkes_intensity"),
    ],
)
def test_empty_inputs_preserve_typed_features(step, column: str) -> None:
    ef = frame({"time": [datetime(2024, 1, 1)], "grid_id": ["A"]}).filter(pl.lit(False))
    result = step.run(ef).collect()
    assert result.is_empty() and column in result.columns


def test_empty_context_does_not_turn_absence_into_an_error() -> None:
    context = pl.DataFrame(schema={"geometry": pl.String, "label": pl.Int64})
    result = (
        SpatialContextJoinStep(context, "geometry", ["label"])
        .run(frame({"value": [1.0]}))
        .collect()
    )
    assert result["label"].to_list() == [None]


def test_trailing_mean_is_partitioned_by_entity() -> None:
    t = datetime(2024, 1, 1)
    ef = frame(
        {
            "time": [t, t + timedelta(days=1), t, t + timedelta(days=1)],
            "site": ["A", "A", "B", "B"],
            "value": [1.0, 2.0, 100.0, 200.0],
        }
    )
    result = MovingAverageStep(["value"], [2], group_cols=["site"]).run(ef).collect()
    assert result["value_ma_2"].to_list() == [None, 1.0, None, 100.0]


def test_hawkes_history_bound_counts_strictly_prior_events() -> None:
    t = datetime(2024, 1, 1)
    ef = frame(
        {"time": [t, t + timedelta(hours=1), t + timedelta(hours=2), t + timedelta(hours=2)]}
    )
    result = HawkesKernelStep(max_history=1).run(ef).collect()
    assert result["hawkes_intensity"].to_list() == pytest.approx(
        [0.1, 0.1 + 0.5 / math.e, 0.1 + 0.5 / math.e, 0.1 + 0.5 / math.e]
    )


def test_reprojection_can_run_in_independent_async_workers() -> None:
    ef = frame({"x": [-87.63, -87.64], "y": [41.88, 41.89]}, crs="EPSG:4326")
    expected = Pipeline([TransformCRSStep("EPSG:26971")]).run(ef).collect()

    async def scenario():
        prepared = await asyncio.gather(
            *[Pipeline([TransformCRSStep("EPSG:26971")]).run_async(ef) for _ in range(3)]
        )
        return await asyncio.gather(*[result.collect_async() for result in prepared])

    for result in asyncio.run(scenario()):
        assert result.equals(expected)
