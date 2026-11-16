"""Analytic reference checks for window-normalized local K/L/H."""

import asyncio
import math
from datetime import datetime

import numpy as np
import polars as pl
import pytest
from pyproj import Transformer
from shapely.geometry import box

from shikong.core import EventFrame, EventMetadata, EventSchema, SpatialDomain
from shikong.core.steps import get_default_registry
from shikong.core.steps.point_process import KFunctionStep


def points(coords=((2.0, 2.0), (4.0, 2.0), (8.0, 8.0)), crs="EPSG:3857"):
    df = pl.DataFrame(
        {
            "t": [datetime(2024, 1, 1)] * len(coords),
            "x": pl.Series([c[0] for c in coords], dtype=pl.Float64),
            "y": pl.Series([c[1] for c in coords], dtype=pl.Float64),
        }
    )
    return EventFrame(
        df.lazy(),
        EventSchema(timestamp_col="t", lon_col="x", lat_col="y"),
        EventMetadata(dataset_name="declared-point-pattern", crs=crs),
    )


def window(scale=1):
    return SpatialDomain(box(0, 0, 10 * scale, 10 * scale), "EPSG:3857")


@pytest.mark.parametrize("edge,expected", [("none", 50.0), ("translation", 62.5)])
def test_analytic_rectangle_and_transform_identity(edge, expected):
    # Area=100; n=3; one neighbor each for the first pair; third point isolated.
    # Translation overlap at displacement (2,0) is (10-2)*10=80, hence weight=1.25.
    outputs = {
        t: KFunctionStep(3, edge_correction=edge, transform=t, window=window())
        .run(points())
        .collect()
        for t in ["K", "L", "H"]
    }
    k = np.array([expected, expected, 0.0])
    np.testing.assert_allclose(outputs["K"]["k_function_local"], k)
    np.testing.assert_allclose(outputs["L"]["l_function_local"], np.sqrt(k / math.pi))
    np.testing.assert_allclose(outputs["H"]["h_function_local"], np.sqrt(k / math.pi) - 3)
    assert outputs["H"]["h_function_local"][-1] == -3
    # Mean local K equals the corresponding area/(n*(n-1)) ordered-pair estimate.
    assert outputs["K"]["k_function_local"].mean() == pytest.approx(2 * expected / 3)


def test_window_and_radius_rescaling_preserve_physical_units():
    for transform, power, col in [
        ("K", 2, "k_function_local"),
        ("L", 1, "l_function_local"),
        ("H", 1, "h_function_local"),
    ]:
        a = KFunctionStep(3, transform=transform, window=window()).run(points()).collect()[col]
        b = (
            KFunctionStep(6, transform=transform, window=window(2))
            .run(points([(4.0, 4.0), (8.0, 4.0), (16.0, 16.0)]))
            .collect()[col]
        )
        np.testing.assert_allclose(b, a.to_numpy() * 2**power)


@pytest.mark.parametrize("coords", [[], [(2.0, 2.0)]])
def test_undefined_pair_intensity_is_typed_null(coords):
    out = KFunctionStep(3, transform="H", window=window()).run(points(coords)).collect()
    assert out["k_function_local"].dtype == out["h_function_local"].dtype == pl.Float64
    assert out["k_function_local"].null_count() == len(coords)
    assert out["h_function_local"].null_count() == len(coords)


def test_observation_window_is_never_inferred_from_points():
    with pytest.raises(ValueError, match="explicit observation window"):
        KFunctionStep().run(points())
    with pytest.raises(ValueError, match="projected CRS"):
        KFunctionStep(window=SpatialDomain(box(-1, -1, 1, 1), "EPSG:4326")).run(points())
    with pytest.raises(ValueError, match="positive finite area"):
        KFunctionStep(window=SpatialDomain(box(0, 0, 0, 0).buffer(0), "EPSG:3857")).run(points())


def test_holes_and_outside_points_are_rejected_explicitly():
    hole = SpatialDomain(box(0, 0, 10, 10).difference(box(4, 4, 6, 6)), "EPSG:3857")
    for coords in [[(5.0, 5.0), (2.0, 2.0)], [(11.0, 2.0), (2.0, 2.0)]]:
        with pytest.raises(ValueError, match="outside the declared window"):
            KFunctionStep(window=hole).run(points(coords))


def test_coincident_records_are_counted_and_not_silently_deduplicated():
    out = (
        KFunctionStep(3, transform="K", window=window())
        .run(points([(2.0, 2.0), (2.0, 2.0)]))
        .collect()
    )
    assert out["k_function_local"].to_list() == [100.0, 100.0]


def test_projection_agrees_with_metric_reference():
    coords = [(2.0, 2.0), (4.0, 2.0), (8.0, 8.0)]
    tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    geo = [tr.transform(x, y) for x, y in coords]
    out = KFunctionStep(3, transform="K", window=window()).run(points(geo, "EPSG:4326")).collect()
    np.testing.assert_allclose(out["k_function_local"], [62.5, 62.5, 0.0], rtol=1e-8)


def test_unsupported_corrections_and_removed_bins_never_silently_pass():
    for correction in ["ripley", "border", "typo"]:
        with pytest.raises(ValueError, match="Supported edge_correction"):
            KFunctionStep(edge_correction=correction)
    with pytest.raises(ValueError, match="n_bins is no longer"):
        KFunctionStep(n_bins=50)
    with pytest.raises(ValueError, match="transform must"):
        KFunctionStep(transform="typo")
    for radius in [-1, 0, float("inf"), float("nan")]:
        with pytest.raises(ValueError, match="positive and finite"):
            KFunctionStep(max_distance=radius)


def test_zero_translation_overlap_is_rejected():
    with pytest.raises(ValueError, match="zero window overlap"):
        KFunctionStep(20, window=window()).run(points([(0.0, 0.0), (10.0, 10.0)]))


@pytest.mark.parametrize("coords", [[(None, 2.0), (2.0, 2.0)], [(float("nan"), 2.0), (2.0, 2.0)]])
def test_missing_and_nonfinite_locations_are_rejected(coords):
    with pytest.raises(ValueError, match="finite non-null"):
        KFunctionStep(window=window()).run(points(coords))


def test_registry_window_and_async_execution():
    step = get_default_registry().create(
        "k_function",
        params={
            "max_distance": 3,
            "transform": "H",
            "window": {"allowed_wkt": [window().geometry.wkt], "crs": "EPSG:3857"},
        },
    )
    expected = step.run(points()).collect()

    async def scenario():
        return await (await step.run_async(points())).collect_async()

    assert asyncio.run(scenario()).equals(expected)
    meta = step.run(points()).schema.feature_provenance["h_function_local"].metadata
    assert meta["units"] == "m" and meta["edge_correction"] == "translation"
