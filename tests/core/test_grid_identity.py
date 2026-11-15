"""Cross-API regressions from the real-data audit; no ID ordering assumptions."""

from datetime import datetime

import polars as pl
import pytest
from shapely.geometry import box

from eventflow.core import EventFrame, EventMetadata, EventSchema, GridSpec, SpatialDomain
from eventflow.core.grid import MAX_GRID_INDEX, grid_cell_id
from eventflow.core.spatial import assign_to_grid, create_grid
from eventflow.core.steps.domain import GridDomainCoverageStep
from eventflow.core.steps.spatial import AssignToGridStep


def points(x, y):
    return EventFrame(
        pl.DataFrame({"t": [datetime(2024, 1, 1)] * len(x), "x": x, "y": y}).lazy(),
        EventSchema(timestamp_col="t", lon_col="x", lat_col="y"),
        EventMetadata(dataset_name="grid-contract", crs="EPSG:3857"),
    )


def test_signed_identity_is_unique_and_matches_lazy_encoding():
    pairs = [(r, c) for r in range(-20, 21) for c in range(-20, 21)]
    ef = points([c + 0.25 for _, c in pairs], [r + 0.25 for r, _ in pairs])
    result = AssignToGridStep(1, origin=(0, 0)).run(ef).collect()
    expected = [grid_cell_id(r, c) for r, c in pairs]
    assert result["grid_id"].to_list() == expected
    assert len(set(expected)) == len(expected)
    assert result["grid_row"].to_list() == [r for r, _ in pairs]


def test_previous_fixed_stride_collision_is_impossible():
    out = AssignToGridStep(1, origin=(0, 0)).run(points([100000.25, 0.25], [0.25, 1.25])).collect()
    assert out["grid_id"].n_unique() == 2  # old encoding returned 100000 for both.


def test_core_pipeline_and_coverage_share_identity():
    ef = points([-0.25, 0.25, 1.25], [0.25, 0.25, 1.25])
    grid = create_grid((-1, 0, 2, 2), 1, "EPSG:3857", origin=(0, 0))
    core = assign_to_grid(ef, grid)
    step = AssignToGridStep(1, origin=(0, 0)).run(ef)
    assert (
        core.collect()
        .select("grid_id", "grid_row", "grid_col")
        .equals(step.collect().select("grid_id", "grid_row", "grid_col"))
    )
    assert core.metadata.custom["grid_spec"] == step.metadata.custom["grid_spec"]
    domain = SpatialDomain(box(-0.5, 0, 2, 2), "EPSG:3857")
    out = GridDomainCoverageStep(domain, grid, "EPSG:3857").run(step).collect()
    assert out["domain_fraction"].to_list() == [0.5, 1.0, 1.0]


def test_extent_shuffling_and_batch_size_do_not_change_anchored_ids():
    ef = points([0.25, 1.25], [0.25, 1.25])
    small = create_grid((0, 0, 2, 2), 1, "EPSG:3857", origin=(0, 0))
    big = create_grid((-2, -2, 4, 4), 1, "EPSG:3857", origin=(0, 0)).reverse()
    assert (
        assign_to_grid(ef, small)
        .collect()["grid_id"]
        .equals(assign_to_grid(ef, big).collect()["grid_id"])
    )
    a = AssignToGridStep(1, origin=(0, 0)).run(ef).collect()
    b = AssignToGridStep(1, origin=(0, 0)).run(ef.with_lazy_frame(ef.lazy_frame.head(1))).collect()
    assert a["grid_id"][0] == b["grid_id"][0]


def test_outside_reference_does_not_alias_an_existing_cell():
    grid = create_grid((0, 0, 1, 1), 1, "EPSG:3857")
    assert assign_to_grid(points([2.25], [0.25]), grid).collect()["grid_id"][0] is None


def test_legacy_ids_and_misaligned_geometry_are_rejected():
    grid = create_grid((0, 0, 2, 2), 1, "EPSG:3857")
    with pytest.raises(ValueError, match="Legacy or incompatible"):
        assign_to_grid(
            points([0.25], [0.25]), grid.with_columns(pl.int_range(0, pl.len()).alias("grid_id"))
        )
    with pytest.raises(ValueError, match="aligned squares"):
        assign_to_grid(
            points([0.25], [0.25]), grid.with_columns((pl.col("maxy") + 0.1).alias("maxy"))
        )
    with pytest.raises(ValueError, match="CRS must agree"):
        assign_to_grid(
            points([0.25], [0.25]), grid.with_columns(pl.lit("EPSG:26971").alias("grid_crs"))
        )


@pytest.mark.parametrize("sign", [-1, 1])
def test_extreme_supported_indices_do_not_overflow(sign):
    idx = sign * MAX_GRID_INDEX
    ef = points([float(idx) + 0.25], [float(idx) + 0.25])
    actual = AssignToGridStep(1, origin=(0, 0)).run(ef).collect()["grid_id"][0]
    assert actual == grid_cell_id(idx, idx) and 0 <= actual < 2**63
    with pytest.raises(ValueError, match="indices must lie"):
        AssignToGridStep(1, origin=(0, 0)).run(points([float(idx * 2)], [0.0])).collect()


def test_spec_fingerprint_distinguishes_lattices():
    a = GridSpec(1, (0, 0), "EPSG:3857").metadata()
    assert a == GridSpec(1.0, (-0.0, 0.0), "EPSG:3857").metadata()
    assert a["fingerprint"] != GridSpec(2, (0, 0), "EPSG:3857").metadata()["fingerprint"]
    assert a["fingerprint"] != GridSpec(1, (1, 0), "EPSG:3857").metadata()["fingerprint"]


def test_empty_batch_requires_an_explicit_anchor():
    ef = points([0.25], [0.25])
    empty = ef.with_lazy_frame(ef.lazy_frame.head(0))
    assert AssignToGridStep(1, origin=(0, 0)).run(empty).collect().is_empty()
    with pytest.raises(ValueError, match="origin from empty"):
        AssignToGridStep(1).run(empty)
