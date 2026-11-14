"""Generic domain preprocessing: geometry, CRS, unknown coverage and composition."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import polars as pl
import pytest
import yaml
from shapely.geometry import Polygon, box
from typer.testing import CliRunner

from eventflow.core import (
    EventFrame,
    EventMetadata,
    EventSchema,
    SpatialDomain,
    annotate_grid_coverage,
)
from eventflow.core.pipeline import Pipeline
from eventflow.core.steps import GridDomainCoverageStep, SpatialDomainMaskStep, get_default_registry


@pytest.fixture
def domain() -> SpatialDomain:
    return SpatialDomain.from_wkt(
        [box(0, 0, 4, 2).wkt], "EPSG:3857", [box(2, 0, 3, 2).wkt], name="survey-footprint"
    )


@pytest.fixture
def events() -> EventFrame:
    return EventFrame(
        pl.DataFrame(
            {
                "when": [datetime(2024, 1, 1)] * 5,
                "east": [1.0, 2.5, 4.0, 5.0, None],
                "north": [1.0] * 5,
            }
        ).lazy(),
        EventSchema(timestamp_col="when", lon_col="east", lat_col="north"),
        EventMetadata(dataset_name="sensor-readings", crs="EPSG:3857"),
    )


@pytest.fixture
def cells() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cell": [10, 20, 30],
            "footprint": [box(0, 0, 2, 2).wkt, box(2, 0, 4, 2).wkt, box(4, 0, 6, 2).wkt],
        }
    )


@pytest.mark.parametrize("mode,expected", [("annotate", 5), ("filter", 2)])
def test_mask_preserves_unknowns_and_filters_only_when_requested(
    domain: SpatialDomain,
    events: EventFrame,
    mode: str,
    expected: int,
) -> None:
    result = SpatialDomainMaskStep(domain, mode=mode).run(events)
    data = result.collect()
    assert data.height == expected
    if mode == "annotate":
        assert data["inside_domain"].to_list() == [True, False, True, False, None]
        assert data["when"].equals(events.collect()["when"])
    assert events.schema.feature_provenance == {}
    assert result.schema.timestamp_col == "when"
    assert result.schema.lon_col == "east"
    assert result.metadata.dataset_name == "sensor-readings"
    assert result.schema.feature_provenance["inside_domain"].metadata["domain"] == domain.metadata()
    assert "spatial_domain" in result.metadata.context_requirements.required_context


def test_boundary_policy_is_explicit(domain: SpatialDomain, events: EventFrame) -> None:
    result = SpatialDomainMaskStep(domain, include_boundary=False).run(events).collect()
    assert result["inside_domain"].to_list() == [True, False, False, False, None]


def test_domain_is_reprojected_without_relabeling_events(events: EventFrame) -> None:
    geographic = SpatialDomain(box(0, 0, 0.00003, 0.00002), "EPSG:4326")
    result = SpatialDomainMaskStep(geographic).run(events)
    assert result.collect()["inside_domain"].to_list() == [True, True, False, False, None]
    assert result.metadata.crs == events.metadata.crs == "EPSG:3857"
    assert geographic.crs == "EPSG:4326"


def test_wkt_geometry_frames_require_whole_geometry_inside(domain: SpatialDomain) -> None:
    frame = EventFrame(
        pl.DataFrame(
            {
                "time": [datetime(2024, 1, 1)] * 4,
                "shape": ["POINT (1 1)", "LINESTRING (1 1, 5 1)", None, "POINT EMPTY"],
            }
        ).lazy(),
        EventSchema(timestamp_col="time", geometry_col="shape"),
        EventMetadata(dataset_name="moving-assets", crs="EPSG:3857"),
    )
    result = SpatialDomainMaskStep(domain).run(frame).collect()
    assert result["inside_domain"].to_list() == [True, False, None, None]


def test_coverage_has_exact_areas_and_keeps_excluded_cells(
    domain: SpatialDomain,
    cells: pl.DataFrame,
) -> None:
    out = annotate_grid_coverage(cells, domain, crs="EPSG:3857", geometry_col="footprint")
    assert out["domain_area_m2"].to_list() == pytest.approx([4, 2, 0])
    assert out["domain_fraction"].to_list() == pytest.approx([1, 0.5, 0])
    assert out["domain_covered"].to_list() == [True, True, False]
    assert out["cell"].equals(cells["cell"])
    assert out["footprint"].equals(cells["footprint"])


def test_unknown_grid_id_is_distinct_from_zero_coverage(
    domain: SpatialDomain,
    cells: pl.DataFrame,
    events: EventFrame,
) -> None:
    source = events.head(4).with_columns(pl.Series("cell", [10, 20, 30, 999]))
    frame = events.with_lazy_frame(source.lazy())
    step = GridDomainCoverageStep(
        domain, cells, "EPSG:3857", grid_id_col="cell", geometry_col="footprint"
    )
    result = step.run(frame)
    out = result.collect().sort("cell")
    assert out["domain_fraction"].to_list() == [1.0, 0.5, 0.0, None]
    assert out["domain_covered"].to_list() == [True, True, False, None]
    assert out.height == source.height
    assert "domain_area_m2" in result.schema.numeric_cols
    assert result.schema.timestamp_col == frame.schema.timestamp_col


def test_yaml_registry_composes_domain_mask(domain: SpatialDomain, events: EventFrame) -> None:
    config = yaml.safe_load("""
    - name: spatial_domain_mask
      params:
        mode: filter
        domain:
          name: configured-area
          crs: EPSG:3857
          allowed_wkt: ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"]
    """)
    pipeline = get_default_registry().build_pipeline(config)
    result = pipeline.run(events).collect()
    assert result.height == 1
    assert result["east"].to_list() == [1.0]


@pytest.mark.parametrize("dataset", ["environmental-sensors", "transport-events"])
def test_same_steps_apply_to_unrelated_datasets(
    domain: SpatialDomain,
    events: EventFrame,
    dataset: str,
) -> None:
    frame = events.with_metadata(dataset_name=dataset)
    result = Pipeline([SpatialDomainMaskStep(domain)]).run(frame)
    assert result.metadata.dataset_name == dataset
    assert result.collect().height == events.collect().height


def test_concurrent_lazy_collection_is_deterministic(
    domain: SpatialDomain, events: EventFrame
) -> None:
    result = SpatialDomainMaskStep(domain).run(events)
    with ThreadPoolExecutor(max_workers=2) as executor:
        frames = list(executor.map(lambda _: result.collect(), range(4)))
    assert all(frame.equals(frames[0]) for frame in frames)


def test_disjoint_parts_overlaps_and_empty_domain() -> None:
    union = SpatialDomain.from_wkt([box(0, 0, 2, 2).wkt, box(1, 0, 3, 2).wkt], "EPSG:3857")
    assert union.geometry.area == 6  # Overlaps are not counted twice.
    disjoint = SpatialDomain.from_wkt([box(0, 0, 1, 1).wkt, box(2, 0, 3, 1).wkt], "EPSG:3857")
    assert disjoint.geometry.geom_type == "MultiPolygon"
    empty = SpatialDomain.from_wkt([box(0, 0, 1, 1).wkt], "EPSG:3857", [box(0, 0, 1, 1).wkt])
    assert empty.geometry.is_empty
    assert empty.to_crs("EPSG:4326").geometry.is_empty


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3435"])
def test_area_metrics_reject_non_metric_grids(
    domain: SpatialDomain, cells: pl.DataFrame, crs: str
) -> None:
    with pytest.raises(ValueError, match="projected CRS in meters"):
        annotate_grid_coverage(cells, domain, crs=crs, geometry_col="footprint")


@pytest.mark.parametrize("shape", ["LINESTRING (0 0, 1 1)", "POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))"])
def test_invalid_domains_are_rejected(shape: str) -> None:
    with pytest.raises(ValueError, match="valid Polygon"):
        SpatialDomain.from_wkt([shape], "EPSG:3857")


def test_empty_inputs_keep_typed_coverage_columns(
    domain: SpatialDomain, cells: pl.DataFrame
) -> None:
    result = annotate_grid_coverage(
        cells.head(0), domain, crs="EPSG:3857", geometry_col="footprint"
    )
    assert result.height == 0
    assert result.schema["domain_area_m2"] == pl.Float64
    assert result.schema["domain_covered"] == pl.Boolean
    assert SpatialDomain(Polygon(), "EPSG:3857").geometry.is_empty


def test_duplicate_ids_and_column_collisions_are_rejected(
    domain: SpatialDomain,
    cells: pl.DataFrame,
    events: EventFrame,
) -> None:
    with pytest.raises(ValueError, match="must be unique"):
        GridDomainCoverageStep(
            domain,
            pl.concat([cells, cells]),
            "EPSG:3857",
            grid_id_col="cell",
            geometry_col="footprint",
        )
    once = SpatialDomainMaskStep(domain).run(events)
    with pytest.raises(ValueError, match="already exists"):
        SpatialDomainMaskStep(domain).run(once)


def test_domain_identity_tracks_geometry_and_crs(domain: SpatialDomain) -> None:
    assert SpatialDomain(domain.geometry, domain.crs).fingerprint == domain.fingerprint
    assert SpatialDomain(box(0, 0, 4, 2), domain.crs).fingerprint != domain.fingerprint


def test_geometry_boundary_policy_is_strict(domain: SpatialDomain) -> None:
    frame = EventFrame(
        pl.DataFrame({"when": [datetime(2024, 1, 1)], "shape": [box(0, 0, 1, 1).wkt]}).lazy(),
        EventSchema(timestamp_col="when", geometry_col="shape"),
        EventMetadata(dataset_name="survey-plots", crs="EPSG:3857"),
    )
    assert SpatialDomainMaskStep(domain).run(frame).collect()["inside_domain"][0] is True
    assert (
        SpatialDomainMaskStep(domain, include_boundary=False)
        .run(frame)
        .collect()["inside_domain"][0]
    ) is False


def test_grid_coverage_reprojects_reference_domain(cells: pl.DataFrame) -> None:
    domain = SpatialDomain(box(0, 0, 0.00003, 0.00002), "EPSG:4326")
    out = annotate_grid_coverage(cells, domain, crs="EPSG:3857", geometry_col="footprint")
    assert out["domain_fraction"][0] == pytest.approx(1.0)
    assert 0 < out["domain_fraction"][1] < 1
    assert out["domain_fraction"][2] == 0


def test_coverage_enrichment_preserves_input_order(
    domain: SpatialDomain,
    cells: pl.DataFrame,
    events: EventFrame,
) -> None:
    source = events.head(4).with_columns(pl.Series("cell", [30, 10, 999, 20]))
    step = GridDomainCoverageStep(
        domain, cells, "EPSG:3857", grid_id_col="cell", geometry_col="footprint"
    )
    out = step.run(events.with_lazy_frame(source.lazy())).collect()
    assert out["cell"].to_list() == [30, 10, 999, 20]
    assert out["domain_fraction"].to_list() == [0.0, 1.0, None, 0.5]


def test_generic_steps_are_discoverable_from_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    import eventflow.cli.main as cli

    monkeypatch.setattr(cli, "_step_registry", None)
    result = CliRunner().invoke(cli.app, ["list-steps", "--tag", "spatial"])
    assert result.exit_code == 0
    assert "spatial_domain_mask" in result.output
    assert "grid_domain_coverage" in result.output
