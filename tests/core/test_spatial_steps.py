"""Tests for spatial pipeline steps."""

from __future__ import annotations

import polars as pl
import pytest
from shapely import geometry

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.spatial import (
    AssignToGridStep,
    AssignToZonesStep,
    ComputeDistancesStep,
    GetisOrdStep,
    GridAggregationStep,
    KDEStep,
    LocalMoranStep,
    SpatialLagStep,
    TransformCRSStep,
)


@pytest.fixture
def sample_event_frame() -> EventFrame:
    """Create a sample EventFrame for testing spatial steps."""
    lf = pl.LazyFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00",
                "2024-01-01T01:00:00",
                "2024-01-01T02:00:00",
                "2024-01-01T03:00:00",
                "2024-01-01T04:00:00",
            ],
            "latitude": [41.8781, 41.8800, 41.8820, 41.8790, 41.8810],
            "longitude": [-87.6298, -87.6300, -87.6310, -87.6295, -87.6305],
            "value": [10.0, 20.0, 15.0, 25.0, 30.0],
            "category": ["A", "B", "A", "B", "A"],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["value"],
        categorical_cols=["category"],
    )
    metadata = EventMetadata(dataset_name="test-spatial", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


@pytest.fixture
def gridded_event_frame(sample_event_frame: EventFrame) -> EventFrame:
    """Create an EventFrame that's already been assigned to a grid."""
    step = AssignToGridStep(grid_size_m=500.0)
    return step.run(sample_event_frame)


class TestAssignToGridStep:
    """Tests for AssignToGridStep."""

    def test_assigns_grid_ids(self, sample_event_frame: EventFrame) -> None:
        """Test that grid IDs are assigned to all events."""
        step = AssignToGridStep(grid_size_m=500.0)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "grid_id" in df.columns
        assert df["grid_id"].null_count() == 0

    def test_registers_feature_provenance(self, sample_event_frame: EventFrame) -> None:
        """Test that grid_id feature is registered with provenance."""
        step = AssignToGridStep(grid_size_m=500.0)
        result = step.run(sample_event_frame)

        assert "grid_id" in result.metadata.feature_provenance
        provenance = result.metadata.feature_provenance["grid_id"]
        assert provenance.produced_by == "AssignToGridStep"
        assert "spatial" in provenance.tags

    def test_updates_metadata(self, sample_event_frame: EventFrame) -> None:
        """Test that grid_size_m is stored in metadata."""
        step = AssignToGridStep(grid_size_m=250.0)
        result = step.run(sample_event_frame)

        assert result.metadata.grid_size_m == 250.0

    def test_custom_origin(self, sample_event_frame: EventFrame) -> None:
        """Test grid assignment with custom origin."""
        step = AssignToGridStep(grid_size_m=500.0, origin=(-88.0, 41.0))
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "grid_id" in df.columns


class TestAssignToZonesStep:
    """Tests for AssignToZonesStep."""

    def test_assigns_zone_ids(self, sample_event_frame: EventFrame) -> None:
        """Test zone assignment with polygon geometries."""
        # Create simple zone geometries
        zones = pl.DataFrame(
            {
                "zone_id": [1, 2],
                "geometry": [
                    geometry.box(-87.64, 41.87, -87.62, 41.89).wkt,
                    geometry.box(-87.62, 41.87, -87.60, 41.89).wkt,
                ],
            }
        )

        step = AssignToZonesStep(zones, zone_id_col="zone_id", zone_geom_col="geometry")
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "zone_id" in df.columns

    def test_registers_provenance(self, sample_event_frame: EventFrame) -> None:
        """Test that zone_id feature has provenance."""
        zones = pl.DataFrame(
            {
                "zone_id": [1],
                "geometry": [geometry.box(-88.0, 41.0, -87.0, 42.0).wkt],
            }
        )

        step = AssignToZonesStep(zones)
        result = step.run(sample_event_frame)

        assert "zone_id" in result.schema.feature_provenance


class TestComputeDistancesStep:
    """Tests for ComputeDistancesStep."""

    def test_computes_distances(self, sample_event_frame: EventFrame) -> None:
        """Test distance computation to POIs."""
        pois = pl.DataFrame(
            {
                "name": ["poi1", "poi2"],
                "longitude": [-87.6298, -87.6350],
                "latitude": [41.8781, 41.8800],
            }
        )

        step = ComputeDistancesStep(pois)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "dist_to_poi1" in df.columns
        assert "dist_to_poi2" in df.columns

    def test_first_event_zero_distance_to_poi1(self, sample_event_frame: EventFrame) -> None:
        """Test that first event has zero distance to co-located POI."""
        pois = pl.DataFrame(
            {
                "name": ["origin"],
                "longitude": [-87.6298],
                "latitude": [41.8781],
            }
        )

        step = ComputeDistancesStep(pois)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["dist_to_origin"][0] == pytest.approx(0.0)


class TestGridAggregationStep:
    """Tests for GridAggregationStep."""

    def test_computes_event_count(self, gridded_event_frame: EventFrame) -> None:
        """Test grid-level event count aggregation."""
        step = GridAggregationStep(grid_col="grid_id", agg_funcs=["count"])
        result = step.run(gridded_event_frame)

        df = result.collect()
        assert "event_count" in df.columns
        assert df["event_count"].sum() >= 5  # At least 5 events total

    def test_computes_value_aggregations(self, gridded_event_frame: EventFrame) -> None:
        """Test grid-level value aggregations."""
        step = GridAggregationStep(
            grid_col="grid_id",
            agg_funcs=["mean", "sum", "std"],
            value_cols=["value"],
        )
        result = step.run(gridded_event_frame)

        df = result.collect()
        assert "value_mean" in df.columns
        assert "value_sum" in df.columns
        assert "value_std" in df.columns

    def test_registers_all_features(self, gridded_event_frame: EventFrame) -> None:
        """Test that all aggregated features are registered."""
        step = GridAggregationStep(
            grid_col="grid_id",
            agg_funcs=["count", "mean"],
            value_cols=["value"],
        )
        result = step.run(gridded_event_frame)

        assert "event_count" in result.metadata.feature_provenance
        assert "value_mean" in result.metadata.feature_provenance


class TestKDEStep:
    """Tests for KDEStep."""

    def test_computes_density(self, sample_event_frame: EventFrame) -> None:
        """Test KDE density computation."""
        step = KDEStep(bandwidth=0.01, output_col="density")
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "density" in df.columns
        assert df["density"].null_count() == 0

    def test_density_positive(self, sample_event_frame: EventFrame) -> None:
        """Test that density values are non-negative."""
        step = KDEStep(bandwidth=0.01)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert (df["kde_density"] >= 0).all()

    def test_registers_provenance_with_params(self, sample_event_frame: EventFrame) -> None:
        """Test that KDE provenance includes bandwidth parameter."""
        step = KDEStep(bandwidth=500.0, kernel="gaussian")
        result = step.run(sample_event_frame)

        provenance = result.metadata.feature_provenance["kde_density"]
        assert provenance.metadata.get("bandwidth") == 500.0


class TestSpatialLagStep:
    """Tests for SpatialLagStep."""

    def test_computes_spatial_lag(self, gridded_event_frame: EventFrame) -> None:
        """Test spatial lag computation."""
        step = SpatialLagStep(value_cols=["value"], neighbor_col="grid_id")
        result = step.run(gridded_event_frame)

        df = result.collect()
        assert "value_spatial_lag" in df.columns

    def test_multiple_value_cols(self, gridded_event_frame: EventFrame) -> None:
        """Test spatial lag with multiple value columns."""
        # Add another numeric column
        lf = gridded_event_frame.lazy_frame.with_columns(
            (pl.col("value") * 2).alias("value2")
        )
        ef = gridded_event_frame.with_lazy_frame(lf)

        step = SpatialLagStep(value_cols=["value", "value2"], neighbor_col="grid_id")
        result = step.run(ef)

        df = result.collect()
        assert "value_spatial_lag" in df.columns
        assert "value2_spatial_lag" in df.columns


class TestLocalMoranStep:
    """Tests for LocalMoranStep."""

    def test_computes_local_moran(self, gridded_event_frame: EventFrame) -> None:
        """Test Local Moran's I computation."""
        step = LocalMoranStep(value_col="value", spatial_unit_col="grid_id")
        result = step.run(gridded_event_frame)

        df = result.collect()
        assert "value_local_moran_i" in df.columns
        assert "value_moran_cluster" in df.columns

    def test_cluster_labels_valid(self, gridded_event_frame: EventFrame) -> None:
        """Test that cluster labels are valid categories."""
        step = LocalMoranStep(value_col="value", spatial_unit_col="grid_id")
        result = step.run(gridded_event_frame)

        df = result.collect()
        valid_labels = {"HH", "HL", "LH", "LL", "NS"}
        assert all(label in valid_labels for label in df["value_moran_cluster"].to_list())


class TestGetisOrdStep:
    """Tests for GetisOrdStep."""

    def test_computes_gi_star(self, gridded_event_frame: EventFrame) -> None:
        """Test Getis-Ord Gi* computation."""
        step = GetisOrdStep(value_col="value", spatial_unit_col="grid_id")
        result = step.run(gridded_event_frame)

        df = result.collect()
        assert "value_gi_star" in df.columns
        assert "value_hotspot" in df.columns

    def test_hotspot_labels_valid(self, gridded_event_frame: EventFrame) -> None:
        """Test that hotspot labels are valid categories."""
        step = GetisOrdStep(value_col="value", spatial_unit_col="grid_id")
        result = step.run(gridded_event_frame)

        df = result.collect()
        valid_labels = {"hot", "cold", "not_significant"}
        assert all(label in valid_labels for label in df["value_hotspot"].to_list())


class TestTransformCRSStep:
    """Tests for TransformCRSStep."""

    def test_skips_same_crs(self, sample_event_frame: EventFrame) -> None:
        """Test that same CRS returns unchanged frame."""
        step = TransformCRSStep(target_crs="EPSG:4326")
        result = step.run(sample_event_frame)

        assert result.metadata.crs == "EPSG:4326"
        # Should not add _proj columns
        df = result.collect()
        assert "longitude_proj" not in df.columns

    def test_transforms_to_projected_crs(self, sample_event_frame: EventFrame) -> None:
        """Test CRS transformation to projected system."""
        step = TransformCRSStep(target_crs="EPSG:26971")  # Illinois State Plane
        result = step.run(sample_event_frame)

        assert result.metadata.crs == "EPSG:26971"
        df = result.collect()
        assert "longitude_proj" in df.columns
        assert "latitude_proj" in df.columns

    def test_updates_context_requirements(self, sample_event_frame: EventFrame) -> None:
        """Test that CRS is recorded in context requirements."""
        step = TransformCRSStep(target_crs="EPSG:26971")
        result = step.run(sample_event_frame)

        assert result.schema.context_requirements.spatial_crs == "EPSG:26971"
