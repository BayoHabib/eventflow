"""Tests for EventMetadata extensions."""

from eventflow.core.schema import EventMetadata


def test_event_metadata_defaults() -> None:
    """EventMetadata should expose sensible defaults for extended fields."""
    metadata = EventMetadata(dataset_name="test-dataset")

    assert metadata.output_modalities == {"table"}
    assert metadata.feature_catalog == {}
    assert metadata.feature_provenance == {}
    assert metadata.context_requirements.spatial_crs is None
    assert metadata.context_requirements.temporal_resolution is None
    assert metadata.context_requirements.required_context == set()
