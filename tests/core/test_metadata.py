"""Tests for EventMetadata extensions."""

from eventflow.core.schema import EventMetadata


def test_event_metadata_defaults():
    """EventMetadata should expose sensible defaults for extended fields."""
    metadata = EventMetadata(dataset_name="test-dataset")

    assert metadata.output_modalities == {"table"}
    assert metadata.feature_catalog == {}
