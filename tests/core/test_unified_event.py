"""Contract tests for cross-project event assertions."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from eventflow.core.unified_event import (
    AssertionKind,
    EventLocation,
    Provenance,
    UnifiedEvent,
)

NOW = datetime(2026, 10, 13, tzinfo=timezone.utc)


def assertion(**changes):
    data = {
        "assertion_id": "example-1",
        "assertion_kind": "observation",
        "event_type": "sensor.motion",
        "occurred_at": NOW,
        "provenance": {"source_id": "demo-sensor", "collected_at": NOW},
        "location": {"reference_system": "site_local", "site_id": "demo-site"},
    }
    data.update(changes)
    return UnifiedEvent.model_validate(data)


def test_observation_roundtrip():
    event = assertion()
    assert UnifiedEvent.model_validate_json(event.model_dump_json()) == event


def test_predictions_require_model():
    with pytest.raises(ValidationError):
        assertion(assertion_kind="prediction")
    assert assertion(assertion_kind="prediction", model_id="poisson-v1").model_id == "poisson-v1"


def test_verification_links_target():
    with pytest.raises(ValidationError):
        assertion(assertion_kind="verification")
    assert assertion(
        assertion_kind="verification", target_assertion_id="example-0"
    ).target_assertion_id == "example-0"


def test_no_model_output_disguised_as_observation():
    with pytest.raises(ValidationError):
        assertion(model_id="vision-v1")


def test_location_coordinate_pair_required():
    with pytest.raises(ValidationError):
        EventLocation(reference_system="EPSG:4326", latitude=41.8)


def test_source_timestamp_requires_timezone():
    with pytest.raises(ValidationError):
        Provenance(source_id="s", collected_at=datetime(2026, 10, 13))


def test_no_direct_self_reference():
    with pytest.raises(ValidationError):
        assertion(provenance={
            "source_id": "s", "collected_at": NOW,
            "upstream_assertion_ids": ["example-1"],
        })


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        assertion(secret_camera_url="unwanted")


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        assertion(confidence=1.2)
