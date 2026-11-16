"""Tests of immutable assertions and contradictory evidence."""
import hashlib
import sqlite3

import pytest

from eventflow.core.evidence_ledger import EvidenceLedger, validate_assertion

SOURCE = b"Public scientific document with a stable source passage"


def claim(identifier="a1", **changes):
    result = {
        "assertion_id": identifier,
        "assertion_kind": "observation",
        "event_type": "literature.statement",
        "claim_text": "The first study supports hypothesis H.",
        "occurred_at": "2026-10-12T12:00:00Z",
        "classification": "public",
        "provenance": {
            "source_id": "public-paper-1",
            "source_uri": "https://example.org/paper/1",
            "passage_id": "section-2-p3",
            "content_sha256": hashlib.sha256(SOURCE).hexdigest(),
            "collected_at": "2026-10-13T09:00:00Z",
        },
    }
    result.update(changes)
    return result


def test_append_read_and_idempotence(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    with EvidenceLedger(path) as ledger:
        digest = ledger.append(claim())
        assert digest == ledger.append(claim())
        assert ledger.get("a1")["claim_text"] == claim()["claim_text"]
        assert ledger.get("missing") is None
    with EvidenceLedger(path) as ledger:
        assert ledger.get("a1") is not None


def test_conflicting_reuse_of_assertion_id_is_forbidden(tmp_path):
    with EvidenceLedger(tmp_path / "ledger.sqlite") as ledger:
        ledger.append(claim())
        with pytest.raises(ValueError, match="collision"):
            ledger.append(claim(claim_text="Now a different assertion"))


def test_disagreement_is_preserved(tmp_path):
    with EvidenceLedger(tmp_path / "ledger.sqlite") as ledger:
        ledger.append(claim())
        ledger.append(claim("a2", claim_text="The replication contradicts H."))
        ledger.relate("a2", "a1", "contradicts")
        assert ledger.get("a1") is not None
        assert ledger.get("a2") is not None
        assert ledger.relationships("a1") == [
            {"source_id": "a2", "target_id": "a1", "relation": "contradicts"}
        ]


def test_unavailable_reference_is_not_silently_accepted(tmp_path):
    with EvidenceLedger(tmp_path / "ledger.sqlite") as ledger:
        ledger.append(claim())
        with pytest.raises(sqlite3.IntegrityError):
            ledger.relate("missing", "a1", "supports")


@pytest.mark.parametrize("update", [
    {"assertion_kind": "unknown"},
    {"occurred_at": "2026-10-13T00:00:00"},
    {"assertion_kind": "prediction"},
    {"assertion_kind": "verification"},
    {"assertion_kind": "observation", "model_id": "llm-v1"},
    {"confidence": 1.2},
    {"secret_camera_url": "rtsp://private"},
])
def test_invalid_assertions_rejected(update):
    with pytest.raises(ValueError):
        validate_assertion(claim(**update))


def test_missing_or_invalid_source_provenance():
    payload = claim()
    payload["provenance"]["content_sha256"] = "invalid"
    with pytest.raises(ValueError):
        validate_assertion(payload)


def test_valid_derived_hypothesis(tmp_path):
    item = claim(
        "hyp-1", assertion_kind="hypothesis",
        claim_text="A possible alternative mechanism explains the results.",
    )
    with EvidenceLedger(tmp_path / "ledger.sqlite") as ledger:
        assert ledger.append(item)
