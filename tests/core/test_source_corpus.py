"""Tests for deterministic passage extraction and content-addressed archive."""
import pytest

from eventflow.core.source_corpus import SourceCorpus, passages


def ingest(root, **updates):
    args = {
        "source_id": "fixture-001",
        "uri": "https://example.org/authorized-document",
        "fetched_at": "2026-10-13T14:00:00Z",
        "content": b"One paragraph.\n\nSecond paragraph.",
        "license_note": "Synthetic test fixture, reusable",
    }
    args.update(updates)
    return SourceCorpus(root).add(**args)


def test_exact_passage_offsets():
    source = "First statement.\n\nSecond statement."
    extracted = passages(source)
    assert len(extracted) == 2
    assert all(source[x["start"]:x["end"]] == x["text"] for x in extracted)


def test_archive_idempotence_and_readback(tmp_path):
    x = ingest(tmp_path)
    assert x == ingest(tmp_path)
    assert SourceCorpus(tmp_path).verify("fixture-001")
    assert len(list((tmp_path / "objects").iterdir())) == 1


def test_source_id_collision_rejected(tmp_path):
    ingest(tmp_path)
    with pytest.raises(ValueError, match="collision"):
        ingest(tmp_path, content=b"Changed source")


def test_modified_object_detected(tmp_path):
    x = ingest(tmp_path)
    (tmp_path / "objects" / f"{x['content_sha256']}.txt").write_bytes(b"tampered")
    assert not SourceCorpus(tmp_path).verify("fixture-001")


@pytest.mark.parametrize("changes", [
    {"fetched_at": "2026-10-13T14:00:00"},
    {"source_id": "../bad"},
    {"uri": "http://example.org/insecure"},
    {"content": b"\xff"},
    {"license_note": ""},
    {"max_bytes": 1},
])
def test_bad_inputs_rejected(tmp_path, changes):
    with pytest.raises((ValueError, UnicodeDecodeError)):
        ingest(tmp_path, **changes)
