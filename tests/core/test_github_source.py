"""Offline contract tests for commit-pinned public GitHub acquisition."""
import base64
import hashlib

import pytest

from eventflow.core.github_source import ingest_github_document
from eventflow.core.source_corpus import SourceCorpus

SHA = "a" * 40
REPO = "BayoHabib/eventflow"
BODY = b"# Architecture\n\nAn explicitly scoped public document.\n"


def response(data=BODY):
    return {
        "type": "file", "encoding": "base64",
        "content": base64.b64encode(data).decode("ascii"),
        "sha": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }


def ingest(tmp_path, fetch_json=None, **changes):
    params = dict(
        corpus=SourceCorpus(tmp_path),
        repository=REPO,
        path="docs/architecture.md",
        commit_sha=SHA,
        license_note="Repository MIT; document permissions must be checked",
        allowed_repositories=frozenset({REPO}),
        fetched_at="2026-10-13T20:30:00Z",
        fetch_json=fetch_json or (lambda _: response()),
    )
    params.update(changes)
    return ingest_github_document(**params)


def test_end_to_end_pinned_document(tmp_path):
    urls = []
    record = ingest(tmp_path, fetch_json=lambda url: (urls.append(url), response())[1])
    assert urls == [f"https://api.github.com/repos/{REPO}/contents/docs/architecture.md?ref={SHA}"]
    assert record["source_uri"].endswith(f"/blob/{SHA}/docs/architecture.md")
    assert record["content_sha256"] == hashlib.sha256(BODY).hexdigest()
    assert len(record["passages"]) == 2
    assert SourceCorpus(tmp_path).verify(record["source_id"])


def test_replay_with_fixed_timestamp_is_idempotent(tmp_path):
    assert ingest(tmp_path) == ingest(tmp_path)


@pytest.mark.parametrize("overrides", [
    {"repository": "untrusted/private"},
    {"commit_sha": "main"},
    {"commit_sha": "b" * 39},
    {"path": "../secrets.md"},
    {"path": "src/hidden.py"},
    {"path": "/README.md"},
    {"license_note": ""},
])
def test_scope_failures_before_fetch(tmp_path, overrides):
    def impossible(_):
        raise AssertionError("network should not be accessed")
    with pytest.raises(ValueError):
        ingest(tmp_path, fetch_json=impossible, **overrides)


def test_refuses_mismatched_blob_hash(tmp_path):
    payload = response()
    payload["sha"] = "0" * 40
    with pytest.raises(ValueError, match="hash mismatch"):
        ingest(tmp_path, fetch_json=lambda _: payload)


def test_refuses_unexpected_api_response(tmp_path):
    with pytest.raises(ValueError):
        ingest(tmp_path, fetch_json=lambda _: {"type": "dir", "encoding": "base64"})


def test_refuses_oversized_content(tmp_path):
    with pytest.raises(ValueError):
        ingest(tmp_path, fetch_json=lambda _: response(b"x" * (2_000_000 + 1)))


def test_rejects_non_utf8_content(tmp_path):
    with pytest.raises(UnicodeDecodeError):
        ingest(tmp_path, fetch_json=lambda _: response(b"\xff"))
