"""Offline contract tests; no network request or GitHub token required."""
import base64
import hashlib

import pytest

from eventflow.core.github_source_adapter import ingest_github_document
from eventflow.core.source_corpus import SourceCorpus

REPO = "BayoHabib/eventflow"
COMMIT = "a" * 40
BODY = b"# A public document\n\nA reported claim in an immutable snapshot."


def response(body=BODY):
    blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
    return {
        "type": "file",
        "encoding": "base64",
        "size": len(body),
        "content": base64.b64encode(body).decode("ascii"),
        "sha": blob,
    }


def ingest(root, *, fetch_json=lambda url: response(), **kwargs):
    return ingest_github_document(
        SourceCorpus(root),
        repository=REPO,
        commit_sha=COMMIT,
        path="docs/architecture.md",
        allowed_repositories={REPO},
        license_note="MIT repository documentation, verify individual assets",
        collected_at="2026-10-13T18:00:00Z",
        fetch_json=fetch_json,
        **kwargs,
    )


def test_pinned_source_is_archived_and_verified(tmp_path):
    item = ingest(tmp_path)
    assert SourceCorpus(tmp_path).verify(item["source_id"])
    assert item["source_uri"].endswith(f"{COMMIT}/docs/architecture.md")
    assert item["passages"][0]["text"] == "# A public document"


def test_replay_is_idempotent(tmp_path):
    assert ingest(tmp_path) == ingest(tmp_path)


def test_git_integrity_rejects_modified_response(tmp_path):
    payload = response()
    payload["sha"] = "f" * 40
    with pytest.raises(ValueError, match="integrity"):
        ingest(tmp_path, fetch_json=lambda url: payload)


@pytest.mark.parametrize("repository,commit_sha,path", [
    ("other/repo", COMMIT, "README.md"),
    (REPO, "main", "README.md"),
    (REPO, COMMIT, "../../secrets.txt"),
    (REPO, COMMIT, "private/config.json"),
    (REPO, COMMIT, "README.md/../secrets.md"),
])
def test_unapproved_inputs_do_not_fetch(tmp_path, repository, commit_sha, path):
    called = False

    def forbidden(url):
        nonlocal called
        called = True
        return response()

    with pytest.raises(ValueError):
        ingest_github_document(
            SourceCorpus(tmp_path),
            repository=repository,
            commit_sha=commit_sha,
            path=path,
            allowed_repositories={REPO},
            license_note="MIT",
            fetch_json=forbidden,
        )
    assert not called


def test_encoded_document_size_enforced(tmp_path):
    with pytest.raises(ValueError, match="size limit"):
        ingest(tmp_path, fetch_json=lambda url: response(b"x" * 1000), max_bytes=10)


def test_wrong_git_blob_type_rejected(tmp_path):
    with pytest.raises(ValueError, match="base64 file"):
        ingest(tmp_path, fetch_json=lambda url: {"type": "dir"})
