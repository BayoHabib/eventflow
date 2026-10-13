"""Read-only, commit-pinned GitHub documentation adapter.

Only explicit allowlisted public repositories, known text extensions and exact
40-hex commits are accepted. The adapter constructs its own GitHub API URL.
A network fetcher can be injected for offline tests. This is not a general
purpose URL fetcher or a replacement for authenticated Github connectors.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Callable
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from eventflow.core.source_corpus import SourceCorpus


_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
_PART = re.compile(r"[A-Za-z0-9_.-]+\Z")
_ALLOWED_SUFFIXES = (".md", ".txt", ".rst")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _github_json(url: str, max_response_bytes: int = 3_000_000) -> dict:
    """Fetch ONLY a caller-constructed API URL, with no redirects."""
    if not url.startswith("https://api.github.com/repos/"):
        raise ValueError("unexpected network endpoint")
    request = Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "eventflow-public-corpus/0.1"},
    )
    response = build_opener(_NoRedirect).open(request, timeout=12)
    with response:
        if response.status != 200:
            raise ValueError(f"GitHub API returned {response.status}")
        body = response.read(max_response_bytes + 1)
        if len(body) > max_response_bytes:
            raise ValueError("GitHub API response too large")
        return json.loads(body)


def ingest_github_document(
    corpus: SourceCorpus,
    *,
    repository: str,
    path: str,
    commit_sha: str,
    license_note: str,
    allowed_repositories: frozenset[str],
    fetch_json: Callable[[str], dict] = _github_json,
    fetched_at: str | None = None,
) -> dict:
    """Fetch a pinned public documentation file into the local source corpus.

    A license note is an attribution record, not a legal determination.
    Never let untrusted documents choose repository allowlists or fetch URLs.
    """
    if repository not in allowed_repositories:
        raise ValueError("repository is not allowlisted")
    components = repository.split("/")
    if len(components) != 2 or any(not _PART.fullmatch(part) for part in components):
        raise ValueError("invalid repository")
    if not _SHA.fullmatch(commit_sha):
        raise ValueError("commit must be a full 40-character SHA")
    parts = path.split("/")
    if (not parts or len(parts) > 12 or any(not _PART.fullmatch(p) or p in {".", ".."} for p in parts)
            or not path.lower().endswith(_ALLOWED_SUFFIXES)):
        raise ValueError("unsupported documentation path")
    if not license_note.strip():
        raise ValueError("license provenance required")

    url = f"https://api.github.com/repos/{repository}/contents/{'/'.join(parts)}?ref={commit_sha.lower()}"
    payload = fetch_json(url)
    if payload.get("type") != "file" or payload.get("encoding") != "base64":
        raise ValueError("GitHub response is not an encoded file")
    encoded = payload.get("content")
    if not isinstance(encoded, str) or len(encoded) > 3_000_000:
        raise ValueError("GitHub content missing or oversized")
    try:
        content = base64.b64decode(encoded, validate=False)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("invalid GitHub base64 content") from exc
    if len(content) > 2_000_000:
        raise ValueError("GitHub source document exceeds size limit")
    # Verify Git's blob hash in addition to our SHA-256 content address.
    git_blob_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if payload.get("sha") != git_blob_sha:
        raise ValueError("GitHub blob hash mismatch")

    canonical_uri = f"https://github.com/{repository}/blob/{commit_sha.lower()}/{path}"
    source_id = hashlib.sha256(canonical_uri.encode("utf-8")).hexdigest()
    timestamp = fetched_at or datetime.now(timezone.utc).isoformat()
    record = corpus.add(
        source_id=source_id,
        uri=canonical_uri,
        fetched_at=timestamp,
        content=content,
        license_note=license_note,
    )
    if not corpus.verify(source_id):
        raise ValueError("stored document integrity check failed")
    return record
