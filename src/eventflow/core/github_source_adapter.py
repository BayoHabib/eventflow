"""Read-only GitHub document adapter with pinned commits and strict network scope.

The adapter is intentionally limited to the official GitHub REST host and
explicitly authorized repositories. It is *not* an arbitrary URL fetcher.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from eventflow.core.source_corpus import SourceCorpus

_REPO = re.compile(r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}\Z")
_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
_ALLOWED_EXTENSIONS = (".md", ".txt", ".rst")
_MAX_BODY = 2_800_000


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("GitHub redirects are disabled for this adapter")


def _github_json(url: str, token: str | None = None) -> dict:
    """Fetch a bounded API JSON body from GitHub. No redirects or arbitrary host."""
    if not url.startswith("https://api.github.com/repos/"):
        raise ValueError("only GitHub REST repository endpoints are allowed")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "eventflow-source-adapter/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers, method="GET")
    opener = build_opener(_RejectRedirects())
    with opener.open(request, timeout=12) as response:
        if response.status != 200:
            raise ValueError(f"GitHub API status {response.status}")
        data = response.read(_MAX_BODY + 1)
    if len(data) > _MAX_BODY:
        raise ValueError("GitHub API response exceeds limit")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("unexpected API response")
    return payload


def _validate_path(path: str) -> None:
    if not isinstance(path, str) or not path or len(path) > 400:
        raise ValueError("invalid path")
    segments = path.split("/")
    if any(segment in {"", ".", ".."} for segment in segments) or "\\" in path:
        raise ValueError("unsafe repository path")
    if not path.lower().endswith(_ALLOWED_EXTENSIONS):
        raise ValueError("only public text documentation formats are permitted")
    if any(ord(char) < 32 for char in path):
        raise ValueError("control character in path")


def ingest_github_document(
    corpus: SourceCorpus,
    *,
    repository: str,
    commit_sha: str,
    path: str,
    allowed_repositories: set[str] | frozenset[str],
    license_note: str,
    fetch_json: Callable[[str], dict] = _github_json,
    max_bytes: int = 1_000_000,
    collected_at: str | None = None,
) -> dict:
    """Archive one source at a full immutable commit, without trusting page text.

    A caller supplies an allowlist and an independently established license note.
    The response's Git blob SHA-1 is checked against decoded content bytes.
    """
    if not _REPO.fullmatch(repository) or repository not in allowed_repositories:
        raise ValueError("repository is not explicitly allowed")
    if not _SHA.fullmatch(commit_sha):
        raise ValueError("full 40-character Git commit SHA required")
    _validate_path(path)
    if max_bytes <= 0 or max_bytes > 2_000_000:
        raise ValueError("invalid max_bytes")
    if not license_note.strip():
        raise ValueError("license note required")

    from urllib.parse import quote

    url = (
        f"https://api.github.com/repos/{repository}/contents/"
        f"{quote(path, safe='/')}?ref={commit_sha}"
    )
    payload = fetch_json(url)
    if payload.get("type") != "file" or payload.get("encoding") != "base64":
        raise ValueError("expected base64 file response")
    if payload.get("size", 0) > max_bytes:
        raise ValueError("source document exceeds size limit")
    raw = base64.b64decode(payload["content"], validate=False)
    if len(raw) > max_bytes:
        raise ValueError("decoded document exceeds size limit")
    raw.decode("utf-8")
    # Git blob identity is SHA1('blob <byte-size>\\0' + bytes), not SHA1(bytes).
    git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
    if payload.get("sha") != git_blob:
        raise ValueError("Git blob integrity mismatch")
    # source IDs remain stable across identical snapshots even with different run dates.
    source_id = hashlib.sha256(
        f"{repository}@{commit_sha}:{path}".encode("utf-8")
    ).hexdigest()[:32]
    timestamp = collected_at or datetime.now(timezone.utc).isoformat()
    source_uri = f"https://github.com/{repository}/blob/{commit_sha}/{quote(path, safe='/')}"
    return corpus.add(
        source_id=source_id,
        uri=source_uri,
        fetched_at=timestamp,
        content=raw,
        license_note=license_note,
        max_bytes=max_bytes,
    )
