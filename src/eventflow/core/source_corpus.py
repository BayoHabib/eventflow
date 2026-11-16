"""Deterministic offline ingestion of authorized UTF-8 documents.

Fetching is intentionally left to connectors which enforce URL policies,
permissions, limits, redirect safety, and content licensing.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import re


def passages(text: str, max_chars: int = 1500) -> list[dict]:
    if max_chars < 64:
        raise ValueError("max_chars must be >= 64")
    result = []
    for m in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text):
        start, stop = m.span()
        while start < stop:
            end = min(stop, start + max_chars)
            if text[start:end].strip():
                result.append({
                    "passage_id": f"p{len(result):05d}", "start": start,
                    "end": end, "text": text[start:end],
                })
            start = end
    return result


class SourceCorpus:
    """Content-addressed archive, not a production security boundary."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / "records").mkdir(parents=True, exist_ok=True)

    def add(self, source_id: str, uri: str, fetched_at: str,
            content: bytes, license_note: str, max_bytes: int = 2_000_000) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", source_id):
            raise ValueError("invalid source_id")
        if not uri.startswith("https://") or not license_note.strip():
            raise ValueError("HTTPS URI and license provenance required")
        try:
            dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            if dt.tzinfo is None or dt.utcoffset() is None:
                raise ValueError("timezone required")
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid timezone-aware fetched_at") from exc
        if not isinstance(content, bytes) or len(content) > max_bytes:
            raise ValueError("invalid or oversized content")
        text = content.decode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        data = {
            "source_id": source_id, "source_uri": uri, "collected_at": fetched_at,
            "content_sha256": digest, "content_length": len(content),
            "license_note": license_note, "passages": passages(text),
        }
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        record_path = self.root / "records" / f"{source_id}.json"
        if record_path.exists() and record_path.read_text("utf-8") != serialized:
            raise ValueError("immutable source_id collision")
        obj_path = self.root / "objects" / f"{digest}.txt"
        if not obj_path.exists():
            obj_path.write_bytes(content)
        if not record_path.exists():
            record_path.write_text(serialized, encoding="utf-8")
        return data

    def verify(self, source_id: str) -> bool:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", source_id):
            raise ValueError("invalid source_id")
        data = json.loads((self.root / "records" / f"{source_id}.json").read_text("utf-8"))
        content = (self.root / "objects" / f"{data['content_sha256']}.txt").read_bytes()
        if hashlib.sha256(content).hexdigest() != data["content_sha256"]:
            return False
        text = content.decode("utf-8")
        return all(text[x["start"]:x["end"]] == x["text"] for x in data["passages"])
