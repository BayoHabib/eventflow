"""Append-only evidence ledger for the first intelligence-platform experiment.

SQLite storage is intentionally independent of any particular graph DB.
It stores *assertions*, not ground truth. Source references and contradictory
claims remain individually inspectable and cannot be overwritten.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any

KINDS = frozenset({"observation", "inference", "prediction", "hypothesis", "verification"})
CLASSIFICATIONS = frozenset({"public", "internal", "restricted"})


def _canonical(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _utc_timestamp(value: str, name: str) -> str:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO8601 timestamp") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{name} must have a timezone")
    return dt.isoformat()


def validate_assertion(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check minimum provenance, classification and claim-type invariants."""
    allowed = {
        "assertion_id", "assertion_kind", "event_type", "occurred_at",
        "provenance", "classification", "model_id", "target_assertion_id",
        "claim_text", "location", "confidence",
    }
    if set(payload) - allowed:
        raise ValueError("unknown assertion fields")
    claim = dict(payload)
    for field in ("assertion_id", "event_type", "claim_text"):
        if not isinstance(claim.get(field), str) or not claim[field].strip():
            raise ValueError(f"{field} is required")
    if len(claim["assertion_id"]) > 256:
        raise ValueError("assertion_id exceeds maximum length")
    if claim.get("assertion_kind") not in KINDS:
        raise ValueError("invalid assertion_kind")
    if claim.get("classification", "internal") not in CLASSIFICATIONS:
        raise ValueError("invalid classification")
    claim.setdefault("classification", "internal")
    claim["occurred_at"] = _utc_timestamp(claim.get("occurred_at"), "occurred_at")
    prov = claim.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError("provenance required")
    for name in ("source_id", "source_uri", "passage_id", "content_sha256"):
        if not isinstance(prov.get(name), str) or not prov[name].strip():
            raise ValueError(f"provenance.{name} required")
    if len(prov["content_sha256"]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in prov["content_sha256"]):
        raise ValueError("invalid source content_sha256")
    prov = dict(prov)
    prov["collected_at"] = _utc_timestamp(prov.get("collected_at"), "collected_at")
    parents = prov.get("upstream_assertion_ids", [])
    if not isinstance(parents, list) or any(not isinstance(x, str) or not x for x in parents):
        raise ValueError("invalid upstream assertions")
    if claim["assertion_id"] in parents:
        raise ValueError("direct self-reference")
    prov["upstream_assertion_ids"] = parents
    claim["provenance"] = prov
    if claim["assertion_kind"] in {"inference", "prediction"} and not claim.get("model_id"):
        raise ValueError("model_id required for model-derived assertions")
    if claim["assertion_kind"] == "observation" and claim.get("model_id"):
        raise ValueError("model output cannot masquerade as observation")
    if claim["assertion_kind"] == "verification" and not claim.get("target_assertion_id"):
        raise ValueError("verification requires target_assertion_id")
    if "confidence" in claim and claim["confidence"] is not None:
        confidence = claim["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not 0 <= confidence <= 1:
            raise ValueError("confidence outside [0,1]")
    return claim


class EvidenceLedger:
    """Single-writer-friendly local proof-of-concept, not a security boundary.

    Do not expose the SQLite file or its query methods to untrusted clients.
    A service gateway must perform authorization and redaction before returning
    entries to OSIRIS or other consumers.
    """

    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS assertions (
                assertion_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                digest TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS relationships (
                source_id TEXT NOT NULL REFERENCES assertions(assertion_id),
                target_id TEXT NOT NULL REFERENCES assertions(assertion_id),
                relation TEXT NOT NULL CHECK(relation IN ('contradicts','supports','revises')),
                PRIMARY KEY(source_id, target_id, relation),
                CHECK(source_id <> target_id)
            );
        """)

    def __enter__(self) -> EvidenceLedger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def append(self, payload: Mapping[str, Any]) -> str:
        claim = validate_assertion(payload)
        canonical = _canonical(claim)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self.connection:
            existing = self.connection.execute(
                "SELECT digest FROM assertions WHERE assertion_id=?", (claim["assertion_id"],)
            ).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError("assertion_id collision: immutable record")
                return digest
            self.connection.execute(
                "INSERT INTO assertions VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                (claim["assertion_id"], canonical, digest),
            )
        return digest

    def get(self, assertion_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload FROM assertions WHERE assertion_id=?", (assertion_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def relate(self, source_id: str, target_id: str, relation: str) -> None:
        if relation not in {"contradicts", "supports", "revises"}:
            raise ValueError("unsupported relationship")
        if source_id == target_id:
            raise ValueError("self relationship forbidden")
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO relationships (source_id,target_id,relation) VALUES (?,?,?)",
                (source_id, target_id, relation),
            )

    def relationships(self, assertion_id: str) -> list[dict[str, str]]:
        rows = self.connection.execute(
            "SELECT source_id,target_id,relation FROM relationships "
            "WHERE source_id=? OR target_id=? ORDER BY source_id,target_id,relation",
            (assertion_id, assertion_id),
        ).fetchall()
        return [{"source_id": a, "target_id": b, "relation": r} for a, b, r in rows]
