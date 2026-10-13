# Experiment 001 — Evidence Ledger implementation note

Date: 2026-10-13. **Scope:** a local proof-of-concept for retaining sourced assertions and explicit contradictions. NOT a production knowledge base or a Web miner.

## Implemented
- `eventflow.core.evidence_ledger.EvidenceLedger`: SQLite append-only-by-API assertions, immutable assertion ID collision checks, idempotent replay, and relations `supports` / `contradicts` / `revises`.
- `validate_assertion`: claim type, timestamps with timezone, source ID/URI, passage reference, source-content SHA256, optional model ID and bounded confidence.
- Regression tests in `tests/core/test_evidence_ledger.py` use **synthetic records only**.

## Example use

```python
from eventflow.core.evidence_ledger import EvidenceLedger

with EvidenceLedger("experiment.sqlite") as ledger:
    ledger.append({
        "assertion_id": "doc-a:claim-1",
        "assertion_kind": "observation",
        "event_type": "literature.statement",
        "claim_text": "A reported finding (example only).",
        "occurred_at": "2026-10-10T12:00:00Z",
        "classification": "public",
        "provenance": {
            "source_id": "doc-a",
            "source_uri": "https://example.org/example",
            "passage_id": "section-1:p2",
            "content_sha256": "a" * 64,
            "collected_at": "2026-10-13T09:00:00Z",
        },
    })
```

This example shows **shape**, not verified scientific evidence. A hash field is accepted as syntax but the ledger does not fetch or verify the source itself.

## Design decision
- SQLite rather than Graphiti for the baseline; avoid AI or graph DB dependencies until a benchmark proves value.
- Append-only means **no record update/delete APIs**; it does **not** mean tamper-resistant or cryptographically auditable storage. Filesystem attackers can modify the database.
- The ledger is not the full `UnifiedEvent` contract from PR #14. These are two experimental schemas, and **must be reconciled before integration**. This experiment requires `claim_text` and source passage identifiers, while PR #14 provides a more general event envelope.
- `observation` in this experiment means an attributed reported statement, not empirically verified truth.

## Known limitations / remaining work
- No HTTP API, authentication/authorization, isolation, quotas, retention, source license enforcement or per-tenant security.
- No actual Agent Reach / Crawl4AI / Docling / OSIRIS / Surveillance Go connector is installed.
- No true source-content verification; URI and hash are user-provided metadata; assertion IDs can collide across untrusted issuers; max payload size not yet enforced.
- Relations are constrained by foreign keys, but there is no transitive cycle check or evidence-weighting model.
- No remote process ingestion or parallel workload evaluation.
- No source corpus collected; no grounded scientific claim or hypothesis test executed.
- Tests added and CI requested through draft PR; report pass/failure only when actually confirmed.

## Next acceptance gate
Build a fixed licensed/authorized corpus with stored content digests and passage offsets. Compare deterministic extraction vs Docling/Crawl4AI, then assess Graphiti against the SQLite baseline for temporal history, contradiction retrieval, cost and repeatability. The ingestion adapter must *not* trust assertions embedded as instructions within collected Web pages.
