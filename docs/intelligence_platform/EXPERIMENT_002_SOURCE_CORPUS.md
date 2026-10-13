# Experiment 002 — Public source corpus and passage provenance

**Date:** 2026-10-13. **Status:** initial offline ingestion module and unit tests; no Web crawler wired yet.

## Why a separate source corpus?

EventFlow's evidence ledger records claims and citation metadata. Before accepting those citations, the system needs a reproducible archive of authorized source bytes, their SHA-256 digests, and precise text offsets. An assertion is not grounded simply because a language model attaches a URL.

## Implemented

- `src/eventflow/core/source_corpus.py`: archive UTF-8 bytes addressed by SHA-256, with immutable per-source metadata.
- Deterministic `passages(text)` with character offsets that can be checked against the original source.
- Validation for timestamps with timezone, source IDs, HTTPS source identifiers, UTF-8 encoding, declared license terms, 2 MB default limit.
- `SourceCorpus.verify` recomputes file SHA-256 and validates source passage offsets.
- `tests/core/test_source_corpus.py`: 10 unit-test cases (synthetic fixtures).
- Local preliminary prototype: 10 tests passed with pytest; **remote GitHub Actions result not yet established**.

## Candidate *real* sources for the first authorized corpus

| Source | Reference | License/access caveat |
|---|---|---|
| EventFlow documentation | https://github.com/BayoHabib/eventflow/blob/main/docs/architecture.md | Repository MIT license; verify individual data provenance |
| EventFlow source design | https://github.com/BayoHabib/eventflow/blob/main/README.md | Same |
| OSIRIS public documentation | https://github.com/simplifaisoul/osiris/blob/master/README.md | Repository MIT license, check contained assets |
| Scientific paper metadata (later) | https://api.openalex.org/ | API use policy and rate limits need validation |

These are source candidates, **not evidence already fetched by the new module**. The first remote fetch connector, content snapshot and benchmark remain pending. The GitHub connector was used to inspect related repository files during design, but those readings have not been ingested into a versioned `SourceCorpus` snapshot.

## Next gates

1. Provide a deliberately scoped GitHub adapter that downloads repository documentation with pinned commit SHA (no user-provided arbitrary URL).
2. Store 10–20 permissible source snapshots with byte digests, source commit, timestamps and explicit attribution.
3. Add per-passage extraction where every generated assertion points to an existing source/passage.
4. Before any wider Web connector (Agent Reach/Crawl4AI), implement domain allowlisting, DNS and redirect SSRF defenses, size/time limits, robots/terms checks and isolated extraction of hostile content.
5. Benchmark Docling/Crawl4AI only after establishing the deterministic baseline.
6. Reconcile the two draft schemas: `UnifiedEvent` (PR #14) and `EvidenceLedger` (PR #16) before joining them.

## Known limitations

- Offline-only; URI is provenance metadata, not a fetching instruction, and `https://` prefix alone is NOT an SSRF safeguard.
- The archive's immutability is API-level, not tamper-resistant or signed.
- Character offsets are for decoded UTF-8 text, not original PDF byte offsets; multimodal documents need source-specific anchors.
- No access control, retention/deletion policy, protected credentials, authorization or production gateway.
- No real sources integrated, no model accuracy results, no OSIRIS connection.

## Reproduction

After editable installation with development dependencies:

```bash
pytest -q -o addopts='' tests/core/test_source_corpus.py
```

Run from the EventFlow repository root. The isolated preliminary Python proof-of-concept passed 10 tests locally; CI remains the canonical verification for the committed version.
