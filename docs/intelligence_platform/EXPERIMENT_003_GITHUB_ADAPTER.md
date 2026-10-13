# Experiment 003 — GitHub pinned-document source adapter

Date: 2026-10-13. Status: **implemented, not yet CI-verified or live-ingestion verified**.

## Delivered

- `eventflow.core.github_source_adapter.ingest_github_document`: fetches allowed repository documentation via GitHub Contents API, at exactly a 40-character commit SHA.
- Explicit per-call `allowed_repositories` permission check; no general URL-fetch interface.
- Accepts UTF-8 Markdown, text and reStructuredText only. Enforces body and decoded size bounds.
- Verifies Git blob SHA-1 over raw bytes before archiving with `SourceCorpus`, which stores an independent SHA-256.
- Rejects redirects in the default network transport. Requests target `https://api.github.com/repos/` only.
- Tests use an injected fake GitHub API client and synthetic data. No network calls in unit tests.

## Illustrative code

```python
from eventflow.core.source_corpus import SourceCorpus
from eventflow.core.github_source_adapter import ingest_github_document

corpus = SourceCorpus("./corpus")
record = ingest_github_document(
    corpus,
    repository="BayoHabib/eventflow",
    commit_sha="<full verified 40-digit commit SHA>",
    path="docs/architecture.md",
    allowed_repositories={"BayoHabib/eventflow"},
    license_note="MIT-licensed repository documentation (check assets separately)",
)
assert corpus.verify(record["source_id"])
```

The placeholder SHA must be replaced with a genuine reviewed commit identifier; the adapter never silently substitutes a moving branch.

## Execution instructions

```bash
python -m pip install -e '.[dev]'
pytest -q -o addopts='' tests/core/test_github_source_adapter.py tests/core/test_source_corpus.py
```

Run a controlled live smoke test only after selecting exact repositories, SHAs, document paths and licensing policy. Github API rate limits and availability can cause failures; the reader does not fall back to unpinned data.

## Limitations / security review

- **No current live fetched documents** are asserted. A GitHub connector was used separately to inspect source code; that is not an archived live test of this adapter.
- `source_id` is stable per repo/commit/path, while `collected_at` defaults to wall-clock time. Replaying to the **same** `SourceCorpus` at a later timestamp causes a deliberate collision; pin `collected_at` for deterministic replays or redesign source identity and collection records separately.
- The current `SourceCorpus` checks only a URI prefix and cannot itself fetch or validate remote sources; **do not expose** its `add` method as a public API.
- SHA-1 verification checks Git blob content consistency, **not** authenticity of an arbitrary supplied commit SHA or the copyright status of the material. Use trusted repo ownership and review licensing.
- README/MD text is untrusted content; prompt injection must not be forwarded as control instructions to language models.
- Not designed for Git LFS or files beyond size limits; no binary documents, PDF/image parsing or Agent Reach integration yet.
- License notes are caller declarations, not a legal verification engine.
- No authentication at the EventFlow level. GitHub public rate limits apply. No OSIRIS exposure.
- URL redirect blocker is a defense for the fixed API endpoint. Network egress policies remain advisable.

## Next gate

1. Verify CI checks on draft PR.
2. Pin a list of 10–20 permissible GitHub docs with full commit SHAs.
3. Execute live ingestion and export object SHA256, source metadata, passage offsets and reproducibility results.
4. Enrich structured claims with cited passages; evaluate against the evidence ledger.
5. Compare Crawl4AI and Docling on the same licensed fixed corpus.
6. Only then develop the Agent Reach multiprovider acquisition boundary.
