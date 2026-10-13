# Experiment 003 — Pinned GitHub source acquisition

Date: 2026-10-13. Status: implementation committed on a feature branch, not released.

## Objective

Acquire **public, explicitly allowlisted** GitHub documentation at an exact immutable commit SHA into the existing EventFlow `SourceCorpus`. This is a narrow first real-source connector and **not** a general Internet crawler or Agent Reach integration.

## Design

- Function: `eventflow.core.github_source.ingest_github_document`.
- Input: `repository` in an explicit `allowed_repositories` set, relative Markdown/text/reStructuredText path, full 40-character Git commit SHA, source license attribution, and an archive instance.
- Construct the fixed `https://api.github.com/repos/<owner>/<repo>/contents/<path>?ref=<sha>` API endpoint internally. Never accept a caller-provided URL or unpinned branch.
- HTTP GET with a bounded response, 12-second timeout, and redirects disabled. Inspect API's encoded-file response, bound decoded content to 2 MB, validate Git blob hash, then archive bytes with SHA-256 and passage offsets.
- Source ID is the SHA-256 of the canonical, pinned GitHub blob URL. Replays return the original archive record only when bytes, source URI, license attribution and archive integrity still match. A conflict raises an error.
- No authentication tokens or private sources in this initial adapter.

## Offline acceptance tests

Tests in `tests/core/test_github_source.py` use an injected fake GitHub API response, so they require no Internet and never contact a public repo. They cover an authorized response, exact pinned URL, content digest, stable replay, unallowlisted repositories, branch refs, path traversal, unsupported file formats, missing license attribution, forged blob hashes, oversized content, malformed API responses and invalid UTF-8.

Run (from repository root after installing development requirements):

```sh
pytest -q -o addopts='' tests/core/test_source_corpus.py tests/core/test_github_source.py
```

**Test-status discipline:** these tests are committed but GitHub CI confirmation is required. No direct execution of the committed branch is claimed here.

## Real-source experiment plan

Initial controlled source: `BayoHabib/eventflow`, documentation files `README.md` and `docs/architecture.md`. Before acquisition, resolve and record the exact target commit hash and document-level reuse permission. The caller must pass an explicit allowlist and full commit; no branch or tag aliases.

For each snapshot, store:
- Repository, commit, source path, retrieval timestamp.
- License note, SHA-256 of bytes, Git blob hash.
- Structured passages with exact character offsets and attributable text.
- Success/failure and completeness indicators.

Next test: 10–20 authorized documents in a fixed corpus; run deterministic claim extraction, link `EvidenceLedger` assertions to archived passage IDs, and compare a temporal graph backend against SQLite.

## Security, privacy and science

Repository allowlisting is necessary but not sufficient. A GitHub document can contain hostile instructions; raw text must never be treated as instructions for an LLM agent. Never copy authentication tokens into corpus metadata or prompt context. No inference is a fact merely because it cites a URL. Source licenses apply independently to repository code, assets and datasets.

## Known gaps

- Public GitHub API rate limits; only text files from allowlisted repos.
- No GitHub token authentication, no GitHub webhook sync, no Agent Reach/Crawl4AI/Docling ingestion.
- `SourceCorpus` currently offers API-level immutability, not cryptographic append-only audit.
- This code does not establish that an external Git commit is signed or that documentation is factually correct.
- The offline fake responses do not prove network interoperability.
- No full integration with OSIRIS or knowledge graph has yet been implemented.
