# Phase 3 — exit criteria (defined before, binary)

Base: `main` @ `91ea351` (merge of PR #43; PR #42 merged at `cf847296`).
Baseline measured 2026-11-15: 838 tests passed locally, coverage 85%,
`make audit` PASS 5/5, Chicago replay 1,908,088 rows × 90 cols,
sum `count` = 60,235.

## Criteria

1. **Suite green, coverage ≥ 85%.** Full pytest passes; ruff, black, mypy clean.
2. **Chicago non-regression.** `make audit` replay still 1,908,088 rows ×
   90 cols, sum `count` = 60,235. No Chicago-specific behaviour changed.
3. **Beijing second-domain proof (TODO item 1).** Replay on the UCI
   Multi-Site Air Quality data (dataset 501), 1 station × 1 year first,
   then all 12 stations:
   - Applicable Steps (lags, calendar, rolling) run with no
     Chicago-specific assumption (no grid, no crime columns).
   - `CompletePanelStep` does NOT turn a missing measurement into 0:
     NA count per value column is identical before and after panel
     completion.
   - No leakage across stations: a station's features never use another
     station's observations (leakage test).
   - Hand-checks: a few station-hours re-derived from the raw CSV.
   - Honest limit documented: timestamps are regular hourly, so this
     proves gaps / missing-stays-missing and per-entity separation, NOT
     irregular sampling.
4. **GeoLife subset replay + boundary statement (TODO item 2).**
   Subset of 1–2 users, one trajectory each (NOT the full ~24.8M rows).
   A written supported/unsupported statement: which Steps apply to
   moving-entity trajectories, which do not, and why.
5. **External connectors (TODO item 3).** Scoped to what the two replays
   actually need; implemented generically (no dataset-specific code in
   `core/`), with tests. No gold-plating.
6. **`make audit` PASS 5/5**, evidence pack in `audit/phase-3/`
   (manifest, Beijing + GeoLife + Chicago replay JSONs, coverage, logs),
   traceability matrix updated with an "After phase 3" section.

Test count / coverage are guardrails, not goals. The maintainer's
go-ahead is the final validation; an ambiguous criterion is marked
partial, never rounded up to success.
