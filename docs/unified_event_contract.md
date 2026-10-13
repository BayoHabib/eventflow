# Unified Event Contract v0.1.0 (draft)

This is the cross-project **assertion envelope** for EventFlow, surveillance-system, Chicago Crime API and (after security review) OSIRIS. The contract is additive: it does not replace `EventFrame` or the existing `EventSchema`.

## Semantic invariant

- `observation`: a measurement or externally reported record; not necessarily ground truth.
- `inference`: an interpretation derived by a model; `model_id` is mandatory.
- `prediction`: model output for an event or horizon; `model_id` is mandatory.
- `hypothesis`: a conjecture requiring independent corroboration.
- `verification`: a new assertion about another assertion; `target_assertion_id` is mandatory.

A `verification` event never modifies its target in place. Use `provenance.upstream_assertion_ids` to construct lineage. Direct self-references are rejected; cycle detection across records must be implemented in the graph store.

`confidence` is an optional number in [0,1], **not** automatically a calibrated probability. Consumers must evaluate calibration before ranking alerts or making consequential decisions.

## Transport and trust

1. Emit versioned JSON over an authenticated service-to-service API or a durable message bus.
2. Require a source identity, unique assertion ID, timezone-aware source/occurrence timestamps, and information classification.
3. Validate message size, allowed event types, rate limits, temporal plausibility and source permissions at ingestion. Deduplicate on trusted issuer + assertion ID.
4. Do not place unredacted video, biometrics, private surveillance targets, secrets or raw identifiable details in the shared event envelope.
5. Store sensitive data privately behind scoped authorization, retention controls and audit logs. Keep OSIRIS-facing projections aggregated/minimized.
6. No inference/hypothesis may be displayed as a confirmed observation. Show uncertainty and source coverage.
7. Validate source trust independently: self-declared `verification_status` is not a trusted endorsement.

## Example (synthetic)

```json
{
  "schema_version": "0.1.0",
  "assertion_id": "example-demo-001",
  "assertion_kind": "observation",
  "event_type": "sensor.motion",
  "occurred_at": "2026-10-13T14:30:00Z",
  "provenance": {
    "source_id": "authorized-demo-sensor",
    "collected_at": "2026-10-13T14:30:02Z",
    "upstream_assertion_ids": []
  },
  "location": {
    "reference_system": "site_local",
    "site_id": "synthetic-site",
    "zone_id": "north-zone"
  },
  "classification": "internal",
  "verification_status": "unverified"
}
```

## Planned adapters (not yet implemented)
- Go surveillance-system alert → UnifiedEvent; do not transmit frames by default.
- Chicago historical event / EventFlow EventFrame row → UnifiedEvent.
- Crime model forecast → prediction assertion with model/version, spatial cell, horizon and calibration provenance.
- OSIRIS gateway → **filtered** and authenticated outward projection; do not send restricted assertions to the unauthenticated SDK stream.

## Explicit non-goals in v0.1.0
- No distributed schema registry, permissions engine, message signing, event graph database, or operational surveillance activation.
- No claim that the current schema is a comprehensive ontology. It defines a minimal interoperable vocabulary requiring future governance.
