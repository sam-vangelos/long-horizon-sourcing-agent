# Quality Operating System

This directory is the shared operating layer for finding and reducing hidden
technical debt in Cloris.

## Tier-0 flows

Start with flows where silent failure is unacceptable:

- Conversational intake produces a reviewable brief.
- Filing a brief persists a valid config.
- Source packet upload keeps source intent and composes a useful draft.
- Route/session resume keeps the user on the intended draft.
- Frontend build and packaged app serve the expected static assets.
- LinkedIn launch/recovery returns to ready or a clearly active run without
  terminal/operator intervention.
- Active sourcing exposes a visible phase and live signal before candidates are
  saved.

## Workflow

1. Log every meaningful bug in `failure-ledger.md`.
2. Map it to an invariant in `invariants/`.
3. Add or strengthen a regression test at the boundary where the bug escaped.
4. Add the narrowest validation command to a focused gate if it should be
   repeatedly checked.
5. If a failure is intentionally not fixed now, put it in known-failures
   with an owner and an expiry date.

The rule: a discovered bug should leave behind a stronger contract, not only a
patch.

## Focused gates

- `make validate-intake`
- `make validate-static-assets`
- `make validate-package-smoke`

`make validate` remains the broad gate. Focused gates exist so agents can check
the relevant contract quickly before the broader suite.

## Invariant Areas

- `invariants/runtime-state.md`

## Domain orchestrators — explicit open gaps

These contracts may still be partial. Do not mark them fixed in the ledger
unless a test or cert guard exists.

| Gap | Intended contract | Current state |
| --- | --- | --- |
| Session draft versioning | `state_revision` / compare-and-swap on intake session PATCH + workers | Revision guards exist on synthesis/compose job blobs inside `state_json`; session-level CAS deferred |
| Shared terminal run statuses | One module for terminal `runs.status` values | `shared/run_status_constants.py` landed; callers still converging |
| Workspace status/judgment browser cert | Browser-observed persistence for all candidate mutations | API cert covers note/status/judgment; browser cert covers note persistence only |
