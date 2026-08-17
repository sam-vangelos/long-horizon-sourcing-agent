# Runtime State Invariants

These contracts cover canonical sourcing runtime state, launch locks, and
recovery. SQLite remains the source of truth; sidecars and JSON files are only
coordination/projection layers.

## Store Initialization Lock Safety

- `RuntimeStateStore` construction must tolerate concurrent initialization
  against the same `runtime_state.sqlite3`.
- Connection setup must install a SQLite busy timeout before any statement that
  can need a write lock.
- WAL negotiation and additive migration setup must retry transient
  `database is locked` failures before surfacing an error.
- Duplicate-column races during additive migrations mean the desired schema is
  already present and must be treated as success.

## Terminal Lock Drain Contract

- A worker sidecar cannot be treated as launch-blocking when canonical SQLite
  says the latest owned run is terminal and drainable.
- Zombie PIDs are not alive workers.
- Sidecar cleanup must only remove `worker.json` when it still points at the
  current owner PID.
- Same-brief launch while the same brief is already active is idempotent
  success, not a user-facing failure.
- Post-run report or market-intel work must not keep the launch lock for a
  fresh sourcing run.

## Canonical Recovery Boundary

- Reconcile reads canonical run state first and uses sidecars only to decide
  whether cleanup is safe.
- Browser loss, API budget failure, and zero-work terminal runs may preserve
  diagnostics, but they must return the selected brief to ready or active
  product state without manual file edits.
- No recovery path may require hand-editing `output/` or draft brief files.
