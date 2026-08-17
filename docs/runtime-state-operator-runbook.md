# Runtime-State Operator Runbook

## Purpose

This is the operator runbook for the second-generation runtime model.

Use it when a GitHub or LinkedIn run needs to be inspected, resumed, repaired,
or intentionally replayed. The official operator surface is:

- `python3 tools/runtime_state_admin.py ...`

Do not edit `progress.json`, stage JSONLs, `candidate_history-*.jsonl`, or
`search_memory-*.json` by hand when `runtime_state.sqlite3` exists. Those files
are projections.

## Mental Model

Normal runs now work like this:

- canonical truth lives in `runtime_state.sqlite3`
- `progress.json` is a projected compatibility checkpoint
- stage JSONLs, candidate history, and search memory are projected artifacts
- direct side-effect artifacts such as `run_log.jsonl`, `saves.jsonl`, and
  `outreach.jsonl` remain direct writes, but they do not control candidate truth
- the runtime-state admin CLI still uses `--output-dir`, but that path should
  now be the mutable `state_dir` under `output/state/<source>/<brief-id>/`

## Resume

Resume is runtime-state-first.

- GitHub: `python3 run_github.py --brief ... --resume`
- LinkedIn: `python3 -m linkedin.session_orchestrator --brief ... --resume`

What happens underneath:

- the latest run for the source + brief is loaded from `runtime_state.sqlite3`
- any orphaned in-flight attempts are reconciled to retryable failure
- any lingering pending candidate side effects are marked interrupted (never
  failed — that status is reserved for a different retry path)
- projections are rebuilt whenever a resume finds an existing run with work
  units — unconditional on resume, not gated on whether reconciliation
  actually changed anything

`progress.json` still matters for session UX, but only as a projection emitted
from the store.

## Rebuild Projections

Rebuild projected artifacts from runtime-state:

```bash
python3 tools/runtime_state_admin.py \
  --output-dir output/state/linkedin/<brief-id> \
  --source linkedin \
  --brief-id <brief-id> \
  rebuild-projections
```

Use this when projections were deleted, manually edited, or left stale after an
interrupted local workflow.

## Restart a LinkedIn String

Restart one LinkedIn string through runtime-state:

```bash
python3 tools/runtime_state_admin.py \
  --output-dir output/state/linkedin/<brief-id> \
  --source linkedin \
  --brief-id <brief-id> \
  restart-linkedin-string \
  --string-id 12
```

This clears terminal state and side-effect idempotency for candidates tied to
 that string, requeues the work unit, rebuilds projections, and preserves audit
history.

## Inspect Stop Reasons

Inspect persisted run stop reasons:

```bash
python3 tools/runtime_state_admin.py \
  --output-dir output/state/github/<brief-id> \
  --source github \
  --brief-id <brief-id> \
  inspect-stop-reasons
```

Important stop reasons include:

- `normal`
- `governor_limit`
- `session_expired`
- `operator_stop`
- `operator_pause`
- `lock_conflict`
- `browser_disconnect_unrecovered`
- `api_budget_exhausted`
- `fatal_runtime_error`
- `worker_missing` — stamped by cloris's own reconciler on a run stranded
  `status='running'` with a dead or missing worker sidecar; this is what
  drives the "Lost track" state

## Inspect Orphans

Inspect orphaned candidate attempts:

```bash
python3 tools/runtime_state_admin.py \
  --output-dir output/state/linkedin/<brief-id> \
  --source linkedin \
  --brief-id <brief-id> \
  inspect-orphans
```

Orphaned attempts should normally already have been reconciled on startup, but
this command is useful when investigating interrupted runs.

## Inspect and Replay Candidate Side Effects

Inspect candidate-scoped side effects:

```bash
python3 tools/runtime_state_admin.py \
  --output-dir output/state/github/<brief-id> \
  --source github \
  --brief-id <brief-id> \
  inspect-side-effects
```

Replay a candidate-scoped side effect intentionally by invalidating the prior
ledger row:

```bash
python3 tools/runtime_state_admin.py \
  --output-dir output/state/github/<brief-id> \
  --source github \
  --brief-id <brief-id> \
  replay-side-effect \
  --identity-key <candidate-identity> \
  --effect-type github_outreach
```

This does not auto-run the side effect by itself. It makes a deliberate replay
possible on the next appropriate execution path.

## Safe Manual Intervention

Safe manual intervention means:

- rebuilding projections from runtime-state
- restarting/requeueing through runtime-state admin commands
- invalidating candidate-scoped side effects intentionally
- inspecting stop reasons and orphaned attempts before changing anything

Unsafe manual intervention means:

- editing `progress.json` to influence resume behavior
- deleting stage JSONLs to try to “un-dedup” candidates
- editing `candidate_history-*.jsonl` or `search_memory-*.json` as if they were
  authoritative
- using raw reset scripts against outputs that already contain
  `runtime_state.sqlite3`

If `runtime_state.sqlite3` exists, prefer the admin surface. Older raw reset
scripts are only for pre-runtime-state outputs and should fail fast otherwise.
