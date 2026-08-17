# AGENTS.md

Canonical guide for AI coding agents (and new humans) working in this repo. Read this before making changes.

## What this repo is

A sourcing platform with four main layers:

1. **Intake, brief authoring, and orchestration**
   - source-packet intake from JD, notes, uploaded docs, and recruiter gap answers (`shared/source_packet.py`, `cloris/api/intake*.py`)
   - conversational readback and human-approved brief filing
   - chief-of-staff orchestration across source specialists (`cloris/chief_of_staff/`)
   - source launcher registry (`cloris/launchers/__init__.py`)

2. **Source adapters** (status per the 2026-07-02 module consolidation)
   - LinkedIn sourcing flow (`linkedin/`) — first-class, active
   - Researcher sourcing flow (`researcher/`) — deprioritized; code present, not under active development
   - Designer portfolio sourcing flow (`designer/`) — SUNSET; directory retained for history, do not extend
   - Exec Search workflow (`exec_search/`) — SUNSET; directory retained for history, do not extend

3. **Shared execution, runtime state, workspace, and identity** (`shared/`)
   - canonical candidate / run / work-unit state
   - resume semantics
   - compatibility projections
   - Cloris-native candidate workspace save destination
   - global cross-source identity store

4. **Post-run intelligence** (`market_intelligence/`, `shared/brief_*`, `tools/iterate_brief.py`)
   - run snapshots
   - market intelligence
   - brief iteration and strategy artifacts

## Hard boundaries

This repo is the Sourcing Agent only. It is **not**:

- TA Ops Agent
- Rosie
- Analytics Hub
- the whole recruiting AI stack

Do not add stack-level architecture docs, cross-repo integration plans, or
platform-wide roadmaps here without explicit instruction.

## Canonical truths

- `runtime_state.sqlite3` is **canonical** for sourcing runtime state.
- JSON/JSONL files in live state dirs are **compatibility projections**, not source-of-truth control state.
- If SQLite and projection files disagree, **trust SQLite**.
- `output/runs/...` is immutable finalized run output.
- `output/market_intelligence/...` is market-scoped synthesis, separate from live per-project runtime state.
- The current product contract starts from **role/source material**, not a pre-authored brief: Cloris compiles source packets, asks gap questions, reads the draft back, and files the brief only after human approval. OPERATING EXCEPTION (2026-07-05, deliberate): the LinkedIn smoke loop currently executes preflight-generated V2 briefs immediately with `reviewed: false` — the human-approval gate is suspended for operator smoke runs, not abandoned as product contract. Both structural audits flagged this divergence; reinstating the gate is Sam's call.
- Registered launch sources are declared in `cloris/launchers/__init__.py`; do not reintroduce source lists in separate docs or UI constants without a synchronization reason.

When investigating complexity, distinguish: canonical state / projections / snapshots / in-memory working state. They behave differently and a bug in one does not imply a bug in the others.

## High-risk files

Edit with elevated care. Scoped rules under `.cursor/rules/high-risk-files.mdc` fire when any of these are open.

| File | Why it's high-risk |
|------|--------------------|
| `shared/runtime_state/store.py` | shared persistence + lifecycle + reconciliation hot spot |
| `shared/runtime_state/linkedin.py` | LinkedIn resume/progress bridge semantics |
| `linkedin/orchestrator.py` | large, policy-heavy; casual broad edits cause regressions |
| `linkedin/browser.py` | brittle browser/runtime behavior; small targeted changes only |
| `market_intelligence/engine.py` | large orchestration surface; refactor carefully |
| `cloris/api/` and `cloris/api/_monolith.py` | Broad HTTP surface coordinating launch, onboarding, workspace, reflection, and COS/conversation; easy to break cross-plane behavior |

Orchestration stacks on **four** interaction planes: per-project `runtime_state.sqlite3` (canonical), JSON/JSONL **projections** (non-authoritative), immutable **`output/runs` snapshots**, and **global orchestration / COS / conversation** state that spans briefs. When debugging, identify which plane a read or write targets before changing code.

- **Targeted edits** over broad cleanup.
- **Preserve behavior** unless the task explicitly calls for changing it.
- When refactoring, **add or strengthen tests first** when feasible.
- Keep changes narrow and commit slices intentional.
- Do not casually mix runtime-state work with unrelated strategy, reconciliation, or config work in one diff.
- Prefer **behavior-preserving extraction** over giant rewrites.
- Cross-cutting changes should be staged deliberately across multiple commits.

## Brief / config discipline

- **Do not edit draft briefs unless explicitly asked.** Files matching `config/brief-*-draft.json` are scratch/in-flight; a pre-edit hook (`.cursor/hooks/guard-protected-paths.sh`, wired only through `.cursor/hooks.json`) blocks edits in Cursor sessions unless the user names the file — there is no equivalent hook in `.claude/settings.json`, so in-harness this is enforced by convention, not a technical guard.
- Treat brief churn carefully — config files often encode product truth.
- Do not make scratch briefs look runnable by accident.

## Output / artifact discipline

- **Avoid editing `output/` directly.** Hand-editing is blocked by the same pre-edit hook in Cursor sessions; in-harness, with no equivalent hook in `.claude/settings.json`, this is enforced by convention, not a technical guard.
- Rebuild projections or artifacts through code paths or tools — never hand-edit under `output/`.
- If a bug looks like it lives in `output/`, the real fix is almost always upstream in the code path that wrote it.

## Testing expectations

- After targeted runtime-state changes, run the narrowest relevant band first:
  - `pytest tests/test_linkedin_runtime_state.py -q`
  - `pyproject.toml` sets `pythonpath = ["."]` so no manual `PYTHONPATH=` prefix is needed.
- Expand outward only if the change crosses boundaries.
- Full-suite gate before declaring done: `make validate` (hygiene + default Python suite — the frontend gate was dropped 2026-08-02; the web surface is a relic). To run frontend checks by hand against the parked copy: `make frontend-validate` (svelte-check + Vitest via `pnpm`; needs Node 20+ and `pnpm` on `PATH`).
- Prefer proving behavior with tests **before** "cleanup" refactors.
- **Never trust a piped pytest exit code** — `pytest | tail` reports tail's exit and can swallow the summary. Capture pytest's own exit (`pytest -q > out.txt 2>&1; echo $?`) and count progress chars when the summary line is absent (a known quirk here).

## Model + shadow-experiment norms (durable, 2026-07-05)

- **Shadow debugging goes through `tools/shadow_replay.py`** (real prompts, real client paths, no browser) — never a live LinkedIn run. The live feed is `tools/shadow_report.py <state-dir> --follow`.
- **Never send a thinking parameter to Opus-family models.** `_thinking_request_kwargs` in `shared/llm_clients.py` is the single guard: claude-fable/claude-mythos only. Fable also rejects sampling params — the Anthropic clients pass none.
- **The run console is sacred**: any new console output on live-run surfaces is ONE prebuilt line per event — no JSON, no multi-line blocks, no raw model prose. Depth belongs in artifacts and the follow feed.
- **Shadow paths are fail-soft end to end** — recorded-never-influencing; a shadow defect must never raise into a primary path.
- If an edit shifts a line pinned in `tools/role_agnosticism_baseline.txt`, re-key that line number only (never add/remove entries), and re-key LAST after all edits to that file.

## Search / inspection defaults

- Prefer `rg` for finding code or text quickly.
- Ask for file-cited explanations when tracing architecture.

## Commit hygiene

- Only commit when explicitly asked.
- When asked, slice intentionally: separate behavior-preserving refactors from behavior changes; do not blob unrelated concerns together.
- Surface adjacent issues rather than silently expanding scope.

## Workflow artifact: `plans/`

Non-trivial work uses a shared plan file at `plans/<topic>.md`. Use
INDEX to find active/current plans; do not broad-scan `plans/` at
startup because completed, generated, and quarantined historical artifacts live
under `plans/archive/`. Template and conventions: `plans/README.md`. This is how
the strategist (Codex or Claude Code) and Cursor hand work back and forth without
losing context.

## How to work

- Before substantive edits, list assumptions and every file you will touch.
- For non-trivial work, start from a plan file under `plans/` before editing.
- Prefer slice-sized plans and commit-sized implementation. If the seam is
  already clear and the slice is a clean whole-file change, Cursor can often
  implement, test, stage, and commit in one loop.
- Slow down into staged-diff review only when selective staging, mixed dirty
  files, high-risk surfaces, or ambiguous boundaries make that extra ceremony
  necessary.

## Cursor seat status

The Cursor seat is DORMANT (2026-07-05 fleet ruling — see CLAUDE
§"Model fleet and seats"). Everything under `.cursor/rules/` and
`.cursor/agents/`, and the playbook in cursor-codex-workflow, is
historical until Sam reactivates the seat. `.cursor/hooks/guard-protected-paths.sh`
is wired only through `.cursor/hooks.json` (Cursor's own preToolUse config);
no equivalent hook exists in `.claude/settings.json`, so for in-harness
sessions the brief/output discipline above is enforced by convention, not by
a technical guard.

## Repo biases

- Runtime-state-first thinking is preferred.
- Behavior-preserving extraction is preferred over giant rewrites.
- Clean commit hygiene matters.
- Cross-cutting changes should be staged deliberately.

## Related docs

- `plans/README.md` — the shared plan-file artifact convention.
