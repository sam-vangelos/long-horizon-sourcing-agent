# Cloris

Cloris is a recruiter-judgment-driven agentic sourcing system. It takes role material such as a job description, intake notes, uploaded documents, recruiter answers, and institutional context; turns that material into a structured sourcing brief; asks for missing judgment; reads the brief back for correction; and files it only after human approval. (Operating exception: the LinkedIn smoke loop currently files preflight-generated V2 briefs immediately with `reviewed: false` — see AGENTS.md § Canonical truths.)

After the brief is approved, a chief-of-staff layer coordinates source specialists across the sources registered in `cloris/launchers/__init__.py` — the single source of truth for what is launchable (currently LinkedIn and GitHub/OSS active, Researcher deprioritized; Designer and Exec Search remain registered but administratively retired). Those specialists use source-native acquisition and evidence evaluation, but save into the same workspace and runtime substrate.

The repo started as an autonomous LinkedIn/GitHub search agent. The current system is broader than that: it includes source-packet intake, conversational brief authoring, a generic launcher registry, source adapters, a shared execution/runtime layer, operator tooling, a candidate workspace, market-level synthesis, and a workflow for turning run evidence back into the next version of a brief.

## What The System Is

The filed brief is the execution contract, but it is no longer assumed to be the starting input. Cloris first captures the role-specific judgment that is usually scattered across job descriptions, intake notes, recruiter memory, search strings, uploaded documents, and ad hoc evaluation habits:

- what kind of work the team actually needs
- where the yes/no boundary sits
- which lookalike profiles usually waste time
- how search should open
- what kinds of evidence matter on each source surface

That judgment is made executable. The result is a human-approved role definition that can drive planning, evaluation, adaptation, reporting, workspace review, and post-run iteration.

The loader still supports older brief formats as compatibility inputs. The current authoring path is source-packet intake through `shared/source_packet.py`, `shared/intake_conversation/`, and the conversation/readback flow; filed briefs then carry the structured schema and source-specific `source_config` used by the launchers.

## How A Run Works

1. Capture source material: job description, intake notes, uploaded PDF/DOCX/TXT/MD files, recruiter gap answers, and role context.
2. Synthesize a structured draft brief, ask gap questions, read the brief back, and file it only after human approval.
3. The chief-of-staff layer selects and orders source specialists for the filed brief.
4. Launch one or more launchable source adapters (the registry in
   `cloris/launchers/__init__.py` is the authoritative list):
   - LinkedIn via browser automation inside Recruiter
   - GitHub/OSS via API-driven search and enrichment
   - Researcher via academic APIs and publication evidence (deprioritized)
5. Discover and evaluate candidates in stages: light evidence triage, deeper review for promising candidates, and terminal source-aware judgment.
6. Persist canonical candidate/work-unit/runtime state in `runtime_state.sqlite3` for each source run.
7. Save/reconcile candidates into the Cloris workspace and rebuild compatibility artifacts as needed.
8. Finalize immutable run snapshots, update market intelligence, and propose next-run strategy/brief changes from run evidence and recruiter feedback.

```mermaid
flowchart LR
    A["Role Material + Uploaded Sources"] --> B["Conversational Intake + Source Packet Compiler"]
    B --> C["Human-Approved Structured Brief"]
    C --> D["Chief-of-Staff Orchestration"]
    D --> E["Source Launcher Registry"]
    E --> F["LinkedIn / GitHub / Researcher / Designer / Exec Search"]
    F --> G["Shared Execution + runtime_state.sqlite3"]
    G --> H["Candidate Workspace"]
    G --> I["Run Snapshots"]
    I --> J["Market Intelligence + Reflection"]
    J --> C
```

## Architecture

The system has seven major layers:

- **Intake and brief layer**
  - source-packet capture, conversational gap questions, human-approved structured role judgment
- **Planning layer**
  - chief-of-staff orchestration, retrieval design, source-native strategy, adaptation, query/lane evolution
- **Source adapters**
  - LinkedIn Recruiter browser workflow
  - GitHub/OSS API search, enrichment, graph expansion, outreach/export side effects
  - Researcher publication/author search
  - Designer portfolio/work-product acquisition and evaluation
  - Exec Search dossier and market-lane workflow
- **Shared execution/runtime layer**
  - canonical candidate lifecycle, attempt tracking, side-effect ledgers, resume state
- **Candidate workspace and identity layer**
  - Cloris-native save destination, per-brief workspace aggregation, cross-source identity store
- **Run artifact layer**
  - compatibility projections, run snapshots, structured reports
- **Market intelligence and brief iteration**
  - per-market synthesis, optional research, draft brief updates from real run evidence

Two docs go deeper on the execution/runtime model:

- [Runtime-State Operator Runbook](docs/runtime-state-operator-runbook.md)
- [Shared Candidate Execution Engine](docs/shared-candidate-execution-engine.md)

For refactor history and what remains as product work rather than architecture debt, see Sourcing-Agent-2nd-Gen-Roadmap.md.

## LinkedIn Capabilities

The LinkedIn adapter connects to a live Chrome session over CDP and works inside LinkedIn Recruiter. It is not just a static string runner.

Current LinkedIn capabilities include:

- two-stage snippet-to-profile evaluation
- role-driven search string generation from the brief
- root query families with sibling variants
- pre-commit experimentation before locking onto a pagination path
- mid-string drift rescue when a once-productive query decays
- runtime-backed search memory and candidate history
- mutation budgeting so search changes stay bounded and auditable
- session orchestration, pacing, budgets, resume/restart support, and optional decoy activity

The important architectural point is that LinkedIn search intelligence is more than “narrow or broaden.” It now tracks search families, experiments within them, and persists that search state in the runtime layer.

## GitHub Capabilities

The GitHub adapter uses API-driven search rather than browser automation. It works across multiple acquisition channels, including:

- user search
- code search
- topic and repository mining
- stargazer and graph expansion from strong candidates

It enriches candidates with repository, contribution, profile, and contact data before running the same kind of structured judgment flow used on LinkedIn.

For saved candidates, the GitHub side can generate outreach copy and export operator-facing CSVs. Strong candidates can also feed graph expansion so the search moves outward from actual signal rather than staying trapped in the original query set.

## Runtime State And Output Model

The shared runtime model is the core of the current architecture.

`runtime_state.sqlite3` is the authoritative record of:

- candidate lifecycle
- work-unit status
- attempt history
- side effects
- resume state

The four storage layers are:

1. **Live project state** in `output/state/...`
   - mutable, project-scoped working state
   - canonical `runtime_state.sqlite3`

2. **Compatibility projections**
   - `progress.json`
   - stage JSONLs
   - `candidate_history-*.jsonl`
   - `search_memory-*.json`
   - useful operational artifacts, but not control-state inputs

3. **Finalized run snapshots** in `output/runs/...`
   - immutable per-run archives
   - used for reporting, replay, and post-run synthesis

4. **Market artifacts** in `output/market_intelligence/...`
   - canonical per-market synthesis
   - separate from per-project LinkedIn or GitHub run state

In practice:

- LinkedIn live state is scoped to the LinkedIn project / brief state key
- GitHub live state is scoped to the GitHub brief ID
- market intelligence is scoped to a derived market key
- projections are rebuildable from runtime state when needed

That distinction matters: live project state, run snapshots, and market artifacts are different layers with different jobs.

## Operating Modes And Safety

On LinkedIn, the system operates through a CDP-connected Chrome session and supports two input modes:

- `concurrent`
  - synthetic mouse/input path that is safer while you keep using the computer
- `away`
  - real mouse/keyboard takeover mode for unattended sessions

The operational model also includes:

- humanized pacing and cadence rather than bursty automation
- bounded search mutation rather than constant query rewriting
- runtime-state-first resume and recovery
- optional decoy activity for safer long-running Recruiter sessions

This repo is not structured around fragile file-edit recovery. When runtime state exists, the canonical recovery/admin surface is `tools/runtime_state_admin.py`.

## Post-Run Learning Loop

Market intelligence is a first-class subsystem, not just report generation.

It consumes finalized run evidence, produces canonical per-market artifacts, can optionally run external research, maintains its own market-keyed artifact/state layer, and feeds both operator strategy and brief iteration.

The post-run loop looks like this:

- finalize a run snapshot
- synthesize what the run actually learned
- update canonical market artifacts
- optionally enrich that synthesis with external research
- draft the next version of the brief from run evidence plus market intel

That is how the system gets continuity across runs instead of treating each session as an isolated sourcing episode.

## Repository Layout

```text
sourcing-agent/
├── config/                # Role briefs and supporting job-description files
├── cloris/                # Desktop app, API package, launchers, chief-of-staff orchestration
├── linkedin/              # LinkedIn Recruiter adapter, browser automation, search intelligence
├── github/                # GitHub/OSS adapter, enrichment, query planning, exports, observability
├── researcher/            # Academic researcher sourcing module
├── designer/              # Portfolio/work-product sourcing module
├── exec_search/           # Executive-search workflow module
├── market_intelligence/   # Post-run synthesis, research backends, artifact generation
├── shared/                # Brief loading, schemas, runtime state, execution engine, utilities
├── tools/                 # Runtime admin, brief iteration, market-intel update helpers
├── docs/                  # Runbooks, cheat sheets, architecture notes, archived campaign docs
└── output/                # Mutable state, finalized run snapshots, exports, market intel
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in the keys required by the effective model roles in `.env`.

- The standard LinkedIn role map requires `ANTHROPIC_API_KEY` and
  `FIREWORKS_API_KEY`; model-role substitutions are configuration-only.
- Other model-provider keys are required only when an effective role selects
  that provider.
- `GITHUB_TOKEN` is required for GitHub sourcing.
- `PERPLEXITY_API_KEY` is optional and only matters if you want external research during market-intelligence updates.

If you plan to run LinkedIn, start Chrome through the helper script and keep your Recruiter session logged in:

```bash
./launch-chrome.sh
```

## Common Commands

These examples use two current briefs for the same role (Principal Research Engineer, Code) that already exist in the repo:

```bash
LINKEDIN_BRIEF=config/prre-code/brief-prre-code-v2.json
GITHUB_BRIEF=config/prre-code/brief-prre-code-github-v2.json
```

### LinkedIn runs

Recommended entry point:

```bash
python3 -m linkedin.session_orchestrator --brief "$LINKEDIN_BRIEF"
```

Useful variants:

```bash
python3 -m linkedin.session_orchestrator --brief "$LINKEDIN_BRIEF" --single-session
python3 -m linkedin.session_orchestrator --brief "$LINKEDIN_BRIEF" --multi-session
python3 -m linkedin.session_orchestrator --brief "$LINKEDIN_BRIEF" --multi-session --with-decoy
python3 -m linkedin.session_orchestrator --brief "$LINKEDIN_BRIEF" --resume
python3 -m linkedin.session_orchestrator --brief "$LINKEDIN_BRIEF" --restart-string 12 --resume
python3 -m linkedin.session_orchestrator --status
python3 -m linkedin.session_orchestrator --decoy-only
```

### GitHub runs

Recommended entry point:

```bash
python3 run_github.py --brief "$GITHUB_BRIEF"
```

Useful variants:

```bash
python3 run_github.py --brief "$GITHUB_BRIEF" --resume
python3 run_github.py --brief "$GITHUB_BRIEF" --status
python3 github/session_orchestrator.py --brief "$GITHUB_BRIEF" --single-session
python3 github/run.py --brief "$GITHUB_BRIEF"  # single-shot, no multi-session cycling
```

### Post-run workflow

Update market intelligence from a finalized run snapshot:

```bash
python3 tools/update_market_intel.py \
  --brief "$LINKEDIN_BRIEF" \
  --run-dir output/runs/linkedin/<brief-id>/<run-stamp>__run-<id> \
  --mode post_run
```

Draft the next version of a brief from a run report:

```bash
python3 -m tools.iterate_brief \
  --brief "$LINKEDIN_BRIEF" \
  --report output/runs/linkedin/<brief-id>/<run-stamp>__run-<id>/run-report.json \
  --search-memory output/runs/linkedin/<brief-id>/<run-stamp>__run-<id>/search_memory-<brief-id>.json \
  --final-judgments output/runs/linkedin/<brief-id>/<run-stamp>__run-<id>/final_judgments.jsonl \
  --output-dir output
```

### Runtime administration

If a run already has `runtime_state.sqlite3`, use the admin surface instead of editing `progress.json` or JSONL files by hand:

```bash
python3 tools/runtime_state_admin.py \
  --output-dir output/state/linkedin/<brief-id> \
  --source linkedin \
  --brief-id <brief-id> \
  rebuild-projections
```

Other supported admin operations include inspecting orphaned attempts, inspecting stop reasons, replaying side effects, requeueing work units, and restarting a specific LinkedIn string.

## Testing

The repo has a broad test suite around runtime state, search intelligence, adapter services, market intelligence, brief iteration, and the shared execution layer.

```bash
make validate
```

For the explicit validation profiles:

- `make validate`
  - repo hygiene checks plus the default green suite
- `make test-default`
  - the default green pytest profile
- `make test-full`
  - the full pytest surface, including heavier dataset replay coverage

The validation policy is documented in [docs/validation-standard.md](docs/validation-standard.md).

## Notes

This is an internal project. The system is intentionally opinionated because it is designed to preserve recruiter judgment alongside automation. The newer parts of the repo reflect that direction: source-packet intake, chief-of-staff orchestration, stronger runtime discipline, clearer operator tooling, richer market-level synthesis, and a tighter loop between what the search learns and how the brief evolves.

## License

Private and internal use only.
