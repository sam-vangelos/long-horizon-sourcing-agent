<!-- provenance: status=canonical era=terminal-first stamped=2026-10-16 -->
# Technical reference

This is the full technical description of the system, kept for engineers who want every mechanism, constant and file. Until 15 October 2026 it was the repository's README; the [README](../README.md) is now the short version.

An autonomous technical recruiting system that plans search strategies, discovers candidates across LinkedIn Recruiter, GitHub, and the open web, evaluates profiles against structured role briefs using tiered language models, and persists every state transition and external side effect to a local SQLite write-ahead log so multi-hour sessions can resume after interruption.

> Curated public snapshot as of 15 October 2026. Private development history, live campaign briefs, and operator run artifacts are omitted.

## The Operating Problem

Sourcing for specialized technical roles requires running dozens of query variations across different platforms, reading hundreds of candidate profiles, cross-referencing external technical evidence (repositories, publications, open-web profiles), and recording calibrated pass/fail rationales. Doing that work with an autonomous agent introduces four failure modes that ordinary request-response scripts do not face:

1. Sessions run for hours against live browser tabs and rate-limited APIs. A crash, network drop, or session cap halfway through a query must not lose progress, re-spend LLM evaluations on candidates already judged under the same brief, or double-click a "Save to pipeline" button in LinkedIn Recruiter.
2. Unbounded LLM and search API retries can burn through a campaign's dollar budget before post-hoc usage logs notice, especially when concurrent workers dispatch requests in parallel or when an upstream response omits billing metadata.
3. Evaluating scraped profiles with LLMs exposes the judgment pipeline to indirect prompt injection from candidate-authored text and to schema drift when models return malformed or truncated JSON.
4. High-velocity development with AI coding agents can produce tests that pass while asserting very little, or feature flags and role-specific heuristics that accumulate without a retirement path.

## How It Works

Each campaign starts from a structured role brief (`shared/brief_v2_schema.py`, loaded and validated by `shared/brief_loader.py`) that defines the role's capability areas, depth distinctions, experience envelope, employer and alumni exclusions, and source-specific retrieval hints. A source session orchestrator acquires an exclusive lock on the campaign's state directory, reconciles any unfinished work units or open attempts from prior runs, and drives a two-stage evaluation loop (`facial_judge` for fast surface triage followed by `full_judge` for deep rubric evaluation with optional public-web evidence augmentation).

```mermaid
flowchart LR
    Brief["Role Brief V2<br/>(shared/brief_v2_schema.py)"] --> Preflight["Preflight & Strategy<br/>(shared/preflight_v2.py)"]
    Preflight --> Adapters["Source Adapters<br/>linkedin/ · github/ · exa/ · researcher/"]
    Adapters --> Store[("SQLite WAL Store<br/>runtime_state.sqlite3")]
    Adapters --> Facial["Stage 1: Facial Triage<br/>(shared/judger.py)"]
    Facial -->|PASS| Full["Stage 2: Full Rubric Eval<br/>+ External Evidence"]
    Facial -->|REJECT| Store
    Full -->|SAVE_FAMILY_DECISIONS| SideEffects["Side-Effect Ledger<br/>(linkedin/side_effects.py)"]
    Full -->|REJECT| Store
    SideEffects --> Store
    Store --> Projections["JSON / JSONL Projections<br/>& Post-Run Snapshots"]
    Projections --> MI["Market Intelligence &<br/>Brief Iteration"]
```

### Repository Map

The snapshot contains `936` tracked files (`818` Python files totaling `407,502` lines: `406` non-test Python modules totaling `204,357` lines and `412` Python files under `tests/` — `404` `test_*.py` test modules plus `8` test support modules — totaling `203,145` lines).

| Path / Package | Files | Lines | Responsibility |
|---|---:|---:|---|
| `shared/` | 150 | 63,106 | Canonical SQLite runtime state (`shared/runtime_state/store.py`), `fcntl.flock` advisory lock (`shared/runtime_state/lock.py`), LLM clients and spend budgets (`shared/llm_clients.py`, `shared/llm_spend_budget.py`), two-stage judger (`shared/judger.py`), untrusted-content fencing (`shared/judgment/tool_contracts.py`), cross-source identity resolution (`shared/identity_resolution.py`), and session rate governance (`shared/governor.py`). |
| `linkedin/` | 70 | 56,329 | Browser-driven LinkedIn Recruiter adapter over Chrome DevTools Protocol (`linkedin/browser.py`), pluggable CDP and macOS HID input backends (`linkedin/input_backends.py`), Boolean and facet lane compilation (`linkedin/boolean_compiler.py`, `linkedin/lane_compiler.py`), Boolean scoring and enrichment (`linkedin/boolean_score/`, `linkedin/boolean_enrichment.py`, `linkedin/boolean_sidecar.py`), adaptive string driver (`linkedin/string_driver.py`), evaluation loop (`linkedin/evaluation_loop.py`), and idempotent pipeline save actuation (`linkedin/side_effects.py`). |
| `github/` | 44 | 16,205 | Open-source maintainer and contributor discovery across GitHub, crates.io, PyPI, npm, Hugging Face, and OSSF criticality data (`github/orchestrator.py`, `github/acquisition.py`, `github/hubs/`). |
| `exa/` | 22 | 12,750 | Open-web candidate discovery and profile synthesis (`exa/orchestrator.py`, `exa/acquisition.py`), pre-dispatch dollar cost reservation (`exa/cost_meter.py`), and employer/location/do-not-save gates (`exa/scope_gate.py`, `exa/identity.py`). |
| `market_intelligence/` | 22 | 18,792 | Post-run snapshot sealing (`market_intelligence/run_snapshots.py`), cross-run market synthesis (`market_intelligence/engine.py`), and brief calibration feedback (`market_intelligence/calibration_to_brief.py`). |
| `researcher/` | 18 | 5,429 | Academic and research-author sourcing via OpenAlex, Semantic Scholar, and arXiv (`researcher/orchestrator.py`, `researcher/sources/`). |
| `decoy/` | 8 | 527 | Non-Recruiter consumer LinkedIn browsing agent (`feed`, `jobs`, `notifications`) on an isolated tab and autocorrelated log-normal burst scheduler (`decoy/agent.py`, `decoy/scheduler.py`, `decoy/actions/`). |
| `tools/` | 66 | 31,176 | Validation runner (`tools/run_validation.py`), diagnostic AST mutation survey (`tools/mutation_survey.py`), browser-free shadow replay harness (`tools/shadow_replay.py`), death watchdog (`tools/cloris_watchdog.py`), and static ratchets (`tools/flag_ledger.py`, `tools/lint_role_agnosticism.py`, `tools/lint_vertical_vocab.py`, `tools/lint_code_bloat.py`). |
| `tests/` | 492 | 223,897 | Offline unit, integration, replay, and mutation-sentinel test suite (`403` `test_*.py` modules, `8` Python support modules, and `80` JSON/HTML/text fixtures). |
| `cloris/` + `exec_search/` | 11 | 4,238 | Chrome CDP launcher (`cloris/chrome_launcher.py`, `617` lines), intake schema (`cloris/intake_sessions.py`, `352` lines), launcher registry (`cloris/launchers/__init__.py`, `1,781` lines), and executive-search evidence assembly (`exec_search/evidence_assembly.py`, `exec_search/signals/`). |
| `config/` + `data/` | 14 | 2,114 | Neutral role brief fixtures (`config/senior-backend-fintech-fixture/`, `config/senior-product-designer-fixture/`, `config/head-of-applied-ai-fixture/`), test-referenced brief fixtures, and OSSF criticality seed data (`data/ossf_criticality_snapshot.csv`). |
| `scripts/`, `docs/`, and root files | 18 | 1,369 | Source entry points and utilities (`run_github.py`, `reset-github.sh`, `launch-chrome.sh`, `scripts/export_linkedin_review_workbook.py`, `scripts/refresh_ossf_criticality.py`), technical design notes (`docs/`), and build configuration (`Makefile`, `pyproject.toml`, `requirements.txt`, `env.example`). |

## Engineering Worth Inspecting

Rather than reading 204,000 lines of application code sequentially, an engineer reviewing this repository can inspect seven self-contained subsystems where the core reliability, browser automation, budgeting, query enrichment, and verification invariants live.

### 1. Crash-Safe SQLite WAL State and Idempotent External Side Effects

[`shared/runtime_state/store.py`](../shared/runtime_state/store.py), [`shared/runtime_state/lock.py`](../shared/runtime_state/lock.py), and [`linkedin/side_effects.py`](../linkedin/side_effects.py) implement the persistence and `--resume` model (detailed in [`docs/runtime-state-and-recovery.md`](../docs/runtime-state-and-recovery.md)).

`RuntimeStateStore` (`CURRENT_SCHEMA_VERSION = "12"`) enforces candidate lifecycle transitions (`_guard_transition` and `_guard_save_family_reentry` across `TARGET_CANDIDATE_LIFECYCLE`) and brief-hash-aware deduplication (`is_dedup_blocked` at `shared/runtime_state/store.py:1495-1529` and `list_terminal_identity_keys` at `shared/runtime_state/store.py:1428-1456`). Candidates with a decision in `SAVE_FAMILY_DECISIONS` or a recruiter `user_status` stay suppressed permanently, while rejected terminal candidates are suppressed while `brief_content_hash` matches the active brief and become re-eligible when the brief changes. External browser mutations (clicking `Save to pipeline` or attaching a note on a LinkedIn Recruiter card) are coordinated by `LinkedInSideEffectsService.handle_save_decision` (`linkedin/side_effects.py:52-357`) and `RuntimeStateStore.begin_candidate_side_effect` / `complete_candidate_side_effect` (`shared/runtime_state/store.py:2196-2489`) against the `side_effects` table (`UNIQUE(candidate_id, effect_type, idempotency_key)` with `effect_type="linkedin_save"`, `idempotency_key="save"`), paired with pre-click DOM verification (`is_already_saved_on_card`, `assert_expected_identity`, and `assert_project_context`). Process-level mutual exclusion is enforced by `RuntimeStateLock` (`shared/runtime_state/lock.py:11-47`), which acquires a non-blocking `fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)` on `runtime_state.lock` in the campaign output directory that the OS kernel releases automatically if the process exits or crashes.

### 2. Fail-Closed Pre-Dispatch Spend Reservation

[`exa/cost_meter.py`](../exa/cost_meter.py) and [`shared/llm_spend_budget.py`](../shared/llm_spend_budget.py) enforce dollar caps before network requests leave the process (detailed in [`docs/spend-metering-and-governance.md`](../docs/spend-metering-and-governance.md)).

In `ExaClient.search` (`exa/client.py:489-509`) and `ExaClient.get_contents` (`exa/client.py:542-561`), the caller prices the worst-case charge from `exa.config.PRICE_TABLE` via `estimate_search_usd()` or `estimate_contents_usd()` and passes `reserve` and `release` hooks into `ExaClient._request`. `ExaCostMeter.reserve()` (`exa/cost_meter.py:141-165`) commits that estimate under a lock prior to dispatch against the per-run cap (`EXA_RUN_BUDGET_USD`, default `$5.00`), raising `ExaBudgetExhaustedError` if the call would cross `cap_usd`. When the response arrives, `ExaCostMeter.settle()` (`exa/cost_meter.py:185-224`) replaces the reservation with `costDollars.total` only if that field is present and finite, and logs an `exa-usage-v1` receipt (`RECEIPT_SCHEMA_VERSION = "exa-usage-v1"`); if billing telemetry is missing or the connection drops after the request was sent, the worst-case estimate stays committed so missing telemetry never creates phantom budget headroom. `reserve_fireworks_spend()` and `settle_fireworks_spend()` (`shared/llm_spend_budget.py:44-127`) apply the same reservation invariant as an opt-in, process-wide guard (`FIREWORKS_PRIMARY_MAX_COST_USD`, off by default) across concurrent LLM worker threads.

### 3. Untrusted Candidate Input Fencing and Fail-Closed JSON Extraction

[`shared/judgment/tool_contracts.py`](../shared/judgment/tool_contracts.py), [`shared/judgment/templates.py`](../shared/judgment/templates.py), and [`shared/llm_json.py`](../shared/llm_json.py) harden the boundary between scraped web/profile text and LLM evaluators.

Before candidate headlines, summaries, or external web snippets enter a prompt, `fence_untrusted()` (`shared/judgment/tool_contracts.py:159-169`) wraps each field in `<UNTRUSTED_CANDIDATE_DATA>`, `<UNTRUSTED_EXTERNAL_EVIDENCE>`, or `<candidate_profile>` delimiters and rewrites any embedded closing or opening fence tag via `_neutralize_control_delimiters()` (`shared/judgment/tool_contracts.py:150-156`, using `_CONTROL_DELIMITER_RE` at `shared/judgment/tool_contracts.py:49-54` to emit `[escaped-delimiter:...]`) so a crafted profile cannot break out of its data block. `untrusted_data_instruction()` (`shared/judgment/tool_contracts.py:171-178`) injects the system-prompt rule that text inside those tags is data to evaluate, never instructions to follow. On the output side, `extract_json()` (`shared/llm_json.py:47-89`) uses `_DECODER.raw_decode` and `_closing_index()` (`shared/llm_json.py:19-44`) from the first `{` or `[` and raises `json.JSONDecodeError` on unclosed brackets rather than attempting heuristic brace repair.

### 4. Browser Automation, Input Backends, and Session Pacing

[`linkedin/input_backends.py`](../linkedin/input_backends.py), [`linkedin/browser.py`](../linkedin/browser.py), [`shared/human_timing.py`](../shared/human_timing.py), [`decoy/`](../decoy/agent.py), [`shared/governor.py`](../shared/governor.py), and [`linkedin/posture_report.py`](../linkedin/posture_report.py) govern how a multi-hour browser session interacts with LinkedIn Recruiter over Chrome DevTools Protocol (detailed in [`docs/browser-automation-and-session-pacing.md`](../docs/browser-automation-and-session-pacing.md)).

Pointer movement, scrolling, and text entry route through `InputBackend` (`linkedin/input_backends.py:315-360`), which supports both `ConcurrentInputBackend` (Playwright mouse/keyboard dispatch paired with `python_ghost_cursor` Bézier trajectories bridged onto `rebrowser_playwright` via `_alias_playwright_to_rebrowser`) and `AwayInputBackend` (macOS CoreGraphics HID event injection via `CGEventPost` on `_CoreGraphicsBridge` with dynamic window-chrome offset calculation in `_locator_screen_point` and cubic Bézier cursor stepping in `_move_mouse`). In `LinkedInBrowser._ghost_click_locator` (`linkedin/browser.py:1210-1270`), elements are scrolled into view before trajectory planning, and the candidate identity guard (`before_click`) executes after cursor arrival and immediately before `page.mouse.down()`. Typing plans (`build_boolean_typing_plan` and `build_note_typing_plan`) model operator pauses (`_BOOLEAN_OPERATORS`), mid-query thinking pauses, bounded typos (`_random_alpha_replacement`) with `Backspace` recovery, home-row key-hold dwell (`_sample_key_dwell`), and fast-bigram rollover (`_is_fast_bigram`) while folding key-hold duration inside inter-key intervals (`interval_includes_dwell=True`) so a 500-character note stays near 53 seconds.

Session timing and volume are bounded across four cooperating mechanisms: `human_delay` and `human_delay_correlated` (`shared/human_timing.py:31-79`) sample log-normal and lag-1 autocorrelated (`_CORRELATION_WEIGHT = 0.3`) delays across isolated named channels; `simulate_profile_read` and `_plan_section_traversal` (`linkedin/browser.py:4759-4908`) plan section-directed traversals (`_TRAVERSAL_STYLES`: `straight`, `backtracker`, `experience_first`, `overshoot`, weighted by `_TRAVERSAL_WEIGHTS`) across `about`, `experience`, and `education` headings with interest-scaled budgets clamped to `_MAX_CHUNK_DWELL = 8.0` seconds per section and `_MAX_PROFILE_READ = 35.0` seconds per profile; `DecoyAgent` (`decoy/agent.py`) and `BurstScheduler` (`decoy/scheduler.py`) interleave read-only bursts (`next_interval`) across consumer LinkedIn (`jobs`, `feed`, `notifications`) on a dedicated tab (`_ensure_tab`) via a cooperative `pause_requested` / `resume_event` checkpoint handoff (`_interleave_loop` in `linkedin/session_orchestrator.py:547-591`); and `SessionGovernor` (`shared/governor.py`) and `GitHubGovernor` (`github/governor.py`) enforce per-session (`MAX_PROFILE_OPENS_PER_SESSION = 200`) and rolling 24-hour (`MAX_PROFILE_OPENS_PER_24H = 400`, `cooldown.get_profile_opens_24h`) profile-open caps, session duration limits (`session_duration_seconds` plus `GOVERNOR_SESSION_BACKSTOP_GRACE_SECONDS = 600`, or `3.5–4.5h` when no duration is passed), and fail-closed `6`-hour rate-limit backoffs (`force_backoff` and `record_forced_backoff` in `shared/cooldown.py`). At session startup, `describe_posture` and `format_posture` (`linkedin/posture_report.py`) log an 8-check `[ok]` / `[MISSING]` verification report, while `compute_fingerprint` (`tools/build_fingerprint.py`) hashes bundled Python source files into a deterministic SHA-256 digest.

### 5. Read-Only Death Watchdog over Live WAL Databases

[`tools/cloris_watchdog.py`](../tools/cloris_watchdog.py) is a standalone health monitor designed to run from `launchd` every two minutes without interfering with active sourcing processes.

Opening an idle SQLite WAL database in standard read-only mode can still create `-wal` and `-shm` sidecar files or contend with a writer. `open_readonly()` (`tools/cloris_watchdog.py:342-357`) inspects whether `-wal` and `-shm` files exist on disk, opens the database with `mode=ro` when WAL sidecars are present or `immutable=1` when none exist, and sets `PRAGMA query_only=ON`. It correlates `runs` rows against open file descriptors on `runtime_state.lock` via `/usr/sbin/lsof` and per-PID heartbeat files (`read_beats`), suppresses false-positive stall alarms during governor cooldown windows, and collapses repeated failure bursts to at most one notification per hour per `(status, reason, process_state)`.

### 6. Browser-Free Shadow Replay Harness

[`tools/shadow_replay.py`](../tools/shadow_replay.py) and [`tests/test_shadow_replay_matrix.py`](../tests/test_shadow_replay_matrix.py) allow prompt and model changes on the strategy, facial, and full evaluation seats to be replayed against captured briefs and profiles without launching a browser or mutating live campaign state (`tools/shadow_replay.py` dispatches live provider calls costing roughly `$1–3` by default, while `tests/test_shadow_replay_matrix.py` tests the matrix runner offline). The matrix runner validates manifest SHA-256 hashes, enforces worst-case dollar caps before dispatching provider calls, isolates per-arm environment variables, and records structured comparison receipts.

### 7. Boolean Query Scoring, Lexicon Enrichment, and Sidecar Integration

[`linkedin/boolean_score/`](../linkedin/boolean_score/score.py), [`linkedin/boolean_enrichment.py`](../linkedin/boolean_enrichment.py), and [`linkedin/boolean_sidecar.py`](../linkedin/boolean_sidecar.py) evaluate and rewrite LinkedIn Recruiter Boolean queries during strategy formation (`_finalize_execution_plan` at `linkedin/strategy.py:1247-1305`) and mid-run adaptation (`_annotate_adaptation_metadata` at `linkedin/strategy.py:707-723`, detailed in [`docs/boolean-scoring-and-enrichment.md`](../docs/boolean-scoring-and-enrichment.md)).

`parse()` (`linkedin/boolean_score/parser.py`) validates LinkedIn Recruiter syntax (flagging `wildcard_operator`, `malformed_operator`, `not_group_contains_and`, and 9 structural warning codes), and `score()` (`linkedin/boolean_score/score.py`) maps terms against a `207`-concept role lexicon (`linkedin/boolean_score/lexicon.py`, `linkedin/boolean_score/lexicons/research-engineer-post-training.json`) to compute $\text{composite} = \text{parsimony} \times \text{SHAPE\_SCORES}[\text{shape}] \times \text{weighted\_mean}(\text{register\_coverage}, \text{variant\_coverage}, \text{cohesion}, \text{relevance})$ (`WEIGHTS = {"register_coverage": 0.30, "variant_coverage": 0.15, "cohesion": 0.10, "relevance": 0.05}`, `PARSIMONY_DECAY = 1.5` over 12 `BLOAT_COEFFICIENTS`). When `LINKEDIN_LEXICON_ENRICHMENT_ENABLED` (`shared/config.py:446-447`, off by default) is enabled, `enrich_plan()` and `enrich_adaptation()` call `enrich()` (`linkedin/boolean_score/enrich.py`), which expands positive `OR` groups with missing `common_forms` and morphology/hyphen twins (`twins_for`) up to `DEFAULT_MAX_CHARS = 2000`, removes superseded `rare` forms (`_removable_rare`), and verifies group-structure preservation (`_structure_preserved`).

For queries the dictionary enricher cannot reach — unseen surface terms, thin concept runs inside multi-concept `OR` groups, and unanchored `lone_sweep` strings missing a class anchor — `enrich_plan_with_sidecar()` and `enrich_adaptation_with_sidecar()` (`linkedin/boolean_sidecar.py`, controlled by `LINKEDIN_ENRICHER_SIDECAR_ENABLED` at `shared/config.py:458-459`, off by default) query a local HTTP sidecar (`POST /enrich` on `127.0.0.1:8765`, `DEFAULT_SAMPLES = 8`). The companion repository [`boolean-search-rl`](https://github.com/sam-vangelos/boolean-search-rl) trains the underlying `Qwen/Qwen2.5-3B-Instruct` LoRA policy with GRPO on a stratified `158`/`49` concept split and serves the merged `Q8_0` GGUF via `llama-server`. Inside `_improve()` (`linkedin/boolean_sidecar.py:700-780`), every sidecar proposal is filtered for morphological padding (`_padded`, `NEAR_DUPLICATE_DISTANCE = 2`), lexical root drift (`_off_root`, `_drops_a_qualifier`, `_unrooted_anchor`), and structural preservation (`_signature`), and is accepted by `_best()` **only if** `report.log_composite > current.log_composite + MIN_GAIN` (`MIN_GAIN = 0.01`), with a two-strike circuit breaker (`_Breaker`) that falls back cleanly if the sidecar is down.

## How Correctness Is Verified

Verification is built around four layers (detailed in [`docs/verification-and-mutation-testing.md`](../docs/verification-and-mutation-testing.md)):

1. The offline pytest suite (`403` `test_*.py` test modules across `412` `.py` files under `tests/`) runs without network access, API keys, or a browser session.
2. `tests/test_mutation_sentinels.py` gates every build in `make validate`: it pairs `20` critical production functions (`SENTINELS` at `tests/test_mutation_sentinels.py:40-147`) with lock tests and verifies in a subprocess that emptying each function via `empty_function` causes its lock test to fail. For broader diagnostic audits, `tools/mutation_survey.py` clones the repo into a hermetic temporary directory, applies two function-level AST operators (`OPERATORS = ("empty_body", "invert_bool")`), and runs a behavior-neutral null control (`mutate_empty_body(..., null=True)`) to verify that failing tests are true behavioral locks rather than source-line pins.
3. `tools/flag_ledger.py` checks every `_env_flag()` in `shared/config.py` (`32` flags, `20` default-off) to verify that every flag is read in `READER_AREAS` outside `shared/config.py` and that every default-off flag not enabled by a `tools/launch_*.sh` script carries a non-expired `# gate: CLO-NNN by YYYY-MM-DD` or `# gate: by-design — <reason>` annotation.
4. Static ratchets enforce repository hygiene (`tools/check_repo_hygiene.py`, checking `.DS_Store`, `__pycache__`, `*.bak-*.json`, brief lifecycle inventory, and markdown provenance headers), role-agnosticism (`tools/lint_role_agnosticism.py`, tested by `tests/test_role_agnosticism_ratchet.py`), lexicon-based vertical-vocabulary neutrality (`tools/lint_vertical_vocab.py`, tested by `tests/test_vertical_vocab_ratchet.py`), and code-bloat ceilings (`tools/lint_code_bloat.py`, tested by `tests/test_code_bloat_ratchet.py`, scanning `9` classes and ratcheting `carried_history`, `duplicated_prose`, and `essay_scripts` against `tools/code_bloat_baseline.txt`).

## Running the Credential-Free Validation Suite and Source Entry Points

From a fresh clone on Python 3.12+:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m rebrowser_playwright install chromium   # add --with-deps on Linux
CLORIS_SKIP_STARTUP_VALIDATION=1 make validate
```

`make validate` runs `tools/check_repo_hygiene.py` and `tools/flag_ledger.py` (`--as-of` `2026-10-15`) before invoking `tools/run_validation.py default` (which runs the offline `pytest` suite, including `tests/test_mutation_sentinels.py` and the static ratchet test modules). No `.env` file, API keys, network connection or signed-in browser is required; the browser tests drive the headless Chromium installed above, and `CLORIS_SKIP_STARTUP_VALIDATION=1` skips the startup check for live API keys.

To run individual verification tools directly:

```bash
.venv/bin/python tools/flag_ledger.py --as-of 2026-10-15
.venv/bin/python tools/lint_role_agnosticism.py
.venv/bin/python tools/lint_vertical_vocab.py
.venv/bin/python tools/lint_code_bloat.py
.venv/bin/pytest tests/test_mutation_sentinels.py tests/test_mutation_survey.py -q
```

To configure credentials for live adapter runs, copy `env.example` to `.env` and populate the provider keys for the sources you intend to run (`ANTHROPIC_API_KEY`, `FIREWORKS_API_KEY`, `GITHUB_TOKEN`, `EXA_API_KEY`, and `CDP_URL` for an authenticated Chrome DevTools Protocol session started via `./launch-chrome.sh`, which runs `python -m cloris.chrome_launcher`). Neutral example role briefs live under `config/senior-backend-fintech-fixture/`, `config/senior-product-designer-fixture/`, and `config/head-of-applied-ai-fixture/` (the remaining directories under `config/` are retained because offline tests reference them). Each source adapter is started via its CLI entry point (`linkedin/session_orchestrator.py`, `exa/run.py`, `run_github.py`, or `researcher/session_orchestrator.py`) and accepts `--brief <path>` and `--resume`.

## What This System Deliberately Does Not Do

This system is a candidate discovery and evaluation engine, not an applicant tracking system (ATS) or an automated outreach mailer.

1. It never logs into LinkedIn or stores platform passwords; `linkedin/browser.py` attaches over localhost CDP to a browser session the operator has already authenticated.
2. It never sends automated messages, connection requests, or emails to candidates. In the LinkedIn adapter, Recruiter pipeline mutations are limited to saving a candidate (`Save to pipeline`) and optionally attaching an evaluation note, while `decoy/` performs read-only browsing (`scroll_feed`, `browse_jobs`, `check_notifications`) on consumer LinkedIn in a separate tab. In the GitHub adapter, `github/outreach.py` generates LLM outreach drafts that are stored locally and never sent; Excel review workbooks are exported by `scripts/export_linkedin_review_workbook.py`.
3. It does not auto-apply post-run brief edits. `market_intelligence/` and `tools/iterate_brief.py` generate proposed brief revisions and market synthesis reports as separate artifacts for human review.
4. An earlier local web UI (`cloris/`) was retired in August 2026 in favor of terminal-first operation; only `cloris/chrome_launcher.py` (`617` lines), `cloris/intake_sessions.py` (`352` lines), and `cloris/launchers/__init__.py` (`1,781` lines) remain in tree for Chrome CDP startup and runtime compatibility with the SQLite store and post-run snapshot reader.

## How This System Was Built

This system was built by a technical recruiter (not a traditional software engineer) by directing AI coding agents (Claude Code, Codex CLI, and Cursor) under an explicit specification, adversarial code review, and mutation-tested verification workflow. Design specifications, schema contracts, and failure-mode analyses were authored before implementation slices; every slice was gated by offline unit and replay tests, static ratchets, and the `20` mutation sentinels in `tests/test_mutation_sentinels.py` (supplemented by diagnostic AST mutation sweeps via `tools/mutation_survey.py`) to verify that tests fail when production invariants are broken.

## License

MIT License. See [`LICENSE`](../LICENSE).
