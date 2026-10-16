<!-- provenance: status=canonical era=terminal-first stamped=2026-10-15 -->
# Verification, Mutation Testing, and Static Ratchets

When a codebase is built at high velocity with AI coding agents, line coverage and green test counts are weak signals on their own: a test can execute a function, assert on a mock or a tautology, and pass even when the production function is emptied or its return value is inverted. This repository pairs its offline pytest suite (`404` test modules across `412` Python files under `tests/`) with a build-gated mutation sentinel suite, an on-request AST mutation survey tool, untrusted-input prompt fencing, and static ratchets.

## Build-Gated Mutation Sentinels (`tests/test_mutation_sentinels.py`)

`tests/test_mutation_sentinels.py` runs inside the default `make validate` pytest suite and gates every build. It defines `SENTINELS` (`tests/test_mutation_sentinels.py:40-147`), a table of `20` `(module, qualname, lock)` tuples pairing critical production functions with the specific unit test (`lock`) that must fail when that function's body is replaced at runtime with a no-op stub (`empty_function`, which swaps the attribute for `_empty_like` returning `None` or an awaiting coroutine returning `None`).

`test_every_sentinel_lock_fails_when_its_function_is_emptied` runs a single subprocess (`_run_locks_with_their_functions_emptied`, loading `-p` `tests.test_mutation_sentinels`) that activates `empty_function` around each of the `20` lock tests via `pytest_runtest_protocol` and asserts that every lock test fails (`outcomes.get(lock) != "failed"`). The `20` sentinels cover:

- `linkedin.side_effects`: `LinkedInSideEffectsService.handle_save_decision`
- `linkedin.candidate_review`: `CandidateReviewer._perform_full_save` and `CandidateReviewer._full_evaluate`
- `shared.runtime_state.store`: `RuntimeStateStore.set_candidate_state`, `RuntimeStateStore.finish_attempt_success`, `RuntimeStateStore.finish_attempt_failure`, `RuntimeStateStore.begin_candidate_side_effect`, `RuntimeStateStore.complete_candidate_side_effect`, `RuntimeStateStore.list_terminal_identity_keys`, and `RuntimeStateStore.reconcile_open_attempts`
- `linkedin.evaluation_loop`: `EvaluationLoop.run_full`
- `linkedin.page_review`: `PageReviewer._review_page_batch` and `PageReviewer._review_page_sequentially`
- `linkedin.string_driver`: `StringDriver._process_string_impl`
- `shared.governor`: `SessionGovernor.check_profile_open_or_raise` and `SessionGovernor.can_start_session`
- `linkedin.geography_gate`: `GeographyGateService._apply_session_location_filter` and `GeographyGateService._verify_session_geography_chips`
- `shared.judgment.templates`: `parse_full_evaluation_response` and `parse_facial_response`

## Diagnostic AST Mutation Survey (`tools/mutation_survey.py`)

While `tests/test_mutation_sentinels.py` gates every build on those `20` critical functions, `tools/mutation_survey.py` is an on-request diagnostic CLI (`map`, `run`, `report`, `status`, `prove`) used to audit broader test locks across the repository. It clones the tracked tree into a hermetic git-initialized temporary directory (`build_pristine`, `_init_throwaway_repository`, and `clone_tree`), maps per-test function coverage, and applies two function-level AST operators (`OPERATORS = ("empty_body", "invert_bool")` at `tools/mutation_survey.py:396`):

| Operator | Implementation (`tools/mutation_survey.py`) | Purpose |
|---|---|---|
| `empty_body` | `mutate_empty_body` inserts `return None` (or `return` for generators) at the top of a function body while keeping the original statements below it as unreachable text so file line counts and source substrings stay intact. | Identifies functions that are covered during tests (`covered > 0`) but whose side effects or return values no test actually asserts (`survivor` vs. `lock`). |
| `invert_bool` | `mutate_invert_bool` wraps `return` expressions (`_value_returns`) in `not (` ... `)` on boolean-returning functions (`_bool_return_kind`). | Identifies boolean predicates whose callers or tests only exercise one truth value. |

When a candidate test fails under `empty_body`, the survey runs a behavior-neutral null control (`_null_control`, calling `build_mutant(..., null=True)`, which inserts `pass` at the exact same line offset without altering execution) to filter out brittle tests that merely read source files or line numbers from disk. Each surveyed function is classified into `lock`, `crash_only`, `survivor`, `no_body`, `no_control`, or `unstable`. `tests/test_mutation_survey.py` tests the AST transformations, coverage mapping, and null-control classification offline.

## Untrusted Candidate Content Fencing (`shared/judgment/tool_contracts.py`)

Candidate profiles scraped from LinkedIn, GitHub repositories, and open-web pages contain arbitrary user-authored text that is interpolated into LLM evaluation prompts. To defend against indirect prompt injection (such as a profile embedding fake delimiter tags or judgment instructions), `fence_untrusted` in `shared/judgment/tool_contracts.py` (`shared/judgment/tool_contracts.py:159-169`) wraps untrusted text in `<UNTRUSTED_CANDIDATE_DATA>`, `<UNTRUSTED_EXTERNAL_EVIDENCE>`, or `<candidate_profile>` tags (`_FENCE_TAGS`). Before wrapping, `_neutralize_control_delimiters` (`shared/judgment/tool_contracts.py:150-156`) uses `_CONTROL_DELIMITER_RE` (`shared/judgment/tool_contracts.py:49-54`) to replace any embedded opening or closing fence tag inside the body with `[escaped-delimiter:...]`, preventing a crafted profile from closing its own fence block early. `untrusted_data_instruction` (`shared/judgment/tool_contracts.py:171-178`) and the prompt builders in `shared/judgment/templates.py` instruct the evaluation model to treat everything inside those tags strictly as untrusted candidate data rather than instructions (`tests/test_untrusted_fence_live_paths.py`).

## Static Ratchets and Hygiene Gates

`make validate` runs two static checks directly before invoking `tools/run_validation.py default`, and the remaining static ratchets execute as pytest modules inside the default validation suite:

1. `tools/check_repo_hygiene.py` (run before pytest and tested by `tests/test_validation_tools.py`) checks that no incidental files matching `is_incidental_tracked_path` (`.DS_Store`, `__pycache__`, or `*.bak-*.json` backup files) are tracked in git, summarizes the `config/` brief lifecycle inventory (`summarize_brief_lifecycle`), and verifies that every tracked `.md` document carries a valid `<!-- provenance:` stamp (`doc_provenance_failures`).
2. `tools/flag_ledger.py` (run before pytest with `--as-of` and tested by `tests/test_flag_ledger.py`) parses every `_env_flag()` declaration in `shared/config.py` (`32` flags, `20` default-off). It verifies that every flag is read in `READER_AREAS` (`linkedin`, `shared`, `cloris`, `market_intelligence`) outside `shared/config.py`, and that any default-off flag not enabled by a `tools/launch_*.sh` script carries a non-expired `# gate: CLO-NNN by YYYY-MM-DD` or `# gate: by-design — <reason>` annotation.
3. `tools/lint_role_agnosticism.py` and `tools/lint_vertical_vocab.py` (run inside pytest via `tests/test_role_agnosticism_ratchet.py` and `tests/test_vertical_vocab_ratchet.py`) scan shared and adapter modules to ensure campaign-specific role terms (`tools/role_agnosticism_baseline.txt`) or lexicon-defined vertical vocabulary terms do not leak into role-agnostic platform code.
4. `tools/lint_code_bloat.py` (run inside pytest via `tests/test_code_bloat_ratchet.py`) scans `9` bloat classes (`carried_history`, `narration`, `stale_references`, `duplicated_prose`, `defensive_layering`, `dead_choice`, `duplicate_helpers`, `test_theater`, and `essay_scripts`), ratcheting the `3` classes in `RATCHET_CLASSES` (`carried_history`, `duplicated_prose`, and `essay_scripts`) against `tools/code_bloat_baseline.txt`.
