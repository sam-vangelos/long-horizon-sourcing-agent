<!-- provenance: status=canonical era=terminal-first stamped=2026-10-15 -->
# Spend Metering and Session Governance

Long-running sourcing campaigns issue hundreds of LLM evaluations, search API queries, and browser profile opens across a run. Without pre-dispatch accounting, concurrent worker tasks or missing provider usage telemetry can overrun a dollar cap or trip platform rate limits before an after-the-fact counter notices. The metering and governance layer enforces limits before requests leave the process.

## Pre-Dispatch Worst-Case Spend Reservation

Both the open-web sourcing adapter (`exa/cost_meter.py`, per-run cap) and the Fireworks inference client (`shared/llm_spend_budget.py`, opt-in process-wide cap) use a fail-closed reservation protocol rather than post-hoc tallying.

### Exa API Cost Meter (`exa/cost_meter.py`)

`ExaCostMeter` guards `/search` and `/contents` calls against a per-run dollar ceiling (`cap_usd`, defaulting to `EXA_RUN_BUDGET_USD` at `$5.00` in `shared/config.py`):

1. In `ExaClient.search` (`exa/client.py:489-509`) and `ExaClient.get_contents` (`exa/client.py:542-561`), the caller computes the worst-case dollar cost from `exa.config.PRICE_TABLE` via `self._meter.estimate_search_usd(...)` or `self._meter.estimate_contents_usd(...)` and passes `reserve=lambda: self._meter.reserve(estimate)` and `release=lambda: self._meter.release(estimate)` into `ExaClient._request` (`exa/client.py:305-418`). Because Exa's `/contents` endpoint serves and bills default text even when `contents` is empty or `None`, `estimate_contents_usd` (`exa/cost_meter.py:120-137`) floors the billable content-type count at `1` so a caller cannot bypass the cap with an untyped request.
2. Inside `ExaClient._request`, `reserve()` is invoked before every attempt. `ExaCostMeter.reserve(estimate_usd)` (`exa/cost_meter.py:141-165`) acquires `self._lock`, computes `projected = self._committed_usd + estimate`, and checks whether `projected > self._cap_usd + _EPSILON`. If the attempt would cross the cap, it raises `ExaBudgetExhaustedError` (a subclass of `ApiBudgetExhaustedError`) before any network I/O occurs.
3. If the HTTP connection fails before the request leaves the machine (DNS, TCP, or TLS faults in `_NEVER_SENT_FAULTS`) or the server returns a non-2xx response whose body carries no usage record (`not _error_reports_usage(text)`), `ExaClient._request` calls `release()`, and `ExaCostMeter.release(estimate_usd)` (`exa/cost_meter.py:167-184`) refunds that attempt's estimate. If a transport fault occurs after the request was transmitted (such as a disconnect or read timeout), the reservation stays committed because Exa may have executed and billed the query.
4. When a `200` response returns, `ExaCostMeter.settle()` (`exa/cost_meter.py:185-224`) extracts `costDollars.total` from the response body. If `costDollars.total` is a finite non-negative number, `settle()` replaces `estimate_usd` with `actual_usd`; if the response omits `costDollars`, the worst-case `estimate_usd` remains committed so missing telemetry never fabricates free budget headroom. Each settled call appends an `exa-usage-v1` receipt (`RECEIPT_SCHEMA_VERSION = "exa-usage-v1"`) to `exa-usage.jsonl`.

### Fireworks LLM Spend Reservation (`shared/llm_spend_budget.py`)

`reserve_fireworks_spend()` (`shared/llm_spend_budget.py:44-95`) provides an opt-in, process-wide pre-dispatch guard across concurrent Fireworks LLM calls (`FIREWORKS_PRIMARY_MAX_COST_USD` in `shared/config.py`, which defaults to `0` / disabled and is enabled when set to a positive dollar cap):

1. Given `cap_usd`, `model`, `input_token_upper_bound`, `max_output_tokens`, and `max_attempts`, `reserve_fireworks_spend()` multiplies the token bounds by `max_attempts` and prices the worst-case logical call via `estimate_usage_cost_usd()` (`shared/llm_usage.py`). If `model` has no verified price entry (`rate_source == "unknown"`), reservation fails immediately with `ApiBudgetExhaustedError`.
2. Under `_LOCK`, the guard computes `projected = _COMMITTED_USD + sum(_ACTIVE.values()) + reservation_usd` and checks `projected > cap + 1e-12` so concurrent in-flight worker reservations cannot collectively exceed `cap_usd`. When concurrent facial workers share a `threading.Barrier` (`FIREWORKS_SPEND_COHORT_CONTEXT_KEY`), `synchronize_fireworks_spend_cohort()` (`shared/llm_spend_budget.py:130-180`) releases the cohort only if every worker in the batch funded its reservation, aborting and refunding all siblings at zero cost if any worker fails to reserve.
3. When the call finishes, `settle_fireworks_spend()` (`shared/llm_spend_budget.py:98-127`) pops the active reservation and commits `measured_cost_usd` only when `usage_complete` is `True`; partial or missing provider usage commits the full worst-case `reserved_usd`.
4. Across all LLM seats (`shared/judger.py`), `ApiBudgetExhaustedError` is re-raised out of candidate evaluation loops rather than absorbed as a candidate-level judgment failure, halting the run cleanly at the budget boundary.

## Session Rate and Cooldown Governance (`shared/governor.py`)

Browser-driven LinkedIn Recruiter sessions are bounded by `SessionGovernor` (`shared/governor.py`) backed by persistent cross-session state in `~/.sourcing-governor/daily_stats.json` (`shared/cooldown.py`).

| Constant / Mechanism | Value / Behavior | Location |
|---|---|---|
| `MAX_PROFILE_OPENS_PER_SESSION` | Hard ceiling of `200` profile opens in a single session invocation. | `shared/governor.py:29` |
| `MAX_PROFILE_OPENS_PER_24H` | Rolling 24-hour ceiling of `400` profile opens across all sessions on the host (`cooldown.get_profile_opens_24h`). | `shared/governor.py:30` |
| `GOVERNOR_LEGACY_SESSION_DURATION_MIN_SECONDS` / `MAX_SECONDS` and `GOVERNOR_SESSION_BACKSTOP_GRACE_SECONDS` | In `SessionGovernor.start_session(session_duration_seconds)`, if `session_duration_seconds` is `None`, the governor draws a random limit between `3.5` and `4.5` hours (`12,600` to `16,200` seconds). When the caller provides `session_duration_seconds` (for example, `linkedin/session_orchestrator.py` samples `4.0` to `5.0` hours via `_sample_session_duration`), the governor sets its hard backstop to `float(session_duration_seconds) + GOVERNOR_SESSION_BACKSTOP_GRACE_SECONDS` (`+600` seconds). | `shared/governor.py:26-28`, `shared/governor.py:111-126` |
| `GovernorStateUnreadable` | Fail-closed startup guard: if `daily_stats.json` is corrupt or unreadable, `cooldown._load_raw` raises `GovernorStateUnreadable` and `SessionGovernor.can_start_session` refuses to start rather than resetting counters to zero. | `shared/cooldown.py:24-58`, `shared/governor.py:106-107` |
| `force_backoff` and `record_forced_backoff` | When a block or rate-limit signal is detected (`blocked_or_rate_limited`), `SessionGovernor.force_backoff` immediately trips the in-memory governor (`GovernorLimitReached`) and persists a `forced_backoff` window (`RATE_LIMIT_BACKOFF_SECONDS = 6 * 3600` by default) to `daily_stats.json` so subsequent sessions refuse to start until the cooldown expires. | `shared/governor.py:160-183`, `shared/cooldown.py:198-224` |

## Multi-Provider LLM Usage Telemetry (`shared/llm_usage.py`)

Every LLM call routed through `shared/llm_clients.py` records a structured usage event to `token-cost-log.jsonl` in the campaign state directory via `record_llm_usage()` (`shared/llm_usage.py`). Each entry captures the provider, effective model identifier, stage (`strategy`, `facial`, `full_eval`, `shadow_facial`, `shadow_full`), prompt and completion token counts, cache-read and cache-creation token breakdowns (for Anthropic prompt caching), latency in milliseconds, and estimated dollar cost. At the end of a session, `shared/cost_rollup.py` aggregates `pipeline_end` and token-cost records so run snapshots and operator reports reflect measured per-run spend.
