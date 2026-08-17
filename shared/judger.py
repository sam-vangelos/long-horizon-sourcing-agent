"""Opus judgment: snippet -> OpusDecision, summary -> OpusDecision.

Prompts are built dynamically from the brief — not hardcoded.
The brief is the single source of truth for evaluation criteria.

V2 briefs use structural templates from judgment_templates.py (claim-and-evidence
procedure with capability mapping + depth test). Old briefs use the original
prompt builders below.
"""

from __future__ import annotations
import concurrent.futures
import contextvars
import functools
import hashlib
import inspect
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence
import shared.config as config
from shared.failures import (
    ApiBudgetExhaustedError,
    is_api_budget_exhausted_error,
    is_failure_decision as _is_failure_decision,
    judgment_failure_decision,
    parse_failure_decision,
)
from shared.schemas import (
    CandidateSnippet,
    CandidateProfileSummary,
    ExternalCandidateEvidence,
    OpusDecision,
)
from shared.llm_clients import (
    _parse_json_response,
    facial_llm,
    opus_llm,
    opus_llm_cached,
    shadow_facial_llm,
    shadow_full_llm,
)
from shared.llm_usage import current_llm_usage_log_path
from shared.llm_policy import (
    FireworksStagePolicy,
    build_fireworks_prompt_cache_key,
)
from shared.brief_loader import Brief
from shared.observability import (
    get_current_observation_id,
    get_current_trace_id,
    get_trace_url,
    observe,
)
from shared.receipts import ReceiptStatus, build_receipt
from shared.storage import log_event
from shared.judgment.templates import (
    assemble_facial_prompt,
    assemble_facial_system,
    assemble_facial_tool_system,
    assemble_full_evaluation_prompt,
    assemble_full_evaluation_system,
    assemble_full_evaluation_tool_system,
    assemble_facial_batch_system,
    _facial_ternary_selected,
    defang_wire_format,
    FacialResult,
    parse_facial_response,
    parse_facial_batch_response,
    parse_full_evaluation_response,
)
from shared.judgment.tool_contracts import (
    FACIAL_CONTRACT_VERSION,
    FULL_CONTRACT_VERSION,
    JudgmentToolContractError,
    facial_tool_contract,
    full_tool_contract,
    generate_opaque_candidate_ids,
    render_facial_tool_user_message,
    render_full_tool_user_message,
    validate_facial_tool_arguments,
    validate_full_tool_arguments,
)

logger = logging.getLogger(__name__)


def _reraise_if_budget_exhausted(e: Exception) -> None:
    """Re-raise provider credit/budget-exhaustion errors so the run pauses.

    Phase-0 shared root. Every judge wraps its LLM call in
    ``except Exception: return judgment_failure_decision(...)``. That absorbs a
    dead/exhausted API key into a ``JUDGMENT_FAILURE`` decision, which the
    runtime treats as RETRYABLE — so the loop re-hits the dead key for every
    remaining candidate for hours and the run finalizes ``status='completed'``.
    Calling this as the FIRST line of each judge except-handler converts the
    budget case into a raise, which the orchestrator's
    ``is_api_budget_exhausted_error`` guard catches to pause the run.

    Scoped to budget/credit errors ONLY: every other exception falls through
    and is absorbed into ``judgment_failure_decision`` exactly as before. This
    is deliberately NOT centralized inside ``judgment_failure_decision`` because
    candidate-level external-evidence calls it and MUST NOT pause the run
    (``shared/external_evidence/provider.py`` contract).
    """

    if is_api_budget_exhausted_error(e):
        raise ApiBudgetExhaustedError(str(e)) from e

# Valid decisions for old-brief prompt contracts.
# Step B of the FACIAL_BORDERLINE promotion plan widens this set to match
# the parser's vocabulary so an old-brief code path that receives a
# ``FACIAL_BORDERLINE`` (e.g. a future config experiment) does not route
# through ``parse_failure_decision``. Persistence stays binary because
# the orchestrator translates BORDERLINE -> YES at the parser-output
# boundary; this set is the validator gate, not the persistence gate.
_VALID_FACIAL = {"FACIAL_YES", "FACIAL_NO", "FACIAL_BORDERLINE"}
# P4 widens the legacy JSON full-eval validator so REVIEW_INFERRED /
# REVIEW_FLAGGED don't route through parse_failure on the old-brief
# code paths. The V2 structural path constructs ``OpusDecision`` directly
# from the parser output (which has its own DECISION matcher) and does
# not consult ``_VALID_FULL``; the legacy JSON paths do.
_VALID_FULL = {"SAVE", "REJECT", "REVIEW_INFERRED", "REVIEW_FLAGGED"}


# ---------------------------------------------------------------------------
# GLM-5.2 shadow judge (Fireworks) — facial-stage AND full-eval A/B
# instrumentation, gated by the SAME config.SHADOW_FACIAL_MODEL_ENABLED
# flag (one experiment, one switch — no second flag was added for the
# full-eval extension).
#
# Design doctrine: the shadow verdict is RECORDED AND COMPARED but has ZERO
# influence on the returned OpusDecision. Every function below is called
# strictly for its side effect (one shadow LLM call + one log_event) after
# the real verdict already exists; none of them return a value the caller
# consumes, which is what makes the zero-influence property structural
# rather than merely tested-for. They are also individually and
# collectively fail-soft: no exception raised in here is allowed to reach
# facial_judge / facial_judge_batch / full_judge's caller.
#
# Scope of the full-eval extension: LinkedIn's V2-structural full_judge
# branch ONLY (assemble_full_evaluation_system + parse_full_evaluation_
# response — the same pairing the primary verdict uses). Two things are
# deliberately NOT covered, matching the existing facial precedent of only
# shadowing what's actually live:
#   - full_judge's legacy old-brief branch: it does its own ad-hoc JSON
#     parsing (_VALID_FULL) rather than parse_full_evaluation_response, so
#     "parse with the SAME parser the primary uses" doesn't apply to it
#     without inventing a second, untested legacy parser (facial's
#     equivalent, _parse_legacy_shadow_decision, exists because the facial
#     ask was symmetric across both branches; the full-eval ask was not).
#   - github.judger's github_full_judge: a different module, different
#     evidence shape, explicitly out of scope.
#
# Where the event lands: shared.storage.log_event() takes an explicit path
# (usually the orchestrator's self.log_path == self.output_dir /
# "run_log.jsonl"), but judger.py is a layer below the orchestrator and has
# no such handle. Every real run entry point (linkedin/orchestrator.py
# Orchestrator.run() and run_full()) opens shared.llm_usage.llm_usage_session
# at self.output_dir / "token-cost-log.jsonl" before any candidate is
# judged, so current_llm_usage_log_path() is available and its parent
# directory is exactly self.output_dir. _shadow_run_log_path() derives the
# sibling run_log.jsonl from that ContextVar rather than plumbing a new
# parameter through every facial_judge call site. Outside of a real run
# (unit tests, rejudge_from_file, ad-hoc facial_judge() calls with no usage
# session open) there is nowhere honest to log to, so the shadow hook is a
# no-op — it still never touches the primary decision either way.
def _shadow_run_log_path() -> Path | None:
    usage_log_path = current_llm_usage_log_path()
    if usage_log_path is None:
        return None
    return usage_log_path.parent / "run_log.jsonl"


# Fire-and-forget dispatch (config.SHADOW_ASYNC_ENABLED, default on). The
# GLM shadow call has ~58s mean latency with outliers to ~240s; on the
# 2026-07-05 live run 33 synchronous shadow calls blocked roughly 30-60
# minutes of a 166-minute session. Nothing downstream consumes the
# comparison in the run itself (it is offline analytics: one run-log event
# + one token-cost row), so the whole comparison body moves to a SINGLE
# background worker thread. Why exactly one worker:
#   - shadow run-log/token-cost writes stay serialized with EACH OTHER,
#     preserving today's per-comparison event ordering;
#   - at most one shadow call is in flight, so the experiment cannot
#     stampede Fireworks or starve the primary path of connections.
# Against the MAIN thread's run-log writes there is no lock to share:
# shared.storage.append_jsonl (the sink under log_event) opens the file
# per call in "a" mode and writes the whole line in one f.write, so
# cross-thread interleaving protection is POSIX O_APPEND append
# atomicity — the same per-line guarantee every existing log_event caller
# already relies on. A judger-local lock could not serialize against the
# orchestrator's own log_event calls anyway (they don't route through
# this module), so none is taken.
#
# Context propagation: _shadow_run_log_path() and record_llm_usage()
# both resolve their sinks via ContextVars (shared/llm_usage.py), and a
# fresh worker thread starts with an EMPTY context — without care the
# shadow would silently no-op off the hot path. Every task is therefore
# submitted through contextvars.copy_context().run so it sees the exact
# context of the judge call that enqueued it, including after the main
# thread's llm_usage_session has closed (the copy is a snapshot; the
# session's reset applies to the main thread's context only). Same idiom
# as shared/intake_conversation/sse_bridge.py's producer thread.
_shadow_executor: concurrent.futures.ThreadPoolExecutor | None = None
_shadow_executor_lock = threading.Lock()
_shadow_pending: set[concurrent.futures.Future] = set()
_shadow_pending_lock = threading.Lock()


def _get_shadow_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazily create the single-worker shadow executor (module lifetime)."""
    global _shadow_executor
    with _shadow_executor_lock:
        if _shadow_executor is None:
            _shadow_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="shadow-judge",
            )
        return _shadow_executor


def _shadow_task_done(future: concurrent.futures.Future) -> None:
    with _shadow_pending_lock:
        _shadow_pending.discard(future)
    exc = future.exception()
    if exc is not None:  # pragma: no cover — comparison bodies are fail-soft
        logger.warning(
            "shadow-judge comparison task failed (primary unaffected): %s", exc
        )


def _dispatch_shadow(task: Callable[[], None]) -> None:
    """Run one shadow comparison — background worker, or inline when
    ``config.SHADOW_ASYNC_ENABLED`` is off (byte-identical to the
    pre-executor behavior; the rollback lever).

    ``task`` is a fully-bound zero-arg callable (one of the
    ``_run_*_shadow_*_sync`` comparison bodies below). Those bodies are
    fail-soft by construction — everything runs inside their own
    try/except — so nothing propagates to the judge caller on either
    path; the done-callback warning above is belt-and-braces for a
    defect in a comparison body itself, mirroring its log line.
    """
    if not config.SHADOW_ASYNC_ENABLED:
        task()
        return
    ctx = contextvars.copy_context()
    future = _get_shadow_executor().submit(ctx.run, task)
    with _shadow_pending_lock:
        _shadow_pending.add(future)
    # Registered AFTER the pending-set add so a task that finishes
    # instantly still gets discarded (the callback fires immediately when
    # the future is already done) instead of leaking a set entry.
    future.add_done_callback(_shadow_task_done)


def drain_shadow_comparisons(timeout: float | None = None) -> bool:
    """Block until every queued shadow comparison has finished.

    Returns True when the queue fully drained (or was already empty),
    False on timeout with tasks still pending. Never raises on task
    failure — the comparison bodies are fail-soft and anything that
    somehow escapes one is logged by ``_shadow_task_done``, not
    re-raised here.

    judger.py itself has no run-finished seam (it is per-candidate judge
    functions; the run lifecycle lives in the orchestrator), so the
    intended caller is linkedin/orchestrator.py's report build,
    immediately before ``_shadow_facial_summary()`` /
    ``_shadow_full_summary()`` read the run log — draining there makes
    the report count every comparison the run enqueued.
    """
    with _shadow_pending_lock:
        pending = list(_shadow_pending)
    if not pending:
        return True
    _done, not_done = concurrent.futures.wait(pending, timeout=timeout)
    return not not_done


def _facial_shadow_call(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    usage_context: dict | None,
    capture: dict | None = None,
) -> tuple[str | None, float | None, str | None]:
    """Call the shadow model once. NEVER raises.

    Returns ``(raw_text, latency_ms, error)`` — exactly one of
    ``raw_text``/``error`` is non-None on return.

    Tags ``usage_context["shadow_stage"] = "facial_shadow"`` so the
    resulting token-cost-log.jsonl row is distinguishable from a
    full-eval shadow row sharing the same ``provider="fireworks"`` —
    see ``_full_shadow_call``'s counterpart tag and
    linkedin/orchestrator.py's ``_shadow_facial_summary`` /
    ``_shadow_full_summary``, which each sum only their own stage's rows.
    """
    start = time.monotonic()
    merged_usage_context = dict(usage_context or {})
    merged_usage_context["shadow_stage"] = "facial_shadow"
    try:
        raw = shadow_facial_llm(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            usage_context=merged_usage_context,
            capture=capture,
        )
    except Exception as e:  # noqa: BLE001 — shadow path is fail-soft by design
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.warning(
            "facial shadow-judge call failed (fail-soft, primary verdict unaffected): %s",
            e,
        )
        return None, latency_ms, str(e)[:240]
    latency_ms = round((time.monotonic() - start) * 1000, 1)
    return raw, latency_ms, None


def _full_shadow_call(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    usage_context: dict | None,
    capture: dict | None = None,
) -> tuple[str | None, float | None, str | None]:
    """Call the shadow model once for full-eval. NEVER raises. Sibling to
    ``_facial_shadow_call`` — identical shape, routed to ``shadow_full_llm``
    and tagged ``usage_context["shadow_stage"] = "full_shadow"`` instead.

    Returns ``(raw_text, latency_ms, error)`` — exactly one of
    ``raw_text``/``error`` is non-None on return.
    """
    start = time.monotonic()
    merged_usage_context = dict(usage_context or {})
    merged_usage_context["shadow_stage"] = "full_shadow"
    try:
        raw = shadow_full_llm(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            usage_context=merged_usage_context,
            capture=capture,
        )
    except Exception as e:  # noqa: BLE001 — shadow path is fail-soft by design
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.warning(
            "full-eval shadow-judge call failed (fail-soft, primary verdict unaffected): %s",
            e,
        )
        return None, latency_ms, str(e)[:240]
    latency_ms = round((time.monotonic() - start) * 1000, 1)
    return raw, latency_ms, None


# Raw-response capture on shadow parse failures. A PARSE_FAILURE event that
# throws away the response it couldn't parse is undiagnosable after the fact
# (2026-07-04 SPL live run: three full-eval PARSE_FAILUREs, no way to tell
# empty-content from malformed-format from reasoning-leak). On failure the
# event gains the first N chars plus the total length (an empty response
# records prefix="" / len=0, which is itself the diagnosis); on success the
# fields are omitted entirely so the happy-path event schema is unchanged.
_SHADOW_RAW_PREFIX_CHARS = 500


def _shadow_raw_capture(raw: str | None, parse_failed: bool) -> dict:
    if not parse_failed or raw is None:
        return {}
    return {
        "shadow_raw_prefix": raw[:_SHADOW_RAW_PREFIX_CHARS],
        "shadow_raw_len": len(raw),
    }


# Full shadow-verdict capture (monitoring channel). The run-log comparison
# event stays compact (decisions + agreement); THIS file carries the shadow
# model's complete output per candidate — the structured judgment text AND
# any separate reasoning the provider returns — plus the exact user prompt,
# so verdicts are attributable to candidates and re-judgeable offline. One
# JSONL per project state dir, sibling to the run log; append-only,
# fail-soft like every shadow write.
_SHADOW_JUDGMENTS_FILENAME = "shadow_judgments.jsonl"
_BATCH_PARSE_FAILURES_FILENAME = "batch_parse_failures.jsonl"


def _record_shadow_judgment(log_path: str, payload: dict) -> None:
    try:
        from shared.storage import append_jsonl

        target = Path(str(log_path)).parent / _SHADOW_JUDGMENTS_FILENAME
        append_jsonl(str(target), payload)
    except Exception as exc:  # noqa: BLE001 — shadow path is best-effort
        logger.warning("shadow_judgments.jsonl write failed: %s", exc)


def _record_batch_parse_failure(
    *,
    candidate_count: int,
    valid_verdicts: int,
    raw: object,
    reason: str = "untrustworthy_batch",
    contract_mode: str = "legacy",
) -> None:
    usage_log_path = current_llm_usage_log_path()
    if usage_log_path is None:
        return

    if isinstance(raw, str):
        raw_text = raw
    else:
        try:
            raw_text = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            raw_text = f"<{type(raw).__module__}.{type(raw).__qualname__}>"
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "candidate_count": candidate_count,
        "valid_verdicts": valid_verdicts,
        "reason": reason,
        "contract_mode": contract_mode,
        "raw_len": len(raw_text),
        "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }
    target = usage_log_path.parent / _BATCH_PARSE_FAILURES_FILENAME

    try:
        try:
            from shared.storage import append_jsonl
        except Exception:  # noqa: BLE001 — fallback is only for import/cycle trouble
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a") as f:
                f.write(json.dumps(payload) + "\n")
        else:
            append_jsonl(str(target), payload)
    except Exception as exc:  # noqa: BLE001 — diagnostic path is best-effort
        logger.warning("batch_parse_failures.jsonl write failed: %s", exc)


# Live console rendering (Sam, 2026-07-05, revised same day): the run
# console is sacred — months-stable, compact, readable. Every completed
# shadow comparison prints EXACTLY ONE line: both decisions by model name,
# the outcome in words, and a running per-tier agreement tally. Full depth
# (chain-of-thought, verdict prose) lives in shadow_judgments.jsonl and the
# live feed (`tools/shadow_report.py <state-dir> --follow`) — never here.
# Each line is built as ONE string and printed once so output from the
# shadow worker thread cannot interleave with the main run's console.
#
# Tally semantics: per-tier counters (facial — batch members included —
# and full, separately) count comparable outcomes only; unparsed / error /
# not-comparable are tracked beside them and rendered in words when
# nonzero, never folded into the percentage.
_SHADOW_TALLY_LOCK = threading.Lock()
_SHADOW_TALLY_KEYS = ("agree", "disagree", "unparsed", "not_comparable", "error")
_shadow_tallies: dict[str, dict[str, int]] = {}


def _reset_shadow_tallies() -> None:
    """Test seam: start the running tallies from zero."""
    with _SHADOW_TALLY_LOCK:
        _shadow_tallies.clear()


def _tally_shadow_outcomes(tier: str, **deltas: int) -> str:
    """Fold one comparison's outcome counts into the tier tally and return
    the rendered running-tally suffix (thread-safe, one atomic step so the
    printed tally always matches the line it rides on)."""
    with _SHADOW_TALLY_LOCK:
        tally = _shadow_tallies.setdefault(
            tier, {key: 0 for key in _SHADOW_TALLY_KEYS}
        )
        for key, delta in deltas.items():
            tally[key] = tally.get(key, 0) + delta
        counts = dict(tally)
    comparable = counts["agree"] + counts["disagree"]
    text = f"{tier}: {counts['agree']}/{comparable} agree"
    if comparable:
        text += f" ({counts['agree'] / comparable * 100:.1f}%)"
    if counts["unparsed"]:
        text += f", {counts['unparsed']} unparsed"
    if counts["not_comparable"]:
        text += f", {counts['not_comparable']} not comparable"
    if counts["error"]:
        n = counts["error"]
        text += f", {n} error" + ("s" if n != 1 else "")
    return text


def _short_model_name(model_name: str | None, fallback: str) -> str:
    """Console-friendly model word for attribution-by-name ('opus', 'glm',
    'fable') — never bare 'primary'/'shadow' jargon on a comparison line."""
    model_id = str(model_name or "").rsplit("/", 1)[-1].lower()
    for word in ("opus", "glm", "fable", "mythos", "sonnet", "haiku", "gpt"):
        if word in model_id:
            return word
    return model_id[:12] or fallback


def _shadow_subject_name(user_prompt: str | None) -> str:
    """Candidate identity for the comparison line. Prefers an explicit
    Name:/Candidate:/Profile: line ANYWHERE in the prompt — the first line
    is sometimes a triage banner, not the candidate (live-caught on the
    2026-07-05 run: headers read '⚠ TRIAGE TIGHTENING ACTIVE…') — and
    falls back to the first non-empty line."""
    first_line = ""
    for line in (user_prompt or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not first_line:
            first_line = stripped
        lowered = stripped.lower()
        for prefix in ("name:", "candidate:", "profile:"):
            if lowered.startswith(prefix):
                value = stripped[len(prefix):].strip()
                if value:
                    return value
    return first_line or "(unknown candidate)"


def _fit_console_field(text: object, width: int) -> str:
    """Collapse to one line (whitespace runs become single spaces), then
    truncate-with-ellipsis or pad to exactly ``width`` — the comparison
    lines keep aligned columns no matter what a name or error contains."""
    flat = " ".join(str(text).split())
    if len(flat) > width:
        return flat[: width - 1] + "…"
    return flat.ljust(width)


def _shadow_outcome_class(
    *,
    agrees: bool | None,
    shadow_decision: str | None,
    shadow_parse_failed: bool,
) -> str:
    """Classify one comparison outcome: 'agree' / 'disagree' / 'unparsed'
    (no usable shadow verdict) / 'not_comparable' (a real verdict that
    isn't classifiable on the comparison axis, e.g. REVIEW_*)."""
    if agrees is True:
        return "agree"
    if agrees is False:
        return "disagree"
    if shadow_decision is None or shadow_parse_failed:
        return "unparsed"
    return "not_comparable"


_SHADOW_OUTCOME_WORDS = {
    "agree": "AGREE",
    "disagree": "DISAGREE",
    "unparsed": "unparsed",
    "not_comparable": "not comparable",
}


def _print_shadow_comparison_line(
    *,
    tier: str,
    tier_tag: str,
    subject: str,
    middle: str,
    tally: str,
) -> None:
    """Print ONE completed shadow comparison as one prebuilt line."""
    try:
        # Middle width 62: the widest real decisions column —
        # "opus=FACIAL_BORDERLINE  glm=FACIAL_BORDERLINE  not comparable"
        # — is 61 chars; anything narrower truncates the outcome word,
        # which is the point of the line (live-caught on the 2026-07-05
        # captures: "DISAGR…").
        line = (
            f"[shadow] {_fit_console_field(tier_tag, 9)} "
            f"{_fit_console_field(subject, 28)} "
            f"{_fit_console_field(middle, 62)} | {tally}"
        )
        print(line, flush=True)
    except Exception as exc:  # noqa: BLE001 — rendering must never hurt the shadow path
        logger.warning("shadow live-console render failed: %s", exc)


def _shadow_single_middle(
    *,
    primary_model: str | None,
    primary_decision: str | None,
    shadow_decision: str | None,
    outcome_class: str,
    error: str | None,
) -> str:
    """The decisions column for a single (non-batch) comparison line."""
    if error:
        return f"SHADOW ERROR: {error}"
    primary_word = _short_model_name(primary_model, "primary")
    shadow_word = _short_model_name(config.SHADOW_FACIAL_MODEL_NAME, "shadow")
    return (
        f"{primary_word}={primary_decision or 'NO VERDICT'}  "
        f"{shadow_word}={shadow_decision or 'NO VERDICT'}  "
        f"{_SHADOW_OUTCOME_WORDS[outcome_class]}"
    )


def _parse_legacy_shadow_decision(raw_text: str) -> str:
    """Parse a shadow response against the legacy (old-brief) JSON contract.

    Mirrors the inline parsing ``facial_judge``'s old-brief branch does on
    the primary Opus response (``result.get("decision")`` checked against
    ``_VALID_FACIAL``), applied to the shadow model's raw text instead of
    an already-parsed dict — the shadow client never JSON-decodes on the
    caller's behalf (see ``shadow_facial_llm``'s docstring for why). Reuses
    ``_parse_json_response`` (the same JSON-repair helper every provider
    branch in llm_clients.py uses) so a shadow model that wraps its JSON in
    prose or code fences parses exactly as leniently as the primary path
    would. Returns "PARSE_FAILURE" — a member of ``FAILURE_DECISIONS`` — on
    any parse or validation problem, never raises.
    """
    try:
        parsed = _parse_json_response(raw_text)
    except Exception:
        return "PARSE_FAILURE"
    if isinstance(parsed, dict):
        decision = parsed.get("decision")
        if decision in _VALID_FACIAL:
            return decision
    return "PARSE_FAILURE"


def _run_facial_shadow_single(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    primary_decision: str,
    parse_fn: Callable[[str], str],
    lane_context: dict | None = None,
) -> None:
    """Shadow-compare ONE facial verdict. Side-effect only; returns nothing.

    Emits exactly one ``facial_shadow_comparison`` run-log event (schema:
    ``batch=False``, ``candidate_count=1``, singular ``primary_decision`` /
    ``shadow_decision`` / ``agrees`` fields) when a run-log sink is
    available. No-op when the flag is off or no sink is available.

    Dispatch: fire-and-forget on the single shadow worker when
    ``config.SHADOW_ASYNC_ENABLED`` (default), inline when it is off —
    see ``_dispatch_shadow``. The enabled-flag check stays HERE, at judge
    time, so flag semantics are identical on both paths; the event
    content is identical either way.
    """
    if not config.SHADOW_FACIAL_MODEL_ENABLED:
        return
    _dispatch_shadow(
        functools.partial(
            _run_facial_shadow_single_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            primary_decision=primary_decision,
            parse_fn=parse_fn,
            lane_context=lane_context,
        )
    )


def _run_facial_shadow_single_sync(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    primary_decision: str,
    parse_fn: Callable[[str], str],
    lane_context: dict | None = None,
) -> None:
    """Comparison body for ``_run_facial_shadow_single`` — the enabled
    flag was already checked at dispatch time. Fail-soft: the entire body
    is inside try/except, so nothing raises to the inline caller or the
    shadow worker."""
    try:
        log_path = _shadow_run_log_path()
        if log_path is None:
            return
        capture: dict = {}
        raw, latency_ms, error = _facial_shadow_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            usage_context=dict(lane_context or {}),
            capture=capture,
        )
        shadow_decision: str | None = None
        shadow_parse_failed = False
        agrees: bool | None = None
        if raw is not None:
            try:
                shadow_decision = parse_fn(raw)
            except Exception as parse_exc:  # noqa: BLE001 — fail-soft
                logger.warning("facial shadow-judge parse failed: %s", parse_exc)
                shadow_decision = None
                shadow_parse_failed = True
            else:
                shadow_parse_failed = is_failure_decision(shadow_decision)
                if not shadow_parse_failed:
                    agrees = shadow_decision == primary_decision
        log_event(
            log_path,
            "facial_shadow_comparison",
            batch=False,
            candidate_count=1,
            primary_decision=primary_decision,
            shadow_decision=shadow_decision,
            agrees=agrees,
            shadow_parse_failed=shadow_parse_failed,
            shadow_model=config.SHADOW_FACIAL_MODEL_NAME,
            latency_ms=latency_ms,
            shadow_error=error,
            **_shadow_raw_capture(raw, shadow_parse_failed),
        )
        _record_shadow_judgment(
            log_path,
            {
                "ts": time.time(),
                "stage": "facial",
                "shadow_model": config.SHADOW_FACIAL_MODEL_NAME,
                "primary_model": config.FACIAL_MODEL_NAME,
                "primary_decision": primary_decision,
                "shadow_decision": shadow_decision,
                "agrees": agrees,
                "shadow_parse_failed": shadow_parse_failed,
                "latency_ms": latency_ms,
                "shadow_error": error,
                "lane_context": dict(lane_context or {}),
                "raw": raw,
                "reasoning_content": capture.get("reasoning_content"),
                "finish_reason": capture.get("finish_reason"),
                "user_prompt": user_prompt,
            },
        )
        outcome = (
            "error"
            if error
            else _shadow_outcome_class(
                agrees=agrees,
                shadow_decision=shadow_decision,
                shadow_parse_failed=shadow_parse_failed,
            )
        )
        _print_shadow_comparison_line(
            tier="facial",
            tier_tag="facial",
            subject=_shadow_subject_name(user_prompt),
            middle=_shadow_single_middle(
                primary_model=config.FACIAL_MODEL_NAME,
                primary_decision=primary_decision,
                shadow_decision=shadow_decision,
                outcome_class=outcome,
                error=error,
            ),
            tally=_tally_shadow_outcomes("facial", **{outcome: 1}),
        )
    except Exception as exc:  # noqa: BLE001 — the whole shadow path is best-effort
        logger.warning("facial shadow-judge comparison failed (primary unaffected): %s", exc)


def _run_facial_shadow_batch(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    candidate_count: int,
    primary_decisions: list[str],
) -> None:
    """Shadow-compare a WHOLE batch facial call with one shadow call.

    Mirrors the real batch call shape (one LLM call for every snippet on
    the page) rather than fanning out one shadow call per candidate.
    Emits exactly one ``facial_shadow_comparison`` run-log event per batch
    (schema: ``batch=True``, plural ``primary_decisions`` /
    ``shadow_decisions`` / ``agrees`` / ``shadow_parse_failed`` arrays,
    one entry per candidate in the same order as ``primary_decisions``).
    No-op when the flag is off or no sink is available.

    Dispatch: fire-and-forget on the single shadow worker when
    ``config.SHADOW_ASYNC_ENABLED`` (default), inline when it is off —
    see ``_dispatch_shadow``. ``primary_decisions`` is snapshotted at
    dispatch time so a caller-side mutation between enqueue and execution
    cannot change what gets compared.
    """
    if not config.SHADOW_FACIAL_MODEL_ENABLED:
        return
    _dispatch_shadow(
        functools.partial(
            _run_facial_shadow_batch_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            candidate_count=candidate_count,
            primary_decisions=list(primary_decisions),
        )
    )


def _run_facial_shadow_batch_sync(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    candidate_count: int,
    primary_decisions: list[str],
) -> None:
    """Comparison body for ``_run_facial_shadow_batch`` — the enabled
    flag was already checked at dispatch time. Fail-soft: the entire body
    is inside try/except, so nothing raises to the inline caller or the
    shadow worker."""
    try:
        log_path = _shadow_run_log_path()
        if log_path is None:
            return
        capture: dict = {}
        raw, latency_ms, error = _facial_shadow_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            usage_context=None,
            capture=capture,
        )
        shadow_decisions: list[str | None] = [None] * candidate_count
        agrees: list[bool | None] = [None] * candidate_count
        shadow_parse_failed: list[bool] = [False] * candidate_count
        if raw is not None:
            try:
                shadow_results = parse_facial_batch_response(raw, candidate_count)
            except Exception as parse_exc:  # noqa: BLE001 — fail-soft
                logger.warning("facial shadow-judge batch parse failed: %s", parse_exc)
                shadow_parse_failed = [True] * candidate_count
            else:
                for idx, shadow_result in enumerate(shadow_results):
                    if idx >= candidate_count:
                        break
                    shadow_decisions[idx] = shadow_result.decision
                    failed = is_failure_decision(shadow_result.decision)
                    shadow_parse_failed[idx] = failed
                    if not failed:
                        agrees[idx] = shadow_result.decision == primary_decisions[idx]
        log_event(
            log_path,
            "facial_shadow_comparison",
            batch=True,
            candidate_count=candidate_count,
            primary_decisions=list(primary_decisions),
            shadow_decisions=shadow_decisions,
            agrees=agrees,
            shadow_parse_failed=shadow_parse_failed,
            shadow_model=config.SHADOW_FACIAL_MODEL_NAME,
            latency_ms=latency_ms,
            shadow_error=error,
            **_shadow_raw_capture(raw, any(shadow_parse_failed)),
        )
        _record_shadow_judgment(
            log_path,
            {
                "ts": time.time(),
                "stage": "facial_batch",
                "shadow_model": config.SHADOW_FACIAL_MODEL_NAME,
                "primary_model": config.FACIAL_MODEL_NAME,
                "candidate_count": candidate_count,
                "primary_decisions": list(primary_decisions),
                "shadow_decisions": shadow_decisions,
                "agrees": agrees,
                "shadow_parse_failed": shadow_parse_failed,
                "latency_ms": latency_ms,
                "shadow_error": error,
                "raw": raw,
                "reasoning_content": capture.get("reasoning_content"),
                "finish_reason": capture.get("finish_reason"),
                "user_prompt": user_prompt,
            },
        )
        # One line for the whole batch; members fold into the facial tally
        # individually (an errored call counts once — the failure was the
        # call, not one per member it never judged).
        outcome_counts = {key: 0 for key in _SHADOW_TALLY_KEYS}
        if error:
            outcome_counts["error"] = 1
            middle = f"SHADOW ERROR: {error}"
        else:
            for idx in range(candidate_count):
                outcome_counts[
                    _shadow_outcome_class(
                        agrees=agrees[idx],
                        shadow_decision=shadow_decisions[idx],
                        shadow_parse_failed=shadow_parse_failed[idx],
                    )
                ] += 1
            middle_parts = [f"{outcome_counts['agree']} agree"]
            for key in ("disagree", "unparsed", "not_comparable"):
                if outcome_counts[key]:
                    middle_parts.append(
                        f"{outcome_counts[key]} {_SHADOW_OUTCOME_WORDS[key].lower()}"
                    )
            middle = ", ".join(middle_parts)
        _print_shadow_comparison_line(
            tier="facial",
            tier_tag=f"facial×{candidate_count}",
            subject="batch",
            middle=middle,
            tally=_tally_shadow_outcomes(
                "facial", **{key: n for key, n in outcome_counts.items() if n}
            ),
        )
    except Exception as exc:  # noqa: BLE001 — the whole shadow path is best-effort
        logger.warning("facial shadow-judge batch comparison failed (primary unaffected): %s", exc)


# Full-eval shadow: decision-CLASS ("axis") agreement, not raw-string
# equality. A SAVE vs INFERENTIAL_SAVE mismatch both land on the SAVE axis
# and count as AGREEMENT for the save/reject instrument this seam measures
# — the raw decisions are still recorded verbatim on the event so a reader
# can see the finer-grained mismatch. REVIEW_INFERRED / REVIEW_FLAGGED /
# PARSE_FAILURE / JUDGMENT_FAILURE / an unrecognized string are not
# classifiable on either axis (None) — same "not comparable" semantics a
# facial shadow parse-failure already uses.
# Imported from contracts, not re-declared — if a new SAVE-family decision
# joins the vocabulary, the shadow axis classifier must follow automatically
# (Opus-review drift finding on the full-eval shadow extension).
from shared.contracts import SAVE_DECISIONS as SAVE_FAMILY_DECISIONS

REJECT_FAMILY_DECISIONS = frozenset({"REJECT"})


def _full_decision_axis(decision: str | None) -> str | None:
    """Collapse a full-eval decision string to its save/reject axis.

    Returns "SAVE", "REJECT", or None (not classifiable on this axis).
    """
    if decision in SAVE_FAMILY_DECISIONS:
        return "SAVE"
    if decision in REJECT_FAMILY_DECISIONS:
        return "REJECT"
    return None


def _full_shadow_agrees(primary_decision: str | None, shadow_decision: str | None) -> bool | None:
    """Decision-class agreement for the full-eval shadow instrument.

    None when either side isn't classifiable on the save/reject axis
    (e.g. either decision is REVIEW_INFERRED/REVIEW_FLAGGED) — not a
    fabricated False.
    """
    primary_axis = _full_decision_axis(primary_decision)
    shadow_axis = _full_decision_axis(shadow_decision)
    if primary_axis is None or shadow_axis is None:
        return None
    return primary_axis == shadow_axis


def _run_full_shadow_single(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    primary_decision: str,
    capability_areas: tuple[str, ...],
    post_save_modifiers: tuple[str, ...],
    lane_context: dict | None = None,
) -> None:
    """Shadow-compare ONE full-eval verdict. Side-effect only; returns nothing.

    Sibling to ``_run_facial_shadow_single`` for the full-eval tier.
    Scope: LinkedIn's V2-structural ``full_judge`` branch only — see the
    shadow-hook module docstring above for why the legacy old-brief branch
    and ``github_full_judge`` are excluded. There is no batch variant:
    ``full_judge`` has no batch call path (unlike facial), so every
    comparison is a singleton, one ``full_shadow_comparison`` event per
    candidate.

    Emits exactly one ``full_shadow_comparison`` run-log event (fields:
    ``primary_decision``, ``shadow_decision``, ``agrees``,
    ``shadow_parse_failed``, ``shadow_model``, ``latency_ms``,
    ``shadow_error``) when a run-log sink is available. No-op when the
    flag is off or no sink is available.

    Dispatch: fire-and-forget on the single shadow worker when
    ``config.SHADOW_ASYNC_ENABLED`` (default), inline when it is off —
    see ``_dispatch_shadow``. The enabled-flag check stays HERE, at judge
    time, so flag semantics are identical on both paths; the event
    content is identical either way.
    """
    if not config.SHADOW_FACIAL_MODEL_ENABLED:
        return
    _dispatch_shadow(
        functools.partial(
            _run_full_shadow_single_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            primary_decision=primary_decision,
            capability_areas=capability_areas,
            post_save_modifiers=post_save_modifiers,
            lane_context=lane_context,
        )
    )


def _run_full_shadow_single_sync(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    primary_decision: str,
    capability_areas: tuple[str, ...],
    post_save_modifiers: tuple[str, ...],
    lane_context: dict | None = None,
) -> None:
    """Comparison body for ``_run_full_shadow_single`` — the enabled
    flag was already checked at dispatch time. Fail-soft: the entire body
    is inside try/except, so nothing raises to the inline caller or the
    shadow worker."""
    try:
        log_path = _shadow_run_log_path()
        if log_path is None:
            return
        capture: dict = {}
        raw, latency_ms, error = _full_shadow_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            usage_context=dict(lane_context or {}),
            capture=capture,
        )
        shadow_decision: str | None = None
        shadow_parse_failed = False
        agrees: bool | None = None
        if raw is not None:
            try:
                shadow_decision = parse_full_evaluation_response(
                    raw,
                    require_semantic_v2=True,
                    capability_areas=capability_areas,
                    post_save_modifiers=post_save_modifiers,
                ).decision
            except Exception as parse_exc:  # noqa: BLE001 — fail-soft
                logger.warning("full-eval shadow-judge parse failed: %s", parse_exc)
                shadow_decision = None
                shadow_parse_failed = True
            else:
                shadow_parse_failed = is_failure_decision(shadow_decision)
                if not shadow_parse_failed:
                    agrees = _full_shadow_agrees(primary_decision, shadow_decision)
        log_event(
            log_path,
            "full_shadow_comparison",
            primary_decision=primary_decision,
            shadow_decision=shadow_decision,
            agrees=agrees,
            shadow_parse_failed=shadow_parse_failed,
            shadow_model=config.SHADOW_FACIAL_MODEL_NAME,
            latency_ms=latency_ms,
            shadow_error=error,
            **_shadow_raw_capture(raw, shadow_parse_failed),
        )
        _record_shadow_judgment(
            log_path,
            {
                "ts": time.time(),
                "stage": "full",
                "shadow_model": config.SHADOW_FACIAL_MODEL_NAME,
                "primary_model": config.FULL_EVAL_MODEL_NAME,
                "primary_decision": primary_decision,
                "shadow_decision": shadow_decision,
                "agrees": agrees,
                "shadow_parse_failed": shadow_parse_failed,
                "latency_ms": latency_ms,
                "shadow_error": error,
                "lane_context": dict(lane_context or {}),
                "raw": raw,
                "reasoning_content": capture.get("reasoning_content"),
                "finish_reason": capture.get("finish_reason"),
                "user_prompt": user_prompt,
            },
        )
        outcome = (
            "error"
            if error
            else _shadow_outcome_class(
                agrees=agrees,
                shadow_decision=shadow_decision,
                shadow_parse_failed=shadow_parse_failed,
            )
        )
        _print_shadow_comparison_line(
            tier="full",
            tier_tag="full",
            subject=_shadow_subject_name(user_prompt),
            middle=_shadow_single_middle(
                primary_model=config.FULL_EVAL_MODEL_NAME,
                primary_decision=primary_decision,
                shadow_decision=shadow_decision,
                outcome_class=outcome,
                error=error,
            ),
            tally=_tally_shadow_outcomes("full", **{outcome: 1}),
        )
    except Exception as exc:  # noqa: BLE001 — the whole shadow path is best-effort
        logger.warning("full-eval shadow-judge comparison failed (primary unaffected): %s", exc)


def _safe_confidence(val, default: float = 0.5) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _as_str_list(value) -> list:
    """Coerce a JSON-deserialized value to a list of trimmed strings.

    Used by the legacy JSON full-eval paths so a model emitting
    ``review_structural_evidence`` as a list, a semicolon-delimited
    string, or omitting it altogether all collapse to a clean
    ``list[str]`` for ``OpusDecision``. P4 specific.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def is_failure_decision(decision: str) -> bool:
    """True if decision represents a non-terminal parse/judgment failure."""
    return _is_failure_decision(decision)


def extract_priority_rank(path: str) -> int:
    # VERTICAL-VOCAB(prompt-layer)
    """Extract capability area rank from a decision path.

    Paths look like 'DIRECT:3. Agentic Systems...' or 'ADJACENT:1. RL Post-Training|TRANSFERABLE'.
    Returns the numeric rank (1-based), or 0 if not found.
    """
    m = re.search(r':(\d+)\.', path)
    return int(m.group(1)) if m else 0

# Test/back-compat fallback only; production callers pass brief= (E1/E2 gate).
_brief: Brief | None = None

_PROMPT_CAPTURE_SCHEMA_VERSION = 1

_FACIAL_LEGACY_CONTRACT_VERSION = "linkedin_facial_legacy_text_v1"
_FULL_LEGACY_CONTRACT_VERSION = "linkedin_full_legacy_text_v2"
_VALID_JUDGMENT_CONTRACT_MODES = frozenset({"legacy", "tool"})


def _v2_judgment_contract_mode(stage: str) -> str:
    attr = (
        "LINKEDIN_V2_FACIAL_CONTRACT"
        if stage == "facial"
        else "LINKEDIN_V2_FULL_CONTRACT"
    )
    mode = str(getattr(config, attr, "legacy") or "legacy").strip().lower()
    if mode not in _VALID_JUDGMENT_CONTRACT_MODES:
        raise RuntimeError(
            f"{attr} must be one of {sorted(_VALID_JUDGMENT_CONTRACT_MODES)}; "
            f"got {mode!r}"
        )
    return mode


def _judgment_usage_context(
    lane_context: dict | None,
    *,
    stage: str,
    contract_mode: str,
    contract_version: str,
    batch_size: int | None = None,
) -> dict[str, object]:
    context: dict[str, object] = dict(lane_context or {})
    context.setdefault("stage", stage)
    context.setdefault("logical_call_id", f"judge-{uuid.uuid4().hex}")
    context["judgment_contract_mode"] = contract_mode
    context["judgment_contract_version"] = contract_version
    if batch_size is not None:
        context["batch_size"] = int(batch_size)
    return context


def _facial_fallback_context(
    lane_context: dict | None,
    *,
    reason: str,
    candidate_index: int,
) -> dict[str, object]:
    """Give each legacy sequential fallback its own attributable call ID."""

    context: dict[str, object] = dict(lane_context or {})
    parent_call_id = str(context.get("logical_call_id") or "")
    if parent_call_id:
        context["parent_logical_call_id"] = parent_call_id
    context["logical_call_id"] = f"judge-{uuid.uuid4().hex}"
    context["fallback_reason"] = reason
    context["fallback_index"] = int(candidate_index)
    context["batch_size"] = 1
    return context


def full_id_mismatch_retry_context(
    lane_context: dict | None,
    *,
    parent_logical_call_id: str,
) -> dict[str, object]:
    """Give a candidate-ID mismatch re-issue its own attributable call ID.

    Sibling of ``_facial_fallback_context``: the re-issue is a distinct
    provider call and must never be accounted against the call it replaces,
    so it carries a fresh ``logical_call_id`` parented to the original.
    """

    context: dict[str, object] = dict(lane_context or {})
    parent_call_id = str(
        parent_logical_call_id or context.get("logical_call_id") or ""
    )
    if parent_call_id:
        context["parent_logical_call_id"] = parent_call_id
    context["logical_call_id"] = f"judge-{uuid.uuid4().hex}"
    context["retry_reason"] = "candidate_id_mismatch"
    return context


def _full_contract_failure_record(
    error: JudgmentToolContractError,
    *,
    expected_candidate_id: str,
    arguments: object,
) -> dict[str, str]:
    """Machine-typed facts about a failed full tool contract.

    ``reason`` is the local validator's own token, never model text, so a
    caller may branch on it without reading the rationale — whose detail
    carries model-returned content by construction.
    """

    actual_candidate_id = ""
    if isinstance(arguments, dict):
        raw_actual = arguments.get("candidate_id")
        if isinstance(raw_actual, str):
            actual_candidate_id = raw_actual
    return {
        "reason": error.reason,
        "expected_candidate_id": expected_candidate_id,
        "actual_candidate_id": actual_candidate_id,
    }


def _fireworks_judgment_policy(
    *,
    stage: str,
    system_prompt: str,
    contract_version: str,
    usage_context: dict[str, object],
) -> FireworksStagePolicy | None:
    if not bool(getattr(config, "FIREWORKS_JUDGMENT_POLICY_ENABLED", False)):
        return None

    if stage == "facial":
        model = config.FACIAL_MODEL_NAME
        effort = str(
            getattr(config, "FIREWORKS_FACIAL_REASONING_EFFORT", "") or ""
        ).strip().lower()
        attempt_timeout = float(
            getattr(config, "FIREWORKS_FACIAL_ATTEMPT_TIMEOUT_SECONDS", 120.0)
        )
        total_deadline = float(
            getattr(config, "FIREWORKS_FACIAL_TOTAL_DEADLINE_SECONDS", 180.0)
        )
        max_attempts = int(
            getattr(config, "FIREWORKS_FACIAL_MAX_ATTEMPTS", 2)
        )
    else:
        model = config.FULL_EVAL_MODEL_NAME
        effort = str(
            getattr(config, "FIREWORKS_FULL_REASONING_EFFORT", "") or ""
        ).strip().lower()
        attempt_timeout = float(
            getattr(config, "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS", 240.0)
        )
        total_deadline = float(
            getattr(config, "FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS", 360.0)
        )
        max_attempts = int(getattr(config, "FIREWORKS_FULL_MAX_ATTEMPTS", 2))

    reasoning_effort = effort or None
    if reasoning_effort not in {None, "high", "max"}:
        raise RuntimeError(
            f"invalid Fireworks {stage} reasoning effort: {reasoning_effort!r}"
        )

    prompt_cache_key = None
    if bool(getattr(config, "FIREWORKS_PROMPT_AFFINITY_ENABLED", False)):
        raw_slot = usage_context.get("batch_slot")
        batch_slot = raw_slot if isinstance(raw_slot, int) else None
        prompt_cache_key = build_fireworks_prompt_cache_key(
            stage=stage,
            model=model,
            contract_version=contract_version,
            stable_prefix=system_prompt,
            lane_id=str(usage_context.get("lane_id") or ""),
            batch_slot=batch_slot,
        )

    return FireworksStagePolicy(
        stage=stage,
        reasoning_effort=reasoning_effort,
        attempt_timeout_seconds=attempt_timeout,
        total_deadline_seconds=total_deadline,
        max_attempts=max_attempts,
        response_transport=(
            "stream"
            if bool(
                getattr(config, "FIREWORKS_JUDGMENT_STREAM_ENABLED", False)
            )
            else "complete"
        ),
        prompt_cache_key=prompt_cache_key,
    )


def init_judger(brief: Brief) -> None:
    """Initialize the judger with a sourcing brief."""
    global _brief
    # Retained for tests/back-compat; production judgment must pass brief=.
    _brief = brief


def _langfuse_context_from_usage_context(
    usage_context: dict[str, object] | None,
) -> dict[str, str | None]:
    langfuse_context = {}
    if isinstance(usage_context, dict):
        raw = usage_context.get("_langfuse")
        if isinstance(raw, dict):
            langfuse_context = raw
    trace_id = langfuse_context.get("trace_id")
    observation_id = langfuse_context.get("observation_id")
    trace_url = langfuse_context.get("trace_url")
    if not isinstance(trace_id, str):
        trace_id = get_current_trace_id()
    if not isinstance(observation_id, str):
        observation_id = get_current_observation_id()
    if not isinstance(trace_url, str):
        trace_url = get_trace_url(trace_id=trace_id) if trace_id else None
    return {
        "trace_id": trace_id if isinstance(trace_id, str) else None,
        "observation_id": (
            observation_id if isinstance(observation_id, str) else None
        ),
        "trace_url": trace_url if isinstance(trace_url, str) else None,
    }


def _build_prompt_capture(
    *,
    stage: str,
    source: str,
    render_route: str,
    llm_caller_name: str,
    expect_json: bool,
    system_prompt: str,
    candidate_text: str,
    usage_context: dict[str, object] | None = None,
) -> dict[str, object]:
    langfuse_context = _langfuse_context_from_usage_context(usage_context)
    return {
        "schema_version": _PROMPT_CAPTURE_SCHEMA_VERSION,
        "stage": stage,
        "source": source,
        "render_route": render_route,
        "llm_caller": llm_caller_name,
        "expect_json": bool(expect_json),
        "candidate_text": candidate_text,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "trace_id": langfuse_context["trace_id"],
        "observation_id": langfuse_context["observation_id"],
        "trace_url": langfuse_context["trace_url"],
        "logical_call_id": (
            usage_context.get("logical_call_id")
            if isinstance(usage_context, dict)
            else None
        ),
        "judgment_contract_mode": (
            usage_context.get("judgment_contract_mode")
            if isinstance(usage_context, dict)
            else None
        ),
        "judgment_contract_version": (
            usage_context.get("judgment_contract_version")
            if isinstance(usage_context, dict)
            else None
        ),
    }


def _attach_prompt_capture(
    decision: OpusDecision,
    prompt_capture: dict[str, object],
) -> OpusDecision:
    capture = dict(prompt_capture)
    source = capture.get("source")
    stage = capture.get("stage")
    render_route = capture.get("render_route")
    if (
        source == "linkedin"
        and stage in {"facial", "full"}
        and isinstance(render_route, str)
        and "judge_receipt" not in capture
    ):
        capture = _with_judge_receipt(
            capture,
            source=source,
            stage=stage,
            render_route=render_route,
            receipt_input={
                "prompt_capture": {
                    key: value
                    for key, value in capture.items()
                    if key != "judge_receipt"
                },
                "candidate_name": decision.candidate_name,
                "profile_url": decision.profile_url,
                "rationale": decision.rationale,
            },
            final_decision=decision.decision,
        )
    decision.prompt_capture = capture
    return decision


def _judge_receipt_status(
    final_decision: str,
    receipt_input: object,
) -> tuple[str, ReceiptStatus]:
    if final_decision == "PARSE_FAILURE":
        if _looks_like_model_refusal(receipt_input):
            return "refused", ReceiptStatus.REFUSED
        return "parse_fail", ReceiptStatus.PARSE_FAIL
    if final_decision == "JUDGMENT_FAILURE":
        return "error", ReceiptStatus.ERROR
    if final_decision == "REFUSED":
        return "refused", ReceiptStatus.REFUSED
    if final_decision == "ABSTAIN":
        return "abstain", ReceiptStatus.ABSTAIN
    return "ok", ReceiptStatus.OK


_MODEL_REFUSAL_MARKERS = (
    "as an ai",
    "i cannot comply",
    "i can't comply",
    "cannot comply",
    "can't comply",
    "unable to comply",
    "i cannot assist",
    "i can't assist",
    "cannot assist",
    "can't assist",
    "decline to",
    "declined to judge",
    "refuse to",
    "refused:",
    "not able to evaluate",
)


def _looks_like_model_refusal(value: object) -> bool:
    if isinstance(value, dict):
        return any(_looks_like_model_refusal(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_looks_like_model_refusal(item) for item in value)
    text = str(value or "").lower()
    return any(marker in text for marker in _MODEL_REFUSAL_MARKERS)


def _build_judge_receipt(
    *,
    source: str,
    stage: str,
    render_route: str,
    receipt_input: object,
    final_decision: str,
) -> dict[str, object]:
    safe_input = _receipt_json_safe(receipt_input)
    parse_status, status = _judge_receipt_status(final_decision, safe_input)
    receipt = build_receipt(
        receipt_type="judge",
        stage=f"{source}_{stage}_judge",
        input_payload={
            "source": source,
            "stage": stage,
            "render_route": render_route,
            "input": safe_input,
        },
        actual_status=status,
        intended_postcondition=(
            f"{source} {stage} judge response resolves to a typed bounded decision"
        ),
        actual_detail={
            "parse_status": parse_status,
            "final_decision": final_decision,
            "render_route": render_route,
        },
        producer="shared.judger",
        version_pins={"shared_judger": "judge-receipts-v1"},
    )
    return receipt.to_dict()


def _receipt_json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _receipt_json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_receipt_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_receipt_json_safe(item) for item in sorted(value, key=str)]
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _with_judge_receipt(
    prompt_capture: dict[str, object],
    *,
    source: str,
    stage: str,
    render_route: str,
    receipt_input: object,
    final_decision: str,
) -> dict[str, object]:
    enriched = dict(prompt_capture)
    enriched["judge_receipt"] = _build_judge_receipt(
        source=source,
        stage=stage,
        render_route=render_route,
        receipt_input=receipt_input,
        final_decision=final_decision,
    )
    return enriched


def _llm_caller_supports_usage_context(llm_caller) -> bool:
    try:
        signature = inspect.signature(llm_caller)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "usage_context" in signature.parameters


def _invoke_llm_caller(
    llm_caller,
    *,
    system_prompt: str,
    user_prompt: str,
    usage_context: dict[str, object],
):
    if _llm_caller_supports_usage_context(llm_caller):
        return llm_caller(
            system_prompt,
            user_prompt,
            usage_context=usage_context,
        )
    return llm_caller(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_preview_scan_section(brief: Brief) -> str:
    """Build preview scan criteria section from brief."""
    psc = brief.raw.get("preview_scan_criteria", {})
    if not psc:
        return ""
    rule = psc.get("rule", "")
    signals = psc.get("signals", [])
    if not signals:
        return ""
    signals_text = "\n".join(f"- {s}" for s in signals)
    return f"\n## Preview Scan Criteria\n{rule}\n{signals_text}\n"


def _build_calibration_section(brief: Brief) -> str:
    """Build calibration examples section from brief."""
    cal = brief.raw.get("calibration_examples", {})
    if not cal:
        return ""
    parts = []
    strong = cal.get("strong_saves", [])
    if strong:
        parts.append("### Strong Saves (correct)")
        for ex in strong:
            parts.append(f"- {ex['name']}: {ex['why']}")
    incorrect = cal.get("incorrect_saves", [])
    if incorrect:
        parts.append("### Incorrect Saves (should have been rejected)")
        for ex in incorrect:
            parts.append(f"- {ex['name']}: {ex['why']}")
    borderline = cal.get("borderline_verify", [])
    if borderline:
        parts.append("### Borderline (verify carefully)")
        for ex in borderline:
            parts.append(f"- {ex['name']}: {ex['why']}")
    if not parts:
        return ""
    return "\n## Calibration Examples\n" + "\n".join(parts) + "\n"


def _build_clear_skips_section(brief: Brief) -> str:
    """Build clear skips section from brief."""
    clear_skips = brief.clear_skips_from_review
    if not clear_skips:
        return ""
    text = "\n".join(f"- {s}" for s in clear_skips)
    return f"\n## Clear Skips from Review\n{text}\n"


YOE_INSTRUCTION = """
## Experience Counting — MANDATORY
When calculating years of experience, you MUST count:
- PhD programs: add the full duration (typically 4-6 years) to career experience
- Master's programs: add 1-2 years
- Research positions (postdoc, research scientist, predoctoral): count as full work experience
- The start date of the EARLIEST of: first degree, first job, or PhD program start

Examples:
- PhD started 2011, graduated 2016, first industry job 2016 → 15 years experience in 2026 (count from 2011)
- MS started 2009, first job 2011 → 17 years experience in 2026 (count from 2009)
- BS 2007, PhD 2011-2016, career 2016-present → 19 years experience in 2026 (count from 2007)

Do NOT count only post-graduation industry years. Advanced degrees are professional development that counts toward total experience."""


def _build_facial_system(brief: Brief) -> str:
    role = brief.role_title
    description = brief.role_description
    minimum_bar = brief.minimum_bar

    hard_skips = brief.hard_skips
    hard_skips_text = "\n".join(f"- {s}" for s in hard_skips) if hard_skips else "None defined"

    archetypes = brief.archetypes
    arch_text = ""
    for a in archetypes:
        arch_text += f"- {a['name']}"
        if a.get("capability_area"):
            arch_text += f" ({a['capability_area']})"
        arch_text += f": {a.get('pattern', '')}\n"
    if not arch_text:
        arch_text = "None defined"

    # Noise archetypes are optional — only include if the brief defines them
    noise_section = ""
    noise = brief.noise_archetypes
    if noise:
        noise_text = ""
        for n in noise:
            noise_text += f"- {n['name']}: {n.get('description', '')}\n"
        noise_section = f"\n## Noise Archetypes (skip these)\n{noise_text}"

    preview_section = _build_preview_scan_section(brief)
    calibration_section = _build_calibration_section(brief)
    clear_skips_section = _build_clear_skips_section(brief)

    # VERTICAL-VOCAB(prompt-layer)
    return f"""You are a senior technical recruiter evaluating candidates for: {role}

{description}

## Minimum Bar
{minimum_bar}
{YOE_INSTRUCTION}

## Non-Fit Patterns (genuinely wrong profiles — wrong career stage, wrong domain entirely)
{hard_skips_text}
{preview_section}{clear_skips_section}
## Target Archetypes (evaluate these FIRST — match before checking non-fit patterns)
{arch_text}{noise_section}{calibration_section}
## Your Task
Make a quick facial-fit judgment. Would a human sourcer open this profile to learn more?

THINK LIKE A RECRUITER, NOT A KEYWORD MATCHER. Synthesize the datapoints:
- What does the combination of employer + title + skills imply about their actual work?
- If someone works at a competitor or adjacent company doing related technical work, they likely have transferable depth even if their profile doesn't spell out every capability.
- A PhD + industry AI role + specific technical signals often means the person operates at a level beyond what their LinkedIn bullet points describe.
- Infer what's probable from context: e.g., someone building RAG systems and agentic frameworks at a major AI company is almost certainly working with training data, evaluation, and model quality — even if they don't say "data curation" verbatim.

The cost of a false positive is MUCH lower than a false negative. When uncertain, lean FACIAL_YES.

Return JSON only:
- "decision": "FACIAL_YES" or "FACIAL_NO"
- "path": most likely archetype name, or "none"
- "confidence": float 0.0-1.0
- "rationale": One concise sentence with specific evidence"""


def _build_full_system(brief: Brief) -> str:
    role = brief.role_title
    description = brief.role_description
    minimum_bar = brief.minimum_bar

    exp_floor = brief.experience_floor
    if isinstance(exp_floor, dict) and exp_floor:
        exp_text = f"Required: {exp_floor.get('required', '')}\nDisqualifying: {exp_floor.get('disqualifying', '')}"
        if exp_floor.get("note"):
            exp_text += f"\nNote: {exp_floor['note']}"
    else:
        exp_text = str(exp_floor) if exp_floor else "Not specified"

    # Pull evaluation fields from raw if available
    evaluation = brief.raw.get("evaluation", {})
    save_threshold = evaluation.get("save_threshold", "")
    capability_areas = evaluation.get("capability_areas", "")

    hard_skips = brief.hard_skips
    hard_skips_text = "\n".join(f"- {s}" for s in hard_skips) if hard_skips else "None"

    clear_skips = brief.clear_skips_from_review
    clear_skips_text = "\n".join(f"- {s}" for s in clear_skips) if clear_skips else "None"

    archetypes = brief.archetypes
    arch_text = ""
    for a in archetypes:
        arch_text += f"\n### {a['name']}"
        if a.get("capability_area"):
            arch_text += f" ({a['capability_area']})"
        arch_text += f"\n{a.get('pattern', '')}\n"
        if a.get("save_signals"):
            arch_text += "Save signals:\n"
            for s in a["save_signals"]:
                arch_text += f"  + {s}\n"
        caution = a.get("caution_signals") or a.get("skip_signals")
        if caution:
            arch_text += "Caution signals (lookalikes — verify, don't auto-reject):\n"
            for s in caution:
                arch_text += f"  ~ {s}\n"
    if not arch_text:
        arch_text = "None defined"

    # Noise archetypes are optional — only include if the brief defines them
    noise_section = ""
    noise = brief.noise_archetypes
    if noise:
        noise_text = ""
        for n in noise:
            noise_text += f"\n### {n['name']}\n{n.get('description', '')}\n"
            if n.get("signals"):
                for s in n["signals"]:
                    noise_text += f"  - {s}\n"
        noise_section = f"\n## Noise Archetypes\n{noise_text}"

    preview_section = _build_preview_scan_section(brief)
    calibration_section = _build_calibration_section(brief)

    return f"""You are a senior technical recruiter deciding if this candidate is worth a conversation for: {role}

{description}

## Minimum Bar
{minimum_bar}
{YOE_INSTRUCTION}

## Experience Floor
{exp_text}

## Save Threshold
{save_threshold}

## Capability Areas
{capability_areas}

## Target Archetypes (evaluate these FIRST)
{arch_text}{noise_section}{calibration_section}

## Non-Fit Patterns (check AFTER archetype evaluation — only if no archetype matches)
{hard_skips_text}
{preview_section}
## Weaker Signal Patterns (not auto-reject — verify against overall profile strength)
{clear_skips_text}

## Your Task
Decide if this candidate is worth a conversation.

SYNTHESIZE, DON'T CHECKLIST. Your job is to evaluate the whole candidate, not to check whether specific phrases appear on their profile.
- Combine employer, title, education, skills, and project descriptions to infer what this person actually does day-to-day — not just what they wrote down.
- Competitor employees doing adjacent work are high-value targets. If they work at a company that does similar work to this role, they almost certainly have relevant depth that isn't fully described on LinkedIn.
- PhD + senior industry role + relevant technical domain = assume depth beyond what's listed. These people don't put everything on LinkedIn.
- Ask: "Would a hiring manager want to talk to this person?" not "Does this profile explicitly mention every capability area?"
- The profile is a partial signal. A 30-minute conversation would reveal whether the depth is there. Your job is to decide if that conversation is worth having.

SAVE when the combination of datapoints makes a compelling case, even if no single datapoint is a perfect match. REJECT when the datapoints collectively point away from the role — wrong domain, wrong depth, wrong trajectory.

Return JSON only:
- "decision": "SAVE" or "REJECT"
- "path": matching archetype name, or "none"
- "confidence": float 0.0-1.0
- "rationale": 1-2 sentences with specific evidence"""


# ---------------------------------------------------------------------------
# V2 helpers: format pipeline schemas → text for structural templates
# ---------------------------------------------------------------------------

def _snippet_to_text(snippet: CandidateSnippet) -> str:
    """Format a CandidateSnippet as plain text for the facial template.

    Phase-0 hardening: candidate-controlled fields are defanged via
    ``defang_wire_format`` before interpolation so a scraped value (including a
    multi-line one with an embedded newline) cannot forge a batch verdict line
    that ``parse_facial_batch_response`` would attribute to a neighbor. Only the
    forge-able ``[N] FACIAL_*`` pattern is neutralized; legitimate content is
    preserved.
    """
    d = defang_wire_format
    lines = [
        f"Name: {d(snippet.name)}",
        f"Headline: {d(snippet.headline)}",
        f"Current Title: {d(snippet.current_title)}",
        f"Current Company: {d(snippet.current_company)}",
        f"Location: {d(snippet.location)}",
        f"Education: {d(snippet.education_snippet)}",
    ]
    if snippet.experience_entries:
        lines.append("")
        lines.append("Career History:")
        for entry in snippet.experience_entries:
            lines.append(f"- {d(entry)}")
    return "\n".join(lines)


def _profile_to_text(summary: CandidateProfileSummary) -> str:
    """Format a CandidateProfileSummary as plain text for the full eval template."""
    lines = [
        f"Name: {summary.name}",
        f"Headline: {summary.headline}",
        "",
        "About:",
        str(getattr(summary, "about", "") or "None listed"),
        "",
        "Experience:",
    ]
    if summary.experiences:
        for e in summary.experiences:
            bullets = "; ".join(e.summary_bullets) if e.summary_bullets else "no details"
            lines.append(f"- {e.title} at {e.company} ({e.start}-{e.end}): {bullets}")
    else:
        lines.append("None listed")

    lines.append("")
    lines.append("Education:")
    if summary.education:
        for e in summary.education:
            lines.append(f"- {e.degree} in {e.field}, {e.school} ({e.start}-{e.end})")
    else:
        lines.append("None listed")

    skills_text = ", ".join(summary.skills_snippet) if summary.skills_snippet else "none listed"
    lines.append("")
    lines.append(f"Skills: {skills_text}")
    return "\n".join(lines)


def _v2_full_profile_text(
    summary: CandidateProfileSummary,
    brief: Brief,
) -> str:
    """Render the same V2 baseline profile body for normal and enriched calls."""

    inner_brief = brief._new_brief
    if getattr(inner_brief, "dossier_mode", False):
        from exec_search.evidence_assembly import assemble_dossier_evidence
        from exec_search.signals import SignalRequestContext

        dossier = assemble_dossier_evidence(
            candidate=summary,
            brief=inner_brief,
            context=SignalRequestContext(
                brief_id=str(getattr(brief, "id", "") or inner_brief.role_title),
                trigger_reason="dossier_full_eval",
            ),
        )
        return dossier.prompt_body
    return _profile_to_text(summary)


def _full_tool_validation_inputs(brief: Brief) -> tuple[list[str], list[str]]:
    inner_brief = brief._new_brief
    capability_areas = list(inner_brief.capability_area_names())
    modifier_names = [
        str(getattr(modifier, "name", "") or "").strip()
        for modifier in (getattr(inner_brief, "post_save_modifiers", None) or [])
        if str(getattr(modifier, "name", "") or "").strip()
    ]
    return capability_areas, modifier_names


def _full_path(
    *,
    match_type: str | None,
    capability_area: str | None,
    transferability: str | None,
) -> str:
    if match_type and capability_area:
        path = f"{match_type}:{capability_area}"
    elif match_type:
        path = match_type.lower()
    else:
        path = capability_area or "none"
    if transferability and transferability not in ("N/A", None):
        path += f"|{transferability}"
    return path


def _full_semantic_evidence(result: object) -> dict[str, object]:
    """Preserve validated full-evaluation assessment evidence."""

    return {
        "match_type": getattr(result, "match_type", None),
        "capability_area": getattr(result, "capability_area", None),
        "capability_evidence": getattr(result, "capability_evidence", "") or "",
        "depth": getattr(result, "depth", None),
        "depth_evidence": getattr(result, "depth_evidence", "") or "",
        "transferability": getattr(result, "transferability", None),
        "transferability_evidence": (
            getattr(result, "transferability_evidence", "") or ""
        ),
        "evidence_recency": getattr(result, "evidence_recency", None),
        "level_alignment": getattr(result, "level_alignment", None),
        "opportunity_coherence": getattr(
            result, "opportunity_coherence", None
        ),
        "caliber": getattr(result, "caliber", None),
        "case_for": getattr(result, "case_for", "") or "",
        "case_against": getattr(result, "case_against", "") or "",
    }


# ---------------------------------------------------------------------------
# Stage 2: Facial judgment
# ---------------------------------------------------------------------------

@observe(name="judge.facial")
def facial_judge(
    snippet: CandidateSnippet,
    brief: Brief | None = None,
    prompt_prefix: str = "",
    lane_context: dict | None = None,
    opaque_candidate_id: str | None = None,
) -> OpusDecision:
    b = brief or _brief
    if not b:
        raise RuntimeError("Judger not initialized. Call init_judger(brief) or pass brief.")

    # --- V2 path: structural templates with prompt caching ---
    if b.has_v2_schema:
        contract_mode = _v2_judgment_contract_mode("facial")
        contract_version = (
            FACIAL_CONTRACT_VERSION
            if contract_mode == "tool"
            else _FACIAL_LEGACY_CONTRACT_VERSION
        )
        system = (
            assemble_facial_tool_system(b._new_brief, batch=False)
            if contract_mode == "tool"
            else assemble_facial_system(b._new_brief)
        )
        snippet_text = _snippet_to_text(snippet)
        candidate_id = (
            opaque_candidate_id or generate_opaque_candidate_ids(1)[0]
            if contract_mode == "tool"
            else ""
        )
        if contract_mode == "tool":
            user_msg = render_facial_tool_user_message(
                [snippet_text],
                [candidate_id],
                prompt_prefix=prompt_prefix,
            )
        else:
            user_msg = f"{prompt_prefix}{snippet_text}" if prompt_prefix else snippet_text
        usage_context = _judgment_usage_context(
            lane_context,
            stage="facial",
            contract_mode=contract_mode,
            contract_version=contract_version,
            batch_size=1,
        )
        policy = _fireworks_judgment_policy(
            stage="facial",
            system_prompt=system,
            contract_version=contract_version,
            usage_context=usage_context,
        )
        allow_borderline = _facial_ternary_selected(b._new_brief)
        render_route = (
            "linkedin.facial.v2_tool_v1"
            if contract_mode == "tool"
            else "linkedin.facial.v2_structural"
        )
        try:
            raw = facial_llm(
                system,
                user_msg,
                expect_json=False,
                usage_context=usage_context,
                policy=policy,
                tool_contract=(
                    facial_tool_contract(allow_borderline=allow_borderline)
                    if contract_mode == "tool"
                    else None
                ),
            )
        except Exception as e:
            _reraise_if_budget_exhausted(e)
            if contract_mode == "tool":
                # A forced-tool transport failure is page/run level. Turning
                # auth/schema/provider exhaustion into a candidate verdict
                # would keep the browser paging while judgment is unavailable.
                raise
            logger.warning("V2 facial judge exception: %s", e)
            return _attach_prompt_capture(
                judgment_failure_decision(
                    stage="facial",
                    candidate_name=snippet.name,
                    profile_url=snippet.profile_url,
                    error=e,
                    source="judgment",
                ),
                _build_prompt_capture(
                    stage="facial",
                    source="linkedin",
                    render_route=render_route,
                    llm_caller_name="facial_llm",
                    expect_json=False,
                    system_prompt=system,
                    candidate_text=user_msg,
                    usage_context=usage_context,
                ),
            )
        if contract_mode == "tool":
            try:
                if not isinstance(raw, dict):
                    raise JudgmentToolContractError(
                        "tool_arguments_not_object", type(raw).__name__
                    )
                result = validate_facial_tool_arguments(
                    raw,
                    expected_ids=[candidate_id],
                    allow_borderline=allow_borderline,
                )[0]
            except JudgmentToolContractError as e:
                logger.warning("V2 facial tool-contract failure: %s", e)
                return _attach_prompt_capture(
                    parse_failure_decision(
                        stage="facial",
                        candidate_name=snippet.name,
                        profile_url=snippet.profile_url,
                        reason=e.reason,
                        detail=e.detail,
                    ),
                    _build_prompt_capture(
                        stage="facial",
                        source="linkedin",
                        render_route=render_route,
                        llm_caller_name="facial_llm",
                        expect_json=False,
                        system_prompt=system,
                        candidate_text=user_msg,
                        usage_context=usage_context,
                    ),
                )
        else:
            result = parse_facial_response(raw)
        # P5.4: the V2 facial contract has no confidence field to parse — the
        # prompt is binary triage (DECISION/REASON only). 1.0 was fabricated.
        # PARSE_FAILURE/JUDGMENT_FAILURE keep 0.0 (matches judgment_failure_decision
        # / parse_failure_decision elsewhere); a genuine valid verdict has no
        # confidence signal at all, so it is None, not a fabricated number.
        confidence = 0.0 if is_failure_decision(result.decision) else None
        # GLM-5.2 shadow judge: fires AFTER the real verdict (result.decision)
        # already exists and returns nothing — the returned OpusDecision below
        # is built from `result`/`confidence` exactly as before this hook
        # existed. See the shadow-hook block's module docstring above
        # facial_judge for the zero-influence doctrine.
        if contract_mode == "legacy":
            _run_facial_shadow_single(
                system_prompt=system,
                user_prompt=user_msg,
                # 8192, not the primary's 2048: sibling of the batch-cap fix
                # (38495f9) — GLM-5.2 spends heavy reasoning tokens before the
                # DECISION line, and the shadow's cap must not be the binding
                # constraint or parse-failure conflates "GLM can't hold the
                # format" with "we cut it off mid-sentence".
                max_tokens=8192,
                primary_decision=result.decision,
                parse_fn=lambda raw_text: parse_facial_response(raw_text).decision,
                lane_context=lane_context,
            )
        return _attach_prompt_capture(
            OpusDecision(
                stage="facial",
                decision=result.decision,
                path="none",
                confidence=confidence,
                rationale=result.reason,
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            ),
            _build_prompt_capture(
                stage="facial",
                source="linkedin",
                render_route=render_route,
                llm_caller_name="facial_llm",
                expect_json=False,
                system_prompt=system,
                candidate_text=user_msg,
                usage_context=usage_context,
            ),
        )

    # --- Old path: original prompt builders ---
    system = _build_facial_system(b)

    career_section = ""
    if snippet.experience_entries:
        career_lines = "\n".join(f"- {e}" for e in snippet.experience_entries)
        career_section = f"\n\nCareer History:\n{career_lines}"

    user_prompt = f"""## Candidate Snippet
Name: {snippet.name}
Headline: {snippet.headline}
Current Title: {snippet.current_title}
Current Company: {snippet.current_company}
Location: {snippet.location}
Education: {snippet.education_snippet}{career_section}

Decide: FACIAL_YES or FACIAL_NO."""

    usage_context: dict[str, object] = {}
    usage_context.setdefault("stage", "facial")
    try:
        result = opus_llm_cached(
            system,
            user_prompt,
            expect_json=True,
            usage_context=usage_context,
            model_name=config.FACIAL_MODEL_NAME,
        )
    except Exception as e:
        _reraise_if_budget_exhausted(e)
        logger.warning("old-brief facial judge exception: %s", e)
        return _attach_prompt_capture(
            judgment_failure_decision(
                stage="facial",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                error=e,
                source="judgment",
            ),
            _build_prompt_capture(
                stage="facial",
                source="linkedin",
                render_route="linkedin.facial.legacy_json",
                llm_caller_name="opus_llm_cached",
                expect_json=True,
                system_prompt=system,
                candidate_text=user_prompt,
                usage_context=usage_context,
            ),
        )

    raw_decision = result.get("decision") if isinstance(result, dict) else None
    if raw_decision not in _VALID_FACIAL:
        logger.warning("facial parse-failure: decision=%r (old-brief path)", raw_decision)
        return _attach_prompt_capture(
            parse_failure_decision(
                stage="facial",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                reason="invalid_decision",
                detail=f"decision={raw_decision!r}",
            ),
            _build_prompt_capture(
                stage="facial",
                source="linkedin",
                render_route="linkedin.facial.legacy_json",
                llm_caller_name="opus_llm_cached",
                expect_json=True,
                system_prompt=system,
                candidate_text=user_prompt,
                usage_context=usage_context,
            ),
        )
    # GLM-5.2 shadow judge: fires AFTER the real verdict (raw_decision)
    # already exists; return value is discarded. See legacy_shadow parser
    # docstring for why this branch parses differently than the V2 branch
    # (legacy old-brief calls are JSON, V2 calls are plain DECISION/REASON
    # text).
    _run_facial_shadow_single(
        system_prompt=system,
        user_prompt=user_prompt,
        max_tokens=8192,
        primary_decision=raw_decision,
        parse_fn=_parse_legacy_shadow_decision,
        lane_context=lane_context,
    )
    return _attach_prompt_capture(
        OpusDecision(
            stage="facial",
            decision=raw_decision,
            path=result.get("path", "none"),
            confidence=_safe_confidence(result.get("confidence", 0.5)),
            rationale=result.get("rationale", ""),
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        ),
        _build_prompt_capture(
            stage="facial",
            source="linkedin",
            render_route="linkedin.facial.legacy_json",
            llm_caller_name="opus_llm_cached",
            expect_json=True,
            system_prompt=system,
            candidate_text=user_prompt,
            usage_context=usage_context,
        ),
    )


# ---------------------------------------------------------------------------
# Stage 4: Full judgment
# ---------------------------------------------------------------------------

@observe(name="judge.full")
def full_judge(
    summary: CandidateProfileSummary,
    brief: Brief | None = None,
    lane_context: dict | None = None,
    opaque_candidate_id: str | None = None,
) -> OpusDecision:
    b = brief or _brief
    if not b:
        raise RuntimeError("Judger not initialized. Call init_judger(brief) or pass brief.")

    # --- V2 path: structural templates with prompt caching ---
    if b.has_v2_schema:
        contract_mode = _v2_judgment_contract_mode("full")
        contract_version = (
            FULL_CONTRACT_VERSION
            if contract_mode == "tool"
            else _FULL_LEGACY_CONTRACT_VERSION
        )
        system = (
            assemble_full_evaluation_tool_system(b._new_brief)
            if contract_mode == "tool"
            else assemble_full_evaluation_system(b._new_brief)
        )
        profile_text = _v2_full_profile_text(summary, b)
        if contract_mode == "legacy":
            profile_text = defang_wire_format(profile_text)
        candidate_id = (
            opaque_candidate_id or generate_opaque_candidate_ids(1)[0]
            if contract_mode == "tool"
            else ""
        )
        user_message = (
            render_full_tool_user_message(profile_text, candidate_id)
            if contract_mode == "tool"
            else profile_text
        )
        usage_context = _judgment_usage_context(
            lane_context,
            stage="full",
            contract_mode=contract_mode,
            contract_version=contract_version,
        )
        policy = _fireworks_judgment_policy(
            stage="full",
            system_prompt=system,
            contract_version=contract_version,
            usage_context=usage_context,
        )
        render_route = (
            "linkedin.full.v2_tool_v2"
            if contract_mode == "tool"
            else "linkedin.full.v2_structural"
        )
        capability_areas, modifier_names = _full_tool_validation_inputs(b)
        try:
            raw = opus_llm_cached(
                system,
                user_message,
                expect_json=False,
                usage_context=usage_context,
                model_name=config.FULL_EVAL_MODEL_NAME,
                policy=policy,
                tool_contract=(
                    full_tool_contract(
                        capability_areas=capability_areas,
                        post_save_modifiers=modifier_names,
                    )
                    if contract_mode == "tool"
                    else None
                ),
            )
        except Exception as e:
            _reraise_if_budget_exhausted(e)
            if contract_mode == "tool":
                raise
            logger.warning("V2 full judge exception: %s", e)
            return _attach_prompt_capture(
                judgment_failure_decision(
                    stage="full",
                    candidate_name=summary.name,
                    profile_url=summary.profile_url,
                    error=e,
                    source="judgment",
                ),
                _build_prompt_capture(
                    stage="full",
                    source="linkedin",
                    render_route=render_route,
                    llm_caller_name="opus_llm_cached",
                    expect_json=False,
                    system_prompt=system,
                    candidate_text=user_message,
                    usage_context=usage_context,
                ),
            )
        if contract_mode == "tool":
            try:
                if not isinstance(raw, dict):
                    raise JudgmentToolContractError(
                        "tool_arguments_not_object", type(raw).__name__
                    )
                result = validate_full_tool_arguments(
                    raw,
                    expected_id=candidate_id,
                    capability_areas=capability_areas,
                    post_save_modifiers=modifier_names,
                )
            except JudgmentToolContractError as e:
                logger.warning("V2 full tool-contract failure: %s", e)
                failure_capture = _build_prompt_capture(
                    stage="full",
                    source="linkedin",
                    render_route=render_route,
                    llm_caller_name="opus_llm_cached",
                    expect_json=False,
                    system_prompt=system,
                    candidate_text=user_message,
                    usage_context=usage_context,
                )
                # Structured channel for callers that must classify this
                # failure without parsing the rationale's model-derived
                # detail (the orchestrator's candidate-ID mismatch re-issue).
                failure_capture["judgment_contract_failure"] = (
                    _full_contract_failure_record(
                        e,
                        expected_candidate_id=candidate_id,
                        arguments=raw,
                    )
                )
                return _attach_prompt_capture(
                    parse_failure_decision(
                        stage="full",
                        candidate_name=summary.name,
                        profile_url=summary.profile_url,
                        reason=e.reason,
                        detail=e.detail,
                    ),
                    failure_capture,
                )
        else:
            result = parse_full_evaluation_response(
                raw,
                require_semantic_v2=True,
                capability_areas=capability_areas,
                post_save_modifiers=modifier_names,
            )
        # GLM-5.2 shadow judge (full-eval extension): fires AFTER the real
        # verdict (result.decision) already exists and returns nothing —
        # the OpusDecision built below from `result` is unaffected. Same
        # system+user prompts as the primary call above (`system`,
        # `profile_text`); max_tokens is SHADOW-OWNED at 16384, not the
        # primary's 8192 — sibling of the batch-cap fix (38495f9) and the
        # facial-single 8192-vs-2048 above. Fireworks counts GLM's
        # reasoning tokens against max_tokens, and both 2026-07-05 live
        # full-eval PARSE_FAILUREs were finish_reason=length (one perfectly
        # formatted response cut off two lines before DECISION after 34K
        # chars of reasoning). The shadow's cap must not be the binding
        # constraint or parse-failure conflates "GLM can't hold the format"
        # with "we cut it off mid-sentence". See the shadow-hook module
        # docstring and _run_full_shadow_single for scope (LinkedIn
        # V2-structural only).
        if contract_mode == "legacy":
            _run_full_shadow_single(
                system_prompt=system,
                user_prompt=profile_text,
                max_tokens=16384,
                primary_decision=result.decision,
                capability_areas=tuple(capability_areas),
                post_save_modifiers=tuple(modifier_names),
                lane_context=lane_context,
            )
        path = _full_path(
            match_type=result.match_type,
            capability_area=result.capability_area,
            transferability=result.transferability,
        )
        return _attach_prompt_capture(
            OpusDecision(
                stage="full",
                decision=result.decision,
                path=path,
                confidence=result.confidence,
                rationale=result.summary or result.case_for or "[parse error]",
                candidate_name=summary.name,
                profile_url=summary.profile_url,
                post_save_modifier=getattr(result, "post_save_modifier", "NONE"),
                outreach_tier=getattr(result, "outreach_tier", None) or "",
                reject_reason=getattr(result, "reject_reason", None) or "",
                # P4: bounded non-save review evidence flows through from
                # the parser. Empty for SAVE / REJECT so to_dict() output
                # is byte-identical to pre-P4 behavior.
                review_reason_code=getattr(result, "review_reason_code", "") or "",
                review_structural_evidence=list(
                    getattr(result, "review_structural_evidence", []) or []
                ),
                review_recommended_next_step=(
                    getattr(result, "review_recommended_next_step", "") or ""
                ),
                semantic_evidence=_full_semantic_evidence(result),
            ),
            _build_prompt_capture(
                stage="full",
                source="linkedin",
                render_route=render_route,
                llm_caller_name="opus_llm_cached",
                expect_json=False,
                system_prompt=system,
                candidate_text=user_message,
                usage_context=usage_context,
            ),
        )

    # --- Old path: original prompt builders ---
    system = _build_full_system(b)

    exp_text = ""
    for e in summary.experiences:
        bullets = "; ".join(e.summary_bullets) if e.summary_bullets else "no details"
        exp_text += f"- {e.title} at {e.company} ({e.start}-{e.end}): {bullets}\n"

    edu_text = ""
    for e in summary.education:
        edu_text += f"- {e.degree} in {e.field}, {e.school} ({e.start}-{e.end})\n"

    skills_text = ", ".join(summary.skills_snippet) if summary.skills_snippet else "none listed"

    user_prompt = f"""## Candidate Profile
Name: {summary.name}
Headline: {summary.headline}

About:
{summary.about or "None listed"}

Experience:
{exp_text if exp_text else "None listed"}

Education:
{edu_text if edu_text else "None listed"}

Skills: {skills_text}

Decide: SAVE or REJECT."""

    usage_context: dict[str, object] = {}
    try:
        result = opus_llm_cached(
            system,
            user_prompt,
            expect_json=True,
            usage_context=usage_context,
            model_name=config.FULL_EVAL_MODEL_NAME,
        )
    except Exception as e:
        _reraise_if_budget_exhausted(e)
        logger.warning("old-brief full judge exception: %s", e)
        return _attach_prompt_capture(
            judgment_failure_decision(
                stage="full",
                candidate_name=summary.name,
                profile_url=summary.profile_url,
                error=e,
                source="judgment",
            ),
            _build_prompt_capture(
                stage="full",
                source="linkedin",
                render_route="linkedin.full.legacy_json",
                llm_caller_name="opus_llm_cached",
                expect_json=True,
                system_prompt=system,
                candidate_text=user_prompt,
                usage_context=usage_context,
            ),
        )

    raw_decision = result.get("decision") if isinstance(result, dict) else None
    if raw_decision not in _VALID_FULL:
        logger.warning("full parse-failure: decision=%r (old-brief path)", raw_decision)
        return _attach_prompt_capture(
            parse_failure_decision(
                stage="full",
                candidate_name=summary.name,
                profile_url=summary.profile_url,
                reason="invalid_decision",
                detail=f"decision={raw_decision!r}",
            ),
            _build_prompt_capture(
                stage="full",
                source="linkedin",
                render_route="linkedin.full.legacy_json",
                llm_caller_name="opus_llm_cached",
                expect_json=True,
                system_prompt=system,
                candidate_text=user_prompt,
                usage_context=usage_context,
            ),
        )
    return _attach_prompt_capture(
        OpusDecision(
            stage="full",
            decision=raw_decision,
            path=result.get("path", "none"),
            confidence=_safe_confidence(result.get("confidence", 0.5)),
            rationale=result.get("rationale", ""),
            candidate_name=summary.name,
            profile_url=summary.profile_url,
            # P4: legacy JSON path — optional review_* fields default to
            # empty when keys are absent so SAVE / REJECT payloads stay
            # byte-identical. The legacy old-brief prompt does not ask
            # for REVIEW_* output, so these are only set if a model
            # spontaneously emits them.
            review_reason_code=str(result.get("review_reason_code", "") or ""),
            review_structural_evidence=_as_str_list(
                result.get("review_structural_evidence")
            ),
            review_recommended_next_step=str(
                result.get("review_recommended_next_step", "") or ""
            ),
        ),
        _build_prompt_capture(
            stage="full",
            source="linkedin",
            render_route="linkedin.full.legacy_json",
            llm_caller_name="opus_llm_cached",
            expect_json=True,
            system_prompt=system,
            candidate_text=user_prompt,
            usage_context=usage_context,
        ),
    )


# ---------------------------------------------------------------------------
# Stage 4 (shadow): Full judgment with external evidence augmentation
# ---------------------------------------------------------------------------
# Sibling to ``full_judge`` for slice 2 of perplexity-evidence-augmentation.
# The v2 path uses the IDENTICAL ``assemble_full_evaluation_system`` system
# prompt as ``full_judge`` so prompt-cache hits on the static prefix are
# preserved; the external-evidence block is appended only to the user message.
# ``full_judge`` itself is not called from here — they are siblings, not
# layered. Failure parity with ``full_judge`` is required so behavior is
# comparable on failure too.


def _format_external_evidence_block(evidence: ExternalCandidateEvidence) -> str:
    """Render external evidence as a deterministic, fenced text block.

    Determinism note: lists that are user-visible (facts, ambiguities,
    do_not_use_for_judgment, evidence_refs) are emitted in input order.
    Fact blocks and inferences are emitted in input order too — the normalizer
    already preserves provider order, and downstream tests assert on stable
    output for the same input.
    """

    lines: list[str] = []
    lines.append("## External Evidence (NOT a judgment — augmentation only)")
    trigger_reason = evidence.trigger_reason or "unspecified"
    identity_conf = (
        f"{evidence.identity_confidence:.2f}"
        if isinstance(evidence.identity_confidence, (int, float))
        else "0.00"
    )
    lines.append(
        f"trigger_reason={trigger_reason} | identity_confidence={identity_conf}"
    )

    lines.append("")
    lines.append("### Sourced facts")
    if evidence.external_fact_blocks:
        for block in evidence.external_fact_blocks:
            topic = block.topic or "(no topic)"
            quality = block.source_quality or "unknown"
            lines.append(f"- topic: {topic} (source_quality={quality})")
            if block.facts:
                for fact in block.facts:
                    refs = ", ".join(ref.url for ref in block.evidence_refs if ref.url)
                    refs_text = f" [refs: {refs}]" if refs else ""
                    lines.append(f"  - {fact}{refs_text}")
            else:
                lines.append("  - (no facts)")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("### Model inferences (not first-party facts; treat with caution)")
    if evidence.external_inferences:
        for inference in evidence.external_inferences:
            try:
                conf_text = f"{float(inference.confidence):.2f}"
            except (TypeError, ValueError):
                conf_text = "0.00"
            basis = ", ".join(ref.url for ref in inference.basis_refs if ref.url)
            basis_text = f" [basis: {basis}]" if basis else ""
            lines.append(
                f"- {inference.claim} (confidence={conf_text}){basis_text}"
            )
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("### Unresolved ambiguities")
    if evidence.unresolved_ambiguities:
        for amb in evidence.unresolved_ambiguities:
            lines.append(f"- {amb}")
    else:
        lines.append("(none)")

    if evidence.do_not_use_for_judgment:
        lines.append("")
        lines.append("### Do not use for judgment")
        for item in evidence.do_not_use_for_judgment:
            lines.append(f"- {item}")

    return "\n".join(lines)


@observe(name="judge.full_with_external_evidence")
def full_judge_with_external_evidence(
    summary: CandidateProfileSummary,
    evidence: ExternalCandidateEvidence,
    brief: Brief | None = None,
    lane_context: dict | None = None,
    opaque_candidate_id: str | None = None,
) -> OpusDecision:
    """Full judgment with external evidence augmentation (shadow path).

    Mirrors ``full_judge`` exactly in v2 / old-brief branching so the two
    paths can be compared 1:1. The system prompt is *unchanged* relative to
    ``full_judge`` to preserve prompt-cache hits on the static prefix; the
    external-evidence block is appended only to the user message.

    This is an optional augmentation after a valid baseline judgment. Its
    provider/contract failures therefore remain fail-soft and return
    ``judgment_failure_decision`` / ``parse_failure_decision`` so the caller
    can preserve the baseline. API-budget exhaustion still propagates. The
    primary ``full_judge`` tool path is intentionally stricter and aborts on
    transport failure.
    """

    b = brief or _brief
    if not b:
        raise RuntimeError("Judger not initialized. Call init_judger(brief) or pass brief.")

    evidence_block = _format_external_evidence_block(evidence)

    if b.has_v2_schema:
        contract_mode = _v2_judgment_contract_mode("full")
        contract_version = (
            FULL_CONTRACT_VERSION
            if contract_mode == "tool"
            else _FULL_LEGACY_CONTRACT_VERSION
        )
        system = (
            assemble_full_evaluation_tool_system(b._new_brief)
            if contract_mode == "tool"
            else assemble_full_evaluation_system(b._new_brief)
        )
        profile_text = _v2_full_profile_text(summary, b)
        if contract_mode == "legacy":
            profile_text = defang_wire_format(profile_text)
            if evidence_block:
                evidence_block = defang_wire_format(evidence_block)
        candidate_id = (
            opaque_candidate_id or generate_opaque_candidate_ids(1)[0]
            if contract_mode == "tool"
            else ""
        )
        user_msg = (
            render_full_tool_user_message(
                profile_text,
                candidate_id,
                external_evidence_block=evidence_block,
            )
            if contract_mode == "tool"
            else profile_text + "\n\n" + evidence_block
        )
        usage_context = _judgment_usage_context(
            lane_context,
            stage="full",
            contract_mode=contract_mode,
            contract_version=contract_version,
        )
        baseline_call_id = str(usage_context.get("logical_call_id") or "")
        if baseline_call_id:
            usage_context["parent_logical_call_id"] = baseline_call_id
        usage_context["logical_call_id"] = f"judge-{uuid.uuid4().hex}"
        usage_context["judgment_variant"] = "external_evidence"
        policy = _fireworks_judgment_policy(
            stage="full",
            system_prompt=system,
            contract_version=contract_version,
            usage_context=usage_context,
        )
        render_route = (
            "linkedin.full.v2_external_evidence_tool_v2"
            if contract_mode == "tool"
            else "linkedin.full.v2_external_evidence"
        )
        capability_areas, modifier_names = _full_tool_validation_inputs(b)
        try:
            raw = opus_llm_cached(
                system,
                user_msg,
                expect_json=False,
                usage_context=usage_context,
                model_name=config.FULL_EVAL_MODEL_NAME,
                policy=policy,
                tool_contract=(
                    full_tool_contract(
                        capability_areas=capability_areas,
                        post_save_modifiers=modifier_names,
                    )
                    if contract_mode == "tool"
                    else None
                ),
            )
        except Exception as e:
            _reraise_if_budget_exhausted(e)
            logger.warning("V2 full judge (with evidence) exception: %s", e)
            return _attach_prompt_capture(
                judgment_failure_decision(
                    stage="full",
                    candidate_name=summary.name,
                    profile_url=summary.profile_url,
                    error=e,
                    source="judgment",
                ),
                _build_prompt_capture(
                    stage="full",
                    source="linkedin",
                    render_route=render_route,
                    llm_caller_name="opus_llm_cached",
                    expect_json=False,
                    system_prompt=system,
                    candidate_text=user_msg,
                    usage_context=usage_context,
                ),
            )
        if contract_mode == "tool":
            try:
                if not isinstance(raw, dict):
                    raise JudgmentToolContractError(
                        "tool_arguments_not_object", type(raw).__name__
                    )
                result = validate_full_tool_arguments(
                    raw,
                    expected_id=candidate_id,
                    capability_areas=capability_areas,
                    post_save_modifiers=modifier_names,
                )
            except JudgmentToolContractError as e:
                logger.warning(
                    "V2 full external-evidence tool-contract failure: %s", e
                )
                return _attach_prompt_capture(
                    parse_failure_decision(
                        stage="full",
                        candidate_name=summary.name,
                        profile_url=summary.profile_url,
                        reason=e.reason,
                        detail=e.detail,
                    ),
                    _build_prompt_capture(
                        stage="full",
                        source="linkedin",
                        render_route=render_route,
                        llm_caller_name="opus_llm_cached",
                        expect_json=False,
                        system_prompt=system,
                        candidate_text=user_msg,
                        usage_context=usage_context,
                    ),
                )
        else:
            result = parse_full_evaluation_response(
                raw,
                require_semantic_v2=True,
                capability_areas=capability_areas,
                post_save_modifiers=modifier_names,
            )
        path = _full_path(
            match_type=result.match_type,
            capability_area=result.capability_area,
            transferability=result.transferability,
        )
        return _attach_prompt_capture(
            OpusDecision(
                stage="full",
                decision=result.decision,
                path=path,
                confidence=result.confidence,
                rationale=result.summary or result.case_for or "[parse error]",
                candidate_name=summary.name,
                profile_url=summary.profile_url,
                post_save_modifier=getattr(result, "post_save_modifier", "NONE"),
                outreach_tier=getattr(result, "outreach_tier", None) or "",
                reject_reason=getattr(result, "reject_reason", None) or "",
                review_reason_code=getattr(result, "review_reason_code", "") or "",
                review_structural_evidence=list(
                    getattr(result, "review_structural_evidence", []) or []
                ),
                review_recommended_next_step=(
                    getattr(result, "review_recommended_next_step", "") or ""
                ),
                semantic_evidence=_full_semantic_evidence(result),
            ),
            _build_prompt_capture(
                stage="full",
                source="linkedin",
                render_route=render_route,
                llm_caller_name="opus_llm_cached",
                expect_json=False,
                system_prompt=system,
                candidate_text=user_msg,
                usage_context=usage_context,
            ),
        )

    system = _build_full_system(b)

    exp_text = ""
    for e in summary.experiences:
        bullets = "; ".join(e.summary_bullets) if e.summary_bullets else "no details"
        exp_text += f"- {e.title} at {e.company} ({e.start}-{e.end}): {bullets}\n"

    edu_text = ""
    for e in summary.education:
        edu_text += f"- {e.degree} in {e.field}, {e.school} ({e.start}-{e.end})\n"

    skills_text = ", ".join(summary.skills_snippet) if summary.skills_snippet else "none listed"

    user_prompt = f"""## Candidate Profile
Name: {summary.name}
Headline: {summary.headline}

About:
{summary.about or "None listed"}

Experience:
{exp_text if exp_text else "None listed"}

Education:
{edu_text if edu_text else "None listed"}

Skills: {skills_text}

{evidence_block}

Decide: SAVE or REJECT."""

    usage_context: dict[str, object] = {}
    try:
        result = opus_llm_cached(
            system,
            user_prompt,
            expect_json=True,
            usage_context=usage_context,
            model_name=config.FULL_EVAL_MODEL_NAME,
        )
    except Exception as e:
        _reraise_if_budget_exhausted(e)
        logger.warning("old-brief full judge (with evidence) exception: %s", e)
        return _attach_prompt_capture(
            judgment_failure_decision(
                stage="full",
                candidate_name=summary.name,
                profile_url=summary.profile_url,
                error=e,
                source="judgment",
            ),
            _build_prompt_capture(
                stage="full",
                source="linkedin",
                render_route="linkedin.full.legacy_external_evidence",
                llm_caller_name="opus_llm_cached",
                expect_json=True,
                system_prompt=system,
                candidate_text=user_prompt,
                usage_context=usage_context,
            ),
        )

    raw_decision = result.get("decision") if isinstance(result, dict) else None
    if raw_decision not in _VALID_FULL:
        logger.warning(
            "full parse-failure (with evidence): decision=%r (old-brief path)",
            raw_decision,
        )
        return _attach_prompt_capture(
            parse_failure_decision(
                stage="full",
                candidate_name=summary.name,
                profile_url=summary.profile_url,
                reason="invalid_decision",
                detail=f"decision={raw_decision!r}",
            ),
            _build_prompt_capture(
                stage="full",
                source="linkedin",
                render_route="linkedin.full.legacy_external_evidence",
                llm_caller_name="opus_llm_cached",
                expect_json=True,
                system_prompt=system,
                candidate_text=user_prompt,
                usage_context=usage_context,
            ),
        )
    return _attach_prompt_capture(
        OpusDecision(
            stage="full",
            decision=raw_decision,
            path=result.get("path", "none"),
            confidence=_safe_confidence(result.get("confidence", 0.5)),
            rationale=result.get("rationale", ""),
            candidate_name=summary.name,
            profile_url=summary.profile_url,
            # P4: legacy external-evidence path mirrors the legacy JSON
            # path. Optional review_* keys are tolerated, defaulting to
            # empty so SAVE / REJECT payloads stay byte-identical.
            review_reason_code=str(result.get("review_reason_code", "") or ""),
            review_structural_evidence=_as_str_list(
                result.get("review_structural_evidence")
            ),
            review_recommended_next_step=str(
                result.get("review_recommended_next_step", "") or ""
            ),
        ),
        _build_prompt_capture(
            stage="full",
            source="linkedin",
            render_route="linkedin.full.legacy_external_evidence",
            llm_caller_name="opus_llm_cached",
            expect_json=True,
            system_prompt=system,
            candidate_text=user_prompt,
            usage_context=usage_context,
        ),
    )


# ---------------------------------------------------------------------------
# GitHub-specific judges
# ---------------------------------------------------------------------------

@observe(name="judge.github_facial")
def github_facial_judge(portfolio_text: str, brief: Brief | None = None) -> OpusDecision:
    """GitHub facial triage — uses portfolio summary from cheap model extraction.

    Unlike LinkedIn facial which receives a CandidateSnippet, GitHub facial
    receives the structured portfolio text (toolchain, repos, contributions).
    """
    from github.judgment_templates import (
        assemble_github_facial_system,
        parse_facial_response,
    )

    b = brief or _brief
    if not b:
        raise RuntimeError("Judger not initialized. Call init_judger(brief) or pass brief.")

    if not b.has_v2_schema:
        raise RuntimeError("GitHub judges require a V2 brief with capability_areas.")

    system = assemble_github_facial_system(b._new_brief)
    usage_context: dict[str, object] = {}
    usage_context.setdefault("stage", "facial")
    try:
        raw = facial_llm(
            system,
            portfolio_text,
            expect_json=False,
            usage_context=usage_context,
        )
    except Exception as e:
        _reraise_if_budget_exhausted(e)
        logger.warning("GitHub facial judge exception: %s", e)
        return _attach_prompt_capture(
            judgment_failure_decision(
                stage="facial",
                candidate_name="",
                profile_url="",
                error=e,
                source="judgment",
            ),
            _build_prompt_capture(
                stage="facial",
                source="github",
                render_route="github.facial.v2_structural",
                llm_caller_name="facial_llm",
                expect_json=False,
                system_prompt=system,
                candidate_text=portfolio_text,
                usage_context=usage_context,
            ),
        )
    result = parse_facial_response(raw)

    return _attach_prompt_capture(
        OpusDecision(
            stage="facial",
            decision=result.decision,
            path="none",
            # P5.4: same reasoning as linkedin facial_judge above — the V2
            # GitHub facial contract has no confidence field; only
            # PARSE_FAILURE/JUDGMENT_FAILURE get the conventional 0.0.
            confidence=0.0 if is_failure_decision(result.decision) else None,
            rationale=result.reason,
            candidate_name="",  # Caller sets this
            profile_url="",     # Caller sets this
        ),
        _build_prompt_capture(
            stage="facial",
            source="github",
            render_route="github.facial.v2_structural",
            llm_caller_name="facial_llm",
            expect_json=False,
            system_prompt=system,
            candidate_text=portfolio_text,
            usage_context=usage_context,
        ),
    )


@observe(name="judge.github_full")
def github_full_judge(evidence_text: str, brief: Brief | None = None) -> OpusDecision:
    """GitHub full evaluation — uses enriched evidence text.

    Unlike LinkedIn full which receives a CandidateProfileSummary, GitHub full
    receives the complete evidence text (toolchain, repos, READMEs, papers, etc.).
    """
    from github.judgment_templates import (
        assemble_github_full_evaluation_system,
        parse_full_evaluation_response,
    )

    b = brief or _brief
    if not b:
        raise RuntimeError("Judger not initialized. Call init_judger(brief) or pass brief.")

    if not b.has_v2_schema:
        raise RuntimeError("GitHub judges require a V2 brief with capability_areas.")

    system = assemble_github_full_evaluation_system(b._new_brief)
    usage_context: dict[str, object] = {}
    try:
        raw = opus_llm_cached(
            system,
            evidence_text,
            expect_json=False,
            usage_context=usage_context,
            model_name=config.FULL_EVAL_MODEL_NAME,
        )
    except Exception as e:
        _reraise_if_budget_exhausted(e)
        logger.warning("GitHub full judge exception: %s", e)
        return _attach_prompt_capture(
            judgment_failure_decision(
                stage="full",
                candidate_name="",
                profile_url="",
                error=e,
                source="judgment",
            ),
            _build_prompt_capture(
                stage="full",
                source="github",
                render_route="github.full.v2_structural",
                llm_caller_name="opus_llm_cached",
                expect_json=False,
                system_prompt=system,
                candidate_text=evidence_text,
                usage_context=usage_context,
            ),
        )
    result = parse_full_evaluation_response(raw)

    # Build path from match_type + capability_area
    if result.match_type and result.capability_area:
        path = f"{result.match_type}:{result.capability_area}"
    elif result.match_type:
        path = result.match_type.lower()
    else:
        path = result.capability_area or "none"
    if result.transferability and result.transferability not in ("N/A", None):
        path += f"|{result.transferability}"

    return _attach_prompt_capture(
        OpusDecision(
            stage="full",
            decision=result.decision,
            path=path,
            confidence=result.confidence,
            rationale=result.summary or result.case_for or "[parse error]",
            candidate_name="",  # Caller sets this
            profile_url="",     # Caller sets this
        ),
        _build_prompt_capture(
            stage="full",
            source="github",
            render_route="github.full.v2_structural",
            llm_caller_name="opus_llm_cached",
            expect_json=False,
            system_prompt=system,
            candidate_text=evidence_text,
            usage_context=usage_context,
        ),
    )


# ---------------------------------------------------------------------------
# Batch facial triage (Phase 2)
# ---------------------------------------------------------------------------

def _count_valid_verdicts(results: list[FacialResult]) -> int:
    """Number of slots that parsed to a real (non-PARSE_FAILURE) verdict."""
    return sum(1 for r in results if not is_failure_decision(r.decision))


def _batch_results_trustworthy(results: list[FacialResult], count: int) -> bool:
    """Whether positional attribution of a parsed batch is safe.

    The batch parser keys results by the LLM-emitted ``[N]`` index and the
    caller attaches each verdict to a snippet by position. That is only sound
    when the model emitted exactly one valid verdict per candidate — i.e. the
    number of distinctly-parsed valid indices equals ``count``. Anything less
    means at least one slot is a gap, and we cannot rule out that the model
    dropped a candidate and renumbered the survivors (so ``[1]`` describes
    snippet[1] rather than snippet[0]). In that case positional attribution is
    untrustworthy and the caller must re-judge sequentially.

    The common well-formed in-order batch (one ``[N]`` line per candidate,
    ``1..count``) yields ``count`` valid verdicts and is therefore trusted —
    byte-identical to the prior fast path.
    """
    return _count_valid_verdicts(results) == count


@observe(name="judge.facial_batch")
def facial_judge_batch(
    snippets: list[CandidateSnippet],
    brief: Brief | None = None,
    prompt_prefix: str = "",
    lane_context: dict | None = None,
    opaque_candidate_ids: Sequence[str] | None = None,
) -> list[OpusDecision]:
    """Batch facial triage — one LLM call for all snippets on a page.

    V2 briefs only. Falls back to sequential for old briefs or on batch failure.
    """
    b = brief or _brief
    if not b:
        raise RuntimeError("Judger not initialized. Call init_judger(brief) or pass brief.")

    if not b.has_v2_schema:
        return [
            facial_judge(
                snippet,
                b,
                prompt_prefix=prompt_prefix,
                lane_context=_facial_fallback_context(
                    lane_context,
                    reason="old_brief_sequential",
                    candidate_index=index,
                ),
            )
            for index, snippet in enumerate(snippets)
        ]

    if not snippets:
        return []

    contract_mode = _v2_judgment_contract_mode("facial")
    contract_version = (
        FACIAL_CONTRACT_VERSION
        if contract_mode == "tool"
        else _FACIAL_LEGACY_CONTRACT_VERSION
    )
    allow_borderline = _facial_ternary_selected(b._new_brief)

    # Build batch user message.  Tool mode carries request-local opaque IDs;
    # legacy mode remains byte-identical to the positional wire format.
    snippet_texts = [_snippet_to_text(s) for s in snippets]
    candidate_ids = (
        (
            tuple(opaque_candidate_ids)
            if opaque_candidate_ids is not None
            else generate_opaque_candidate_ids(len(snippets))
        )
        if contract_mode == "tool"
        else ()
    )
    if contract_mode == "tool" and (
        len(candidate_ids) != len(snippets)
        or len(set(candidate_ids)) != len(candidate_ids)
        or any(not re.fullmatch(r"cand_[0-9a-f]{24}", value) for value in candidate_ids)
    ):
        raise ValueError("opaque_candidate_ids must be unique cand_<24 hex> IDs")
    if contract_mode == "tool":
        user_msg = render_facial_tool_user_message(
            snippet_texts,
            candidate_ids,
            prompt_prefix=prompt_prefix,
        )
        system = assemble_facial_tool_system(b._new_brief, batch=True)
        render_route = "linkedin.facial.batch_v2_tool_v1"
    else:
        numbered = "\n\n".join(
            f"[{i+1}] {text}" for i, text in enumerate(snippet_texts)
        )
        user_msg = f"{prompt_prefix}{numbered}" if prompt_prefix else numbered
        system = assemble_facial_batch_system(b._new_brief)
        render_route = "linkedin.facial.batch_v2_structural"

    usage_context = _judgment_usage_context(
        lane_context,
        stage="facial",
        contract_mode=contract_mode,
        contract_version=contract_version,
        batch_size=len(snippets),
    )
    policy = _fireworks_judgment_policy(
        stage="facial",
        system_prompt=system,
        contract_version=contract_version,
        usage_context=usage_context,
    )

    def capture_for(decision: OpusDecision) -> OpusDecision:
        return _attach_prompt_capture(
            decision,
            _build_prompt_capture(
                stage="facial",
                source="linkedin",
                render_route=render_route,
                llm_caller_name="facial_llm",
                expect_json=False,
                system_prompt=system,
                candidate_text=user_msg,
                usage_context=usage_context,
            ),
        )

    try:
        raw = facial_llm(
            system,
            user_msg,
            expect_json=False,
            # Live-measured GLM reasoning envelope; mirrors the shadow call below.
            max_tokens=16384,
            usage_context=usage_context,
            policy=policy,
            tool_contract=(
                facial_tool_contract(allow_borderline=allow_borderline)
                if contract_mode == "tool"
                else None
            ),
        )
    except Exception as e:
        _reraise_if_budget_exhausted(e)
        if policy is not None:
            # The explicit Fireworks runner has already applied its bounded,
            # provider-aware retry policy.  A terminal/exhausted batch error
            # must abort the page; fanning it out into N sequential calls
            # defeats the attempt/deadline cap and can amplify an outage or
            # billing failure into a costly retry storm.  Legacy behavior is
            # preserved only when the explicit policy is disabled.
            logger.warning(
                "Batch facial policy call failed without sequential fanout: "
                "type=%s status=%s",
                type(e).__name__,
                getattr(e, "status_code", None),
            )
            raise
        if contract_mode == "tool":
            logger.warning(
                "Batch facial tool judge failed without sequential fanout: "
                "type=%s status=%s",
                type(e).__name__,
                getattr(e, "status_code", None),
            )
            raise
        logger.warning("Batch facial judge failed, falling back to sequential: %s", e)
        return [
            facial_judge(
                snippet,
                b,
                prompt_prefix=prompt_prefix,
                lane_context=_facial_fallback_context(
                    lane_context,
                    reason="batch_transport_failure",
                    candidate_index=index,
                ),
            )
            for index, snippet in enumerate(snippets)
        ]

    if contract_mode == "tool":
        try:
            if not isinstance(raw, dict):
                raise JudgmentToolContractError(
                    "tool_arguments_not_object", type(raw).__name__
                )
            results = list(
                validate_facial_tool_arguments(
                    raw,
                    expected_ids=candidate_ids,
                    allow_borderline=allow_borderline,
                )
            )
        except JudgmentToolContractError as e:
            logger.warning("Batch facial tool-contract failure: %s", e)
            _record_batch_parse_failure(
                candidate_count=len(snippets),
                valid_verdicts=0,
                raw=raw,
                reason=e.reason,
                contract_mode=contract_mode,
            )
            return [
                capture_for(
                    parse_failure_decision(
                        stage="facial",
                        candidate_name=snippet.name,
                        profile_url=snippet.profile_url,
                        reason=e.reason,
                        detail=e.detail,
                    )
                )
                for snippet in snippets
            ]
    else:
        results = parse_facial_batch_response(raw, len(snippets))
    prompt_capture = _build_prompt_capture(
        stage="facial",
        source="linkedin",
        render_route=render_route,
        llm_caller_name="facial_llm",
        expect_json=False,
        system_prompt=system,
        candidate_text=user_msg,
        usage_context=usage_context,
    )

    # Fail-loud against verdict mis-attribution: positional attribution is only
    # trustworthy when the model emitted exactly one valid verdict per
    # candidate (distinct valid indices == count). Fewer valid verdicts than
    # snippets is the renumbering/drop signal — the model may have dropped a
    # candidate and renumbered survivors 1..K, in which case [1] no longer
    # describes snippet[0]. We cannot tell which survivor maps to which
    # snippet, so the whole batch is untrustworthy and every snippet is
    # re-judged sequentially (each sequential call re-attaches to the right
    # person). This deliberately supersedes prior keep-prefix/retry-tail
    # behavior, which silently mis-attributed renumbered survivors.
    valid_verdicts = _count_valid_verdicts(results)
    if (
        contract_mode == "legacy"
        and not _batch_results_trustworthy(results, len(snippets))
    ):
        logger.warning(
            "Batch facial judge produced %s valid verdict(s) for %s snippet(s) "
            "(renumbering/drop suspected); re-judging the batch sequentially",
            valid_verdicts,
            len(snippets),
        )
        _record_batch_parse_failure(
            candidate_count=len(snippets),
            valid_verdicts=valid_verdicts,
            raw=raw,
            reason="untrustworthy_positional_batch",
            contract_mode=contract_mode,
        )
        return [
            facial_judge(
                snippet,
                b,
                prompt_prefix=prompt_prefix,
                lane_context=_facial_fallback_context(
                    lane_context,
                    reason="untrustworthy_positional_batch",
                    candidate_index=index,
                ),
            )
            for index, snippet in enumerate(snippets)
        ]

    # GLM-5.2 shadow judge: ONE shadow call for the whole trustworthy batch,
    # mirroring the real batch call shape rather than fanning out per
    # candidate. `results` is 1:1 with `snippets` here (the trustworthiness
    # check above already guaranteed that). Fires after the real per-
    # candidate verdicts already exist; return value is discarded. Note:
    # batch and single shadow events are mutually exclusive per batch by
    # construction — this hook only fires on the trustworthy branch, where
    # trustworthiness means zero failure verdicts, so failed_indexes is
    # empty and no sequential retry (with its own single-comparison event)
    # can follow. An untrustworthy batch takes the sequential path and
    # emits single events only.
    if contract_mode == "legacy":
        _run_facial_shadow_batch(
            system_prompt=system,
            user_prompt=user_msg,
            # 16384, not the primary's 4096: GLM-5.2 spends heavy reasoning
            # tokens before the DECISION lines (live-measured: a batch pinned
            # 4096 exactly and truncated, failing all candidates at once —
            # SPL test run, 2026-07-03). The shadow's cap must not be the
            # binding constraint or parse-failure conflates "GLM can't hold
            # the format" with "we cut it off mid-sentence".
            max_tokens=16384,
            candidate_count=len(snippets),
            primary_decisions=[result.decision for result in results],
        )

    decisions: list[OpusDecision | None] = []
    failed_indexes: list[int] = []
    for idx, (snippet, result) in enumerate(zip(snippets, results)):
        if is_failure_decision(result.decision):
            failed_indexes.append(idx)
            decisions.append(None)
            continue

        decisions.append(
            _attach_prompt_capture(
                OpusDecision(
                    stage="facial",
                    decision=result.decision,
                    path="none",
                    # P5.4: reached only for non-failure verdicts (the
                    # is_failure_decision branch above already continued);
                    # the V2 facial contract has no confidence to report.
                    confidence=None,
                    rationale=result.reason,
                    candidate_name=snippet.name,
                    profile_url=snippet.profile_url,
                ),
                prompt_capture,
            )
        )

    if failed_indexes and contract_mode == "legacy":
        logger.warning(
            "Batch facial judge had %s parse failure(s); retrying sequentially for those entries",
            len(failed_indexes),
        )
        for idx in failed_indexes:
            decisions[idx] = facial_judge(
                snippets[idx],
                b,
                prompt_prefix=prompt_prefix,
                lane_context=_facial_fallback_context(
                    lane_context,
                    reason="batch_slot_parse_failure",
                    candidate_index=idx,
                ),
            )

    return [decision for decision in decisions if decision is not None]


@observe(name="judge.github_facial_batch")
def github_facial_judge_batch(
    portfolio_texts: list[tuple[str, str, str]],
    brief: Brief | None = None,
) -> list[OpusDecision]:
    """Batch GitHub facial triage — one LLM call for multiple candidates.

    Args:
        portfolio_texts: list of (username, profile_url, portfolio_text) tuples
        brief: optional Brief override

    V2 briefs only. Falls back to sequential on batch failure.
    """
    from github.judgment_templates import (
        assemble_github_facial_batch_system,
    )

    b = brief or _brief
    if not b:
        raise RuntimeError("Judger not initialized.")

    if not b.has_v2_schema:
        raise RuntimeError("GitHub batch facial requires a V2 brief.")

    if not portfolio_texts:
        return []

    system = assemble_github_facial_batch_system(b._new_brief)

    # Build batch user message (reuse LinkedIn batch format: [N] content).
    # Defang candidate-controlled portfolio_text symmetrically with the LinkedIn
    # path so an embedded "[N] FACIAL_*" line cannot forge a neighbor's verdict.
    numbered = "\n\n".join(
        f"[{i+1}] {defang_wire_format(text)}" for i, (_, _, text) in enumerate(portfolio_texts)
    )
    usage_context: dict[str, object] = {}
    usage_context.setdefault("stage", "facial")

    try:
        raw = facial_llm(
            system,
            numbered,
            expect_json=False,
            max_tokens=4096,
            usage_context=usage_context,
        )
    except Exception as e:
        logger.warning("GitHub batch facial failed, falling back to sequential: %s", e)
        decisions = []
        for candidate_name, profile_url, text in portfolio_texts:
            decision = github_facial_judge(text, b)
            decision.candidate_name = candidate_name
            decision.profile_url = profile_url
            decisions.append(decision)
        return decisions

    results = parse_facial_batch_response(raw, len(portfolio_texts))
    prompt_capture = _build_prompt_capture(
        stage="facial",
        source="github",
        render_route="github.facial.batch_v2_structural",
        llm_caller_name="facial_llm",
        expect_json=False,
        system_prompt=system,
        candidate_text=numbered,
        usage_context=usage_context,
    )

    # Same fail-loud mis-attribution guard as the LinkedIn batch path (this path
    # reuses the identical [N] wire format): if the model did not emit exactly
    # one valid verdict per candidate, positional attribution is untrustworthy
    # (renumbering/drop), so re-judge the whole batch sequentially.
    def _sequential_all() -> list[OpusDecision]:
        out = []
        for candidate_name, profile_url, text in portfolio_texts:
            decision = github_facial_judge(text, b)
            decision.candidate_name = candidate_name
            decision.profile_url = profile_url
            out.append(decision)
        return out

    if not _batch_results_trustworthy(results, len(portfolio_texts)):
        logger.warning(
            "GitHub batch facial produced %s valid verdict(s) for %s candidate(s) "
            "(renumbering/drop suspected); re-judging the batch sequentially",
            _count_valid_verdicts(results),
            len(portfolio_texts),
        )
        return _sequential_all()

    decisions: list[OpusDecision | None] = []
    failed_indexes: list[int] = []
    for idx, ((candidate_name, profile_url, _), result) in enumerate(zip(portfolio_texts, results)):
        if is_failure_decision(result.decision):
            failed_indexes.append(idx)
            decisions.append(None)
            continue

        decisions.append(
            _attach_prompt_capture(
                OpusDecision(
                    stage="facial",
                    decision=result.decision,
                    path="none",
                    # P5.4: reached only for non-failure verdicts (the
                    # is_failure_decision branch above already continued);
                    # the V2 facial contract has no confidence to report.
                    confidence=None,
                    rationale=result.reason,
                    candidate_name=candidate_name,
                    profile_url=profile_url,
                ),
                prompt_capture,
            )
        )

    if failed_indexes:
        logger.warning(
            "GitHub batch facial had %s parse failure(s); retrying sequentially for those entries",
            len(failed_indexes),
        )
        for idx in failed_indexes:
            candidate_name, profile_url, text = portfolio_texts[idx]
            decision = github_facial_judge(text, b)
            decision.candidate_name = candidate_name
            decision.profile_url = profile_url
            decisions[idx] = decision

    return [decision for decision in decisions if decision is not None]


# ---------------------------------------------------------------------------
# Researcher evaluators — Slice 5
# ---------------------------------------------------------------------------
#
# Per Researcher Module Spec Opinion 6, the full evaluator MUST populate
# `rationale` + `confidence` exactly so the source-agnostic read-model
# `extract_save_reason_and_confidence` finds them at
# `terminal_payload["full_decision"]`. Both judges produce OpusDecision
# directly; the orchestrator wraps with envelope and writes through the
# shared SharedExecutionRuntime path.
#
# Pre-LLM deterministic gates apply at the facial layer ONLY: the
# resolver in `researcher.discipline_defaults` returns the layered
# floor; if the candidate's h_index or papers_in_window is below the
# floor, fast-exit FACIAL_NO with a recruiter-readable rationale (no
# engineer-vocab leak). The full evaluator never sees a fast-exit
# candidate; the orchestrator only escalates FACIAL_YES /
# FACIAL_BORDERLINE to full eval.


@observe(name="judge.researcher_facial_batch")
def researcher_facial_judge_batch(
    snippets: list,
    brief: Brief | None = None,
    *,
    source_config: dict | None = None,
    llm_caller=None,
) -> list[OpusDecision]:
    """Researcher batch facial triage.

    Pre-LLM deterministic gates: resolve the floor via
    `researcher.discipline_defaults.resolve_floors` and fast-exit any
    snippet whose h_index or papers_in_window is below the floor with
    a recruiter-readable FACIAL_NO rationale.

    Surviving snippets get a single batched LLM call. The
    `llm_caller` parameter is injectable for tests; defaults to the
    real `facial_llm` from `shared.llm_clients`.
    """

    from researcher.discipline_defaults import resolve_floors
    from researcher.judgment_templates import (
        assemble_facial_system,
        fast_exit_rationale_for_h_index,
        fast_exit_rationale_for_papers,
        render_facial_user_prompt,
    )

    b = brief or _brief
    if b is None:
        raise RuntimeError(
            "Researcher judger requires a Brief — pass `brief=` or call init_judger first."
        )

    floors = resolve_floors(source_config or {})
    discipline = ""
    if isinstance(source_config, dict):
        discipline = str(source_config.get("discipline") or "").strip().lower()

    decisions: list[OpusDecision] = []
    survivors: list[tuple[int, Any]] = []  # (output_index, snippet)
    output_template: list[OpusDecision | None] = [None] * len(snippets)

    for idx, snippet in enumerate(snippets):
        # Fast-exit gates per Spec Slice 5.
        if snippet.papers_in_window < floors["papers_in_window_floor"]:
            output_template[idx] = OpusDecision(
                stage="facial",
                decision="FACIAL_NO",
                path="fast_exit:papers_in_window",
                confidence=1.0,
                rationale=fast_exit_rationale_for_papers(
                    papers_in_window=snippet.papers_in_window,
                    papers_in_window_floor=floors["papers_in_window_floor"],
                    papers_in_window_months=floors["papers_in_window_months"],
                    discipline=discipline,
                ),
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            continue
        if snippet.h_index < floors["h_index_floor"]:
            output_template[idx] = OpusDecision(
                stage="facial",
                decision="FACIAL_NO",
                path="fast_exit:h_index",
                confidence=1.0,
                rationale=fast_exit_rationale_for_h_index(
                    h_index=snippet.h_index,
                    h_index_floor=floors["h_index_floor"],
                    discipline=discipline,
                ),
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            )
            continue
        survivors.append((idx, snippet))

    if survivors:
        if llm_caller is None:
            def _default_caller(
                system: str,
                user: str,
                *,
                usage_context: dict[str, object] | None = None,
            ) -> str:
                return facial_llm(
                    system,
                    user,
                    expect_json=True,
                    max_tokens=4096,
                    usage_context=usage_context,
                )
            llm_caller = _default_caller
            llm_caller_name = "facial_llm"
        else:
            llm_caller_name = getattr(llm_caller, "__name__", "custom_llm_caller")

        system = assemble_facial_system(b)
        for idx, snippet in survivors:
            user_prompt = render_facial_user_prompt(snippet)
            usage_context: dict[str, object] = {}
            usage_context.setdefault("stage", "facial")
            try:
                raw = _invoke_llm_caller(
                    llm_caller,
                    system_prompt=system,
                    user_prompt=user_prompt,
                    usage_context=usage_context,
                )
            except Exception as exc:  # noqa: BLE001 - LLM client errors surface as failure decisions
                _reraise_if_budget_exhausted(exc)
                logger.warning("Researcher facial judge failed for %s: %s", snippet.name, exc)
                output_template[idx] = _attach_prompt_capture(
                    judgment_failure_decision(
                        stage="facial",
                        candidate_name=snippet.name,
                        profile_url=snippet.profile_url,
                        error=exc,
                        source="judgment",
                    ),
                    _build_prompt_capture(
                        stage="facial",
                        source="researcher",
                        render_route="researcher.facial.json",
                        llm_caller_name=llm_caller_name,
                        expect_json=True,
                        system_prompt=system,
                        candidate_text=user_prompt,
                        usage_context=usage_context,
                    ),
                )
                continue
            output_template[idx] = _attach_prompt_capture(
                _parse_researcher_decision(
                    raw,
                    stage="facial",
                    snippet=snippet,
                ),
                _build_prompt_capture(
                    stage="facial",
                    source="researcher",
                    render_route="researcher.facial.json",
                    llm_caller_name=llm_caller_name,
                    expect_json=True,
                    system_prompt=system,
                    candidate_text=user_prompt,
                    usage_context=usage_context,
                ),
            )

    return [d for d in output_template if d is not None]


@observe(name="judge.researcher_full")
def researcher_full_judge(
    candidate,
    brief: Brief | None = None,
    *,
    llm_caller=None,
) -> OpusDecision:
    """Researcher full evaluation — single LLM call per candidate.

    Returns an OpusDecision whose `rationale` + `confidence` MUST
    populate exactly per Spec Opinion 6 (the wire contract for every
    module). The orchestrator writes the decision dict at
    `terminal_payload["full_decision"]` via SharedExecutionRuntime.
    """

    from researcher.judgment_templates import (
        assemble_full_evaluation_system,
        render_full_user_prompt,
    )

    b = brief or _brief
    if b is None:
        raise RuntimeError(
            "Researcher judger requires a Brief — pass `brief=` or call init_judger first."
        )

    if llm_caller is None:
        # A.2 cache-gap remediation: researcher full-judge sends a 4-8KB
        # system prompt per candidate; caching it cuts repeat-call cost
        # by ~90% within the 5-minute TTL. Same signature as opus_llm.
        def _default_caller(
            system: str,
            user: str,
            *,
            usage_context: dict[str, object] | None = None,
        ) -> str:
            return opus_llm_cached(
                system,
                user,
                expect_json=True,
                max_tokens=4096,
                usage_context=usage_context,
            )
        llm_caller = _default_caller
        llm_caller_name = "opus_llm_cached"
    else:
        llm_caller_name = getattr(llm_caller, "__name__", "custom_llm_caller")

    system = assemble_full_evaluation_system(b)
    user_prompt = render_full_user_prompt(candidate)
    usage_context: dict[str, object] = {}
    try:
        raw = _invoke_llm_caller(
            llm_caller,
            system_prompt=system,
            user_prompt=user_prompt,
            usage_context=usage_context,
        )
    except Exception as exc:  # noqa: BLE001
        _reraise_if_budget_exhausted(exc)
        logger.warning("Researcher full judge failed for %s: %s", candidate.name, exc)
        return _attach_prompt_capture(
            judgment_failure_decision(
                stage="full",
                candidate_name=candidate.name,
                profile_url=candidate.profile_url,
                error=exc,
                source="judgment",
            ),
            _build_prompt_capture(
                stage="full",
                source="researcher",
                render_route="researcher.full.json",
                llm_caller_name=llm_caller_name,
                expect_json=True,
                system_prompt=system,
                candidate_text=user_prompt,
                usage_context=usage_context,
            ),
        )

    return _attach_prompt_capture(
        _parse_researcher_decision(
            raw,
            stage="full",
            snippet=candidate,
        ),
        _build_prompt_capture(
            stage="full",
            source="researcher",
            render_route="researcher.full.json",
            llm_caller_name=llm_caller_name,
            expect_json=True,
            system_prompt=system,
            candidate_text=user_prompt,
            usage_context=usage_context,
        ),
    )


def _parse_researcher_decision(raw, *, stage: str, snippet) -> OpusDecision:
    """Parse the LLM's JSON output into an OpusDecision.

    Handles both the dict-already case (when llm_caller returns dict)
    and the json-string case. Defensive against missing keys.
    """

    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return judgment_failure_decision(
                stage=stage,
                candidate_name=getattr(snippet, "name", ""),
                profile_url=getattr(snippet, "profile_url", ""),
                error=ValueError(f"non-JSON response: {raw!r}"),
                source="parse",
            )
    else:
        payload = {}

    decision = str(payload.get("decision") or "").strip().upper()
    if not decision:
        return judgment_failure_decision(
            stage=stage,
            candidate_name=getattr(snippet, "name", ""),
            profile_url=getattr(snippet, "profile_url", ""),
            error=ValueError("missing `decision` in LLM output"),
            source="parse",
        )

    rationale = str(payload.get("rationale") or "").strip() or "[no rationale]"
    confidence = _safe_confidence(payload.get("confidence"), default=0.5)
    path = str(payload.get("path") or "").strip() or "none"

    return OpusDecision(
        stage=stage,
        decision=decision,
        path=path,
        confidence=confidence,
        rationale=rationale,
        candidate_name=getattr(snippet, "name", ""),
        profile_url=getattr(snippet, "profile_url", ""),
    )
