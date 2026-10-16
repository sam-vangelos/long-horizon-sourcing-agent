"""Run strategy and preflight comparisons in a non-blocking shadow path.

When ``SHADOW_STRATEGY_ENABLED`` is on, each strategy-formation and preflight
call submits a comparison to ``SHADOW_STRATEGY_MODEL_NAME`` and persists its
raw response plus deterministic metrics under ``<state_dir>/shadow_strategy/``.
Preflight reuses the primary prompts byte-for-byte; formation rebuilds a fresh
user prompt without prior-run data or lane feedback so the comparison is brief-
driven. Shadow artifacts never influence the primary path.

A single background worker serializes shadow calls and the primary never waits.
The worker snapshots the caller's context so usage rows reach the active cost
sink. Every exception, including model refusals, is recorded as ``shadow_error``
or logged without escaping. There is no synchronous fallback: shadow work stays
off the run-critical formation path. The client passes no temperature to Fable,
and its captured thinking summary is persisted with the artifact.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import functools
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from shared.boolean_metrics import (
    _OPERATORS,
    _OR_GROUP,
    _QUOTED,
    _TOKEN,
    _T_RUN,
    _skeletonize,
)
import shared.config as config
from shared.llm_clients import _parse_json_response, opus_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic plan metrics
#
# Computed twice per comparison — once on the primary response at the call
# site (folded into ``primary_meta``) and once on the shadow response in the
# worker — against each side's own ``reference_text`` (system prompt plus user
# prompt), so the two sides of every artifact are directly comparable
# offline. These are the item-18/19 formation metrics: skeleton diversity,
# max-skeleton share, AND-gate depth, NOT usage, vocabulary novelty.
# ---------------------------------------------------------------------------

def plan_metrics(
    response: object,
    reference_text: str = "",
    novelty_reference: str = "system",
) -> dict:
    """Deterministic structural metrics for a strategy-shaped response.

    ``response`` is either an already-parsed plan dict (the primary path —
    ``opus_llm(expect_json=True)`` hands the call site a dict) or a raw
    string (the shadow path); strings go through ``_parse_json_response``,
    the same repair-tolerant parser every primary provider branch uses, so
    a shadow model that fences its JSON parses exactly as leniently as the
    primary would. Booleans are read from ``generated_strings[*].boolean``.

    ``reference_text`` feeds vocabulary novelty: the fraction of distinct
    quoted terms (case-insensitive) that do NOT appear in it. Callers pass
    the prompt text being used as the novelty baseline, making novelty
    "terms the model reached for beyond what the prompt handed it" — the
    anchoring metric from plan item 18. ``None`` when the response has no
    quoted terms. ``novelty_reference`` labels that baseline in artifacts.

    Never raises. Unparseable (or non-dict) responses return
    ``{"parse_failed": True, "novelty_reference": novelty_reference}`` — a
    refusal that arrives as prose lands here rather than as an exception.
    """
    try:
        if isinstance(response, (dict, list)):
            parsed = response
        elif isinstance(response, str):
            parsed = _parse_json_response(response)
        else:
            return {"parse_failed": True, "novelty_reference": novelty_reference}
    except Exception:
        return {"parse_failed": True, "novelty_reference": novelty_reference}
    if not isinstance(parsed, dict):
        return {"parse_failed": True, "novelty_reference": novelty_reference}
    try:
        booleans: list[str] = []
        raw_strings = parsed.get("generated_strings")
        if isinstance(raw_strings, list):
            for entry in raw_strings:
                if isinstance(entry, dict):
                    boolean = entry.get("boolean")
                    if isinstance(boolean, str) and boolean.strip():
                        booleans.append(boolean.strip())
        n_strings = len(booleans)

        skeleton_counts: dict[str, int] = {}
        and_counts: list[int] = []
        not_hits = 0
        quoted_terms: dict[str, str] = {}  # casefold key -> first spelling
        for boolean in booleans:
            skeleton = _skeletonize(boolean)
            skeleton_counts[skeleton] = skeleton_counts.get(skeleton, 0) + 1
            # Operator counts run on the quote-stripped text so an "and"
            # INSIDE a quoted phrase ("search and rescue") never counts.
            stripped = _QUOTED.sub("T", boolean)
            ops = [
                tok.upper()
                for tok in _TOKEN.findall(stripped)
                if tok.upper() in _OPERATORS
            ]
            and_counts.append(sum(1 for op in ops if op == "AND"))
            if "NOT" in ops:
                not_hits += 1
            for term in _QUOTED.findall(boolean):
                term = term.strip()
                if term:
                    quoted_terms.setdefault(term.casefold(), term)

        reference = (reference_text or "").casefold()
        novel_count = sum(1 for key in quoted_terms if key not in reference)
        return {
            "parse_failed": False,
            "novelty_reference": novelty_reference,
            "n_strings": n_strings,
            "distinct_skeletons": len(skeleton_counts),
            "skeleton_counts": dict(
                sorted(skeleton_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "max_skeleton_share": (
                round(max(skeleton_counts.values()) / n_strings, 3)
                if n_strings
                else None
            ),
            "mean_and_count": (
                round(sum(and_counts) / n_strings, 3) if n_strings else None
            ),
            "not_usage_rate": (
                round(not_hits / n_strings, 3) if n_strings else None
            ),
            "n_quoted_terms": len(quoted_terms),
            "vocab_novelty": (
                round(novel_count / len(quoted_terms), 3) if quoted_terms else None
            ),
        }
    except Exception:  # noqa: BLE001 — metric helper must never raise
        return {"parse_failed": True, "novelty_reference": novelty_reference}


# ---------------------------------------------------------------------------
# Fire-and-forget dispatch (single worker, module lifetime)
# ---------------------------------------------------------------------------

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
                thread_name_prefix="shadow-strategy",
            )
        return _shadow_executor


def _shadow_task_done(future: concurrent.futures.Future) -> None:
    with _shadow_pending_lock:
        _shadow_pending.discard(future)
    exc = future.exception()
    if exc is not None:  # pragma: no cover — the worker body is fail-soft
        logger.warning(
            "shadow-strategist task failed (primary unaffected): %s", exc
        )


def dispatch_strategy_shadow(
    *,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    shadow_dir: Path | None,
    primary_meta: dict,
    shadow_prompt_context: str = "byte-identical",
    primary_prompt_included_prior_run_data: bool | None = None,
) -> None:
    """Fire one shadow-strategist comparison; returns immediately.

    No-op unless ``config.SHADOW_STRATEGY_ENABLED`` and ``shadow_dir`` is
    not None. ``system_prompt``/``user_prompt`` are the prompts the SHADOW
    runs on: byte-identical to the primary's by default (preflight), or a
    fresh-context rebuild (formation — see the module docstring). The
    caller declares which contract applies via ``shadow_prompt_context``
    ("byte-identical" / "fresh"), persisted in the artifact so a reader
    never has to guess whether prompt divergence was intentional;
    ``primary_prompt_included_prior_run_data`` records honestly whether
    the PRIMARY's prompt carried past-run data the shadow didn't see
    (None where the question doesn't apply, e.g. preflight).
    ``primary_meta`` is the caller's snapshot of the primary side (its
    model name + ``plan_metrics`` of its response) and is persisted
    verbatim in the artifact.

    Never raises: a defect in dispatch itself is logged and swallowed so
    the strategy/preflight path cannot be dragged down by its own shadow.
    """
    try:
        if not config.SHADOW_STRATEGY_ENABLED or shadow_dir is None:
            return
        task = functools.partial(
            _run_strategy_shadow_sync,
            stage=stage,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            shadow_dir=Path(shadow_dir),
            primary_meta=primary_meta,
            shadow_prompt_context=shadow_prompt_context,
            primary_prompt_included_prior_run_data=(
                primary_prompt_included_prior_run_data
            ),
        )
        # Snapshot the caller's context so the worker's opus_llm call
        # resolves the run's token-cost sink (ContextVar) instead of
        # silently recording nowhere — same idiom as shared/judger.py.
        ctx = contextvars.copy_context()
        future = _get_shadow_executor().submit(ctx.run, task)
        with _shadow_pending_lock:
            _shadow_pending.add(future)
        # Registered AFTER the pending-set add so a task that finishes
        # instantly still gets discarded (the callback fires immediately
        # when the future is already done) instead of leaking a set entry.
        future.add_done_callback(_shadow_task_done)
    except Exception as e:  # noqa: BLE001 — shadow must never hurt the primary
        logger.warning(
            "shadow-strategist dispatch failed (primary unaffected): %s", e
        )


def _run_strategy_shadow_sync(
    *,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    shadow_dir: Path,
    primary_meta: dict,
    shadow_prompt_context: str = "byte-identical",
    primary_prompt_included_prior_run_data: bool | None = None,
) -> None:
    """Worker body: call the shadow model once, persist one artifact.

    Fail-soft by construction. The inner try captures the shadow CALL —
    including the RuntimeError ``opus_llm`` raises on a non-end_turn
    stop_reason (a Fable refusal) — as ``shadow_error`` in the artifact.
    The outer try covers metric computation and the artifact write itself;
    nothing propagates to the executor beyond a log line.
    """
    try:
        start = time.monotonic()
        raw: str | None = None
        error: str | None = None
        # opus_llm fills this BEFORE its non-end_turn raise, so a refusal
        # or truncation still leaves the shadow's reasoning + stop_reason
        # here for the artifact.
        capture: dict = {}
        try:
            raw = opus_llm(
                system_prompt,
                user_prompt,
                expect_json=False,
                max_tokens=max_tokens,
                usage_context={
                    "stage": f"{stage}_shadow",
                    # Mirrors judger's shadow_stage tag discipline so
                    # token-cost rows group cleanly per experiment.
                    "shadow_stage": "strategy_shadow",
                },
                model_name=config.SHADOW_STRATEGY_MODEL_NAME,
                capture=capture,
            )
        except Exception as e:  # noqa: BLE001 — refusals surface here
            error = str(e)[:500]
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        artifact = {
            "stage": stage,
            "shadow_model": config.SHADOW_STRATEGY_MODEL_NAME,
            "latency_ms": latency_ms,
            "shadow_error": error,
            "shadow_stop_reason": capture.get("stop_reason"),
            "raw_response": raw,
            # The shadow model's summarized reasoning (always-thinking
            # models only; None elsewhere) — the "why" behind its plan.
            "thinking_summary": capture.get("thinking_summary"),
            "metrics": (
                plan_metrics(
                    raw,
                    reference_text=system_prompt + "\n" + user_prompt,
                    novelty_reference="system+user",
                )
                if raw is not None
                else None
            ),
            "primary_meta": primary_meta,
            # Which prompt contract this comparison ran under, and whether
            # the primary's own prompt carried past-run data the shadow
            # deliberately didn't see (fresh-context formation shadows).
            "shadow_prompt_context": shadow_prompt_context,
            "primary_prompt_included_prior_run_data": (
                primary_prompt_included_prior_run_data
            ),
            # The exact prompts THIS SHADOW CALL ran on (the shadow's own —
            # equal to the primary's only under the byte-identical
            # contract). Persisted so an artifact is self-contained:
            # re-judging or replaying it offline needs no live run.
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        shadow_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S-%f")
        artifact_path = shadow_dir / f"shadow-{stage}-{timestamp}.json"
        tmp_path = artifact_path.with_name(f"{artifact_path.name}.tmp")
        tmp_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_path, artifact_path)
        _print_strategy_shadow_live(artifact, artifact_path)
    except Exception as e:  # noqa: BLE001 — fail-soft to the very end
        logger.warning(
            "shadow-strategist artifact failed (primary unaffected): %s", e
        )


# Live console contract: one prebuilt line per completed comparison, carrying
# deterministic metrics and model attribution. Full prompts, thinking, and raw
# responses stay in the artifact/feed so the run console remains bounded.


def _short_model_name(model_name: str | None, fallback: str) -> str:
    """Console-friendly model word for attribution-by-name ('fable',
    'opus') — never bare 'primary'/'shadow' jargon on a comparison line."""
    model_id = str(model_name or "").rsplit("/", 1)[-1].lower()
    for word in ("opus", "glm", "fable", "mythos", "sonnet", "haiku", "gpt"):
        if word in model_id:
            return word
    return model_id[:12] or fallback


def _metrics_summary(metrics: dict | None) -> str | None:
    """'19 strings/5 skeletons/novelty 0.61' — None when the response
    wasn't plan-shaped (unparseable, or zero generated strings; a
    preflight brief lands here by design)."""
    if not isinstance(metrics, dict) or metrics.get("parse_failed"):
        return None
    if not metrics.get("n_strings"):
        return None
    text = (
        f"{metrics['n_strings']} strings/"
        f"{metrics.get('distinct_skeletons')} skeletons"
    )
    novelty = metrics.get("vocab_novelty")
    if novelty is not None:
        text += f"/novelty {novelty:.2f}"
    return text


def _metrics_compact(metrics: dict | None) -> str | None:
    """The primary's side of the comparison, compressed: '18/3/0.40'."""
    if not isinstance(metrics, dict) or metrics.get("parse_failed"):
        return None
    if not metrics.get("n_strings"):
        return None
    text = f"{metrics['n_strings']}/{metrics.get('distinct_skeletons')}"
    novelty = metrics.get("vocab_novelty")
    if novelty is not None:
        text += f"/{novelty:.2f}"
    return text


def _print_strategy_shadow_live(artifact: dict, artifact_path: Path) -> None:
    """Print ONE completed strategy-shadow comparison as one prebuilt line."""
    try:
        shadow_word = _short_model_name(artifact.get("shadow_model"), "shadow")
        primary_meta = artifact.get("primary_meta") or {}
        primary_word = _short_model_name(
            primary_meta.get("primary_model"), "primary"
        )
        prefix = f"[shadow] strategist({shadow_word}) {artifact.get('stage')}"
        latency_s = f"{(artifact.get('latency_ms') or 0) / 1000:.0f}s"
        if artifact.get("shadow_error"):
            error = " ".join(str(artifact["shadow_error"]).split())[:120]
            line = f"{prefix} SHADOW ERROR: {error} — artifact: {artifact_path}"
        else:
            shadow_summary = _metrics_summary(artifact.get("metrics"))
            if shadow_summary:
                primary_compact = _metrics_compact(primary_meta.get("metrics"))
                versus = (
                    f"vs {primary_word} {primary_compact}"
                    if primary_compact
                    else f"vs {primary_word} (no plan metrics)"
                )
                line = (
                    f"{prefix} done {latency_s} — {shadow_summary} {versus}"
                    " — detail: feed or artifact"
                )
            else:
                # Not plan-shaped (a preflight brief, a refusal in prose):
                # size + artifact path, never the response itself.
                size_kb = len(artifact.get("raw_response") or "") / 1024
                line = (
                    f"{prefix} done {latency_s} — response {size_kb:.1f}KB "
                    f"(not plan-shaped) — artifact: {artifact_path}"
                )
        print(line, flush=True)
    except Exception as e:  # noqa: BLE001 — rendering must never hurt the shadow path
        logger.warning("shadow-strategist live-console render failed: %s", e)


def drain_strategy_shadows(timeout: float | None = None) -> bool:
    """Block until every queued shadow-strategist task has finished.

    Returns True when the queue fully drained (or was already empty),
    False on timeout with tasks still pending. Never raises on task
    failure — worker bodies are fail-soft and anything that somehow
    escapes one is logged by ``_shadow_task_done``, not re-raised here.
    Intended caller: linkedin/orchestrator.py's report build, beside the
    judge-shadow ``drain_shadow_comparisons`` call.
    """
    with _shadow_pending_lock:
        pending = list(_shadow_pending)
    if not pending:
        return True
    _done, not_done = concurrent.futures.wait(pending, timeout=timeout)
    return not not_done
