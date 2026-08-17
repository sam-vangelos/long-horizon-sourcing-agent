#!/usr/bin/env python3
"""Offline facial-gate variant comparison harness — EXPERIMENT ONLY.

This is an experiment harness. It is NOT production tooling. It does NOT
change facial-stage behavior in any run. It does NOT mutate runtime state,
projections, snapshots, or briefs. It reads stored snippet evidence and
runs alternative facial-gate prompts against them so a human can decide
whether the production gate should move tighter, looser, or ternary.

The deliverable is a tool that ANSWERS the tighter/looser/ternary
question; it does not MOVE the gate. To actually change the production
prompt, edit ``shared/judgment/templates.py`` separately under the
normal review process.

Hard guarantees:

- No edits to ``shared/judgment/templates.py``, ``shared/judger.py``,
  the production prompt, briefs, or any production code path.
- No live LLM or network calls under test (callers inject ``facial_call``).
- Imports from ``linkedin/`` are limited to ``assemble_facial_system``
  for the variant-builder; no orchestrator / browser / acquisition imports.
- Reads ``snippets.jsonl``, optionally ``profile_summaries.jsonl`` and
  ``final_judgments.jsonl``. NEVER opens a profile or scrapes LinkedIn.
- CLI requires ``--experiment`` to run; without it the harness exits
  with code 1 and a message.

Variants:

- ``baseline`` — the current production facial prompt (calls
  ``assemble_facial_system`` exactly as-is). Comparison reference.
- ``looser`` — string-mutates the baseline to remove the
  "Ambiguity favors NO" sentence, the "Do NOT open a profile just to
  ..." sentence, and the "DIRECTLY connects" YES bullet. Keeps
  ``fast_exit_block`` and ``non_fit_block``. If any of the targeted
  substrings is missing (template churn), raises ``RuntimeError`` so a
  silent no-op cannot ship.
- ``ternary`` — appends an experimental block that allows
  ``FACIAL_BORDERLINE`` as a third decision. The production code never
  sees this output; the harness records borderline as a distinct
  outcome and (under the default policy) treats it as opening the
  candidate for full eval.

Token proxy: ``len(prompt_string) // 4``. This is a CHARACTER-based
proxy, not a real token count. Use it for relative comparison across
variants on the same input set, not for cost projection in dollars.

CLI shape::

    python tools/experiments/facial_gate_experiment.py
        --experiment
        --snippets PATH                           # or '-' for stdin
        --brief PATH
        [--variants baseline looser ternary]
        [--profile-summaries PATH]
        [--final-judgments PATH]
        [--limit N]                               # default 50; 0 = no limit
        [--json-out PATH]
        [--max-candidates N]                      # default 50; deterministic front-truncate
        [--ternary-policy {open_borderline,skip_borderline}]   # default open_borderline
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


# Make ``python tools/experiments/facial_gate_experiment.py ...`` runnable
# without ``PYTHONPATH=.`` (mirrors ``tools/aggregate_shadow_judgments.py``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Constants — exact substrings stripped from baseline to produce ``looser``.
# These are pinned to the production template at
# ``shared/judgment/templates.py:FACIAL_TRIAGE_TEMPLATE``. If the template
# changes, these strings will fail to match and the harness will raise so a
# future template churn cannot silently produce a no-op ``looser`` variant.
# ---------------------------------------------------------------------------

LOOSER_STRIP_AMBIGUITY = (
    "Ambiguity favors NO. A FACIAL_YES requires at least one STRONG "
    "positive signal — a title, company, or trajectory element that "
    "directly connects to a required capability area. Generic seniority "
    "+ generic capability-area keywords is NOT sufficient for YES."
)

LOOSER_STRIP_DO_NOT_OPEN = (
    "Do NOT open a profile just to \"verify\" or \"assess depth\" — if "
    "the snippet does not contain a clear positive signal, the answer "
    "is FACIAL_NO. The cost of opening a non-fit profile (60+ seconds "
    "of session budget, detection risk, wasted Opus tokens) exceeds "
    "the cost of missing an ambiguous candidate who can be found "
    "through other search strings."
)

LOOSER_STRIP_DIRECTLY_BULLET = (
    "- FACIAL_YES: At least one position shows a title, employer, or "
    "transition that DIRECTLY connects to a capability area. The "
    "connection must be specific, not generic."
)

LOOSER_REPLACEMENT_BULLET = (
    "- FACIAL_YES: The snippet shows a plausible connection to a "
    "capability area worth a full read."
)


TERNARY_APPENDIX = """

=== EXPERIMENTAL TERNARY OUTPUT ===
You may now choose THREE outcomes instead of two:
- FACIAL_YES: confident open.
- FACIAL_BORDERLINE: snippet is genuinely ambiguous; open with explicit low-expectation tagging.
- FACIAL_NO: clearly outside scope.
When unsure between YES and NO, prefer FACIAL_BORDERLINE. The downstream
system can decide whether to spend the next-stage budget.
Output exactly:
DECISION: FACIAL_YES | FACIAL_BORDERLINE | FACIAL_NO
REASON: one sentence
"""


# Save decisions in the recorded full-eval projection that count as a
# realized save for recovery analysis. Mirrors the orchestrator's own
# treatment of the save set (and ``tools/aggregate_shadow_judgments.py``).
_SAVE_DECISIONS = frozenset({"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"})

VALID_VARIANTS = ("baseline", "looser", "ternary")
TERNARY_POLICIES = ("open_borderline", "skip_borderline")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DecisionRow:
    """One per-snippet outcome under a single variant."""

    profile_url: str
    candidate_name: str
    decision: str  # FACIAL_YES | FACIAL_NO | FACIAL_BORDERLINE | PARSE_FAILURE
    reason: str
    latency_seconds: float
    input_token_proxy: int
    output_token_proxy: int


@dataclass
class VariantResult:
    """Per-variant counters plus the full per-decision row stream."""

    variant: str
    ternary_policy: str
    total_snippets: int = 0
    facial_yes: int = 0
    facial_no: int = 0
    facial_borderline: int = 0
    parse_failures: int = 0
    reach_full_eval: int = 0
    latency_total_seconds: float = 0.0
    latency_p50_seconds: float = 0.0
    latency_p95_seconds: float = 0.0
    input_token_proxy_total: int = 0
    output_token_proxy_total: int = 0
    cost_per_reached_full_eval_proxy: float = 0.0
    rows: list[DecisionRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "ternary_policy": self.ternary_policy,
            "total_snippets": self.total_snippets,
            "facial_yes": self.facial_yes,
            "facial_no": self.facial_no,
            "facial_borderline": self.facial_borderline,
            "parse_failures": self.parse_failures,
            "reach_full_eval": self.reach_full_eval,
            "latency_total_seconds": round(self.latency_total_seconds, 4),
            "latency_p50_seconds": round(self.latency_p50_seconds, 4),
            "latency_p95_seconds": round(self.latency_p95_seconds, 4),
            "input_token_proxy_total": self.input_token_proxy_total,
            "output_token_proxy_total": self.output_token_proxy_total,
            "cost_per_reached_full_eval_proxy": round(
                self.cost_per_reached_full_eval_proxy, 3
            ),
        }


# ---------------------------------------------------------------------------
# Snippet / projection loading
# ---------------------------------------------------------------------------


def load_snippets(path_or_stdin: str, *, max_candidates: int):
    """Load CandidateSnippets from a JSONL file (or '-' for stdin).

    ``max_candidates`` is a deterministic front-truncate after parsing,
    not a sample. Default 50 (caller's responsibility).
    """
    from shared.schemas import CandidateSnippet

    if path_or_stdin == "-":
        text = sys.stdin.read()
        lines = text.splitlines()
    else:
        p = Path(path_or_stdin)
        if not p.exists() or not p.is_file():
            return []
        with open(p, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()

    snippets = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        try:
            snippets.append(CandidateSnippet.from_dict(row))
        except (KeyError, TypeError, ValueError):
            continue
        if len(snippets) >= max_candidates:
            break

    return snippets


def load_profile_summaries_index(path: str | None) -> dict:
    """Map ``profile_url`` -> ``CandidateProfileSummary``. Empty dict on no path."""
    if not path:
        return {}
    from shared.schemas import CandidateProfileSummary

    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}

    index: dict = {}
    with open(p, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            try:
                summary = CandidateProfileSummary.from_dict(row)
            except (KeyError, TypeError, ValueError):
                continue
            url = (summary.profile_url or "").strip()
            if url:
                index[url] = summary
    return index


def load_final_judgments_index(path: str | None) -> dict:
    """Map ``profile_url`` -> raw final-judgment row dict. Empty dict on no path.

    Tolerant to both flat ``OpusDecision.to_dict()``-shaped rows and rows
    that nest the decision under ``final_decision`` / ``decision_payload``
    keys (different historic projection writers).
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}

    index: dict = {}
    with open(p, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            url = _extract_profile_url(row)
            if url:
                index[url] = row
    return index


def _extract_profile_url(row: dict) -> str:
    """Pull a profile_url out of a final-judgment row regardless of shape."""
    for key in ("profile_url", "candidate_url"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    nested = row.get("final_decision") or row.get("decision_payload") or {}
    if isinstance(nested, dict):
        v = nested.get("profile_url")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_recorded_decision(row: dict) -> str:
    """Pull a decision string out of a final-judgment row regardless of shape."""
    v = row.get("decision")
    if isinstance(v, str) and v.strip():
        return v.strip()
    nested = row.get("final_decision") or row.get("decision_payload") or {}
    if isinstance(nested, dict):
        v = nested.get("decision")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ---------------------------------------------------------------------------
# Variant prompt builder
# ---------------------------------------------------------------------------


def build_variant_system_prompt(brief, variant: str) -> str:
    """Return the cacheable system prompt for ``variant``.

    - ``baseline``: ``assemble_facial_system(brief._new_brief)`` exactly as-is.
    - ``looser``: baseline minus the "Ambiguity favors NO" sentence, the
      "Do NOT open a profile" sentence, and the "DIRECTLY connects" YES
      bullet (the latter rewritten to a softer plausibility bullet so the
      output schema still has a YES branch). Raises ``RuntimeError`` if any
      target substring is missing.
    - ``ternary``: baseline plus an experimental block that lets the model
      emit ``FACIAL_BORDERLINE``.
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown variant: {variant!r} (allowed: {VALID_VARIANTS})")

    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is None:
        # Allow callers to pass the new brief directly (helpful in tests).
        new_brief = brief

    from linkedin.judgment_templates import assemble_facial_system

    baseline_prompt = assemble_facial_system(new_brief)

    if variant == "baseline":
        return baseline_prompt

    if variant == "looser":
        prompt = baseline_prompt
        for idx, target in enumerate(
            (LOOSER_STRIP_AMBIGUITY, LOOSER_STRIP_DO_NOT_OPEN), start=1
        ):
            if target not in prompt:
                raise RuntimeError(
                    f"looser variant: substring {idx} not found in baseline "
                    f"prompt — production facial template likely changed; "
                    f"update LOOSER_STRIP_* in tools/experiments/"
                    f"facial_gate_experiment.py to match the new template."
                )
            prompt = prompt.replace(target, "")
        if LOOSER_STRIP_DIRECTLY_BULLET not in prompt:
            raise RuntimeError(
                "looser variant: substring 3 (DIRECTLY-connects YES bullet) "
                "not found in baseline prompt — production facial template "
                "likely changed; update LOOSER_STRIP_DIRECTLY_BULLET in "
                "tools/experiments/facial_gate_experiment.py to match."
            )
        prompt = prompt.replace(
            LOOSER_STRIP_DIRECTLY_BULLET, LOOSER_REPLACEMENT_BULLET
        )
        return prompt

    if variant == "ternary":
        return baseline_prompt + TERNARY_APPENDIX

    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Variant response parser
# ---------------------------------------------------------------------------


_DECISION_LINE = re.compile(r"^\s*DECISION\s*:\s*(.+?)\s*$", re.IGNORECASE)


def parse_variant_response(raw: str, variant: str) -> tuple[str, str]:
    """Parse a variant's raw response into a ``(decision, reason)`` tuple.

    Decision values:
      - ``FACIAL_YES``
      - ``FACIAL_NO``
      - ``FACIAL_BORDERLINE`` (only legal under ``variant == "ternary"``;
        for any other variant a borderline response is ``PARSE_FAILURE``)
      - ``PARSE_FAILURE`` for unparseable / illegal output.

    Mirrors but does not import ``linkedin.judgment_templates.parse_facial_response``
    so the harness is decoupled from the production parser's evolution.
    """
    if not isinstance(raw, str):
        return "PARSE_FAILURE", "non-string response"
    text = raw.strip()
    if not text:
        return "PARSE_FAILURE", "empty response"

    # Scan for a DECISION: line first (most specific path).
    decision_token = ""
    for line in text.splitlines():
        m = _DECISION_LINE.match(line)
        if m:
            decision_token = m.group(1).strip().upper()
            break

    # Fall back to scanning the whole body if no DECISION: line.
    if not decision_token:
        upper = text.upper()
        if "FACIAL_BORDERLINE" in upper:
            decision_token = "FACIAL_BORDERLINE"
        elif "FACIAL_YES" in upper:
            decision_token = "FACIAL_YES"
        elif "FACIAL_NO" in upper:
            decision_token = "FACIAL_NO"

    if not decision_token:
        return "PARSE_FAILURE", "no decision token found"

    # Normalize tokens. "FACIAL_BORDERLINE" must win over "FACIAL_NO" because
    # "BORDERLINE" contains no overlapping substring with the other tokens, but
    # we still order checks defensively.
    if "BORDERLINE" in decision_token:
        decided = "FACIAL_BORDERLINE"
    elif "YES" in decision_token:
        decided = "FACIAL_YES"
    elif "NO" in decision_token:
        decided = "FACIAL_NO"
    else:
        return "PARSE_FAILURE", f"unrecognized decision token: {decision_token!r}"

    if decided == "FACIAL_BORDERLINE" and variant != "ternary":
        return (
            "PARSE_FAILURE",
            f"FACIAL_BORDERLINE is illegal under variant={variant!r}",
        )

    reason = _extract_reason(text)
    return decided, reason


def _extract_reason(text: str) -> str:
    """Pull the REASON: field out of a response, or return empty string."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("REASON:"):
            return stripped.split(":", 1)[1].strip() if ":" in stripped else ""
    return ""


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------


def _candidate_user_prompt(snippet) -> str:
    """Build the user-message text for a single snippet.

    The harness owns its own user-prompt shape so it does not depend on
    ``assemble_facial_prompt`` (which is a one-shot template that bakes
    snippet text into the system prompt — wrong shape for cached
    system-prompt usage). The shape is intentionally minimal and stable
    so ``len // 4`` is a meaningful proxy across variants.
    """
    lines = []
    lines.append(f"NAME: {snippet.name}")
    if getattr(snippet, "headline", ""):
        lines.append(f"HEADLINE: {snippet.headline}")
    if getattr(snippet, "current_title", ""):
        lines.append(f"CURRENT_TITLE: {snippet.current_title}")
    if getattr(snippet, "current_company", ""):
        lines.append(f"CURRENT_COMPANY: {snippet.current_company}")
    if getattr(snippet, "location", ""):
        lines.append(f"LOCATION: {snippet.location}")
    if getattr(snippet, "education_snippet", ""):
        lines.append(f"EDUCATION: {snippet.education_snippet}")
    experiences = getattr(snippet, "experience_entries", None) or []
    if experiences:
        lines.append("CAREER_HISTORY:")
        for entry in experiences:
            lines.append(f"  - {entry}")
    return "\n".join(lines)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    # Nearest-rank percentile; deterministic and dependency-free.
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    rank = min(rank, len(ordered))
    return ordered[rank - 1]


def run_variant_against_snippets(
    *,
    snippets: list,
    brief,
    variant: str,
    facial_call: Callable[[str, str], str],
    ternary_policy: str = "open_borderline",
) -> VariantResult:
    """Drive a single variant across a snippet set. Pure modulo ``facial_call``.

    ``facial_call(system_prompt, user_prompt) -> raw_text``. Tests inject a
    deterministic mock; the production default invokes
    ``shared.llm_clients.facial_llm`` (lazy-imported in ``run()``).
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown variant: {variant!r}")
    if ternary_policy not in TERNARY_POLICIES:
        raise ValueError(f"unknown ternary_policy: {ternary_policy!r}")

    system_prompt = build_variant_system_prompt(brief, variant)
    input_proxy_system = len(system_prompt) // 4

    result = VariantResult(variant=variant, ternary_policy=ternary_policy)
    latencies: list[float] = []

    for snippet in snippets:
        user_prompt = _candidate_user_prompt(snippet)
        input_proxy = input_proxy_system + len(user_prompt) // 4
        t0 = time.perf_counter()
        try:
            raw = facial_call(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 — harness must keep aggregating
            raw = f"FACIAL_CALL_EXCEPTION: {type(exc).__name__}: {exc}"
        latency = time.perf_counter() - t0
        decision, reason = parse_variant_response(raw, variant)
        output_proxy = len(raw) // 4

        row = DecisionRow(
            profile_url=getattr(snippet, "profile_url", ""),
            candidate_name=getattr(snippet, "name", ""),
            decision=decision,
            reason=reason,
            latency_seconds=latency,
            input_token_proxy=input_proxy,
            output_token_proxy=output_proxy,
        )
        result.rows.append(row)
        result.total_snippets += 1
        result.input_token_proxy_total += input_proxy
        result.output_token_proxy_total += output_proxy
        latencies.append(latency)

        if decision == "FACIAL_YES":
            result.facial_yes += 1
        elif decision == "FACIAL_NO":
            result.facial_no += 1
        elif decision == "FACIAL_BORDERLINE":
            result.facial_borderline += 1
        else:
            result.parse_failures += 1

    if variant == "ternary":
        if ternary_policy == "open_borderline":
            result.reach_full_eval = result.facial_yes + result.facial_borderline
        else:  # skip_borderline
            result.reach_full_eval = result.facial_yes
    else:
        result.reach_full_eval = result.facial_yes

    result.latency_total_seconds = sum(latencies)
    result.latency_p50_seconds = _percentile(latencies, 50.0)
    result.latency_p95_seconds = _percentile(latencies, 95.0)
    if result.reach_full_eval > 0:
        result.cost_per_reached_full_eval_proxy = (
            result.input_token_proxy_total / result.reach_full_eval
        )
    else:
        result.cost_per_reached_full_eval_proxy = float(
            result.input_token_proxy_total
        )

    return result


# ---------------------------------------------------------------------------
# Cross-variant analysis (agreement, disagreement, recovery, false negatives)
# ---------------------------------------------------------------------------


def analyze_recovery(
    *,
    baseline_result: VariantResult,
    variant_result: VariantResult,
    final_judgments_index: dict,
    profile_summaries_index: dict,
    brief,
) -> dict:
    """Compute pairwise comparison of ``variant_result`` against ``baseline_result``.

    Pure: no I/O, no clocks, no randomness. Same input -> same output.

    The "recovery" numbers are interpreted relative to the recorded run
    represented by ``final_judgments_index`` (typically the run that
    actually executed under the production gate). When that index is empty
    the recovery counters are 0 and ``recovery_evidence_available`` is False.
    """
    base_by_url = {row.profile_url: row for row in baseline_result.rows}
    var_by_url = {row.profile_url: row for row in variant_result.rows}

    # Use the overlap of profile_urls; the harness expects both runs were
    # driven over the same input set, but we defend against the asymmetric
    # case (one variant skipped).
    shared_urls = sorted(set(base_by_url.keys()) & set(var_by_url.keys()))

    agreement = 0
    disagreement_total = 0
    baseline_yes_variant_no = 0
    baseline_no_variant_yes = 0
    baseline_no_variant_borderline = 0
    baseline_yes_variant_borderline = 0
    other_disagreement = 0  # parse failures or borderline vs no, etc.

    for url in shared_urls:
        b = base_by_url[url].decision
        v = var_by_url[url].decision
        if b == v:
            agreement += 1
            continue
        disagreement_total += 1
        if b == "FACIAL_YES" and v == "FACIAL_NO":
            baseline_yes_variant_no += 1
        elif b == "FACIAL_NO" and v == "FACIAL_YES":
            baseline_no_variant_yes += 1
        elif b == "FACIAL_NO" and v == "FACIAL_BORDERLINE":
            baseline_no_variant_borderline += 1
        elif b == "FACIAL_YES" and v == "FACIAL_BORDERLINE":
            baseline_yes_variant_borderline += 1
        else:
            other_disagreement += 1

    # --- Recovery analysis vs the recorded run ------------------------
    recovery_evidence_available = bool(final_judgments_index)
    baseline_saves_recovered = 0
    variant_saves_recovered = 0
    variant_only_recovered_saves = 0

    if recovery_evidence_available:
        for url in shared_urls:
            row = final_judgments_index.get(url)
            if not row:
                continue
            recorded = _extract_recorded_decision(row)
            if recorded not in _SAVE_DECISIONS:
                continue
            base_open = base_by_url[url].decision == "FACIAL_YES"
            # Variant "opens" if it would have reached full eval under the
            # configured ternary policy. For binary variants this == YES.
            var_open = _opens_full_eval(
                variant_result, var_by_url[url].decision
            )
            if base_open:
                baseline_saves_recovered += 1
            if var_open:
                variant_saves_recovered += 1
                if not base_open:
                    variant_only_recovered_saves += 1

    # --- Likely false negatives heuristic -----------------------------
    likely_false_negatives_under_variant = 0
    likely_false_negative_urls: list[str] = []
    if profile_summaries_index:
        # Only meaningful for variants that emitted at least one NO.
        from shared.external_evidence import should_request_external_evidence

        for url, row in var_by_url.items():
            if row.decision != "FACIAL_NO":
                continue
            summary = profile_summaries_index.get(url)
            if summary is None:
                # No profile evidence to gate on; do NOT count.
                continue
            try:
                trigger = should_request_external_evidence(
                    summary=summary, brief=brief
                )
            except Exception:  # noqa: BLE001 — heuristic must never crash analysis
                continue
            if trigger.should_run and trigger.reason in (
                "academic_context",
                "sparse_profile",
            ):
                likely_false_negatives_under_variant += 1
                likely_false_negative_urls.append(url)

    return {
        "baseline_variant": baseline_result.variant,
        "compared_variant": variant_result.variant,
        "ternary_policy": variant_result.ternary_policy,
        "shared_total": len(shared_urls),
        "agreement": agreement,
        "disagreement": disagreement_total,
        "baseline_yes_variant_no": baseline_yes_variant_no,
        "baseline_no_variant_yes": baseline_no_variant_yes,
        "baseline_no_variant_borderline": baseline_no_variant_borderline,
        "baseline_yes_variant_borderline": baseline_yes_variant_borderline,
        "other_disagreement": other_disagreement,
        "recovery_evidence_available": recovery_evidence_available,
        "baseline_saves_recovered": baseline_saves_recovered,
        "variant_saves_recovered": variant_saves_recovered,
        "variant_only_recovered_saves": variant_only_recovered_saves,
        "likely_false_negatives_under_variant": likely_false_negatives_under_variant,
        "likely_false_negative_urls": sorted(likely_false_negative_urls),
    }


def _opens_full_eval(result: VariantResult, decision: str) -> bool:
    if decision == "FACIAL_YES":
        return True
    if decision == "FACIAL_BORDERLINE":
        return (
            result.variant == "ternary"
            and result.ternary_policy == "open_borderline"
        )
    return False


def materially_different_rows(
    *,
    baseline_result: VariantResult,
    variant_result: VariantResult,
) -> list[tuple[str, str, str]]:
    """Return ``(profile_url, baseline_decision, variant_decision)`` for disagreements.

    Sorted by ``profile_url`` for stable output.
    """
    base_by_url = {row.profile_url: row.decision for row in baseline_result.rows}
    var_by_url = {row.profile_url: row.decision for row in variant_result.rows}
    out: list[tuple[str, str, str]] = []
    for url in sorted(set(base_by_url.keys()) & set(var_by_url.keys())):
        b = base_by_url[url]
        v = var_by_url[url]
        if b != v:
            out.append((url, b, v))
    return out


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _truncate(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: max(0, n - 1)] + "…"


def format_summary(
    *,
    results: dict,
    comparisons: dict,
    has_baseline: bool,
) -> str:
    """Render a human-readable per-variant summary block."""
    lines: list[str] = []
    lines.append("=== Facial-Gate Experiment — Per-Variant Counters ===")
    lines.append(
        "Token proxy: input/output character count // 4 (NOT real tokens, NOT dollars)."
    )
    lines.append("")
    for variant, vr in results.items():
        d = vr.to_dict()
        lines.append(f"-- variant: {variant} (ternary_policy={d['ternary_policy']}) --")
        rows = [
            ("total_snippets", d["total_snippets"]),
            ("facial_yes", d["facial_yes"]),
            ("facial_no", d["facial_no"]),
            ("facial_borderline", d["facial_borderline"]),
            ("parse_failures", d["parse_failures"]),
            ("reach_full_eval", d["reach_full_eval"]),
            ("latency_total_seconds", d["latency_total_seconds"]),
            ("latency_p50_seconds", d["latency_p50_seconds"]),
            ("latency_p95_seconds", d["latency_p95_seconds"]),
            ("input_token_proxy_total", d["input_token_proxy_total"]),
            ("output_token_proxy_total", d["output_token_proxy_total"]),
            ("cost_per_reached_full_eval_proxy", d["cost_per_reached_full_eval_proxy"]),
        ]
        width = max(len(label) for label, _ in rows)
        for label, value in rows:
            lines.append(f"  {label.ljust(width)} : {value}")
        lines.append("")

    lines.append("=== Pairwise Comparisons (vs baseline) ===")
    if not has_baseline:
        lines.append(
            "(skipped — baseline variant was not run; pairwise comparisons "
            "require both baseline and at least one other variant)"
        )
        lines.append("")
        return "\n".join(lines)

    if not comparisons:
        lines.append("(no non-baseline variants to compare)")
        lines.append("")
        return "\n".join(lines)

    for compared, cmp in comparisons.items():
        lines.append(f"-- baseline vs {compared} --")
        rows = [
            ("shared_total", cmp["shared_total"]),
            ("agreement", cmp["agreement"]),
            ("disagreement", cmp["disagreement"]),
            ("baseline_yes_variant_no", cmp["baseline_yes_variant_no"]),
            ("baseline_no_variant_yes", cmp["baseline_no_variant_yes"]),
            ("baseline_no_variant_borderline", cmp["baseline_no_variant_borderline"]),
            ("baseline_yes_variant_borderline", cmp["baseline_yes_variant_borderline"]),
            ("other_disagreement", cmp["other_disagreement"]),
            ("recovery_evidence_available", cmp["recovery_evidence_available"]),
            ("baseline_saves_recovered", cmp["baseline_saves_recovered"]),
            ("variant_saves_recovered", cmp["variant_saves_recovered"]),
            ("variant_only_recovered_saves", cmp["variant_only_recovered_saves"]),
            (
                "likely_false_negatives_under_variant",
                cmp["likely_false_negatives_under_variant"],
            ),
        ]
        width = max(len(label) for label, _ in rows)
        for label, value in rows:
            lines.append(f"  {label.ljust(width)} : {value}")
        lines.append("")

    lines.append(
        "Note: likely_false_negatives_under_variant counts only candidates "
        "with a stored profile_summary; missing summaries are excluded "
        "(no evidence to gate on)."
    )
    return "\n".join(lines)


def format_materially_different(
    *,
    comparisons: dict,
    results: dict,
    snippets_by_url: dict,
    final_judgments_index: dict,
    profile_summaries_index: dict,
    brief,
    limit: int,
) -> str:
    """Render the per-pair "Materially different cases" listing."""
    lines: list[str] = ["=== Materially Different Cases ==="]
    if not comparisons:
        lines.append("(no comparisons to render)")
        return "\n".join(lines)

    # Lazy import — only needed if we have any rows to render.
    try:
        from shared.external_evidence import should_request_external_evidence
    except Exception:  # pragma: no cover — defensive
        should_request_external_evidence = None  # type: ignore[assignment]

    baseline_result = results.get("baseline")

    for compared_variant, _cmp in comparisons.items():
        variant_result = results.get(compared_variant)
        if baseline_result is None or variant_result is None:
            continue
        rows = materially_different_rows(
            baseline_result=baseline_result,
            variant_result=variant_result,
        )
        lines.append("")
        lines.append(f"-- baseline vs {compared_variant} --")
        if not rows:
            lines.append("  (no disagreements)")
            continue
        if limit and limit > 0:
            shown = rows[:limit]
            remainder = max(0, len(rows) - limit)
        else:
            shown = rows
            remainder = 0
        for url, base_dec, var_dec in shown:
            recorded_row = final_judgments_index.get(url) or {}
            recorded = _extract_recorded_decision(recorded_row) if recorded_row else "n/a"
            if not recorded:
                recorded = "n/a"
            trigger_reason = "none"
            if should_request_external_evidence is not None and profile_summaries_index:
                summary = profile_summaries_index.get(url)
                if summary is not None:
                    try:
                        td = should_request_external_evidence(
                            summary=summary, brief=brief
                        )
                        if td.should_run and td.reason:
                            trigger_reason = td.reason
                    except Exception:  # noqa: BLE001
                        pass
            snippet_first_line = _truncate(
                _snippet_preview(snippets_by_url.get(url)), 120
            )
            lines.append(
                f"- {url} | baseline={base_dec} variant={var_dec} | "
                f"recorded_decision={recorded} | "
                f"trigger_external={trigger_reason} | "
                f"snippet_first_line={snippet_first_line}"
            )
        if remainder > 0:
            lines.append(f"  ... and {remainder} more")

    return "\n".join(lines)


def _snippet_preview(snippet) -> str:
    if snippet is None:
        return ""
    headline = getattr(snippet, "headline", "") or ""
    if headline:
        return headline
    title = getattr(snippet, "current_title", "") or ""
    company = getattr(snippet, "current_company", "") or ""
    if title and company:
        return f"{title} @ {company}"
    return title or company or getattr(snippet, "name", "") or ""


# ---------------------------------------------------------------------------
# JSON-out writer (atomic; mirrors tools/aggregate_shadow_judgments.py)
# ---------------------------------------------------------------------------


def write_summary_json(summary: dict, path: Path) -> None:
    """Atomically write ``summary`` to ``path`` as pretty JSON.

    Writes to a sibling temp file then ``os.replace`` so a crash mid-write
    cannot leave a half-written summary on disk.
    """
    try:
        path = Path(path)
        if path.exists() and path.is_dir():
            print(
                f"ERROR: --json-out path is a directory: {path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        parent = path.parent if str(path.parent) else Path(".")
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: failed to write --json-out {path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _build_default_facial_call() -> Callable[[str, str], str]:
    """Default ``facial_call`` that invokes the production facial provider.

    Lazy-imported so tests that always inject a mock never touch the
    anthropic SDK or any network code path.
    """

    def _call(system: str, user: str) -> str:
        from shared.llm_clients import facial_llm

        result = facial_llm(system, user, expect_json=False)
        if isinstance(result, dict):
            # facial_llm can return dict when expect_json=True; we asked for
            # text so this is defensive only.
            return json.dumps(result)
        return str(result)

    return _call


def run(
    args: argparse.Namespace,
    *,
    facial_call: Optional[Callable[[str, str], str]] = None,
) -> int:
    """Top-level entry. Returns process exit code."""
    if not getattr(args, "experiment", False):
        print(
            "ERROR: this is an experiment harness; pass --experiment to confirm",
            file=sys.stderr,
        )
        return 1

    variants = list(getattr(args, "variants", None) or VALID_VARIANTS)
    for v in variants:
        if v not in VALID_VARIANTS:
            print(
                f"ERROR: unknown variant: {v!r} (allowed: {VALID_VARIANTS})",
                file=sys.stderr,
            )
            return 1

    ternary_policy = getattr(args, "ternary_policy", "open_borderline")
    if ternary_policy not in TERNARY_POLICIES:
        print(
            f"ERROR: unknown --ternary-policy: {ternary_policy!r}",
            file=sys.stderr,
        )
        return 1

    max_candidates = int(getattr(args, "max_candidates", 50) or 50)
    if max_candidates <= 0:
        print(
            "ERROR: --max-candidates must be > 0 (got "
            f"{max_candidates}); this is a safety cap",
            file=sys.stderr,
        )
        return 1

    snippets = load_snippets(args.snippets, max_candidates=max_candidates)
    if not snippets:
        print(
            "ERROR: no snippets resolved from "
            f"{args.snippets!r}; nothing to compare",
            file=sys.stderr,
        )
        return 1

    # Brief loading is deferred so tests can mock load_brief.
    from shared.brief_loader import load_brief

    brief = load_brief(args.brief)
    if getattr(brief, "_new_brief", None) is None:
        # The variant builder needs the v2 brief surface (capability_areas,
        # facial_calibration, etc.). Old-format briefs are out of scope.
        print(
            "ERROR: the experiment harness requires a V2 brief (with "
            "capability_areas + facial_calibration); the loader returned a "
            "legacy-format brief without _new_brief",
            file=sys.stderr,
        )
        return 1

    profile_summaries_index = load_profile_summaries_index(
        getattr(args, "profile_summaries", None)
    )
    final_judgments_index = load_final_judgments_index(
        getattr(args, "final_judgments", None)
    )

    if facial_call is None:
        facial_call = _build_default_facial_call()

    results: dict = {}
    for variant in variants:
        results[variant] = run_variant_against_snippets(
            snippets=snippets,
            brief=brief,
            variant=variant,
            facial_call=facial_call,
            ternary_policy=ternary_policy,
        )

    has_baseline = "baseline" in results
    comparisons: dict = {}
    if has_baseline:
        for variant, vr in results.items():
            if variant == "baseline":
                continue
            comparisons[variant] = analyze_recovery(
                baseline_result=results["baseline"],
                variant_result=vr,
                final_judgments_index=final_judgments_index,
                profile_summaries_index=profile_summaries_index,
                brief=brief,
            )

    print(
        format_summary(
            results=results,
            comparisons=comparisons,
            has_baseline=has_baseline,
        )
    )
    print()
    snippets_by_url = {s.profile_url: s for s in snippets if s.profile_url}
    print(
        format_materially_different(
            comparisons=comparisons,
            results=results,
            snippets_by_url=snippets_by_url,
            final_judgments_index=final_judgments_index,
            profile_summaries_index=profile_summaries_index,
            brief=brief,
            limit=int(getattr(args, "limit", 50) or 50),
        )
    )

    if getattr(args, "json_out", None):
        summary = {
            "variants": {v: r.to_dict() for v, r in results.items()},
            "comparisons": comparisons,
            "config": {
                "snippets_path": args.snippets,
                "brief_path": args.brief,
                "profile_summaries_path": getattr(args, "profile_summaries", None),
                "final_judgments_path": getattr(args, "final_judgments", None),
                "max_candidates": max_candidates,
                "ternary_policy": ternary_policy,
                "variants_run": list(results.keys()),
                "token_proxy_note": "len(prompt_string) // 4; not real tokens",
            },
        }
        write_summary_json(summary, Path(args.json_out))

    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "EXPERIMENT ONLY — offline facial-gate variant comparison. "
            "Reads stored snippets, runs alternative facial prompts (baseline, "
            "looser, ternary), and prints per-variant counters plus pairwise "
            "agreement / recovery / likely-false-negative analysis. Does not "
            "modify production behavior."
        )
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        default=False,
        help="Required confirmation flag; without it the harness exits 1.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VALID_VARIANTS),
        choices=VALID_VARIANTS,
        help="Which variants to run (default: all three).",
    )
    parser.add_argument(
        "--snippets",
        required=True,
        help="Path to a snippets.jsonl file (or '-' for stdin).",
    )
    parser.add_argument(
        "--brief",
        required=True,
        help="Path to a brief JSON file consumable by load_brief.",
    )
    parser.add_argument(
        "--profile-summaries",
        default=None,
        help=(
            "Optional path to a profile_summaries.jsonl file. Required for "
            "the likely-false-negative analysis; missing summaries exclude "
            "candidates from that bucket (no evidence to gate on)."
        ),
    )
    parser.add_argument(
        "--final-judgments",
        default=None,
        help=(
            "Optional path to a final_judgments.jsonl file from a recorded "
            "run; required for save-recovery analysis."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Cap for the materially-different listing per pair. 0 = no limit.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help=(
            "Optional path to write the structured summary as JSON. Default: "
            "nothing is written to disk."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help=(
            "Hard cap on input size to bound spend. Deterministic "
            "front-truncate after parsing. Default 50."
        ),
    )
    parser.add_argument(
        "--ternary-policy",
        choices=TERNARY_POLICIES,
        default="open_borderline",
        help=(
            "Under 'ternary', whether FACIAL_BORDERLINE counts toward "
            "reach_full_eval (open_borderline, default) or not "
            "(skip_borderline)."
        ),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
