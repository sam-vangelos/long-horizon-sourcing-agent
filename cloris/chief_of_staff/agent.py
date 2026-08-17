"""Chief-of-staff cross-source synthesis backend.

After ≥2 sources contribute candidates in a multi-module run, this
agent produces a team-level read for the principal: a paragraph in
Cloris voice, an independent 0.0-1.0 trust weight per contributing
specialist, and a one-sentence priority for what to look at first.

Two backends mirror :mod:`market_intelligence.briefing_polish`:

- :class:`ChiefOfStaffAgent`: Opus LLM call. Falls through to the
  heuristic synthesizer on any of six cascade conditions (see
  :func:`ChiefOfStaffAgent.synthesize` docstring).
- :class:`HeuristicChiefOfStaffSynthesizer`: deterministic builder
  from the per-source signals + the single editorial briefing.
  Always grounded; never produces engineer prose.

Confidence is computed PROGRAMMATICALLY (not LLM self-rating):
- Heuristic: signal-density across the contributing sources.
- LLM: containment check pass = 1.0; fail cascades to heuristic.

Banned tokens and snake_case identifier discipline reuse the
:mod:`market_intelligence.briefing_polish` primitives directly
(``BANNED_BRIEFING_TOKENS``, ``SNAKE_CASE_IDENTIFIER_RE``,
``_has_llm_access``) — the cascade discipline is shared, not
forked.

The new (chief-of-staff-specific) cascade route is
``specialist_weight_invalid``: ``per_specialist_weight`` whose keys
reference sources that did not contribute candidates this run.
This is the synthesis-equivalent of
:func:`market_intelligence.brief_polish._role_title_drift` — a
preservation contract enforcing that the agent never invents
specialists.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cloris.chief_of_staff.decision import DispatchPlan, DispatchStep
from cloris.launchers import LAUNCHERS
from cloris.chief_of_staff.prompts import (
    build_chief_of_staff_system_prompt,
    build_chief_of_staff_user_prompt,
    build_dispatch_system_prompt,
    build_dispatch_user_prompt,
)
from market_intelligence.briefing_polish import (
    BANNED_BRIEFING_TOKENS,
    SNAKE_CASE_IDENTIFIER_RE,
    _has_llm_access,
    _normalize_text,
)
from market_intelligence.schema import MarketIdentity
from shared.llm_clients import opus_llm_cached as opus_llm
from shared.observability import observe
from shared.output_paths import resolve_orchestration_db_path
from shared.runtime_state.orchestration_store import OrchestrationStateStore

if TYPE_CHECKING:
    from shared.brief_loader import Brief


# Re-export the shared discipline primitives so callers (and tests)
# don't have to reach into briefing_polish for them. Same values; one
# source of truth at briefing_polish.
__all__ = [
    "BANNED_BRIEFING_TOKENS",
    "SNAKE_CASE_IDENTIFIER_RE",
    "ChiefOfStaffAgent",
    "ChiefOfStaffSynthesis",
    "DispatchPlan",
    "DispatchStep",
    "HeuristicChiefOfStaffSynthesizer",
]


# Minimum paragraph length below which output is treated as degenerate.
# Mirrors :data:`market_intelligence.briefing_polish.MIN_PARAGRAPH_CHARS`.
# A 30-character paragraph catches one-sentence stubs that pass JSON
# validation but offer no signal ("I don't know yet.").
MIN_PARAGRAPH_CHARS = 30

# Synthesis prompt token budget — modest because the input is already
# a structured per-source dict plus the existing single-briefing
# paragraph. Mirrors the briefing-polish budget posture.
SYNTHESIS_MAX_TOKENS = 2000


_FALLBACK_REASON_RE = re.compile(r"\bfallback reason=([A-Za-z_][A-Za-z0-9_]*)\b")


def _emit_stage(message: str) -> None:
    """Emit a stage log line to stderr + bridge cascade reasons to Langfuse.

    Mirrors the per-call telemetry helper used elsewhere in the repo
    (see :func:`market_intelligence.briefing_polish._emit_stage`).
    The bracketed prefix is the subsystem; the message body is a
    dot-separated namespace + colon-separated event, matching the
    repo convention (``[chief-of-staff] synthesis:start ...``,
    ``[market-intel] reflection.polish:start ...``).

    Phase 1 of Langfuse adoption: when the message carries a
    ``fallback reason=<reason>`` token (the cascade-route convention
    used by both :meth:`ChiefOfStaffAgent.dispatch` and
    :meth:`ChiefOfStaffAgent.synthesize`), the helper ALSO emits the
    reason as a Langfuse span attribute under
    ``cascade.fallback_reason``. Single bridging point so the 13
    fallback call sites in this module don't each have to know about
    the observability layer. No-op when the Langfuse client is null /
    disabled / network-degraded.
    """

    print(f"[chief-of-staff] {message}", file=sys.stderr, flush=True)

    # Cascade-route attribution. The regex is permissive (snake_case
    # token after ``fallback reason=``) so future cascade routes pick
    # this up automatically as long as they follow the existing message
    # convention.
    match = _FALLBACK_REASON_RE.search(message)
    if match is not None:
        try:
            from shared.observability import update_current_observation

            update_current_observation(
                metadata={"cascade.fallback_reason": match.group(1)}
            )
        except Exception:  # noqa: BLE001 — Langfuse path is fail-soft
            pass


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChiefOfStaffSynthesis:
    """Team-level read produced by the chief-of-staff agent.

    ``paragraph`` is 2-4 sentences in Cloris voice. ``per_specialist_weight``
    maps each contributing source key (``"linkedin"``, ``"github"``, ...)
    to ``{"weight": 0.0..1.0, "rationale": "..."}`` — independent
    trust scores, NOT a normalized share. ``priority_for_principal``
    is one sentence naming what the recruiter should look at first
    when they open the workspace. ``confidence`` is the programmatic
    grounding score (heuristic: signal-density; LLM: containment-
    check pass = 1.0). ``source`` is one of ``"llm"``,
    ``"deterministic"``, ``"empty"`` so the operator can tell at a
    glance which path fired.
    """

    paragraph: str
    per_specialist_weight: dict[str, dict] = field(default_factory=dict)
    priority_for_principal: str = ""
    confidence: float = 0.0
    source: str = "empty"

    def to_dict(self) -> dict:
        return {
            "paragraph": self.paragraph,
            "per_specialist_weight": {
                key: {
                    "weight": round(float(value.get("weight", 0.0)), 2),
                    "rationale": str(value.get("rationale", "") or ""),
                }
                for key, value in (self.per_specialist_weight or {}).items()
            },
            "priority_for_principal": self.priority_for_principal,
            "confidence": round(float(self.confidence), 2),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Source-key humanization
# ---------------------------------------------------------------------------


_SOURCE_DISPLAY: dict[str, str] = {
    "linkedin": "LinkedIn",
    "github": "GitHub",
    "researcher": "Researcher",
    "designer": "Designer",
    "exec_search": "Executive Search",
}


def _humanize_source(source: str) -> str:
    """Map a raw source key to a recruiter-readable display name.

    Falls back to title-casing with underscores → spaces so a future
    source key gets a sensible default before it lands in the
    explicit map. The heuristic backend uses this so its paragraph
    never quotes a raw lowercase identifier.
    """

    raw = (source or "").strip().lower()
    if raw in _SOURCE_DISPLAY:
        return _SOURCE_DISPLAY[raw]
    if not raw:
        return ""
    return " ".join(part.capitalize() for part in raw.split("_") if part) or raw


def _humanize_lane(lane: str | None) -> str:
    """Humanize a lane name so it reads as recruiter prose.

    ``forward_deployed_engineering`` → ``Forward Deployed Engineering``.
    ``ml-platform`` → ``ml-platform`` (already humanized — left alone).
    Empty / None → empty string.
    """

    text = (lane or "").strip()
    if not text:
        return ""
    if "_" in text:
        return " ".join(part.capitalize() for part in text.split("_") if part)
    return text


# ---------------------------------------------------------------------------
# Heuristic backend — deterministic team-level read
# ---------------------------------------------------------------------------


class HeuristicChiefOfStaffSynthesizer:
    """Deterministic chief-of-staff synthesis. Always grounded.

    Builds the paragraph from per-source candidate / save counts and
    the strongest source's top lane. Builds ``per_specialist_weight``
    from per-source save density: a contributing source with no
    saves still gets a moderate weight (the negative read is
    informative); a source with strong save density gets a higher
    weight. Builds ``priority_for_principal`` from the source that
    surfaced the densest signal.

    Confidence = signal-density across the inputs: how many of the
    contributing sources surfaced saves, how many surfaced top-lane
    signal, whether the existing single-briefing paragraph is non-
    empty. Single-source briefs are not the heuristic's job — the
    integration layer guards against ``len(sources) < 2`` upstream.
    """

    def synthesize(
        self,
        *,
        market_identity: MarketIdentity,
        per_source_signals: dict[str, dict],
        briefing_paragraph: str,
        deterministic_summary: dict | None = None,
        prior_handoff_payloads: dict[str, dict] | None = None,
    ) -> ChiefOfStaffSynthesis:
        # Audit Move #1: heuristic backend ignores prior_handoff_payloads
        # — its deterministic narrative is built from per_source_signals
        # alone (saves count + top lane). The kwarg is accepted to keep
        # the call signature uniform with the LLM backend so callers
        # (and the LLM-backend's fallback paths) can pass it through
        # without per-backend routing.
        del prior_handoff_payloads
        sources = sorted(per_source_signals.keys())
        if not sources:
            return ChiefOfStaffSynthesis(
                paragraph=(
                    "I don't have enough from this run to draw a team-level "
                    "read yet — let me read the broader market."
                ),
                per_specialist_weight={},
                priority_for_principal=(
                    "There's no specialist read to surface yet."
                ),
                confidence=0.0,
                source="empty",
            )

        normalized: dict[str, dict] = {
            source: _normalize_per_source(per_source_signals.get(source) or {})
            for source in sources
        }

        paragraph = _heuristic_paragraph(
            sources=sources,
            normalized=normalized,
            briefing_paragraph=briefing_paragraph or "",
        )
        weights = {
            source: _heuristic_weight_for_source(normalized[source])
            for source in sources
        }
        priority = _heuristic_priority(
            sources=sources, normalized=normalized
        )
        confidence = _heuristic_confidence(
            sources=sources,
            normalized=normalized,
            briefing_paragraph=briefing_paragraph or "",
        )

        return ChiefOfStaffSynthesis(
            paragraph=paragraph,
            per_specialist_weight=weights,
            priority_for_principal=priority,
            confidence=confidence,
            source="deterministic",
        )


def _normalize_per_source(raw: dict) -> dict:
    """Coerce a per-source signal record into a stable shape.

    Always returns ``{"candidate_count", "save_count", "top_lane"}``
    with sane defaults. Non-int counts coerce to 0; a missing top
    lane is ``None``.
    """

    try:
        candidate_count = int(raw.get("candidate_count", 0) or 0)
    except (TypeError, ValueError):
        candidate_count = 0
    try:
        save_count = int(raw.get("save_count", 0) or 0)
    except (TypeError, ValueError):
        save_count = 0
    top_lane_raw = raw.get("top_lane")
    top_lane = (
        _normalize_text(top_lane_raw).strip()
        if isinstance(top_lane_raw, str)
        else None
    )
    return {
        "candidate_count": candidate_count,
        "save_count": save_count,
        "top_lane": top_lane or None,
    }


def _heuristic_paragraph(
    *,
    sources: list[str],
    normalized: dict[str, dict],
    briefing_paragraph: str,
) -> str:
    """Build a 2-4 sentence Cloris-voice team-level paragraph.

    Sentence 1 — what the team produced this run (per-source counts).
    Sentence 2 — which specialist's read carried the densest signal,
                 with the lane name when present.
    Sentence 3 (conditional) — a recruiter-facing acknowledgment when
                 no source surfaced saves yet (negative-read framing).
    """

    clauses = [
        f"{_humanize_source(source)} ({normalized[source]['candidate_count']} "
        f"candidate{'s' if normalized[source]['candidate_count'] != 1 else ''}, "
        f"{normalized[source]['save_count']} "
        f"save{'s' if normalized[source]['save_count'] != 1 else ''})"
        for source in sources
    ]
    if len(clauses) == 2:
        run_clause = f"{clauses[0]} and {clauses[1]}"
    elif len(clauses) > 2:
        run_clause = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    else:
        # Single source — defensive; integration layer guards but the
        # heuristic should still produce sensible output if called.
        run_clause = clauses[0]

    sentences = [f"Across {run_clause}, here's what the team turned up."]

    strongest = max(
        sources,
        key=lambda s: (
            normalized[s]["save_count"],
            normalized[s]["candidate_count"],
        ),
    )
    strongest_signals = normalized[strongest]
    top_lane_humanized = _humanize_lane(strongest_signals["top_lane"])
    if strongest_signals["save_count"] > 0:
        if top_lane_humanized:
            sentences.append(
                f"{_humanize_source(strongest)} carried the densest save "
                f"signal — {strongest_signals['save_count']} "
                f"save{'s' if strongest_signals['save_count'] != 1 else ''} "
                f"with {top_lane_humanized} as the strongest lane."
            )
        else:
            sentences.append(
                f"{_humanize_source(strongest)} carried the densest save "
                f"signal — {strongest_signals['save_count']} "
                f"save{'s' if strongest_signals['save_count'] != 1 else ''} "
                f"on {strongest_signals['candidate_count']} "
                f"candidate{'s' if strongest_signals['candidate_count'] != 1 else ''}."
            )
    else:
        # Negative-read framing — none of the specialists surfaced saves.
        # The cross-source read is still informative.
        sentences.append(
            "None of the specialists surfaced saves yet — too small a "
            "sample to weigh their reads against each other."
        )

    return " ".join(sentences)


def _heuristic_weight_for_source(signals: dict) -> dict:
    """Map a per-source signal record to {weight, rationale}.

    Independent 0.0..1.0 score, not normalized. Bounded to [0.3, 0.95]
    so a contributing source never reads as zero-trust (they ran;
    that's a baseline) and never reads as absolute trust (this is a
    heuristic, not the LLM's calibrated read).
    """

    candidate_count = int(signals["candidate_count"] or 0)
    save_count = int(signals["save_count"] or 0)
    if candidate_count <= 0:
        weight = 0.3
        rationale = (
            "Contributed no candidates this run — minimal trust until they "
            "produce something."
        )
    elif save_count == 0:
        weight = 0.4
        rationale = (
            f"Returned {candidate_count} "
            f"candidate{'s' if candidate_count != 1 else ''} and surfaced "
            f"no saves — the negative read is informative."
        )
    else:
        save_rate = save_count / max(candidate_count, 1)
        # Anchor at 0.5; reward save density up to 0.95.
        weight = min(0.95, 0.5 + (0.45 * min(save_rate * 5.0, 1.0)))
        rationale = (
            f"Surfaced {save_count} "
            f"save{'s' if save_count != 1 else ''} on {candidate_count} "
            f"candidate{'s' if candidate_count != 1 else ''} — "
            f"weight scales with save density."
        )
    return {"weight": round(weight, 2), "rationale": rationale}


def _heuristic_priority(
    *, sources: list[str], normalized: dict[str, dict]
) -> str:
    """Pick the strongest source and write a one-sentence action.

    ``Start with the LinkedIn saves first.`` is the canonical shape
    when at least one specialist surfaced saves; the negative-read
    case names the broader market when none did.
    """

    strongest = max(
        sources,
        key=lambda s: (
            normalized[s]["save_count"],
            normalized[s]["candidate_count"],
        ),
    )
    signals = normalized[strongest]
    if signals["save_count"] > 0:
        return (
            f"Start with the {_humanize_source(strongest)} "
            f"save{'s' if signals['save_count'] != 1 else ''} first."
        )
    return (
        "Read the broader market before sequencing the next pass — none of "
        "the specialists surfaced saves to triage yet."
    )


def _heuristic_confidence(
    *,
    sources: list[str],
    normalized: dict[str, dict],
    briefing_paragraph: str,
) -> float:
    """Heuristic confidence formula: populated_fields / 5.

    The 5 scored fields:
      1. ≥2 sources contributing
      2. ≥1 source with saves > 0
      3. ≥1 source with a top lane
      4. ≥1 source with candidates > 0 (sanity)
      5. existing single-briefing paragraph is non-empty
    """

    populated = 0
    if len(sources) >= 2:
        populated += 1
    if any(int(normalized[s]["save_count"]) > 0 for s in sources):
        populated += 1
    if any((normalized[s]["top_lane"] or "") for s in sources):
        populated += 1
    if any(int(normalized[s]["candidate_count"]) > 0 for s in sources):
        populated += 1
    if _normalize_text(briefing_paragraph):
        populated += 1
    return round(populated / 5.0, 2)


# ---------------------------------------------------------------------------
# LLM backend — Opus, with six-route failure cascade to heuristic
# ---------------------------------------------------------------------------


def _dispatch_modules_from_brief(brief: object) -> list[str]:
    """Return declared modules in order from the brief contract."""

    direct = getattr(brief, "target_modules", None)
    if isinstance(direct, list):
        return [str(m) for m in direct if isinstance(m, str) and m]

    raw = getattr(brief, "raw", None)
    if isinstance(raw, dict):
        raw_modules = raw.get("target_modules")
        if isinstance(raw_modules, list):
            return [str(m) for m in raw_modules if isinstance(m, str) and m]

    new_brief = getattr(brief, "_new_brief", None)
    nested = getattr(new_brief, "target_modules", None)
    if isinstance(nested, list):
        return [str(m) for m in nested if isinstance(m, str) and m]

    return []


def _brief_id_for_dispatch(brief: object) -> str:
    direct = getattr(brief, "id", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    raw = getattr(brief, "raw", None)
    if isinstance(raw, dict):
        for key in ("brief_id", "name"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    role_title = getattr(brief, "role_title", None)
    if isinstance(role_title, str) and role_title.strip():
        return role_title.strip()
    return "unknown"


def _principal_id_for_dispatch(brief: object) -> str:
    direct = getattr(brief, "principal_id", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    raw = getattr(brief, "raw", None)
    if isinstance(raw, dict):
        value = raw.get("principal_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _known_launcher_source_keys() -> tuple[str, ...]:
    """Stable tuple of registered launcher keys (orchestrator dispatch allowlist)."""

    return tuple(sorted(LAUNCHERS.keys()))


def _exec_search_mfm_ready() -> bool:
    """Whether exec_search may run under multi-module-foundation rules.

    Stubbed True for Slice 2.6 — Multi-module foundation / workspace
    entries land in Slice 7 (per multi-agent execution plan). Replace
    this body with a real workspace_entries / prerequisite check when
    that slice ships; callers should treat False as
    ``mfm_dependency_unsatisfied`` and fall back to the heuristic
    dispatch planner without re-architecting.
    """

    return True


def _validate_dispatch_schema(raw: Any) -> str | None:
    """Return None if ``raw`` matches the LLM dispatch JSON contract."""

    if not isinstance(raw, dict):
        return "not_dict"
    steps = raw.get("steps")
    if not isinstance(steps, list):
        return "steps_not_list"
    if not steps:
        return "steps_empty"
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            return f"step_not_dict:{idx}"
        name = step.get("module_name")
        if not isinstance(name, str) or not name.strip():
            return f"module_name_empty:{idx}"
        cond = step.get("handoff_condition")
        if cond is not None and not isinstance(cond, str):
            return f"handoff_condition_bad_type:{idx}"
    return None


def _dispatch_plan_from_llm_dict(raw: dict) -> DispatchPlan:
    """Build a :class:`DispatchPlan` after :func:`_validate_dispatch_schema` passes."""

    steps_out: list[DispatchStep] = []
    for step in raw["steps"]:
        hn = step.get("handoff_condition")
        if isinstance(hn, str) and not hn.strip():
            hn = None
        steps_out.append(
            DispatchStep(
                module_name=str(step["module_name"]).strip(),
                handoff_condition=hn,
            )
        )
    return DispatchPlan(steps=steps_out)


def _dispatch_proposes_unknown_source(
    raw: dict, known_sources: tuple[str, ...]
) -> str | None:
    """Return an offending module name if any step is not in ``known_sources``."""

    allowed = frozenset(known_sources)
    steps = raw.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, dict):
            continue
        mod = step.get("module_name")
        if isinstance(mod, str) and mod.strip() and mod not in allowed:
            return mod
    return None


def _dispatch_handoff_self_reference(plan: DispatchPlan) -> bool:
    """True if any step's handoff text references its own ``module_name``.

    Conservative stand-in until ``handoff_condition`` becomes a typed
    reference in v2 — substring check on lowercase forms.
    """

    for step in plan.steps:
        mod = (step.module_name or "").strip().lower()
        cond = (step.handoff_condition or "").strip().lower()
        if mod and mod in cond:
            return True
    return False


def _persist_dispatch_run(*, brief: object, plan: DispatchPlan) -> None:
    store = OrchestrationStateStore(resolve_orchestration_db_path())
    invocation_order = [step.module_name for step in plan.steps]
    store.insert_chief_of_staff_run(
        brief_id=_brief_id_for_dispatch(brief),
        principal_id=_principal_id_for_dispatch(brief),
        status="running",
        dispatch_plan=plan.to_dict(),
        invocation_order=invocation_order,
        handoff_payloads={},
        synthesis_output={},
    )


class ChiefOfStaffAgent:
    """Opus-driven chief-of-staff synthesis. Falls through to heuristic.

    Single entry point: :meth:`synthesize`. Six failure modes converge
    on :class:`HeuristicChiefOfStaffSynthesizer`:

      1. ``llm_raise`` — ``opus_llm`` raises (network, rate-limit,
         timeout, parse error).
      2. ``schema_invalid`` — JSON valid but: ``paragraph`` missing /
         empty / shorter than ``MIN_PARAGRAPH_CHARS``;
         ``per_specialist_weight`` not a dict OR any value not a dict
         OR weight not numeric / outside [0,1] / rationale empty;
         ``priority_for_principal`` missing or empty.
      3. ``banned_token`` — paragraph contains any token in
         :data:`market_intelligence.briefing_polish.BANNED_BRIEFING_TOKENS`.
      4. ``snake_case_token`` — paragraph contains a snake_case
         identifier (engine-vocab leak).
      5. ``specialist_weight_invalid`` — ``per_specialist_weight``
         keys reference a source not in the contributing-sources set.
         The synthesis-specific hallucination check.
      6. ``containment_failed`` — paragraph names no specific value
         from the per-source signals (no candidate count, save count,
         humanized source name, or top lane name).

    Each failure emits ``_emit_stage`` with ``reason=`` so the
    cascade is traceable in logs.
    """

    def __init__(
        self, fallback: HeuristicChiefOfStaffSynthesizer | None = None
    ) -> None:
        self.fallback = fallback or HeuristicChiefOfStaffSynthesizer()

    def _heuristic_dispatch(
        self, brief: Brief, prior_runs: list[dict] | None
    ) -> DispatchPlan:
        del prior_runs  # Heuristic ignores prior-run context (Slice 2.5).
        modules = _dispatch_modules_from_brief(brief)
        return DispatchPlan(
            steps=[
                DispatchStep(module_name=module, handoff_condition=None)
                for module in modules
            ]
        )

    @observe(name="chief_of_staff.dispatch")
    def dispatch(
        self,
        brief: Brief,
        prior_runs: list[dict] | None,
        *,
        persist: bool = True,
    ) -> DispatchPlan:
        """Build dispatch order: Opus JSON plan, or heuristic on failure.

        Six cascade fall-throughs (each emits ``dispatch:fallback`` with a
        ``reason=``) converge on :meth:`_heuristic_dispatch` — the same
        Slice 2.5 deterministic order and empty handoff fields.

          1. ``no_llm_access`` — briefing-polish gate is off.
          2. ``llm_raise`` — ``opus_llm`` raised.
          3. ``schema_invalid`` — JSON shape doesn't match ``DispatchPlan``.
          4. ``unknown_source_proposed`` — a ``module_name`` not in
             ``LAUNCHERS`` keys.
          5. ``mfm_dependency_unsatisfied`` — ``exec_search`` in the plan
             while :func:`_exec_search_mfm_ready` is false (stubbed true
             until multi-module foundation ships).
          6. ``dispatch_loops_back`` — conservative self-reference in
             ``handoff_condition`` (see :func:`_dispatch_handoff_self_reference`).

        When ``persist`` is True, persists exactly one row with the plan
        that ultimately won (LLM or heuristic).
        """

        brief_label = _brief_id_for_dispatch(brief)
        known_sources = _known_launcher_source_keys()

        def finalize(plan: DispatchPlan) -> DispatchPlan:
            if persist:
                _persist_dispatch_run(brief=brief, plan=plan)
            return plan

        if not _has_llm_access():
            _emit_stage(
                f"dispatch:fallback reason=no_llm_access brief={brief_label}"
            )
            return finalize(self._heuristic_dispatch(brief, prior_runs))

        t0 = time.monotonic()
        _emit_stage(
            f"dispatch:start backend=ChiefOfStaffAgent brief={brief_label} "
            f"known_sources={','.join(known_sources)}"
        )
        try:
            raw = opus_llm(
                build_dispatch_system_prompt(),
                build_dispatch_user_prompt(
                    brief=brief,
                    prior_runs=list(prior_runs or []),
                    known_sources=known_sources,
                ),
                expect_json=True,
                max_tokens=SYNTHESIS_MAX_TOKENS,
                usage_context={
                    "stage": "chief_of_staff_dispatch",
                    "brief_id": brief_label,
                },
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"dispatch:fallback reason=llm_raise "
                f"exc={exc.__class__.__name__} elapsed_ms={elapsed_ms} "
                f"brief={brief_label}"
            )
            return finalize(self._heuristic_dispatch(brief, prior_runs))

        validation_failure = _validate_dispatch_schema(raw)
        if validation_failure is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"dispatch:fallback reason=schema_invalid "
                f"detail={validation_failure} elapsed_ms={elapsed_ms} "
                f"brief={brief_label}"
            )
            return finalize(self._heuristic_dispatch(brief, prior_runs))

        bad_source = _dispatch_proposes_unknown_source(raw, known_sources)
        if bad_source is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"dispatch:fallback reason=unknown_source_proposed "
                f"detail=module_name={bad_source!r} elapsed_ms={elapsed_ms} "
                f"brief={brief_label}"
            )
            return finalize(self._heuristic_dispatch(brief, prior_runs))

        plan = _dispatch_plan_from_llm_dict(raw)

        if any(
            step.module_name == "exec_search" and not _exec_search_mfm_ready()
            for step in plan.steps
        ):
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"dispatch:fallback reason=mfm_dependency_unsatisfied "
                f"elapsed_ms={elapsed_ms} brief={brief_label}"
            )
            return finalize(self._heuristic_dispatch(brief, prior_runs))

        if _dispatch_handoff_self_reference(plan):
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"dispatch:fallback reason=dispatch_loops_back "
                f"elapsed_ms={elapsed_ms} brief={brief_label}"
            )
            return finalize(self._heuristic_dispatch(brief, prior_runs))

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _emit_stage(
            f"dispatch:done elapsed_ms={elapsed_ms} brief={brief_label} "
            f"steps={len(plan.steps)}"
        )
        return finalize(plan)

    @observe(name="chief_of_staff.synthesize")
    def synthesize(
        self,
        *,
        market_identity: MarketIdentity,
        per_source_signals: dict[str, dict],
        briefing_paragraph: str,
        deterministic_summary: dict | None = None,
        prior_handoff_payloads: dict[str, dict] | None = None,
    ) -> ChiefOfStaffSynthesis:
        contributing_sources = sorted(per_source_signals.keys())
        sources_label = ",".join(contributing_sources)

        if not _has_llm_access():
            _emit_stage(
                f"synthesis:fallback reason=no_llm_access "
                f"sources={sources_label}"
            )
            return self.fallback.synthesize(
                market_identity=market_identity,
                per_source_signals=per_source_signals,
                briefing_paragraph=briefing_paragraph,
                deterministic_summary=deterministic_summary,
                prior_handoff_payloads=prior_handoff_payloads,
            )

        t0 = time.monotonic()
        # Audit Move #1: surface handoff context in the start log so
        # operators can see at-a-glance whether the synthesis call is
        # consuming the persisted per-source payloads or running on
        # per_source_signals alone (pre-Move-1 behavior).
        handoff_context_size = (
            len(prior_handoff_payloads) if prior_handoff_payloads else 0
        )
        _emit_stage(
            f"synthesis:start backend=ChiefOfStaffAgent sources={sources_label} "
            f"prior_handoff_payloads={handoff_context_size}"
        )
        try:
            raw = opus_llm(
                build_chief_of_staff_system_prompt(),
                build_chief_of_staff_user_prompt(
                    market_identity=market_identity,
                    per_source_signals=per_source_signals,
                    briefing_paragraph=briefing_paragraph,
                    deterministic_summary=deterministic_summary,
                    prior_handoff_payloads=prior_handoff_payloads,
                ),
                expect_json=True,
                max_tokens=SYNTHESIS_MAX_TOKENS,
                usage_context={
                    "stage": "chief_of_staff_synthesis",
                    "market_key": market_identity.market_key,
                    "sources": sources_label,
                },
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"synthesis:fallback reason=llm_raise "
                f"exc={exc.__class__.__name__} elapsed_ms={elapsed_ms}"
            )
            return self.fallback.synthesize(
                market_identity=market_identity,
                per_source_signals=per_source_signals,
                briefing_paragraph=briefing_paragraph,
                deterministic_summary=deterministic_summary,
                prior_handoff_payloads=prior_handoff_payloads,
            )

        # Route 2: schema validity (includes per_specialist_weight value
        # validation — weight type / range, rationale non-empty).
        validation_failure = _validate_schema(raw)
        if validation_failure is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"synthesis:fallback reason=schema_invalid "
                f"detail={validation_failure} elapsed_ms={elapsed_ms}"
            )
            return self.fallback.synthesize(
                market_identity=market_identity,
                per_source_signals=per_source_signals,
                briefing_paragraph=briefing_paragraph,
                deterministic_summary=deterministic_summary,
                prior_handoff_payloads=prior_handoff_payloads,
            )

        paragraph = _normalize_text(raw.get("paragraph"))
        per_specialist_weight = _normalize_weights(
            raw.get("per_specialist_weight")
        )
        priority = _normalize_text(raw.get("priority_for_principal"))

        # Route 3: banned-token check (run before snake_case so a
        # banned-and-snake-case output cascades on the more specific
        # signal first when both are present).
        banned_hit = _banned_token_hit(paragraph)
        if banned_hit is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"synthesis:fallback reason=banned_token "
                f"token={banned_hit!r} elapsed_ms={elapsed_ms}"
            )
            return self.fallback.synthesize(
                market_identity=market_identity,
                per_source_signals=per_source_signals,
                briefing_paragraph=briefing_paragraph,
                deterministic_summary=deterministic_summary,
                prior_handoff_payloads=prior_handoff_payloads,
            )

        # Route 4: snake_case identifier check. Same posture as the
        # briefing-polish backend's route 5 — engine identifiers
        # (lane keys, family keys) are jargon by construction, the
        # prompt instructs the LLM to humanize them, and this regex
        # enforces it after the fact.
        snake_hit = _snake_case_token_hit(paragraph)
        if snake_hit is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"synthesis:fallback reason=snake_case_token "
                f"token={snake_hit!r} elapsed_ms={elapsed_ms}"
            )
            return self.fallback.synthesize(
                market_identity=market_identity,
                per_source_signals=per_source_signals,
                briefing_paragraph=briefing_paragraph,
                deterministic_summary=deterministic_summary,
                prior_handoff_payloads=prior_handoff_payloads,
            )

        # Route 5: specialist_weight_invalid. The synthesis-specific
        # preservation contract — keys in per_specialist_weight MUST
        # be drawn from the contributing-sources set. Catches the
        # "invented specialist" hallucination (the LLM names a
        # researcher / designer that didn't actually run this brief).
        # Modeled on _role_title_drift in market_intelligence/brief_polish.py.
        invented = _specialist_weight_drift(
            per_specialist_weight=per_specialist_weight,
            contributing_sources=set(contributing_sources),
        )
        if invented:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"synthesis:fallback reason=specialist_weight_invalid "
                f"detail=invented_sources={sorted(invented)!r} "
                f"elapsed_ms={elapsed_ms}"
            )
            return self.fallback.synthesize(
                market_identity=market_identity,
                per_source_signals=per_source_signals,
                briefing_paragraph=briefing_paragraph,
                deterministic_summary=deterministic_summary,
                prior_handoff_payloads=prior_handoff_payloads,
            )

        # Route 6: containment check. Mirror briefing_polish:_containment_check
        # — permissive substring match against needles built from
        # the per-source signals. If needles is empty (degenerate
        # input), pass-through; we don't penalize the LLM for not
        # citing values that don't exist.
        if not _containment_check(
            paragraph=paragraph, per_source_signals=per_source_signals
        ):
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"synthesis:fallback reason=containment_failed "
                f"elapsed_ms={elapsed_ms}"
            )
            return self.fallback.synthesize(
                market_identity=market_identity,
                per_source_signals=per_source_signals,
                briefing_paragraph=briefing_paragraph,
                deterministic_summary=deterministic_summary,
                prior_handoff_payloads=prior_handoff_payloads,
            )

        # Success: LLM produced a grounded, in-voice synthesis.
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        result = ChiefOfStaffSynthesis(
            paragraph=paragraph,
            per_specialist_weight=per_specialist_weight,
            priority_for_principal=priority,
            confidence=1.0,
            source="llm",
        )
        _emit_stage(
            f"synthesis:done elapsed_ms={elapsed_ms} "
            f"source={result.source} confidence={result.confidence:.2f}"
        )
        return result


# ---------------------------------------------------------------------------
# Cascade route helpers
# ---------------------------------------------------------------------------


def _validate_schema(raw: Any) -> str | None:
    """Return None on valid schema, else a short failure-reason string.

    Validates:
    - ``raw`` is a dict
    - ``paragraph`` is a string ≥ ``MIN_PARAGRAPH_CHARS``
    - ``per_specialist_weight`` is a dict, every value is a dict,
      ``weight`` is numeric and within [0.0, 1.0], ``rationale`` is a
      non-empty string
    - ``priority_for_principal`` is a non-empty string
    """

    if not isinstance(raw, dict):
        return "not_dict"
    paragraph = raw.get("paragraph")
    if not isinstance(paragraph, str) or len(_normalize_text(paragraph)) < MIN_PARAGRAPH_CHARS:
        return "paragraph_missing_or_short"

    weights = raw.get("per_specialist_weight")
    if not isinstance(weights, dict):
        return "per_specialist_weight_not_dict"
    if not weights:
        return "per_specialist_weight_empty"
    for source_key, value in weights.items():
        if not isinstance(value, dict):
            return f"weight_value_not_dict:{source_key}"
        weight_raw = value.get("weight")
        if not isinstance(weight_raw, (int, float)) or isinstance(
            weight_raw, bool
        ):
            return f"weight_not_numeric:{source_key}"
        weight_val = float(weight_raw)
        if weight_val < 0.0 or weight_val > 1.0:
            return f"weight_out_of_range:{source_key}={weight_val}"
        rationale = value.get("rationale")
        if not isinstance(rationale, str) or not _normalize_text(rationale):
            return f"rationale_missing_or_empty:{source_key}"

    priority = raw.get("priority_for_principal")
    if not isinstance(priority, str) or not _normalize_text(priority):
        return "priority_missing_or_empty"
    return None


def _normalize_weights(raw: Any) -> dict[str, dict]:
    """Normalize the weight payload for output, post-validation.

    Returns ``{source: {weight: float, rationale: str}}``. Defensive
    only — :func:`_validate_schema` has already passed when this is
    called on the success path.
    """

    out: dict[str, dict] = {}
    if not isinstance(raw, dict):
        return out
    for source_key, value in raw.items():
        if not isinstance(value, dict):
            continue
        weight_raw = value.get("weight")
        if not isinstance(weight_raw, (int, float)) or isinstance(
            weight_raw, bool
        ):
            continue
        rationale = _normalize_text(value.get("rationale"))
        out[str(source_key)] = {
            "weight": round(float(weight_raw), 2),
            "rationale": rationale,
        }
    return out


def _banned_token_hit(paragraph: str) -> str | None:
    """Return the first banned token present in paragraph, else None.

    Reuses :data:`market_intelligence.briefing_polish.BANNED_BRIEFING_TOKENS`
    so the chief-of-staff synthesis enforces the same engineer-vocab
    discipline as the per-run editorial briefing — one source of truth.
    """

    lowered = paragraph.lower()
    for token in BANNED_BRIEFING_TOKENS:
        if token in lowered:
            return token
    return None


def _snake_case_token_hit(paragraph: str) -> str | None:
    """Return the first snake_case identifier in paragraph, else None.

    Reuses :data:`market_intelligence.briefing_polish.SNAKE_CASE_IDENTIFIER_RE`.
    Engine identifiers (lane keys, family keys) are jargon by
    construction; the prompt instructs the LLM to humanize them; this
    regex enforces it after the fact. Same lowercase-only posture as
    briefing_polish — uppercase-or-mixed-case underscored words in
    human prose are vanishingly rare and not worth false-positiving.
    """

    match = SNAKE_CASE_IDENTIFIER_RE.search(paragraph)
    return match.group(0) if match else None


def _specialist_weight_drift(
    *,
    per_specialist_weight: dict[str, dict],
    contributing_sources: set[str],
) -> set[str]:
    """Return the set of weight keys NOT in the contributing-sources set.

    Models :func:`market_intelligence.brief_polish._role_title_drift`
    — a preservation contract enforced post-LLM. A non-empty return
    set indicates the LLM hallucinated a specialist that didn't run
    this brief. Empty set = all keys are valid.

    Comparison is exact (case-sensitive). The system prompt
    instructs the LLM to use the exact source-key strings from
    ``per_source_signals``, so a case mismatch is itself a violation
    worth catching.
    """

    if not isinstance(per_specialist_weight, dict):
        return set()
    return {
        key for key in per_specialist_weight.keys() if key not in contributing_sources
    }


def _containment_check(
    *, paragraph: str, per_source_signals: dict[str, dict]
) -> bool:
    """Check that paragraph names ≥1 specific value from per-source signals.

    Mirrors the shape of
    :func:`market_intelligence.briefing_polish._containment_check` —
    permissive substring match (case-insensitive). Rewards grounding
    without penalizing creative phrasing. The minimum bar: at least
    one countable, named, or measured signal from the input is also
    in the output.

    Needles, drawn from per-source signals:
    - Each per-source ``candidate_count`` (>0) as a digit string
    - Each per-source ``save_count`` (>0) as a digit string
    - Each contributing source name lowercased (``linkedin``,
      ``github``) AND its humanized form lowercased (``LinkedIn`` →
      ``linkedin`` after lower; ``GitHub`` → ``github``). The
      humanized form already maps to lowercase for these examples,
      but for sources whose humanized form differs from a simple
      lowercase (``Executive Search``), both surfaces are included.
    - Each per-source ``top_lane`` lowercased AND its humanized form
      lowercased (so ``forward_deployed_engineering`` and
      ``Forward Deployed Engineering`` both pass).

    Empty needles → return True (degenerate input — don't penalize).
    """

    paragraph_l = paragraph.lower()
    needles: list[str] = []
    for source, signals in (per_source_signals or {}).items():
        if not isinstance(signals, dict):
            continue
        candidate_count = int(signals.get("candidate_count", 0) or 0)
        save_count = int(signals.get("save_count", 0) or 0)
        if candidate_count > 0:
            needles.append(str(candidate_count))
        if save_count > 0:
            needles.append(str(save_count))
        source_lower = (source or "").strip().lower()
        if source_lower:
            needles.append(source_lower)
        humanized_source = _humanize_source(source).lower().strip()
        if humanized_source and humanized_source != source_lower:
            needles.append(humanized_source)
        top_lane = signals.get("top_lane")
        if isinstance(top_lane, str) and top_lane.strip():
            raw_lane = top_lane.lower().strip()
            needles.append(raw_lane)
            humanized_lane = _humanize_lane(top_lane).lower().strip()
            if humanized_lane and humanized_lane != raw_lane:
                needles.append(humanized_lane)

    if not needles:
        return True
    return any(needle in paragraph_l for needle in needles)
