"""Editorial briefing polish — Cloris-voice reflection over planner output.

Owns the transformation from planner output (engineer-shape) to a
recruiter-facing editorial briefing (Cloris-voice paragraph + intentions).
Decoupled from ``planner_summary`` so the briefing is computed from
STRUCTURED signals (planner_result fields + deterministic_summary), not
from whatever the planner happened to write into ``planner_summary``.

Two backends:
- :class:`BriefingPolishBackend`: Opus LLM rewrite. Fails through to
  heuristic on any of four conditions (see :func:`polish` docstring).
- :class:`HeuristicBriefingBackend`: deterministic builder from
  structured signals. Always grounded; never surfaces "Tracking N
  hypotheses" or other engineer prose.

Confidence is computed PROGRAMMATICALLY (not LLM self-rating):
- Heuristic: signal-density. populated_fields / 6.
- LLM: containment check. Pass = paragraph contains >=1 specific value
  from the structured input (number, lane name, channel name, percent).
  Pass -> 1.0; fail -> 0.4 + cascade to heuristic.

Banned tokens (see :data:`BANNED_BRIEFING_TOKENS`) are jargon the
planner uses but the recruiter shouldn't see. Both the prompt and the
post-LLM check enforce them.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import shared.config as shared_config
# A.2 cache-gap remediation: briefing polish system prompt is large
# (~7KB voice/preservation contract); caching cuts repeat-call cost
# by ~90% within the 5-minute TTL. Same signature as opus_llm.
from shared.llm_clients import opus_llm_cached as opus_llm
from shared.observability import observe

from market_intelligence.research_prompts import (
    build_briefing_polish_system_prompt,
    build_briefing_polish_user_prompt,
    lane_display_name,
)
from market_intelligence.schema import MarketIdentity


# Catches engineer-vocab leaking into recruiter-facing output. The word
# "snake_case" pattern (`devprod_genai`, `lane_key`, `family_key`,
# `forward_deployed_engineering`) is jargon by construction — recruiters
# don't write or read identifiers with underscores. The polish prompt
# instructs the LLM to translate engine identifiers to recruiter-readable
# names; this regex enforces it after the fact. Triggered as a 5th
# failure route in the cascade (snake_case_token_detected) → heuristic
# fallback. Lowercase letters only because the artifact's lane keys and
# family keys are lowercase by convention.
SNAKE_CASE_IDENTIFIER_RE = re.compile(r"[a-z]+(?:_[a-z]+)+")


# Module-shared so prompt + check use one source of truth. The prompt
# tells the LLM not to use these; the post-LLM check enforces it.
# Lowercased; the check normalizes the paragraph before comparison.
BANNED_BRIEFING_TOKENS: tuple[str, ...] = (
    "hypothesis",
    "tracking",
    "lane_key",
    "planner",
    "critic",
    "artifact",
)

# Minimum paragraph length below which we treat output as degenerate.
# 30 chars catches one-sentence stubs ("Cloris read the market.") that
# pass JSON validation but offer no signal.
MIN_PARAGRAPH_CHARS = 30

# Intentions cap. The LLM is told to return up to N; we trim defensively.
MAX_INTENTIONS = 4

# Polish prompt token budget — modest because input is structured
# summaries, not raw run data.
POLISH_MAX_TOKENS = 3000


def _has_llm_access() -> bool:
    """True when the Anthropic API key is configured.

    Mirrors :func:`market_intelligence.agent_backends._has_llm_access`
    behavior including the test-environment short-circuit.
    """

    import os

    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return bool(getattr(shared_config, "ANTHROPIC_API_KEY", "").strip())


_FALLBACK_REASON_RE = re.compile(r"\bfallback reason=([A-Za-z_][A-Za-z0-9_]*)\b")


def _emit_stage(message: str) -> None:
    """Mirror engine.py _emit_stage so polish stage logs interleave cleanly.

    Phase 1 of Langfuse adoption: when the message carries a
    ``fallback reason=<reason>`` token (the cascade-route convention
    used by :class:`BriefingPolishBackend.polish`), the helper ALSO
    emits the reason as a Langfuse span attribute under
    ``cascade.fallback_reason``. Single bridging point so the polish
    backend's fallback call sites don't each have to know about the
    observability layer. No-op when the Langfuse client is null /
    disabled / network-degraded. Same pattern as
    :func:`cloris.chief_of_staff.agent._emit_stage`.
    """

    import sys

    print(f"[market-intel] {message}", file=sys.stderr, flush=True)

    match = _FALLBACK_REASON_RE.search(message)
    if match is not None:
        try:
            from shared.observability import update_current_observation

            update_current_observation(
                metadata={"cascade.fallback_reason": match.group(1)}
            )
        except Exception:  # noqa: BLE001 — Langfuse path is fail-soft
            pass


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


@dataclass
class EditorialBriefing:
    """Recruiter-facing reflection over a sourcing run.

    ``paragraph`` is 2-4 sentences in Cloris voice. ``intentions`` is a
    list of one-line items (text + priority). ``confidence`` is the
    programmatic grounding score (heuristic: signal-density; LLM:
    containment-check pass=1.0/fail=0.4-cascade). ``source`` is one of
    ``"llm"``, ``"deterministic"``, ``"empty"`` so the operator can
    tell at a glance which path fired.
    """

    paragraph: str
    intentions: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "empty"

    def to_dict(self) -> dict:
        return {
            "paragraph": self.paragraph,
            "intentions": list(self.intentions),
            "confidence": round(float(self.confidence), 2),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Heuristic — deterministic builder from structured signals
# ---------------------------------------------------------------------------


class HeuristicBriefingBackend:
    """Deterministic briefing builder. Always grounded in concrete signals.

    Composes the paragraph from ``deterministic_summary.aggregate_metrics``
    (run count, save count, save rate, channel skew) plus the strongest
    lane from ``deterministic_summary.lane_intelligence`` if present.
    Never produces output of the form "Tracking N active hypotheses" or
    "Run 3 executed family X" — those are engineer narratives.

    Confidence = signal-density: count of populated structured fields
    divided by max possible (currently 6). Single-run zero-saves scores
    ~0.33; multi-run multi-lane scores ~1.0.
    """

    def polish(
        self,
        *,
        market_identity: MarketIdentity,
        deterministic_summary: dict,
        planner_result: Any,  # PlannerResult; Any to avoid circular import
        steering_notes: list[str] | None = None,
    ) -> EditorialBriefing:
        agg = (deterministic_summary or {}).get("aggregate_metrics") or {}
        run_count = int(agg.get("run_count") or 0)
        saved_count = int(agg.get("saved_count") or 0)
        save_rate = float(agg.get("save_rate") or 0.0)
        channel_volumes = agg.get("candidate_volume_by_channel") or {}
        candidate_volume = sum(
            int(v or 0) for v in channel_volumes.values()
        )
        lanes = (deterministic_summary or {}).get("lane_intelligence") or []
        top_lane = _top_lane(lanes)
        intentions = _intentions_from_focus(planner_result)
        steering_notes = list(steering_notes or [])

        # Compose paragraph based on signal density.
        paragraph = _heuristic_paragraph(
            run_count=run_count,
            saved_count=saved_count,
            save_rate=save_rate,
            candidate_volume=candidate_volume,
            channel_volumes=channel_volumes,
            top_lane=top_lane,
            intentions=intentions,
            steering_notes=steering_notes,
        )

        confidence = _signal_density_confidence(
            run_count=run_count,
            saved_count=saved_count,
            save_rate=save_rate,
            candidate_volume=candidate_volume,
            channel_volumes=channel_volumes,
            top_lane=top_lane,
            intentions=intentions,
        )

        if not paragraph.strip():
            return EditorialBriefing(
                paragraph=(
                    "I don't have enough from this run to draw conclusions yet "
                    "— let me read the broader market."
                ),
                intentions=intentions,
                confidence=0.0,
                source="empty",
            )

        return EditorialBriefing(
            paragraph=paragraph,
            intentions=intentions[:MAX_INTENTIONS],
            confidence=confidence,
            source="deterministic",
        )


def _heuristic_paragraph(
    *,
    run_count: int,
    saved_count: int,
    save_rate: float,
    candidate_volume: int,
    channel_volumes: dict,
    top_lane: dict | None,
    intentions: list[dict],
    steering_notes: list[str],
) -> str:
    """Build a 2-4 sentence Cloris-voice paragraph from structured signals.

    The paragraph structure is:
      1. What happened (concrete numbers from this run / runs).
      2. Where the strongest signal came from (lane, if any).
      3. What I want to do next (derived from intentions).
      4. (Optional) Acknowledgment of the operator's steering note.

    Each clause is conditional on signal — empty signals collapse to a
    shorter paragraph rather than padding with filler. Cold-start
    single-run case lands here gracefully.
    """

    sentences: list[str] = []

    # Sentence 1: what happened.
    if candidate_volume > 0 and run_count > 0:
        runs_phrase = (
            "your first run"
            if run_count == 1
            else f"{run_count} runs"
        )
        if saved_count == 0:
            sentences.append(
                f"I read {candidate_volume} candidates from {runs_phrase}. "
                f"None landed in your save list yet."
            )
        else:
            pct_phrase = (
                f" — about {int(round(save_rate * 100))}%"
                if save_rate > 0
                else ""
            )
            saves_phrase = "save" if saved_count == 1 else "saves"
            sentences.append(
                f"I read {candidate_volume} candidates across {runs_phrase} "
                f"and saved {saved_count}{pct_phrase}."
            )
    elif run_count > 0:
        sentences.append(
            f"Your last {('run' if run_count == 1 else f'{run_count} runs')} "
            f"finished without surfacing candidates."
        )
    else:
        # No runs at all — defer to the empty-state path in caller.
        return ""

    # Sentence 2: lane signal.
    if top_lane is not None:
        # Translate via shared lane_display_name so engine identifiers
        # (e.g. "devprod_genai", "forward_deployed_engineering") never
        # leak verbatim into recruiter-facing prose. Mirrors the LLM
        # prompt's lane_translation_table contract — both surfaces
        # produce the same humanized name for any given lane_key.
        lane_label = _normalize_text(lane_display_name(top_lane))
        lane_saves = int(top_lane.get("saved_count") or 0)
        if lane_label and lane_saves > 0:
            saves_phrase = "save" if lane_saves == 1 else "saves"
            sentences.append(
                f"The strongest signal came from {lane_label} "
                f"with {lane_saves} of those {saves_phrase}."
            )
        elif lane_label:
            sentences.append(
                f"Most of the volume came through {lane_label}, "
                f"but it didn't surface saves yet."
            )

    # Sentence 3: what I want to look into next.
    if intentions:
        if saved_count == 0:
            # Cold-start framing — name that the sample is small.
            sentences.append(
                "Too small a sample to draw lane conclusions. "
                "I want to read what the broader market looks like before our next pass."
            )
        elif len(intentions) == 1:
            sentences.append(
                f"I want to find out whether {_intention_as_predicate(intentions)} "
                f"before our next run."
            )
        else:
            sentences.append(
                "I want to look more closely at adjacent talent pools "
                "and search posture before our next run."
            )
    elif saved_count == 0:
        sentences.append(
            "I want to read the broader market before suggesting changes to the brief."
        )

    # Sentence 4 (optional): steering ack.
    if steering_notes:
        most_recent = _normalize_text(steering_notes[-1])
        if most_recent:
            sentences.append(
                f"Per your note, I'll also factor in: \"{most_recent}\"."
            )

    return " ".join(sentences)


def _intentions_from_focus(planner_result: Any) -> list[dict]:
    """Translate planner external_research_focus into editorial bullets.

    Each item: {"text": "...", "priority": "high|medium|low"}.
    Falls back to edge_case_research_focus, then to a single generic
    bullet so the UI always has something to render.
    """

    items: list[dict] = []
    for raw in (
        getattr(planner_result, "external_research_focus", []) or []
    ):
        if not isinstance(raw, dict):
            continue
        focus = _normalize_text(raw.get("focus"))
        if not focus:
            continue
        priority = _normalize_text(raw.get("priority")).lower() or "medium"
        items.append({"text": focus, "priority": priority})
    if not items:
        for raw in (
            getattr(planner_result, "edge_case_research_focus", []) or []
        ):
            if not isinstance(raw, dict):
                continue
            focus = _normalize_text(raw.get("focus"))
            if not focus:
                continue
            priority = _normalize_text(raw.get("priority")).lower() or "medium"
            items.append({"text": focus, "priority": priority})
    if not items:
        items.append(
            {
                "text": (
                    "Whether anything in the broader market should change "
                    "how we sequence the next search."
                ),
                "priority": "medium",
            }
        )
    return items


def _intention_as_predicate(intentions: list[dict]) -> str:
    """Render the first intention as a predicate that follows
    "I want to find out whether ...".

    The planner / intention list naturally phrases items as
    "Whether X" questions. We strip the leading "Whether " so the
    paragraph builder can prefix "I want to find out whether ..." and
    get one grammatical sentence.

    For non-Whether-style intentions (rare but possible), we fall
    back to a simpler "what's true about the broader market" frame so
    the paragraph never reads broken.
    """

    if not intentions:
        return "what's shifting in the broader market"
    text = _normalize_text(intentions[0].get("text"))
    if not text:
        return "what's shifting in the broader market"
    # Strip leading "Whether " so the predicate slots into
    # "I want to find out whether <X>".
    if text.lower().startswith("whether "):
        return text[8:].rstrip(".") or "what's shifting in the broader market"
    # Non-Whether-style intention: use the full text as a noun phrase.
    return text.rstrip(".")


def _top_lane(lanes: list[dict]) -> dict | None:
    """Pick the strongest lane by saved_count then candidate_volume."""

    if not lanes:
        return None
    sortable = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        saved = int(lane.get("saved_count") or 0)
        volume = int(lane.get("candidate_volume") or 0)
        sortable.append((saved, volume, lane))
    if not sortable:
        return None
    sortable.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return sortable[0][2]


def _signal_density_confidence(
    *,
    run_count: int,
    saved_count: int,
    save_rate: float,
    candidate_volume: int,
    channel_volumes: dict,
    top_lane: dict | None,
    intentions: list[dict],
) -> float:
    """Heuristic confidence formula: populated_fields / 6.

    The 6 fields scored:
      1. run_count > 0
      2. saved_count > 0
      3. save_rate > 0
      4. top_lane present (with a name)
      5. channel skew computable (>=1 channel with volume)
      6. intentions count > 0
    """

    populated = 0
    if run_count > 0:
        populated += 1
    if saved_count > 0:
        populated += 1
    if save_rate > 0:
        populated += 1
    if top_lane is not None and _normalize_text(
        top_lane.get("display_name") or top_lane.get("lane_key")
    ):
        populated += 1
    if any(int(v or 0) > 0 for v in (channel_volumes or {}).values()):
        populated += 1
    if intentions:
        populated += 1
    return round(populated / 6.0, 2)


# ---------------------------------------------------------------------------
# LLM polish — Opus, with four-route failure cascade to heuristic
# ---------------------------------------------------------------------------


class BriefingPolishBackend:
    """Opus-driven editorial polish. Falls through to heuristic on failure.

    Single entry point: :func:`polish`. Four failure modes converge on
    :class:`HeuristicBriefingBackend`:

      1. ``opus_llm`` raises (network, rate-limit, timeout, parse error)
      2. JSON valid but schema invalid: ``paragraph`` missing/empty,
         or ``intentions`` not a list
      3. Containment check fails: paragraph contains no specific value
         from the structured input
      4. Banned-token check fails: paragraph contains any token in
         :data:`BANNED_BRIEFING_TOKENS`

    Each failure emits ``_emit_stage`` with ``reason=`` so the cascade
    is traceable in logs.
    """

    def __init__(self, fallback: HeuristicBriefingBackend | None = None) -> None:
        self.fallback = fallback or HeuristicBriefingBackend()

    @observe(name="market_intel.briefing_polish")
    def polish(
        self,
        *,
        market_identity: MarketIdentity,
        deterministic_summary: dict,
        planner_result: Any,  # PlannerResult; Any to avoid circular import
        steering_notes: list[str] | None = None,
    ) -> EditorialBriefing:
        if not _has_llm_access():
            _emit_stage("reflection.polish:fallback reason=no_llm_access")
            return self.fallback.polish(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                planner_result=planner_result,
                steering_notes=steering_notes,
            )

        t0 = time.monotonic()
        _emit_stage(
            "reflection.polish:start backend=BriefingPolishBackend "
            f"market={market_identity.market_key}"
        )
        try:
            raw = opus_llm(
                build_briefing_polish_system_prompt(),
                build_briefing_polish_user_prompt(
                    market_identity=market_identity,
                    deterministic_summary=deterministic_summary,
                    planner_result=planner_result,
                    steering_notes=list(steering_notes or []),
                ),
                expect_json=True,
                max_tokens=POLISH_MAX_TOKENS,
                usage_context={
                    "stage": "market_intel_briefing_polish",
                    "market_key": market_identity.market_key,
                },
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"reflection.polish:fallback reason=llm_raise "
                f"exc={exc.__class__.__name__} elapsed_ms={elapsed_ms}"
            )
            return self.fallback.polish(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                planner_result=planner_result,
                steering_notes=steering_notes,
            )

        # Route 2: schema validity.
        validation_failure = _validate_schema(raw)
        if validation_failure is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"reflection.polish:fallback reason=schema_invalid "
                f"detail={validation_failure} elapsed_ms={elapsed_ms}"
            )
            return self.fallback.polish(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                planner_result=planner_result,
                steering_notes=steering_notes,
            )

        paragraph = _normalize_text(raw.get("paragraph"))
        intentions = _normalize_intentions(raw.get("intentions"))

        # Route 4: banned-token check (run before containment so a
        # banned-and-grounded output still falls through correctly).
        banned_hit = _banned_token_hit(paragraph)
        if banned_hit is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"reflection.polish:fallback reason=banned_token "
                f"token={banned_hit!r} elapsed_ms={elapsed_ms}"
            )
            return self.fallback.polish(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                planner_result=planner_result,
                steering_notes=steering_notes,
            )

        # Route 5: snake_case identifier check. Any underscore-bearing
        # identifier in the polished paragraph is engineer-vocab leaking
        # into recruiter-facing output. The prompt instructs the LLM to
        # translate engine identifiers (lane keys, family keys, etc.) to
        # recruiter-readable display names; this enforces it programmatically.
        # Same heuristic-fallback posture as the four routes above.
        snake_hit = _snake_case_token_hit(paragraph)
        if snake_hit is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"reflection.polish:fallback reason=snake_case_token_detected "
                f"token={snake_hit!r} elapsed_ms={elapsed_ms}"
            )
            return self.fallback.polish(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                planner_result=planner_result,
                steering_notes=steering_notes,
            )

        # Route 3: containment check. If the paragraph doesn't ground
        # itself in at least one specific value from the input, treat as
        # ungrounded and cascade — same posture as a degenerate response.
        contained = _containment_check(
            paragraph=paragraph,
            deterministic_summary=deterministic_summary,
        )
        if not contained:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"reflection.polish:fallback reason=containment_failed "
                f"elapsed_ms={elapsed_ms}"
            )
            return self.fallback.polish(
                market_identity=market_identity,
                deterministic_summary=deterministic_summary,
                planner_result=planner_result,
                steering_notes=steering_notes,
            )

        # Success: LLM produced a grounded, in-voice paragraph.
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        result = EditorialBriefing(
            paragraph=paragraph,
            intentions=intentions[:MAX_INTENTIONS],
            confidence=1.0,
            source="llm",
        )
        _emit_stage(
            f"reflection.polish:done elapsed_ms={elapsed_ms} "
            f"source={result.source} confidence={result.confidence:.2f}"
        )
        return result


def _validate_schema(raw: Any) -> str | None:
    """Return None on valid schema, else a short failure reason string."""

    if not isinstance(raw, dict):
        return "not_dict"
    paragraph = raw.get("paragraph")
    if not isinstance(paragraph, str) or len(_normalize_text(paragraph)) < MIN_PARAGRAPH_CHARS:
        return "paragraph_missing_or_short"
    intentions = raw.get("intentions")
    if not isinstance(intentions, list):
        return "intentions_not_list"
    return None


def _normalize_intentions(raw: Any) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = _normalize_text(item.get("text"))
        if not text:
            continue
        priority = _normalize_text(item.get("priority")).lower()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        out.append({"text": text, "priority": priority})
    return out


def _banned_token_hit(paragraph: str) -> str | None:
    """Return the first banned token present in paragraph (case-insensitive), else None."""

    lowered = paragraph.lower()
    for token in BANNED_BRIEFING_TOKENS:
        if token in lowered:
            return token
    return None


def _snake_case_token_hit(paragraph: str) -> str | None:
    """Return the first snake_case identifier present in paragraph, else None.

    Engine identifiers (lane keys like ``devprod_genai``, family keys
    like ``forward_deployed_engineering``) are jargon by construction
    — recruiters don't read or write underscore-bearing identifiers.
    The polish prompt instructs the LLM to translate to display names;
    this regex enforces it after the fact.

    Lowercase only because the artifact's keys are lowercase by
    convention (``derive_market_key`` / ``normalize_family_key`` both
    slugify to lowercase). Uppercase or mixed-case underscored words
    in human prose (``ASCII_TEXT``) are vanishingly rare in this
    surface and not worth false-positiving.
    """

    match = SNAKE_CASE_IDENTIFIER_RE.search(paragraph)
    return match.group(0) if match else None


def _containment_check(
    *, paragraph: str, deterministic_summary: dict
) -> bool:
    """Check that paragraph names >=1 specific value from the structured input.

    Acceptable references: any of run_count, saved_count, candidate_volume
    rendered as a number; the top lane name; a channel name (linkedin /
    github); or a save_rate rendered as a percent (with or without %).

    The check is permissive — we want to reward grounding without
    penalizing creative phrasing. The minimum bar is: at least one
    countable, named, or measured signal from the input is also in the
    output.
    """

    paragraph_l = paragraph.lower()
    agg = (deterministic_summary or {}).get("aggregate_metrics") or {}
    run_count = int(agg.get("run_count") or 0)
    saved_count = int(agg.get("saved_count") or 0)
    save_rate = float(agg.get("save_rate") or 0.0)
    candidate_volume = sum(
        int(v or 0)
        for v in (agg.get("candidate_volume_by_channel") or {}).values()
    )

    needles: list[str] = []
    for n in (run_count, saved_count, candidate_volume):
        if n > 0:
            needles.append(str(n))
    if save_rate > 0:
        pct = int(round(save_rate * 100))
        if pct > 0:
            needles.append(f"{pct}%")
            needles.append(str(pct))
    for channel in (agg.get("candidate_volume_by_channel") or {}).keys():
        normalized = _normalize_text(channel).lower()
        if normalized:
            needles.append(normalized)
    top_lane = _top_lane(
        (deterministic_summary or {}).get("lane_intelligence") or []
    )
    if top_lane is not None:
        # Include both the humanized form (what the LLM SHOULD produce
        # per the prompt's translation rule) and the raw lane_key (in
        # case the LLM violates and quotes raw — the snake_case route
        # then catches it). Both as lowercase needles. This makes
        # correctly-humanized output pass containment without giving
        # raw-quoting output a free pass.
        humanized = lane_display_name(top_lane).lower().strip()
        if humanized:
            needles.append(humanized)
        raw_key = _normalize_text(top_lane.get("lane_key")).lower()
        if raw_key:
            needles.append(raw_key)

    if not needles:
        # No signals to ground against. Don't penalize the LLM for
        # not citing values that don't exist; treat as "passed".
        return True
    return any(needle in paragraph_l for needle in needles)
