"""Source-packet driven V2 brief synthesis with field provenance."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import shared.config as shared_config
from market_intelligence.brief_polish import (
    _banned_token_in_v2_draft,
    _snake_case_in_v2_draft,
)
from market_intelligence.briefing_polish import _has_llm_access
from shared.brief_v2_schema import (
    BriefSchemaError,
    normalize_generated_engagement_context,
    validate_v2_brief,
)
from shared.intake_conversation.insights import (
    HIRING_MANAGER_PICTURE_KEY,
    is_generic_trope,
    normalize_hiring_manager_success_image,
)
from shared.llm_clients import opus_llm_cached
from shared.llm_usage import llm_usage_session
from shared.source_packet import normalize_source_text
from shared.source_capabilities import (
    recommend_source_strategy_from_text,
    source_capability_prompt_block,
    target_modules_from_strategy,
)


SOURCE_PACKET_SYNTHESIS_SYSTEM = """You are Cloris authoring a V2 sourcing brief from recruiter-provided source material.

Return one JSON object only. No markdown.

The output must be a valid V2 brief draft plus provenance siblings:
- role_title, role_level, role_summary
- capability_areas: non-empty array of {name, description}
- depth_distinction: {builder_definition, user_definition, edge_case_guidance}
- non_fit_patterns: array of {label, why_not}
- employer_signal_rules: array
- facial_calibration: object with expected_yes_rate_low/high and pattern arrays
- minimum_years_experience: number
- minimum_bar_description: string
- market_density: "sparse" | "moderate" | "dense"
- engagement_context: {selectivity_posture: "selective" | "coverage", optional hiring_company, engagement_description, talent_bar_statement}; use coverage only for sparse markets
- target_modules: array of source keys from the source capability manifest
- source_strategy: array of {source, role, rationale}; role is primary, secondary, corroborating, or investigation_first

ALSO emit (as a sibling, NOT inside any v2 field) one intake-insight key:

- hiring_manager_success_image: {
    "summary": "ONE vivid sentence picturing the person the hiring manager actually wants — role-anchored, recruiter-readable.",
    "proof_points": ["evidence the hiring manager would recognize as real"],
    "screening_translation": "how this picture changes screening behavior",
    "confidence": 0.0-1.0,
    "source": "source_packet",
    "corrected_by_recruiter": false
  }

The picture is load-bearing: when the JD or source material supports a vivid read, emit it. Otherwise omit the key — empty is better than corporate trope. Forbidden phrasing in the summary: "strong communication skills", "team player", "self-starter", "rockstar", "wears many hats". Synthesis writes never set ``corrected_by_recruiter: true``; that flag is reserved for the conversational extractor when the recruiter explicitly corrects the picture.

For each logical field group, add a sibling key ending in __provenance whose leaves carry:
{"confidence": 0.0-1.0, "evidence": "short quote or explanation", "alternatives": []}

Confidence rubric:
- >0.8: directly supported by source material
- 0.4-0.8: reasonable synthesis from multiple cues
- <0.4: filler/gap that needs recruiter validation

Use the current draft as continuity context only. Preserve recruiter-specific corrections when they conflict with weaker source evidence.

SOURCE CAPABILITY MANIFEST:
Evidence boundaries are not permission to skip a module; they define what a module cannot prove alone and what companion evidence completes the read.
""" + source_capability_prompt_block() + """

Do not use these recruiter-facing tokens: hypothesis, tracking, lane_key, planner, critic, artifact.
Do not write snake_case identifiers inside recruiter-facing strings."""

SYNTHESIS_SOURCE_CHAR_BUDGET = 20_000


def _emit_stage(message: str) -> None:
    import sys

    print(f"[intake] {message}", file=sys.stderr, flush=True)


def _resolve_log_path() -> Path:
    base = Path(getattr(shared_config, "OUTPUT_DIR", Path("."))) / "intake_logs"
    base.mkdir(parents=True, exist_ok=True)
    return base / "source_packet_synthesis.jsonl"


def _certify_synthesis_delay_ms() -> int:
    """Read ``CLORIS_CERTIFY_SYNTHESIS_DELAY_MS`` as a non-negative int.

    Returns ``0`` when unset, empty, non-numeric, or negative. Only the
    deterministic/stub synthesis path consults this — the real LLM path
    never sleeps on this value.
    """

    raw = os.getenv("CLORIS_CERTIFY_SYNTHESIS_DELAY_MS", "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, value)


@dataclass(frozen=True)
class SourcePacketSynthesisResult:
    v2_draft: dict[str, Any]
    field_provenance: dict[str, dict[str, Any]]
    confidence_overall: float
    synthesized_at: str
    source: Literal["llm", "deterministic"]
    intake_insights: dict[str, Any] = field(default_factory=dict)
    source_truncated: bool = False
    source_char_count: int = 0

    def to_polish_meta_dict(self) -> dict[str, Any]:
        return {
            "source": "source_packet_synthesis"
            if self.source == "llm"
            else "deterministic",
            "confidence": self.confidence_overall,
            "polished_at": self.synthesized_at,
        }


def synthesize_v2_from_source_packet(
    *,
    source_text: str,
    job_description_text: str = "",
    intake_notes_text: str = "",
    current_v2_draft: dict[str, Any] | None = None,
    field_provenance: dict[str, Any] | None = None,
    geography: str | None = None,
    exemplar_block: str = "",
    recruiter_preferences: str = "",
    session_id: int | None = None,
) -> SourcePacketSynthesisResult:
    """Synthesize a valid V2 draft from source material."""

    t0 = time.monotonic()
    source_text = normalize_source_text(source_text)
    synthesized_at = datetime.now(timezone.utc).isoformat()
    _emit_stage(
        f"intake.source_packet:start session_id={session_id} chars={len(source_text)}"
    )

    def done(
        v2: dict[str, Any],
        provenance: dict[str, dict[str, Any]],
        source: Literal["llm", "deterministic"],
        intake_insights: dict[str, Any] | None = None,
    ) -> SourcePacketSynthesisResult:
        defaults = _default_provenance_for_v2(v2)
        merged = {**defaults, **provenance}
        confidence = _overall_confidence(merged)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _emit_stage(
            f"intake.source_packet:done source={source} confidence={confidence} elapsed_ms={elapsed_ms}"
        )
        source_char_count = len(source_text)
        # Only the LLM path caps the prompt at SYNTHESIS_SOURCE_CHAR_BUDGET; the
        # deterministic/heuristic path consumes the full text, so it never truncates.
        source_truncated = (
            source == "llm" and source_char_count > SYNTHESIS_SOURCE_CHAR_BUDGET
        )
        return SourcePacketSynthesisResult(
            v2_draft=v2,
            field_provenance=merged,
            confidence_overall=confidence,
            synthesized_at=synthesized_at,
            source=source,
            intake_insights=dict(intake_insights or {}),
            source_truncated=source_truncated,
            source_char_count=source_char_count,
        )

    disable_llm = os.getenv("CLORIS_DISABLE_INTAKE_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if len(source_text) < 80 or disable_llm or not _has_llm_access():
        reason = (
            "short_source"
            if len(source_text) < 80
            else "certification_disabled_llm"
            if disable_llm
            else "no_llm_access"
        )
        _emit_stage(f"intake.source_packet:fallback reason={reason}")
        # Cert-only artificial delay: lets the browser-observed
        # certification flow reliably observe the ``running`` state,
        # the "Updating draft" indicator, and ``/api/status`` liveness.
        # Honored only in the deterministic/stub path; the real LLM
        # path is untouched.
        delay_ms = _certify_synthesis_delay_ms()
        if delay_ms > 0:
            _emit_stage(
                f"intake.source_packet:cert_delay_ms={delay_ms} reason={reason}"
            )
            time.sleep(delay_ms / 1000.0)
        v2, provenance, insights = _heuristic_synthesize(
            source_text=source_text,
            job_description_text=job_description_text,
            intake_notes_text=intake_notes_text,
            current_v2_draft=current_v2_draft,
        )
        return done(v2, provenance, "deterministic", intake_insights=insights)

    prompt = _build_user_prompt(
        source_text=source_text,
        current_v2_draft=current_v2_draft,
        field_provenance=field_provenance,
        geography=geography,
        exemplar_block=exemplar_block,
        recruiter_preferences=recruiter_preferences,
    )
    try:
        with llm_usage_session(
            _resolve_log_path(),
            session_id=session_id,
            brief_id=None,
            stage="source_packet_synthesis",
        ):
            raw = opus_llm_cached(
                SOURCE_PACKET_SYNTHESIS_SYSTEM,
                prompt,
                expect_json=True,
                max_tokens=12000,
                usage_context={
                    "stage": "source_packet_synthesis",
                    "session_id": session_id,
                    "source_chars": len(source_text),
                },
            )
    except Exception as exc:  # noqa: BLE001 - LLM cascade fallback
        _emit_stage(
            f"intake.source_packet:fallback reason=llm_raise exc={exc.__class__.__name__}"
        )
        v2, provenance, insights = _heuristic_synthesize(
            source_text=source_text,
            job_description_text=job_description_text,
            intake_notes_text=intake_notes_text,
            current_v2_draft=current_v2_draft,
        )
        return done(v2, provenance, "deterministic", intake_insights=insights)

    if not isinstance(raw, dict):
        _emit_stage("intake.source_packet:fallback reason=invalid_json_root")
        v2, provenance, insights = _heuristic_synthesize(
            source_text=source_text,
            job_description_text=job_description_text,
            intake_notes_text=intake_notes_text,
            current_v2_draft=current_v2_draft,
        )
        return done(v2, provenance, "deterministic", intake_insights=insights)

    v2, provenance, insights_raw = _strip_provenance_keys(raw)
    _normalize_v2(v2, job_description_text=job_description_text, intake_notes_text=intake_notes_text)
    if not _valid_for_intake(v2):
        _emit_stage("intake.source_packet:fallback reason=schema_invalid")
        v2, provenance, insights = _heuristic_synthesize(
            source_text=source_text,
            job_description_text=job_description_text,
            intake_notes_text=intake_notes_text,
            current_v2_draft=current_v2_draft,
        )
        return done(v2, provenance, "deterministic", intake_insights=insights)
    banned = _banned_token_in_v2_draft(v2)
    if banned is not None:
        _emit_stage(f"intake.source_packet:fallback reason=banned_token path={banned[0]}")
        v2, provenance, insights = _heuristic_synthesize(
            source_text=source_text,
            job_description_text=job_description_text,
            intake_notes_text=intake_notes_text,
            current_v2_draft=current_v2_draft,
        )
        return done(v2, provenance, "deterministic", intake_insights=insights)
    snake = _snake_case_in_v2_draft(v2)
    if snake is not None:
        _emit_stage(f"intake.source_packet:fallback reason=snake_case path={snake[0]}")
        v2, provenance, insights = _heuristic_synthesize(
            source_text=source_text,
            job_description_text=job_description_text,
            intake_notes_text=intake_notes_text,
            current_v2_draft=current_v2_draft,
        )
        return done(v2, provenance, "deterministic", intake_insights=insights)

    # Normalize insights from the LLM output through the shared product
    # rule. Trope-shaped or below-floor pictures are dropped here; the
    # synthesis worker never sets ``corrected_by_recruiter: true`` (that
    # flag is reserved for the conversational extractor on a recruiter
    # current-turn override).
    insights = _normalize_synthesis_insights(
        insights_raw,
        v2_draft=v2,
        job_description_text=job_description_text,
        intake_notes_text=intake_notes_text,
        source_text=source_text,
    )
    return done(v2, provenance, "llm", intake_insights=insights)


def _build_user_prompt(
    *,
    source_text: str,
    current_v2_draft: dict[str, Any] | None,
    field_provenance: dict[str, Any] | None,
    geography: str | None,
    exemplar_block: str,
    recruiter_preferences: str,
) -> str:
    payload = {
        "source_packet": source_text[:SYNTHESIS_SOURCE_CHAR_BUDGET],
        "current_v2_draft": current_v2_draft if isinstance(current_v2_draft, dict) else {},
        "field_provenance": field_provenance if isinstance(field_provenance, dict) else {},
        "geography": geography or "",
        "exemplar_patterns": exemplar_block,
        "recruiter_preferences": recruiter_preferences,
    }
    return "Synthesize the brief from this source packet.\n\nINPUT:\n" + json.dumps(
        payload,
        indent=2,
    )


def _heuristic_synthesize(
    *,
    source_text: str,
    job_description_text: str,
    intake_notes_text: str,
    current_v2_draft: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    existing = current_v2_draft if isinstance(current_v2_draft, dict) else {}
    text = source_text or job_description_text or intake_notes_text
    # Empty defaults signal "needs user input"; non-empty defaults like "Role"
    # or "Core role scope" looked like data, confused recruiters, and polluted
    # the read-back chapter's distillation when LLM polish fell back to the
    # heuristic. Each field is left empty if neither LLM synthesis nor the
    # existing v2_draft has a real value.
    title = str(existing.get("role_title") or _first_title(text) or "").strip()
    summary = _first_paragraph(text)
    capability = _capability_from_text(text)
    v2: dict[str, Any] = {
        **existing,
        "role_title": title,
        "role_summary": str(existing.get("role_summary") or summary),
        "capability_areas": _existing_or_default_capabilities(existing, capability),
        "depth_distinction": _existing_or_default_depth(existing),
        "non_fit_patterns": existing.get("non_fit_patterns")
        if isinstance(existing.get("non_fit_patterns"), list)
        else [],
        "source_strategy": existing.get("source_strategy")
        if isinstance(existing.get("source_strategy"), list)
        else recommend_source_strategy_from_text(text),
        "target_modules": _target_modules(existing, text),
        "facial_calibration": existing.get("facial_calibration")
        if isinstance(existing.get("facial_calibration"), dict)
        else {
            "expected_yes_rate_low": 0.25,
            "expected_yes_rate_high": 0.55,
            # P6 (Wave 2): code-filled default band is attributable, never
            # indistinguishable from an authored one (audit R6 provenance).
            "band_source": "synthesis_default",
            "fast_exit_patterns": [],
            "trajectory_yes_patterns": [],
            "trajectory_ambiguous_patterns": [],
            "trajectory_no_patterns": [],
        },
        "minimum_years_experience": existing.get("minimum_years_experience")
        if isinstance(existing.get("minimum_years_experience"), (int, float))
        else 0,
        "minimum_bar_description": str(
            existing.get("minimum_bar_description")
            or _minimum_bar_from_text(text)
        ),
        "market_density": str(existing.get("market_density") or "moderate"),
        "employer_signal_rules": existing.get("employer_signal_rules")
        if isinstance(existing.get("employer_signal_rules"), list)
        else [],
    }
    if job_description_text:
        v2["jd_text"] = job_description_text
    if intake_notes_text:
        v2["intake_notes"] = intake_notes_text
    if not v2["non_fit_patterns"]:
        v2["non_fit_patterns"] = _non_fit_patterns_from_text(text)
    _normalize_v2(v2, job_description_text=job_description_text, intake_notes_text=intake_notes_text)
    insights = _heuristic_hiring_manager_picture(
        v2_draft=v2,
        text=text,
        job_description_text=job_description_text,
        intake_notes_text=intake_notes_text,
    )
    return v2, _default_provenance_for_v2(v2, evidence=summary[:320]), insights


# Known JD section-header lines that aren't role titles. Skip when
# detected so the next line gets a chance to be considered.
_JD_SECTION_HEADERS = frozenset(
    {
        "JOB DESCRIPTION",
        "INTAKE NOTES",
        "RECRUITER GAP ANSWERS",
        "RESPONSIBILITIES",
        "REQUIREMENTS",
        "QUALIFICATIONS",
        "ABOUT THE ROLE",
        "ABOUT US",
        "ABOUT THE COMPANY",
        "ABOUT THE TEAM",
        "DUTIES",
        "WHAT YOU WILL DO",
        "WHAT YOU'LL DO",
        "WHAT WE ARE LOOKING FOR",
        "WHAT WE'RE LOOKING FOR",
        "BENEFITS",
        "COMPENSATION",
        "SALARY",
        "LOCATION",
    }
)

# Phrasal markers — if any of these appear in a candidate line, it's
# JD section-header prose ("Responsibilities include but are not
# limited to:") not a role title.
_JD_HEADER_MARKERS = (
    "responsibilities include",
    "requirements include",
    "qualifications include",
    "what you will do",
    "what you'll do",
    "what we are looking for",
    "what we're looking for",
)


def _first_title(text: str) -> str:
    # A real JD's role title is at the very top, either as the first
    # line or immediately after a section header like "JOB DESCRIPTION".
    # Once we hit any body content (prose paragraph, bullet, section
    # header) we stop looking — title candidates AFTER the body are
    # body lines that happen to be short, not titles.
    #
    # Earlier versions iterated through every line looking for one
    # that matched the title shape. That dragged section-header
    # trailers ("Responsibilities include but are not limited to")
    # and bullet items ("- Partnership and corporate taxation") into
    # role_title. The strict-first-line rule rejects those: a JD
    # opening with a prose paragraph or bullets simply yields empty
    # role_title, and the role chapter asks the recruiter to fill it
    # in themselves. That's the correct UX — recruiter knows the
    # title even when the JD's structure is messy.
    for line in text.strip().splitlines():
        raw = line.rstrip()
        s = line.strip(" #:\t")
        if not s:
            continue
        upper = s.upper()
        if (
            upper in _JD_SECTION_HEADERS
            or upper.startswith("UPLOADED FILE")
            or upper.startswith("UPLOADED ")
        ):
            # Section header — skip and check the next line ONLY for a
            # title candidate. If that next line also fails the title
            # shape, we return "" without looking further down.
            # ``UPLOADED `` covers ``compose_source_packet_text`` group
            # headers like ``UPLOADED JOB DESCRIPTION FILES`` so the JD
            # title underneath, not the wrapper, becomes role_title.
            continue
        # Disqualifiers — if the FIRST content line fails any of these,
        # return "" rather than looking deeper.
        if raw.endswith(":"):
            return ""
        if raw.startswith(("- ", "* ", "•", "·")) or (raw[:3].rstrip(".") in {"1", "2", "3", "4", "5"} and raw[1:3].startswith(". ")):
            return ""
        lower = s.lower()
        if any(marker in lower for marker in _JD_HEADER_MARKERS):
            return ""
        if any(p in s for p in (". ", "! ", "? ")) or s.endswith((".", "!", "?")):
            return ""
        if len(s.split()) > 10:
            return ""
        # Title candidate passed every check.
        return s[:120]
    return ""


def _first_paragraph(text: str) -> str:
    # Empty input → empty output. The earlier placeholder string
    # ("Derived from the recruiter's source material.") leaked into the
    # read-back as if it were real recruiter input.
    stripped = text.strip()
    if not stripped:
        return ""
    paragraphs = re.split(r"\n\s*\n", stripped)
    return (paragraphs[0] or stripped).replace("\n", " ")[:2000]


def _capability_from_text(text: str) -> str:
    paragraphs = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs[1][:2000]
    return _first_paragraph(text)


def _minimum_bar_from_text(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        lower = sentence.lower()
        if any(token in lower for token in ("must have", "required", "minimum", "needs")):
            return sentence.strip()[:500]
    return _first_paragraph(text)[:500]


def _non_fit_patterns_from_text(text: str) -> list[dict[str, str]]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        lower = sentence.lower()
        if any(token in lower for token in ("screen out", "screened out", "not demos", "not a fit")):
            cleaned = sentence.strip()
            return [
                {
                    "label": "Screened-out profile",
                    "description": cleaned[:300],
                    "why_not": cleaned[:300],
                }
            ]
    return []


def _existing_or_default_capabilities(
    existing: dict[str, Any], description: str
) -> list[dict[str, Any]]:
    # One seeded card with empty `name` and the JD-extracted `description`.
    # The brief_v2_schema validator requires a non-empty capability_areas
    # list with name+description keys present. The earlier bug wasn't that
    # we seeded a capability — it was that we hardcoded `name="Core role
    # scope"` (generic placeholder), which then read as if the recruiter
    # had written it. Empty name + real JD-derived description is useful
    # starting content the recruiter can edit; no placeholder leakage.
    current = existing.get("capability_areas")
    if isinstance(current, list) and current:
        return [x for x in current if isinstance(x, dict)]
    return [{"name": "", "description": description}]


def _existing_or_default_depth(existing: dict[str, Any]) -> dict[str, str]:
    # All three sub-fields default empty. The earlier
    # "When the source packet is thin, keep the profile in review rather than
    # inventing evidence." default was a system-prompt-style guardrail that
    # somehow ended up as a user-facing field default, then propagated into
    # the read-back as if the recruiter had written it.
    current = existing.get("depth_distinction")
    if isinstance(current, dict):
        return {
            "builder_definition": str(current.get("builder_definition") or ""),
            "user_definition": str(current.get("user_definition") or ""),
            "edge_case_guidance": str(current.get("edge_case_guidance") or ""),
        }
    return {
        "builder_definition": "",
        "user_definition": "",
        "edge_case_guidance": "",
    }


def _target_modules(existing: dict[str, Any], text: str = "") -> list[str]:
    raw = existing.get("target_modules")
    if isinstance(raw, list):
        vals = [str(x) for x in raw if str(x).strip()]
        if vals:
            return sorted(set(vals))
    strategy = existing.get("source_strategy")
    if isinstance(strategy, list):
        vals = target_modules_from_strategy(
            [x for x in strategy if isinstance(x, dict)]
        )
        if vals:
            return vals
    return target_modules_from_strategy(recommend_source_strategy_from_text(text))


def _normalize_v2(
    v2: dict[str, Any],
    *,
    job_description_text: str,
    intake_notes_text: str,
) -> None:
    strategy = v2.get("source_strategy")
    if not isinstance(strategy, list) or not strategy:
        basis = "\n".join(
            str(v2.get(k) or "")
            for k in ("role_title", "role_summary", "jd_text", "intake_notes")
        )
        strategy = recommend_source_strategy_from_text(basis)
    else:
        strategy = [x for x in strategy if isinstance(x, dict)]
    v2["source_strategy"] = strategy
    if "target_modules" not in v2 or not isinstance(v2.get("target_modules"), list) or not v2["target_modules"]:
        v2["target_modules"] = target_modules_from_strategy(strategy)
    caps = v2.get("capability_areas")
    if not isinstance(caps, list) or not caps:
        # Coerce missing / non-list / empty to a single structural
        # empty-stub card (empty strings, schema-valid). See
        # _existing_or_default_capabilities for the rationale.
        v2["capability_areas"] = [{"name": "", "description": ""}]
    depth = v2.get("depth_distinction")
    if not isinstance(depth, dict):
        depth = {}
    v2["depth_distinction"] = {
        "builder_definition": str(depth.get("builder_definition") or ""),
        "user_definition": str(depth.get("user_definition") or ""),
        "edge_case_guidance": str(depth.get("edge_case_guidance") or ""),
    }
    patterns = v2.get("non_fit_patterns")
    clean_patterns: list[dict[str, Any]] = []
    if isinstance(patterns, list):
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            label = str(pattern.get("label") or "").strip()
            why = str(pattern.get("why_not") or pattern.get("description") or "").strip()
            if label and why:
                clean_patterns.append({"label": label, "why_not": why})
    v2["non_fit_patterns"] = clean_patterns
    if job_description_text:
        v2["jd_text"] = job_description_text
    if intake_notes_text:
        v2["intake_notes"] = intake_notes_text
    normalize_generated_engagement_context(v2)


def _valid_for_intake(v2: dict[str, Any]) -> bool:
    try:
        validate_v2_brief(v2)
        return True
    except BriefSchemaError:
        return False


def _default_provenance_for_v2(
    v2: dict[str, Any],
    *,
    evidence: str = "no direct source support",
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def put(path: str, confidence: float) -> None:
        out[path] = {
            "source": "source_packet_synthesis",
            "confidence": confidence,
            "evidence": evidence,
            "alternatives": [],
        }

    if v2.get("role_title"):
        put("role_title", 0.45)
    for i, area in enumerate(v2.get("capability_areas") or []):
        if isinstance(area, dict):
            put(f"capability_areas[{i}].name", 0.35)
            put(f"capability_areas[{i}].description", 0.45)
    depth = v2.get("depth_distinction")
    if isinstance(depth, dict):
        for key in ("builder_definition", "user_definition", "edge_case_guidance"):
            put(f"depth_distinction.{key}", 0.25 if not depth.get(key) else 0.4)
    if "minimum_bar_description" in v2:
        put("minimum_bar_description", 0.25 if not v2.get("minimum_bar_description") else 0.45)
    return out


def _join_provenance_path(prefix: str, key: str | int) -> str:
    k = str(key)
    if not prefix:
        return k
    if k.isdigit():
        return f"{prefix}[{k}]"
    return f"{prefix}.{k}"


def _flatten_provenance(prefix: str, obj: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(obj, dict):
        return out
    for key, value in obj.items():
        path = _join_provenance_path(prefix, key)
        if isinstance(value, dict) and "confidence" in value and "evidence" in value:
            try:
                confidence = max(0.0, min(1.0, float(value.get("confidence"))))
            except (TypeError, ValueError):
                confidence = 0.5
            out[path] = {
                "source": "source_packet_synthesis",
                "confidence": confidence,
                "evidence": str(value.get("evidence") or ""),
                "alternatives": list(value.get("alternatives") or []),
            }
        elif isinstance(value, dict):
            out.update(_flatten_provenance(path, value))
    return out


_INSIGHT_TOP_LEVEL_KEYS: frozenset[str] = frozenset({HIRING_MANAGER_PICTURE_KEY})


def _strip_provenance_keys(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Partition LLM output into v2 fields, provenance siblings, and intake
    insights.

    Insights are peeled off here — same place provenance is — so they
    never enter the v2 schema validation gate or get scrubbed by the
    banned-token / snake-case checks (which are recruiter-language gates
    targeted at v2 fields).
    """

    v2: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    insights_raw: dict[str, Any] = {}
    for key, value in raw.items():
        if key.endswith("__provenance") and isinstance(value, dict):
            provenance.update(_flatten_provenance(key[: -len("__provenance")], value))
        elif key in _INSIGHT_TOP_LEVEL_KEYS:
            insights_raw[key] = value
        else:
            v2[key] = value
    return v2, provenance, insights_raw


def _overall_confidence(provenance: dict[str, dict[str, Any]]) -> float:
    if not provenance:
        return 0.0
    vals: list[float] = []
    for item in provenance.values():
        try:
            vals.append(float(item.get("confidence") or 0.0))
        except (TypeError, ValueError):
            vals.append(0.0)
    return round(sum(vals) / max(len(vals), 1), 2)


def _normalize_synthesis_insights(
    raw: dict[str, Any],
    *,
    v2_draft: dict[str, Any],
    job_description_text: str,
    intake_notes_text: str,
    source_text: str,
) -> dict[str, Any]:
    """Run synthesis-time insight values through the shared product rule.

    Synthesis is the source-packet path; ``corrected_by_recruiter`` is
    always coerced to ``False`` here regardless of what the LLM emits —
    only the conversational extractor is allowed to set the recruiter
    correction lock.
    """

    if not isinstance(raw, dict) or not raw:
        return {}
    role_context = _synthesis_role_context(
        v2_draft=v2_draft,
        job_description_text=job_description_text,
        intake_notes_text=intake_notes_text,
        source_text=source_text,
    )
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key == HIRING_MANAGER_PICTURE_KEY:
            normalized = normalize_hiring_manager_success_image(
                value, role_context, source="source_packet"
            )
            if normalized is None:
                continue
            normalized["corrected_by_recruiter"] = False
            normalized["source"] = "source_packet"
            out[key] = normalized
    return out


def _synthesis_role_context(
    *,
    v2_draft: dict[str, Any],
    job_description_text: str,
    intake_notes_text: str,
    source_text: str,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    for key in ("role_title", "role_summary"):
        value = v2_draft.get(key) if isinstance(v2_draft, dict) else None
        if isinstance(value, str):
            ctx[key] = value
    if isinstance(v2_draft, dict):
        cap = v2_draft.get("capability_areas")
        if isinstance(cap, list):
            ctx["capability_areas"] = cap
    text_chunks = [
        chunk
        for chunk in (job_description_text, intake_notes_text, source_text)
        if isinstance(chunk, str) and chunk.strip()
    ]
    if text_chunks:
        ctx["text"] = "\n\n".join(text_chunks)
    return ctx


def _heuristic_hiring_manager_picture(
    *,
    v2_draft: dict[str, Any],
    text: str,
    job_description_text: str,
    intake_notes_text: str,
) -> dict[str, Any]:
    """Best-effort hiring-manager picture from the heuristic path.

    Used when the LLM is disabled (certification mode) or unreachable.
    The shape must satisfy ``normalize_hiring_manager_success_image``:
    role-anchored, non-trope, with at least one proof point and a
    screening translation. The heuristic anchors on whatever the v2
    draft and source text actually provide:

    1. ``role_title`` is required (without it there is no role to
       picture).
    2. Capability area names are used to phrase the summary when
       present; otherwise the role title alone carries the anchor.
    3. Proof points come from capability descriptions first, then the
       minimum bar, then the first content paragraph of the source
       text.
    4. The shared ``is_generic_trope`` rule is the final gate — if the
       fallback reads as corporate slop we return ``{}`` so the CTA
       recovery flow surfaces the gap rather than the sidebar showing
       a trope.
    """

    role_title = (
        v2_draft.get("role_title") if isinstance(v2_draft.get("role_title"), str) else ""
    )
    role_title = role_title.strip()
    if not role_title:
        return {}

    cap_areas: list[dict[str, Any]] = []
    raw_cap = v2_draft.get("capability_areas")
    if isinstance(raw_cap, list):
        for item in raw_cap:
            if isinstance(item, dict):
                cap_areas.append(item)
    cap_names = [str(item.get("name") or "").strip() for item in cap_areas]
    cap_names = [n for n in cap_names if n]

    minimum_bar = v2_draft.get("minimum_bar_description")
    if isinstance(minimum_bar, str):
        minimum_bar = minimum_bar.strip()
    else:
        minimum_bar = ""

    if cap_names:
        if len(cap_names) == 1:
            cap_phrase = cap_names[0].lower()
        elif len(cap_names) == 2:
            cap_phrase = f"{cap_names[0].lower()} and {cap_names[1].lower()}"
        else:
            cap_phrase = (
                ", ".join(n.lower() for n in cap_names[:-1])
                + f", and {cap_names[-1].lower()}"
            )
        summary = (
            f"A {role_title} who has personally owned {cap_phrase} — "
            "not someone who has only been around the work."
        )
        screening_anchor = cap_names[0].lower()
    else:
        # Fall back on the role title alone — still a role-anchored
        # summary, just without sub-capability detail.
        summary = (
            f"A {role_title} who has personally shipped the work, "
            f"not someone who has only advised on it."
        )
        screening_anchor = role_title.lower()

    proof_points: list[str] = []
    for item in cap_areas[:3]:
        desc = item.get("description")
        if isinstance(desc, str) and desc.strip():
            proof_points.append(desc.strip())
    if not proof_points and minimum_bar:
        proof_points.append(minimum_bar)
    if not proof_points:
        # Last-resort: the first substantive paragraph of the source
        # text. ``_first_paragraph`` handles section headers + bullets.
        paragraph = _first_paragraph(text or job_description_text or intake_notes_text)
        if paragraph:
            proof_points.append(paragraph[:400])
    if not proof_points:
        return {}

    if minimum_bar:
        screening_translation = (
            f"Reject candidates who cannot show direct, hands-on evidence "
            f"of {screening_anchor} — {minimum_bar}"
        )
    else:
        screening_translation = (
            f"Reject candidates who have only advised on {screening_anchor} "
            "rather than personally owning the work in a comparable setting."
        )

    role_context = {
        "role_title": role_title,
        "role_summary": v2_draft.get("role_summary")
        if isinstance(v2_draft.get("role_summary"), str)
        else "",
        "capability_areas": cap_areas,
        "text": "\n\n".join(
            chunk
            for chunk in (job_description_text, intake_notes_text, text)
            if isinstance(chunk, str) and chunk.strip()
        ),
    }
    if is_generic_trope(summary, role_context):
        return {}

    picture = normalize_hiring_manager_success_image(
        {
            "summary": summary,
            "proof_points": proof_points,
            "screening_translation": screening_translation,
            "confidence": 0.4,
            "source": "source_packet",
            "corrected_by_recruiter": False,
        },
        role_context,
        source="source_packet",
    )
    if picture is None:
        return {}
    return {HIRING_MANAGER_PICTURE_KEY: picture}
