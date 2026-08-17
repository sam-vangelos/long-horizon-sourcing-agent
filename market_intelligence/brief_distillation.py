"""Recruiter-facing read-back for an in-flight V2 brief draft."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import shared.config as shared_config
from market_intelligence.briefing_polish import (
    BANNED_BRIEFING_TOKENS,
    SNAKE_CASE_IDENTIFIER_RE,
    _has_llm_access,
)
from shared.llm_clients import opus_llm_cached
from shared.llm_usage import llm_usage_session


DISTILL_SYSTEM = """You are Cloris writing a brief back to the recruiter as a concise memo, not a form.

Return JSON only:
{
  "prose": "<2-4 short paragraphs. Plain, precise, no headings or bullets.>",
  "structure_map": {"spans": [{"char_start": 0, "char_end": 10, "field": "role_title"}]}
}

Every non-empty load-bearing field should appear in prose. Do not add claims not grounded in the structured brief.
If a field is thin, say it is thin instead of inventing.
Avoid banned internal tokens and snake_case identifiers."""


# Strings that earlier versions of source_packet_synthesis seeded as
# defaults but were never real recruiter input. We detect them in
# v2_draft so the distill fallback can refuse to compose prose around
# them.
PLACEHOLDER_STRINGS: tuple[str, ...] = (
    "Core role scope",
    "Derived from the source packet.",
    "Derived from the recruiter's source material.",
    "When the source packet is thin, keep the profile in review rather than inventing evidence.",
    "Role",
)


def _looks_like_placeholder(value: str, *, kind: str | None = None) -> bool:
    """Return True iff `value` is one of the known placeholder defaults,
    or — when `kind == "role_title"` — looks like JD prose dumped into
    a title field (long + sentence-shaped).
    """

    s = (value or "").strip()
    if not s:
        return False
    if s in PLACEHOLDER_STRINGS:
        return True
    # role_title is a short label, not a sentence. A long sentence-shaped
    # value here is JD prose that leaked through _first_title's heuristic.
    if kind == "role_title" and len(s) > 100 and ("." in s or "\n" in s):
        return True
    return False


@dataclass(frozen=True)
class FaithfulnessReport:
    passes: bool
    missing_fields: list[str]
    weak_fields: list[str]
    overall_overlap: float
    placeholder_fields: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # frozen dataclass post-init dance for the default; keeps the
        # serialization shape backwards compatible.
        if self.placeholder_fields is None:
            object.__setattr__(self, "placeholder_fields", [])


@dataclass(frozen=True)
class DistillationResult:
    prose: str
    structure_map: dict[str, Any]
    faithfulness: FaithfulnessReport
    source: str
    generated_at: str
    deficits: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.deficits is None:
            object.__setattr__(self, "deficits", [])

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "prose": self.prose,
            "structure_map": self.structure_map,
            "faithfulness": {
                "passes": self.faithfulness.passes,
                "missing_fields": self.faithfulness.missing_fields,
                "weak_fields": self.faithfulness.weak_fields,
                "overall_overlap": self.faithfulness.overall_overlap,
                "placeholder_fields": self.faithfulness.placeholder_fields,
            },
            # The read-back chapter renders an editorial empty-state when
            # `deficits` is non-empty AND prose is empty, listing which
            # fields the recruiter still needs to fill in. Better than
            # the slot-fill paragraph the old heuristic emitted.
            "deficits": self.deficits,
            "source": self.source,
            "generated_at": self.generated_at,
        }


def distill_brief(
    *,
    v2_draft: dict[str, Any],
    field_provenance: dict[str, Any] | None = None,
    source_text: str | None = None,
    session_id: int | None = None,
) -> DistillationResult:
    """Distill a structured brief draft into recruiter-facing prose."""

    generated_at = datetime.now(timezone.utc).isoformat()
    if os.getenv("CLORIS_DISABLE_INTAKE_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or not _has_llm_access():
        return _finish(_heuristic_distill(v2_draft), v2_draft, "deterministic", generated_at)
    payload = {
        "v2_draft": v2_draft,
        "field_provenance": field_provenance if isinstance(field_provenance, dict) else {},
        "source_excerpt": (source_text or "")[:4000],
    }
    try:
        with llm_usage_session(
            _log_path(), session_id=session_id, brief_id=None, stage="brief_distillation"
        ):
            raw = opus_llm_cached(
                DISTILL_SYSTEM,
                "Distill this brief.\n\nINPUT:\n" + json.dumps(payload, indent=2),
                expect_json=True,
                max_tokens=4000,
                usage_context={"stage": "brief_distillation", "session_id": session_id},
            )
    except Exception:
        return _finish(_heuristic_distill(v2_draft), v2_draft, "deterministic", generated_at)
    if not isinstance(raw, dict):
        return _finish(_heuristic_distill(v2_draft), v2_draft, "deterministic", generated_at)
    prose = str(raw.get("prose") or "")
    if _banned(prose) or SNAKE_CASE_IDENTIFIER_RE.search(prose):
        return _finish(_heuristic_distill(v2_draft), v2_draft, "deterministic", generated_at)
    report = _faithfulness(prose, v2_draft)
    if not report.passes:
        return _finish(_heuristic_distill(v2_draft), v2_draft, "deterministic", generated_at)
    # Even when LLM output passes faithfulness, refuse to surface prose
    # that was grounded in placeholder-shaped fields. The user would see
    # "Tax Associate" as a role title but the prose would have echoed
    # the original placeholder, masking that the input was bad.
    if report.placeholder_fields:
        return _finish(_heuristic_distill(v2_draft), v2_draft, "deterministic", generated_at)
    return DistillationResult(
        prose=prose,
        structure_map=raw.get("structure_map") if isinstance(raw.get("structure_map"), dict) else {"spans": []},
        faithfulness=report,
        source="llm",
        generated_at=generated_at,
        deficits=[],
    )


def _finish(payload: dict[str, Any], v2_draft: dict[str, Any], source: str, generated_at: str) -> DistillationResult:
    prose = str(payload.get("prose") or "")
    raw_deficits = payload.get("deficits")
    deficits = [str(d) for d in raw_deficits] if isinstance(raw_deficits, list) else []
    # When the heuristic refused to compose, mark the source as "deficits"
    # so callers and the UI can distinguish "we tried and the inputs were
    # bad" from "LLM was unavailable so we fell back."
    effective_source = "deficits" if deficits and not prose else source
    return DistillationResult(
        prose=prose,
        structure_map=payload.get("structure_map") if isinstance(payload.get("structure_map"), dict) else {"spans": []},
        faithfulness=_faithfulness(prose, v2_draft),
        source=effective_source,
        generated_at=generated_at,
        deficits=deficits,
    )


def _heuristic_distill(v2: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback for the read-back.

    Earlier versions of this function slot-filled three paragraphs from
    whatever was in role_title / role_summary / capability_areas /
    depth_distinction. When those slots contained placeholders or raw
    JD prose, the output was readable gibberish that landed in front of
    recruiters as "Cloris's draft." Sam called it incoherent
    belligerent gibberish on 2026-05-13.

    New contract: this function REFUSES to compose prose if the inputs
    look like placeholders or are missing the load-bearing fields a
    recruiter would expect Cloris to have read. When refusing, it
    returns empty prose plus a `deficits` list naming the gaps. The
    review chapter then renders an editorial empty-state pointing the
    user at the specific fields that still need real input.

    When inputs ARE clean, the function composes plain connective prose
    (no taxonomy jargon, no "Cloris would run this as…" service idiom).
    """

    deficits: list[str] = []
    placeholders: list[str] = []

    role = str(v2.get("role_title") or "").strip()
    summary = str(v2.get("role_summary") or "").strip()
    if not role:
        deficits.append("role_title")
    elif _looks_like_placeholder(role, kind="role_title"):
        placeholders.append("role_title")
        deficits.append("role_title")
    # role_summary is optional in the new contract — if missing,
    # the read-back composes around just role_title + capabilities +
    # depth. Only flag as a placeholder if it matches a known default
    # string. Long paragraphs are fine here.
    if summary and _looks_like_placeholder(summary):
        placeholders.append("role_summary")
        summary = ""

    real_areas: list[tuple[str, str]] = []
    for i, area in enumerate(v2.get("capability_areas") or []):
        if not isinstance(area, dict):
            continue
        name = str(area.get("name") or "").strip()
        desc = str(area.get("description") or "").strip()
        if _looks_like_placeholder(name) or _looks_like_placeholder(desc):
            placeholders.append(f"capability_areas[{i}]")
            continue
        if name or desc:
            real_areas.append((name, desc))
    if not real_areas:
        deficits.append("capability_areas")

    depth = v2.get("depth_distinction") if isinstance(v2.get("depth_distinction"), dict) else {}
    builder = str(depth.get("builder_definition") or "").strip()
    user = str(depth.get("user_definition") or "").strip()
    edge = str(depth.get("edge_case_guidance") or "").strip()
    depth_had_placeholders = False
    if _looks_like_placeholder(builder):
        placeholders.append("depth_distinction.builder_definition")
        depth_had_placeholders = True
        builder = ""
    if _looks_like_placeholder(user):
        placeholders.append("depth_distinction.user_definition")
        depth_had_placeholders = True
        user = ""
    if _looks_like_placeholder(edge):
        placeholders.append("depth_distinction.edge_case_guidance")
        depth_had_placeholders = True
        edge = ""
    # Only flag depth_distinction as a deficit when we DETECTED placeholders
    # in it. Bare-empty depth is fine — the read-back composes around what
    # the recruiter has actually written (role + capabilities). Depth is
    # additional texture, not a hard requirement.
    if depth_had_placeholders and not any((builder, user, edge)):
        deficits.append("depth_distinction")

    if deficits:
        # Refuse to compose. The review chapter handles empty-prose +
        # non-empty-deficits as the editorial empty-state.
        return {
            "prose": "",
            "structure_map": {"spans": []},
            "deficits": deficits,
            "placeholder_fields": placeholders,
        }

    # Inputs are clean — compose plain connective prose. No taxonomy
    # jargon, no service-idiom openings. Two short paragraphs: the
    # role + what good looks like, then the depth distinction in the
    # recruiter's own language.
    area_strs = []
    for name, desc in real_areas:
        d = (desc or "").rstrip(".").strip()
        n = (name or "").strip()
        if n and d:
            area_strs.append(f"{n} — {d}")
        else:
            area_strs.append(n or d)
    capability_sentence = (
        "Cloris will judge candidates on " + "; ".join(area_strs) + "."
    ) if area_strs else ""

    depth_pieces = []
    if builder:
        depth_pieces.append(f"What they need to be able to do: {builder.rstrip('.').strip()}.")
    if user:
        depth_pieces.append(f"What looks similar but isn't enough: {user.rstrip('.').strip()}.")
    if edge:
        depth_pieces.append(f"Where to be careful: {edge.rstrip('.').strip()}.")
    depth_sentence = " ".join(depth_pieces).strip()

    paragraphs: list[str] = []
    opening = f"{role}." if not summary else f"{role}. {summary}"
    if opening.strip():
        paragraphs.append(opening.strip())
    if capability_sentence:
        paragraphs.append(capability_sentence)
    if depth_sentence:
        paragraphs.append(depth_sentence)

    prose = "\n\n".join(p for p in paragraphs if p.strip())
    return {
        "prose": prose,
        "structure_map": {"spans": []},
        "deficits": [],
        "placeholder_fields": [],
    }


def _faithfulness(prose: str, v2: dict[str, Any]) -> FaithfulnessReport:
    load = _load_bearing_values(v2)
    # Surface placeholder pollution alongside missing/weak overlap so
    # callers can decide independently whether to trust the prose.
    placeholders: list[str] = []
    for field, value in load:
        kind = "role_title" if field == "role_title" else None
        if _looks_like_placeholder(value, kind=kind):
            placeholders.append(field)
    if not load:
        return FaithfulnessReport(True, [], [], 1.0, placeholders)
    low = prose.lower()
    missing: list[str] = []
    weak: list[str] = []
    hits = 0
    for field, value in load:
        tokens = [t for t in re_words(value) if len(t) >= 4][:8]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in low) / len(tokens)
        if overlap == 0:
            missing.append(field)
        elif overlap < 0.25:
            weak.append(field)
        else:
            hits += 1
    overall = round(hits / max(len(load), 1), 2)
    return FaithfulnessReport(overall >= 0.4, missing, weak, overall, placeholders)


def _load_bearing_values(v2: dict[str, Any]) -> list[tuple[str, str]]:
    vals: list[tuple[str, str]] = []
    for key in ("role_title", "role_summary", "minimum_bar_description"):
        val = v2.get(key)
        if isinstance(val, str) and val.strip():
            vals.append((key, val))
    for i, area in enumerate(v2.get("capability_areas") or []):
        if isinstance(area, dict):
            for key in ("name", "description"):
                val = area.get(key)
                if isinstance(val, str) and val.strip():
                    vals.append((f"capability_areas[{i}].{key}", val))
    depth = v2.get("depth_distinction")
    if isinstance(depth, dict):
        for key in ("builder_definition", "user_definition", "edge_case_guidance"):
            val = depth.get(key)
            if isinstance(val, str) and val.strip():
                vals.append((f"depth_distinction.{key}", val))
    return vals


def re_words(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def _banned(text: str) -> bool:
    low = text.lower()
    return any(token in low for token in BANNED_BRIEFING_TOKENS)


def _log_path() -> Path:
    base = Path(getattr(shared_config, "OUTPUT_DIR", Path("."))) / "intake_logs"
    base.mkdir(parents=True, exist_ok=True)
    return base / "brief_distillation.jsonl"
