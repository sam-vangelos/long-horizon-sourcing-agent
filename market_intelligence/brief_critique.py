"""Natural-language brief critique to structured edit proposals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import shared.config as shared_config
from market_intelligence.briefing_polish import _has_llm_access
from shared.llm_clients import opus_llm_cached
from shared.llm_usage import llm_usage_session


CRITIQUE_SYSTEM = """Convert a recruiter's natural-language critique into JSON edit operations for a V2 sourcing brief.

Return JSON only:
{"edits": [{"field": "<path>", "op": "set", "value": <json value>}]}

Allowed field paths include:
- role_title, role_summary, role_level, minimum_years_experience, minimum_bar_description, market_density, instructions, notes
- depth_distinction.builder_definition, depth_distinction.user_definition, depth_distinction.edge_case_guidance
- capability_areas[0].name, capability_areas[0].description
- non_fit_patterns[0].label, non_fit_patterns[0].why_not

Only emit edits clearly requested by the critique. Do not rewrite unrelated fields."""


@dataclass
class CritiqueResult:
    edits: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    stage_errors: list[str] = field(default_factory=list)
    source: Literal["llm", "deterministic"] = "deterministic"


class BriefCritiqueBackend:
    """Parse recruiter critique into structured field edits."""

    def parse(
        self,
        *,
        critique_text: str,
        v2_draft: dict[str, Any],
        affirmed_fields: list[str] | None = None,
        field_provenance: dict[str, Any] | None = None,
        session_id: int | None = None,
    ) -> CritiqueResult:
        del affirmed_fields, field_provenance
        direct = _try_json_critique(critique_text)
        if direct is not None:
            return CritiqueResult(edits=direct, source="deterministic")
        if not _has_llm_access():
            return CritiqueResult(edits=_heuristic_edits(critique_text), source="deterministic")
        payload = {"v2_draft": v2_draft, "critique": critique_text}
        try:
            with llm_usage_session(
                _log_path(), session_id=session_id, brief_id=None, stage="brief_critique"
            ):
                raw = opus_llm_cached(
                    CRITIQUE_SYSTEM,
                    "Turn the critique into structured edits.\n\nINPUT:\n"
                    + json.dumps(payload, indent=2),
                    expect_json=True,
                    max_tokens=4096,
                    usage_context={"stage": "brief_critique", "session_id": session_id},
                )
        except Exception as exc:  # noqa: BLE001
            return CritiqueResult(stage_errors=[str(exc)], source="deterministic")
        if not isinstance(raw, dict):
            return CritiqueResult(stage_errors=["critique response not an object"], source="llm")
        return CritiqueResult(edits=_normalize_edits(raw.get("edits")), source="llm")


def _try_json_critique(critique_text: str) -> list[dict[str, Any]] | None:
    s = critique_text.strip()
    if not s.startswith("{"):
        return None
    try:
        raw = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return _normalize_edits(raw.get("edits"))


def _normalize_edits(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        if not field:
            continue
        out.append({"field": field, "op": str(item.get("op") or "set").lower(), "value": item.get("value")})
    return out


def _heuristic_edits(critique_text: str) -> list[dict[str, Any]]:
    low = critique_text.lower()
    edits: list[dict[str, Any]] = []
    if "title" in low and " to " in low:
        tail = critique_text.rsplit(" to ", 1)[-1].strip().strip(".")
        if 2 <= len(tail) <= 120:
            edits.append({"field": "role_title", "op": "set", "value": tail})
    if "minimum" in low and "bar" in low:
        edits.append(
            {
                "field": "minimum_bar_description",
                "op": "set",
                "value": critique_text.strip(),
            }
        )
    return edits


def _log_path() -> Path:
    base = Path(getattr(shared_config, "OUTPUT_DIR", Path("."))) / "intake_logs"
    base.mkdir(parents=True, exist_ok=True)
    return base / "brief_critique.jsonl"
