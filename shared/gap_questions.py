"""Gap-question generation from draft provenance and schema shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LOAD_BEARING_FIELDS: tuple[tuple[str, str], ...] = (
    ("role_title", "What exact title and level should Cloris use for this search?"),
    (
        "capability_areas[0].description",
        "What are the two or three capabilities that matter most for this hire?",
    ),
    (
        "depth_distinction.builder_definition",
        "What does true builder-level depth look like for this role?",
    ),
    (
        "depth_distinction.user_definition",
        "What background is merely adjacent or user-level, but not enough?",
    ),
    (
        "depth_distinction.edge_case_guidance",
        "What borderline profiles should Cloris be careful with?",
    ),
    (
        "minimum_bar_description",
        "What is the minimum bar Cloris should refuse to go below?",
    ),
)


@dataclass(frozen=True)
class GapQuestion:
    id: str
    field: str
    question: str
    reason: str
    confidence: float


def generate_gap_questions(
    *,
    v2_draft: dict[str, Any],
    field_provenance: dict[str, Any] | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Return prioritized recruiter questions for missing/weak fields."""

    provenance = field_provenance if isinstance(field_provenance, dict) else {}
    questions: list[GapQuestion] = []
    for field, question in LOAD_BEARING_FIELDS:
        value = _read_path(v2_draft, field)
        prov = provenance.get(field) if isinstance(provenance, dict) else None
        confidence = _confidence(prov)
        missing = _is_empty(value)
        weak = confidence < 0.55
        if not missing and not weak:
            continue
        reason = "missing" if missing else "low confidence"
        questions.append(
            GapQuestion(
                id=_question_id(field),
                field=field,
                question=question,
                reason=reason,
                confidence=confidence,
            )
        )
    questions.sort(key=lambda q: (0 if q.reason == "missing" else 1, q.confidence))
    return [
        {
            "id": q.id,
            "field": q.field,
            "question": q.question,
            "reason": q.reason,
            "confidence": q.confidence,
        }
        for q in questions[:limit]
    ]


def _question_id(field: str) -> str:
    return "gap_" + "".join(ch if ch.isalnum() else "_" for ch in field).strip("_")


def _confidence(value: Any) -> float:
    if not isinstance(value, dict):
        return 0.0
    raw = value.get("confidence")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _read_path(root: dict[str, Any], path: str) -> Any:
    cur: Any = root
    for part in path.split("."):
        if "[" in part and part.endswith("]"):
            key, raw_idx = part[:-1].split("[", 1)
            if not isinstance(cur, dict):
                return None
            arr = cur.get(key)
            if not isinstance(arr, list):
                return None
            try:
                idx = int(raw_idx)
            except ValueError:
                return None
            if idx < 0 or idx >= len(arr):
                return None
            cur = arr[idx]
        else:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
    return cur
