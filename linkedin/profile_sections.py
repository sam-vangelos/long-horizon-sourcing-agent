"""Profile section anchors for LinkedIn Recruiter profile panels.

Selectors anchor on stable heading text and semantic classes — never on per-deploy
build hashes. See plans/wave2-profile-dom-reference.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_COUNT_SUFFIX_RE = re.compile(r"\s*\(\d+\)$")

SECTION_SELECTORS: dict[str, dict[str, object]] = {
    "about": {
        "headings": ("summary", "about"),
        "block": ".summary-card",
    },
    "experience": {
        "headings": ("experience",),
        "block": ".experience-card",
        "entry": "li.position-item, li .position-item",
    },
    "education": {
        "headings": ("education",),
        "block": None,
        "entry": "li.education-entity, li .education-entity",
    },
    "skills": {
        "headings": ("skills",),
        "block": None,
    },
}

_HEADING_EVAL_JS = """(root) => [...root.querySelectorAll('h1,h2,h3')].map(h => {
  const r = h.getBoundingClientRect(); const rr = root.getBoundingClientRect();
  return { text: (h.textContent||'').trim(), offset: (r.top - rr.top) + (root.scrollTop||0) };
})"""


@dataclass(frozen=True)
class SectionAnchor:
    name: str
    heading_text: str
    offset: float


def _normalize_heading_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = _COUNT_SUFFIX_RE.sub("", normalized)
    return normalized.strip()


def _heading_matches_section(normalized: str, headings: tuple[str, ...]) -> bool:
    return any(
        normalized == candidate or normalized.startswith(candidate)
        for candidate in headings
    )


def _map_sections(raw_headings: list[dict]) -> list[SectionAnchor]:
    seen: set[str] = set()
    anchors: list[SectionAnchor] = []

    for raw in raw_headings:
        text = str(raw.get("text", ""))
        offset = float(raw.get("offset", 0.0))
        normalized = _normalize_heading_text(text)

        for section_name, spec in SECTION_SELECTORS.items():
            if section_name in seen:
                continue
            headings = spec["headings"]
            assert isinstance(headings, tuple)
            if not _heading_matches_section(normalized, headings):
                continue
            seen.add(section_name)
            anchors.append(
                SectionAnchor(
                    name=section_name,
                    heading_text=normalized,
                    offset=offset,
                )
            )
            break

    return anchors


async def locate_sections(container) -> list[SectionAnchor]:
    try:
        raw = await container.evaluate(_HEADING_EVAL_JS)
    except Exception:
        return []
    return _map_sections(raw or [])
