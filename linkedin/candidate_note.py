"""Render LinkedIn Recruiter notes from candidate judgments."""

from __future__ import annotations

from datetime import datetime
import math
import re
import unicodedata

from shared import config
from shared.schemas import OpusDecision, SearchString


# The duplicate-detection window. A note carries no authorship marker, so a
# retried write recognises its own note by that note's OWN opening line. Sixty
# characters because Recruiter's collapsed note render length is not recorded
# (notes are `defer` in docs/linkedin-recruiter-dom-map.md): widen this only
# against a live measurement.
NOTE_PROBE_MAX_CHARS = 60
_SENTENCE_BOUNDARY = re.compile(r'[.!?](?:["”’\']?)(?=\s|$)')
_WORD_BOUNDARY = re.compile(r"\s+")


def note_probe_text(note_text: object) -> str:
    """Return the duplicate-detection window for a rendered note.

    The window is the note's first line, whitespace-normalized (so rendered
    line wrapping cannot break a match) and bounded. An empty note yields an
    empty probe, which callers MUST refuse rather than spend as a match: `"" in
    anything` is always true.
    """
    first_line = str(note_text or "").lstrip().split("\n", 1)[0]
    return " ".join(first_line.split())[:NOTE_PROBE_MAX_CHARS]


def _sanitize(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("@", "at")
    text = "".join(
        char
        for char in text
        if char == "\n" or unicodedata.category(char) != "Cc"
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _outside_quotes(text: str) -> list[bool]:
    outside = [True]
    in_quote = False
    for char in text:
        if char in {'"', "“"}:
            in_quote = not in_quote if char == '"' else True
        elif char == "”":
            in_quote = False
        outside.append(not in_quote)
    return outside


def _truncate_at_word(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    if max_chars == 1:
        return "…"

    limit = max_chars - 1
    boundaries = [match.start() for match in _WORD_BOUNDARY.finditer(text, 0, limit + 1)]
    end = boundaries[-1] if boundaries else limit
    return f"{text[:end].rstrip()}…"


def _truncate_evidence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return "…" if max_chars == 1 else ""

    limit = max_chars - 1
    outside = _outside_quotes(text)
    sentence_floor = math.ceil(limit * 0.6)
    sentence_ends = [
        match.end()
        for match in _SENTENCE_BOUNDARY.finditer(text, 0, limit + 1)
        if sentence_floor <= match.end() <= limit and outside[match.end()]
    ]
    if sentence_ends:
        end = sentence_ends[-1]
    else:
        word_ends = [
            match.start()
            for match in _WORD_BOUNDARY.finditer(text, 0, limit + 1)
            if outside[match.start()]
        ]
        if not word_ends:
            word_ends = [
                match.start()
                for match in _WORD_BOUNDARY.finditer(text, 0, limit + 1)
            ]
        end = word_ends[-1] if word_ends else limit
    return f"{text[:end].rstrip()}…"


def build_candidate_note(
    decision: OpusDecision,
    search_string: SearchString,
    *,
    now: datetime,
    max_chars: int | None = None,
    evidence_max_chars: int | None = None,
) -> str:
    """Render the evaluation itself — bounded, mention-safe, no side effects.

    The note is the evaluation and nothing else: the rationale as the first
    paragraph, the capability evidence as a second one when there is evidence.
    No authorship marker, no date, no string id, no labels — an operator must
    not be able to tell an agent wrote it. `search_string` and `now` stay in
    the signature because the caller contract passes them; neither reaches the
    rendered text.

    An empty judgment renders an empty string, and the caller then writes no
    note at all.
    """
    limit = config.LINKEDIN_NOTE_MAX_CHARS if max_chars is None else max_chars
    evidence_limit = (
        config.LINKEDIN_NOTE_EVIDENCE_MAX_CHARS
        if evidence_max_chars is None
        else evidence_max_chars
    )

    rationale = _sanitize(getattr(decision, "rationale", ""))
    semantic_evidence = getattr(decision, "semantic_evidence", None) or {}
    evidence = _truncate_evidence(
        _sanitize(semantic_evidence.get("capability_evidence")),
        evidence_limit,
    )

    if evidence:
        separator = "\n\n" if rationale else ""
        # Evidence shrinks before the rationale: the judgment's own reasoning
        # is what the operator is reading for.
        evidence = _truncate_evidence(
            evidence,
            limit - len(rationale) - len(separator),
        )
        if evidence:
            return f"{rationale}{separator}{evidence}"

    return _truncate_at_word(rationale, limit)
