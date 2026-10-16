"""Voice-property assertions for Cloris turns (Phase C9, used by C10).

Each helper raises :class:`AssertionError` when the input violates the
voice contract. The set is split into per-response checks (a single
Cloris turn) and per-conversation checks (the whole transcript).

Source of truth for the lexical banlist is
:data:`shared.intake_conversation.prompts.LEXICAL_TIC_BANLIST`. Both
the orchestrator system prompt and these helpers regex against the
same constant — change the list, both update. No drift.

Voice rules encoded:

- §VIII-1 stacked fragments: out-of-scope here (hard to detect cleanly
  without false positives); covered by manual review during the C12
  acid test instead.
- §VIII-2 rhetorical-question transitions ("What does this mean?"):
  :func:`assert_no_rhetorical_question_transitions`.
- §VIII-3 unspaced em-dashes: :func:`assert_spaced_em_dashes_only`.
- §VIII-4 announcing the reframe ("Crucially…"): folded into
  :func:`assert_no_lexical_tics`.
- §VIII-5 explaining-why-evidence-matters: out-of-scope for v0.
- §VIII-6 generic claims: :func:`assert_back_reference` (the
  specificity rule).
- §VIII-7 hedging: out-of-scope for v0.
- §VIII-8 register slip: :func:`assert_no_register_slip` (warning-level
  via a secondary banlist; false-positive prone).
- §VIII-9 bullet points: :func:`assert_no_bullets`.
- §VIII-10 throat-clearing openers: :func:`assert_no_throat_clearing_openers`.

Sentence count is bounded so Cloris turns don't drift into mini-essays
(the conversational surface should feel like a chat, not a memo).
"""

from __future__ import annotations

import re
from typing import Any

from shared.intake_conversation.prompts import (
    LEXICAL_TIC_BANLIST,
    LEXICAL_TIC_VERB_BANLIST,
)
from shared.intake_conversation.voice_contract import (
    looks_like_brief_dump,
    phrase_cooldown_violations,
    source_overlap_ratio,
)


# ---------------------------------------------------------------------
# Per-response helpers (single Cloris turn).
# ---------------------------------------------------------------------


def assert_no_lexical_tics(text: str) -> None:
    """Raise if ``text`` contains any banned lexical tic.

    Case-insensitive. Multi-word phrases are matched as substrings;
    single-word tics use word-boundary regex so substrings inside
    longer words don't false-positive ("substrate" doesn't match
    inside "substratosphere", though the banlist has no such word).
    The verb-only banlist (currently just "engage") matches as a
    whole word so the noun "engagement" passes.
    """

    lowered = text.lower()
    for tic in LEXICAL_TIC_BANLIST:
        if " " in tic or "-" in tic or "'" in tic:
            if tic in lowered:
                raise AssertionError(
                    f"Forbidden lexical tic {tic!r} found in: {text!r}"
                )
        else:
            pattern = r"\b" + re.escape(tic) + r"\b"
            if re.search(pattern, lowered):
                raise AssertionError(
                    f"Forbidden lexical tic {tic!r} found in: {text!r}"
                )

    # Verb-form banlist: match the bare verb form only ("engage", not
    # "engagement"). Crude — catches "engages" / "engaged" / "engaging"
    # via word-stem trailing-letter set; passes "engagement" because it
    # has no word boundary after "engage" matching the pattern.
    for verb in LEXICAL_TIC_VERB_BANLIST:
        pattern = r"\b" + re.escape(verb) + r"(s|d|ing)?\b"
        if re.search(pattern, lowered):
            raise AssertionError(
                f"Forbidden verb-form tic {verb!r} found in: {text!r}"
            )


def assert_no_bullets(text: str) -> None:
    """Raise if ``text`` contains a bulleted-list line.

    Detects lines starting with ``- ``, ``* ``, ``• ``, or numbered
    lists like ``1. ``. The conversational surface is prose, not a
    checklist (voice guide §VIII-9).
    """

    bullet_re = re.compile(r"(?m)^\s*(?:[-*•]|\d+\.)\s+\S")
    if bullet_re.search(text):
        raise AssertionError(
            f"Cloris turn contains bullet-list formatting: {text!r}"
        )


def assert_back_reference(
    text: str,
    prior_messages: list[dict[str, Any]],
    source_packet: dict[str, Any] | None,
) -> None:
    """Raise if a question-bearing Cloris turn lacks a back-reference.

    Implements the §V specificity rule (encoded as a hard behavioral
    constraint in :func:`shared.intake_conversation.prompts.build_orchestrator_system_prompt`).

    Heuristic:
    1. If the turn doesn't ask a question (no ``?``), pass — only
       question-bearing turns must back-reference.
    2. Extract grounding tokens from the prior 3 recruiter turns + the
       source packet's text fields. Tokens are: capitalized non-initial
       words (likely proper nouns), numbers, and 4+ char lowercased
       words that aren't common stopwords.
    3. Assert at least one grounding token (case-insensitive substring)
       appears in the Cloris turn.

    NOT a strict semantic check. A crude grounding signal that catches
    "tell me about the role" on turn 3 — the failure mode the §V rule
    exists to prevent.
    """

    if "?" not in text:
        return  # not a question turn, no back-reference required

    grounding_tokens = _grounding_tokens(prior_messages, source_packet)
    if not grounding_tokens:
        # No grounding context (e.g. opener turn with no source packet
        # and no prior recruiter turns). The specificity rule can't
        # apply — pass silently.
        return

    lowered = text.lower()
    for token in grounding_tokens:
        if token.lower() in lowered:
            return

    raise AssertionError(
        f"Cloris turn asks a question without back-referencing prior "
        f"context. Question turn: {text!r}. Grounding tokens available: "
        f"{sorted(grounding_tokens)[:20]}"
    )


def assert_spaced_em_dashes_only(text: str) -> None:
    """Raise if ``text`` contains an unspaced em-dash or en-dash.

    Voice guide §II-C / §VIII-3: always ``word — word``, never
    ``word—word``. The regex matches a non-whitespace character on
    either side of the dash; spaced dashes have whitespace adjacent.
    """

    if re.search(r"\S—\S|\S–\S", text):
        raise AssertionError(
            f"Cloris turn uses unspaced em/en-dash. "
            f"Always 'word — word'. Found in: {text!r}"
        )


def assert_sentence_count(
    text: str, *, max_sentences: int = 5, mean_max: float = 3.0
) -> None:
    """Raise when ``text`` is too long-winded for a chat turn.

    A "sentence" is a non-empty fragment terminated by ``.``, ``!``,
    or ``?``. ``max_sentences`` caps any single Cloris turn; the
    ``mean_max`` parameter is reserved for the per-conversation pass
    (transcript-level mean across Cloris turns) and isn't enforced
    here for a single-turn check.
    """

    sentences = _split_sentences(text)
    n = len(sentences)
    if n > max_sentences:
        raise AssertionError(
            f"Cloris turn has {n} sentences (max {max_sentences}). "
            f"Turn: {text!r}"
        )


# ---------------------------------------------------------------------
# Per-conversation helpers (whole transcript).
# ---------------------------------------------------------------------


def assert_no_rhetorical_question_transitions(
    messages: list[dict[str, Any]],
) -> None:
    """Raise if any Cloris turn opens with a rhetorical-question
    transition (voice guide §VIII-2).

    Patterns matched: "What does this mean", "What does that mean",
    "Why does this matter", "What's the takeaway", "What does X mean"
    when X is a short noun phrase.
    """

    pattern = re.compile(
        r"^\s*(?:so\s+)?what(?:'s|\s+(?:does|is|was))\s+(?:this|that|the\s+\w+)\s+(?:mean|matter)",
        re.IGNORECASE,
    )
    matter_pattern = re.compile(
        r"^\s*why\s+(?:does\s+)?(?:this|that)\s+matter",
        re.IGNORECASE,
    )

    for msg in messages:
        if msg.get("role") != "cloris":
            continue
        content = str(msg.get("content") or "")
        for sentence in _split_sentences(content):
            if pattern.match(sentence) or matter_pattern.match(sentence):
                raise AssertionError(
                    f"Cloris uses a rhetorical-question transition: "
                    f"{sentence!r}"
                )


def assert_no_throat_clearing_openers(
    messages: list[dict[str, Any]],
) -> None:
    """Raise if any Cloris turn opens with a throat-clearing phrase.

    Voice guide §VIII-10: start with the substance. Detects "Just to
    clarify", "It's worth noting", "I should mention", "Quick note",
    "Real quick", "I want to take a moment", and similar.
    """

    openers = (
        "just to clarify",
        "it's worth noting",
        "i should mention",
        "i want to mention",
        "quick note",
        "real quick",
        "i want to take a moment",
        "before we go further",
        "just so you know",
    )

    for msg in messages:
        if msg.get("role") != "cloris":
            continue
        content = str(msg.get("content") or "").strip().lower()
        for opener in openers:
            if content.startswith(opener):
                raise AssertionError(
                    f"Cloris turn opens with throat-clearing phrase "
                    f"{opener!r}. Turn: {msg.get('content')!r}"
                )


def assert_no_register_slip(messages: list[dict[str, Any]]) -> None:
    """Raise if any Cloris turn contains a corporate-cheery exclamation.

    Voice guide §VIII-8 + §IV: register matches the substance. The
    secondary banlist below catches the most common slip — analytical
    or warm-precise register sliding into SaaS-rep cheery
    interjections. False-positive prone (e.g. quoted recruiter speech),
    so callers should treat failures here as a warning to inspect the
    transcript rather than a hard ship-blocker until the heuristic
    stabilizes.
    """

    banlist = (
        "Awesome!",
        "Got it!",
        "Perfect!",
        "Sounds good!",
        "Amazing!",
        "Love it!",
    )

    for msg in messages:
        if msg.get("role") != "cloris":
            continue
        content = str(msg.get("content") or "")
        for phrase in banlist:
            if phrase in content:
                raise AssertionError(
                    f"Cloris turn contains register-slip exclamation "
                    f"{phrase!r}. Turn: {content!r}"
                )


def assert_phrase_cooldown(messages: list[dict[str, Any]]) -> None:
    """Raise when a cooldown phrase repeats sooner than allowed."""

    violations = phrase_cooldown_violations(messages)
    if violations:
        raise AssertionError(
            "Cloris violated phrase cooldowns: " + "; ".join(violations)
        )


def assert_no_brief_dump_shape(text: str) -> None:
    """Raise when a Cloris turn looks like a brief pasted into chat."""

    if looks_like_brief_dump(text):
        raise AssertionError(
            f"Cloris turn looks like a brief dump in chat: {text[:200]!r}..."
        )


def assert_source_question_not_redundant(
    text: str,
    source_packet: dict[str, Any] | None,
    *,
    max_overlap: float = 0.75,
) -> None:
    """Raise when a question mostly repeats source-packet wording.

    Implements the inference-first path for detailed JDs: if the overlap
    ratio exceeds ``max_overlap``, Cloris should have inferred the answer
    instead of asking.
    """

    if "?" not in text:
        return
    ratio = source_overlap_ratio(text, source_packet)
    if ratio > max_overlap:
        raise AssertionError(
            f"Cloris question overlaps source packet at {ratio:.0%} "
            f"(max {max_overlap:.0%}). Question: {text!r}"
        )


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_GROUNDING_STOPWORDS = frozenset(
    {
        "and",
        "the",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "they",
        "their",
        "would",
        "could",
        "should",
        "about",
        "into",
        "onto",
        "your",
        "yours",
        "ours",
        "what",
        "when",
        "where",
        "which",
        "while",
        "after",
        "before",
        "still",
        "going",
        "doing",
        "trying",
        "looking",
    }
)


def _split_sentences(text: str) -> list[str]:
    fragments = _SENTENCE_SPLIT_RE.split(text.strip())
    return [f for f in fragments if f.strip()]


def _grounding_tokens(
    prior_messages: list[dict[str, Any]],
    source_packet: dict[str, Any] | None,
) -> set[str]:
    """Extract a set of grounding-candidate tokens from prior context.

    Tokens are:
    - Capitalized words appearing mid-sentence (likely proper nouns).
    - Numbers (digits + percent signs).
    - Lowercase words >= 4 chars that aren't in _GROUNDING_STOPWORDS.

    Source is the most recent 3 recruiter turns (per the §V rule
    window) plus any string fields on the source_packet (JD body,
    intake notes, geography).
    """

    sources: list[str] = []

    recruiter_turns = [
        str(m.get("content") or "")
        for m in prior_messages
        if m.get("role") == "recruiter"
    ]
    sources.extend(recruiter_turns[-3:])

    if source_packet:
        for key, val in source_packet.items():
            if isinstance(val, str) and val.strip():
                sources.append(val)

    tokens: set[str] = set()
    for source in sources:
        # Capitalized non-initial words (proper nouns).
        for match in re.finditer(r"(?<=[\s.,;:!?\-—])([A-Z][a-zA-Z]{2,})", source):
            tokens.add(match.group(1))
        # Numbers / percentages.
        for match in re.finditer(r"\b(\d+(?:\.\d+)?%?)\b", source):
            tokens.add(match.group(1))
        # 4+ char lowercase words.
        for match in re.finditer(r"\b([a-z]{4,})\b", source.lower()):
            word = match.group(1)
            if word in _GROUNDING_STOPWORDS:
                continue
            tokens.add(word)

    return tokens


__all__ = [
    "assert_no_lexical_tics",
    "assert_no_bullets",
    "assert_back_reference",
    "assert_spaced_em_dashes_only",
    "assert_sentence_count",
    "assert_no_rhetorical_question_transitions",
    "assert_no_throat_clearing_openers",
    "assert_no_register_slip",
    "assert_phrase_cooldown",
    "assert_no_brief_dump_shape",
    "assert_source_question_not_redundant",
    "source_overlap_ratio",
]
