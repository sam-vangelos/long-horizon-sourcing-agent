"""Behavioral transcript tests (Phase C10).

Drives five canonical conversation fixtures through the
extraction-merge-sufficiency pipeline and asserts:

1. Final ``v2_draft`` matches the per-fixture snapshot.
2. ``ready_to_compose`` flips correctly across the conversation.
3. Conversation length stays under the per-fixture bound.
4. No placeholder strings appear in the final ``v2_draft``.
5. Per-response voice properties hold for every Cloris turn (lexical
   tics, bullets, back-reference, spaced em-dashes, sentence count).
6. Per-conversation voice properties hold (no rhetorical-question
   transitions, no throat-clearing openers, no register slip).

The fixture format is a JSONL where each line is one of:

- ``{"kind": "meta", "scenario": "...", "expected_ready_to_compose": bool,
   "expected_max_cloris_turns": int, "source_packet": {...}?}`` — must
   be the FIRST line.
- ``{"kind": "cloris", "content": "..."}`` — a scripted Cloris turn.
- ``{"kind": "recruiter", "content": "..."}`` — a recruiter turn.
- ``{"kind": "extraction", "updates": {...}}`` — a scripted slot
  extraction that runs after the prior Cloris turn (mimicking the C5
  endpoint's flow).

Cloris turns are hand-written so the voice asserts have a known
substrate. Replacing them when the orchestrator prompt evolves means
generating fresh transcripts manually + re-snapshotting; ``v0`` is
deliberate scripting rather than recorded LLM output to keep the
suite offline-runnable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_intelligence.brief_distillation import _looks_like_placeholder
from shared.intake_conversation.state import merge_extracted
from shared.intake_conversation.sufficiency import is_ready_to_compose
from tests.intake_conversation.voice_asserts import (
    assert_back_reference,
    assert_no_brief_dump_shape,
    assert_no_bullets,
    assert_no_lexical_tics,
    assert_no_register_slip,
    assert_no_rhetorical_question_transitions,
    assert_no_throat_clearing_openers,
    assert_sentence_count,
    assert_spaced_em_dashes_only,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "conversation_transcripts"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _all_fixtures() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.jsonl"))


@pytest.mark.parametrize(
    "fixture_path",
    _all_fixtures(),
    ids=lambda p: p.stem,
)
def test_transcript_fixture(fixture_path: Path) -> None:
    """Drive one canonical transcript through the pipeline and pin all
    behavioral + voice contracts.

    Failures here indicate either a voice regression (Cloris turn
    drifted into a forbidden pattern), an extraction-merge regression
    (final v2_draft no longer matches the snapshot), or a sufficiency
    regression (ready_to_compose threshold moved).
    """

    rows = _load_jsonl(fixture_path)
    assert rows, f"empty fixture {fixture_path}"

    meta = rows[0]
    assert meta["kind"] == "meta", "first line of fixture must be meta"
    expected_ready: bool = bool(meta.get("expected_ready_to_compose", True))
    expected_max_cloris_turns: int = int(meta.get("expected_max_cloris_turns", 12))
    source_packet = meta.get("source_packet")

    messages: list[dict] = []
    v2_draft: dict = {}
    ready_history: list[bool] = [False]
    cloris_turn_count = 0

    for row in rows[1:]:
        kind = row.get("kind")
        if kind == "cloris":
            content = row["content"]
            cloris_turn_count += 1

            # Per-response voice asserts. The fixture's Cloris turns
            # are hand-written so a regression here is a fixture-level
            # bug, not a model regression — but the helpers ensure the
            # gold-standard transcripts themselves still embody the
            # voice contract.
            assert_no_lexical_tics(content)
            assert_no_brief_dump_shape(content)
            assert_no_bullets(content)
            assert_back_reference(content, messages, source_packet)
            assert_spaced_em_dashes_only(content)
            assert_sentence_count(content, max_sentences=5)

            messages.append(
                {
                    "role": "cloris",
                    "content": content,
                    "ts": f"2026-05-13T12:00:{cloris_turn_count:02d}+00:00",
                }
            )
        elif kind == "recruiter":
            messages.append(
                {
                    "role": "recruiter",
                    "content": row["content"],
                    "ts": f"2026-05-13T12:00:{len(messages):02d}+00:00",
                }
            )
        elif kind == "extraction":
            updates = row.get("updates", {})
            v2_draft = merge_extracted(
                v2_draft, updates, manually_edited_keys=set()
            )
            ready, _ = is_ready_to_compose(v2_draft)
            ready_history.append(ready)
        else:
            raise AssertionError(f"unknown row kind {kind!r} in {fixture_path}")

    # Per-conversation voice asserts.
    assert_no_rhetorical_question_transitions(messages)
    assert_no_throat_clearing_openers(messages)
    assert_no_register_slip(messages)

    # Conversation length bound.
    assert cloris_turn_count <= expected_max_cloris_turns, (
        f"{fixture_path.name}: {cloris_turn_count} Cloris turns exceeds "
        f"expected_max_cloris_turns={expected_max_cloris_turns}"
    )

    # Final v2_draft snapshot match.
    snapshot_path = fixture_path.with_suffix(".expected_v2_draft.json")
    if snapshot_path.exists():
        expected_draft = json.loads(snapshot_path.read_text())
        assert v2_draft == expected_draft, (
            f"{fixture_path.name}: final v2_draft does not match snapshot at "
            f"{snapshot_path.name}"
        )

    # No placeholder strings anywhere in the final draft.
    for s in _walk_strings(v2_draft):
        assert not _looks_like_placeholder(s), (
            f"{fixture_path.name}: placeholder string {s!r} survived to final "
            f"v2_draft"
        )

    # ready_to_compose flips correctly.
    assert ready_history[-1] == expected_ready, (
        f"{fixture_path.name}: final ready_to_compose={ready_history[-1]}, "
        f"expected {expected_ready}"
    )
