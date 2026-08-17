"""The experience floor reaches FACIAL triage, and reaches it as a prior.

Until 2026-07-27 `_experience_bar_line` was rendered only at the two
FULL_EVALUATION_TEMPLATE sites. Facial — the stage that exists to decide what
NOT to spend a profile open on — never saw the floor, so a candidate years short
could clear facial on vocabulary signal and burn the expensive read before any
floor applied.

The fix must stay ADVISORY. A hard facial floor would delete the population this
kind of role most wants: visible career span understates real experience for
anyone with an advanced degree, a research career, or omitted early roles. Sam's
framing (2026-07-27): six years plus a standout track record should reach the
full read. Several tests below lock that property specifically, because a
"tighten it up" edit that turns the prior into a gate would otherwise pass.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from linkedin import judgment_templates as jt
from linkedin.judgment_templates import _facial_experience_floor_line
from shared.brief_loader import _load_v2_brief
from shared.storage import read_json

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "config" / "senior-backend-fintech-fixture" / "brief.json"


@pytest.fixture()
def brief():
    """The v2 schema object the templates actually format against."""
    return _load_v2_brief(read_json(str(FIXTURE)))._new_brief


def _facial_renderings(brief) -> dict[str, str]:
    return {
        "facial_system": jt.assemble_facial_system(brief),
        "facial_batch_system": jt.assemble_facial_batch_system(brief),
        "facial_tool_single": jt.assemble_facial_tool_system(brief, batch=False),
        "facial_tool_batch": jt.assemble_facial_tool_system(brief, batch=True),
        "facial_prompt": jt.assemble_facial_prompt(brief, "SNIPPET"),
    }


def test_every_facial_path_carries_the_experience_prior(brief) -> None:
    assert brief.minimum_years_experience == 6
    for name, rendered in _facial_renderings(brief).items():
        assert "EXPERIENCE PRIOR" in rendered, name
        assert "6+ years" in rendered, name


def test_ternary_facial_templates_carry_it_too(brief, monkeypatch) -> None:
    # Template selection is per-brief; both branches must render, and a missing
    # format key raises KeyError rather than degrading quietly.
    monkeypatch.setattr(jt, "_facial_ternary_selected", lambda _b: True)

    assert "EXPERIENCE PRIOR" in jt.assemble_facial_system(brief)
    assert "EXPERIENCE PRIOR" in jt.assemble_facial_batch_system(brief)


def test_the_prior_is_advisory_not_a_gate(brief) -> None:
    block = _facial_experience_floor_line(brief)

    # The near-floor override is the whole point: it is what lets 6 years plus a
    # standout record reach the full read.
    assert "not a reason to reject" in block
    assert "standout track record" in block
    # And it must never license rejection on arithmetic.
    assert "Never reject on date arithmetic alone" in block
    for gate_word in ("automatic reject", "HARD", "regardless of other signals"):
        assert gate_word not in block, f"prior reads as a gate: {gate_word!r}"


def test_the_prior_still_kills_the_clearly_junior_snippet(brief) -> None:
    # The waste this exists to stop. Advisory must not mean toothless.
    block = _facial_experience_floor_line(brief)

    assert "FACIAL_NO" in block
    assert "half the floor" in block


def test_full_evaluation_keeps_its_own_bar_and_does_not_get_the_facial_block(
    brief,
) -> None:
    # Two instruments, two stages: the full read keeps the authoritative bar,
    # facial gets the prior. Collapsing them would put facial's softer language
    # in front of the stage that owns the decision.
    full = jt.assemble_full_evaluation_system(brief)

    assert "MINIMUM BAR" in full
    assert "EXPERIENCE PRIOR" not in full


@pytest.mark.parametrize("floor", [None, 0, -3, True, "7", 7.5])
def test_a_brief_without_a_usable_floor_renders_nothing(brief, floor) -> None:
    # bool is an int subclass — True must not render "1+ years".
    stub = copy.copy(brief)
    stub.minimum_years_experience = floor

    assert _facial_experience_floor_line(stub) == ""
    assert "EXPERIENCE PRIOR" not in jt.assemble_facial_system(stub)


def test_a_missing_attribute_is_not_an_error(brief) -> None:
    # Briefs load through several paths; one of them yields an object without
    # the attribute at all. Facial must still assemble.
    class Bare:
        pass

    assert _facial_experience_floor_line(Bare()) == ""
