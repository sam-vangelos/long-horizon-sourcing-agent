"""A resumed run judges with the seed's operator-owned experience and level fields.

A preflight-born LinkedIn run resumes on the state dir's frozen
``preflight_v2_brief.json`` (linkedin/evaluation_loop.py, the resume reload).
Before this overlay the only seed field that reached a resumed run was
``alumni_blacklist``, so an operator ruling written into the seed after the
campaign started (a hard experience cap added to a field-engineering seed)
never reached the judge: the frozen brief kept an advisory ceiling, an
experience measure that counted research years, and instructions saying
years are "not a hard ceiling".

The fixture is a synthetic frozen brief with the shape and advisory wording of
a real preflight-generated one. The seed here is a compact stand-in shaped
like a seed after such a ruling.
"""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

FROZEN_BRIEF = (
    Path(__file__).parent
    / "fixtures"
    / "field_engineering_resume"
    / "preflight_v2_brief.json"
)

# The frozen brief's advisory wording. None of it may reach the judge once the
# seed carries the hard cap.
CONTRADICTIONS = (
    "not a hard ceiling",
    "never alone a reject",
    "never alone a reason to reject",
)

OPERATOR_KEYS = (
    "minimum_years_experience",
    "maximum_years_experience",
    "maximum_years_experience_is_hard",
    "experience_measure",
    "role_level",
    "minimum_bar_description",
    "instructions",
)

# A plain label: shared.brief_schema.Brief.is_senior_role substring-matches
# DIRECTOR / VP / HEAD / EXECUTIVE / L7-L14 in role_level and, on a hit, adds
# the executive calibration blocks to every full judgment. The reject titles
# live in minimum_bar_description and the instructions instead.
ROLE_LEVEL = "Early-career individual contributor (about 1-4 years of full-time work)"
# Headers of the blocks is_senior_role() switches on (shared/brief_schema.py).
SENIOR_BLOCK_MARKERS = ("SENIORITY CALIBRATION (", "EXECUTIVE BUILDER CALIBRATION", "L7+")
MINIMUM_BAR = (
    "Three things together: (1) a customer-facing technical role; (2) hands-on "
    "Python and LLM or model-API work; and (3) the level, which is a hard gate: "
    "an individual contributor with about 1 to 4 years of full-time "
    "professional work, counted from the first full-time role."
)
MEASURE = (
    "Years of full-time professional work, counted from the start of the first "
    "full-time role; internships, co-ops, and student, research-assistant, or "
    "teaching-assistant roles do not count."
)
INSTRUCTIONS = [
    "THE TARGET: a pre-sales, solutions, or field engineer, the technical person "
    "in customer conversations at an AI, ML, data, or developer-tools company.",
    "EXPERIENCE CEILING (operator ruling 2026-10-01): about 4 years of full-time "
    "professional work is a hard cap, counted from the first full-time role. "
    "Internships, co-ops, and student, research-assistant, or teaching-assistant "
    "roles are excluded from the count. A current Senior, Lead, Staff, "
    "Principal, Head-of, Director, or people-manager title is a REJECT.",
    "Excluded employers: candidates CURRENTLY employed at OpenAI or Anthropic "
    "are never leads; Babbage is the hiring company and is never sourced.",
]


def _hard_cap_seed(**overrides) -> dict:
    """A legacy-shaped seed (needs preflight) carrying the operator's hard cap."""

    seed = {
        "name": "field-engineering-test-seed",
        "role_title": "AI Solutions Engineer, Field Engineering",
        "linkedin_project": "Test project",
        # Matches the harness's project-scoped search URL, so run start
        # treats the page as this brief's own and never navigates.
        "linkedin_project_id": "test-project",
        "kit_url": "",
        "permanent_filters": {},
        "alumni_blacklist": ["Babbage"],
        "employer_blacklist": ["Babbage", "OpenAI", "Anthropic"],
        "instructions": list(INSTRUCTIONS),
        "minimum_years_experience": 1,
        "maximum_years_experience": 2,
        "maximum_years_experience_is_hard": True,
        "experience_measure": MEASURE,
        "role_level": ROLE_LEVEL,
        "minimum_bar_description": MINIMUM_BAR,
        "intake_notes": "Early-career to mid-level individual contributor.",
        "jd": "The Role\n\nBabbage is hiring an AI Solutions Engineer.",
    }
    seed.update(overrides)
    return seed


def _frozen() -> dict:
    return json.loads(FROZEN_BRIEF.read_text())


class _Stop(Exception):
    """Ends the run right after the reload, before any browser work."""


def _resume_through_the_evaluation_loop(seed_raw: dict, frozen: dict):
    """Drive ``run_full(resume=True)`` through the real resume reload.

    Returns the brief object the run holds after the reload — the object every
    judge call reads (candidate_review passes ``context.brief_obj``).
    """

    from linkedin.evaluation_loop import EvaluationLoop
    from shared.brief_loader import normalize_brief
    from shared.schemas import Progress, SearchString
    from tests.test_linkedin_pipeline import _PROJECT_SEARCH_URL, _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.state.context.brief_obj = normalize_brief(copy.deepcopy(seed_raw))
        assert p.state.context.brief_obj.needs_preflight()
        (Path(td) / "preflight_v2_brief.json").write_text(json.dumps(frozen))
        Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1, name="resume owner", boolean="one", status="in_progress"
                )
            ],
        ).save(str(p.state.progress_path))
        p.state.context.browser.connect = AsyncMock()
        p.state.context.browser.disconnect = AsyncMock()
        p.state.context.browser.page = MagicMock(url=_PROJECT_SEARCH_URL)
        p.state.surface._get_project_url = lambda: ""
        p.state.surface._apply_session_location_filter = AsyncMock()
        p.state._run_report_service._print_summary = lambda: None
        p.state._run_report_service._generate_run_report = lambda progress: None

        with patch("linkedin.evaluation_loop.init_judger"), patch.object(
            EvaluationLoop,
            "_record_judgment_runtime_profile_event",
            side_effect=_Stop,
        ):
            with pytest.raises(_Stop):
                asyncio.run(p.run_full(resume=True))
        return p.state.context.brief_obj


def _judge_system_prompt(brief, contract_mode: str, monkeypatch) -> str:
    """The full-evaluation system prompt ``full_judge`` sends for ``brief``."""

    from shared import config, judger
    from shared.schemas import CandidateProfileSummary, Experience

    monkeypatch.setattr(config, "LINKEDIN_V2_FULL_CONTRACT", contract_mode)
    monkeypatch.setattr(config, "FIREWORKS_JUDGMENT_POLICY_ENABLED", False)
    captured: dict[str, str] = {}

    def _capture(system, user_message, **_kwargs):
        captured["system"] = system
        raise _Stop

    summary = CandidateProfileSummary(
        name="Test Candidate",
        profile_url="/talent/profile/test",
        headline="Forward Deployed Engineer",
        experiences=[
            Experience(
                title="Forward Deployed Engineer",
                company="Example AI",
                start="2019",
                end="Present",
                summary_bullets=["Runs customer pilots in Python."],
            )
        ],
    )
    with patch("shared.judger.opus_llm_cached", side_effect=_capture):
        try:
            judger.full_judge(summary, brief=brief)
        except _Stop:
            pass
    assert "system" in captured, "full_judge never reached the model call"
    return captured["system"]


# ---------------------------------------------------------------- the red test


@pytest.mark.parametrize("contract_mode", ["legacy", "tool"])
def test_resumed_run_judges_with_the_seeds_hard_level_cap(
    contract_mode, monkeypatch, capsys
) -> None:
    brief = _resume_through_the_evaluation_loop(_hard_cap_seed(), _frozen())
    console = capsys.readouterr().out

    assert brief.has_v2_schema, "resume did not reload the generated V2 brief"
    system = _judge_system_prompt(brief, contract_mode, monkeypatch)

    assert "HARD LEVELING GATE" in system
    assert "EXPERIENCE BAND: 1-2 years" in system
    assert f"Years are measured as: {MEASURE}" in system
    assert f"({ROLE_LEVEL})" in system, "the seed's role_level is not the ROLE line"
    assert MINIMUM_BAR in system
    assert INSTRUCTIONS[1] in system
    for phrase in CONTRADICTIONS:
        assert phrase not in system, phrase
    # The frozen brief's advisory measure and level are gone, not appended to.
    assert "including research and advanced-degree years" not in system
    assert "ADVISORY LEVELING" not in system

    # One prebuilt console line names what the seed replaced.
    overlay_lines = [line for line in console.splitlines() if line.startswith("  [seed]")]
    assert len(overlay_lines) == 1, console
    for key in OPERATOR_KEYS:
        assert key in overlay_lines[0], key
    # The reload line must not claim the original criteria after the seed
    # replaced them.
    assert "Reloaded generated V2 brief from state dir" in console
    assert "same criteria as the original run" not in console
    assert "criteria updated from the seed" in console

    # The level label must not switch on the executive calibration.
    assert brief._new_brief.is_senior_role() is False
    for marker in SENIOR_BLOCK_MARKERS:
        assert marker not in system, marker


def test_a_resume_the_seed_changes_nothing_still_says_same_criteria(capsys) -> None:
    seed_raw = _hard_cap_seed()
    for key in OPERATOR_KEYS:
        seed_raw.pop(key)
    seed_raw.pop("alumni_blacklist")
    _resume_through_the_evaluation_loop(seed_raw, _frozen())
    console = capsys.readouterr().out

    assert (
        "  Reloaded generated V2 brief from state dir "
        "(same criteria as the original run)"
    ) in console
    assert "[seed]" not in console and "criteria updated" not in console


@pytest.mark.parametrize(
    "key,bad",
    [
        ("maximum_years_experience_is_hard", "true"),
        ("maximum_years_experience_is_hard", 1),
        ("maximum_years_experience", True),
        ("maximum_years_experience", "2"),
        ("minimum_years_experience", 1.5),
        ("minimum_years_experience", None),
        ("role_level", ["Early-career"]),
        ("experience_measure", 4),
        ("minimum_bar_description", {"text": "x"}),
    ],
)
def test_a_malformed_seed_field_is_skipped_and_named(key, bad, capsys) -> None:
    """Types match what shared.brief_loader._load_v2_brief accepts; anything
    else would load as a silent default (a string "true" hardness reads as
    False). The malformed key keeps the frozen value and is named on the one
    [seed] line."""

    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief, preflight_overrides_from_brief

    frozen = _frozen()
    seed = normalize_brief(_hard_cap_seed(**{key: bad}))
    merged = overlay_seed_onto_resumed_brief(copy.deepcopy(frozen), seed)
    lines = [line for line in capsys.readouterr().out.splitlines() if "[seed]" in line]

    assert merged[key] == frozen[key]
    assert len(lines) == 1, lines
    assert f"skipped as malformed: {key}" in lines[0]
    # The well-formed keys still apply.
    others = [k for k in OPERATOR_KEYS if k != key]
    for other in others:
        assert merged[other] == _hard_cap_seed()[other], other
    # A fresh run skips it too.
    assert key not in preflight_overrides_from_brief(seed, {}, geography="")


def test_a_null_maximum_is_a_deliberate_no_ceiling_not_malformed(capsys) -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

    merged = overlay_seed_onto_resumed_brief(
        _frozen(),
        normalize_brief(
            _hard_cap_seed(maximum_years_experience=None, maximum_years_experience_is_hard=False)
        ),
    )

    assert merged["maximum_years_experience"] is None
    assert "malformed" not in capsys.readouterr().out


def test_only_malformed_fields_still_print_one_line(capsys) -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

    frozen = _frozen()
    seed_raw = _hard_cap_seed()
    for key in OPERATOR_KEYS:
        seed_raw.pop(key)
    seed_raw.pop("alumni_blacklist")
    seed_raw["maximum_years_experience_is_hard"] = "yes"
    merged = overlay_seed_onto_resumed_brief(copy.deepcopy(frozen), normalize_brief(seed_raw))
    out = capsys.readouterr().out

    assert merged == frozen
    assert out.splitlines() == [
        "  [seed] no operator field from the seed applied; skipped as malformed: "
        "maximum_years_experience_is_hard"
    ]


# ------------------------------------------------------------ overlay contract


def test_a_seed_without_operator_fields_resumes_on_the_frozen_brief_unchanged(
    capsys,
) -> None:
    """Every other campaign's resume is byte-for-byte what it was."""

    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

    seed_raw = _hard_cap_seed()
    for key in OPERATOR_KEYS:
        seed_raw.pop(key)
    # The frozen brief predates alumni_blacklist; that overlay is pre-existing
    # behaviour with its own test below.
    seed_raw.pop("alumni_blacklist")
    frozen = _frozen()
    merged = overlay_seed_onto_resumed_brief(
        copy.deepcopy(frozen), normalize_brief(seed_raw)
    )

    assert merged == frozen
    assert capsys.readouterr().out == ""


def test_only_keys_the_seed_carries_overlay_and_only_changes_are_named() -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_operator_fields

    frozen = _frozen()
    seed_raw = _hard_cap_seed()
    for key in OPERATOR_KEYS:
        seed_raw.pop(key)
    seed_raw["maximum_years_experience_is_hard"] = True
    # Same instructions as the frozen brief: present, but not a change.
    seed_raw["instructions"] = list(frozen["instructions"])

    merged, changed = overlay_seed_operator_fields(
        copy.deepcopy(frozen), normalize_brief(seed_raw)
    )

    assert changed == ["maximum_years_experience_is_hard"]
    assert merged["maximum_years_experience_is_hard"] is True
    for key in OPERATOR_KEYS:
        if key != "maximum_years_experience_is_hard":
            assert merged[key] == frozen[key], key
    # Idempotent: a second pass changes nothing.
    again, changed_again = overlay_seed_operator_fields(merged, normalize_brief(seed_raw))
    assert again == merged and changed_again == []


def test_instructions_are_never_added_to_a_brief_generated_without_them(capsys) -> None:
    """Briefs generated before finalize_preflight_v2 copied the seed's
    instructions carry none, and their runs never judged with them (nine live
    campaigns' state dirs on 2026-10-01). Seed instructions must not appear on
    their resume; the experience fields still apply."""

    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief, overlay_seed_operator_fields

    frozen = _frozen()
    frozen.pop("instructions")
    merged, changed = overlay_seed_operator_fields(
        copy.deepcopy(frozen), normalize_brief(_hard_cap_seed())
    )

    assert "instructions" not in merged
    assert "instructions" not in changed
    assert merged["maximum_years_experience_is_hard"] is True

    # An instructions-only seed resumes such a brief byte-for-byte unchanged.
    seed_raw = _hard_cap_seed()
    for key in OPERATOR_KEYS:
        if key != "instructions":
            seed_raw.pop(key)
    seed_raw.pop("alumni_blacklist")
    assert overlay_seed_onto_resumed_brief(
        copy.deepcopy(frozen), normalize_brief(seed_raw)
    ) == frozen
    assert capsys.readouterr().out == ""


def test_the_overlay_never_aliases_the_seeds_lists() -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_operator_fields

    seed = normalize_brief(_hard_cap_seed())
    merged, _changed = overlay_seed_operator_fields(_frozen(), seed)
    merged["instructions"].append("mutated")

    assert seed.raw["instructions"] == INSTRUCTIONS


def test_a_brief_with_no_raw_dict_overlays_nothing() -> None:
    """Only the legacy seed carries operator fields in ``raw``; a mock or any
    object without a dict ``raw`` is a no-op, never a crash."""

    from shared.preflight_v2 import overlay_seed_operator_fields

    frozen = _frozen()
    merged, changed = overlay_seed_operator_fields(copy.deepcopy(frozen), MagicMock())

    assert merged == frozen and changed == []


def test_the_alumni_overlay_still_applies_and_prints_its_own_line(capsys) -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

    frozen = _frozen()
    frozen.pop("alumni_blacklist", None)
    seed_raw = _hard_cap_seed()
    for key in OPERATOR_KEYS:
        seed_raw.pop(key)
    merged = overlay_seed_onto_resumed_brief(frozen, normalize_brief(seed_raw))

    assert merged["alumni_blacklist"] == ["Babbage"]
    assert capsys.readouterr().out == (
        "  [alumni] seed alumni_blacklist applied to the resumed brief: Babbage\n"
    )


def test_resume_lint_prints_one_line_and_never_blocks(capsys) -> None:
    """A hard cap the instructions never authorize is a lint ERROR, printed on
    one line; the resume still proceeds on the merged brief."""

    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

    unauthorized = [item for item in INSTRUCTIONS if "EXPERIENCE CEILING" not in item]
    merged = overlay_seed_onto_resumed_brief(
        _frozen(), normalize_brief(_hard_cap_seed(instructions=unauthorized))
    )
    lines = capsys.readouterr().out.splitlines()
    lint_lines = [line for line in lines if line.startswith("  [preflight-lint]")]

    assert merged["maximum_years_experience_is_hard"] is True
    assert len(lint_lines) == 1, lines
    assert "hard_experience_ceiling_without_operator_authorization" in lint_lines[0]
    assert "experience_measure_unstated_qualifier" in lint_lines[0]


def test_an_authorized_hard_cap_lints_without_errors(capsys) -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

    overlay_seed_onto_resumed_brief(_frozen(), normalize_brief(_hard_cap_seed()))
    lines = capsys.readouterr().out.splitlines()
    lint_lines = [line for line in lines if line.startswith("  [preflight-lint]")]

    # The frozen brief carries two pre-existing warnings of its own; the
    # overlaid fields add nothing.
    assert len(lint_lines) == 1, lines
    assert "ERROR" not in lint_lines[0] and "error" not in lint_lines[0]
    assert "hard_experience_ceiling" not in lint_lines[0]
    assert "experience_measure_unstated_qualifier" not in lint_lines[0]


def test_a_lint_crash_never_blocks_the_resume(capsys) -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

    with patch("shared.brief_lint.lint_generated_brief", side_effect=ValueError("boom")):
        merged = overlay_seed_onto_resumed_brief(
            _frozen(), normalize_brief(_hard_cap_seed())
        )
    out = capsys.readouterr().out

    assert merged["maximum_years_experience"] == 2
    assert "  [preflight-lint] resumed brief: lint did not run (ValueError: boom)" in out


# ------------------------------------------------------------------ fresh run


def test_a_fresh_run_applies_the_seed_operator_fields_after_generation() -> None:
    from shared.brief_loader import normalize_brief
    from shared.judgment.templates import assemble_full_evaluation_system
    from shared.preflight_v2 import (
        PreflightV2Generation,
        finalize_preflight_v2,
        preflight_overrides_from_brief,
    )

    seed = normalize_brief(_hard_cap_seed())
    overrides = preflight_overrides_from_brief(seed, {}, geography="")
    for key in OPERATOR_KEYS:
        assert overrides[key] == _hard_cap_seed()[key], key

    generated = _frozen()
    for key in ("instructions", "alumni_blacklist", "provenance"):
        generated.pop(key, None)
    execution = finalize_preflight_v2(
        seed,
        PreflightV2Generation(
            system_prompt="",
            user_prompt="",
            raw_response="",
            data=generated,
            findings=(),
            usage_context={},
            source_label="JD",
        ),
    )
    for key in OPERATOR_KEYS:
        assert execution.brief_json[key] == _hard_cap_seed()[key], key
    system = assemble_full_evaluation_system(execution.brief._new_brief)
    assert "HARD LEVELING GATE" in system
    assert "EXPERIENCE BAND: 1-2 years" in system
    for phrase in CONTRADICTIONS:
        assert phrase not in system, phrase


def test_fresh_overrides_without_operator_fields_are_unchanged() -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import preflight_overrides_from_brief

    seed_raw = _hard_cap_seed()
    for key in OPERATOR_KEYS:
        seed_raw.pop(key)
    overrides = preflight_overrides_from_brief(
        normalize_brief(seed_raw), {}, geography=""
    )

    assert not set(OPERATOR_KEYS) & set(overrides)


# ----------------------------------------------------------------- shadow replay


def test_shadow_replay_judges_the_resumed_view_when_given_the_seed(
    tmp_path, capsys
) -> None:
    from shared.brief_loader import load_brief
    from tools.shadow_replay import seed_overlaid_v2_brief

    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps(_hard_cap_seed()))
    out_dir = tmp_path / "replay"
    out_dir.mkdir()

    path = seed_overlaid_v2_brief(FROZEN_BRIEF, seed_path, out_dir)

    assert path.parent == out_dir and path != FROZEN_BRIEF
    assert json.loads(FROZEN_BRIEF.read_text()) == _frozen(), "the input was edited"
    brief = load_brief(str(path))
    assert brief._new_brief.maximum_years_experience == 2
    assert brief._new_brief.maximum_years_experience_is_hard is True
    assert brief._new_brief.role_level == ROLE_LEVEL
    assert "  [seed]" in capsys.readouterr().out


def test_shadow_replay_leaves_the_v2_brief_alone_for_a_v2_seed(tmp_path) -> None:
    """A V2-shaped seed never resumes through the overlay in production."""

    from tools.shadow_replay import seed_overlaid_v2_brief

    v2_seed = tmp_path / "v2_seed.json"
    v2_seed.write_text(json.dumps(_frozen()))

    assert seed_overlaid_v2_brief(FROZEN_BRIEF, v2_seed, tmp_path) == FROZEN_BRIEF
