"""Tests for the 2026-05-13 distillation redesign: refuse-to-compose
when inputs are placeholder-shaped or missing, with explicit deficits.

Before this redesign `_heuristic_distill` slot-filled three paragraphs
from whatever was in role_title / role_summary / capability_areas /
depth_distinction — including the hardcoded placeholder strings that
`shared.source_packet_synthesis` used to seed by default. The output
was "Cloris would run this as Northwind is seeking a Tax Associate to work
within the Accounting Team… JOB DESCRIPTION Northwind is seeking a Tax
Associate…" — readable gibberish in front of recruiters.

The new contract: empty prose plus a `deficits` list when inputs are
missing or look like placeholders. Clean inputs still produce real
connective prose.
"""

from market_intelligence.brief_distillation import (
    _heuristic_distill,
    _looks_like_placeholder,
    PLACEHOLDER_STRINGS,
)


def test_empty_v2_returns_empty_prose_and_required_deficits() -> None:
    # Hard requirements: role_title + capability_areas. Empty
    # depth_distinction is fine (not flagged) — depth only triggers
    # a deficit when its fields are placeholder-shaped, not just empty.
    result = _heuristic_distill({})
    assert result["prose"] == ""
    deficits = result["deficits"]
    assert "role_title" in deficits
    assert "capability_areas" in deficits
    # depth_distinction NOT flagged when bare-empty.
    assert "depth_distinction" not in deficits


def test_placeholder_role_title_flagged_and_refuses_to_compose() -> None:
    polluted = {
        "role_title": (
            "Northwind is seeking a Tax Associate to work within the Accounting "
            "Team and assist with all tax related matters. You will be "
            "responsible for liaising with external tax consultants."
        ),
        "capability_areas": [
            {"name": "Core role scope", "description": "Responsibilities include but are not limited to:"}
        ],
        "depth_distinction": {
            "builder_definition": "",
            "user_definition": "",
            "edge_case_guidance": (
                "When the source packet is thin, keep the profile in review "
                "rather than inventing evidence."
            ),
        },
    }
    result = _heuristic_distill(polluted)
    assert result["prose"] == ""
    assert "role_title" in result["deficits"]
    assert "capability_areas" in result["deficits"]
    assert "depth_distinction" in result["deficits"]
    # Each placeholder field is reported by name so the UI can flag them.
    assert "role_title" in result["placeholder_fields"]
    assert any("capability_areas" in p for p in result["placeholder_fields"])
    assert "depth_distinction.edge_case_guidance" in result["placeholder_fields"]


def test_clean_v2_produces_real_connective_prose() -> None:
    clean = {
        "role_title": "Tax Associate",
        "role_summary": (
            "Owns sales tax compliance end-to-end and backs up the "
            "Director of Tax on partnership and corporate work."
        ),
        "capability_areas": [
            {
                "name": "Multi-state sales tax",
                "description": "Files returns across all 50 states; has done at least one Avalara implementation.",
            },
            {
                "name": "Partnership tax mechanics",
                "description": "K-1 prep, basis tracking, allocations on film LP structures.",
            },
        ],
        "depth_distinction": {
            "builder_definition": "Has prepared and filed returns themselves, not just supervised them.",
            "user_definition": "Tax memo writers without recent return-prep experience.",
            "edge_case_guidance": "Watch for managers who can describe compliance but haven't touched it in 2+ years.",
        },
    }
    result = _heuristic_distill(clean)
    assert result["deficits"] == []
    assert result["placeholder_fields"] == []
    prose = result["prose"]
    # The role + summary appear in the opening.
    assert "Tax Associate" in prose
    # The capability_area descriptions are surfaced.
    assert "Avalara" in prose
    # Depth distinction renders with recruiter-language labels (not "builder definition" jargon).
    assert "What they need to be able to do" in prose
    assert "Where to be careful" in prose
    # No service-idiom opening.
    assert "Cloris would run this as" not in prose


def test_looks_like_placeholder_catches_known_defaults() -> None:
    for s in PLACEHOLDER_STRINGS:
        assert _looks_like_placeholder(s), f"should flag known placeholder {s!r}"


def test_looks_like_placeholder_passes_clean_short_titles() -> None:
    assert not _looks_like_placeholder("Tax Associate", kind="role_title")
    assert not _looks_like_placeholder("Senior FDE", kind="role_title")


def test_looks_like_placeholder_flags_jd_paragraph_as_title() -> None:
    paragraph = (
        "Northwind is seeking a Tax Associate to work within the Accounting "
        "Team and assist with all tax related matters."
    )
    assert _looks_like_placeholder(paragraph, kind="role_title")


def test_looks_like_placeholder_allows_long_summaries() -> None:
    # role_summary can legitimately be a long paragraph; the kind=None
    # default path only catches exact PLACEHOLDER_STRINGS, not "long".
    summary = (
        "Owns sales tax compliance end-to-end and backs up the Director of Tax "
        "on partnership and corporate work, with a 6-month migration off "
        "spreadsheets onto Avalara as the first concrete deliverable."
    )
    assert not _looks_like_placeholder(summary)


def test_mixed_clean_and_placeholder_caps_filters_only_real() -> None:
    mixed = {
        "role_title": "Tax Associate",
        "role_summary": "Owns sales tax.",
        "capability_areas": [
            {"name": "Multi-state sales tax", "description": "Files returns across 50 states."},
            {"name": "Core role scope", "description": "Responsibilities include but are not limited to:"},
        ],
        "depth_distinction": {
            "builder_definition": "Has prepared and filed returns.",
            "user_definition": "",
            "edge_case_guidance": "",
        },
    }
    result = _heuristic_distill(mixed)
    # The clean capability appears, the placeholder is dropped, prose composes.
    assert result["deficits"] == []
    assert "Multi-state sales tax" in result["prose"]
    assert "Core role scope" not in result["prose"]
