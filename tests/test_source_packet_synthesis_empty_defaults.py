"""Tests for the 2026-05-13 source-packet synthesis redesign: empty
defaults instead of placeholder strings.

Before, `_existing_or_default_capabilities` returned
`[{"name": "Core role scope", "description": <text>}]` and
`_existing_or_default_depth` seeded `edge_case_guidance` with a
system-prompt-style guardrail string. These looked like recruiter
input and propagated into the read-back as gibberish.

Now the defaults are empty — the review chapter renders an editorial
empty-state pointing the recruiter at fields they need to fill in.
"""

from shared.source_packet_synthesis import (
    _existing_or_default_capabilities,
    _existing_or_default_depth,
    _first_title,
    _first_paragraph,
    _heuristic_synthesize,
)


def test_capability_default_uses_extracted_description_with_empty_name() -> None:
    # Empty name + the caller-provided description (extracted from the
    # JD). Real starting content for the recruiter, no "Core role scope"
    # placeholder.
    out = _existing_or_default_capabilities({}, "Owns sales tax compliance.")
    assert out == [{"name": "", "description": "Owns sales tax compliance."}]


def test_capability_default_empty_description_yields_empty_stub() -> None:
    # When no JD content extracts cleanly, fall through to a fully
    # empty structural stub (still schema-valid).
    out = _existing_or_default_capabilities({}, "")
    assert out == [{"name": "", "description": ""}]


def test_capability_preserves_existing_real_areas() -> None:
    existing = {
        "capability_areas": [
            {"name": "Multi-state sales tax", "description": "Files returns across all 50 states."}
        ]
    }
    out = _existing_or_default_capabilities(existing, "ignored")
    assert out == [
        {"name": "Multi-state sales tax", "description": "Files returns across all 50 states."}
    ]


def test_depth_default_has_no_system_prompt_string() -> None:
    out = _existing_or_default_depth({})
    assert out == {
        "builder_definition": "",
        "user_definition": "",
        "edge_case_guidance": "",
    }
    # Specifically: the old guardrail string is gone.
    assert "When the source packet is thin" not in out["edge_case_guidance"]


def test_depth_preserves_existing_real_values() -> None:
    existing = {
        "depth_distinction": {
            "builder_definition": "Hands-on returns work.",
            "user_definition": "Memo writers.",
            "edge_case_guidance": "Watch for stale practitioners.",
        }
    }
    out = _existing_or_default_depth(existing)
    assert out["builder_definition"] == "Hands-on returns work."
    assert out["user_definition"] == "Memo writers."
    assert out["edge_case_guidance"] == "Watch for stale practitioners."


def test_first_title_returns_short_label() -> None:
    text = "Tax Associate\n\nA24 is seeking someone…"
    assert _first_title(text) == "Tax Associate"


def test_first_title_rejects_jd_paragraph() -> None:
    # The earlier heuristic returned the full first content line up to
    # 120 chars, so this paragraph became the role_title. Now the
    # stricter heuristic returns empty.
    text = (
        "Northwind is seeking a Tax Associate to work within the Accounting Team "
        "and assist with all tax related matters."
    )
    assert _first_title(text) == ""


def test_first_title_rejects_sentence_with_punctuation() -> None:
    assert _first_title("This is a sentence with a period.") == ""


def test_first_title_rejects_long_label() -> None:
    # 11 words → too long for a title.
    assert _first_title("one two three four five six seven eight nine ten eleven") == ""


def test_first_paragraph_empty_input_returns_empty() -> None:
    assert _first_paragraph("") == ""
    assert _first_paragraph("   \n\n  ") == ""


def test_heuristic_synthesize_leaves_empty_when_jd_is_pure_prose() -> None:
    jd = (
        "Northwind is seeking a Tax Associate to work within the Accounting Team "
        "and assist with all tax related matters.\n\n"
        "JOB DESCRIPTION\nA24 is seeking a Tax Associate."
    )
    v2, _provenance, _insights = _heuristic_synthesize(
        source_text=jd,
        job_description_text=jd,
        intake_notes_text="",
        current_v2_draft=None,
    )
    # role_title is empty — the recruiter fills it in via the review chapter.
    assert v2["role_title"] == ""
    # capability_areas is a single card with empty name + the JD-derived
    # description as starting content. The recruiter edits this on the
    # review chapter. No "Core role scope" placeholder.
    assert len(v2["capability_areas"]) == 1
    assert v2["capability_areas"][0]["name"] == ""
    # description is the extracted JD paragraph (whatever survives
    # _capability_from_text) — non-empty for a JD with content.
    assert v2["capability_areas"][0]["description"]
    # depth_distinction is all-empty — no system-prompt-string leakage.
    assert v2["depth_distinction"] == {
        "builder_definition": "",
        "user_definition": "",
        "edge_case_guidance": "",
    }


def test_heuristic_synthesize_extracts_clean_title_from_short_first_line() -> None:
    jd = "Senior Tax Associate\n\nWe're looking for someone who can…"
    v2, _, _insights = _heuristic_synthesize(
        source_text=jd,
        job_description_text=jd,
        intake_notes_text="",
        current_v2_draft=None,
    )
    assert v2["role_title"] == "Senior Tax Associate"
