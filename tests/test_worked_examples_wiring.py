"""Focused wiring test for the `worked_examples` brief field.

Mirrors the established `transferability_examples` thread (schema -> loader ->
Brief -> block). Two properties:

(a) A brief that POPULATES worked_examples renders them — proves the lever is
    wired end-to-end through the real loader path that the prompt reads
    (``b._new_brief``).
(b) A brief with EMPTY worked_examples renders "" exactly — the load-bearing
    invariant. The judgment-template helper
    (``_calibration_worked_examples_or_default``) only uses the block's value
    ``if rendered``; an empty string is falsy, so it falls through to the
    vertical-agnostic fallback and the assembled full-evaluation prompt stays
    byte-identical for every brief that does not populate this field. If the
    block ever returned a header on empty, byte-equality would break and the
    characterization tests would too — ``test_empty_brief_renders_empty_string_exactly``
    red-fails on that one-line revert.

The empty Brief is constructed via ``_load_v2_brief(_minimal_v2_raw())._new_brief``,
the same idiom the sibling ``transferability_examples`` empty-input test uses
(``tests/test_calibration_brief_fields.py``): ``brief_schema.Brief`` has required
positional args, so a bare ``Brief()`` does not construct — the minimal-raw load
is the faithful "no worked_examples" brief.
"""

from shared import brief_schema
from shared.brief_loader import load_brief, _load_v2_brief
from linkedin.judgment_templates import assemble_full_evaluation_system


COLOMBIA_BRIEF = "config/FDL-Colombia/brief-fdl-colombia-v4.json"


def _minimal_v2_raw() -> dict:
    """Minimum viable V2 brief raw dict with NO worked_examples populated."""
    return {
        "role_title": "Senior Marketing Lead",
        "role_level": "L5",
        "role_summary": "Generic vertical-agnostic test role.",
        "geography": "New York",
        "linkedin_project": "Test Project",
        "capability_areas": [
            {
                "name": "Capability A",
                "description": "Owns campaign launches.",
                "builder_signals": ["launched campaign"],
                "user_signals": ["read campaign reports"],
                "key_terms": ["launch"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns the launch.",
            "user_definition": "Watches the launch.",
            "edge_case_guidance": "Defer to evidence.",
        },
        "non_fit_patterns": [],
        "employer_signal_rules": [],
        "minimum_years_experience": 5,
        "minimum_bar_description": "Five years of launch ownership.",
        "facial_calibration": {
            "expected_yes_rate_low": 0.25,
            "expected_yes_rate_high": 0.55,
            "fast_exit_patterns": [],
            "trajectory_yes_patterns": [],
            "trajectory_ambiguous_patterns": [],
            "trajectory_no_patterns": [],
        },
    }


def test_populated_brief_renders_worked_examples():
    b = load_brief(COLOMBIA_BRIEF)
    new_brief = b._new_brief

    # The loader hydrates the structured Brief the prompt actually reads.
    assert len(new_brief.worked_examples) == 4
    assert all(isinstance(we, brief_schema.WorkedExample) for we in new_brief.worked_examples)

    rendered = new_brief.worked_examples_block()
    assert rendered != ""
    # Content threads through: a SAVE exemplar and the REJECT floor are present.
    assert "Candidate W1" in rendered
    assert "REJECT" in rendered
    # Every authored decision label surfaces in the rendered block.
    for decision in ("SAVE", "TRANSFERABLE_SAVE", "INFERENTIAL_SAVE", "REJECT"):
        assert decision in rendered


def test_empty_brief_renders_empty_string_exactly():
    # Load-bearing: byte-equality of the full-eval prompt for non-populated
    # briefs depends on this returning "" exactly. A minimal V2 raw with no
    # worked_examples must yield an empty list and an empty block.
    nb = _load_v2_brief(_minimal_v2_raw())._new_brief
    assert nb.worked_examples == []
    # Exact equality — not just falsy. A one-line revert that returned a header
    # ("- WORKED EXAMPLES" / "[...]") on empty would make this assertion fail.
    assert nb.worked_examples_block() == ""


def test_empty_worked_examples_assembles_the_vertical_agnostic_fallback():
    """End-to-end byte-equality guard through the real prompt producer.

    The two characterization tests
    (``test_judgment_template_calibration.py`` / ``test_seam_judgment.py``)
    assert STRUCTURAL facts about the assembled full-eval prompt — forbidden
    strings absent, every placeholder filled, populated vocab present — not an
    exact golden string. They therefore do NOT catch a worked_examples_block
    that returns a non-forbidden header on empty. This test closes that gap: it
    asserts the assembled prompt for a brief with no worked_examples carries the
    vertical-agnostic fallback verbatim, so the WORKED EXAMPLES slot is
    byte-for-byte the legacy output. Red-fails on the same one-line revert that
    the block-level test above catches, but pins the invariant where it actually
    matters — the assembled prompt the LLM sees.
    """
    nb = _load_v2_brief(_minimal_v2_raw())._new_brief
    prompt = assemble_full_evaluation_system(nb)
    assert "(No worked examples were provided in the brief." in prompt
