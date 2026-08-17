"""P8.1 salvage parity — the salvage path runs the SAME post-parse pipeline.

Guardrails for plans/sourcing-rigor-hardening.md P8.1 (audit R2-F4): a plan that
enters ``form_strategy``'s salvage branch must not silently skip pipeline steps
the success path runs (edge-case rebalance, search-memory demotion,
``original_architecture`` stamping). Both tests drive ``form_strategy`` through
its production entry, never the helper in isolation.

REACHABILITY CAVEAT (test-honesty lens, Wave 3 slice 11): the fixture error is
lab-shaped, not production-shaped. ``_try_salvage_strategy`` can only recover a
plan when the exception MESSAGE carries a complete top-level JSON object, and
production forecloses that for realistic payloads: a genuinely truncated Opus
response raises "Opus response truncated: stop_reason=..." BEFORE parse
(shared/llm_clients.py:345-348, no JSON in the message), and a malformed
completed response raises with the text capped at 500 chars
(shared/llm_clients.py:950) — a truncated top-level object can never yield a
complete prefix. These tests therefore pin the PARITY property for whenever the
branch is entered; salvage reachability itself is a recorded product finding
(dead-lever class), not something this fixture pretends to exercise.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from shared.brief_loader import Brief
from linkedin.strategy import form_strategy


def _make_brief() -> Brief:
    """Synthetic compat brief that trips the edge-case rebalance.

    ``role_description`` carries a tapped-market trigger so
    ``_brief_targets_edge_case_opening`` fires; the canonical/edge-case pattern
    mirrors give ``_opening_priority`` real vocabulary to classify against.
    """
    brief = Brief(
        id="salvage-parity-test",
        role_title="Workflow Automation Engineer",
        role_description=(
            "The obvious pool is tapped — prior recruiters exhausted the "
            "canonical claims-platform population; open with edge cases."
        ),
        kit_url="",
        linkedin_project="proj-salvage-parity",
        linkedin_project_id="",
        minimum_bar="3y hands-on claims automation.",
        archetypes=[{"name": "Claims automation builder"}],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )
    brief.canonical_company_patterns = ["acme claims", "globex insurance"]
    brief.canonical_title_patterns = ["claims platform engineer"]
    brief.edge_case_patterns = ["adjuster copilot", "claims intake workbench"]
    return brief


def _plan_payload() -> dict:
    """Fresh payload per call — the pipeline mutates string dicts in place.

    The fixture decouples the two reordering mechanisms so each assertion can
    only pass if ITS mechanism ran (test-honesty lens, Wave 3 slice 11):

    - canonical_claims_platform: canonical bucket, NOT exhausted. Rebalance
      (and only rebalance) moves it off the opening slot.
    - adjuster_copilot: edge-case bucket, NOT exhausted. Ends up first only
      if rebalance ran.
    - claims_intake_workbench: edge-case bucket, EXHAUSTED. Rebalance alone
      would keep it near the FRONT (edge-case); only search-memory demotion
      moves it to the back.

    Both mechanisms ran  → [adjuster, canonical, workbench]
    No rebalance         → canonical stays first
    No demotion          → canonical (bucket 2) is last, workbench is not
    """
    return {
        "architecture": "titration",
        "architecture_rationale": "mock",
        "architecture_success_criteria": [],
        "architecture_pivot_triggers": [],
        "strategy_rationale": "mock",
        "generated_strings": [
            {
                "boolean": '("acme claims" OR "globex insurance") AND ("claims platform engineer")',
                "rationale": "Canonical claims-platform cleanup string",
                "vocabulary_sources": "mock",
                "family_key": "canonical_claims_platform",
                "novelty_bucket": "canonical",
                "domain_lane": "claims_platforms",
            },
            {
                "boolean": '("adjuster copilot") AND ("production" OR "deployed")',
                "rationale": "Adjuster copilot edge-case population",
                "vocabulary_sources": "mock",
                "family_key": "adjuster_copilot",
                "novelty_bucket": "edge_case",
                "domain_lane": "adjuster_tools",
            },
            {
                "boolean": '("claims intake workbench") AND ("built" OR "shipped")',
                "rationale": "Claims intake workbench edge-case population",
                "vocabulary_sources": "mock",
                "family_key": "claims_intake_workbench",
                "novelty_bucket": "edge_case",
                "domain_lane": "adjuster_tools",
            },
        ],
        "coverage_gaps": [],
        "noise_predictions": [],
    }


def _prior_run_data() -> dict:
    return {
        "search_memory_summary": {
            "overall": {
                "families_tracked": 1,
                "save_rate": 0.02,
                "duplicate_rate": 0.51,
                "novelty_mix": {"edge_case_saves": 2, "canonical_saves": 10},
            },
            "families": [
                {
                    "family_key": "claims_intake_workbench",
                    "novelty_bucket": "edge_case",
                    "domain_lane": "adjuster_tools",
                    "status": "exhausted",
                    "status_reason": "Repeated family with high duplicate overlap.",
                    "save_rate": 0.01,
                    "duplicate_rate": 0.58,
                    "dominant_anchors": ["workbench"],
                }
            ],
        }
    }


def _salvage_error() -> RuntimeError:
    """A lab-shaped error carrying a complete payload + truncated tail.

    Production's 500-char message cap means this shape is only reachable for
    payloads far smaller than a real strategy response — see the module
    docstring's reachability caveat. It exists to ENTER the salvage branch,
    which is the only way to test the branch's pipeline behavior.
    """
    full = json.dumps(_plan_payload())
    truncated_tail = ', "coverage_gaps": [{"gap": "cut mid-str'
    return RuntimeError(
        f"Could not parse JSON from LLM response: {full}{truncated_tail}"
    )


def test_salvage_and_success_paths_produce_identical_plans():
    """Pipeline parity: identical payload in → byte-identical plan out."""
    brief = _make_brief()

    with patch("linkedin.strategy.opus_llm", return_value=_plan_payload()):
        success_plan = form_strategy(brief, [], prior_run_data=_prior_run_data())

    with patch("linkedin.strategy.opus_llm", side_effect=_salvage_error()):
        salvage_plan = form_strategy(brief, [], prior_run_data=_prior_run_data())

    assert salvage_plan.to_dict() == success_plan.to_dict()


def test_truncated_strategy_response_still_runs_full_pipeline():
    """Salvage seam through the production path: the skipped steps now run."""
    brief = _make_brief()

    with patch("linkedin.strategy.opus_llm", side_effect=_salvage_error()):
        plan = form_strategy(brief, [], prior_run_data=_prior_run_data())

    # original_architecture is stamped (was "" on the salvage path).
    assert plan.original_architecture == "titration"

    # The exact final order is producible ONLY when BOTH previously-skipped
    # steps ran (see _plan_payload's decoupling notes): rebalance moved the
    # canonical string off the opening slot, AND demotion moved the exhausted
    # edge-case family behind the canonical one (rebalance alone would keep
    # it ahead of canonical; demotion alone would leave canonical first).
    assert [s["family_key"] for s in plan.generated_strings] == [
        "adjuster_copilot",
        "canonical_claims_platform",
        "claims_intake_workbench",
    ]

    # Lint attach ran (shared with the success path — invariant, not new).
    assert all("boolean_lint" in s for s in plan.generated_strings)
