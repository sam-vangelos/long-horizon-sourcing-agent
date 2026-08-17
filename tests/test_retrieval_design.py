"""Tests for layered retrieval-design helpers and brief loading compatibility."""

import tempfile
from pathlib import Path

import pytest

from shared.brief_loader import load_brief
from shared.retrieval_design import (
    derive_legacy_search_views,
    render_retrieval_design,
    retrieval_design_from_payload,
)
from shared.storage import read_json, write_json

_ROOT = Path(__file__).resolve().parent.parent
_FDE_BRIEF = _ROOT / "config" / "Forward-Deployed-Engineer-NYC" / "brief-forward-deployed-engineer-us-v1.4.json"
_HEAD_AI_V2 = _ROOT / "config" / "brief-head-ai-lab-nyc-v2.json"


@pytest.mark.skipif(not _FDE_BRIEF.is_file(), reason="Optional FDE brief JSON not under config/")
def test_legacy_brief_load_keeps_flat_fields_and_derives_retrieval_design():
    brief = load_brief("config/Forward-Deployed-Engineer-NYC/brief-forward-deployed-engineer-us-v1.4.json")

    assert len(brief.search_priorities) == 7
    assert len(brief.additional_search_terms) == 57
    assert brief.retrieval_design["derived_from_legacy"] is True
    assert brief.retrieval_design["families"]


def test_explicit_retrieval_design_derives_legacy_views_and_rendered_variants():
    design = retrieval_design_from_payload(
        {
            "families": [
                {
                    "family_id": "delivery_builders",
                    "label": "Delivery builders",
                    "objective": "Broad entry plus builder proof.",
                    "priority": 90,
                    "enabled": True,
                    "variants_to_emit": 1,
                    "entry_signals": [
                        {"item_id": "entry_1", "label": "Delivery", "terms": ["deployment engineer", "implementation engineer"]}
                    ],
                    "capability_proxies": [
                        {"item_id": "cap_1", "label": "Capability", "terms": ["workflow orchestration", "tool calling"]}
                    ],
                    "reality_filters": [
                        {"item_id": "real_1", "label": "Reality", "terms": ["production", "deployed"]}
                    ],
                    "context_constraints": [],
                    "anti_noise": [],
                    "hypothesis_ids": ["hidden_pool_delivery"],
                }
            ],
            "shared_layers": {},
            "edge_case_hypotheses": [
                {
                    "hypothesis_id": "hidden_pool_delivery",
                    "label": "Hidden delivery pool",
                    "hidden_cohort": "Delivery engineers who do not use FDE titles",
                    "why_missed": "They use implementation language rather than canonical titles.",
                    "entry_signal_variants": [],
                    "capability_proxy_variants": [],
                    "reality_filter_variants": [],
                    "context_constraint_variants": [],
                    "anti_noise_variants": [],
                    "validation_rule": "Promote after repeated strong saves.",
                }
            ],
        }
    )

    priorities, terms = derive_legacy_search_views(design)
    families, strings = render_retrieval_design(design)

    assert families[0]["family_id"] == "delivery_builders"
    assert strings[0]["retrieval_recipe"]["family_id"] == "delivery_builders"
    assert any("Delivery builders" in item for item in priorities)
    assert "workflow orchestration" in terms


@pytest.mark.skipif(not _FDE_BRIEF.is_file(), reason="Optional FDE brief JSON not under config/")
def test_loader_does_not_treat_derived_retrieval_design_as_explicit_opt_in():
    raw = read_json("config/Forward-Deployed-Engineer-NYC/brief-forward-deployed-engineer-us-v1.4.json")
    raw["retrieval_design"] = load_brief(
        "config/Forward-Deployed-Engineer-NYC/brief-forward-deployed-engineer-us-v1.4.json"
    ).retrieval_design

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/brief.json"
        write_json(path, raw)
        brief = load_brief(path)

    assert len(brief.search_priorities) == 7
    assert len(brief.additional_search_terms) == 57
    assert brief.retrieval_design["derived_from_legacy"] is True


@pytest.mark.skipif(not _HEAD_AI_V2.is_file(), reason="Optional Head AI Lab V2 brief JSON not under config/")
def test_loader_rejects_invalid_explicit_retrieval_design():
    raw = read_json(_HEAD_AI_V2)
    raw["retrieval_design"] = {
        "families": [
            {
                "family_id": "payments_builders",
                "label": "Payments builders",
                "objective": "Open payments and transaction-banking builder lanes.",
                "priority": 90,
                "enabled": True,
                "variants_to_emit": 1,
                "entry_signals": [
                    {
                        "item_id": "entry_payments",
                        "label": "Payments personas",
                        "terms": ["payments platform", "transaction banking"],
                    }
                ],
                "capability_proxies": [
                    {
                        "item_id": "cap_orchestration",
                        "label": "Workflow builders",
                        "terms": ["payment orchestration", "fraud workflow"],
                    }
                ],
                "reality_filters": [],
                "context_constraints": [],
                "anti_noise": [],
                "hypothesis_ids": ["payments_hidden_pool"],
            }
        ],
        "shared_layers": {},
        "edge_case_hypotheses": [
            {
                "hypothesis_id": "payments_hidden_pool",
                "label": "Payments hidden pool",
                "hidden_cohort": "Payments-platform builders who do not use AI-leadership titles",
                "why_missed": "They present through workflow and platform language rather than executive AI language.",
                "entry_signal_variants": [],
                "capability_proxy_variants": [],
                "reality_filter_variants": [],
                "context_constraint_variants": [],
                "anti_noise_variants": [],
                "validation_rule": "Promote only after repeated strong saves.",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/brief.json"
        write_json(path, raw)
        with pytest.raises(ValueError, match="Invalid explicit retrieval_design"):
            load_brief(path)
