from __future__ import annotations

from cloris.launchers import known_sources
from shared.source_capabilities import (
    recommend_source_strategy_from_text,
    source_capability_keys,
    source_capability_prompt_block,
)


def test_source_capability_manifest_covers_registered_sources() -> None:
    assert set(source_capability_keys()) == set(known_sources())


def test_source_capability_prompt_lists_every_source_without_weak_for() -> None:
    prompt = source_capability_prompt_block()
    for source in source_capability_keys():
        assert f"`{source}`" in prompt
    assert "weak_for" not in prompt


def test_applied_ai_lab_strategy_uses_linkedin_with_corroboration() -> None:
    strategy = recommend_source_strategy_from_text(
        "Head of Applied AI Lab for BFS. Needs applied AI leadership, "
        "research depth, artifacts, and market mapping for senior targets."
    )
    by_source = {item["source"]: item["role"] for item in strategy}

    assert by_source["linkedin"] == "primary"
    assert by_source["github"] == "corroborating"
    assert by_source["researcher"] == "corroborating"
    assert by_source["exec_search"] == "investigation_first"
