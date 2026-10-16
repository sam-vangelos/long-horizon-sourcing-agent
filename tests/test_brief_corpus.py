from __future__ import annotations

from pathlib import Path

from shared.brief_corpus import build_exemplar_block, index_v2_brief, query_corpus
from shared.runtime_state.store import RuntimeStateStore


def test_index_and_query_accepted_brief(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    index_v2_brief(
        store,
        brief_key="brief-1",
        title="Staff Platform Engineer",
        v2_json={
            "role_title": "Staff Platform Engineer",
            "role_summary": "Owns developer infrastructure.",
            "capability_areas": [
                {
                    "name": "Developer platform",
                    "description": "Builds internal platform systems.",
                }
            ],
            "depth_distinction": {
                "builder_definition": "Ships the platform.",
                "user_definition": "Uses the platform.",
                "edge_case_guidance": "Check for ownership.",
            },
        },
    )
    hits = query_corpus(store, source_excerpt="developer platform infrastructure")
    assert hits
    assert hits[0].brief_key == "brief-1"


def test_build_exemplar_block_returns_used_ids(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    index_v2_brief(
        store,
        brief_key="brief-2",
        title="ML Infra Lead",
        v2_json={
            "role_title": "ML Infra Lead",
            "capability_areas": [{"name": "ML infra", "description": "Serving systems."}],
            "depth_distinction": {},
        },
    )
    block, used = build_exemplar_block(store, "ML serving systems")
    assert "ML Infra Lead" in block
    assert used == ["brief-2"]
