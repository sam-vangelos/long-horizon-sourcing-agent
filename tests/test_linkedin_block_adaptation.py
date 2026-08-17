"""Unit tests for the extracted LinkedIn BlockAdaptationService cluster."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from shared.schemas import Progress, SearchString
from shared.strict_seniority import is_strict_seniority_brief

from linkedin.block_adaptation import BlockAdaptationDeps, BlockAdaptationService


def _non_strict_brief() -> SimpleNamespace:
    return SimpleNamespace(
        role_description="Platform engineer for a growth-stage startup.",
        role_summary="",
        minimum_bar="",
        minimum_bar_description="",
        intake_notes="",
        notes="",
        instructions=[],
        minimum_years_experience=5,
        raw={},
    )


def _strict_brief() -> SimpleNamespace:
    return SimpleNamespace(
        role_description=(
            "Executive Director analog leader for financial services applied AI."
        ),
        role_summary="",
        minimum_bar="",
        minimum_bar_description="",
        intake_notes="",
        notes="",
        instructions=[],
        minimum_years_experience=15,
        raw={},
    )


def _make_service(holder: dict | None = None) -> BlockAdaptationService:
    state = holder if holder is not None else {}
    if "lint" not in state:
        state["lint"] = []
    if "brief" not in state:
        state["brief"] = _non_strict_brief()
    if "wus" not in state:
        state["wus"] = _FakeWorkUnitService()
    if "memory" not in state:
        state["memory"] = None
    if "ensure_calls" not in state:
        state["ensure_calls"] = 0

    def ensure_services() -> None:
        state["ensure_calls"] += 1

    def set_search_memory(value: dict) -> None:
        state["memory"] = value

    deps = BlockAdaptationDeps(
        log_path=Path(os.devnull),
        get_brief_obj=lambda: state["brief"],
        get_lint_blocked_strings=lambda: state["lint"],
        ensure_services=ensure_services,
        get_work_unit_service=lambda: state["wus"],
        set_search_memory=set_search_memory,
        normalize_candidate_name_key=lambda name: (name or "").lower().strip(),
    )
    return BlockAdaptationService(deps)


class _FakeWorkUnitService:
    def __init__(self, *, sentinel: str = "default") -> None:
        self.sentinel = sentinel
        self.cleared_progress: list[Progress | None] = []
        self.updated_blocks: list[list[SearchString]] = []

    def clear_pending_block_adaptation(self, progress: Progress | None) -> None:
        self.cleared_progress.append(progress)

    def update_search_memory_from_block(self, block_strings: list[SearchString]) -> dict:
        self.updated_blocks.append(block_strings)
        return {"sentinel": self.sentinel, "count": len(block_strings)}


def test_merge_reorder_actions_priority_tie_and_ordering():
    """Higher _priority wins; next beats non-next on ties; next before last; strips internals."""
    service = _make_service()
    base = [
        {"string_id": 1, "move_to": "last", "_priority": 100},
        {"string_id": 2, "move_to": "last", "_priority": 50},
    ]
    overlay = [
        {"string_id": 1, "move_to": "next", "_priority": 100},
        {"string_id": 2, "move_to": "next", "_priority": 200},
        {"string_id": 3, "move_to": "last", "_priority": 10},
    ]
    merged = service._merge_reorder_actions(base, overlay)
    assert [item["string_id"] for item in merged] == [2, 1, 3]
    assert all("_priority" not in item and "_sequence" not in item for item in merged)
    assert merged[0]["move_to"] == "next"
    assert merged[1]["move_to"] == "next"
    assert merged[2]["move_to"] == "last"


def test_reprioritize_new_strings_for_exploitation_tiers_and_tiebreak():
    """Family outranks lane outranks neither; edge_case first within tier; index tiebreak."""
    service = _make_service()
    new_strings = [
        {"family_key": "cold", "domain_lane": "lane_b", "novelty_bucket": "canonical"},
        {"family_key": "hot", "domain_lane": "lane_a", "novelty_bucket": "canonical"},
        {"family_key": "other", "domain_lane": "lane_a", "novelty_bucket": "canonical"},
        {"family_key": "other", "domain_lane": "lane_a", "novelty_bucket": "edge_case"},
        {"family_key": "other", "domain_lane": "lane_a", "novelty_bucket": "canonical"},
    ]
    ordered = service._reprioritize_new_strings_for_exploitation(
        new_strings,
        proven_family_keys={"hot"},
        proven_domain_lanes={"lane_a"},
    )
    assert [item["family_key"] for item in ordered] == [
        "hot",
        "other",
        "other",
        "other",
        "cold",
    ]
    lane_a_others = [item for item in ordered if item["family_key"] == "other"]
    assert lane_a_others[0]["novelty_bucket"] == "edge_case"
    assert lane_a_others[1]["novelty_bucket"] == "canonical"
    assert lane_a_others[2]["novelty_bucket"] == "canonical"


def test_apply_reorder_actions_only_moves_queued_strings():
    """Only queued strings move; next inserts before first queued; last appends."""
    service = _make_service()
    strings = [
        SearchString(id=1, name="queued-first", boolean="a", block="b", status="queued"),
        SearchString(id=2, name="queued-second", boolean="a", block="b", status="queued"),
        SearchString(id=3, name="done", boolean="a", block="b", status="done"),
        SearchString(id=4, name="in-progress", boolean="a", block="b", status="in_progress"),
    ]
    progress = Progress(brief_name="test", strings=strings)
    service._apply_reorder_actions(
        progress,
        [
            {"string_id": 2, "move_to": "next"},
            {"string_id": 4, "move_to": "last"},
        ],
    )
    assert [ss.id for ss in progress.strings] == [2, 1, 3, 4]
    assert [ss.status for ss in progress.strings] == [
        "queued",
        "queued",
        "done",
        "in_progress",
    ]

    progress = Progress(
        brief_name="test",
        strings=[
            SearchString(id=10, name="q1", boolean="a", block="b", status="queued"),
            SearchString(id=11, name="q2", boolean="a", block="b", status="queued"),
        ],
    )
    service._apply_reorder_actions(progress, [{"string_id": 10, "move_to": "last"}])
    assert [ss.id for ss in progress.strings] == [11, 10]


def test_saved_profile_snapshots_dedups_and_skips_missing_index():
    """Normalize callback dedups; names missing from the index are skipped."""
    service = _make_service()
    profile_index = {
        "alice smith": {
            "name": "Alice Smith",
            "headline": "Builder",
            "experiences": [{"title": "Eng", "company": "Acme"}],
        }
    }
    snapshots = service._saved_profile_snapshots(
        ["Alice Smith", "alice smith", "Bob Missing"],
        profile_index,
    )
    assert len(snapshots) == 1
    assert snapshots[0]["name"] == "Alice Smith"
    assert snapshots[0]["title"] == "Eng"


def test_queue_lint_gate_empty_boolean_warning_and_error_paths(capsys):
    """Empty boolean -> {}; errors block; non-error findings return the report dict."""
    service = _make_service()
    assert service._queue_lint_gate(
        {"boolean": "   "},
        boolean_key="boolean",
        source="generated",
        lint_context=None,
    ) == {}

    holder = {"lint": []}
    service = _make_service(holder)
    blocked = service._queue_lint_gate(
        {"boolean": '("deployment engineer"', "rationale": "bad"},
        boolean_key="boolean",
        source="generated",
        lint_context=None,
    )
    assert blocked is None
    assert len(holder["lint"]) == 1
    assert holder["lint"][0]["source"] == "generated"
    assert "unbalanced" in holder["lint"][0]["codes"][0]

    holder = {"lint": []}
    service = _make_service(holder)
    report = service._queue_lint_gate(
        {
            "boolean": '("trust & safety" OR "platform ops")',
            "rationale": "warn",
        },
        boolean_key="boolean",
        source="generated",
        lint_context=None,
    )
    assert report is not None
    assert isinstance(report, dict)
    assert report.get("findings")
    assert holder["lint"] == []


def test_record_lint_blocked_reads_lint_list_live_not_snapshotted():
    """Rebinding lint_blocked_strings must route appends to the new list."""
    holder: dict = {"lint": []}
    service = _make_service(holder)
    old_list = holder["lint"]
    service._record_lint_blocked(
        {"boolean": "x", "rationale": "test"},
        boolean_key="boolean",
        source="generated",
        codes=["code"],
        messages=["msg"],
        repair_hints=["fix"],
    )
    assert len(old_list) == 1

    holder["lint"] = []
    service._record_lint_blocked(
        {"boolean": "y", "rationale": "rebind"},
        boolean_key="boolean",
        source="generated",
        codes=["code2"],
        messages=["msg2"],
        repair_hints=[],
    )
    assert old_list == [{"source": "generated", "name": "test", "family_key": "", "boolean": "x", "codes": ["code"], "messages": ["msg"], "repair_hints": ["fix"]}]
    assert len(holder["lint"]) == 1
    assert holder["lint"][0]["name"] == "rebind"


def test_hydrate_search_string_metadata_reads_brief_live_not_snapshotted():
    """Rebinding brief from non-strict to strict must stamp seniority fields."""
    holder: dict = {"brief": _non_strict_brief()}
    service = _make_service(holder)
    search_string = SearchString(
        id=1,
        name="VP Engineering",
        boolean='"VP" AND "engineering"',
        block="b",
    )
    service._hydrate_search_string_metadata(search_string)
    assert search_string.seniority_risk == ""
    assert search_string.opening_eligible is None

    holder["brief"] = _strict_brief()
    assert is_strict_seniority_brief(holder["brief"])
    service._hydrate_search_string_metadata(search_string)
    assert search_string.seniority_risk
    assert search_string.title_bucket_risk
    assert search_string.opening_eligible is not None


def test_clear_pending_block_adaptation_reads_work_unit_service_live_not_snapshotted():
    """Rebinding work_unit_service must hit the new fake, not the old one."""
    first = _FakeWorkUnitService(sentinel="first")
    holder: dict = {"wus": first}
    service = _make_service(holder)
    progress = Progress(brief_name="test", strings=[])
    service._clear_pending_block_adaptation(progress)
    assert first.cleared_progress == [progress]

    second = _FakeWorkUnitService(sentinel="second")
    holder["wus"] = second
    service._clear_pending_block_adaptation(progress)
    assert second.cleared_progress == [progress]
    assert len(first.cleared_progress) == 1


def test_update_search_memory_from_block_reads_work_unit_and_setter_live():
    """Rebinding work_unit_service and set_search_memory must route live."""
    first = _FakeWorkUnitService(sentinel="first")
    holder: dict = {"wus": first, "memory": None}
    service = _make_service(holder)
    block = [SearchString(id=1, name="s", boolean="a", block="b")]
    service._update_search_memory_from_block(block)
    assert holder["memory"] == {"sentinel": "first", "count": 1}
    assert holder["ensure_calls"] == 1

    second = _FakeWorkUnitService(sentinel="second")
    holder["wus"] = second
    service._update_search_memory_from_block(block)
    assert holder["memory"] == {"sentinel": "second", "count": 1}
    assert second.updated_blocks[-1] == block
