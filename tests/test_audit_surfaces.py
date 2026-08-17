"""Unit tests for the Cloris UI audit pipeline rule checkers.

These tests do NOT exercise the Playwright walker (that's an integration
concern requiring a running backend). Instead, they validate that each
rule checker produces the expected RuleResult records for synthetic
ElementFact / SurfaceCapture inputs.
"""

from __future__ import annotations

from typing import Any

import pytest

from tools.audit_common import ElementFact, SurfaceCapture
from tools.audit_rules import (
    check_class_0_placeholder,
    check_class_1_zombie_runs,
    check_R2_no_raw_ids_in_titles,
    check_R9_no_filesystem_paths,
    check_R17_mono_caps_floor,
    check_R21_register_separation,
    check_R24_canonical_wording,
)


def _fact(
    *,
    tag: str = "h1",
    text: str = "",
    classes: list[str] | None = None,
    font_size_px: float = 16.0,
    text_transform: str = "none",
    is_visible: bool = True,
) -> ElementFact:
    return ElementFact(
        tag=tag,
        text=text,
        classes=classes or [],
        font_size_px=font_size_px,
        text_transform=text_transform,
        role=None,
        aria_label=None,
        selector=f"{tag}.test",
        is_visible=is_visible,
    )


def _capture(facts: list[ElementFact], slug: str = "test@1280") -> SurfaceCapture:
    return SurfaceCapture(
        slug=slug,
        route="/#/test",
        viewport_w=1280,
        viewport_h=900,
        description="test",
        full_page_screenshot="",
        viewport_screenshot="",
        dom_html="",
        title="",
        facts=facts,
    )


# ---- R2 — no raw IDs in titles --------------------------------------------

class TestR2:
    def test_pure_numeric_h1_fails(self) -> None:
        cap = _capture([_fact(tag="h1", text="3000000007")])
        results = check_R2_no_raw_ids_in_titles(cap, {})
        assert len(results) == 1
        assert results[0].rule_id == "R2"
        assert results[0].class_id == "2"
        assert not results[0].passed

    def test_numeric_with_suffix_fails(self) -> None:
        cap = _capture([_fact(tag="h1", text="3000000006 Clean")])
        results = check_R2_no_raw_ids_in_titles(cap, {})
        assert len(results) == 1

    def test_human_title_passes(self) -> None:
        cap = _capture([_fact(tag="h1", text="Research Engineer Colombia")])
        results = check_R2_no_raw_ids_in_titles(cap, {})
        assert results == []

    def test_filesystem_path_in_title_fails(self) -> None:
        cap = _capture([_fact(tag="h1", text="/Users/operator/foo")])
        results = check_R2_no_raw_ids_in_titles(cap, {})
        assert len(results) == 1

    def test_invisible_element_skipped(self) -> None:
        cap = _capture([_fact(tag="h1", text="9999999999", is_visible=False)])
        results = check_R2_no_raw_ids_in_titles(cap, {})
        assert results == []

    def test_h2_also_checked(self) -> None:
        cap = _capture([_fact(tag="h2", text="9999999999")])
        results = check_R2_no_raw_ids_in_titles(cap, {})
        assert len(results) == 1

    def test_paragraph_not_checked(self) -> None:
        # R2 is for titles, not body prose.
        cap = _capture([_fact(tag="p", text="Run #3000000007 finished")])
        results = check_R2_no_raw_ids_in_titles(cap, {})
        assert results == []


# ---- R9 — no filesystem paths ---------------------------------------------

class TestR9:
    def test_users_path_fails(self) -> None:
        cap = _capture([_fact(text="/Users/operator/Projects/foo")])
        results = check_R9_no_filesystem_paths(cap, {})
        assert len(results) == 1
        assert results[0].rule_id == "R9"
        assert results[0].class_id == "5"

    def test_output_state_path_fails(self) -> None:
        cap = _capture([_fact(text="output/state/linkedin/foo")])
        results = check_R9_no_filesystem_paths(cap, {})
        assert len(results) == 1

    def test_relative_config_path_passes(self) -> None:
        # R9 forbids absolute paths, not all relative paths. config/foo.json
        # is a separate concern (Phase 1B replaces it with a role title).
        cap = _capture([_fact(text="config/brief-foo.json")])
        results = check_R9_no_filesystem_paths(cap, {})
        assert results == []

    def test_clean_text_passes(self) -> None:
        cap = _capture([_fact(text="Forward Deployed Engineer · Colombia")])
        results = check_R9_no_filesystem_paths(cap, {})
        assert results == []


# ---- R17 — mono-caps ≥ 14px -----------------------------------------------

class TestR17:
    def test_uppercase_below_floor_fails(self) -> None:
        cap = _capture([_fact(
            tag="span",
            text="282 BRIEFS",
            text_transform="uppercase",
            font_size_px=10.0,
        )])
        results = check_R17_mono_caps_floor(cap, {})
        assert len(results) == 1
        assert results[0].rule_id == "R17"

    def test_uppercase_at_floor_passes(self) -> None:
        cap = _capture([_fact(
            tag="span",
            text="STARTED",
            text_transform="uppercase",
            font_size_px=14.0,
        )])
        results = check_R17_mono_caps_floor(cap, {})
        assert results == []

    def test_uppercase_above_floor_passes(self) -> None:
        cap = _capture([_fact(
            tag="h2",
            text="WHERE TO START",
            text_transform="uppercase",
            font_size_px=18.0,
        )])
        results = check_R17_mono_caps_floor(cap, {})
        assert results == []

    def test_non_uppercase_skipped(self) -> None:
        cap = _capture([_fact(
            tag="span",
            text="hello",
            text_transform="none",
            font_size_px=10.0,
        )])
        results = check_R17_mono_caps_floor(cap, {})
        assert results == []


# ---- R24 — canonical wording (no raw enums) -------------------------------

class TestR24:
    def test_raw_stop_reason_fails(self) -> None:
        cap = _capture([_fact(text="Stop reason: governor_limit_reached")])
        results = check_R24_canonical_wording(cap, {})
        assert len(results) == 1
        assert results[0].rule_id == "R24"

    def test_humanized_stop_reason_passes(self) -> None:
        cap = _capture([_fact(text="Stop reason: governor limit")])
        results = check_R24_canonical_wording(cap, {})
        assert results == []

    def test_reference_slip_exception(self) -> None:
        # Reference Slip is the canonical home for raw values — should NOT fail.
        cap = _capture([_fact(
            text="governor_limit_reached",
            classes=["reference-slip-value"],
        )])
        results = check_R24_canonical_wording(cap, {})
        assert results == []


# ---- R21 — register separation --------------------------------------------

class TestR21:
    def test_voice_metaphor_in_button_fails(self) -> None:
        cap = _capture([_fact(
            tag="a",
            text="Back to the front of the file",
            classes=["homescreen-look-through"],
        )])
        results = check_R21_register_separation(cap, {})
        assert len(results) == 1
        assert results[0].rule_id == "R21"

    def test_voice_metaphor_in_decorative_passes(self) -> None:
        # Italic deck paragraphs are voice-zone, not operational.
        cap = _capture([_fact(
            tag="p",
            text="She filed it away for safekeeping.",
            classes=["section-deck"],
        )])
        results = check_R21_register_separation(cap, {})
        assert results == []

    def test_plain_button_text_passes(self) -> None:
        cap = _capture([_fact(
            tag="button",
            text="Stop",
        )])
        results = check_R21_register_separation(cap, {})
        assert results == []


# ---- Class 1 — zombie runs ------------------------------------------------

class TestClass1Zombies:
    def test_running_with_missing_worker_fails(self) -> None:
        api_status: dict[str, Any] = {
            "entries": [{
                "source": "github",
                "state_key": "fde_v1",
                "worker_state": "missing",
                "latest_run": {"id": 4, "status": "running", "started_at": "2026-04-11"},
            }]
        }
        cap = _capture([_fact()], slug="home@1280")
        results = check_class_1_zombie_runs(cap, api_status)
        assert len(results) == 1
        assert results[0].class_id == "1"
        assert results[0].severity == "critical"

    def test_running_with_alive_worker_passes(self) -> None:
        api_status: dict[str, Any] = {
            "entries": [{
                "source": "github",
                "state_key": "fde_v1",
                "worker_state": "alive",
                "latest_run": {"id": 4, "status": "running", "started_at": "2026-04-30"},
            }]
        }
        cap = _capture([_fact()], slug="home@1280")
        results = check_class_1_zombie_runs(cap, api_status)
        assert results == []

    def test_only_emitted_once_per_audit(self) -> None:
        api_status: dict[str, Any] = {
            "entries": [{
                "source": "gh",
                "state_key": "x",
                "worker_state": "missing",
                "latest_run": {"id": 1, "status": "running", "started_at": ""},
            }]
        }
        # Non-home slug should NOT re-fire the API check.
        cap = _capture([_fact()], slug="run-li-int@1280")
        results = check_class_1_zombie_runs(cap, api_status)
        assert results == []


# ---- Class 0 — placeholder text -------------------------------------------

class TestClass0Placeholders:
    def test_placeholder_text_fails(self) -> None:
        cap = _capture([_fact(tag="h1", text="Onboarding placeholder")])
        results = check_class_0_placeholder(cap, {})
        assert len(results) == 1
        assert results[0].class_id == "0"

    def test_coming_soon_fails(self) -> None:
        cap = _capture([_fact(text="This feature is coming soon")])
        results = check_class_0_placeholder(cap, {})
        assert len(results) == 1

    def test_real_content_passes(self) -> None:
        cap = _capture([_fact(text="Authoring is in development.")])
        # 'in development' is acceptable when it's the genuine, designed state.
        # Only pure placeholder words trigger the check.
        results = check_class_0_placeholder(cap, {})
        # 'development' contains 'TBD' / 'TODO' substring — let's confirm it
        # doesn't false-positive.
        assert results == []
