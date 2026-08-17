"""Tests for the brief polish backend (Phase D Slice D4).

Coverage map (mirrors plan §"Tests"):

- ``TestHeuristicBriefPolish``: snapshot tests against rich / partial /
  minimal chapter-capture fixtures + confidence-formula assertions.
- ``TestBriefPolishCascade``: parameterized across all seven failure
  routes. All converge on :class:`HeuristicBriefPolishBackend` output.
- ``TestPath3Preservation``: source_config.linkedin.project_id MUST
  survive heuristic AND LLM (Route 5).
- ``TestRoleTitlePreservation``: role_title MUST survive (Route 7).
- ``TestHallucinationGuard``: low-overlap LLM output triggers Route 6.
- ``TestSchemaInvalidCascade``: malformed LLM output triggers Route 2.
- ``TestUndoBuffer``: snapshot/restore/restore-with-polish-meta
  semantics — exercised through the API endpoints because the buffer
  mechanism lives at the endpoint layer, not the polish backend.
- ``TestTelemetryLines``: ``_emit_stage`` called with expected log
  shapes per route (start, hallucination_check, fallback, done).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from market_intelligence.brief_polish import (
    BANNED_BRIEFING_TOKENS,
    BriefPolishBackend,
    BriefPolishResult,
    HALLUCINATION_OVERLAP_THRESHOLD,
    HEURISTIC_CONFIDENCE_DENOMINATOR,
    HeuristicBriefPolishBackend,
    MIN_SUBSTANTIVE_CHARS,
    SNAKE_CASE_IDENTIFIER_RE,
    _capability_area_overlap,
    _path3_drift,
    _role_title_drift,
    build_brief_polish_system_prompt,
    build_brief_polish_user_prompt,
)
from shared.brief_v2_schema import validate_v2_brief


# ---------------------------------------------------------------------------
# Fixtures — chapter-capture shapes
# ---------------------------------------------------------------------------


def _captures_rich() -> dict[str, Any]:
    """All four chapters populated. Heuristic confidence should hit 1.0."""

    return {
        "role": {
            "title": "Forward Deployed Engineer",
            "framing": (
                "Embeds with our pilot customers to ship custom workflows "
                "on top of the core platform. Owns the integration end-to-end "
                "from spike to handoff."
            ),
        },
        "good_looks": {
            "prose": (
                "Ships customer-facing systems end-to-end. Comfortable "
                "embedded with PMs and customers, not just other engineers. "
                "Has wrangled at least one PostgreSQL schema migration in "
                "production. Writes Python that other engineers actually "
                "want to read. Knows when to spike and when to harden."
            ),
        },
        "lookalikes": {
            "exemplars_prose": (
                "People who shipped Looker integrations end-to-end at "
                "data startups. The sort of engineer who joins early and "
                "writes both the SDK and the docs."
            ),
            "non_fit_prose": (
                "Pure backend engineers who haven't touched a customer call. "
                "Anyone who needs three weeks of spec before writing code."
            ),
        },
        "where_to_look": {
            "target_modules": ["linkedin"],
            "linkedin_project_id": "3001",
            "linkedin_project_name": "FDE Search 2026",
            "anything_else": "",
        },
    }


def _captures_partial() -> dict[str, Any]:
    """Only role + good_looks. Heuristic should be ~3/7."""

    return {
        "role": {
            "title": "Staff Engineer",
            "framing": "",
        },
        "good_looks": {
            "prose": (
                "Builds platform abstractions for application teams. "
                "Defines the patterns we expect other teams to follow."
            ),
        },
    }


def _captures_minimal() -> dict[str, Any]:
    """Only role.title. Heuristic should be 1/7."""

    return {
        "role": {"title": "Senior Engineer"},
    }


def _captures_empty() -> dict[str, Any]:
    """No usable captures. Heuristic should mark source=empty."""

    return {}


# ---------------------------------------------------------------------------
# Heuristic backend — confidence formulas + scaffolding shape
# ---------------------------------------------------------------------------


class TestHeuristicBriefPolish:
    def test_rich_fixture_passes_validate(self) -> None:
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_rich()
        )
        # The heuristic seed must always pass validate_v2_brief — it's
        # the safety net under the LLM cascade. If this regresses, the
        # cascade-fallback output would itself fail downstream
        # validation, which is the failure mode we're guarding against.
        validate_v2_brief(out.v2_draft)
        assert out.source == "deterministic"

    def test_rich_fixture_full_signal_density(self) -> None:
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_rich()
        )
        # All seven scoring fields populated → confidence = 7/7 = 1.0.
        # Range cap because rounding 7/7 → 1.00 exact.
        assert out.confidence == 1.0

    def test_rich_fixture_promotes_path3_into_source_config(self) -> None:
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_rich()
        )
        # Path 3: linkedin_project_id from where_to_look chapter must
        # land in source_config.linkedin.project_id, NOT at the legacy
        # flat linkedin_project_id key. project_name comes along when
        # present. Mirrors the frontend seeder at OnboardingFlow.svelte:299.
        assert out.v2_draft["source_config"]["linkedin"]["project_id"] == "3001"
        assert (
            out.v2_draft["source_config"]["linkedin"]["project_name"]
            == "FDE Search 2026"
        )

    def test_partial_fixture_partial_signal_density(self) -> None:
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_partial()
        )
        # role.title (1) + good_looks.prose >=30 chars (1) +
        # target_modules defaults to [] in input → not counted.
        # role.framing is empty → not counted.
        # Expected populated fields: 2 (role.title, good_looks.prose).
        # 2/7 ≈ 0.29.
        assert out.source == "deterministic"
        assert out.confidence == round(2 / HEURISTIC_CONFIDENCE_DENOMINATOR, 2)

    def test_minimal_fixture_one_field_signal(self) -> None:
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_minimal()
        )
        # Just role.title → 1/7 ≈ 0.14.
        assert out.source == "deterministic"
        assert out.confidence == round(1 / HEURISTIC_CONFIDENCE_DENOMINATOR, 2)

    def test_empty_fixture_marks_source_empty(self) -> None:
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_empty()
        )
        # No title, no prose → source=empty, confidence=0.0.
        # The Reference Slip surfaces this honestly.
        assert out.source == "empty"
        assert out.confidence == 0.0

    def test_role_title_fallback_prefers_chapter_capture(self) -> None:
        """role.title from captures wins over the role_title hint argument."""
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures={"role": {"title": "Captured Title"}},
            role_title="Hint Title",
        )
        assert out.v2_draft["role_title"] == "Captured Title"

    def test_role_title_fallback_uses_hint_when_capture_empty(self) -> None:
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures={"good_looks": {"prose": "Some prose " * 10}},
            role_title="Hint Title",
        )
        assert out.v2_draft["role_title"] == "Hint Title"

    def test_target_modules_default_when_absent(self) -> None:
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_minimal()
        )
        assert out.v2_draft["target_modules"] == ["linkedin"]

    def test_target_modules_canonical_sort(self) -> None:
        """Brief disk-shape stays stable: target_modules sorted dedup."""
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures={
                "role": {"title": "X"},
                "where_to_look": {
                    "target_modules": ["linkedin", "github", "linkedin"]
                },
            }
        )
        assert out.v2_draft["target_modules"] == ["github", "linkedin"]

    def test_no_source_config_when_no_linkedin_project(self) -> None:
        """Empty source_config dict shouldn't survive — clutters the brief."""
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_minimal()
        )
        assert "source_config" not in out.v2_draft

    def test_substantive_chars_threshold_excludes_short_prose(self) -> None:
        """Sub-MIN_SUBSTANTIVE_CHARS prose doesn't count toward confidence."""
        short = "x" * (MIN_SUBSTANTIVE_CHARS - 1)
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures={
                "role": {"title": "X", "framing": short},
                "good_looks": {"prose": short},
            }
        )
        # Only role.title counts (framing + prose both below threshold).
        assert out.confidence == round(1 / HEURISTIC_CONFIDENCE_DENOMINATOR, 2)


# ---------------------------------------------------------------------------
# Cascade — seven routes converge on heuristic
# ---------------------------------------------------------------------------


class TestBriefPolishCascade:
    """All seven failure modes route to HeuristicBriefPolishBackend output.

    Mirrors the structure of test_briefing_polish.TestPolishCascade.
    Monkey-patches _has_llm_access in each test (default is False under
    PYTEST_CURRENT_TEST so the no_llm_access path would short-circuit).
    """

    @pytest.fixture(autouse=True)
    def _enable_llm_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "market_intelligence.brief_polish._has_llm_access",
            lambda: True,
        )

    def _expected_heuristic(self) -> BriefPolishResult:
        return HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_rich()
        )

    def _polish(self) -> BriefPolishResult:
        return BriefPolishBackend().polish(
            chapter_captures=_captures_rich(),
            role_title="Forward Deployed Engineer",
        )

    def test_route_1_llm_raise_falls_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("network blip")

        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", _raise
        )
        out = self._polish()
        expected = self._expected_heuristic()
        assert out.source == expected.source
        assert out.v2_draft == expected.v2_draft

    def test_route_2_schema_invalid_not_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm",
            lambda *a, **k: ["not", "a", "dict"],
        )
        out = self._polish()
        assert out.source == "deterministic"

    def test_route_2_schema_invalid_missing_required_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Missing depth_distinction → validate_v2_brief raises.
        bad = {
            "role_title": "Forward Deployed Engineer",
            "capability_areas": [{"name": "Foo", "description": "Bar."}],
            "non_fit_patterns": [],
            "target_modules": ["linkedin"],
        }
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: bad
        )
        out = self._polish()
        assert out.source == "deterministic"

    def test_route_3_banned_token_in_capability_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = self._valid_polished_dict()
        bad["capability_areas"][0]["description"] = (
            "Tracking customer hypothesis end-to-end ships systems."
        )
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: bad
        )
        out = self._polish()
        assert out.source == "deterministic"

    def test_route_4_snake_case_in_capability_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = self._valid_polished_dict()
        bad["capability_areas"][0]["name"] = "forward_deployed_engineer"
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: bad
        )
        out = self._polish()
        assert out.source == "deterministic"

    def test_route_5_path3_drift_drops_project_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = self._valid_polished_dict()
        # Drop source_config entirely — drift detector should fire.
        bad.pop("source_config", None)
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: bad
        )
        out = self._polish()
        assert out.source == "deterministic"
        # The fallback heuristic preserves Path 3 from the seed.
        assert (
            out.v2_draft["source_config"]["linkedin"]["project_id"] == "3001"
        )

    def test_route_5_path3_drift_changes_project_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = self._valid_polished_dict()
        bad["source_config"]["linkedin"]["project_id"] = "9999"
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: bad
        )
        out = self._polish()
        assert out.source == "deterministic"
        # Heuristic restores the seed's project_id.
        assert (
            out.v2_draft["source_config"]["linkedin"]["project_id"] == "3001"
        )

    def test_route_6_hallucination_low_overlap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = self._valid_polished_dict()
        # Capability area description with words that don't appear in
        # _captures_rich().good_looks.prose. Avoids banned tokens and
        # snake_case.
        bad["capability_areas"] = [
            {
                "name": "Marine biology",
                "description": (
                    "Researching octopus camouflage adaptations across "
                    "tropical reef ecosystems."
                ),
            }
        ]
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: bad
        )
        out = self._polish()
        assert out.source == "deterministic"

    def test_route_7_role_title_drift(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = self._valid_polished_dict()
        bad["role_title"] = "Senior Engineer"
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: bad
        )
        out = self._polish()
        assert out.source == "deterministic"
        assert out.v2_draft["role_title"] == "Forward Deployed Engineer"

    def test_success_path_marks_source_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good = self._valid_polished_dict()
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: good
        )
        out = self._polish()
        assert out.source == "llm"
        assert out.confidence == 1.0
        # Polished payload survives end-to-end.
        assert out.v2_draft["capability_areas"][0]["name"] == "Customer-facing systems"

    def _valid_polished_dict(self) -> dict:
        """A shape that passes all seven cascade routes against _captures_rich.

        Capability area description deliberately ECHOES words from
        _captures_rich().good_looks.prose so the hallucination overlap
        check passes (>= HALLUCINATION_OVERLAP_THRESHOLD).
        """

        return {
            "role_title": "Forward Deployed Engineer",
            "capability_areas": [
                {
                    "name": "Customer-facing systems",
                    "description": (
                        "Ships customer-facing systems end-to-end. "
                        "Comfortable embedded with PMs and customers."
                    ),
                },
                {
                    "name": "Production database work",
                    "description": (
                        "Has wrangled at least one PostgreSQL schema "
                        "migration in production. Knows when to spike "
                        "and when to harden."
                    ),
                },
            ],
            "depth_distinction": {
                "builder_definition": (
                    "Owns the integration end-to-end from spike to "
                    "handoff. Writes Python other engineers want to read."
                ),
                "user_definition": "",
                "edge_case_guidance": "",
            },
            "non_fit_patterns": [
                {
                    "label": "Spec-bound engineers",
                    "why_not": (
                        "Anyone who needs three weeks of spec before "
                        "writing code."
                    ),
                }
            ],
            "target_modules": ["linkedin"],
            "source_config": {
                "linkedin": {
                    "project_id": "3001",
                    "project_name": "FDE Search 2026",
                }
            },
        }


# ---------------------------------------------------------------------------
# Targeted preservation + hallucination tests
# ---------------------------------------------------------------------------


class TestPath3Preservation:
    def test_heuristic_preserves_project_id(self) -> None:
        out = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_rich()
        )
        assert (
            out.v2_draft["source_config"]["linkedin"]["project_id"] == "3001"
        )

    def test_path3_drift_detector_dropped(self) -> None:
        seeded = {"source_config": {"linkedin": {"project_id": "3001"}}}
        polished = {}
        assert "dropped" in (_path3_drift(seeded, polished) or "")

    def test_path3_drift_detector_changed(self) -> None:
        seeded = {"source_config": {"linkedin": {"project_id": "3001"}}}
        polished = {"source_config": {"linkedin": {"project_id": "9999"}}}
        assert "changed" in (_path3_drift(seeded, polished) or "")

    def test_path3_drift_detector_no_seed_no_drift(self) -> None:
        """No project_id in seed → no drift to detect."""
        assert _path3_drift({}, {"source_config": {"linkedin": {"project_id": "x"}}}) is None

    def test_path3_drift_detector_match(self) -> None:
        seeded = {"source_config": {"linkedin": {"project_id": "3001"}}}
        polished = {"source_config": {"linkedin": {"project_id": "3001"}}}
        assert _path3_drift(seeded, polished) is None


class TestRoleTitlePreservation:
    def test_role_title_drift_detector_dropped(self) -> None:
        assert _role_title_drift(
            seeded_role_title="Forward Deployed Engineer",
            polished_role_title="",
        ) == "dropped"

    def test_role_title_drift_detector_changed(self) -> None:
        out = _role_title_drift(
            seeded_role_title="Forward Deployed Engineer",
            polished_role_title="Senior Engineer",
        )
        assert out is not None
        assert "changed" in out

    def test_role_title_drift_detector_no_seed_no_drift(self) -> None:
        assert _role_title_drift(
            seeded_role_title="",
            polished_role_title="Anything",
        ) is None

    def test_role_title_drift_detector_match(self) -> None:
        assert _role_title_drift(
            seeded_role_title="X",
            polished_role_title="X",
        ) is None


class TestHallucinationGuard:
    def test_overlap_high_when_descriptions_share_prose_words(self) -> None:
        prose = "ships customer-facing systems end-to-end with PMs"
        v2 = {
            "capability_areas": [
                {
                    "name": "X",
                    "description": "Ships customer-facing systems end-to-end.",
                }
            ]
        }
        avg, per_area = _capability_area_overlap(
            v2_draft=v2, good_looks_prose=prose
        )
        # Most tokens shared → high overlap, well above threshold.
        assert avg > HALLUCINATION_OVERLAP_THRESHOLD
        assert per_area[0] > HALLUCINATION_OVERLAP_THRESHOLD

    def test_overlap_low_when_descriptions_invent_topic(self) -> None:
        prose = "ships customer-facing systems end-to-end with PMs"
        v2 = {
            "capability_areas": [
                {
                    "name": "X",
                    "description": (
                        "Researching octopus camouflage adaptations "
                        "across tropical reef ecosystems."
                    ),
                }
            ]
        }
        avg, _ = _capability_area_overlap(
            v2_draft=v2, good_looks_prose=prose
        )
        assert avg < HALLUCINATION_OVERLAP_THRESHOLD

    def test_overlap_returns_one_when_prose_empty(self) -> None:
        """No prose to ground against → don't penalize. Caller gates this."""
        v2 = {"capability_areas": [{"name": "X", "description": "Anything."}]}
        avg, per_area = _capability_area_overlap(
            v2_draft=v2, good_looks_prose=""
        )
        assert avg == 1.0
        assert per_area == [1.0]

    def test_overlap_per_area_distribution_is_returned(self) -> None:
        """Per-area list (not just average) — needed for telemetry analysis."""
        prose = "alpha beta gamma delta"
        v2 = {
            "capability_areas": [
                {"name": "matches", "description": "alpha beta gamma"},
                {"name": "doesn't", "description": "octopus camouflage reef"},
            ]
        }
        avg, per_area = _capability_area_overlap(
            v2_draft=v2, good_looks_prose=prose
        )
        assert len(per_area) == 2
        # First area has high overlap, second has zero. Average sits in middle.
        assert per_area[0] > 0.5
        assert per_area[1] < 0.1
        assert 0.2 < avg < 0.6


# ---------------------------------------------------------------------------
# Schema-invalid cascade — mirrors TestBriefPolishCascade route 2 but
# calls out the structured detail surfaced in fallback log.
# ---------------------------------------------------------------------------


class TestSchemaInvalidCascade:
    @pytest.fixture(autouse=True)
    def _enable_llm_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "market_intelligence.brief_polish._has_llm_access",
            lambda: True,
        )

    def test_missing_depth_distinction_fires_schema_invalid_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = {
            "role_title": "Forward Deployed Engineer",
            "capability_areas": [{"name": "X", "description": "Y."}],
            "non_fit_patterns": [],
            "target_modules": ["linkedin"],
        }
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: bad
        )
        out = BriefPolishBackend().polish(chapter_captures=_captures_rich())
        # Cascade fired — heuristic seed prevails.
        assert out.source == "deterministic"
        # Heuristic depth_distinction is the three empty strings shape.
        assert "depth_distinction" in out.v2_draft
        assert out.v2_draft["depth_distinction"] == {
            "builder_definition": "",
            "user_definition": "",
            "edge_case_guidance": "",
        }


# ---------------------------------------------------------------------------
# Telemetry — log lines per route via _emit_stage
# ---------------------------------------------------------------------------


class _Recorder:
    """Capture every _emit_stage call for assertions."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: str) -> None:
        self.lines.append(message)


class TestTelemetryLines:
    """Verify _emit_stage receives the expected log shapes per route.

    These assertions are deliberately substring-based (not regex full-
    match) so prompt or wording tweaks don't churn the test suite — but
    every key field that operator tooling will grep for is asserted.
    """

    @pytest.fixture(autouse=True)
    def _enable_llm_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "market_intelligence.brief_polish._has_llm_access",
            lambda: True,
        )

    def _patch_recorder(self, monkeypatch: pytest.MonkeyPatch) -> _Recorder:
        recorder = _Recorder()
        monkeypatch.setattr(
            "market_intelligence.brief_polish._emit_stage", recorder
        )
        return recorder

    def test_start_logs_input_richness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = self._patch_recorder(monkeypatch)
        # Make the LLM raise so we get start + fallback + done quickly.
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
        )
        BriefPolishBackend().polish(
            chapter_captures=_captures_rich(),
            role_title="X",
            session_id=42,
        )
        start_lines = [l for l in recorder.lines if "brief.polish:start" in l]
        assert len(start_lines) == 1
        line = start_lines[0]
        # Every operator-grep field present.
        assert "session_id=42" in line
        assert "good_looks_chars=" in line
        assert "exemplars_chars=" in line
        assert "non_fit_chars=" in line
        assert "has_role_title=true" in line
        assert "has_linkedin_project=true" in line

    def test_done_logs_source_confidence_elapsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = self._patch_recorder(monkeypatch)
        # Empty captures → done line emits source=empty without LLM call.
        BriefPolishBackend().polish(chapter_captures=_captures_empty())
        done_lines = [l for l in recorder.lines if "brief.polish:done" in l]
        assert len(done_lines) == 1
        line = done_lines[0]
        assert "source=empty" in line
        assert "confidence=0.00" in line
        assert "elapsed_ms=" in line

    def test_hallucination_check_always_logs_on_llm_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = self._patch_recorder(monkeypatch)
        # Use a polished payload that PASSES all routes including
        # hallucination — the check should still log overlap_avg +
        # per-area + threshold + n_areas.
        good = TestBriefPolishCascade()._valid_polished_dict()
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: good
        )
        BriefPolishBackend().polish(chapter_captures=_captures_rich())
        hall_lines = [
            l for l in recorder.lines if "brief.polish:hallucination_check" in l
        ]
        assert len(hall_lines) == 1
        line = hall_lines[0]
        assert "overlap_avg=" in line
        assert "overlap_per_area=" in line
        assert f"threshold={HALLUCINATION_OVERLAP_THRESHOLD:.2f}" in line
        assert "n_areas=" in line

    def test_fallback_logs_route_specific_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = self._patch_recorder(monkeypatch)
        bad = TestBriefPolishCascade()._valid_polished_dict()
        bad["capability_areas"][0]["name"] = "forward_deployed_eng"
        monkeypatch.setattr(
            "market_intelligence.brief_polish.opus_llm", lambda *a, **k: bad
        )
        BriefPolishBackend().polish(chapter_captures=_captures_rich())
        fallback_lines = [
            l for l in recorder.lines if "brief.polish:fallback" in l
        ]
        assert len(fallback_lines) == 1
        line = fallback_lines[0]
        assert "reason=snake_case_token" in line
        # The token + path are the route-specific detail per the plan
        # — operators need both to diagnose which field misbehaved.
        assert "token=" in line
        assert "path=capability_areas[0].name" in line

    def test_fallback_no_llm_access_emits_fallback_then_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = self._patch_recorder(monkeypatch)
        # Disable LLM access for this test — overrides the autouse fixture
        # because we're testing the no_llm_access cascade specifically.
        monkeypatch.setattr(
            "market_intelligence.brief_polish._has_llm_access",
            lambda: False,
        )
        BriefPolishBackend().polish(chapter_captures=_captures_rich())
        # Three lines: start, fallback (no_llm_access), done.
        kinds = [l.split(":", 2)[1].split(" ")[0] for l in recorder.lines]
        assert kinds == ["start", "fallback", "done"]
        assert "reason=no_llm_access" in recorder.lines[1]


# ---------------------------------------------------------------------------
# Prompt builders — smoke tests for shape
# ---------------------------------------------------------------------------


class TestPromptBuilders:
    def test_system_prompt_encodes_preservation_contracts(self) -> None:
        prompt = build_brief_polish_system_prompt()
        assert "PRESERVATION RULES" in prompt
        assert "role_title" in prompt
        assert "source_config.linkedin.project_id" in prompt
        assert "HALLUCINATION GUARD" in prompt
        # Banned tokens enumerated so the LLM has the same list the
        # cascade route will check against.
        for token in BANNED_BRIEFING_TOKENS:
            assert token in prompt

    def test_user_prompt_carries_chapter_captures_and_seed(self) -> None:
        seeded = HeuristicBriefPolishBackend().polish(
            chapter_captures=_captures_rich()
        ).v2_draft
        prompt = build_brief_polish_user_prompt(
            chapter_captures=_captures_rich(),
            seeded_v2_draft=seeded,
        )
        assert "chapter_captures" in prompt
        assert "seeded_v2_draft" in prompt
        assert "Forward Deployed Engineer" in prompt
        # Path 3 project_id flows through.
        assert "3001" in prompt


# ---------------------------------------------------------------------------
# Undo buffer — exercised through the API endpoints (TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with intake DB + config dir isolated to tmp_path.

    Mirrors the fixture in test_intake_complete.py — both API surfaces
    share the same persistence boundary (intake_sessions table at
    output/intake/intake_sessions.sqlite3).
    """

    from cloris.app import create_app
    from cloris import api as cloris_api

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLORIS_OUTPUT_ROOT", str(output_dir))

    return TestClient(create_app())


def _create_session(client: TestClient, **state: Any) -> int:
    """Create a session and patch state_json with the given fields."""

    create = client.post("/api/intake/sessions", json={})
    assert create.status_code == 201, create.text
    session_id: int = create.json()["session"]["id"]
    if state:
        patch_resp = client.patch(
            f"/api/intake/sessions/{session_id}",
            json={"state_json": state},
        )
        assert patch_resp.status_code == 200, patch_resp.text
    return session_id


class TestUndoBuffer:
    """Snapshot/restore semantics + restore-with-polish-meta lineage.

    Covers the workflows enumerated in the plan's "Undo buffer (one-deep)"
    subsection:
      - polish snapshots prior v2_draft into v2_draft_prev
      - restore pops v2_draft_prev into v2_draft, deletes the buffer key
      - restore preserves polish_meta lineage (LLM polish → restore stays
        source=llm; seed → restore clears polish_meta)
      - restore 404s when no v2_draft_prev exists
    """

    def test_polish_snapshots_prior_v2_draft_into_prev(
        self, client: TestClient
    ) -> None:
        """First polish — prior v2_draft (the seed) goes into v2_draft_prev."""
        session_id = _create_session(
            client,
            role={"title": "Test Role"},
            good_looks={"prose": "Ships customer systems."},
            v2_draft={"role_title": "Pre-polish", "capability_areas": [], "depth_distinction": {}},
        )

        response = client.post(f"/api/intake/sessions/{session_id}/polish")
        assert response.status_code == 200, response.text

        state = response.json()["session"]["state_json"]
        assert "v2_draft_prev" in state
        assert state["v2_draft_prev"]["v2_draft"]["role_title"] == "Pre-polish"

    def test_polish_omits_polish_meta_in_buffer_when_absent(
        self, client: TestClient
    ) -> None:
        """First polish → buffer.polish_meta absent (no prior polish)."""
        session_id = _create_session(
            client,
            role={"title": "Test Role"},
            good_looks={"prose": "Ships customer systems."},
            v2_draft={"role_title": "Pre-polish"},
        )
        response = client.post(f"/api/intake/sessions/{session_id}/polish")
        state = response.json()["session"]["state_json"]
        assert "polish_meta" not in state["v2_draft_prev"]

    def test_polish_writes_polish_meta(self, client: TestClient) -> None:
        """polish_meta MUST be present after polish — Reference Slip reads from it."""
        session_id = _create_session(
            client,
            role={"title": "Test Role"},
            good_looks={"prose": "Ships customer systems end-to-end."},
        )
        response = client.post(f"/api/intake/sessions/{session_id}/polish")
        state = response.json()["session"]["state_json"]
        assert "v2_draft_polish_meta" in state
        meta = state["v2_draft_polish_meta"]
        # Without LLM access (test env), source defaults to deterministic.
        assert meta["source"] in ("deterministic", "llm", "empty")
        assert "confidence" in meta
        assert "polished_at" in meta

    def test_restore_pops_prev_into_v2_draft(self, client: TestClient) -> None:
        session_id = _create_session(
            client,
            role={"title": "Test Role"},
            good_looks={"prose": "Ships customer systems end-to-end."},
            v2_draft={"role_title": "Original", "capability_areas": [], "depth_distinction": {}},
        )

        # Polish to populate v2_draft_prev.
        polish_resp = client.post(f"/api/intake/sessions/{session_id}/polish")
        assert polish_resp.status_code == 200

        # Now restore.
        restore_resp = client.post(
            f"/api/intake/sessions/{session_id}/restore_prev_draft"
        )
        assert restore_resp.status_code == 200, restore_resp.text
        state = restore_resp.json()["session"]["state_json"]
        # v2_draft is the pre-polish shape.
        assert state["v2_draft"]["role_title"] == "Original"
        # Buffer is consumed (one-shot).
        assert "v2_draft_prev" not in state

    def test_restore_clears_polish_meta_when_buffer_had_no_meta(
        self, client: TestClient
    ) -> None:
        """Restoring back to a state with no polish lineage clears polish_meta.

        Critical for honest lineage: the Reference Slip should not lie
        about an unpolished restored draft. Pre-seeds a v2_draft (the
        frontend-seeded scaffold shape) so the polish snapshot has
        something to capture into v2_draft_prev — without this seed
        the polish endpoint sees no prior v2_draft and skips the
        snapshot, so restore would 404 instead of clearing polish_meta.
        """
        session_id = _create_session(
            client,
            role={"title": "Test Role"},
            good_looks={"prose": "Ships customer systems end-to-end."},
            v2_draft={
                "role_title": "Test Role",
                "capability_areas": [{"name": "Cap 1", "description": "x"}],
                "depth_distinction": {
                    "builder_definition": "",
                    "user_definition": "",
                    "edge_case_guidance": "",
                },
            },
        )
        client.post(f"/api/intake/sessions/{session_id}/polish")
        restore_resp = client.post(
            f"/api/intake/sessions/{session_id}/restore_prev_draft"
        )
        assert restore_resp.status_code == 200, restore_resp.text
        state = restore_resp.json()["session"]["state_json"]
        assert "v2_draft_polish_meta" not in state

    def test_restore_preserves_polish_meta_when_buffer_had_meta(
        self, client: TestClient
    ) -> None:
        """Polish twice, then restore — the prior polish_meta is restored."""
        session_id = _create_session(
            client,
            role={"title": "Test Role"},
            good_looks={"prose": "Ships customer systems end-to-end."},
        )
        # Polish #1 — buffer captures the seed (no polish_meta).
        client.post(f"/api/intake/sessions/{session_id}/polish")
        # Polish #2 — buffer captures the post-polish-1 state, which
        # has a polish_meta. After restore we should get THAT polish_meta
        # back, even though the next snapshot would be the post-polish-2
        # state.
        client.post(f"/api/intake/sessions/{session_id}/polish")
        restore_resp = client.post(
            f"/api/intake/sessions/{session_id}/restore_prev_draft"
        )
        state = restore_resp.json()["session"]["state_json"]
        # polish_meta survives because the buffer captured the polish_meta
        # written by polish #1.
        assert "v2_draft_polish_meta" in state

    def test_restore_404s_when_no_prev_draft(self, client: TestClient) -> None:
        session_id = _create_session(client, role={"title": "Test Role"})
        response = client.post(
            f"/api/intake/sessions/{session_id}/restore_prev_draft"
        )
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert detail["error"] == "no_prev_draft"

    def test_restore_404s_when_session_missing(self, client: TestClient) -> None:
        response = client.post(
            "/api/intake/sessions/999999/restore_prev_draft"
        )
        assert response.status_code == 404
        assert response.json()["detail"]["error"] == "intake_session_not_found"

    def test_polish_404s_when_session_missing(self, client: TestClient) -> None:
        response = client.post("/api/intake/sessions/999999/polish")
        assert response.status_code == 404

    def test_polish_then_hand_edit_then_polish_then_restore(
        self, client: TestClient
    ) -> None:
        """The workflow Sam called out: hand-edits between polishes survive restore.

        Sequence:
          1. Polish → buffer = seed, v2_draft = polished-1.
          2. Hand-edit a depth field via PATCH.
          3. Polish again → buffer = polished-1-with-edit, v2_draft = polished-2.
          4. Restore → v2_draft = polished-1-with-edit (the hand-edit is preserved).
        """

        session_id = _create_session(
            client,
            role={"title": "Test Role"},
            good_looks={
                "prose": (
                    "Ships customer-facing systems end-to-end. Comfortable "
                    "embedded with PMs and customers."
                )
            },
        )

        # Polish #1.
        client.post(f"/api/intake/sessions/{session_id}/polish")

        # Hand-edit depth_distinction.builder_definition. Read-modify-write.
        get_resp = client.get(f"/api/intake/sessions/{session_id}")
        assert get_resp.status_code == 200
        current_state = get_resp.json()["session"]["state_json"]
        edited_v2 = dict(current_state["v2_draft"])
        edited_v2["depth_distinction"] = {
            **edited_v2.get("depth_distinction", {}),
            "builder_definition": "EDITED VALUE THAT MUST SURVIVE",
        }
        patch_resp = client.patch(
            f"/api/intake/sessions/{session_id}",
            json={"state_json": {**current_state, "v2_draft": edited_v2}},
        )
        assert patch_resp.status_code == 200

        # Polish #2.
        client.post(f"/api/intake/sessions/{session_id}/polish")

        # Restore — should get back the hand-edited polished-1 draft.
        restore_resp = client.post(
            f"/api/intake/sessions/{session_id}/restore_prev_draft"
        )
        assert restore_resp.status_code == 200
        restored_v2 = restore_resp.json()["session"]["state_json"]["v2_draft"]
        assert (
            restored_v2["depth_distinction"]["builder_definition"]
            == "EDITED VALUE THAT MUST SURVIVE"
        )
