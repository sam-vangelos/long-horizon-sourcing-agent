"""Tests for the editorial briefing polish backend.

Coverage:
- TestHeuristicBriefing: signal-density gradients (zero-signal,
  single-run-zero-saves, multi-run-multi-lane). Asserts that the
  cold-start case reads warm not bare. Asserts the heuristic NEVER
  surfaces "Tracking N hypotheses" or other engineer prose.
- TestPolishCascade: parameterized over the four failure modes
  (LLM raise, schema invalid, containment fail, banned-token). All
  converge on HeuristicBriefingBackend output.
- TestConfidence: heuristic signal-density formula across gradients;
  LLM containment-check pass/fail.
- TestSnapshotsAgainstRealArtifacts: read the four real on-disk
  market intel artifacts and assert the heuristic produces grounded
  briefings against their aggregate_metrics shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from market_intelligence.agent_backends import PlannerResult
from market_intelligence.briefing_polish import (
    BANNED_BRIEFING_TOKENS,
    BriefingPolishBackend,
    EditorialBriefing,
    HeuristicBriefingBackend,
    _containment_check,
    _signal_density_confidence,
    _top_lane,
)
from market_intelligence.schema import MarketIdentity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _market_identity(role: str = "Principal Forward Deployed AI Engineer") -> MarketIdentity:
    return MarketIdentity(
        market_key="test_market",
        role_title=role,
        role_level="IC6",
        geography="New York, New York, United States",
        channels_seen=["linkedin"],
        brief_ids_seen=["test"],
        brief_versions_seen=["v1"],
    )


def _multi_run_summary() -> dict:
    """Multi-run multi-lane shape — Principal FDE NYC analog."""

    return {
        "aggregate_metrics": {
            "run_count": 2,
            "saved_count": 3,
            "rejected_count": 44,
            "save_rate": 0.064,
            "facial_yes_rate": 0.21,
            "candidate_volume_by_channel": {"linkedin": 47},
        },
        "lane_intelligence": [
            {
                "display_name": "forward-deployed-engineering",
                "saved_count": 2,
                "candidate_volume": 28,
            },
            {
                "display_name": "ml-platform",
                "saved_count": 1,
                "candidate_volume": 12,
            },
        ],
    }


def _single_run_zero_saves_summary() -> dict:
    """Cold-start case: 1 run, 8 candidates, 0 saves, no lanes."""

    return {
        "aggregate_metrics": {
            "run_count": 1,
            "saved_count": 0,
            "rejected_count": 8,
            "save_rate": 0.0,
            "facial_yes_rate": 0.0,
            "candidate_volume_by_channel": {"linkedin": 8},
        },
        "lane_intelligence": [],
    }


def _zero_signal_summary() -> dict:
    """No runs at all — empty signal."""

    return {
        "aggregate_metrics": {
            "run_count": 0,
            "saved_count": 0,
            "rejected_count": 0,
            "save_rate": 0.0,
            "facial_yes_rate": 0.0,
            "candidate_volume_by_channel": {},
        },
        "lane_intelligence": [],
    }


def _planner_with_focus() -> PlannerResult:
    return PlannerResult(
        planner_summary="Tracking 1 active hypotheses across 2 run(s).",
        external_research_focus=[
            {
                "focus": "Whether the comp band is realistic for Staff Engineers in NYC right now",
                "priority": "high",
            },
            {"focus": "Adjacent talent pools", "priority": "medium"},
        ],
    )


# ---------------------------------------------------------------------------
# Heuristic backend — signal-density gradients
# ---------------------------------------------------------------------------


class TestHeuristicBriefing:
    def test_multi_run_reads_grounded(self) -> None:
        out = HeuristicBriefingBackend().polish(
            market_identity=_market_identity(),
            deterministic_summary=_multi_run_summary(),
            planner_result=_planner_with_focus(),
        )
        # The cold-start failure mode is bareness; the multi-run failure
        # mode is engineer-speak. Assert specific recruiter-facing
        # numbers are present and engineer phrasing is absent.
        assert "47 candidates" in out.paragraph
        assert "saved 3" in out.paragraph
        assert "6%" in out.paragraph
        assert "forward-deployed-engineering" in out.paragraph
        assert out.source == "deterministic"
        assert out.confidence >= 0.83  # 5+/6 fields populated

    def test_single_run_zero_saves_reads_warm_not_bare(self) -> None:
        out = HeuristicBriefingBackend().polish(
            market_identity=_market_identity(),
            deterministic_summary=_single_run_zero_saves_summary(),
            planner_result=PlannerResult(),
        )
        # Cold-start framing — must name "first run", "save list",
        # "broader market" so the recruiter feels Cloris saw their run
        # rather than reading a generic non-answer.
        assert "8 candidates" in out.paragraph
        assert "first run" in out.paragraph
        assert "save list" in out.paragraph.lower()
        assert "broader market" in out.paragraph.lower()
        assert out.source == "deterministic"

    def test_zero_signal_collapses_to_editorial_dead_end(self) -> None:
        out = HeuristicBriefingBackend().polish(
            market_identity=_market_identity(),
            deterministic_summary=_zero_signal_summary(),
            planner_result=PlannerResult(),
        )
        assert out.source == "empty"
        assert out.confidence == 0.0
        # Editorial dead-end copy — Cloris voice but doesn't pretend
        # to have signal she doesn't.
        assert "broader market" in out.paragraph.lower()

    def test_steering_note_is_acknowledged(self) -> None:
        out = HeuristicBriefingBackend().polish(
            market_identity=_market_identity(),
            deterministic_summary=_multi_run_summary(),
            planner_result=_planner_with_focus(),
            steering_notes=["Also check Stripe payments experience"],
        )
        assert "Per your note" in out.paragraph
        assert "Stripe payments experience" in out.paragraph

    def test_banned_tokens_never_in_heuristic_output(self) -> None:
        # Run heuristic across all signal-density fixtures; the
        # heuristic builder is hand-written so banned tokens shouldn't
        # appear, but we enforce it as a regression test.
        for ds in (
            _multi_run_summary(),
            _single_run_zero_saves_summary(),
            _zero_signal_summary(),
        ):
            out = HeuristicBriefingBackend().polish(
                market_identity=_market_identity(),
                deterministic_summary=ds,
                planner_result=_planner_with_focus(),
            )
            paragraph_lower = out.paragraph.lower()
            for token in BANNED_BRIEFING_TOKENS:
                assert token not in paragraph_lower, (
                    f"Heuristic produced banned token {token!r} "
                    f"in paragraph: {out.paragraph!r}"
                )

    def test_intentions_carry_priorities(self) -> None:
        out = HeuristicBriefingBackend().polish(
            market_identity=_market_identity(),
            deterministic_summary=_multi_run_summary(),
            planner_result=_planner_with_focus(),
        )
        assert len(out.intentions) == 2
        priorities = [i["priority"] for i in out.intentions]
        assert "high" in priorities
        assert "medium" in priorities


# ---------------------------------------------------------------------------
# Confidence formulas
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_signal_density_zero(self) -> None:
        confidence = _signal_density_confidence(
            run_count=0,
            saved_count=0,
            save_rate=0.0,
            candidate_volume=0,
            channel_volumes={},
            top_lane=None,
            intentions=[],
        )
        assert confidence == 0.0

    def test_signal_density_full(self) -> None:
        confidence = _signal_density_confidence(
            run_count=2,
            saved_count=3,
            save_rate=0.06,
            candidate_volume=47,
            channel_volumes={"linkedin": 47},
            top_lane={"display_name": "fde", "saved_count": 2},
            intentions=[{"text": "x", "priority": "high"}],
        )
        assert confidence == 1.0

    def test_signal_density_cold_start(self) -> None:
        # Single-run zero-saves: run_count > 0 + candidate_volume > 0
        # + intentions populated = 3 of 6 = 0.5.
        confidence = _signal_density_confidence(
            run_count=1,
            saved_count=0,
            save_rate=0.0,
            candidate_volume=8,
            channel_volumes={"linkedin": 8},
            top_lane=None,
            intentions=[{"text": "x", "priority": "medium"}],
        )
        assert confidence == 0.5

    def test_containment_check_passes_with_number(self) -> None:
        ds = {
            "aggregate_metrics": {
                "run_count": 2,
                "saved_count": 3,
                "save_rate": 0.06,
                "candidate_volume_by_channel": {"linkedin": 47},
            },
            "lane_intelligence": [
                {"display_name": "fde", "saved_count": 2}
            ],
        }
        # Paragraph cites the candidate count.
        assert _containment_check(
            paragraph="I read 47 candidates this run.",
            deterministic_summary=ds,
        )

    def test_containment_check_passes_with_lane_name(self) -> None:
        ds = {
            "aggregate_metrics": {"run_count": 1},
            "lane_intelligence": [
                {"display_name": "fde-engineering", "saved_count": 1}
            ],
        }
        assert _containment_check(
            paragraph="The strongest signal came from fde-engineering.",
            deterministic_summary=ds,
        )

    def test_containment_check_passes_with_channel_name(self) -> None:
        ds = {
            "aggregate_metrics": {
                "run_count": 1,
                "candidate_volume_by_channel": {"linkedin": 12},
            },
            "lane_intelligence": [],
        }
        # Paragraph references the channel name even without specific count.
        assert _containment_check(
            paragraph="Saw 12 LinkedIn candidates this run.",
            deterministic_summary=ds,
        )

    def test_containment_check_fails_when_paragraph_is_generic(self) -> None:
        ds = {
            "aggregate_metrics": {
                "run_count": 2,
                "saved_count": 3,
                "save_rate": 0.06,
                "candidate_volume_by_channel": {"linkedin": 47},
            },
            "lane_intelligence": [
                {"display_name": "fde-engineering", "saved_count": 2}
            ],
        }
        # Generic paragraph — no specific value cited.
        assert not _containment_check(
            paragraph="The market has interesting signals worth investigating further.",
            deterministic_summary=ds,
        )

    def test_containment_check_passes_when_no_signals_to_cite(self) -> None:
        # When the input has no signals to cite, the check should not
        # penalize the paragraph for failing to cite values that don't
        # exist. Returns True (treated as passed).
        assert _containment_check(
            paragraph="Anything goes here.",
            deterministic_summary=_zero_signal_summary(),
        )


# ---------------------------------------------------------------------------
# Top-lane selector
# ---------------------------------------------------------------------------


class TestTopLane:
    def test_picks_highest_saved_count(self) -> None:
        lanes = [
            {"display_name": "low-saves", "saved_count": 1, "candidate_volume": 100},
            {"display_name": "high-saves", "saved_count": 5, "candidate_volume": 30},
        ]
        top = _top_lane(lanes)
        assert top is not None
        assert top["display_name"] == "high-saves"

    def test_breaks_tie_by_volume(self) -> None:
        lanes = [
            {"display_name": "low-volume", "saved_count": 2, "candidate_volume": 10},
            {"display_name": "high-volume", "saved_count": 2, "candidate_volume": 50},
        ]
        top = _top_lane(lanes)
        assert top is not None
        assert top["display_name"] == "high-volume"

    def test_returns_none_for_empty(self) -> None:
        assert _top_lane([]) is None
        assert _top_lane(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Polish cascade — four failure modes converge on heuristic
# ---------------------------------------------------------------------------


# Skip the entire polish cascade class when the LLM access guard is
# enabled (it short-circuits to heuristic in test env). We monkey-patch
# _has_llm_access in each test to true so we exercise the actual cascade
# routes against a mocked opus_llm.


class TestPolishCascade:
    """All four failure modes route to HeuristicBriefingBackend output.

    The success path is covered separately (test_llm_success_path).
    """

    @pytest.fixture(autouse=True)
    def _enable_llm_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "market_intelligence.briefing_polish._has_llm_access",
            lambda: True,
        )

    def _expected_heuristic(self) -> EditorialBriefing:
        return HeuristicBriefingBackend().polish(
            market_identity=_market_identity(),
            deterministic_summary=_multi_run_summary(),
            planner_result=_planner_with_focus(),
        )

    def _polish(self) -> EditorialBriefing:
        return BriefingPolishBackend().polish(
            market_identity=_market_identity(),
            deterministic_summary=_multi_run_summary(),
            planner_result=_planner_with_focus(),
        )

    def test_route_1_llm_raises(self) -> None:
        with patch(
            "market_intelligence.briefing_polish.opus_llm",
            side_effect=RuntimeError("network down"),
        ):
            out = self._polish()
        expected = self._expected_heuristic()
        assert out.source == "deterministic"
        assert out.paragraph == expected.paragraph

    def test_route_2_schema_invalid_not_dict(self) -> None:
        with patch(
            "market_intelligence.briefing_polish.opus_llm",
            return_value="not a dict",
        ):
            out = self._polish()
        assert out.source == "deterministic"

    def test_route_2_schema_invalid_paragraph_short(self) -> None:
        with patch(
            "market_intelligence.briefing_polish.opus_llm",
            return_value={"paragraph": "Too short.", "intentions": []},
        ):
            out = self._polish()
        assert out.source == "deterministic"

    def test_route_2_schema_invalid_intentions_not_list(self) -> None:
        with patch(
            "market_intelligence.briefing_polish.opus_llm",
            return_value={
                "paragraph": "I read 47 candidates and saved 3 in your most recent run.",
                "intentions": "not a list",
            },
        ):
            out = self._polish()
        assert out.source == "deterministic"

    def test_route_3_containment_fails(self) -> None:
        # Paragraph passes schema and ban-tokens but cites no specific
        # value from the input.
        with patch(
            "market_intelligence.briefing_polish.opus_llm",
            return_value={
                "paragraph": "The market has interesting signals worth investigating further.",
                "intentions": [{"text": "look into stuff", "priority": "medium"}],
            },
        ):
            out = self._polish()
        assert out.source == "deterministic"

    @pytest.mark.parametrize("token", list(BANNED_BRIEFING_TOKENS))
    def test_route_4_banned_token_in_paragraph(self, token: str) -> None:
        # Build a paragraph that contains the banned token plus a real
        # signal so it would otherwise pass containment.
        with patch(
            "market_intelligence.briefing_polish.opus_llm",
            return_value={
                "paragraph": (
                    f"I read 47 candidates this run. The {token} suggests we should "
                    f"refine the search."
                ),
                "intentions": [{"text": "Refine the search.", "priority": "medium"}],
            },
        ):
            out = self._polish()
        assert out.source == "deterministic"

    @pytest.mark.parametrize(
        "snake_token",
        [
            "devprod_genai",  # the actual real-world regression
            "forward_deployed_engineering",
            "colombian_academic_ml",
            "lane_x_y_z",  # multi-segment
        ],
    )
    def test_route_5_snake_case_identifier_in_paragraph(
        self, snake_token: str
    ) -> None:
        # 5th cascade route — snake_case identifier leaked into output.
        # Build a paragraph that's grounded (passes containment) and
        # has no banned tokens, but contains a snake_case identifier.
        with patch(
            "market_intelligence.briefing_polish.opus_llm",
            return_value={
                "paragraph": (
                    f"I read 47 candidates from the {snake_token} lane "
                    f"and saved 3 in your most recent run."
                ),
                "intentions": [{"text": "Look further.", "priority": "medium"}],
            },
        ):
            out = self._polish()
        # Falls through to heuristic; snake_case never reaches output.
        assert out.source == "deterministic"
        from market_intelligence.briefing_polish import _snake_case_token_hit

        assert _snake_case_token_hit(out.paragraph) is None, (
            f"Heuristic fallback shouldn't produce snake_case either; "
            f"got paragraph: {out.paragraph!r}"
        )

    def test_llm_success_path(self) -> None:
        # Grounded paragraph (cites "47 candidates" + "fde"), no banned
        # tokens, valid schema. Returns LLM source with confidence 1.0.
        good_payload = {
            "paragraph": (
                "I read 47 candidates across 2 runs and saved 3. "
                "The strongest signal came from forward-deployed-engineering."
            ),
            "intentions": [
                {
                    "text": "Whether the comp band is realistic for Staff Engineers in NYC.",
                    "priority": "high",
                },
            ],
        }
        with patch(
            "market_intelligence.briefing_polish.opus_llm",
            return_value=good_payload,
        ):
            out = self._polish()
        assert out.source == "llm"
        assert out.confidence == 1.0
        assert "47 candidates" in out.paragraph


# ---------------------------------------------------------------------------
# Snapshot tests against the four real on-disk market intel artifacts
# ---------------------------------------------------------------------------


_REAL_MARKETS = [
    "principal_forward_deployed_ai_engineer__new_york_new_york_united_states__ic6",
    "forward_deployed_engineer__new_york_new_york_united_states__ic5_ic6",
    "head_of_applied_ai_lab__new_york_city_metropolitan_area__l8_l9",
    "junior_frontier_data_lead__colombia__ic4",
]


class TestSnapshotsAgainstRealArtifacts:
    """Run the heuristic against on-disk artifacts and assert grounding.

    The artifacts have a real ``aggregate_metrics`` block we can derive
    the deterministic_summary shape from. We assert the heuristic
    output is grounded (contains specific numbers from the input) and
    never contains banned tokens.
    """

    @staticmethod
    def _deterministic_summary_from_artifact(market_dir: Path) -> dict | None:
        artifact_path = market_dir / "market-intel.json"
        if not artifact_path.exists():
            return None
        artifact = json.loads(artifact_path.read_text())
        agg = artifact.get("aggregate_metrics") or {}
        if not agg:
            return None
        # Build a minimal deterministic_summary shape — the briefing
        # polish only reads aggregate_metrics + lane_intelligence.
        return {
            "aggregate_metrics": agg,
            "lane_intelligence": artifact.get("lane_intelligence") or [],
        }

    @pytest.mark.parametrize("market_key", _REAL_MARKETS)
    def test_heuristic_against_real_artifact(self, market_key: str) -> None:
        market_dir = (
            Path(__file__).parent.parent
            / "output"
            / "market_intelligence"
            / market_key
        )
        ds = self._deterministic_summary_from_artifact(market_dir)
        if ds is None:
            pytest.skip(f"no on-disk artifact for {market_key}")

        out = HeuristicBriefingBackend().polish(
            market_identity=_market_identity(),
            deterministic_summary=ds,
            planner_result=PlannerResult(),
        )

        # Always grounded — never the engineer-speak the planner used to
        # surface ("Tracking N hypotheses").
        for token in BANNED_BRIEFING_TOKENS:
            assert token not in out.paragraph.lower(), (
                f"{market_key}: banned token {token!r} in paragraph: "
                f"{out.paragraph!r}"
            )

        # Trial-blocking regression check: NO snake_case identifier from
        # the engine layer (lane keys, family keys) leaks into the
        # paragraph. Catches the `devprod_genai` class of bug. The
        # heuristic backend must humanize lane keys before citing them;
        # the LLM backend has the cascade route that catches violations.
        from market_intelligence.briefing_polish import _snake_case_token_hit

        snake_hit = _snake_case_token_hit(out.paragraph)
        assert snake_hit is None, (
            f"{market_key}: snake_case identifier {snake_hit!r} leaked "
            f"into paragraph: {out.paragraph!r}"
        )

        assert out.source in {"deterministic", "empty"}
        # When there's a run, we expect the paragraph to say "I read N
        # candidates" or similar — assert at least one digit is present
        # if there's any candidate volume.
        agg = ds.get("aggregate_metrics") or {}
        candidate_volume = sum(
            int(v or 0)
            for v in (agg.get("candidate_volume_by_channel") or {}).values()
        )
        if candidate_volume > 0:
            assert any(ch.isdigit() for ch in out.paragraph), (
                f"{market_key}: expected a digit in grounded paragraph, "
                f"got: {out.paragraph!r}"
            )
