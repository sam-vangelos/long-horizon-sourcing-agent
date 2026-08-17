"""Tests for the chief-of-staff cross-source synthesis backend.

Coverage:

- TestHeuristicSynthesis: signal-density gradients (rich, one-silent,
  minimal, empty). Asserts grounded paragraphs, in-range weights,
  no banned tokens / snake_case identifiers, and a stable confidence
  formula.
- TestSynthesisCascade: parameterized over the six failure modes
  (LLM raise, schema invalid — including the per_specialist_weight
  value-validation slot — banned token, snake_case identifier,
  specialist_weight_invalid hallucination, containment fail) plus
  the success path. All failure modes converge on the heuristic
  output.
- TestTelemetryLines: asserts the ``[chief-of-staff] synthesis:*``
  log line shape so the cascade is observable and matches the
  ``[market-intel] reflection.polish:*`` convention used elsewhere
  in the repo.
- TestSpecialistWeightDrift / TestContainment: focused unit tests on
  the two cascade-route helpers most likely to change shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from cloris.chief_of_staff.agent import (
    BANNED_BRIEFING_TOKENS,
    ChiefOfStaffAgent,
    ChiefOfStaffSynthesis,
    HeuristicChiefOfStaffSynthesizer,
    _containment_check,
    _heuristic_confidence,
    _normalize_per_source,
    _snake_case_token_hit,
    _specialist_weight_drift,
    _validate_schema,
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
        channels_seen=["linkedin", "github"],
        brief_ids_seen=["test"],
        brief_versions_seen=["v1"],
    )


def _two_source_rich() -> dict[str, dict]:
    """Both LinkedIn and GitHub contributed candidates and saves.

    Mirrors the canonical multi-source flagship case named in the
    plan ("Across LinkedIn (47 candidates, 3 saves) and GitHub (22
    candidates, 1 save), LinkedIn carried the denser signal").
    """

    return {
        "linkedin": {
            "candidate_count": 47,
            "save_count": 3,
            "top_lane": "forward_deployed_engineering",
        },
        "github": {
            "candidate_count": 22,
            "save_count": 1,
            "top_lane": "rust ml",
        },
    }


def _two_source_one_silent() -> dict[str, dict]:
    """LinkedIn surfaced saves; GitHub returned candidates but no saves."""

    return {
        "linkedin": {
            "candidate_count": 47,
            "save_count": 3,
            "top_lane": "forward_deployed_engineering",
        },
        "github": {
            "candidate_count": 22,
            "save_count": 0,
            "top_lane": None,
        },
    }


def _two_source_minimal() -> dict[str, dict]:
    """Both contributed candidates; neither surfaced saves yet."""

    return {
        "linkedin": {"candidate_count": 12, "save_count": 0, "top_lane": None},
        "github": {"candidate_count": 8, "save_count": 0, "top_lane": None},
    }


def _empty_signals() -> dict[str, dict]:
    """Defensive: empty input. The integration layer guards this; the
    heuristic should still produce sane editorial-dead-end output if
    invoked."""

    return {}


def _example_briefing_paragraph() -> str:
    return (
        "I read 47 candidates across 2 runs and saved 3 — about 6%. The "
        "strongest signal came from forward-deployed-engineering with 2 "
        "of those saves."
    )


def _good_llm_payload() -> dict:
    return {
        "paragraph": (
            "Across LinkedIn (47 candidates, 3 saves) and GitHub (22 "
            "candidates, 1 save), LinkedIn carried the densest save "
            "signal this run with Forward Deployed Engineering as the "
            "strongest lane."
        ),
        "per_specialist_weight": {
            "linkedin": {
                "weight": 0.85,
                "rationale": (
                    "3 saves on 47 candidates with a clean lane "
                    "concentration."
                ),
            },
            "github": {
                "weight": 0.55,
                "rationale": (
                    "22 maintainers, 1 save — useful breadth but thinner "
                    "save signal."
                ),
            },
        },
        "priority_for_principal": (
            "Start with the LinkedIn saves first."
        ),
    }


# ---------------------------------------------------------------------------
# Heuristic backend — signal-density gradients
# ---------------------------------------------------------------------------


class TestHeuristicSynthesis:
    def test_rich_signals_produce_grounded_paragraph(self) -> None:
        out = HeuristicChiefOfStaffSynthesizer().synthesize(
            market_identity=_market_identity(),
            per_source_signals=_two_source_rich(),
            briefing_paragraph=_example_briefing_paragraph(),
        )
        assert out.source == "deterministic"
        # Grounded numbers and humanized source names land in prose.
        assert "47 candidate" in out.paragraph
        assert "3 save" in out.paragraph
        assert "22 candidate" in out.paragraph
        assert "1 save" in out.paragraph
        assert "LinkedIn" in out.paragraph
        assert "GitHub" in out.paragraph
        # Lane name is humanized (no underscores in prose).
        assert "Forward Deployed Engineering" in out.paragraph
        # Both contributing sources have weight entries.
        assert set(out.per_specialist_weight.keys()) == {"linkedin", "github"}
        for source, entry in out.per_specialist_weight.items():
            assert 0.0 <= entry["weight"] <= 1.0, (
                f"weight for {source} out of range: {entry}"
            )
            assert entry["rationale"], f"empty rationale for {source}"
        # Stronger source has higher weight than the weaker one.
        assert (
            out.per_specialist_weight["linkedin"]["weight"]
            > out.per_specialist_weight["github"]["weight"]
        )
        # Priority names the strongest source as a concrete first action.
        assert "LinkedIn" in out.priority_for_principal
        assert "save" in out.priority_for_principal.lower()

    def test_one_silent_source_gets_lower_weight_with_negative_read(
        self,
    ) -> None:
        out = HeuristicChiefOfStaffSynthesizer().synthesize(
            market_identity=_market_identity(),
            per_source_signals=_two_source_one_silent(),
            briefing_paragraph=_example_briefing_paragraph(),
        )
        assert out.source == "deterministic"
        # Silent source gets lower weight; its rationale names the
        # negative read so the principal sees the framing.
        github_entry = out.per_specialist_weight["github"]
        linkedin_entry = out.per_specialist_weight["linkedin"]
        # Silent source caps at the silent-path default (0.4).
        assert github_entry["weight"] <= 0.5
        # Save-producing source is materially above the silent-path
        # default. The heuristic anchors at 0.5 and rewards save
        # density, so any non-zero save signal should clear 0.55.
        assert linkedin_entry["weight"] >= 0.55
        # Relative ordering — the actual substantive assertion.
        assert linkedin_entry["weight"] > github_entry["weight"]
        assert "negative read" in github_entry["rationale"]
        assert "0 saves" in out.paragraph or "0 save" in out.paragraph
        assert "3 save" in out.paragraph
        # Both sources still named in the run-clause.
        assert "LinkedIn" in out.paragraph
        assert "GitHub" in out.paragraph

    def test_minimal_signals_produces_negative_read_priority(self) -> None:
        out = HeuristicChiefOfStaffSynthesizer().synthesize(
            market_identity=_market_identity(),
            per_source_signals=_two_source_minimal(),
            briefing_paragraph="",
        )
        assert out.source == "deterministic"
        # No specialist surfaced saves yet — paragraph names that.
        assert "no saves" in out.paragraph.lower() or (
            "0 saves" in out.paragraph or "0 save" in out.paragraph
        )
        # Priority defaults to broader-market framing rather than a
        # bogus "Start with..." action.
        assert "broader market" in out.priority_for_principal.lower()
        # Weights cluster at the low-confidence default for sources
        # that contributed but didn't save.
        for entry in out.per_specialist_weight.values():
            assert 0.3 <= entry["weight"] <= 0.5

    def test_empty_signals_collapses_to_editorial_dead_end(self) -> None:
        out = HeuristicChiefOfStaffSynthesizer().synthesize(
            market_identity=_market_identity(),
            per_source_signals=_empty_signals(),
            briefing_paragraph="",
        )
        assert out.source == "empty"
        assert out.confidence == 0.0
        # Editorial dead-end copy — Cloris voice, doesn't pretend.
        assert "broader market" in out.paragraph.lower()
        assert out.per_specialist_weight == {}

    def test_banned_tokens_never_in_heuristic_output(self) -> None:
        for sigs in (
            _two_source_rich(),
            _two_source_one_silent(),
            _two_source_minimal(),
            _empty_signals(),
        ):
            out = HeuristicChiefOfStaffSynthesizer().synthesize(
                market_identity=_market_identity(),
                per_source_signals=sigs,
                briefing_paragraph=_example_briefing_paragraph(),
            )
            paragraph_lower = out.paragraph.lower()
            for token in BANNED_BRIEFING_TOKENS:
                assert token not in paragraph_lower, (
                    f"Heuristic produced banned token {token!r}: "
                    f"{out.paragraph!r}"
                )

    def test_snake_case_never_in_heuristic_output(self) -> None:
        for sigs in (
            _two_source_rich(),
            _two_source_one_silent(),
            _two_source_minimal(),
        ):
            out = HeuristicChiefOfStaffSynthesizer().synthesize(
                market_identity=_market_identity(),
                per_source_signals=sigs,
                briefing_paragraph=_example_briefing_paragraph(),
            )
            assert _snake_case_token_hit(out.paragraph) is None, (
                f"Heuristic leaked snake_case identifier in: "
                f"{out.paragraph!r}"
            )
            for entry in out.per_specialist_weight.values():
                assert (
                    _snake_case_token_hit(entry["rationale"]) is None
                ), f"Heuristic leaked snake_case in rationale: {entry}"

    def test_to_dict_round_trip_is_stable(self) -> None:
        out = HeuristicChiefOfStaffSynthesizer().synthesize(
            market_identity=_market_identity(),
            per_source_signals=_two_source_rich(),
            briefing_paragraph=_example_briefing_paragraph(),
        )
        d = out.to_dict()
        assert d["source"] == "deterministic"
        assert isinstance(d["paragraph"], str)
        assert isinstance(d["per_specialist_weight"], dict)
        assert isinstance(d["priority_for_principal"], str)
        assert isinstance(d["confidence"], float)
        for source, entry in d["per_specialist_weight"].items():
            assert isinstance(entry["weight"], float)
            assert 0.0 <= entry["weight"] <= 1.0
            assert isinstance(entry["rationale"], str)


# ---------------------------------------------------------------------------
# Confidence + helpers
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_signal_density_full(self) -> None:
        normalized = {
            "linkedin": _normalize_per_source(
                {
                    "candidate_count": 47,
                    "save_count": 3,
                    "top_lane": "forward_deployed_engineering",
                }
            ),
            "github": _normalize_per_source(
                {"candidate_count": 22, "save_count": 1, "top_lane": "rust ml"}
            ),
        }
        c = _heuristic_confidence(
            sources=sorted(normalized.keys()),
            normalized=normalized,
            briefing_paragraph=_example_briefing_paragraph(),
        )
        assert c == 1.0

    def test_signal_density_minimal(self) -> None:
        normalized = {
            "linkedin": _normalize_per_source(
                {"candidate_count": 12, "save_count": 0, "top_lane": None}
            ),
            "github": _normalize_per_source(
                {"candidate_count": 8, "save_count": 0, "top_lane": None}
            ),
        }
        # 2 sources contributing + ≥1 has candidates = 2/5 = 0.4
        c = _heuristic_confidence(
            sources=sorted(normalized.keys()),
            normalized=normalized,
            briefing_paragraph="",
        )
        assert c == 0.4


class TestSpecialistWeightDrift:
    """The synthesis-specific preservation contract — keys must be in
    the contributing-sources set."""

    def test_no_drift_when_keys_match(self) -> None:
        invented = _specialist_weight_drift(
            per_specialist_weight={
                "linkedin": {"weight": 0.8, "rationale": "x"},
                "github": {"weight": 0.5, "rationale": "y"},
            },
            contributing_sources={"linkedin", "github"},
        )
        assert invented == set()

    def test_drift_when_specialist_invented(self) -> None:
        invented = _specialist_weight_drift(
            per_specialist_weight={
                "linkedin": {"weight": 0.8, "rationale": "x"},
                "researcher": {"weight": 0.6, "rationale": "y"},  # didn't run
            },
            contributing_sources={"linkedin", "github"},
        )
        assert invented == {"researcher"}

    def test_case_mismatch_is_drift(self) -> None:
        # System prompt says use exact source-key strings; case
        # mismatch is a violation worth catching.
        invented = _specialist_weight_drift(
            per_specialist_weight={
                "LinkedIn": {"weight": 0.8, "rationale": "x"},
            },
            contributing_sources={"linkedin"},
        )
        assert invented == {"LinkedIn"}

    def test_non_dict_input_is_no_drift(self) -> None:
        # Defensive — schema validation catches non-dict elsewhere.
        assert (
            _specialist_weight_drift(
                per_specialist_weight="not a dict",  # type: ignore[arg-type]
                contributing_sources={"linkedin"},
            )
            == set()
        )


class TestContainment:
    """Mirrors briefing_polish:_containment_check shape — substring
    match against needles built from per-source signals."""

    def test_passes_with_candidate_count(self) -> None:
        sigs = _two_source_rich()
        assert _containment_check(
            paragraph="The team read 47 candidates this run.",
            per_source_signals=sigs,
        )

    def test_passes_with_source_name(self) -> None:
        sigs = _two_source_rich()
        assert _containment_check(
            paragraph="LinkedIn carried the read this run.",
            per_source_signals=sigs,
        )

    def test_passes_with_humanized_top_lane(self) -> None:
        sigs = _two_source_rich()
        assert _containment_check(
            paragraph="Forward Deployed Engineering was the densest lane.",
            per_source_signals=sigs,
        )

    def test_passes_with_raw_top_lane(self) -> None:
        sigs = _two_source_rich()
        # Cascade ordering: snake_case route fires BEFORE containment,
        # so a paragraph with raw snake_case is rejected upstream.
        # But the containment check itself is permissive: raw-form
        # values are still grounded inputs and pass containment.
        assert _containment_check(
            paragraph="The forward_deployed_engineering lane was densest.",
            per_source_signals=sigs,
        )

    def test_fails_when_paragraph_is_generic(self) -> None:
        sigs = _two_source_rich()
        assert not _containment_check(
            paragraph="The team turned in some interesting work this run.",
            per_source_signals=sigs,
        )

    def test_passes_when_no_signals_to_cite(self) -> None:
        # Defensive: empty needles → don't penalize.
        assert _containment_check(
            paragraph="Anything goes.",
            per_source_signals={},
        )


# ---------------------------------------------------------------------------
# Schema validator — covers per_specialist_weight value validation
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    @staticmethod
    def _good_payload() -> dict:
        return _good_llm_payload()

    def test_passes_on_good_payload(self) -> None:
        assert _validate_schema(self._good_payload()) is None

    def test_rejects_non_dict(self) -> None:
        assert _validate_schema("not a dict") == "not_dict"

    def test_rejects_short_paragraph(self) -> None:
        payload = self._good_payload()
        payload["paragraph"] = "Too short."
        assert _validate_schema(payload) == "paragraph_missing_or_short"

    def test_rejects_non_dict_weights(self) -> None:
        payload = self._good_payload()
        payload["per_specialist_weight"] = ["not a dict"]
        assert _validate_schema(payload) == "per_specialist_weight_not_dict"

    def test_rejects_empty_weights(self) -> None:
        payload = self._good_payload()
        payload["per_specialist_weight"] = {}
        assert _validate_schema(payload) == "per_specialist_weight_empty"

    def test_rejects_weight_not_numeric(self) -> None:
        payload = self._good_payload()
        payload["per_specialist_weight"]["linkedin"]["weight"] = "high"
        assert _validate_schema(payload) == "weight_not_numeric:linkedin"

    @pytest.mark.parametrize("bad_weight", [-0.01, 1.01, 1.5, -1.0, 2.0])
    def test_rejects_weight_out_of_range(self, bad_weight: float) -> None:
        payload = self._good_payload()
        payload["per_specialist_weight"]["linkedin"]["weight"] = bad_weight
        result = _validate_schema(payload)
        assert result is not None
        assert result.startswith("weight_out_of_range:linkedin=")

    def test_rejects_weight_boolean(self) -> None:
        # Defensive: bool is a subclass of int in Python; reject it
        # explicitly so the LLM can't smuggle True/False as a weight.
        payload = self._good_payload()
        payload["per_specialist_weight"]["linkedin"]["weight"] = True
        assert _validate_schema(payload) == "weight_not_numeric:linkedin"

    def test_rejects_empty_rationale(self) -> None:
        payload = self._good_payload()
        payload["per_specialist_weight"]["linkedin"]["rationale"] = ""
        assert (
            _validate_schema(payload) == "rationale_missing_or_empty:linkedin"
        )

    def test_rejects_missing_priority(self) -> None:
        payload = self._good_payload()
        payload["priority_for_principal"] = ""
        assert _validate_schema(payload) == "priority_missing_or_empty"


# ---------------------------------------------------------------------------
# Cascade — six routes converge on the heuristic
# ---------------------------------------------------------------------------


class TestSynthesisCascade:
    """All six failure modes route to the heuristic synthesizer.

    The success path is covered separately (test_llm_success_path).
    """

    @pytest.fixture(autouse=True)
    def _enable_llm_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._has_llm_access", lambda: True
        )

    def _expected_heuristic(self) -> ChiefOfStaffSynthesis:
        return HeuristicChiefOfStaffSynthesizer().synthesize(
            market_identity=_market_identity(),
            per_source_signals=_two_source_rich(),
            briefing_paragraph=_example_briefing_paragraph(),
        )

    def _synthesize(self) -> ChiefOfStaffSynthesis:
        return ChiefOfStaffAgent().synthesize(
            market_identity=_market_identity(),
            per_source_signals=_two_source_rich(),
            briefing_paragraph=_example_briefing_paragraph(),
        )

    def test_route_1_llm_raises(self) -> None:
        with patch(
            "cloris.chief_of_staff.agent.opus_llm",
            side_effect=RuntimeError("network down"),
        ):
            out = self._synthesize()
        assert out.source == "deterministic"
        assert out.paragraph == self._expected_heuristic().paragraph

    def test_route_2_schema_invalid_not_dict(self) -> None:
        with patch(
            "cloris.chief_of_staff.agent.opus_llm",
            return_value="not a dict",
        ):
            out = self._synthesize()
        assert out.source == "deterministic"

    def test_route_2_schema_invalid_paragraph_short(self) -> None:
        bad = _good_llm_payload()
        bad["paragraph"] = "Short."
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"

    def test_route_2_schema_invalid_weight_not_numeric(self) -> None:
        bad = _good_llm_payload()
        bad["per_specialist_weight"]["linkedin"]["weight"] = "high"
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"

    def test_route_2_schema_invalid_weight_out_of_range(self) -> None:
        bad = _good_llm_payload()
        bad["per_specialist_weight"]["linkedin"]["weight"] = 1.5
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"

    def test_route_2_schema_invalid_rationale_empty(self) -> None:
        bad = _good_llm_payload()
        bad["per_specialist_weight"]["linkedin"]["rationale"] = ""
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"

    def test_route_2_schema_invalid_priority_missing(self) -> None:
        bad = _good_llm_payload()
        bad.pop("priority_for_principal")
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"

    @pytest.mark.parametrize("token", list(BANNED_BRIEFING_TOKENS))
    def test_route_3_banned_token(self, token: str) -> None:
        bad = _good_llm_payload()
        bad["paragraph"] = (
            f"Across LinkedIn (47 candidates, 3 saves) and GitHub "
            f"(22 candidates, 1 save). The {token} suggests review."
        )
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"

    @pytest.mark.parametrize(
        "snake_token",
        [
            "devprod_genai",
            "forward_deployed_engineering",
            "colombian_academic_ml",
            "lane_x_y_z",
        ],
    )
    def test_route_4_snake_case_token(self, snake_token: str) -> None:
        bad = _good_llm_payload()
        bad["paragraph"] = (
            f"Across LinkedIn (47 candidates, 3 saves) and GitHub "
            f"(22 candidates, 1 save), the {snake_token} lane was "
            f"strongest."
        )
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"
        # Heuristic fallback never produces snake_case either.
        assert _snake_case_token_hit(out.paragraph) is None

    def test_route_5_specialist_weight_invalid_invented_source(
        self,
    ) -> None:
        # researcher didn't contribute candidates this run; the LLM
        # invented it. Cascade must catch.
        bad = _good_llm_payload()
        bad["per_specialist_weight"]["researcher"] = {
            "weight": 0.7,
            "rationale": "Pulled some great citation graphs.",
        }
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"
        # Heuristic only references actually-contributing sources.
        assert "researcher" not in out.per_specialist_weight

    def test_route_5_specialist_weight_invalid_only_invented_keys(
        self,
    ) -> None:
        # All keys invented (defensive — both linkedin/github replaced).
        bad = _good_llm_payload()
        bad["per_specialist_weight"] = {
            "researcher": {"weight": 0.7, "rationale": "x"},
            "designer": {"weight": 0.5, "rationale": "y"},
        }
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"

    def test_route_6_containment_fails(self) -> None:
        bad = _good_llm_payload()
        bad["paragraph"] = (
            "The team turned in some interesting work this run, with "
            "things worth investigating further over the next pass."
        )
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            out = self._synthesize()
        assert out.source == "deterministic"

    def test_llm_success_path(self) -> None:
        with patch(
            "cloris.chief_of_staff.agent.opus_llm",
            return_value=_good_llm_payload(),
        ):
            out = self._synthesize()
        assert out.source == "llm"
        assert out.confidence == 1.0
        assert "47 candidates" in out.paragraph
        assert set(out.per_specialist_weight.keys()) == {"linkedin", "github"}
        assert out.priority_for_principal


# ---------------------------------------------------------------------------
# Telemetry shape — [chief-of-staff] synthesis:* line discipline
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)


class TestTelemetryLines:
    @pytest.fixture(autouse=True)
    def _enable_llm_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._has_llm_access", lambda: True
        )

    def test_emits_start_and_done_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._emit_stage", recorder
        )
        with patch(
            "cloris.chief_of_staff.agent.opus_llm",
            return_value=_good_llm_payload(),
        ):
            ChiefOfStaffAgent().synthesize(
                market_identity=_market_identity(),
                per_source_signals=_two_source_rich(),
                briefing_paragraph=_example_briefing_paragraph(),
            )
        joined = "\n".join(recorder.messages)
        assert "synthesis:start backend=ChiefOfStaffAgent" in joined
        assert "sources=github,linkedin" in joined  # sorted
        assert "synthesis:done" in joined
        assert "source=llm" in joined
        # No fallback line on the success path.
        assert "synthesis:fallback" not in joined

    def test_emits_fallback_with_route_specific_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._emit_stage", recorder
        )
        bad = _good_llm_payload()
        bad["per_specialist_weight"]["researcher"] = {
            "weight": 0.7,
            "rationale": "invented",
        }
        with patch(
            "cloris.chief_of_staff.agent.opus_llm", return_value=bad
        ):
            ChiefOfStaffAgent().synthesize(
                market_identity=_market_identity(),
                per_source_signals=_two_source_rich(),
                briefing_paragraph=_example_briefing_paragraph(),
            )
        joined = "\n".join(recorder.messages)
        assert (
            "synthesis:fallback reason=specialist_weight_invalid" in joined
        ), joined
        assert "researcher" in joined

    def test_emits_fallback_with_no_llm_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._emit_stage", recorder
        )
        # Override the autouse fixture for this test.
        monkeypatch.setattr(
            "cloris.chief_of_staff.agent._has_llm_access", lambda: False
        )
        ChiefOfStaffAgent().synthesize(
            market_identity=_market_identity(),
            per_source_signals=_two_source_rich(),
            briefing_paragraph=_example_briefing_paragraph(),
        )
        joined = "\n".join(recorder.messages)
        assert "synthesis:fallback reason=no_llm_access" in joined
        # synthesis:start NOT emitted when LLM access is missing — we
        # short-circuit before the start line.
        assert "synthesis:start" not in joined
