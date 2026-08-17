"""Langfuse observability layer — Phase 1 acceptance tests.

Pins the contract the Phase 1 wrap targets establish:

1. ``@observe()`` wrappers don't change call semantics (input /
   output identical with vs without Langfuse keys).
2. The no-op stub fires when keys are absent / disabled.
3. Cascade-route attribution surfaces correctly through the
   ``_emit_stage`` helpers in ``cloris/chief_of_staff/agent.py``,
   ``market_intelligence/briefing_polish.py``, and
   ``market_intelligence/brief_polish.py`` — i.e., a stage log line
   carrying ``fallback reason=schema_invalid`` triggers
   ``update_current_observation(metadata={"cascade.fallback_reason": "schema_invalid"})``.
4. Vision-eval guard attributes (``vision.layer1_*`` /
   ``vision.layer4_hard_reject`` / ``vision.fallback_reason``)
   surface from a fixture vision-eval result.
5. Byte-equivalence of JSONL ``run_log`` records when Langfuse is
   wired vs disabled — the Phase 1 cost-source parity check.

The Langfuse SDK itself is NOT installed in CI, so all tests here
exercise the null-stub / passthrough path. The contract pinned is
"nothing changes when Langfuse is wired absent OR disabled" —
Langfuse-cloud-active integration smoke tests live elsewhere
(``docs/`` or operator runbook), not in unit tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from shared.observability import (
    get_current_observation_id,
    get_current_trace_id,
    get_trace_url,
    is_active,
    observe,
    update_current_observation,
)
from shared.observability.langfuse_client import (
    get_client,
    reset_for_testing,
)


@pytest.fixture(autouse=True)
def _fresh_singleton():
    reset_for_testing()
    yield
    reset_for_testing()


# ---------------------------------------------------------------------------
# (1) Decorator semantics unchanged with vs without keys
# ---------------------------------------------------------------------------


class TestDecoratorSemantics:
    def test_observe_disables_sdk_input_output_capture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        decorator_calls: list[tuple[tuple, dict]] = []
        fake_langfuse = ModuleType("langfuse")

        def _real_observe(*args, **kwargs):
            decorator_calls.append((args, kwargs))
            return lambda fn: fn

        fake_langfuse.observe = _real_observe  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)

        with patch("shared.observability.is_active", return_value=True):

            @observe(name="test_fn", as_type="generation")
            def add(a: int, b: int) -> int:
                return a + b

        assert add(2, 3) == 5
        assert decorator_calls == [
            (
                (),
                {
                    "name": "test_fn",
                    "as_type": "generation",
                    "capture_input": False,
                    "capture_output": False,
                },
            )
        ]

    def test_observe_passthrough_when_keys_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_DISABLE", raising=False)
        reset_for_testing()

        @observe(name="test_fn")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5
        assert add.__name__ == "add"

    def test_observe_preserves_kwargs_and_default_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        reset_for_testing()

        @observe(name="kw_fn")
        def kw_fn(x, *, y=5, z=10):
            return x + y + z

        assert kw_fn(1) == 16
        assert kw_fn(1, y=20) == 31
        assert kw_fn(1, z=100) == 106

    def test_observe_preserves_exceptions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        reset_for_testing()

        @observe(name="raises_fn")
        def boom() -> None:
            raise ValueError("intended")

        with pytest.raises(ValueError, match="intended"):
            boom()

    def test_record_llm_usage_keeps_operational_metadata(
        self, tmp_path: Path
    ) -> None:
        from shared.llm_usage import llm_usage_session, record_llm_usage

        with patch("shared.observability.update_current_observation") as update:
            with llm_usage_session(
                tmp_path / "usage.jsonl",
                brief_id="brief-metadata",
            ):
                record_llm_usage(
                    provider="anthropic",
                    model="claude-opus",
                    usage={
                        "input_tokens": 1000,
                        "output_tokens": 500,
                        "cache_read_input_tokens": 200,
                        "cache_creation_input_tokens": 50,
                    },
                    usage_context={"stage": "judge.facial"},
                )

        observation = update.call_args.kwargs
        assert observation["model"] == "claude-opus"
        assert observation["usage"] == {
            "input": 1000,
            "output": 500,
            "total": 1500,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 50,
        }
        assert observation["metadata"]["stage"] == "judge.facial"
        assert observation["metadata"]["brief_id"] == "brief-metadata"
        assert observation["metadata"]["estimated_cost_usd"] == 0.017913


# ---------------------------------------------------------------------------
# (2) No-op stub when keys absent OR LANGFUSE_DISABLE=1
# ---------------------------------------------------------------------------


class TestNullStub:
    def test_get_client_returns_null_when_keys_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        reset_for_testing()

        client = get_client()
        assert getattr(client, "is_null", False) is True
        assert is_active() is False

    def test_get_client_returns_null_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake_pub")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake_sec")
        monkeypatch.setenv("LANGFUSE_DISABLE", "1")
        reset_for_testing()

        assert is_active() is False

    def test_update_current_observation_is_noop_on_null_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        reset_for_testing()

        # Calling update_current_observation should never raise even
        # though there's no active span / no real client wired.
        update_current_observation(metadata={"key": "value"})
        update_current_observation()  # empty kwargs

    def test_trace_helpers_return_none_on_null_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        reset_for_testing()

        assert get_current_trace_id() is None
        assert get_current_observation_id() is None
        assert get_trace_url() is None

    def test_secret_key_not_echoed_in_repr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LANGFUSE_SECRET_KEY must never appear in the client's
        repr — that's our pinned secret-handling constraint."""

        secret = "super-sensitive-secret-do-not-leak"
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake_pub")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret)
        monkeypatch.delenv("LANGFUSE_DISABLE", raising=False)
        reset_for_testing()

        client = get_client()
        assert secret not in repr(client)

    def test_trace_helpers_proxy_to_active_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeClient:
            is_null = False

            def get_current_trace_id(self) -> str:
                return "trace-123"

            def get_current_observation_id(self) -> str:
                return "obs-456"

            def get_trace_url(self, *, trace_id: str | None = None) -> str:
                return f"https://langfuse.test/{trace_id or 'trace-123'}"

        import shared.observability.langfuse_client as client_module

        monkeypatch.setattr(client_module, "_CLIENT", _FakeClient())
        monkeypatch.setattr(client_module, "_INSTANTIATED", True)

        assert get_current_trace_id() == "trace-123"
        assert get_current_observation_id() == "obs-456"
        assert get_trace_url() == "https://langfuse.test/trace-123"
        assert get_trace_url("trace-999") == "https://langfuse.test/trace-999"


# ---------------------------------------------------------------------------
# (3) Cascade-route attribution via _emit_stage
# ---------------------------------------------------------------------------


class TestCascadeAttribution:
    def test_chief_of_staff_emit_stage_pushes_cascade_attribute(self) -> None:
        """The chief-of-staff ``_emit_stage`` parses out the
        ``fallback reason=<reason>`` token and pushes it as a
        cascade.fallback_reason attribute. Verified by intercepting
        the update_current_observation call."""

        captured: list[dict] = []

        from cloris.chief_of_staff import agent

        def _capture(**kwargs):
            captured.append(kwargs)

        with patch(
            "shared.observability.update_current_observation",
            _capture,
        ):
            agent._emit_stage(
                "synthesis:fallback reason=schema_invalid detail=bad_json"
            )

        assert captured == [
            {"metadata": {"cascade.fallback_reason": "schema_invalid"}}
        ]

    def test_chief_of_staff_emit_stage_does_not_emit_for_non_fallback(
        self,
    ) -> None:
        captured: list[dict] = []

        from cloris.chief_of_staff import agent

        def _capture(**kwargs):
            captured.append(kwargs)

        with patch(
            "shared.observability.update_current_observation",
            _capture,
        ):
            agent._emit_stage("synthesis:start backend=ChiefOfStaffAgent")
            agent._emit_stage("dispatch:done elapsed_ms=12 brief=foo")

        assert captured == []

    def test_briefing_polish_emit_stage_pushes_cascade_attribute(self) -> None:
        """Same contract for the market-intel polish backend's
        ``_emit_stage``."""

        captured: list[dict] = []

        from market_intelligence import briefing_polish

        def _capture(**kwargs):
            captured.append(kwargs)

        with patch(
            "shared.observability.update_current_observation",
            _capture,
        ):
            briefing_polish._emit_stage(
                "reflection.polish:fallback reason=banned_token "
                "token='hypothesis'"
            )

        assert captured == [
            {"metadata": {"cascade.fallback_reason": "banned_token"}}
        ]

    def test_brief_polish_emit_stage_pushes_cascade_attribute(self) -> None:
        """And for the intake-time brief-polish ``_emit_stage``."""

        captured: list[dict] = []

        from market_intelligence import brief_polish

        def _capture(**kwargs):
            captured.append(kwargs)

        with patch(
            "shared.observability.update_current_observation",
            _capture,
        ):
            brief_polish._emit_stage(
                "brief.polish:fallback reason=llm_raise exc=ConnectionError"
            )

        assert captured == [
            {"metadata": {"cascade.fallback_reason": "llm_raise"}}
        ]


# ---------------------------------------------------------------------------
# (4) Vision-eval guard attributes
# ---------------------------------------------------------------------------


class TestVisionGuardAttribution:
    def test_emit_vision_attributes_for_clean_eval(self) -> None:
        """A successful eval (no fallback_reason, all principles
        anchor-consistent) emits layer1=true, layer2=true,
        layer3=1.0, layer4=false."""

        from designer.vision_evaluation import (
            VisionEvaluationResult,
            VisualJudgment,
            VisualJudgmentPrinciple,
            _emit_vision_guard_attributes,
        )

        judgment = VisualJudgment(
            model="gemini-2.5-pro",
            principles=(
                VisualJudgmentPrinciple(
                    name="hierarchy",
                    score=2,
                    anchor="good",
                    reasoning="Strong visual hierarchy.",
                    image_ids=(0, 1),
                    anchor_consistency_pass=True,
                ),
                VisualJudgmentPrinciple(
                    name="typography",
                    score=3,
                    anchor="excellent",
                    reasoning="Crisp type system.",
                    image_ids=(0,),
                    anchor_consistency_pass=True,
                ),
            ),
            overall_verdict="yes",
            overall_confidence=0.9,
            fallback_reason="",
            cost_estimate_usd=0.05,
        )
        result = VisionEvaluationResult(
            judgment=judgment, asset_references=(), raw_response={}
        )

        captured: list[dict] = []

        def _capture(**kwargs):
            captured.append(kwargs)

        with patch(
            "shared.observability.update_current_observation",
            _capture,
        ):
            _emit_vision_guard_attributes(result)

        assert len(captured) == 1
        meta = captured[0]["metadata"]
        assert meta["vision.layer1_schema_validity"] is True
        assert meta["vision.layer2_image_grounding"] is True
        assert meta["vision.layer3_anchor_consistency"] == 1.0
        assert meta["vision.layer4_hard_reject"] is False
        assert meta["vision.fallback_reason"] == ""
        assert meta["vision.model"] == "gemini-2.5-pro"

    def test_emit_vision_attributes_for_layer1_failure(self) -> None:
        """A schema-invalid fallback flips layer1 to false."""

        from designer.vision_evaluation import (
            VisionEvaluationResult,
            VisualJudgment,
            _emit_vision_guard_attributes,
        )

        judgment = VisualJudgment(
            model="gemini-2.5-pro",
            principles=(),
            overall_verdict="borderline",
            overall_confidence=0.0,
            fallback_reason="schema_invalid:no_principles",
            cost_estimate_usd=0.0,
        )
        result = VisionEvaluationResult(
            judgment=judgment, asset_references=(), raw_response={}
        )

        captured: list[dict] = []

        def _capture(**kwargs):
            captured.append(kwargs)

        with patch(
            "shared.observability.update_current_observation",
            _capture,
        ):
            _emit_vision_guard_attributes(result)

        meta = captured[0]["metadata"]
        assert meta["vision.layer1_schema_validity"] is False
        assert meta["vision.fallback_reason"] == "schema_invalid:no_principles"

    def test_emit_vision_attributes_for_layer4_hard_reject(self) -> None:
        """A hard-reject fallback flips layer4 to true (a SUCCESSFUL
        policy outcome)."""

        from designer.vision_evaluation import (
            VisionEvaluationResult,
            VisualJudgment,
            _emit_vision_guard_attributes,
        )

        judgment = VisualJudgment(
            model="gemini-2.5-pro",
            principles=(),
            overall_verdict="no",
            overall_confidence=1.0,
            fallback_reason="hard_reject:offensive_content",
            cost_estimate_usd=0.05,
        )
        result = VisionEvaluationResult(
            judgment=judgment, asset_references=(), raw_response={}
        )

        captured: list[dict] = []

        def _capture(**kwargs):
            captured.append(kwargs)

        with patch(
            "shared.observability.update_current_observation",
            _capture,
        ):
            _emit_vision_guard_attributes(result)

        meta = captured[0]["metadata"]
        assert meta["vision.layer4_hard_reject"] is True
        assert meta["vision.layer1_schema_validity"] is True
        assert meta["vision.layer2_image_grounding"] is True


# ---------------------------------------------------------------------------
# (5) Byte-equivalence of JSONL run_log records
# ---------------------------------------------------------------------------


class TestByteEquivalence:
    def test_record_llm_usage_jsonl_byte_equivalent_with_or_without_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audit Phase 1 verification step (5): ``record_llm_usage``'s
        JSONL output must be byte-equivalent whether Langfuse is
        wired or not (modulo timestamps, which are the documented
        allowed delta).

        We run record_llm_usage twice with identical inputs — once
        with keys absent (LANGFUSE_DISABLE=1), once with keys absent.
        Both runs MUST produce identical JSONL records byte-for-byte
        except the ``timestamp`` field. A drift means the Langfuse
        path leaked into the JSONL sink.
        """

        from shared.llm_usage import llm_usage_session, record_llm_usage

        log_path_a = tmp_path / "log_disabled.jsonl"
        log_path_b = tmp_path / "log_keys_absent.jsonl"

        # First run: Langfuse explicitly disabled.
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake_pub")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake_sec")
        monkeypatch.setenv("LANGFUSE_DISABLE", "1")
        reset_for_testing()

        with llm_usage_session(log_path_a, brief_id="brief-byteq", source="linkedin"):
            record_llm_usage(
                provider="anthropic",
                model="claude-opus",
                usage={
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cache_read_input_tokens": 200,
                },
                request={
                    "system_prompt_chars": 2000,
                    "stop_reason": "end_turn",
                },
                usage_context={"stage": "judge.facial"},
            )

        # Second run: keys absent.
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_DISABLE", raising=False)
        reset_for_testing()

        with llm_usage_session(log_path_b, brief_id="brief-byteq", source="linkedin"):
            record_llm_usage(
                provider="anthropic",
                model="claude-opus",
                usage={
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cache_read_input_tokens": 200,
                },
                request={
                    "system_prompt_chars": 2000,
                    "stop_reason": "end_turn",
                },
                usage_context={"stage": "judge.facial"},
            )

        # Diff the two JSONL files. Same shape + same values modulo
        # the timestamp delta documented in the verification step.
        record_a = json.loads(log_path_a.read_text().strip())
        record_b = json.loads(log_path_b.read_text().strip())

        # Documented allowed delta: timestamp is wall-clock-derived.
        record_a.pop("timestamp")
        record_b.pop("timestamp")
        # Second allowed delta (P10 actuate #4): the receipt's created_at is
        # now real wall-clock time too (previously pinned to a fake epoch,
        # which is why this test never had to exclude it before), and
        # receipt_id is content-addressed OVER created_at, so it legitimately
        # differs between the two calls as well.
        record_a["receipt"].pop("created_at")
        record_b["receipt"].pop("created_at")
        record_a["receipt"].pop("receipt_id")
        record_b["receipt"].pop("receipt_id")

        assert record_a == record_b, (
            "JSONL records diverged between disabled vs keys-absent "
            "Langfuse paths — Langfuse sink leaked into the JSONL "
            "writer."
        )
        # Sanity: both records carry the documented stable fields.
        assert record_a["provider"] == "anthropic"
        assert record_a["model"] == "claude-opus"
        assert record_a["input_tokens"] == 1000
        assert record_a["estimated_cost_usd"] is not None  # cost mapping fired
        assert record_a["stage"] == "judge.facial"
