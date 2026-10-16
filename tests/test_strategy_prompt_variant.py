"""The strategy prompt variant, and the optional effort control beside it.

Two contracts, both narrow. The ``live`` prompt variant must render exactly
what production renders today, and an Anthropic request with no effort set
must carry exactly the kwargs it carried before effort existed.
"""

from __future__ import annotations

import pytest

import shared.config as config
import shared.llm_clients as clients
from linkedin import strategy


@pytest.fixture
def brief():
    from tests.test_lane_key_integrity import _make_brief

    return _make_brief()


def _render(monkeypatch, variant: str, brief) -> str:
    monkeypatch.setattr(config, "STRATEGY_PROMPT_VARIANT", variant)
    monkeypatch.setattr(strategy._config, "STRATEGY_PROMPT_VARIANT", variant)
    return strategy._build_strategy_system_deprescribed(brief)


def test_the_register_variant_renders_differently(monkeypatch, brief):
    live = _render(monkeypatch, "live", brief)
    register = _render(monkeypatch, "register", brief)
    assert register != live
    assert strategy._LIVE_OR_GROUP_PARAGRAPH in live
    assert strategy._LIVE_OR_GROUP_PARAGRAPH not in register
    assert strategy._REGISTER_OR_GROUP_PARAGRAPH in register


def test_the_register_variant_keeps_the_contract_byte_identical(monkeypatch, brief):
    live = _render(monkeypatch, "live", brief)
    register = _render(monkeypatch, "register", brief)
    assert strategy._DEPRESCRIBED_TASK_AND_CONTRACT in live
    assert strategy._DEPRESCRIBED_TASK_AND_CONTRACT in register


def test_an_unknown_variant_renders_live(monkeypatch, brief):
    assert _render(monkeypatch, "nonsense", brief) == _render(monkeypatch, "live", brief)


@pytest.mark.structural
def test_effort_is_not_the_thinking_parameter():
    """The Opus thinking guard is untouched by the effort control."""
    assert clients._thinking_request_kwargs("claude-opus-5") == {}
    assert clients._output_config_kwargs(None) == {}
    assert clients._output_config_kwargs("max") == {"output_config": {"effort": "max"}}
    with pytest.raises(ValueError):
        clients._output_config_kwargs("enormous")


def _captured_request(monkeypatch, **kwargs) -> dict:
    seen: dict = {}

    class _Message:
        content = [type("Block", (), {"text": "{}", "type": "text"})()]
        stop_reason = "end_turn"
        usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()

    monkeypatch.setattr(clients, "get_llm_client", lambda *a, **k: object())
    monkeypatch.setattr(
        clients,
        "_anthropic_message",
        lambda client, **request: (seen.update(request), _Message())[1],
    )
    monkeypatch.setattr(clients.config, "ANTHROPIC_API_KEY", "test-key")
    clients.opus_llm("system", "user", expect_json=False, **kwargs)
    return seen


def test_an_unset_effort_leaves_the_request_unchanged(monkeypatch):
    request = _captured_request(monkeypatch)
    assert "output_config" not in request
    assert "thinking" not in request
    assert set(request) == {"model", "max_tokens", "system", "messages"}


def test_a_set_effort_adds_only_the_output_config(monkeypatch):
    baseline = set(_captured_request(monkeypatch))
    request = _captured_request(monkeypatch, effort="max")
    assert request["output_config"] == {"effort": "max"}
    assert set(request) - baseline == {"output_config"}
    assert "thinking" not in request


# The two money paths of tools/strategy_prompt_experiment.py, neither of which
# had a test when the experiment was handed over: the client wrap that carries
# `effort` into the real formation call, and the cost capture that reads what
# the client actually wrote back.


class _StubPlan:
    def to_dict(self) -> dict:
        return {"generated_strings": [{"boolean": '"research engineer" AND "rlhf"'}]}


def _stub_form_strategy(brief, kit_strings, prior_run_data=None, **kwargs):
    """Stand in for form_strategy, calling through the same way it does.

    It passes a usage_context of its own, exactly as the real formation call
    does, so the assertion that the tool's context reaches the client is a
    real one: the wrap has to override this dict, not sit beside it.
    """
    strategy.opus_llm(
        "system",
        "user",
        expect_json=True,
        max_tokens=98304,
        usage_context={"stage": "linkedin_strategy_form", "brief_id": "stub"},
        model_name="claude-opus-5",
    )
    return _StubPlan()


def _recording_client(recorded: list):
    """A client that records its kwargs and records usage the way Opus does."""

    def _client(system_prompt, user_prompt, **kwargs):
        recorded.append(kwargs)
        from shared.llm_usage import record_llm_usage

        record_llm_usage(
            provider="anthropic",
            model=kwargs.get("model_name") or "claude-opus-5",
            usage={
                "input_tokens": 12,
                "output_tokens": 3400,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 22000,
            },
            request={"system_prompt_chars": len(system_prompt)},
            usage_context=kwargs.get("usage_context"),
        )
        return {"generated_strings": []}

    return _client


def test_the_experiment_carries_effort_into_every_sample(monkeypatch, tmp_path):
    from tools import strategy_prompt_experiment as experiment

    monkeypatch.setattr(config, "STRATEGY_PROMPT_VARIANT", "live")
    recorded: list = []
    client = _recording_client(recorded)
    monkeypatch.setattr(strategy, "opus_llm", client)
    monkeypatch.setattr(strategy, "form_strategy", _stub_form_strategy)

    plans, usage = experiment.run_samples(
        object(),
        model="claude-opus-5",
        variant="register",
        samples=2,
        effort="max",
        out_dir=tmp_path,
    )

    assert len(plans) == 2 and len(usage) == 2
    assert [call.get("effort") for call in recorded] == ["max", "max"]
    # One attempt per sample: the client retries five times by default,
    # which under a call cap buys four more calls nobody asked for.
    assert [call.get("max_attempts") for call in recorded] == [1, 1]
    # The tool's context wins over formation's own, or the cost capture reads
    # a dict nobody filled.
    assert [call["usage_context"]["stage"] for call in recorded] == [
        "prompt_experiment",
        "prompt_experiment",
    ]
    # One fresh context per call: a shared dict leaks one sample's markers
    # into the next (shared/llm_clients.py:281).
    assert recorded[0]["usage_context"] is not recorded[1]["usage_context"]
    assert strategy.opus_llm is client


def test_the_experiment_restores_the_binding_when_a_sample_raises(
    monkeypatch, tmp_path
):
    from tools import strategy_prompt_experiment as experiment

    monkeypatch.setattr(config, "STRATEGY_PROMPT_VARIANT", "live")
    client = _recording_client([])
    monkeypatch.setattr(strategy, "opus_llm", client)

    def _blow_up(*args, **kwargs):
        raise RuntimeError("formation blew up")

    monkeypatch.setattr(strategy, "form_strategy", _blow_up)

    with pytest.raises(RuntimeError, match="formation blew up"):
        experiment.run_samples(
            object(),
            model="claude-opus-5",
            variant="live",
            samples=1,
            effort="max",
            out_dir=tmp_path,
        )
    assert strategy.opus_llm is client


def _filled_usage_context(monkeypatch) -> dict:
    """Run one real cached call over a fake transport, return its context."""

    class _Message:
        content = [type("Block", (), {"text": "{}", "type": "text"})()]
        stop_reason = "end_turn"
        usage = type(
            "U",
            (),
            {
                "input_tokens": 12,
                "output_tokens": 3400,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 22000,
            },
        )()

    monkeypatch.setattr(clients, "get_llm_client", lambda *a, **k: object())
    monkeypatch.setattr(clients, "_anthropic_message", lambda client, **r: _Message())
    monkeypatch.setattr(clients.config, "ANTHROPIC_API_KEY", "test-key")
    context: dict = {"stage": "prompt_experiment"}
    clients.opus_llm_cached(
        "system",
        "user",
        expect_json=False,
        usage_context=context,
        model_name="claude-opus-5",
        effort="max",
    )
    return context


def test_the_cost_capture_reads_the_keys_the_client_actually_fills(monkeypatch):
    from tools import strategy_prompt_experiment as experiment

    context = _filled_usage_context(monkeypatch)

    # The client writes no token counts onto the dict it was handed. It writes
    # one receipt, and the tokens and the priced cost live inside it.
    assert "input_tokens" not in context
    assert "_llm_receipt" in context

    row = experiment._usage_row(context)
    assert row["input_tokens"] == 12
    assert row["output_tokens"] == 3400
    assert row["cache_creation_input_tokens"] == 22000
    assert row["model"] == "claude-opus-5"
    # 12 in, 3400 out, 22000 cache-written, at the repo's own Opus rates.
    assert row["cost_usd"] == pytest.approx(0.22256, abs=1e-6)


def test_a_sample_whose_call_never_landed_stops_the_run():
    from tools import strategy_prompt_experiment as experiment

    with pytest.raises(RuntimeError, match="no LLM receipt"):
        experiment._usage_row({"stage": "prompt_experiment"})


def _errored_usage_context(monkeypatch) -> dict:
    """Drive one real cached call whose transport fails, return its context."""

    def _blow_up(client, **request):
        raise RuntimeError("Request timed out or interrupted")

    monkeypatch.setattr(clients, "get_llm_client", lambda *a, **k: object())
    monkeypatch.setattr(clients, "_anthropic_message", _blow_up)
    monkeypatch.setattr(clients.config, "ANTHROPIC_API_KEY", "test-key")
    context: dict = {"stage": "prompt_experiment"}
    with pytest.raises(RuntimeError):
        clients.opus_llm_cached(
            "system",
            "user",
            expect_json=False,
            usage_context=context,
            model_name="claude-opus-5",
            effort="max",
            max_attempts=1,
        )
    return context


def test_a_failed_call_stops_the_run_though_it_left_a_receipt(monkeypatch):
    """A receipt's presence is not success: the error path writes one too."""
    from tools import strategy_prompt_experiment as experiment

    context = _errored_usage_context(monkeypatch)
    assert context["_llm_receipt"]["actual_status"] == "error"

    with pytest.raises(RuntimeError, match="did not succeed"):
        experiment._usage_row(context)


def test_a_sample_that_produced_no_strings_stops_the_run(monkeypatch, tmp_path):
    """A truncated call leaves an ``ok`` receipt and an empty portfolio.

    Only the string count catches it: the receipt is written before the
    stop_reason check, so a response that ran into max_tokens is priced and
    recorded as a success. One such Fable sample cost $5.02 and returned
    nothing on 2026-09-21, and the run went on to the next sample.
    """
    from tools import strategy_prompt_experiment as experiment

    monkeypatch.setattr(config, "STRATEGY_PROMPT_VARIANT", "live")
    client = _recording_client([])
    monkeypatch.setattr(strategy, "opus_llm", client)

    class _EmptyPlan:
        def to_dict(self) -> dict:
            return {"generated_strings": []}

    def _empty_form_strategy(brief, kit_strings, prior_run_data=None, **kwargs):
        strategy.opus_llm(
            "system",
            "user",
            usage_context={"stage": "linkedin_strategy_form"},
            model_name="claude-fable-5-1",
        )
        return _EmptyPlan()

    monkeypatch.setattr(strategy, "form_strategy", _empty_form_strategy)

    with pytest.raises(RuntimeError, match="no strings"):
        experiment.run_samples(
            object(),
            model="claude-fable-5-1",
            variant="live",
            samples=3,
            effort="max",
            out_dir=tmp_path,
        )
    # Stopped on the first empty sample, not after all three.
    assert (tmp_path / "claude-fable-5-1-live-0.json").is_file()
    assert not (tmp_path / "claude-fable-5-1-live-1.json").exists()
    assert strategy.opus_llm is client


# The CLI seat. The experiment re-runs on subscription usage rather than API
# credits, which means the strategy tier has to reach claude_cli_chat and the
# tool has to be able to hand it a timeout long enough for a formation call.


def test_the_strategy_tier_routes_a_claude_cli_model_to_the_cli(monkeypatch):
    """`form_strategy` calls opus_llm_cached, so that is the route to pin."""
    seen: dict = {}

    def _fake_cli_chat(**kwargs):
        seen.update(kwargs)
        if kwargs.get("usage_context") is not None:
            from shared.llm_usage import record_llm_usage

            record_llm_usage(
                provider="claude_cli",
                model=kwargs["model"],
                usage={"input_tokens": 7, "output_tokens": 9},
                request={},
                usage_context=kwargs["usage_context"],
            )
        return "{}"

    monkeypatch.setattr(clients, "claude_cli_chat", _fake_cli_chat)
    context: dict = {"stage": "prompt_experiment"}
    clients.opus_llm_cached(
        "system",
        "user",
        expect_json=False,
        usage_context=context,
        model_name="claude-cli:claude-opus-5",
        timeout_seconds=1500,
    )

    # The prefix selects the transport and is stripped before the CLI sees it.
    assert seen["model"] == "claude-opus-5"
    assert seen["timeout_seconds"] == 1500
    assert seen["system_prompt"] == "system"
    # No Anthropic API key is consulted on this path.
    assert "output_config" not in seen and "thinking" not in seen
    # And it is the binding form_strategy actually calls.
    assert strategy.opus_llm is clients.opus_llm_cached


def test_the_experiment_hands_the_client_its_timeout(monkeypatch, tmp_path):
    from tools import strategy_prompt_experiment as experiment

    monkeypatch.setattr(config, "STRATEGY_PROMPT_VARIANT", "live")
    recorded: list = []
    client = _recording_client(recorded)
    monkeypatch.setattr(strategy, "opus_llm", client)
    monkeypatch.setattr(strategy, "form_strategy", _stub_form_strategy)

    experiment.run_samples(
        object(),
        model="claude-cli:claude-opus-5",
        variant="live",
        samples=1,
        effort=None,
        timeout_seconds=1500,
        out_dir=tmp_path,
    )

    assert recorded[0]["timeout_seconds"] == 1500
    # Default effort: the seat takes no effort argument, so none is sent.
    assert recorded[0].get("effort") is None
    # A model id with a colon must not become a colon in a filename.
    assert (tmp_path / "claude-cli-claude-opus-5-live-0.json").is_file()
