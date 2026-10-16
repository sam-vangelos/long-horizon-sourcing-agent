"""Strategy formation survives one unparseable reply and keeps the evidence.

2026-09-30 field-engineering launch: formation on claude-opus-5 ended
``end_turn`` after 25,066 output tokens, ``_parse_json_response`` could not
parse the reply, and ``form_strategy`` returned zero strings. The run executed
nothing and was rejected as ``green_but_useless``; the reply itself was never
saved. These tests drive ``form_strategy`` / ``adapt_after_block`` through
their production entry with a fake model whose replies go through the REAL
parser, so the failure is production-shaped, not a hand-built exception.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from linkedin.strategy import adapt_after_block, form_strategy
from shared.brief_loader import Brief
from shared.llm_clients import _parse_json_response
from shared.schemas import AdaptationResponse, BlockReport, Progress, SearchString


PREFIX = "Could not parse JSON from LLM response: "

# Fails every repair the parser tries: no closing bracket for the array, and
# the force-close fallback still leaves an unbalanced object.
MALFORMED = '{ "architecture": "title_first", "generated_strings": [1, 2 }'


def _brief() -> Brief:
    return Brief(
        id="parse-retry-test",
        role_title="Solutions Engineer",
        role_description="Customer-facing engineer who ships integrations.",
        kit_url="",
        linkedin_project="proj-parse-retry",
        linkedin_project_id="",
        minimum_bar="Has shipped customer integrations.",
        archetypes=[{"name": "Integration builder"}],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )


def _plan_text() -> str:
    return json.dumps(
        {
            "architecture": "title_first",
            "architecture_rationale": "mock",
            "architecture_success_criteria": [],
            "architecture_pivot_triggers": [],
            "strategy_rationale": "mock",
            "generated_strings": [
                {
                    "boolean": '("solutions engineer") AND ("api integration")',
                    "rationale": "Title doorway into integration builders",
                    "vocabulary_sources": "mock",
                    "family_key": "se_integration",
                    "novelty_bucket": "canonical",
                    "domain_lane": "general",
                },
                {
                    "boolean": '("forward deployed engineer")',
                    "rationale": "Adjacent customer-facing title",
                    "vocabulary_sources": "mock",
                    "family_key": "fde",
                    "novelty_bucket": "edge_case",
                    "domain_lane": "general",
                },
            ],
            "coverage_gaps": [],
            "noise_predictions": [],
        }
    )


class _ScriptedModel:
    """Fake ``opus_llm``: each call parses the next scripted reply for real."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, system_prompt, user_prompt, expect_json=True, **kwargs):
        self.calls.append(
            {
                "system": system_prompt,
                "user": user_prompt,
                "usage_context": dict(kwargs.get("usage_context") or {}),
                "max_tokens": kwargs.get("max_tokens"),
                "model_name": kwargs.get("model_name"),
            }
        )
        return _parse_json_response(self.replies.pop(0))


def _forensics(state_dir: Path) -> list[Path]:
    root = state_dir / "forensics"
    return sorted(root.rglob("*unparseable*")) if root.exists() else []


# --- the parse exception -------------------------------------------------


def test_parse_exception_stays_a_runtime_error_and_carries_the_full_reply():
    from shared.llm_clients import UnparseableLLMResponseError

    long_reply = MALFORMED[:-1] + ", " + ", ".join(['"pad"'] * 400) + " }"
    assert len(long_reply) > 2000
    with pytest.raises(RuntimeError) as caught:
        _parse_json_response(long_reply)

    exc = caught.value
    assert isinstance(exc, UnparseableLLMResponseError)
    # The message is byte-identical to before: prefix + first 500 chars.
    assert str(exc) == PREFIX + long_reply[:500]
    # The full reply survives, not just the 500-char excerpt.
    assert exc.raw_text == long_reply
    assert isinstance(exc.decode_error, json.JSONDecodeError)
    assert exc.json_text == long_reply
    try:
        json.loads(long_reply)
    except json.JSONDecodeError as direct:
        assert exc.decode_error.pos == direct.pos
        assert (exc.decode_error.lineno, exc.decode_error.colno) == (direct.lineno, direct.colno)


def test_parse_exception_position_indexes_the_fence_stripped_text():
    from shared.llm_clients import UnparseableLLMResponseError

    fenced = "```json\n" + MALFORMED + "\n```"
    with pytest.raises(UnparseableLLMResponseError) as caught:
        _parse_json_response(fenced)
    exc = caught.value
    assert exc.raw_text == fenced
    assert exc.json_text == MALFORMED
    try:
        json.loads(MALFORMED)
    except json.JSONDecodeError as direct:
        assert exc.decode_error.pos == direct.pos


def test_parser_behaviour_is_unchanged_for_parseable_and_repairable_replies():
    assert _parse_json_response('{"a": 1}') == {"a": 1}
    assert _parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_response('prose {"a": 1} prose') == {"a": 1}
    # The force-close repair still salvages a truncated object.
    assert _parse_json_response('{"a": "b", "c": "d"') == {"a": "b", "c": "d"}


# --- formation retry ------------------------------------------------------


def test_formation_retries_once_on_unparseable_reply_and_returns_the_plan(tmp_path, capsys):
    model = _ScriptedModel([MALFORMED, _plan_text()])
    with patch("linkedin.strategy.opus_llm", side_effect=model):
        plan = form_strategy(_brief(), [], prior_run_data=None, state_dir=tmp_path)

    assert len(model.calls) == 2
    # Same prompt, same call shape, on the retry.
    assert model.calls[0]["system"] == model.calls[1]["system"]
    assert model.calls[0]["user"] == model.calls[1]["user"]
    assert model.calls[0]["max_tokens"] == model.calls[1]["max_tokens"] == 98304
    # The first attempt's usage context is today's, byte for byte; the retry
    # is labelled so the cost log can tell the two calls apart.
    assert model.calls[0]["usage_context"] == {
        "stage": "linkedin_strategy_form",
        "brief_id": "parse-retry-test",
    }
    assert model.calls[1]["usage_context"] == {
        "stage": "linkedin_strategy_form",
        "brief_id": "parse-retry-test",
        "parse_retry_attempt": 2,
    }
    assert [s["family_key"] for s in plan.generated_strings] == ["se_integration", "fde"]

    out = capsys.readouterr()
    attempt_lines = [line for line in out.out.splitlines() if "not valid JSON" in line]
    assert len(attempt_lines) == 1
    assert "1/2" in attempt_lines[0] and "retrying once" in attempt_lines[0]
    assert "{" not in attempt_lines[0]
    assert "Strategy formation failed" not in out.err
    assert len(_forensics(tmp_path)) == 1


def test_formation_unparseable_twice_keeps_todays_failure_and_saves_both_replies(tmp_path, capsys):
    # The second reply is long and multi-line, like the live 25k-token one, so
    # a saver that truncated to the 500-char message excerpt would show.
    second = (
        MALFORMED.replace("title_first", "sniper")[:-1]
        + ",\n"
        + ",\n".join(['"pad"'] * 600)
        + "\n}"
    )
    assert len(second) > 4000
    model = _ScriptedModel([MALFORMED, second])
    with patch("linkedin.strategy.opus_llm", side_effect=model):
        plan = form_strategy(_brief(), [], prior_run_data=None, state_dir=tmp_path)

    assert len(model.calls) == 2
    assert plan.generated_strings == []
    assert plan.strategy_rationale.startswith("Strategy formation failed")

    out = capsys.readouterr()
    assert f"[warn] Strategy formation failed ({PREFIX}{second[:500]}) — no strings to execute" in out.err
    attempt_lines = [line for line in out.out.splitlines() if "not valid JSON" in line]
    assert len(attempt_lines) == 2
    assert "1/2" in attempt_lines[0] and "retrying once" in attempt_lines[0]
    assert "2/2" in attempt_lines[1] and "giving up" in attempt_lines[1]

    files = _forensics(tmp_path)
    assert [f.name.rsplit("-", 1)[-1] for f in files] == ["attempt1.txt", "attempt2.txt"]
    for path, reply in zip(files, (MALFORMED, second)):
        assert path.parent == tmp_path / "forensics" / "strategy"
        assert path.name.startswith("strategy-unparseable-")
        body = path.read_text(encoding="utf-8")
        # The full raw reply is the file's tail, verbatim.
        assert body.endswith("\n" + reply)
        try:
            json.loads(reply)
        except json.JSONDecodeError as direct:
            assert direct.msg in body
            assert f"char {direct.pos}" in body
            assert f"line {direct.lineno}" in body
            assert f"column {direct.colno}" in body
        assert "linkedin_strategy_form" in body


def test_formation_unparseable_twice_without_state_dir_still_retries_and_never_raises(capsys):
    model = _ScriptedModel([MALFORMED, MALFORMED])
    with patch("linkedin.strategy.opus_llm", side_effect=model):
        plan = form_strategy(_brief(), [], prior_run_data=None)

    assert len(model.calls) == 2
    assert plan.generated_strings == []
    assert "not saved" in capsys.readouterr().out


@pytest.mark.parametrize("breakage", ["state_dir_is_a_file", "write_raises"])
def test_forensics_write_failure_never_raises_into_formation(tmp_path, breakage, capsys):
    if breakage == "state_dir_is_a_file":
        state_dir = tmp_path / "not-a-dir"
        state_dir.write_text("occupied", encoding="utf-8")
        context = contextlib.nullcontext()
    else:
        state_dir = tmp_path

        def _explode(*_args, **_kwargs):
            raise OSError("disk full")

        context = patch.object(Path, "write_text", _explode)

    model = _ScriptedModel([MALFORMED, _plan_text()])
    with patch("linkedin.strategy.opus_llm", side_effect=model), context:
        plan = form_strategy(_brief(), [], prior_run_data=None, state_dir=state_dir)

    assert len(model.calls) == 2
    assert len(plan.generated_strings) == 2
    attempt_line = next(
        line for line in capsys.readouterr().out.splitlines() if "not valid JSON" in line
    )
    assert "not saved" in attempt_line


def test_formation_does_not_retry_other_failures():
    """Only the parse exception earns a retry: a truncation or a plain
    RuntimeError carrying the old message (the salvage fixtures) is one call."""
    for error in (
        RuntimeError("Opus response truncated: stop_reason=max_tokens."),
        RuntimeError(PREFIX + "malformed"),
    ):
        calls = []

        def _raise(*args, _error=error, **kwargs):
            calls.append(1)
            raise _error

        with patch("linkedin.strategy.opus_llm", side_effect=_raise):
            plan = form_strategy(_brief(), [], prior_run_data=None)
        assert len(calls) == 1
        assert plan.generated_strings == []


# --- adaptation retry -----------------------------------------------------


def _block_report() -> BlockReport:
    return BlockReport(
        block_name="Opening",
        strings_run=1,
        strings_with_saves=0,
        total_results=50,
        total_saves=0,
        zero_save_string_ids=[1],
        string_details=[],
    )


def _adaptation_text() -> str:
    return json.dumps(
        {
            "new_strings": [
                {
                    "boolean": '("solutions architect") AND ("api integration")',
                    "rationale": "Adjacent title",
                    "family_key": "sa_integration",
                    "novelty_bucket": "edge_case",
                    "domain_lane": "general",
                }
            ],
            "skip_remaining": [],
            "reorder": [],
            "noise_updates": [],
        }
    )


def test_adaptation_retries_once_on_unparseable_reply(tmp_path):
    model = _ScriptedModel([MALFORMED, _adaptation_text()])
    with patch("linkedin.strategy.opus_llm", side_effect=model):
        adaptation = adapt_after_block(
            _brief(),
            _block_report(),
            [SearchString(id=2, name="Queued", boolean='("queued")')],
            state_dir=tmp_path,
        )

    assert len(model.calls) == 2
    assert model.calls[1]["usage_context"]["parse_retry_attempt"] == 2
    assert [s["family_key"] for s in adaptation.new_strings] == ["sa_integration"]
    files = _forensics(tmp_path)
    assert len(files) == 1
    assert files[0].name.startswith("adaptation-unparseable-")
    assert files[0].name.endswith("-attempt1.txt")


def test_adaptation_unparseable_twice_still_raises_to_its_caller(tmp_path):
    from shared.llm_clients import UnparseableLLMResponseError

    model = _ScriptedModel([MALFORMED, MALFORMED])
    with patch("linkedin.strategy.opus_llm", side_effect=model):
        with pytest.raises(UnparseableLLMResponseError):
            adapt_after_block(
                _brief(),
                _block_report(),
                [SearchString(id=2, name="Queued", boolean='("queued")')],
                state_dir=tmp_path,
            )
    assert len(model.calls) == 2
    assert len(_forensics(tmp_path)) == 2


def test_block_adaptation_hands_the_state_dir_to_adaptation():
    import tempfile

    from tests.test_linkedin_pipeline import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        seen: dict = {}

        def spy_adapt_fn(brief, report, remaining, **kwargs):
            seen.update(kwargs)
            return AdaptationResponse(no_change=True)

        done = SearchString(
            id=1,
            name="s1",
            boolean='("x")',
            status="done",
            result_count=10,
            candidates_count=5,
            pages_reviewed=1,
        )
        queued = SearchString(id=2, name="s2", boolean='("y")', status="queued")
        progress = Progress(brief_name="test", strings=[done, queued])
        asyncio.run(
            p.state.planner._run_block_adaptation("Block 1", [done], progress, spy_adapt_fn)
        )
        expected = p.state.state_dir

    assert seen["state_dir"] == expected
