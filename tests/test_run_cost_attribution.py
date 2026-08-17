"""Integration-style attribution for cheap_llm rows in token-cost JSONL."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from shared.llm_clients import cheap_llm
from shared.llm_usage import estimate_usage_cost_usd, llm_usage_session
from shared.storage import read_jsonl


def _fake_openai_response(text: str, prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text)),
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class _FakeOpenAICompletions:
    def __init__(self, response_factory):
        self._response_factory = response_factory

    def create(self, **kwargs):
        return self._response_factory()


class _FakeOpenAIChat:
    def __init__(self, response_factory):
        self.completions = _FakeOpenAICompletions(response_factory)


class _FakeOpenAIClientCtor:
    def __init__(self, response_factory):
        self._factory = response_factory

    def __call__(self, **kwargs):
        return SimpleNamespace(chat=_FakeOpenAIChat(self._factory))


class _FakeGoogleModels:
    def __init__(self, text: str, usage_metadata):
        self._text = text
        self._metadata = usage_metadata

    def generate_content(self, **kwargs):
        return SimpleNamespace(text=self._text, usage_metadata=self._metadata)


class _FakeGoogleClient:
    def __init__(self, **kwargs) -> None:
        self.models = _FakeGoogleModels(
            '{"z": []}',
            SimpleNamespace(prompt_token_count=150, candidates_token_count=75),
        )


def test_multiple_openai_cheap_llm_rows_sum_to_logged_total_cost():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        sequence = iter(
            (
                ('{"stage": "one"}', 1000, 10),
                ('{"stage": "two"}', 2000, 20),
            )
        )

        def next_payload():
            text, inp, outp = next(sequence)
            return _fake_openai_response(text, inp, outp)

        fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(next_payload))

        expected = []
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "openai"), patch(
                "shared.config.CHEAP_MODEL_NAME", "gpt-4o-mini"
            ):
                with llm_usage_session(
                    log_path,
                    pipeline="attribution",
                    run_id="run-sum-test",
                ):
                    cheap_llm("s", "first", expect_json=True)
                    cheap_llm("s", "second", expect_json=True)

                    for inp, outp in ((1000, 10), (2000, 20)):
                        c, _src = estimate_usage_cost_usd(
                            model="gpt-4o-mini",
                            input_tokens=inp,
                            output_tokens=outp,
                        )
                        assert c is not None
                        expected.append(c)

        rows = read_jsonl(log_path)
        assert len(rows) == 2
        assert rows[0]["provider"] == "openai"
        assert rows[1]["provider"] == "openai"
        assert rows[0]["pipeline"] == "attribution"
        assert rows[0]["run_id"] == "run-sum-test"

        logged_total = round(sum(float(r["estimated_cost_usd"] or 0) for r in rows), 6)
        expected_total = round(sum(expected), 6)
        assert logged_total == expected_total


def test_openai_then_google_providers_each_logged_once():
    pytest.importorskip("google.genai")

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        fake_openai = SimpleNamespace(
            OpenAI=_FakeOpenAIClientCtor(
                lambda: _fake_openai_response('{"a": null}', prompt_tokens=50, completion_tokens=5),
            ),
        )

        openai_manual, _ = estimate_usage_cost_usd(
            model="gpt-4o-mini",
            input_tokens=50,
            output_tokens=5,
        )
        google_manual, _ = estimate_usage_cost_usd(
            model="gemini-2.0-flash",
            input_tokens=150,
            output_tokens=75,
        )
        assert openai_manual is not None and google_manual is not None
        manual_total = round(openai_manual + google_manual, 6)

        with llm_usage_session(log_path, pipeline="mixed_providers"):
            with patch.dict(sys.modules, {"openai": fake_openai}):
                with patch("shared.config.CHEAP_MODEL_PROVIDER", "openai"), patch(
                    "shared.config.CHEAP_MODEL_NAME", "gpt-4o-mini"
                ):
                    cheap_llm("sys-a", "user-a", expect_json=True)

            with patch("google.genai.Client", side_effect=lambda **kwargs: _FakeGoogleClient()):
                with patch("shared.config.CHEAP_MODEL_PROVIDER", "google"):
                    cheap_llm("sys-b", "user-b", expect_json=True)

        rows = read_jsonl(log_path)
        assert len(rows) == 2
        providers = [r["provider"] for r in rows]
        assert providers == ["openai", "google"]
        logged_total = round(sum(float(r["estimated_cost_usd"] or 0) for r in rows), 6)
        assert logged_total == manual_total
