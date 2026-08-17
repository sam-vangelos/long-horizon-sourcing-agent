"""GLM-5.2 (Fireworks) shadow-judge seam — facial-stage instrumentation.

Covers the load-bearing doctrine from the shadow-judge build: the shadow
verdict is RECORDED AND COMPARED but has ZERO influence on the returned
decision, the whole shadow path is fail-soft (a shadow exception never
reaches the caller and never changes the primary verdict), and the batch
path shadows the WHOLE batch call once rather than fanning out per
candidate.

Run with: python -m pytest tests/test_judger_shadow_facial.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import shared.config
from shared.llm_usage import llm_usage_session
from shared.schemas import CandidateSnippet
from shared.storage import read_jsonl


@pytest.fixture(autouse=True)
def _synchronous_shadow(monkeypatch):
    """Pin these tests to the INLINE comparison path.

    They assert on run_log.jsonl immediately after the judge call and
    patch the shadow client inside ``with`` blocks — a background-worker
    comparison could run after the patch reverts and hit the real client.
    SHADOW_ASYNC_ENABLED=False is the supported escape hatch and is
    byte-identical to the pre-executor synchronous behavior, so every
    assertion here keeps its original meaning. The async dispatch
    property itself is pinned in tests/test_judger_shadow_async.py.

    Also zeroes the module-level running tallies so console-line tests
    assert against their own comparison counts, not accumulated history.
    """
    monkeypatch.setattr(shared.config, "SHADOW_ASYNC_ENABLED", False)
    from shared.judger import _reset_shadow_tallies

    _reset_shadow_tallies()


def _make_snippet(**kwargs) -> CandidateSnippet:
    defaults = {
        "name": "Test Person",
        "headline": "ML Engineer",
        "current_title": "ML Engineer",
        "current_company": "Acme Corp",
        "location": "San Francisco",
        "education_snippet": "BS CS Stanford",
        "profile_url": "/talent/profile/test123",
        "source_string_id": 1,
        "source_string_name": "test",
        "page": 1,
        "result_rank": 1,
    }
    defaults.update(kwargs)
    return CandidateSnippet(**defaults)


def _v2_brief() -> MagicMock:
    brief = MagicMock()
    brief.has_v2_schema = True
    brief._new_brief = MagicMock()
    return brief


def _run_log_events(log_dir: Path) -> list[dict]:
    return read_jsonl(log_dir / "run_log.jsonl")


# ---------------------------------------------------------------------------
# Zero-influence: the returned verdict must be byte-identical with the
# shadow flag on (shadow mocked, even to a DIFFERENT decision) vs off.
# ---------------------------------------------------------------------------


def test_facial_judge_verdict_is_byte_identical_with_shadow_on_vs_off():
    snippet = _make_snippet()
    primary_response = "DECISION: FACIAL_YES\nREASON: strong builder signal"

    def _judge_once() -> dict:
        with patch("shared.judger.assemble_facial_system", return_value="system"), \
             patch("shared.judger.facial_llm", return_value=primary_response), \
             patch("shared.judger.shadow_facial_llm", return_value="DECISION: FACIAL_NO\nREASON: disagreeing shadow"):
            from shared.judger import facial_judge

            decision = facial_judge(snippet, _v2_brief())
            return {
                "decision": decision.decision,
                "rationale": decision.rationale,
                "confidence": decision.confidence,
                "path": decision.path,
                "candidate_name": decision.candidate_name,
                "profile_url": decision.profile_url,
            }

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                off = _judge_once()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                on = _judge_once()

    assert on == off
    assert on["decision"] == "FACIAL_YES"  # sanity: primary verdict, not the disagreeing shadow


def test_facial_judge_legacy_branch_verdict_unaffected_by_shadow():
    """Old-brief (non-V2) path — the shadow hook must not affect this branch either."""
    snippet = _make_snippet()
    primary_result = {"decision": "FACIAL_YES", "path": "none", "confidence": 0.7, "rationale": "ok"}

    def _judge_once() -> str:
        brief = MagicMock()
        brief.has_v2_schema = False
        with patch("shared.judger._build_facial_system", return_value="system"), \
             patch("shared.judger.opus_llm_cached", return_value=primary_result), \
             patch("shared.judger.shadow_facial_llm", return_value='{"decision": "FACIAL_NO"}'):
            from shared.judger import facial_judge

            return facial_judge(snippet, brief).decision

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                off = _judge_once()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                on = _judge_once()

    assert on == off == "FACIAL_YES"


# ---------------------------------------------------------------------------
# Fail-soft: a shadow exception must never reach the caller and must never
# change the primary verdict; it must be recorded as shadow_error.
# ---------------------------------------------------------------------------


def test_shadow_exception_is_fail_soft_and_recorded():
    snippet = _make_snippet()
    primary_response = "DECISION: FACIAL_YES\nREASON: strong builder signal"

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_facial_system", return_value="system"), \
                     patch("shared.judger.facial_llm", return_value=primary_response), \
                     patch(
                         "shared.judger.shadow_facial_llm",
                         side_effect=RuntimeError("fireworks account suspended"),
                     ):
                    from shared.judger import facial_judge

                    decision = facial_judge(snippet, _v2_brief())

            events = _run_log_events(Path(td))

    assert decision.decision == "FACIAL_YES"  # primary verdict unaffected
    shadow_events = [e for e in events if e.get("event") == "facial_shadow_comparison"]
    assert len(shadow_events) == 1
    event = shadow_events[0]
    assert event["primary_decision"] == "FACIAL_YES"
    assert event["shadow_decision"] is None
    assert event["agrees"] is None
    assert event["shadow_parse_failed"] is False
    assert "fireworks account suspended" in event["shadow_error"]


def test_shadow_disabled_by_default_never_calls_shadow_or_logs():
    snippet = _make_snippet()
    primary_response = "DECISION: FACIAL_YES\nREASON: strong builder signal"

    with tempfile.TemporaryDirectory() as td:
        # SHADOW_FACIAL_MODEL_ENABLED defaults to False — no patch needed,
        # this pins the real default doctrine (off by default).
        with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
            with patch("shared.judger.assemble_facial_system", return_value="system"), \
                 patch("shared.judger.facial_llm", return_value=primary_response), \
                 patch("shared.judger.shadow_facial_llm") as mock_shadow:
                from shared.judger import facial_judge

                decision = facial_judge(snippet, _v2_brief())

        events = _run_log_events(Path(td))

    assert decision.decision == "FACIAL_YES"
    mock_shadow.assert_not_called()
    assert not [e for e in events if e.get("event") == "facial_shadow_comparison"]


def test_shadow_parse_failure_is_recorded_and_does_not_count_as_agreement():
    snippet = _make_snippet()
    primary_response = "DECISION: FACIAL_YES\nREASON: strong builder signal"

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_facial_system", return_value="system"), \
                     patch("shared.judger.facial_llm", return_value=primary_response), \
                     patch("shared.judger.shadow_facial_llm", return_value="garbage, not a verdict"):
                    from shared.judger import facial_judge

                    decision = facial_judge(snippet, _v2_brief())

            events = _run_log_events(Path(td))

    assert decision.decision == "FACIAL_YES"
    event = next(e for e in events if e.get("event") == "facial_shadow_comparison")
    assert event["shadow_parse_failed"] is True
    assert event["agrees"] is None
    # Parse failures persist the raw response for post-mortems — a
    # PARSE_FAILURE that throws away what it couldn't parse is
    # undiagnosable (2026-07-04 SPL live run).
    assert event["shadow_raw_prefix"] == "garbage, not a verdict"
    assert event["shadow_raw_len"] == len("garbage, not a verdict")


# ---------------------------------------------------------------------------
# Batch: shadow the WHOLE batch call once, not a per-candidate fan-out.
# ---------------------------------------------------------------------------


def test_batch_shadow_makes_exactly_one_call_for_whole_batch():
    snippets = [
        _make_snippet(name="Alice", profile_url="/alice"),
        _make_snippet(name="Bob", profile_url="/bob"),
        _make_snippet(name="Carol", profile_url="/carol"),
    ]
    primary_batch_response = (
        "[1] FACIAL_YES | alice reason\n"
        "[2] FACIAL_NO | bob reason\n"
        "[3] FACIAL_YES | carol reason\n"
    )
    shadow_batch_response = (
        "[1] FACIAL_NO | shadow disagrees on alice\n"
        "[2] FACIAL_NO | shadow agrees on bob\n"
        "[3] FACIAL_YES | shadow agrees on carol\n"
    )

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_facial_batch_system", return_value="system"), \
                     patch("shared.judger.facial_llm", return_value=primary_batch_response), \
                     patch(
                         "shared.judger.shadow_facial_llm",
                         return_value=shadow_batch_response,
                     ) as mock_shadow:
                    from shared.judger import facial_judge_batch

                    decisions = facial_judge_batch(snippets, _v2_brief())

            events = _run_log_events(Path(td))

    # Primary verdicts are exactly what the batch response said — unaffected
    # by the disagreeing shadow batch response.
    by_name = {d.candidate_name: d.decision for d in decisions}
    assert by_name == {"Alice": "FACIAL_YES", "Bob": "FACIAL_NO", "Carol": "FACIAL_YES"}

    # ONE shadow call for the whole batch, not three.
    assert mock_shadow.call_count == 1

    shadow_events = [e for e in events if e.get("event") == "facial_shadow_comparison"]
    assert len(shadow_events) == 1
    event = shadow_events[0]
    assert event["batch"] is True
    assert event["candidate_count"] == 3
    assert event["primary_decisions"] == ["FACIAL_YES", "FACIAL_NO", "FACIAL_YES"]
    assert event["shadow_decisions"] == ["FACIAL_NO", "FACIAL_NO", "FACIAL_YES"]
    assert event["agrees"] == [False, True, True]
    assert event["shadow_parse_failed"] == [False, False, False]
    # Raw-capture fields are failure-only: the happy-path event schema is
    # unchanged.
    assert "shadow_raw_prefix" not in event
    assert "shadow_raw_len" not in event


def test_batch_shadow_parse_failure_records_raw_prefix():
    """An unparseable shadow BATCH response marks every candidate failed and
    persists the raw text once on the batch event."""
    snippets = [
        _make_snippet(name="Alice", profile_url="/alice"),
        _make_snippet(name="Bob", profile_url="/bob"),
    ]
    primary_batch_response = (
        "[1] FACIAL_YES | alice reason\n"
        "[2] FACIAL_NO | bob reason\n"
    )
    shadow_garbage = "reasoning preamble that never reaches a verdict line"

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_facial_batch_system", return_value="system"), \
                     patch("shared.judger.facial_llm", return_value=primary_batch_response), \
                     patch("shared.judger.shadow_facial_llm", return_value=shadow_garbage):
                    from shared.judger import facial_judge_batch

                    decisions = facial_judge_batch(snippets, _v2_brief())

            events = _run_log_events(Path(td))

    assert [d.decision for d in decisions] == ["FACIAL_YES", "FACIAL_NO"]
    event = next(e for e in events if e.get("event") == "facial_shadow_comparison")
    assert event["batch"] is True
    assert all(event["shadow_parse_failed"])
    assert event["shadow_raw_prefix"] == shadow_garbage
    assert event["shadow_raw_len"] == len(shadow_garbage)


# ---------------------------------------------------------------------------
# Run-console contract (Sam, 2026-07-05, revised same day): EXACTLY ONE
# compact line per completed comparison — decisions by model name, outcome
# in words, running facial tally (batch members fold in individually).
# Reasoning stays in shadow_judgments.jsonl and the --follow feed.
# ---------------------------------------------------------------------------


def _shadow_console_lines(out: str) -> list[str]:
    return [l for l in out.splitlines() if l.startswith("[shadow] ")]


def _console_model_names(monkeypatch):
    monkeypatch.setattr(shared.config, "FACIAL_MODEL_NAME", "claude-opus-4-6")
    monkeypatch.setattr(
        shared.config,
        "SHADOW_FACIAL_MODEL_NAME",
        "accounts/fireworks/models/glm-5p2",
    )


def test_facial_single_prints_one_compact_comparison_line(capsys, monkeypatch):
    _console_model_names(monkeypatch)
    snippet = _make_snippet()
    primary_response = "DECISION: FACIAL_YES\nREASON: strong builder signal"

    def _fake_shadow(system_prompt, user_prompt, **kwargs):
        capture = kwargs.get("capture")
        if isinstance(capture, dict):
            capture["reasoning_content"] = "Trajectory says platform builder."
        return "DECISION: FACIAL_YES\nREASON: agreeing shadow"

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_facial_system", return_value="system"), \
                     patch("shared.judger.facial_llm", return_value=primary_response), \
                     patch("shared.judger.shadow_facial_llm", side_effect=_fake_shadow):
                    from shared.judger import facial_judge

                    facial_judge(snippet, _v2_brief())

    out = capsys.readouterr().out
    lines = _shadow_console_lines(out)
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[shadow] facial")
    assert "Test Person" in line
    assert "opus=FACIAL_YES" in line
    assert "glm=FACIAL_YES" in line
    assert "AGREE" in line
    assert "facial: 1/1 agree (100.0%)" in line
    # Reasoning never hits the run console.
    assert "Trajectory says platform builder." not in out


def test_facial_batch_prints_one_line_and_members_fold_into_tally(capsys, monkeypatch):
    _console_model_names(monkeypatch)
    snippets = [
        _make_snippet(name="Alice", profile_url="/alice"),
        _make_snippet(name="Bob", profile_url="/bob"),
        _make_snippet(name="Carol", profile_url="/carol"),
    ]
    primary_batch_response = (
        "[1] FACIAL_YES | alice reason\n"
        "[2] FACIAL_NO | bob reason\n"
        "[3] FACIAL_YES | carol reason\n"
    )
    shadow_batch_response = (
        "[1] FACIAL_NO | shadow disagrees on alice\n"
        "[2] FACIAL_NO | shadow agrees on bob\n"
        "[3] FACIAL_YES | shadow agrees on carol\n"
    )

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_facial_batch_system", return_value="system"), \
                     patch("shared.judger.facial_llm", return_value=primary_batch_response), \
                     patch("shared.judger.shadow_facial_llm", return_value=shadow_batch_response):
                    from shared.judger import facial_judge_batch

                    facial_judge_batch(snippets, _v2_brief())

    out = capsys.readouterr().out
    lines = _shadow_console_lines(out)
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[shadow] facial×3")
    assert "2 agree, 1 disagree" in line
    assert "facial: 2/3 agree (66.7%)" in line
    # Per-candidate verdict prose stays off the run console.
    assert "shadow disagrees on alice" not in out
