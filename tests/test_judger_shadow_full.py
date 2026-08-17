"""GLM-5.2 (Fireworks) shadow-judge seam — FULL-EVAL instrumentation.

Sibling to tests/test_judger_shadow_facial.py, same doctrine and idioms
applied to shared.judger.full_judge's V2-structural branch: the shadow
verdict is RECORDED AND COMPARED but has ZERO influence on the returned
decision, the whole shadow path is fail-soft, and agreement is measured on
the SAVE-family-vs-REJECT decision AXIS (a SAVE vs INFERENTIAL_SAVE
mismatch on the raw decision strings still counts as agreement).

Run with: python -m pytest tests/test_judger_shadow_full.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import shared.config
from shared.llm_usage import llm_usage_session
from shared.schemas import CandidateProfileSummary
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

    Also zeroes the module-level running tallies so each test's console
    line asserts against ITS OWN comparison counts, not the file's
    accumulated history.
    """
    monkeypatch.setattr(shared.config, "SHADOW_ASYNC_ENABLED", False)
    from shared.judger import _reset_shadow_tallies

    _reset_shadow_tallies()


def _make_summary(**kwargs) -> CandidateProfileSummary:
    defaults = {
        "name": "Test Person",
        "headline": "ML Engineer",
        "profile_url": "/talent/profile/test123",
        "experiences": [],
        "education": [],
        "skills_snippet": [],
    }
    defaults.update(kwargs)
    return CandidateProfileSummary(**defaults)


def _v2_brief() -> MagicMock:
    brief = MagicMock()
    brief.has_v2_schema = True
    brief._new_brief = MagicMock()
    brief._new_brief.dossier_mode = False
    brief._new_brief.capability_area_names.return_value = ["ML Infra"]
    brief._new_brief.post_save_modifiers = []
    return brief


def _run_log_events(log_dir: Path) -> list[dict]:
    return read_jsonl(log_dir / "run_log.jsonl")


_PRIMARY_SAVE_RAW = (
    "STEP_1_MATCH: DIRECT\n"
    "STEP_1_AREA: ML Infra\n"
    "STEP_1_EVIDENCE: strong infra background\n"
    "STEP_1_RECENCY: CURRENT\n"
    "STEP_2_DEPTH: BUILDER\n"
    "STEP_2_EVIDENCE: built training pipelines\n"
    "STEP_3_TRANSFERABILITY: N/A\n"
    "STEP_3_EVIDENCE: n/a\n"
    "STEP_4_LEVEL: ALIGNED\n"
    "STEP_5_COHERENCE: COHERENT\n"
    "STEP_6_CALIBER: STRONG\n"
    "CASE_FOR: strong fit\n"
    "CASE_AGAINST: none\n"
    "REJECT_REASON: NONE\n"
    "OUTREACH_TIER: STANDARD\n"
    "DECISION: SAVE\n"
    "CONFIDENCE: 0.9\n"
    "POST_SAVE_MODIFIER: NONE\n"
    "SUMMARY: Strong candidate for the role.\n"
)


def _shadow_reject_raw() -> str:
    return (
        _PRIMARY_SAVE_RAW.replace(
            "STEP_1_RECENCY: CURRENT", "STEP_1_RECENCY: RECENT"
        )
        .replace("STEP_4_LEVEL: ALIGNED", "STEP_4_LEVEL: BELOW")
        .replace("STEP_5_COHERENCE: COHERENT", "STEP_5_COHERENCE: INCOHERENT")
        .replace("STEP_6_CALIBER: STRONG", "STEP_6_CALIBER: WEAK")
        .replace(
            "REJECT_REASON: NONE", "REJECT_REASON: CAPABILITY_INSUFFICIENT"
        )
        .replace("OUTREACH_TIER: STANDARD", "OUTREACH_TIER: NONE")
        .replace("DECISION: SAVE", "DECISION: REJECT")
    )


def _shadow_review_raw() -> str:
    return (
        _PRIMARY_SAVE_RAW.replace(
            "STEP_1_RECENCY: CURRENT", "STEP_1_RECENCY: RECENT"
        )
        .replace("STEP_4_LEVEL: ALIGNED", "STEP_4_LEVEL: UNCLEAR")
        .replace("STEP_5_COHERENCE: COHERENT", "STEP_5_COHERENCE: UNCLEAR")
        .replace("STEP_6_CALIBER: STRONG", "STEP_6_CALIBER: UNKNOWN")
        .replace("OUTREACH_TIER: STANDARD", "OUTREACH_TIER: NONE")
        .replace("DECISION: SAVE", "DECISION: REVIEW_FLAGGED")
        .replace(
            "SUMMARY: Strong candidate for the role.",
            "REVIEW_REASON: needs_more_evidence\n"
            "RECOMMENDED_NEXT_STEP: Verify current ownership scope.\n"
            "SUMMARY: Strong candidate for the role.",
        )
    )


def _shadow_inferential_save_raw() -> str:
    return _PRIMARY_SAVE_RAW.replace("DECISION: SAVE", "DECISION: INFERENTIAL_SAVE")


# ---------------------------------------------------------------------------
# Zero-influence: the returned verdict must be byte-identical with the
# shadow flag on (shadow mocked, even to a DISAGREEING decision) vs off.
# ---------------------------------------------------------------------------


def test_full_judge_verdict_is_byte_identical_with_shadow_on_vs_off():
    summary = _make_summary()

    def _judge_once() -> dict:
        with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
             patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
             patch("shared.judger.shadow_full_llm", return_value=_shadow_reject_raw()):
            from shared.judger import full_judge

            decision = full_judge(summary, _v2_brief())
            return {
                "decision": decision.decision,
                "rationale": decision.rationale,
                "confidence": decision.confidence,
                "path": decision.path,
                "candidate_name": decision.candidate_name,
                "profile_url": decision.profile_url,
                "post_save_modifier": decision.post_save_modifier,
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
    assert on["decision"] == "SAVE"  # sanity: primary verdict, not the disagreeing shadow


def test_full_judge_dossier_mode_verdict_unaffected_by_shadow():
    """Dossier-mode (exec_search) branch also builds profile_text differently
    (assemble_dossier_evidence) — the shadow hook must not affect it either."""
    summary = _make_summary()
    brief = _v2_brief()
    brief._new_brief.dossier_mode = True

    dossier = MagicMock()
    dossier.prompt_body = "dossier evidence body"

    def _judge_once() -> str:
        with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
             patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
             patch("shared.judger.shadow_full_llm", return_value=_shadow_reject_raw()), \
             patch(
                 "exec_search.evidence_assembly.assemble_dossier_evidence",
                 return_value=dossier,
             ):
            from shared.judger import full_judge

            return full_judge(summary, brief).decision

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                off = _judge_once()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                on = _judge_once()

    assert on == off == "SAVE"


# ---------------------------------------------------------------------------
# Fail-soft: a shadow exception must never reach the caller and must never
# change the primary verdict; it must be recorded as shadow_error.
# ---------------------------------------------------------------------------


def test_full_shadow_exception_is_fail_soft_and_recorded():
    summary = _make_summary()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                     patch(
                         "shared.judger.shadow_full_llm",
                         side_effect=RuntimeError("fireworks account suspended"),
                     ):
                    from shared.judger import full_judge

                    decision = full_judge(summary, _v2_brief())

            events = _run_log_events(Path(td))

    assert decision.decision == "SAVE"  # primary verdict unaffected
    shadow_events = [e for e in events if e.get("event") == "full_shadow_comparison"]
    assert len(shadow_events) == 1
    event = shadow_events[0]
    assert event["primary_decision"] == "SAVE"
    assert event["shadow_decision"] is None
    assert event["agrees"] is None
    assert event["shadow_parse_failed"] is False
    assert "fireworks account suspended" in event["shadow_error"]


def test_full_shadow_disabled_by_default_never_calls_shadow_or_logs():
    summary = _make_summary()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                     patch("shared.judger.shadow_full_llm") as mock_shadow:
                    from shared.judger import full_judge

                    decision = full_judge(summary, _v2_brief())

            events = _run_log_events(Path(td))

    assert decision.decision == "SAVE"
    mock_shadow.assert_not_called()
    assert not [e for e in events if e.get("event") == "full_shadow_comparison"]


def test_full_shadow_no_usage_session_is_a_no_op():
    """No llm_usage_session open -> _shadow_run_log_path() returns None ->
    the whole shadow hook no-ops even with the flag on (facial's precedent:
    unit tests / rejudge_from_file / ad-hoc calls have nowhere honest to
    log to)."""
    summary = _make_summary()

    with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
        with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
             patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
             patch("shared.judger.shadow_full_llm") as mock_shadow:
            from shared.judger import full_judge

            decision = full_judge(summary, _v2_brief())

    assert decision.decision == "SAVE"
    mock_shadow.assert_not_called()


def test_full_shadow_parse_failure_is_recorded_and_does_not_count_as_agreement():
    summary = _make_summary()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                     patch("shared.judger.shadow_full_llm", return_value="garbage, not a verdict"):
                    from shared.judger import full_judge

                    decision = full_judge(summary, _v2_brief())

            events = _run_log_events(Path(td))

    assert decision.decision == "SAVE"
    event = next(e for e in events if e.get("event") == "full_shadow_comparison")
    assert event["shadow_parse_failed"] is True
    assert event["agrees"] is None
    # Parse failures persist the raw response for post-mortems — the
    # 2026-07-04 SPL run's full-eval PARSE_FAILUREs were undiagnosable
    # without it. An empty response records prefix="" / len=0, which is
    # itself the diagnosis.
    assert event["shadow_raw_prefix"] == "garbage, not a verdict"
    assert event["shadow_raw_len"] == len("garbage, not a verdict")


# ---------------------------------------------------------------------------
# Decision-CLASS agreement (save/reject axis) — the load-bearing new
# semantics for full-eval that facial's binary decision didn't need.
# ---------------------------------------------------------------------------


def test_full_shadow_save_vs_inferential_save_counts_as_agreement_but_records_raw_decisions():
    summary = _make_summary()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                     patch(
                         "shared.judger.shadow_full_llm",
                         return_value=_shadow_inferential_save_raw(),
                     ):
                    from shared.judger import full_judge

                    decision = full_judge(summary, _v2_brief())

            events = _run_log_events(Path(td))

    assert decision.decision == "SAVE"
    event = next(e for e in events if e.get("event") == "full_shadow_comparison")
    # Raw decisions recorded verbatim...
    assert event["primary_decision"] == "SAVE"
    assert event["shadow_decision"] == "INFERENTIAL_SAVE"
    # ...but agreement is on the save/reject AXIS: both are save-family.
    assert event["agrees"] is True
    assert event["shadow_parse_failed"] is False


def test_full_shadow_save_vs_reject_disagrees():
    summary = _make_summary()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                     patch("shared.judger.shadow_full_llm", return_value=_shadow_reject_raw()):
                    from shared.judger import full_judge

                    decision = full_judge(summary, _v2_brief())

            events = _run_log_events(Path(td))

    assert decision.decision == "SAVE"
    event = next(e for e in events if e.get("event") == "full_shadow_comparison")
    assert event["primary_decision"] == "SAVE"
    assert event["shadow_decision"] == "REJECT"
    assert event["agrees"] is False


def test_full_shadow_review_decision_is_not_comparable_on_save_reject_axis():
    """A shadow REVIEW_INFERRED/REVIEW_FLAGGED verdict isn't classifiable on
    the save/reject axis — `agrees` must be None, not a fabricated
    False, and it must not be flagged as a parse failure either (it's a
    real, valid full-eval decision, just not on this axis)."""
    summary = _make_summary()
    shadow_review_raw = _shadow_review_raw()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                     patch("shared.judger.shadow_full_llm", return_value=shadow_review_raw):
                    from shared.judger import full_judge

                    decision = full_judge(summary, _v2_brief())

            events = _run_log_events(Path(td))

    assert decision.decision == "SAVE"
    event = next(e for e in events if e.get("event") == "full_shadow_comparison")
    assert event["shadow_decision"] == "REVIEW_FLAGGED"
    assert event["agrees"] is None
    assert event["shadow_parse_failed"] is False


# ---------------------------------------------------------------------------
# Live console surface (Sam, 2026-07-05, revised same day): the run console
# is sacred. Each completed shadow comparison prints EXACTLY ONE line —
# both decisions by model name, the outcome in words, and the running
# per-tier tally. Reasoning and verdict prose live in
# shadow_judgments.jsonl and the --follow feed, never on the run console.
# ---------------------------------------------------------------------------


def _shadow_console_lines(out: str) -> list[str]:
    return [l for l in out.splitlines() if l.startswith("[shadow] ")]


def _console_model_names(monkeypatch):
    """Pin the model names the console line derives its words from."""
    monkeypatch.setattr(shared.config, "FULL_EVAL_MODEL_NAME", "claude-opus-4-6")
    monkeypatch.setattr(
        shared.config,
        "SHADOW_FACIAL_MODEL_NAME",
        "accounts/fireworks/models/glm-5p2",
    )


def test_full_shadow_prints_one_compact_comparison_line(capsys, monkeypatch):
    _console_model_names(monkeypatch)
    summary = _make_summary()

    def _fake_shadow(system_prompt, user_prompt, **kwargs):
        capture = kwargs.get("capture")
        if isinstance(capture, dict):
            capture["reasoning_content"] = "Weighing infra depth against the bar."
            capture["finish_reason"] = "stop"
        return _shadow_reject_raw()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                     patch("shared.judger.shadow_full_llm", side_effect=_fake_shadow):
                    from shared.judger import full_judge

                    full_judge(summary, _v2_brief())

    out = capsys.readouterr().out
    lines = _shadow_console_lines(out)
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[shadow] full")
    assert "Test Person" in line  # candidate identity, by name
    assert "opus=SAVE" in line
    assert "glm=REJECT" in line
    assert "DISAGREE" in line
    assert "full: 0/1 agree (0.0%)" in line  # the running tally rides the line
    # Reasoning and verdict prose stay OFF the run console — they live in
    # shadow_judgments.jsonl and the --follow feed.
    assert "Weighing infra depth against the bar." not in out
    assert "DECISION: REJECT" not in out


def test_full_shadow_parse_failure_console_names_the_failure_in_words(capsys, monkeypatch):
    """No naked N/A and no raw-text dump: an unparseable shadow response
    renders as `glm=PARSE_FAILURE` + the word `unparsed`, counted beside
    the tally (never inside its percentage)."""
    _console_model_names(monkeypatch)
    summary = _make_summary()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                     patch("shared.judger.shadow_full_llm", return_value="garbage, not a verdict"):
                    from shared.judger import full_judge

                    full_judge(summary, _v2_brief())

    out = capsys.readouterr().out
    lines = _shadow_console_lines(out)
    assert len(lines) == 1
    line = lines[0]
    assert "opus=SAVE" in line
    assert "glm=PARSE_FAILURE" in line
    assert "unparsed" in line
    assert "full: 0/0 agree, 1 unparsed" in line
    # The raw text does NOT hit the run console (it persists in
    # shadow_judgments.jsonl and renders, bounded, on the feed).
    assert "garbage, not a verdict" not in out


def test_full_shadow_tally_accumulates_across_comparisons(capsys, monkeypatch):
    """Tally arithmetic across a mixed sequence: agree, disagree, REVIEW
    (not comparable), transport error — comparable outcomes drive the
    percentage; everything else is appended in words."""
    _console_model_names(monkeypatch)
    summary = _make_summary()
    shadow_review_raw = _shadow_review_raw()
    shadow_responses = [
        _shadow_inferential_save_raw(),          # save-axis AGREE
        _shadow_reject_raw(),                    # DISAGREE
        shadow_review_raw,                       # not comparable
        RuntimeError("fireworks account suspended"),  # SHADOW ERROR
    ]

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                     patch("shared.judger.shadow_full_llm", side_effect=shadow_responses):
                    from shared.judger import full_judge

                    for _ in shadow_responses:
                        full_judge(summary, _v2_brief())

    out = capsys.readouterr().out
    lines = _shadow_console_lines(out)
    assert len(lines) == 4
    assert "full: 1/1 agree (100.0%)" in lines[0]
    assert "full: 1/2 agree (50.0%)" in lines[1]
    assert "not comparable" in lines[2]
    assert "full: 1/2 agree (50.0%), 1 not comparable" in lines[2]
    assert "SHADOW ERROR: fireworks account suspended" in lines[3]
    assert "full: 1/2 agree (50.0%), 1 not comparable, 1 error" in lines[3]


# ---------------------------------------------------------------------------
# Prompt parity: the shadow call must receive the SAME system+user prompts
# as the primary call. The token ceiling is deliberately NOT parity: it is
# shadow-owned at 16384 (Fireworks counts GLM's reasoning tokens against
# max_tokens; at the primary's 8192 both 2026-07-05 live full-eval parse
# failures were finish_reason=length truncations, one of them a perfectly
# formatted response cut off two lines before DECISION).
# ---------------------------------------------------------------------------


def test_full_shadow_receives_same_prompts_but_shadow_owned_max_tokens():
    summary = _make_summary()
    captured = {}

    def _fake_shadow(system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["max_tokens"] = kwargs.get("max_tokens")
        return _shadow_reject_raw()

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_full_evaluation_system", return_value="the system prompt"), \
                     patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW) as mock_primary, \
                     patch("shared.judger.shadow_full_llm", side_effect=_fake_shadow):
                    from shared.judger import full_judge

                    full_judge(summary, _v2_brief())

    primary_call_args = mock_primary.call_args
    assert captured["system_prompt"] == "the system prompt" == primary_call_args.args[0]
    assert captured["user_prompt"] == primary_call_args.args[1]
    assert captured["max_tokens"] == 16384
