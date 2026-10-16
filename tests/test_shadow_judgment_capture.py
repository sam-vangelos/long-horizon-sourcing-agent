"""shadow_judgments.jsonl — the full shadow-verdict monitoring channel.

The run-log comparison events stay compact (decisions + agreement); this
file is where the shadow model's complete output — structured judgment
text, separate reasoning_content, and the exact user prompt — lands per
candidate so verdicts are attributable and re-judgeable offline.
"""

import json

import shared.judger as judger


def _stub_shadow(raw_text: str, reasoning: str | None):
    def stub(system_prompt, user_prompt, max_tokens=0, usage_context=None, capture=None):
        if capture is not None:
            capture["reasoning_content"] = reasoning
            capture["finish_reason"] = "stop"
        return raw_text

    return stub


def _full_shadow_raw() -> str:
    return """STEP_1_MATCH: DIRECT
STEP_1_AREA: Core Systems
STEP_1_EVIDENCE: Built the target system.
STEP_2_DEPTH: BUILDER
STEP_2_EVIDENCE: Designed and owned it.
STEP_3_TRANSFERABILITY: N/A
STEP_3_EVIDENCE: Direct match.
STEP_1_RECENCY: CURRENT
STEP_4_LEVEL: ALIGNED
STEP_5_COHERENCE: COHERENT
STEP_6_CALIBER: STRONG
CASE_FOR: Strong direct evidence.
CASE_AGAINST: Limited scale detail.
DECISION: SAVE
CONFIDENCE: 0.8
REJECT_REASON: NONE
OUTREACH_TIER: PRIORITY
POST_SAVE_MODIFIER: NONE
SUMMARY: Strong candidate."""


def _records(tmp_path):
    target = tmp_path / "shadow_judgments.jsonl"
    return [json.loads(line) for line in target.read_text().splitlines()]


def test_full_shadow_capture_writes_full_record(tmp_path, monkeypatch):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(judger, "_shadow_run_log_path", lambda: str(log_path))
    monkeypatch.setattr(
        judger,
        "shadow_full_llm",
        _stub_shadow(_full_shadow_raw(), "chain of thought here"),
    )

    judger._run_full_shadow_single_sync(
        system_prompt="sys",
        user_prompt="PROFILE: Jane Doe",
        max_tokens=64,
        primary_decision="REJECT",
        capability_areas=("Core Systems",),
        post_save_modifiers=(),
        lane_context={"stage": "full_eval"},
    )

    records = _records(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["stage"] == "full"
    assert "DECISION: SAVE" in rec["raw"]
    assert rec["reasoning_content"] == "chain of thought here"
    assert rec["user_prompt"] == "PROFILE: Jane Doe"
    assert rec["primary_decision"] == "REJECT"
    assert rec["shadow_decision"] == "SAVE"
    assert rec["agrees"] is False
    assert rec["lane_context"] == {"stage": "full_eval"}


def test_facial_shadow_capture_writes_record(tmp_path, monkeypatch):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(judger, "_shadow_run_log_path", lambda: str(log_path))
    monkeypatch.setattr(
        judger,
        "shadow_facial_llm",
        _stub_shadow("DECISION: FACIAL_YES", None),
    )

    judger._run_facial_shadow_single_sync(
        system_prompt="sys",
        user_prompt="SNIPPET: Jane Doe",
        max_tokens=64,
        primary_decision="FACIAL_YES",
        parse_fn=lambda raw: "FACIAL_YES",
        lane_context=None,
    )

    records = _records(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["stage"] == "facial"
    assert rec["raw"] == "DECISION: FACIAL_YES"
    assert rec["reasoning_content"] is None
    assert rec["agrees"] is True
    assert rec["user_prompt"] == "SNIPPET: Jane Doe"


def test_capture_failure_never_breaks_the_shadow_path(tmp_path, monkeypatch):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(judger, "_shadow_run_log_path", lambda: str(log_path))
    monkeypatch.setattr(
        judger, "shadow_full_llm", _stub_shadow(_full_shadow_raw(), None)
    )

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(judger, "_record_shadow_judgment", judger._record_shadow_judgment)
    monkeypatch.setattr("shared.storage.append_jsonl", boom)

    # Must not raise: the capture write is fail-soft like every shadow write.
    judger._run_full_shadow_single_sync(
        system_prompt="sys",
        user_prompt="PROFILE: X",
        max_tokens=64,
        primary_decision="SAVE",
        capability_areas=("Core Systems",),
        post_save_modifiers=(),
        lane_context=None,
    )
