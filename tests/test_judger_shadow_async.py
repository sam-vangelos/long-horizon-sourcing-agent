"""GLM-5.2 (Fireworks) shadow-judge seam — fire-and-forget dispatch.

Pins the SHADOW_ASYNC_ENABLED property the 2026-07-05 live run motivated:
with the flag on (the default), the judge call returns while the shadow
comparison is still in flight on the single background worker, and
``drain_shadow_comparisons`` flushes the queue so the comparison event
exists afterward with exactly the same content the synchronous path
emits (tests/test_judger_shadow_full.py / _facial.py pin that content on
the inline path; those files force SHADOW_ASYNC_ENABLED=False).

The blocked-shadow stub is an Event, not a sleep: the judge returning
while ``release_shadow`` is still unset PROVES non-blocking dispatch
(a synchronous regression would sit inside the stub until its bounded
wait expires and then fail the no-event-yet assertion — deterministic
red, no timing flake). Everything, including drain, happens INSIDE the
``with patch`` blocks so the worker can never touch a real client.

Context propagation is pinned implicitly: the worker resolves the
run-log sink from the ContextVar snapshot captured at dispatch
(shared/llm_usage.py's _USAGE_LOG_PATH) — if that snapshot were lost the
shadow would no-op and the after-drain event assertions would fail. One
test additionally closes the llm_usage_session BEFORE releasing the
shadow, pinning that the snapshot survives the session's reset.

Run with: python -m pytest tests/test_judger_shadow_async.py -v
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from shared.llm_usage import llm_usage_session
from shared.schemas import CandidateProfileSummary, CandidateSnippet
from shared.storage import read_jsonl


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

_SHADOW_REJECT_RAW = (
    _PRIMARY_SAVE_RAW.replace("STEP_1_RECENCY: CURRENT", "STEP_1_RECENCY: RECENT")
    .replace("STEP_4_LEVEL: ALIGNED", "STEP_4_LEVEL: BELOW")
    .replace("STEP_5_COHERENCE: COHERENT", "STEP_5_COHERENCE: INCOHERENT")
    .replace("STEP_6_CALIBER: STRONG", "STEP_6_CALIBER: WEAK")
    .replace("REJECT_REASON: NONE", "REJECT_REASON: CAPABILITY_INSUFFICIENT")
    .replace("OUTREACH_TIER: STANDARD", "OUTREACH_TIER: NONE")
    .replace("DECISION: SAVE", "DECISION: REJECT")
)


def test_full_judge_returns_before_shadow_completes_and_drain_flushes_event():
    """The required async property, end to end: judge returns while the
    shadow is still blocked; drain times out while it is blocked; after
    release + drain the full_shadow_comparison event exists with the same
    content the synchronous path emits — even though the usage session
    closed before the shadow finished (ContextVar snapshot)."""
    summary = _make_summary()
    release_shadow = threading.Event()

    def _blocked_shadow(system_prompt, user_prompt, **kwargs):
        # Bounded wait so a regression to synchronous dispatch fails the
        # no-event-yet assertion below instead of hanging the suite.
        assert release_shadow.wait(timeout=10), "shadow never released"
        return _SHADOW_REJECT_RAW

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True), \
             patch("shared.config.SHADOW_ASYNC_ENABLED", True):
            with patch("shared.judger.assemble_full_evaluation_system", return_value="system"), \
                 patch("shared.judger.opus_llm_cached", return_value=_PRIMARY_SAVE_RAW), \
                 patch("shared.judger.shadow_full_llm", side_effect=_blocked_shadow):
                from shared.judger import drain_shadow_comparisons, full_judge

                with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                    decision = full_judge(summary, _v2_brief())

                    # The judge already returned; the shadow CANNOT have
                    # completed (it is blocked on release_shadow), so no
                    # comparison event may exist yet.
                    assert decision.decision == "SAVE"
                    assert not [
                        e
                        for e in _run_log_events(Path(td))
                        if e.get("event") == "full_shadow_comparison"
                    ]
                    # Drain with the task still blocked: times out, False.
                    assert drain_shadow_comparisons(timeout=0.05) is False

                # Usage session is now CLOSED; the worker's ContextVar
                # snapshot must still route the event to this run dir.
                release_shadow.set()
                assert drain_shadow_comparisons(timeout=10) is True

            events = _run_log_events(Path(td))

    shadow_events = [e for e in events if e.get("event") == "full_shadow_comparison"]
    assert len(shadow_events) == 1
    event = shadow_events[0]
    # Same content the synchronous path emits (pinned in
    # tests/test_judger_shadow_full.py) — the dispatch mode must not
    # change the event schema or values.
    assert event["primary_decision"] == "SAVE"
    assert event["shadow_decision"] == "REJECT"
    assert event["agrees"] is False
    assert event["shadow_parse_failed"] is False
    assert event["shadow_error"] is None
    assert event["latency_ms"] is not None


def test_facial_judge_returns_before_shadow_completes_and_drain_flushes_event():
    """Facial sibling of the full-eval async test — same worker, same
    drain, singular facial_shadow_comparison event schema."""
    snippet = _make_snippet()
    release_shadow = threading.Event()

    def _blocked_shadow(system_prompt, user_prompt, **kwargs):
        assert release_shadow.wait(timeout=10), "shadow never released"
        return "DECISION: FACIAL_NO\nREASON: disagreeing shadow"

    with tempfile.TemporaryDirectory() as td:
        with patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", True), \
             patch("shared.config.SHADOW_ASYNC_ENABLED", True):
            with llm_usage_session(Path(td) / "token-cost-log.jsonl"):
                with patch("shared.judger.assemble_facial_system", return_value="system"), \
                     patch(
                         "shared.judger.facial_llm",
                         return_value="DECISION: FACIAL_YES\nREASON: strong builder signal",
                     ), \
                     patch("shared.judger.shadow_facial_llm", side_effect=_blocked_shadow):
                    from shared.judger import drain_shadow_comparisons, facial_judge

                    decision = facial_judge(snippet, _v2_brief())

                    assert decision.decision == "FACIAL_YES"
                    assert not [
                        e
                        for e in _run_log_events(Path(td))
                        if e.get("event") == "facial_shadow_comparison"
                    ]

                    release_shadow.set()
                    assert drain_shadow_comparisons(timeout=10) is True

            events = _run_log_events(Path(td))

    shadow_events = [e for e in events if e.get("event") == "facial_shadow_comparison"]
    assert len(shadow_events) == 1
    event = shadow_events[0]
    assert event["batch"] is False
    assert event["candidate_count"] == 1
    assert event["primary_decision"] == "FACIAL_YES"
    assert event["shadow_decision"] == "FACIAL_NO"
    assert event["agrees"] is False


def test_drain_with_empty_queue_returns_true_immediately():
    from shared.judger import drain_shadow_comparisons

    assert drain_shadow_comparisons() is True
    assert drain_shadow_comparisons(timeout=0) is True
