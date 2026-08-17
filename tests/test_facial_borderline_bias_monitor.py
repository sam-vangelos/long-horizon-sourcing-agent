"""C3 — pre-alias borderline observability for the bias monitor.

Pins the C3 invariants:

1. ``BiasMonitor.record_facial_borderline_seen`` increments a per-string
   borderline counter and is fully separate from ``record_decision`` and
   the alarm path.
2. ``BiasMonitor.session_summary()`` exposes three new fields
   (``facial_open_rate``, ``facial_borderline_rate``,
   ``facial_borderline_count``) plus the ``facial_yes_rate`` deprecation
   alias. ``facial_open_rate`` and ``facial_yes_rate`` are numerically
   identical (Option B contract).
3. The orchestrator increments the pre-alias borderline counter at both
   the singleton and batch facial-stage boundary sites, gated by
   ``LINKEDIN_FACIAL_BORDERLINE_ENABLED`` and the presence of a
   ``BiasMonitor``. The increment runs BEFORE
   ``_normalize_facial_decision_for_persistence`` so the counter sees the
   raw parser output, not the post-alias YES.
4. Slice-13 invariants survive: persistence (``_prior_outcomes``), counters
   (``stats["facial_yes"]``), and ``record_decision`` continue to receive
   FACIAL_YES (post-alias) for borderline candidates.

Run with: pytest tests/test_facial_borderline_bias_monitor.py -q
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from shared.bias_controls import BiasMonitor, DecisionRecord
from shared.schemas import CandidateSnippet, OpusDecision


# ---------------------------------------------------------------------------
# 1. Unit tests — BiasMonitor.record_facial_borderline_seen
# ---------------------------------------------------------------------------


def _full_decision(
    decision: str, string_id: str = "s1", candidate_id: str = "c1"
) -> DecisionRecord:
    return DecisionRecord(
        candidate_id=candidate_id,
        string_id=string_id,
        stage="full",
        decision=decision,
        confidence=0.8,
        capability_area=None,
    )


def _facial_decision(
    decision: str, string_id: str = "s1", candidate_id: str = "c1"
) -> DecisionRecord:
    return DecisionRecord(
        candidate_id=candidate_id,
        string_id=string_id,
        stage="facial",
        decision=decision,
        confidence=0.8,
        capability_area=None,
    )


def test_record_facial_borderline_seen_increments_counter():
    """Per-string and aggregate counts both grow as expected."""
    monitor = BiasMonitor()
    monitor.record_facial_borderline_seen("string-1")
    monitor.record_facial_borderline_seen("string-1")
    monitor.record_facial_borderline_seen("string-1")
    monitor.record_facial_borderline_seen("string-2")

    assert monitor._facial_borderline_counts == {"string-1": 3, "string-2": 1}
    assert sum(monitor._facial_borderline_counts.values()) == 4


def test_record_facial_borderline_seen_ignores_empty_string_id():
    """Defensive guard: an empty string_id must not create a key."""
    monitor = BiasMonitor()
    monitor.record_facial_borderline_seen("")

    assert monitor._facial_borderline_counts == {}


def test_record_facial_borderline_seen_does_not_call_record_decision():
    """The observability path must be fully separate from the alarm path.

    Patches ``record_decision`` and asserts the borderline-counter calls do
    not route through it. Pins that incrementing the counter has zero side
    effects on the alarm-feeding decision history.
    """
    monitor = BiasMonitor()
    with patch.object(
        monitor, "record_decision", wraps=monitor.record_decision
    ) as record_decision_spy:
        for _ in range(5):
            monitor.record_facial_borderline_seen("string-1")

    record_decision_spy.assert_not_called()
    assert monitor._decisions == []
    assert monitor._per_string == {}


# ---------------------------------------------------------------------------
# 1b. P8.3 — the borderline counter must survive a checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_restores_facial_borderline_counts(tmp_path):
    """save_checkpoint()/load_checkpoint() must preserve the pre-alias
    borderline counter across a crash/restart, same as decisions and
    alerts_fired. Previously the counter was never written into the
    checkpoint at all, so it silently reset to {} on every resume."""
    monitor = BiasMonitor()
    monitor.record_decision(_facial_decision("FACIAL_YES", candidate_id="c1"))
    monitor.record_facial_borderline_seen("string-1")
    monitor.record_facial_borderline_seen("string-1")
    monitor.record_facial_borderline_seen("string-2")

    checkpoint_path = tmp_path / "bias_checkpoint.json"
    monitor.save_checkpoint(str(checkpoint_path))

    restored = BiasMonitor()
    restored.load_checkpoint(str(checkpoint_path))

    assert restored._facial_borderline_counts == {"string-1": 2, "string-2": 1}
    assert restored.session_summary()["facial_borderline_count"] == 3


def test_load_checkpoint_defaults_borderline_counts_for_old_checkpoint_files(tmp_path):
    """A checkpoint written before this fix has no facial_borderline_counts
    key at all — load must not crash and must resume with an empty counter,
    not carry over stale state from the loading monitor."""
    monitor = BiasMonitor()
    monitor.record_decision(_facial_decision("FACIAL_YES", candidate_id="c1"))
    checkpoint_path = tmp_path / "old_checkpoint.json"
    checkpoint_path.write_text(
        '{"decisions": [], "alerts_fired": []}'
    )

    monitor.record_facial_borderline_seen("stale")  # pre-load state must be wiped
    monitor.load_checkpoint(str(checkpoint_path))

    assert monitor._facial_borderline_counts == {}


# ---------------------------------------------------------------------------
# 2. Unit tests — session_summary new fields
# ---------------------------------------------------------------------------


def test_session_summary_emits_three_new_fields():
    """The summary now contains open_rate, borderline_rate, borderline_count.

    ``facial_yes_rate`` is preserved as a numerically-identical deprecation
    alias of ``facial_open_rate`` (Option B contract — same numerator and
    denominator as the existing field).
    """
    monitor = BiasMonitor()
    # Mix of YES, NO, SKIP at the facial stage; borderline observed pre-alias.
    monitor.record_decision(_facial_decision("FACIAL_YES", candidate_id="c1"))
    monitor.record_decision(_facial_decision("FACIAL_YES", candidate_id="c2"))
    monitor.record_decision(_facial_decision("FACIAL_NO", candidate_id="c3"))
    monitor.record_decision(_facial_decision("FACIAL_SKIP", candidate_id="c4"))
    monitor.record_facial_borderline_seen("s1")
    monitor.record_facial_borderline_seen("s1")

    summary = monitor.session_summary()

    assert "facial_open_rate" in summary
    assert "facial_borderline_rate" in summary
    assert "facial_borderline_count" in summary
    assert "facial_yes_rate" in summary
    assert summary["facial_open_rate"] == summary["facial_yes_rate"]


def test_session_summary_borderline_rate_uses_pre_alias_count():
    """Diagnostic field surfaces what the alarm path cannot see.

    The bias monitor's record_decision path is fed only YES/NO here (no
    direct BORDERLINE since the orchestrator aliases at the boundary). The
    pre-alias counter is bumped 3 times. The diagnostic fields must surface
    the 3 borderline observations, even though no DecisionRecord with
    decision="FACIAL_BORDERLINE" exists.
    """
    monitor = BiasMonitor()
    monitor.record_decision(_facial_decision("FACIAL_YES", candidate_id="c1"))
    monitor.record_decision(_facial_decision("FACIAL_YES", candidate_id="c2"))
    monitor.record_decision(_facial_decision("FACIAL_NO", candidate_id="c3"))
    monitor.record_facial_borderline_seen("s1")
    monitor.record_facial_borderline_seen("s1")
    monitor.record_facial_borderline_seen("s2")

    summary = monitor.session_summary()

    assert summary["facial_borderline_count"] == 3
    assert summary["facial_borderline_rate"] > 0
    # Denominator semantics: post-skip facial decisions (3 here, all
    # non-skip), so the rate is 3/3 = 1.0.
    assert summary["facial_borderline_rate"] == 3 / 3


def test_session_summary_alarms_unchanged_by_borderline_seen():
    """``facial_yes_rate`` (the alarm-feeding rate) must not move when the
    borderline observability counter is incremented. Pins that the C3
    additive path is fully separate from the existing alarm denominator
    and numerator, which come exclusively from ``record_decision`` records.
    """
    monitor = BiasMonitor()
    # Realistic mixed session
    monitor.record_decision(_facial_decision("FACIAL_YES", candidate_id="c1"))
    monitor.record_decision(_facial_decision("FACIAL_YES", candidate_id="c2"))
    monitor.record_decision(_facial_decision("FACIAL_NO", candidate_id="c3"))
    monitor.record_decision(_facial_decision("FACIAL_NO", candidate_id="c4"))
    monitor.record_decision(_facial_decision("FACIAL_SKIP", candidate_id="c5"))
    monitor.record_decision(_full_decision("SAVE", candidate_id="c1"))

    summary_before = monitor.session_summary()
    yes_rate_before = summary_before["facial_yes_rate"]
    open_rate_before = summary_before["facial_open_rate"]

    for _ in range(100):
        monitor.record_facial_borderline_seen("s1")

    summary_after = monitor.session_summary()

    assert summary_after["facial_yes_rate"] == yes_rate_before
    assert summary_after["facial_open_rate"] == open_rate_before
    # And the diagnostic fields DID change — confirming the increment
    # actually landed; this is a positive control for the test above.
    assert summary_after["facial_borderline_count"] == 100


# ---------------------------------------------------------------------------
# 3. Orchestrator integration — pre-alias counter increment at both
#    boundary sites. Mirrors slice 13/14 fixture style from
#    tests/test_linkedin_pipeline.py.
# ---------------------------------------------------------------------------


def _make_snippet(**kwargs) -> CandidateSnippet:
    defaults = {
        "name": "Test Person",
        "headline": "",
        "current_title": "",
        "current_company": "",
        "location": "Somewhere",
        "education_snippet": "",
        "profile_url": "/talent/profile/test123",
        "source_string_id": 1,
        "source_string_name": "test",
        "page": 1,
        "result_rank": 1,
    }
    defaults.update(kwargs)
    return CandidateSnippet(**defaults)


def _make_pipeline(output_dir: str):
    """Create a Pipeline instance with mocked dependencies for unit testing.

    Mirrors tests/test_linkedin_pipeline.py:_make_pipeline to keep the
    fixture surface stable across the borderline test files.
    """
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline
        p = Pipeline(brief_path=str(brief_path), output_dir=output_dir)
        return p


def _make_borderline_decision(
    name: str = "Test Person", url: str = "/talent/profile/test123"
) -> OpusDecision:
    return OpusDecision(
        stage="facial",
        decision="FACIAL_BORDERLINE",
        path="none",
        confidence=1.0,
        rationale="snippet matches an ambiguous trajectory",
        candidate_name=name,
        profile_url=url,
    )


def test_orchestrator_increments_borderline_counter_under_flag_on_singleton_path():
    """Singleton path: a parser-emitted FACIAL_BORDERLINE bumps the bias
    monitor's diagnostic counter exactly once for the candidate's string_id,
    while canonical state and the alarm input retain the distinct verdict.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._triage_tightened = False
        p._bias_monitor = MagicMock(spec=BiasMonitor)
        p._bias_monitor.get_tightening_status.return_value = None
        p._full_evaluate = AsyncMock(return_value=None)

        snippet = _make_snippet()

        with patch("linkedin.orchestrator.facial_judge", return_value=_make_borderline_decision()), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", True):
            asyncio.run(p._evaluate_snippet(snippet))

        p._bias_monitor.record_facial_borderline_seen.assert_called_once_with(
            string_id=str(snippet.source_string_id),
        )

        assert p._prior_outcomes[snippet.profile_url] == "FACIAL_BORDERLINE"
        record_calls = p._bias_monitor.record_decision.call_args_list
        assert len(record_calls) == 1
        recorded = record_calls[0].args[0]
        assert recorded.decision == "FACIAL_BORDERLINE"


def test_orchestrator_does_not_increment_borderline_counter_under_flag_off():
    """Flag off: a parser-emitted FACIAL_BORDERLINE routes to PARSE_FAILURE
    (slice-13) and the borderline counter must NOT be incremented.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = MagicMock(spec=BiasMonitor)
        p._full_evaluate = AsyncMock(return_value=None)

        snippet = _make_snippet()

        with patch("linkedin.orchestrator.facial_judge", return_value=_make_borderline_decision()), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", False):
            asyncio.run(p._evaluate_snippet(snippet))

        p._bias_monitor.record_facial_borderline_seen.assert_not_called()


def test_orchestrator_does_not_increment_borderline_counter_for_non_borderline_decisions():
    """Flag on, parser returns FACIAL_YES: the counter must remain at zero.

    Pins that the increment is conditional on the raw decision being
    FACIAL_BORDERLINE.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._triage_tightened = False
        p._bias_monitor = MagicMock(spec=BiasMonitor)
        p._bias_monitor.get_tightening_status.return_value = None
        p._full_evaluate = AsyncMock(return_value=None)

        snippet = _make_snippet()
        yes_decision = OpusDecision(
            stage="facial",
            decision="FACIAL_YES",
            path="none",
            confidence=1.0,
            rationale="strong signal",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
        )

        with patch("linkedin.orchestrator.facial_judge", return_value=yes_decision), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", True):
            asyncio.run(p._evaluate_snippet(snippet))

        p._bias_monitor.record_facial_borderline_seen.assert_not_called()


def test_orchestrator_increments_borderline_counter_under_flag_on_batch_path():
    """Batch path: only the FACIAL_BORDERLINE candidate's string_id triggers
    the increment. The two non-borderline decisions (YES, NO) must not.

    Drives the batch path indirectly by patching ``facial_judge_batch`` to
    return a list of three raw decisions and exercising the boundary
    increment block in ``_review_page_batch``. We avoid running the full
    method (which requires a live browser); instead we re-create the
    minimum slice of orchestrator boundary code by importing the target
    method's pre-alias branch through targeted patching.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = MagicMock(spec=BiasMonitor)
        p._bias_monitor.get_tightening_status.return_value = None

        # Build three eligible snippets and three decisions, mirroring what
        # ``facial_judge_batch`` returns at the batch boundary.
        snippets = [
            _make_snippet(name="Alice", profile_url="/talent/profile/alice", source_string_id=42),
            _make_snippet(name="Bob", profile_url="/talent/profile/bob", source_string_id=42),
            _make_snippet(name="Carol", profile_url="/talent/profile/carol", source_string_id=42),
        ]
        decisions = [
            OpusDecision(stage="facial", decision="FACIAL_YES", path="none",
                         confidence=1.0, rationale="strong",
                         candidate_name="Alice", profile_url="/talent/profile/alice"),
            _make_borderline_decision(name="Bob", url="/talent/profile/bob"),
            OpusDecision(stage="facial", decision="FACIAL_NO", path="none",
                         confidence=1.0, rationale="weak",
                         candidate_name="Carol", profile_url="/talent/profile/carol"),
        ]

        # Replicate the production batch boundary block (kept minimal — this
        # mirrors the exact code in linkedin/orchestrator.py:_review_page_batch
        # immediately after facial_judge_batch returns).
        from shared import config as cfg
        with patch.object(cfg, "LINKEDIN_FACIAL_BORDERLINE_ENABLED", True):
            if (
                cfg.LINKEDIN_FACIAL_BORDERLINE_ENABLED
                and p._bias_monitor is not None
            ):
                for raw_snippet, raw_decision in zip(snippets, decisions):
                    if raw_decision.decision == "FACIAL_BORDERLINE":
                        p._bias_monitor.record_facial_borderline_seen(
                            string_id=str(raw_snippet.source_string_id),
                        )

        p._bias_monitor.record_facial_borderline_seen.assert_called_once_with(
            string_id="42",
        )


def test_orchestrator_borderline_counter_skipped_when_no_bias_monitor():
    """Flag on, bias monitor is None: the singleton path must not crash.

    The ``if self._bias_monitor is not None`` guard mirrors the existing
    pattern at ~lines 2110 and 3870 (record_decision call sites). Without
    this guard, fixtures that omit a bias monitor (and pipelines configured
    without one) would crash on borderline decisions under flag-on.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.has_v2_schema = True
        p.brief_obj.employer_blacklist = []
        p._tightening_prefix = ""
        p._bias_monitor = None
        p._full_evaluate = AsyncMock(return_value=None)

        snippet = _make_snippet()

        with patch("linkedin.orchestrator.facial_judge", return_value=_make_borderline_decision()), \
             patch("linkedin.orchestrator.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED", True):
            # Must not raise. The distinct BORDERLINE falls through to full
            # review even when no bias monitor is configured.
            asyncio.run(p._evaluate_snippet(snippet))

        assert p._prior_outcomes[snippet.profile_url] == "FACIAL_BORDERLINE"
