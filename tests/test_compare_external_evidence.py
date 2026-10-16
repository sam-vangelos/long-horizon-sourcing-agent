"""Tests for ``tools/compare_external_evidence.py``.

Hard rules pinned by these tests:

- No live calls. ``full_judge``, ``full_judge_with_external_evidence``,
  ``should_request_external_evidence``, and ``fetch_external_candidate_evidence``
  are all patched.
- The tool never raises ``ApiBudgetExhaustedError``: a provider returning
  ``ExternalEvidenceFailure(reason="quota_exhausted")`` must degrade to
  ``(None, "quota_exhausted", trigger)`` and never propagate budget
  exceptions.
- ``--skip-external`` short-circuits the gate and the provider entirely.
- ``--evidence-fixture`` short-circuits the gate and the provider entirely.
- ``run_comparison`` returns 0 on success and 1 on hard input errors only.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.schemas import (
    CandidateProfileSummary,
    Education,
    EvidenceRef,
    Experience,
    ExternalCandidateEvidence,
    ExternalEvidenceFailure,
    ExternalFactBlock,
    ExternalInference,
    OpusDecision,
    TriggerDecision,
)

# Allow importing the script as a module.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import compare_external_evidence as cee  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_summary() -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name="Jane Doe",
        profile_url="https://www.linkedin.com/in/janedoe",
        headline="ML Researcher",
        experiences=[
            Experience(
                title="Research Scientist",
                company="OpenLab",
                start="2020",
                end="present",
                summary_bullets=["LLM evals", "RLHF stability"],
            )
        ],
        education=[Education(degree="PhD", school="MIT", field="ML")],
        skills_snippet=["python", "pytorch"],
    )


def _make_evidence(
    *,
    extra_urls: int = 0,
) -> ExternalCandidateEvidence:
    refs = [
        EvidenceRef(
            url="https://example.edu/thesis",
            title="Thesis",
            source_quality="high",
        ),
        EvidenceRef(
            url="https://example.edu/cv",
            title="CV",
            source_quality="medium",
        ),
    ]
    fact_block = ExternalFactBlock(
        topic="phd_thesis",
        facts=["Published thesis on RLHF stability."],
        evidence_refs=list(refs),
        source_quality="high",
    )
    inferences = [
        ExternalInference(
            claim="Likely deep RL knowledge.",
            basis_refs=[refs[0]],
            confidence=0.6,
        )
    ]
    extra_inferences = []
    for i in range(extra_urls):
        extra_inferences.append(
            ExternalInference(
                claim=f"Extra claim {i}",
                basis_refs=[
                    EvidenceRef(
                        url=f"https://example.edu/extra-{i}",
                        title=f"Extra {i}",
                        source_quality="medium",
                    )
                ],
                confidence=0.5,
            )
        )
    return ExternalCandidateEvidence(
        trigger_reason="academic_context",
        identity_confidence=0.7,
        profile_facts_used_for_matching=["name=Jane Doe"],
        external_fact_blocks=[fact_block],
        external_inferences=inferences + extra_inferences,
        unresolved_ambiguities=["A second Jane Doe at MIT publishes in vision."],
        do_not_use_for_judgment=[],
        raw_provider_model="sonar-deep-research",
        normalizer_model="",
    )


def _make_decision(
    *,
    decision: str = "REJECT",
    path: str = "none",
    confidence: float = 0.55,
    rationale: str = "Baseline rationale",
    name: str = "Jane Doe",
    profile_url: str = "https://www.linkedin.com/in/janedoe",
) -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision=decision,
        path=path,
        confidence=confidence,
        rationale=rationale,
        candidate_name=name,
        profile_url=profile_url,
    )


def _build_args(
    **overrides,
) -> argparse.Namespace:
    defaults = dict(
        profile_summary="-",
        brief="dummy-brief.json",
        skip_external=False,
        force_trigger=None,
        evidence_fixture=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# load_profile_summary
# ---------------------------------------------------------------------------


class TestLoadProfileSummary:
    def test_happy_path_from_file(self):
        summary = _make_summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            with path.open("w") as fh:
                json.dump(summary.to_dict(), fh)
            loaded = cee.load_profile_summary(str(path))
        assert isinstance(loaded, CandidateProfileSummary)
        assert loaded.name == summary.name
        assert loaded.profile_url == summary.profile_url
        assert len(loaded.experiences) == 1
        assert loaded.experiences[0].company == "OpenLab"
        assert len(loaded.education) == 1
        assert loaded.education[0].school == "MIT"

    def test_from_stdin(self, monkeypatch):
        summary = _make_summary()
        payload = json.dumps(summary.to_dict())
        monkeypatch.setattr("sys.stdin", io.StringIO(payload))
        loaded = cee.load_profile_summary("-")
        assert isinstance(loaded, CandidateProfileSummary)
        assert loaded.name == summary.name

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "nope.json")
            with pytest.raises(FileNotFoundError):
                cee.load_profile_summary(missing)


# ---------------------------------------------------------------------------
# load_evidence_fixture
# ---------------------------------------------------------------------------


class TestLoadEvidenceFixture:
    def test_round_trip(self):
        original = _make_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            with path.open("w") as fh:
                json.dump(original.to_dict(), fh)
            loaded = cee.load_evidence_fixture(str(path))
        assert isinstance(loaded, ExternalCandidateEvidence)
        assert loaded == original

    def test_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                cee.load_evidence_fixture(str(Path(tmp) / "nope.json"))


# ---------------------------------------------------------------------------
# format_evidence_block
# ---------------------------------------------------------------------------


class TestFormatEvidenceBlock:
    def test_evidence_present_renders_status_and_urls(self):
        evidence = _make_evidence()
        block = cee.format_evidence_block(
            evidence,
            status="evidence_present",
            trigger_reason="academic_context",
        )
        assert "=== External Evidence ===" in block
        assert "status: evidence_present" in block
        assert "trigger_reason: academic_context" in block
        assert "identity_confidence: 0.70" in block
        assert "fact_blocks: 1" in block
        assert "inferences: 1" in block
        assert "unresolved_ambiguities: 1" in block
        assert "https://example.edu/thesis" in block
        assert "https://example.edu/cv" in block
        # Cap-at-10 not triggered with only 2 unique URLs.
        assert "more" not in block

    def test_evidence_present_caps_at_ten_urls(self):
        # 1 fact-block ref (thesis) + 1 fact-block ref (cv) + 1 inference ref (thesis dup) +
        # extra_urls=15 inferences each adding a new URL = 17 unique URLs total.
        evidence = _make_evidence(extra_urls=15)
        block = cee.format_evidence_block(
            evidence,
            status="evidence_present",
            trigger_reason="academic_context",
        )
        # The header should reflect total unique URLs.
        assert "evidence_refs (count=17)" in block
        # Only 10 should be printed individually, then a "and N more" summary.
        printed_urls = [
            line for line in block.splitlines() if line.strip().startswith("- http")
        ]
        assert len(printed_urls) == 10
        assert "and 7 more" in block

    def test_failure_path_renders_zero_refs(self):
        block = cee.format_evidence_block(
            None,
            status="quota_exhausted",
            trigger_reason="academic_context",
        )
        assert "status: quota_exhausted" in block
        assert "evidence_refs (count=0)" in block
        assert "identity_confidence: n/a" in block
        assert "fact_blocks: 0" in block
        # No URL bullets should appear.
        assert "- http" not in block


# ---------------------------------------------------------------------------
# format_diff_block
# ---------------------------------------------------------------------------


class TestFormatDiffBlock:
    def test_same_decision_diff(self):
        diff = {
            "computed": True,
            "decision_changed": False,
            "decision_baseline": "SAVE",
            "decision_enriched": "SAVE",
            "path_changed": False,
            "path_baseline": "DIRECT:Data Curation",
            "path_enriched": "DIRECT:Data Curation",
            "rationale_changed": True,
            "confidence_delta": 0.05,
        }
        block = cee.format_diff_block(diff)
        assert "=== Diff ===" in block
        assert "decision_changed: False" in block
        assert "baseline=SAVE | enriched=SAVE" in block
        assert "path_changed:     False" in block
        assert "rationale_changed: True" in block
        assert "confidence_delta: +0.050" in block

    def test_changed_decision_diff_with_baseline_enriched(self):
        baseline = _make_decision(decision="REJECT", confidence=0.7)
        enriched = _make_decision(decision="SAVE", confidence=0.82)
        diff = {
            "computed": True,
            "decision_changed": True,
            "decision_baseline": "REJECT",
            "decision_enriched": "SAVE",
            "path_changed": True,
            "path_baseline": "none",
            "path_enriched": "DIRECT:Data Curation",
            "rationale_changed": True,
            "confidence_delta": 0.12,
        }
        block = cee.format_diff_block(diff, baseline=baseline, enriched=enriched)
        assert "decision_changed: True" in block
        assert "baseline=REJECT | enriched=SAVE" in block
        assert "path_changed:     True" in block
        assert "confidence_delta: +0.120" in block
        assert "baseline=0.700" in block
        assert "enriched=0.820" in block

    def test_not_computed(self):
        diff = {"computed": False, "reason": "skipped_by_flag"}
        block = cee.format_diff_block(diff)
        assert "diff: not computed (reason=skipped_by_flag)" in block


# ---------------------------------------------------------------------------
# gather_evidence
# ---------------------------------------------------------------------------


class TestGatherEvidence:
    def test_skip_external_short_circuits(self):
        summary = _make_summary()
        brief = MagicMock()
        with patch.object(cee, "fetch_external_candidate_evidence") as mock_fetch:
            with patch.object(cee, "should_request_external_evidence") as mock_gate:
                evidence, status, trigger = cee.gather_evidence(
                    summary=summary,
                    brief=brief,
                    force_trigger=None,
                    skip_external=True,
                    evidence_fixture=None,
                    perplexity_api_key="real-key",
                )
        assert evidence is None
        assert status == "skipped_by_flag"
        assert trigger is None
        mock_fetch.assert_not_called()
        mock_gate.assert_not_called()

    def test_evidence_fixture_loaded(self):
        summary = _make_summary()
        brief = MagicMock()
        evidence_fixture = _make_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            with path.open("w") as fh:
                json.dump(evidence_fixture.to_dict(), fh)
            with patch.object(cee, "fetch_external_candidate_evidence") as mock_fetch:
                with patch.object(cee, "should_request_external_evidence") as mock_gate:
                    evidence, status, trigger = cee.gather_evidence(
                        summary=summary,
                        brief=brief,
                        force_trigger=None,
                        skip_external=False,
                        evidence_fixture=str(path),
                        perplexity_api_key="",
                    )
        assert isinstance(evidence, ExternalCandidateEvidence)
        assert evidence == evidence_fixture
        assert status == "fixture_loaded"
        assert trigger is None
        mock_fetch.assert_not_called()
        mock_gate.assert_not_called()

    def test_no_api_key_no_fixture_returns_disabled_no_api_key(self):
        summary = _make_summary()
        brief = MagicMock()
        trigger = TriggerDecision(
            should_run=True,
            reason="academic_context",
            skip_reason="",
            signals={"fired": "academic_context"},
        )
        with patch.object(
            cee, "should_request_external_evidence", return_value=trigger
        ) as mock_gate:
            with patch.object(cee, "fetch_external_candidate_evidence") as mock_fetch:
                evidence, status, returned_trigger = cee.gather_evidence(
                    summary=summary,
                    brief=brief,
                    force_trigger=None,
                    skip_external=False,
                    evidence_fixture=None,
                    perplexity_api_key="",
                )
        assert evidence is None
        assert status == "disabled_no_api_key"
        assert returned_trigger == trigger
        mock_gate.assert_called_once()
        mock_fetch.assert_not_called()

    def test_force_trigger_bypasses_gate(self):
        summary = _make_summary()
        brief = MagicMock()
        evidence_obj = _make_evidence()
        with patch.object(
            cee, "should_request_external_evidence"
        ) as mock_gate:
            with patch.object(
                cee, "fetch_external_candidate_evidence", return_value=evidence_obj
            ) as mock_fetch:
                evidence, status, trigger = cee.gather_evidence(
                    summary=summary,
                    brief=brief,
                    force_trigger="academic_context",
                    skip_external=False,
                    evidence_fixture=None,
                    perplexity_api_key="real-key",
                )
        assert evidence is evidence_obj
        assert status == "evidence_present"
        assert trigger is not None
        assert trigger.should_run is True
        assert trigger.reason == "academic_context"
        mock_gate.assert_not_called()
        mock_fetch.assert_called_once()

    def test_provider_quota_exhausted_does_not_raise(self):
        """Pins slice 1 invariant: ``ApiBudgetExhaustedError`` must NOT propagate.

        The provider returns ``ExternalEvidenceFailure(reason="quota_exhausted")``
        as a typed result; the tool must absorb it the same way slice 1 +
        slice 2 do.
        """

        from shared.failures import ApiBudgetExhaustedError  # noqa: F401  (asserts importable)

        summary = _make_summary()
        brief = MagicMock()
        failure = ExternalEvidenceFailure(
            reason="quota_exhausted",
            detail="credit balance is too low",
            provider="perplexity",
            http_status=402,
        )
        trigger = TriggerDecision(
            should_run=True,
            reason="academic_context",
            skip_reason="",
            signals={"fired": "academic_context"},
        )
        with patch.object(
            cee, "should_request_external_evidence", return_value=trigger
        ):
            with patch.object(
                cee, "fetch_external_candidate_evidence", return_value=failure
            ):
                evidence, status, returned_trigger = cee.gather_evidence(
                    summary=summary,
                    brief=brief,
                    force_trigger=None,
                    skip_external=False,
                    evidence_fixture=None,
                    perplexity_api_key="real-key",
                )
        assert evidence is None
        assert status == "quota_exhausted"
        assert returned_trigger == trigger


# ---------------------------------------------------------------------------
# run_comparison
# ---------------------------------------------------------------------------


def _write_summary_to(tmp: Path, summary: CandidateProfileSummary) -> Path:
    path = tmp / "summary.json"
    with path.open("w") as fh:
        json.dump(summary.to_dict(), fh)
    return path


def _write_evidence_to(tmp: Path, evidence: ExternalCandidateEvidence) -> Path:
    path = tmp / "evidence.json"
    with path.open("w") as fh:
        json.dump(evidence.to_dict(), fh)
    return path


class TestRunComparison:
    def test_end_to_end_with_fixture_evidence(self, capsys):
        summary = _make_summary()
        evidence = _make_evidence()
        baseline_decision = _make_decision(decision="REJECT", confidence=0.55)
        enriched_decision = _make_decision(decision="SAVE", confidence=0.78)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_path = _write_summary_to(tmp_path, summary)
            evidence_path = _write_evidence_to(tmp_path, evidence)
            args = _build_args(
                profile_summary=str(summary_path),
                brief="dummy-brief.json",
                evidence_fixture=str(evidence_path),
            )

            mock_brief = MagicMock(id="brief-test", role_title="ML Engineer")
            with patch.object(cee, "load_brief", return_value=mock_brief):
                rc = cee.run_comparison(
                    args,
                    opus_baseline=lambda summary, brief: baseline_decision,
                    opus_enriched=lambda summary, evidence, brief: enriched_decision,
                )

        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "=== Candidate ===" in out
        assert "name: Jane Doe" in out
        assert "brief: brief-test" in out
        assert "=== External Evidence ===" in out
        assert "status: fixture_loaded" in out
        assert "https://example.edu/thesis" in out
        assert "=== Baseline judgment (canonical) ===" in out
        assert '"decision": "REJECT"' in out
        assert "=== Enriched judgment (shadow / debug) ===" in out
        assert '"decision": "SAVE"' in out
        assert "=== Diff ===" in out
        assert "decision_changed: True" in out
        assert "baseline=REJECT | enriched=SAVE" in out
        assert "baseline=0.550" in out
        assert "enriched=0.780" in out

    def test_skip_external_prints_enriched_null_and_diff_not_computed(self, capsys):
        summary = _make_summary()
        baseline_decision = _make_decision(decision="REJECT", confidence=0.6)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_path = _write_summary_to(tmp_path, summary)
            args = _build_args(
                profile_summary=str(summary_path),
                brief="dummy-brief.json",
                skip_external=True,
            )

            mock_brief = MagicMock(id="brief-test", role_title="ML Engineer")
            opus_enriched = MagicMock()
            with patch.object(cee, "load_brief", return_value=mock_brief):
                rc = cee.run_comparison(
                    args,
                    opus_baseline=lambda summary, brief: baseline_decision,
                    opus_enriched=opus_enriched,
                )

        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "status: skipped_by_flag" in out
        assert '"decision": "REJECT"' in out
        assert "enriched: null (reason=skipped_by_flag)" in out
        assert "diff: not computed (reason=skipped_by_flag)" in out
        opus_enriched.assert_not_called()

    def test_bad_profile_summary_json_returns_1(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            with bad.open("w") as fh:
                fh.write("{ not json")
            args = _build_args(
                profile_summary=str(bad),
                brief="dummy-brief.json",
            )

            with patch.object(cee, "load_brief") as mock_load_brief:
                rc = cee.run_comparison(args)

        assert rc == 1
        captured = capsys.readouterr()
        # Error must be on stderr, not stdout.
        assert "ERROR" in captured.err
        # We bailed before loading the brief.
        mock_load_brief.assert_not_called()

    def test_missing_profile_summary_returns_1(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            args = _build_args(
                profile_summary=str(Path(tmp) / "nope.json"),
                brief="dummy-brief.json",
            )
            rc = cee.run_comparison(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
