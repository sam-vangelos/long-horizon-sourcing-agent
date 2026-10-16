"""Tests for slice 1 of the perplexity-evidence-augmentation feature.

Covers:
  - dataclass round-trips (ExternalCandidateEvidence, ExternalEvidenceFailure)
  - trigger gate heuristics (sparse_profile, academic_context, no_trigger)
  - normalizer (parse failure, weak citations, happy path with explicit
    fact / inference / ambiguity separation)
  - provider failure modes (disabled-no-key, disabled-by-config, quota,
    timeout, unexpected exception)

Hard rules pinned by these tests:
  - The provider returns typed ExternalEvidenceFailure on any error path.
    It NEVER raises ApiBudgetExhaustedError. Quota errors must be classified
    as reason="quota_exhausted", not "unknown".
  - No imports from linkedin/, github/, or market_intelligence/.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from shared.brief_schema import (
    BiasControls,
    Brief,
    CapabilityArea,
    DepthDistinction,
    EmployerSignalRule,
    FacialCalibration,
    NonFitPattern,
)
from shared.external_evidence import (
    fetch_external_candidate_evidence,
    normalize_perplexity_response,
    should_request_external_evidence,
)
from shared.failures import ApiBudgetExhaustedError
from shared.schemas import (
    CandidateProfileSummary,
    Education,
    EvidenceRef,
    Experience,
    ExternalCandidateEvidence,
    ExternalEvidenceFailure,
    ExternalFactBlock,
    ExternalInference,
    TriggerDecision,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brief(**overrides) -> Brief:
    """Minimal valid Brief for slice 1 tests; the gate ignores brief content."""
    defaults = dict(
        role_title="Test Role",
        role_level="IC5",
        role_summary="Build the thing.",
        geography="United States",
        linkedin_project="proj-test",
        capability_areas=[
            CapabilityArea(
                name="Test Area",
                description="What the work looks like.",
                builder_signals=["builds X"],
                user_signals=[],
            )
        ],
        depth_distinction=DepthDistinction(
            builder_definition="Builds X end-to-end.",
            user_definition="Calls a hosted X API.",
            edge_case_guidance="Borderline cases default to user.",
        ),
        non_fit_patterns=[
            NonFitPattern(
                label="Adjacent thing",
                description="Looks like X but isn't.",
                why_not="Different stack.",
            )
        ],
        employer_signal_rules=[
            EmployerSignalRule(
                tier="neutral",
                employer_patterns=["Some Co"],
                evidence_required="hands-on X",
                save_on_employer_alone=False,
            )
        ],
        minimum_years_experience=3,
        minimum_bar_description="3y hands-on X.",
        facial_calibration=FacialCalibration(
            expected_yes_rate_low=0.25,
            expected_yes_rate_high=0.55,
            fast_exit_patterns=["pure ops"],
            trajectory_yes_patterns=["X engineering"],
            trajectory_ambiguous_patterns=["mixed"],
            trajectory_no_patterns=["pure frontend"],
        ),
        bias_controls=BiasControls(),
    )
    defaults.update(overrides)
    return Brief(**defaults)


def _make_summary(
    *,
    name: str = "Jane Doe",
    experiences: list[Experience] | None = None,
    education: list[Education] | None = None,
) -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name=name,
        profile_url="https://www.linkedin.com/in/janedoe",
        headline="Engineer",
        experiences=experiences if experiences is not None else [],
        education=education if education is not None else [],
        skills_snippet=[],
    )


def _mock_openai_module(*, raises: BaseException | None = None, response: object | None = None):
    """Build a mock ``openai`` module patched into ``sys.modules``.

    Either ``raises`` (an exception ``client.responses.create`` should raise)
    or ``response`` (the object it should return) must be provided. Returns
    ``(mock_module, mock_client)``.
    """
    mock_client = MagicMock()
    if raises is not None:
        mock_client.responses.create.side_effect = raises
    else:
        mock_client.responses.create.return_value = response
    mock_module = MagicMock()
    mock_module.OpenAI.return_value = mock_client
    return mock_module, mock_client


# ---------------------------------------------------------------------------
# Schema round-trips
# ---------------------------------------------------------------------------

class TestSchemaRoundTrip:
    def test_external_candidate_evidence_round_trip(self):
        original = ExternalCandidateEvidence(
            trigger_reason="academic_context",
            identity_confidence=0.75,
            profile_facts_used_for_matching=["name=Jane Doe", "school=MIT"],
            external_fact_blocks=[
                ExternalFactBlock(
                    topic="phd_thesis",
                    facts=["Thesis on RLHF stability."],
                    evidence_refs=[
                        EvidenceRef(
                            url="https://example.edu/thesis",
                            title="Thesis",
                            source_quality="high",
                        )
                    ],
                    source_quality="high",
                )
            ],
            external_inferences=[
                ExternalInference(
                    claim="Likely strong RL background.",
                    basis_refs=[
                        EvidenceRef(
                            url="https://example.edu/thesis",
                            title="Thesis",
                            source_quality="high",
                        )
                    ],
                    confidence=0.6,
                )
            ],
            unresolved_ambiguities=["Common name; multiple Jane Does in publications."],
            do_not_use_for_judgment=["unverified blog claim"],
            raw_provider_model="sonar-deep-research",
            normalizer_model="",
        )
        round_tripped = ExternalCandidateEvidence.from_dict(original.to_dict())
        assert round_tripped == original
        assert round_tripped.identity_confidence == pytest.approx(0.75)
        assert round_tripped.external_inferences[0].confidence == pytest.approx(0.6)

    def test_external_evidence_failure_round_trip(self):
        original = ExternalEvidenceFailure(
            reason="quota_exhausted",
            detail="credit balance is too low",
            provider="perplexity",
            http_status=402,
        )
        round_tripped = ExternalEvidenceFailure.from_dict(original.to_dict())
        assert round_tripped == original

    def test_trigger_decision_round_trip(self):
        original = TriggerDecision(
            should_run=True,
            reason="sparse_profile",
            skip_reason="",
            signals={"experience_count": 1, "bullet_total": 0},
        )
        round_tripped = TriggerDecision.from_dict(original.to_dict())
        assert round_tripped == original


# ---------------------------------------------------------------------------
# Trigger gate
# ---------------------------------------------------------------------------

class TestTriggerGate:
    def test_phd_education_triggers_academic_context(self):
        summary = _make_summary(
            education=[
                Education(degree="PhD", school="MIT", field="ML"),
            ],
            experiences=[
                Experience(
                    title="Researcher",
                    company="Lab X",
                    summary_bullets=["a", "b", "c", "d"],
                ),
                Experience(
                    title="Researcher",
                    company="Lab Y",
                    summary_bullets=["e", "f", "g"],
                ),
            ],
        )
        decision = should_request_external_evidence(summary=summary, brief=_make_brief())
        assert decision.should_run is True
        assert decision.reason == "academic_context"
        assert decision.signals["has_phd"] is True
        assert decision.signals["fired"] == "academic_context"

    def test_phd_variants_match(self):
        for variant in ("Ph.D", "Ph.D.", "Doctor of Philosophy", "phd"):
            summary = _make_summary(
                education=[Education(degree=variant, school="X", field="Y")],
            )
            decision = should_request_external_evidence(
                summary=summary, brief=_make_brief()
            )
            assert decision.should_run is True
            assert decision.reason == "academic_context", (
                f"variant {variant!r} should trigger academic_context"
            )

    def test_sparse_one_experience_triggers_sparse_profile(self):
        summary = _make_summary(
            experiences=[
                Experience(
                    title="Engineer",
                    company="Acme",
                    summary_bullets=["one bullet"],
                )
            ],
            education=[Education(degree="BS", school="State U", field="CS")],
        )
        decision = should_request_external_evidence(summary=summary, brief=_make_brief())
        assert decision.should_run is True
        assert decision.reason == "sparse_profile"
        assert decision.signals["fired"] == "sparse_profile"

    def test_rich_profile_no_trigger(self):
        summary = _make_summary(
            experiences=[
                Experience(
                    title="Engineer",
                    company="Acme",
                    summary_bullets=["a", "b", "c"],
                ),
                Experience(
                    title="Engineer",
                    company="Beta",
                    summary_bullets=["d", "e", "f"],
                ),
                Experience(
                    title="Engineer",
                    company="Gamma",
                    summary_bullets=["g", "h"],
                ),
            ],
            education=[Education(degree="BS", school="State U", field="CS")],
        )
        decision = should_request_external_evidence(summary=summary, brief=_make_brief())
        assert decision.should_run is False
        assert decision.skip_reason == "no_trigger_matched"
        assert decision.signals["fired"] == "none"

    def test_brief_is_not_consulted_in_slice_1(self):
        """The brief argument is reserved for slice 2/3; gate must ignore it."""
        summary = _make_summary(
            education=[Education(degree="PhD", school="MIT", field="ML")],
        )
        decision_default = should_request_external_evidence(
            summary=summary, brief=_make_brief()
        )
        decision_alt = should_request_external_evidence(
            summary=summary,
            brief=_make_brief(role_title="Totally Different Role"),
        )
        assert decision_default == decision_alt


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_parse_failure_on_malformed_json(self):
        result = normalize_perplexity_response(
            raw_text="this is not json {{{",
            raw_sources=[],
            trigger_reason="academic_context",
            raw_provider_model="sonar",
        )
        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "parse_failure"
        assert result.provider == "perplexity"

    def test_weak_citations_rejected(self):
        payload = json.dumps(
            {
                "trigger_reason": "academic_context",
                "identity_confidence": 0.8,
                "profile_facts_used_for_matching": ["name=Jane Doe"],
                "external_fact_blocks": [
                    {
                        "topic": "employer_overview",
                        "facts": ["Acme is a robotics startup."],
                        "evidence_refs": [],
                        "source_quality": "low",
                    }
                ],
                "external_inferences": [],
                "unresolved_ambiguities": [],
            }
        )
        result = normalize_perplexity_response(
            raw_text=payload,
            raw_sources=[],
            trigger_reason="academic_context",
            raw_provider_model="sonar",
            min_citations=2,
        )
        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "weak_citations"

    def test_happy_path_preserves_separation(self):
        payload = json.dumps(
            {
                "trigger_reason": "academic_context",
                "identity_confidence": 0.7,
                "profile_facts_used_for_matching": ["name=Jane Doe", "school=MIT"],
                "external_fact_blocks": [
                    {
                        "topic": "phd_thesis",
                        "facts": ["Thesis on RLHF stability published 2023."],
                        "evidence_refs": [
                            {
                                "url": "https://example.edu/thesis",
                                "title": "Thesis page",
                                "source_quality": "high",
                            }
                        ],
                        "source_quality": "high",
                    }
                ],
                "external_inferences": [
                    {
                        "claim": "Likely deep RL knowledge.",
                        "basis_refs": [
                            {
                                "url": "https://example.edu/thesis",
                                "title": "Thesis page",
                                "source_quality": "high",
                            }
                        ],
                        "confidence": 0.55,
                    },
                    {
                        # This inference has no basis_refs — must be PRESERVED.
                        "claim": "Plausibly familiar with industrial RLHF.",
                        "basis_refs": [],
                        "confidence": 0.2,
                    },
                ],
                "unresolved_ambiguities": [
                    "A second 'Jane Doe' at MIT publishes in vision.",
                ],
                "do_not_use_for_judgment": ["unverified blog claim"],
            }
        )
        result = normalize_perplexity_response(
            raw_text=payload,
            raw_sources=[],
            trigger_reason="academic_context",
            raw_provider_model="sonar-deep-research",
            min_citations=2,
        )
        assert isinstance(result, ExternalCandidateEvidence)
        assert result.trigger_reason == "academic_context"
        assert result.identity_confidence == pytest.approx(0.7)
        assert len(result.external_fact_blocks) == 1
        assert result.external_fact_blocks[0].topic == "phd_thesis"
        assert len(result.external_fact_blocks[0].evidence_refs) == 1
        assert result.external_fact_blocks[0].source_quality == "high"
        assert len(result.external_inferences) == 2
        assert result.external_inferences[1].claim.startswith("Plausibly")
        assert result.external_inferences[1].basis_refs == []
        assert "Jane Doe" in result.unresolved_ambiguities[0]
        assert result.do_not_use_for_judgment == ["unverified blog claim"]
        assert result.raw_provider_model == "sonar-deep-research"
        assert result.normalizer_model == ""

    def test_missing_identity_confidence_defaults_to_zero(self):
        payload = json.dumps(
            {
                "trigger_reason": "sparse_profile",
                "external_fact_blocks": [
                    {
                        "topic": "employer",
                        "facts": ["Fact 1.", "Fact 2."],
                        "evidence_refs": [
                            {"url": "https://a.example.com", "source_quality": "medium"},
                            {"url": "https://b.example.com", "source_quality": "low"},
                        ],
                    }
                ],
                "external_inferences": [],
            }
        )
        result = normalize_perplexity_response(
            raw_text=payload,
            raw_sources=[],
            trigger_reason="sparse_profile",
            raw_provider_model="sonar",
            min_citations=2,
        )
        assert isinstance(result, ExternalCandidateEvidence)
        assert result.identity_confidence == 0.0


# ---------------------------------------------------------------------------
# Provider — failure-mode classification
# ---------------------------------------------------------------------------

class _FakeBudgetError(Exception):
    """A stand-in error that contains quota markers in its message."""


class _FakeHttpError(Exception):
    """A stand-in HTTP-shaped error with a status_code attribute."""

    def __init__(
        self,
        message: str,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(
            status_code=status_code,
            headers=headers or {},
        )


def _provider_call(**overrides):
    """Invoke the provider with sensible defaults for tests."""
    summary = overrides.pop("summary", None) or _make_summary()
    brief = overrides.pop("brief", None) or _make_brief()
    trigger = overrides.pop("trigger", None) or TriggerDecision(
        should_run=True, reason="academic_context"
    )
    identity_hints = overrides.pop("identity_hints", None)
    return fetch_external_candidate_evidence(
        summary=summary,
        brief=brief,
        trigger=trigger,
        identity_hints=identity_hints,
    )


class TestProviderDisabled:
    def test_disabled_no_api_key(self):
        """No PERPLEXITY_API_KEY → returns disabled_no_api_key, never imports openai."""
        from shared.external_evidence import provider as provider_mod

        fake_config = MagicMock(
            PERPLEXITY_API_KEY="",
            LINKEDIN_EXTERNAL_EVIDENCE_ENABLED=True,
            LINKEDIN_EXTERNAL_EVIDENCE_MODEL="",
            LINKEDIN_EXTERNAL_EVIDENCE_TIMEOUT_SECONDS=90.0,
            LINKEDIN_EXTERNAL_EVIDENCE_MAX_OUTPUT_TOKENS=4096,
            LINKEDIN_EXTERNAL_EVIDENCE_MIN_CITATIONS=2,
            LINKEDIN_EXTERNAL_EVIDENCE_PERPLEXITY_PRESET="",
        )

        # If the provider tried to import openai, this sentinel would explode.
        sentinel_openai = MagicMock()
        sentinel_openai.OpenAI.side_effect = AssertionError(
            "openai must NOT be imported when API key is missing"
        )

        orig_config = provider_mod.config
        try:
            provider_mod.config = fake_config
            with patch.dict("sys.modules", {"openai": sentinel_openai}):
                result = _provider_call()
        finally:
            provider_mod.config = orig_config

        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "disabled_no_api_key"
        sentinel_openai.OpenAI.assert_not_called()

    def test_disabled_by_config(self):
        """ENABLED=False with key present → returns disabled_by_config."""
        from shared.external_evidence import provider as provider_mod

        fake_config = MagicMock(
            PERPLEXITY_API_KEY="present-key",
            LINKEDIN_EXTERNAL_EVIDENCE_ENABLED=False,
            LINKEDIN_EXTERNAL_EVIDENCE_MODEL="",
            LINKEDIN_EXTERNAL_EVIDENCE_TIMEOUT_SECONDS=90.0,
            LINKEDIN_EXTERNAL_EVIDENCE_MAX_OUTPUT_TOKENS=4096,
            LINKEDIN_EXTERNAL_EVIDENCE_MIN_CITATIONS=2,
            LINKEDIN_EXTERNAL_EVIDENCE_PERPLEXITY_PRESET="",
        )

        sentinel_openai = MagicMock()
        sentinel_openai.OpenAI.side_effect = AssertionError(
            "openai must NOT be imported when feature is config-disabled"
        )

        orig_config = provider_mod.config
        try:
            provider_mod.config = fake_config
            with patch.dict("sys.modules", {"openai": sentinel_openai}):
                result = _provider_call()
        finally:
            provider_mod.config = orig_config

        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "disabled_by_config"
        sentinel_openai.OpenAI.assert_not_called()


class TestProviderFailureClassification:
    def _enabled_config(self) -> MagicMock:
        return MagicMock(
            PERPLEXITY_API_KEY="present-key",
            LINKEDIN_EXTERNAL_EVIDENCE_ENABLED=True,
            LINKEDIN_EXTERNAL_EVIDENCE_MODEL="sonar",
            LINKEDIN_EXTERNAL_EVIDENCE_TIMEOUT_SECONDS=90.0,
            LINKEDIN_EXTERNAL_EVIDENCE_MAX_OUTPUT_TOKENS=4096,
            LINKEDIN_EXTERNAL_EVIDENCE_MIN_CITATIONS=2,
            LINKEDIN_EXTERNAL_EVIDENCE_PERPLEXITY_PRESET="",
        )

    def _run_with_mock(self, mock_module, sleep_calls: list[float] | None = None):
        from shared.external_evidence import provider as provider_mod

        orig_config = provider_mod.config
        try:
            provider_mod.config = self._enabled_config()
            with patch.dict("sys.modules", {"openai": mock_module}), patch(
                "shared.llm_clients.time.sleep",
                side_effect=(sleep_calls.append if sleep_calls is not None else None),
            ):
                return _provider_call()
        finally:
            provider_mod.config = orig_config

    def test_quota_exhausted_classified_explicitly(self):
        """Quota errors MUST classify as quota_exhausted, not unknown.

        This pins the most important slice 1 invariant: candidate-level
        Perplexity quota exhaustion must NEVER raise ApiBudgetExhaustedError
        (which would pause the LinkedIn run). It must return a typed failure.
        """
        mock_module, mock_client = _mock_openai_module(
            raises=_FakeBudgetError("credit balance is too low; please purchase credits")
        )

        # Wrap in try/except to assert the function never raises, especially
        # never ApiBudgetExhaustedError.
        try:
            result = self._run_with_mock(mock_module)
        except ApiBudgetExhaustedError as exc:  # pragma: no cover — test fails on raise
            pytest.fail(
                "fetch_external_candidate_evidence must NEVER raise "
                f"ApiBudgetExhaustedError; got: {exc!r}"
            )
        except Exception as exc:  # pragma: no cover — test fails on raise
            pytest.fail(
                "fetch_external_candidate_evidence must NEVER raise; "
                f"got: {type(exc).__name__}: {exc!r}"
            )

        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "quota_exhausted", (
            f"expected reason='quota_exhausted', got {result.reason!r}. "
            "Slice 1 must classify Perplexity quota errors explicitly so the "
            "LinkedIn run does not pause."
        )
        assert result.reason != "unknown"
        assert result.provider == "perplexity"
        assert mock_client.responses.create.call_count == 1

    def test_quota_exhausted_via_http_402(self):
        mock_module, mock_client = _mock_openai_module(
            raises=_FakeHttpError("Payment required", status_code=402)
        )
        result = self._run_with_mock(mock_module)
        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "quota_exhausted"
        assert result.http_status == 402
        assert mock_client.responses.create.call_count == 1

    def test_timeout_classified(self):
        mock_module, mock_client = _mock_openai_module(
            raises=TimeoutError("request timed out")
        )
        result = self._run_with_mock(mock_module)
        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "timeout"
        assert mock_client.responses.create.call_count == 3

    def test_http_error_with_status(self):
        mock_module, mock_client = _mock_openai_module(
            raises=_FakeHttpError("Bad gateway", status_code=502)
        )
        result = self._run_with_mock(mock_module)
        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "http_error"
        assert result.http_status == 502
        assert mock_client.responses.create.call_count == 3

    def test_rate_limit_retries_three_attempts_and_honors_retry_after(self):
        mock_module, mock_client = _mock_openai_module(
            raises=_FakeHttpError(
                "Rate limited",
                status_code=429,
                headers={"Retry-After": "7"},
            )
        )
        sleep_calls: list[float] = []

        result = self._run_with_mock(mock_module, sleep_calls)

        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "http_error"
        assert result.http_status == 429
        assert mock_client.responses.create.call_count == 3
        assert sleep_calls == [7.0, 7.0]

    def test_unknown_unexpected_exception(self):
        mock_module, mock_client = _mock_openai_module(raises=RuntimeError("boom"))
        result = self._run_with_mock(mock_module)
        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "unknown"
        assert "boom" in result.detail
        assert mock_client.responses.create.call_count == 1

    def test_provider_failure_records_error_receipt(self, monkeypatch):
        from shared.external_evidence import provider as provider_mod

        usage_calls = []
        monkeypatch.setattr(
            provider_mod,
            "record_llm_usage",
            lambda **kwargs: usage_calls.append(kwargs),
        )
        mock_module, _ = _mock_openai_module(raises=RuntimeError("boom"))

        result = self._run_with_mock(mock_module)

        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "unknown"
        assert len(usage_calls) == 1
        call = usage_calls[0]
        assert call["provider"] == "perplexity"
        assert call["actual_status"] == "error"
        assert call["usage"]["input_tokens"] == 0
        assert call["request"]["error_type"] == "RuntimeError"
        assert call["request"]["error_message"] == "boom"
        assert call["usage_context"]["stage"] == "linkedin_external_evidence"
        assert call["usage_context"]["trigger_reason"] == "academic_context"
