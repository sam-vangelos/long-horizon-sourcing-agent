"""Opt-in live Perplexity integration band for the external-evidence provider.

This file is the live counterpart to ``tests/test_external_evidence.py``. The
unit-test file is fully mocked and offline. This file makes a small, bounded
number of real Perplexity API calls so we can verify that the provider's
typed-response parsing, failure classification, and the no-``ApiBudgetExhaustedError``
invariant hold against the actual API surface — not just our mocks of it.

Opt-in gate
-----------

Tests in this file are skipped unless BOTH:

- ``PERPLEXITY_API_KEY`` is set (a real key), AND
- ``PERPLEXITY_LIVE_INTEGRATION=1`` (explicit opt-in flag).

The gate is applied at the **class** level (one decorator, shared by all tests)
because every test in the file shares the same opt-in semantics.

Run as::

    PERPLEXITY_LIVE_INTEGRATION=1 pytest tests/test_external_evidence_integration.py -q

Without the opt-in env vars set, ``pytest tests/`` collects this file and
skips every test in it (no errors, no failures).

What we assert vs. what we don't
---------------------------------

Live tests are guaranteed-flaky if they assert on exact Perplexity content. We
assert ONLY on:

- Status classes (``ExternalCandidateEvidence`` vs ``ExternalEvidenceFailure``).
- Schema shape (types, list-of-refs, URL parseability, round-trip equality).
- The universal invariant: ``ApiBudgetExhaustedError`` MUST NOT propagate out
  of the candidate-level provider regardless of upstream API status. This is
  the most important property of the live band — Perplexity quota errors at
  the candidate level must NEVER pause LinkedIn runs.

When a specific failure mode (``quota_exhausted``, ``weak_citations``) cannot
be deterministically forced from outside, the corresponding test
``pytest.skip(...)`` with a clear reason rather than ``pytest.fail(...)``.

Live API call budget
--------------------

- 1 happy-path call, issued by a module-scoped fixture and shared by tests 1,
  2, and 4.
- 1 weak-citations probe call, issued by test 3.

**Live API call count: ≤2 per pytest invocation. Hard cap: ≤3.**

Identity choices (rationale)
----------------------------

- Happy path: Geoffrey Hinton — Turing-Award laureate, decades of stable
  publicly-cited biography (U Toronto, Google Brain, etc.), low controversy,
  not a current employee of a sensitive company. Built as a sparse profile
  with PhD education so the trigger gate's ``academic_context`` rule fires
  deterministically.
- Weak-citations probe: ``John Smith`` at ``Acme Corp`` with no PhD, no field
  of study, no school. The deliberately ambiguous identity is *expected but
  not guaranteed* to produce few citations. If the provider still returns
  usable evidence (e.g., a real John Smith happens to dominate the public
  web for that headline), the test ``skip``s rather than ``fail``s — honest
  signal over fake-quota theater.

Strict no-prod-import rule
--------------------------

This file imports only from ``shared.external_evidence``, ``shared.schemas``,
and ``shared.failures`` — the same surface the offline unit tests use. It
deliberately does not import from ``linkedin/``, ``github/``, or
``market_intelligence/``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import pytest

import shared.config as config
from shared.external_evidence import (
    fetch_external_candidate_evidence,
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
    TriggerDecision,
)


# ---------------------------------------------------------------------------
# Opt-in gate
# ---------------------------------------------------------------------------

LIVE_INTEGRATION_REQUIRED = pytest.mark.skipif(
    not (
        os.getenv("PERPLEXITY_API_KEY")
        and os.getenv("PERPLEXITY_LIVE_INTEGRATION") == "1"
    ),
    reason=(
        "Live Perplexity integration band: requires PERPLEXITY_API_KEY and "
        "PERPLEXITY_LIVE_INTEGRATION=1. Run as: "
        "PERPLEXITY_LIVE_INTEGRATION=1 pytest tests/test_external_evidence_integration.py -q"
    ),
)


# ---------------------------------------------------------------------------
# Brief + summary builders (kept local; do not depend on the offline test file)
# ---------------------------------------------------------------------------

def _make_brief():
    """Minimal valid Brief — same shape as the offline-unit-test helper.

    Imports of ``Brief`` and friends are local to keep the module top-level
    import list as small (and as obviously prod-safe) as possible.
    """
    from shared.brief_schema import (
        BiasControls,
        Brief,
        CapabilityArea,
        DepthDistinction,
        EmployerSignalRule,
        FacialCalibration,
        NonFitPattern,
    )

    return Brief(
        role_title="Research Engineer",
        role_level="IC5",
        role_summary="Build deep learning systems.",
        geography="United States",
        linkedin_project="proj-live-integration",
        capability_areas=[
            CapabilityArea(
                name="Deep Learning",
                description="Model architecture and training.",
                builder_signals=["builds models"],
                user_signals=[],
            )
        ],
        depth_distinction=DepthDistinction(
            builder_definition="Builds end-to-end.",
            user_definition="Calls hosted APIs.",
            edge_case_guidance="Borderline cases default to user.",
        ),
        non_fit_patterns=[
            NonFitPattern(
                label="Adjacent",
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
        minimum_bar_description="3y hands-on.",
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


def _hinton_summary() -> CandidateProfileSummary:
    """Stable, well-cited public scientist profile for the happy-path probe.

    Built deliberately sparse and PhD-tagged so the trigger gate's
    ``academic_context`` rule fires.
    """
    return CandidateProfileSummary(
        name="Geoffrey Hinton",
        profile_url="https://www.linkedin.com/in/geoffrey-hinton-public-figure",
        headline="Professor Emeritus, University of Toronto",
        experiences=[
            Experience(
                title="Professor Emeritus",
                company="University of Toronto",
                summary_bullets=["Neural networks research."],
            ),
        ],
        education=[
            Education(
                degree="PhD",
                school="University of Edinburgh",
                field="Artificial Intelligence",
            ),
        ],
        skills_snippet=[],
    )


def _ambiguous_summary() -> CandidateProfileSummary:
    """Deliberately ambiguous identity for the weak-citations probe.

    Common name + generic employer + no PhD + no school. Expected (not
    guaranteed) to produce few citations.
    """
    return CandidateProfileSummary(
        name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-ambiguous",
        headline="Engineer",
        experiences=[
            Experience(
                title="Engineer",
                company="Acme Corp",
                summary_bullets=["Built things."],
            ),
        ],
        education=[],
        skills_snippet=[],
    )


# ---------------------------------------------------------------------------
# Provider invocation helper
# ---------------------------------------------------------------------------

def _force_enabled_config_namespace() -> SimpleNamespace:
    """Mirror the real ``shared.config`` values but force the feature flag on.

    The provider checks ``config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED`` at call
    time. That flag defaults to ``False`` in production. The opt-in spec for
    this live band is ``PERPLEXITY_API_KEY + PERPLEXITY_LIVE_INTEGRATION=1``;
    we deliberately do NOT require the user to also set
    ``LINKEDIN_EXTERNAL_EVIDENCE_ENABLED=true`` in their shell. To bridge the
    gap without editing production code, we monkey-patch the provider's view
    of ``config`` for the duration of the live call only.

    The real ``PERPLEXITY_API_KEY`` is pulled from ``config`` (which loaded
    it from ``.env`` / environment via ``python-dotenv``).
    """
    return SimpleNamespace(
        PERPLEXITY_API_KEY=getattr(config, "PERPLEXITY_API_KEY", "")
        or os.getenv("PERPLEXITY_API_KEY", ""),
        LINKEDIN_EXTERNAL_EVIDENCE_ENABLED=True,
        LINKEDIN_EXTERNAL_EVIDENCE_MODEL=getattr(
            config, "LINKEDIN_EXTERNAL_EVIDENCE_MODEL", ""
        ),
        LINKEDIN_EXTERNAL_EVIDENCE_TIMEOUT_SECONDS=getattr(
            config, "LINKEDIN_EXTERNAL_EVIDENCE_TIMEOUT_SECONDS", 90.0
        ),
        LINKEDIN_EXTERNAL_EVIDENCE_MAX_OUTPUT_TOKENS=getattr(
            config, "LINKEDIN_EXTERNAL_EVIDENCE_MAX_OUTPUT_TOKENS", 4096
        ),
        LINKEDIN_EXTERNAL_EVIDENCE_MIN_CITATIONS=getattr(
            config, "LINKEDIN_EXTERNAL_EVIDENCE_MIN_CITATIONS", 2
        ),
        LINKEDIN_EXTERNAL_EVIDENCE_MIN_IDENTITY_CONFIDENCE=getattr(
            config, "LINKEDIN_EXTERNAL_EVIDENCE_MIN_IDENTITY_CONFIDENCE", 0.5
        ),
        LINKEDIN_EXTERNAL_EVIDENCE_PERPLEXITY_PRESET=getattr(
            config, "LINKEDIN_EXTERNAL_EVIDENCE_PERPLEXITY_PRESET", ""
        ),
    )


def _call_provider_live(
    *,
    summary: CandidateProfileSummary,
    trigger: TriggerDecision,
) -> ExternalCandidateEvidence | ExternalEvidenceFailure:
    """Issue a single live provider call with the feature flag forced on.

    Uses the same ``provider_mod.config`` monkeypatch pattern as the offline
    unit tests so we don't have to mutate global env state. ``ApiBudgetExhaustedError``
    is intentionally NOT caught here — if it ever propagates, it must surface
    to the test layer so we can ``pytest.fail`` on it.
    """
    from shared.external_evidence import provider as provider_mod

    fake_config = _force_enabled_config_namespace()
    orig_config = provider_mod.config
    try:
        provider_mod.config = fake_config
        return fetch_external_candidate_evidence(
            summary=summary,
            brief=_make_brief(),
            trigger=trigger,
            identity_hints=None,
        )
    finally:
        provider_mod.config = orig_config


# ---------------------------------------------------------------------------
# Module-scoped cached happy-path call
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cached_happy_path_call() -> dict[str, Any]:
    """Issue ONE live happy-path call and cache it for the whole module.

    Returns a dict with one of three shapes:

    - ``{"kind": "result", "result": <ExternalCandidateEvidence | ExternalEvidenceFailure>, "trigger": <TriggerDecision>}``
    - ``{"kind": "budget_leak", "exception": <ApiBudgetExhaustedError>}``
      — the bug we are explicitly guarding against; consumer tests must
      ``pytest.fail`` on this.
    - ``{"kind": "transient_error", "exception": <Exception>}``
      — anything else (network, etc.); consumer tests should ``pytest.skip``
      rather than mass-fail the band on a transient.

    Module scope (NOT session scope) so the cache cannot bleed across pytest
    invocations.
    """
    summary = _hinton_summary()
    trigger = should_request_external_evidence(summary=summary, brief=_make_brief())
    if not trigger.should_run:
        return {
            "kind": "transient_error",
            "exception": AssertionError(
                "happy-path identity did not trigger gate; pick a different identity"
            ),
        }

    try:
        result = _call_provider_live(summary=summary, trigger=trigger)
    except ApiBudgetExhaustedError as exc:
        return {"kind": "budget_leak", "exception": exc}
    except Exception as exc:  # noqa: BLE001 — broad on purpose; see fixture docstring.
        return {"kind": "transient_error", "exception": exc}

    return {"kind": "result", "result": result, "trigger": trigger}


# ---------------------------------------------------------------------------
# Shared schema-shape helpers
# ---------------------------------------------------------------------------

_KNOWN_FAILURE_REASONS: frozenset[str] = frozenset(
    {
        "disabled_no_api_key",
        "disabled_by_config",
        "quota_exhausted",
        "timeout",
        "http_error",
        "parse_failure",
        "weak_citations",
        "unknown",
        "normalizer_failure",
    }
)


def _assert_evidence_well_formed(
    result: ExternalCandidateEvidence,
    *,
    expected_trigger_reason: str,
) -> None:
    """Schema-shape assertions only — never assertions on exact content."""
    assert isinstance(result.identity_confidence, float)
    assert 0.0 <= result.identity_confidence <= 1.0, (
        f"identity_confidence out of [0, 1]: {result.identity_confidence}"
    )

    assert isinstance(result.external_fact_blocks, list)
    for block in result.external_fact_blocks:
        assert isinstance(block, ExternalFactBlock)
        assert isinstance(block.evidence_refs, list)
        for ref in block.evidence_refs:
            assert isinstance(ref, EvidenceRef)
            assert ref.url, "EvidenceRef.url must be non-empty"
            parsed = urlparse(ref.url)
            assert parsed.scheme in {"http", "https"}, (
                f"EvidenceRef.url has unsupported scheme: {ref.url!r}"
            )

    assert isinstance(result.raw_provider_model, str)
    assert result.raw_provider_model, "raw_provider_model must be non-empty"

    assert result.trigger_reason == expected_trigger_reason, (
        f"trigger_reason should round-trip the input "
        f"({expected_trigger_reason!r}); got {result.trigger_reason!r}"
    )

    round_tripped = ExternalCandidateEvidence.from_dict(result.to_dict())
    assert round_tripped == result, "to_dict / from_dict round-trip must be exact"


# ---------------------------------------------------------------------------
# Live integration test class
# ---------------------------------------------------------------------------

@LIVE_INTEGRATION_REQUIRED
@pytest.mark.live_integration
class TestLivePerplexityIntegration:
    """Live integration band — guarded by the opt-in env vars above."""

    def test_live_happy_path_returns_well_formed_evidence(
        self, cached_happy_path_call: dict[str, Any]
    ) -> None:
        """Single live call against a stable identity-clear scientist.

        Asserts ONLY on schema shape and status class; never on content.
        Most importantly: asserts the no-``ApiBudgetExhaustedError`` invariant.
        """
        kind = cached_happy_path_call["kind"]

        if kind == "budget_leak":
            pytest.fail(
                "Perplexity quota leaked as ApiBudgetExhaustedError out of "
                "fetch_external_candidate_evidence. This is the most important "
                "invariant of the live band and it MUST NOT happen — candidate-"
                "level external evidence must never pause a LinkedIn run. "
                f"exception={cached_happy_path_call['exception']!r}"
            )

        if kind == "transient_error":
            pytest.skip(
                f"shared live happy-path call raised: "
                f"{cached_happy_path_call['exception']!r}"
            )

        assert kind == "result"
        result = cached_happy_path_call["result"]
        trigger: TriggerDecision = cached_happy_path_call["trigger"]

        assert isinstance(result, (ExternalCandidateEvidence, ExternalEvidenceFailure)), (
            f"provider must return a typed result; got {type(result).__name__}"
        )

        if isinstance(result, ExternalEvidenceFailure):
            if result.reason == "disabled_no_api_key":
                pytest.skip(
                    "provider reports disabled_no_api_key despite opt-in env "
                    "vars; check key plumbing in shared/config.py"
                )
            assert result.reason in _KNOWN_FAILURE_REASONS, (
                f"failure reason must be in the documented set; got {result.reason!r}"
            )
            assert result.provider == "perplexity"
            return

        _assert_evidence_well_formed(
            result, expected_trigger_reason=trigger.reason
        )

    def test_live_quota_exhaustion_classifies_correctly_when_observed(
        self, cached_happy_path_call: dict[str, Any]
    ) -> None:
        """Opportunistic: assert classification IF the cached call hit quota.

        We do NOT issue a separate live call here — we read the cached
        happy-path result. If it didn't surface ``quota_exhausted``, we skip
        with an honest signal. We never fake a quota error.
        """
        kind = cached_happy_path_call["kind"]

        if kind == "budget_leak":
            pytest.fail(
                "Perplexity quota leaked as ApiBudgetExhaustedError; the no-"
                "ApiBudgetExhaustedError invariant is broken. "
                f"exception={cached_happy_path_call['exception']!r}"
            )

        if kind == "transient_error":
            pytest.skip(
                f"cannot classify quota path; shared live call raised: "
                f"{cached_happy_path_call['exception']!r}"
            )

        result = cached_happy_path_call["result"]

        if not (
            isinstance(result, ExternalEvidenceFailure)
            and result.reason == "quota_exhausted"
        ):
            pytest.skip(
                "quota_exhausted path not exercised this run; cannot "
                "deterministically force quota exhaustion from outside. "
                f"observed status={type(result).__name__}"
                + (
                    f"(reason={result.reason!r})"
                    if isinstance(result, ExternalEvidenceFailure)
                    else ""
                )
            )

        assert isinstance(result, ExternalEvidenceFailure)
        assert result.reason == "quota_exhausted"
        assert result.provider == "perplexity"
        assert result.detail, "wrapped quota error message must survive into detail"

    def test_live_weak_citations_classifies_correctly_when_observed(self) -> None:
        """Issue ONE additional live call against an ambiguous identity.

        Live call #2 in this file. Total live API budget remains ≤2 per
        pytest invocation.
        """
        summary = _ambiguous_summary()

        # The trigger gate fires ``sparse_profile`` for this shape (one
        # experience, one bullet, no PhD).
        trigger = should_request_external_evidence(summary=summary, brief=_make_brief())
        if not trigger.should_run:
            pytest.skip(
                "ambiguous-identity probe did not trigger the gate; cannot "
                "exercise the provider path"
            )

        try:
            result = _call_provider_live(summary=summary, trigger=trigger)
        except ApiBudgetExhaustedError as exc:
            pytest.fail(
                "Perplexity quota leaked as ApiBudgetExhaustedError on the "
                "weak-citations probe call. The no-ApiBudgetExhaustedError "
                f"invariant is broken. exception={exc!r}"
            )
        except Exception as exc:  # noqa: BLE001 — transient network errors should skip, not fail.
            pytest.skip(f"weak-citations probe raised transiently: {exc!r}")

        if isinstance(result, ExternalCandidateEvidence):
            pytest.skip(
                "weak_citations path not exercised; provider returned usable "
                "evidence on the ambiguous-identity probe"
            )

        assert isinstance(result, ExternalEvidenceFailure), (
            f"provider must return a typed result; got {type(result).__name__}"
        )

        if result.reason != "weak_citations":
            pytest.skip(
                f"unexpected status reason={result.reason!r} on weak-citations "
                "probe; cannot deterministically assert weak_citations"
            )

        assert result.provider == "perplexity"
        assert result.detail, "weak_citations failure must include a detail string"
        # Spec note: detail should mention citation count or threshold; we
        # don't bind to exact phrasing since the normalizer's wording is its
        # own concern. We assert detail is non-empty above; that's enough at
        # the live boundary.

    def test_live_no_apibudgetexhaustederror_invariant(
        self, cached_happy_path_call: dict[str, Any]
    ) -> None:
        """Headline invariant: Perplexity quota errors MUST NEVER pause LinkedIn runs.

        This test is intentionally narrow and named explicitly so a future
        contributor reading the file sees the no-``ApiBudgetExhaustedError``
        property as a first-class concern. It reuses the cached happy-path
        call (no extra live API spend).
        """
        kind = cached_happy_path_call["kind"]

        if kind == "budget_leak":
            pytest.fail(
                "fetch_external_candidate_evidence raised ApiBudgetExhaustedError. "
                "Candidate-level external evidence must NEVER raise this — "
                "doing so pauses the LinkedIn run, which is exactly the failure "
                "mode this band exists to prevent. "
                f"exception={cached_happy_path_call['exception']!r}"
            )

        if kind == "transient_error":
            # A non-budget exception still propagated — but the spec says the
            # provider must never raise. If a non-ApiBudgetExhaustedError
            # propagated out of the provider, that's a separate (also bad)
            # bug. Surface it as a fail rather than a skip so it can't be
            # silently lost.
            exc = cached_happy_path_call["exception"]
            pytest.fail(
                "fetch_external_candidate_evidence raised an exception; the "
                "provider must always return a typed result. "
                f"exception={exc!r}"
            )

        # Cached call returned a typed result without raising. The invariant
        # holds for this run.
        assert kind == "result"
        result = cached_happy_path_call["result"]
        assert isinstance(
            result, (ExternalCandidateEvidence, ExternalEvidenceFailure)
        )


# Without the opt-in env vars, every test above is skipped at class level by
# the ``LIVE_INTEGRATION_REQUIRED`` decorator, and the file collects cleanly
# under default ``pytest tests/``. The offline coverage for this slice lives
# in ``tests/test_external_evidence.py``.
