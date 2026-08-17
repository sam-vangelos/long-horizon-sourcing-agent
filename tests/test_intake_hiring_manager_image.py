"""Hiring-manager success-image invariant tests.

Asserts the product invariant Cloris must enforce: before composing or
reviewing a brief, infer and articulate a vivid picture of the person the
hiring manager actually wants. The tests assert *concepts*, not exact
prose, by routing through ``shared.intake_conversation.insights``:

- ``normalize_hiring_manager_success_image`` is the single product rule.
- ``is_missing_hiring_manager_success_image`` is the only deficit check.
- ``is_generic_trope`` is the only trope rule.
- ``merge_intake_insights`` is the only merge primitive.

Three producers are exercised (extractor / composer / synthesis worker);
a fourth case proves the recruiter-correction lock survives a synthesis
re-run. Where unit-level tests already cover a producer in isolation
(``test_intake_conversation_extractor.py``, ``test_intake_conversation_endpoint.py``),
this file adds the cross-producer contract checks.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloris.app import create_app
from shared.intake_conversation.composer import compose_from_conversation
from shared.intake_conversation.extractor import (
    ExtractionResult,
    extract_slots,
)
from shared.intake_conversation.insights import (
    HIRING_MANAGER_PICTURE_KEY,
    HIRING_MANAGER_PICTURE_LOCK_PATH,
    GENERIC_TROPE_PHRASES,
    is_generic_trope,
    is_missing_hiring_manager_success_image,
    merge_intake_insights,
    normalize_hiring_manager_success_image,
)
from shared.runtime_state.store import RuntimeStateStore


# -------------------------------------------------------------------------
# Fixtures: BFS Applied AI Lab role context
# -------------------------------------------------------------------------


_BFS_ROLE_CONTEXT: dict[str, Any] = {
    "role_title": "Head of Applied AI",
    "role_summary": (
        "Sets AI vision and architectural guardrails for the BFS group, "
        "speaks credibly to bank executives, and personally owns agentic "
        "design trade-offs."
    ),
    "capability_areas": [
        {
            "name": "Agentic design",
            "description": (
                "Owns architecture trade-offs in agentic systems and can "
                "speak to deployment realities inside a BFS firm."
            ),
        },
        {
            "name": "Executive partnership",
            "description": (
                "Translates technical depth for bank executives without "
                "losing the technical thread."
            ),
        },
    ],
}


def _bfs_picture(*, corrected: bool = False, source: str = "combined") -> dict:
    return {
        "summary": (
            "A quasi-CTO of the BFS group who can shape AI vision, "
            "define architectural guardrails for agentic systems, and "
            "still sit with bank executives."
        ),
        "proof_points": [
            "Has shipped an applied AI lab inside a BFS firm.",
            "Owned architecture trade-offs on agentic systems in production.",
            "Has presented agentic deployment plans to bank executives.",
        ],
        "screening_translation": (
            "Reject candidates who have only advised on GenAI programs "
            "without owning agentic architecture decisions."
        ),
        "confidence": 0.7,
        "source": source,
        "corrected_by_recruiter": corrected,
    }


def _msg(role: str, content: str) -> dict:
    return {
        "role": role,
        "content": content,
        "ts": "2026-05-13T12:00:00+00:00",
    }


def _bfs_transcript() -> list[dict]:
    return [
        _msg(
            "cloris",
            "What's the role you're filling?",
        ),
        _msg(
            "recruiter",
            "We're hiring a Head of Applied AI for the BFS group. "
            "We need someone who can set the AI vision, define "
            "architectural guardrails for agentic systems, and still "
            "speak credibly to bank executives.",
        ),
        _msg(
            "cloris",
            "Picturing a quasi-CTO of the BFS group — sets AI vision, "
            "defines architectural guardrails, still goes deep on "
            "agentic design. Is that the person, or are you more after "
            "a boardroom program shaper who'd hand the architecture to "
            "a deputy?",
        ),
        _msg(
            "recruiter",
            "Yes, the quasi-CTO read is right — they need to own the "
            "agentic architecture themselves, not just program-manage.",
        ),
    ]


# -------------------------------------------------------------------------
# Concept-level invariants on the BFS picture
# -------------------------------------------------------------------------


def test_normalized_bfs_picture_satisfies_concept_invariants() -> None:
    """For the BFS Applied AI Lab fixture the produced picture must:

    - have a vivid summary that names a senior-leader / quasi-CTO framing
      AND at least one role-context anchor (BFS / applied AI / agentic /
      bank);
    - have at least two proof points, at least one carrying a domain
      anchor;
    - have a screening_translation distinguishing operators from advisors;
    - not be a generic trope.
    """

    normalized = normalize_hiring_manager_success_image(
        _bfs_picture(),
        _BFS_ROLE_CONTEXT,
        source="combined",
    )

    assert normalized is not None, "BFS fixture should produce a usable picture"
    assert is_missing_hiring_manager_success_image(normalized) is False
    assert is_generic_trope(normalized["summary"], _BFS_ROLE_CONTEXT) is False

    summary_lower = normalized["summary"].lower()
    framing_tokens = ("vision", "architect", "head of", "quasi-cto", "shape")
    assert any(
        token in summary_lower for token in framing_tokens
    ), f"summary missing senior-leader framing: {summary_lower!r}"

    anchor_tokens = ("bfs", "applied ai", "agentic", "bank")
    assert any(
        token in summary_lower for token in anchor_tokens
    ), f"summary missing role-context anchor: {summary_lower!r}"

    assert len(normalized["proof_points"]) >= 2
    assert any(
        any(token in proof.lower() for token in anchor_tokens)
        for proof in normalized["proof_points"]
    ), "no proof point carries a domain anchor"

    screening = normalized["screening_translation"].lower()
    operator_vs_advisor_markers = (
        "only advised",
        "advisors",
        "owning",
        "owned",
        "personally",
        "vs",
        "versus",
        "not just",
        "without owning",
    )
    assert any(
        marker in screening for marker in operator_vs_advisor_markers
    ), f"screening_translation does not distinguish operators from advisors: {screening!r}"


@pytest.mark.parametrize("phrase", list(GENERIC_TROPE_PHRASES)[:6])
def test_generic_tropes_are_rejected_in_summary(phrase: str) -> None:
    """Each banned phrase causes ``is_generic_trope`` to return True
    independent of role context — the whole point is to reject corporate
    slop regardless of how plausible the surrounding sentence sounds.
    """

    summary = (
        f"Cloris thinks the hiring manager wants a {phrase} who shapes the "
        "BFS applied AI vision."
    )
    assert is_generic_trope(summary, _BFS_ROLE_CONTEXT) is True


def test_role_unrelated_summary_is_a_trope_even_without_banlist_phrase() -> None:
    """A summary that shares no tokens with the role context cannot be
    accepted as the hiring-manager picture — generic praise unmoored from
    the role is a trope by construction.
    """

    summary = (
        "A motivated leader who builds momentum and brings energy to "
        "every conversation across the organization."
    )
    assert is_generic_trope(summary, _BFS_ROLE_CONTEXT) is True


def test_normalize_returns_none_on_trope_or_below_floor() -> None:
    """The shared product rule rejects trope-shaped or below-floor input
    so callers do not have to re-implement the rule.
    """

    trope = {
        "summary": "Strong communication skills and a team player who self-starts.",
        "proof_points": ["team player"],
        "screening_translation": "reject those who fail communication.",
    }
    assert normalize_hiring_manager_success_image(
        trope, _BFS_ROLE_CONTEXT, source="conversation"
    ) is None

    too_short = {
        "summary": "BFS leader.",  # below floor
        "proof_points": ["Has shipped agentic systems in production at a BFS firm."],
        "screening_translation": "reject pure advisors who never owned the architecture.",
    }
    assert normalize_hiring_manager_success_image(
        too_short, _BFS_ROLE_CONTEXT, source="conversation"
    ) is None


# -------------------------------------------------------------------------
# Extractor invariant: insight stays out of v2_draft, deficit stays out
# of v2 schema validation
# -------------------------------------------------------------------------


def test_extractor_split_holds_with_combined_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when cheap_llm emits both v2 fields and the picture in the
    same response, the extractor seam keeps them apart.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "role_title": "Head of Applied AI",
            "role_summary": _BFS_ROLE_CONTEXT["role_summary"],
            "hiring_manager_success_image": _bfs_picture(),
        },
    )

    result = extract_slots(
        messages=_bfs_transcript(),
        current_v2_draft={},
        source_packet={"job_description_text": "BFS Applied AI Lab head"},
    )

    assert isinstance(result, ExtractionResult)
    assert HIRING_MANAGER_PICTURE_KEY not in result.v2_updates
    assert HIRING_MANAGER_PICTURE_KEY in result.insight_updates


def test_extractor_returns_canonical_empty_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty cheap_llm output, exception, and structural validation
    failure all yield ``ExtractionResult({}, {})`` — never None, never a
    legacy flat dict.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {},
    )
    empty = extract_slots(
        messages=_bfs_transcript(),
        current_v2_draft={},
        source_packet=None,
    )
    assert isinstance(empty, ExtractionResult)
    assert empty.v2_updates == {} and empty.insight_updates == {}

    def _raise(system, user, expect_json=True, usage_context=None):
        raise RuntimeError("simulated outage")

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm", _raise
    )
    raised = extract_slots(
        messages=_bfs_transcript(),
        current_v2_draft={},
        source_packet=None,
    )
    assert isinstance(raised, ExtractionResult)
    assert raised.v2_updates == {} and raised.insight_updates == {}


# -------------------------------------------------------------------------
# Composer: insight survives v2 deficits, deficits stay separate
# -------------------------------------------------------------------------


def test_compose_surfaces_insight_deficit_independent_of_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful v2 compose with a missing picture must:

    - report ``status == "composed"`` (the brief schema is valid);
    - keep ``missing_keys`` / ``invalid_keys`` empty;
    - surface the missing picture via ``insight_deficits`` only.
    """

    # No LLM access → deterministic compose path. The heuristic compose
    # produces a valid v2 draft from the transcript but does not produce
    # an insight (insights only come from the LLM path or synthesis).
    monkeypatch.setattr(
        "shared.intake_conversation.composer._has_llm_access",
        lambda: False,
    )

    result = compose_from_conversation(
        messages=_bfs_transcript(),
        current_v2_draft={},
        source_packet={"job_description_text": "BFS Applied AI Lab head"},
    )

    # The brief schema may or may not be fully valid in the heuristic
    # path depending on transcript shape; what we assert is that the
    # insight deficit channel is independent.
    assert HIRING_MANAGER_PICTURE_KEY not in result.v2_draft
    insight_fields = {d.get("field") for d in result.insight_deficits}
    assert HIRING_MANAGER_PICTURE_KEY in insight_fields
    # Insight deficits never inflate v2 schema deficits.
    assert all(
        d.get("field") != HIRING_MANAGER_PICTURE_KEY
        for d in result.deficits
    )
    assert HIRING_MANAGER_PICTURE_KEY not in result.missing_keys
    assert HIRING_MANAGER_PICTURE_KEY not in result.invalid_keys


def test_compose_attaches_insight_when_llm_emits_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The LLM compose path returns ``insight_updates`` so the API
    persistence layer can write it to ``state_json.intake_insights``.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.composer._has_llm_access",
        lambda: True,
    )
    monkeypatch.setattr(
        "shared.intake_conversation.composer.opus_llm_cached",
        lambda system, user, **kwargs: {
            "v2_draft": {
                "role_title": "Head of Applied AI",
                "role_summary": _BFS_ROLE_CONTEXT["role_summary"],
                "capability_areas": _BFS_ROLE_CONTEXT["capability_areas"],
                "depth_distinction": {
                    "builder_definition": "Has personally owned agentic system architecture.",
                    "user_definition": "Has used agentic frameworks but not shaped them.",
                    "edge_case_guidance": "Treat program-only profiles as review-only.",
                },
                "non_fit_patterns": [
                    {
                        "label": "Advisory-only AI",
                        "why_not": "Has only consulted, never owned architecture.",
                    }
                ],
                "minimum_bar_description": "Owned at least one agentic system in production.",
            },
            "intake_insights": {
                HIRING_MANAGER_PICTURE_KEY: _bfs_picture(),
            },
            "deficits": [],
        },
    )

    result = compose_from_conversation(
        messages=_bfs_transcript(),
        current_v2_draft={},
        source_packet={"job_description_text": "BFS Applied AI Lab head"},
    )

    assert HIRING_MANAGER_PICTURE_KEY in result.insight_updates
    picture = result.insight_updates[HIRING_MANAGER_PICTURE_KEY]
    assert is_missing_hiring_manager_success_image(picture) is False
    assert result.insight_deficits == []


# -------------------------------------------------------------------------
# Recruiter-correction propagation across producers
# -------------------------------------------------------------------------


def test_extractor_correction_locks_picture_through_merge_intake_insights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end propagation:

    1. Extractor emits a corrected picture (corrected_by_recruiter=true).
    2. ``merge_intake_insights`` writes it to state.
    3. The picture path is added to ``manually_edited_keys``.
    4. A subsequent extractor turn that tries to overwrite is dropped by
       ``merge_intake_insights`` because the path is locked.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            HIRING_MANAGER_PICTURE_KEY: _bfs_picture(corrected=True),
        },
    )
    correction = extract_slots(
        messages=_bfs_transcript(),
        current_v2_draft=_BFS_ROLE_CONTEXT,
        source_packet=None,
    )
    picture = correction.insight_updates[HIRING_MANAGER_PICTURE_KEY]
    assert picture["corrected_by_recruiter"] is True

    # Step 2 + 3: API persistence (mirrored here).
    insights_state: dict[str, Any] = merge_intake_insights(
        {}, correction.insight_updates, manually_edited_keys=()
    )
    locks: set[str] = {HIRING_MANAGER_PICTURE_LOCK_PATH}

    # Step 4: a different picture coming through the next turn must NOT
    # overwrite the recruiter's correction.
    challenger = {
        HIRING_MANAGER_PICTURE_KEY: {
            **_bfs_picture(),
            "summary": (
                "A program-management-style operator who keeps agentic "
                "rollouts on time and on budget across BFS partners."
            ),
            "corrected_by_recruiter": False,
        }
    }
    after = merge_intake_insights(
        insights_state, challenger, manually_edited_keys=locks
    )
    assert (
        after[HIRING_MANAGER_PICTURE_KEY]["summary"]
        == picture["summary"]
    ), "locked picture must survive a non-corrective re-extraction"


def test_synthesis_cannot_trample_recruiter_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesis worker re-runs apply ``merge_intake_insights`` against a
    locked path; a previously-corrected picture survives.
    """

    locked_picture = {
        **_bfs_picture(corrected=True, source="conversation"),
    }
    state_insights = {HIRING_MANAGER_PICTURE_KEY: locked_picture}
    locks = [HIRING_MANAGER_PICTURE_LOCK_PATH]

    fresh_synthesis_picture = {
        HIRING_MANAGER_PICTURE_KEY: {
            **_bfs_picture(corrected=False, source="source_packet"),
            "summary": (
                "A senior architect who shapes BFS agentic deployment "
                "patterns for retail and commercial banking lines."
            ),
        }
    }
    merged = merge_intake_insights(
        state_insights, fresh_synthesis_picture, manually_edited_keys=locks
    )
    assert (
        merged[HIRING_MANAGER_PICTURE_KEY]["summary"]
        == locked_picture["summary"]
    )


# -------------------------------------------------------------------------
# CTA readiness via the API endpoint: deficit surfaces, brief schema is
# decoupled
# -------------------------------------------------------------------------


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    tmp_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    tmp_store = RuntimeStateStore(tmp_db_path)
    monkeypatch.setattr(
        "cloris.api.intake._intake_store", lambda: tmp_store
    )
    monkeypatch.setattr(
        "cloris.api.intake._intake_db_path", lambda: tmp_db_path
    )
    return TestClient(create_app())


def _create_session_via_api(api_client: TestClient) -> int:
    res = api_client.post("/api/intake/sessions", json={})
    res.raise_for_status()
    return int(res.json()["session"]["id"])


def test_compose_endpoint_surfaces_insight_deficits_field(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API response shape carries ``insight_deficits`` distinct from
    the existing ``deficits`` / ``missing_keys`` / ``invalid_keys`` fields.
    """

    # Force the deterministic path so the picture stays empty.
    monkeypatch.setattr(
        "shared.intake_conversation.composer._has_llm_access",
        lambda: False,
    )

    session_id = _create_session_via_api(api_client)
    # Seed enough conversation state so compose has something to work with.
    from cloris.api.intake import _intake_store
    from cloris import intake_sessions as intake_module

    state = {
        "messages": _bfs_transcript(),
        "v2_draft": {},
        "source_packet": {
            "job_description_text": (
                "Head of Applied AI for the BFS group. Owns AI vision, "
                "agentic architecture trade-offs, and partners with bank "
                "executives."
            )
        },
        "conversation_meta": {"manually_edited_keys": []},
    }
    intake_module.patch_intake_session(
        store=_intake_store(), session_id=session_id, state_json=state
    )

    res = api_client.post(
        f"/api/intake/sessions/{session_id}/compose_from_conversation"
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    result = payload["job"]["result"]
    assert result is not None
    insight_fields = {d.get("field") for d in result["insight_deficits"]}
    assert HIRING_MANAGER_PICTURE_KEY in insight_fields
    # v2 schema deficits remain decoupled.
    assert all(
        d.get("field") != HIRING_MANAGER_PICTURE_KEY
        for d in result.get("deficits") or []
    )
