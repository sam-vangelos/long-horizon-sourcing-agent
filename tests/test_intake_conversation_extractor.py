"""Tests for the conversational intake slot extractor (Phase C3).

The extractor is the bridge from natural-language conversation to the
structured ``v2_draft`` AND the ``state_json.intake_insights`` bag.
Four load-bearing contracts to verify:

1. **Deliberate seam.** ``extract_slots`` returns
   :class:`ExtractionResult` carrying split ``v2_updates`` and
   ``insight_updates`` payloads. Empty / failure / validation failure
   all return ``ExtractionResult({}, {})`` — never ``None``, never a
   tuple, never a flat dict.

2. **Insight isolation.** ``hiring_manager_success_image`` (and any
   future insight key) is partitioned into ``insight_updates`` BEFORE
   the v2 schema validation gate runs. Insights never enter
   ``v2_draft`` even temporarily, and a malformed v2 round does not
   drop a normalized insight.

3. **Manual-edit conflicts.** cheap_llm receives the manually-edited
   dot-paths and the explicit rule. The
   :mod:`shared.intake_conversation.state` merge backstop catches
   misbehaving extractor output for v2 fields; the
   :mod:`shared.intake_conversation.insights.merge_intake_insights`
   backstop catches misbehavior for insight fields. Recruiter
   corrections to the picture surface via
   ``corrected_by_recruiter: true`` on the emitted insight.

4. **Placeholder + trope gating.** Any v2 string matching
   ``PLACEHOLDER_STRINGS`` is dropped pre-merge. Insight payloads run
   through ``normalize_hiring_manager_success_image``, which rejects
   trope-shaped or below-floor output.

cheap_llm is monkeypatched at the
``shared.intake_conversation.extractor.cheap_llm`` binding so tests
don't require an API key.
"""

from __future__ import annotations

import pytest

from shared.intake_conversation.extractor import (
    ExtractionResult,
    extract_slots,
)
from shared.intake_conversation.state import merge_extracted


def _msg(role: str, content: str) -> dict:
    return {
        "role": role,
        "content": content,
        "ts": "2026-05-13T12:00:00+00:00",
    }


# -------------------------------------------------------------------------
# Happy path — clean extraction
# -------------------------------------------------------------------------


def test_extract_slots_returns_clean_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "role_title": "Senior Tax Associate",
            "role_summary": "Owns multi-state sales tax filings end-to-end.",
        },
    )

    result = extract_slots(
        messages=[
            _msg("cloris", "What's the role?"),
            _msg("recruiter", "Senior tax associate at Northwind, owns sales tax filings."),
        ],
        current_v2_draft={},
        source_packet=None,
    )

    assert isinstance(result, ExtractionResult)
    assert result.v2_updates["role_title"] == "Senior Tax Associate"
    assert "sales tax filings" in result.v2_updates["role_summary"]
    assert result.insight_updates == {}


def test_extract_slots_accepts_source_recommendations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "target_modules": ["linkedin", "github", "researcher"],
            "source_strategy": [
                {
                    "source": "linkedin",
                    "role": "primary",
                    "rationale": "Broad people coverage.",
                },
                {
                    "source": "github",
                    "role": "corroborating",
                    "rationale": "Public work can confirm depth.",
                },
            ],
        },
    )

    result = extract_slots(
        messages=[
            _msg("cloris", "I'd start with LinkedIn and corroborate on GitHub."),
            _msg("recruiter", "Yes, include Researcher too."),
        ],
        current_v2_draft={},
        source_packet=None,
    )

    assert result.v2_updates["target_modules"] == ["linkedin", "github", "researcher"]
    assert result.v2_updates["source_strategy"][1]["role"] == "corroborating"
    assert result.insight_updates == {}


def test_extract_slots_returns_empty_when_no_messages() -> None:
    """No conversation = nothing to extract. cheap_llm not even called.

    Empty case is the canonical ``ExtractionResult({}, {})``.
    """

    result = extract_slots(
        messages=[],
        current_v2_draft={},
        source_packet=None,
    )

    assert isinstance(result, ExtractionResult)
    assert result.v2_updates == {}
    assert result.insight_updates == {}


# -------------------------------------------------------------------------
# Failure modes — extractor non-fatal
# -------------------------------------------------------------------------


def test_extract_slots_returns_empty_on_cheap_llm_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(system, user, expect_json=True, usage_context=None):
        raise RuntimeError("simulated cheap_llm outage")

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm", _raise
    )

    result = extract_slots(
        messages=[_msg("recruiter", "tax associate")],
        current_v2_draft={},
        source_packet=None,
    )

    assert isinstance(result, ExtractionResult)
    assert result.v2_updates == {}
    assert result.insight_updates == {}


def test_extract_slots_returns_empty_on_non_dict_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """cheap_llm returning a list (not a dict) is a contract violation —
    drop the round rather than coerce. Empty payload, both sides.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: ["not", "a", "dict"],
    )

    result = extract_slots(
        messages=[_msg("recruiter", "tax associate")],
        current_v2_draft={},
        source_packet=None,
    )

    assert isinstance(result, ExtractionResult)
    assert result.v2_updates == {}
    assert result.insight_updates == {}


# -------------------------------------------------------------------------
# Placeholder gating
# -------------------------------------------------------------------------


def test_extract_slots_drops_placeholder_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLACEHOLDER_STRINGS values must be scrubbed before write."""

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "role_title": "Senior Tax Associate",
            "role_summary": "Derived from the source packet.",  # placeholder
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "tax associate")],
        current_v2_draft={},
        source_packet=None,
    )

    assert result.v2_updates["role_title"] == "Senior Tax Associate"
    assert "role_summary" not in result.v2_updates


def test_extract_slots_drops_jd_dumped_into_role_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """role_title must be a label, not a sentence. _looks_like_placeholder
    drops long sentence-shaped values for that field.
    """

    long_jd_paste = (
        "We are a fast-growing media company seeking a Senior Tax Associate "
        "to own our multi-state sales tax compliance function, partner with "
        "our finance leadership, and provide hands-on tax planning support "
        "across all production entities. The ideal candidate brings 5+ years."
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "role_title": long_jd_paste,
            "role_summary": "Owns sales tax compliance.",
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "see attached JD")],
        current_v2_draft={},
        source_packet=None,
    )

    assert "role_title" not in result.v2_updates
    assert result.v2_updates["role_summary"] == "Owns sales tax compliance."


def test_extract_slots_drops_placeholders_in_nested_capability_areas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "capability_areas": [
                {"name": "Sales tax", "description": "Owns multi-state filings."},
                {"name": "Reporting", "description": "Core role scope"},  # placeholder
            ]
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "tax stuff")],
        current_v2_draft={},
        source_packet=None,
    )

    caps = result.v2_updates["capability_areas"]
    # Reporting capability gets dropped down to {name: "Reporting"} — the
    # placeholder description scrubs out, and an item with only a name is
    # not useful, so the scrubber drops the empty sub-object too. Sales tax
    # survives intact.
    assert len(caps) == 2
    assert caps[0]["name"] == "Sales tax"
    assert "description" in caps[0]
    assert caps[1]["name"] == "Reporting"
    # The description got scrubbed out — only `name` remains. The C5
    # endpoint persists it; sufficiency check (C4) won't accept this
    # capability area as "real" until a description lands.
    assert "description" not in caps[1]


# -------------------------------------------------------------------------
# Manual-edit handling — prompt-passing
# -------------------------------------------------------------------------


def test_extract_slots_passes_manually_edited_keys_to_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The system prompt must enumerate the locked dot-paths so cheap_llm
    can apply the manual-edit rule.
    """

    captured = {}

    def _capturing(system, user, expect_json=True, usage_context=None):
        captured["system"] = system
        captured["user"] = user
        return {}

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm", _capturing
    )

    extract_slots(
        messages=[_msg("recruiter", "tax associate")],
        current_v2_draft={"role_title": "Manual Title"},
        source_packet=None,
        manually_edited_keys={"role_title", "depth_distinction.builder_definition"},
    )

    system = captured["system"]
    assert "MANUALLY EDITED SLOTS" in system
    assert "role_title" in system
    assert "depth_distinction.builder_definition" in system
    assert "MOST RECENT" in system  # the rule


def test_extract_slots_no_locked_keys_uses_no_lock_block(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _capturing(system, user, expect_json=True, usage_context=None):
        captured["system"] = system
        return {}

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm", _capturing
    )

    extract_slots(
        messages=[_msg("recruiter", "tax associate")],
        current_v2_draft={},
        source_packet=None,
    )

    assert "MANUALLY EDITED SLOTS: (none yet" in captured["system"]


# -------------------------------------------------------------------------
# Manual-edit handling — backstop via merge_extracted
# -------------------------------------------------------------------------


def test_manual_edit_preserved_when_extractor_misbehaves(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end safety: even if cheap_llm IGNORES the manually-edited
    rule and returns an update for a locked slot, the
    :func:`merge_extracted` backstop drops it before persistence.

    This test exercises the integration of (extractor returns updates) +
    (merge_extracted in C5 endpoint applies the locks). The extractor
    itself doesn't drop locked slots from its output — that's deliberately
    the merger's job, so validation can run on the LLM's actual intent.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "role_title": "Extractor Override (should be dropped)",
            "role_summary": "OK to land",
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "still the same role")],
        current_v2_draft={"role_title": "Manual Title"},
        source_packet=None,
        manually_edited_keys={"role_title"},
    )

    # Extractor returned the override (showing the prompt rule didn't fire
    # in this fake), but merge_extracted backstops it.
    merged = merge_extracted(
        {"role_title": "Manual Title"},
        result.v2_updates,
        manually_edited_keys={"role_title"},
    )

    assert merged["role_title"] == "Manual Title"
    assert merged["role_summary"] == "OK to land"


# -------------------------------------------------------------------------
# Validation gate
# -------------------------------------------------------------------------


def test_extract_slots_accepts_incomplete_drafts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing required v2 keys (capability_areas, depth_distinction) is
    the normal mid-intake state. Updates land regardless.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "role_title": "Senior Tax Associate",
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "tax associate at Northwind")],
        current_v2_draft={},
        source_packet=None,
    )

    assert result.v2_updates["role_title"] == "Senior Tax Associate"


def test_extract_slots_drops_only_the_invalid_key_on_structural_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P9.4: one structurally-invalid top-level key (capability_areas item
    missing required name/description) must not discard the whole round.
    The valid ``depth_distinction`` update still lands; only
    ``capability_areas`` is dropped and named in ``dropped_keys``.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "capability_areas": [
                {"foo": "bar"},  # missing required name + description
            ],
            "depth_distinction": {
                "builder_definition": "x",
                "user_definition": "y",
                "edge_case_guidance": "z",
            },
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "noise")],
        current_v2_draft={},
        source_packet=None,
    )

    assert isinstance(result, ExtractionResult)
    assert "capability_areas" not in result.v2_updates
    assert result.v2_updates["depth_distinction"] == {
        "builder_definition": "x",
        "user_definition": "y",
        "edge_case_guidance": "z",
    }
    assert result.dropped_keys == ("capability_areas",)
    assert result.insight_updates == {}


def test_extract_slots_extraction_partial_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The dropped-key marker must be visible in the log, not silent."""

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "capability_areas": [{"foo": "bar"}],
            "role_title": "Senior Tax Associate",
        },
    )

    # depth_distinction already present on the draft from an earlier turn
    # so validate_v2_brief reaches the shape-check for capability_areas
    # instead of short-circuiting on a missing-required-key.
    existing_draft = {
        "depth_distinction": {
            "builder_definition": "x",
            "user_definition": "y",
            "edge_case_guidance": "z",
        }
    }

    with caplog.at_level(
        "WARNING", logger="shared.intake_conversation.extractor"
    ):
        result = extract_slots(
            messages=[_msg("recruiter", "noise")],
            current_v2_draft=existing_draft,
            source_packet=None,
        )

    assert result.dropped_keys == ("capability_areas",)
    assert result.v2_updates["role_title"] == "Senior Tax Associate"
    partial_logs = [
        rec for rec in caplog.records if "extraction_partial" in rec.getMessage()
    ]
    assert len(partial_logs) == 1
    assert "capability_areas" in partial_logs[0].getMessage()


def test_extract_slots_no_dropped_keys_on_clean_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structurally clean round reports no dropped keys."""

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "role_title": "Senior Tax Associate",
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "tax associate")],
        current_v2_draft={},
        source_packet=None,
    )

    assert result.dropped_keys == ()


def test_extract_slots_passes_when_updates_complete_a_valid_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "capability_areas": [
                {"name": "Tax compliance", "description": "Owns SUT filings."}
            ],
            "depth_distinction": {
                "builder_definition": "Builds.",
                "user_definition": "Uses.",
                "edge_case_guidance": "Borderline goes to ___.",
            },
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "tax stuff")],
        current_v2_draft={},
        source_packet=None,
    )

    assert "capability_areas" in result.v2_updates
    assert "depth_distinction" in result.v2_updates


# -------------------------------------------------------------------------
# Source packet handling
# -------------------------------------------------------------------------


def test_extract_slots_includes_source_packet_in_user_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _capturing(system, user, expect_json=True, usage_context=None):
        captured["user"] = user
        return {}

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm", _capturing
    )

    extract_slots(
        messages=[_msg("recruiter", "see attached JD")],
        current_v2_draft={},
        source_packet={"raw_text": "JD body about tax associate role"},
    )

    assert "source_packet" in captured["user"]
    assert "tax associate" in captured["user"]


# -------------------------------------------------------------------------
# Insight split + isolation (the deliberate seam)
# -------------------------------------------------------------------------


_BFS_ROLE_DRAFT = {
    "role_title": "Head of Applied AI",
    "role_summary": "Sets AI vision for the BFS group, partners with bank executives.",
    "capability_areas": [
        {"name": "Agentic design", "description": "Owns trade-offs in agentic systems."},
        {"name": "Executive partnership", "description": "Speaks credibly to bank leadership."},
    ],
}


def _bfs_picture_payload(*, corrected: bool = False) -> dict:
    return {
        "summary": (
            "A quasi-CTO of the BFS group who can shape AI vision, "
            "define architectural guardrails for agentic systems, and "
            "still sit with bank executives."
        ),
        "proof_points": [
            "Has shipped an applied AI lab inside a BFS firm.",
            "Owned architecture trade-offs on agentic systems in production.",
        ],
        "screening_translation": (
            "Reject candidates who have only advised on GenAI programs "
            "without owning agentic architecture decisions."
        ),
        "confidence": 0.7,
        "source": "combined",
        "corrected_by_recruiter": corrected,
    }


def test_extract_slots_splits_insight_keys_off_v2_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A combined response (v2 fields + hiring_manager_success_image)
    must surface as ``v2_updates`` (without the insight key) and
    ``insight_updates`` (with only the insight key).
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "role_title": "Head of Applied AI",
            "role_summary": "Sets AI vision for the BFS group.",
            "hiring_manager_success_image": _bfs_picture_payload(),
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "We need a Head of Applied AI for BFS")],
        current_v2_draft=_BFS_ROLE_DRAFT,
        source_packet={"job_description_text": "BFS group hiring Head of Applied AI"},
    )

    assert "hiring_manager_success_image" not in result.v2_updates
    assert result.v2_updates["role_title"] == "Head of Applied AI"
    picture = result.insight_updates["hiring_manager_success_image"]
    assert picture["summary"].lower().startswith("a quasi-cto")
    assert picture["source"] in {"combined", "conversation", "source_packet"}
    assert picture["corrected_by_recruiter"] is False


def test_extract_slots_normalizes_insight_through_shared_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trope-shaped insight output must be dropped at the extractor
    boundary via ``normalize_hiring_manager_success_image``.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "hiring_manager_success_image": {
                "summary": "Strong communication skills and a team player who self-starts.",
                "proof_points": ["team player", "self-starter"],
                "screening_translation": "reject candidates who fail communication.",
                "confidence": 0.5,
                "source": "conversation",
            },
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "We need a Head of Applied AI for BFS")],
        current_v2_draft=_BFS_ROLE_DRAFT,
        source_packet=None,
    )

    assert result.insight_updates == {}
    assert result.v2_updates == {}


def test_extract_slots_keeps_insight_when_v2_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed v2 key drops only that key from ``v2_updates`` (P9.4);
    a normalized insight in the same round survives independently, and
    the still-valid ``depth_distinction`` update also survives.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "capability_areas": [
                {"foo": "bar"},  # missing required name + description
            ],
            "depth_distinction": {
                "builder_definition": "x",
                "user_definition": "y",
                "edge_case_guidance": "z",
            },
            "hiring_manager_success_image": _bfs_picture_payload(),
        },
    )

    result = extract_slots(
        messages=[_msg("recruiter", "still BFS Applied AI")],
        current_v2_draft=_BFS_ROLE_DRAFT,
        source_packet=None,
    )

    assert "capability_areas" not in result.v2_updates
    assert result.dropped_keys == ("capability_areas",)
    assert "depth_distinction" in result.v2_updates
    assert "hiring_manager_success_image" in result.insight_updates


def test_extract_slots_carries_corrected_by_recruiter_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the recruiter explicitly corrects the picture in their most
    recent turn, the extractor surface preserves the
    ``corrected_by_recruiter`` flag for the API persistence layer.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        lambda system, user, expect_json=True, usage_context=None: {
            "hiring_manager_success_image": _bfs_picture_payload(corrected=True),
        },
    )

    result = extract_slots(
        messages=[
            _msg("cloris", "Picturing a quasi-CTO."),
            _msg(
                "recruiter",
                "Actually, the picture is more an architect who can sit with bank executives.",
            ),
        ],
        current_v2_draft=_BFS_ROLE_DRAFT,
        source_packet=None,
    )

    picture = result.insight_updates["hiring_manager_success_image"]
    assert picture["corrected_by_recruiter"] is True


def test_extract_slots_threads_current_intake_insights_into_user_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cheap_llm user prompt must include the current intake
    insights so the model can preserve a recruiter-corrected picture
    rather than re-inventing one each turn.
    """

    captured = {}

    def _capturing(system, user, expect_json=True, usage_context=None):
        captured["user"] = user
        return {}

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm", _capturing
    )

    extract_slots(
        messages=[_msg("recruiter", "still BFS Applied AI")],
        current_v2_draft=_BFS_ROLE_DRAFT,
        source_packet=None,
        current_intake_insights={
            "hiring_manager_success_image": _bfs_picture_payload(corrected=True)
        },
    )

    assert "current_intake_insights" in captured["user"]
    assert "quasi-CTO" in captured["user"]


def test_extractor_schema_documents_hiring_manager_picture_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extractor system prompt must document the new slot and the
    recruiter-correction emission rule so cheap_llm knows when to set
    ``corrected_by_recruiter: true``.
    """

    captured = {}

    def _capturing(system, user, expect_json=True, usage_context=None):
        captured["system"] = system
        return {}

    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm", _capturing
    )

    extract_slots(
        messages=[_msg("recruiter", "tax associate")],
        current_v2_draft={},
        source_packet=None,
    )

    system = captured["system"]
    assert "hiring_manager_success_image" in system
    assert "corrected_by_recruiter" in system
    # Trope ban is named.
    assert "strong communication skills" in system.lower()
