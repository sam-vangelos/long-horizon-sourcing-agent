"""Slice A.4 + A.5 — save destination interface + implementations.

Pins the AbstractSaveDestination contract:

- A.4: interface (``shared/save_destination/__init__.py``) + LinkedIn
  wrapper (``shared/save_destination/linkedin_recruiter.py``).
- A.5: workspace destination
  (``shared/save_destination/candidate_workspace.py``).

The full LinkedIn body-move (per spec §3.1) is a behavior-preserving
follow-up; A.4 ships the interface only and the LinkedIn wrapper
raises ``NotImplementedError`` on ``save()`` until the body lands.
The workspace destination is fully wired and writes to the
``workspace_entries`` table added by Slice A.3.

Tests pin:

- The interface is importable + correctly abstract (cannot be
  instantiated directly).
- ``LinkedInRecruiterSaveDestination.supports()`` returns True for
  ``source="linkedin"``, False for non-linkedin sources.
- ``CandidateWorkspaceSaveDestination.supports()`` returns True for
  every (brief, source).
- ``CandidateWorkspaceSaveDestination.save()`` writes a workspace
  row with the expected shape, idempotently merges on re-save, and
  returns ``SaveResult(status="succeeded", workspace_entry_id=...)``.
- ``surface_type`` override via ``evidence_payload`` flows through
  to the row.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from shared.runtime_state.store import RuntimeStateStore
from shared.save_destination import (
    SAVE_DECISION_FAMILY,
    AbstractSaveDestination,
    SaveResult,
)
from shared.save_destination.candidate_workspace import (
    CandidateWorkspaceSaveDestination,
)
from shared.save_destination.linkedin_recruiter import (
    LinkedInRecruiterSaveDestination,
)


@dataclass
class _StubBrief:
    id: str = "brief-test"
    target_modules: list[str] | None = None


@dataclass
class _StubDecision:
    """Minimal OpusDecision-shaped object for testing."""

    decision: str = "SAVE"
    confidence: float = 0.85
    rationale: str = "Strong evidence of capability."
    path: str = "DIRECT:1. Capability"


@dataclass
class _StubEnvelope:
    brief_id: str
    candidate_id: int
    runtime_state_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_store_with_candidate(
    tmp_path: Path,
    *,
    brief_id: str = "brief-test",
    identity_key: str = "cand-x",
) -> tuple[RuntimeStateStore, int]:
    """Create a runtime store + insert a candidate row; return (store, cand_id)."""

    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    now = _utc_now()
    with store.connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO candidates(
                source, brief_id, identity_key, display_name,
                current_lifecycle_state, terminal_decision,
                first_seen_at, last_seen_at
            )
            VALUES ('linkedin', ?, ?, 'Test Candidate',
                    'evaluated', 'SAVE', ?, ?)
            """,
            (brief_id, identity_key, now, now),
        )
        cand_id = int(cur.lastrowid)
    return store, cand_id


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


def test_save_decision_family_is_frozen() -> None:
    assert isinstance(SAVE_DECISION_FAMILY, frozenset)
    assert SAVE_DECISION_FAMILY == frozenset({
        "SAVE",
        "INFERENTIAL_SAVE",
        "TRANSFERABLE_SAVE",
        "SIGNAL_SAVE",
    })


def test_review_decisions_excluded_from_save_family() -> None:
    """P4: ``REVIEW_INFERRED`` and ``REVIEW_FLAGGED`` are bounded non-save
    review outcomes. They MUST NOT appear in ``SAVE_DECISION_FAMILY``;
    the save-destination registry treats them as non-dispatchable, which
    is what keeps the LinkedIn save-click and workspace ``save_decision``
    write from firing for ambiguous candidates.
    """

    assert "REVIEW_INFERRED" not in SAVE_DECISION_FAMILY
    assert "REVIEW_FLAGGED" not in SAVE_DECISION_FAMILY


def test_abstract_save_destination_cannot_be_instantiated() -> None:
    """ABC enforcement — direct instantiation must raise."""

    with pytest.raises(TypeError):
        AbstractSaveDestination()  # type: ignore[abstract]


def test_save_result_carries_destination_and_status() -> None:
    result = SaveResult(destination="linkedin_recruiter", status="succeeded")
    assert result.destination == "linkedin_recruiter"
    assert result.status == "succeeded"
    assert result.side_effect_id is None
    assert result.payload == {}
    assert result.error is None


# ---------------------------------------------------------------------------
# LinkedInRecruiterSaveDestination — wrap (Slice A.4)
# ---------------------------------------------------------------------------


def test_linkedin_destination_supports_only_linkedin_source() -> None:
    """Per spec §3.1: only LinkedIn-sourced candidates dispatch to Recruiter."""

    dest = LinkedInRecruiterSaveDestination(side_effects_service=None)  # type: ignore[arg-type]
    brief = _StubBrief()

    assert dest.supports(brief=brief, source="linkedin") is True
    assert dest.supports(brief=brief, source="github") is False
    assert dest.supports(brief=brief, source="researcher") is False
    assert dest.supports(brief=brief, source="designer") is False
    assert dest.supports(brief=brief, source="exec_search") is False


def test_linkedin_destination_save_raises_until_body_move_lands() -> None:
    """A.4 ships the wrap only; ``save()`` raises NotImplementedError until
    the behavior-preserving body-move follow-up lands."""

    dest = LinkedInRecruiterSaveDestination(side_effects_service=None)  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="body-move"):
        dest.save(
            envelope=None,
            decision=None,
            evidence_payload={},
            attempt_id=None,
        )


def test_linkedin_destination_name_is_canonical() -> None:
    dest = LinkedInRecruiterSaveDestination(side_effects_service=None)  # type: ignore[arg-type]
    assert dest.name == "linkedin_recruiter"


# ---------------------------------------------------------------------------
# CandidateWorkspaceSaveDestination — full writer (Slice A.5)
# ---------------------------------------------------------------------------


def test_workspace_destination_supports_every_brief_and_source() -> None:
    """Universal applicability — workspace is the cross-source surface."""

    dest = CandidateWorkspaceSaveDestination()
    brief = _StubBrief()

    for source in (
        "linkedin",
        "github",
        "researcher",
        "designer",
        "exec_search",
        "future_module",
    ):
        assert dest.supports(brief=brief, source=source) is True


def test_workspace_destination_name_is_canonical() -> None:
    dest = CandidateWorkspaceSaveDestination()
    assert dest.name == "candidate_workspace"


def test_workspace_destination_writes_row_with_decision_fields(
    tmp_path: Path,
) -> None:
    """``save()`` writes a workspace_entries row carrying the decision."""

    store, cand_id = _make_store_with_candidate(tmp_path, brief_id="brief-w1")
    dest = CandidateWorkspaceSaveDestination()

    result = dest.save(
        envelope=_StubEnvelope(
            brief_id="brief-w1",
            candidate_id=cand_id,
            runtime_state_path=store.db_path,
        ),
        decision=_StubDecision(
            decision="SAVE",
            confidence=0.9,
            rationale="Top-decile signal.",
            path="DIRECT:1. Signal",
        ),
        evidence_payload={"signal_score": 0.92},
        attempt_id=None,
    )

    assert result.status == "succeeded"
    assert result.destination == "candidate_workspace"
    assert result.error is None
    entry_id = result.payload["workspace_entry_id"]
    assert entry_id > 0

    with store.connect() as conn:
        row = conn.execute(
            "SELECT * FROM workspace_entries WHERE id = ?", (entry_id,)
        ).fetchone()
    assert row is not None
    assert row["brief_id"] == "brief-w1"
    assert row["candidate_id"] == cand_id
    assert row["save_decision"] == "SAVE"
    assert row["save_confidence"] == 0.9
    assert row["save_rationale"] == "Top-decile signal."
    assert row["save_path"] == "DIRECT:1. Signal"
    assert row["surface_type"] == "save"  # default
    assert row["review_status"] == "unreviewed"
    assert row["outreach_status"] == "none"

    contextualization = json.loads(row["contextualization_payload_json"])
    assert contextualization == {"signal_score": 0.92}


def test_workspace_destination_save_is_idempotent_via_on_conflict(
    tmp_path: Path,
) -> None:
    """A re-save for the same (brief, candidate) updates rather than duplicates.

    Pins the ``ON CONFLICT(brief_id, candidate_id) DO UPDATE`` semantics.
    """

    store, cand_id = _make_store_with_candidate(tmp_path, brief_id="brief-w2")
    dest = CandidateWorkspaceSaveDestination()

    first = dest.save(
        envelope=_StubEnvelope(
            brief_id="brief-w2",
            candidate_id=cand_id,
            runtime_state_path=store.db_path,
        ),
        decision=_StubDecision(decision="SAVE", confidence=0.7),
        evidence_payload={"first": True},
        attempt_id=None,
    )
    second = dest.save(
        envelope=_StubEnvelope(
            brief_id="brief-w2",
            candidate_id=cand_id,
            runtime_state_path=store.db_path,
        ),
        decision=_StubDecision(decision="INFERENTIAL_SAVE", confidence=0.6),
        evidence_payload={"second": True},
        attempt_id=None,
    )

    assert first.status == "succeeded"
    assert second.status == "succeeded"
    # Same row id — UPDATE in place rather than new INSERT.
    assert first.payload["workspace_entry_id"] == second.payload["workspace_entry_id"]

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT save_decision, save_confidence, contextualization_payload_json "
            "FROM workspace_entries WHERE brief_id = ?",
            ("brief-w2",),
        ).fetchall()
    assert len(rows) == 1, "re-save must merge, not duplicate"
    # Latest values win.
    assert rows[0]["save_decision"] == "INFERENTIAL_SAVE"
    assert rows[0]["save_confidence"] == 0.6
    assert json.loads(rows[0]["contextualization_payload_json"]) == {"second": True}


def test_workspace_destination_surface_type_override_flows_through(
    tmp_path: Path,
) -> None:
    """Designer's ``hitl_visual_review`` and Exec Search's ``exec_search_dossier``
    surface-type overrides land in the row via evidence_payload."""

    store, cand_id = _make_store_with_candidate(tmp_path, brief_id="brief-w3")
    dest = CandidateWorkspaceSaveDestination()

    result = dest.save(
        envelope=_StubEnvelope(
            brief_id="brief-w3",
            candidate_id=cand_id,
            runtime_state_path=store.db_path,
        ),
        decision=_StubDecision(),
        evidence_payload={
            "surface_type": "hitl_visual_review",
            "visual_judgment": {"per_principle": [{"score": 3}]},
        },
        attempt_id=None,
    )

    assert result.status == "succeeded"
    assert result.payload["surface_type"] == "hitl_visual_review"

    with store.connect() as conn:
        row = conn.execute(
            "SELECT surface_type, contextualization_payload_json "
            "FROM workspace_entries WHERE id = ?",
            (result.payload["workspace_entry_id"],),
        ).fetchone()
    assert row["surface_type"] == "hitl_visual_review"
    contextualization = json.loads(row["contextualization_payload_json"])
    # The surface_type key is consumed by the destination, not stored
    # in contextualization (it has its own column).
    assert "surface_type" not in contextualization
    assert contextualization == {
        "visual_judgment": {"per_principle": [{"score": 3}]}
    }


def test_workspace_destination_failed_save_returns_failed_result(
    tmp_path: Path,
) -> None:
    """Missing brief_id / candidate_id surfaces as failed, not raise."""

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    dest = CandidateWorkspaceSaveDestination()

    result = dest.save(
        envelope=_StubEnvelope(
            brief_id="",  # empty
            candidate_id=0,
            runtime_state_path=store.db_path,
        ),
        decision=_StubDecision(),
        evidence_payload={},
        attempt_id=None,
    )

    assert result.status == "failed"
    assert "brief_id" in (result.error or "")


def test_workspace_destination_missing_db_path_returns_failed_result(
    tmp_path: Path,
) -> None:
    """Envelope without db_path / runtime_state_path returns failed."""

    dest = CandidateWorkspaceSaveDestination()

    @dataclass
    class _BadEnvelope:
        brief_id: str = "brief-x"
        candidate_id: int = 1

    result = dest.save(
        envelope=_BadEnvelope(),
        decision=_StubDecision(),
        evidence_payload={},
        attempt_id=None,
    )

    assert result.status == "failed"
    assert "runtime_state_path" in (result.error or "")
