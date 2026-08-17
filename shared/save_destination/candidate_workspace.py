"""Cloris-native candidate workspace save destination — Slice A.5.

Per ``docs/cloris-save-destination-abstraction.md`` §3.2 + the
schema at ``docs/cloris-candidate-workspace-spec.md`` §3 (Slice A.3
ships the migration). Writes a row to ``workspace_entries`` for
every SAVE-family decision, dispatched per brief's
``save_destinations`` declaration.

Universal applicability: ``supports()`` returns True for every
(brief, source) — the workspace is the cross-source aggregation
surface so every saved candidate lands here regardless of module.
LinkedIn briefs typically declare both ``["linkedin_recruiter",
"candidate_workspace"]`` so saves land in both surfaces; non-LinkedIn
briefs declare just ``["candidate_workspace"]``.

Idempotency: the writer uses ``INSERT ... ON CONFLICT(brief_id,
candidate_id) DO UPDATE`` so retries don't duplicate rows. The
``UNIQUE(brief_id, candidate_id)`` constraint added in Slice A.3
makes this safe.

Today's slice ships the writer; the recruiter-mutation endpoints
(PATCH ``/api/workspace/entry/{entry_id}``, POST
``/api/workspace/entry/{entry_id}/review``) and frontend HITL card
rendering land in Phase C / G alongside the Designer pipeline that
exercises the ``surface_type: "hitl_visual_review"`` payload first.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.save_destination import AbstractSaveDestination, SaveResult


# Per spec at docs/cloris-candidate-workspace-spec.md:50-72 the
# default ``surface_type`` is ``"save"`` — Designer overrides to
# ``"hitl_visual_review"`` via the evidence_payload from Phase C.3,
# Exec Search overrides to ``"exec_search_dossier"`` via Phase D.4,
# other modules can introduce their own surface types.
_DEFAULT_SURFACE_TYPE = "save"

# Per spec, ``review_status`` starts at ``"unreviewed"`` and the
# recruiter walks it through confirmed / rejected / borderline at
# the workspace card. ``outreach_status`` starts at ``"none"`` and
# the recruiter walks it through queued / sent / replied / declined.
_DEFAULT_REVIEW_STATUS = "unreviewed"
_DEFAULT_OUTREACH_STATUS = "none"


def install_schema(conn: sqlite3.Connection) -> None:
    """Install candidate-workspace tables into the shared SQLite DB."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brief_id TEXT NOT NULL,
            candidate_id INTEGER NOT NULL,
            person_id INTEGER,
            surface_type TEXT NOT NULL DEFAULT 'save',
            save_decision TEXT NOT NULL,
            save_confidence REAL,
            save_rationale TEXT NOT NULL DEFAULT '',
            save_path TEXT NOT NULL DEFAULT '',
            review_status TEXT NOT NULL DEFAULT 'unreviewed',
            review_marked_at TEXT,
            outreach_status TEXT NOT NULL DEFAULT 'none',
            outreach_marked_at TEXT,
            recruiter_notes TEXT NOT NULL DEFAULT '',
            contextualization_payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(brief_id, candidate_id),
            FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_workspace_brief
        ON workspace_entries(brief_id, review_status, outreach_status);
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace_review_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_entry_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(workspace_entry_id) REFERENCES workspace_entries(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_workspace_review_events_entry
        ON workspace_review_events(workspace_entry_id, created_at);
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace_outreach_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_entry_id INTEGER NOT NULL,
            generator_module TEXT NOT NULL,
            template_version TEXT NOT NULL DEFAULT '',
            subject TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(workspace_entry_id) REFERENCES workspace_entries(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_workspace_outreach_artifacts_entry
        ON workspace_outreach_artifacts(workspace_entry_id, created_at);
        """
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path(envelope: Any) -> Path:
    """Pull the canonical SQLite path from the envelope.

    The envelope shape varies across module pipelines (LinkedIn carries
    a ``runtime_state_path`` attr; researcher / designer / exec_search
    each have their own). The destination accepts a flexible
    extraction so per-module code can pass whatever they have.

    Falls back to a no-op path if the envelope shape doesn't carry a
    db reference; the resulting save raises clearly rather than
    silently writing nowhere.
    """

    db_path = getattr(envelope, "runtime_state_path", None) or getattr(
        envelope, "db_path", None
    )
    if db_path is None:
        raise ValueError(
            "CandidateWorkspaceSaveDestination requires the envelope to "
            "expose a `runtime_state_path` or `db_path` attribute pointing "
            "at the brief's runtime_state.sqlite3."
        )
    return Path(db_path)


class CandidateWorkspaceSaveDestination(AbstractSaveDestination):
    """Cloris-native candidate workspace save destination.

    Writes to the ``workspace_entries`` table added by Slice A.3.
    Universal applicability — every (brief, source) is supported,
    so this destination joins ``LinkedInRecruiterSaveDestination``
    on LinkedIn briefs (both ``["linkedin_recruiter",
    "candidate_workspace"]``) and stands alone on non-LinkedIn
    briefs.

    The destination is module-agnostic: it doesn't know whether the
    candidate came from LinkedIn, GitHub, Researcher, Designer, or
    Exec Search. The per-module ``evidence_payload`` carries the
    module-specific context (e.g., Designer's ``visual_judgment``
    block, Researcher's publication-record summary) into
    ``contextualization_payload_json`` for the workspace card to
    render.
    """

    name: str = "candidate_workspace"

    def supports(self, *, brief: Any, source: str) -> bool:
        """Universal — every (brief, source) is supported.

        The workspace is the cross-source aggregation surface; saves
        from any module land here. The brief's ``save_destinations``
        declaration controls whether to dispatch to this destination
        on a per-brief basis (e.g., a LinkedIn-only brief that
        explicitly opts out of workspace surfacing).
        """

        del brief, source
        return True

    def save(
        self,
        *,
        envelope: Any,
        decision: Any,
        evidence_payload: dict[str, Any],
        attempt_id: int | None,
    ) -> SaveResult:
        """Write a ``workspace_entries`` row for this save.

        Pulls ``brief_id`` + ``candidate_id`` from the envelope,
        decision details (``decision``, ``confidence``, ``rationale``,
        ``path``) from the OpusDecision, and module-specific context
        from ``evidence_payload`` (carrying ``surface_type`` if the
        module wants to override the default ``"save"``).

        Idempotent via ``ON CONFLICT(brief_id, candidate_id) DO UPDATE``:
        a re-save for the same (brief, candidate) updates the existing
        row rather than failing on the UNIQUE constraint. Per the spec
        at ``docs/cloris-candidate-workspace-spec.md`` the workspace
        is brief-scoped; cross-brief candidates surface as separate
        entries.

        ``attempt_id`` is currently unused (the workspace doesn't
        wire to ``side_effects`` rows the way LinkedIn does); it's
        accepted for interface uniformity with
        :class:`LinkedInRecruiterSaveDestination`.
        """

        del attempt_id

        brief_id = getattr(envelope, "brief_id", None)
        candidate_id = getattr(envelope, "candidate_id", None)
        if not brief_id or not candidate_id:
            return SaveResult(
                destination=self.name,
                status="failed",
                error=(
                    "envelope missing brief_id / candidate_id; cannot "
                    "write workspace_entries row"
                ),
            )

        try:
            db_path = _resolve_db_path(envelope)
        except ValueError as exc:
            return SaveResult(
                destination=self.name,
                status="failed",
                error=str(exc),
            )

        # Surface-type override: per-module write paths can pass a
        # non-default ``surface_type`` to route the workspace card
        # to the appropriate render mode. Designer's Phase C.3
        # writes ``"hitl_visual_review"``; Exec Search's Phase D.4
        # writes ``"exec_search_dossier"``.
        surface_type = str(
            evidence_payload.get("surface_type", _DEFAULT_SURFACE_TYPE)
        )

        # Module-specific evidence (Designer's ``visual_judgment``
        # block, Researcher's publication record, Exec Search's
        # dossier sections) lands in ``contextualization_payload_json``
        # so the workspace card renderer has a uniform field to read.
        # The destination doesn't interpret the payload; it just
        # serializes it.
        contextualization = {
            k: v for k, v in evidence_payload.items() if k != "surface_type"
        }

        save_decision = getattr(decision, "decision", "SAVE")
        confidence = getattr(decision, "confidence", None)
        rationale = getattr(decision, "rationale", "") or ""
        path = getattr(decision, "path", "") or ""

        now = _utc_now()
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA busy_timeout=5000")
                cur = conn.execute(
                    """
                    INSERT INTO workspace_entries(
                        brief_id, candidate_id, surface_type,
                        save_decision, save_confidence, save_rationale,
                        save_path, review_status, outreach_status,
                        contextualization_payload_json,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(brief_id, candidate_id) DO UPDATE SET
                        surface_type = excluded.surface_type,
                        save_decision = excluded.save_decision,
                        save_confidence = excluded.save_confidence,
                        save_rationale = excluded.save_rationale,
                        save_path = excluded.save_path,
                        contextualization_payload_json =
                            excluded.contextualization_payload_json,
                        updated_at = excluded.updated_at
                    RETURNING id
                    """,
                    (
                        brief_id,
                        candidate_id,
                        surface_type,
                        save_decision,
                        confidence,
                        rationale,
                        path,
                        _DEFAULT_REVIEW_STATUS,
                        _DEFAULT_OUTREACH_STATUS,
                        json.dumps(contextualization, sort_keys=True),
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return SaveResult(
                destination=self.name,
                status="failed",
                error=f"workspace_entries write raised: {exc}",
            )

        entry_id = int(row["id"]) if row is not None else None
        return SaveResult(
            destination=self.name,
            status="succeeded",
            payload={"workspace_entry_id": entry_id, "surface_type": surface_type},
        )


__all__ = ["CandidateWorkspaceSaveDestination"]
