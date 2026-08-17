"""Researcher runtime-state bridge — Slice 6.

Mirrors :class:`shared.runtime_state.github.GitHubRuntimeStateBridge`
but for researcher author-query work_units. Per Researcher Module Spec
Slice 6:

- Resume semantics: re-read ``work_units WHERE status IN
  ('queued', 'in_progress')``; pagination cursor lives in
  ``checkpoint_json``.
- ``identity_key`` resolution per Slice 4: ORCID-when-present,
  ``openalex:{author_id}`` composite fallback (computed by
  :func:`researcher.schemas.identity_key_for_candidate`).
- Saves stay in the ``candidates`` table (no per-source side-effects
  table) per Spec Opinion 4 — workspace is the only save destination.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researcher.schemas import (
    ResearcherCandidate,
    ResearcherSnippet,
    identity_key_for_candidate,
)
from shared.execution import CandidateExecutionEngine
from shared.runtime_state.store import (
    RESEARCHER_AUTHOR_QUERY_KIND,
    RuntimeStateStore,
)
from shared.schemas import OpusDecision


SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}


def _resolve_recruiter_id() -> int | None:
    """Resolve the acting recruiter for a run-start, fail-soft to None.

    reopen Stage 2 (R5a-3): stamps ``runs.recruiter_id`` so the read-only
    taste aggregator (R5a-4) can attribute adaptation decisions. The
    resolver is the single auth seam (``shared.recruiter_context``); we
    catch broadly because a run launch must never die on recruiter
    resolution — a None recruiter_id is a clean "unknown" (the aggregator
    skips it), whereas a raised exception here would abort the run.
    """

    try:
        from shared.recruiter_context import get_current_recruiter_id

        return get_current_recruiter_id()
    except Exception:  # noqa: BLE001 — resolution must never break a run launch
        return None


class ResearcherRuntimeStateBridge:
    """Keep researcher runtime semantics DB-authoritative.

    Slice 6 ships the minimum needed for the orchestrator to:
    1. Start or resume a run.
    2. Upsert each query as a work_unit (paginated; cursor in
       ``checkpoint_json``).
    3. Record candidate discovery + facial + full attempts via the
       shared :class:`CandidateExecutionEngine`.
    4. Write terminal decisions with `full_decision` populating the
       wire contract per Spec Opinion 6.
    """

    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        output_dir: str | Path,
        brief_id: str,
        brief_name: str,
        brief_path: str | None = None,
    ) -> None:
        self.store = store
        self.output_dir = Path(output_dir)
        self.brief_id = brief_id
        self.brief_name = brief_name
        self.brief_path = brief_path
        self._execution_engine = CandidateExecutionEngine(
            store=self.store,
            output_dir=str(self.output_dir),
            brief_id=self.brief_id,
            source="researcher",
        )

    # -----------------------------------------------------------------
    # Run lifecycle
    # -----------------------------------------------------------------

    def has_runtime_state(self) -> bool:
        latest_run = self.store.get_latest_run(
            source="researcher", brief_id=self.brief_id
        )
        return bool(
            latest_run
            or self.store.has_candidates(source="researcher", brief_id=self.brief_id)
        )

    def start_or_resume_run(self, *, resume: bool) -> int:
        """Start a new run or resume the latest one.

        Mirrors :class:`GitHubRuntimeStateBridge.start_or_resume_run`
        shape — same reconciliation calls, same brief-identity pinning.
        Returns the new run_id.
        """

        self.store.reconcile_open_attempts(
            source="researcher", brief_id=self.brief_id
        )
        self.store.reconcile_pending_side_effects(
            source="researcher", brief_id=self.brief_id
        )
        latest_run = self.store.get_latest_run(
            source="researcher", brief_id=self.brief_id
        )

        from shared.brief_identity import compute_brief_identity

        identity = (
            compute_brief_identity(self.brief_path) if self.brief_path else None
        )
        identity_kwargs: dict[str, Any] = {}
        if identity is not None:
            identity_kwargs = {
                "brief_path_at_launch": identity["brief_path_at_launch"],
                "brief_content_hash": identity["brief_content_hash"],
                "brief_snapshot_json": identity["brief_snapshot_json"],
            }

        recruiter_id = _resolve_recruiter_id()

        if resume and latest_run and self.store.has_work_units(int(latest_run["id"])):
            return self.store.start_run(
                source="researcher",
                brief_id=self.brief_id,
                output_dir=str(self.output_dir),
                mode="resume",
                resume_state=self.store.get_run_resume_state(int(latest_run["id"])),
                resumed_from_run_id=int(latest_run["id"]),
                clone_work_units_from_run_id=int(latest_run["id"]),
                recruiter_id=recruiter_id,
                **identity_kwargs,
            )

        return self.store.start_run(
            source="researcher",
            brief_id=self.brief_id,
            output_dir=str(self.output_dir),
            mode="resume" if resume else "fresh",
            resume_state={"brief_name": self.brief_name},
            resumed_from_run_id=int(latest_run["id"]) if resume and latest_run else None,
            recruiter_id=recruiter_id,
            **identity_kwargs,
        )

    # -----------------------------------------------------------------
    # Work-unit upsert (one per researcher query)
    # -----------------------------------------------------------------

    def upsert_query_work_unit(
        self,
        *,
        run_id: int,
        query: dict,
        status: str = "queued",
        ordering_index: int | None = None,
        cursor: str = "*",
        candidates_discovered: int = 0,
        facial_yes_count: int = 0,
        facial_no_count: int = 0,
        facial_borderline_count: int = 0,
        saves_count: int = 0,
        rejected_count: int = 0,
    ) -> None:
        """Write/update one query as a work_unit.

        ``checkpoint_json`` carries the pagination cursor + per-query
        counters; resume reads cursor from here.
        """

        idx = (
            ordering_index
            if ordering_index is not None
            else int(query.get("id") or 0)
        )
        topic_lane = "+".join(str(item) for item in query.get("topic_concepts") or [])
        venue_lane = "+".join(str(item) for item in query.get("venue_filter") or [])
        family_key = topic_lane or venue_lane or "researcher"
        domain_lane = venue_lane or topic_lane
        novelty_bucket = "adapted" if query.get("adapted_from_batch") else "initial"
        self.store.upsert_work_unit(
            run_id=run_id,
            source="researcher",
            brief_id=self.brief_id,
            kind=RESEARCHER_AUTHOR_QUERY_KIND,
            source_unit_id=str(query.get("id") or idx),
            display_name=str(query.get("name") or f"query-{idx}"),
            ordering_index=idx,
            status=status,
            payload=dict(query),
            checkpoint={
                "cursor": cursor,
                "topic_concepts": list(query.get("topic_concepts") or []),
                "venue_filter": list(query.get("venue_filter") or []),
            },
            metrics={},
            family_key=family_key,
            novelty_bucket=novelty_bucket,
            domain_lane=domain_lane,
            counters={
                "result_count": candidates_discovered,
                "candidates_discovered": candidates_discovered,
                "facial_yes_count": facial_yes_count,
                "facial_no_count": facial_no_count,
                "facial_borderline_count": facial_borderline_count,
                "saves_count": saves_count,
                "rejected_count": rejected_count,
            },
            notes="",
        )

    def load_pending_queries(
        self,
        run_id: int,
    ) -> list[dict]:
        """Return the work-unit payloads for queries still queued or
        in-progress (used by Slice 6 resume).
        """

        rows = self.store.list_work_units(
            run_id=run_id,
            kind=RESEARCHER_AUTHOR_QUERY_KIND,
        )
        pending: list[dict] = []
        for row in rows:
            status = row.get("status") if isinstance(row, dict) else getattr(row, "status", "")
            if status not in ("queued", "in_progress"):
                continue
            payload_json = (
                row.get("payload_json")
                if isinstance(row, dict)
                else getattr(row, "payload_json", "{}")
            )
            try:
                payload = json.loads(payload_json or "{}")
            except (json.JSONDecodeError, TypeError):
                payload = {}
            pending.append(payload)
        return pending

    # -----------------------------------------------------------------
    # Per-candidate execution
    # -----------------------------------------------------------------

    def record_candidate_discovery(
        self,
        *,
        run_id: int,
        query_id: int,
        candidate: ResearcherCandidate,
    ) -> int | None:
        """Record a researcher candidate discovery.

        Returns the candidate row id (None if the candidate has no
        identity anchor — defensive against acquisition bugs).
        """

        identity_key = identity_key_for_candidate(candidate)
        if not identity_key:
            return None
        return self.store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=None,  # The bridge keeps work_unit lookup off the hot path.
            source="researcher",
            brief_id=self.brief_id,
            identity_key=identity_key,
            display_name=candidate.name,
            profile_url=candidate.profile_url,
        )

    def record_facial_decision(
        self,
        *,
        run_id: int,
        snippet: ResearcherSnippet,
        decision: OpusDecision,
    ) -> None:
        """Walk a candidate through ``discovered → snippet_extracted →
        facial_started → facial_terminal``; write the facial decision.

        For FACIAL_NO, the candidate's ``terminal_decision`` is set so
        the workspace surface filters it out of the SAVE list. For
        FACIAL_YES / FACIAL_BORDERLINE, no terminal_decision is set —
        the orchestrator escalates to full eval and that path writes
        the canonical terminal payload via :meth:`record_full_decision`.
        """

        identity_key = self._snippet_identity_key(snippet)
        if not identity_key:
            return
        # Walk the canonical lifecycle: discovered → snippet_extracted.
        snippet_attempt = self.store.start_attempt(
            run_id=run_id,
            source="researcher",
            brief_id=self.brief_id,
            identity_key=identity_key,
            stage="snippet",
            display_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        if snippet_attempt is None:
            return
        self.store.finish_attempt_success(
            attempt_id=snippet_attempt,
            new_state="snippet_extracted",
            payload={"snippet": snippet.to_dict()},
            run_id=run_id,
        )

        # snippet_extracted → facial_started.
        self.store.set_candidate_state(
            run_id=run_id,
            source="researcher",
            brief_id=self.brief_id,
            identity_key=identity_key,
            new_state="facial_started",
        )
        facial_attempt = self.store.start_attempt(
            run_id=run_id,
            source="researcher",
            brief_id=self.brief_id,
            identity_key=identity_key,
            stage="facial",
            display_name=snippet.name,
            profile_url=snippet.profile_url,
        )
        if facial_attempt is None:
            return
        terminal_decision = (
            decision.decision if decision.decision == "FACIAL_NO" else None
        )
        attempt_payload: dict[str, Any] = {"facial_decision": decision.to_dict()}
        terminal_payload: dict[str, Any] = {"facial_decision": decision.to_dict()}
        prompt_capture = getattr(decision, "prompt_capture", None)
        if isinstance(prompt_capture, dict) and prompt_capture:
            attempt_payload["prompt_capture"] = dict(prompt_capture)
            observability = {
                key: value
                for key in ("trace_id", "observation_id", "trace_url")
                if isinstance((value := prompt_capture.get(key)), str) and value
            }
            observability["prompt_capture_schema_version"] = prompt_capture.get(
                "schema_version"
            )
            terminal_payload["observability"] = observability
        self.store.finish_attempt_success(
            attempt_id=facial_attempt,
            new_state="facial_terminal",
            terminal_decision=terminal_decision,
            payload=attempt_payload,
            terminal_payload=terminal_payload,
            run_id=run_id,
        )

    def record_full_decision(
        self,
        *,
        run_id: int,
        candidate: ResearcherCandidate,
        decision: OpusDecision,
        needs_identity_confirmation: bool = False,
        identity_review_note: str = "",
    ) -> None:
        """Walk a candidate through full eval; write the canonical
        terminal payload with ``full_decision`` per Spec Opinion 6.

        The wire contract is non-negotiable — this is the path
        :func:`shared.runtime_state.read_models.extract_save_reason_and_confidence`
        reads to surface save reason + confidence on the workspace card.

        ``needs_identity_confirmation`` is the Move #26 plumb: when the
        Slice-4 disambiguator flagged this candidate as a common-name
        collision (≥2 ORCID-less authors with the same normalized name),
        the flag rides on the terminal payload as
        ``needs_identity_confirmation: true`` + an
        ``identity_review_note`` so the recruiter workspace card can
        surface a "needs manual review" affordance. Researcher's
        workspace card lands in audit Move #4; until then the flag is
        readable from terminal_payload_json by any consumer.
        """

        identity_key = identity_key_for_candidate(candidate)
        if not identity_key:
            return
        # The orchestrator always calls record_facial_decision first
        # (the candidate is in `facial_terminal` by the time we get
        # here). Walk facial_terminal → full_started → full_terminal.
        # We do NOT re-call record_candidate_discovery — it would reset
        # `current_lifecycle_state = 'discovered'` and break the guard.
        self.store.set_candidate_state(
            run_id=run_id,
            source="researcher",
            brief_id=self.brief_id,
            identity_key=identity_key,
            new_state="full_started",
        )
        attempt_id = self.store.start_attempt(
            run_id=run_id,
            source="researcher",
            brief_id=self.brief_id,
            identity_key=identity_key,
            stage="full",
            display_name=candidate.name,
            profile_url=candidate.profile_url,
        )
        if attempt_id is None:
            return
        payload: dict[str, Any] = {
            "full_decision": decision.to_dict(),
            "candidate_record": candidate.to_dict(),
        }
        if needs_identity_confirmation:
            payload["needs_identity_confirmation"] = True
            if identity_review_note:
                payload["identity_review_note"] = identity_review_note
        attempt_payload = dict(payload)
        terminal_payload = dict(payload)
        prompt_capture = getattr(decision, "prompt_capture", None)
        if isinstance(prompt_capture, dict) and prompt_capture:
            attempt_payload["prompt_capture"] = dict(prompt_capture)
            observability = {
                key: value
                for key in ("trace_id", "observation_id", "trace_url")
                if isinstance((value := prompt_capture.get(key)), str) and value
            }
            observability["prompt_capture_schema_version"] = prompt_capture.get(
                "schema_version"
            )
            terminal_payload["observability"] = observability
        self.store.finish_attempt_success(
            attempt_id=attempt_id,
            new_state="full_terminal",
            terminal_decision=decision.decision,
            payload=attempt_payload,
            terminal_payload=terminal_payload,
            run_id=run_id,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _snippet_identity_key(self, snippet: ResearcherSnippet) -> str:
        """Resolve identity key from snippet's profile_url (ORCID URL or
        OpenAlex URL). Falls back to a hash of the URL when neither is
        recognized — the orchestrator should normally not hit this
        path because acquisition supplies the structured ResearcherCandidate
        first.
        """

        url = snippet.profile_url or ""
        if "orcid.org/" in url:
            tail = url.rsplit("/", 1)[-1]
            return f"orcid:{tail}"
        if "openalex.org/" in url:
            tail = url.rsplit("/", 1)[-1]
            return f"openalex:{tail}"
        return ""
