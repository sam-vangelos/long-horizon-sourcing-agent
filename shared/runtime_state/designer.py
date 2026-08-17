"""Designer runtime-state bridge."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from designer.schemas import DesignerCandidate, DesignerSearchQuery, DesignerSnippet
from shared.runtime_state.store import (
    DESIGNER_BEHANCE_QUERY_KIND,
    DESIGNER_CSE_QUERY_KIND,
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


class DesignerRuntimeStateBridge:
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

    def start_or_resume_run(self, *, resume: bool) -> int:
        self.store.reconcile_open_attempts(source="designer", brief_id=self.brief_id)
        self.store.reconcile_pending_side_effects(source="designer", brief_id=self.brief_id)
        latest_run = self.store.get_latest_run(source="designer", brief_id=self.brief_id)

        from shared.brief_identity import compute_brief_identity

        identity = compute_brief_identity(self.brief_path) if self.brief_path else None
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
                source="designer",
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
            source="designer",
            brief_id=self.brief_id,
            output_dir=str(self.output_dir),
            mode="resume" if resume else "fresh",
            resume_state={"brief_name": self.brief_name},
            resumed_from_run_id=int(latest_run["id"]) if resume and latest_run else None,
            recruiter_id=recruiter_id,
            **identity_kwargs,
        )

    def query_kind(self, query: DesignerSearchQuery) -> str:
        return (
            DESIGNER_CSE_QUERY_KIND
            if query.source == "google_cse"
            else DESIGNER_BEHANCE_QUERY_KIND
        )

    def query_source_unit_id(self, query: DesignerSearchQuery, ordering_index: int) -> str:
        return f"{query.source}:{ordering_index}:{query.query_text.lower().strip()}"

    def upsert_query_work_unit(
        self,
        *,
        run_id: int,
        query: DesignerSearchQuery,
        ordering_index: int,
        status: str,
        counters: dict[str, int] | None = None,
    ) -> int:
        counters = counters or {}
        return self.store.upsert_work_unit(
            run_id=run_id,
            source="designer",
            brief_id=self.brief_id,
            kind=self.query_kind(query),
            source_unit_id=self.query_source_unit_id(query, ordering_index),
            display_name=f"{query.source}: {query.query_text}",
            ordering_index=ordering_index,
            status=status,
            payload=asdict(query),
            checkpoint={"source": query.source, "discipline": query.discipline},
            metrics={},
            family_key=query.capability_area_name or query.query_text,
            novelty_bucket="adapted" if query.extra_filters.get("adapted") else "initial",
            domain_lane=query.discipline or query.capability_area_name,
            counters=counters,
        )

    def record_discovery(
        self,
        *,
        run_id: int,
        work_unit_id: int,
        snippet: DesignerSnippet,
    ) -> int:
        return self.store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=work_unit_id,
            source="designer",
            brief_id=self.brief_id,
            identity_key=snippet.identity_key,
            display_name=snippet.display_name,
            profile_url=snippet.profile_url,
            payload={"snippet": _snippet_payload(snippet)},
        )

    def record_facial_decision(
        self,
        *,
        run_id: int,
        work_unit_id: int,
        snippet: DesignerSnippet,
        decision: OpusDecision,
    ) -> None:
        identity_key = snippet.identity_key
        snippet_attempt = self.store.start_attempt(
            run_id=run_id,
            source="designer",
            brief_id=self.brief_id,
            identity_key=identity_key,
            stage="snippet",
            work_unit_id=work_unit_id,
            display_name=snippet.display_name,
            profile_url=snippet.profile_url,
        )
        self.store.finish_attempt_success(
            attempt_id=snippet_attempt,
            new_state="snippet_extracted",
            payload={"snippet": _snippet_payload(snippet)},
            run_id=run_id,
        )
        self.store.set_candidate_state(
            run_id=run_id,
            source="designer",
            brief_id=self.brief_id,
            identity_key=identity_key,
            new_state="facial_started",
            last_work_unit_id=work_unit_id,
        )
        facial_attempt = self.store.start_attempt(
            run_id=run_id,
            source="designer",
            brief_id=self.brief_id,
            identity_key=identity_key,
            stage="facial",
            work_unit_id=work_unit_id,
            display_name=snippet.display_name,
            profile_url=snippet.profile_url,
        )
        terminal_decision = decision.decision if decision.decision == "FACIAL_NO" else None
        self.store.finish_attempt_success(
            attempt_id=facial_attempt,
            new_state="facial_terminal",
            terminal_decision=terminal_decision,
            payload={"facial_decision": decision.to_dict()},
            terminal_payload={"facial_decision": decision.to_dict()},
            run_id=run_id,
        )

    def record_full_decision(
        self,
        *,
        run_id: int,
        work_unit_id: int,
        candidate: DesignerCandidate,
        decision: OpusDecision,
        terminal_payload: dict[str, Any],
    ) -> None:
        snippet = candidate.snippet
        self.store.set_candidate_state(
            run_id=run_id,
            source="designer",
            brief_id=self.brief_id,
            identity_key=snippet.identity_key,
            new_state="full_started",
            last_work_unit_id=work_unit_id,
        )
        attempt_id = self.store.start_attempt(
            run_id=run_id,
            source="designer",
            brief_id=self.brief_id,
            identity_key=snippet.identity_key,
            stage="full",
            work_unit_id=work_unit_id,
            display_name=snippet.display_name,
            profile_url=snippet.profile_url,
        )
        payload = {
            "full_decision": decision.to_dict(),
            "candidate_record": _candidate_payload(candidate),
            **terminal_payload,
        }
        self.store.finish_attempt_success(
            attempt_id=attempt_id,
            new_state="full_terminal",
            terminal_decision=decision.decision,
            payload=payload,
            terminal_payload=payload,
            run_id=run_id,
        )


def _snippet_payload(snippet: DesignerSnippet) -> dict[str, Any]:
    return {
        "source": snippet.source,
        "identity_key": snippet.identity_key,
        "display_name": snippet.display_name,
        "profile_url": snippet.profile_url,
        "location": snippet.location,
        "headline": snippet.headline,
        "fields": list(snippet.fields),
        "tools": list(snippet.tools),
        "top_project_titles": list(snippet.top_project_titles),
        "appreciation_count_total": snippet.appreciation_count_total,
        "social_links": [list(pair) for pair in snippet.social_links],
    }


def _candidate_payload(candidate: DesignerCandidate) -> dict[str, Any]:
    return {
        "snippet": _snippet_payload(candidate.snippet),
        "project_summaries": [
            {
                "project_id": project.project_id,
                "title": project.title,
                "cover_image_url": project.cover_image_url,
                "appreciation_count": project.appreciation_count,
                "view_count": project.view_count,
                "published_at": project.published_at,
                "fields": list(project.fields),
                "description": project.description,
            }
            for project in candidate.project_summaries
        ],
        "raw_payload": dict(candidate.raw_payload),
    }
