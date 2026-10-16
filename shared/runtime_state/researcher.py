"""Researcher plans, evidence and stages on the shared canonical runtime."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from researcher.schemas import ResearcherCandidate, ResearcherSnippet, identity_key_for_candidate
from shared.contracts import FAILURE_DECISIONS, LEGACY_REVIEW_DECISIONS, SAVE_DECISIONS
from shared.execution import CandidateExecutionEngine
from shared.runtime_state.store import RESEARCHER_AUTHOR_QUERY_KIND, RuntimeStateStore
from shared.schemas import OpusDecision


def _resolve_recruiter_id() -> int | None:
    try:
        from shared.recruiter_context import get_current_recruiter_id
        return get_current_recruiter_id()
    except Exception:
        return None


class ResearcherRuntimeStateBridge:
    def __init__(self, *, store: RuntimeStateStore, output_dir: str | Path,
                 brief_id: str, brief_name: str, brief_path: str | None = None):
        self.store = store
        self.output_dir = Path(output_dir)
        self.brief_id = brief_id
        self.brief_name = brief_name
        self.brief_path = brief_path
        self._execution_engine = CandidateExecutionEngine(
            store=store, output_dir=str(output_dir), brief_id=brief_id, source="researcher",
        )
        self.runtime = self._execution_engine.runtime

    def has_runtime_state(self) -> bool:
        return bool(self.store.get_latest_run(source="researcher", brief_id=self.brief_id)
                    or self.store.has_candidates(source="researcher", brief_id=self.brief_id))

    def start_or_resume_run(self, *, resume: bool, replan: bool = False,
                            limits: dict | None = None) -> int:
        from shared.brief_identity import compute_brief_identity

        latest = self.store.get_latest_run(source="researcher", brief_id=self.brief_id)
        identity = compute_brief_identity(self.brief_path) if self.brief_path else None
        if resume and not latest:
            raise ValueError("Resume requires an existing campaign")
        if resume and identity and latest["brief_content_hash"] != identity["brief_content_hash"] and not replan:
            raise ValueError("Brief hash changed; use --resume --replan")
        prior = self.store.get_run_resume_state(latest["id"]) if resume else {}
        frontier = prior.setdefault("frontier", {})
        frontier.setdefault("reference_date", date.today().isoformat())
        frontier.setdefault("plan_revision", 1)
        if replan and identity and latest and latest["brief_content_hash"] != identity["brief_content_hash"]:
            frontier["plan_revision"] += 1
            frontier["replan_required"] = True
        if limits is not None:
            if frontier.get("limits") and frontier["limits"] != limits:
                frontier.setdefault("limit_amendments", []).append({"before": frontier["limits"], "after": limits})
            frontier["limits"] = dict(limits)
        orphans = self.store.list_orphaned_attempts(source="researcher", brief_id=self.brief_id)
        self.store.reconcile_open_attempts(source="researcher", brief_id=self.brief_id)
        if latest and latest["status"] == "running":
            self.store.finish_run(latest["id"], "interrupted", stop_reason="worker_missing")
        run_id = self.store.start_run(
            source="researcher", brief_id=self.brief_id, output_dir=str(self.output_dir),
            mode="resume" if resume else "fresh", resume_state=prior,
            resumed_from_run_id=latest["id"] if resume else None,
            clone_work_units_from_run_id=latest["id"] if resume else None,
            recruiter_id=_resolve_recruiter_id(), backfill_legacy_candidate_hashes=False,
            **(identity or {}),
        )
        for orphan in orphans:
            self.store.record_event(run_id=run_id, event_type="researcher_unknown_request_outcome",
                                    payload={"attempt_id": orphan["id"], "stage": orphan["stage"], "cost": "reserved estimate retained"})
        if resume:
            for row in self.store.list_work_units(run_id):
                checkpoint = json.loads(row["checkpoint_json"])
                if checkpoint.get("request_started") and not checkpoint.get("received_page") and not checkpoint.get("raw_response"):
                    self.store.record_event(run_id=run_id, event_type="researcher_unknown_request_outcome", payload={
                        "query_id": row["source_unit_id"], "stage": "acquisition", "cost": "reserved estimate retained"})
        return run_id

    def campaign(self, run_id: int) -> dict:
        return self.store.get_run_resume_state(run_id).get("frontier", {})

    def save_campaign(self, run_id: int, campaign: dict) -> None:
        state = self.store.get_run_resume_state(run_id)
        state["frontier"] = campaign
        self.store.update_run_resume_state(run_id, state)

    def query_row(self, run_id: int, query_id) -> dict | None:
        return self.store.get_work_unit_by_source_id(run_id, kind=RESEARCHER_AUTHOR_QUERY_KIND, source_unit_id=str(query_id))

    def upsert_query_work_unit(self, *, run_id: int, query: dict, status: str = "queued",
                              ordering_index: int | None = None, cursor: str | None = None,
                              checkpoint: dict | None = None, metrics: dict | None = None,
                              **counters) -> None:
        row = self.query_row(run_id, query.get("id"))
        saved_checkpoint = json.loads(row["checkpoint_json"]) if row else {"cursor": "*"}
        saved_checkpoint.update(checkpoint or {})
        if cursor is not None:
            saved_checkpoint["cursor"] = cursor
        saved_metrics = json.loads(row["metrics_json"]) if row else {}
        saved_metrics.update(metrics or {})
        count_keys = ("result_count", "candidates_discovered", "facial_yes_count", "facial_no_count",
                      "facial_borderline_count", "saves_count", "rejected_count")
        saved_counts = {k: row[k] for k in count_keys} if row else {}
        saved_counts.update({k: v for k, v in counters.items() if k in count_keys})
        self.store.upsert_work_unit(
            run_id=run_id, source="researcher", brief_id=self.brief_id,
            kind=RESEARCHER_AUTHOR_QUERY_KIND, source_unit_id=str(query.get("id")),
            display_name=query.get("name", ""), ordering_index=ordering_index or (row["ordering_index"] if row else int(query.get("id") or 0)),
            status=status, payload=query, checkpoint=saved_checkpoint, metrics=saved_metrics,
            family_key=query.get("family") or "+".join(query.get("topic_concepts") or query.get("venue_filter") or ["researcher"]),
            novelty_bucket="adapted" if query.get("adapted_from_batch") else "initial",
            domain_lane="+".join(query.get("venue_filter") or query.get("topic_concepts") or []),
            counters=saved_counts,
        )

    def load_pending_queries(self, run_id: int) -> list[dict]:
        return [json.loads(row["payload_json"]) for row in self.store.list_work_units(run_id, kind=RESEARCHER_AUTHOR_QUERY_KIND)
                if row["status"] in {"queued", "in_progress", "error"}]

    def envelope(self, run_id: int, candidate: ResearcherCandidate, query_id=0, snippet=None):
        return self._execution_engine.envelope(
            source="researcher", brief_id=self.brief_id, run_id=run_id,
            work_unit_kind=RESEARCHER_AUTHOR_QUERY_KIND, work_unit_source_id=str(query_id),
            identity_key=identity_key_for_candidate(candidate), display_name=candidate.name,
            profile_url=candidate.profile_url, snippet=snippet,
            source_cursor={"query_id": query_id},
        )

    def candidate_row(self, key: str) -> dict | None:
        return self.store.get_candidate(source="researcher", brief_id=self.brief_id, identity_key=key)

    def candidate_rows(self) -> list[dict]:
        with self.store.connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM candidates WHERE source='researcher' AND brief_id=? ORDER BY id", (self.brief_id,),
            )]

    def candidate_record(self, key: str) -> tuple[ResearcherCandidate, int]:
        row = self.candidate_row(key)
        if not row:
            raise ValueError("Unknown researcher identity")
        with self.store.connect() as conn:
            events = conn.execute(
                "SELECT payload_json FROM events WHERE candidate_id=? AND event_type IN ('researcher_evidence','candidate_discovered') ORDER BY id DESC",
                (row["id"],),
            ).fetchall()
        for event in events:
            payload = json.loads(event[0])
            if payload.get("candidate_record"):
                candidate = ResearcherCandidate.from_dict(payload["candidate_record"])
                candidate.local_identity_key = key
                return candidate, payload.get("query_id", 0)
        payload = json.loads(row["terminal_payload_json"] or "{}")
        if payload.get("candidate_record"):
            candidate = ResearcherCandidate.from_dict(payload["candidate_record"])
            candidate.local_identity_key = key
            return candidate, payload.get("cursor", {}).get("query_id", 0)
        raise ValueError("Historical researcher record has no recoverable evidence")

    def record_candidate_discovery(self, *, run_id: int, query_id: int,
                                   candidate: ResearcherCandidate) -> int | None:
        from researcher.identity import reconcile_candidate
        prior = []
        for row in self.candidate_rows():
            try:
                prior.append(self.candidate_record(row["identity_key"])[0])
            except ValueError:
                continue  # A legacy row without evidence cannot establish an alias.
        original_key = identity_key_for_candidate(candidate)
        previous = reconcile_candidate(candidate, prior)
        key = identity_key_for_candidate(candidate)
        if not key:
            return None
        candidate.local_identity_key = key
        if previous and original_key != key:
            for identity_key in (original_key, key):
                row = self.candidate_row(identity_key)
                if row:
                    self.store.ensure_candidate(source="researcher", brief_id=self.brief_id,
                                                identity_key=identity_key, person_key="researcher:" + key)
            self.store.record_event(run_id=run_id, event_type="researcher_identity_linked",
                                    payload={"alias": original_key, "canonical": key, "references": candidate.identity_refs})
        payload = {"candidate_record": candidate.to_dict(), "query_id": query_id}
        existing = self.candidate_row(key)
        if existing:
            candidate_id = existing["id"]
            self.store.record_event(run_id=run_id, candidate_id=candidate_id,
                                    event_type="researcher_evidence", payload=payload)
        else:
            candidate_id = self.runtime.record_discovery(self.envelope(run_id, candidate, query_id), payload=payload)
        return candidate_id

    def start_facial(self, *, run_id: int, snippet: ResearcherSnippet) -> int:
        key = self._snippet_identity_key(snippet)
        candidate, query_id = self.candidate_record(key)
        envelope = self.envelope(run_id, candidate, query_id, snippet)
        row = self.candidate_row(key)
        if row["current_lifecycle_state"] != "snippet_extracted":
            self.runtime.record_snippet_extracted(envelope, payload={"snippet": snippet.to_dict(), "candidate_record": candidate.to_dict()})
        return self.runtime.start_stage(envelope, stage="facial", payload={"candidate_record": candidate.to_dict()})

    def record_facial_decision(self, *, run_id: int, snippet: ResearcherSnippet,
                               decision: OpusDecision, attempt_id: int | None = None) -> None:
        candidate, query_id = self.candidate_record(self._snippet_identity_key(snippet))
        if attempt_id is None:
            attempt_id = self.start_facial(run_id=run_id, snippet=snippet)
        self._finish(run_id, candidate, query_id, "facial", decision, attempt_id, snippet=snippet)

    def start_full(self, *, run_id: int, candidate: ResearcherCandidate) -> int:
        _, query_id = self.candidate_record(identity_key_for_candidate(candidate))
        return self.runtime.start_stage(self.envelope(run_id, candidate, query_id), stage="full",
                                        payload={"candidate_record": candidate.to_dict()})

    def record_full_decision(self, *, run_id: int, candidate: ResearcherCandidate,
                             decision: OpusDecision, needs_identity_confirmation=False,
                             identity_review_note="", attempt_id: int | None = None) -> None:
        _, query_id = self.candidate_record(identity_key_for_candidate(candidate))
        if attempt_id is None:
            attempt_id = self.start_full(run_id=run_id, candidate=candidate)
        extra = {}
        if needs_identity_confirmation:
            extra = {"needs_identity_confirmation": True, "identity_review_note": identity_review_note}
        self._finish(run_id, candidate, query_id, "full", decision, attempt_id, extra=extra)

    def _finish(self, run_id, candidate, query_id, stage, decision, attempt_id, *, snippet=None, extra=None):
        envelope = self.envelope(run_id, candidate, query_id, snippet)
        payload = {"candidate_record": candidate.to_dict(), **(extra or {})}
        payload["judgment_brief_hash"] = (self.store.get_run(run_id) or {}).get("brief_content_hash")
        if decision.decision in FAILURE_DECISIONS:
            self.runtime.finish_stage_failure(attempt_id=attempt_id, envelope=envelope, stage=stage,
                                              error_or_failure_decision=decision, extra_payload=payload)
        else:
            self.runtime.finish_stage_success(attempt_id=attempt_id, envelope=envelope, stage=stage,
                                              decision=decision, extra_payload=payload)

    def pending_candidates(self) -> list[dict]:
        return [row for row in self.candidate_rows() if not (row.get("person_key") and self.store.is_person_key_blocked(brief_id=self.brief_id, person_key=row["person_key"])) and not self.store.is_dedup_blocked(
            source="researcher", brief_id=self.brief_id, identity_key=row["identity_key"],
        )]

    def defer_open_attempts(self, run_id: int) -> None:
        """Budget refusal leaves an eligible obligation without consuming a model retry."""
        for attempt in self.store.list_orphaned_attempts(source="researcher", brief_id=self.brief_id):
            if attempt["run_id"] != run_id:
                continue
            candidate, query_id = self.candidate_record(attempt["identity_key"])
            self.runtime.finish_stage_failure(
                attempt_id=attempt["id"], envelope=self.envelope(run_id, candidate, query_id), stage=attempt["stage"],
                error_or_failure_decision=RuntimeError("Dispatch deferred by campaign budget"),
                extra_payload={"failure_kind_override": "budget_deferred", "force_retryable": True},
            )

    def persist_page(self, *, run_id: int, query: dict, page: dict, after_candidates=None) -> None:
        """Write every returned person before advancing the query cursor; replay is idempotent."""
        row = self.query_row(run_id, query["id"])
        previous = json.loads(row["checkpoint_json"]) if row else {}
        known = set((previous.get("received_page") or {}).get("known_identity_keys", [])) if previous.get("received_page") else {r["identity_key"] for r in self.candidate_rows()}
        if not previous.get("received_page"):
            cached = {**page, "candidates": [c.to_dict() for c in page["candidates"]], "known_identity_keys": sorted(known)}
            self.upsert_query_work_unit(run_id=run_id, query=query, status="in_progress", checkpoint={"received_page": cached})
            self.store.record_event(run_id=run_id, event_type="researcher_page_received", payload={
                "query_id": query["id"], "results_returned": page.get("results_returned", len(page["candidates"]))})
        duplicates = 0
        for candidate in page["candidates"]:
            self.record_candidate_discovery(run_id=run_id, query_id=query["id"], candidate=candidate)
            duplicates += identity_key_for_candidate(candidate) in known
        if after_candidates:
            after_candidates()
        row = self.query_row(run_id, query["id"])
        checkpoint = json.loads(row["checkpoint_json"]) if row else {}
        checkpoint.update(
            cursor=page.get("next_cursor"), exhausted=page.get("exhausted", False),
            pages_fetched=checkpoint.get("pages_fetched", 0) + 1,
            results_returned=checkpoint.get("results_returned", 0) + page.get("results_returned", len(page["candidates"])),
            coverage=page.get("coverage", "bounded request"),
            raw_response=None, received_page=None, request_started=False,
            duplicates=checkpoint.get("duplicates", 0) + duplicates,
            matched_works=checkpoint.get("matched_works", 0) + page.get("matched_works", 0),
        )
        self.upsert_query_work_unit(run_id=run_id, query=query,
                                    status="done" if page.get("exhausted") or not page.get("next_cursor") else "in_progress",
                                    checkpoint=checkpoint)

    def revisit(self, *, run_id: int, key: str) -> None:
        row = self.candidate_row(key)
        if not row or row["terminal_decision"] not in LEGACY_REVIEW_DECISIONS or row.get("user_status"):
            raise ValueError("Only an existing non-save, non-human-decided review may be revisited")
        self.store.record_event(run_id=run_id, candidate_id=row["id"], event_type="researcher_revisit",
                                payload={"previous_judgment": json.loads(row["terminal_payload_json"]),
                                         "brief_hash": row["brief_content_hash"]})
        self.store.clear_candidate_terminal_state(source="researcher", brief_id=self.brief_id, identity_key=key)
        candidate, query_id = self.candidate_record(key)
        candidate.evidence_status = "refresh_required"
        self.record_candidate_discovery(run_id=run_id, query_id=query_id, candidate=candidate)

    @staticmethod
    def _snippet_identity_key(snippet: ResearcherSnippet) -> str:
        if snippet.identity_key:
            return snippet.identity_key
        from urllib.parse import urlsplit
        parsed = urlsplit(snippet.profile_url)
        if parsed.hostname == "orcid.org":
            return "orcid:" + parsed.path.strip("/")
        if parsed.hostname == "openalex.org":
            return "openalex:" + parsed.path.strip("/")
        return ""
