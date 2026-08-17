"""Legacy LinkedIn runtime import helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection

from shared.runtime_state.store import RuntimeStateStore
from shared.schemas import CandidateProfileSummary, CandidateSnippet, OpusDecision, Progress, SearchString
from shared.storage import read_jsonl


@dataclass(frozen=True)
class LinkedInLegacyImportContext:
    store: RuntimeStateStore
    run_id: int
    brief_id: str
    progress_path: Path
    history_path: Path
    search_memory_path: Path
    snippets_path: Path
    facial_path: Path
    profiles_path: Path
    final_path: Path
    read_legacy_progress: Callable[[], Progress | None]
    sync_progress: Callable[..., None]
    search_string_for_id: Callable[[Progress | None, int], SearchString]
    record_snippet_extracted: Callable[..., int | None]
    start_stage_attempt: Callable[..., int | None]
    finish_failure_decision: Callable[..., None]
    finish_stage_success: Callable[..., None]
    save_decisions: Collection[str]


def import_linkedin_legacy_state(context: LinkedInLegacyImportContext) -> None:
    with context.store.connect() as conn:
        imported = conn.execute(
            """
            SELECT 1
            FROM events
            WHERE run_id = ? AND event_type = 'linkedin_legacy_import_complete'
            LIMIT 1
            """,
            (context.run_id,),
        ).fetchone()
        if imported:
            return

    progress = context.read_legacy_progress()
    if progress:
        # Hydration write: the imported progress.json may carry statuses this
        # version does not know; they must be replayed, not rejected.
        context.sync_progress(context.run_id, progress, validate_status=False)

    snippets_by_url = _load_snippets_by_url(context.snippets_path)
    facial_by_url = _load_single_record_by_url(context.facial_path)
    profiles_by_url = _load_single_record_by_url(context.profiles_path)
    finals_by_url = _load_single_record_by_url(context.final_path)

    _replay_snippets_and_judgments(
        context=context,
        progress=progress,
        snippets_by_url=snippets_by_url,
        facial_by_url=facial_by_url,
        profiles_by_url=profiles_by_url,
        finals_by_url=finals_by_url,
    )
    _replay_history(context=context)

    context.store.record_event(
        run_id=context.run_id,
        event_type="linkedin_legacy_import_complete",
        payload={
            "progress_exists": context.progress_path.exists(),
            "history_exists": context.history_path.exists(),
            "search_memory_exists": context.search_memory_path.exists(),
        },
    )


def _load_snippets_by_url(path: Path) -> dict[str, list[dict[str, Any]]]:
    snippets_by_url: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return snippets_by_url
    for record in read_jsonl(path):
        url = record.get("profile_url", "")
        if not url:
            continue
        snippets_by_url.setdefault(url, []).append(record)
    return snippets_by_url


def _load_single_record_by_url(path: Path) -> dict[str, dict[str, Any]]:
    records_by_url: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records_by_url
    for record in read_jsonl(path):
        url = record.get("profile_url", "")
        if url:
            records_by_url[url] = record
    return records_by_url


def _replay_snippets_and_judgments(
    *,
    context: LinkedInLegacyImportContext,
    progress: Progress | None,
    snippets_by_url: dict[str, list[dict[str, Any]]],
    facial_by_url: dict[str, dict[str, Any]],
    profiles_by_url: dict[str, dict[str, Any]],
    finals_by_url: dict[str, dict[str, Any]],
) -> None:
    for url, snippet_records in snippets_by_url.items():
        for record in snippet_records:
            snippet = CandidateSnippet.from_dict(record)
            context.record_snippet_extracted(
                run_id=context.run_id,
                search_string=context.search_string_for_id(progress, snippet.source_string_id),
                snippet=snippet,
            )

        facial_record = facial_by_url.get(url)
        if facial_record:
            snippet = CandidateSnippet.from_dict(snippet_records[-1])
            search_string = context.search_string_for_id(progress, snippet.source_string_id)
            attempt_id = context.start_stage_attempt(
                run_id=context.run_id,
                search_string=search_string,
                snippet=snippet,
                stage="facial",
            )
            decision = OpusDecision.from_dict(facial_record)
            if decision.decision in {"PARSE_FAILURE", "JUDGMENT_FAILURE"}:
                context.finish_failure_decision(
                    run_id=context.run_id,
                    attempt_id=attempt_id,
                    snippet=snippet,
                    decision=decision,
                )
            else:
                context.finish_stage_success(
                    run_id=context.run_id,
                    attempt_id=attempt_id,
                    stage="facial",
                    snippet=snippet,
                    decision=decision,
                )

        final_record = finals_by_url.get(url)
        if final_record:
            snippet = CandidateSnippet.from_dict(snippet_records[-1])
            search_string = context.search_string_for_id(progress, snippet.source_string_id)
            profile_summary = None
            if url in profiles_by_url:
                profile_summary = CandidateProfileSummary.from_dict(profiles_by_url[url])
            candidate = context.store.get_candidate(
                source="linkedin",
                brief_id=context.brief_id,
                identity_key=url,
            )
            if candidate and candidate["current_lifecycle_state"] not in {"facial_terminal", "full_started", "full_terminal"}:
                context.store.set_candidate_state(
                    run_id=context.run_id,
                    source="linkedin",
                    brief_id=context.brief_id,
                    identity_key=url,
                    new_state="facial_terminal",
                    terminal_decision="FACIAL_YES",
                    terminal_payload={"source_string_id": snippet.source_string_id},
                )
            attempt_id = context.start_stage_attempt(
                run_id=context.run_id,
                search_string=search_string,
                snippet=snippet,
                stage="full",
                payload={"profile_summary": profile_summary.to_dict() if profile_summary else {}},
            )
            decision = OpusDecision.from_dict(final_record)
            if decision.decision in {"PARSE_FAILURE", "JUDGMENT_FAILURE"}:
                context.finish_failure_decision(
                    run_id=context.run_id,
                    attempt_id=attempt_id,
                    snippet=snippet,
                    decision=decision,
                    payload={"profile_summary": profile_summary.to_dict() if profile_summary else {}},
                )
            else:
                context.finish_stage_success(
                    run_id=context.run_id,
                    attempt_id=attempt_id,
                    stage="full",
                    snippet=snippet,
                    decision=decision,
                    profile_summary=profile_summary,
                )


def _replay_history(*, context: LinkedInLegacyImportContext) -> None:
    if not context.history_path.exists():
        return
    for record in read_jsonl(context.history_path):
        url = record.get("profile_url", "")
        outcome = record.get("outcome", "")
        if not url or not outcome:
            continue
        if not context.store.get_candidate(source="linkedin", brief_id=context.brief_id, identity_key=url):
            context.store.record_candidate_discovery(
                run_id=context.run_id,
                work_unit_id=None,
                source="linkedin",
                brief_id=context.brief_id,
                identity_key=url,
                display_name=record.get("candidate_name", ""),
                profile_url=url,
                payload={"legacy_import": True},
            )
        candidate = context.store.get_candidate(
            source="linkedin",
            brief_id=context.brief_id,
            identity_key=url,
        )
        current_state = candidate["current_lifecycle_state"] if candidate else "discovered"
        if current_state == "discovered":
            context.store.set_candidate_state(
                run_id=context.run_id,
                source="linkedin",
                brief_id=context.brief_id,
                identity_key=url,
                new_state="snippet_extracted",
            )
            current_state = "snippet_extracted"
        if current_state == "snippet_extracted":
            context.store.set_candidate_state(
                run_id=context.run_id,
                source="linkedin",
                brief_id=context.brief_id,
                identity_key=url,
                new_state="facial_started",
            )
            current_state = "facial_started"
        if outcome in context.save_decisions or outcome == "REJECT":
            if current_state == "facial_started":
                context.store.set_candidate_state(
                    run_id=context.run_id,
                    source="linkedin",
                    brief_id=context.brief_id,
                    identity_key=url,
                    new_state="facial_terminal",
                    terminal_decision="FACIAL_YES",
                    terminal_payload={
                        "legacy_import": True,
                        "source_string_id": record.get("source_string_id"),
                    },
                )
                current_state = "facial_terminal"
            if current_state == "facial_terminal":
                context.store.set_candidate_state(
                    run_id=context.run_id,
                    source="linkedin",
                    brief_id=context.brief_id,
                    identity_key=url,
                    new_state="full_started",
                )
            state = "full_terminal"
        else:
            state = "facial_terminal"
        context.store.set_candidate_state(
            run_id=context.run_id,
            source="linkedin",
            brief_id=context.brief_id,
            identity_key=url,
            new_state=state,
            terminal_decision=outcome,
            terminal_payload={
                "confidence": record.get("confidence", 0.0),
                "source_string_id": record.get("source_string_id"),
                "timestamp": record.get("timestamp"),
            },
        )
