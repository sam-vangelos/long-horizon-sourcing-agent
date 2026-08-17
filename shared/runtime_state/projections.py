"""Compatibility projections derived from runtime_state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from github.schemas import GitHubProgress
from shared.schemas import Progress, SearchString
from shared.search_memory import update_search_memory

from .store import GITHUB_QUERY_KIND, LINKEDIN_STRING_KIND, RuntimeStateStore


def project_github_progress(store: RuntimeStateStore, run_id: int) -> GitHubProgress:
    return store.load_github_progress(run_id)


def write_github_progress_projection(store: RuntimeStateStore, run_id: int, path: str | Path) -> GitHubProgress:
    progress = project_github_progress(store, run_id)
    _write_json_atomic(path, progress.to_dict())
    return progress


def project_github_snippets(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    run_id: int | None = None,
) -> list[dict]:
    return _project_attempt_payloads(
        store,
        source="github",
        brief_id=brief_id,
        run_id=run_id,
        stage="facial",
        payload_key="snippet",
    )


def project_github_facial_judgments(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    run_id: int | None = None,
) -> list[dict]:
    return _project_attempt_payloads(
        store,
        source="github",
        brief_id=brief_id,
        run_id=run_id,
        stage="facial",
        payload_key="facial_decision",
    )


def project_github_profile_summaries(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    run_id: int | None = None,
) -> list[dict]:
    return _project_attempt_payloads(
        store,
        source="github",
        brief_id=brief_id,
        run_id=run_id,
        stage="full",
        payload_key="profile_summary",
    )


def project_github_final_judgments(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    run_id: int | None = None,
) -> list[dict]:
    return _project_attempt_payloads(
        store,
        source="github",
        brief_id=brief_id,
        run_id=run_id,
        stage="full",
        payload_key="full_decision",
    )


def write_github_stage_projections(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    output_dir: str | Path,
    run_id: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    _write_jsonl_atomic(output_dir / "snippets.jsonl", project_github_snippets(store, brief_id=brief_id, run_id=run_id))
    _write_jsonl_atomic(output_dir / "facial_judgments.jsonl", project_github_facial_judgments(store, brief_id=brief_id, run_id=run_id))
    _write_jsonl_atomic(output_dir / "profile_summaries.jsonl", project_github_profile_summaries(store, brief_id=brief_id, run_id=run_id))
    _write_jsonl_atomic(output_dir / "final_judgments.jsonl", project_github_final_judgments(store, brief_id=brief_id, run_id=run_id))


def project_linkedin_progress(store: RuntimeStateStore, run_id: int) -> Progress:
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"run not found: {run_id}")
    resume_state = _json_loads(run.get("resume_state_json"))
    work_units = store.list_work_units(run_id, kind=LINKEDIN_STRING_KIND)
    strings = []
    for row in work_units:
        payload = _json_loads(row["payload_json"])
        metrics = _json_loads(row["metrics_json"] if "metrics_json" in row.keys() else None)
        checkpoint = _json_loads(row["checkpoint_json"])
        string_id = _coerce_projection_int(
            payload.get("id"),
            fallback=_coerce_projection_int(row["source_unit_id"]),
        )
        string_name = str(
            payload.get("name") or row["display_name"] or f"string-{string_id}"
        ).strip()
        full_funnel = _project_linkedin_full_funnel_counts(
            row=row,
            payload=payload,
            metrics=metrics,
        )
        payload.update(
            {
                "id": string_id,
                "name": string_name,
                "status": row["status"],
                "result_count": row["result_count"],
                "pages_reviewed": metrics.get("pages_reviewed", checkpoint.get("pages_reviewed", payload.get("pages_reviewed", 0))),
                "facial_yes_count": row["facial_yes_count"],
                "facial_no_count": row["facial_no_count"],
                "facial_borderline_count": row["facial_borderline_count"],
                **full_funnel,
                "candidates_count": row["candidates_discovered"],
                "duplicates_count": metrics.get("duplicates_count", checkpoint.get("duplicates_count", payload.get("duplicates_count", 0))),
                "notes": row["notes"] or payload.get("notes", ""),
                "family_key": row["family_key"],
                "novelty_bucket": row["novelty_bucket"],
                "domain_lane": row["domain_lane"],
            }
        )
        strings.append(SearchString.from_dict(payload))
    return Progress(
        brief_name=resume_state.get("brief_name", run["brief_id"]),
        strings=strings,
        candidates_saved=int(resume_state.get("candidates_saved", 0)),
        candidates_rejected=int(resume_state.get("candidates_rejected", 0)),
        current_string_id=resume_state.get("current_string_id"),
        current_page=int(resume_state.get("current_page", 0)),
        pending_block_name=resume_state.get("pending_block_name", ""),
        pending_block_string_ids=list(resume_state.get("pending_block_string_ids", [])),
        pending_block_ready=bool(resume_state.get("pending_block_ready", False)),
        pivot_count=int(resume_state.get("pivot_count", 0)),
    )


def write_linkedin_progress_projection(store: RuntimeStateStore, run_id: int, path: str | Path) -> Progress:
    progress = project_linkedin_progress(store, run_id)
    _write_json_atomic(path, progress.to_dict())
    return progress


def project_linkedin_candidate_history(store: RuntimeStateStore, *, brief_id: str) -> list[dict]:
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT identity_key, display_name, profile_url, terminal_decision, terminal_payload_json, last_seen_at
            FROM candidates
            WHERE source = 'linkedin' AND brief_id = ? AND terminal_decision IS NOT NULL
            ORDER BY last_seen_at ASC, id ASC
            """,
            (brief_id,),
        ).fetchall()
    history = []
    for row in rows:
        payload = _json_loads(row["terminal_payload_json"])
        history.append(
            {
                "profile_url": row["profile_url"] or row["identity_key"],
                "candidate_name": row["display_name"],
                # Compatibility projections must not collapse the canonical
                # three-way facial decision. Resume code handles YES and
                # BORDERLINE as separate full-review-eligible outcomes.
                "outcome": row["terminal_decision"],
                "confidence": payload.get("confidence", 0.0),
                "source_string_id": payload.get("source_string_id"),
                "timestamp": payload.get("timestamp", row["last_seen_at"]),
            }
        )
    return history


def write_linkedin_candidate_history_projection(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    path: str | Path,
) -> list[dict]:
    history = project_linkedin_candidate_history(store, brief_id=brief_id)
    _write_jsonl_atomic(path, history)
    return history


def project_linkedin_search_memory(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    run_id: int | None = None,
) -> dict:
    memory: dict = {}
    with store.connect() as conn:
        sql = """
            SELECT run_id, source_unit_id, display_name, payload_json, checkpoint_json, family_key, novelty_bucket, domain_lane, result_count,
                   candidates_discovered, facial_yes_count, facial_no_count, facial_borderline_count,
                   saves_count, rejected_count, notes, metrics_json
            FROM work_units
            WHERE source = 'linkedin' AND brief_id = ? AND kind = ? AND status = 'done'
        """
        params: list[Any] = [brief_id, LINKEDIN_STRING_KIND]
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        # P3.2: run-ordered so epoch transitions (brief revisions) archive a
        # family's counters exactly once per revision, in run order.
        sql += " ORDER BY run_id ASC, ordering_index ASC, id ASC"
        rows = conn.execute(sql, tuple(params)).fetchall()
        # P3.2: per-run brief content hash = the family-memory epoch.
        run_epochs: dict[int, str] = {
            int(r["id"]): str(r["brief_content_hash"] or "")
            for r in conn.execute(
                "SELECT id, brief_content_hash FROM runs WHERE source = 'linkedin' AND brief_id = ?",
                (brief_id,),
            ).fetchall()
        }
    strings = []
    for row in rows:
        payload = _json_loads(row["payload_json"])
        checkpoint = _json_loads(row["checkpoint_json"])
        metrics = _json_loads(row["metrics_json"] if "metrics_json" in row.keys() else None)
        string_id = _coerce_projection_int(
            payload.get("id"),
            fallback=_coerce_projection_int(row["source_unit_id"]),
        )
        string_name = str(
            payload.get("name") or row["display_name"] or f"string-{string_id}"
        ).strip()
        full_funnel = _project_linkedin_full_funnel_counts(
            row=row,
            payload=payload,
            metrics=metrics,
        )
        strings.append(
            SearchString.from_dict(
                {
                    "id": string_id,
                    "name": string_name,
                    "boolean": payload.get("boolean", ""),
                    "status": "done",
                    "result_count": row["result_count"],
                    "pages_reviewed": metrics.get("pages_reviewed", checkpoint.get("pages_reviewed", payload.get("pages_reviewed", 0))),
                    "saves": list(payload.get("saves", [])),
                    "notes": row["notes"] or payload.get("notes", ""),
                    "block": payload.get("block", ""),
                    "subblock": payload.get("subblock", ""),
                    "string_type": payload.get("string_type", ""),
                    "facial_yes_count": row["facial_yes_count"],
                    "facial_no_count": row["facial_no_count"],
                    "facial_borderline_count": row["facial_borderline_count"],
                    **full_funnel,
                    "candidates_count": row["candidates_discovered"],
                    "duplicates_count": metrics.get("duplicates_count", checkpoint.get("duplicates_count", payload.get("duplicates_count", 0))),
                    "suppressed_prior_session_count": metrics.get(
                        "suppressed_prior_session_count",
                        checkpoint.get(
                            "suppressed_prior_session_count",
                            payload.get("suppressed_prior_session_count", 0),
                        ),
                    ),
                    "phase": payload.get("phase", "scout"),
                    "original_boolean": payload.get("original_boolean", ""),
                    "refinement_stack": payload.get("refinement_stack", []),
                    "family_key": row["family_key"],
                    "novelty_bucket": row["novelty_bucket"],
                    "domain_lane": row["domain_lane"],
                    "retrieval_recipe": payload.get("retrieval_recipe", {}),
                    "retrieval_hypothesis_ids": payload.get("retrieval_hypothesis_ids", []),
                }
            )
        )
        # P3.2: attach the string's brief epoch (its run's content hash) so
        # update_search_memory can archive counters across brief revisions.
        setattr(strings[-1], "brief_epoch", run_epochs.get(int(row["run_id"]), ""))
        # P7 Stage C: attach the run id the same way so update_search_memory
        # can compute run-over-run family_key stability (Jaccard overlap).
        setattr(strings[-1], "run_id", int(row["run_id"]))
    if strings:
        memory = update_search_memory(memory, brief_id, strings)
    return memory


def write_linkedin_search_memory_projection(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    path: str | Path,
    run_id: int | None = None,
) -> dict:
    memory = project_linkedin_search_memory(store, brief_id=brief_id, run_id=run_id)
    _write_json_atomic(path, memory)
    return memory


def project_linkedin_snippets(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    run_id: int | None = None,
) -> list[dict]:
    return _project_attempt_payloads(
        store,
        source="linkedin",
        brief_id=brief_id,
        run_id=run_id,
        stage="snippet",
        payload_key="snippet",
    )


def project_linkedin_facial_judgments(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    run_id: int | None = None,
) -> list[dict]:
    return _project_attempt_payloads(
        store,
        source="linkedin",
        brief_id=brief_id,
        run_id=run_id,
        stage="facial",
        payload_key="facial_decision",
    )


def project_linkedin_profile_summaries(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    run_id: int | None = None,
) -> list[dict]:
    return _project_attempt_payloads(
        store,
        source="linkedin",
        brief_id=brief_id,
        run_id=run_id,
        stage="full",
        payload_key="profile_summary",
    )


def project_linkedin_final_judgments(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    run_id: int | None = None,
) -> list[dict]:
    # Canonical SQLite stores full-stage decisions under "full_decision"
    # (per SharedExecutionRuntime._build_stage_payload, which uses
    # f"{stage}_decision"). Mirror the GitHub equivalent above.
    return _project_attempt_payloads(
        store,
        source="linkedin",
        brief_id=brief_id,
        run_id=run_id,
        stage="full",
        payload_key="full_decision",
    )


def write_linkedin_stage_projections(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    output_dir: str | Path,
    run_id: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    _write_jsonl_atomic(output_dir / "snippets.jsonl", project_linkedin_snippets(store, brief_id=brief_id, run_id=run_id))
    _write_jsonl_atomic(output_dir / "facial_judgments.jsonl", project_linkedin_facial_judgments(store, brief_id=brief_id, run_id=run_id))
    _write_jsonl_atomic(output_dir / "profile_summaries.jsonl", project_linkedin_profile_summaries(store, brief_id=brief_id, run_id=run_id))
    _write_jsonl_atomic(output_dir / "final_judgments.jsonl", project_linkedin_final_judgments(store, brief_id=brief_id, run_id=run_id))


def _write_json_atomic(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _write_jsonl_atomic(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    tmp.replace(path)


def _json_loads(raw: str | None) -> Any:
    if not raw:
        return {}
    return json.loads(raw)


def _coerce_projection_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _project_linkedin_full_funnel_counts(
    *,
    row: Any,
    payload: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, int]:
    """Read settled full-profile counts from canonical SQLite state.

    New work units store the semantic funnel in ``metrics_json``. Older rows
    predate those keys, so their durable payload/column values provide a
    conservative compatibility fallback: physical saves imply outreach and
    ``rejected_count`` implies a full rejection. Explicit metric values,
    including zero, always win over those fallbacks.
    """

    def count(metric_key: str, payload_key: str, fallback: int = 0) -> int:
        if metric_key in metrics:
            raw = metrics[metric_key]
        elif payload_key in payload:
            raw = payload[payload_key]
        else:
            raw = fallback
        return max(0, _coerce_projection_int(raw, fallback=max(0, fallback)))

    outreach = count(
        "full_outreach",
        "full_outreach_count",
        fallback=_coerce_projection_int(row["saves_count"]),
    )
    review = count("full_review", "full_review_count")
    reject = count(
        "full_reject",
        "full_reject_count",
        fallback=_coerce_projection_int(row["rejected_count"]),
    )
    reviewed = count(
        "full_reviewed",
        "full_reviewed_count",
        fallback=outreach + review + reject,
    )
    # A corrupt or legacy undercount must not claim fewer settled reviews
    # than the mutually exclusive terminal buckets it contains.
    reviewed = max(reviewed, outreach + review + reject)
    return {
        "full_reviewed_count": reviewed,
        "full_outreach_count": outreach,
        "full_review_count": review,
        "full_reject_count": reject,
    }


def _project_attempt_payloads(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    run_id: int | None,
    stage: str,
    payload_key: str,
) -> list[dict]:
    with store.connect() as conn:
        sql = """
            SELECT ca.payload_json
            FROM candidate_attempts ca
            JOIN candidates c ON c.id = ca.candidate_id
            WHERE c.source = ? AND c.brief_id = ? AND ca.stage = ?
        """
        params: list[Any] = [source, brief_id, stage]
        if run_id is not None:
            sql += " AND ca.run_id = ?"
            params.append(run_id)
        sql += " ORDER BY ca.started_at ASC, ca.id ASC"
        rows = conn.execute(sql, tuple(params)).fetchall()
    projected: list[dict] = []
    for row in rows:
        payload = _json_loads(row["payload_json"])
        record = payload.get(payload_key)
        if record:
            projected.append(record)
    return projected
