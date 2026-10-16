"""Native researcher dossiers and projections derived from committed SQLite state."""

from __future__ import annotations

import html
import json
from pathlib import Path

from researcher.budget import cost_summary
from researcher.judgment_templates import evidence_packet
from researcher.schemas import ResearcherCandidate
from shared.contracts import SAVE_DECISIONS, LEGACY_REVIEW_DECISIONS
from shared.runtime_state.projections import _write_json_atomic, _write_jsonl_atomic, _project_attempt_payloads


def build_report(bridge, run_id: int) -> dict:
    run = bridge.store.get_run(run_id)
    campaign = bridge.campaign(run_id)
    snapshot = json.loads(run["brief_snapshot_json"] or "{}")
    groups = {key: [] for key in ("accepted", "previously_saved", "review", "rejected", "failed", "pending", "linked_observations")}
    pending = {row["id"] for row in bridge.pending_candidates()}
    rows = bridge.candidate_rows()
    for row in rows:
        payload = json.loads(row["terminal_payload_json"] or "{}")
        decision = row["terminal_decision"]
        provenance = payload.get("judgment_brief_hash")
        candidate = payload.get("candidate_record")
        if not candidate:
            try:
                candidate = bridge.candidate_record(row["identity_key"])[0].to_dict()
            except ValueError:
                candidate = {"name": row["display_name"], "profile_url": row["profile_url"], "payload_version": 1}
        detail = payload.get("full_decision") or payload.get("facial_decision") or {}
        record = {"identity_key": row["identity_key"], "person_key": row.get("person_key"), "candidate": candidate,
                  "decision": decision, "rationale": detail.get("rationale", ""), "terminal_payload": payload,
                  "judgment_brief_hash": provenance, "lifecycle": row["current_lifecycle_state"]}
        if decision in SAVE_DECISIONS:
            group = "accepted" if provenance and provenance == run["brief_content_hash"] else "previously_saved"
        elif decision in LEGACY_REVIEW_DECISIONS:
            group = "review"
        elif decision in {"REJECT", "FACIAL_NO"}:
            group = "rejected"
        elif row["current_lifecycle_state"].startswith("failed"):
            group = "failed"
            with bridge.store.connect() as conn:
                attempt = conn.execute("SELECT failure_kind, failure_reason FROM candidate_attempts WHERE candidate_id=? ORDER BY id DESC LIMIT 1", (row["id"],)).fetchone()
            if attempt:
                record["failure"] = dict(attempt)
                if attempt["failure_kind"] == "budget_deferred":
                    group = "pending"
        else:
            group = "pending" if row["id"] in pending else "linked_observations"
        groups[group].append(record)
    queries = []
    for row in bridge.store.list_work_units(run_id):
        query = json.loads(row["payload_json"])
        checkpoint = json.loads(row["checkpoint_json"])
        checkpoint.pop("raw_response", None)
        queries.append({"query": query, "status": row["status"], "checkpoint": checkpoint,
                        "metrics": json.loads(row["metrics_json"])})
    counts = {key: len(value) for key, value in groups.items()}
    counts.update(discovered=len({r.get("person_key") or r["identity_key"] for r in rows}), observations=len(rows),
                  evaluated=sum(counts[k] for k in ("accepted", "previously_saved", "review", "rejected")),
                  queries_failed=sum(q["status"] == "error" for q in queries),
                  queries_pending=sum(q["status"] in {"queued", "in_progress", "error"} for q in queries),
                  api_results=sum(q["checkpoint"].get("results_returned", 0) for q in queries))
    return {"schema_version": 2, "source": "researcher", "run_metadata": {
        "run_id": run_id, "role_title": snapshot.get("role_title") or bridge.brief_name,
        "brief_name": bridge.brief_name, "brief_content_hash": run["brief_content_hash"],
        "brief_version": snapshot.get("version", ""), "plan_revision": campaign.get("plan_revision"),
        "role_mode": snapshot.get("source_config", {}).get("researcher", {}).get("role_mode"),
        "started_at": run["started_at"], "ended_at": run["ended_at"], "status": run["status"], "stop_reason": run["stop_reason"],
    }, "metrics_summary": {"candidates_evaluated": counts["evaluated"], "candidates_saved": counts["accepted"]},
        "counts": counts, "limits": campaign.get("limits", {}), "cost": cost_summary(campaign.get("cost_receipts", [])),
        "queries": queries, "candidates": groups}


def _text(value) -> str:
    return html.escape(str(value or ""), quote=False).replace("[", "\\[").replace("]", "\\]")


def render_report(report: dict) -> str:
    metadata, counts, cost = report["run_metadata"], report["counts"], report["cost"]
    lines = [f"# Researcher slate: {_text(metadata['role_title'])}", "",
             f"Role: {_text(metadata['role_mode'])}. Session {metadata['run_id']}: {_text(metadata['status'])}; stop: {_text(metadata['stop_reason'])}.",
             f"Brief: {_text(metadata['brief_content_hash'])}; revision {metadata['plan_revision']}.", "",
             f"Accepted for this brief: {counts['accepted']}; REVIEW: {counts['review']}; rejected: {counts['rejected']}; failed candidates: {counts['failed']}.",
             f"Previously saved under older/unknown briefs: {counts['previously_saved']}. Pending candidates: {counts['pending']}; pending queries: {counts['queries_pending']}; failed queries: {counts['queries_failed']}.",
             f"Discovered people: {counts['discovered']}; candidate observations: {counts['observations']}; API results: {counts['api_results']}.",
             f"Usage-priced spend: ${cost['measured_usd']:.6f}; reserved/estimated spend: ${cost['estimated_usd']:.6f}; usage unavailable for {cost['unavailable_calls']} calls. Provider invoices may differ.",
             f"Campaign/session limits: {_text(json.dumps(report['limits'], sort_keys=True))}.", "", "## Search coverage", ""]
    for query in report["queries"]:
        q, checkpoint = query["query"], query["checkpoint"]
        lines.append(f"- {_text(q.get('intent') or q.get('name'))}: {_text(q.get('provider'))}/{_text(q.get('route'))}, {_text(query['status'])}; {_text(checkpoint.get('coverage', 'not acquired'))}; cap: {_text(checkpoint.get('cap_reason', 'none'))}.")
    labels = {"accepted": "Accepted recommendations", "previously_saved": "Previously saved under older or unknown briefs", "review": "REVIEW candidates", "rejected": "Rejected candidates", "failed": "Runtime failures", "pending": "Pending candidates", "linked_observations": "Linked observations"}
    for group, label in labels.items():
        lines += ["", f"## {label}", ""]
        records = report["candidates"][group]
        if not records:
            lines += ["None."]
        for record in records:
            candidate = ResearcherCandidate.from_dict(record["candidate"])
            lines += [f"### {_text(candidate.name)}", "", f"{_text(record['decision'] or record['lifecycle'])}: {_text(record['rationale'])}",
                      f"Identity: {_text(record['identity_key'])}; source IDs: {_text(json.dumps(candidate.identity_refs, sort_keys=True))}.", ""]
            _, index = evidence_packet(candidate)
            for evidence in index.values():
                from researcher.sources.public_web import public_reference
                try:
                    url = public_reference(evidence["url"]).replace(")", "%29")
                except ValueError:
                    continue
                lines += [f"[Evidence: {_text(evidence.get('title') or evidence['id'])}]({url}) — source date {_text(evidence.get('source_date') or 'unknown')}; relationship {_text(evidence['relationship'])}.", ""]
                lines += ["> " + _text(line) for line in evidence["excerpt"].splitlines()]
                lines += [""]
            for observation in candidate.professional_context:
                lines += [f"Professional context: {_text(observation.get('claim') or observation.get('excerpt'))} (source date {_text(observation.get('source_date') or 'unknown')}; {_text(observation.get('kind') or observation.get('relationship'))})."]
            questions = candidate.identity_ambiguities + [str(g) for g in candidate.evidence_gaps]
            questions += (record["terminal_payload"].get("full_decision", {}).get("semantic_evidence") or {}).get("unresolved_questions", [])
            lines += ["Unresolved questions: " + (_text("; ".join(questions)) if questions else "None recorded.")]
            if record.get("failure"):
                lines += ["Failure: " + _text(json.dumps(record["failure"]))]
    return "\n".join(lines) + "\n"


def write_projections(bridge, run_id: int, report: dict | None = None) -> dict:
    report = report or build_report(bridge, run_id)
    output = bridge.output_dir
    _write_json_atomic(output / "progress.json", {"brief_name": bridge.brief_name, "counts": report["counts"], "strings": report["queries"], "candidates_saved": report["counts"]["accepted"]})
    _write_json_atomic(output / "execution_plan.json", {"generated_strings": bridge.campaign(run_id).get("plan", [])})
    final = [record for group in ("accepted", "previously_saved", "review", "rejected") for record in report["candidates"][group]]
    _write_jsonl_atomic(output / "final_judgments.jsonl", final)
    for filename, stage, key in (("snippets.jsonl", "snippet", "snippet"), ("facial_judgments.jsonl", "facial", "facial_decision")):
        _write_jsonl_atomic(output / filename, _project_attempt_payloads(bridge.store, source="researcher", brief_id=bridge.brief_id, run_id=None, stage=stage, payload_key=key))
    return report


def publish_report(bridge, run_id: int) -> dict:
    report = write_projections(bridge, run_id)
    _write_json_atomic(bridge.output_dir / "run-report.json", report)
    text = render_report(report)
    temporary = bridge.output_dir / "researcher-slate.md.tmp"
    temporary.write_text(text)
    temporary.replace(bridge.output_dir / "researcher-slate.md")
    from market_intelligence.researcher_reflection import build_researcher_ecosystem_narrative
    records = [r for key in ("accepted", "review", "rejected") for r in report["candidates"][key]
               if r["judgment_brief_hash"] == report["run_metadata"]["brief_content_hash"]]
    _write_json_atomic(bridge.output_dir / "researcher-research-input.json", build_researcher_ecosystem_narrative(final_judgments=records))
    if bridge.brief_path:
        from market_intelligence.run_snapshots import finalize_run_snapshot
        snapshot = finalize_run_snapshot(source="researcher", brief_path=bridge.brief_path, state_dir=bridge.output_dir, run_id=run_id)
        campaign = bridge.campaign(run_id)
        campaign["published_snapshot"] = str(snapshot)
        campaign.pop("artifact_degradation", None)
        bridge.save_campaign(run_id, campaign)
    return report
