"""Post-session markdown report generator.

Reads all four layer output files and produces session_*_report.md.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_report(
    strategy_path: Path,
    graph_path: Path,
    candidates_path: Path,
    metrics_path: Path,
    report_path: Path,
    session_id: str,
    duration_seconds: float,
):
    """Read all layer files and produce a markdown report."""
    # Load data
    strategy_events = _read_jsonl(strategy_path)
    graph_data = _read_json(graph_path)
    candidates_data = _read_json(candidates_path)
    metrics_events = _read_jsonl(metrics_path)

    lines: list[str] = []

    # --- 1. Executive Summary ---
    lines.append(f"# Session Report: {session_id}")
    lines.append("")

    final_metrics = next((e for e in reversed(metrics_events) if e.get("event") == "session_final"), {})
    final_stats = final_metrics.get("final_stats", {})
    saves_count = final_stats.get("saved", 0)
    fy = final_stats.get("facial_yes", 0)
    save_rate = f"{saves_count / fy * 100:.1f}%" if fy > 0 else "n/a"
    m = int(duration_seconds // 60)

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Saves**: {saves_count}")
    lines.append(f"- **Save rate** (of FACIAL_YES): {save_rate}")
    lines.append(f"- **Candidates discovered**: {final_stats.get('candidates_discovered', 0)}")
    lines.append(f"- **Candidates enriched**: {final_stats.get('candidates_enriched', 0)}")
    lines.append(f"- **Duration**: {m} minutes")
    lines.append(f"- **Queries completed**: {final_metrics.get('queries_completed', 0)}")
    lines.append("")

    # --- 1.5 Run Cost + Run Health ---
    # P4.2: cost_usd/cost_per_save_usd are only present on final_stats when
    # the run's token-cost-log.jsonl produced a usable total — never an
    # affirmative $0.00 when cost is unknown (mirrors
    # shared.run_report_schema.render_run_report_markdown).
    cost_usd = final_stats.get("cost_usd")
    if cost_usd is not None:
        lines.append("## Run Cost")
        lines.append("")
        lines.append(f"- **Total cost**: ${float(cost_usd):.4f}")
        cost_per_save = final_stats.get("cost_per_save_usd")
        if cost_per_save is not None:
            lines.append(f"- **Cost per save**: ${float(cost_per_save):.4f}")
        lines.append("")

    # P4.3: run health judged against this brief's own historical baseline
    # (shared.observability_monitors.compute_run_health). Only rendered
    # when the monitor actually produced a verdict — no runtime state /
    # lookup errors degrade to silence rather than a false "healthy" claim.
    run_health = final_stats.get("run_health") or {}
    if run_health.get("status") == "ok":
        lines.append("## Run Health")
        lines.append("")
        if run_health.get("degraded"):
            reasons = ", ".join(str(r) for r in (run_health.get("degraded_reasons") or []))
            lines.append(f"- **Degraded: Yes** — reasons: {reasons}")
        else:
            lines.append("- Degraded: No")
        lines.append("")

    # P6.4 follow-up: bias_summary is only written to session_final when a
    # monitor was active this run (metrics_layer.write_final's
    # only-when-present discipline — None never lands). Also gate on
    # total_decisions > 0 so a monitor that ran but saw nothing doesn't
    # render an empty section (mirrors the run_health "no verdict" silence
    # pattern above).
    bias_summary = final_metrics.get("bias_summary") or {}
    if bias_summary.get("total_decisions", 0) > 0:
        lines.append("## Bias Monitor")
        lines.append("")
        lines.append(f"- **Total decisions**: {bias_summary.get('total_decisions', 0)}")
        lines.append(f"- **Facial YES rate**: {bias_summary.get('facial_yes_rate', 0):.1%}")
        lines.append(f"- **Save rate**: {bias_summary.get('save_rate', 0):.1%}")
        lines.append(f"- **Parse failures**: {bias_summary.get('parse_failures', 0)} "
                    f"({bias_summary.get('parse_failure_rate', 0):.1%})")
        alerts = bias_summary.get("alerts_fired", [])
        lines.append(f"- **Alerts fired**: {len(alerts)}")
        if alerts:
            for alert in alerts:
                lines.append(f"  - {alert}")
        lines.append("")

    # --- 2. Strategy Evolution ---
    lines.append("## Strategy Evolution")
    lines.append("")

    formed = next((e for e in strategy_events if e.get("event") == "strategy_formed"), None)
    if formed:
        lines.append(f"**Initial thesis** ({formed.get('query_count', 0)} queries):")
        lines.append(f"> {formed.get('rationale', 'n/a')}")
        lines.append("")
        lines.append(f"Channel distribution: {json.dumps(formed.get('channel_distribution', {}))}")
        lines.append("")

    adaptations = [e for e in strategy_events if e.get("event") == "adaptation_checkpoint"]
    if adaptations:
        lines.append("### Adaptations")
        lines.append("")
        for a in adaptations:
            lines.append(f"**Checkpoint {a.get('checkpoint', '?')}** — +{a.get('new_queries_added', 0)} queries, "
                        f"{len(a.get('skipped_ids', []))} skipped")
            lines.append(f"> {a.get('rationale', '')}")
            lines.append("")

    narrative = next((e for e in strategy_events if e.get("event") == "session_narrative"), None)
    if narrative and narrative.get("lessons_learned"):
        lines.append("### Lessons Learned")
        lines.append("")
        for lesson in narrative["lessons_learned"]:
            lines.append(f"- {lesson}")
        lines.append("")

    # --- 3. Saves Table ---
    saves = candidates_data.get("saves", [])
    if saves:
        lines.append("## Saves")
        lines.append("")
        lines.append("| Name | Capability | Confidence | Channel | Contact |")
        lines.append("|------|-----------|------------|---------|---------|")
        for s in saves:
            emails = ", ".join(s.get("contact_emails", [])[:2]) or "—"
            lines.append(f"| {s.get('name', '?')} | {s.get('capability_area', '?')} | "
                        f"{s.get('confidence', 0):.2f} | {s.get('query_channel', '?')} | {emails} |")
        lines.append("")

    # --- 4. Save Reasoning Chains ---
    if saves:
        lines.append("## Save Reasoning Chains")
        lines.append("")
        for s in saves:
            lines.append(f"### {s.get('name', '?')} ({s.get('username', '?')})")
            lines.append(f"- **Query**: {s.get('query_name', '?')} [{s.get('query_channel', '?')}]")
            lines.append(f"- **Decision**: {s.get('decision_type', '?')} ({s.get('confidence', 0):.2f})")
            lines.append(f"- **Capability**: {s.get('capability_area', '?')}")
            lines.append(f"- **Rationale**: {s.get('rationale', 'n/a')}")
            lines.append("")

    # --- 5. Close Rejects ---
    close_rejects = candidates_data.get("close_rejects", [])
    if close_rejects:
        lines.append("## Close Rejects (FACIAL_YES → REJECT)")
        lines.append("")
        lines.append("| Name | Capability Area | Confidence | Reason |")
        lines.append("|------|----------------|------------|--------|")
        for cr in close_rejects:
            reason = (cr.get("reason", "") or "")[:80]
            lines.append(f"| {cr.get('name', '?')} | {cr.get('capability_area', '?')} | "
                        f"{cr.get('confidence', 0):.2f} | {reason} |")
        lines.append("")

    # --- 6. FACIAL_NO Patterns ---
    facial_no = candidates_data.get("facial_no_patterns", {})
    if facial_no:
        lines.append("## FACIAL_NO Patterns")
        lines.append("")
        for q_key, data in facial_no.items():
            lines.append(f"**{q_key}** — {data.get('total', 0)} total")
            cats = data.get("categories", {})
            if cats:
                sorted_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)
                for cat, count in sorted_cats:
                    lines.append(f"  - {cat}: {count}")
            lines.append("")

    # --- 7. Graph Traversal ---
    if graph_data:
        branch_summary = graph_data.get("branch_summary", [])
        cross_refs = graph_data.get("cross_references", [])

        if branch_summary:
            lines.append("## Graph Traversal")
            lines.append("")
            productive = [b for b in branch_summary if b.get("productive")]
            lines.append(f"Productive queries: {len(productive)}/{len(branch_summary)}")
            lines.append("")
            if productive:
                lines.append("| Query | Channel | Candidates | Saves |")
                lines.append("|-------|---------|------------|-------|")
                for b in productive:
                    lines.append(f"| {b.get('root', '?')} | {b.get('channel', '?')} | "
                                f"{b.get('candidates_found', 0)} | {b.get('saves', 0)} |")
                lines.append("")

        if cross_refs:
            lines.append("### Cross-References (candidates found via multiple paths)")
            lines.append("")
            for cr in cross_refs:
                lines.append(f"- **{cr['candidate']}**: {', '.join(cr['found_via'])}")
            lines.append("")

    # --- 8. Resource Dashboard ---
    checkpoints = [e for e in metrics_events if e.get("event") == "checkpoint"]
    if checkpoints:
        lines.append("## Resource Dashboard")
        lines.append("")
        lines.append("| Checkpoint | Queries | Saves | Dedup% | Save Rate | Cost/Save |")
        lines.append("|-----------|---------|-------|--------|-----------|-----------|")
        for cp in checkpoints:
            cps = cp.get("cost_per_save", {})
            lines.append(f"| {cp.get('checkpoint', '?')} | {cp.get('queries_completed', 0)} | "
                        f"{cp.get('cumulative_saves', 0)} | {cp.get('dedup_rate', 0):.0%} | "
                        f"{cp.get('rolling_save_rate_last_10q', 0):.2%} | "
                        f"{cps.get('candidates_evaluated', 0):.0f} evals |")
        lines.append("")

    # --- 9. Capability Distribution ---
    cap_dist = candidates_data.get("capability_distribution", {})
    if cap_dist:
        lines.append("## Capability Distribution")
        lines.append("")
        lines.append("| Capability Area | Total | Direct | Transferable |")
        lines.append("|----------------|-------|--------|-------------|")
        for area, counts in sorted(cap_dist.items(), key=lambda x: x[1]["count"], reverse=True):
            lines.append(f"| {area} | {counts['count']} | {counts['direct']} | {counts['transferable']} |")
        lines.append("")

    # Write
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}
