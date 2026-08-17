"""Structured run-report schema and markdown rendering helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _require_dict(name: str, value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")
    return value


def _require_list(name: str, value: Any) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _stringify_list(value: list[Any]) -> list[str]:
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


@dataclass
class RunDebriefAnalysis:
    """Model-authored analytical sections for a run debrief."""

    winning_lanes: list[dict]
    underperforming_lanes: list[dict]
    coverage_gaps: list[dict]
    noise_patterns: list[dict]
    saved_candidate_patterns: dict
    adaptation_assessment: dict
    recommendations: dict
    brief_iteration_hints: dict

    @classmethod
    def from_dict(cls, data: dict) -> "RunDebriefAnalysis":
        if not isinstance(data, dict):
            raise ValueError("run debrief analysis must be a dict")
        required = (
            "winning_lanes",
            "underperforming_lanes",
            "coverage_gaps",
            "noise_patterns",
            "saved_candidate_patterns",
            "adaptation_assessment",
            "recommendations",
            "brief_iteration_hints",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"run debrief analysis missing keys: {', '.join(missing)}")

        return cls(
            winning_lanes=[item for item in _require_list("winning_lanes", data["winning_lanes"]) if isinstance(item, dict)],
            underperforming_lanes=[item for item in _require_list("underperforming_lanes", data["underperforming_lanes"]) if isinstance(item, dict)],
            coverage_gaps=[item for item in _require_list("coverage_gaps", data["coverage_gaps"]) if isinstance(item, dict)],
            noise_patterns=[item for item in _require_list("noise_patterns", data["noise_patterns"]) if isinstance(item, dict)],
            saved_candidate_patterns=_require_dict("saved_candidate_patterns", data["saved_candidate_patterns"]),
            adaptation_assessment=_require_dict("adaptation_assessment", data["adaptation_assessment"]),
            recommendations=_require_dict("recommendations", data["recommendations"]),
            brief_iteration_hints=_require_dict("brief_iteration_hints", data["brief_iteration_hints"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StructuredRunReport:
    """Validated machine-readable end-of-run report."""

    schema_version: int
    run_metadata: dict
    metrics_summary: dict
    string_performance: list[dict]
    winning_lanes: list[dict]
    underperforming_lanes: list[dict]
    coverage_gaps: list[dict]
    noise_patterns: list[dict]
    saved_candidate_patterns: dict
    adaptation_assessment: dict
    recommendations: dict
    brief_iteration_hints: dict

    @classmethod
    def from_parts(cls, snapshot: dict, analysis: RunDebriefAnalysis) -> "StructuredRunReport":
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must be a dict")
        required = ("run_metadata", "metrics_summary", "string_performance")
        missing = [key for key in required if key not in snapshot]
        if missing:
            raise ValueError(f"run-report snapshot missing keys: {', '.join(missing)}")

        return cls(
            schema_version=int(snapshot.get("schema_version", 1)),
            run_metadata=_require_dict("run_metadata", snapshot["run_metadata"]),
            metrics_summary=_require_dict("metrics_summary", snapshot["metrics_summary"]),
            string_performance=[item for item in _require_list("string_performance", snapshot["string_performance"]) if isinstance(item, dict)],
            winning_lanes=analysis.winning_lanes,
            underperforming_lanes=analysis.underperforming_lanes,
            coverage_gaps=analysis.coverage_gaps,
            noise_patterns=analysis.noise_patterns,
            saved_candidate_patterns=analysis.saved_candidate_patterns,
            adaptation_assessment=analysis.adaptation_assessment,
            recommendations=analysis.recommendations,
            brief_iteration_hints=analysis.brief_iteration_hints,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredRunReport":
        if not isinstance(data, dict):
            raise ValueError("structured run report must be a dict")
        required = (
            "schema_version",
            "run_metadata",
            "metrics_summary",
            "string_performance",
            "winning_lanes",
            "underperforming_lanes",
            "coverage_gaps",
            "noise_patterns",
            "saved_candidate_patterns",
            "adaptation_assessment",
            "recommendations",
            "brief_iteration_hints",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"structured run report missing keys: {', '.join(missing)}")
        analysis = RunDebriefAnalysis.from_dict(
            {
                "winning_lanes": data["winning_lanes"],
                "underperforming_lanes": data["underperforming_lanes"],
                "coverage_gaps": data["coverage_gaps"],
                "noise_patterns": data["noise_patterns"],
                "saved_candidate_patterns": data["saved_candidate_patterns"],
                "adaptation_assessment": data["adaptation_assessment"],
                "recommendations": data["recommendations"],
                "brief_iteration_hints": data["brief_iteration_hints"],
            }
        )
        return cls.from_parts(
            {
                "schema_version": data["schema_version"],
                "run_metadata": data["run_metadata"],
                "metrics_summary": data["metrics_summary"],
                "string_performance": data["string_performance"],
            },
            analysis,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _render_named_section(title: str, items: list[dict], heading_key: str, detail_keys: list[str]) -> list[str]:
    lines = [f"## {title}"]
    if not items:
        lines.append("- None")
        lines.append("")
        return lines
    for item in items:
        heading = str(item.get(heading_key, "Unnamed")).strip() or "Unnamed"
        lines.append(f"- **{heading}**")
        for key in detail_keys:
            value = item.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                rendered = ", ".join(_stringify_list(value))
            else:
                rendered = str(value).strip()
            if rendered:
                label = key.replace("_", " ")
                lines.append(f"  {label}: {rendered}")
    lines.append("")
    return lines


def render_run_report_markdown(report: StructuredRunReport) -> str:
    """Render human-readable markdown from a validated structured report."""
    meta = report.run_metadata
    metrics = report.metrics_summary
    saved_patterns = report.saved_candidate_patterns or {}
    adaptation = report.adaptation_assessment or {}
    recs = report.recommendations or {}
    hints = report.brief_iteration_hints or {}

    title = meta.get("role_title") or meta.get("brief_name") or "Run Debrief"
    lines = [f"# End-of-Run Debrief Report: {title}", ""]

    # P1.3: a run where >20% of save attempts physically failed must say so
    # at the top of the report — before any productivity claims.
    save_health = metrics.get("pipeline_save_health") or {}
    save_failure_rate = float(save_health.get("failure_rate") or 0.0)
    if save_health.get("attempted") and save_failure_rate > 0.20:
        failed_total = int(save_health.get("failed", 0)) + int(
            save_health.get("failed_permanent", 0)
        )
        lines.extend(
            [
                f"> **⚠ PIPELINE-SAVE DEGRADATION: {save_failure_rate:.0%} of save "
                f"attempts failed ({failed_total} of {save_health.get('attempted')}).** "
                "Saved counts below understate judged-SAVE candidates; failed saves "
                "retry on rediscovery/resume. See Pipeline-Save Health.",
                "",
            ]
        )

    # P9.3: a brief generated by preflight_v2 and never human-reviewed must
    # say so before any productivity claims — the stamp only exists on
    # briefs that carry ``provenance`` (P9.5's absence-en-bloc detector's
    # future positive counterpart); hand-authored / legacy briefs have no
    # ``provenance`` key and render nothing here.
    provenance = meta.get("provenance") or {}
    if provenance.get("reviewed") is False:
        generated_by = str(provenance.get("generated_by") or "preflight_v2")
        lines.extend(
            [
                f"> **⚠ UNREVIEWED BRIEF: this run's evaluation criteria were "
                f"machine-generated by `{generated_by}` and have not been "
                "human-reviewed.**",
                "",
            ]
        )
    # The generated brief's own open questions (preflight_confidence_notes)
    # render beside the unreviewed-brief stamp — the operator-review slot —
    # so a question preflight asked cannot live only in a scrolled-away
    # console print.
    confidence_notes = str(meta.get("preflight_confidence_notes") or "").strip()
    if confidence_notes:
        lines.extend(
            [
                f"> **⚠ PREFLIGHT CONFIDENCE NOTES (operator review):** {confidence_notes}",
                "",
            ]
        )

    # P3b: the session-geography receipt — what pool this run searched, as a
    # recorded fact. Renders right under the banner warnings so an off-geo
    # question is answered before any productivity claims.
    session_geography = meta.get("session_geography") or {}
    if isinstance(session_geography, dict) and session_geography.get("intended"):
        intended = "; ".join(str(v) for v in session_geography.get("intended") or [])
        verified = bool(session_geography.get("verified_applied"))
        reasserts = int(session_geography.get("reasserts", 0) or 0)
        status = "verified applied" if verified else "NOT verified"
        suffix = f", {reasserts} re-assert(s)" if reasserts else ""
        # P3a Stage B: candidate→facet resolutions are part of the receipt —
        # what the model rewrote is a recorded fact, never a silent edit.
        resolutions = session_geography.get("resolutions") or []
        if resolutions:
            rendered = ", ".join(
                f"{r.get('candidate', '?')}→{r.get('resolved', '?')}"
                for r in resolutions
                if isinstance(r, dict)
            )
            suffix += f"; facet names model-resolved: {rendered}"
        lines.extend([f"> Geography: {intended} — {status}{suffix}.", ""])

    lines.append("## Executive Summary")
    overall_summary = str(meta.get("overall_summary", "")).strip()
    if overall_summary:
        lines.append(overall_summary)
        lines.append("")
    lines.extend(
        [
            "| Metric | Value |",
            "|---|---|",
            f"| Strings executed | {metrics.get('strings_executed', 0)} |",
            f"| Strings skipped | {metrics.get('strings_skipped', 0)} |",
            f"| Total results | {metrics.get('total_results', 0)} |",
            f"| Pages reviewed | {metrics.get('total_pages_reviewed', 0)} |",
            f"| Candidates evaluated | {metrics.get('candidates_evaluated', 0)} |",
            f"| Facial YES | {metrics.get('facial_yes', 0)} |",
            f"| Facial NO | {metrics.get('facial_no', 0)} |",
            f"| Saved | {metrics.get('saved', 0)} |",
            f"| Rejected | {metrics.get('rejected', 0)} |",
            f"| High-pressure candidates seen | {metrics.get('high_pressure_candidates_seen', 0)} |",
            f"| Activity-saturated preview skips | {metrics.get('activity_saturated_preview_skips', 0)} |",
            f"| High-fit low-novelty saves | {metrics.get('high_fit_low_novelty_saves', 0)} |",
            f"| Overall save rate | {metrics.get('overall_save_rate', 0):.1%} |",
            f"| Facial YES rate | {metrics.get('facial_yes_rate', 0):.1%} |",
            "",
        ]
    )

    # P2.2: structured-filter actuator health — per-dimension apply rates and
    # the strings that silently ran keyword-only.
    actuator = metrics.get("structured_filter_actuator") or {}
    per_dimension = actuator.get("per_dimension") or {}
    if per_dimension:
        lines.extend(
            [
                "## Structured-Filter Actuator",
                "| Dimension | Requested values | Applied | Apply rate |",
                "|---|---|---|---|",
            ]
        )
        for dim, row in per_dimension.items():
            lines.append(
                f"| {dim} | {row.get('requested', 0)} | {row.get('applied', 0)} "
                f"| {float(row.get('apply_rate', 0.0)):.0%} |"
            )
        fallback_ids = actuator.get("strings_fell_back_to_keyword") or []
        if fallback_ids:
            lines.append("")
            lines.append(
                f"**{len(fallback_ids)} string(s) ran as keyword-only fallback** "
                f"(ids: {', '.join(str(i) for i in fallback_ids)}) — their "
                "performance is not evidence about their filter-bounded lanes."
            )
        lines.append("")

    # P3b: constraint-ownership manifest — who owns each stated constraint,
    # plus the defer-dimension counter (requested-but-unsupported structured
    # controls). Only-when-present.
    constraint_manifest = metrics.get("constraint_manifest") or {}
    manifest_classes = constraint_manifest.get("classes") or {}
    if manifest_classes:
        lines.extend(
            [
                "## Constraint Manifest",
                "| Constraint | Stated | Status | Owner |",
                "|---|---|---|---|",
            ]
        )
        for name, entry in manifest_classes.items():
            stated = "yes" if entry.get("stated_in_brief") else "no"
            owner = str(entry.get("owner_layer") or "—")
            lines.append(
                f"| {name} | {stated} | {entry.get('status', '?')} | {owner} |"
            )
        unsupported = constraint_manifest.get("requested_but_unsupported") or {}
        if unsupported:
            rendered = ", ".join(
                f"{dim}×{count}" for dim, count in sorted(unsupported.items())
            )
            lines.append("")
            lines.append(
                f"**Requested-but-unsupported structured dimensions:** {rendered} — "
                "these constraints rode Boolean/title surfaces only."
            )
        lines.append("")

    # P7 Stage C (Wave 3): family-key churn — the learning loop aggregates on
    # family_key; low run-over-run overlap means history never accumulates.
    # Only-when-warning — a stable key set renders nothing here.
    key_stability = metrics.get("key_stability") or {}
    if key_stability.get("warning"):
        jaccard = float(key_stability.get("jaccard") or 0.0)
        lines.extend(
            [
                "## ⚠ FAMILY-KEY CHURN",
                (
                    f"> Jaccard overlap with the prior run is {jaccard:g} (<0.3): "
                    f"family keys churned; learning is not accumulating. "
                    f"({key_stability.get('prior_family_count', '?')} prior vs "
                    f"{key_stability.get('current_family_count', '?')} current families, "
                    f"runs {key_stability.get('prior_run_id', '?')}→"
                    f"{key_stability.get('current_run_id', '?')}.)"
                ),
                "",
            ]
        )

    # P5 (Wave 2): strings refused at queue build by the wired Boolean lint
    # (error-severity findings / ubiquity gate). Only-when-present — a clean
    # run renders nothing here.
    lint_blocked = metrics.get("lint_blocked") or []
    if lint_blocked:
        lines.extend(
            [
                "## Boolean Craft",
                f"**{len(lint_blocked)} string(s) blocked at queue build** — "
                "error-severity lint findings; they never executed.",
            ]
        )
        for item in lint_blocked:
            label = str(item.get("family_key") or item.get("name") or "unnamed")
            codes = ", ".join(str(c) for c in (item.get("codes") or []))
            lines.append(f"- **{label}** ({item.get('source', '?')}): {codes}")
            for hint in item.get("repair_hints") or []:
                lines.append(f"  repair: {hint}")
        lines.append("")

    if save_health.get("attempted"):
        lines.extend(
            [
                "## Pipeline-Save Health",
                "| Metric | Value |",
                "|---|---|",
                f"| Save attempts | {save_health.get('attempted', 0)} |",
                f"| Succeeded | {save_health.get('succeeded', 0)} |",
                f"| Already in pipeline | {save_health.get('already_present', 0)} |",
                f"| Failed (retryable) | {save_health.get('failed', 0)} |",
                f"| Failed permanently | {save_health.get('failed_permanent', 0)} |",
                f"| Interrupted | {save_health.get('interrupted', 0)} |",
                f"| Retried from prior attempt | {save_health.get('retried_from_prior', 0)} |",
                f"| Failure rate | {save_failure_rate:.1%} |",
            ]
        )
        failed_by_reason = save_health.get("failed_by_reason") or {}
        for reason, count in sorted(failed_by_reason.items()):
            lines.append(f"| Failure reason: {reason} | {count} |")
        lines.append("")

    # P4.2: run-level cost, rendered only when the run's token-cost-log.jsonl
    # produced a usable total. Never an affirmative $0.00 when the log is
    # absent/empty/unresolved — that's "unknown", not "free".
    cost_summary = metrics.get("cost_summary") or {}
    if cost_summary.get("status") == "ok":
        lines.extend(
            [
                "## Run Cost",
                f"- Total cost: ${float(cost_summary.get('cost_usd') or 0.0):.4f}",
            ]
        )
        cost_per_save = cost_summary.get("cost_per_save_usd")
        if cost_per_save is not None:
            lines.append(f"- Cost per save: ${float(cost_per_save):.4f}")
        lines.append("")

    # P4.3: this run judged against its own db's historical baseline
    # (shared.observability_monitors.compute_run_health). Only rendered
    # when the monitor produced a verdict — no runtime state / lookup
    # errors degrade to silence rather than a false "healthy" claim.
    run_health = metrics.get("run_health") or {}
    if run_health.get("status") == "ok":
        lines.append("## Run Health")
        if run_health.get("degraded"):
            reasons = ", ".join(str(r) for r in (run_health.get("degraded_reasons") or []))
            lines.append(f"- **Degraded: Yes** — reasons: {reasons}")
        else:
            lines.append("- Degraded: No")
        lines.append("")

    # GLM-5.2 shadow-judge evaluation seam (Fireworks, opt-in via
    # SHADOW_FACIAL_MODEL_ENABLED). Rendered only when the run actually
    # logged >=1 facial_shadow_comparison event — this block's key
    # (metrics_summary["shadow_facial"]) is entirely absent otherwise
    # (linkedin/orchestrator.py's _shadow_facial_summary returns None on
    # zero comparisons), so a plain-dict presence check is the gate here
    # rather than the "status" idiom the Run Cost/Run Health blocks use.
    # Any individual rate can still be None (undefined denominator, e.g.
    # zero comparable candidates) and is rendered as "n/a", never a
    # fabricated 0.0%.
    shadow_facial = metrics.get("shadow_facial")
    shadow_full = metrics.get("shadow_full")

    def _fmt_rate(value: Any) -> str:
        return f"{float(value):.1%}" if value is not None else "n/a"

    def _fmt_ms(value: Any) -> str:
        return f"{float(value):.0f}ms" if value is not None else "n/a"

    if isinstance(shadow_facial, dict):
        lines.extend(
            [
                "## Shadow Judge",
                f"- Shadow model: {shadow_facial.get('model', 'unknown')}",
                f"- Comparisons: {shadow_facial.get('comparisons', 0)}",
                f"- Agreement rate: {_fmt_rate(shadow_facial.get('agreement_rate'))}",
                f"- Shadow parse-failure rate: {_fmt_rate(shadow_facial.get('shadow_parse_failure_rate'))}",
                f"- Primary YES rate: {_fmt_rate(shadow_facial.get('primary_yes_rate'))}",
                f"- Shadow YES rate: {_fmt_rate(shadow_facial.get('shadow_yes_rate'))}",
                f"- Mean shadow latency: {_fmt_ms(shadow_facial.get('mean_latency_ms'))}",
            ]
        )
        # Cache-hit visibility (Fireworks automatic prefix caching):
        # only-when-present — the key is absent entirely (never a
        # fabricated 0.0%) unless at least one token-cost-log row for this
        # tier carried real token counts (linkedin/orchestrator.py's
        # _shadow_cache_hit_rate).
        if "mean_cache_hit_rate" in shadow_facial:
            lines.append(
                f"- Mean cache-hit rate: {_fmt_rate(shadow_facial.get('mean_cache_hit_rate'))}"
            )
        lines.append("")

    # Full-eval sibling block — same only-when-present discipline,
    # independent of whether the facial block above rendered (a run can
    # have facial-tier comparisons with zero full-eval ones, or vice versa
    # if SHADOW_FACIAL_MODEL_ENABLED was flipped on partway through).
    if isinstance(shadow_full, dict):
        lines.extend(
            [
                "### Full Evaluation (shadow)",
                f"- Shadow model: {shadow_full.get('model', 'unknown')}",
                f"- Comparisons: {shadow_full.get('comparisons', 0)}",
                f"- Agreement rate: {_fmt_rate(shadow_full.get('agreement_rate'))}",
                f"- Shadow parse-failure rate: {_fmt_rate(shadow_full.get('shadow_parse_failure_rate'))}",
                f"- Primary save rate: {_fmt_rate(shadow_full.get('primary_save_rate'))}",
                f"- Shadow save rate: {_fmt_rate(shadow_full.get('shadow_save_rate'))}",
                f"- Mean shadow latency: {_fmt_ms(shadow_full.get('mean_latency_ms'))}",
            ]
        )
        if "mean_cache_hit_rate" in shadow_full:
            lines.append(
                f"- Mean cache-hit rate: {_fmt_rate(shadow_full.get('mean_cache_hit_rate'))}"
            )
        # Sampling-bias caveat: enriched re-judgments (external-evidence
        # path) are not shadowed, and they concentrate the hardest calls.
        lines.append(
            "- Note: agreement excludes external-evidence re-judgments "
            "(hardest sub-population un-shadowed); read as an upper bound."
        )
        lines.append("")

    # P3.6: facial calibration closes the loop — compare the run's actual
    # facial YES rate against the brief-authored expected band, and surface
    # a clearly-visible drift warning once it's been out-of-band for 2+
    # consecutive runs. Silent when there were no facial verdicts or the
    # brief never authored a band (status != "ok").
    facial_calibration = metrics.get("facial_calibration") or {}
    if facial_calibration.get("status") == "ok":
        actual_rate = float(facial_calibration.get("actual_yes_rate") or 0.0)
        low = float(facial_calibration.get("authored_low") or 0.0)
        high = float(facial_calibration.get("authored_high") or 0.0)
        # P6 (Wave 3): the band's provenance renders beside the band — a
        # loader/synthesis default band is a template artifact, and its
        # out-of-band judgments read differently from a reasoned band's.
        band_source = str(facial_calibration.get("band_source") or "")
        source_suffix = f", band source: {band_source}" if band_source else ""
        lines.extend(
            [
                "## Facial Calibration",
                f"- Facial YES rate: {actual_rate:.1%} "
                f"(authored band {low:.1%}–{high:.1%}{source_suffix})",
            ]
        )
        if facial_calibration.get("calibration_drift_warning"):
            lines.append(
                "> **⚠ FACIAL-CALIBRATION DRIFT: observed YES rate has been "
                "outside the authored band for 2+ consecutive runs.** Consider "
                "proposing a recalibration hunk."
            )
        lines.append("")

    lines.extend(
        _render_named_section(
            "Top Performing Lanes",
            report.winning_lanes,
            "lane",
            ["string_ids", "candidate_examples", "evidence", "why_it_worked", "recommended_action"],
        )
    )
    lines.extend(
        _render_named_section(
            "Underperforming Lanes",
            report.underperforming_lanes,
            "lane",
            ["string_ids", "issue", "evidence", "recommended_action"],
        )
    )
    lines.extend(
        _render_named_section(
            "Coverage Gaps",
            report.coverage_gaps,
            "gap",
            ["why_it_matters", "suggested_search_strategy"],
        )
    )
    lines.extend(
        _render_named_section(
            "Noise Patterns",
            report.noise_patterns,
            "pattern",
            ["evidence", "mitigation"],
        )
    )

    lines.append("## Saved Candidate Patterns")
    standout = saved_patterns.get("standout_candidates", [])
    if standout:
        lines.append("### Standout Candidates")
        for item in standout:
            lines.append(f"- **{item.get('name', 'Unnamed')}**: {item.get('why', '').strip()}")
    common_employers = saved_patterns.get("common_employers", [])
    if common_employers:
        lines.append("### Common Employers")
        for item in common_employers:
            note = f" — {item.get('note', '').strip()}" if item.get("note") else ""
            lines.append(f"- **{item.get('employer', 'Unknown')}**: {item.get('count', 0)}{note}")
    common_titles = saved_patterns.get("common_titles", [])
    if common_titles:
        lines.append("### Common Titles")
        for item in common_titles:
            note = f" — {item.get('note', '').strip()}" if item.get("note") else ""
            lines.append(f"- **{item.get('title_family', 'Unknown')}**: {item.get('count', 0)}{note}")
    archetypes = saved_patterns.get("archetype_distribution", [])
    if archetypes:
        lines.append("### Archetype Distribution")
        for item in archetypes:
            note = f" — {item.get('note', '').strip()}" if item.get("note") else ""
            lines.append(f"- **{item.get('archetype', 'Unknown')}**: {item.get('count', 0)}{note}")
    seniority_notes = _stringify_list(saved_patterns.get("seniority_notes", []))
    if seniority_notes:
        lines.append("### Seniority Notes")
        for item in seniority_notes:
            lines.append(f"- {item}")
    lines.append("")

    lines.append("## Adaptation Assessment")
    # P4.5: pure per-string-stat ROI numbers (realized saves of strings an
    # adaptation call inserted vs. displaced) feeding the model-authored
    # narrative below. Only rendered when the run actually logged
    # block_adaptation events — "no_adaptation_events" degrades to silence.
    adaptation_roi = metrics.get("adaptation_roi") or {}
    if adaptation_roi.get("status") == "ok":
        roi_events = adaptation_roi.get("events") or []
        lines.extend(
            [
                "### Measured Adaptation ROI",
                f"- Adaptation events: {len(roi_events)}",
                f"- Inserted saves: {adaptation_roi.get('total_inserted_saves', 0)}",
                f"- Displaced saves: {adaptation_roi.get('total_displaced_saves', 0)}",
                f"- Net saves gained: {adaptation_roi.get('net_saves_gained', 0)}",
                "",
            ]
        )
    if adaptation.get("summary"):
        lines.append(str(adaptation["summary"]).strip())
    effective = _stringify_list(adaptation.get("effective_refinements", []))
    if effective:
        lines.append("")
        lines.append("### Effective Refinements")
        for item in effective:
            lines.append(f"- {item}")
    questionable = _stringify_list(adaptation.get("questionable_or_skipped", []))
    if questionable:
        lines.append("")
        lines.append("### Questionable or Skipped")
        for item in questionable:
            lines.append(f"- {item}")
    operational = _stringify_list(adaptation.get("operational_notes", []))
    if operational:
        lines.append("")
        lines.append("### Operational Notes")
        for item in operational:
            lines.append(f"- {item}")
    lines.append("")

    lines.append("## Recommendations")
    for heading, key in (
        ("Try Next", "try_next"),
        ("Avoid Next", "avoid_next"),
        ("Prioritize Pipeline", "prioritize_pipeline"),
    ):
        values = _stringify_list(recs.get(key, []))
        if values:
            lines.append(f"### {heading}")
            for item in values:
                lines.append(f"- {item}")
    lines.append("")

    locked_cautions = _stringify_list(hints.get("locked_field_cautions", []))
    if locked_cautions:
        lines.append("## Brief Iteration Hints")
        for item in locked_cautions:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
