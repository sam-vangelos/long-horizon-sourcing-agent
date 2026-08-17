"""Structured per-run research context for market-intelligence backends."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from market_intelligence.schema import MarketEvidenceBatch, MarketIdentity
from shared.search_memory import extract_dominant_anchors, get_search_memory_families
from shared.storage import read_json, read_jsonl, write_json


SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}
ADAPTATION_EVENTS = {
    "block_adaptation",
    "forced_narrow",
    "early_exit",
    "string_error",
    "bias_alert",
    "architecture_pivot",
    "adaptation_decision",
}

MAX_STRING_PERFORMANCE = 12
MAX_WINNING_LANES = 5
MAX_UNDERPERFORMING_LANES = 5
MAX_GAPS = 5
MAX_NOISE_PATTERNS = 5
MAX_SEARCH_MEMORY_FAMILIES = 8
MAX_SAVED_EXAMPLES = 12
MAX_REJECTED_EXAMPLES = 12
MAX_ADAPTATION_EVENTS = 25
MAX_FULL_PACKETS = 3


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sort_by_generated_at(values: list[tuple[datetime | None, Any]]) -> list[Any]:
    return [
        item
        for _dt, item in sorted(
            values,
            key=lambda row: row[0] or datetime.min,
        )
    ]


def _limit_records(items: list[dict], limit: int, sort_fields: tuple[str, ...]) -> list[dict]:
    def _sort_key(item: dict) -> tuple[Any, ...]:
        key: list[Any] = []
        for field in sort_fields:
            value = item.get(field)
            if isinstance(value, (int, float)):
                key.append(-value)
            else:
                key.append(str(value))
        return tuple(key)

    return sorted(items, key=_sort_key)[:limit]


def _load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return [item for item in read_jsonl(path) if isinstance(item, dict)]
    except Exception:
        return []


def _top_string_performance(report_input: dict | None, report: dict | None, runtime_summary: dict) -> list[dict]:
    entries = []
    if isinstance(report_input, dict):
        entries = [
            item for item in report_input.get("string_performance", []) if isinstance(item, dict)
        ]
    if not entries and isinstance(report, dict):
        entries = [
            item for item in report.get("string_performance", []) if isinstance(item, dict)
        ]
    if not entries:
        for work_unit in runtime_summary.get("work_units", []):
            if not isinstance(work_unit, dict):
                continue
            entries.append(
                {
                    "string_id": work_unit.get("source_unit_id", ""),
                    "name": work_unit.get("display_name", ""),
                    "status": "done",
                    "result_count": work_unit.get("candidate_volume", 0),
                    "pages_reviewed": work_unit.get("pages_reviewed", 0),
                    "saves": work_unit.get("saves_count", 0),
                    "save_rate": round(
                        int(work_unit.get("saves_count", 0))
                        / max(int(work_unit.get("candidate_volume", 0)), 1),
                        4,
                    ),
                    "saved_candidates": [],
                    "notes": work_unit.get("boolean", ""),
                    "facial_yes_count": work_unit.get("facial_yes_count", 0),
                    "facial_no_count": work_unit.get("facial_no_count", 0),
                    "candidates_count": work_unit.get("candidate_volume", 0),
                    "duplicates_count": work_unit.get("duplicates_count", 0),
                    "family_key": work_unit.get("family_key", ""),
                    "novelty_bucket": work_unit.get("novelty_bucket", ""),
                    "domain_lane": work_unit.get("domain_lane", ""),
                }
            )
    return _limit_records(entries, MAX_STRING_PERFORMANCE, ("saves", "save_rate", "candidates_count"))


def _lane_execution_summary(report_input: dict | None, string_performance: list[dict]) -> list[dict]:
    """Extract lane execution summary, preferring the pre-built key when present.

    For snapshots built by D1+ the ``lane_execution_summary`` key is present
    directly. For older snapshots we group ``string_performance`` by
    ``domain_lane`` (falling back to ``family_key``) to reconstruct it.
    """
    if isinstance(report_input, dict):
        direct = report_input.get("lane_execution_summary")
        if isinstance(direct, list) and direct:
            return [item for item in direct if isinstance(item, dict)]

    # Reconstruct from string_performance for old snapshots
    buckets: dict[str, dict] = {}
    for sp in string_performance:
        if not isinstance(sp, dict):
            continue
        lane_key = (
            str(sp.get("domain_lane") or sp.get("family_key") or "legacy").strip()
            or "legacy"
        )
        if lane_key not in buckets:
            buckets[lane_key] = {
                "lane_id": lane_key,
                "lane_name": lane_key,
                "family_keys": [],
                "acquisition_modes": [],
                "string_count": 0,
                "result_count": 0,
                "pages_reviewed": 0,
                "candidates_evaluated": 0,
                "facial_yes": 0,
                "facial_no": 0,
                "saves": 0,
                "save_rate": 0.0,
                "string_ids": [],
            }
        b = buckets[lane_key]
        b["string_count"] += 1
        b["result_count"] += int(sp.get("result_count", 0) or 0)
        b["pages_reviewed"] += int(sp.get("pages_reviewed", 0) or 0)
        b["candidates_evaluated"] += int(sp.get("candidates_count", 0) or 0)
        b["facial_yes"] += int(sp.get("facial_yes_count", 0) or 0)
        b["facial_no"] += int(sp.get("facial_no_count", 0) or 0)
        b["saves"] += int(sp.get("saves", 0) or 0)
        b["string_ids"].append(sp.get("string_id", ""))
        fk = str(sp.get("family_key", "") or "").strip()
        if fk and fk not in b["family_keys"]:
            b["family_keys"].append(fk)
    for b in buckets.values():
        total = b["candidates_evaluated"]
        b["save_rate"] = round(b["saves"] / max(total, 1), 4)
    return list(buckets.values())


def _lane_evidence_from_snapshot(
    report_input: dict | None,
    report_analysis: dict,
    string_performance: list[dict],
) -> dict:
    """Build lane evidence for research prompt consumption.

    Combines lane execution summary with report analysis winning/underperforming
    lanes to produce a lane-level evidence section the research prompt can reason
    about: committed variants, abandoned lanes, result window misses.
    """
    winning = report_analysis.get("winning_lanes", []) if isinstance(report_analysis, dict) else []
    underperforming = report_analysis.get("underperforming_lanes", []) if isinstance(report_analysis, dict) else []

    # Extract committed/abandoned from string_performance search_intelligence
    committed_variants: list[dict] = []
    abandoned_variants: list[dict] = []
    for sp in string_performance:
        si = sp.get("search_intelligence") if isinstance(sp, dict) else None
        if not isinstance(si, dict):
            continue
        bv = si.get("best_variant")
        if isinstance(bv, dict) and bv.get("variant_id"):
            committed_variants.append({
                "string_id": sp.get("string_id"),
                "lane_id": sp.get("domain_lane") or sp.get("family_key") or "",
                "variant_id": bv.get("variant_id"),
                "variant_kind": bv.get("variant_kind", ""),
            })

        variants = si.get("variants")
        if isinstance(variants, list):
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                if variant.get("status") != "abandoned":
                    continue
                abandoned_variants.append({
                    "string_id": sp.get("string_id"),
                    "lane_id": sp.get("domain_lane") or sp.get("family_key") or "",
                    "variant_id": variant.get("variant_id", ""),
                    "variant_kind": variant.get("variant_kind", ""),
                    "reason": variant.get("lifecycle_reason") or variant.get("decision_rationale", ""),
                })

        last_decision = si.get("last_variant_decision")
        if isinstance(last_decision, dict) and last_decision.get("action") == "abandon":
            abandoned_variants.append({
                "string_id": sp.get("string_id"),
                "lane_id": sp.get("domain_lane") or sp.get("family_key") or "",
                "variant_id": last_decision.get("variant_id", ""),
                "variant_kind": "",
                "reason": last_decision.get("reason", ""),
            })

    # Deduplicate committed variants by variant_id
    seen_variants: set[str] = set()
    deduped: list[dict] = []
    for cv in committed_variants:
        vid = cv.get("variant_id", "")
        if vid and vid not in seen_variants:
            seen_variants.add(vid)
            deduped.append(cv)
    committed_variants = deduped[:12]

    seen_abandoned: set[str] = set()
    deduped_abandoned: list[dict] = []
    for item in abandoned_variants:
        vid = str(item.get("variant_id", "") or "")
        key = vid or f"{item.get('string_id')}:{item.get('reason')}"
        if key in seen_abandoned:
            continue
        seen_abandoned.add(key)
        deduped_abandoned.append(item)
    abandoned_variants = deduped_abandoned[:8]

    return {
        "winning_lanes": winning[:5],
        "underperforming_lanes": underperforming[:5],
        "committed_variants": committed_variants,
        "abandoned_variants": abandoned_variants,
    }


def _bounded_search_memory_summary(report_input: dict | None, search_memory: dict | None) -> dict:
    summary = (
        (report_input or {}).get("search_memory_summary")
        if isinstance(report_input, dict)
        else None
    )
    if isinstance(summary, dict):
        overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
        families = [item for item in summary.get("families", []) if isinstance(item, dict)]
    else:
        overall = (search_memory or {}).get("overall") if isinstance(search_memory, dict) else {}
        families = get_search_memory_families(search_memory)
    return {
        "overall": overall or {},
        "families": _limit_records(
            families,
            MAX_SEARCH_MEMORY_FAMILIES,
            ("saves", "save_rate", "strings_seen"),
        ),
    }


def _profile_lookup(profile_summaries: list[dict], snippets: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    by_name: dict[str, dict] = {}
    by_url: dict[str, dict] = {}
    for record in profile_summaries + snippets:
        if not isinstance(record, dict):
            continue
        name = _normalize_text(record.get("name") or record.get("candidate_name"))
        profile_url = _normalize_text(record.get("profile_url"))
        if name and name not in by_name:
            by_name[name.lower()] = record
        if profile_url and profile_url not in by_url:
            by_url[profile_url] = record
    return by_name, by_url


def _profile_context(record: dict) -> dict:
    experiences = record.get("experiences") or []
    latest = experiences[0] if experiences and isinstance(experiences[0], dict) else {}
    bullets = latest.get("summary_bullets") or []
    summary_excerpt = " ".join(str(item).strip() for item in bullets[:2] if str(item).strip())
    return {
        "headline": _normalize_text(record.get("headline")),
        "current_title": _normalize_text(record.get("current_title") or latest.get("title")),
        "current_company": _normalize_text(record.get("current_company") or latest.get("company")),
        "location": _normalize_text(record.get("location") or latest.get("location")),
        "summary_excerpt": summary_excerpt[:500],
    }


def _fallback_candidate_examples(report_input: dict | None, key: str) -> list[dict]:
    if not isinstance(report_input, dict):
        return []
    items = report_input.get(key, [])
    if not isinstance(items, list):
        return []
    examples: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        examples.append(
            {
                "candidate_name": _normalize_text(item.get("candidate_name") or item.get("name")),
                "decision": _normalize_text(item.get("decision")),
                "path": _normalize_text(item.get("path")),
                "confidence": (
                    float(item["confidence"])
                    if isinstance(item.get("confidence"), (int, float))
                    and not isinstance(item.get("confidence"), bool)
                    else None
                ),
                "rationale": _normalize_text(item.get("rationale") or item.get("why")),
            }
        )
    return examples


def _build_candidate_examples(
    final_judgments: list[dict],
    profile_summaries: list[dict],
    snippets: list[dict],
    report_input: dict | None,
) -> dict:
    by_name, by_url = _profile_lookup(profile_summaries, snippets)
    saved: list[dict] = []
    rejected: list[dict] = []

    for record in final_judgments:
        if not isinstance(record, dict):
            continue
        decision = _normalize_text(record.get("decision")).upper()
        name = _normalize_text(record.get("candidate_name") or record.get("name"))
        profile_url = _normalize_text(record.get("profile_url"))
        profile = by_url.get(profile_url) or by_name.get(name.lower()) or {}
        example = {
            "candidate_name": name,
            "decision": decision,
            "path": _normalize_text(record.get("path")),
            "confidence": (
                float(record["confidence"])
                if isinstance(record.get("confidence"), (int, float))
                and not isinstance(record.get("confidence"), bool)
                else None
            ),
            "rationale": _normalize_text(record.get("rationale") or record.get("reason")),
            "profile_url": profile_url,
            **_profile_context(profile),
        }
        if decision in SAVE_DECISIONS:
            saved.append(example)
        elif decision:
            rejected.append(example)

    if not saved:
        saved = _fallback_candidate_examples(report_input, "saved_candidate_summaries")
    if not rejected:
        rejected = _fallback_candidate_examples(report_input, "rejected_candidate_summaries")

    return {
        "saved_examples": saved[:MAX_SAVED_EXAMPLES],
        "rejected_examples": rejected[:MAX_REJECTED_EXAMPLES],
    }


def _summarize_adaptation_event(event: dict) -> dict:
    report = event.get("report") if isinstance(event.get("report"), dict) else {}
    metrics = event.get("metrics") if isinstance(event.get("metrics"), dict) else {}
    source_payload = (
        event.get("source_payload") if isinstance(event.get("source_payload"), dict) else {}
    )
    batch_report = (
        source_payload.get("batch_report")
        if isinstance(source_payload.get("batch_report"), dict)
        else {}
    )
    # Summary fallback chain: LinkedIn shape (report.summary / event.message)
    # → AdaptationDecision shape (event.rationale, source_payload.batch_report.summary).
    # Without the rationale fallback, every non-LinkedIn adaptation event
    # would render in market-intel reports with an empty summary line.
    summary = (
        _normalize_text(report.get("summary"))
        or _normalize_text(event.get("message"))
        or _normalize_text(batch_report.get("summary"))
        or _normalize_text(event.get("rationale"))
    )
    return {
        "timestamp": _normalize_text(event.get("timestamp")),
        "event": _normalize_text(event.get("event")),
        "source": _normalize_text(event.get("source")),
        "string_id": str(event.get("string_id", "")),
        "page": int(event.get("page", 0) or 0),
        "phase": _normalize_text(event.get("phase")),
        "action": _normalize_text(event.get("action")),
        "lane": _normalize_text(event.get("lane")),
        "block": _normalize_text(event.get("block")),
        "severity": _normalize_text(event.get("severity")),
        "alert_type": _normalize_text(event.get("alert_type")),
        "rationale": _normalize_text(event.get("rationale")),
        "saves": int(metrics.get("saves", 0) or 0),
        "candidates_discovered": int(metrics.get("candidates_discovered", 0) or 0),
        "summary": summary,
    }


def _build_adaptation_timeline(run_log: list[dict]) -> list[dict]:
    selected = [
        _summarize_adaptation_event(event)
        for event in run_log
        if isinstance(event, dict) and event.get("event") in ADAPTATION_EVENTS
    ]
    ordered = _sort_by_generated_at(
        [(_parse_dt(item.get("timestamp")), item) for item in selected]
    )
    return ordered[-MAX_ADAPTATION_EVENTS:]


def _build_saved_candidate_patterns(candidate_evidence: dict, profile_summaries: list[dict]) -> dict:
    saved_examples = candidate_evidence.get("saved_examples", [])
    employers = Counter()
    titles = Counter()
    archetypes = Counter()
    seniority_notes: list[str] = []
    profile_lookup = {
        _normalize_text(item.get("name")).lower(): item
        for item in profile_summaries
        if isinstance(item, dict) and _normalize_text(item.get("name"))
    }

    for example in saved_examples:
        profile = profile_lookup.get(
            _normalize_text(example.get("candidate_name")).lower()
        ) or {}
        experiences = profile.get("experiences") or []
        latest = experiences[0] if experiences and isinstance(experiences[0], dict) else {}
        employer = _normalize_text(example.get("current_company") or latest.get("company"))
        title = _normalize_text(example.get("current_title") or latest.get("title"))
        if employer:
            employers[employer] += 1
        if title:
            titles[title] += 1
        text = " ".join(
            [
                _normalize_text(example.get("headline")),
                _normalize_text(example.get("rationale")),
                _normalize_text(example.get("summary_excerpt")),
            ]
        )
        anchors = extract_dominant_anchors(text, limit=3)
        if anchors:
            archetypes[", ".join(anchors[:2])] += 1
        if title and title not in seniority_notes:
            seniority_notes.append(f"Saved example titles included: {title}")

    return {
        "standout_candidates": [
            {
                "name": _normalize_text(item.get("candidate_name")),
                "why": _normalize_text(item.get("rationale"))[:220],
            }
            for item in saved_examples[:5]
            if _normalize_text(item.get("candidate_name"))
        ],
        "common_employers": [
            {"employer": employer, "count": count, "note": "Recurring among saved examples."}
            for employer, count in employers.most_common(5)
        ],
        "common_titles": [
            {"title_family": title, "count": count, "note": "Recurring saved-title pattern."}
            for title, count in titles.most_common(5)
        ],
        "archetype_distribution": [
            {"archetype": label, "count": count, "note": "Derived from saved-candidate evidence."}
            for label, count in archetypes.most_common(5)
        ],
        "seniority_notes": seniority_notes[:5],
    }


def _reconstruct_report_analysis(
    batch: MarketEvidenceBatch,
    candidate_evidence: dict,
    adaptation_timeline: list[dict],
) -> dict:
    work_units = batch.runtime_summary.get("work_units", [])
    families = _bounded_search_memory_summary(None, batch.search_memory).get("families", [])
    winning_lanes: list[dict] = []
    underperforming_lanes: list[dict] = []

    family_by_key = {
        _normalize_text(item.get("family_key")): item
        for item in families
        if isinstance(item, dict) and _normalize_text(item.get("family_key"))
    }

    for work_unit in work_units:
        if not isinstance(work_unit, dict):
            continue
        saves = int(work_unit.get("saves_count", 0) or 0)
        candidates = int(work_unit.get("candidate_volume", 0) or 0)
        save_rate = round(saves / max(candidates, 1), 4)
        family = family_by_key.get(_normalize_text(work_unit.get("family_key")), {})
        lane_label = _normalize_text(work_unit.get("display_name")) or _normalize_text(
            work_unit.get("family_key")
        )
        dominant_anchors = family.get("dominant_anchors", [])
        if saves > 0:
            winning_lanes.append(
                {
                    "lane": lane_label,
                    "string_ids": [_normalize_text(work_unit.get("source_unit_id"))],
                    "candidate_examples": [
                        _normalize_text(item.get("candidate_name"))
                        for item in candidate_evidence.get("saved_examples", [])[:3]
                        if _normalize_text(item.get("candidate_name"))
                    ],
                    "evidence": f"{saves} saves across {candidates} evaluated candidates ({save_rate:.1%} save rate).",
                    "why_it_worked": (
                        "Dominant anchors suggest strong fit around "
                        f"{', '.join(dominant_anchors[:4])}."
                        if dominant_anchors
                        else "This lane produced repeatable save signal."
                    ),
                    "recommended_action": "Keep this lane active and test adjacent anchor variants.",
                }
            )
        elif candidates >= 10:
            underperforming_lanes.append(
                {
                    "lane": lane_label,
                    "string_ids": [_normalize_text(work_unit.get("source_unit_id"))],
                    "issue": f"0 saves across {candidates} evaluated candidates.",
                    "evidence": (
                        f"Candidate volume was {candidates} with "
                        f"{int(work_unit.get('facial_no_count', 0) or 0)} facial NO decisions."
                    ),
                    "recommended_action": "Tighten the lane with more explicit builder and workflow constraints.",
                }
            )

    bias_alerts = [
        item for item in adaptation_timeline if _normalize_text(item.get("event")) == "bias_alert"
    ]
    coverage_gaps = []
    if not winning_lanes:
        coverage_gaps.append(
            {
                "gap": "No clearly winning lane emerged from reconstructed evidence.",
                "why_it_matters": "The run needs at least one strong signal family to prioritize.",
                "suggested_search_strategy": "Re-run using the top search-memory anchors with narrower workflow language.",
            }
        )
    noise_patterns = []
    if underperforming_lanes:
        noise_patterns.append(
            {
                "pattern": "High-volume lanes with zero saves",
                "evidence": underperforming_lanes[0]["evidence"],
                "mitigation": "Require more explicit hands-on builder evidence in the boolean.",
            }
        )
    if bias_alerts:
        noise_patterns.append(
            {
                "pattern": "Bias-monitor intervention",
                "evidence": _normalize_text(bias_alerts[-1].get("summary")),
                "mitigation": "Review calibration examples before reusing this lane family.",
            }
        )

    candidate_patterns = _build_saved_candidate_patterns(
        candidate_evidence,
        [],
    )

    try_next = [item["lane"] for item in winning_lanes[:3] if _normalize_text(item.get("lane"))]
    if not try_next:
        try_next = [
            ", ".join(item.get("dominant_anchors", [])[:3])
            for item in families[:3]
            if item.get("dominant_anchors")
        ]
    avoid_next = [item["lane"] for item in underperforming_lanes[:3] if _normalize_text(item.get("lane"))]
    search_priorities = [item["lane"] for item in winning_lanes[:3] if _normalize_text(item.get("lane"))]
    additional_terms = []
    for family in families[:3]:
        additional_terms.extend(
            anchor for anchor in family.get("dominant_anchors", [])[:3] if anchor
        )

    return {
        "winning_lanes": winning_lanes[:MAX_WINNING_LANES],
        "underperforming_lanes": underperforming_lanes[:MAX_UNDERPERFORMING_LANES],
        "coverage_gaps": coverage_gaps[:MAX_GAPS],
        "noise_patterns": noise_patterns[:MAX_NOISE_PATTERNS],
        "saved_candidate_patterns": candidate_patterns,
        "adaptation_assessment": {
            "summary": (
                f"Reconstructed from {len(adaptation_timeline)} adaptation events and "
                f"{len(candidate_evidence.get('saved_examples', []))} saved examples."
            ),
            "effective_refinements": [
                _normalize_text(item.get("summary") or item.get("rationale"))
                for item in adaptation_timeline
                if _normalize_text(item.get("action")) in {"narrow", "continue"}
            ][:5],
            "questionable_or_skipped": [
                _normalize_text(item.get("summary") or item.get("rationale"))
                for item in adaptation_timeline
                if _normalize_text(item.get("event")) in {"string_error", "early_exit"}
            ][:5],
            "operational_notes": [
                _normalize_text(item.get("summary") or item.get("rationale"))
                for item in adaptation_timeline
                if _normalize_text(item.get("summary") or item.get("rationale"))
            ][:5],
        },
        "recommendations": {
            "try_next": try_next[:5],
            "avoid_next": avoid_next[:5],
            "prioritize_pipeline": [
                _normalize_text(item.get("candidate_name"))
                for item in candidate_evidence.get("saved_examples", [])[:5]
                if _normalize_text(item.get("candidate_name"))
            ],
        },
        "brief_iteration_hints": {
            "instructions": ["This section was reconstructed from raw run artifacts."],
            "search_priorities": search_priorities[:5],
            "additional_search_terms": list(dict.fromkeys(additional_terms))[:8],
            "intake_notes": "Use reconstructed market-intel context with caution; original debrief unavailable.",
        },
    }


def build_linkedin_research_input_packet(
    *,
    batch: MarketEvidenceBatch,
    report_input: dict | None,
    profile_summaries: list[dict],
    snippets: list[dict],
    run_log: list[dict],
    reconstruct_report_analysis: bool,
    artifact_paths: dict[str, str],
    research_input_path: Path,
    live_advisory_summary: dict | None,
) -> dict:
    report = batch.report or {}
    candidate_evidence = _build_candidate_examples(
        batch.final_judgments,
        profile_summaries,
        snippets,
        report_input,
    )
    adaptation_timeline = _build_adaptation_timeline(run_log)

    if report:
        report_analysis = {
            "winning_lanes": [
                item for item in report.get("winning_lanes", []) if isinstance(item, dict)
            ][:MAX_WINNING_LANES],
            "underperforming_lanes": [
                item
                for item in report.get("underperforming_lanes", [])
                if isinstance(item, dict)
            ][:MAX_UNDERPERFORMING_LANES],
            "coverage_gaps": [
                item for item in report.get("coverage_gaps", []) if isinstance(item, dict)
            ][:MAX_GAPS],
            "noise_patterns": [
                item for item in report.get("noise_patterns", []) if isinstance(item, dict)
            ][:MAX_NOISE_PATTERNS],
            "saved_candidate_patterns": dict(report.get("saved_candidate_patterns", {})),
            "adaptation_assessment": dict(report.get("adaptation_assessment", {})),
            "recommendations": dict(report.get("recommendations", {})),
            "brief_iteration_hints": dict(report.get("brief_iteration_hints", {})),
        }
        context_quality = "original_report"
        analysis_provenance = "original_report"
    elif reconstruct_report_analysis:
        report_analysis = _reconstruct_report_analysis(
            batch,
            candidate_evidence,
            adaptation_timeline,
        )
        context_quality = "reconstructed_report"
        analysis_provenance = "reconstructed_from_raw"
    else:
        report_analysis = {
            "winning_lanes": [],
            "underperforming_lanes": [],
            "coverage_gaps": [],
            "noise_patterns": [],
            "saved_candidate_patterns": {},
            "adaptation_assessment": {},
            "recommendations": {},
            "brief_iteration_hints": {},
        }
        context_quality = "raw_only"
        analysis_provenance = "none"

    top_sp = _top_string_performance(report_input, report, batch.runtime_summary)
    deterministic_snapshot = {
        "run_metadata": dict(
            (report_input or {}).get("run_metadata")
            or report.get("run_metadata", {})
        ),
        "metrics_summary": dict(
            (report_input or {}).get("metrics_summary")
            or report.get("metrics_summary", {})
            or batch.metrics_summary
        ),
        "string_performance": top_sp,
        "lane_execution_summary": _lane_execution_summary(report_input, top_sp),
        "search_memory_summary": _bounded_search_memory_summary(
            report_input,
            batch.search_memory,
        ),
        "bias_monitor_summary": _normalize_text(
            (report_input or {}).get("bias_monitor_summary")
        ),
        "lane_evidence": _lane_evidence_from_snapshot(
            report_input, report_analysis, top_sp
        ),
    }

    packet = {
        "context_metadata": {
            "run_ref": batch.run_ref,
            "source": batch.source,
            "output_dir": batch.output_dir,
            "generated_at": batch.generated_at,
            "brief_version": batch.brief_version,
            "context_quality": context_quality,
            "analysis_provenance": analysis_provenance,
            "artifact_paths_used": artifact_paths,
            "research_input_path": str(research_input_path),
        },
        "deterministic_snapshot": deterministic_snapshot,
        "report_analysis": report_analysis,
        "candidate_evidence": candidate_evidence,
        "adaptation_timeline": adaptation_timeline,
    }
    if isinstance(live_advisory_summary, dict) and live_advisory_summary:
        packet["live_advisory_summary"] = live_advisory_summary
    return packet


def maybe_build_and_persist_research_packet(
    batch: MarketEvidenceBatch,
    *,
    reconstruct_report_analysis: bool,
) -> MarketEvidenceBatch:
    if batch.source != "linkedin":
        return batch

    output_dir = Path(batch.output_dir)
    research_input_path = output_dir / "market-intel-research-input.json"
    report_input = _load_optional_json(output_dir / "run-report-input.json")
    run_log = _load_optional_jsonl(output_dir / "run_log.jsonl")
    profile_summaries = _load_optional_jsonl(output_dir / "profile_summaries.jsonl")
    snippets = _load_optional_jsonl(output_dir / "snippets.jsonl")
    live_advisory_summary = _load_optional_json(output_dir / "market_intel" / "live-summary.json")

    artifact_paths = {
        name: str(path)
        for name, path in {
            "run_report_input": output_dir / "run-report-input.json",
            "run_report": output_dir / "run-report.json",
            "final_judgments": output_dir / "final_judgments.jsonl",
            "runtime_state": output_dir / "runtime_state.sqlite3",
            "run_log": output_dir / "run_log.jsonl",
            "profile_summaries": output_dir / "profile_summaries.jsonl",
            "snippets": output_dir / "snippets.jsonl",
            "live_advisory_summary": output_dir / "market_intel" / "live-summary.json",
            "live_advisories": output_dir / "market_intel" / "live-advisories.jsonl",
        }.items()
        if path.exists()
    }
    search_memory_files = sorted(output_dir.glob("search_memory-*.json"))
    if search_memory_files:
        artifact_paths["search_memory"] = str(search_memory_files[0])

    packet = build_linkedin_research_input_packet(
        batch=batch,
        report_input=report_input,
        profile_summaries=profile_summaries,
        snippets=snippets,
        run_log=run_log,
        reconstruct_report_analysis=reconstruct_report_analysis,
        artifact_paths=artifact_paths,
        research_input_path=research_input_path,
        live_advisory_summary=live_advisory_summary,
    )
    write_json(research_input_path, packet)
    batch.report_input = report_input
    batch.research_context = packet
    batch.research_input_path = str(research_input_path)
    metadata = packet.get("context_metadata", {})
    batch.context_quality = _normalize_text(metadata.get("context_quality"))
    batch.analysis_provenance = _normalize_text(metadata.get("analysis_provenance"))
    return batch


def _aggregate_metrics(evidence_batches: list[MarketEvidenceBatch]) -> dict:
    per_channel: dict[str, dict] = defaultdict(lambda: {"run_count": 0, "saved": 0, "candidate_volume": 0})
    total_saved = 0
    total_candidates = 0
    for batch in evidence_batches:
        metrics = batch.metrics_summary
        per_channel[batch.source]["run_count"] += int(metrics.get("run_count", 0) or 1)
        per_channel[batch.source]["saved"] += int(metrics.get("saved", 0))
        per_channel[batch.source]["candidate_volume"] += int(metrics.get("candidate_volume", 0))
        total_saved += int(metrics.get("saved", 0))
        total_candidates += int(metrics.get("candidate_volume", 0))
    return {
        "run_count": len(evidence_batches),
        "saved_count": total_saved,
        "candidate_volume": total_candidates,
        "save_rate": round(total_saved / max(total_candidates, 1), 4),
        "per_channel_summary": dict(per_channel),
    }


def _lane_rollup(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    counts: dict[str, dict] = defaultdict(lambda: {"run_refs": set(), "winning_count": 0, "underperforming_count": 0, "anchors": Counter()})
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        for winner in analysis.get("winning_lanes", []):
            lane = _normalize_text(winner.get("lane"))
            if not lane:
                continue
            counts[lane]["run_refs"].add(batch.run_ref)
            counts[lane]["winning_count"] += 1
            counts[lane]["anchors"].update(
                extract_dominant_anchors(
                    " ".join(
                        [
                            _normalize_text(winner.get("why_it_worked")),
                            _normalize_text(winner.get("evidence")),
                        ]
                    ),
                    limit=4,
                )
            )
        for loser in analysis.get("underperforming_lanes", []):
            lane = _normalize_text(loser.get("lane"))
            if not lane:
                continue
            counts[lane]["run_refs"].add(batch.run_ref)
            counts[lane]["underperforming_count"] += 1
    rollup = []
    for lane, stats in counts.items():
        rollup.append(
            {
                "lane": lane,
                "run_refs": sorted(stats["run_refs"]),
                "winning_count": stats["winning_count"],
                "underperforming_count": stats["underperforming_count"],
                "dominant_anchors": [anchor for anchor, _ in stats["anchors"].most_common(5)],
            }
        )
    return sorted(rollup, key=lambda item: (-item["winning_count"], item["lane"]))[:8]


def _rollup_counter(evidence_batches: list[MarketEvidenceBatch], path: tuple[str, ...], label_key: str) -> list[dict]:
    counter = Counter()
    for batch in evidence_batches:
        packet = batch.research_context or {}
        current: Any = packet
        for key in path:
            current = current.get(key, {}) if isinstance(current, dict) else {}
        if not isinstance(current, list):
            continue
        for item in current:
            if not isinstance(item, dict):
                continue
            label = _normalize_text(item.get(label_key))
            count = int(item.get("count", 1) or 1)
            if label:
                counter[label] += count
    return [
        {"label": label, "count": count}
        for label, count in counter.most_common(8)
    ]


def _select_run_packets(evidence_batches: list[MarketEvidenceBatch]) -> tuple[list[dict], list[MarketEvidenceBatch]]:
    candidates = [batch for batch in evidence_batches if isinstance(batch.research_context, dict) and batch.research_context]
    if not candidates:
        return [], []

    def _metric(batch: MarketEvidenceBatch, key: str) -> float:
        value = batch.metrics_summary.get(key, 0)
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    ordered: list[MarketEvidenceBatch] = []
    newest = max(candidates, key=lambda batch: _parse_dt(batch.generated_at) or datetime.min)
    ordered.append(newest)
    highest_signal = max(
        candidates,
        key=lambda batch: (_metric(batch, "saved"), _metric(batch, "save_rate")),
    )
    ordered.append(highest_signal)
    highest_volume = max(candidates, key=lambda batch: _metric(batch, "candidate_volume"))
    ordered.append(highest_volume)

    deduped: list[MarketEvidenceBatch] = []
    seen: set[str] = set()
    for batch in ordered:
        if batch.run_ref in seen:
            continue
        seen.add(batch.run_ref)
        deduped.append(batch)
        if len(deduped) >= MAX_FULL_PACKETS:
            break

    packets = [batch.research_context for batch in deduped if batch.research_context]
    return packets, deduped


def _dedupe_projection_records(items: list[dict], *, key_fields: tuple[str, ...]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = tuple(_normalize_text(item.get(field)).lower() for field in key_fields)
        if not any(key):
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _observed_success_patterns(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        for winner in analysis.get("winning_lanes", [])[:2]:
            lane = _normalize_text(winner.get("lane"))
            summary = _normalize_text(winner.get("why_it_worked")) or _normalize_text(
                winner.get("evidence")
            )
            if lane:
                records.append(
                    {
                        "label": lane,
                        "summary": summary or "Observed as a winning lane in completed sourcing.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
        saved_patterns = analysis.get("saved_candidate_patterns", {})
        for archetype in (saved_patterns.get("archetype_distribution") or [])[:2]:
            label = _normalize_text(archetype.get("archetype"))
            summary = _normalize_text(archetype.get("note"))
            if label:
                records.append(
                    {
                        "label": label,
                        "summary": summary or "Repeatedly surfaced among saved candidates.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _observed_failures_and_false_negatives(
    evidence_batches: list[MarketEvidenceBatch],
) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        for item in analysis.get("underperforming_lanes", [])[:3]:
            lane = _normalize_text(item.get("lane"))
            summary = _normalize_text(item.get("issue")) or _normalize_text(
                item.get("evidence")
            )
            if lane:
                records.append(
                    {
                        "label": lane,
                        "summary": summary or "Observed as a weak or noisy lane.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
        for gap in analysis.get("coverage_gaps", [])[:3]:
            label = _normalize_text(gap.get("gap"))
            summary = _normalize_text(gap.get("why_it_matters")) or _normalize_text(
                gap.get("suggested_search_strategy")
            )
            if label:
                records.append(
                    {
                        "label": label,
                        "summary": summary or "Coverage gap that may hide a false negative pool.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _title_and_archetype_blind_spots(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        for gap in analysis.get("coverage_gaps", [])[:4]:
            label = _normalize_text(gap.get("gap"))
            strategy = _normalize_text(gap.get("suggested_search_strategy"))
            if label:
                records.append(
                    {
                        "label": label,
                        "summary": strategy or "Potential blind spot in title or archetype coverage.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
        saved_patterns = analysis.get("saved_candidate_patterns", {})
        for title_family in (saved_patterns.get("common_titles") or [])[:2]:
            label = _normalize_text(title_family.get("title_family"))
            summary = _normalize_text(title_family.get("note"))
            if label:
                records.append(
                    {
                        "label": label,
                        "summary": summary or "Observed title family worth checking for adjacent variants.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _employer_signal_gaps(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        common_employers = (
            analysis.get("saved_candidate_patterns", {}).get("common_employers", [])
        )
        if common_employers:
            continue
        records.append(
            {
                "label": "Employer clustering remains thin",
                "summary": "Internal sourcing evidence does not yet show repeat employer clusters for this run.",
                "supporting_run_refs": [batch.run_ref],
            }
        )
    if not records and evidence_batches:
        return [
            {
                "label": "Employer coverage could still broaden",
                "summary": "Even with repeat employers present, external research should test whether adjacent employer clusters are missing.",
                "supporting_run_refs": [batch.run_ref for batch in evidence_batches[:3]],
            }
        ]
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:5]


def _candidate_evidence_coverage_gaps(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        candidate_evidence = (
            packet.get("candidate_evidence", {}) if isinstance(packet, dict) else {}
        )
        saved_count = len(candidate_evidence.get("saved_examples", []))
        rejected_count = len(candidate_evidence.get("rejected_examples", []))
        if saved_count >= 3 and rejected_count >= 3:
            continue
        records.append(
            {
                "label": "Profile-level evidence is sparse",
                "summary": (
                    f"Only {saved_count} saved and {rejected_count} rejected candidate examples were available for detailed analysis."
                ),
                "supporting_run_refs": [batch.run_ref],
            }
        )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:5]


def _adaptation_lessons(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        assessment = analysis.get("adaptation_assessment", {}) if isinstance(analysis, dict) else {}
        summary = _normalize_text(assessment.get("summary"))
        if summary:
            records.append(
                {
                    "label": "Adaptation assessment",
                    "summary": summary,
                    "supporting_run_refs": [batch.run_ref],
                }
            )
        for lesson in (assessment.get("effective_refinements") or [])[:3]:
            text = _normalize_text(lesson)
            if text:
                records.append(
                    {
                        "label": "Effective refinement",
                        "summary": text,
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
        timeline = packet.get("adaptation_timeline", []) if isinstance(packet, dict) else []
        for event in timeline[:3]:
            action = _normalize_text(event.get("action"))
            # P3.7: the timeline producer emits "summary" — "report_summary"
            # never existed, so this always fell to event-name noise.
            message = _normalize_text(event.get("summary") or event.get("event"))
            if action or message:
                records.append(
                    {
                        "label": action or "Adaptation event",
                        "summary": message or "Adaptation checkpoint recorded during sourcing.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _hidden_pool_risk_signals(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        if analysis.get("coverage_gaps"):
            records.append(
                {
                    "label": "Coverage gaps suggest hidden supply",
                    "summary": (
                        f"{len(analysis.get('coverage_gaps', []))} coverage gap(s) indicate the run may have missed adjacent title or archetype pools."
                    ),
                    "supporting_run_refs": [batch.run_ref],
                }
            )
        search_summary = (
            packet.get("search_memory_summary", {}) if isinstance(packet, dict) else {}
        )
        families = search_summary.get("families", []) if isinstance(search_summary, dict) else []
        edge_case_families = [
            family
            for family in families
            if _normalize_text(family.get("novelty_bucket")).lower() == "edge_case"
            and int(family.get("saves", 0) or 0) > 0
        ]
        if edge_case_families:
            records.append(
                {
                    "label": "Saved signal is concentrated in novelty-heavy families",
                    "summary": (
                        f"{len(edge_case_families)} search-memory family/families with edge-case novelty produced saves, suggesting hidden-pool risk."
                    ),
                    "supporting_run_refs": [batch.run_ref],
                }
            )
        candidate_evidence = (
            packet.get("candidate_evidence", {}) if isinstance(packet, dict) else {}
        )
        if not candidate_evidence.get("saved_examples"):
            records.append(
                {
                    "label": "Sparse saved-profile evidence limits market visibility",
                    "summary": "The run lacks enough detailed saved profiles to rule out missed title families or adjacent pools.",
                    "supporting_run_refs": [batch.run_ref],
                }
            )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _edge_case_lane_signals(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        deterministic = (
            packet.get("deterministic_snapshot", {}) if isinstance(packet, dict) else {}
        )
        string_performance = (
            deterministic.get("string_performance", [])
            if isinstance(deterministic, dict)
            else []
        )
        for item in string_performance[:12]:
            if not isinstance(item, dict):
                continue
            novelty_bucket = _normalize_text(item.get("novelty_bucket")).lower()
            saves = int(item.get("saves", 0) or 0)
            label = _normalize_text(item.get("name")) or _normalize_text(item.get("family_key"))
            if novelty_bucket != "edge_case" or not label:
                continue
            summary = (
                f"Edge-case lane with {saves} save(s) across {int(item.get('candidates_count', 0) or 0)} candidate(s)."
                if saves > 0
                else "Edge-case lane was explored but did not yet produce saves."
            )
            records.append(
                {
                    "label": label,
                    "summary": summary,
                    "supporting_run_refs": [batch.run_ref],
                }
            )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _novelty_mix_summary(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    counts: Counter[str] = Counter()
    run_refs_by_bucket: defaultdict[str, set[str]] = defaultdict(set)
    for batch in evidence_batches:
        packet = batch.research_context or {}
        deterministic = (
            packet.get("deterministic_snapshot", {}) if isinstance(packet, dict) else {}
        )
        for item in deterministic.get("string_performance", []) if isinstance(deterministic, dict) else []:
            if not isinstance(item, dict):
                continue
            bucket = _normalize_text(item.get("novelty_bucket")).lower() or "unknown"
            counts[bucket] += 1
            run_refs_by_bucket[bucket].add(batch.run_ref)
    records = [
        {
            "label": bucket.replace("_", " "),
            "summary": f"{count} string(s) fell into the {bucket.replace('_', ' ')} novelty bucket across observed runs.",
            "supporting_run_refs": sorted(run_refs_by_bucket[bucket]),
        }
        for bucket, count in counts.most_common()
    ]
    return records[:5]


def _false_negative_hypotheses_from_internal_evidence(
    evidence_batches: list[MarketEvidenceBatch],
) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        for gap in analysis.get("coverage_gaps", [])[:4]:
            if not isinstance(gap, dict):
                continue
            label = _normalize_text(gap.get("gap"))
            strategy = _normalize_text(gap.get("suggested_search_strategy"))
            if label:
                records.append(
                    {
                        "label": f"Hidden pool hypothesis: {label}",
                        "summary": strategy or "The current search may be missing a title or archetype slice of the market.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
        for item in analysis.get("underperforming_lanes", [])[:3]:
            lane = _normalize_text(item.get("lane"))
            issue = _normalize_text(item.get("issue"))
            if lane and issue:
                records.append(
                    {
                        "label": f"False-negative risk in {lane}",
                        "summary": f"{issue} This may reflect title/archetype mismatch rather than a truly empty pool.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _self_labeling_risk_indicators(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        saved_patterns = analysis.get("saved_candidate_patterns", {})
        for title_family in (saved_patterns.get("common_titles") or [])[:4]:
            label = _normalize_text(title_family.get("title_family"))
            note = _normalize_text(title_family.get("note"))
            if label:
                records.append(
                    {
                        "label": label,
                        "summary": note or "Observed title family suggests candidates may self-label differently than the target title.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
        for gap in analysis.get("coverage_gaps", [])[:3]:
            summary = _normalize_text(gap.get("suggested_search_strategy"))
            if summary:
                records.append(
                    {
                        "label": "Title self-labeling risk",
                        "summary": summary,
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _title_fragmentation_indicators(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        common_titles = analysis.get("saved_candidate_patterns", {}).get("common_titles", [])
        if len(common_titles) >= 2:
            titles = ", ".join(
                _normalize_text(item.get("title_family"))
                for item in common_titles[:4]
                if _normalize_text(item.get("title_family"))
            )
            if titles:
                records.append(
                    {
                        "label": "Observed title fragmentation among saves",
                        "summary": f"Saved candidates already span multiple title families: {titles}.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
        search_summary = (
            packet.get("search_memory_summary", {}) if isinstance(packet, dict) else {}
        )
        families = search_summary.get("families", []) if isinstance(search_summary, dict) else []
        fragmented = [
            item
            for item in families
            if len(item.get("dominant_anchors", []) or []) >= 3
        ]
        if fragmented:
            records.append(
                {
                    "label": "Search-memory anchor spread suggests fragmented labeling",
                    "summary": "Dominant anchors vary materially across families, increasing the odds that relevant candidates self-present under different title patterns.",
                    "supporting_run_refs": [batch.run_ref],
                }
            )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _archetype_boundary_confusion(evidence_batches: list[MarketEvidenceBatch]) -> list[dict]:
    records: list[dict] = []
    for batch in evidence_batches:
        packet = batch.research_context or {}
        analysis = packet.get("report_analysis", {}) if isinstance(packet, dict) else {}
        saved_patterns = analysis.get("saved_candidate_patterns", {})
        for archetype in (saved_patterns.get("archetype_distribution") or [])[:4]:
            label = _normalize_text(archetype.get("archetype"))
            note = _normalize_text(archetype.get("note"))
            if label:
                records.append(
                    {
                        "label": label,
                        "summary": note or "Observed archetype may overlap with adjacent pools that are easy to confuse in sourcing.",
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
        for note in (saved_patterns.get("seniority_notes") or [])[:2]:
            text = _normalize_text(note)
            if text:
                records.append(
                    {
                        "label": "Seniority / archetype boundary risk",
                        "summary": text,
                        "supporting_run_refs": [batch.run_ref],
                    }
                )
    return _dedupe_projection_records(records, key_fields=("label", "summary"))[:8]


def _edge_case_context(evidence_batches: list[MarketEvidenceBatch]) -> dict:
    return {
        "hidden_pool_risk_signals": _hidden_pool_risk_signals(evidence_batches),
        "edge_case_lane_signals": _edge_case_lane_signals(evidence_batches),
        "novelty_mix_summary": _novelty_mix_summary(evidence_batches),
        "false_negative_hypotheses_from_internal_evidence": _false_negative_hypotheses_from_internal_evidence(
            evidence_batches
        ),
        "self_labeling_risk_indicators": _self_labeling_risk_indicators(
            evidence_batches
        ),
        "title_fragmentation_indicators": _title_fragmentation_indicators(
            evidence_batches
        ),
        "archetype_boundary_confusion": _archetype_boundary_confusion(
            evidence_batches
        ),
        "candidate_evidence_blind_spots": _candidate_evidence_coverage_gaps(
            evidence_batches
        ),
    }


def build_research_context_bundle(
    market_identity: MarketIdentity,
    evidence_batches: list[MarketEvidenceBatch],
) -> dict:
    run_packets, selected_batches = _select_run_packets(evidence_batches)
    selected_refs = {batch.run_ref for batch in selected_batches}
    omitted_batches = [batch for batch in evidence_batches if batch.run_ref not in selected_refs]

    omitted_rollup = {
        "additional_run_count": len(omitted_batches),
        "run_refs": [batch.run_ref for batch in omitted_batches],
        "lane_highlights": _lane_rollup(omitted_batches),
        "coverage_gap_themes": _rollup_counter(
            omitted_batches,
            ("report_analysis", "coverage_gaps"),
            "gap",
        ),
        "noise_pattern_themes": _rollup_counter(
            omitted_batches,
            ("report_analysis", "noise_patterns"),
            "pattern",
        ),
    }

    return {
        "market_identity": market_identity.to_dict(),
        "cross_run_aggregate": {
            **_aggregate_metrics(evidence_batches),
            "lane_trend_summary": _lane_rollup(evidence_batches),
            "employer_rollup": _rollup_counter(
                evidence_batches,
                ("report_analysis", "saved_candidate_patterns", "common_employers"),
                "employer",
            ),
            "archetype_rollup": _rollup_counter(
                evidence_batches,
                ("report_analysis", "saved_candidate_patterns", "archetype_distribution"),
                "archetype",
            ),
        },
        "observed_success_patterns": _observed_success_patterns(evidence_batches),
        "observed_failures_and_false_negatives": _observed_failures_and_false_negatives(
            evidence_batches
        ),
        "title_and_archetype_blind_spots": _title_and_archetype_blind_spots(
            evidence_batches
        ),
        "employer_signal_gaps": _employer_signal_gaps(evidence_batches),
        "candidate_evidence_coverage_gaps": _candidate_evidence_coverage_gaps(
            evidence_batches
        ),
        "adaptation_lessons": _adaptation_lessons(evidence_batches),
        "edge_case_context": _edge_case_context(evidence_batches),
        "run_packets": run_packets,
        "historical_rollup": omitted_rollup,
    }
