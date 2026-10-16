"""Search policy calculations; no retained run, browser, or persistence state."""
from __future__ import annotations

import re as _re
from typing import Any
from shared import config
from shared.schemas import CandidateSnippet, GlanceResult, Progress, SearchString
from linkedin.search_intelligence import (
    LinkedInSearchVariant,
    LinkedInPageInsights,
    LinkedInDriftAssessment,
    LinkedInVariantSnapshot,
    LinkedInStructuredFilters,
    result_window_for_count,
    _copy_filters,
)

_SENIORITY_PREFIXES = _re.compile(
    r'^(senior|sr\.?|lead|principal|staff|junior|jr\.?|associate|chief|head of|director of|vp of)\s+',
    _re.IGNORECASE,
)


_TRAILING_LEVELS = _re.compile(r'\s+(i{1,4}|iv|[1-6])$', _re.IGNORECASE)


_TITLE_SYNONYMS: dict[str, str] = {
    "software developer": "software engineer",
    "swe": "software engineer",
    "dev ops": "devops engineer",
    "data science": "data scientist",
    "programme manager": "program manager",
}


def _normalize_title_family(title: str) -> str:
    """Collapse a job title to a canonical family for clustering."""
    t = title.strip().lower()
    t = _SENIORITY_PREFIXES.sub('', t)
    t = _TRAILING_LEVELS.sub('', t)
    t = t.strip()
    for pattern, canonical in _TITLE_SYNONYMS.items():
        if t == pattern:
            t = canonical
            break
    return t


def _remaining_queued_strings(progress: Progress, *, current_string_id: int) -> int:
    return sum(
        1
        for string in progress.strings
        if string.id != current_string_id
        and string.status not in {"done", "skipped"}
    )


def _initial_search_mode(result_count: int) -> str:
    return "recon" if result_count >= 500 else "paginate"


def _page_candidates(all_candidates: list[dict], page_num: int) -> list[dict]:
    return [candidate for candidate in all_candidates if candidate.get("page") == page_num]


def _build_page_insights(
    *,
    page_num: int,
    result_count: int,
    preview_snippets: list[CandidateSnippet],
    all_candidates: list[dict],
    glance_result: GlanceResult | None,
) -> LinkedInPageInsights:
    title_counts: dict[str, int] = {}
    company_counts: dict[str, int] = {}
    for snippet in preview_snippets:
        title = _normalize_title_family(snippet.current_title or snippet.headline or "")
        if title:
            title_counts[title] = title_counts.get(title, 0) + 1
        company = (snippet.current_company or "").strip()
        if company:
            company_counts[company] = company_counts.get(company, 0) + 1

    page_candidates = _page_candidates(all_candidates, page_num)
    title_signal_counts: dict[str, int] = {}
    company_signal_counts: dict[str, int] = {}
    settled_positive_outcomes = {"save", "save_failed", "review"}
    for candidate in page_candidates:
        if candidate.get("outcome") not in settled_positive_outcomes:
            continue
        title = _normalize_title_family(str(candidate.get("title") or ""))
        if title:
            title_signal_counts[title] = title_signal_counts.get(title, 0) + 1
        company = _normalize_company_rail_label(str(candidate.get("company") or ""))
        if company:
            company_signal_counts[company] = company_signal_counts.get(company, 0) + 1
    signal_anchors = [
        f"{candidate.get('title') or 'Unknown'} at {candidate.get('company') or 'Unknown'}"
        for candidate in page_candidates
        if candidate.get("outcome") in settled_positive_outcomes
    ][:5]
    noise_anchors = [
        f"{candidate.get('title') or 'Unknown'} at {candidate.get('company') or 'Unknown'}"
        for candidate in page_candidates
        if candidate.get("outcome") in {"facial_no", "reject"}
    ][:5]
    dominant_non_fit_patterns: list[str] = []
    if glance_result and glance_result.action == "reformulate":
        dominant_non_fit_patterns.append(glance_result.summary)
    if noise_anchors and not dominant_non_fit_patterns:
        dominant_non_fit_patterns.append("preview signal dominated by facial-no candidates")

    window = result_window_for_count(result_count)
    result_window = f"{window[0]}-{window[1]}" if window else "direct_paginate"
    return LinkedInPageInsights(
        page=page_num,
        result_count=result_count,
        result_window=result_window,
        title_clusters=[
            {
                "label": label,
                "title": label,
                "count": count,
                "signal_count": title_signal_counts.get(label, 0),
            }
            for label, count in sorted(title_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        ],
        company_clusters=[
            {
                "label": label,
                "company": label,
                "count": count,
                "signal_count": company_signal_counts.get(
                    _normalize_company_rail_label(label),
                    0,
                ),
            }
            for label, count in sorted(company_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        ],
        signal_anchors=signal_anchors,
        noise_anchors=noise_anchors,
        dominant_non_fit_patterns=dominant_non_fit_patterns,
        glance_action=glance_result.action if glance_result else "",
        glance_summary=glance_result.summary if glance_result else "",
    )


def _normalize_company_rail_label(company: str) -> str:
    return " ".join(str(company or "").strip().lower().split())


def _cluster_signal_count(cluster: dict[str, Any]) -> int:
    try:
        return int(cluster.get("signal_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _thresholded_structured_promotions(
    insights: Any | None,
) -> tuple[set[str], set[str]]:
    title_clusters = getattr(insights, "title_clusters", []) or []
    company_clusters = getattr(insights, "company_clusters", []) or []
    title_labels = {
        _normalize_title_family(
            str(cluster.get("label") or cluster.get("title") or "")
        )
        for cluster in title_clusters
        if isinstance(cluster, dict) and _cluster_signal_count(cluster) > 0
    }
    company_labels = {
        _normalize_company_rail_label(
            str(cluster.get("label") or cluster.get("company") or "")
        )
        for cluster in company_clusters
        if isinstance(cluster, dict) and _cluster_signal_count(cluster) > 0
    }
    return (
        {label for label in title_labels if label},
        {label for label in company_labels if label},
    )


def _scout_metrics(
    *,
    page_stats: dict[str, int],
    page_insights: LinkedInPageInsights,
) -> dict[str, object]:
    full_outreach = int(
        page_stats.get("full_outreach", page_stats.get("saves", 0))
    )
    full_review = int(page_stats.get("full_review", 0))
    full_reject = int(
        page_stats.get("full_reject", page_stats.get("rejects", 0))
    )
    full_reviewed = int(
        page_stats.get(
            "full_reviewed",
            full_outreach + full_review + full_reject,
        )
    )
    facial_no = int(page_stats.get("facial_no", 0))
    page_signal = full_outreach + full_review
    noise_dominant = any(
        (
            page_insights.glance_action == "reformulate",
            bool(page_insights.dominant_non_fit_patterns),
            facial_no + full_reject >= max(8, page_signal * 2),
            len(page_insights.noise_anchors) >= len(page_insights.signal_anchors) + 2,
        )
    )
    # A SAVE-family full judgment is strong directional signal. Human
    # REVIEW is weak signal. Full rejects are negative, never productive.
    real_signal = full_outreach > 0
    weak_but_real_signal = full_review > 0 and not real_signal
    return {
        "page_signal": page_signal,
        "full_reviewed": full_reviewed,
        "full_outreach": full_outreach,
        "full_review": full_review,
        "full_reject": full_reject,
        # Compatibility telemetry only; neither field drives signal.
        "saves": int(page_stats.get("saves", 0)),
        "facial_yes": int(page_stats.get("facial_yes", 0)),
        "rejects": full_reject,
        "facial_no": facial_no,
        "noise_dominant": noise_dominant,
        "real_signal": real_signal,
        "strong_scout_signal": real_signal,
        "weak_but_real_signal": weak_but_real_signal,
    }


def _scout_quality_for_recon(
    *,
    result_count: int,
    scout_metrics: dict[str, object],
    precommit_recovery_attempts_used: int,
) -> tuple[str, str, str]:
    page_signal = int(scout_metrics["page_signal"])
    noise_dominant = bool(scout_metrics["noise_dominant"])
    real_signal = bool(scout_metrics["real_signal"])
    weak_but_real_signal = bool(scout_metrics["weak_but_real_signal"])
    recovery_budget_remaining = max(
        0,
        config.PRECOMMIT_MAX_RECOVERY_ATTEMPTS - precommit_recovery_attempts_used,
    )

    if result_count < 500:
        if page_signal == 0 and noise_dominant:
            return (
                "stop",
                "small pool is already dead/noisy on page 1, so it should stop rather than spend rescue budget",
                "small_pool_dead_stop",
            )
        return (
            "commit",
            "result pool is already small enough that direct pagination is cheaper than pre-commit rescue branching",
            "small_pool_direct",
        )
    if result_count < 5000:
        if page_signal > 0:
            return (
                "commit",
                "mid-sized pool already shows enough directional scout signal to paginate without pre-commit rescue branching",
                "mid_pool_signal_commit",
            )
        if noise_dominant and recovery_budget_remaining > 0:
            return (
                "experiment",
                "mid-sized pool is dead/noisy on the scout page, so one bounded rescue attempt is justified before stop",
                "mid_pool_dead_noisy_recovery",
            )
        return (
            "stop",
            "mid-sized pool is not surfacing signal and lacks a coherent scout-time rescue path, so stop cleanly",
            "mid_pool_unsalvageable_stop",
        )

    if page_signal == 0 and noise_dominant:
        if recovery_budget_remaining > 0:
            return (
                "experiment",
                "large/noisy scout page is dead, so spend a bounded pre-commit rescue attempt before abandoning the family",
                "precommit_dead_noisy_recovery",
            )
        return (
            "stop",
            "pre-commit rescue budget is exhausted and the large pool is still dead/noisy, so stop instead of paying more pagination tax",
            "precommit_recovery_exhausted_stop",
        )
    if real_signal and noise_dominant:
        if recovery_budget_remaining > 0:
            return (
                "experiment",
                "large pool shows real signal, but dominant noise means it should get one bounded rescue attempt before direct pagination",
                "precommit_real_signal_noisy_recovery",
            )
        return (
            "commit",
            "large pool still shows real signal and rescue budget is exhausted, so fall back to direct pagination rather than stopping a live lane",
            "precommit_real_signal_budget_exhausted_commit",
        )
    if real_signal:
        return (
            "commit",
            "root scout already shows real signal without dominant noise, so it has earned direct pagination",
            "root_real_signal_commit",
        )
    if weak_but_real_signal:
        if recovery_budget_remaining > 0:
            return (
                "experiment",
                "scout page has only weak signal, so it should survive a bounded rescue attempt before the root query earns pagination",
                "precommit_weak_signal_recovery",
            )
        return (
            "stop",
            "pre-commit rescue budget is exhausted and the large pool still only shows weak scout signal, so stop instead of paying more pagination tax",
            "precommit_recovery_exhausted_stop",
        )
    return (
        "stop",
        "large pool is not surfacing signal and the scout did not produce a coherent rescue path, so stop cleanly",
        "precommit_unsalvageable_stop",
    )


def _variant_has_earned_commit(variant: LinkedInSearchVariant) -> bool:
    # Facial positives only justify opening a profile. A variant earns
    # pagination after at least one settled full judgment clears outreach.
    return int(getattr(variant, "full_outreach", 0) or 0) > 0


def _assess_pagination_drift(
    *,
    experiment_state: LinkedInExperimentState,
    page_num: int,
    result_count: int,
    page_stats: dict[str, int],
    page_insights: LinkedInPageInsights,
    remaining_queued_strings: int,
) -> LinkedInDriftAssessment:
    if experiment_state.committed_variant_id is None:
        return LinkedInDriftAssessment(decision="continue", rationale="", eligible=False)
    if experiment_state.active_variant_id != experiment_state.committed_variant_id:
        return LinkedInDriftAssessment(decision="continue", rationale="", eligible=False)
    if experiment_state.drift_attempt_count >= config.SEARCH_EXPERIMENT_DRIFT_BUDGET:
        return LinkedInDriftAssessment(
            decision="continue",
            rationale="drift budget already used for this committed variant",
            eligible=False,
        )
    if experiment_state.pages_since_last_mutation < 1:
        return LinkedInDriftAssessment(
            decision="continue",
            rationale="need at least one fully reviewed page between search mutations",
            eligible=False,
        )
    active_variant = experiment_state.active_variant
    if active_variant.pages_reviewed < 2 or page_num < 2 or result_count < 500:
        return LinkedInDriftAssessment(decision="continue", rationale="", eligible=False)
    if not experiment_state.real_signal_seen():
        return LinkedInDriftAssessment(
            decision="continue",
            rationale="committed variant has not shown enough real signal to justify rescue",
            eligible=False,
        )

    page_signal = int(
        page_stats.get("full_outreach", page_stats.get("saves", 0))
    ) + int(page_stats.get("full_review", 0))
    noisy_page = page_signal == 0 and (
        page_insights.glance_action == "reformulate"
        or len(page_insights.noise_anchors) >= max(2, len(page_insights.signal_anchors) + 1)
        or bool(page_insights.dominant_non_fit_patterns)
    )
    if not noisy_page:
        return LinkedInDriftAssessment(decision="continue", rationale="", eligible=False)

    opportunity_cost_ok = (
        experiment_state.family_outreach_total > 0
        or remaining_queued_strings <= 3
    )
    if not opportunity_cost_ok:
        return LinkedInDriftAssessment(
            decision="continue",
            rationale="other untouched families still have higher expected value than rescuing this one",
            eligible=False,
        )

    early = experiment_state.early_signal_snapshot
    recent = experiment_state.recent_noise_snapshot
    early_titles = [cluster.get("label", "") for cluster in (early.title_clusters if early else []) if cluster.get("label")]
    recent_titles = [cluster.get("label", "") for cluster in (recent.title_clusters if recent else []) if cluster.get("label")]
    overfit_risk = "high" if early and len(early.signal_anchors) <= 1 and remaining_queued_strings > 1 else "medium"
    keyword_hypothesis = ""
    if early_titles:
        keyword_hypothesis = f"tighten around early signal titles like {early_titles[0]}"
    elif recent_titles:
        keyword_hypothesis = f"exclude late-page leakage titles like {recent_titles[0]}"
    future_filter_hypothesis = ""
    if recent_titles:
        future_filter_hypothesis = f"title filter could exclude {recent_titles[0]}"
    elif recent and recent.company_clusters:
        future_filter_hypothesis = f"company filter could demote {recent.company_clusters[0].get('label', '')}"

    if overfit_risk == "high":
        decision = "spawn_recall_sibling"
        rationale = "later pages are drifting, but the early signal sample is narrow enough that rescue should stay recall-friendly"
    else:
        decision = "refine_committed"
        rationale = "later pages are leaking noise and the committed variant has enough prior value to justify one bounded keyword rescue"
    return LinkedInDriftAssessment(
        decision=decision,
        rationale=rationale,
        eligible=True,
        overfit_risk=overfit_risk,
        keyword_hypothesis=keyword_hypothesis,
        future_filter_hypothesis=future_filter_hypothesis,
    )


def _acquisition_mode_for_variant(
    search_string: SearchString,
    variant: LinkedInSearchVariant,
) -> str:
    """Resolve the acquisition mode for a variant.

    A variant with non-empty structured filters runs as ``linkedin_hybrid``;
    upgrade ``search_string.acquisition_mode`` in place so the persisted record
    satisfies that invariant. ``apply_variant`` rejects a non-empty-filter variant
    unless the mode is ``linkedin_hybrid``, so without this upgrade a mid-run
    promote on a boolean lane would be silently denied. A keyword/empty-filter variant keeps the lane's
    mode. A deliberate demotion clears filters while the hybrid mode remains,
    yielding the established hybrid-plus-empty keyword path.
    """
    if not variant.structured_filters.is_empty():
        if search_string.acquisition_mode != "linkedin_hybrid":
            search_string.acquisition_mode = "linkedin_hybrid"
        return "linkedin_hybrid"
    return search_string.acquisition_mode or "linkedin_boolean"


def _resolve_structured_controls(
    raw_variant: dict[str, Any],
    inherited: LinkedInStructuredFilters,
    *,
    structured_lever_open: bool = True,
    allowed_promoted_titles: set[str] | None = None,
    allowed_promoted_companies: set[str] | None = None,
    company_scope: Any = None,
    refused: list[str] | None = None,
) -> tuple[LinkedInStructuredFilters, str, str, bool]:
    """Resolve proposal surface and structured controls into variant state.

    ``company_scope`` is the brief's employer allowlist
    (shared/employer_allowlist.py), or None. Under one, the companies
    dimension is the campaign's competitor bound: a ``companies`` demote is
    refused and an unlisted company is never promoted into it; each refusal
    is appended to ``refused`` (``"demote:companies"``, ``"promote:<name>"``).
    A title-restricted employer (a client's data side) is never promoted, and
    a string its facet bounds keeps its titles: a ``titles`` demote and any
    promoted title outside the allowlist's families are refused
    (``"demote:titles"``, ``"promote:<title>"``).

    Both variant planners use the same semantics: copy inherited filters, add
    promoted titles/companies, drop demoted dimensions, resolve the surface,
    and mark non-empty controls as ``structured_filter``. Return the resolved
    filters, surface, kind, and whether filters changed.

    When ``structured_lever_open`` is false, drop promoted controls and coerce
    the surface to ``boolean``; still honor demotions and retain their
    deliberate-demotion marker. Title/company promotions also require their
    literal label in caller-supplied evidence; a promotion stripped by that gate
    does not keep ``structured_filter`` provenance.
    """
    # Harden the unpack: a disobeying model can hand back a non-dict truthy
    # structured_controls (list/str/int) — `or {}` only rescues FALSY values, so
    # an isinstance gate is what keeps controls.get(...) from raising
    # (AttributeError would escape the drift planner's try/except, aborting the run).
    raw_controls = raw_variant.get("structured_controls")
    controls = raw_controls if isinstance(raw_controls, dict) else {}
    # Closed breaker: coerce the surface to "boolean" — the lever offers no
    # hybrid/structured_only, so a disobeying model's surface is overridden, not
    # trusted. (Open path keeps the model's surface, default "boolean".)
    if structured_lever_open:
        surface = str(raw_variant.get("surface") or "boolean").strip() or "boolean"
    else:
        surface = "boolean"
    filters = _copy_filters(inherited)
    client_string = company_scope is not None and any(
        company_scope.is_restricted(company_scope.canonical_for(company))
        for company in inherited.companies or []
    )

    # PROMOTE is the only RE-PROMOTE vector, so it is the one the closed breaker
    # refuses: with the lever closed, titles/companies adds are dropped entirely
    # (the inherited filters carry forward unchanged; demotes below still run).
    # Each dimension is likewise list-guarded: a str value (e.g. "NotAList")
    # otherwise iterates per-character into garbage titles/companies.
    ran_structured_control = False
    if structured_lever_open:
        raw_titles = controls.get("titles")
        promoted_titles = [
            str(t).strip()
            for t in (raw_titles if isinstance(raw_titles, list) else [])
            if str(t).strip()
        ]
        graduated_titles = [
            title
            for title in promoted_titles
            if allowed_promoted_titles is None
            or _normalize_title_family(title) in allowed_promoted_titles
        ]
        if client_string:
            for title in [t for t in graduated_titles if not company_scope.title_in_family(t)]:
                if refused is not None:
                    refused.append(f"promote:{title}")
            graduated_titles = [t for t in graduated_titles if company_scope.title_in_family(t)]
        # A title promotion must span the role's title families or it is dropped
        # whole and the variant stays keyword-only. Count families rather than
        # labels, and judge the prospective inherited-plus-promoted set: rail
        # volume can evidence individual modal titles while the set remains too
        # narrow. Fail closed so an undersized filter does not hide the pool.
        prospective_families = {
            family
            for family in (
                _normalize_title_family(title)
                for title in (*filters.titles, *graduated_titles)
            )
            if family
        }
        if (
            graduated_titles
            and len(prospective_families) < config.LINKEDIN_TITLE_FILTER_MIN_FAMILIES
        ):
            graduated_titles = []
        for title in graduated_titles:
            if title not in filters.titles:
                filters.titles.append(title)
                ran_structured_control = True
        raw_companies = controls.get("companies")
        promoted_companies = [
            str(c).strip()
            for c in (raw_companies if isinstance(raw_companies, list) else [])
            if str(c).strip()
        ]
        for company in promoted_companies:
            if (
                allowed_promoted_companies is not None
                and _normalize_company_rail_label(company)
                not in allowed_promoted_companies
            ):
                continue
            if company_scope is not None and (
                not company_scope.canonical_for(company)
                or company_scope.is_restricted(company_scope.canonical_for(company))
            ):
                if refused is not None:
                    refused.append(f"promote:{company}")
                continue
            if company not in filters.companies:
                filters.companies.append(company)
                ran_structured_control = True

    # DEMOTE: drop the named dimension(s) entirely (the over-narrowing/leaking
    # filter the loop is retiring). Matches _drop_one_filter's dimension set.
    raw_demote = controls.get("demote")
    demote_dims = [
        str(d).strip()
        for d in (raw_demote if isinstance(raw_demote, list) else [])
        if str(d).strip()
    ]
    for dim in demote_dims:
        if dim == "companies" and company_scope is not None:
            if refused is not None:
                refused.append("demote:companies")
            continue
        if dim == "titles" and client_string:
            if refused is not None:
                refused.append("demote:titles")
            continue
        if dim in {"titles", "companies", "skills", "assessments"}:
            ran_structured_control = True
            getattr(filters, dim).clear()
        elif dim in filters.sidebar_filters:
            ran_structured_control = True
            filters.sidebar_filters.pop(dim, None)
        elif dim in filters.advanced_filters:
            ran_structured_control = True
            filters.advanced_filters.pop(dim, None)

    structured_changed = filters.to_dict() != inherited.to_dict()
    # Make the "structured_filter" marker PROVENANCE-true, not LABEL-true: the kind
    # is keyed on by is_deliberate_boolean_demotion (seeding gate), so only a variant
    # that actually ran a structured control may wear it. A real structured move is
    # one that either carries non-empty controls (a promote, or a demote that named a
    # dimension — full demote-to-boolean included) OR changed the structured state.
    # An EMPTY-controls proposal that merely self-labels "structured_filter" while
    # changing nothing structural is a keyword variant; fall through to its keyword
    # kind so it is NOT mistaken for a deliberate demote and silently denied the
    # lane's structured-filter seed on a hybrid lane.
    #
    # Closed breaker: drop the bool(controls) term. With the lever closed a promote
    # was already stripped above, so a promote-only payload's still-truthy `controls`
    # had NO effect; keying the marker on it would mint a structured_filter kind for
    # a variant that ran no structured move and re-promote nothing — exactly the
    # disobeying-model re-promote the breaker exists to deny. Only a demote that
    # actually changed the filters (structured_changed) is a structured move when
    # closed, so a demote-to-empty keeps its deliberate-demotion provenance while a
    # stripped promote falls through to a plain keyword kind.
    ran_structured_move = ran_structured_control or structured_changed
    if ran_structured_move:
        variant_kind = "structured_filter"
    else:
        raw_kind = str(raw_variant.get("variant_kind") or "precision")
        variant_kind = "precision" if raw_kind == "structured_filter" else raw_kind
    # A structured move that emptied the filters is a deliberate boolean demotion;
    # normalize its surface so the marker (surface=="boolean" + kind=="structured_filter")
    # is unambiguous for the seeding gate.
    if ran_structured_move and filters.is_empty():
        surface = "boolean"
    elif not ran_structured_move and filters.is_empty():
        surface = "boolean"
    return filters, surface, variant_kind, structured_changed


def _fallback_drift_boolean(
    *,
    current_boolean: str,
    decision: str,
    early_snapshot: LinkedInVariantSnapshot | None,
    recent_snapshot: LinkedInVariantSnapshot | None,
) -> str:
    if decision == "spawn_recall_sibling":
        if recent_snapshot and recent_snapshot.title_clusters:
            label = str(recent_snapshot.title_clusters[0].get("label", "")).strip()
            if label:
                return f"({current_boolean}) NOT (\"{label}\")"
        return ""
    if early_snapshot and early_snapshot.title_clusters:
        label = str(early_snapshot.title_clusters[0].get("label", "")).strip()
        if label and label.lower() not in current_boolean.lower():
            return f"({current_boolean}) AND (\"{label}\")"
    if recent_snapshot and recent_snapshot.title_clusters:
        label = str(recent_snapshot.title_clusters[0].get("label", "")).strip()
        if label:
            return f"({current_boolean}) NOT (\"{label}\")"
    return ""


def _should_force_narrow_in_scout(
    *,
    search_string: SearchString,
    page_num: int,
    result_count: int,
    string_stats: dict,
    glance_result: GlanceResult | None,
) -> bool:
    """Allow forced narrow only for page-1 scout strings that show zero signal."""
    return (
        search_string.phase == "scout"
        and page_num == 1
        and result_count >= 500
        and not search_string.refinement_stack
        and string_stats.get("full_outreach", 0) == 0
        and string_stats.get("full_review", 0) == 0
        and glance_result is not None
        and glance_result.action == "reformulate"
    )
