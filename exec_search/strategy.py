"""Executive Search strategy formation."""

from __future__ import annotations

from typing import Any

from shared.brief_loader import Brief
from shared.schemas import ExecutionPlan


def form_exec_search_strategy(
    brief: Brief,
    prior_run_data: dict | None = None,
    *,
    investigation_packet: dict[str, Any] | None = None,
) -> ExecutionPlan:
    """Build company/scope/career-path lanes for exec-search discovery."""

    raw = _brief_raw(brief)
    companies = _target_companies(raw, investigation_packet)
    titles = _target_titles(raw, brief)
    career_paths = _career_paths(raw, investigation_packet)
    lanes: list[dict[str, Any]] = []
    lane_id = 1
    for company in companies:
        for title in titles[:3]:
            lanes.append(
                {
                    "id": lane_id,
                    "name": f"{title} at {company}",
                    "lane_type": "target_company",
                    "company": company,
                    "title": title,
                    "scope": _scope_hint(raw),
                    "career_path_hypothesis": career_paths[0] if career_paths else "",
                    "query": f'"{title}" "{company}" executive profile',
                }
            )
            lane_id += 1
    if not lanes:
        for title in titles[:5]:
            lanes.append(
                {
                    "id": lane_id,
                    "name": f"Market map: {title}",
                    "lane_type": "market_map",
                    "company": "",
                    "title": title,
                    "scope": _scope_hint(raw),
                    "career_path_hypothesis": career_paths[0] if career_paths else "",
                    "query": f'"{title}" executive profile',
                }
            )
            lane_id += 1

    return ExecutionPlan(
        strategy_rationale=(
            "Company/seniority-first executive search lanes derived from the "
            "brief and pre-launch investigation."
        ),
        generated_strings=lanes,
        architecture="company_first",
        architecture_rationale="Exec search starts from company scope and career-path hypotheses.",
    )


def form_strategy_for_registry(
    brief: Brief,
    prior_run_data: dict | None = None,
) -> ExecutionPlan:
    return form_exec_search_strategy(brief, prior_run_data)


def _brief_raw(brief: Brief) -> dict[str, Any]:
    raw = getattr(brief, "_new_brief", None)
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "raw_dict"):
        candidate = raw.raw_dict()
        if isinstance(candidate, dict):
            return candidate
    return {}


def _target_companies(
    raw: dict[str, Any],
    investigation_packet: dict[str, Any] | None,
) -> list[str]:
    source_config = raw.get("source_config") or {}
    exec_config = source_config.get("exec_search") if isinstance(source_config, dict) else {}
    candidates = []
    if isinstance(exec_config, dict):
        candidates.extend(exec_config.get("target_companies") or [])
        candidates.extend(exec_config.get("company_allowlist") or [])
    if investigation_packet:
        for finding in investigation_packet.get("findings") or []:
            if isinstance(finding, dict):
                company = finding.get("company") or finding.get("target_company")
                if isinstance(company, str):
                    candidates.append(company)
    seen: set[str] = set()
    out: list[str] = []
    for item in candidates:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
    return out[:20]


def _target_titles(raw: dict[str, Any], brief: Brief) -> list[str]:
    source_config = raw.get("source_config") or {}
    exec_config = source_config.get("exec_search") if isinstance(source_config, dict) else {}
    titles: list[str] = []
    if isinstance(exec_config, dict):
        titles.extend(item for item in exec_config.get("target_titles") or [] if isinstance(item, str))
    role_title = str(raw.get("role_title") or getattr(brief, "role_title", "") or "").strip()
    if role_title:
        titles.append(role_title)
    defaults = ["Chief Product Officer", "VP Product", "Chief Operating Officer", "VP Operations"]
    titles.extend(defaults)
    seen: set[str] = set()
    out: list[str] = []
    for title in titles:
        cleaned = title.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out


def _career_paths(
    raw: dict[str, Any],
    investigation_packet: dict[str, Any] | None,
) -> list[str]:
    source_config = raw.get("source_config") or {}
    exec_config = source_config.get("exec_search") if isinstance(source_config, dict) else {}
    paths = []
    if isinstance(exec_config, dict):
        paths.extend(exec_config.get("career_path_hypotheses") or [])
    if investigation_packet:
        paths.extend(investigation_packet.get("sourcing_recommendations") or [])
    return [str(path).strip() for path in paths if isinstance(path, str) and path.strip()]


def _scope_hint(raw: dict[str, Any]) -> str:
    calibration = raw.get("executive_calibration")
    if not isinstance(calibration, dict):
        return ""
    pieces = [
        str(calibration.get("sector") or "").strip(),
        str(calibration.get("stage") or "").strip(),
        str(calibration.get("pnl_scale_usd") or "").strip(),
    ]
    return " / ".join(piece for piece in pieces if piece)
