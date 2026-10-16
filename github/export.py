"""CSV export for Gem/Greenhouse import.

Reads canonical runtime state when ``runtime_state.sqlite3`` is present,
otherwise falls back to legacy pipeline JSONL projections. Joins saved
candidates on identity (username / profile URL), never display name alone
when ambiguity exists.

Usage:
    python github_export.py output/github/
    python github_export.py output/github/ --out custom_path.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from shared.judger import extract_priority_rank
from shared.runtime_state import RuntimeStateStore
from shared.storage import read_jsonl


# Column order — Gem standard fields first, then GitHub-specific custom fields
CSV_COLUMNS = [
    # Gem auto-map fields
    "First Name",
    "Last Name",
    "Email",
    "LinkedIn URL",
    "Company",
    "Title",
    "Location",
    "Source",
    # GitHub-specific (Gem custom fields)
    "GitHub URL",
    "GitHub Username",
    "Decision",
    "Confidence",
    "Maintainership Level",
    "Maintainership Confidence",
    "Maintainership Evidence",
    "Maintainership Target Project",
    "Capability Area",
    "Evaluation Summary",
    "Top Repos",
    "Toolchain",
    "ML Signal",
    "Papers",
    "Website",
    "Outreach Subject",
    "Outreach Message",
    "Priority Rank",
    "Source Query",
    "Source Channel",
    # Identity handles for recruiter reconciliation (after Gem/Greenhouse columns)
    "github_username",
    "github_profile_url",
    "person_key",
    "linkedin_url",
]

SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}


def _split_name(full_name: str, username: str = "") -> tuple[str, str]:
    """Split a full name into first and last. Falls back to username."""
    name = full_name.strip()
    if not name:
        return (username, "")
    parts = name.split(None, 1)
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[0], parts[1])


def _extract_username(record: dict) -> str:
    """Extract username from a candidate or judgment record."""
    # candidates.jsonl has top-level "username" and nested user.username
    if "username" in record:
        return record["username"]
    if "user" in record and isinstance(record["user"], dict):
        return record["user"].get("username", "")
    # judgments may carry username directly
    return record.get("username", "") or record.get("candidate_name", "")


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    return []


def _maintainership_payload(candidate: dict) -> dict:
    payload = candidate.get("maintainership")
    return payload if isinstance(payload, dict) else {}


def _format_maintainership_confidence(payload: dict) -> str:
    confidence = payload.get("confidence")
    if isinstance(confidence, (int, float)):
        return f"{confidence:.2f}"
    if isinstance(confidence, str):
        stripped = confidence.strip()
        if not stripped:
            return ""
        try:
            return f"{float(stripped):.2f}"
        except ValueError:
            return stripped
    return ""


def _evidence_sources(payload: dict) -> list[str]:
    sources = _string_list(payload.get("evidence_sources"))
    signals = payload.get("signals")
    if isinstance(signals, dict) and signals.get("budget_exhausted"):
        sources.append("budget_exhausted")
    return sources


def _project_from_evidence_token(source: str) -> str | None:
    """Extract owner/repo from a maintainership evidence token.

    Both token families are target-project-scoped by construction: the
    classifier only runs against ``brief.target_projects``, and declared
    entries are filtered to target projects at the classify seam (W3-PX1)
    before they can mint ``declared:`` tokens. So any owner/repo segment
    here is safe for the Maintainership Target Project column.
    """
    if not isinstance(source, str):
        return None
    for part in source.split(":")[1:]:
        text = part.strip()
        if "/" in text:
            return text
    return None


def _maintainership_target_projects(payload: dict) -> str:
    explicit = (
        _string_list(payload.get("target_projects"))
        or _string_list(payload.get("target_project"))
        or _string_list(payload.get("projects"))
        or _string_list(payload.get("project"))
        or _string_list(payload.get("repo"))
    )
    if explicit:
        return ", ".join(dict.fromkeys(explicit))

    projects: list[str] = []
    for source in _evidence_sources(payload):
        project = _project_from_evidence_token(source)
        if project:
            projects.append(project)
    return ", ".join(dict.fromkeys(projects))


def _normalize_github_profile_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/")
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if netloc.endswith("github.com"):
        scheme = "https"
    if scheme in {"http", "https"} and netloc:
        return urlunsplit((scheme, netloc, path, "", ""))
    return raw.rstrip("/")


def _load_json_dict(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw or raw == "{}":
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _resolve_github_brief_ids(store: RuntimeStateStore) -> list[str]:
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT brief_id FROM runs
            WHERE source = 'github'
            ORDER BY brief_id
            """
        ).fetchall()
        if rows:
            return [str(row["brief_id"]) for row in rows]
        rows = conn.execute(
            """
            SELECT DISTINCT brief_id FROM candidates
            WHERE source = 'github'
            ORDER BY brief_id
            """
        ).fetchall()
        return [str(row["brief_id"]) for row in rows]


def _build_csv_row(
    *,
    candidate: dict,
    judgment: dict,
    outreach: dict,
    username: str,
    person_key: str,
    github_profile_url: str,
    linkedin_url: str,
) -> dict[str, str]:
    user = candidate.get("user", {}) if isinstance(candidate.get("user"), dict) else {}
    contact = candidate.get("contact", {}) if isinstance(candidate.get("contact"), dict) else {}
    portfolio = candidate.get("portfolio_summary", {}) if isinstance(
        candidate.get("portfolio_summary"), dict
    ) else {}

    decision = str(judgment.get("decision", "") or "")
    confidence_raw = judgment.get("confidence", 0)
    try:
        confidence_value = float(confidence_raw or 0)
    except (TypeError, ValueError):
        confidence_value = 0.0

    full_name = str(user.get("name", "") or username)
    first_name, last_name = _split_name(full_name, username)

    emails = contact.get("emails", [])
    email = emails[0] if isinstance(emails, list) and emails else ""

    title = str(candidate.get("synthesized_headline", "") or user.get("bio", "") or "")
    if len(title) > 200:
        title = title[:197] + "..."

    top_repos_list = candidate.get("top_repos", [])[:3] if isinstance(
        candidate.get("top_repos"), list
    ) else []
    top_repos = ", ".join(
        f"{r.get('name', '')} ({r.get('stars', 0)} stars)"
        for r in top_repos_list
        if isinstance(r, dict)
    )

    toolchain_data = portfolio.get("toolchain_detected", {})
    toolchain = ""
    if isinstance(toolchain_data, dict):
        toolchain = ", ".join(toolchain_data.get("frameworks", []) or [])

    ml_signal = str(portfolio.get("ml_signal_strength", "") or "")
    paper_titles = candidate.get("paper_titles", [])
    papers = "; ".join(paper_titles) if isinstance(paper_titles, list) and paper_titles else ""
    website = str(contact.get("website", "") or user.get("blog", "") or "")

    path = str(judgment.get("path", "") or "")
    cap_area = ""
    if ":" in path:
        cap_area = path.split(":", 1)[1].split("|")[0]
    priority_rank = extract_priority_rank(path)

    maintainership = _maintainership_payload(candidate)
    maintainership_evidence = _evidence_sources(maintainership)

    gem_linkedin = str(contact.get("linkedin_url", "") or "")
    profile_url = github_profile_url or str(user.get("profile_url", "") or "")

    return {
        "First Name": first_name,
        "Last Name": last_name,
        "Email": email,
        "LinkedIn URL": gem_linkedin,
        "Company": str(user.get("company", "") or ""),
        "Title": title,
        "Location": str(user.get("location", "") or ""),
        "Source": "GitHub Sourcing Agent",
        "GitHub URL": profile_url,
        "GitHub Username": username,
        "Decision": decision,
        "Confidence": f"{confidence_value:.2f}",
        "Maintainership Level": str(maintainership.get("level", "") or ""),
        "Maintainership Confidence": _format_maintainership_confidence(maintainership),
        "Maintainership Evidence": "; ".join(maintainership_evidence),
        "Maintainership Target Project": _maintainership_target_projects(maintainership),
        "Capability Area": cap_area,
        "Evaluation Summary": str(judgment.get("rationale", "") or ""),
        "Top Repos": top_repos,
        "Toolchain": toolchain,
        "ML Signal": ml_signal,
        "Papers": papers,
        "Website": website,
        "Outreach Subject": str(outreach.get("subject_line", "") or ""),
        "Outreach Message": str(outreach.get("message", "") or ""),
        "Priority Rank": str(priority_rank) if priority_rank else "",
        "Source Query": str(candidate.get("source_query", "") or ""),
        "Source Channel": str(candidate.get("source_strategy", "") or ""),
        "github_username": username,
        "github_profile_url": profile_url,
        "person_key": person_key,
        "linkedin_url": linkedin_url or gem_linkedin,
    }


def _resolve_legacy_candidate(
    judgment: dict,
    candidate_by_username: dict[str, dict],
    candidate_by_profile_url: dict[str, dict],
    usernames_by_name: dict[str, list[str]],
) -> tuple[dict, str] | None:
    """Match a legacy judgment row to a candidate record by identity.

    Username on the judgment row is preferred; profile URL is next. Display
    name is used only when it maps to exactly one candidate. Rows with
    duplicate display names and no username/profile_url on the judgment
    remain ambiguous and are skipped.
    """
    judgment_username = str(judgment.get("username", "") or "").strip()
    if judgment_username:
        candidate = candidate_by_username.get(judgment_username)
        if candidate:
            return candidate, judgment_username

    judgment_profile_url = _normalize_github_profile_url(
        str(judgment.get("profile_url", "") or "")
    )
    if judgment_profile_url:
        candidate = candidate_by_profile_url.get(judgment_profile_url)
        if candidate:
            user = candidate.get("user", {}) if isinstance(candidate.get("user"), dict) else {}
            username = str(
                candidate.get("username", "") or user.get("username", "") or ""
            ).strip()
            if username:
                return candidate, username

    candidate_name = str(judgment.get("candidate_name", "") or "").strip()
    if not candidate_name:
        return None
    matching_usernames = usernames_by_name.get(candidate_name, [])
    if len(matching_usernames) == 1:
        username = matching_usernames[0]
        candidate = candidate_by_username.get(username)
        if candidate:
            return candidate, username
    return None


def _export_rows_from_runtime_state(store: RuntimeStateStore) -> list[dict[str, str]]:
    brief_ids = _resolve_github_brief_ids(store)
    if not brief_ids:
        return []

    rows: list[dict[str, str]] = []
    for brief_id in brief_ids:
        for identity_key in store.list_terminal_identity_keys(source="github", brief_id=brief_id):
            candidate_row = store.get_candidate(
                source="github",
                brief_id=brief_id,
                identity_key=identity_key,
            )
            if not candidate_row:
                continue
            decision = str(candidate_row.get("terminal_decision") or "")
            if decision not in SAVE_DECISIONS:
                continue

            payload = _load_json_dict(candidate_row.get("terminal_payload_json"))
            judgment = payload.get("full_decision")
            if not isinstance(judgment, dict):
                judgment = {}
            candidate_record = payload.get("candidate_record")
            if not isinstance(candidate_record, dict):
                candidate_record = {}

            username = _extract_username(candidate_record) or identity_key
            outreach = candidate_record.get("outreach_copy")
            if not isinstance(outreach, dict):
                outreach = {}

            user = candidate_record.get("user", {}) if isinstance(
                candidate_record.get("user"), dict
            ) else {}
            profile_url = str(
                user.get("profile_url", "") or candidate_row.get("profile_url", "") or ""
            ).strip()
            contact = candidate_record.get("contact", {}) if isinstance(
                candidate_record.get("contact"), dict
            ) else {}
            linkedin_url = str(contact.get("linkedin_url", "") or "").strip()

            person_key = str(candidate_row.get("person_key") or "") or identity_key

            rows.append(
                _build_csv_row(
                    candidate=candidate_record,
                    judgment={**judgment, "decision": decision},
                    outreach=outreach,
                    username=username,
                    person_key=person_key,
                    github_profile_url=profile_url,
                    linkedin_url=linkedin_url,
                )
            )
    return rows


def _export_rows_from_jsonl(output_dir: Path) -> list[dict[str, str]]:
    candidates = read_jsonl(output_dir / "candidates.jsonl")
    judgments = read_jsonl(output_dir / "final_judgments.jsonl")
    outreach_records = read_jsonl(output_dir / "outreach.jsonl")

    candidate_by_username: dict[str, dict] = {}
    candidate_by_profile_url: dict[str, dict] = {}
    usernames_by_name: dict[str, list[str]] = {}
    for candidate in candidates:
        username = _extract_username(candidate)
        if not username:
            continue
        candidate_by_username[username] = candidate
        user = candidate.get("user", {}) if isinstance(candidate.get("user"), dict) else {}
        profile_url = _normalize_github_profile_url(str(user.get("profile_url", "") or ""))
        if profile_url:
            candidate_by_profile_url[profile_url] = candidate
        candidate_name = str(user.get("name", "") or "").strip()
        if candidate_name:
            bucket = usernames_by_name.setdefault(candidate_name, [])
            if username not in bucket:
                bucket.append(username)
        username_bucket = usernames_by_name.setdefault(username, [])
        if username not in username_bucket:
            username_bucket.append(username)

    outreach_by_username: dict[str, dict] = {}
    for outreach in outreach_records:
        outreach_username = str(outreach.get("username", "") or "").strip()
        if outreach_username:
            outreach_by_username[outreach_username] = outreach

    rows: list[dict[str, str]] = []
    for judgment in judgments:
        if judgment.get("stage") != "full":
            continue
        decision = judgment.get("decision", "")
        if decision not in SAVE_DECISIONS:
            continue

        resolved = _resolve_legacy_candidate(
            judgment,
            candidate_by_username,
            candidate_by_profile_url,
            usernames_by_name,
        )
        if not resolved:
            continue
        candidate, username = resolved
        outreach = outreach_by_username.get(username, {})
        if not isinstance(outreach, dict):
            outreach = {}

        user = candidate.get("user", {}) if isinstance(candidate.get("user"), dict) else {}
        profile_url = str(user.get("profile_url", "") or "").strip()
        contact = candidate.get("contact", {}) if isinstance(candidate.get("contact"), dict) else {}
        linkedin_url = str(contact.get("linkedin_url", "") or "").strip()
        person_key = f"gh:{username.lower()}"

        rows.append(
            _build_csv_row(
                candidate=candidate,
                judgment=judgment,
                outreach=outreach,
                username=username,
                person_key=person_key,
                github_profile_url=profile_url,
                linkedin_url=linkedin_url,
            )
        )
    return rows


def export_saved_candidates_csv(
    output_dir: str | Path,
    csv_path: str | Path | None = None,
) -> Path:
    """Export saved candidates as a Gem/Greenhouse-compatible CSV.

    Returns the path to the written CSV file.
    """
    output_dir = Path(output_dir)
    if csv_path is None:
        csv_path = output_dir / "saved_candidates.csv"
    csv_path = Path(csv_path)

    runtime_db_path = output_dir / "runtime_state.sqlite3"
    if runtime_db_path.is_file():
        store = RuntimeStateStore(runtime_db_path)
        rows = _export_rows_from_runtime_state(store)
    else:
        rows = _export_rows_from_jsonl(output_dir)

    rows.sort(key=lambda row: float(row.get("Confidence", "0") or "0"), reverse=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python github_export.py <output_dir> [--out <csv_path>]")
        sys.exit(1)

    output_dir = sys.argv[1]
    csv_out = None
    if "--out" in sys.argv:
        idx = sys.argv.index("--out")
        if idx + 1 < len(sys.argv):
            csv_out = sys.argv[idx + 1]

    path = export_saved_candidates_csv(output_dir, csv_out)
    print(f"Exported to: {path}")
