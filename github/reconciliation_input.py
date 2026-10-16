"""Prepare GitHub-sourced leads for LinkedIn reconciliation and experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from github.schemas import ContactInfo
from shared.identity_experiment_schemas import (
    IdentityResolutionExperimentLead,
    IdentityResolutionGoldLabel,
)
from shared.identity_resolution import (
    build_person_lookup_name,
    normalize_company_name,
    normalize_person_name,
    summarize_email_domains,
)
from shared.reconciliation_schemas import LinkedInIdentityHints
from shared.storage import read_jsonl, write_json

SAVE_DECISIONS = {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}
PRIMARY_EXPERIMENT_BUCKETS = (
    "easy_exact_name",
    "common_name_ambiguous",
    "name_variant",
    "stale_employer_changed_role",
)


@dataclass
class GitHubReconciliationLead:
    username: str
    candidate_name: str
    github_url: str
    company: str = ""
    location: str = ""
    title: str = ""
    decision: str = ""
    confidence: float = 0.0
    rationale: str = ""
    source_query: str = ""
    source_channel: str = ""
    linkedin_hints: LinkedInIdentityHints | None = None
    candidate_payload: dict = field(default_factory=dict)
    judgment_payload: dict = field(default_factory=dict)
    outreach_payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.linkedin_hints is None:
            payload["linkedin_hints"] = None
        return payload


@dataclass
class GitHubReconciliationLoadStats:
    total_saved_judgments: int = 0
    leads_loaded: int = 0
    skipped_missing_name: int = 0
    skipped_ambiguous_name: int = 0
    skipped_unmatched_profile_url: int = 0
    skipped_missing_candidate: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GitHubReconciliationLoadBatch:
    leads: list[GitHubReconciliationLead]
    stats: GitHubReconciliationLoadStats


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


def _build_contact(record: dict) -> ContactInfo:
    contact = record.get("contact", {}) if isinstance(record.get("contact"), dict) else {}
    return ContactInfo(
        emails=list(contact.get("emails", []) or []),
        linkedin_url=str(contact.get("linkedin_url", "") or "").strip(),
        twitter_url=str(contact.get("twitter_url", "") or "").strip(),
        website=str(contact.get("website", "") or "").strip(),
    )


def build_identity_hints(candidate_record: dict, judgment_record: dict | None = None) -> LinkedInIdentityHints:
    user = candidate_record.get("user", {}) if isinstance(candidate_record.get("user"), dict) else {}
    contact = _build_contact(candidate_record)
    emails = [email for email in contact.emails if "@" in email]
    profile_url = str(user.get("profile_url", "") or "").strip()
    title = (
        str(candidate_record.get("synthesized_headline", "") or "").strip()
        or str(user.get("bio", "") or "").strip()
    )
    candidate_name = str(user.get("name", "") or "").strip() or str(user.get("username", "") or "").strip()
    source_query = str(candidate_record.get("source_query", "") or "").strip()
    source_channel = str(candidate_record.get("source_strategy", "") or "").strip()
    if judgment_record and not title:
        title = str(judgment_record.get("candidate_title", "") or "").strip()

    return LinkedInIdentityHints(
        candidate_name=candidate_name,
        github_username=str(user.get("username", "") or "").strip(),
        github_url=profile_url,
        linkedin_url_hint=contact.linkedin_url,
        company=str(user.get("company", "") or "").strip(),
        location=str(user.get("location", "") or "").strip(),
        title=title,
        emails=emails,
        email_domains=summarize_email_domains(emails),
        source_query=source_query,
        source_channel=source_channel,
    )


def load_github_reconciliation_batch(output_dir: str | Path) -> GitHubReconciliationLoadBatch:
    output_dir = Path(output_dir)
    candidates = read_jsonl(output_dir / "candidates.jsonl")
    judgments = read_jsonl(output_dir / "final_judgments.jsonl")
    outreach = read_jsonl(output_dir / "outreach.jsonl")
    stats = GitHubReconciliationLoadStats()

    candidate_by_username: dict[str, dict] = {}
    candidate_by_profile_url: dict[str, dict] = {}
    usernames_by_name: dict[str, list[str]] = {}
    for record in candidates:
        user = record.get("user", {}) if isinstance(record.get("user"), dict) else {}
        username = str(record.get("username", "") or user.get("username", "") or "").strip()
        if not username:
            continue
        candidate_by_username[username] = record
        profile_url = _normalize_github_profile_url(str(user.get("profile_url", "") or "").strip())
        if profile_url:
            candidate_by_profile_url[profile_url] = record
        candidate_name = str(user.get("name", "") or "").strip()
        if candidate_name:
            name_bucket = usernames_by_name.setdefault(candidate_name, [])
            if username not in name_bucket:
                name_bucket.append(username)
        username_bucket = usernames_by_name.setdefault(username, [])
        if username not in username_bucket:
            username_bucket.append(username)

    outreach_by_username = {
        str(record.get("username", "") or "").strip(): record
        for record in outreach
        if str(record.get("username", "") or "").strip()
    }

    leads: list[GitHubReconciliationLead] = []
    for judgment in judgments:
        if judgment.get("stage") != "full":
            continue
        if judgment.get("decision") not in SAVE_DECISIONS:
            continue
        stats.total_saved_judgments += 1
        candidate_name = str(judgment.get("candidate_name", "") or "").strip()
        if not candidate_name:
            stats.skipped_missing_name += 1
            continue
        judgment_profile_url = _normalize_github_profile_url(str(judgment.get("profile_url", "") or "").strip())
        candidate_record = candidate_by_profile_url.get(judgment_profile_url)
        username = ""
        if candidate_record:
            user = candidate_record.get("user", {}) if isinstance(candidate_record.get("user"), dict) else {}
            username = str(candidate_record.get("username", "") or user.get("username", "") or "").strip()
        elif judgment_profile_url:
            stats.skipped_unmatched_profile_url += 1
            continue
        else:
            matching_usernames = usernames_by_name.get(candidate_name, [])
            if len(matching_usernames) == 1:
                username = matching_usernames[0]
                candidate_record = candidate_by_username.get(username)
            else:
                stats.skipped_ambiguous_name += 1
                continue
        if not candidate_record:
            stats.skipped_missing_candidate += 1
            continue
        user = candidate_record.get("user", {}) if isinstance(candidate_record.get("user"), dict) else {}
        hints = build_identity_hints(candidate_record, judgment)
        leads.append(
            GitHubReconciliationLead(
                username=username,
                candidate_name=hints.candidate_name or candidate_name,
                github_url=str(user.get("profile_url", "") or "").strip(),
                company=hints.company,
                location=hints.location,
                title=hints.title,
                decision=str(judgment.get("decision", "") or "").strip(),
                confidence=float(judgment.get("confidence", 0.0) or 0.0),
                rationale=str(judgment.get("rationale", "") or "").strip(),
                source_query=hints.source_query,
                source_channel=hints.source_channel,
                linkedin_hints=hints,
                candidate_payload=candidate_record,
                judgment_payload=judgment,
                outreach_payload=outreach_by_username.get(username, {}),
            )
        )
    stats.leads_loaded = len(leads)
    return GitHubReconciliationLoadBatch(leads=leads, stats=stats)


def load_saved_github_leads(output_dir: str | Path) -> list[GitHubReconciliationLead]:
    return load_github_reconciliation_batch(output_dir).leads


def load_saved_github_reconciliation_batch_with_fallback(
    output_dir: str | Path,
) -> GitHubReconciliationLoadBatch:
    """Load saved GitHub leads, falling back to saves.jsonl when full judgments are absent."""
    batch = load_github_reconciliation_batch(output_dir)
    if batch.leads:
        return batch

    experiment_leads = _build_experiment_leads_from_saves(output_dir)
    leads: list[GitHubReconciliationLead] = []
    for lead in experiment_leads:
        hints = LinkedInIdentityHints(
            candidate_name=lead.candidate_name,
            github_username=lead.github_username,
            github_url=lead.github_url,
            linkedin_url_hint=lead.linkedin_url_hint,
            company=lead.company,
            location=lead.location,
            title=lead.title,
            source_query=lead.source_query,
            source_channel=lead.source_channel,
        )
        leads.append(
            GitHubReconciliationLead(
                username=lead.github_username,
                candidate_name=lead.candidate_name,
                github_url=lead.github_url,
                company=lead.company,
                location=lead.location,
                title=lead.title,
                decision=str(lead.judgment_payload.get("decision", "SAVE") or "SAVE").strip(),
                confidence=float(lead.judgment_payload.get("confidence", 0.0) or 0.0),
                rationale=str(lead.judgment_payload.get("rationale", "") or "").strip(),
                source_query=lead.source_query,
                source_channel=lead.source_channel,
                linkedin_hints=hints,
                candidate_payload=lead.candidate_payload,
                judgment_payload=lead.judgment_payload,
                outreach_payload=lead.outreach_payload,
            )
        )

    stats = GitHubReconciliationLoadStats(
        total_saved_judgments=len(leads),
        leads_loaded=len(leads),
    )
    return GitHubReconciliationLoadBatch(leads=leads, stats=stats)


def _build_experiment_leads_from_saves(output_dir: str | Path) -> list[IdentityResolutionExperimentLead]:
    """Fallback experiment source when full-judgment reconciliation inputs are empty."""
    output_dir = Path(output_dir)
    candidates = read_jsonl(output_dir / "candidates.jsonl")
    saves = read_jsonl(output_dir / "saves.jsonl")
    outreach = read_jsonl(output_dir / "outreach.jsonl")

    candidate_by_username: dict[str, dict] = {}
    candidate_by_profile_url: dict[str, dict] = {}
    for record in candidates:
        user = record.get("user", {}) if isinstance(record.get("user"), dict) else {}
        username = str(record.get("username", "") or user.get("username", "") or "").strip()
        if not username:
            continue
        candidate_by_username[username] = record
        profile_url = _normalize_github_profile_url(str(user.get("profile_url", "") or "").strip())
        if profile_url:
            candidate_by_profile_url[profile_url] = record

    outreach_by_username = {
        str(record.get("username", "") or "").strip(): record
        for record in outreach
        if str(record.get("username", "") or "").strip()
    }

    leads: list[IdentityResolutionExperimentLead] = []
    seen_usernames: set[str] = set()
    for save in saves:
        decision = str(save.get("decision", "") or "").strip()
        if decision not in SAVE_DECISIONS:
            continue
        username = str(save.get("username", "") or "").strip()
        github_url = _normalize_github_profile_url(str(save.get("github_url", "") or "").strip())
        candidate_record = candidate_by_username.get(username)
        if candidate_record is None and github_url:
            candidate_record = candidate_by_profile_url.get(github_url)
        if candidate_record is None:
            continue
        user = candidate_record.get("user", {}) if isinstance(candidate_record.get("user"), dict) else {}
        resolved_username = str(candidate_record.get("username", "") or user.get("username", "") or username).strip()
        if not resolved_username or resolved_username in seen_usernames:
            continue
        seen_usernames.add(resolved_username)

        contact = _build_contact(candidate_record)
        title = (
            str(candidate_record.get("synthesized_headline", "") or "").strip()
            or str(user.get("bio", "") or "").strip()
            or str(save.get("bio", "") or "").strip()
        )
        company = str(user.get("company", "") or "").strip() or str(save.get("company", "") or "").strip()
        location = str(user.get("location", "") or "").strip() or str(save.get("location", "") or "").strip()
        github_profile_url = str(user.get("profile_url", "") or "").strip() or str(save.get("github_url", "") or "").strip()
        candidate_name = (
            str(user.get("name", "") or "").strip()
            or str(save.get("name", "") or "").strip()
            or resolved_username
        )
        leads.append(
            IdentityResolutionExperimentLead(
                github_username=resolved_username,
                candidate_name=candidate_name,
                github_url=github_profile_url,
                company=company,
                location=location,
                title=title,
                lookup_name=build_person_lookup_name(candidate_name, resolved_username),
                source_query=str(candidate_record.get("source_query", "") or "").strip(),
                source_channel=str(candidate_record.get("source_strategy", "") or "").strip(),
                has_direct_linkedin_hint=bool(contact.linkedin_url),
                linkedin_url_hint=contact.linkedin_url,
                candidate_payload=candidate_record,
                judgment_payload=save,
                outreach_payload=outreach_by_username.get(resolved_username, {}),
            )
        )
    return leads


def _name_token_count(name: str) -> int:
    return len([part for part in normalize_person_name(name).split() if part])


def _looks_name_variant(name: str, username: str) -> bool:
    raw = str(name or "").strip()
    lowered = raw.lower()
    if any(marker in raw for marker in ("-", "'", ".", "(", ")", "/")):
        return True
    if _name_token_count(raw) >= 4:
        return True
    if any(len(part) == 1 for part in raw.replace(".", " ").split()):
        return True
    normalized_name = normalize_person_name(raw)
    username_tokens = normalize_person_name(username).split()
    name_tokens = normalized_name.split()
    if normalized_name and username_tokens and name_tokens:
        overlap = set(username_tokens) & set(name_tokens)
        if not overlap:
            return True
    return False


def _looks_stale_or_changed_role(lead: GitHubReconciliationLead) -> bool:
    if not lead.company or not lead.title:
        return True
    source_channel = (lead.source_channel or "").lower()
    if source_channel in {"code_search", "repo_mining"}:
        return True
    judgment_title = str(lead.judgment_payload.get("candidate_title", "") or "").strip()
    if judgment_title and normalize_person_name(judgment_title) != normalize_person_name(lead.title):
        return True
    company = normalize_company_name(lead.company)
    if company and company not in normalize_person_name(lead.title):
        return True
    return False


def _classify_experiment_bucket(
    lead: GitHubReconciliationLead,
    *,
    duplicate_name_keys: set[str],
) -> tuple[str, str]:
    name_key = normalize_person_name(lead.candidate_name)
    if name_key in duplicate_name_keys:
        return "common_name_ambiguous", "duplicate_normalized_name"
    if _looks_name_variant(lead.candidate_name, lead.username):
        return "name_variant", "name_format_or_username_drift"
    if _looks_stale_or_changed_role(lead):
        return "stale_employer_changed_role", "company_or_role_signal_unstable"
    return "easy_exact_name", "unique_clean_name"


def _lead_sort_key(lead: GitHubReconciliationLead) -> tuple:
    return (
        -float(lead.confidence or 0.0),
        normalize_person_name(lead.candidate_name),
        str(lead.username or ""),
    )


def build_identity_resolution_experiment_cohort(
    output_dir: str | Path,
    *,
    primary_bucket_size: int = 10,
    sanity_size: int = 10,
) -> dict[str, list[IdentityResolutionExperimentLead]]:
    """Construct a deterministic experiment cohort from a GitHub run output dir."""
    batch = load_github_reconciliation_batch(output_dir)
    if batch.leads:
        leads = sorted(batch.leads, key=_lead_sort_key)
    else:
        fallback = _build_experiment_leads_from_saves(output_dir)
        leads = sorted(
            [
                GitHubReconciliationLead(
                    username=lead.github_username,
                    candidate_name=lead.candidate_name,
                    github_url=lead.github_url,
                    company=lead.company,
                    location=lead.location,
                    title=lead.title,
                    decision=str(lead.judgment_payload.get("decision", "") or "").strip(),
                    confidence=float(lead.judgment_payload.get("confidence", 0.0) or 0.0),
                    rationale=str(lead.judgment_payload.get("decision_path", "") or "").strip(),
                    source_query=lead.source_query,
                    source_channel=lead.source_channel,
                    linkedin_hints=build_identity_hints(lead.candidate_payload, lead.judgment_payload)
                    if lead.candidate_payload
                    else None,
                    candidate_payload=lead.candidate_payload,
                    judgment_payload=lead.judgment_payload,
                    outreach_payload=lead.outreach_payload,
                )
                for lead in fallback
            ],
            key=_lead_sort_key,
        )
    duplicate_counts: dict[str, int] = {}
    for lead in leads:
        key = normalize_person_name(lead.candidate_name)
        if not key:
            continue
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
    duplicate_keys = {key for key, count in duplicate_counts.items() if count > 1}

    hinted: list[IdentityResolutionExperimentLead] = []
    buckets: dict[str, list[IdentityResolutionExperimentLead]] = {
        bucket: [] for bucket in PRIMARY_EXPERIMENT_BUCKETS
    }
    fallback_pool: list[IdentityResolutionExperimentLead] = []

    for lead in leads:
        has_hint = bool(lead.linkedin_hints and lead.linkedin_hints.linkedin_url_hint)
        experiment_lead = IdentityResolutionExperimentLead(
            github_username=lead.username,
            candidate_name=lead.candidate_name,
            github_url=lead.github_url,
            company=lead.company,
            location=lead.location,
            title=lead.title,
            lookup_name=build_person_lookup_name(lead.candidate_name, lead.username),
            cohort_kind="sanity" if has_hint else "primary",
            source_query=lead.source_query,
            source_channel=lead.source_channel,
            has_direct_linkedin_hint=has_hint,
            linkedin_url_hint=lead.linkedin_hints.linkedin_url_hint if lead.linkedin_hints else "",
            candidate_payload=lead.candidate_payload,
            judgment_payload=lead.judgment_payload,
            outreach_payload=lead.outreach_payload,
        )
        if has_hint:
            hinted.append(experiment_lead)
            continue

        bucket, sampling_reason = _classify_experiment_bucket(
            lead,
            duplicate_name_keys=duplicate_keys,
        )
        experiment_lead.cohort_bucket = bucket
        experiment_lead.sampling_reason = sampling_reason
        buckets[bucket].append(experiment_lead)
        fallback_pool.append(experiment_lead)

    selected_primary: list[IdentityResolutionExperimentLead] = []
    selected_keys: set[str] = set()
    for bucket in PRIMARY_EXPERIMENT_BUCKETS:
        bucket_rows = sorted(buckets[bucket], key=lambda item: (item.github_username, item.candidate_name))
        for item in bucket_rows[:primary_bucket_size]:
            selected_primary.append(item)
            selected_keys.add(item.github_username)

    total_primary_target = primary_bucket_size * len(PRIMARY_EXPERIMENT_BUCKETS)
    if len(selected_primary) < total_primary_target:
        remaining = [
            lead for lead in sorted(fallback_pool, key=lambda item: (item.github_username, item.candidate_name))
            if lead.github_username not in selected_keys
        ]
        for lead in remaining[: max(total_primary_target - len(selected_primary), 0)]:
            selected_primary.append(lead)
            selected_keys.add(lead.github_username)

    selected_sanity = sorted(hinted, key=lambda item: (item.github_username, item.candidate_name))[:sanity_size]
    return {
        "primary": selected_primary[:total_primary_target],
        "sanity": selected_sanity,
    }


def build_identity_resolution_gold_template(
    leads: list[IdentityResolutionExperimentLead],
) -> list[IdentityResolutionGoldLabel]:
    template: list[IdentityResolutionGoldLabel] = []
    for lead in leads:
        template.append(
            IdentityResolutionGoldLabel(
                github_username=lead.github_username,
                candidate_name=lead.candidate_name,
                cohort_kind=lead.cohort_kind,
                cohort_bucket=lead.cohort_bucket,
                github_url=lead.github_url,
                gold_company=lead.company,
                gold_location=lead.location,
                gold_title=lead.title,
                source_query=lead.source_query,
                source_channel=lead.source_channel,
            )
        )
    return template


def export_identity_resolution_experiment_cohort(
    output_dir: str | Path,
    export_dir: str | Path,
    *,
    primary_bucket_size: int = 10,
    sanity_size: int = 10,
) -> dict[str, Path]:
    """Write cohort and blank gold-label template artifacts for the experiment."""
    export_dir = Path(export_dir)
    cohort = build_identity_resolution_experiment_cohort(
        output_dir,
        primary_bucket_size=primary_bucket_size,
        sanity_size=sanity_size,
    )
    primary_leads = cohort["primary"]
    sanity_leads = cohort["sanity"]
    gold_template = build_identity_resolution_gold_template(primary_leads + sanity_leads)

    primary_path = export_dir / "identity_resolution_primary_cohort.json"
    sanity_path = export_dir / "identity_resolution_sanity_cohort.json"
    gold_path = export_dir / "identity_resolution_gold_template.jsonl"
    write_json(primary_path, [lead.to_dict() for lead in primary_leads])
    write_json(sanity_path, [lead.to_dict() for lead in sanity_leads])
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gold_path, "w", encoding="utf-8") as handle:
        for row in gold_template:
            handle.write(json.dumps(row.to_dict()) + "\n")
    return {
        "primary": primary_path,
        "sanity": sanity_path,
        "gold_template": gold_path,
    }
