"""Schemas for retrieval-only identity-resolution experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class IdentityResolutionExperimentLead:
    """One GitHub lead selected into an identity-resolution experiment cohort."""

    github_username: str
    candidate_name: str
    github_url: str = ""
    company: str = ""
    location: str = ""
    title: str = ""
    lookup_name: str = ""
    cohort_kind: str = "primary"
    cohort_bucket: str = ""
    source_query: str = ""
    source_channel: str = ""
    has_direct_linkedin_hint: bool = False
    linkedin_url_hint: str = ""
    sampling_reason: str = ""
    candidate_payload: dict = field(default_factory=dict)
    judgment_payload: dict = field(default_factory=dict)
    outreach_payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityResolutionExperimentLead":
        return cls(
            github_username=str(data.get("github_username", "") or "").strip(),
            candidate_name=str(data.get("candidate_name", "") or "").strip(),
            github_url=str(data.get("github_url", "") or "").strip(),
            company=str(data.get("company", "") or "").strip(),
            location=str(data.get("location", "") or "").strip(),
            title=str(data.get("title", "") or "").strip(),
            lookup_name=str(data.get("lookup_name", "") or "").strip(),
            cohort_kind=str(data.get("cohort_kind", "primary") or "primary").strip(),
            cohort_bucket=str(data.get("cohort_bucket", "") or "").strip(),
            source_query=str(data.get("source_query", "") or "").strip(),
            source_channel=str(data.get("source_channel", "") or "").strip(),
            has_direct_linkedin_hint=bool(data.get("has_direct_linkedin_hint", False)),
            linkedin_url_hint=str(data.get("linkedin_url_hint", "") or "").strip(),
            sampling_reason=str(data.get("sampling_reason", "") or "").strip(),
            candidate_payload=data.get("candidate_payload", {}) if isinstance(data.get("candidate_payload"), dict) else {},
            judgment_payload=data.get("judgment_payload", {}) if isinstance(data.get("judgment_payload"), dict) else {},
            outreach_payload=data.get("outreach_payload", {}) if isinstance(data.get("outreach_payload"), dict) else {},
        )


@dataclass
class IdentityResolutionGoldLabel:
    """Manual gold label for one experiment lead."""

    github_username: str
    candidate_name: str
    cohort_kind: str = "primary"
    cohort_bucket: str = ""
    gold_outcome: str = ""
    gold_public_linkedin_url: str = ""
    gold_display_name: str = ""
    gold_company: str = ""
    gold_location: str = ""
    gold_title: str = ""
    adjudication_note: str = ""
    github_url: str = ""
    source_query: str = ""
    source_channel: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityResolutionGoldLabel":
        return cls(
            github_username=str(data.get("github_username", "") or "").strip(),
            candidate_name=str(data.get("candidate_name", "") or "").strip(),
            cohort_kind=str(data.get("cohort_kind", "primary") or "primary").strip(),
            cohort_bucket=str(data.get("cohort_bucket", "") or "").strip(),
            gold_outcome=str(data.get("gold_outcome", "") or "").strip(),
            gold_public_linkedin_url=str(data.get("gold_public_linkedin_url", "") or "").strip(),
            gold_display_name=str(data.get("gold_display_name", "") or "").strip(),
            gold_company=str(data.get("gold_company", "") or "").strip(),
            gold_location=str(data.get("gold_location", "") or "").strip(),
            gold_title=str(data.get("gold_title", "") or "").strip(),
            adjudication_note=str(data.get("adjudication_note", "") or "").strip(),
            github_url=str(data.get("github_url", "") or "").strip(),
            source_query=str(data.get("source_query", "") or "").strip(),
            source_channel=str(data.get("source_channel", "") or "").strip(),
        )


@dataclass
class SurfacedProfileCandidate:
    """One candidate surfaced by a retrieval strategy."""

    profile_url: str = ""
    public_profile_url: str = ""
    display_name: str = ""
    headline: str = ""
    company: str = ""
    location: str = ""
    rank: int = 0
    source_surface: str = ""
    raw_evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SurfacedProfileCandidate":
        return cls(
            profile_url=str(data.get("profile_url", "") or "").strip(),
            public_profile_url=str(data.get("public_profile_url", "") or "").strip(),
            display_name=str(data.get("display_name", "") or "").strip(),
            headline=str(data.get("headline", "") or "").strip(),
            company=str(data.get("company", "") or "").strip(),
            location=str(data.get("location", "") or "").strip(),
            rank=int(data.get("rank", 0) or 0),
            source_surface=str(data.get("source_surface", "") or "").strip(),
            raw_evidence=str(data.get("raw_evidence", "") or "").strip(),
        )


@dataclass
class StrategyExecutionRecord:
    """One strategy execution for one lead."""

    strategy_name: str
    github_username: str
    candidate_name: str
    cohort_kind: str = "primary"
    cohort_bucket: str = ""
    surfaced_candidates: list[SurfacedProfileCandidate] = field(default_factory=list)
    query: str = ""
    location_filter: str = ""
    blocker_state: str = ""
    duration_seconds: float = 0.0
    interaction_count: int = 0
    started_at: str = ""
    finished_at: str = ""
    final_url: str = ""
    page_title: str = ""
    body_excerpt: str = ""
    top1_profile_url: str = ""
    top3_profile_urls: list[str] = field(default_factory=list)
    top1_correct: bool = False
    top3_contains_correct: bool = False
    wrong_person_top1: bool = False
    ambiguous_only: bool = False
    manual_review_required: bool = False
    no_candidate: bool = False
    aborted: bool = False
    abort_reason: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["surfaced_candidates"] = [candidate.to_dict() for candidate in self.surfaced_candidates]
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyExecutionRecord":
        return cls(
            strategy_name=str(data.get("strategy_name", "") or "").strip(),
            github_username=str(data.get("github_username", "") or "").strip(),
            candidate_name=str(data.get("candidate_name", "") or "").strip(),
            cohort_kind=str(data.get("cohort_kind", "primary") or "primary").strip(),
            cohort_bucket=str(data.get("cohort_bucket", "") or "").strip(),
            surfaced_candidates=[
                SurfacedProfileCandidate.from_dict(item)
                for item in data.get("surfaced_candidates", [])
                if isinstance(item, dict)
            ],
            query=str(data.get("query", "") or "").strip(),
            location_filter=str(data.get("location_filter", "") or "").strip(),
            blocker_state=str(data.get("blocker_state", "") or "").strip(),
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
            interaction_count=int(data.get("interaction_count", 0) or 0),
            started_at=str(data.get("started_at", "") or "").strip(),
            finished_at=str(data.get("finished_at", "") or "").strip(),
            final_url=str(data.get("final_url", "") or "").strip(),
            page_title=str(data.get("page_title", "") or "").strip(),
            body_excerpt=str(data.get("body_excerpt", "") or "").strip(),
            top1_profile_url=str(data.get("top1_profile_url", "") or "").strip(),
            top3_profile_urls=[str(item).strip() for item in data.get("top3_profile_urls", []) if str(item).strip()],
            top1_correct=bool(data.get("top1_correct", False)),
            top3_contains_correct=bool(data.get("top3_contains_correct", False)),
            wrong_person_top1=bool(data.get("wrong_person_top1", False)),
            ambiguous_only=bool(data.get("ambiguous_only", False)),
            manual_review_required=bool(data.get("manual_review_required", False)),
            no_candidate=bool(data.get("no_candidate", False)),
            aborted=bool(data.get("aborted", False)),
            abort_reason=str(data.get("abort_reason", "") or "").strip(),
            notes=[str(item).strip() for item in data.get("notes", []) if str(item).strip()],
        )
