"""Shared schemas for LinkedIn activity-aware novelty and GitHub reconciliation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RecruiterActivitySnapshot:
    """LinkedIn Recruiter activity visible in list view or profile view."""

    message_count: int = 0
    project_count: int = 0
    view_count: int = 0
    saved_by: str = ""
    raw_activity_text: str = ""
    recent_activity: list[str] = field(default_factory=list)
    last_outbound_contact: str = ""
    reachout_status: str = ""
    sequences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RecruiterActivitySnapshot | None":
        if not isinstance(data, dict):
            return None
        return cls(
            message_count=int(data.get("message_count", 0) or 0),
            project_count=int(data.get("project_count", 0) or 0),
            view_count=int(data.get("view_count", 0) or 0),
            saved_by=str(data.get("saved_by", "") or "").strip(),
            raw_activity_text=str(data.get("raw_activity_text", "") or "").strip(),
            recent_activity=[str(item).strip() for item in data.get("recent_activity", []) if str(item).strip()],
            last_outbound_contact=str(data.get("last_outbound_contact", "") or "").strip(),
            reachout_status=str(data.get("reachout_status", "") or "").strip(),
            sequences=[str(item).strip() for item in data.get("sequences", []) if str(item).strip()],
        )


@dataclass
class LinkedInIdentityHints:
    """Candidate identity hints used to resolve a GitHub lead to LinkedIn."""

    candidate_name: str
    github_username: str = ""
    github_url: str = ""
    linkedin_url_hint: str = ""
    company: str = ""
    location: str = ""
    title: str = ""
    emails: list[str] = field(default_factory=list)
    email_domains: list[str] = field(default_factory=list)
    source_query: str = ""
    source_channel: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LinkedInIdentityHints":
        return cls(
            candidate_name=str(data.get("candidate_name", "") or "").strip(),
            github_username=str(data.get("github_username", "") or "").strip(),
            github_url=str(data.get("github_url", "") or "").strip(),
            linkedin_url_hint=str(data.get("linkedin_url_hint", "") or "").strip(),
            company=str(data.get("company", "") or "").strip(),
            location=str(data.get("location", "") or "").strip(),
            title=str(data.get("title", "") or "").strip(),
            emails=[str(item).strip() for item in data.get("emails", []) if str(item).strip()],
            email_domains=[str(item).strip() for item in data.get("email_domains", []) if str(item).strip()],
            source_query=str(data.get("source_query", "") or "").strip(),
            source_channel=str(data.get("source_channel", "") or "").strip(),
        )


@dataclass
class LinkedInMatchResult:
    """Ranked LinkedIn candidate match for a GitHub lead."""

    matched_profile_url: str = ""
    matched_name: str = ""
    matched_company: str = ""
    matched_title: str = ""
    matched_location: str = ""
    match_confidence: float = 0.0
    match_method: str = ""
    evidence: list[str] = field(default_factory=list)
    ambiguity_reasons: list[str] = field(default_factory=list)
    recruiter_activity: RecruiterActivitySnapshot | None = None
    novelty_pressure: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.recruiter_activity is None:
            payload["recruiter_activity"] = None
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "LinkedInMatchResult":
        return cls(
            matched_profile_url=str(data.get("matched_profile_url", "") or "").strip(),
            matched_name=str(data.get("matched_name", "") or "").strip(),
            matched_company=str(data.get("matched_company", "") or "").strip(),
            matched_title=str(data.get("matched_title", "") or "").strip(),
            matched_location=str(data.get("matched_location", "") or "").strip(),
            match_confidence=float(data.get("match_confidence", 0.0) or 0.0),
            match_method=str(data.get("match_method", "") or "").strip(),
            evidence=[str(item).strip() for item in data.get("evidence", []) if str(item).strip()],
            ambiguity_reasons=[str(item).strip() for item in data.get("ambiguity_reasons", []) if str(item).strip()],
            recruiter_activity=RecruiterActivitySnapshot.from_dict(data.get("recruiter_activity")),
            novelty_pressure=str(data.get("novelty_pressure", "") or "").strip(),
        )


@dataclass
class ReconciliationAssessment:
    """Assessment of whether a LinkedIn profile validates a GitHub lead."""

    same_person: str = "unknown"
    fit_confirmation: str = "unclear"
    reachout_status: str = ""
    novelty_value: str = ""
    summary: str = ""
    contradictions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReconciliationAssessment":
        return cls(
            same_person=str(data.get("same_person", "unknown") or "unknown").strip(),
            fit_confirmation=str(data.get("fit_confirmation", "unclear") or "unclear").strip(),
            reachout_status=str(data.get("reachout_status", "") or "").strip(),
            novelty_value=str(data.get("novelty_value", "") or "").strip(),
            summary=str(data.get("summary", "") or "").strip(),
            contradictions=[str(item).strip() for item in data.get("contradictions", []) if str(item).strip()],
        )


@dataclass
class ReconciliationDecision:
    """Final recruiter-facing decision for a GitHub→LinkedIn reconciliation pass."""

    action: str
    rationale: str
    match_result: LinkedInMatchResult | None = None
    assessment: ReconciliationAssessment | None = None

    def to_dict(self) -> dict:
        payload = {
            "action": self.action,
            "rationale": self.rationale,
            "match_result": self.match_result.to_dict() if self.match_result else None,
            "assessment": self.assessment.to_dict() if self.assessment else None,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "ReconciliationDecision":
        return cls(
            action=str(data.get("action", "") or "").strip(),
            rationale=str(data.get("rationale", "") or "").strip(),
            match_result=LinkedInMatchResult.from_dict(data["match_result"]) if isinstance(data.get("match_result"), dict) else None,
            assessment=ReconciliationAssessment.from_dict(data["assessment"]) if isinstance(data.get("assessment"), dict) else None,
        )
