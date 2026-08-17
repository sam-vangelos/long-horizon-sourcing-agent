"""GitHub-specific data types and adapter functions.

Defines intermediate schemas for GitHub API data, plus adapter methods that
map into the existing CandidateSnippet/CandidateProfileSummary schemas used
by judger.py. The LinkedIn schemas are not modified — this module bridges
GitHub data into the existing evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional
import json

from shared.schemas import (
    CandidateSnippet,
    CandidateProfileSummary,
    ContactInfo,
    Experience,
    Education,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# GitHub API data types
# ---------------------------------------------------------------------------

@dataclass
class GitHubUser:
    """Raw user data from GET /users/{username}."""
    username: str
    name: str = ""
    bio: str = ""
    company: str = ""
    location: str = ""
    email: str = ""
    blog: str = ""
    hireable: Optional[bool] = None
    followers: int = 0
    following: int = 0
    public_repos: int = 0
    created_at: str = ""
    profile_url: str = ""
    twitter_username: str = ""
    avatar_url: str = ""
    account_type: str = "User"  # "User" | "Organization"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_api(cls, data: dict) -> GitHubUser:
        """Parse from GitHub API response."""
        return cls(
            username=data.get("login", ""),
            name=data.get("name", "") or "",
            bio=data.get("bio", "") or "",
            company=data.get("company", "") or "",
            location=data.get("location", "") or "",
            email=data.get("email", "") or "",
            blog=data.get("blog", "") or "",
            hireable=data.get("hireable"),
            followers=data.get("followers", 0),
            following=data.get("following", 0),
            public_repos=data.get("public_repos", 0),
            created_at=data.get("created_at", ""),
            profile_url=data.get("html_url", ""),
            twitter_username=data.get("twitter_username", "") or "",
            avatar_url=data.get("avatar_url", "") or "",
            account_type=data.get("type", "User") or "User",
        )


@dataclass
class GitHubRepo:
    """Repository data relevant to candidate assessment."""
    name: str
    full_name: str = ""
    description: str = ""
    language: str = ""
    stars: int = 0
    forks: int = 0
    topics: list[str] = field(default_factory=list)
    pushed_at: str = ""
    created_at: str = ""
    is_fork: bool = False
    html_url: str = ""
    owner_login: str = ""  # to distinguish own repos from org repos

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_api(cls, data: dict) -> GitHubRepo:
        """Parse from GitHub API response."""
        return cls(
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            description=data.get("description", "") or "",
            language=data.get("language", "") or "",
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            topics=data.get("topics", []),
            pushed_at=data.get("pushed_at", ""),
            created_at=data.get("created_at", ""),
            is_fork=data.get("fork", False),
            html_url=data.get("html_url", ""),
            owner_login=data.get("owner", {}).get("login", ""),
        )


# ---------------------------------------------------------------------------
# Registry evidence rendering helpers
# ---------------------------------------------------------------------------

_DOWNLOADS_WINDOW_LABELS: dict[str, str] = {
    "last-month": "last month",
    "last-90-days": "last 90 days",
}


def _format_registry_downloads(
    count: int | float | None,
    *,
    downloads_window: str | None = None,
) -> str | None:
    if count is None or isinstance(count, bool):
        return None
    label = f"{int(count):,} downloads"
    window_label = _DOWNLOADS_WINDOW_LABELS.get(downloads_window or "")
    if window_label:
        return f"{label} ({window_label})"
    return label


def _package_downloads_label(pkg: dict) -> str | None:
    return _format_registry_downloads(
        pkg.get("downloads_last_month"),
        downloads_window=pkg.get("downloads_window"),
    )


def _format_declared_maintainership_ref(evidence_sources: list[str]) -> str:
    """Format ``source_file in owner/repo`` from a declared evidence source."""

    for src in evidence_sources:
        if not isinstance(src, str) or not src.startswith("declared:"):
            continue
        _, _, remainder = src.partition("declared:")
        repo, _, source_file = remainder.partition(":")
        if repo and source_file:
            return f"{source_file} in {repo}"
        if repo:
            return repo
    return "declared roster"


@dataclass
class GitHubCandidate:
    """Enriched candidate combining user profile + repos + contributions + contacts.

    This is the intermediate representation. The adapter methods map it into
    CandidateSnippet and CandidateProfileSummary for the evaluation pipeline.
    """
    user: GitHubUser
    top_repos: list[GitHubRepo] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)  # language -> total bytes
    contact: ContactInfo = field(default_factory=ContactInfo)
    contribution_months: dict[str, int] = field(default_factory=dict)  # last-push-month histogram, one entry per top repo (github/enricher.py) — not commit counts
    readme_text: str = ""  # profile README content
    source_strategy: str = ""  # "user_search" | "code_search" | "repo_mining" | "org_exploration"
    source_query: str = ""
    data_sufficiency: str = "sufficient"  # "sufficient" | "insufficient" | "minimal"

    # These are populated by the cheap model synthesis step
    synthesized_headline: str = ""
    synthesized_experience_entries: list[str] = field(default_factory=list)
    synthesized_experiences: list[dict] = field(default_factory=list)  # raw dicts for Experience
    synthesized_skills: list[str] = field(default_factory=list)

    # --- Enrichment extensions (Phase 2) ---
    # Runtime field name kept for payload compat; render labels use
    # "target project contributions" (S1b rename is render-level).
    frontier_contributions: list[dict] = field(default_factory=list)
    website_text: str = ""
    paper_links: list[str] = field(default_factory=list)
    paper_titles: list[str] = field(default_factory=list)
    repo_readmes: dict[str, str] = field(default_factory=dict)  # repo_name -> readme_text

    # --- Portfolio extraction (cheap model structured output) ---
    portfolio_summary: dict = field(default_factory=dict)

    # --- Capability-aware synthesis ---
    capability_mapping: list[dict] = field(default_factory=list)
    builder_evidence: list[str] = field(default_factory=list)
    user_evidence: list[str] = field(default_factory=list)
    repo_analysis: list[dict] = field(default_factory=list)

    # --- Outreach (stored, never sent) ---
    outreach_copy: dict = field(default_factory=dict)

    # OSS Maintainers Slice 6: optional maintainership classification
    # for the candidate's relationship to the brief's
    # ``target_projects``. Populated by
    # :func:`github.maintainership.classify` when
    # ``brief.target_projects`` is non-empty (spec §11 behavior-
    # preserving contract: classic github briefs leave this ``None``
    # and the evidence + prompt blocks render unchanged). Stored as a
    # plain dict on the candidate (not a typed field) because
    # serialization round-trips through ``to_dict``/JSONL — the
    # classifier dataclass exposes :meth:`MaintainershipClassification.to_dict`
    # for that purpose. Consumers that want the typed shape rebuild
    # it from the dict.
    maintainership: Optional[dict] = None

    # OSS Maintainers multi-hub: optional registry-sourced evidence.
    # Populated by a sibling discovery slice when a candidate is found
    # via npm/crates declared maintainership. Shape (plain dict for JSONL
    # round-trip):
    #   {
    #     "declared_roles": [
    #       {
    #         "hub": "npm" | "crates",
    #         "handle": str,
    #         "package": str,
    #         "role": "maintainer" | "owner",
    #         "corroborated_github_login": str | "",
    #       },
    #     ],
    #     "packages": [
    #       {
    #         "hub": str,
    #         "name": str,
    #         "downloads_last_month": int | None,
    #         "downloads_window": "last-month" | "last-90-days" | None,
    #         "reverse_dependencies": int | None,
    #         "latest_release": str | "",
    #         "release_cadence": str | "",
    #         "deprecated": bool,
    #       },
    #     ],
    #   }
    # When ``None``, ``to_evidence_text()`` and ``to_portfolio_text()``
    # omit the registry section byte-identically (same contract as
    # ``maintainership``).
    registry_evidence: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "user": self.user.to_dict(),
            "top_repos": [r.to_dict() for r in self.top_repos],
            "languages": self.languages,
            "contact": self.contact.to_dict(),
            "contribution_months": self.contribution_months,
            "readme_text": self.readme_text[:500] if self.readme_text else "",
            "source_strategy": self.source_strategy,
            "source_query": self.source_query,
            "data_sufficiency": self.data_sufficiency,
            "synthesized_headline": self.synthesized_headline,
            "synthesized_experience_entries": self.synthesized_experience_entries,
            "synthesized_skills": self.synthesized_skills,
            "frontier_contributions": self.frontier_contributions,
            "website_text": self.website_text[:500] if self.website_text else "",
            "paper_links": self.paper_links,
            "paper_titles": self.paper_titles,
            "repo_readmes": {k: v[:500] for k, v in self.repo_readmes.items()},
            "portfolio_summary": self.portfolio_summary,
            "capability_mapping": self.capability_mapping,
            "builder_evidence": self.builder_evidence,
            "user_evidence": self.user_evidence,
            "maintainership": self.maintainership,
            "registry_evidence": self.registry_evidence,
            "repo_analysis": self.repo_analysis,
            "outreach_copy": self.outreach_copy,
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_snippet(self, source_string_id: int = 0, source_string_name: str = "", page: int = 0, result_rank: int = 0) -> CandidateSnippet:
        """Map to CandidateSnippet for facial judgment.

        Uses synthesized fields from the cheap model. If synthesis hasn't run,
        falls back to raw GitHub data.
        """
        headline = self.synthesized_headline or self.user.bio or ""
        experience_entries = self.synthesized_experience_entries or [
            f"{r.name}: {r.description} ({r.language}, {r.stars}★)"
            for r in self.top_repos[:5] if not r.is_fork
        ]

        return CandidateSnippet(
            name=self.user.name or self.user.username,
            headline=headline,
            current_title=self.user.company or "",
            current_company=self.user.company or "",
            location=self.user.location or "",
            education_snippet="",  # GitHub doesn't expose education
            profile_url=self.user.profile_url,
            source_string_id=source_string_id,
            source_string_name=source_string_name,
            page=page,
            result_rank=result_rank,
            experience_entries=experience_entries,
        )

    def to_profile_summary(self) -> CandidateProfileSummary:
        """Map to CandidateProfileSummary for full judgment.

        Uses synthesized fields from the cheap model for experiences.
        Falls back to raw repo data if synthesis hasn't run.
        """
        # Build experiences from synthesized data or raw repos
        experiences = []
        if self.synthesized_experiences:
            for exp_dict in self.synthesized_experiences:
                experiences.append(Experience(
                    title=exp_dict.get("title", ""),
                    company=exp_dict.get("company", ""),
                    location=exp_dict.get("location", ""),
                    start=exp_dict.get("start", ""),
                    end=exp_dict.get("end", ""),
                    summary_bullets=exp_dict.get("summary_bullets", []),
                ))
        else:
            # Fallback: repos as pseudo-experiences
            for repo in self.top_repos[:10]:
                if repo.is_fork:
                    continue
                bullets = [repo.description] if repo.description else []
                if repo.topics:
                    bullets.append(f"Topics: {', '.join(repo.topics)}")
                bullets.append(f"{repo.stars} stars, {repo.forks} forks")
                experiences.append(Experience(
                    title=f"Maintainer — {repo.name}",
                    company="GitHub (open source)",
                    start=repo.created_at[:10] if repo.created_at else "",
                    end=repo.pushed_at[:10] if repo.pushed_at else "",
                    summary_bullets=bullets,
                ))

        # Skills from synthesized or raw language data
        skills = self.synthesized_skills or sorted(
            self.languages.keys(),
            key=lambda l: self.languages[l],
            reverse=True,
        )[:15]

        headline = self.synthesized_headline or self.user.bio or ""

        return CandidateProfileSummary(
            name=self.user.name or self.user.username,
            profile_url=self.user.profile_url,
            headline=headline,
            experiences=experiences,
            education=[],  # GitHub doesn't expose education
            skills_snippet=skills,
        )

    def to_portfolio_text(self) -> str:
        """Format portfolio summary for GitHub facial triage template."""
        ps = self.portfolio_summary
        if not ps:
            # Fallback to raw data if portfolio extraction hasn't run
            lines = [
                f"Username: {self.user.username}",
                f"Name: {self.user.name}",
                f"Bio: {self.user.bio or '(none)'}",
                f"Company: {self.user.company or '(none)'}",
                f"Location: {self.user.location or '(none)'}",
                f"Followers: {self.user.followers}",
                f"Public repos: {self.user.public_repos}",
                f"Account created: {self.user.created_at[:10] if self.user.created_at else 'unknown'}",
                "",
            ]
            if self.top_repos:
                lines.append("Top repositories:")
                for r in self.top_repos[:5]:
                    topics = f" [{', '.join(r.topics)}]" if r.topics else ""
                    lines.append(f"  - {r.name} ({r.language}, {r.stars} stars){topics}: {r.description or '(no description)'}")
            if self.readme_text:
                lines.append(f"\nProfile README excerpt: {self.readme_text[:500]}")
            self._append_registry_portfolio_lines(lines)
            return "\n".join(lines)

        # Format from structured portfolio summary
        lines = [f"Username: {self.user.username}"]
        lines.append(f"Name: {self.user.name}")
        lines.append(f"Location: {self.user.location or '(none)'}")

        if ps.get("profile_summary"):
            lines.append(f"Profile: {ps['profile_summary']}")

        # Toolchain detection (most important signal)
        tc = ps.get("toolchain_detected", {})
        if tc.get("frameworks"):
            lines.append(f"\nDomain Toolchain Detected: {', '.join(tc['frameworks'])}")
            if tc.get("evidence"):
                for ev in tc["evidence"]:
                    lines.append(f"  - {ev}")
            if tc.get("capability_areas_signaled"):
                lines.append(f"  Capability areas signaled: {', '.join(tc['capability_areas_signaled'])}")

        # Repo summaries
        repos = ps.get("repo_summaries", [])
        if repos:
            lines.append(f"\nRepositories ({len(repos)}):")
            for r in repos:
                fork_tag = " [FORK]" if r.get("is_fork") else ""
                frameworks = f" — frameworks: {', '.join(r['frameworks_used'])}" if r.get("frameworks_used") else ""
                lines.append(f"  - {r['name']} ({r.get('stars', 0)} stars){fork_tag}{frameworks}")
                if r.get("readme_gist"):
                    lines.append(f"    {r['readme_gist']}")
                if r.get("builder_or_user"):
                    lines.append(f"    Assessment: {r['builder_or_user']}")

        # Target project contributions (legacy key for resumed pre-rename checkpoints)
        fc = ps.get("target_project_contributions") or ps.get("frontier_contributions", [])
        if fc:
            lines.append(f"\nTarget Project Contributions: {', '.join(fc)}")

        # Website/papers
        wp = ps.get("website_papers", [])
        if wp:
            lines.append(f"\nWebsite/Papers: {', '.join(wp)}")

        lines.append(f"\nML Signal Strength: {ps.get('ml_signal_strength', 'unknown')}")

        self._append_registry_portfolio_lines(lines)
        return "\n".join(lines)

    def _append_registry_portfolio_lines(self, lines: list[str]) -> None:
        """Compact declared-registry block for facial triage."""
        if not isinstance(self.registry_evidence, dict):
            return
        declared_roles = self.registry_evidence.get("declared_roles") or []
        packages = self.registry_evidence.get("packages") or []
        if not declared_roles and not packages:
            return
        package_facts = {
            (p.get("hub", ""), p.get("name", "")): p
            for p in packages
            if isinstance(p, dict)
        }
        lines.append("\nRegistry Maintainership (declared):")
        for role in declared_roles:
            if not isinstance(role, dict):
                continue
            hub = role.get("hub", "")
            package = role.get("package", "")
            role_label = role.get("role", "maintainer")
            facts = package_facts.get((hub, package), {})
            detail_parts: list[str] = []
            downloads_label = _package_downloads_label(facts)
            if downloads_label:
                detail_parts.append(downloads_label)
            reverse_deps = facts.get("reverse_dependencies")
            if reverse_deps is not None:
                detail_parts.append(f"{reverse_deps:,} reverse dependencies")
            detail_suffix = ""
            if detail_parts:
                detail_suffix = f" ({', '.join(detail_parts)})"
            source_file = role.get("source_file", "")
            source_suffix = f" in {source_file}" if source_file else ""
            lines.append(
                f"  Declared {role_label} of {package} on {hub}"
                f"{detail_suffix}{source_suffix}"
            )

    def to_evidence_text(self) -> str:
        """Format full enriched profile for GitHub full evaluation template."""
        lines = [
            f"Username: {self.user.username}",
            f"Name: {self.user.name}",
            f"Bio: {self.user.bio or '(none)'}",
            f"Company: {self.user.company or '(none)'}",
            f"Location: {self.user.location or '(none)'}",
            f"Followers: {self.user.followers}",
            f"Public repos: {self.user.public_repos}",
            f"Account created: {self.user.created_at[:10] if self.user.created_at else 'unknown'}",
            f"Profile URL: {self.user.profile_url}",
        ]
        account_age_line = self._format_account_age_line()
        if account_age_line:
            lines.insert(-1, account_age_line)

        # Portfolio summary (from cheap model extraction)
        ps = self.portfolio_summary
        if ps:
            tc = ps.get("toolchain_detected", {})
            if tc.get("frameworks"):
                lines.append(f"\n═══ DOMAIN TOOLCHAIN DETECTED ═══")
                lines.append(f"Frameworks: {', '.join(tc['frameworks'])}")
                if tc.get("evidence"):
                    for ev in tc["evidence"]:
                        lines.append(f"  - {ev}")
                if tc.get("capability_areas_signaled"):
                    lines.append(f"Capability areas signaled: {', '.join(tc['capability_areas_signaled'])}")

        # Target project contributions (runtime field: frontier_contributions)
        if self.frontier_contributions:
            lines.append(f"\n═══ TARGET PROJECT CONTRIBUTIONS ═══")
            for fc in self.frontier_contributions:
                lines.append(f"  - {fc.get('repo', '')}: {fc.get('type', '')} — {fc.get('detail', '')}")

        # OSS Maintainers Slice 6: maintainership classification.
        # Renders ONLY when populated (i.e., classifier ran because
        # `brief.target_projects` was non-empty). Behavior-preserving
        # for classic github briefs — `maintainership is None` ⇒
        # this block is omitted byte-identically. Section header
        # signals the LLM that the evidence is authoritative for
        # the recruiter-named projects (per spec §11), not generic
        # OSS prestige.
        if isinstance(self.maintainership, dict) and self.maintainership.get("level"):
            level = self.maintainership.get("level", "contributor")
            confidence = self.maintainership.get("confidence", 0.0)
            evidence_sources = self.maintainership.get("evidence_sources", []) or []
            role_certainty = self.maintainership.get("role_certainty", "inferred")
            lines.append(f"\n═══ MAINTAINERSHIP EVIDENCE (recruiter-named projects) ═══")
            if role_certainty == "declared":
                declared_ref = _format_declared_maintainership_ref(evidence_sources)
                lines.append(f"  Declared level: {level} ({declared_ref})")
                corroboration = self.maintainership.get("corroboration")
                if isinstance(corroboration, dict) and corroboration.get("level"):
                    corr_level = corroboration.get("level", "contributor")
                    corr_confidence = corroboration.get("confidence")
                    if isinstance(corr_confidence, (int, float)):
                        lines.append(
                            f"  Classifier corroborates at {corr_level} "
                            f"({corr_confidence:.2f})"
                        )
                    else:
                        lines.append(
                            f"  Classifier corroborates at {corr_level}"
                        )
            else:
                lines.append("  Role certainty: inferred (signal-based)")
                if isinstance(confidence, (int, float)):
                    lines.append(
                        f"  Classified level: {level} (confidence {confidence:.2f})"
                    )
                else:
                    lines.append(f"  Classified level: {level}")
            if evidence_sources:
                lines.append(f"  Evidence sources:")
                for src in evidence_sources:
                    lines.append(f"    - {src}")
            if self.maintainership.get("signals", {}).get("budget_exhausted"):
                lines.append(
                    "  NOTE: classifier API budget exhausted before all "
                    "signals scored — evidence may be partial."
                )

        # OSS Maintainers multi-hub: declared registry maintainership.
        # Renders ONLY when populated. ``registry_evidence is None`` ⇒
        # byte-identical omission (same invariant as maintainership).
        if isinstance(self.registry_evidence, dict) and (
            self.registry_evidence.get("declared_roles")
            or self.registry_evidence.get("packages")
        ):
            lines.append(f"\n═══ REGISTRY MAINTAINERSHIP (declared) ═══")
            package_facts = {
                (p.get("hub", ""), p.get("name", "")): p
                for p in (self.registry_evidence.get("packages") or [])
                if isinstance(p, dict)
            }
            for role in self.registry_evidence.get("declared_roles") or []:
                if not isinstance(role, dict):
                    continue
                hub = role.get("hub", "")
                package = role.get("package", "")
                role_label = role.get("role", "maintainer")
                handle = role.get("handle", "")
                facts = package_facts.get((hub, package), {})
                reverse_deps = facts.get("reverse_dependencies")
                detail_parts: list[str] = []
                downloads_label = _package_downloads_label(facts)
                if downloads_label:
                    detail_parts.append(downloads_label)
                if reverse_deps is not None:
                    detail_parts.append(
                        f"{reverse_deps:,} reverse dependencies"
                    )
                detail_suffix = ""
                if detail_parts:
                    detail_suffix = f" ({', '.join(detail_parts)})"
                source_file = role.get("source_file", "")
                source_suffix = f" in {source_file}" if source_file else ""
                lines.append(
                    f"  Declared {role_label} of {package} on {hub}"
                    f"{detail_suffix}{source_suffix}"
                )
                if handle:
                    lines.append(f"    Registry handle: {handle}")
                corroborated = role.get("corroborated_github_login", "")
                if corroborated:
                    lines.append(
                        f"    Corroborated GitHub login: {corroborated}"
                    )
            rendered_packages = {
                (r.get("hub", ""), r.get("package", ""))
                for r in (self.registry_evidence.get("declared_roles") or [])
                if isinstance(r, dict)
            }
            for pkg in self.registry_evidence.get("packages") or []:
                if not isinstance(pkg, dict):
                    continue
                key = (pkg.get("hub", ""), pkg.get("name", ""))
                if key in rendered_packages:
                    continue
                hub = pkg.get("hub", "")
                name = pkg.get("name", "")
                detail_parts: list[str] = []
                downloads_label = _package_downloads_label(pkg)
                if downloads_label:
                    detail_parts.append(downloads_label)
                reverse_deps = pkg.get("reverse_dependencies")
                if reverse_deps is not None:
                    detail_parts.append(f"{reverse_deps:,} reverse dependencies")
                latest_release = pkg.get("latest_release", "")
                if latest_release:
                    detail_parts.append(f"latest release {latest_release}")
                release_cadence = pkg.get("release_cadence", "")
                if release_cadence:
                    detail_parts.append(f"release cadence {release_cadence}")
                if pkg.get("deprecated"):
                    detail_parts.append("deprecated")
                line = f"  Package {name} on {hub}"
                if detail_parts:
                    line += f" ({', '.join(detail_parts)})"
                lines.append(line)

        # Top repos with READMEs
        lines.append(f"\n═══ REPOSITORIES ═══")
        for r in self.top_repos[:10]:
            fork_tag = " [FORK]" if r.is_fork else ""
            topics = f" Topics: {', '.join(r.topics)}" if r.topics else ""
            lines.append(f"\n{r.name} ({r.language}, {r.stars} stars, {r.forks} forks){fork_tag}")
            lines.append(f"  Description: {r.description or '(none)'}")
            if topics:
                lines.append(f"  {topics}")
            lines.append(f"  Last pushed: {r.pushed_at[:10] if r.pushed_at else 'unknown'}")
            # Include README if available
            readme = self.repo_readmes.get(r.name, "")
            if readme:
                lines.append(f"  README excerpt:\n    {readme[:1500]}")

        # Repo analysis from portfolio extraction
        project_quality = (self.portfolio_summary or {}).get("project_quality")
        if self.repo_analysis or project_quality:
            lines.append(f"\n═══ REPO ANALYSIS (from enrichment) ═══")
            if isinstance(project_quality, dict) and project_quality.get("band"):
                pq_repo = project_quality.get("repo", "")
                pq_band = project_quality.get("band", "")
                pq_score = project_quality.get("score")
                pq_line = f"  Project quality ({pq_repo}): {pq_band}"
                if pq_score is not None:
                    pq_line += f" (score {pq_score:.2f})"
                lines.append(pq_line)
            for ra in self.repo_analysis:
                lines.append(f"  {ra.get('name', '')}: {ra.get('what_it_does', '')}")
                if ra.get("builder_signals"):
                    lines.append(f"    Builder signals: {', '.join(ra['builder_signals'])}")
                lines.append(f"    Relevance: {ra.get('relevance', 'unknown')}")

        # Website and papers
        if self.website_text or self.paper_titles:
            lines.append(f"\n═══ WEBSITE & PAPERS ═══")
            if self.paper_titles:
                lines.append("Papers:")
                for pt in self.paper_titles:
                    lines.append(f"  - {pt}")
            if self.website_text:
                lines.append(f"Website content excerpt:\n  {self.website_text[:1000]}")

        # Profile README
        if self.readme_text:
            lines.append(f"\n═══ PROFILE README ═══")
            lines.append(self.readme_text[:2000])

        # Languages
        if self.languages:
            sorted_langs = sorted(self.languages.items(), key=lambda x: x[1], reverse=True)
            lang_str = ", ".join(f"{lang} ({weight})" for lang, weight in sorted_langs[:10])
            lines.append(f"\n═══ LANGUAGE DISTRIBUTION ═══")
            lines.append(lang_str)

        repo_activity_lines = self._format_repo_activity_lines()
        if repo_activity_lines:
            lines.extend(repo_activity_lines)

        # Builder/user evidence from synthesis
        if self.builder_evidence:
            lines.append(f"\n═══ BUILDER EVIDENCE (from enrichment) ═══")
            for be in self.builder_evidence:
                lines.append(f"  + {be}")
        if self.user_evidence:
            lines.append(f"\n═══ USER-LEVEL EVIDENCE (from enrichment) ═══")
            for ue in self.user_evidence:
                lines.append(f"  - {ue}")

        # Contact info
        # OSS Maintainers Slice 8: include linkedin_url when discovered.
        # Provenance label ("blog" / "bio" / "readme") is rendered
        # alongside so the LLM (and downstream resolver) can band
        # confidence — a blog-field URL is recruiter-deliberate; a
        # bio/readme extraction is corroborating.
        if (
            self.contact.emails
            or self.contact.website
            or self.contact.linkedin_url
        ):
            lines.append(f"\n═══ CONTACT ═══")
            if self.contact.emails:
                lines.append(f"  Emails: {', '.join(self.contact.emails)}")
            if self.contact.website:
                lines.append(f"  Website: {self.contact.website}")
            if self.contact.linkedin_url:
                source = self.contact.linkedin_url_source or "unknown"
                lines.append(
                    f"  LinkedIn: {self.contact.linkedin_url} (via {source})"
                )

        return "\n".join(lines)

    def _format_account_age_line(self) -> str | None:
        created_at = self.user.created_at
        if not created_at or not str(created_at).strip():
            return None
        try:
            created = datetime.fromisoformat(
                str(created_at).replace("Z", "+00:00")
            )
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            now = _now_utc()
            years = (now - created).total_seconds() / (365.25 * 24 * 3600)
            today = now.strftime("%Y-%m-%d")
            return f"Account age: {years:.1f} years (as of {today})"
        except (ValueError, TypeError, OverflowError):
            return None

    def _format_repo_activity_lines(self) -> list[str]:
        repos = self.top_repos
        if not repos:
            return []

        non_fork = [r for r in repos if not r.is_fork]
        use_repos = non_fork if non_fork else repos
        all_forks = not non_fork

        created_dates: list[datetime] = []
        pushed_dates: list[datetime] = []
        recent_push_count = 0
        now = _now_utc()
        twelve_months_ago = now - timedelta(days=365.25)

        for repo in use_repos:
            created = self._parse_repo_datetime(repo.created_at)
            if created is not None:
                created_dates.append(created)
            pushed = self._parse_repo_datetime(repo.pushed_at)
            if pushed is not None:
                pushed_dates.append(pushed)
                if pushed >= twelve_months_ago:
                    recent_push_count += 1

        if not created_dates and not pushed_dates:
            return []

        lines = ["\n═══ REPO ACTIVITY SPAN ═══"]
        if created_dates:
            earliest = min(created_dates)
            lines.append(f"First owned repo created: {earliest.strftime('%Y-%m')}")
        if pushed_dates:
            latest = max(pushed_dates)
            lines.append(f"Most recent push: {latest.strftime('%Y-%m')}")
        suffix = " (all forks)" if all_forks else ""
        lines.append(
            f"Repos pushed to in the last 12 months: {recent_push_count} of {len(use_repos)}{suffix}"
        )
        return lines

    @staticmethod
    def _parse_repo_datetime(date_str: str) -> datetime | None:
        if not date_str or not str(date_str).strip():
            return None
        try:
            dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError, OverflowError):
            return None

    def assess_data_sufficiency(self) -> str:
        """Determine if there's enough data to justify an Opus evaluation call."""
        non_fork_repos = [r for r in self.top_repos if not r.is_fork]
        has_bio = bool(self.user.bio and len(self.user.bio) > 10)
        has_repos = len(non_fork_repos) >= 1
        has_activity = any(c > 0 for c in self.contribution_months.values())

        # Strong prior: frontier contributions, high followers, or starred repos
        has_strong_prior = (
            bool(self.frontier_contributions) or
            self.user.followers > 200 or
            any(r.stars > 50 for r in non_fork_repos)
        )

        if has_strong_prior:
            self.data_sufficiency = "sufficient"
        elif has_repos and (has_bio or has_activity):
            self.data_sufficiency = "sufficient"
        elif has_repos or has_bio:
            self.data_sufficiency = "minimal"
        else:
            self.data_sufficiency = "insufficient"

        return self.data_sufficiency


# ---------------------------------------------------------------------------
# GitHub search query (analogous to SearchString for LinkedIn)
# ---------------------------------------------------------------------------

@dataclass
class GitHubSearchQuery:
    """A single search query to execute against GitHub API."""
    id: int
    name: str  # human-readable description
    query: str  # GitHub search query string
    channel: str  # "user_search" | "code_search" | "repo_mining" | "org_exploration" | "topic_search" | "stargazer_mining" | "graph_expansion"
    status: str = "queued"  # "queued" | "in_progress" | "done" | "skipped" | "error"
    result_count: int = 0
    candidates_discovered: int = 0
    saves: list[str] = field(default_factory=list)
    notes: str = ""
    # For repo mining: the specific repo to mine
    target_repo: str = ""  # "owner/repo"
    # For org exploration: the specific org
    target_org: str = ""
    # For registry discovery: ecosystem and named packages to target
    target_ecosystem: str = ""  # "npmjs.org" | "crates.io" | ...
    target_packages: list[str] = field(default_factory=list)  # ["lodash", "tokio"]
    # Auto-segmentation tracking
    parent_query_id: Optional[int] = None  # if this was auto-segmented from a larger query
    hit_result_cap: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GitHubSearchQuery:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# GitHub progress checkpoint
# ---------------------------------------------------------------------------

@dataclass
class GitHubProgress:
    """Resumable checkpoint for GitHub sourcing sessions."""
    brief_name: str
    queries: list[GitHubSearchQuery] = field(default_factory=list)
    candidates_discovered: int = 0
    candidates_enriched: int = 0
    candidates_saved: int = 0
    candidates_rejected: int = 0
    candidates_insufficient: int = 0
    current_query_id: Optional[int] = None
    discovered_usernames: list[str] = field(default_factory=list)  # global dedup (serialized as list)
    mined_repos: list[str] = field(default_factory=list)  # repos whose contributors have been fetched
    api_calls_made: int = 0
    graph_expansion_queue: list[dict] = field(default_factory=list)
    # Each entry: {"username": "...", "reason": "SAVE", "confidence": 0.85, "capability_area": "...", "added_at": "..."}
    graph_expansion_processed: list[str] = field(default_factory=list)  # usernames already processed

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> GitHubProgress:
        queries = [GitHubSearchQuery.from_dict(q) for q in d.get("queries", [])]
        return cls(
            brief_name=d["brief_name"],
            queries=queries,
            candidates_discovered=d.get("candidates_discovered", 0),
            candidates_enriched=d.get("candidates_enriched", 0),
            candidates_saved=d.get("candidates_saved", 0),
            candidates_rejected=d.get("candidates_rejected", 0),
            candidates_insufficient=d.get("candidates_insufficient", 0),
            current_query_id=d.get("current_query_id"),
            discovered_usernames=d.get("discovered_usernames", []),
            mined_repos=d.get("mined_repos", []),
            api_calls_made=d.get("api_calls_made", 0),
            graph_expansion_queue=d.get("graph_expansion_queue", []),
            graph_expansion_processed=d.get("graph_expansion_processed", []),
        )

    @classmethod
    def from_file(cls, path: str) -> GitHubProgress:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


# ---------------------------------------------------------------------------
# GitHub batch report (for strategy adaptation)
# ---------------------------------------------------------------------------

@dataclass
class GitHubBatchReport:
    """Summary of a batch of queries, sent to Opus for adaptation."""
    batch_name: str
    queries_run: int = 0
    queries_with_saves: int = 0
    total_candidates_discovered: int = 0
    total_saves: int = 0
    total_rejects: int = 0
    total_insufficient: int = 0
    top_performing_queries: list[dict] = field(default_factory=list)
    zero_save_query_ids: list[int] = field(default_factory=list)
    common_languages_in_saves: list[str] = field(default_factory=list)
    common_repos_in_saves: list[str] = field(default_factory=list)
    queries_hitting_result_cap: list[int] = field(default_factory=list)
    query_details: list[dict] = field(default_factory=list)
    # Each: {"query_id": int, "name": str, "query_string": str, "channel": str, "saves": int, "candidates": int}
    channel_metrics: list[dict] = field(default_factory=list)
    signal_markers: list[dict] = field(default_factory=list)
    noise_markers: list[dict] = field(default_factory=list)
    exhaustion_markers: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_text(self) -> str:
        lines = [f'Batch "{self.batch_name}" — {self.queries_run} queries complete.']
        lines.append(f"- {self.total_candidates_discovered} candidates discovered, {self.total_saves} saved, {self.total_rejects} rejected, {self.total_insufficient} insufficient data")
        if self.top_performing_queries:
            top = ", ".join(
                f"Query #{q['query_id']} \"{q.get('name', '')}\" ({q.get('saves', 0)} saves)"
                for q in self.top_performing_queries
            )
            lines.append(f"- Top performers: {top}")
        if self.zero_save_query_ids:
            lines.append(f"- Zero-save queries: {', '.join(f'#{qid}' for qid in self.zero_save_query_ids)}")
        if self.common_languages_in_saves:
            lines.append(f"- Common languages in saves: {', '.join(self.common_languages_in_saves)}")
        if self.common_repos_in_saves:
            lines.append(f"- Common repos in saves: {', '.join(self.common_repos_in_saves)}")
        if self.queries_hitting_result_cap:
            lines.append(f"- Queries hitting 1,000 cap: {', '.join(f'#{qid}' for qid in self.queries_hitting_result_cap)}")
        if self.channel_metrics:
            lines.append("- Channel yield:")
            for metric in self.channel_metrics:
                lines.append(
                    f"  {metric.get('channel', '?')}: "
                    f"{metric.get('queries', 0)} queries, "
                    f"{metric.get('candidates', 0)} candidates, "
                    f"{metric.get('saves', 0)} saves"
                )
        if self.signal_markers:
            lines.append("- Signal markers:")
            for marker in self.signal_markers:
                lines.append(
                    f"  {marker.get('label', marker.get('kind', 'signal'))}: "
                    f"{marker.get('count', 0)}"
                )
        if self.noise_markers:
            lines.append("- Noise markers:")
            for marker in self.noise_markers:
                lines.append(
                    f"  {marker.get('label', marker.get('kind', 'noise'))}: "
                    f"{marker.get('count', 0)}"
                )
        if self.exhaustion_markers:
            lines.append("- Channel exhaustion/degradation:")
            for marker in self.exhaustion_markers:
                lines.append(
                    f"  {marker.get('channel', '?')}: {marker.get('reason', '')}"
                )
        if self.query_details:
            lines.append("- Per-query breakdown:")
            for qd in self.query_details:
                status = f"{qd['saves']} saves" if qd.get('saves') else "zero saves"
                lines.append(f"  #{qd['query_id']} [{qd.get('channel', '?')}] [{status}, {qd.get('candidates', 0)} candidates]: {qd.get('query_string', '')[:150]}")
        return "\n".join(lines)
