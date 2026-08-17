"""GitHub profile enrichment pipeline.

Takes a username, fetches data from multiple GitHub API endpoints, then uses
the cheap model to synthesize sparse GitHub data into CandidateSnippet and
CandidateProfileSummary objects compatible with the existing evaluation pipeline.

The enrichment pipeline aggregates:
    1. User profile (bio, company, location, email)
    2. Top repositories (descriptions, topics, stars, languages)
    3. Language distribution across all repos
    4. Contribution frequency (monthly commit proxy via repo push dates)
    5. Profile README (self-description)
    6. Contact info (commit emails, social links)

Usage:
    async with GitHubClient() as client:
        enricher = GitHubEnricher(client)
        candidate = await enricher.enrich("torvalds")
        if candidate.data_sufficiency != "insufficient":
            snippet = candidate.to_snippet()
            profile = candidate.to_profile_summary()
"""

from __future__ import annotations

import json
import logging
import re
import base64
from html.parser import HTMLParser
from typing import Optional

from github.client import GitHubClient
from github.schemas import (
    GitHubUser,
    GitHubRepo,
    GitHubCandidate,
    ContactInfo,
)
from shared.contact_discovery import discover_contacts
from shared.failures import ApiBudgetExhaustedError, is_api_budget_exhausted_error
from shared.llm_clients import cheap_llm
import github.config as gc
from shared.safety import fetch_text_if_safe, validate_public_url

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML text extraction helper
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Simple HTML to text extractor."""
    def __init__(self):
        super().__init__()
        self._text = []
        self._skip = False
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True
    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
    def handle_data(self, data):
        if not self._skip:
            self._text.append(data.strip())
    def get_text(self) -> str:
        return " ".join(t for t in self._text if t)

def _html_to_text(html: str, max_length: int = 10000) -> str:
    """Extract text from HTML, capped at max_length."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html[:50000])  # Don't parse huge pages
        return extractor.get_text()[:max_length]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Synthesis prompts
# ---------------------------------------------------------------------------

_PORTFOLIO_EXTRACTION_SYSTEM = """You are a technical analysis assistant. Given raw GitHub profile data, extract structured signals for a recruiting evaluation pipeline.

Your output must be JSON with these fields:
{{
    "toolchain_detected": {{
        "frameworks": ["list of brief-relevant frameworks found in repos"],
        "evidence": ["specific evidence for each framework — e.g., 'repo X imports <package>', 'repo Y has domain-specific configs'"],
        "capability_areas_signaled": ["which capability areas these frameworks signal — use the area names provided below"]
    }},
    "repo_summaries": [
        {{
            "name": "repo name",
            "stars": 0,
            "description": "what the repo does",
            "readme_gist": "2-3 sentence summary of the README content",
            "frameworks_used": ["frameworks detected in this repo"],
            "topics": ["repo topics"],
            "is_fork": false,
            "builder_or_user": "builder — custom training loop, not default config"
        }}
    ],
    "target_project_contributions": ["list of contributions to brief-named target projects"],
    "profile_summary": "one-line summary: role, followers, account age, key signals",
    "website_papers": ["any paper titles or website highlights"],
    "primary_languages": ["top languages by usage"],
    "ml_signal_strength": "strong/moderate/weak/none — how likely is this person an ML practitioner?"
}}

CAPABILITY AREAS for this role:
{capability_areas}

CODE SIGNALS — these are practitioner fingerprints from the brief. Finding these in repos is high-signal evidence:
{toolchain_list}

Guidelines:
- Focus on identifying WHICH brief-relevant frameworks/tools are used in repos — this is the #1 signal
- For each repo, determine if it shows BUILDER work (custom implementations, training scripts, eval harnesses) or USER work (forks with minimal changes, API wrappers, tutorial notebooks)
- If repos import or configure brief-named code signals, note this prominently
- Map framework usage to capability areas
- Be specific about what each repo actually does based on its README and description
- If data is very sparse, set ml_signal_strength to "none" or "weak"
"""

_PORTFOLIO_EXTRACTION_USER = """Here is the GitHub profile data to analyze:

**Username:** {username}
**Name:** {name}
**Bio:** {bio}
**Company:** {company}
**Location:** {location}
**Account created:** {created_at}
**Followers:** {followers}
**Public repos:** {public_repos}

**Profile README excerpt:**
{readme}

**Top repositories (by stars, non-fork):**
{repos_text}

**Repo README excerpts:**
{repo_readmes_text}

**Language distribution:**
{languages_text}

**Target project contributions:**
{target_project_text}

**Website/paper content:**
{website_text}

Analyze this profile and output the structured JSON format."""


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------

class GitHubEnricher:
    """Multi-step profile enrichment pipeline."""

    def __init__(self, client: GitHubClient, brief=None, safety_event_recorder=None):
        self._client = client
        self._brief = brief
        self._target_project_contributor_cache: dict[str, set[str]] = {}  # repo -> set of contributor logins
        self._safety_event_recorder = safety_event_recorder

    async def light_enrich(
        self,
        username: str,
        source_strategy: str = "",
        source_query: str = "",
    ) -> Optional[GitHubCandidate]:
        """Fetch ONLY the user profile (1 API call) for pre-enrichment geo gating.

        Returns a GitHubCandidate with just the user profile populated.
        Call full_enrich() on the result to complete enrichment if the
        candidate passes the geo check.
        """
        user_data = await self._client.get_user(username)
        if not user_data:
            return None

        user = GitHubUser.from_api(user_data)
        return GitHubCandidate(
            user=user,
            source_strategy=source_strategy,
            source_query=source_query,
        )

    async def full_enrich(
        self,
        candidate: GitHubCandidate,
        skip_synthesis: bool = False,
    ) -> GitHubCandidate:
        """Complete enrichment on a light-enriched candidate.

        Performs steps 2-8 (repos, languages, READMEs, contacts, synthesis)
        on a candidate that already has its user profile fetched.
        """
        username = candidate.user.username
        return await self._enrich_remaining(candidate, username, skip_synthesis)

    async def enrich(
        self,
        username: str,
        source_strategy: str = "",
        source_query: str = "",
        skip_synthesis: bool = False,
    ) -> Optional[GitHubCandidate]:
        """Enrich a GitHub user into a full GitHubCandidate.

        Returns None only on critical API failure (user doesn't exist).
        Returns GitHubCandidate with data_sufficiency="insufficient" for
        sparse profiles — caller decides whether to evaluate.
        """
        # Step 1: Fetch user profile
        user_data = await self._client.get_user(username)
        if not user_data:
            return None

        user = GitHubUser.from_api(user_data)
        candidate = GitHubCandidate(
            user=user,
            source_strategy=source_strategy,
            source_query=source_query,
        )

        return await self._enrich_remaining(candidate, username, skip_synthesis)

    async def _enrich_remaining(
        self,
        candidate: GitHubCandidate,
        username: str,
        skip_synthesis: bool = False,
    ) -> GitHubCandidate:
        """Steps 2-8 of enrichment (everything after user profile fetch)."""

        # Step 2: Fetch repositories
        repos_data = await self._client.get_user_repos(username, max_repos=gc.MAX_REPOS_PER_USER)
        candidate.top_repos = [GitHubRepo.from_api(r) for r in repos_data]

        # Sort non-fork repos by stars for synthesis
        non_fork = [r for r in candidate.top_repos if not r.is_fork]
        non_fork.sort(key=lambda r: r.stars, reverse=True)
        candidate.top_repos = non_fork[:gc.MAX_REPOS_PER_USER]

        # Step 3: Aggregate languages across repos
        candidate.languages = self._aggregate_languages(candidate.top_repos)

        # Step 4: Profile README (optional — 404 is fine)
        readme = await self._client.get_profile_readme(username)
        if readme:
            candidate.readme_text = readme

        # Step 4b: Fetch repo READMEs (top non-fork repos)
        await self._fetch_repo_readmes(candidate)

        # Step 4c: Check target-project contributions
        await self._check_target_project_contributions(candidate)

        # Step 4d: Crawl website and papers
        await self._crawl_website_and_papers(candidate)

        # Step 5: Contact discovery
        candidate.contact = await discover_contacts(self._client, username, candidate.top_repos[:5])

        # Step 6: Contribution frequency proxy
        candidate.contribution_months = self._estimate_contribution_months(candidate.top_repos)

        # Step 7: Assess data sufficiency
        candidate.assess_data_sufficiency()

        # Step 8: Cheap model synthesis (only if sufficient data)
        if not skip_synthesis and candidate.data_sufficiency != "insufficient":
            await self._extract_portfolio(candidate)

        return candidate

    async def enrich_from_search_result(
        self,
        search_item: dict,
        source_strategy: str = "",
        source_query: str = "",
    ) -> Optional[GitHubCandidate]:
        """Enrich from a search result dict (has login but minimal data)."""
        username = search_item.get("login", "")
        if not username:
            return None
        return await self.enrich(username, source_strategy, source_query)

    # ── Portfolio Extraction ────────────────────────────────────────

    async def _extract_portfolio(self, candidate: GitHubCandidate) -> None:
        """Use cheap model to extract structured portfolio signals from GitHub data."""
        repos_text = self._format_repos(candidate.top_repos[:10])
        languages_text = self._format_languages(candidate.languages)
        readme_excerpt = (candidate.readme_text[:1000] + "...") if len(candidate.readme_text) > 1000 else candidate.readme_text

        # Format repo READMEs
        repo_readmes_parts = []
        for name, content in list(candidate.repo_readmes.items())[:5]:
            repo_readmes_parts.append(f"--- {name} ---\n{content[:800]}")
        repo_readmes_text = "\n\n".join(repo_readmes_parts) if repo_readmes_parts else "(no repo READMEs fetched)"

        # Format target-project contributions
        target_project_text = "\n".join(
            f"- {fc.get('repo', '')}: {fc.get('type', '')} — {fc.get('detail', '')}"
            for fc in candidate.frontier_contributions
        ) if candidate.frontier_contributions else "(none detected)"

        # Format website/papers
        website_parts = []
        if candidate.paper_titles:
            website_parts.append("Papers: " + ", ".join(candidate.paper_titles))
        if candidate.website_text:
            website_parts.append(f"Website: {candidate.website_text[:500]}")
        website_text = "\n".join(website_parts) if website_parts else "(none)"

        # Build capability areas and toolchain list for the prompt
        capability_areas = ""
        toolchain_list = ""
        if self._brief:
            if hasattr(self._brief, '_new_brief') and self._brief._new_brief:
                nb = self._brief._new_brief
                capability_areas = "\n".join(f"- {ca.name}: {ca.description}" for ca in nb.capability_areas)
                # Collect code signals
                for ca in nb.capability_areas:
                    if hasattr(ca, 'github_code_signals') and ca.github_code_signals:
                        toolchain_list += f"{ca.name}: {', '.join(ca.github_code_signals)}\n"
        if not toolchain_list:
            toolchain_list = "(not specified)"

        system_prompt = _PORTFOLIO_EXTRACTION_SYSTEM.format(
            capability_areas=capability_areas or "(not specified)",
            toolchain_list=toolchain_list,
        )

        user_prompt = _PORTFOLIO_EXTRACTION_USER.format(
            username=candidate.user.username,
            name=candidate.user.name,
            bio=candidate.user.bio,
            company=candidate.user.company,
            location=candidate.user.location,
            created_at=candidate.user.created_at[:10] if candidate.user.created_at else "",
            followers=candidate.user.followers,
            public_repos=candidate.user.public_repos,
            readme=readme_excerpt or "(no profile README)",
            repos_text=repos_text or "(no repositories)",
            repo_readmes_text=repo_readmes_text,
            languages_text=languages_text or "(no language data)",
            target_project_text=target_project_text,
            website_text=website_text,
        )

        usage_context = {
            "stage": "github_portfolio_extraction",
            "source": "github",
            "username": candidate.user.username,
            "source_strategy": candidate.source_strategy,
            "source_query": candidate.source_query,
        }

        try:
            result = cheap_llm(
                system_prompt,
                user_prompt,
                expect_json=True,
                usage_context=usage_context,
            )
            if isinstance(result, dict):
                candidate.portfolio_summary = result
                candidate.capability_mapping = [result.get("toolchain_detected", {})]
                candidate.builder_evidence = []
                candidate.user_evidence = []
                candidate.repo_analysis = result.get("repo_summaries", [])
                # Extract builder/user evidence from repo analysis
                for repo in candidate.repo_analysis:
                    assessment = repo.get("builder_or_user", "")
                    if "builder" in assessment.lower():
                        candidate.builder_evidence.append(f"{repo.get('name', '')}: {assessment}")
                    elif "user" in assessment.lower():
                        candidate.user_evidence.append(f"{repo.get('name', '')}: {assessment}")
                # Also keep backward compat fields
                candidate.synthesized_headline = result.get("profile_summary", "")
                candidate.synthesized_skills = result.get("primary_languages", [])
        except Exception as e:
            if is_api_budget_exhausted_error(e):
                raise ApiBudgetExhaustedError(str(e)) from e
            print(f"    [enricher] Portfolio extraction failed for {candidate.user.username}: {e}")

    # ── Enrichment Helpers ───────────────────────────────────────────

    async def _fetch_repo_readmes(self, candidate: GitHubCandidate) -> None:
        """Fetch README content from top non-fork repos."""
        non_fork = [r for r in candidate.top_repos if not r.is_fork]
        for repo in non_fork[:gc.MAX_READMES_PER_CANDIDATE]:
            try:
                readme = await self._client.get_repo_readme(repo.full_name)
                if readme:
                    candidate.repo_readmes[repo.name] = readme
            except Exception:
                pass  # Not critical — skip silently

    async def _check_target_project_contributions(self, candidate: GitHubCandidate) -> None:
        """Check if candidate has contributed to brief-named target projects."""
        target_projects: list[str] = []
        if self._brief is not None:
            target_projects = list(getattr(self._brief, "target_projects", []) or [])

        for target_repo in target_projects:
            if not isinstance(target_repo, str):
                continue
            target_repo = target_repo.strip()
            if not target_repo:
                continue

            if target_repo not in self._target_project_contributor_cache:
                try:
                    contributors = await self._client.get_repo_contributors(
                        target_repo, max_contributors=500
                    )
                    self._target_project_contributor_cache[target_repo] = {
                        c.get("login", "").lower() for c in contributors
                    }
                except Exception:
                    self._target_project_contributor_cache[target_repo] = set()

            if candidate.user.username.lower() in self._target_project_contributor_cache[target_repo]:
                candidate.frontier_contributions.append({
                    "repo": target_repo,
                    "type": "contributor",
                    "detail": f"Listed as contributor to {target_repo}",
                })

    # Valid arXiv ID formats: new-style "2301.00001" or old-style "hep-th/9901001"
    _ARXIV_ID_RE = re.compile(
        r'^(\d{4}\.\d{4,5}(v\d+)?|[a-zA-Z-]+(\.[a-zA-Z-]+)?/\d{7}(v\d+)?)$'
    )

    _REDIRECT_STATUSES = {301, 302, 303, 307, 308}
    _MAX_REDIRECTS = 5

    async def _crawl_website_and_papers(self, candidate: GitHubCandidate) -> None:
        """Crawl personal website and discover papers.

        All candidate-controlled URLs are validated against SSRF blocklists
        before fetching, and redirects are followed manually with per-hop
        validation.
        """
        username = candidate.user.username
        urls_to_check: list[str] = []

        # Blog URL from profile
        blog = candidate.user.blog
        if blog and "linkedin.com" not in blog.lower():
            if not blog.startswith("http"):
                blog = f"https://{blog}"
            safe, reason = await validate_public_url(blog)
            if safe:
                urls_to_check.append(blog)
            else:
                self._record_safety_event(
                    "blocked_external_url",
                    {"url": blog, "reason": reason, "source": "profile_blog"},
                )
                logger.warning("Blocked blog URL for %s: %s (%s)", username, blog, reason)

        # Look for URLs in profile README
        if candidate.readme_text:
            url_pattern = r'https?://[^\s\)\]>\"\'<]+'
            found_urls = re.findall(url_pattern, candidate.readme_text)
            for url in found_urls:
                url_lower = url.lower()
                if any(domain in url_lower for domain in ("arxiv.org", "scholar.google", "semanticscholar.org", "dl.acm.org")):
                    safe, reason = await validate_public_url(url)
                    if safe:
                        candidate.paper_links.append(url)
                    else:
                        self._record_safety_event(
                            "blocked_external_url",
                            {"url": url, "reason": reason, "source": "readme_paper_link"},
                        )
                        logger.warning("Blocked paper URL for %s: %s (%s)", username, url, reason)
                elif "linkedin.com" not in url_lower and "github.com" not in url_lower:
                    if url not in urls_to_check:
                        safe, reason = await validate_public_url(url)
                        if safe:
                            urls_to_check.append(url)
                        else:
                            self._record_safety_event(
                                "blocked_external_url",
                                {"url": url, "reason": reason, "source": "readme_link"},
                            )
                            logger.warning("Blocked README URL for %s: %s (%s)", username, url, reason)

        # Fetch website content (redirect-safe)
        for url in urls_to_check[:2]:  # Max 2 websites
            try:
                import aiohttp
                import ssl
                import certifi
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                connector = aiohttp.TCPConnector(ssl=ssl_ctx)
                async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=10)) as session:
                    result = await fetch_text_if_safe(
                        session=session,
                        url=url,
                        max_redirects=self._MAX_REDIRECTS,
                        on_event=self._async_record_safety_event,
                    )
                    if result.status == "ok":
                        candidate.website_text = _html_to_text(result.body, gc.MAX_WEBSITE_FETCH_SIZE)
                        paper_urls = re.findall(r'https?://arxiv\.org/abs/[^\s\)\]>\"\'<]+', result.body)
                        candidate.paper_links.extend(paper_urls)
            except Exception as exc:
                self._record_safety_event(
                    "enrichment_failure",
                    {"kind": "website_fetch", "identity_key": username, "url": url},
                )
                logger.debug(
                    "enrichment failure (website_fetch) for %s url=%s: %s",
                    username,
                    url,
                    exc,
                )

        # Deduplicate paper links
        candidate.paper_links = list(dict.fromkeys(candidate.paper_links))

        # Extract paper titles from arxiv links
        for link in candidate.paper_links[:5]:  # Max 5 papers
            if "arxiv.org" in link:
                try:
                    arxiv_id = link.split("/abs/")[-1].rstrip("/").rstrip(".")
                    if not arxiv_id or not self._ARXIV_ID_RE.match(arxiv_id):
                        continue
                    import aiohttp
                    import ssl
                    import certifi
                    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
                    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=10)) as session:
                        async with session.get(
                            "https://export.arxiv.org/api/query",
                            params={"id_list": arxiv_id},
                        ) as resp:
                            if resp.status == 200:
                                xml = await resp.text()
                                # Simple title extraction from Atom XML
                                title_match = re.search(r'<title[^>]*>(.*?)</title>', xml, re.DOTALL)
                                if title_match:
                                    title = title_match.group(1).strip()
                                    if title and title != "ArXiv Query":
                                        candidate.paper_titles.append(title)
                except Exception as exc:
                    self._record_safety_event(
                        "enrichment_failure",
                        {"kind": "arxiv_fetch", "identity_key": username, "url": link},
                    )
                    logger.debug(
                        "enrichment failure (arxiv_fetch) for %s url=%s: %s",
                        username,
                        link,
                        exc,
                    )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _aggregate_languages(repos: list[GitHubRepo]) -> dict[str, int]:
        """Aggregate language distribution from repo primary languages.

        This is a rough estimate — for precise byte counts we'd need
        GET /repos/{owner}/{repo}/languages per repo, but that's expensive.
        We use the repo's primary language as a proxy.
        """
        langs: dict[str, int] = {}
        for repo in repos:
            if repo.language:
                langs[repo.language] = langs.get(repo.language, 0) + max(repo.stars, 1)
        return langs

    @staticmethod
    def _estimate_contribution_months(repos: list[GitHubRepo]) -> dict[str, int]:
        """Estimate monthly contribution frequency from repo push dates.

        Not precise (actual commits would need per-repo API calls), but
        gives a signal of activity recency and consistency.
        """
        months: dict[str, int] = {}
        for repo in repos:
            if repo.pushed_at:
                month_key = repo.pushed_at[:7]  # "2025-03"
                months[month_key] = months.get(month_key, 0) + 1
        return months

    @staticmethod
    def _format_repos(repos: list[GitHubRepo]) -> str:
        """Format repos for the synthesis prompt."""
        lines = []
        for i, r in enumerate(repos, 1):
            topics = f" [{', '.join(r.topics)}]" if r.topics else ""
            lines.append(
                f"{i}. {r.name} ({r.language or 'unknown'}, {r.stars}★, {r.forks} forks){topics}\n"
                f"   {r.description or '(no description)'}\n"
                f"   Last pushed: {r.pushed_at[:10] if r.pushed_at else 'unknown'}"
            )
        return "\n".join(lines) if lines else "(no repositories)"

    @staticmethod
    def _format_languages(languages: dict[str, int]) -> str:
        """Format language distribution for the synthesis prompt."""
        if not languages:
            return "(no language data)"
        sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)
        return ", ".join(f"{lang} ({weight})" for lang, weight in sorted_langs[:15])

    def _record_safety_event(self, event_type: str, payload: dict) -> None:
        if self._safety_event_recorder:
            self._safety_event_recorder(event_type, payload)

    async def _async_record_safety_event(self, event_type: str, payload: dict) -> None:
        self._record_safety_event(event_type, payload)
