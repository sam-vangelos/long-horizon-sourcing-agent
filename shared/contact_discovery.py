"""Contact information discovery from GitHub profiles.

Extracts emails from:
    1. User profile (public email field)
    2. Commit history (git author email)
    3. Social links (Twitter, LinkedIn, blog)

Filters out noreply@github.com and other bot/service addresses.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from shared.schemas import ContactInfo

if TYPE_CHECKING:
    from github.client import GitHubClient
    from github.schemas import GitHubRepo

log = logging.getLogger(__name__)


# Addresses to filter out
_NOREPLY_PATTERNS = [
    "noreply@github.com",
    "users.noreply.github.com",
    "noreply@",
    "no-reply@",
    "github-actions",
    "dependabot",
    "greenkeeper",
    "renovate",
]


def _is_real_email(email: str) -> bool:
    """Check if an email looks like a real person's address."""
    if not email or "@" not in email:
        return False
    email_lower = email.lower().strip()
    for pattern in _NOREPLY_PATTERNS:
        if pattern in email_lower:
            return False
    # Basic format check
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email_lower):
        return False
    return True


async def discover_contacts(
    client: "GitHubClient",
    username: str,
    top_repos: list[GitHubRepo],
) -> ContactInfo:
    """Discover contact information for a GitHub user.

    Checks profile email, commit history, and social links.
    Returns ContactInfo with deduplicated, validated addresses.
    """
    emails: set[str] = set()
    contact = ContactInfo()

    # 1. Profile email (from get_user, already fetched)
    # The caller should have the user data; we check commit emails here

    # 2. Commit emails from their repos
    for repo in top_repos[:3]:  # Limit API calls
        if repo.is_fork:
            continue
        try:
            commits = await client.get_user_commits(
                repo.full_name or f"{username}/{repo.name}",
                author=username,
            )
            for commit in commits:
                commit_data = commit.get("commit", {})
                author_data = commit_data.get("author", {})
                email = author_data.get("email", "")
                if _is_real_email(email):
                    emails.add(email.lower().strip())
        except Exception:
            # Not fatal — commit fetch can fail for many reasons
            continue

    contact.emails = sorted(emails)

    # 3. Social links are extracted from the user profile by the enricher
    # (twitter_username, blog fields) — we just format them here if present

    return contact


# OSS Maintainers Slice 8: regex used to scan bio + profile README
# for embedded LinkedIn URLs. Requires the full ``linkedin.com/in/<handle>``
# shape — bare keywords like "ex-LinkedIn" don't match (per spec §12
# false-positive mitigation). Captures either ``in/`` (personal
# profiles) or ``company/`` (company pages); the resolver only acts
# on personal profiles, but the parser captures both so we can log
# discoveries we choose not to use.
_LINKEDIN_URL_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/(?:in|pub|company)/[A-Za-z0-9_\-./%]+",
    re.IGNORECASE,
)


def _extract_linkedin_url(text: str) -> str:
    """Return the first ``linkedin.com/in/<handle>`` URL in ``text``, else ''."""

    if not text:
        return ""
    match = _LINKEDIN_URL_RE.search(text)
    if match is None:
        return ""
    url = match.group(0)
    # Trim trailing punctuation that often clings to URLs in prose.
    return url.rstrip(".,;:)]}>\"'")


# P3.8: a `linkedin.com/company/...` URL is a company page, not a person —
# it must never be assigned to `contact.linkedin_url` (the field the
# cross-source identity resolver treats as a personal-profile join key).
# The regex above deliberately still captures company URLs (so callers can
# log discoveries they choose not to use); this helper is the filter at the
# assignment sites, not at the parser.
def _is_company_page_url(url: str) -> bool:
    return "linkedin.com/company/" in url.lower()


def merge_profile_contact(
    contact: ContactInfo,
    user_email: str,
    twitter: str,
    blog: str,
    *,
    bio: str = "",
    readme_text: str = "",
) -> ContactInfo:
    """Merge additional contact info from the user profile into ContactInfo.

    OSS Maintainers Slice 8: the optional ``bio`` and ``readme_text``
    arguments enable LinkedIn URL discovery beyond the historical
    ``blog`` field. Provenance is recorded on
    ``contact.linkedin_url_source`` so downstream consumers (the
    cross-source identity resolver in
    :mod:`shared.identity_resolution_service`, the recruiter
    workspace) can band confidence: ``"blog"`` is highest, ``"bio"``
    and ``"readme"`` are medium. Per spec §12, the bio/readme
    extractors require a full URL match (regex
    :data:`_LINKEDIN_URL_RE`), so passing references like "ex-
    LinkedIn" do not falsely trigger a cross-link.

    Resolution order: ``blog`` wins if present (deliberate recruiter
    placement). ``bio`` is checked next; ``readme_text`` is the
    final fallback. The resolver never re-asserts a higher-
    confidence source if a lower one was already populated — i.e.,
    if a previous call set ``linkedin_url`` from ``"readme"``, this
    call's ``"blog"`` URL wins.
    """

    # Add profile email
    if _is_real_email(user_email):
        email_lower = user_email.lower().strip()
        if email_lower not in contact.emails:
            contact.emails = [email_lower] + contact.emails  # Profile email first

    # Twitter
    if twitter:
        contact.twitter_url = f"https://twitter.com/{twitter}"

    # Blog/website (spec: blog-derived URLs are highest confidence)
    if blog:
        url = blog if blog.startswith("http") else f"https://{blog}"
        contact.website = url
        # Check if blog is a LinkedIn URL
        if "linkedin.com" in blog.lower():
            if _is_company_page_url(url):
                log.debug(
                    "contact_discovery: blog field is a LinkedIn company page, "
                    "not assigning as personal linkedin_url: %s",
                    url,
                )
            else:
                contact.linkedin_url = url
                contact.linkedin_url_source = "blog"

    # Bio scan (spec: medium confidence — bio prose may include
    # full LinkedIn URLs the recruiter would want surfaced).
    if not contact.linkedin_url:
        bio_url = _extract_linkedin_url(bio)
        if bio_url:
            if _is_company_page_url(bio_url):
                log.debug(
                    "contact_discovery: bio LinkedIn URL is a company page, "
                    "not assigning as personal linkedin_url: %s",
                    bio_url,
                )
            else:
                contact.linkedin_url = bio_url
                contact.linkedin_url_source = "bio"

    # Profile README scan (spec: same as bio — full URL match
    # required, lower than blog).
    if not contact.linkedin_url:
        readme_url = _extract_linkedin_url(readme_text)
        if readme_url:
            if _is_company_page_url(readme_url):
                log.debug(
                    "contact_discovery: README LinkedIn URL is a company page, "
                    "not assigning as personal linkedin_url: %s",
                    readme_url,
                )
            else:
                contact.linkedin_url = readme_url
                contact.linkedin_url_source = "readme"

    return contact
