"""Designer ↔ LinkedIn cross-module identity adapter — Slice C.8.

Per ``docs/designer-hitl-module-spec.md`` §9-10 and
``docs/cloris-cross-module-identity-resolution-spec.md`` §5.4.

The adapter encodes the matching strategy that turns a designer
candidate (Behance-profile-anchored, portfolio-URL-anchored) into a
confident link to the same person's LinkedIn candidate.

Resolution methods (confidence order — resolver applies first match):

1. **Behance URL in LinkedIn profile** — the LinkedIn candidate's
   social/website links contain the designer's Behance profile URL.
   Confidence 0.95; auto-merge.
2. **Portfolio URL match** — both sides carry the same personal
   portfolio URL (Cargo, Squarespace, personal domain). Confidence
   0.90; auto-merge.
3. **Name + company match** — designer's display name matches the
   LinkedIn candidate's name AND the designer's headline contains a
   company substring that matches the LinkedIn candidate's company.
   Confidence 0.65; auto-medium / pending_merge_decisions.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from shared.cross_module_identity.researcher_to_linkedin import IdentityMatch

# Confidence bands per ``docs/cloris-cross-module-identity-resolution-spec.md``
# §"Auto-merge thresholds." Designer↔LinkedIn uses the standard
# auto-strong (≥0.85) + auto-medium (0.65-0.85) bands.
BEHANCE_URL_MATCH_CONFIDENCE: float = 0.95
PORTFOLIO_URL_MATCH_CONFIDENCE: float = 0.90
NAME_PLUS_COMPANY_BASE_CONFIDENCE: float = 0.65
NAME_PLUS_COMPANY_MAX_CONFIDENCE: float = 0.80

_BEHANCE_USERNAME_RE = re.compile(r"behance\.net/([A-Za-z0-9_-]+)")


def normalize_portfolio_url(url: str) -> str:
    """Normalize a portfolio URL for comparison.

    Strips protocol, ``www.``, trailing slash, query params, and
    fragment. Lowercases. Returns empty string on empty/whitespace input.
    """
    if not isinstance(url, str):
        return ""
    cleaned = url.strip()
    if not cleaned:
        return ""
    parsed = urlparse(cleaned if "://" in cleaned else f"https://{cleaned}")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    if not host:
        return ""
    return f"{host}{path}"


def _extract_behance_username(url: str) -> str:
    """Extract the Behance username from a URL, or return empty string."""
    m = _BEHANCE_USERNAME_RE.search(url)
    return m.group(1).lower() if m else ""


def match_by_behance_url(
    designer_profile_url: str,
    linkedin_urls: list[str],
) -> IdentityMatch | None:
    """Match when a LinkedIn profile links to the designer's Behance.

    Extracts the Behance username from ``designer_profile_url`` and
    checks whether any URL in ``linkedin_urls`` contains the same
    ``behance.net/<username>`` pattern.
    """
    designer_username = _extract_behance_username(designer_profile_url)
    if not designer_username:
        return None
    for url in linkedin_urls:
        linkedin_username = _extract_behance_username(url)
        if linkedin_username and linkedin_username == designer_username:
            return IdentityMatch(
                confidence=BEHANCE_URL_MATCH_CONFIDENCE,
                method="behance_url_in_linkedin",
                evidence=(
                    f"Behance username: {designer_username}",
                    f"LinkedIn URL: {url}",
                ),
            )
    return None


def match_by_portfolio_url(
    designer_urls: list[str],
    linkedin_urls: list[str],
) -> IdentityMatch | None:
    """Match when both sides share the same personal portfolio URL.

    Normalizes all URLs before comparison (strips protocol, www,
    trailing slash, query params). Returns the first overlap found.
    """
    if not designer_urls or not linkedin_urls:
        return None
    designer_normalized = {normalize_portfolio_url(u) for u in designer_urls if u}
    designer_normalized.discard("")
    if not designer_normalized:
        return None
    for url in linkedin_urls:
        normalized = normalize_portfolio_url(url)
        if normalized and normalized in designer_normalized:
            return IdentityMatch(
                confidence=PORTFOLIO_URL_MATCH_CONFIDENCE,
                method="portfolio_url_match",
                evidence=(f"Shared portfolio URL: {normalized}",),
            )
    return None


def _normalize_company_substring(value: str) -> str:
    """Lowercase + strip common company-name suffixes for substring matching."""
    if not isinstance(value, str):
        return ""
    out = value.strip().lower()
    for suffix in (
        ", inc.",
        ", inc",
        " inc.",
        " inc",
        ", llc",
        " llc",
        ", ltd.",
        ", ltd",
        " ltd.",
        " ltd",
        ", l.l.c.",
    ):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
            break
    return out


def match_by_name_and_company(
    *,
    designer_name: str,
    designer_headline: str,
    linkedin_name: str,
    linkedin_company: str,
) -> IdentityMatch | None:
    """Match when name + company align.

    Both names should be pre-normalized via ``normalize_person_name``.
    Headline↔company matching is substring-based after suffix
    stripping: a designer headline "Senior Designer at Linear" matches
    a LinkedIn company "Linear" because "linear" is contained in the
    normalized headline.

    Returns ``None`` when names don't match, when either company
    signal is empty, or when the substring check fails.
    """
    if not designer_name or not linkedin_name:
        return None
    if designer_name != linkedin_name:
        return None
    headline_norm = _normalize_company_substring(designer_headline)
    company_norm = _normalize_company_substring(linkedin_company)
    if not headline_norm or not company_norm:
        return None
    if company_norm not in headline_norm and headline_norm not in company_norm:
        return None
    return IdentityMatch(
        confidence=NAME_PLUS_COMPANY_BASE_CONFIDENCE,
        method="name_plus_company",
        evidence=(
            f"Name: {designer_name}",
            f"Company overlap: {designer_headline} ↔ {linkedin_company}",
        ),
    )


def best_match(
    *,
    designer_profile_url: str,
    designer_social_urls: list[str],
    designer_name: str,
    designer_headline: str,
    linkedin_urls: list[str],
    linkedin_name: str,
    linkedin_company: str,
) -> IdentityMatch | None:
    """Try all matchers in confidence order, return the first hit."""
    m = match_by_behance_url(designer_profile_url, linkedin_urls)
    if m is not None:
        return m
    m = match_by_portfolio_url(designer_social_urls, linkedin_urls)
    if m is not None:
        return m
    return match_by_name_and_company(
        designer_name=designer_name,
        designer_headline=designer_headline,
        linkedin_name=linkedin_name,
        linkedin_company=linkedin_company,
    )


__all__ = [
    "IdentityMatch",
    "BEHANCE_URL_MATCH_CONFIDENCE",
    "PORTFOLIO_URL_MATCH_CONFIDENCE",
    "NAME_PLUS_COMPANY_BASE_CONFIDENCE",
    "NAME_PLUS_COMPANY_MAX_CONFIDENCE",
    "normalize_portfolio_url",
    "match_by_behance_url",
    "match_by_portfolio_url",
    "match_by_name_and_company",
    "best_match",
]
