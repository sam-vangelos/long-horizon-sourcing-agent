"""Deterministic identity and novelty scoring helpers."""

from __future__ import annotations

import re
from collections import Counter

from shared.reconciliation_schemas import (
    LinkedInIdentityHints,
    LinkedInMatchResult,
    RecruiterActivitySnapshot,
)

_CREDENTIAL_SUFFIX_RE = re.compile(
    r"\b(phd|ph\.d|mba|m\.s|ms|m\.sc|msc|dr|cfa|pmp|pe|frsa)\b\.?",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|llc|ltd|corp|corporation|holdings|group|co|company|technologies|technology)\b\.?",
    re.IGNORECASE,
)
_STOPWORDS = {
    "and",
    "of",
    "the",
    "in",
    "at",
    "for",
    "to",
    "new",
    "york",
    "city",
    "area",
    "greater",
    "metropolitan",
    "united",
    "states",
}
_LOCATION_ALIAS_MAP = {
    "nyc": "New York City Metropolitan Area",
    "new york city": "New York City Metropolitan Area",
    "new york city metropolitan area": "New York City Metropolitan Area",
    "greater new york city area": "New York City Metropolitan Area",
    "sf": "San Francisco Bay Area",
    "san francisco": "San Francisco Bay Area",
    "san francisco bay area": "San Francisco Bay Area",
    "bay area": "San Francisco Bay Area",
    "seattle metro": "Seattle Metropolitan Area",
    "greater seattle area": "Seattle Metropolitan Area",
}
_AUTHOR_HINT_RE = re.compile(r"@author\s+([^)]+)", re.IGNORECASE)
_CAMELCASE_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_USERNAME_STRIP_URL_RE = re.compile(r"^https?://(?:www\.)?github\.com/", re.IGNORECASE)
_USERNAME_TOKEN_SPLIT_RE = re.compile(r"[_\-\s]+|\d+")
_ALPHA_ONLY_RE = re.compile(r"^[A-Za-z]+$")


def _normalize_text(value: str) -> str:
    value = str(value or "").lower().strip()
    value = _CREDENTIAL_SUFFIX_RE.sub(" ", value)
    value = _PUNCT_RE.sub(" ", value)
    value = " ".join(value.split())
    return value


def _tokenize(value: str) -> list[str]:
    normalized = _normalize_text(value)
    return [token for token in normalized.split() if token and token not in _STOPWORDS]


def normalize_person_name(value: str) -> str:
    return _normalize_text(value)


def normalize_company_name(value: str) -> str:
    value = _COMPANY_SUFFIX_RE.sub(" ", value or "")
    return _normalize_text(value)


def normalize_location_text(value: str) -> str:
    return _normalize_text(value)


def canonicalize_location_label(value: str) -> str:
    normalized = normalize_location_text(value)
    if not normalized:
        return ""
    if "manhattan" in normalized and "new york" in normalized:
        return "New York City Metropolitan Area"
    if "brooklyn" in normalized and "new york" in normalized:
        return "New York City Metropolitan Area"
    if "queens" in normalized and "new york" in normalized:
        return "New York City Metropolitan Area"
    if "new york" in normalized:
        return "New York City Metropolitan Area"
    if "san francisco" in normalized or "bay area" in normalized:
        return "San Francisco Bay Area"
    if "seattle" in normalized:
        return "Seattle Metropolitan Area"
    if normalized in _LOCATION_ALIAS_MAP:
        return _LOCATION_ALIAS_MAP[normalized]
    for alias, canonical in _LOCATION_ALIAS_MAP.items():
        if alias in normalized:
            return canonical
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return ""
    trimmed = tokens[:4]
    return " ".join(token.capitalize() for token in trimmed)


def normalize_public_linkedin_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    normalized = raw.split("?", 1)[0].rstrip("/")
    if "linkedin.com/in/" in normalized:
        return normalized.lower()
    return normalized


# Phase F Slice F3: cross-module identity needs the public-handle slug
# (the lowercased path segment) so candidates discovered via LinkedIn and
# candidates discovered via GitHub-with-LinkedIn-hint can match on the
# same canonical handle. Mirrors the regex shape in
# `linkedin/recruiter_identity_resolver.py:88-91`; ported here so
# `shared.*` modules don't import from the LinkedIn package (cycle-prone
# and violates the module-direction rule).
_PUBLIC_LINKEDIN_SLUG_RE = re.compile(
    r"(?:https?://)?(?:[a-z]+\.)?linkedin\.com/in/([A-Za-z0-9._\-%]+)|^/in/([A-Za-z0-9._\-%]+)",
    re.IGNORECASE,
)


def normalize_public_linkedin_handle(value: str) -> str:
    """Return the lowercase slug for a public LinkedIn URL, or empty.

    Recruiter URLs (``/talent/profile/...``) carry no public handle and
    return empty so they can't accidentally collide across humans.
    """

    if not value:
        return ""
    match = _PUBLIC_LINKEDIN_SLUG_RE.search(str(value))
    if not match:
        return ""
    slug = match.group(1) or match.group(2) or ""
    slug = slug.strip().lower()
    return slug.rstrip("/").split("?", 1)[0].split("#", 1)[0]


def build_person_lookup_name(candidate_name: str, github_username: str = "") -> str:
    raw = str(candidate_name or "").strip()
    if not raw:
        raw = str(github_username or "").strip()
    if not raw:
        return ""
    author_match = _AUTHOR_HINT_RE.search(raw)
    if author_match:
        raw = author_match.group(1).strip()
    raw = _CAMELCASE_BOUNDARY_RE.sub(" ", raw)
    raw = raw.replace("_", " ").replace("-", " ")
    raw = raw.replace("(", " ").replace(")", " ")
    raw = " ".join(raw.split())
    tokens = raw.split()
    if tokens and all(token.isupper() for token in tokens):
        raw = " ".join(token.capitalize() for token in tokens)
        tokens = raw.split()
    lookup = raw.strip()

    # P0 fallback: when the scraped candidate_name is a single token (e.g. "Michael"),
    # a plain Recruiter name search devolves into "every Michael in NYC". Derive a
    # safe second token from the GitHub username when possible. Strict rule: never
    # invent characters not present in the source -- no apostrophes, no speculative
    # initials-splitting.
    if len(tokens) == 1:
        extra = _extra_tokens_from_username(github_username, tokens[0])
        if extra:
            lookup = " ".join([tokens[0], *extra])

    return lookup


def _extra_tokens_from_username(
    github_username: str,
    candidate_single_token: str,
) -> list[str]:
    """Derive a list of safe title-cased surname-candidate tokens from a GitHub username.

    Never invents punctuation or characters. Returns at most 2 tokens ordered by length
    descending. Returns an empty list when no safe usable token is available.
    """
    raw = str(github_username or "").strip()
    if not raw:
        return []
    raw = _USERNAME_STRIP_URL_RE.sub("", raw).strip().strip("/")
    if not raw:
        return []
    candidate_lower = candidate_single_token.lower()

    # Prefer explicit-boundary splits first: underscores, hyphens, whitespace, digits.
    boundary_parts = [p for p in _USERNAME_TOKEN_SPLIT_RE.split(raw) if p]
    boundary_tokens: list[str] = []
    for part in boundary_parts:
        # Further split by camelcase inside each part, so "mldAngelo" -> ["mld", "Angelo"].
        for piece in _CAMELCASE_BOUNDARY_RE.split(part):
            if _is_safe_username_token(piece, candidate_lower):
                boundary_tokens.append(piece.lower())

    if boundary_tokens:
        unique = _ordered_unique(boundary_tokens)
        unique.sort(key=len, reverse=True)
        return [_title_case_token(token) for token in unique[:2]]

    # No usable boundary-derived token. Fall back to appending the full username as a
    # single title-cased token iff it is purely alphabetic, >= 2 chars, and not equal
    # to the candidate's single token case-insensitively.
    if _is_safe_username_token(raw, candidate_lower):
        return [_title_case_token(raw.lower())]

    return []


def _is_safe_username_token(token: str, candidate_lower: str) -> bool:
    if not token or len(token) < 2:
        return False
    if not _ALPHA_ONLY_RE.match(token):
        return False
    if token.lower() == candidate_lower:
        return False
    return True


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _title_case_token(token: str) -> str:
    return token[:1].upper() + token[1:].lower() if token else token


def build_candidate_lookup_queries(hints: LinkedInIdentityHints) -> list[str]:
    """Build bounded LinkedIn Recruiter keyword queries for a GitHub lead."""
    name = hints.candidate_name.strip()
    company = normalize_company_name(hints.company)
    location_tokens = _tokenize(hints.location)
    title_tokens = _tokenize(hints.title)
    queries: list[str] = []

    if not name:
        return queries

    quoted_name = f'"{name}"'
    company_terms = sorted({f'"{token}"' for token in company.split() if token})
    location_terms = sorted({f'"{token}"' for token in location_tokens[:3]})
    title_terms = sorted({f'"{token}"' for token in title_tokens[:4]})

    if company:
        company_query = f"{quoted_name} AND ({' OR '.join(company_terms)})"
        queries.append(company_query)

    if location_tokens:
        location_query = f"{quoted_name} AND ({' OR '.join(location_terms)})"
        queries.append(location_query)

    if company and location_tokens:
        combined = (
            f"{quoted_name} AND "
            f"({' OR '.join(company_terms)}) AND "
            f"({' OR '.join(location_terms)})"
        )
        queries.insert(0, combined)

    if title_tokens:
        title_query = f"{quoted_name} AND ({' OR '.join(title_terms)})"
        queries.append(title_query)

    queries.append(quoted_name)

    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        query = " ".join(query.split())
        if query and query not in seen:
            seen.add(query)
            deduped.append(query)
    return deduped[:5]


def resolve_direct_linkedin_hint(
    hints: LinkedInIdentityHints,
) -> LinkedInMatchResult | None:
    """Return a direct-hint match when GitHub already points to LinkedIn."""
    url = (hints.linkedin_url_hint or "").strip()
    if not url:
        return None
    ambiguity_reasons: list[str] = []
    if "/talent/profile/" not in url:
        ambiguity_reasons.append("Recruiter activity unavailable until a Recruiter profile is resolved")
    return LinkedInMatchResult(
        matched_profile_url=url,
        matched_name=hints.candidate_name,
        matched_company=hints.company,
        matched_title=hints.title,
        matched_location=hints.location,
        match_confidence=0.96,
        match_method="direct_linkedin_hint",
        evidence=["Direct LinkedIn URL hint present"],
        ambiguity_reasons=ambiguity_reasons,
        recruiter_activity=None,
        novelty_pressure="",
    )


def classify_recruiter_activity_pressure(activity: RecruiterActivitySnapshot | None) -> str:
    """Convert recruiter activity into a conservative novelty pressure label."""
    if not activity:
        return "low"
    if activity.message_count >= 6 or (activity.project_count >= 3 and activity.view_count >= 3):
        return "high"
    if activity.message_count >= 3 or activity.project_count >= 2 or activity.view_count >= 4:
        return "medium"
    return "low"


def infer_reachout_status(activity: RecruiterActivitySnapshot | None) -> str:
    if not activity:
        return ""
    if activity.last_outbound_contact:
        return "recent_outbound_contact"
    if activity.message_count > 0:
        return "messaged"
    if activity.project_count > 0:
        return "in_projects"
    return "unworked"


def score_linkedin_identity_match(
    hints: LinkedInIdentityHints,
    *,
    matched_name: str,
    matched_company: str = "",
    matched_title: str = "",
    matched_location: str = "",
    matched_profile_url: str = "",
    recruiter_activity: RecruiterActivitySnapshot | None = None,
    match_method: str = "search",
) -> LinkedInMatchResult:
    """Score a likely LinkedIn identity match from normalized hints."""
    evidence: list[str] = []
    ambiguity_reasons: list[str] = []
    score = 0.0

    expected_name = normalize_person_name(hints.candidate_name)
    actual_name = normalize_person_name(matched_name)
    if expected_name and actual_name:
        if expected_name == actual_name:
            score += 0.55
            evidence.append("Exact name match")
        else:
            expected_tokens = set(_tokenize(expected_name))
            actual_tokens = set(_tokenize(actual_name))
            overlap = len(expected_tokens & actual_tokens)
            if overlap >= max(1, min(len(expected_tokens), len(actual_tokens)) - 1):
                score += 0.35
                evidence.append("Strong partial name overlap")
            else:
                ambiguity_reasons.append("Name mismatch")

    expected_company_tokens = set(_tokenize(normalize_company_name(hints.company)))
    actual_company_tokens = set(_tokenize(normalize_company_name(matched_company)))
    if expected_company_tokens and actual_company_tokens:
        overlap = len(expected_company_tokens & actual_company_tokens)
        if overlap:
            score += min(0.2, 0.08 * overlap)
            evidence.append("Company overlap")
        else:
            ambiguity_reasons.append("Company mismatch")

    expected_location_tokens = set(_tokenize(normalize_location_text(hints.location)))
    actual_location_tokens = set(_tokenize(normalize_location_text(matched_location)))
    if expected_location_tokens and actual_location_tokens:
        overlap = len(expected_location_tokens & actual_location_tokens)
        if overlap:
            score += min(0.1, 0.04 * overlap)
            evidence.append("Location overlap")
        else:
            ambiguity_reasons.append("Location mismatch")

    expected_title = _normalize_text(hints.title)
    actual_title = _normalize_text(matched_title)
    title_overlap = len(set(_tokenize(hints.title)) & set(_tokenize(matched_title)))
    if expected_title and actual_title and expected_title == actual_title:
        score += 0.15
        evidence.append("Exact title match")
    elif title_overlap:
        score += min(0.1, 0.03 * title_overlap)
        evidence.append("Title overlap")

    if recruiter_activity and recruiter_activity.saved_by:
        evidence.append(f"Recruiter activity visible (saved by {recruiter_activity.saved_by})")

    score = min(score, 1.0)
    novelty_pressure = classify_recruiter_activity_pressure(recruiter_activity)
    return LinkedInMatchResult(
        matched_profile_url=matched_profile_url,
        matched_name=matched_name,
        matched_company=matched_company,
        matched_title=matched_title,
        matched_location=matched_location,
        match_confidence=round(score, 3),
        match_method=match_method,
        evidence=evidence,
        ambiguity_reasons=ambiguity_reasons,
        recruiter_activity=recruiter_activity,
        novelty_pressure=novelty_pressure,
    )


def choose_best_match(matches: list[LinkedInMatchResult]) -> tuple[str, LinkedInMatchResult | None]:
    """Classify a set of match candidates into high/manual/none buckets."""
    if not matches:
        return "no_confident_match", None
    ranked = sorted(matches, key=lambda item: item.match_confidence, reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    if best.match_confidence >= 0.85 and (second is None or best.match_confidence - second.match_confidence >= 0.15):
        return "high_confidence_match", best
    if best.match_confidence >= 0.6:
        return "manual_review", best
    return "no_confident_match", None


def summarize_email_domains(emails: list[str]) -> list[str]:
    domains = [email.split("@", 1)[1].lower() for email in emails if "@" in email]
    counts = Counter(domains)
    return [domain for domain, _count in counts.most_common(3)]
