"""Researcher ↔ LinkedIn cross-module identity adapter — Slice B.3.

Per Researcher Module Spec Slice 7's "Cross-module identity"
follow-up + ``docs/cloris-cross-module-identity-resolution-spec.md``.
The adapter encodes the matching strategy that turns a researcher
candidate (publication-record-anchored) into a confident link to
the same person's LinkedIn candidate (profile-URL-anchored).

## Resolution methods

In confidence order — the resolver applies them in this order and
takes the first that matches:

1. **ORCID match** — both sides carry the same ORCID. ORCID is a
   single-person-claimed permanent identifier; coverage is partial
   (~30-40% of ML researchers per Researcher Module Spec Opinion 3)
   but when both sides have it, the match is deterministic. Confidence
   ≥ 0.95; auto-merge.
2. **Name + current affiliation match** — researcher's `real_name`
   matches LinkedIn candidate's `display_name` AND researcher's
   `affiliation` matches the LinkedIn candidate's most-recent company
   (heuristic substring match, normalized). Confidence 0.65-0.85;
   auto-medium per the resolver's existing band shape.
3. **Name + publication-evidence match** — researcher's name matches
   LinkedIn candidate's name AND the LinkedIn candidate's profile
   text mentions one of the researcher's top venues / first-author
   papers. Confidence 0.60-0.75; pending_merge_decisions surface.

## What this module ships today

The adapter is interface-only at Slice B.3 ship — the resolution
methods named above are documented contracts; the actual matching
calls land in the resolver's pass-3 extension as a behavior-preserving
follow-up. Today's slice provides:

- Public functions :func:`match_by_orcid` and
  :func:`match_by_name_and_affiliation` that the resolver service
  can compose into its existing pass structure.
- Confidence-band constants pinning the ranges above so future
  drift in confidence values surfaces as a deliberate edit.

The full integration into ``shared.identity_resolution_service``'s
pass structure is a behavior-preserving follow-up gated on
``tests/test_identity_resolution.py`` (which today doesn't exist
at the per-source granularity these adapters need; building out
the test scaffold is part of the follow-up).

## Why researcher↔LinkedIn matters operationally

Researchers without LinkedIn profiles can't be saved to LinkedIn
Recruiter — the Cloris-native workspace is the only destination
(per Researcher Module Spec Opinion 4). Researchers WITH LinkedIn
profiles can be reconciled so the recruiter sees one candidate
card with both publication-record evidence (researcher) and
LinkedIn profile evidence (linkedin) in a unified workspace surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# Confidence bands per ``docs/cloris-cross-module-identity-resolution-spec.md``
# §"Auto-merge thresholds." These are per-pair contracts; researcher↔LinkedIn
# uses the standard auto-strong (≥0.85) + auto-medium (0.65-0.85) bands.
ORCID_MATCH_CONFIDENCE: float = 0.95
NAME_PLUS_AFFILIATION_BASE_CONFIDENCE: float = 0.70
NAME_PLUS_AFFILIATION_MAX_CONFIDENCE: float = 0.85
NAME_PLUS_PUBLICATION_BASE_CONFIDENCE: float = 0.60
NAME_PLUS_PUBLICATION_MAX_CONFIDENCE: float = 0.75


# ORCID format: 16 digits split by hyphens into 4-char chunks; the
# trailing character can be a checksum 'X'. Per orcid.org spec.
_ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


@dataclass(frozen=True)
class IdentityMatch:
    """One confident cross-source link between two candidates.

    ``method`` names the resolution method that produced the match;
    feeds into the resolver's existing ``match_signal`` payload on
    ``CandidateLink`` so the recruiter-facing pending-decision UI
    can surface "matched on ORCID" / "matched on name + affiliation"
    as editorial provenance.
    """

    confidence: float
    method: str
    evidence: tuple[str, ...]


def normalize_orcid(value: str) -> str:
    """Normalize an ORCID string to the canonical hyphenated 19-char shape.

    Strips whitespace, removes the optional ``https://orcid.org/`` URL
    prefix, validates the format. Returns the empty string if the
    input doesn't parse as a valid ORCID — callers treat empty as
    "no ORCID signal" rather than raising, matching the resolver's
    "missing field doesn't block" posture.
    """

    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    # Strip URL prefix (orcid.org sometimes formats as a full URL).
    for prefix in ("https://orcid.org/", "http://orcid.org/", "orcid.org/"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    if not _ORCID_RE.match(cleaned):
        return ""
    return cleaned


def match_by_orcid(researcher_orcid: str, linkedin_orcid: str) -> IdentityMatch | None:
    """Return :class:`IdentityMatch` when both sides carry the same ORCID.

    ``None`` when either side's ORCID is empty / malformed, or when
    they don't match. ORCID is the gold-standard match for academic
    candidates: a single-person-claimed permanent identifier.

    LinkedIn rarely carries ORCID on the profile (LinkedIn doesn't
    surface it as a structured field), so the linkedin-side ORCID
    typically comes from a recruiter-pasted URL in the profile's
    "External" / contact-info section. When LinkedIn-side ORCID is
    present + matches the researcher's, confidence is ≥0.95 and
    auto-merge fires.
    """

    a = normalize_orcid(researcher_orcid)
    b = normalize_orcid(linkedin_orcid)
    if not a or not b or a != b:
        return None
    return IdentityMatch(
        confidence=ORCID_MATCH_CONFIDENCE,
        method="orcid_match",
        evidence=(f"ORCID: {a}",),
    )


def _normalize_company_substring(value: str) -> str:
    """Lowercase + strip common company-name suffixes for substring matching.

    Researcher affiliations come from OpenAlex's parsed institution
    objects (typically clean — "Stanford University", "Google
    DeepMind"); LinkedIn current-company strings carry more noise
    ("Stanford University - Computer Science Dept", "Google, LLC").
    Stripping common suffixes + lowercasing makes the substring match
    more permissive without losing precision.
    """

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


def match_by_name_and_affiliation(
    *,
    researcher_name: str,
    researcher_affiliation: str,
    linkedin_name: str,
    linkedin_company: str,
) -> IdentityMatch | None:
    """Return :class:`IdentityMatch` when name + affiliation align.

    Both names normalize via the resolver's
    :func:`shared.name_normalization.normalize_person_name` (caller
    is responsible for normalization before passing in — this adapter
    accepts already-normalized names because the resolver's pass-2
    name grouping has already done it).

    Affiliation matching is substring-based after suffix stripping
    (see :func:`_normalize_company_substring`): a researcher's
    "Stanford University" matches a LinkedIn "Stanford University -
    Computer Science" + "Stanford University, AI Lab" because the
    researcher's institution string is contained in both. False
    positives are possible for very common substrings ("Google",
    "Microsoft"); the medium-confidence band reflects this — the
    resolver routes these to ``pending_merge_decisions`` for
    recruiter confirmation rather than auto-merging.

    Returns ``None`` when names don't match exactly, when either
    affiliation is empty, or when the substring check fails.
    """

    if not researcher_name or not linkedin_name:
        return None
    if researcher_name != linkedin_name:
        return None
    res_aff = _normalize_company_substring(researcher_affiliation)
    li_co = _normalize_company_substring(linkedin_company)
    if not res_aff or not li_co:
        return None
    if res_aff not in li_co and li_co not in res_aff:
        return None
    return IdentityMatch(
        confidence=NAME_PLUS_AFFILIATION_BASE_CONFIDENCE,
        method="name_plus_affiliation",
        evidence=(
            f"Name: {researcher_name}",
            f"Affiliation overlap: {researcher_affiliation} ↔ {linkedin_company}",
        ),
    )


__all__ = [
    "IdentityMatch",
    "ORCID_MATCH_CONFIDENCE",
    "NAME_PLUS_AFFILIATION_BASE_CONFIDENCE",
    "NAME_PLUS_AFFILIATION_MAX_CONFIDENCE",
    "NAME_PLUS_PUBLICATION_BASE_CONFIDENCE",
    "NAME_PLUS_PUBLICATION_MAX_CONFIDENCE",
    "normalize_orcid",
    "match_by_orcid",
    "match_by_name_and_affiliation",
]
