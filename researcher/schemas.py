"""Researcher dataclasses — Slice 4.

Two shapes:

- :class:`ResearcherCandidate` — the full record we evaluate (per
  Researcher Module Spec Slice 4 fields).
- :class:`ResearcherSnippet` — the compact view passed to the facial
  evaluator (analogous to LinkedIn's
  :class:`shared.schemas.CandidateSnippet`).

Identity helpers live alongside: :func:`identity_key_for_candidate`
implements the ORCID-when-present + ``openalex:{author_id}`` composite
fallback per Spec Opinion 3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ResearcherPaper:
    """One published work — minimal fields the evaluator cites."""

    title: str
    venue: str = ""
    year: int = 0
    citation_count: int = 0
    is_first_author: bool = False
    openalex_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearcherCandidate:
    """Full researcher record per Spec Slice 4 contract."""

    author_id: str  # OpenAlex author id (e.g., "A1234567890")
    orcid: str  # ORCID URL or bare id; empty if unknown
    name: str
    affiliations: list[str] = field(default_factory=list)
    top_papers: list[ResearcherPaper] = field(default_factory=list)
    h_index: int = 0
    citation_count: int = 0
    works_count: int = 0
    # Number of papers in the recruiter-relevant time window (e.g., 36
    # months for the universal minimum, or 24 for an NLP discipline
    # default). Computed at acquisition time so the deterministic gate
    # at Slice 5 can read a single int.
    papers_in_window: int = 0
    # Source-of-truth URL for the recruiter (ORCID URL when present,
    # OpenAlex author URL otherwise).
    profile_url: str = ""
    # Free-form raw OpenAlex payload kept for diagnostic + downstream
    # cross-validation paths (Semantic Scholar h-index reconciliation,
    # etc.). Not required reading at evaluation time.
    raw_openalex: dict = field(default_factory=dict)
    # arXiv preprint categories the candidate has authored / co-authored
    # in the recent window (audit Move #4 R1). Populated by
    # :func:`researcher.acquisition.enrich_with_arxiv` when an
    # ArxivClient is wired into the pipeline. Empty when the
    # OpenAlex-only path runs (pre-R1 behavior). Distinct from
    # ``raw_openalex.x_concepts`` which covers PUBLISHED works only —
    # arxiv_categories surface preprints OpenAlex hasn't indexed yet,
    # which matters for early-career researchers and for "who's
    # working on this right now" reads.
    arxiv_categories: list[str] = field(default_factory=list)
    # Semantic Scholar's h-index for this author when the
    # reconciliation path fired (audit Move #4 R1). Zero when S2 wasn't
    # consulted OR the lookup failed. The orchestrator's facial /
    # full judges read ``h_index`` (the canonical value) which gets
    # bumped to S2's number when S2's is materially higher than
    # OpenAlex's; this column carries the raw S2 value for diagnostic
    # provenance.
    s2_h_index: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "ResearcherCandidate":
        papers = [
            ResearcherPaper(**p) if isinstance(p, dict) else p
            for p in d.get("top_papers") or []
        ]
        return cls(
            author_id=d.get("author_id", ""),
            orcid=d.get("orcid", ""),
            name=d.get("name", ""),
            affiliations=list(d.get("affiliations") or []),
            top_papers=papers,
            h_index=int(d.get("h_index", 0)),
            citation_count=int(d.get("citation_count", 0)),
            works_count=int(d.get("works_count", 0)),
            papers_in_window=int(d.get("papers_in_window", 0)),
            profile_url=d.get("profile_url", ""),
            raw_openalex=d.get("raw_openalex") or {},
            arxiv_categories=list(d.get("arxiv_categories") or []),
            s2_h_index=int(d.get("s2_h_index", 0)),
        )


@dataclass
class ResearcherSnippet:
    """Compact facial-input view for the researcher evaluator.

    Mirrors :class:`shared.schemas.CandidateSnippet` shape so the
    runtime-state primitives accept it; carries the fields the facial
    evaluator at Slice 5 prompts on (name, current affiliation, top
    5 papers, h-index, arXiv categories).
    """

    name: str
    current_affiliation: str
    h_index: int
    citation_count: int
    papers_in_window: int
    top_paper_titles: list[str] = field(default_factory=list)
    arxiv_categories: list[str] = field(default_factory=list)
    profile_url: str = ""
    # Identity of the parent query that surfaced this snippet — keeps
    # the facial-eval batch correctly bucketed under its work_unit.
    source_query_id: int = 0
    source_query_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Identity key resolution — Spec Opinion 3
# ---------------------------------------------------------------------------


def identity_key_for_candidate(candidate: ResearcherCandidate | dict) -> str:
    """Resolve the researcher's stable identity key.

    Per Researcher Module Spec Opinion 3:
    - ORCID-anchored when present: ``orcid:{normalized_orcid}``
    - Otherwise composite fallback: ``openalex:{author_id}``

    The :class:`shared.runtime_state.store.RuntimeStateStore` candidate
    table's ``UNIQUE(brief_id, source, identity_key)`` constraint
    accepts any string, so this key is the canonical dedup primitive
    across runs of the same brief.

    Returns empty string if the candidate has neither an ORCID nor an
    OpenAlex author id (which would be a bug — surface the candidate as
    INSUFFICIENT_DATA at evaluation time, not silently lose it here).
    """

    if isinstance(candidate, ResearcherCandidate):
        orcid = candidate.orcid
        author_id = candidate.author_id
    else:
        orcid = candidate.get("orcid", "")
        author_id = candidate.get("author_id", "")

    normalized_orcid = _normalize_orcid(orcid)
    if normalized_orcid:
        return f"orcid:{normalized_orcid}"
    if author_id:
        return f"openalex:{_strip_openalex_prefix(author_id)}"
    return ""


def _normalize_orcid(value: Any) -> str:
    """Return the bare 16-digit ORCID (with hyphens), stripped of URL."""

    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if not candidate:
        return ""
    # Strip an "https://orcid.org/" prefix if present.
    for prefix in ("https://orcid.org/", "http://orcid.org/", "orcid.org/"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break
    # ORCID format: 0000-0001-2345-6789 (16 digits + 3 hyphens). The
    # last digit can be 'X' as a checksum. We don't validate the
    # checksum here — that's an OpenAlex concern; we just normalize.
    return candidate.strip()


def _strip_openalex_prefix(value: str) -> str:
    """OpenAlex author URLs have form ``https://openalex.org/A1234567890``.

    We store the bare id in the identity key so downstream comparisons
    don't depend on the URL form.
    """

    candidate = value.strip()
    for prefix in ("https://openalex.org/", "http://openalex.org/", "openalex.org/"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break
    return candidate
