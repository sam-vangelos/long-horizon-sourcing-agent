"""Researcher acquisition — Slice 4 (audit Move #4 R1: arXiv + S2 enrichment).

Execute one researcher query against OpenAlex, paginate, dedup by
``author_id``, hand off to disambiguation. The result is a list of
:class:`ResearcherCandidate` records ready for the deterministic gates
+ LLM evaluators at Slice 5.

Audit Move #4 R1 wires the two existing source clients
(:mod:`researcher.sources.arxiv` + :mod:`researcher.sources.semantic_scholar`)
into the acquisition pipeline as optional enrichers:

- :func:`enrich_with_arxiv` populates each candidate's
  ``arxiv_categories`` so the facial evaluator can read recent
  preprint signal that OpenAlex hasn't indexed yet (matters most
  for early-career researchers).
- :func:`reconcile_h_index_with_semantic_scholar` cross-checks each
  candidate's h_index against Semantic Scholar's; when S2's number
  is materially higher (>20% by default), the candidate's h_index
  is bumped to S2's. Closes Spec Opinion 1's "all three sources
  contribute" requirement that pre-R1 the MVP only had OpenAlex
  for.

The query dict shape is the one produced by
:func:`researcher.strategy.form_strategy` — see
:data:`researcher.strategy.RESEARCHER_QUERY_SCHEMA_KEYS`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from researcher.schemas import ResearcherCandidate, ResearcherPaper
from researcher.sources.openalex import OpenAlexClient

logger = logging.getLogger(__name__)


# Default per-query result cap. The orchestrator (Slice 6) can override.
_DEFAULT_MAX_AUTHORS = 200

# Default fraction by which Semantic Scholar's h-index must exceed
# OpenAlex's for the reconciliation path to bump the candidate's
# canonical h_index. Below the threshold, the OpenAlex value stays.
_DEFAULT_H_INDEX_RECONCILIATION_THRESHOLD = 0.20


@dataclass
class AcquisitionResult:
    """Outcome of a single query execution."""

    query_id: int
    query_name: str
    candidates: list[ResearcherCandidate] = field(default_factory=list)
    pages_fetched: int = 0
    raw_authors_seen: int = 0
    duplicates_skipped: int = 0
    truncated: bool = False  # True when we hit max_authors before exhausting

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query_name": self.query_name,
            "candidate_count": len(self.candidates),
            "pages_fetched": self.pages_fetched,
            "raw_authors_seen": self.raw_authors_seen,
            "duplicates_skipped": self.duplicates_skipped,
            "truncated": self.truncated,
        }


def execute_query(
    *,
    query: dict,
    client: OpenAlexClient,
    papers_in_window_months: int = 36,
    max_authors: int = _DEFAULT_MAX_AUTHORS,
    per_page: int = 25,
    now: datetime | None = None,
    arxiv_client: Any = None,
    semantic_scholar_client: Any = None,
) -> AcquisitionResult:
    """Execute a researcher query against OpenAlex; return deduped candidates.

    The query dict matches the strategy output:
    ``{topic_concepts, venue_filter, min_year, min_citations,
    ror_country_filter, id, name, ...}``.

    Pagination uses OpenAlex's cursor protocol (``cursor=*`` initial,
    then ``meta.next_cursor`` from each response). Stops when:
    - ``meta.next_cursor`` is null/empty (exhaustion), or
    - we've collected ``max_authors`` unique candidates (truncation).

    Audit Move #4 R1 — when ``arxiv_client`` is supplied, candidates
    are enriched with ``arxiv_categories`` from author-lastname-matched
    preprints. When ``semantic_scholar_client`` is supplied, each
    candidate's h_index is reconciled against Semantic Scholar's
    (S2 wins on materially-higher values). Both enrichers are
    fail-soft: any per-candidate API error logs + skips that
    candidate's enrichment without aborting the broader acquisition.

    Pre-R1 callers (orchestrator with no clients supplied) are
    byte-identical to today's behavior — both kwargs default to None.
    """

    now = now or datetime.now(timezone.utc)
    window_start_year = now.year - max(0, int(papers_in_window_months) // 12)

    seen_author_ids: set[str] = set()
    candidates: list[ResearcherCandidate] = []
    pages_fetched = 0
    raw_authors_seen = 0
    duplicates_skipped = 0
    cursor = "*"
    truncated = False

    concept_ids = _as_str_list(query.get("topic_concepts"))
    country_codes = _as_str_list(query.get("ror_country_filter"))
    min_citations = int(query.get("min_citations") or 0)

    while cursor and len(candidates) < max_authors:
        response = client.search_authors(
            concept_ids=concept_ids or None,
            country_codes=country_codes or None,
            min_citations=min_citations or None,
            cursor=cursor,
            per_page=per_page,
        )
        pages_fetched += 1

        results = response.get("results") or []
        for raw in results:
            raw_authors_seen += 1
            author_id = _extract_author_id(raw)
            if not author_id:
                continue
            if author_id in seen_author_ids:
                duplicates_skipped += 1
                continue
            seen_author_ids.add(author_id)
            candidates.append(
                _build_candidate(
                    raw,
                    window_start_year=window_start_year,
                )
            )
            if len(candidates) >= max_authors:
                truncated = True
                break

        cursor = ((response.get("meta") or {}).get("next_cursor") or "") if not truncated else ""

    # Audit Move #4 R1: arXiv + Semantic Scholar enrichment passes.
    # Both are no-ops when the clients aren't supplied (the default
    # for callers that haven't opted into the multi-source path).
    if arxiv_client is not None:
        enrich_with_arxiv(
            candidates,
            arxiv_client=arxiv_client,
            arxiv_categories_query=_as_str_list(query.get("arxiv_categories")),
        )
    if semantic_scholar_client is not None:
        reconcile_h_index_with_semantic_scholar(
            candidates,
            semantic_scholar_client=semantic_scholar_client,
        )

    return AcquisitionResult(
        query_id=int(query.get("id") or 0),
        query_name=str(query.get("name") or ""),
        candidates=candidates,
        pages_fetched=pages_fetched,
        raw_authors_seen=raw_authors_seen,
        duplicates_skipped=duplicates_skipped,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# arXiv enrichment — audit Move #4 R1
# ---------------------------------------------------------------------------


def enrich_with_arxiv(
    candidates: list[ResearcherCandidate],
    *,
    arxiv_client: Any,
    arxiv_categories_query: list[str] | None = None,
    max_authors_to_enrich: int = 50,
) -> None:
    """Populate ``candidate.arxiv_categories`` from arXiv preprint search.

    For each candidate (up to ``max_authors_to_enrich`` to bound API
    cost), runs an arXiv author-lastname search filtered to the
    declared ``arxiv_categories_query`` (if any), and collects the
    distinct primary_category strings from the returned preprints.

    Mutates candidates in place. Per-candidate failures (network,
    parse) log a warning and skip that candidate; the acquisition
    pipeline never aborts on enrichment failures.
    """

    if not candidates:
        return
    for candidate in candidates[:max_authors_to_enrich]:
        lastname = _last_name(candidate.name)
        if not lastname:
            continue
        try:
            response = arxiv_client.search(
                categories=list(arxiv_categories_query or []) or None,
                author_lastname=lastname,
                max_results=10,
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft per-candidate
            logger.warning(
                "researcher.acquisition: arxiv enrichment failed for %s (%s)",
                candidate.name,
                exc,
            )
            continue

        seen_categories: list[str] = []
        for entry in response.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            primary = (entry.get("primary_category") or "").strip()
            if primary and primary not in seen_categories:
                seen_categories.append(primary)
            for cat in entry.get("categories") or []:
                cat_str = (str(cat) if cat else "").strip()
                if cat_str and cat_str not in seen_categories:
                    seen_categories.append(cat_str)
        candidate.arxiv_categories = seen_categories


# ---------------------------------------------------------------------------
# Semantic Scholar h-index reconciliation — audit Move #4 R1
# ---------------------------------------------------------------------------


def reconcile_h_index_with_semantic_scholar(
    candidates: list[ResearcherCandidate],
    *,
    semantic_scholar_client: Any,
    threshold_fraction: float = _DEFAULT_H_INDEX_RECONCILIATION_THRESHOLD,
    max_authors_to_reconcile: int = 50,
) -> None:
    """Cross-check each candidate's h_index against Semantic Scholar.

    Per Spec Opinion 1's "all three sources contribute" requirement.
    OpenAlex sometimes undercounts citations from authors with name
    aliases or pre-2010 publications; Semantic Scholar often has a
    fuller picture. When S2's h-index is more than
    ``threshold_fraction`` higher than OpenAlex's, this helper bumps
    the candidate's canonical ``h_index`` to S2's value. The raw S2
    number lands in ``candidate.s2_h_index`` regardless for diagnostic
    provenance.

    Strategy for resolving each candidate to an S2 author:
    1. ORCID (when present) — S2 supports ORCID lookup directly.
    2. Name search (best-effort) — picks the highest-citation match.

    Mutates candidates in place. Per-candidate failures log a warning
    and skip; acquisition continues. Bounded at
    ``max_authors_to_reconcile`` so a 200-candidate run doesn't
    consume the full S2 free-tier quota.
    """

    if not candidates:
        return
    for candidate in candidates[:max_authors_to_reconcile]:
        try:
            s2_author = _resolve_s2_author(candidate, semantic_scholar_client)
        except Exception as exc:  # noqa: BLE001 — fail-soft
            logger.warning(
                "researcher.acquisition: semantic scholar lookup failed for %s (%s)",
                candidate.name,
                exc,
            )
            continue
        if not s2_author:
            continue
        s2_h = _coerce_int(s2_author.get("hIndex"))
        if s2_h <= 0:
            continue
        candidate.s2_h_index = s2_h
        # Bump the canonical h_index when S2's is materially higher.
        # This isn't a "trust S2 more" claim; it's an observation that
        # OpenAlex's per-author rollup misses citation channels S2 sees
        # (preprints, conference proceedings outside the major venues).
        # When the delta is small, OpenAlex's number stays — we don't
        # want noise-driven flips at the floor boundary.
        if candidate.h_index > 0:
            delta_fraction = (s2_h - candidate.h_index) / candidate.h_index
            if delta_fraction > threshold_fraction:
                candidate.h_index = s2_h
        elif s2_h > 0:
            # OpenAlex returned 0 (missing summary_stats); use S2's.
            candidate.h_index = s2_h


def _resolve_s2_author(
    candidate: ResearcherCandidate,
    semantic_scholar_client: Any,
) -> dict | None:
    """Resolve a candidate to an S2 author record. ORCID first, then name."""

    orcid = _bare_orcid(candidate.orcid)
    if orcid:
        try:
            return semantic_scholar_client.get_author(
                f"ORCID:{orcid}",
                fields=["hIndex", "citationCount", "paperCount", "name"],
            )
        except Exception as exc:  # noqa: BLE001 — fall through to name
            logger.debug(
                "researcher.acquisition: s2 ORCID lookup failed for %s (%s)",
                candidate.name,
                exc,
            )
    if not candidate.name:
        return None
    response = semantic_scholar_client.search_authors(
        candidate.name,
        limit=5,
        fields=["hIndex", "citationCount", "paperCount", "name"],
    )
    matches = response.get("data") or []
    if not matches:
        return None
    # Pick the highest-citation match — best heuristic for "the right
    # person" when ORCID is absent. The disambiguation pass at Slice 4
    # already filters by country / concept upstream, so the name list
    # here is narrow.
    matches.sort(
        key=lambda r: -_coerce_int((r or {}).get("citationCount", 0))
    )
    top = matches[0]
    if not isinstance(top, dict):
        return None
    return top


def _last_name(full_name: str) -> str:
    """Extract the surname from a full-name string. Defensive."""

    cleaned = (full_name or "").strip()
    if not cleaned:
        return ""
    # Split on whitespace; surname is the last non-empty token.
    parts = [p for p in re.split(r"\s+", cleaned) if p]
    if not parts:
        return ""
    return parts[-1]


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bare_orcid(orcid: str) -> str:
    """Strip the ORCID URL prefix, leaving ``0000-0001-...`` or empty."""

    text = (orcid or "").strip()
    if not text:
        return ""
    for prefix in ("https://orcid.org/", "http://orcid.org/", "orcid.org/"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _build_candidate(
    raw: dict,
    *,
    window_start_year: int,
) -> ResearcherCandidate:
    """Build a :class:`ResearcherCandidate` from one OpenAlex author dict."""

    author_id = _extract_author_id(raw)
    orcid = str(raw.get("orcid") or "").strip()
    name = str(raw.get("display_name") or "").strip()

    affiliations = _extract_affiliations(raw)
    summary = raw.get("summary_stats") or {}
    h_index = int(summary.get("h_index") or 0)
    citation_count = int(raw.get("cited_by_count") or 0)
    works_count = int(raw.get("works_count") or 0)

    counts_by_year = raw.get("counts_by_year") or []
    papers_in_window = sum(
        int(c.get("works_count") or 0)
        for c in counts_by_year
        if isinstance(c, dict) and int(c.get("year") or 0) >= window_start_year
    )

    profile_url = (
        f"https://orcid.org/{_bare_orcid(orcid)}"
        if orcid
        else (raw.get("id") or f"https://openalex.org/{author_id}")
    )

    top_papers = _extract_top_papers(raw)

    return ResearcherCandidate(
        author_id=author_id,
        orcid=orcid,
        name=name,
        affiliations=affiliations,
        top_papers=top_papers,
        h_index=h_index,
        citation_count=citation_count,
        works_count=works_count,
        papers_in_window=papers_in_window,
        profile_url=profile_url,
        raw_openalex=raw,
    )


def _extract_author_id(raw: dict) -> str:
    """OpenAlex authors carry an id like ``https://openalex.org/A1234``.

    We store the bare ``A1234`` for stable comparison.
    """

    raw_id = str(raw.get("id") or "").strip()
    if not raw_id:
        return ""
    for prefix in (
        "https://openalex.org/",
        "http://openalex.org/",
        "openalex.org/",
    ):
        if raw_id.startswith(prefix):
            return raw_id[len(prefix):]
    return raw_id


def _extract_affiliations(raw: dict) -> list[str]:
    affiliations: list[str] = []
    for inst in raw.get("last_known_institutions") or []:
        if not isinstance(inst, dict):
            continue
        name = str(inst.get("display_name") or "").strip()
        country = str(inst.get("country_code") or "").strip()
        if name and country:
            affiliations.append(f"{name} ({country})")
        elif name:
            affiliations.append(name)
    return affiliations


def _extract_top_papers(raw: dict) -> list[ResearcherPaper]:
    """OpenAlex author payloads optionally embed top works under
    ``top_works`` (Cloris's enrichment may also stuff the works list
    here at acquisition time). Defensive: handle absence.
    """

    works = raw.get("top_works") or raw.get("works") or []
    papers: list[ResearcherPaper] = []
    for w in works:
        if not isinstance(w, dict):
            continue
        title = str(w.get("title") or w.get("display_name") or "").strip()
        if not title:
            continue
        venue = ""
        primary_loc = w.get("primary_location") or {}
        source = primary_loc.get("source") if isinstance(primary_loc, dict) else None
        if isinstance(source, dict):
            venue = str(source.get("display_name") or "").strip()
        publication_year = int(w.get("publication_year") or 0)
        cited_by = int(w.get("cited_by_count") or 0)
        first_author_position = False
        authorships = w.get("authorships") or []
        if isinstance(authorships, list) and authorships:
            first = authorships[0]
            if isinstance(first, dict):
                first_author_position = (
                    str(first.get("author_position") or "").lower() == "first"
                )
        openalex_id = str(w.get("id") or "").strip()
        papers.append(
            ResearcherPaper(
                title=title,
                venue=venue,
                year=publication_year,
                citation_count=cited_by,
                is_first_author=first_author_position,
                openalex_id=openalex_id,
            )
        )
    return papers[:5]


def _bare_orcid(orcid: str) -> str:
    candidate = orcid.strip()
    for prefix in ("https://orcid.org/", "http://orcid.org/", "orcid.org/"):
        if candidate.startswith(prefix):
            return candidate[len(prefix):]
    return candidate


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, str) and str(v).strip()]
