"""Researcher acquisition and identity-matched evidence.

Frontier execution fetches bounded pages and persists them through the runtime
bridge. Optional legacy arXiv/S2 enrichment accepts corroborated matches only."""

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
    from researcher.strategy import compile_query
    compiled = compile_query(query, client)
    if compiled["route"] == "work_search":
        result = AcquisitionResult(query_id=int(query.get("id") or 0), query_name=query.get("name", ""))
        cursor = "*"
        seen = set()
        while cursor and len(result.candidates) < max_authors:
            page = fetch_query_page(query=compiled, client=client, cursor=cursor)
            result.pages_fetched += 1
            result.raw_authors_seen += page["results_returned"]
            for candidate in page["candidates"]:
                if candidate.author_id in seen:
                    result.duplicates_skipped += 1
                else:
                    seen.add(candidate.author_id)
                    result.candidates.append(candidate)
            cursor = page["next_cursor"]
        result.truncated = bool(cursor or len(result.candidates) > max_authors)
        result.candidates = result.candidates[:max_authors]
        return result
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
            candidate.evidence_gaps.append({"provider": "arxiv", "status": "failed", "reason": type(exc).__name__})
            continue

        seen_categories: list[str] = []
        for entry in response.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            known_ids = {p.doi for p in candidate.top_papers if p.doi} | {v.get("url", "") for p in candidate.top_papers for v in p.versions}
            authors = {str(a.get("name", "") if isinstance(a, dict) else a).casefold() for a in entry.get("authors", [])}
            if candidate.name.casefold() not in authors or not known_ids.intersection({entry.get("doi"), entry.get("id"), entry.get("link")}):
                continue
            primary = (entry.get("primary_category") or "").strip()
            if primary and primary not in seen_categories:
                seen_categories.append(primary)
            for cat in entry.get("categories") or []:
                cat_str = (str(cat) if cat else "").strip()
                if cat_str and cat_str not in seen_categories:
                    seen_categories.append(cat_str)
        candidate.arxiv_categories = seen_categories
        if not seen_categories:
            candidate.evidence_gaps.append({"provider": "arxiv", "status": "unmatched", "reason": "No corroborated work and author match"})


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
            candidate.evidence_gaps.append({"provider": "semantic_scholar", "status": "failed", "reason": type(exc).__name__})
            continue
        if not s2_author:
            candidate.evidence_gaps.append({"provider": "semantic_scholar", "status": "unmatched", "reason": "No corroborated author identifier"})
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
        if candidate.h_index is not None and candidate.h_index > 0:
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
    """Accept only an ORCID-correlated response; never substitute a namesake."""

    orcid = _bare_orcid(candidate.orcid)
    if orcid:
        try:
            author = semantic_scholar_client.get_author(
                f"ORCID:{orcid}",
                fields=["hIndex", "citationCount", "paperCount", "name", "externalIds"],
            )
            if _bare_orcid(str((author.get("externalIds") or {}).get("ORCID", ""))) == orcid:
                return author
        except Exception as exc:
            logger.debug(
                "researcher.acquisition: s2 ORCID lookup failed for %s (%s)",
                candidate.name,
                exc,
            )
            raise
    return None


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


def _optional_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


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
    h_index = _coerce_int(summary.get("h_index"))
    citation_count = _coerce_int(raw.get("cited_by_count"))
    works_count = _coerce_int(raw.get("works_count"))

    counts_by_year = raw.get("counts_by_year") or []
    papers_in_window = sum(
        _coerce_int(c.get("works_count"))
        for c in counts_by_year
        if isinstance(c, dict) and _coerce_int(c.get("year")) >= window_start_year
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


def _extract_top_papers(raw: dict, *, limit: int | None = 5) -> list[ResearcherPaper]:
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
        authorships = w.get("authorships") or []
        matched = next((a for a in authorships if isinstance(a, dict) and _extract_author_id(raw) and _extract_author_id(a.get("author") or {}) == _extract_author_id(raw)), {})
        if authorships and not matched:
            continue
        position = str(matched.get("author_position") or "").lower()
        openalex_id = str(w.get("id") or "").strip()
        doi = str(w.get("doi") or "").removeprefix("https://doi.org/").lower()
        url = primary_loc.get("landing_page_url") or w.get("doi") or openalex_id
        abstract = _abstract_text(w.get("abstract_inverted_index"))
        version = primary_loc.get("version") or "unknown"
        papers.append(
            ResearcherPaper(
                title=title,
                venue=venue,
                year=publication_year,
                citation_count=cited_by,
                is_first_author=position == "first",
                openalex_id=openalex_id,
                evidence_id=f"doi:{doi}" if doi else openalex_id,
                doi=doi, url=url, publication_date=w.get("publication_date") or "",
                version=version, status="withdrawn" if w.get("is_retracted") else ("published" if version == "publishedVersion" else "unknown"),
                content=abstract, content_location="OpenAlex abstract_inverted_index",
                matched_author_id=_extract_author_id(matched.get("author") or {}), author_position=position,
                equal_contribution=matched.get("equal_contribution"),
                contribution_statement=str(matched.get("contribution_statement") or "")[:4000],
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                availability="available" if abstract else "metadata_only",
                versions=[{"url": url, "version": version, "publication_date": w.get("publication_date"), "provider_id": openalex_id, "content": abstract, "withdrawn": bool(w.get("is_retracted"))}],
            )
        )
    return merge_papers(papers)[:limit]


def _abstract_text(index: dict | None) -> str:
    if not isinstance(index, dict):
        return ""
    tokens = [(pos, word) for word, positions in index.items() for pos in positions if type(pos) is int and 0 <= pos < 4000]
    return " ".join(word for _, word in sorted(tokens))[:16000]


def _publication_date(work: dict) -> str | None:
    from datetime import date
    value = work.get("publication_date")
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def merge_papers(papers: list[ResearcherPaper]) -> list[ResearcherPaper]:
    grouped = {}
    for paper in papers:
        key = paper.evidence_id or paper.openalex_id or paper.url
        if not key:  # Legacy records with no identifiers remain distinct observations.
            key = f"legacy:{len(grouped)}"
        if key in grouped:
            old = grouped[key]
            for version in paper.versions:
                if version not in old.versions:
                    old.versions.append(version)
            if not old.content and paper.content:
                old.content, old.availability = paper.content, paper.availability
        else:
            grouped[key] = paper
    return list(grouped.values())


def fetch_query_page(*, query: dict, client, cursor="*", persist_compiled=None) -> dict:
    """One bounded request, normalized without another paid acquisition call."""
    from researcher.strategy import compile_query
    compiled = compile_query(query, client)
    if persist_compiled:
        persist_compiled(compiled)
    params = {**compiled["compiled_params"], "cursor": cursor}
    if compiled["route"] == "work_search":
        response = client.search_works(**params)
    elif compiled["route"] == "people_search":
        response = client.search_authors(**params)
    else:
        raise ValueError("Public-web queries require the Exa transport")
    if not isinstance(response.get("results"), list) or not isinstance(response.get("meta"), dict):
        raise ValueError("OpenAlex response lacks results/continuation metadata")
    if "next_cursor" not in response["meta"] or (response["meta"]["next_cursor"] is not None and not isinstance(response["meta"]["next_cursor"], str)):
        raise ValueError("OpenAlex response has missing or invalid continuation metadata")
    results = response["results"]
    candidates = {}
    excluded = 0
    for raw in results:
        if compiled["route"] == "people_search":
            records = [raw]
        else:
            published = _publication_date(raw)
            venue_ids = {str((loc.get("source") or {}).get("id", "")).rsplit("/", 1)[-1] for loc in raw.get("locations", []) + [raw.get("primary_location") or {}]}
            wanted = set(compiled["resolved_entities"]["sources"].values())
            if (params.get("from_date") and (not published or published < params["from_date"])) or (params.get("to_date") and (not published or published > params["to_date"])) or (wanted and not wanted.intersection(venue_ids)):
                excluded += 1
                continue
            records = [{**a["author"], "top_works": [raw], "last_known_institutions": a.get("institutions", [])} for a in raw.get("authorships", []) if (a.get("author") or {}).get("id")]
        for record in records:
            key = _extract_author_id(record)
            if not key:
                continue
            candidate = _build_candidate(record, window_start_year=datetime.now(timezone.utc).year - 2)
            candidate.payload_version = query.get("payload_version", 1)
            candidate.reference_date = query.get("reference_date", "")
            candidate.h_index = _optional_count((record.get("summary_stats") or {}).get("h_index"))
            if candidate.payload_version >= 2:
                candidate.papers_in_window = None  # Year buckets cannot prove the rolling date boundary.
            candidate.discovery_queries = [str(query.get("id", ""))]
            candidate.identity_refs = {"openalex": key}
            if candidate.orcid:
                candidate.identity_refs["orcid"] = _bare_orcid(candidate.orcid)
            candidate.professional_context = [{"claim": a, "kind": "publication_affiliation", "source_date": record.get("updated_date"), "url": candidate.profile_url} for a in candidate.affiliations]
            if key in candidates:
                candidates[key].top_papers = merge_papers(candidates[key].top_papers + candidate.top_papers)
            else:
                candidates[key] = candidate
    next_cursor = response["meta"].get("next_cursor")
    return {"candidates": list(candidates.values()), "next_cursor": next_cursor, "exhausted": not bool(next_cursor),
            "results_returned": len(results), "matched_works": len(results) - excluded if compiled["route"] == "work_search" else 0,
            "coverage": "provider results filtered by exact dates and venue; bounded page", "excluded_controls": excluded}


def enrich_research_candidate(candidate: ResearcherCandidate, *, client, source_config: dict, persist=lambda candidate: None, public_fetcher=None) -> None:
    """Retrieve actual matched works before full judgment; persist each usable response."""
    from researcher.strategy import rolling_window_start
    from researcher.sources.public_web import fetch_public_text, corroborate_public_identity
    public_fetcher = public_fetcher or fetch_public_text
    corroborate_public_identity(candidate, fetcher=public_fetcher)
    months = source_config.get("papers_in_window_months", 36)
    if candidate.evidence_status == "complete" and candidate.evidence_window_months == months:
        return
    if not candidate.author_id:
        return
    reference = candidate.reference_date or datetime.now(timezone.utc).date().isoformat()
    if "summary_stats" not in candidate.raw_openalex:
        raw = client.get_author(candidate.author_id)
        if _extract_author_id(raw) != candidate.author_id:
            raise ValueError("OpenAlex author identity mismatch")
        candidate.h_index = _optional_count((raw.get("summary_stats") or {}).get("h_index"))
        candidate.raw_openalex = raw
        if raw.get("orcid"):
            candidate.orcid = raw["orcid"]
            candidate.identity_refs["orcid"] = _bare_orcid(candidate.orcid)
        persist(candidate)
    start = rolling_window_start(reference, months)
    response = client.search_works(author_id=candidate.author_id, from_date=start, to_date=reference, per_page=25)
    if not isinstance(response.get("results"), list) or not isinstance(response.get("meta"), dict):
        raise ValueError("OpenAlex works evidence unavailable")
    works = [w for w in response["results"] if _publication_date(w) and start <= _publication_date(w) <= reference]
    papers = _extract_top_papers({"id": candidate.author_id, "top_works": works}, limit=None)
    candidate.top_papers = merge_papers(candidate.top_papers + papers)[:5]
    ambiguous = any(not _publication_date(w) or not any(_extract_author_id(a.get("author") or {}) == candidate.author_id for a in w.get("authorships", [])) for w in response["results"])
    complete = "next_cursor" in response["meta"] and not response["meta"]["next_cursor"]
    candidate.papers_in_window = len(papers) if complete and not ambiguous else None
    if ambiguous:
        candidate.evidence_gaps.append({"provider": "openalex", "status": "unknown", "reason": "A work lacks an exact date or matched authorship; rolling publication count is unknown"})
    if not complete:
        candidate.evidence_gaps.append({"provider": "openalex", "status": "capped", "reason": "Author work page is incomplete; publication count unknown"})
    persist(candidate)
    for paper in candidate.top_papers[:3]:
        if paper.url and not paper.contribution_statement and candidate.name.casefold() not in paper.content.casefold():
            try:
                fetched = public_fetcher(paper.url)
            except (ValueError, OSError) as exc:
                fetched = {"status": "unavailable", "reason": type(exc).__name__}
            if fetched.get("status") == "ok" and fetched.get("text"):
                paper.content = fetched["text"][:16000]
                paper.content_location = fetched.get("url") or paper.url
                paper.availability = "available"
            else:
                candidate.evidence_gaps.append({"provider": "public_web", "status": fetched.get("status", "unavailable"), "reason": "Original paper text unavailable"})
            persist(candidate)
    candidate.evidence_status = "complete"
    candidate.evidence_window_months = months
    persist(candidate)


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
