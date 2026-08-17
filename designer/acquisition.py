"""Designer module — candidate acquisition from per-source clients.

Designer Slices 2-3. Executes :class:`designer.schemas.DesignerSearchQuery`
objects against the per-source clients (Behance in Slice 2; Google CSE
in Slice 3) and produces a deduped stream of
:class:`designer.schemas.DesignerSnippet` instances ready for the
text-based facial-stage judge.

Cross-source dedup happens by ``identity_key``. Behance candidates
carry ``behance:<username>``; Google CSE candidates carry
``cse:<host>/<first-path-segment>``. Slice 8's identity-resolution
layer joins across sources at the workspace surface (e.g., a Behance
profile + that designer's personal site discovered via CSE land as
the same canonical person); this module only dedups within the
current acquisition stream.
"""

from __future__ import annotations

from typing import AsyncIterator

from designer.schemas import (
    DesignerSearchQuery,
    DesignerSnippet,
    behance_user_to_snippet,
    cse_item_to_snippet,
)
from designer.sources.behance import BehanceClient
from designer.sources.google_cse import (
    PORTFOLIO_HOST_DOMAINS,
    GoogleCSEClient,
)


# Cap on candidates per query so a single broad query (e.g.,
# capability-area name = "design") doesn't burn the per-hour API
# budget on one search. Behance returns 12 per page by default; one
# page per query is the Slice-2 default. Wider sweeps land in Slice 5+
# once the orchestrator has a per-run governor that can spend budget
# adaptively.
DEFAULT_MAX_USERS_PER_QUERY = 12


async def acquire_behance_candidates(
    queries: list[DesignerSearchQuery],
    *,
    client: BehanceClient,
    max_users_per_query: int = DEFAULT_MAX_USERS_PER_QUERY,
) -> AsyncIterator[DesignerSnippet]:
    """Run each Behance query, yield deduped snippets.

    Generator shape so the orchestrator can apply backpressure (a
    full-stage judge call costs money; we don't want to acquire 500
    candidates before the first eval lands).

    Per-query failure: if a query 4xx/5xx's, log and continue rather
    than aborting the run. The orchestrator surfaces stop-reason
    `acquisition_partial_failure` only when ALL queries fail.
    """

    seen_identity_keys: set[str] = set()

    for query in queries:
        if query.source != "behance":
            continue

        params: dict[str, str] = {}
        if "country" in query.extra_filters and isinstance(
            query.extra_filters["country"], str
        ):
            params["country"] = query.extra_filters["country"]

        try:
            status, body = await client.search_users(
                query=query.query_text,
                country=params.get("country"),
                sort=query.sort,
                page=1,
            )
        except Exception:
            # Per-query failure is recoverable; continue with remaining
            # queries. The orchestrator's stop-reason logic decides
            # whether the run as a whole fails.
            continue

        if status != 200 or not isinstance(body, dict):
            continue

        users = body.get("users") or []
        if not isinstance(users, list):
            continue

        for user in users[:max_users_per_query]:
            if not isinstance(user, dict):
                continue
            snippet = behance_user_to_snippet(user)
            if not snippet.identity_key or snippet.identity_key == "behance:_unknown_":
                continue
            if snippet.identity_key in seen_identity_keys:
                continue
            seen_identity_keys.add(snippet.identity_key)
            yield snippet


# Per CSE query, how many results to surface. CSE returns 10/page;
# Slice-3 takes the first page only.
DEFAULT_MAX_RESULTS_PER_CSE_QUERY = 10


async def acquire_google_cse_candidates(
    queries: list[DesignerSearchQuery],
    *,
    client: GoogleCSEClient,
    portfolio_hosts: tuple[str, ...] = PORTFOLIO_HOST_DOMAINS,
    max_results_per_query: int = DEFAULT_MAX_RESULTS_PER_CSE_QUERY,
) -> AsyncIterator[DesignerSnippet]:
    """Run each CSE query against each portfolio host, yield deduped snippets.

    Each strategy-formed CSE query fans out across the portfolio-host
    set (cargo.site, squarespace.com, …) — the strategy doesn't
    site-restrict at query-formation time so the host fanout stays
    explicit at the acquisition layer.

    Per-query failure: same as Behance — log and continue rather
    than aborting the run.
    """

    seen_identity_keys: set[str] = set()

    for query in queries:
        if query.source != "google_cse":
            continue

        for host in portfolio_hosts:
            try:
                status, body = await client.search(
                    query=query.query_text,
                    site_filter=host,
                    start=1,
                )
            except Exception:
                # Per-(query, host) failure is recoverable; continue
                # with remaining hosts and queries.
                continue

            if status != 200 or not isinstance(body, dict):
                continue

            items = body.get("items") or []
            if not isinstance(items, list):
                continue

            for item in items[:max_results_per_query]:
                if not isinstance(item, dict):
                    continue
                snippet = cse_item_to_snippet(item)
                if snippet is None:
                    continue
                if snippet.identity_key in seen_identity_keys:
                    continue
                seen_identity_keys.add(snippet.identity_key)
                yield snippet


async def acquire_designer_candidates(
    queries: list[DesignerSearchQuery],
    *,
    behance_client: BehanceClient | None = None,
    google_cse_client: GoogleCSEClient | None = None,
) -> AsyncIterator[DesignerSnippet]:
    """Cross-source acquisition: Behance + Google CSE merged stream.

    Iterates Behance first (structured taxonomy → higher-precision
    matches generally), then Google CSE (broader portfolio-host
    coverage). Cross-source dedup via ``identity_key`` — Slice 8's
    cross-source identity layer eventually joins ``behance:joe`` with
    ``cse:joe.cargo.site/portfolio`` at the workspace surface; for
    the acquisition-stream layer, they remain distinct rows here.

    Either client may be ``None`` (a brief that doesn't have a
    Behance key configured runs CSE-only, and vice versa). Passing
    both ``None`` yields an empty stream.
    """

    seen: set[str] = set()
    behance_queries = [q for q in queries if q.source == "behance"]
    cse_queries = [q for q in queries if q.source == "google_cse"]

    if behance_client is not None and behance_queries:
        async for snippet in acquire_behance_candidates(
            behance_queries, client=behance_client
        ):
            if snippet.identity_key in seen:
                continue
            seen.add(snippet.identity_key)
            yield snippet

    if google_cse_client is not None and cse_queries:
        async for snippet in acquire_google_cse_candidates(
            cse_queries, client=google_cse_client
        ):
            if snippet.identity_key in seen:
                continue
            seen.add(snippet.identity_key)
            yield snippet


def dedup_snippets(snippets: list[DesignerSnippet]) -> list[DesignerSnippet]:
    """Cross-source dedup by ``identity_key``.

    Useful when a Behance acquisition stream and a Google CSE stream
    (Slice 3) merge: the same person discovered via both surfaces
    deduplicates here. Returns the snippets in original order; first
    occurrence wins (the source that surfaced the candidate first
    keeps the metadata).
    """

    seen: set[str] = set()
    out: list[DesignerSnippet] = []
    for snippet in snippets:
        if snippet.identity_key in seen:
            continue
        seen.add(snippet.identity_key)
        out.append(snippet)
    return out
