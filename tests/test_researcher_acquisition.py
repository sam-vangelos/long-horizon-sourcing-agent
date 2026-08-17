"""Researcher Slice 4 — acquisition coverage.

Asserts:
- Single-page query: results land as ResearcherCandidate; raw OpenAlex
  payload preserved.
- Multi-page pagination: cursor protocol drives a second call with the
  next_cursor; dedup by author_id across pages.
- Truncation: hitting max_authors before exhaustion sets ``truncated``.
- papers_in_window is computed from counts_by_year against the window.
- identity_key_for_candidate prefers ORCID over OpenAlex id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from researcher.acquisition import execute_query
from researcher.schemas import (
    ResearcherCandidate,
    ResearcherPaper,
    identity_key_for_candidate,
)


# ---------------------------------------------------------------------------
# Stub OpenAlex client
# ---------------------------------------------------------------------------


class _StubOpenAlexClient:
    """Replays canned responses in order; records every call's params."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []
        self._idx = 0

    def search_authors(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        if self._idx >= len(self.responses):
            raise AssertionError(
                f"_StubOpenAlexClient: no canned response for call #{self._idx + 1}"
            )
        response = self.responses[self._idx]
        self._idx += 1
        return response


def _author_payload(
    *,
    author_id: str,
    name: str,
    orcid: str = "",
    h_index: int = 10,
    citation_count: int = 100,
    works_count: int = 20,
    counts_by_year: list[dict] | None = None,
    institutions: list[dict] | None = None,
    x_concepts: list[dict] | None = None,
) -> dict:
    return {
        "id": f"https://openalex.org/{author_id}",
        "display_name": name,
        "orcid": orcid,
        "summary_stats": {"h_index": h_index},
        "cited_by_count": citation_count,
        "works_count": works_count,
        "counts_by_year": counts_by_year or [
            {"year": 2024, "works_count": 3},
            {"year": 2023, "works_count": 4},
            {"year": 2020, "works_count": 5},
        ],
        "last_known_institutions": institutions
        or [{"display_name": "MIT", "country_code": "US"}],
        "x_concepts": x_concepts
        or [
            {"id": "https://openalex.org/C2778407487", "display_name": "NLP"}
        ],
    }


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def test_execute_query_single_page_returns_candidates() -> None:
    client = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [
                    _author_payload(author_id="A1", name="Jane Researcher"),
                    _author_payload(author_id="A2", name="Bob Builder"),
                ],
            }
        ]
    )
    query = {
        "id": 7,
        "name": "RLHF · NeurIPS",
        "topic_concepts": ["C2778407487"],
        "ror_country_filter": ["US"],
        "min_citations": 50,
    }
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)

    result = execute_query(query=query, client=client, now=now)

    assert result.query_id == 7
    assert result.query_name == "RLHF · NeurIPS"
    assert result.pages_fetched == 1
    assert result.raw_authors_seen == 2
    assert result.duplicates_skipped == 0
    assert not result.truncated
    assert len(result.candidates) == 2
    first = result.candidates[0]
    assert isinstance(first, ResearcherCandidate)
    assert first.author_id == "A1"
    assert first.name == "Jane Researcher"
    assert "MIT (US)" in first.affiliations
    # Filters propagate into the OpenAlex client params.
    assert client.calls[0]["concept_ids"] == ["C2778407487"]
    assert client.calls[0]["country_codes"] == ["US"]
    assert client.calls[0]["min_citations"] == 50


def test_execute_query_paginates_and_dedups_across_pages() -> None:
    client = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": "page2"},
                "results": [
                    _author_payload(author_id="A1", name="Alice One"),
                    _author_payload(author_id="A2", name="Bob Two"),
                ],
            },
            {
                "meta": {"next_cursor": ""},
                "results": [
                    # A2 duplicate across pages
                    _author_payload(author_id="A2", name="Bob Two"),
                    _author_payload(author_id="A3", name="Carol Three"),
                ],
            },
        ]
    )
    result = execute_query(query={"id": 1, "name": "q1"}, client=client)

    assert result.pages_fetched == 2
    assert result.raw_authors_seen == 4
    assert result.duplicates_skipped == 1
    assert [c.author_id for c in result.candidates] == ["A1", "A2", "A3"]
    # Second call uses the next_cursor returned by the first.
    assert client.calls[1]["cursor"] == "page2"


def test_execute_query_truncates_at_max_authors() -> None:
    client = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": "page2"},
                "results": [
                    _author_payload(author_id=f"A{i}", name=f"Author {i}")
                    for i in range(5)
                ],
            },
        ]
    )
    result = execute_query(query={}, client=client, max_authors=3)
    assert len(result.candidates) == 3
    assert result.truncated is True
    # We do not fetch the second page once truncated.
    assert client._idx == 1


def test_papers_in_window_counts_recent_works() -> None:
    """Window is `now.year - papers_in_window_months // 12`. With now=
    2025 and 36 months → window starts at 2022. Only 2024+2023 land
    inside; 2020 does not."""

    client = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [
                    _author_payload(
                        author_id="A1",
                        name="Author",
                        counts_by_year=[
                            {"year": 2024, "works_count": 3},
                            {"year": 2023, "works_count": 4},
                            {"year": 2020, "works_count": 5},
                        ],
                    )
                ],
            }
        ]
    )
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    result = execute_query(
        query={},
        client=client,
        papers_in_window_months=36,
        now=now,
    )
    assert result.candidates[0].papers_in_window == 7  # 3 + 4 (2020 excluded)


def test_acquisition_skips_authors_without_id() -> None:
    client = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [
                    {"display_name": "No ID", "orcid": ""},
                    _author_payload(author_id="A1", name="Has ID"),
                ],
            }
        ]
    )
    result = execute_query(query={}, client=client)
    assert len(result.candidates) == 1
    assert result.candidates[0].author_id == "A1"


# ---------------------------------------------------------------------------
# Identity key resolution (Spec Opinion 3)
# ---------------------------------------------------------------------------


def test_identity_key_prefers_orcid_when_present() -> None:
    candidate = ResearcherCandidate(
        author_id="A1234",
        orcid="https://orcid.org/0000-0001-2345-6789",
        name="Jane R.",
    )
    assert (
        identity_key_for_candidate(candidate)
        == "orcid:0000-0001-2345-6789"
    )


def test_identity_key_falls_back_to_openalex_when_orcid_missing() -> None:
    candidate = ResearcherCandidate(
        author_id="A1234",
        orcid="",
        name="Jane R.",
    )
    assert identity_key_for_candidate(candidate) == "openalex:A1234"


def test_identity_key_strips_openalex_url_prefix() -> None:
    candidate = ResearcherCandidate(
        author_id="https://openalex.org/A1234",
        orcid="",
        name="Jane R.",
    )
    assert identity_key_for_candidate(candidate) == "openalex:A1234"


def test_identity_key_empty_when_no_anchor() -> None:
    candidate = ResearcherCandidate(author_id="", orcid="", name="No Anchor")
    assert identity_key_for_candidate(candidate) == ""


def test_identity_key_deterministic_across_runs() -> None:
    """Same OpenAlex payload → same identity key. This is the dedup
    primitive across runs of the same brief.
    """

    a = ResearcherCandidate(
        author_id="A1234",
        orcid="0000-0001-2345-6789",
        name="X",
    )
    b = ResearcherCandidate(
        author_id="A1234",
        orcid="0000-0001-2345-6789",
        name="X",
    )
    assert identity_key_for_candidate(a) == identity_key_for_candidate(b)


def test_identity_key_accepts_dict_shape() -> None:
    """Hydration paths (e.g., from terminal_payload_json) may pass the
    candidate as a dict — the helper should not require dataclass
    construction for the lookup.
    """

    payload = {"author_id": "A1234", "orcid": ""}
    assert identity_key_for_candidate(payload) == "openalex:A1234"


# ---------------------------------------------------------------------------
# ResearcherCandidate / ResearcherPaper round-trip
# ---------------------------------------------------------------------------


def test_researcher_candidate_round_trips_via_dict() -> None:
    candidate = ResearcherCandidate(
        author_id="A1",
        orcid="0000-0001",
        name="X",
        affiliations=["MIT (US)"],
        top_papers=[
            ResearcherPaper(
                title="Survey of RLHF",
                venue="NeurIPS",
                year=2024,
                citation_count=50,
                is_first_author=True,
            )
        ],
        h_index=12,
        citation_count=300,
        works_count=20,
        papers_in_window=4,
    )
    rehydrated = ResearcherCandidate.from_dict(candidate.to_dict())
    assert rehydrated.author_id == candidate.author_id
    assert rehydrated.affiliations == candidate.affiliations
    assert rehydrated.top_papers[0].title == "Survey of RLHF"
    assert rehydrated.top_papers[0].is_first_author is True


# ---------------------------------------------------------------------------
# Audit Move #4 R1 — arXiv + Semantic Scholar enrichment
# ---------------------------------------------------------------------------


class _StubArxivClient:
    """Replays canned arXiv responses + records every call."""

    def __init__(self, response_by_lastname: dict[str, dict]) -> None:
        self.response_by_lastname = response_by_lastname
        self.calls: list[dict] = []

    def search(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        lastname = kwargs.get("author_lastname") or ""
        return self.response_by_lastname.get(lastname, {"entries": []})


class _StubS2Client:
    """Replays canned Semantic Scholar responses + records calls."""

    def __init__(
        self,
        author_by_orcid: dict[str, dict] | None = None,
        author_by_name_search: dict[str, dict] | None = None,
    ) -> None:
        self.author_by_orcid = author_by_orcid or {}
        self.author_by_name_search = author_by_name_search or {}
        self.get_author_calls: list[str] = []
        self.search_authors_calls: list[str] = []

    def get_author(self, author_id: str, *, fields=None) -> dict:
        self.get_author_calls.append(author_id)
        if author_id.startswith("ORCID:"):
            orcid = author_id[len("ORCID:"):]
            if orcid in self.author_by_orcid:
                return self.author_by_orcid[orcid]
            raise RuntimeError("orcid not found")
        raise RuntimeError("non-ORCID lookup not stubbed")

    def search_authors(self, query: str, *, limit=10, fields=None) -> dict:
        self.search_authors_calls.append(query)
        match = self.author_by_name_search.get(query)
        if match is None:
            return {"data": []}
        return {"data": [match]}


def test_execute_query_enriches_arxiv_categories_when_arxiv_client_supplied() -> None:
    openalex = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [
                    _author_payload(author_id="A1", name="Jane Researcher"),
                ],
            }
        ]
    )
    arxiv = _StubArxivClient(
        response_by_lastname={
            "Researcher": {
                "entries": [
                    {
                        "primary_category": "cs.CL",
                        "categories": ["cs.CL", "cs.LG"],
                    },
                    {
                        "primary_category": "cs.LG",
                        "categories": ["cs.LG"],
                    },
                ],
            }
        }
    )
    result = execute_query(
        query={"id": 1, "name": "q1", "topic_concepts": ["C1"]},
        client=openalex,
        arxiv_client=arxiv,
    )
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    # Categories should be deduped + ordered (primary categories first;
    # then any additional categories from any entry).
    assert "cs.CL" in candidate.arxiv_categories
    assert "cs.LG" in candidate.arxiv_categories


def test_execute_query_skips_arxiv_enrichment_when_lastname_empty() -> None:
    """A candidate with no name shouldn't trigger an arXiv call."""

    openalex = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [_author_payload(author_id="A1", name="")],
            }
        ]
    )
    arxiv = _StubArxivClient(response_by_lastname={})
    execute_query(
        query={"id": 1, "name": "q1", "topic_concepts": ["C1"]},
        client=openalex,
        arxiv_client=arxiv,
    )
    assert arxiv.calls == []


def test_execute_query_arxiv_enrichment_failure_logs_and_continues() -> None:
    """Per-candidate API failure shouldn't abort the broader acquisition."""

    class _RaisingArxivClient:
        def search(self, **_kwargs: Any) -> dict:
            raise RuntimeError("arxiv outage")

    openalex = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [
                    _author_payload(author_id="A1", name="Jane Researcher"),
                ],
            }
        ]
    )
    result = execute_query(
        query={"id": 1, "name": "q1"},
        client=openalex,
        arxiv_client=_RaisingArxivClient(),
    )
    # Acquisition continues; arxiv_categories stays empty.
    assert len(result.candidates) == 1
    assert result.candidates[0].arxiv_categories == []


def test_execute_query_reconciles_h_index_with_semantic_scholar() -> None:
    """When S2's h-index is materially higher than OpenAlex's, the
    canonical h_index gets bumped + s2_h_index records the raw S2
    value for diagnostic provenance."""

    openalex = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [
                    _author_payload(
                        author_id="A1",
                        name="Jane Researcher",
                        orcid="https://orcid.org/0000-0001-1111-1111",
                        h_index=10,
                    ),
                ],
            }
        ]
    )
    s2 = _StubS2Client(
        author_by_orcid={
            "0000-0001-1111-1111": {
                "hIndex": 18,  # 80% higher than OpenAlex
                "citationCount": 5000,
                "name": "Jane Researcher",
            }
        }
    )
    result = execute_query(
        query={"id": 1, "name": "q1"},
        client=openalex,
        semantic_scholar_client=s2,
    )
    candidate = result.candidates[0]
    assert candidate.s2_h_index == 18
    assert candidate.h_index == 18  # bumped


def test_execute_query_keeps_openalex_h_index_when_s2_delta_below_threshold() -> None:
    """Small deltas (< threshold) should not flip the canonical h_index
    — the threshold guard avoids noise-driven flips at the floor
    boundary that would destabilize per-run save eligibility."""

    openalex = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [
                    _author_payload(
                        author_id="A1",
                        name="Jane Researcher",
                        orcid="https://orcid.org/0000-0001-1111-1111",
                        h_index=15,
                    ),
                ],
            }
        ]
    )
    s2 = _StubS2Client(
        author_by_orcid={
            "0000-0001-1111-1111": {"hIndex": 16, "name": "Jane Researcher"}
        }
    )
    result = execute_query(
        query={"id": 1, "name": "q1"},
        client=openalex,
        semantic_scholar_client=s2,
    )
    candidate = result.candidates[0]
    # S2 raw value still recorded for diagnostics, but canonical
    # h_index stays at OpenAlex's 15 (< 20% delta threshold).
    assert candidate.s2_h_index == 16
    assert candidate.h_index == 15


def test_execute_query_falls_back_to_s2_name_search_when_orcid_missing() -> None:
    """Candidates without ORCID resolve via name search."""

    openalex = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [
                    _author_payload(
                        author_id="A1",
                        name="Anonymous Researcher",
                        orcid="",
                        h_index=5,
                    ),
                ],
            }
        ]
    )
    s2 = _StubS2Client(
        author_by_name_search={
            "Anonymous Researcher": {"hIndex": 12, "name": "Anonymous Researcher"}
        }
    )
    result = execute_query(
        query={"id": 1, "name": "q1"},
        client=openalex,
        semantic_scholar_client=s2,
    )
    candidate = result.candidates[0]
    assert candidate.s2_h_index == 12
    assert candidate.h_index == 12  # bumped (>20% delta from 5)
    assert s2.search_authors_calls == ["Anonymous Researcher"]


def test_execute_query_s2_failure_logs_and_continues() -> None:
    """Per-candidate S2 failure shouldn't abort acquisition."""

    class _RaisingS2Client:
        def get_author(self, *_args, **_kwargs):
            raise RuntimeError("s2 outage")

        def search_authors(self, *_args, **_kwargs):
            raise RuntimeError("s2 outage")

    openalex = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [
                    _author_payload(author_id="A1", name="Jane Researcher", h_index=10),
                ],
            }
        ]
    )
    result = execute_query(
        query={"id": 1, "name": "q1"},
        client=openalex,
        semantic_scholar_client=_RaisingS2Client(),
    )
    candidate = result.candidates[0]
    assert candidate.h_index == 10  # untouched
    assert candidate.s2_h_index == 0  # never set


def test_execute_query_pre_r1_path_unchanged_when_clients_omitted() -> None:
    """The default OpenAlex-only path is byte-identical to pre-R1."""

    openalex = _StubOpenAlexClient(
        [
            {
                "meta": {"next_cursor": ""},
                "results": [_author_payload(author_id="A1", name="X", h_index=10)],
            }
        ]
    )
    result = execute_query(query={"id": 1, "name": "q1"}, client=openalex)
    candidate = result.candidates[0]
    assert candidate.arxiv_categories == []
    assert candidate.s2_h_index == 0
    assert candidate.h_index == 10  # untouched
