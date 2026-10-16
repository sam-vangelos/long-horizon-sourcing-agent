"""Exa source module S3 — acquisition coverage.

Offline: a fake client returns responses parsed from the recorded
fixtures in ``tests/fixtures/exa/`` and records the search kwargs it was
called with. Nothing here touches the network or the cheap tier.

Three properties carry the weight. ONE Exa call per query with contents
attached is the cost contract — a second ``/contents`` call would pay
again for pages the search already returned. Only transient failures are
absorbed, so a dead key or an exhausted budget stops the run instead of
producing eighty empty queries that look healthy. And a query's target
firm is search INTENT, written into a record only when the retrieved
page is the firm's own site.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiohttp
import pytest

from exa.acquisition import (
    AcquisitionOutcome,
    data_sufficiency,
    firm_is_grounded_by_page,
    run_query,
)
from exa.client import ExaAuthError, ExaCreditsError, ExaSearchResponse
from exa.cost_meter import ExaBudgetExhaustedError
from exa.extraction import ParsedProfile
from exa.schemas import ExaSearchQuery


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "exa"

_TALIA_QUOTE = "Talia Brennan is Chief of Staff at Cedarline Capital"
_OMAR_QUOTE = "Omar Vestergaard, Senior Associate"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


class _FakeClient:
    """Duck-typed stand-in for ``ExaClient``: one ``search``, one status."""

    def __init__(self, payload: dict | None = None, *, error: Exception | None = None):
        self.payload = payload or {}
        self.error = error
        self.calls: list[dict] = []

    async def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self.error is not None:
            raise self.error
        return ExaSearchResponse.from_api(self.payload)

    def status_line(self) -> str:
        return "Exa spend: $0.0160 of $5.00 cap | 1 calls"


def _query(**overrides) -> ExaSearchQuery:
    payload = dict(
        id=3,
        name="people_search · Northwind Capital",
        lane="people_search",
        firm="Northwind Capital",
        query="Chief of Staff at Northwind Capital SF Bay Area",
        category="people",
        include_domains=["linkedin.com"],
        num_results=10,
    )
    payload.update(overrides)
    return ExaSearchQuery(**payload)


def _team_query(**overrides) -> ExaSearchQuery:
    payload = dict(
        lane="team_page",
        firm="Cedarline Capital",
        category=None,
        include_domains=["cedarlinecapital.com"],
        query="Cedarline Capital team page investors",
    )
    payload.update(overrides)
    return _query(**payload)


def _run(query, client, **kwargs) -> AcquisitionOutcome:
    return asyncio.run(run_query(query, client=client, **kwargs))


# ---------------------------------------------------------------------------
# people_search
# ---------------------------------------------------------------------------


def test_people_search_builds_candidates_with_evidence_urls() -> None:
    client = _FakeClient(_fixture("search_people_ok.json"))

    outcome = _run(_query(), client)

    assert outcome.query_id == 3
    assert outcome.lane == "people_search"
    assert outcome.result_count == 3
    assert outcome.exa_calls == 1
    assert outcome.notes == ""

    # The riverstonepartners.com hit is a real page but not a profile,
    # so it is a drop with a reason, not a silent loss.
    assert outcome.dropped == {"not_profile": 1}

    assert [c.name for c in outcome.candidates] == ["Dana Okafor", "Priya Raghunathan"]

    dana = outcome.candidates[0]
    assert dana.current_title == "Chief of Staff to the CEO"
    assert dana.current_firm == "Northwind Capital"
    assert dana.identity_key == "linkedin.com/in/dana-okafor-chiefofstaff"
    assert dana.source_query_id == 3
    assert dana.source_lane == "people_search"
    assert dana.evidence_urls() == [
        "https://www.linkedin.com/in/dana-okafor-chiefofstaff"
    ]
    assert dana.evidence[0].highlights  # the judge gets the retrieved text
    assert dana.verify_before_outreach is True

    # A profile whose page title names a different firm keeps that firm.
    assert outcome.candidates[1].current_firm == "Alder Peak"


def test_archetype_lane_takes_the_people_search_path() -> None:
    """S8b: an archetype query returns profiles, so it is parsed as profiles.

    The lane differs from ``people_search`` only in how its text was
    written — a career shape instead of a firm × title string — and both
    filter to ``linkedin.com``. Routing it to the extractor instead
    would pay for an LLM call per profile page to re-derive structure
    the render already carries, and would put a firm-hint on a query
    that has no firm.
    """

    client = _FakeClient(_fixture("search_people_ok.json"))

    def _never(*args, **kwargs):
        raise AssertionError("the archetype lane must not call the extractor")

    outcome = _run(
        _query(
            lane="archetype",
            name="archetype · a01-mbb-to-vc",
            firm="",
            query="Former McKinsey consultants now investing in the Bay Area",
            num_results=25,
        ),
        client,
        extractor=_never,
    )

    assert outcome.lane == "archetype"
    assert [c.name for c in outcome.candidates] == ["Dana Okafor", "Priya Raghunathan"]
    assert all(c.source_lane == "archetype" for c in outcome.candidates)

    # The non-profile hit is dropped with a reason, exactly as it is on
    # the people_search lane — not handed to the extractor.
    assert outcome.dropped == {"not_profile": 1}

    # The query carries no firm, so no candidate inherits one it did not
    # earn: Dana's firm comes off her own profile render.
    assert outcome.candidates[0].current_firm == "Northwind Capital"
    assert outcome.candidates[1].current_firm == "Alder Peak"

    # And the author's result count reaches the API unchanged.
    assert client.calls[0]["num_results"] == 25


def test_people_search_one_exa_call_with_contents() -> None:
    client = _FakeClient(_fixture("search_people_ok.json"))

    _run(_query(), client)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["query"] == "Chief of Staff at Northwind Capital SF Bay Area"
    assert call["category"] == "people"
    assert call["num_results"] == 10
    assert call["include_domains"] == ["linkedin.com"]
    assert call["contents"] == {
        "highlights": {
            "query": "Chief of Staff at Northwind Capital SF Bay Area",
            "maxCharacters": 400,
        },
        "text": {"maxCharacters": 4000},
    }

    # An empty domain list means "no filter", not "filter on nothing".
    bare = _FakeClient(_fixture("search_people_ok.json"))
    _run(_query(include_domains=[]), bare)
    assert bare.calls[0]["include_domains"] is None


def test_data_sufficiency_bands() -> None:
    assert data_sufficiency("Principal", "Benchmark") == "sufficient"
    assert data_sufficiency("Principal", "") == "minimal"
    assert data_sufficiency("", "Benchmark") == "minimal"
    assert data_sufficiency("", "") == "insufficient"

    payload = {
        "results": [
            {
                "url": "https://www.linkedin.com/in/full-record",
                "title": "Ada Fullerton - Principal - Northwind Capital | LinkedIn",
                "text": "",
                "highlights": [],
            },
            {
                "url": "https://www.linkedin.com/in/name-only",
                "title": "Basil Ferro | LinkedIn",
                "text": "",
                "highlights": [],
            },
        ]
    }

    outcome = _run(_query(), _FakeClient(payload))

    bands = {c.name: c.data_sufficiency for c in outcome.candidates}
    # A page title that names no firm leaves the record saying so. The
    # query's firm is what the search was LOOKING for, and a LinkedIn
    # result is not the firm's own page, so it stays out of the record.
    assert bands == {
        "Ada Fullerton": "sufficient",
        "Basil Ferro": "insufficient",
    }


def test_query_firm_not_assumed_off_domain() -> None:
    payload = {
        "results": [
            {
                "url": "https://www.linkedin.com/in/name-only",
                "title": "Basil Ferro | LinkedIn",
                "text": "",
                "highlights": [],
            }
        ]
    }

    outcome = _run(_query(), _FakeClient(payload))

    assert outcome.candidates[0].current_firm == ""
    # The identity key falls back to the handle either way, so nothing
    # about dedupe depends on the guess we declined to make.
    assert outcome.candidates[0].identity_key == "linkedin.com/in/name-only"


def test_query_firm_grounded_on_firm_domain_page() -> None:
    """The firm's own site IS evidence that its people work there."""

    assert firm_is_grounded_by_page(
        firm="Cedarline Capital",
        evidence_url="https://www.cedarlinecapital.com/team",
        include_domains=[],
    )
    assert firm_is_grounded_by_page(
        firm="Wexford Peak Partners",
        evidence_url="https://wexfordpeak.com/about/people",
        include_domains=["wexfordpeak.com"],
    )
    # linkedin.com is where everybody's profile lives, so a domain
    # filter on it confirms nothing about any one firm.
    assert not firm_is_grounded_by_page(
        firm="Northwind Capital",
        evidence_url="https://www.linkedin.com/in/someone",
        include_domains=["linkedin.com"],
    )

    def _extractor(evidence, *, firm_hint, usage_context=None):
        if "cedarline" not in evidence.url:
            return []
        return [
            ParsedProfile(
                name="Omar Vestergaard",
                title="Senior Associate",
                firm="",
                evidence_quote=_OMAR_QUOTE,
            )
        ]

    payload = _fixture("search_team_page_ok.json")
    outcome = _run(_team_query(), _FakeClient(payload), extractor=_extractor)

    assert outcome.candidates[0].current_firm == "Cedarline Capital"
    # Title off the page, firm off the page's host — both evidenced.
    assert outcome.candidates[0].current_title == "Senior Associate"
    assert outcome.candidates[0].data_sufficiency == "sufficient"


def test_candidates_sharing_identity_key_merge_evidence() -> None:
    payload = {
        "results": [
            {
                "url": "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
                "title": "Dana Okafor - Chief of Staff - Northwind Capital | LinkedIn",
                "text": "Chief of Staff at Northwind Capital.",
                "highlights": ["Chief of Staff at Northwind Capital."],
            },
            {
                "url": "https://linkedin.com/in/Dana-Okafor-ChiefOfStaff/",
                "title": "Dana Okafor - Operating Partner - Northwind | LinkedIn",
                "text": "Second retrieved view of the same profile.",
                "highlights": [],
            },
        ]
    }

    outcome = _run(_query(), _FakeClient(payload))

    assert len(outcome.candidates) == 1
    merged = outcome.candidates[0]
    # A field both reads filled keeps the first value...
    assert merged.current_title == "Chief of Staff"
    assert merged.current_firm == "Northwind Capital"
    # ...and both pages stay attached as evidence.
    assert merged.evidence_urls() == [
        "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
        "https://linkedin.com/in/Dana-Okafor-ChiefOfStaff/",
    ]


def test_merge_fills_blank_title_from_later_read() -> None:
    """A blank is an absence, not a finding, so a later read may fill it."""

    payload = {
        "results": [
            {
                "url": "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
                "title": "Dana Okafor | LinkedIn",
                "text": "",
                "highlights": [],
            },
            {
                "url": "https://linkedin.com/in/Dana-Okafor-ChiefOfStaff/",
                "title": "Dana Okafor - Chief of Staff - Northwind Capital | LinkedIn",
                "text": "",
                "highlights": [],
            },
        ]
    }

    outcome = _run(_query(), _FakeClient(payload))

    assert len(outcome.candidates) == 1
    merged = outcome.candidates[0]
    assert merged.current_title == "Chief of Staff"
    assert merged.current_firm == "Northwind Capital"
    assert merged.data_sufficiency == "sufficient"


def test_merge_same_url_merges_highlights() -> None:
    """Two reads of one page carry two quotes; neither is thrown away."""

    payload = {
        "results": [
            {
                "url": "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
                "title": "Dana Okafor - Chief of Staff - Northwind Capital | LinkedIn",
                "text": "",
                "highlights": ["first retrieved highlight"],
            },
            {
                "url": "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
                "title": "Dana Okafor - Chief of Staff - Northwind Capital | LinkedIn",
                "text": "",
                "highlights": ["second retrieved highlight"],
            },
        ]
    }

    outcome = _run(_query(), _FakeClient(payload))

    assert len(outcome.candidates) == 1
    evidence = outcome.candidates[0].evidence
    assert len(evidence) == 1
    assert evidence[0].highlights == [
        "first retrieved highlight",
        "second retrieved highlight",
    ]


# ---------------------------------------------------------------------------
# Namesake guard
# ---------------------------------------------------------------------------


def test_namesake_exclude_terms_drop_results() -> None:
    payload = {
        "results": [
            {
                "url": "https://www.linkedin.com/in/quinn-avila-accel",
                "title": "Quinn Avila - Partner - Accel | LinkedIn",
                "text": "Partner at Accel, early-stage team, San Francisco.",
                "highlights": [],
            },
            {
                "url": "https://www.linkedin.com/in/wade-kessler-akkr",
                "title": "Wade Kessler - Senior Associate - Accel-KKR | LinkedIn",
                "text": "Senior Associate at a technology-focused PE firm.",
                "highlights": [],
            },
        ]
    }

    outcome = _run(_query(firm="Accel", exclude_terms=["Accel-KKR"]), _FakeClient(payload))

    assert [c.name for c in outcome.candidates] == ["Quinn Avila"]
    assert outcome.dropped == {"namesake": 1}
    assert outcome.candidates[0].current_firm == "Accel"


def test_namesake_matches_across_separators() -> None:
    """One firm written three ways is still one firm."""

    for written in ("Accel-KKR", "Accel KKR", "accel_kkr"):
        payload = {
            "results": [
                {
                    "url": "https://www.linkedin.com/in/wade-kessler",
                    "title": f"Wade Kessler - Senior Associate - {written} | LinkedIn",
                    "text": "",
                    "highlights": [],
                }
            ]
        }
        outcome = _run(
            _query(firm="Accel", exclude_terms=["Accel-KKR"]), _FakeClient(payload)
        )
        assert outcome.candidates == [], written
        assert outcome.dropped == {"namesake": 1}, written

    # And the URL is scanned too, for a page whose title hides the firm.
    payload = {
        "results": [
            {
                "url": "https://www.accel-kkr.com/team/wade-kessler",
                "title": "Wade Kessler | Team",
                "text": "",
                "highlights": [],
            }
        ]
    }
    outcome = _run(
        _team_query(firm="Accel", exclude_terms=["Accel KKR"], include_domains=[]),
        _FakeClient(payload),
    )
    assert outcome.dropped == {"namesake": 1}


def test_namesake_does_not_drop_on_prose_mention() -> None:
    """A co-investor mention is not employment.

    Page text is out of scope for the guard: a genuine Accel employee
    whose profile mentions a deal alongside Accel-KKR is exactly the
    person the campaign wants.
    """

    payload = {
        "results": [
            {
                "url": "https://www.linkedin.com/in/nadia-ferrand",
                "title": "Nadia Ferrand - Vice President - Accel | LinkedIn",
                "text": "Vice President at Accel. Co-invested with Accel-KKR in 2024.",
                "highlights": ["Co-invested alongside Accel-KKR."],
            }
        ]
    }

    outcome = _run(
        _query(firm="Accel", exclude_terms=["Accel-KKR"]), _FakeClient(payload)
    )

    assert [c.name for c in outcome.candidates] == ["Nadia Ferrand"]
    assert outcome.dropped == {}


# ---------------------------------------------------------------------------
# team_page / operating_signal
# ---------------------------------------------------------------------------


def test_team_page_runs_extractor_per_page() -> None:
    seen: list[str] = []

    def _extractor(evidence, *, firm_hint, usage_context=None):
        seen.append(evidence.url)
        assert firm_hint == "Cedarline Capital"
        assert usage_context["lane_id"] == "team_page"
        assert usage_context["string_id"] == 3
        if "cedarline" not in evidence.url:
            return []
        return [
            ParsedProfile(
                name="Talia Brennan",
                title="Chief of Staff",
                firm="Cedarline Capital",
                evidence_quote=_TALIA_QUOTE,
            ),
            ParsedProfile(
                name="Omar Vestergaard",
                title="Senior Associate",
                firm="",
                evidence_quote=_OMAR_QUOTE,
            ),
        ]

    client = _FakeClient(_fixture("search_team_page_ok.json"))
    outcome = _run(_team_query(), client, extractor=_extractor)

    assert seen == [
        "https://www.cedarlinecapital.com/team",
        "https://www.wexfordpeak.com/about/people",
    ]
    assert [c.name for c in outcome.candidates] == ["Talia Brennan", "Omar Vestergaard"]
    # A page that yields nobody is a counted outcome, not a silent zero.
    assert outcome.dropped == {"extraction_empty": 1}
    # The person the page gave no firm for gets it from the page's own
    # host, not from the query's intent.
    assert outcome.candidates[1].current_firm == "Cedarline Capital"
    assert outcome.candidates[0].evidence_urls() == [
        "https://www.cedarlinecapital.com/team"
    ]


def test_assembly_recheck_uses_full_grounding() -> None:
    """An injected extractor cannot smuggle a title past assembly.

    The shipped extractor grounds its own output, but the extractor is a
    parameter and the record is born here — so the invariant the record
    publishes is enforced here.
    """

    def _extractor(evidence, *, firm_hint, usage_context=None):
        return [
            ParsedProfile(
                name="Talia Brennan",
                title="Managing Director",
                firm="FakeCo",
                location="Reykjavik, Iceland",
                linkedin_url="https://www.linkedin.com/in/talia-brennan",
                evidence_quote=_TALIA_QUOTE,
            ),
            ParsedProfile(
                name="Ghost Person",
                title="Managing Director",
                firm="Cedarline Capital",
                evidence_quote=_TALIA_QUOTE,
            ),
        ]

    payload = _fixture("search_team_page_ok.json")
    payload["results"] = payload["results"][:1]
    # A firm-less query, so nothing can be filled from the page host and
    # the blanking is visible on its own.
    outcome = _run(
        _team_query(firm="", include_domains=[]),
        _FakeClient(payload),
        extractor=_extractor,
    )

    assert [c.name for c in outcome.candidates] == ["Talia Brennan"]
    assert outcome.dropped == {"fabrication_guard": 1}

    talia = outcome.candidates[0]
    assert talia.current_title == ""
    assert talia.current_firm == ""
    assert talia.location == ""
    assert talia.linkedin_url == ""
    assert talia.data_sufficiency == "insufficient"


def test_grounded_quote_is_retained_on_candidate_evidence() -> None:
    """The sentence that earned the person their place reaches the judge.

    ``to_evidence_text`` renders highlights and suppresses the excerpt
    whenever any highlight exists (`exa/schemas.py:292`), so a validated
    quote that was not carried onto the record would be invisible to the
    model that decides.
    """

    def _extractor(evidence, *, firm_hint, usage_context=None):
        return [
            ParsedProfile(
                name="Talia Brennan",
                title="Chief of Staff",
                firm="Cedarline Capital",
                evidence_quote=_TALIA_QUOTE,
            )
        ]

    payload = _fixture("search_team_page_ok.json")
    payload["results"] = payload["results"][:1]
    outcome = _run(_team_query(), _FakeClient(payload), extractor=_extractor)

    evidence = outcome.candidates[0].evidence[0]
    assert evidence.highlights[0] == _TALIA_QUOTE
    # The page's own highlights survive underneath it.
    assert len(evidence.highlights) == 3
    assert _TALIA_QUOTE in outcome.candidates[0].to_evidence_text()


def test_quote_not_leaked_across_candidates_on_same_page() -> None:
    """Five people off one team page must not share five quotes."""

    def _extractor(evidence, *, firm_hint, usage_context=None):
        return [
            ParsedProfile(
                name="Talia Brennan",
                title="Chief of Staff",
                evidence_quote=_TALIA_QUOTE,
            ),
            ParsedProfile(
                name="Omar Vestergaard",
                title="Senior Associate",
                evidence_quote=_OMAR_QUOTE,
            ),
        ]

    payload = _fixture("search_team_page_ok.json")
    payload["results"] = payload["results"][:1]
    outcome = _run(_team_query(), _FakeClient(payload), extractor=_extractor)

    talia, omar = outcome.candidates
    assert talia.evidence[0].highlights[0] == _TALIA_QUOTE
    assert omar.evidence[0].highlights[0] == _OMAR_QUOTE
    assert _OMAR_QUOTE not in talia.evidence[0].highlights
    assert _TALIA_QUOTE not in omar.evidence[0].highlights
    # Two records, two evidence objects, one underlying page.
    assert talia.evidence[0] is not omar.evidence[0]
    assert talia.evidence[0].url == omar.evidence[0].url


# ---------------------------------------------------------------------------
# Failure modes and the console
# ---------------------------------------------------------------------------


def test_budget_exhausted_propagates() -> None:
    client = _FakeClient(error=ExaBudgetExhaustedError("cap reached"))

    with pytest.raises(ExaBudgetExhaustedError):
        _run(_query(), client)


def test_auth_and_credits_errors_propagate() -> None:
    """A dead key and an empty account do not fix themselves on query two."""

    for error in (
        ExaAuthError("401 INVALID_API_KEY"),
        ExaCreditsError("402 OUT_OF_CREDITS"),
    ):
        with pytest.raises(type(error)):
            _run(_query(), _FakeClient(error=error))


def test_validation_error_propagates() -> None:
    """A malformed request is a compiler bug every query carries."""

    client = _FakeClient(error=ValueError("Exa numResults must be between 1 and 100"))

    with pytest.raises(ValueError):
        _run(_query(), client)


def test_absorbed_transport_error_counts_the_dispatch() -> None:
    client = _FakeClient(error=aiohttp.ClientError("connection reset"))

    outcome = _run(_query(), client)

    assert outcome.candidates == []
    assert outcome.result_count == 0
    # The request went out and may have been billed. The meter is the
    # authority on spend; this is the call ledger.
    assert outcome.exa_calls == 1
    assert "connection reset" in outcome.notes

    timeout = _run(_query(), _FakeClient(error=asyncio.TimeoutError()))
    assert timeout.exa_calls == 1


def test_client_error_returns_empty_outcome_with_note() -> None:
    client = _FakeClient(error=RuntimeError("Exa API error 500:\nupstream\nfailure"))

    outcome = _run(_query(), client)

    assert outcome.candidates == []
    assert outcome.result_count == 0
    assert outcome.exa_calls == 1
    assert "upstream failure" in outcome.notes
    assert "\n" not in outcome.notes


def test_console_line_is_single_line_per_query(capsys) -> None:
    client = _FakeClient(_fixture("search_people_ok.json"))

    _run(_query(), client)

    printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(printed) == 1
    assert printed[0] == (
        "    [exa] q3 people_search · Northwind Capital: 3 results → "
        "2 candidates (not_profile 1) | Exa spend: $0.0160 of $5.00 cap | 1 calls"
    )

    # The error path is one line too, and the failing client is the one
    # whose output is most likely to be multi-line.
    _run(_query(), _FakeClient(error=RuntimeError("boom\nboom")))
    error_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    assert len(error_lines) == 1
    assert error_lines[0].startswith(
        "    [exa] q3 people_search · Northwind Capital: error —"
    )


def test_no_drops_reads_as_no_drops(capsys) -> None:
    payload = {
        "results": [
            {
                "url": "https://www.linkedin.com/in/clean-hit",
                "title": "Ada Fullerton - Principal - Northwind Capital | LinkedIn",
                "text": "",
                "highlights": [],
            }
        ]
    }

    outcome = _run(_query(), _FakeClient(payload))

    assert outcome.dropped == {}
    assert "(no drops)" in capsys.readouterr().out
