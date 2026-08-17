"""Designer Slice 2 — acquisition + dedup.

Pins the contract for the acquisition layer:

- ``acquire_behance_candidates`` calls the Behance client with each
  query's parameters, maps the response to ``DesignerSnippet``, and
  yields deduped snippets keyed by ``identity_key``.
- A per-query failure (5xx / network error) does NOT abort the run —
  the next query continues. The orchestrator owns the
  whole-run-failed decision.
- ``dedup_snippets`` is a pure function that keeps first occurrence.
- The ``behance:_unknown_`` identity-key fallback never escapes — a
  user record without a username gets dropped, not surfaced as a
  near-empty snippet.
"""

from __future__ import annotations

from typing import Any

import pytest

from designer.acquisition import (
    DEFAULT_MAX_USERS_PER_QUERY,
    acquire_behance_candidates,
    dedup_snippets,
)
from designer.schemas import (
    DesignerSearchQuery,
    DesignerSnippet,
    behance_user_to_snippet,
    behance_project_to_summary,
)
from designer.sources.behance import BehanceClient


# ---------------------------------------------------------------------------
# Fake client — records calls; returns canned response per call
# ---------------------------------------------------------------------------


class _FakeBehanceClient:
    """Stand-in for :class:`BehanceClient` for the acquisition tests."""

    def __init__(self, *, queue: list[tuple[int, dict | None]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._queue: list[tuple[int, dict | None]] = list(queue or [])

    async def search_users(
        self,
        *,
        query: str,
        country: str | None = None,
        sort: str = "appreciations",
        page: int = 1,
    ) -> tuple[int, dict | None]:
        self.calls.append(
            {"query": query, "country": country, "sort": sort, "page": page}
        )
        if not self._queue:
            return 200, {"users": []}
        return self._queue.pop(0)


def _fake_user(username: str, *, display_name: str = "", fields: tuple[str, ...] = ()) -> dict:
    return {
        "username": username,
        "display_name": display_name or f"{username.title()} Person",
        "occupation": "Designer",
        "city": "City",
        "country": "Country",
        "url": f"https://www.behance.net/{username}",
        "fields": list(fields) or ["UI/UX"],
        "stats": {"appreciations": 100, "followers": 10},
    }


# ---------------------------------------------------------------------------
# acquire_behance_candidates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquisition_yields_one_snippet_per_unique_user() -> None:
    queries = [
        DesignerSearchQuery(
            source="behance",
            query_text="design systems",
            sort="appreciations",
            capability_area_name="Design systems",
        )
    ]
    client = _FakeBehanceClient(
        queue=[
            (
                200,
                {
                    "users": [
                        _fake_user("designerone"),
                        _fake_user("designertwo"),
                    ]
                },
            )
        ]
    )

    seen: list[DesignerSnippet] = [
        s async for s in acquire_behance_candidates(queries, client=client)
    ]

    assert {s.identity_key for s in seen} == {
        "behance:designerone",
        "behance:designertwo",
    }


@pytest.mark.asyncio
async def test_acquisition_dedups_users_appearing_in_multiple_queries() -> None:
    queries = [
        DesignerSearchQuery(
            source="behance",
            query_text="systems",
            capability_area_name="Design systems",
        ),
        DesignerSearchQuery(
            source="behance",
            query_text="components",
            capability_area_name="Design systems",
        ),
    ]
    client = _FakeBehanceClient(
        queue=[
            (200, {"users": [_fake_user("alice"), _fake_user("bob")]}),
            (200, {"users": [_fake_user("bob"), _fake_user("carol")]}),  # bob duplicates
        ]
    )

    seen: list[DesignerSnippet] = [
        s async for s in acquire_behance_candidates(queries, client=client)
    ]
    assert [s.identity_key for s in seen] == [
        "behance:alice",
        "behance:bob",
        "behance:carol",
    ]


@pytest.mark.asyncio
async def test_acquisition_skips_query_on_failure() -> None:
    queries = [
        DesignerSearchQuery(source="behance", query_text="failing", capability_area_name="A"),
        DesignerSearchQuery(source="behance", query_text="working", capability_area_name="A"),
    ]
    client = _FakeBehanceClient(
        queue=[
            (500, None),  # first query 500s
            (200, {"users": [_fake_user("survivor")]}),  # second query fine
        ]
    )

    seen: list[DesignerSnippet] = [
        s async for s in acquire_behance_candidates(queries, client=client)
    ]
    assert [s.identity_key for s in seen] == ["behance:survivor"]


@pytest.mark.asyncio
async def test_acquisition_caps_users_per_query_at_default() -> None:
    """One query returning 50 users should not blast all 50 through —
    the per-query cap protects budget."""

    queries = [
        DesignerSearchQuery(source="behance", query_text="broad", capability_area_name="A")
    ]
    many_users = [_fake_user(f"user{i:03d}") for i in range(50)]
    client = _FakeBehanceClient(queue=[(200, {"users": many_users})])

    seen: list[DesignerSnippet] = [
        s async for s in acquire_behance_candidates(queries, client=client)
    ]
    assert len(seen) == DEFAULT_MAX_USERS_PER_QUERY


@pytest.mark.asyncio
async def test_acquisition_skips_user_without_username() -> None:
    """Users missing a username get dropped, not surfaced as
    `behance:_unknown_`."""

    queries = [
        DesignerSearchQuery(source="behance", query_text="x", capability_area_name="A")
    ]
    bad_user = _fake_user("ok")
    blank_user = {**_fake_user("noname"), "username": ""}
    client = _FakeBehanceClient(queue=[(200, {"users": [bad_user, blank_user]})])

    seen: list[DesignerSnippet] = [
        s async for s in acquire_behance_candidates(queries, client=client)
    ]
    assert [s.identity_key for s in seen] == ["behance:ok"]


@pytest.mark.asyncio
async def test_acquisition_threads_country_filter_into_client_call() -> None:
    queries = [
        DesignerSearchQuery(
            source="behance",
            query_text="x",
            capability_area_name="A",
            extra_filters={"country": "US"},
        )
    ]
    client = _FakeBehanceClient(queue=[(200, {"users": [_fake_user("u")]})])

    _ = [s async for s in acquire_behance_candidates(queries, client=client)]
    assert client.calls[0]["country"] == "US"


@pytest.mark.asyncio
async def test_acquisition_skips_non_behance_queries() -> None:
    """Slice 3 will add CSE queries; today the acquisition loop ignores
    them so a mixed-source brief from Slice 3+ doesn't accidentally
    invoke behance.search_users with CSE-style query text."""

    queries = [
        DesignerSearchQuery(
            source="google_cse",
            query_text="site:cargo.site design system",
            capability_area_name="A",
        )
    ]
    client = _FakeBehanceClient(queue=[])

    seen: list[DesignerSnippet] = [
        s async for s in acquire_behance_candidates(queries, client=client)
    ]
    assert seen == []
    assert client.calls == []


# ---------------------------------------------------------------------------
# Pure helpers — schema mappers + dedup
# ---------------------------------------------------------------------------


def test_behance_user_to_snippet_maps_fields_cleanly() -> None:
    snippet = behance_user_to_snippet(_fake_user("alice", fields=("Branding", "Logo")))
    assert snippet.identity_key == "behance:alice"
    assert snippet.profile_url == "https://www.behance.net/alice"
    assert snippet.fields == ("Branding", "Logo")
    assert snippet.location == "City, Country"


def test_behance_user_to_snippet_handles_missing_optional_fields() -> None:
    bare_user = {"username": "bare"}
    snippet = behance_user_to_snippet(bare_user)
    assert snippet.identity_key == "behance:bare"
    assert snippet.location == ""
    assert snippet.fields == ()
    assert snippet.profile_url == "https://www.behance.net/bare"


def test_behance_project_to_summary_picks_largest_cover_first() -> None:
    project = {
        "id": 42,
        "name": "X",
        "covers": {
            "115": "small.jpg",
            "404": "medium.jpg",
            "808": "large.jpg",
            "original": "original.jpg",
        },
        "stats": {"appreciations": 5, "views": 100},
        "fields": ["UI/UX"],
        "published_on": 1700000000,
    }
    summary = behance_project_to_summary(project)
    assert summary.cover_image_url == "original.jpg"
    assert summary.appreciation_count == 5
    assert summary.fields == ("UI/UX",)


def test_dedup_snippets_keeps_first_occurrence() -> None:
    a = behance_user_to_snippet(_fake_user("alice"))
    b = behance_user_to_snippet(_fake_user("bob"))
    a_again = behance_user_to_snippet(_fake_user("alice", display_name="Alice Two"))

    deduped = dedup_snippets([a, b, a_again])
    assert [s.identity_key for s in deduped] == ["behance:alice", "behance:bob"]
    # First occurrence's display name wins.
    assert deduped[0].display_name == a.display_name
