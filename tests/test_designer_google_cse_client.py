"""Designer Slice 3 — Google Custom Search Engine client.

Pins the contract for :class:`designer.sources.google_cse.GoogleCSEClient`:

- Constructor reads ``GOOGLE_CSE_API_KEY`` and ``GOOGLE_CSE_ID``.
- Every request injects both as query parameters.
- ``search`` accepts ``site_filter`` and renders ``q="site:<host> <query>"``.
- 401/403 raise :class:`GoogleCSEAuthError`.
- The CSE result mappers (``cse_result_to_identity_key``,
  ``cse_result_to_display_name``, ``cse_result_thumbnail_url``)
  produce stable identities and best-effort display names.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from designer.sources.google_cse import (
    GOOGLE_CSE_API_BASE,
    PORTFOLIO_HOST_DOMAINS,
    GoogleCSEAuthError,
    GoogleCSEClient,
    cse_result_thumbnail_url,
    cse_result_to_display_name,
    cse_result_to_identity_key,
)


# ---------------------------------------------------------------------------
# Fixture data — recorded shape from CSE v1 API
# ---------------------------------------------------------------------------


_CSE_SEARCH_FIXTURE = {
    "items": [
        {
            "title": "Joe Designer — Cargo",
            "link": "https://joedesigner.cargo.site/",
            "displayLink": "joedesigner.cargo.site",
            "snippet": "Joe Designer is a senior product designer based in NYC.",
            "pagemap": {
                "metatags": [
                    {
                        "og:title": "Joe Designer — Senior Product Designer",
                        "og:image": "https://joedesigner.cargo.site/og.png",
                    }
                ],
                "cse_thumbnail": [
                    {
                        "src": "https://encrypted-tbn0.gstatic.com/images?q=tbn:joe123",
                        "width": "300",
                        "height": "200",
                    }
                ],
            },
        },
        {
            "title": "Sara Smith Portfolio | Squarespace",
            "link": "https://sarasmith.squarespace.com/work/branding-2024",
            "displayLink": "sarasmith.squarespace.com",
            "snippet": "Sara is a brand designer specializing in editorial work.",
            "pagemap": {
                "metatags": [{"og:title": "Sara Smith — Brand Designer"}],
                "cse_thumbnail": [
                    {
                        "src": "https://encrypted-tbn0.gstatic.com/images?q=tbn:sara456",
                        "width": "300",
                        "height": "200",
                    }
                ],
            },
        },
    ],
    "queries": {"request": [{"totalResults": "2"}]},
}


# ---------------------------------------------------------------------------
# Fake session helpers (mirrors test_designer_behance_client.py)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        body: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._text = text
        self.headers = headers or {}

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def json(self) -> Any:
        return self._body

    async def text(self) -> str:
        if self._text:
            return self._text
        return json.dumps(self._body) if self._body is not None else ""


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[_FakeResponse] = []

    def queue(self, response: _FakeResponse) -> None:
        self.responses.append(response)

    def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        self.calls.append((url, dict(params or {})))
        if not self.responses:
            return _FakeResponse(status=500, text="no fixture queued")
        return self.responses.pop(0)

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_session() -> _FakeSession:
    return _FakeSession()


@pytest.fixture
def client_factory(fake_session: _FakeSession, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "fake-cse-key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "fake-cse-id")

    def _build() -> GoogleCSEClient:
        return GoogleCSEClient(session_factory=lambda: fake_session)

    return _build


# ---------------------------------------------------------------------------
# Construction / auth
# ---------------------------------------------------------------------------


def test_constructor_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CSE_ID", "x")
    with pytest.raises(RuntimeError, match="GOOGLE_CSE_API_KEY"):
        GoogleCSEClient()


def test_constructor_raises_when_cse_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "x")
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_CSE_ID"):
        GoogleCSEClient()


def test_default_portfolio_hosts_carry_documented_baseline() -> None:
    """Slice-3 baseline portfolio-host domains; expansion is a follow-up."""

    assert "cargo.site" in PORTFOLIO_HOST_DOMAINS
    assert "squarespace.com" in PORTFOLIO_HOST_DOMAINS
    assert "format.com" in PORTFOLIO_HOST_DOMAINS
    assert "semplice.com" in PORTFOLIO_HOST_DOMAINS
    assert "awwwards.com" in PORTFOLIO_HOST_DOMAINS
    assert "siteinspire.com" in PORTFOLIO_HOST_DOMAINS


# ---------------------------------------------------------------------------
# Query plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_injects_key_and_cx(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=200, body=_CSE_SEARCH_FIXTURE))

    async with client_factory() as client:
        status, body = await client.search(query="design system")

    assert status == 200
    assert body is not None and len(body["items"]) == 2

    url, params = fake_session.calls[0]
    assert url == GOOGLE_CSE_API_BASE
    assert params["key"] == "fake-cse-key"
    assert params["cx"] == "fake-cse-id"
    assert params["q"] == "design system"
    assert params["start"] == "1"
    assert params["num"] == "10"


@pytest.mark.asyncio
async def test_search_with_site_filter_renders_site_query(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=200, body=_CSE_SEARCH_FIXTURE))

    async with client_factory() as client:
        await client.search(query="design system", site_filter="cargo.site")

    _, params = fake_session.calls[0]
    assert params["q"] == "site:cargo.site design system"


@pytest.mark.asyncio
async def test_401_raises_google_cse_auth_error(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=401, text='{"error":"invalid key"}'))

    async with client_factory() as client:
        with pytest.raises(GoogleCSEAuthError):
            await client.search(query="anything")


@pytest.mark.asyncio
async def test_validate_credentials_succeeds_on_200(
    fake_session: _FakeSession, client_factory: Any
) -> None:
    fake_session.queue(_FakeResponse(status=200, body=_CSE_SEARCH_FIXTURE))

    async with client_factory() as client:
        await client.validate_credentials()  # No raise.


# ---------------------------------------------------------------------------
# Result mappers
# ---------------------------------------------------------------------------


def test_identity_key_collapses_subpaths_within_same_first_segment() -> None:
    """Two URLs sharing host + first path segment collapse to one
    identity. Different first segments are kept distinct (intentional —
    a `/work` page vs a `/blog` page from the same site can be
    different surfaces; cross-source identity layer in Slice 8 handles
    cross-segment merge if needed)."""

    a = cse_result_to_identity_key("https://joedesigner.cargo.site/work/2024")
    b = cse_result_to_identity_key("https://joedesigner.cargo.site/work/2025")
    assert a == b == "cse:joedesigner.cargo.site/work"

    # Different first segments DO produce different keys.
    c = cse_result_to_identity_key("https://joedesigner.cargo.site/about")
    assert c != a
    assert c == "cse:joedesigner.cargo.site/about"


def test_identity_key_uses_host_for_root_pages() -> None:
    key = cse_result_to_identity_key("https://exampledesigner.com/")
    assert key == "cse:exampledesigner.com"


def test_display_name_prefers_og_title() -> None:
    item = _CSE_SEARCH_FIXTURE["items"][0]
    assert cse_result_to_display_name(item) == "Joe Designer — Senior Product Designer"


def test_display_name_strips_host_suffix_when_no_og_title() -> None:
    item = {"title": "Sara Smith Portfolio | Squarespace", "displayLink": "sarasmith.squarespace.com"}
    assert cse_result_to_display_name(item) == "Sara Smith Portfolio"


def test_display_name_falls_back_to_display_link_when_title_blank() -> None:
    item = {"title": "", "displayLink": "lastresort.cargo.site"}
    assert cse_result_to_display_name(item) == "lastresort.cargo.site"


def test_thumbnail_url_extracted_from_pagemap() -> None:
    item = _CSE_SEARCH_FIXTURE["items"][0]
    src = cse_result_thumbnail_url(item)
    assert src.startswith("https://encrypted-tbn0.gstatic.com/")


def test_thumbnail_url_empty_when_pagemap_missing() -> None:
    assert cse_result_thumbnail_url({"link": "https://x"}) == ""
