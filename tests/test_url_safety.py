"""Tests for shared/url_safety.py — SSRF protection for candidate-controlled URLs.

All DNS resolution is mocked for determinism. No real network calls.
Integration tests stub aiohttp and exercise _crawl_website_and_papers.

Run with: python -m pytest tests/test_url_safety.py -v
"""

import sys
import types
import asyncio
import socket
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

import shared.llm_clients as _llm
from shared.url_safety import check_url, check_ip

# ---------------------------------------------------------------------------
# Stub heavy deps so we can import github.enricher without aiohttp installed
# ---------------------------------------------------------------------------

for _mod_name in ("aiohttp", "certifi", "github.client", "shared.contact_discovery"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

_aiohttp = sys.modules["aiohttp"]
_aiohttp.ClientSession = MagicMock
_aiohttp.TCPConnector = MagicMock
_aiohttp.ClientTimeout = MagicMock

_certifi = sys.modules["certifi"]
_certifi.where = MagicMock(return_value="")

_client = sys.modules["github.client"]
_client.GitHubClient = MagicMock

_contact = sys.modules["shared.contact_discovery"]
_contact.discover_contacts = MagicMock()

_llm.cheap_llm = MagicMock()

from github.schemas import GitHubUser, GitHubCandidate
from github.enricher import GitHubEnricher


# ---------------------------------------------------------------------------
# Helper — mock async DNS resolution
# ---------------------------------------------------------------------------

def _mock_getaddrinfo(ip: str):
    """Return an AsyncMock that resolves any hostname to the given IP."""
    async def _getaddrinfo(host, port, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, port or 80))]
    return _getaddrinfo


def _mock_getaddrinfo_fail():
    """Return an AsyncMock that raises gaierror (DNS failure)."""
    async def _getaddrinfo(host, port, **kw):
        raise socket.gaierror("Name or service not known")
    return _getaddrinfo


def _patch_dns(ip: str):
    """Patch asyncio.get_running_loop to return a loop with mocked getaddrinfo."""
    mock_loop = MagicMock()
    mock_loop.getaddrinfo = _mock_getaddrinfo(ip)
    return patch("asyncio.get_running_loop", return_value=mock_loop)


def _patch_dns_fail():
    mock_loop = MagicMock()
    mock_loop.getaddrinfo = _mock_getaddrinfo_fail()
    return patch("asyncio.get_running_loop", return_value=mock_loop)


# ---------------------------------------------------------------------------
# check_ip (synchronous)
# ---------------------------------------------------------------------------

class TestCheckIp:
    def test_public_ip_allowed(self):
        safe, _ = check_ip("93.184.216.34")
        assert safe

    def test_loopback_blocked(self):
        safe, reason = check_ip("127.0.0.1")
        assert not safe
        assert "loopback" in reason

    def test_private_blocked(self):
        safe, reason = check_ip("10.0.0.1")
        assert not safe
        assert "private" in reason

    def test_invalid_ip(self):
        safe, reason = check_ip("not-an-ip")
        assert not safe
        assert "invalid" in reason


# ---------------------------------------------------------------------------
# check_url — scheme / parse errors (no DNS needed)
# ---------------------------------------------------------------------------

class TestCheckUrlParsing:
    def test_empty_blocked(self):
        safe, reason = asyncio.run(check_url(""))
        assert not safe
        assert "empty" in reason

    def test_file_scheme_blocked(self):
        safe, reason = asyncio.run(check_url("file:///etc/passwd"))
        assert not safe
        assert "scheme" in reason

    def test_ftp_scheme_blocked(self):
        safe, reason = asyncio.run(check_url("ftp://evil.com/payload"))
        assert not safe
        assert "scheme" in reason

    def test_no_scheme_blocked(self):
        safe, reason = asyncio.run(check_url("not-a-url"))
        assert not safe
        assert "scheme" in reason


# ---------------------------------------------------------------------------
# check_url — hostname blocklist (no DNS needed, blocked before resolution)
# ---------------------------------------------------------------------------

class TestCheckUrlHostnameBlocklist:
    def test_localhost_blocked(self):
        safe, reason = asyncio.run(check_url("http://localhost:8080/"))
        assert not safe
        assert "blocked hostname" in reason

    def test_metadata_google_blocked(self):
        safe, reason = asyncio.run(check_url("http://metadata.google.internal/"))
        assert not safe
        assert "blocked hostname" in reason

    def test_metadata_goog_blocked(self):
        safe, reason = asyncio.run(check_url("http://metadata.goog/"))
        assert not safe
        assert "blocked hostname" in reason


# ---------------------------------------------------------------------------
# check_url — raw IP literals (no DNS needed)
# ---------------------------------------------------------------------------

class TestCheckUrlRawIp:
    def test_loopback_ip_blocked(self):
        safe, reason = asyncio.run(check_url("http://127.0.0.1/"))
        assert not safe
        assert "loopback" in reason

    def test_loopback_alt_blocked(self):
        safe, reason = asyncio.run(check_url("http://127.0.0.2/"))
        assert not safe
        assert "loopback" in reason

    def test_private_10_blocked(self):
        safe, reason = asyncio.run(check_url("http://10.0.0.1/admin"))
        assert not safe
        assert "private" in reason

    def test_private_172_blocked(self):
        safe, reason = asyncio.run(check_url("http://172.16.0.1/"))
        assert not safe
        assert "private" in reason

    def test_private_192_blocked(self):
        safe, reason = asyncio.run(check_url("http://192.168.1.1/"))
        assert not safe
        assert "private" in reason

    def test_link_local_blocked(self):
        safe, reason = asyncio.run(check_url("http://169.254.169.254/latest/meta-data/"))
        assert not safe
        assert "169.254.169.254" in reason

    def test_ipv6_loopback_blocked(self):
        safe, reason = asyncio.run(check_url("http://[::1]/"))
        assert not safe
        assert "loopback" in reason

    def test_zero_ip_blocked(self):
        safe, reason = asyncio.run(check_url("http://0.0.0.0/"))
        assert not safe

    def test_multicast_blocked(self):
        safe, reason = asyncio.run(check_url("http://224.0.0.1/"))
        assert not safe
        assert "multicast" in reason

    def test_reserved_blocked(self):
        safe, reason = asyncio.run(check_url("http://240.0.0.1/"))
        assert not safe
        assert "240.0.0.1" in reason


# ---------------------------------------------------------------------------
# check_url — DNS resolution (mocked)
# ---------------------------------------------------------------------------

class TestCheckUrlDns:
    def test_public_https_allowed(self):
        with _patch_dns("93.184.216.34"):
            safe, _ = asyncio.run(check_url("https://example.com"))
        assert safe

    def test_public_http_allowed(self):
        with _patch_dns("93.184.216.34"):
            safe, _ = asyncio.run(check_url("http://example.com"))
        assert safe

    def test_arxiv_allowed(self):
        with _patch_dns("151.101.1.42"):
            safe, _ = asyncio.run(check_url("https://arxiv.org/abs/2301.00001"))
        assert safe

    def test_dns_to_private_blocked(self):
        with _patch_dns("127.0.0.1"):
            safe, reason = asyncio.run(check_url("http://evil.com/"))
        assert not safe
        assert "loopback" in reason

    def test_dns_to_link_local_blocked(self):
        with _patch_dns("169.254.169.254"):
            safe, reason = asyncio.run(check_url("http://sneaky.com/"))
        assert not safe
        assert "169.254.169.254" in reason

    def test_dns_resolution_failure(self):
        with _patch_dns_fail():
            safe, reason = asyncio.run(check_url("http://nonexistent.invalid/"))
        assert not safe
        assert "DNS resolution failed" in reason


# ---------------------------------------------------------------------------
# Helpers — mock aiohttp for enricher integration tests
# ---------------------------------------------------------------------------

def _make_candidate(**overrides) -> GitHubCandidate:
    """Minimal GitHubCandidate for testing."""
    user = GitHubUser(username="testuser")
    c = GitHubCandidate(user=user)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def _make_enricher() -> GitHubEnricher:
    """GitHubEnricher with a mocked client."""
    client = MagicMock()
    return GitHubEnricher(client)


def _mock_response(status, headers=None, body=""):
    """Create a mock aiohttp response usable as an async context manager."""
    resp = MagicMock()
    resp.status = status
    resp.headers = headers or {}
    resp.text = AsyncMock(return_value=body)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _mock_session(responses):
    """Create a mock aiohttp.ClientSession returning responses in order.

    Returns (session_context_manager, recorded_calls) where
    recorded_calls is a list of (url, kwargs) tuples.
    """
    session = MagicMock()
    call_idx = {"i": 0}
    recorded_calls = []

    def _get(url, **kw):
        recorded_calls.append((url, kw))
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx < len(responses):
            return responses[idx]
        return _mock_response(404)

    session.get = _get
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, recorded_calls


# ---------------------------------------------------------------------------
# Integration: redirect safety in _crawl_website_and_papers
# ---------------------------------------------------------------------------

def _patch_aiohttp(session_ctx):
    """Patch the aiohttp stub so ClientSession() returns our mock session."""
    aiohttp_mod = sys.modules["aiohttp"]
    return patch.object(aiohttp_mod, "ClientSession", return_value=session_ctx)


class TestRedirectIntegration:
    def test_redirect_to_private_blocked(self):
        """302 redirect to http://10.0.0.1 → website_text not populated."""
        candidate = _make_candidate()
        candidate.user.blog = "https://myblog.com"
        enricher = _make_enricher()

        redirect_resp = _mock_response(302, {"Location": "http://10.0.0.1/admin"})
        session_ctx, calls = _mock_session([redirect_resp])

        async def _run():
            with _patch_dns("93.184.216.34"):
                with _patch_aiohttp(session_ctx):
                    await enricher._crawl_website_and_papers(candidate)

        asyncio.run(_run())
        assert candidate.website_text == ""

    def test_relative_redirect_resolved(self):
        """Relative Location: /page → resolved with urljoin, fetch continues."""
        candidate = _make_candidate()
        candidate.user.blog = "https://myblog.com"
        enricher = _make_enricher()

        redirect_resp = _mock_response(302, {"Location": "/page"})
        ok_resp = _mock_response(200, {"Content-Type": "text/html"}, "<p>Hello world</p>")
        session_ctx, calls = _mock_session([redirect_resp, ok_resp])

        async def _run():
            with _patch_dns("93.184.216.34"):
                with _patch_aiohttp(session_ctx):
                    await enricher._crawl_website_and_papers(candidate)

        asyncio.run(_run())
        assert candidate.website_text != ""
        # Verify the second request used the resolved absolute URL
        assert len(calls) >= 2
        assert calls[1][0] == "https://myblog.com/page"


# ---------------------------------------------------------------------------
# Integration: arXiv old-style ID extraction
# ---------------------------------------------------------------------------

class TestArxivIdExtraction:
    def _run_arxiv_fetch(self, paper_link: str):
        """Run _crawl_website_and_papers and return (recorded_calls, candidate)."""
        candidate = _make_candidate()
        candidate.paper_links = [paper_link]
        enricher = _make_enricher()

        arxiv_xml = """<?xml version="1.0"?>
        <feed><entry><title>Test Paper Title</title></entry></feed>"""
        arxiv_resp = _mock_response(200, {"Content-Type": "application/xml"}, arxiv_xml)
        session_ctx, calls = _mock_session([arxiv_resp])

        async def _run():
            with _patch_dns("93.184.216.34"):
                with _patch_aiohttp(session_ctx):
                    await enricher._crawl_website_and_papers(candidate)

        asyncio.run(_run())
        return calls, candidate

    def test_old_style_slash(self):
        """hep-th/9901001 is not dropped by extraction."""
        calls, candidate = self._run_arxiv_fetch("https://arxiv.org/abs/hep-th/9901001")
        assert len(calls) == 1
        assert calls[0][1]["params"]["id_list"] == "hep-th/9901001"
        assert "Test Paper Title" in candidate.paper_titles

    def test_dotted_archive_prefix(self):
        """math.GT/0309136 is not dropped by extraction."""
        calls, candidate = self._run_arxiv_fetch("https://arxiv.org/abs/math.GT/0309136")
        assert len(calls) == 1
        assert calls[0][1]["params"]["id_list"] == "math.GT/0309136"
        assert "Test Paper Title" in candidate.paper_titles
