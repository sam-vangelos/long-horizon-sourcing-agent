"""Offline contract tests for the Exa HTTP client (S1).

No network: every test drives a fake aiohttp session or an AsyncMock on
``_request``. Recorded fixtures live in ``tests/fixtures/exa/``.
"""

import asyncio
import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

import exa.config as exa_config
from exa.client import (
    ExaAuthError,
    ExaClient,
    ExaCreditsError,
    _QpsPacer,
)
from exa.cost_meter import ExaBudgetExhaustedError, ExaCostMeter


# One default search (auto, no contents) costs this much to reserve.
_SEARCH_ESTIMATE = 0.007
# The recorded people fixture's measured total.
_FIXTURE_ACTUAL = 0.016


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "exa"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fake aiohttp session
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(
        self, status, *, payload=None, text=None, headers=None, text_error=None
    ):
        self.status = status
        self._payload = payload
        if text is not None:
            self._text = text
        elif payload is not None:
            self._text = json.dumps(payload)
        else:
            self._text = ""
        self.headers = headers or {}
        self._text_error = text_error

    async def json(self):
        return self._payload

    async def text(self):
        if self._text_error is not None:
            raise self._text_error
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeFault:
    """A queued transport fault — raises when the request context is entered."""

    def __init__(self, exc):
        self.exc = exc

    async def __aenter__(self):
        raise self.exc

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Returns queued responses; the last one repeats for further attempts."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, method, url, json=None, **kwargs):
        self.calls.append({"method": method, "url": url, "json": json})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _make_client(responses, *, cap_usd=0.0, sink_path=None):
    """Client wired to a fake session with every wait zeroed.

    ``pacer_min_interval=0.0`` keeps QPS pacing out of the captured sleep
    list so retry assertions see only retry waits.
    """

    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    client = ExaClient(
        api_key="dummy",
        meter=ExaCostMeter(cap_usd=cap_usd, sink_path=sink_path),
        sleep=fake_sleep,
        pacer_min_interval=0.0,
    )
    session = _FakeSession(responses)
    client._session = session
    return client, session, sleeps


# ---------------------------------------------------------------------------
# Search parsing
# ---------------------------------------------------------------------------

def test_search_parses_recorded_people_fixture():
    payload = _fixture("search_people_ok.json")
    client, session, _ = _make_client([_FakeResponse(200, payload=payload)])

    response = asyncio.run(
        client.search(
            "chief of staff to the CEO venture capital",
            category="people",
            num_results=3,
            contents={"text": True, "highlights": True, "summary": {"query": "role"}},
        )
    )

    assert response.request_id == "b1f0c6f4e2a94f0f8f2b3c5d7e9a0b12"
    assert response.search_time == pytest.approx(1834.51)
    assert len(response.results) == 3

    first = response.results[0]
    assert first.id == "https://www.linkedin.com/in/dana-okafor-chiefofstaff"
    assert first.url == "https://www.linkedin.com/in/dana-okafor-chiefofstaff"
    assert first.title.startswith("Dana Okafor - Chief of Staff")
    assert first.author == "Dana Okafor"
    assert first.published_date == "2026-05-14T00:00:00.000Z"
    assert "Northwind Capital" in first.text
    assert len(first.highlights) == 2
    assert first.highlight_scores == [pytest.approx(0.7421), pytest.approx(0.6188)]
    assert first.summary.startswith("Chief of Staff to the CEO at Northwind Capital")

    # Null publishedDate / author / empty highlights must not crash the parse.
    third = response.results[2]
    assert third.published_date == ""
    assert third.author == ""
    assert third.highlights == []
    assert third.highlight_scores == []

    assert response.cost_dollars["total"] == pytest.approx(0.016)
    assert response.cost_dollars["search"]["neural"] == pytest.approx(0.007)
    assert response.to_dict()["results"][0]["url"] == first.url


# ---------------------------------------------------------------------------
# Pre-network validation
# ---------------------------------------------------------------------------

def test_search_rejects_invalid_num_results_before_network():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])
    client._request = AsyncMock()

    for bad in (0, 101):
        with pytest.raises(ValueError, match="numResults"):
            asyncio.run(client.search("q", num_results=bad))

    client._request.assert_not_awaited()
    assert session.calls == []
    assert client.meter.committed_usd == 0.0


def test_search_rejects_unknown_type_and_category_before_network():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])
    client._request = AsyncMock()

    with pytest.raises(ValueError, match="search type"):
        asyncio.run(client.search("q", search_type="turbo"))
    with pytest.raises(ValueError, match="category"):
        asyncio.run(client.search("q", category="humans"))

    client._request.assert_not_awaited()
    assert session.calls == []
    assert client.meter.committed_usd == 0.0


def test_domains_reject_bare_string():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])

    # A bare string has a length and iterates — a length-only check would
    # send "exa.ai" as six single-character domains and match nothing.
    with pytest.raises(ValueError, match="must be a list of domain strings"):
        asyncio.run(client.search("q", include_domains="exa.ai"))
    with pytest.raises(ValueError, match="non-empty strings"):
        asyncio.run(client.search("q", exclude_domains=["ok.com", ""]))
    with pytest.raises(ValueError, match="non-empty strings"):
        asyncio.run(client.search("q", include_domains=["ok.com", None]))

    assert session.calls == []
    assert client.meter.committed_usd == 0.0


def test_user_location_rejects_non_ascii():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])

    # "ÜS".isalpha() is True — only isascii() catches it.
    with pytest.raises(ValueError, match="ASCII country code"):
        asyncio.run(client.search("q", user_location="ÜS"))
    with pytest.raises(ValueError, match="ASCII country code"):
        asyncio.run(client.search("q", user_location="USA"))

    assert session.calls == []


def test_contents_unknown_key_rejected():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])

    with pytest.raises(ValueError, match="Unknown Exa contents key"):
        asyncio.run(client.search("q", contents={"txt": True}))
    with pytest.raises(ValueError, match="Unknown Exa contents key"):
        asyncio.run(client.get_contents(["https://a"], contents={"query": "x"}))

    assert session.calls == []
    assert client.meter.committed_usd == 0.0


def test_contents_map_cannot_override_ids():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])

    # `ids` is not an allowlisted contents key, so it never reaches the
    # body-merge in the first place.
    with pytest.raises(ValueError, match="Unknown Exa contents key"):
        asyncio.run(
            client.get_contents(
                ["https://real"],
                contents={"ids": ["https://substituted"], "text": True},
            )
        )
    assert session.calls == []

    # And an allowlisted key that collides with a request field loses to
    # the explicit argument.
    asyncio.run(
        client.get_contents(
            ["https://real"],
            contents={"text": True, "maxAgeHours": 24},
            max_age_hours=1,
        )
    )
    body = session.calls[0]["json"]
    assert body["ids"] == ["https://real"]
    assert body["maxAgeHours"] == 1


def test_contents_none_values_are_dropped():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])

    asyncio.run(client.search("q", contents={"text": True, "highlights": None}))
    assert session.calls[0]["json"]["contents"] == {"text": True}

    asyncio.run(client.get_contents(["https://a"], contents={"text": True, "summary": None}))
    body = session.calls[1]["json"]
    assert body == {"text": True, "ids": ["https://a"]}
    assert "summary" not in body


def test_search_contents_max_age_validated_before_reserve():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])
    client.meter.reserve = MagicMock()

    with pytest.raises(ValueError, match="maxAgeHours"):
        asyncio.run(client.search("q", contents={"text": True, "maxAgeHours": 721}))
    with pytest.raises(ValueError, match="maxAgeHours"):
        asyncio.run(client.get_contents(["https://a"], max_age_hours=-2))

    client.meter.reserve.assert_not_called()
    assert session.calls == []


def test_search_body_omits_unset_optional_fields():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])

    asyncio.run(client.search("head of talent"))

    body = session.calls[0]["json"]
    assert body == {"query": "head of talent", "type": "auto", "numResults": 10}
    assert "contents" not in body
    assert "category" not in body
    assert "includeDomains" not in body
    assert "userLocation" not in body
    assert all(value is not None for value in body.values())


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_401_raises_auth_error_without_retry():
    payload = _fixture("search_error_401.json")
    client, session, sleeps = _make_client([_FakeResponse(401, payload=payload)])

    with pytest.raises(ExaAuthError, match="INVALID_API_KEY"):
        asyncio.run(client.search("q"))

    assert len(session.calls) == 1
    assert sleeps == []


def test_402_raises_credits_error_without_retry():
    payload = {
        "requestId": "f8a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5",
        "error": "Insufficient credits to complete this request.",
    }
    client, session, sleeps = _make_client([_FakeResponse(402, payload=payload)])

    with pytest.raises(ExaCreditsError, match="out of credits"):
        asyncio.run(client.search("q"))

    assert len(session.calls) == 1
    assert sleeps == []


def test_429_retries_and_honors_retry_after():
    rate_limited = _FakeResponse(
        429,
        payload=_fixture("search_error_429.json"),
        headers={"Retry-After": "2.5"},
    )
    ok = _FakeResponse(200, payload=_fixture("search_people_ok.json"))
    client, session, sleeps = _make_client([rate_limited, ok])

    response = asyncio.run(client.search("q", num_results=3))

    assert len(session.calls) == 2
    assert sleeps == [pytest.approx(2.5)]
    assert len(response.results) == 3


def test_5xx_retries_then_raises_after_max_retries():
    client, session, sleeps = _make_client(
        [_FakeResponse(500, text="upstream failure")]
    )

    with pytest.raises(RuntimeError, match="failed after 5 retries"):
        asyncio.run(client.search("q"))

    assert len(session.calls) == 5
    # Four waits between five attempts — the last attempt raises rather
    # than sleeping into a wait nobody is waiting for.
    assert len(sleeps) == 4
    # Every 5xx released its attempt's reservation.
    assert client.meter.committed_usd == pytest.approx(0.0)


def test_504_is_retried():
    client, session, _ = _make_client(
        [
            _FakeResponse(504, text="gateway timeout"),
            _FakeResponse(200, payload=_fixture("search_people_ok.json")),
        ]
    )

    response = asyncio.run(client.search("q", num_results=3))

    assert len(session.calls) == 2
    assert len(response.results) == 3
    assert client.meter.committed_usd == pytest.approx(_FIXTURE_ACTUAL)


def test_final_429_does_not_sleep():
    client, session, sleeps = _make_client(
        [
            _FakeResponse(
                429,
                payload=_fixture("search_error_429.json"),
                headers={"Retry-After": "3600"},
            )
        ]
    )

    with pytest.raises(RuntimeError, match="failed after 5 retries"):
        asyncio.run(client.search("q"))

    # An hour-long Retry-After on the LAST attempt would hang the run for
    # an hour before raising anyway.
    assert len(session.calls) == 5
    assert sleeps == [pytest.approx(3600.0)] * 4


def test_401_with_unreadable_body_still_raises_auth_error():
    client, session, sleeps = _make_client(
        [_FakeResponse(401, text_error=aiohttp.ClientError("body read failed"))]
    )

    # A failed body read must not demote a 401 into a retryable transport
    # fault — the status already decided the outcome.
    with pytest.raises(ExaAuthError, match="INVALID_API_KEY"):
        asyncio.run(client.search("q"))

    assert len(session.calls) == 1
    assert sleeps == []


# ---------------------------------------------------------------------------
# Per-attempt budget accounting
# ---------------------------------------------------------------------------

def test_transport_fault_retry_keeps_prior_estimate_charged():
    client, session, _ = _make_client(
        [
            _FakeFault(aiohttp.ClientError("connection reset by peer")),
            _FakeResponse(200, payload=_fixture("search_people_ok.json")),
        ]
    )

    asyncio.run(client.search("q", num_results=3))

    # Exa may have run the lost attempt and billed for it. Two dispatches,
    # one measured — the unknown one stays charged.
    assert len(session.calls) == 2
    assert client.meter.committed_usd == pytest.approx(
        _FIXTURE_ACTUAL + _SEARCH_ESTIMATE
    )
    assert client.meter.call_count == 1


def test_429_retry_releases_that_attempts_estimate():
    client, session, _ = _make_client(
        [
            _FakeResponse(
                429,
                payload=_fixture("search_error_429.json"),
                headers={"Retry-After": "1.5"},
            ),
            _FakeResponse(200, payload=_fixture("search_people_ok.json")),
        ]
    )

    asyncio.run(client.search("q", num_results=3))

    # The server answered and did no work, so that attempt bought nothing.
    assert len(session.calls) == 2
    assert client.meter.committed_usd == pytest.approx(_FIXTURE_ACTUAL)


def test_reserve_per_attempt_can_exhaust_cap_mid_retry():
    client, session, _ = _make_client(
        [_FakeFault(aiohttp.ClientError("connection reset by peer"))],
        cap_usd=_SEARCH_ESTIMATE,
    )

    # The cap funds exactly one attempt. The first is charged (outcome
    # unknown), so the retry cannot be funded and stops before dispatch.
    with pytest.raises(ExaBudgetExhaustedError):
        asyncio.run(client.search("q"))

    assert len(session.calls) == 1
    assert client.meter.committed_usd == pytest.approx(_SEARCH_ESTIMATE)


# ---------------------------------------------------------------------------
# QPS pacer
# ---------------------------------------------------------------------------

def test_qps_pacer_spaces_requests():
    sleeps: list[float] = []
    now = {"t": 0.0}

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        now["t"] += seconds

    pacer = _QpsPacer(10, sleep=fake_sleep, clock=lambda: now["t"])

    async def three_acquires():
        for _ in range(3):
            await pacer.acquire()

    asyncio.run(three_acquires())

    assert len(sleeps) == 2
    assert all(seconds >= 0.1 - 1e-9 for seconds in sleeps)


def test_qps_pacer_serializes_concurrent_acquires():
    """Three acquires racing under asyncio.gather must still be spaced.

    The sequential test passes even with the lock removed. This one does
    not: without the lock, the second and third acquires both compute
    their gap from the same stale timestamp, both sleep once, and both
    land at the same instant.
    """

    now = {"t": 0.0}
    completions: list[float] = []

    async def fake_sleep(seconds):
        now["t"] += seconds
        await asyncio.sleep(0)  # yield, so an unlocked pacer can interleave

    pacer = _QpsPacer(10, sleep=fake_sleep, clock=lambda: now["t"])

    async def acquire_and_stamp():
        await pacer.acquire()
        completions.append(now["t"])

    async def race():
        await asyncio.gather(*(acquire_and_stamp() for _ in range(3)))

    asyncio.run(race())

    assert sorted(completions) == [
        pytest.approx(0.0),
        pytest.approx(0.1),
        pytest.approx(0.2),
    ]


def test_console_lines_never_contain_newlines(capsys):
    client, session, _ = _make_client(
        [
            _FakeFault(aiohttp.ClientError("boom\nsecond line\r\nthird line")),
            _FakeResponse(200, payload={}),
        ]
    )

    asyncio.run(client.search("q"))

    printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(printed) == 1
    assert printed[0].startswith("    [exa] Connection error")
    assert "second line" in printed[0]
    assert "third line" in printed[0]


# ---------------------------------------------------------------------------
# Contents
# ---------------------------------------------------------------------------

def test_get_contents_empty_urls_makes_no_request():
    client, session, _ = _make_client([_FakeResponse(200, payload={})])

    response = asyncio.run(client.get_contents([]))

    assert session.calls == []
    assert response.results == []
    assert response.request_id == ""
    assert client.meter.call_count == 0
    assert client.meter.committed_usd == 0.0


def test_contents_statuses_surface_per_url_errors():
    payload = _fixture("contents_ok.json")
    client, session, _ = _make_client([_FakeResponse(200, payload=payload)])

    response = asyncio.run(
        client.get_contents(
            [
                "https://www.riverstonepartners.com/team/marcus-lindqvist",
                "https://www.aldpeak-internal.example/team/priya",
            ],
            contents={"text": True, "highlights": True},
        )
    )

    assert [entry["status"] for entry in response.statuses] == ["success", "error"]
    assert response.statuses[1]["error"]["tag"] == "CRAWL_NOT_FOUND"
    assert len(response.results) == 1
    assert response.cost_dollars["total"] == pytest.approx(0.002)

    body = session.calls[0]["json"]
    assert body["ids"] == [
        "https://www.riverstonepartners.com/team/marcus-lindqvist",
        "https://www.aldpeak-internal.example/team/priya",
    ]
    assert body["text"] is True
    assert body["highlights"] is True


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def test_validate_credentials_uses_cheapest_call():
    client, session, _ = _make_client(
        [_FakeResponse(200, payload={"requestId": "r", "results": []})]
    )

    asyncio.run(client.validate_credentials())

    assert len(session.calls) == 1
    body = session.calls[0]["json"]
    assert session.calls[0]["url"] == "https://api.exa.ai/search"
    assert body["type"] == "fast"
    assert body["numResults"] == 1
    assert "contents" not in body


def test_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(exa_config, "EXA_API_KEY", "")

    with pytest.raises(RuntimeError, match="EXA_API_KEY not set"):
        ExaClient()


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

def test_exa_api_key_uses_agent_key_pattern(monkeypatch):
    """The AGENT_KEY_PREFIX override applies to EXA_API_KEY like every other key."""

    import shared.config as shared_config

    monkeypatch.setenv("AGENT_KEY_PREFIX", "EXA")
    monkeypatch.setenv("EXA_API_KEY", "shared-key")
    monkeypatch.setenv("EXA_EXA_API_KEY", "prefixed-key")
    try:
        reloaded = importlib.reload(shared_config)
        assert reloaded.EXA_API_KEY == "prefixed-key"
        assert reloaded.EXA_RUN_BUDGET_USD == pytest.approx(5.00)

        # exa.config must resolve THROUGH shared.config, not re-read the
        # env itself — a raw os.getenv there would silently ignore the
        # agent prefix and bill the wrong account.
        reloaded_exa = importlib.reload(exa_config)
        assert reloaded_exa.EXA_API_KEY == "prefixed-key"
    finally:
        monkeypatch.undo()
        importlib.reload(shared_config)
        importlib.reload(exa_config)


# ---------------------------------------------------------------------------
# TLS trust: certifi plus an operator-supplied CA (Cloudflare Gateway)
# ---------------------------------------------------------------------------


def test_client_ssl_context_is_certifi_when_no_extra_ca(monkeypatch):
    import ssl
    from shared import tls

    monkeypatch.delenv(tls.EXTRA_CA_ENV, raising=False)
    ctx = tls.client_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_client_ssl_context_loads_extra_ca_when_configured(tmp_path, monkeypatch):
    import certifi
    from shared import tls

    extra = tmp_path / "gateway-ca.pem"
    extra.write_text(open(certifi.where()).read())  # any valid PEM bundle
    monkeypatch.setenv(tls.EXTRA_CA_ENV, str(extra))
    before = len(tls.client_ssl_context().get_ca_certs())
    monkeypatch.delenv(tls.EXTRA_CA_ENV)
    base = len(tls.client_ssl_context().get_ca_certs())
    assert before >= base  # loaded without error; duplicates are tolerated


def test_missing_extra_ca_file_raises_at_construction(tmp_path, monkeypatch):
    import pytest as _pytest
    from shared import tls

    monkeypatch.setenv(tls.EXTRA_CA_ENV, str(tmp_path / "absent.pem"))
    with _pytest.raises(ValueError, match="CLORIS_EXTRA_CA_FILE"):
        tls.client_ssl_context()


# ---------------------------------------------------------------------------
# Attempts Exa never billed release their reservation (2026-10-10 DNS outage)
# ---------------------------------------------------------------------------
#
# The SPL competitor run stopped on api_budget_exhausted with $0.744 of real
# Exa spend: six queries hit a DNS outage, every one of their five attempts
# reserved the $0.297 worst case, and the transport-fault branch kept every
# reservation charged. A request that never reached Exa, or that Exa answered
# with an error and no usage record, bought nothing and must cost nothing.

import socket
import types

# A production-shaped archetype search: 100 results, highlights + text.
_PROD_NUM_RESULTS = 100
_PROD_CONTENTS = {"highlights": {"maxCharacters": 400}, "text": {"maxCharacters": 4000}}
_PROD_ESTIMATE = 0.297


def _conn_key():
    return types.SimpleNamespace(host="api.exa.ai", port=443, ssl=True, is_ssl=True)


def _dns_fault():
    return _FakeFault(
        aiohttp.ClientConnectorDNSError(
            _conn_key(),
            socket.gaierror(8, "nodename nor servname provided, or not known"),
        )
    )


def _prod_search(client):
    return client.search(
        "people who run coding data programs",
        num_results=_PROD_NUM_RESULTS,
        contents=_PROD_CONTENTS,
    )


def test_prod_search_estimate_is_the_phantom_unit():
    meter = ExaCostMeter(cap_usd=0.0)
    estimate = meter.estimate_search_usd(
        search_type="auto", num_results=_PROD_NUM_RESULTS, contents=_PROD_CONTENTS
    )
    assert estimate == pytest.approx(_PROD_ESTIMATE)


def test_three_failed_calls_then_success_commit_only_the_success():
    """Three searches lost to DNS, then one that lands: only the landing bills."""

    fail = [_dns_fault()] * 15  # three calls x five attempts each
    ok = _FakeResponse(200, payload=_fixture("search_people_ok.json"))
    client, session, _ = _make_client([*fail, ok], cap_usd=1.00)

    for _ in range(3):
        with pytest.raises(aiohttp.ClientConnectorDNSError):
            asyncio.run(_prod_search(client))
    response = asyncio.run(_prod_search(client))

    assert len(session.calls) == 16
    assert len(response.results) == 3
    assert client.meter.committed_usd == pytest.approx(_FIXTURE_ACTUAL)
    assert client.meter.call_count == 1


def test_dns_faults_inside_one_call_then_success_commit_only_the_success():
    client, session, _ = _make_client(
        [_dns_fault(), _dns_fault(), _dns_fault(),
         _FakeResponse(200, payload=_fixture("search_people_ok.json"))],
        cap_usd=10.00,
    )

    asyncio.run(_prod_search(client))

    assert len(session.calls) == 4
    assert client.meter.committed_usd == pytest.approx(_FIXTURE_ACTUAL)


@pytest.mark.parametrize(
    "fault",
    [
        aiohttp.ClientConnectorError(_conn_key(), OSError(61, "Connection refused")),
        aiohttp.ClientConnectorCertificateError(_conn_key(), Exception("bad cert")),
        aiohttp.ClientConnectorSSLError(_conn_key(), OSError(1, "handshake failed")),
        aiohttp.ConnectionTimeoutError("Connection timeout to host https://api.exa.ai"),
    ],
    ids=["refused", "certificate", "tls_handshake", "connect_timeout"],
)
def test_connection_never_established_releases_every_attempt(fault):
    client, session, _ = _make_client([_FakeFault(fault)], cap_usd=10.00)

    with pytest.raises(type(fault)):
        asyncio.run(_prod_search(client))

    assert len(session.calls) == 5
    assert client.meter.committed_usd == pytest.approx(0.0)


def test_dns_outage_cannot_exhaust_the_cap():
    """The overnight failure, replayed: six dead queries leave the cap whole."""

    client, session, _ = _make_client([_dns_fault()], cap_usd=10.00)
    for _ in range(6):
        with pytest.raises(aiohttp.ClientConnectorDNSError):
            asyncio.run(_prod_search(client))

    assert len(session.calls) == 30
    assert client.meter.committed_usd == pytest.approx(0.0)


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_client_error_status_without_usage_record_releases(status):
    client, session, _ = _make_client(
        [_FakeResponse(status, payload={"requestId": "r", "error": "bad", "tag": "X"})],
        cap_usd=10.00,
    )

    with pytest.raises(RuntimeError, match=f"Exa API error {status}"):
        asyncio.run(_prod_search(client))

    assert len(session.calls) == 1
    assert client.meter.committed_usd == pytest.approx(0.0)


def test_auth_and_credit_refusals_release_their_attempt():
    for status, error in ((401, ExaAuthError), (402, ExaCreditsError)):
        client, _, _ = _make_client(
            [_FakeResponse(status, payload={"requestId": "r", "error": "no"})],
            cap_usd=10.00,
        )
        with pytest.raises(error):
            asyncio.run(_prod_search(client))
        assert client.meter.committed_usd == pytest.approx(0.0)


def test_error_status_carrying_a_usage_record_stays_charged():
    """A non-2xx that says Exa billed it is not provably free."""

    client, _, _ = _make_client(
        [
            _FakeResponse(
                400,
                payload={"requestId": "r", "error": "partial", "costDollars": {"total": 0.05}},
            )
        ],
        cap_usd=10.00,
    )

    with pytest.raises(RuntimeError, match="Exa API error 400"):
        asyncio.run(_prod_search(client))

    assert client.meter.committed_usd == pytest.approx(_PROD_ESTIMATE)


def test_lost_response_after_dispatch_still_stays_charged():
    """Disconnect after send: Exa may have run it, so the attempt stays charged."""

    client, session, _ = _make_client(
        [
            _FakeFault(aiohttp.ServerDisconnectedError()),
            _FakeResponse(200, payload=_fixture("search_people_ok.json")),
        ],
        cap_usd=10.00,
    )

    asyncio.run(_prod_search(client))

    assert len(session.calls) == 2
    assert client.meter.committed_usd == pytest.approx(_FIXTURE_ACTUAL + _PROD_ESTIMATE)
