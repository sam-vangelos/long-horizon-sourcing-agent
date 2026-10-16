"""Async Exa API client with QPS pacing and a fail-closed spend meter.

Wraps the two Exa endpoints this module needs:
- ``POST /search``   — neural/keyword web search, optionally with contents
- ``POST /contents`` — content retrieval for already-known URLs

Every call reserves a worst-case charge before dispatch and settles
against the response's ``costDollars``. Auth (401) and credit (402)
failures are terminal; 429 and 5xx retry with backoff.

Usage:
    async with ExaClient() as client:
        response = await client.search("head of talent", category="people")
        pages = await client.get_contents([r.url for r in response.results])
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Optional

import aiohttp

import exa.config as ec
from exa.cost_meter import ExaCostMeter
from shared.human_timing import human_delay
from shared.tls import client_ssl_context


# ---------------------------------------------------------------------------
# Retry config
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5

# Transport faults raised before the request left this machine: DNS, TCP
# connect, TLS handshake and certificate checks, a proxy refusal, and the
# connect-phase timeout. Exa never saw the request, so it cannot have billed
# it. Every other transport fault (a disconnect or read timeout after the
# request was sent) leaves the outcome unknown and stays charged.
_NEVER_SENT_FAULTS: tuple[type[BaseException], ...] = (
    aiohttp.ClientConnectorError,
    aiohttp.ConnectionTimeoutError,
    aiohttp.ServerFingerprintMismatch,
)

# Keys either endpoint accepts in a contents map. Anything else is a
# caller mistake, and forwarding it risks overwriting a request field
# (``ids``) or spending on a shape the API will reject.
_CONTENTS_KEYS = frozenset(
    {
        "text",
        "highlights",
        "summary",
        "extras",
        "maxAgeHours",
        "livecrawl",
        "livecrawlTimeout",
        "subpages",
    }
)


class ExaAuthError(RuntimeError):
    """Raised when Exa rejects the configured API key. Never retried."""

    status_code = 401


class ExaCreditsError(RuntimeError):
    """Raised when the Exa account is out of credits. Never retried."""

    status_code = 402


# ---------------------------------------------------------------------------
# Response types (S1 owns transport shapes only; S2 owns candidate schemas)
# ---------------------------------------------------------------------------

@dataclass
class ExaResult:
    """One result from /search or /contents. Tolerant of absent fields."""

    id: str = ""
    url: str = ""
    title: str = ""
    published_date: str = ""
    author: str = ""
    text: str = ""
    highlights: list[str] = field(default_factory=list)
    highlight_scores: list[float] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_api(cls, data: dict) -> ExaResult:
        """Parse from an Exa result object. Missing or null fields become empty."""

        data = data or {}
        highlights = data.get("highlights") or []
        scores = data.get("highlightScores") or []
        return cls(
            id=data.get("id", "") or "",
            url=data.get("url", "") or "",
            title=data.get("title", "") or "",
            published_date=data.get("publishedDate", "") or "",
            author=data.get("author", "") or "",
            text=data.get("text", "") or "",
            highlights=[str(item) for item in highlights if item is not None],
            highlight_scores=[
                float(score)
                for score in scores
                if isinstance(score, (int, float)) and not isinstance(score, bool)
            ],
            summary=data.get("summary", "") or "",
        )


@dataclass
class ExaSearchResponse:
    """A parsed /search response, with the API's own cost telemetry attached."""

    request_id: str = ""
    results: list[ExaResult] = field(default_factory=list)
    cost_dollars: dict = field(default_factory=dict)
    search_time: float | None = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "results": [result.to_dict() for result in self.results],
            "cost_dollars": self.cost_dollars,
            "search_time": self.search_time,
        }

    @classmethod
    def from_api(cls, data: dict) -> ExaSearchResponse:
        data = data or {}
        return cls(
            request_id=data.get("requestId", "") or "",
            results=[ExaResult.from_api(item) for item in (data.get("results") or [])],
            cost_dollars=data.get("costDollars") or {},
            search_time=_optional_float(data.get("searchTime")),
        )


@dataclass
class ExaContentsResponse:
    """A parsed /contents response. ``statuses`` carries per-URL crawl outcomes."""

    request_id: str = ""
    results: list[ExaResult] = field(default_factory=list)
    statuses: list[dict] = field(default_factory=list)
    cost_dollars: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "results": [result.to_dict() for result in self.results],
            "statuses": self.statuses,
            "cost_dollars": self.cost_dollars,
        }

    @classmethod
    def from_api(cls, data: dict) -> ExaContentsResponse:
        data = data or {}
        return cls(
            request_id=data.get("requestId", "") or "",
            results=[ExaResult.from_api(item) for item in (data.get("results") or [])],
            statuses=[item for item in (data.get("statuses") or []) if isinstance(item, dict)],
            cost_dollars=data.get("costDollars") or {},
        )


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# ---------------------------------------------------------------------------
# QPS pacer
# ---------------------------------------------------------------------------

class _QpsPacer:
    """Minimum-interval pacer for one endpoint.

    ``shared.rate_limiter.RateLimiter`` is hardwired to ``github.config``
    (in ``shared.rate_limiter.RateLimiter.__init__``), so Exa gets its own tiny pacer
    rather than a shared abstraction bent to fit two callers. ``sleep``
    and ``clock`` are injectable so tests can run it without wall time.
    """

    def __init__(
        self,
        qps: float,
        *,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
        clock: Optional[Callable[[], float]] = None,
        min_interval: Optional[float] = None,
    ):
        if min_interval is not None:
            self._min_interval = max(float(min_interval), 0.0)
        else:
            self._min_interval = 1.0 / float(qps) if qps and qps > 0 else 0.0
        self._sleep = sleep or asyncio.sleep
        self._clock = clock or time.monotonic
        self._last: float | None = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Hold at or below the endpoint's QPS. Returns the seconds waited."""

        async with self._lock:
            now = self._clock()
            waited = 0.0
            if self._last is not None and self._min_interval > 0:
                gap = self._min_interval - (now - self._last)
                if gap > 0:
                    await self._sleep(gap)
                    waited = gap
                    now = self._clock()
            self._last = now
            return waited


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ExaClient:
    """Async Exa REST client with per-endpoint pacing and spend metering."""

    SEARCH_PATH = "/search"
    CONTENTS_PATH = "/contents"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        meter: Optional[ExaCostMeter] = None,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
        clock: Optional[Callable[[], float]] = None,
        pacer_min_interval: Optional[float] = None,
    ):
        self._api_key = api_key or ec.EXA_API_KEY
        if not self._api_key:
            raise RuntimeError(
                "EXA_API_KEY not set. Add it to .env or pass to ExaClient()."
            )
        self._base = ec.EXA_API_BASE
        self._session: Optional[aiohttp.ClientSession] = None
        self._sleep = sleep or asyncio.sleep
        if meter is None:
            from shared.config import EXA_RUN_BUDGET_USD

            meter = ExaCostMeter(cap_usd=EXA_RUN_BUDGET_USD)
        self._meter = meter
        self._pacers: dict[str, _QpsPacer] = {
            self.SEARCH_PATH: _QpsPacer(
                ec.SEARCH_QPS,
                sleep=sleep,
                clock=clock,
                min_interval=pacer_min_interval,
            ),
            self.CONTENTS_PATH: _QpsPacer(
                ec.CONTENTS_QPS,
                sleep=sleep,
                clock=clock,
                min_interval=pacer_min_interval,
            ),
        }

    async def __aenter__(self) -> ExaClient:
        # certifi plus CLORIS_EXTRA_CA_FILE — api.exa.ai can be re-signed by
        # a TLS-inspecting gateway (shared/tls.py).
        ssl_ctx = client_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "sourcing-agent/1.0",
            },
            timeout=aiohttp.ClientTimeout(total=ec.REQUEST_TIMEOUT_SECONDS),
        )
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    @property
    def meter(self) -> ExaCostMeter:
        return self._meter

    # ── Core request method ──────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        json_body: dict,
        *,
        reserve: Optional[Callable[[], None]] = None,
        release: Optional[Callable[[], None]] = None,
    ) -> tuple[int, dict | None, dict]:
        """Make a paced request. Returns (status_code, json_body, headers).

        Retries 429 (honoring ``Retry-After``) and any 5xx, plus
        transport faults. 401 and 402 raise immediately — a bad key or an
        empty account will not fix itself on attempt two.

        Budget hooks. ``reserve`` is called before EVERY dispatch, because
        every dispatch can be billed: a retry after a lost response pays
        twice. ``release`` refunds an attempt Exa provably never billed:
        one whose connection was never established (DNS, TCP, TLS — see
        ``_NEVER_SENT_FAULTS``), and one the server answered with a
        non-2xx status and no usage record (429, 5xx, and the 4xx that
        raise, 401 and 402 included). A fault AFTER the request was sent
        — a disconnect, a read timeout — leaves the outcome unknown: Exa
        may have run the search and lost the reply, so that attempt stays
        charged. A ``reserve`` that cannot fit under the cap raises out
        of this method, stopping the retry loop mid-flight.
        """

        url = path if path.startswith("http") else f"{self._base}{path}"
        pacer = self._pacers.get(path)
        last_attempt_index = _MAX_RETRIES - 1

        for attempt in range(_MAX_RETRIES):
            if pacer is not None:
                await pacer.acquire()
            if reserve is not None:
                reserve()

            try:
                async with self._session.request(method, url, json=json_body) as resp:
                    status = resp.status
                    headers = dict(resp.headers)

                    if status == 200:
                        return status, await resp.json(), headers

                    # Best-effort body read: a failure to read the body of a
                    # 401 must not be reclassified as a retryable transport
                    # fault. The status already decided the outcome.
                    text = await _safe_text(resp)
                    tag = _error_tag(text)

                    # The server answered with an error. Unless the error
                    # body itself reports a charge, Exa did no billable
                    # work, so this attempt's reservation goes back — on
                    # every path out of here, the raising ones included.
                    if release is not None and not _error_reports_usage(text):
                        release()

                    if status == 401:
                        raise ExaAuthError(
                            "Exa API authentication failed: 401 "
                            f"{tag or 'INVALID_API_KEY'}. Check EXA_API_KEY in "
                            f".env. Response: {text[:300]}"
                        )

                    if status == 402:
                        raise ExaCreditsError(
                            "Exa API refused the call: 402 "
                            f"{tag or 'OUT_OF_CREDITS'}. The Exa account is out "
                            f"of credits. Response: {text[:300]}"
                        )

                    if status == 429 or _is_retryable_status(status):
                        # The server answered and did not do the work; the
                        # reservation was returned above.
                        if attempt >= last_attempt_index:
                            break
                        wait = None
                        if status == 429:
                            wait = _retry_after_seconds(headers)
                        if wait is None:
                            wait = (2 ** attempt) + human_delay(0.5, 2.0)
                        print(
                            _one_line(
                                f"    [exa] HTTP {status}, retrying in "
                                f"{wait:.1f}s (attempt {attempt + 1})"
                            )
                        )
                        await self._sleep(wait)
                        continue

                    suffix = f" ({tag})" if tag else ""
                    raise RuntimeError(f"Exa API error {status}{suffix}: {text[:500]}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if isinstance(e, _NEVER_SENT_FAULTS):
                    # The request never reached Exa; it bought nothing.
                    if release is not None:
                        release()
                # Otherwise the outcome is unknown and the attempt stays charged.
                if attempt >= last_attempt_index:
                    raise
                wait = (2 ** attempt) + human_delay(0.5, 2.0)
                print(
                    _one_line(
                        f"    [exa] Connection error ({e}), retrying in {wait:.1f}s"
                    )
                )
                await self._sleep(wait)

        raise RuntimeError(
            f"Exa API request failed after {_MAX_RETRIES} retries: {url}"
        )

    # ── Search ───────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        search_type: str = "auto",
        num_results: int = ec.DEFAULT_NUM_RESULTS,
        category: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        contents: Optional[dict] = None,
        user_location: Optional[str] = None,
    ) -> ExaSearchResponse:
        """Run one Exa search.

        Validation runs before any network or budget activity; the body
        carries only the fields the caller actually set.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Exa search requires a non-empty query")
        _validate_num_results(num_results)
        if search_type not in ec.SEARCH_TYPES:
            raise ValueError(
                f"Unknown Exa search type {search_type!r}; expected one of "
                f"{sorted(ec.SEARCH_TYPES)}"
            )
        if category is not None and category not in ec.CATEGORIES:
            raise ValueError(
                f"Unknown Exa category {category!r}; expected one of "
                f"{sorted(ec.CATEGORIES)}"
            )
        _validate_domains("includeDomains", include_domains)
        _validate_domains("excludeDomains", exclude_domains)
        if user_location is not None and not (
            isinstance(user_location, str)
            and len(user_location) == 2
            and user_location.isascii()
            and user_location.isalpha()
        ):
            raise ValueError(
                "Exa userLocation must be a 2-letter ASCII country code, got "
                f"{user_location!r}"
            )
        normalized_contents = _normalize_contents(contents)

        body: dict = {
            "query": query,
            "type": search_type,
            "numResults": int(num_results),
        }
        if category is not None:
            body["category"] = category
        if include_domains is not None:
            body["includeDomains"] = list(include_domains)
        if exclude_domains is not None:
            body["excludeDomains"] = list(exclude_domains)
        if start_published_date is not None:
            body["startPublishedDate"] = start_published_date
        if end_published_date is not None:
            body["endPublishedDate"] = end_published_date
        if user_location is not None:
            body["userLocation"] = user_location
        if normalized_contents is not None:
            body["contents"] = normalized_contents

        estimate = self._meter.estimate_search_usd(
            search_type=search_type,
            num_results=int(num_results),
            contents=normalized_contents,
        )

        status, payload, headers = await self._request(
            "POST",
            self.SEARCH_PATH,
            body,
            reserve=lambda: self._meter.reserve(estimate),
            release=lambda: self._meter.release(estimate),
        )
        data = payload or {}
        self._meter.settle(
            endpoint=self.SEARCH_PATH,
            estimate_usd=estimate,
            cost_dollars=data.get("costDollars"),
            request_id=data.get("requestId") or headers.get("x-request-id"),
            num_results=len(data.get("results") or []),
        )
        return ExaSearchResponse.from_api(data)

    # ── Contents ─────────────────────────────────────────────────────

    async def get_contents(
        self,
        urls: list[str],
        *,
        contents: Optional[dict] = None,
        max_age_hours: Optional[int] = None,
    ) -> ExaContentsResponse:
        """Fetch page contents for already-known URLs.

        An empty ``urls`` list is a no-op: no request, no charge, an
        empty response.
        """

        if not urls:
            return ExaContentsResponse()
        if max_age_hours is not None:
            _validate_max_age_hours(max_age_hours)
        normalized_contents = _normalize_contents(contents)

        # ``/contents`` takes the content flags at the TOP level, so the
        # caller's map is merged FIRST and the call's own fields are
        # written last — a contents map can never overwrite the URL list
        # or the explicit freshness bound.
        body: dict = dict(normalized_contents or {})
        body["ids"] = list(urls)
        if max_age_hours is not None:
            body["maxAgeHours"] = max_age_hours

        estimate = self._meter.estimate_contents_usd(
            num_urls=len(urls),
            contents=normalized_contents,
        )

        status, payload, headers = await self._request(
            "POST",
            self.CONTENTS_PATH,
            body,
            reserve=lambda: self._meter.reserve(estimate),
            release=lambda: self._meter.release(estimate),
        )
        data = payload or {}
        self._meter.settle(
            endpoint=self.CONTENTS_PATH,
            estimate_usd=estimate,
            cost_dollars=data.get("costDollars"),
            request_id=data.get("requestId") or headers.get("x-request-id"),
            num_results=len(data.get("results") or []),
        )
        return ExaContentsResponse.from_api(data)

    # ── Preflight ────────────────────────────────────────────────────

    async def validate_credentials(self) -> None:
        """Fail fast before a run starts, on the cheapest real Exa call.

        ``fast`` type, one result, no contents — the minimum billable
        search. Auth and credit failures propagate as their typed errors
        so the readiness probe can name the actual blocker.
        """

        try:
            await self.search("exa", search_type="fast", num_results=1)
        except (ExaAuthError, ExaCreditsError):
            raise
        except RuntimeError as exc:
            raise RuntimeError(
                f"Exa credential preflight failed: {exc}. Check EXA_API_KEY in .env."
            ) from exc

    # ── Status ───────────────────────────────────────────────────────

    def status_line(self) -> str:
        """One-line spend status for console output."""

        return self._meter.status_line()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_num_results(num_results: int) -> None:
    if isinstance(num_results, bool) or not isinstance(num_results, int):
        raise ValueError("Exa numResults must be an integer")
    if not (1 <= num_results <= ec.MAX_NUM_RESULTS):
        raise ValueError(
            f"Exa numResults must be between 1 and {ec.MAX_NUM_RESULTS}, got "
            f"{num_results}"
        )


def _validate_domains(field_name: str, domains: list[str] | None) -> None:
    """Reject anything that is not a real list of domain strings.

    A bare ``str`` is the trap: it has a length and iterates, so a
    length-only check would send ``"exa.ai"`` to the API as seven
    single-character domains and silently return nothing.
    """

    if domains is None:
        return
    if isinstance(domains, (str, bytes)) or not isinstance(domains, (list, tuple)):
        raise ValueError(
            f"Exa {field_name} must be a list of domain strings, got "
            f"{type(domains).__name__}"
        )
    if len(domains) > ec.MAX_DOMAIN_FILTER_ENTRIES:
        raise ValueError(
            f"Exa {field_name} accepts at most "
            f"{ec.MAX_DOMAIN_FILTER_ENTRIES} entries, got {len(domains)}"
        )
    for entry in domains:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                f"Exa {field_name} entries must be non-empty strings, got "
                f"{entry!r}"
            )


def _validate_max_age_hours(max_age_hours: object) -> None:
    if isinstance(max_age_hours, bool) or not isinstance(max_age_hours, int):
        raise ValueError("Exa maxAgeHours must be an integer")
    if not (ec.MIN_MAX_AGE_HOURS <= max_age_hours <= ec.MAX_MAX_AGE_HOURS):
        raise ValueError(
            "Exa maxAgeHours must be between "
            f"{ec.MIN_MAX_AGE_HOURS} and {ec.MAX_MAX_AGE_HOURS}, got "
            f"{max_age_hours}"
        )


def _normalize_contents(contents: dict | None) -> dict | None:
    """Validate and clean a contents map for either endpoint.

    One helper for both, because both endpoints take the same flags —
    ``/search`` nested under ``contents``, ``/contents`` at the top
    level. Unknown keys are rejected rather than forwarded: the
    allowlist is what stops a caller's ``ids`` or ``query`` from riding
    into the request body. ``None`` values are dropped so they never
    reach the wire as JSON nulls, and ``maxAgeHours`` is bounds-checked
    here so an out-of-range value fails BEFORE anything is reserved or
    dispatched.
    """

    if contents is None:
        return None
    if not isinstance(contents, dict):
        raise ValueError(
            f"Exa contents must be a mapping, got {type(contents).__name__}"
        )
    unknown = sorted(str(key) for key in contents if key not in _CONTENTS_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown Exa contents key(s) {unknown}; expected a subset of "
            f"{sorted(_CONTENTS_KEYS)}"
        )
    normalized = {key: value for key, value in contents.items() if value is not None}
    if "maxAgeHours" in normalized:
        _validate_max_age_hours(normalized["maxAgeHours"])
    return normalized


def _is_retryable_status(status: int) -> bool:
    """Any 5xx is a server-side fault worth another attempt (504 included)."""

    return 500 <= status < 600


async def _safe_text(resp) -> str:
    """Read an error body without letting the read change the diagnosis."""

    try:
        return await resp.text()
    except Exception:
        return ""


def _one_line(text: str) -> str:
    """Collapse anything destined for the run console onto one line.

    The leading indent is preserved (the console idiom is a four-space
    prefix); every other whitespace run, newlines included, becomes a
    single space. Raw provider and transport messages are multi-line
    often enough that this has to be structural, not a habit.
    """

    raw = str(text)
    indent = raw[: len(raw) - len(raw.lstrip(" "))]
    return indent + " ".join(raw.replace("\r", " ").replace("\n", " ").split())


def _error_tag(text: str) -> str:
    """Pull ``tag`` out of Exa's ``{requestId, error, tag}`` envelope."""

    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    tag = payload.get("tag")
    return str(tag) if tag else ""


def _error_reports_usage(text: str) -> bool:
    """Whether an error body carries Exa's own usage record (``costDollars``).

    Exa's error envelope is ``{requestId, error, tag}``; a body that ALSO
    reports a charge says the call was billed, so its reservation is not
    returned. Anything unparseable reports nothing.
    """

    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("costDollars") not in (None, {}, "")


def _retry_after_seconds(headers: dict) -> float | None:
    """Parse ``Retry-After`` (seconds form). Returns None when absent/unparseable."""

    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ExaAuthError",
    "ExaClient",
    "ExaContentsResponse",
    "ExaCreditsError",
    "ExaResult",
    "ExaSearchResponse",
]
