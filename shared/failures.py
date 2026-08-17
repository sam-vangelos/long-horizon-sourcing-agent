"""Shared failure semantics for judgment and runtime retry decisions."""

from __future__ import annotations

from dataclasses import dataclass

from shared.contracts import FAILURE_DECISIONS as CONTRACT_FAILURE_DECISIONS
from shared.schemas import OpusDecision

PARSE_FAILURE = "PARSE_FAILURE"
JUDGMENT_FAILURE = "JUDGMENT_FAILURE"

RECOVERABLE_ERROR = "RECOVERABLE_ERROR"
TERMINAL_ERROR = "TERMINAL_ERROR"


class ApiBudgetExhaustedError(RuntimeError):
    """Raised when provider credits are exhausted and the run should pause."""

FAILURE_DECISIONS = CONTRACT_FAILURE_DECISIONS
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

_API_BUDGET_EXHAUSTED_PATTERNS: tuple[str, ...] = (
    "credit balance is too low",
    "purchase credits",
    "plans & billing",
    "billing hard limit",
    "insufficient credits",
    "exceeded your current quota",
    "quota exceeded",
    "spending limit",
    "failure to pay past invoices",
)

_RECOVERABLE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("rate_limit", "provider", "rate_limit"),
    ("rate limit", "provider", "rate_limit"),
    ("429", "provider", "http_429"),
    ("502", "provider", "http_502"),
    ("503", "provider", "http_503"),
    ("504", "provider", "http_504"),
    ("529", "provider", "http_529"),
    ("overloaded", "provider", "capacity"),
    ("capacity", "provider", "capacity"),
    ("timeout", "network", "timeout"),
    ("timed out", "network", "timeout"),
    ("connection reset", "network", "connection_reset"),
    ("connection closed", "network", "connection_closed"),
    ("connection aborted", "network", "connection_aborted"),
    ("target crashed", "browser", "browser_disconnect"),
    ("target closed", "browser", "browser_disconnect"),
    ("page closed", "browser", "browser_disconnect"),
    ("context closed", "browser", "browser_disconnect"),
    ("session closed", "browser", "browser_disconnect"),
    ("browser has been closed", "browser", "browser_disconnect"),
    ("cannot get world", "browser", "browser_disconnect"),
)

_TERMINAL_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("stop_reason=", "provider", "truncated_response"),
    ("finish_reason=", "provider", "truncated_response"),
    ("invalid api key", "auth", "invalid_api_key"),
    ("authentication", "auth", "authentication_failed"),
    ("permission denied", "auth", "permission_denied"),
    ("forbidden", "auth", "forbidden"),
    ("unsupported", "request", "unsupported_request"),
)


def _provider_error_body_message(exc: BaseException) -> str:
    """The provider body's message, tolerating both SDK conventions.

    Anthropic's SDK stores the decoded response as-is, so the message lives
    at ``body["error"]["message"]``; the OpenAI SDK (the Fireworks transport)
    unwraps ``body["error"]`` before constructing the exception, so the same
    message lives at ``body["message"]``. Reading only one nesting made the
    412 typed branch dead code against real Fireworks errors (wave-1 review
    finding) — read both.
    """

    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return ""
    error_body = body.get("error")
    if isinstance(error_body, dict):
        return str(error_body.get("message", ""))
    return str(body.get("message", ""))


@dataclass(frozen=True)
class FailureClassification:
    kind: str
    domain: str
    reason: str
    detail: str = ""
    status_code: int | None = None

    @property
    def retryable(self) -> bool:
        return self.kind == RECOVERABLE_ERROR


def is_failure_decision(decision: str) -> bool:
    """True if decision represents a non-terminal parse or judgment failure."""
    return decision in FAILURE_DECISIONS


def is_api_budget_exhausted_error(exc: BaseException | str) -> bool:
    """Detect provider billing/credit exhaustion errors that should pause the run.

    P8.4(b): prefer the typed error the Anthropic SDK exposes on
    ``anthropic.APIStatusError`` — ``status_code`` (int) and ``type`` (parsed
    from ``body["error"]["type"]``) — over parsing the human-readable message.
    The SDK now has a dedicated ``billing_error`` type for credit/quota
    exhaustion; when present it's a direct signal and needs no message check.
    The older/still-current shape for this specific failure is HTTP 400 with
    the generic ``invalid_request_error`` type, which is too broad on its own
    (it covers many unrelated 400s) — for that type only, the body's message
    is checked against the same phrase list as the fallback below.

    Fireworks reports the same condition as HTTP 412 (``PRECONDITION_FAILED``)
    with an "Account … is suspended, possibly due to reaching the monthly
    spending limit or failure to pay past invoices" message. The 412-scoped
    branch accepts "suspended"/"spending limit" there, where the status code
    bounds the looser phrase; bare "suspended" is deliberately NOT in the
    generic fallback list because arbitrary error text (a browser error
    quoting page copy, say) could carry it.

    Fallback: substring match on ``str(exc)``. Needed for errors that don't
    expose the typed attributes at all (raw httpx failures, other providers,
    mocked exceptions in tests, or a future SDK response shape this hasn't
    been updated for). Risk: a coincidental substring hit on unrelated error
    text could misclassify it as budget exhaustion — this is why the typed
    path is checked first and preferred whenever it's available.
    """
    if isinstance(exc, ApiBudgetExhaustedError):
        return True

    status_code = _coerce_status_code(getattr(exc, "status_code", None))
    error_type = getattr(exc, "type", None)
    if status_code == 400 and error_type == "billing_error":
        return True
    if status_code == 400 and error_type == "invalid_request_error":
        body_message = _provider_error_body_message(exc)
        if body_message and any(
            pattern in body_message.lower() for pattern in _API_BUDGET_EXHAUSTED_PATTERNS
        ):
            return True
    if status_code == 412:
        candidate = (_provider_error_body_message(exc) or str(exc)).lower()
        if "suspended" in candidate or "spending limit" in candidate:
            return True

    detail = _clip_detail(str(exc) or exc.__class__.__name__).lower()
    return any(pattern in detail for pattern in _API_BUDGET_EXHAUSTED_PATTERNS)


def classify_runtime_failure(exc: Exception, source: str = "runtime") -> FailureClassification:
    """Classify an exception into shared retry semantics."""
    detail = _clip_detail(str(exc) or exc.__class__.__name__)
    status_code = _coerce_status_code(getattr(exc, "status_code", None))
    lowered = detail.lower()

    if is_api_budget_exhausted_error(exc):
        return FailureClassification(
            kind=TERMINAL_ERROR,
            domain="provider",
            reason="api_budget_exhausted",
            detail=detail,
            status_code=status_code,
        )

    if status_code in RETRYABLE_STATUS_CODES:
        return FailureClassification(
            kind=RECOVERABLE_ERROR,
            domain="provider",
            reason=f"http_{status_code}",
            detail=detail,
            status_code=status_code,
        )

    if status_code in {400, 401, 403, 404, 422}:
        return FailureClassification(
            kind=TERMINAL_ERROR,
            domain="provider",
            reason=f"http_{status_code}",
            detail=detail,
            status_code=status_code,
        )

    # Anthropic and OpenAI expose provider-agnostic APIConnectionError classes
    # that do not inherit Python's built-in ConnectionError. Their canonical
    # message is simply "Connection error.", so neither the isinstance check
    # nor the more specific reset/closed/aborted patterns below recognize them.
    # Keep this after status handling so request-specific 400/422 responses
    # remain terminal request failures even if their prose mentions a
    # connection error.
    if exc.__class__.__name__ == "APIConnectionError" or any(
        pattern in lowered for pattern in ("connection error", "connection-error")
    ):
        return FailureClassification(
            kind=RECOVERABLE_ERROR,
            domain="network",
            reason="connection_error",
            detail=detail,
            status_code=status_code,
        )

    for pattern, domain, reason in _TERMINAL_PATTERNS:
        if pattern in lowered:
            return FailureClassification(
                kind=TERMINAL_ERROR,
                domain=domain,
                reason=reason,
                detail=detail,
                status_code=status_code,
            )

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return FailureClassification(
            kind=RECOVERABLE_ERROR,
            domain="network",
            reason="timeout" if isinstance(exc, TimeoutError) else "connection_error",
            detail=detail,
            status_code=status_code,
        )

    for pattern, domain, reason in _RECOVERABLE_PATTERNS:
        if pattern in lowered:
            return FailureClassification(
                kind=RECOVERABLE_ERROR,
                domain=domain,
                reason=reason,
                detail=detail,
                status_code=status_code,
            )

    return FailureClassification(
        kind=TERMINAL_ERROR,
        domain=source,
        reason="unclassified",
        detail=detail,
        status_code=status_code,
    )


def format_failure_rationale(
    decision: str,
    classification: FailureClassification | None = None,
    detail: str = "",
) -> str:
    """Build a stable rationale string for failure decisions."""
    clipped_detail = _clip_detail(detail)
    if not classification:
        return f"[{decision}: {clipped_detail}]"

    status_suffix = ""
    if classification.status_code is not None:
        status_suffix = f" status={classification.status_code}"

    meta = (
        f"{classification.kind.lower()}/"
        f"{classification.domain}/"
        f"{classification.reason}"
        f"{status_suffix}"
    )
    if clipped_detail:
        return f"[{decision}: {meta}] {clipped_detail}"
    return f"[{decision}: {meta}]"


def judgment_failure_decision(
    stage: str,
    candidate_name: str,
    profile_url: str,
    error: Exception,
    path: str = "none",
    source: str = "judgment",
) -> OpusDecision:
    """Build a standardized JUDGMENT_FAILURE decision."""
    classification = classify_runtime_failure(error, source=source)
    return OpusDecision(
        stage=stage,
        decision=JUDGMENT_FAILURE,
        path=path,
        confidence=0.0,
        rationale=format_failure_rationale(
            JUDGMENT_FAILURE,
            classification=classification,
            detail=str(error),
        ),
        candidate_name=candidate_name,
        profile_url=profile_url,
    )


def parse_failure_decision(
    stage: str,
    candidate_name: str,
    profile_url: str,
    detail: str,
    reason: str = "parse_error",
    path: str = "none",
) -> OpusDecision:
    """Build a standardized PARSE_FAILURE decision."""
    classification = FailureClassification(
        kind=TERMINAL_ERROR,
        domain="parse",
        reason=reason,
        detail=_clip_detail(detail),
    )
    return OpusDecision(
        stage=stage,
        decision=PARSE_FAILURE,
        path=path,
        confidence=0.0,
        rationale=format_failure_rationale(
            PARSE_FAILURE,
            classification=classification,
            detail=detail,
        ),
        candidate_name=candidate_name,
        profile_url=profile_url,
    )


def _coerce_status_code(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _clip_detail(detail: str, limit: int = 240) -> str:
    cleaned = (detail or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."
