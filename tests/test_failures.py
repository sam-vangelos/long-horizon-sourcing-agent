from __future__ import annotations

from unittest.mock import patch

from shared.failures import (
    ApiBudgetExhaustedError,
    JUDGMENT_FAILURE,
    PARSE_FAILURE,
    RECOVERABLE_ERROR,
    TERMINAL_ERROR,
    classify_runtime_failure,
    is_api_budget_exhausted_error,
    judgment_failure_decision,
    parse_failure_decision,
)
from shared.llm_clients import _retry_with_backoff


class _StatusError(RuntimeError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class _TypedApiStatusError(RuntimeError):
    """Mimics the shape anthropic.APIStatusError actually exposes:
    status_code (int), type (parsed from body["error"]["type"]), and body
    (the raw decoded response). Used to prove is_api_budget_exhausted_error
    prefers this typed shape over the top-level exception message."""

    def __init__(self, message: str, status_code: int, error_type: str, body: dict):
        super().__init__(message)
        self.status_code = status_code
        self.type = error_type
        self.body = body


def test_classify_retryable_provider_status_code():
    classification = classify_runtime_failure(_StatusError("overloaded", 503), source="llm")
    assert classification.kind == RECOVERABLE_ERROR
    assert classification.domain == "provider"
    assert classification.reason == "http_503"
    assert classification.retryable is True


def test_classify_terminal_truncated_response():
    messages = [
        "Opus response truncated: stop_reason=max_tokens",
        "Fireworks response truncated: finish_reason=length. Increase max_tokens.",
    ]
    for message in messages:
        classification = classify_runtime_failure(RuntimeError(message), source="llm")
        assert classification.kind == TERMINAL_ERROR
        assert classification.domain == "provider"
        assert classification.reason == "truncated_response"
        assert classification.retryable is False


def test_classify_api_budget_exhausted_before_generic_http_400():
    exc = _StatusError(
        "Your credit balance is too low to access the Anthropic API. "
        "Please go to Plans & Billing to upgrade or purchase credits.",
        400,
    )

    classification = classify_runtime_failure(exc, source="llm")

    assert classification.kind == TERMINAL_ERROR
    assert classification.domain == "provider"
    assert classification.reason == "api_budget_exhausted"
    assert classification.retryable is False
    assert is_api_budget_exhausted_error(exc) is True
    assert is_api_budget_exhausted_error(ApiBudgetExhaustedError("credits exhausted")) is True


# ---------------------------------------------------------------------------
# P8.4(b): typed detection via the SDK's status_code/type/body shape, not
# message-copy parsing.
# ---------------------------------------------------------------------------


def test_typed_billing_error_type_detected_with_no_matching_message_text():
    """The SDK's dedicated billing_error type is a direct signal — this must
    be detected even when NEITHER the top-level exception message NOR the
    body message contains any of the known substring patterns. Proves
    detection is genuinely typed, not just checking a different string."""
    exc = _TypedApiStatusError(
        "Bad Request",
        400,
        "billing_error",
        {"error": {"type": "billing_error", "message": "Organization is over its monthly spend cap."}},
    )
    assert is_api_budget_exhausted_error(exc) is True


def test_typed_invalid_request_error_checks_body_message_not_top_level_message():
    """Real anthropic.APIStatusError instances carry the informative text in
    body["error"]["message"], not necessarily in the top-level exception
    message. The generic invalid_request_error type alone is too broad to
    trust without a body-message check."""
    exc = _TypedApiStatusError(
        "Please check your request and try again.",  # generic top-level message
        400,
        "invalid_request_error",
        {"error": {"type": "invalid_request_error", "message": "Your account has insufficient credits to complete this request."}},
    )
    assert is_api_budget_exhausted_error(exc) is True


def test_typed_invalid_request_error_without_matching_body_message_is_not_budget_exhausted():
    """A generic invalid_request_error whose body message doesn't match any
    known budget-exhaustion phrase must NOT be misclassified — this is the
    over-broad-type risk the body-message check exists to guard against."""
    exc = _TypedApiStatusError(
        "Bad Request",
        400,
        "invalid_request_error",
        {"error": {"type": "invalid_request_error", "message": "max_tokens must be greater than 0"}},
    )
    assert is_api_budget_exhausted_error(exc) is False


def test_classify_retryable_browser_disconnect():
    classification = classify_runtime_failure(
        RuntimeError("Target closed while reading browser page"),
        source="browser",
    )
    assert classification.kind == RECOVERABLE_ERROR
    assert classification.domain == "browser"
    assert classification.reason == "browser_disconnect"


def test_judgment_failure_decision_includes_shared_classification():
    decision = judgment_failure_decision(
        stage="full",
        candidate_name="Test Person",
        profile_url="/profile/test",
        error=_StatusError("provider overloaded", 503),
        source="judgment",
    )
    assert decision.decision == JUDGMENT_FAILURE
    assert decision.confidence == 0.0
    assert "recoverable_error/provider/http_503" in decision.rationale


def test_parse_failure_decision_includes_shared_classification():
    decision = parse_failure_decision(
        stage="facial",
        candidate_name="Test Person",
        profile_url="/profile/test",
        reason="invalid_decision",
        detail="decision='YOLO'",
    )
    assert decision.decision == PARSE_FAILURE
    assert decision.confidence == 0.0
    assert "terminal_error/parse/invalid_decision" in decision.rationale
    assert "decision='YOLO'" in decision.rationale


def test_retry_with_backoff_retries_recoverable_error():
    attempts = {"count": 0}

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _StatusError("provider overloaded", 503)
        return "ok"

    with patch("shared.llm_clients.random.uniform", return_value=0.0), patch("shared.llm_clients.time.sleep") as sleep:
        result = _retry_with_backoff(flaky_call, label="test")

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleep.call_count == 2


def test_retry_with_backoff_does_not_retry_terminal_error():
    attempts = {"count": 0}

    def truncated_call():
        attempts["count"] += 1
        raise RuntimeError("Opus response truncated: stop_reason=max_tokens")

    with patch("shared.llm_clients.random.uniform", return_value=0.0), patch("shared.llm_clients.time.sleep") as sleep:
        try:
            _retry_with_backoff(truncated_call, label="test")
        except RuntimeError as exc:
            assert "stop_reason=max_tokens" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    assert attempts["count"] == 1
    assert sleep.call_count == 0


_FIREWORKS_412_BODY_MESSAGE = (
    "Account samvangelos is suspended, possibly due to reaching the monthly "
    "spending limit or failure to pay past invoices. Please go to "
    "https://fireworks.ai/account/billing for more information."
)
_FIREWORKS_412_STR = (
    "Error code: 412 - {'error': {'message': '" + _FIREWORKS_412_BODY_MESSAGE + "', "
    "'param': None, 'code': 'PRECONDITION_FAILED', 'type': 'error'}}"
)


def test_fireworks_412_account_suspension_is_budget_exhaustion():
    """The REAL wire shape: the OpenAI SDK (Fireworks' transport) unwraps
    body["error"] before constructing APIStatusError, so .body is the FLAT
    inner dict. str(exc) is deliberately message-free here so the TYPED
    branch, not the str-fallback, is what this test proves (wave-1 review
    finding: the first version asserted an Anthropic-style nested body the
    OpenAI SDK never produces)."""
    exc = _TypedApiStatusError(
        "Error code: 412",
        status_code=412,
        error_type="error",
        body={
            "message": _FIREWORKS_412_BODY_MESSAGE,
            "param": None,
            "code": "PRECONDITION_FAILED",
            "type": "error",
        },
    )
    assert is_api_budget_exhausted_error(exc) is True
    classification = classify_runtime_failure(exc, source="llm")
    assert classification.kind == TERMINAL_ERROR
    assert classification.reason == "api_budget_exhausted"


def test_fireworks_412_nested_body_also_classifies():
    """The Anthropic nesting convention (body["error"]["message"]) must keep
    working too — the extractor reads both shapes."""
    exc = _TypedApiStatusError(
        "Error code: 412",
        status_code=412,
        error_type="error",
        body={
            "error": {
                "message": _FIREWORKS_412_BODY_MESSAGE,
                "code": "PRECONDITION_FAILED",
                "type": "error",
            }
        },
    )
    assert is_api_budget_exhausted_error(exc) is True


def test_fireworks_412_message_only_fallback():
    assert is_api_budget_exhausted_error(Exception(_FIREWORKS_412_STR)) is True


def test_plain_412_without_billing_language_is_not_budget():
    exc = _TypedApiStatusError(
        "Error code: 412 - precondition failed: stale request nonce",
        status_code=412,
        error_type="error",
        body={"error": {"message": "precondition failed: stale request nonce"}},
    )
    assert is_api_budget_exhausted_error(exc) is False


def test_generic_text_with_suspended_alone_is_not_budget():
    assert (
        is_api_budget_exhausted_error(
            Exception("profile page reported: account suspended")
        )
        is False
    )
