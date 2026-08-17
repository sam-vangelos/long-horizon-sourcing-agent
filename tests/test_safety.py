"""Production-safety boundary tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from shared.safety import LinkedInRecoveryService, fetch_text_if_safe


class _FakeResponse:
    def __init__(self, status: int, *, headers: dict[str, str] | None = None, body: str = ""):
        self.status = status
        self.headers = headers or {}
        self._body = body

    async def text(self) -> str:
        return self._body


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)

    def get(self, *_args, **_kwargs):
        return _FakeRequestContext(self._responses.pop(0))


def test_safe_fetch_blocks_unsafe_redirect():
    events: list[tuple[str, dict]] = []
    session = _FakeSession(
        [
            _FakeResponse(
                302,
                headers={"Location": "http://127.0.0.1/admin"},
            )
        ]
    )

    async def _run():
        return await fetch_text_if_safe(
            session=session,
            url="https://example.com",
            on_event=lambda event_type, payload: events.append((event_type, payload)),
        )

    with patch("shared.safety.egress.validate_public_url", new=AsyncMock(side_effect=[(True, "ok"), (False, "loopback")])):
        result = asyncio.run(_run())

    assert result.status == "blocked_redirect"
    assert events[0][0] == "blocked_redirect"


def test_linkedin_recovery_service_records_abandoned_recovery():
    coordinator = MagicMock()
    browser = SimpleNamespace(
        recover_from_target_crash=AsyncMock(return_value=False),
        disconnect=AsyncMock(return_value=None),
        connect=AsyncMock(side_effect=RuntimeError("chrome unavailable")),
        navigate_to_search=AsyncMock(return_value=None),
    )
    service = LinkedInRecoveryService(
        coordinator=coordinator,
        browser=browser,
        max_attempts=2,
        wait_seconds=0,
    )

    recovered = asyncio.run(service.recover(run_id=7, recovery_url="https://www.linkedin.com/talent/search"))

    assert recovered is False
    statuses = [call.kwargs["status"] for call in coordinator.record_browser_recovery_event.call_args_list]
    assert statuses == ["attempt", "failed", "attempt", "failed", "abandoned"]
