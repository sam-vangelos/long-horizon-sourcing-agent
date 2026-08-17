"""Safe external URL egress helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin

from shared.url_safety import check_url


@dataclass(frozen=True)
class SafeFetchResult:
    status: str
    final_url: str = ""
    body: str = ""
    content_type: str = ""
    reason: str = ""
    redirects_followed: int = 0


async def validate_public_url(url: str) -> tuple[bool, str]:
    return await check_url(url)


async def fetch_text_if_safe(
    *,
    session: Any,
    url: str,
    max_redirects: int = 5,
    allowed_content_substring: str = "html",
    on_event: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
) -> SafeFetchResult:
    safe, reason = await validate_public_url(url)
    if not safe:
        if on_event:
            maybe_awaitable = on_event("blocked", {"url": url, "reason": reason})
            if maybe_awaitable is not None:
                await maybe_awaitable
        return SafeFetchResult(status="blocked", final_url=url, reason=reason)

    target = url
    redirects_followed = 0
    for _ in range(max_redirects + 1):
        async with session.get(target, allow_redirects=False) as resp:
            if resp.status in {301, 302, 303, 307, 308}:
                location = resp.headers.get("Location")
                if not location:
                    return SafeFetchResult(
                        status="redirect_missing_location",
                        final_url=target,
                        reason="redirect missing location",
                        redirects_followed=redirects_followed,
                    )
                location = urljoin(target, location)
                safe, reason = await validate_public_url(location)
                if not safe:
                    if on_event:
                        maybe_awaitable = on_event(
                            "blocked_redirect",
                            {"url": target, "redirect_url": location, "reason": reason},
                        )
                        if maybe_awaitable is not None:
                            await maybe_awaitable
                    return SafeFetchResult(
                        status="blocked_redirect",
                        final_url=location,
                        reason=reason,
                        redirects_followed=redirects_followed,
                    )
                target = location
                redirects_followed += 1
                continue

            content_type = resp.headers.get("Content-Type", "")
            if allowed_content_substring not in content_type:
                return SafeFetchResult(
                    status="unsupported_content_type",
                    final_url=target,
                    content_type=content_type,
                    redirects_followed=redirects_followed,
                )
            return SafeFetchResult(
                status="ok",
                final_url=target,
                body=await resp.text(),
                content_type=content_type,
                redirects_followed=redirects_followed,
            )

    return SafeFetchResult(
        status="too_many_redirects",
        final_url=target,
        reason="too many redirects",
        redirects_followed=redirects_followed,
    )
