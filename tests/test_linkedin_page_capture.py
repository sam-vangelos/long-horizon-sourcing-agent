"""Tests for linkedin.page_capture — fail-soft page capture primitive."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from linkedin.page_capture import capture_page_state


def _make_browser(
    *,
    url: str = "https://www.linkedin.com/talent/search",
    title: str = "LinkedIn Recruiter",
    body_text: str = "Page body content for capture.",
    screenshot_raises: bool = False,
) -> MagicMock:
    browser = MagicMock()
    page = MagicMock()

    type(page).url = PropertyMock(return_value=url)
    page.title = AsyncMock(return_value=title)

    body_locator = MagicMock()
    body_locator.inner_text = AsyncMock(return_value=body_text)
    page.locator = MagicMock(return_value=body_locator)

    async def _screenshot(**kwargs: object) -> None:
        path = kwargs.get("path")
        if path is not None:
            path.write_bytes(b"\x89PNG\r\n")

    if screenshot_raises:
        page.screenshot = AsyncMock(side_effect=RuntimeError("screenshot failed"))
    else:
        page.screenshot = AsyncMock(side_effect=_screenshot)

    browser.page = page
    return browser


def test_capture_writes_meta_body_and_screenshot(tmp_path):
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        title="Recruiter Search",
        body_text="Candidate results body text.",
    )

    result = asyncio.run(
        capture_page_state(
            browser,
            tmp_path,
            reason="health_something_went_wrong",
            run_id=7,
        )
    )

    captures_dir = tmp_path / "captures"
    assert captures_dir.is_dir()
    capture_dirs = [p for p in captures_dir.iterdir() if p.is_dir()]
    assert len(capture_dirs) == 1

    capture_dir = capture_dirs[0]
    meta_path = capture_dir / "meta.json"
    body_path = capture_dir / "body.txt"
    screenshot_path = capture_dir / "screenshot.png"

    assert meta_path.is_file()
    assert body_path.is_file()
    assert screenshot_path.is_file()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["url"] == (
        "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    assert meta["title"] == "Recruiter Search"
    assert meta["reason"] == "health_something_went_wrong"
    assert meta["run_id"] == 7
    assert meta["timestamp"]

    assert body_path.read_text(encoding="utf-8") == "Candidate results body text."
    assert screenshot_path.read_bytes().startswith(b"\x89PNG")

    browser.page.screenshot.assert_awaited_once()
    screenshot_kwargs = browser.page.screenshot.await_args.kwargs
    assert screenshot_kwargs["path"] == screenshot_path
    assert screenshot_kwargs["full_page"] is False

    assert result == meta


def test_capture_returns_none_and_does_not_raise_when_page_is_dead(tmp_path):
    browser = MagicMock()
    type(browser).page = PropertyMock(side_effect=RuntimeError("Target closed"))

    result = asyncio.run(
        capture_page_state(browser, tmp_path, reason="health_login_redirect")
    )

    assert result is None


def test_capture_survives_screenshot_failure_and_still_writes_body(tmp_path):
    browser = _make_browser(
        body_text="Body survives screenshot failure.",
        screenshot_raises=True,
    )

    result = asyncio.run(
        capture_page_state(browser, tmp_path, reason="post_recovery")
    )

    assert result is not None

    captures_dir = tmp_path / "captures"
    capture_dir = next(p for p in captures_dir.iterdir() if p.is_dir())

    meta = json.loads((capture_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["reason"] == "post_recovery"
    assert (capture_dir / "body.txt").read_text(encoding="utf-8") == (
        "Body survives screenshot failure."
    )
    assert not (capture_dir / "screenshot.png").exists()
    assert result == meta


def test_capture_refuses_past_the_directory_cap(tmp_path):
    captures_dir = tmp_path / "captures"
    captures_dir.mkdir(parents=True)
    for index in range(201):
        (captures_dir / f"existing-{index:03d}").mkdir()

    browser = _make_browser()
    before_count = sum(1 for p in captures_dir.iterdir() if p.is_dir())

    result = asyncio.run(
        capture_page_state(browser, tmp_path, reason="should_not_write")
    )

    after_count = sum(1 for p in captures_dir.iterdir() if p.is_dir())

    assert result is None
    assert before_count == 201
    assert after_count == 201
