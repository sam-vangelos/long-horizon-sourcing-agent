"""CLO-243 / CLO-250 slice 2: LinkedIn sign-in walls are named, not crashed into.

The 2026-09-28 SPL run landed on LinkedIn's one-session-only page
(``/enterprise-authentication/sessions``), read it as a missing sidebar, and
spent its whole absorb budget starting new sessions into the same wall
before a traceback killed it. These tests pin the replacement contract:

- ``classify_auth_wall`` names the wall from the real capture and from
  synthetic URLs for each variant, and never names a Recruiter surface;
- every seam (navigate_to_search, check_and_recover, the health classifier,
  recovery, the geography gate) raises or reports the typed condition and
  touches nothing on the page;
- through the production call path, a wall ends the day cycle after exactly
  one session, as ``recruiter_reauth_required``, with zero absorb-budget use.

``linkedin.auth_wall`` is imported inside the tests, not at module scope, so
the production-path tests fail on BEHAVIOUR (sessions absorbed into the wall)
against code that predates the module, rather than on a collection error.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "auth_wall"
# The real 2026-09-28 wall URL with the seat's accountId/appInstanceId removed.
WALL_URL = json.loads((FIXTURES / "session_conflict_meta.json").read_text())["url"]
WALL_BODY = (FIXTURES / "session_conflict_body.txt").read_text()
SEARCH_URL = "https://www.linkedin.com/talent/hire/9000000003/discover/recruiterSearch"
SESSION_CONFLICT_SENTENCE = (
    "LinkedIn Recruiter says another sign-in to this seat is active and only "
    "one session is allowed; Cloris stopped without choosing a session — open "
    "Recruiter, choose the session, then relaunch with resume."
)


def _aw():
    import linkedin.auth_wall as auth_wall

    return auth_wall


# ---------------------------------------------------------------------------
# A fake Recruiter page that records every interaction that could change it.
# ---------------------------------------------------------------------------


class _FakeLocator:
    def __init__(self, page: "_FakePage", selector: str):
        self._page = page
        self._selector = selector

    @property
    def first(self) -> "_FakeLocator":
        return self

    def nth(self, _index: int) -> "_FakeLocator":
        return self

    def locator(self, selector: str) -> "_FakeLocator":
        return _FakeLocator(self._page, f"{self._selector} {selector}")

    def filter(self, **_kwargs) -> "_FakeLocator":
        return self

    async def is_visible(self, timeout=None) -> bool:
        return False

    async def count(self) -> int:
        return 0

    async def inner_text(self, timeout=None) -> str:
        if self._selector == "body":
            return self._page.body
        raise TimeoutError(f"locator {self._selector!r} timed out")

    async def all_inner_texts(self) -> list[str]:
        return []

    async def _touch(self, kind: str, *args, **kwargs):
        self._page.touches.append((kind, self._selector))

    async def click(self, *args, **kwargs):
        await self._touch("click")

    async def hover(self, *args, **kwargs):
        await self._touch("hover")

    async def fill(self, *args, **kwargs):
        await self._touch("fill")

    async def press(self, *args, **kwargs):
        await self._touch("press")

    async def type(self, *args, **kwargs):
        await self._touch("type")

    async def evaluate(self, *args, **kwargs):
        await self._touch("evaluate")


class _FakeInput:
    def __init__(self, page: "_FakePage", kind: str):
        self._page = page
        self._kind = kind

    def __getattr__(self, name: str):
        async def _record(*_args, **_kwargs):
            self._page.touches.append((self._kind, name))

        return _record


class _FakePage:
    """Stays wherever it starts, like a tab redirected to a wall.

    ``goto`` is recorded but does not move the page: LinkedIn's redirect
    lands every navigation back on the wall, which is what the 28 Sep log
    shows for nine sessions in a row.
    """

    def __init__(self, url: str, body: str = ""):
        self.url = url
        self.body = body
        self.gotos: list[str] = []
        self.touches: list[tuple[str, str]] = []
        self.keyboard = _FakeInput(self, "keyboard")
        self.mouse = _FakeInput(self, "mouse")

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self, selector)

    async def goto(self, url, **_kwargs):
        self.gotos.append(url)

    async def wait_for_timeout(self, _ms):
        return None

    async def title(self) -> str:
        return "LinkedIn"

    async def screenshot(self, **_kwargs):
        return None

    async def evaluate(self, *_args, **_kwargs):
        self.touches.append(("evaluate", "page"))

    async def reload(self, *_args, **_kwargs):
        self.touches.append(("reload", "page"))


def _browser_on(page: _FakePage, state_dir: Path | None):
    from linkedin.browser import LinkedInBrowser
    from shared.governor import UNGOVERNED_FOR_TESTS

    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS, state_dir=state_dir)
    browser._page = page
    return browser


def _capture_reasons(state_dir: Path) -> list[str]:
    captures = state_dir / "captures"
    if not captures.is_dir():
        return []
    return sorted(
        json.loads((entry / "meta.json").read_text())["reason"]
        for entry in captures.iterdir()
        if (entry / "meta.json").is_file()
    )


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


def test_real_capture_classifies_as_session_conflict():
    classify = _aw().classify_auth_wall

    assert classify(WALL_URL, WALL_BODY) == "session_conflict"
    # The URL alone names it (check_and_recover reads no body)...
    assert classify(WALL_URL, None) == "session_conflict"
    # ...and so does the body alone, for a wall served under another path.
    assert classify("https://www.linkedin.com/some/other/page", WALL_BODY) == (
        "session_conflict"
    )
    assert classify("", WALL_BODY) == "session_conflict"


@pytest.mark.parametrize(
    ("url", "variant"),
    [
        (
            "https://www.linkedin.com/enterprise-authentication/sessions?accountId=1&appInstanceId=2&application=recruiter",
            "session_conflict",
        ),
        (
            "https://www.linkedin.com/enterprise-authentication/sso?application=recruiter",
            "enterprise_reauth",
        ),
        (
            "https://www.linkedin.com/enterprise-authentication/redirect?appInstanceId=9",
            "enterprise_reauth",
        ),
        ("https://www.linkedin.com/login", "login_redirect"),
        (
            "https://www.linkedin.com/uas/login?session_redirect=%2Ftalent%2Fsearch",
            "login_redirect",
        ),
        ("https://www.linkedin.com/checkpoint/challenge/AgEx", "checkpoint"),
        ("https://www.linkedin.com/checkpoint/lg/login-submit", "checkpoint"),
        ("https://www.linkedin.com/challenge/verify", "checkpoint"),
    ],
)
def test_each_variant_classifies_from_its_url(url, variant):
    assert _aw().classify_auth_wall(url, None) == variant


@pytest.mark.parametrize(
    "url",
    [
        SEARCH_URL,
        SEARCH_URL + "?searchContextId=abc&start=0",
        # A redirect target in the QUERY is not where the page is.
        SEARCH_URL + "?next=%2Flogin&ref=/enterprise-authentication/sessions",
        "https://www.linkedin.com/talent/search",
        "https://www.linkedin.com/talent/profile/AEMAAAB",
        "https://www.linkedin.com/feed/",
        "about:blank",
        "",
    ],
)
def test_ordinary_pages_are_not_walls(url):
    assert _aw().classify_auth_wall(url, None) is None


def test_recruiter_surface_is_never_a_wall_whatever_its_body_says():
    """On /talent/ pages the body is candidate copy; no phrase there may end
    a day cycle."""
    assert _aw().classify_auth_wall(SEARCH_URL, WALL_BODY) is None


def test_auth_wall_error_first_line_names_the_reason_variant_and_sentence():
    aw = _aw()
    error = aw.AuthWallError("session_conflict", url=WALL_URL)

    assert isinstance(error, RuntimeError)
    assert error.variant == "session_conflict"
    assert str(error).splitlines()[0] == (
        f"recruiter_reauth_required: session_conflict: {SESSION_CONFLICT_SENTENCE}"
    )
    # The seat's identifiers never reach a message, a record or a console.
    assert "accountId" not in str(error)
    assert "appInstanceId" not in str(error)
    assert error.url == "https://www.linkedin.com/enterprise-authentication/sessions"


@pytest.mark.parametrize(
    "variant", ["session_conflict", "enterprise_reauth", "login_redirect", "checkpoint"]
)
def test_every_variant_has_a_plain_sentence(variant):
    error = _aw().AuthWallError(variant)
    first = str(error).splitlines()[0]
    assert first.startswith(f"recruiter_reauth_required: {variant}: ")
    assert "Cloris stopped" in first


def test_wall_is_found_through_wrappers_and_legacy_text():
    aw = _aw()
    wall = aw.AuthWallError("enterprise_reauth")

    assert aw.auth_wall_variant_of(wall) == "enterprise_reauth"
    try:
        try:
            raise wall
        except RuntimeError as inner:
            raise ValueError("page-local recovery refused") from inner
    except ValueError as outer:
        assert aw.auth_wall_variant_of(outer) == "enterprise_reauth"
    try:
        try:
            raise wall
        except RuntimeError:
            raise ValueError("recovery refused unhealthy surface")
    except ValueError as outer:
        assert aw.auth_wall_variant_of(outer) == "enterprise_reauth"

    # The exact text the 28 Sep run died on (URL truncated at 120 chars).
    legacy = RuntimeError(
        "navigate_to_search failed: sidebar elements not found after 16s. "
        "Page URL: https://www.linkedin.com/enterprise-authentication/sessions"
        "?accountId=REDACTED&appInstanceId=REDACTED&application=recr"
    )
    assert aw.auth_wall_variant_of(legacy) == "session_conflict"
    # A lost type keeps its name through the message prefix.
    assert aw.auth_wall_variant_of(
        RuntimeError("page-local recovery refused unhealthy surface: "
                     "recruiter_reauth_required: checkpoint: x")
    ) == "checkpoint"
    # search_surface's recovery refusal quotes the health state by name.
    assert aw.auth_wall_variant_of(
        RuntimeError("page-local recovery refused unhealthy surface: "
                     "recruiter_reauth_required")
    ) == "enterprise_reauth"

    generic = RuntimeError(
        "navigate_to_search failed: sidebar elements not found after 16s. "
        f"Page URL: {SEARCH_URL}"
    )
    assert aw.auth_wall_variant_of(generic) is None
    assert aw.auth_wall_variant_of(None) is None


# ---------------------------------------------------------------------------
# Browser seams
# ---------------------------------------------------------------------------


def test_navigate_to_search_names_the_wall_captures_it_and_touches_nothing(tmp_path):
    aw = _aw()
    page = _FakePage(WALL_URL, WALL_BODY)
    browser = _browser_on(page, tmp_path)

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(browser.navigate_to_search(SEARCH_URL))

    assert raised.value.variant == "session_conflict"
    assert _capture_reasons(tmp_path) == ["auth_wall_session_conflict"]
    # One navigation (the one that landed on the wall), then reads only.
    assert page.gotos == [SEARCH_URL]
    assert page.touches == []


def test_navigate_to_search_on_a_non_wall_page_keeps_the_generic_failure(tmp_path):
    aw = _aw()
    page = _FakePage(SEARCH_URL, "Some unrelated recruiter copy")
    browser = _browser_on(page, tmp_path)

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(browser.navigate_to_search(SEARCH_URL))

    assert not isinstance(raised.value, aw.AuthWallError)
    assert str(raised.value).startswith(
        "navigate_to_search failed: sidebar elements not found after 16s."
    )
    assert _capture_reasons(tmp_path) == ["navigate_sidebar_missing"]


@pytest.mark.parametrize(
    ("url", "variant"),
    [
        ("https://www.linkedin.com/login", "login_redirect"),
        (
            "https://www.linkedin.com/uas/login?session_redirect=/talent/search",
            "login_redirect",
        ),
        (WALL_URL, "session_conflict"),
        (
            "https://www.linkedin.com/enterprise-authentication/sso?application=recruiter",
            "enterprise_reauth",
        ),
    ],
)
def test_check_and_recover_raises_the_typed_wall(tmp_path, url, variant):
    aw = _aw()
    page = _FakePage(url)
    browser = _browser_on(page, tmp_path)

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(browser.check_and_recover())

    assert raised.value.variant == variant
    assert _capture_reasons(tmp_path) == [f"auth_wall_{variant}"]
    assert page.touches == []
    assert page.gotos == []


def test_check_and_recover_leaves_checkpoints_to_the_detection_backoff(tmp_path):
    """A /checkpoint/ is the health classifier's blocked_or_rate_limited trip
    (a six-hour detection backoff); check_and_recover must not pre-empt it."""
    page = _FakePage("https://www.linkedin.com/checkpoint/challenge/AgEx")
    browser = _browser_on(page, tmp_path)

    assert asyncio.run(browser.check_and_recover()) is False
    assert page.touches == []


def test_check_and_recover_on_a_recruiter_page_is_unchanged(tmp_path):
    page = _FakePage(SEARCH_URL)
    browser = _browser_on(page, tmp_path)

    assert asyncio.run(browser.check_and_recover()) is False
    assert _capture_reasons(tmp_path) == []


# ---------------------------------------------------------------------------
# Recruiter recovery
# ---------------------------------------------------------------------------


def _recovery_browser(url: str, body: str, state_dir: Path | None):
    from unittest.mock import AsyncMock

    page = _FakePage(url, body)
    return SimpleNamespace(
        page=page,
        _state_dir=state_dir,
        _project_id="9000000003",
        _governor=None,
        refresh_active_tab=AsyncMock(return_value=True),
        disconnect=AsyncMock(),
        connect=AsyncMock(),
        _bind_existing_recruiter_page=AsyncMock(return_value=True),
        navigate_to_search=AsyncMock(),
    )


def test_health_classifier_names_the_wall_instead_of_unclassified(tmp_path):
    from linkedin.recruiter_recovery import RECRUITER_HEALTH_STATES, detect_recruiter_health

    browser = _recovery_browser(WALL_URL, WALL_BODY, tmp_path)

    assert asyncio.run(detect_recruiter_health(browser)) == "recruiter_reauth_required"
    assert "recruiter_reauth_required" in RECRUITER_HEALTH_STATES
    assert _capture_reasons(tmp_path) == ["auth_wall_session_conflict"]


def test_health_classifier_keeps_checkpoint_and_login_classifications(tmp_path):
    from linkedin.recruiter_recovery import detect_recruiter_health

    checkpoint = _recovery_browser(
        "https://www.linkedin.com/checkpoint/challenge/AgEx", "", tmp_path
    )
    login = _recovery_browser("https://www.linkedin.com/login", "", tmp_path)
    search = _recovery_browser(SEARCH_URL, WALL_BODY, tmp_path)

    assert asyncio.run(detect_recruiter_health(checkpoint)) == "blocked_or_rate_limited"
    assert asyncio.run(detect_recruiter_health(login)) == "logged_out"
    assert asyncio.run(detect_recruiter_health(search)) == "healthy"


def test_recovery_refuses_to_refresh_or_navigate_into_a_wall(tmp_path):
    from linkedin.recruiter_recovery import (
        RecruiterRecoverySnapshot,
        recover_recruiter_context,
    )

    aw = _aw()
    browser = _recovery_browser(WALL_URL, WALL_BODY, tmp_path)
    snapshot = RecruiterRecoverySnapshot(search_url=SEARCH_URL, project_id="9000000003")

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(recover_recruiter_context(browser, snapshot))

    assert raised.value.variant == "session_conflict"
    browser.refresh_active_tab.assert_not_awaited()
    browser.disconnect.assert_not_awaited()
    browser.connect.assert_not_awaited()
    browser.navigate_to_search.assert_not_awaited()
    assert browser.page.touches == []


def test_recovery_stops_when_a_wall_appears_mid_recovery(tmp_path):
    """A refresh that lands on the wall must not be followed by a
    navigation back into it (and a swallowed failure, and another try)."""
    from linkedin.recruiter_recovery import (
        RecruiterRecoverySnapshot,
        recover_recruiter_context,
    )

    aw = _aw()
    browser = _recovery_browser("about:blank", "", tmp_path)

    async def _refresh_lands_on_wall():
        browser.page.url = WALL_URL
        browser.page.body = WALL_BODY
        return True

    browser.refresh_active_tab.side_effect = _refresh_lands_on_wall
    snapshot = RecruiterRecoverySnapshot(search_url=SEARCH_URL, project_id="9000000003")

    with pytest.raises(aw.AuthWallError):
        asyncio.run(recover_recruiter_context(browser, snapshot, max_attempts=3))

    assert browser.refresh_active_tab.await_count == 1
    browser.navigate_to_search.assert_not_awaited()


# ---------------------------------------------------------------------------
# Geography gate
# ---------------------------------------------------------------------------


def _gate(browser, tmp_path):
    from linkedin.geography_gate import GeographyGateDeps, GeographyGateService

    brief = SimpleNamespace(
        permanent_filters={"Location": "Greater Seattle Area"},
        geography_source="operator",
        id="test-brief",
    )
    return GeographyGateService(
        GeographyGateDeps(
            get_browser=lambda: browser,
            get_brief_obj=lambda: brief,
            log_path=tmp_path / "run_log.jsonl",
            stats={},
        )
    )


def test_geography_gate_reraises_a_wall_never_wraps_it(tmp_path):
    from linkedin.orchestrator import GeographyRegimeError

    aw = _aw()
    wall = aw.AuthWallError("session_conflict", url=WALL_URL)

    async def _apply(*_args, **_kwargs):
        raise wall

    browser = SimpleNamespace(page=_FakePage(WALL_URL, WALL_BODY), _state_dir=None)
    browser.apply_location_filter = _apply

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(_gate(browser, tmp_path)._apply_session_location_filter())

    assert raised.value is wall
    assert not isinstance(raised.value, GeographyRegimeError)


def test_geography_gate_names_a_wall_behind_a_generic_failure(tmp_path):
    """The real apply path on a wall page fails in require_recruiter_tab
    with a generic RuntimeError; the gate must read the live page and raise
    the typed wall, not the terminal 'fix your facet names' stop (R4)."""
    from linkedin.orchestrator import GeographyRegimeError

    aw = _aw()
    page = _FakePage(WALL_URL, WALL_BODY)
    browser = _browser_on(page, tmp_path)

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(_gate(browser, tmp_path)._apply_session_location_filter())

    assert raised.value.variant == "session_conflict"
    assert not isinstance(raised.value, GeographyRegimeError)
    assert page.touches == []


def test_geography_gate_keeps_the_regime_error_on_a_normal_page(tmp_path):
    from linkedin.orchestrator import GeographyRegimeError

    async def _apply(*_args, **_kwargs):
        raise ValueError("typeahead editor never opened")

    browser = SimpleNamespace(page=_FakePage(SEARCH_URL, "candidate copy"), _state_dir=None)
    browser.apply_location_filter = _apply

    with pytest.raises(GeographyRegimeError):
        asyncio.run(_gate(browser, tmp_path)._apply_session_location_filter())


# ---------------------------------------------------------------------------
# Session classification backstop
# ---------------------------------------------------------------------------


def test_session_error_shutdown_maps_every_wall_shape_to_the_stable_reason(monkeypatch):
    import linkedin.session_orchestrator as so
    from shared.cooldown import ShutdownKind

    aw = _aw()
    legacy = RuntimeError(
        "navigate_to_search failed: sidebar elements not found after 16s. "
        "Page URL: https://www.linkedin.com/enterprise-authentication/sessions?accountId=X"
    )
    for exc in (aw.AuthWallError("session_conflict"), legacy):
        assert so._session_error_shutdown(exc) == (
            "recruiter_reauth_required",
            ShutdownKind.ERROR,
        )
        assert so._classify_session_exception(exc) == "recruiter_reauth_required"

    # Classified outside the guarded orchestrator import, like the regime
    # errors: a classifier fault must not demote it to absorbable free text.
    monkeypatch.setitem(sys.modules, "linkedin.orchestrator", None)
    assert so._session_error_shutdown(aw.AuthWallError("checkpoint"))[0] == (
        "recruiter_reauth_required"
    )

    monkeypatch.setattr(so.config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(so.config, "LINKEDIN_SESSION_ERROR_RETRIES", 5)
    assert so._should_retry_session_error(
        "recruiter_reauth_required", consecutive_error_resumes=0
    ) is False


def test_generic_session_errors_keep_their_generic_reason():
    import linkedin.session_orchestrator as so

    reason, _kind = so._session_error_shutdown(
        RuntimeError(f"navigate_to_search failed: sidebar elements not found after 16s. Page URL: {SEARCH_URL}")
    )
    assert reason.startswith("error: navigate_to_search failed")


# ---------------------------------------------------------------------------
# The production call path: day cycle -> session -> pipeline -> seam
# ---------------------------------------------------------------------------


@pytest.fixture
def day_cycle(monkeypatch, tmp_path):
    """Run the real day cycle and real ``_run_sourcing_session`` against a
    fake Pipeline whose ``run_full`` is supplied per test. Governor state,
    the browser lock, network probes and sleeps are all isolated."""
    import linkedin.orchestrator as orchestrator
    import linkedin.session_orchestrator as so
    from shared import cooldown
    from shared.runtime_state import RuntimeStateLock

    recorded: dict = {
        "pipelines": 0,
        "session_end_reasons": [],
        "forced_backoffs": [],
        "network_probes": 0,
        "error_backoffs": 0,
        "dormant_samples": 0,
    }
    behaviours: list = []

    class FakeGovernor:
        shutdown_reason = None

        def can_start_session(self, session_type="linkedin_sourcing"):
            return True, "ok"

        def start_session(self, session_duration_seconds=None):
            return None

        def end_session(self):
            return {"profile_opens_session": 0}

        def status_line(self):
            return "status"

    class FakePipeline:
        @staticmethod
        def _validate_judgment_runtime_configuration():
            return None

        def __init__(self, **_kwargs):
            recorded["pipelines"] += 1
            index = recorded["pipelines"] - 1
            self.stats = {}
            self._behaviour = behaviours[min(index, len(behaviours) - 1)]
            self.browser = None

        async def run_full(self, resume=False, restart_string_id=None, restart_string_ids=None):
            await self._behaviour(self)

    async def fake_network_is_reachable():
        recorded["network_probes"] += 1
        return True

    def fake_error_backoff(_attempt):
        recorded["error_backoffs"] += 1
        return 0.0

    def fake_dormant():
        recorded["dormant_samples"] += 1
        return 0.0

    def fake_record_session_end(**kwargs):
        recorded["session_end_reasons"].append(kwargs["reason"])

    monkeypatch.setattr(cooldown, "GOVERNOR_DIR", tmp_path / "governor")
    monkeypatch.setattr(cooldown, "DAILY_STATS_FILE", tmp_path / "governor" / "daily_stats.json")
    monkeypatch.setattr(cooldown, "SESSIONS_LOG", tmp_path / "governor" / "sessions.jsonl")
    monkeypatch.setattr(cooldown, "record_session_start", lambda session_type: 42)
    monkeypatch.setattr(cooldown, "get_sessions_recorded_today", lambda session_type=None: 1)
    monkeypatch.setattr(cooldown, "get_profile_opens_24h", lambda: 0)
    monkeypatch.setattr(cooldown, "record_session_end", fake_record_session_end)
    monkeypatch.setattr(cooldown, "get_active_backoff", lambda: None)
    monkeypatch.setattr(
        cooldown,
        "record_forced_backoff",
        lambda *args, **kwargs: recorded["forced_backoffs"].append((args, kwargs)),
    )
    monkeypatch.setattr(orchestrator, "Pipeline", FakePipeline)
    monkeypatch.setattr(so, "SessionGovernor", FakeGovernor)
    monkeypatch.setattr(
        so,
        "_linkedin_browser_lock",
        lambda: RuntimeStateLock(
            tmp_path, filename="recruiter-browser.lock", resource_name="test browser"
        ),
    )
    monkeypatch.setattr(so, "_network_is_reachable", fake_network_is_reachable)
    monkeypatch.setattr(so, "_sample_session_duration", lambda: 16666.0)
    monkeypatch.setattr(so, "_sample_error_backoff", fake_error_backoff)
    monkeypatch.setattr(so, "_sample_dormant_duration", fake_dormant)
    monkeypatch.setattr(so, "_resume_has_pending_work", lambda *_a, **_k: True)
    monkeypatch.setattr(so, "_assert_resume_target_exists", lambda *_a, **_k: None)
    monkeypatch.setattr(so.signal, "signal", lambda *_a, **_k: None)
    monkeypatch.setattr(so.config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(so.config, "LINKEDIN_SESSION_ERROR_RETRIES", 5)

    def run(*session_behaviours, multi_session: bool = True) -> dict:
        behaviours.extend(session_behaviours)
        try:
            asyncio.run(
                so.run_day_cycle(
                    brief_path="brief.json",
                    output_dir=None,
                    multi_session=multi_session,
                    resume=True,
                )
            )
        except BaseException as exc:  # noqa: BLE001 — the test inspects it
            recorded["raised"] = exc
        return recorded

    return run


def _navigate_into(tmp_path: Path, url: str, body: str):
    async def _behaviour(pipeline):
        page = _FakePage(url, body)
        pipeline.browser = _browser_on(page, tmp_path)
        pipeline.page = page
        await pipeline.browser.navigate_to_search(SEARCH_URL)

    return _behaviour


def _apply_geography_on(tmp_path: Path, url: str, body: str):
    async def _behaviour(pipeline):
        page = _FakePage(url, body)
        pipeline.browser = _browser_on(page, tmp_path)
        pipeline.page = page
        await _gate(pipeline.browser, tmp_path)._apply_session_location_filter()

    return _behaviour


async def _completes(_pipeline):
    return None


def _assert_stopped_on_first_sight(recorded: dict, out: str) -> None:
    assert recorded["pipelines"] == 1, (
        f"{recorded['pipelines']} sessions were started into the wall; "
        f"session end reasons: {recorded['session_end_reasons']}"
    )
    assert recorded["session_end_reasons"] == ["recruiter_reauth_required"]
    aw = _aw()
    assert isinstance(recorded.get("raised"), aw.AuthWallError)
    assert recorded["raised"].variant == "session_conflict"
    # Zero absorb-budget use: no absorb, no backoff, no dormant gap, no probe.
    assert "Session error absorbed" not in out
    assert recorded["error_backoffs"] == 0
    assert recorded["dormant_samples"] == 0
    assert recorded["network_probes"] == 0
    # Not the six-hour detection backoff.
    assert recorded["forced_backoffs"] == []
    # One plain operator line carrying the variant's sentence.
    assert out.count(SESSION_CONFLICT_SENTENCE) == 1
    assert "Session error:" not in out


def test_wall_under_navigate_to_search_ends_the_day_cycle_after_one_session(
    day_cycle, tmp_path, capsys
):
    recorded = day_cycle(
        _navigate_into(tmp_path, WALL_URL, WALL_BODY),
        _completes,
    )

    _assert_stopped_on_first_sight(recorded, capsys.readouterr().out)
    assert _capture_reasons(tmp_path) == ["auth_wall_session_conflict"]


def test_wall_under_the_geography_gate_ends_the_day_cycle_after_one_session(
    day_cycle, tmp_path, capsys
):
    recorded = day_cycle(
        _apply_geography_on(tmp_path, WALL_URL, WALL_BODY),
        _completes,
    )

    _assert_stopped_on_first_sight(recorded, capsys.readouterr().out)


def test_single_session_mode_also_names_the_wall(day_cycle, tmp_path, capsys):
    recorded = day_cycle(
        _navigate_into(tmp_path, WALL_URL, WALL_BODY),
        multi_session=False,
    )

    _assert_stopped_on_first_sight(recorded, capsys.readouterr().out)


def test_generic_sidebar_missing_on_a_normal_page_is_still_absorbed(
    day_cycle, tmp_path, capsys
):
    """Today's behaviour for a non-wall failure is unchanged: absorbed once
    under its family budget, then the next session runs."""
    recorded = day_cycle(
        _navigate_into(tmp_path, SEARCH_URL, "candidate copy"),
        _completes,
    )

    out = capsys.readouterr().out
    assert "raised" not in recorded
    assert recorded["pipelines"] == 2
    assert recorded["session_end_reasons"][0].startswith(
        "error: navigate_to_search failed"
    )
    assert "Session error absorbed (RuntimeError 1/5)" in out


# ---------------------------------------------------------------------------
# Review fixes: variant precedence, checkpoint backoff, the remaining routes
# by which a wall reached a generic or misnamed stop.
# ---------------------------------------------------------------------------

CHECKPOINT_URL = "https://www.linkedin.com/checkpoint/challenge/AgEx"


class _RecordingGovernor:
    """A test governor that records forced backoffs and checks nothing."""

    def __init__(self):
        self.forced: list[str] = []

    def check_profile_open_or_raise(self) -> None:
        return None

    def record_profile_open(self) -> None:
        return None

    def force_backoff(self, reason: str, **_kwargs) -> None:
        self.forced.append(reason)


def _governed_browser_on(page: _FakePage, state_dir: Path | None, governor):
    from linkedin.browser import LinkedInBrowser

    browser = LinkedInBrowser(governor=governor, state_dir=state_dir)
    browser._page = page
    return browser


def test_a_bare_name_does_not_hide_the_specific_variant_behind_it():
    """search_surface's page-local refusal quotes the health state with no
    variant; the AuthWallError it was raised over still names the wall, and
    the operator must be told to choose the session, not to sign in."""
    aw = _aw()
    try:
        try:
            raise aw.AuthWallError("session_conflict", url=WALL_URL)
        except RuntimeError:
            raise RuntimeError(
                "page-local recovery refused unhealthy surface: "
                "recruiter_reauth_required"
            )
    except RuntimeError as outer:
        assert aw.auth_wall_variant_of(outer) == "session_conflict"
        assert aw.auth_wall_operator_line(outer).startswith(SESSION_CONFLICT_SENTENCE)


def test_checkpoint_under_navigate_to_search_stops_and_keeps_the_detection_backoff(tmp_path):
    """A /checkpoint/ is a detection signal everywhere else (the health
    classifier's blocked_or_rate_limited, a six-hour forced backoff). Named
    under navigate_to_search it must still trip that backoff, so a relaunch
    cannot re-probe a flagged seat at once."""
    aw = _aw()
    governor = _RecordingGovernor()
    page = _FakePage(CHECKPOINT_URL, "Let's do a quick security check")
    browser = _governed_browser_on(page, tmp_path, governor)

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(browser.navigate_to_search(SEARCH_URL))

    assert raised.value.variant == "checkpoint"
    assert governor.forced == ["blocked_or_rate_limited"]
    assert _capture_reasons(tmp_path) == ["auth_wall_checkpoint"]
    assert page.touches == []


def test_sign_in_walls_do_not_trip_the_detection_backoff(tmp_path):
    aw = _aw()
    governor = _RecordingGovernor()
    page = _FakePage(WALL_URL, WALL_BODY)
    browser = _governed_browser_on(page, tmp_path, governor)

    with pytest.raises(aw.AuthWallError):
        asyncio.run(browser.navigate_to_search(SEARCH_URL))

    assert governor.forced == []


def test_checkpoint_under_the_geography_gate_keeps_the_detection_backoff(tmp_path):
    from linkedin.orchestrator import GeographyRegimeError

    aw = _aw()
    governor = _RecordingGovernor()

    async def _apply(*_args, **_kwargs):
        raise RuntimeError("require_recruiter_tab: not a Recruiter tab")

    browser = SimpleNamespace(
        page=_FakePage(CHECKPOINT_URL, ""), _state_dir=None, _governor=governor
    )
    browser.apply_location_filter = _apply

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(_gate(browser, tmp_path)._apply_session_location_filter())

    assert raised.value.variant == "checkpoint"
    assert not isinstance(raised.value, GeographyRegimeError)
    assert governor.forced == ["blocked_or_rate_limited"]


@pytest.mark.parametrize("source", ["operator", "preflight"])
def test_geography_gate_names_a_wall_when_apply_returns_false(tmp_path, source):
    """apply_location_filter returns False (no editor, unreadable chips) on a
    page that is no longer Recruiter; that must be the wall, never a
    retryable GeographyRegimeError that starts a second session into it, and
    never a model resolution typed into the wall page."""
    from linkedin.orchestrator import GeographyRegimeError

    aw = _aw()

    async def _apply(*_args, **_kwargs):
        return False

    browser = SimpleNamespace(page=_FakePage(WALL_URL, WALL_BODY), _state_dir=None)
    browser.apply_location_filter = _apply
    gate = _gate(browser, tmp_path)
    gate.deps.get_brief_obj().geography_source = source
    resolutions: list = []

    async def _resolve(values):
        resolutions.append(values)
        return list(values)

    gate._resolve_and_reapply_geography = _resolve

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(gate._apply_session_location_filter())

    assert raised.value.variant == "session_conflict"
    assert not isinstance(raised.value, GeographyRegimeError)
    assert resolutions == []
    assert browser.page.touches == []


def test_geography_gate_apply_false_on_a_normal_page_keeps_the_retryable_error(tmp_path):
    from linkedin.orchestrator import GeographyRegimeError

    async def _apply(*_args, **_kwargs):
        return False

    browser = SimpleNamespace(page=_FakePage(SEARCH_URL, "candidate copy"), _state_dir=None)
    browser.apply_location_filter = _apply

    with pytest.raises(GeographyRegimeError) as raised:
        asyncio.run(_gate(browser, tmp_path)._apply_session_location_filter())

    assert raised.value.retryable is True


def test_geography_gate_passes_through_an_error_that_quotes_a_wall_url(tmp_path):
    """The live page has moved on (not a wall now), but the failure itself
    quotes the wall URL: the error is passed through, never wrapped into the
    terminal 'fix your facet names' stop."""
    from linkedin.orchestrator import GeographyRegimeError

    aw = _aw()
    quoted = RuntimeError(
        "apply_location_filter: navigation failed. Page URL: "
        "https://www.linkedin.com/enterprise-authentication/sessions?application=recruiter"
    )

    async def _apply(*_args, **_kwargs):
        raise quoted

    browser = SimpleNamespace(page=_FakePage(SEARCH_URL, "candidate copy"), _state_dir=None)
    browser.apply_location_filter = _apply

    with pytest.raises(Exception) as raised:
        asyncio.run(_gate(browser, tmp_path)._apply_session_location_filter())

    assert raised.value is quoted
    assert not isinstance(raised.value, GeographyRegimeError)
    assert aw.auth_wall_variant_of(raised.value) == "session_conflict"


@pytest.mark.parametrize("chips_raise", [False, True])
def test_chip_verification_names_a_wall_instead_of_a_regime_error(
    monkeypatch, tmp_path, chips_raise
):
    """Under facet-lifecycle clearing, unreadable chips on a wall page are
    the wall, not geography_regime_error ('fix your facet values')."""
    from linkedin.orchestrator import GeographyRegimeError
    from shared import config

    aw = _aw()
    monkeypatch.setattr(config, "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True)

    async def _chips():
        if chips_raise:
            raise RuntimeError("location wrapper not found")
        return None

    browser = SimpleNamespace(page=_FakePage(WALL_URL, WALL_BODY), _state_dir=None)
    browser.read_applied_location_chips = _chips

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(_gate(browser, tmp_path)._verify_session_geography_chips())

    assert raised.value.variant == "session_conflict"
    assert not isinstance(raised.value, GeographyRegimeError)


def test_chip_verification_on_a_normal_page_keeps_the_regime_error(monkeypatch, tmp_path):
    from linkedin.orchestrator import GeographyRegimeError
    from shared import config

    monkeypatch.setattr(config, "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True)

    async def _chips():
        return None

    browser = SimpleNamespace(page=_FakePage(SEARCH_URL, "candidate copy"), _state_dir=None)
    browser.read_applied_location_chips = _chips

    with pytest.raises(GeographyRegimeError):
        asyncio.run(_gate(browser, tmp_path)._verify_session_geography_chips())


def test_recovery_reraises_a_wall_raised_by_its_own_navigation(tmp_path):
    """The health check passes as an ordinary wrong surface, then the
    recovery navigation itself lands on the wall: it must re-raise, not
    swallow the wall and try again."""
    from linkedin.recruiter_recovery import (
        RecruiterRecoverySnapshot,
        recover_recruiter_context,
    )

    aw = _aw()
    browser = _recovery_browser("about:blank", "", tmp_path)
    wall = aw.AuthWallError("session_conflict", url=WALL_URL)
    browser.navigate_to_search.side_effect = wall
    snapshot = RecruiterRecoverySnapshot(search_url=SEARCH_URL, project_id="9000000003")

    with pytest.raises(aw.AuthWallError) as raised:
        asyncio.run(recover_recruiter_context(browser, snapshot, max_attempts=3))

    assert raised.value is wall
    assert browser.refresh_active_tab.await_count == 1
    assert browser.navigate_to_search.await_count == 1


def _swallow_the_wall_then_fail_generically(tmp_path: Path):
    """search_surface.reanchor_search_surface is fail-soft: it catches the
    wall raised under navigate_to_search and returns False, and the next
    readiness gate refuses with a generic error that names no wall."""

    async def _behaviour(pipeline):
        page = _FakePage(WALL_URL, WALL_BODY)
        pipeline.browser = _browser_on(page, tmp_path)
        pipeline.page = page
        try:
            await pipeline.browser.navigate_to_search(SEARCH_URL)
        except Exception:
            pass
        raise RuntimeError("search surface not ready: foreign_search_page")

    return _behaviour


def test_a_swallowed_wall_still_ends_the_day_cycle_on_first_sight(
    day_cycle, tmp_path, capsys
):
    recorded = day_cycle(
        _swallow_the_wall_then_fail_generically(tmp_path),
        _completes,
    )

    _assert_stopped_on_first_sight(recorded, capsys.readouterr().out)
    assert "auth_wall_session_conflict" in _capture_reasons(tmp_path)


def test_checkpoint_in_the_day_cycle_stops_after_one_session_with_the_backoff(
    day_cycle, tmp_path, capsys
):
    aw = _aw()
    governor = _RecordingGovernor()

    async def _behaviour(pipeline):
        page = _FakePage(CHECKPOINT_URL, "")
        pipeline.browser = _governed_browser_on(page, tmp_path, governor)
        pipeline.page = page
        await pipeline.browser.navigate_to_search(SEARCH_URL)

    recorded = day_cycle(_behaviour, _completes)

    assert recorded["pipelines"] == 1
    assert recorded["session_end_reasons"] == ["recruiter_reauth_required"]
    assert isinstance(recorded.get("raised"), aw.AuthWallError)
    assert recorded["raised"].variant == "checkpoint"
    assert governor.forced == ["blocked_or_rate_limited"]
    assert recorded["error_backoffs"] == 0
