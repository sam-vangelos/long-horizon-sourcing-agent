<!-- provenance: status=canonical era=terminal-first stamped=2026-10-15 -->
# Browser Automation, Input Backends, and Session Pacing

## Operating Problem

LinkedIn sourcing runs inside a single authenticated LinkedIn Recruiter browser context over multi-hour sessions (`SESSION_DURATION_MIN_SECONDS = 4 * 3600` to `SESSION_DURATION_MAX_SECONDS = 5 * 3600` in `linkedin/session_orchestrator.py`). Unlike stateless HTTP scrapers that rotate proxies or disposable sessions, the browser agent attaches over the Chrome DevTools Protocol (`CDP_URL`) via `rebrowser-playwright` (`rebrowser_playwright` at import time) to a persistent Chrome profile (`LinkedInBrowser.connect` in `linkedin/browser.py`, launched by `cloris/chrome_launcher.py`).

Two failure modes dominate a multi-hour browser session:

1. DOM and target drift inside Chrome and the Recruiter single-page application. When attaching over CDP on Chrome 148+, adopting an existing tab target can leave a zombie renderer where `Runtime.evaluate` hangs; when `_bind_existing_recruiter_page` finds no responsive tab, `LinkedInBrowser._bind_created_recruiter_page` (`linkedin/browser.py:1091-1155`) discovers the live Recruiter URL via `_discover_recruiter_url_via_cdp` (`Target.getTargets` matching `"linkedin.com/talent"`), opens a fresh page via `context.new_page()`, and waits for `_settle_search_rail`. Once bound, `LinkedInBrowser.require_recruiter_tab` (`linkedin/browser.py:1157-1197`) pins the session to the project URL (`/talent/hire/`, parsed by `_recruiter_page_project_id`) and raises `RuntimeError` if a navigation lands on consumer LinkedIn or an unverified project.
2. Unbounded or mechanical interaction patterns across hours of card scanning, profile reads, filter edits, and note writing. If pointer movement teleports to element centers, keystrokes fire at zero-variance intervals, profile opens run without a ceiling, or the browser does nothing but Recruiter pagination for five hours straight, the session either trips rate limits or degrades the underlying account state.

The repository addresses those concerns through six cooperating layers: pluggable input backends (`linkedin/input_backends.py`), log-normal and autocorrelated human timing (`shared/human_timing.py`), section-directed profile reading (`linkedin/browser.py`), interleaved decoy browsing on a separate consumer tab (`decoy/`), persistent session and rolling-window governors (`shared/governor.py`, `shared/cooldown.py`, `github/governor.py`), and startup posture and build-fingerprint verification (`linkedin/posture_report.py`, `tools/build_fingerprint.py`).

---

## Input Backends and Ghost-Cursor Trajectories

All pointer, scroll, and keyboard interactions in `LinkedInBrowser` route through the `InputBackend` base class in `linkedin/input_backends.py`. `normalize_input_mode` resolves the `--input-mode` setting (`input_mode` parameter on `LinkedInBrowser`, `LinkedInOrchestrator`, and `linkedin/session_orchestrator.py`) into one of two concrete implementations constructed by `create_input_backend`:

| Mode | Aliases accepted by `normalize_input_mode` | Backend class | Dispatch mechanism |
|---|---|---|---|
| `concurrent` | `concurrent`, `synthetic`, `ghostcursor`, `ghost_cursor` (plus `None`) | `ConcurrentInputBackend` | Playwright CDP mouse/keyboard events paired with `python_ghost_cursor` Bézier trajectories |
| `away` | `away`, `takeover`, `away_from_keyboard`, `afk` | `AwayInputBackend` | macOS CoreGraphics HID event injection (`CGEventPost` on `kCGHIDEventTap`) |

### Rebrowser Module Aliasing and Ghost-Cursor Dispatch

The runtime depends on `rebrowser-playwright` (`rebrowser_playwright`) rather than stock `playwright`, so that CDP `Runtime.Enable` leaks are patched at the driver layer. Because `python_ghost_cursor.playwright_async` imports `from playwright.async_api import Page, ElementHandle, CDPSession`, `_alias_playwright_to_rebrowser` in `linkedin/input_backends.py` inspects `importlib.util.find_spec("playwright")` when `LINKEDIN_GHOST_CURSOR_ENABLED` is true. When stock `playwright` is absent, it registers `rebrowser_playwright` and `rebrowser_playwright.async_api` under `sys.modules["playwright"]` and `sys.modules["playwright.async_api"]` before importing `create_cursor`.

In `LinkedInBrowser._ghost_click_locator` (`linkedin/browser.py:1210-1270`), clicking a locator proceeds across a strict commit boundary:

1. The element is scrolled into view (`locator.scroll_into_view_if_needed(timeout=2000)`) before any bounding-box coordinates are computed, preventing `python_ghost_cursor` from aiming at off-viewport coordinates.
2. `ConcurrentInputBackend.click_locator` (`linkedin/input_backends.py:434-455`) resolves `locator.element_handle(timeout=3000)` and moves the pointer along a Bézier trajectory via `self._cursor.move(handle)` (or, if Ghost Cursor is unavailable, `_ghost_click_locator` falls back to `locator.hover(timeout=5000)` so Playwright completes its actionability wait and positions the pointer before the guard runs).
3. If the caller supplied a `before_click` callback (passed as `about_to_commit` to `click_locator`), that callback runs *after* pointer positioning and *immediately before* `page.mouse.down()`. Nothing that can scroll or move the page runs after `before_click`, so if a card shifts or a slide-in panel swaps candidate identity while the cursor is in flight, the guard aborts before any click event is dispatched.
4. `page.mouse.down()` and `page.mouse.up()` fire to commit the click.

### OS-Level HID Takeover (`AwayInputBackend`)

When the operator runs with `--input-mode away`, `AwayInputBackend` (`linkedin/input_backends.py:708-1044`) drives the macOS cursor and keyboard through `_CoreGraphicsBridge` (`linkedin/input_backends.py:595-706`), which binds `ApplicationServices` and `CoreFoundation` via `ctypes` (`CGEventCreate`, `CGEventGetLocation`, `CGEventCreateMouseEvent`, `CGEventCreateKeyboardEvent`, `CGEventCreateScrollWheelEvent`, `CGEventPost` at `kCGHIDEventTap = 0`, and `CFRelease`).

To translate DOM coordinates into screen coordinates without hardcoded pixel offsets, `AwayInputBackend._locator_screen_point` (`linkedin/input_backends.py:914-960`) queries `window.screenX`, `window.screenY`, `window.outerWidth`, `window.outerHeight`, `window.innerWidth`, and `window.innerHeight`, deriving left border width as `max(0.0, (outerWidth - innerWidth) / 2.0)` and top window chrome height (tab strip, omnibox, bookmarks bar) as `max(0.0, outerHeight - innerHeight)`. `AwayInputBackend._move_mouse` (`linkedin/input_backends.py:962-1006`) reads the current OS cursor position via `CGEventGetLocation` and steps through an 8-to-28-point cubic Bézier curve whose two control points are offset by `±24.0` pixels, posting `kCGEventMouseMoved` events paced on channel `os_pointer_micro`. Keystrokes map ASCII characters and shifted symbols through `AwayInputBackend.CHAR_KEY_CODES` and `KEY_CODES` (`_type_char` and `_press_with_modifiers`), falling back to `page.keyboard.insert_text` if an unmapped character is encountered.

### Boolean and Prose Typing Plans

Both backends share pure plan builders in `linkedin/input_backends.py` that turn an input string into a sequence of `TypingStep` records (`TypingPlan`) before any key event is sent:

1. `build_boolean_typing_plan` (`linkedin/input_backends.py:130-259`) models typing a Recruiter boolean query. Base inter-key intervals are drawn via `_char_delay` (`LINKEDIN_SEARCH_TYPING_CHAR_MIN_SECONDS` default `0.055` to `LINKEDIN_SEARCH_TYPING_CHAR_MAX_SECONDS` default `0.11` in `shared/config.py`). Operator boundaries (`_BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}`, plus closing quotes and closing parentheses) add a `0.18–0.45s` pause (`LINKEDIN_SEARCH_TYPING_OPERATOR_PAUSE_MIN_SECONDS` to `LINKEDIN_SEARCH_TYPING_OPERATOR_PAUSE_MAX_SECONDS`). Queries of 45 characters or longer add a mid-query thinking pause (`0.30–0.90s`, `LINKEDIN_SEARCH_TYPING_THOUGHT_PAUSE_MIN_SECONDS` to `LINKEDIN_SEARCH_TYPING_THOUGHT_PAUSE_MAX_SECONDS`) with 60% probability at a whitespace boundary at or after `_THINKING_PAUSE_MIN_INDEX = 18`. To avoid machine-perfect entry on long boolean strings, `build_boolean_typing_plan` injects one typo with probability `0.15` (`LINKEDIN_SEARCH_TYPING_MEDIUM_TYPO_PROBABILITY`) on 25–59 character strings and `0.30` (`LINKEDIN_SEARCH_TYPING_LONG_TYPO_PROBABILITY`) on `>= 60` character strings (with a `0.10` `LINKEDIN_SEARCH_TYPING_SECOND_TYPO_PROBABILITY` chance of a second typo separated by at least 20 characters, capped at `LINKEDIN_SEARCH_TYPING_MAX_TYPOS` default `2`), choosing only from alphabetic tokens of length `>= 5` outside `_BOOLEAN_OPERATORS`. Each typo emits a replacement letter via `_random_alpha_replacement`, a `Backspace` step, the corrected character (`is_correction=True`), and a 4-character post-correction slowdown (`+0.02–0.04s`).
2. `build_note_typing_plan` (`linkedin/input_backends.py:262-312`) models typing a plain-text candidate note (`_note_char_delay` using `LINKEDIN_NOTE_TYPING_CHAR_MIN_SECONDS` default `0.05` to `LINKEDIN_NOTE_TYPING_CHAR_MAX_SECONDS` default `0.10`, plus `0.02–0.06s` word pauses and `0.30–0.60s` sentence pauses after `.`, `!`, `?`). It sets `interval_includes_dwell=True` on `TypingPlan`. When `LINKEDIN_TYPING_DWELL_ENABLED` is active (`shared/config.py`), `ConcurrentInputBackend._type_char_with_dwell` (`linkedin/input_backends.py:482-531`) dispatches explicit `page.keyboard.down` and `page.keyboard.up` events separated by `_sample_key_dwell` (log-normal with `_DWELL_LOG_MEDIAN_SECONDS = 0.09`, `_DWELL_LOG_SIGMA = 0.35`, clamped to `_DWELL_FLOOR_SECONDS = 0.03` and `_DWELL_CEILING_SECONDS = 0.30`, scaled by `_HOME_ROW_DWELL_MULTIPLIER = 0.8` on `_HOME_ROW_KEYS` `asdfjkl;` and space). Because `interval_includes_dwell` is true on prose notes, `type_text` subtracts `held_seconds` from `step.delay_seconds` instead of summing them, keeping a 500-character note near 53 seconds rather than doubling it. On consecutive distinct alphabetic characters (`_is_fast_bigram`), `_type_char_with_dwell` depresses the next key before releasing the current key with probability `_FAST_BIGRAM_OVERLAP_PROBABILITY = 0.25`.

---

## Human Timing and Section-Directed Profile Reading

### Log-Normal and Autocorrelated Delays (`shared/human_timing.py`)

`shared/human_timing.py` replaces uniform random sleeps (`random.uniform`) with two primitives grounded in human response-time distributions:

- `human_delay(low, high, mu=0.0, sigma=0.6)` (`shared/human_timing.py:31-43`) sets `mu = math.log(max((low + high) / 2.0, 0.1))` when `mu == 0`, draws from `random.lognormvariate(mu, sigma)`, and clamps the result to `[low, high]`.
- `human_delay_correlated(base, spread=0.6, *, channel="default")` (`shared/human_timing.py:54-79`) maintains a per-channel dictionary `_last_delay_by_channel: dict[str, float]`. Each call draws `raw = random.lognormvariate(math.log(max(base, 0.1)), spread)` and blends it with the prior draw on the same `channel` using `_CORRELATION_WEIGHT = 0.3`:
  $$\text{delay} = \operatorname{clamp}\bigl(0.7 \cdot \text{raw} + 0.3 \cdot \text{last\_delay},\; 0.3 \cdot \text{base},\; 4.0 \cdot \text{base}\bigr)$$
  Using separate named channels (`scroll`, `page_turn`, `profile_read`, `profile_read_return`, `profile_linger`, `card_glance`, `snippet_scan`, `preview_no`, `preview_skip`, `note_compose`, `note_submit`, `os_pointer`, `os_key`, `cadence_interval`, `cadence_pause`, plus `${channel}_micro` and `${channel}_pause` sub-streams) ensures that high-frequency wheel ticks never drag down the next `profile_read` or `page_turn` pause.

### Chunked Wheel Scrolling (`decoy/actions/_utils.py`)

Both `ConcurrentInputBackend.scroll` and the decoy actions delegate wheel scrolling to `human_scroll` in `decoy/actions/_utils.py` (`decoy/actions/_utils.py:37-88`). Rather than jumping `delta_y` pixels in one event, `human_scroll` breaks the distance into wheel ticks drawn from `int(random.lognormvariate(math.log(85), 0.35))` (floored at `20` pixels), dispatches each tick via `page.mouse.wheel(0, chunk * direction)`, waits on `human_delay_correlated(0.03, spread=0.45, channel=f"{channel}_micro")`, inserts a `0.12–0.35s` pause on `channel=f"{channel}_pause"` on 10% of ticks, and on scrolls of at least `220` pixels injects an 18%-probability corrective reverse nudge (`12–36` pixels opposite the scroll direction, followed 75% of the time by a forward restoration tick).

### Section-Directed Profile Reading (`linkedin/browser.py`)

When the sourcing loop opens a candidate slide-in panel (`div.profile__main-container`), `LinkedInBrowser.simulate_profile_read` (`linkedin/browser.py:4759-4859`) simulates human inspection of the profile before judgment or note writing:

- When `LINKEDIN_SECTION_DIRECTED_READ_ENABLED` is true (default), `simulate_profile_read` locates the `about`, `experience`, and `education` DOM headings (`locate_sections` in `linkedin/profile_sections.py`, matched against `_SECTION_DIRECTED_NAMES`) and calls `_read_section_directed` and `_plan_section_traversal` (`linkedin/browser.py:4861-4908`). `_plan_section_traversal` skips `about` with probability `_ABOUT_SKIP_PROBABILITY = 0.30` and selects one of four `_TRAVERSAL_STYLES` weighted by `_TRAVERSAL_WEIGHTS = (45, 25, 20, 10)`: `straight` (`45`), `backtracker` (`25`, which revisits `experience` after reaching the end), `experience_first` (`20`, jumping to `experience` first), or `overshoot` (`10`, passing a section heading by `40–120` pixels in `_scroll_to_section` and correcting back).
- Total read budget scales with the upstream pre-screen interest score via `_interest_read_budget_seconds(interest) = min(3.0 + clamped * 12.0, 35.0)` (`linkedin/browser.py:4714-4717`). Every per-section dwell passes through `_profile_read_dwell` and `_clamp_profile_read_dwell` (`linkedin/browser.py:4720-4757`), which enforce two hard ceilings regardless of interest score or section count: `_MAX_CHUNK_DWELL = 8.0` seconds per section stop and `_MAX_PROFILE_READ = 35.0` seconds cumulative across the profile.
- When `LINKEDIN_SECTION_DIRECTED_READ_ENABLED` is false, `simulate_profile_read` selects among the four whole-container scroll patterns in `_PATTERN_NAMES` (`focused_reader`, `skipper`, `skimmer`, `section_hopper`, implemented by `_read_focused`, `_read_skipper`, `_read_skimmer`, and `_read_section_hopper`) weighted by `_PATTERN_WEIGHTS` for `"short"` (`<= 2` chunks), `"medium"` (`<= 5` chunks), or `"long"` profiles.

---

## Decoy Browsing and Cooperative Interleaving (`decoy/`)

To prevent a browser session from consisting solely of Recruiter search and profile-open calls, the `decoy/` package generates non-Recruiter browsing activity on consumer LinkedIn (`https://www.linkedin.com/feed/`).

| Module | Primary symbols | Role |
|---|---|---|
| `decoy/agent.py` | `DecoyAgent`, `ACTIONS` | Owns a dedicated consumer LinkedIn tab (`_ensure_tab`), executes weighted 1–3 action bursts (`execute_burst`), and appends telemetry to `~/.sourcing-governor/decoy.jsonl` (`DECOY_LOG`) |
| `decoy/scheduler.py` | `BurstScheduler` | Samples autocorrelated log-normal inter-burst intervals (`next_interval`) in `"dormant"` (`10–90` min) or `"interleave"` (`15–60` min) mode |
| `decoy/actions/jobs.py` | `browse_jobs` | Navigates to `/jobs` via `#global-nav`, scrolls 2–5 viewports, ghost-clicks 1–3 visible job cards, and dwells `10–30s` per listing |
| `decoy/actions/feed.py` | `scroll_feed` | Passively scrolls 3–10 feed viewports with `1–5s` reading pauses on 30–50% of stops and zero likes, comments, or shares |
| `decoy/actions/notifications.py` | `check_notifications` | Opens the `#global-nav` notifications bell, scans for `5–15s`, and with 20% probability clicks one of the top 5 notification cards |
| `decoy/actions/_utils.py` | `ghost_click`, `human_scroll` | Shared Ghost Cursor click wrapper and chunked wheel scroller with overshoot correction |

### Tab Isolation and Action Weights

`DecoyAgent._ensure_tab` (`decoy/agent.py:66-79`) never reuses an existing tab or the Recruiter tab; it always opens a dedicated tab via `await self._context.new_page()` (`self._owns_page = True`), navigates to `https://www.linkedin.com/feed/`, initializes its own `python_ghost_cursor` instance via `_init_cursor()`, and closes only its owned tab on `DecoyAgent.close()`. Each `execute_burst()` call selects `random.randint(1, 3)` actions via `_weighted_pick` using `ACTIONS`: `("feed", scroll_feed, 3)`, `("notifications", check_notifications, 1)`, and `("jobs", browse_jobs, 5)`. All actions are read-only on the social graph: `scroll_feed` never reacts or comments, and `browse_jobs` only clicks job cards to view descriptions.

### Scheduler Modes and Cooperative Handoff

`BurstScheduler.next_interval` (`decoy/scheduler.py:37-62`) draws log-normal intervals (`math.exp(self._mu + self._sigma * random.gauss(0, 1)) * 60`) and blends each draw with `self._last_interval` (`self._weight = 0.3`):

- In `"dormant"` mode (used by `DecoyAgent.run_dormant_loop` between active sourcing sessions and by `run_decoy_only` in `linkedin/session_orchestrator.py`), `_mu = 3.2`, `_sigma = 0.6` (median `~25` minutes), clamped to `[600, 5400]` seconds (`10–90` minutes).
- In `"interleave"` mode (used while a sourcing session is running with `--with-decoy`), `_mu = 3.4`, `_sigma = 0.5` (median `~30` minutes), clamped to `[900, 3600]` seconds (`15–60` minutes).

During active sourcing, `_interleave_loop` in `linkedin/session_orchestrator.py` (`linkedin/session_orchestrator.py:547-591`) coordinates with `PageReviewer._review_page_sequentially` and `PageReviewer._review_page_batch` in `linkedin/page_review.py` through two `asyncio.Event` primitives (`pause_requested` and `resume_event`) so that two coroutines never drive the browser concurrently:

```
_interleave_loop (session_orchestrator.py)          page_review.py (between candidate cards)
------------------------------------------          ----------------------------------------
resume_event.clear()
pause_requested.set()
wait up to 60s for pause_requested.is_set()==False
                                                    sees pause_requested.is_set() == True
                                                    allocator._checkpoint_progress(progress)
                                                    pause_requested.clear()
                                                    awaits resume_event.wait() (timeout 300s)
await decoy.execute_burst()
await browser.bring_to_front()
resume_event.set()
```

If the sourcing loop is mid-evaluation and does not reach a card boundary within 60 seconds (`for _ in range(120): await asyncio.sleep(0.5)`), `_interleave_loop` clears `pause_requested`, sets `resume_event`, logs a skipped burst, and waits for the next scheduled interval rather than racing on the browser context. Inside the search loop, `SearchSurface._maybe_cadence_pause` (`linkedin/search_surface.py:1624-1662`) also inserts periodic in-loop pauses sampled by `_sample_cadence_pause` around `CADENCE_PAUSE_SECONDS` every `_sample_cadence_interval` (`CADENCE_INTERVAL_MINUTES`).

---

## Session Governors, Cooldown State, and Rate-Limit Backoff

### Mandatory Browser Governance (`shared/governor.py` and `shared/cooldown.py`)

`LinkedInBrowser.__init__` (`linkedin/browser.py:760-796`) requires a `SessionGovernor` instance and raises `ValueError` if `governor=None` is passed (unit tests that do not exercise governance pass the explicit sentinel `UNGOVERNED_FOR_TESTS` from `shared/governor.py`). Both `LinkedInBrowser.open_profile` and `LinkedInBrowser.open_profile_by_url` call `self._governor.check_profile_open_or_raise()` before touching the DOM and `self._governor.record_profile_open()` once the slide-in panel opens.

`SessionGovernor` (`shared/governor.py`) and `shared/cooldown.py` enforce three hard limits persisted via atomic temp-file rename (`_save_raw`) in `~/.sourcing-governor/daily_stats.json`:

| Constant | Value | Enforcement point |
|---|---|---|
| `MAX_PROFILE_OPENS_PER_SESSION` | `200` | `SessionGovernor._check_limits_or_raise` / `check_profile_open_or_raise` (raises `GovernorLimitReached`) |
| `MAX_PROFILE_OPENS_PER_24H` | `400` | `cooldown.get_profile_opens_24h` across a rolling `WINDOW_SECONDS = 24 * 3600` window of per-open Unix timestamps |
| Session duration limit | `4.0–5.0h` (`14,400–18,000s`) | Sampled per session by `_sample_session_duration` (`linkedin/session_orchestrator.py:73-78`) and passed to `SessionGovernor.start_session(session_duration_seconds)`, which adds `GOVERNOR_SESSION_BACKSTOP_GRACE_SECONDS = 600` (`shared/governor.py:111-122`; if `session_duration_seconds` is `None`, `start_session` draws from `GOVERNOR_LEGACY_SESSION_DURATION_MIN_SECONDS = 3.5 * 3600` to `GOVERNOR_LEGACY_SESSION_DURATION_MAX_SECONDS = 4.5 * 3600`) |

If `~/.sourcing-governor/daily_stats.json` contains corrupt JSON, `cooldown._load_raw` raises `GovernorStateUnreadable` rather than resetting counters to zero, and `SessionGovernor.can_start_session` fails closed (`(False, "governor state unreadable — refusing to start")`). When pruning `sessions_today`, `cooldown._reconcile_stale_sessions` uses `_pid_is_alive` (`shared/cooldown.py:88-98`, probing `os.kill(pid, 0)`) to close orphaned in-progress session entries left by dead processes.

### Rate-Limit Detection, Forced Backoff, and Multi-Session Cycling

Before opening profiles or recovering from navigation errors, `detect_recruiter_health` in `linkedin/recruiter_recovery.py` (`linkedin/recruiter_recovery.py:316-386`) classifies the current page into `RecruiterHealthState` (`"healthy"`, `"aw_snap"`, `"target_crashed"`, `"slow_or_unresponsive"`, `"logged_out"`, `"wrong_surface"`, `"lost_project_context"`, `"stale_search_context"`, `"cdp_attach_failed"`, `"blocked_or_rate_limited"`, or `"recruiter_reauth_required"`). To prevent a candidate whose profile text mentions `"rate limiting"` or `"1,429 followers"` from triggering a false positive, `_peek_page_diagnostic_parts` (`linkedin/recruiter_recovery.py:287-313`) separates the lowercased document `<title>` from the first `2,000` characters of `body` text, and `_matches_rate_limit_vocabulary` (`linkedin/recruiter_recovery.py:164-186`) holds them to different rules: `_RATE_LIMIT_TITLE_PATTERNS` (`"rate limit"`, `"too many requests"`) match against the title only, whereas body text matches only `_RATE_LIMIT_BODY_PATTERNS` (`"too many requests"`), status-qualified `429` patterns (`_RATE_LIMIT_STATUS_REGEXES`), or reader-addressed notices (`_RATE_LIMIT_NOTICE_REGEXES`).

The orchestrator reacts differently depending on the failure class:

- When `detect_recruiter_health` returns `"blocked_or_rate_limited"`, `recover_recruiter_context` (`linkedin/recruiter_recovery.py:672-715`) never refreshes or retries. It calls `governor.force_backoff("blocked_or_rate_limited")`, which persists a `RATE_LIMIT_BACKOFF_SECONDS = 6 * 3600` (`6`-hour) `forced_backoff` entry via `cooldown.record_forced_backoff` (`shared/cooldown.py:201-209`) and returns `reason="blocked_or_rate_limited_backoff"`. In `linkedin/session_orchestrator.py`, `_wait_for_governor_window` treats an active `forced_backoff` or `GovernorStateUnreadable` as an immediate stop rather than a rolling-window cap to sleep through.
- When `detect_recruiter_health` returns `"recruiter_reauth_required"` (classified via `classify_auth_wall` in `linkedin/auth_wall.py`), `recover_recruiter_context` calls `_raise_auth_wall`, raising `AuthWallError`, and `run_day_cycle` (`linkedin/session_orchestrator.py:743-770`) halts without writing a 6-hour rate-limit block, because an expired login requires operator re-authentication rather than cooldown time.
- Between sessions in `run_day_cycle`, the orchestrator enters `DecoyAgent.run_dormant_loop` for a log-normal dormant interval sampled by `_sample_dormant_duration` (`linkedin/session_orchestrator.py:81-95`, configured by `LINKEDIN_DORMANT_MEDIAN_MINUTES`, `LINKEDIN_DORMANT_MIN_MINUTES`, and `LINKEDIN_DORMANT_MAX_MINUTES` in `shared/config.py`) before re-checking `governor.can_start_session()`.

### API-Scoped Governance (`github/governor.py`)

`GitHubGovernor` (`github/governor.py`, configured by `github/config.py`) implements the corresponding session lifecycle (`can_start_session`, `start_session`, `end_session`, `record_enrichment`, `check_limits`, `check_limits_or_raise`) against `~/.sourcing-governor/github/daily_stats.json`. Because GitHub sourcing runs over authenticated REST and GraphQL APIs rather than a single browser session, browser-specific session caps are lifted (`MAX_SESSION_DURATION_SECONDS` default `999 * 3600`, `MAX_ENRICHMENTS_PER_SESSION` default `999_999`, `MAX_SESSIONS_PER_DAY` default `999` in `github/config.py`), while `GitHubGovernor.should_enter_enrichment_only(limiter_remaining)` (`github/governor.py:211-213`) switches the pipeline from search discovery to enrichment-only mode whenever remaining GitHub REST quota drops below `500` requests.

---

## Startup Posture Reporting and Build Fingerprinting

### Posture Report (`linkedin/posture_report.py`)

At the start of every LinkedIn session (`linkedin/session_orchestrator.py:808`), `describe_posture` and `format_posture` in `linkedin/posture_report.py` inspect the live runtime configuration and emit an 8-row `[ok]` / `[MISSING]` checklist to both the console and `run_log.jsonl`:

1. `ghost cursor`: calls `_probe_ghost_cursor()`, which invokes `_alias_playwright_to_rebrowser()` when `LINKEDIN_GHOST_CURSOR_ENABLED` is true and imports `python_ghost_cursor.playwright_async.create_cursor` to confirm whether Bézier trajectories are `"active (rebrowser alias)"`, `"loaded"`, or unavailable.
2. `input backend mode`: reports the configured `input_mode` (`"concurrent"`, `"away"`, or `"unset"`).
3. `decoy`: reports `"enabled"` (`[ok]`) when `with_decoy` is true or `"dormant"` (`[MISSING]`) when disabled.
4. `cadence pause`: reports `f"interval ~{config.CADENCE_INTERVAL_MINUTES} min"`.
5. `MAX_PROFILE_OPENS_PER_SESSION`: reports `str(MAX_PROFILE_OPENS_PER_SESSION)` (`200`).
6. `MAX_PROFILE_OPENS_PER_24H`: reports `str(MAX_PROFILE_OPENS_PER_24H)` (`400`).
7. `forced backoff`: queries `cooldown.get_active_backoff()` and reports whether a forced backoff is active (`f"active until {backoff['until']}"`) or `"none"`.
8. `driver package`: inspects `importlib.metadata.version("rebrowser-playwright")` and flags `[MISSING]` if the package is not installed.

### Source Build Fingerprint (`tools/build_fingerprint.py`)

When long-running sessions or packaged bundles execute across restarts, `git rev-parse HEAD` does not detect uncommitted edits or stale copied trees. `compute_fingerprint` in `tools/build_fingerprint.py` (`tools/build_fingerprint.py:82-99`) walks every non-`__pycache__`, non-`.pyc` Python file under `BUNDLED_SOURCE_ROOTS` (`cloris`, `shared`, `market_intelligence`) via `iter_bundled_python_files` and hashes `<relative_posix_path>\0<sha256(file_bytes)>\0` into a deterministic 64-character SHA-256 hex digest.

---

## What the Test Suite Verifies

Seven dedicated test modules (plus `tests/test_browser_read_dwell.py` and `tests/test_cooldown.py`) verify the subsystem without requiring a live Chrome session:

| Test file | What it verifies |
|---|---|
| `tests/test_linkedin_input_backends.py` | Verifies `normalize_input_mode` and `create_input_backend`; checks that `build_boolean_typing_plan` injects operator pauses, mid-query thinking pauses, and typos (`_random_alpha_replacement`) followed by `Backspace` and `is_correction=True` while keeping `_BOOLEAN_OPERATORS` (`AND`/`OR`/`NOT`) untouched; verifies that `build_note_typing_plan` sets `interval_includes_dwell=True` and holds a 500-character note budget near `~53s`; exercises `ConcurrentInputBackend.type_text` to confirm keydown/keyup dwell (`_sample_key_dwell`), home-row scaling (`_HOME_ROW_DWELL_MULTIPLIER`), and fast-bigram overlap (`_is_fast_bigram`); tests `AwayInputBackend._locator_screen_point` window-chrome offset math and `_move_mouse` cubic Bézier stepping against a mocked `_CoreGraphicsBridge`; and checks that `LinkedInBrowser._ghost_click_locator` scrolls locators into view and runs `before_click` after cursor motion but before `mouse.down`. |
| `tests/test_ghost_cursor_alias.py` | Exercises `_alias_playwright_to_rebrowser` in isolation, verifying that `sys.modules["playwright"]` and `sys.modules["playwright.async_api"]` alias to `rebrowser_playwright` when stock `playwright` is absent, that existing `playwright` modules are never overwritten, and that `LINKEDIN_GHOST_CURSOR_ENABLED=false` disables the alias. |
| `tests/test_human_timing.py` | Verifies `human_delay` clamping (`[low, high]`, degenerate `low == high`, sub-`0.1` bounds) and tests `human_delay_correlated` channel isolation (`_last_delay_by_channel`), `[0.3 * base, 4.0 * base]` bounds, and positive lag-1 autocorrelation (`_CORRELATION_WEIGHT = 0.3`) compared against independent samples. |
| `tests/test_decoy_agent.py` | Verifies `BurstScheduler.next_interval` clamping (`600–5400s` in `"dormant"` mode, `900–3600s` in `"interleave"` mode) and autocorrelation; confirms `DecoyAgent._ensure_tab` always opens a new tab via `context.new_page()` rather than hijacking existing tabs and closes only its owned tab; tests `execute_burst` failure resilience and JSONL logging; and runs `scroll_feed`, `browse_jobs`, and `check_notifications` against mock Playwright pages to confirm passive scrolling and card dwell behavior. |
| `tests/test_shared_governor.py` | Verifies `SessionGovernor` session lifecycle, per-session (`200`) and rolling 24-hour (`400`, `cooldown.get_profile_opens_24h`) profile-open caps, `session_duration_seconds` backstop enforcement, `force_backoff` persistence (`6`-hour block), fail-closed behavior (`GovernorStateUnreadable`) when `daily_stats.json` contains corrupt JSON, and the `UNGOVERNED_FOR_TESTS` sentinel. |
| `tests/test_linkedin_browser_governor.py` | Verifies that `LinkedInBrowser(..., governor=None)` raises `ValueError`, that `open_profile` and `open_profile_by_url` raise `GovernorLimitReached` before any DOM interaction when the session or 24-hour cap is exhausted, that successful opens increment `profile_opens_session`, and that failed opens do not burn a profile-open credit. |
| `tests/test_linkedin_posture_report.py` | Verifies `describe_posture` and `format_posture` when Ghost Cursor is active (`[ok]`), when `_probe_ghost_cursor` reports an import failure (`[MISSING]`), when `cooldown.get_active_backoff` returns an active rate-limit backoff, and when `with_decoy` is enabled. |
