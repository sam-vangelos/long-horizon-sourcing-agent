#!/usr/bin/env python3
"""Canonical LinkedIn sourcing launcher.

It manages:
  - Session Governor (hard safety limits)
  - Sourcing Agent (one session by default)
  - Explicit Multi-Session Cycling (sprint → dormant → sprint)
  - Optional Decoy Agent interleaving

Usage:
    python -m linkedin.session_orchestrator --brief config/brief-X.json
    python -m linkedin.session_orchestrator --brief config/brief-X.json --resume
    python -m linkedin.session_orchestrator --brief config/brief-X.json --multi-session
    python -m linkedin.session_orchestrator --brief config/brief-X.json --multi-session --with-decoy
"""

import argparse
import asyncio
import math
import os
import random
import signal
import socket
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("AGENT_KEY_PREFIX", "LINKEDIN")
from shared import config
from shared import cooldown
from shared.cooldown import GovernorStateUnreadable, ShutdownKind
from shared.console_tee import enable_console_tee
from shared.output_paths import resolve_linkedin_state_dir
from shared.runtime_state import RuntimeStateLock
from shared.runtime_state.read_models import (
    has_pending_work,
    latest_run_in_state_dir,
    _RUNTIME_DB_FILENAME,
)
from shared.governor import (
    CAP_SHAPED_SHUTDOWN_PREFIXES,
    SessionGovernor,
    SessionExpired,
    OperatorStopRequested,
    GovernorLimitReached,
    MAX_PROFILE_OPENS_PER_24H,
)
from shared.failures import ApiBudgetExhaustedError
from shared.constraint_manifest import ConstraintManifestError
from shared.preflight_v2 import PreflightRegimeError
from linkedin.auth_wall import (
    AuthWallError,
    RECRUITER_REAUTH_REQUIRED,
    auth_wall_operator_line,
    auth_wall_variant_of,
    live_auth_wall_error,
)
from decoy.agent import DecoyAgent
from decoy.scheduler import BurstScheduler
from shared.human_timing import human_delay


# ──────────────────────────────────────────────────────────────────────
# Session duration sampling
# ──────────────────────────────────────────────────────────────────────

SESSION_DURATION_MIN_SECONDS = 4 * 3600
SESSION_DURATION_MAX_SECONDS = 5 * 3600


def _sample_session_duration() -> float:
    """Sample a fresh uniform session duration in the 4-5 hour band, in seconds.

    A fresh draw avoids a fixed detectable session length.
    """
    return random.uniform(SESSION_DURATION_MIN_SECONDS, SESSION_DURATION_MAX_SECONDS)


def _sample_dormant_duration() -> float:
    """Log-normal dormant period between sessions, in seconds.

    Shape is operator-tunable (LINKEDIN_DORMANT_{MEDIAN,MIN,MAX}_MINUTES);
    defaults reproduce a median of about 110 minutes clamped to 75-180. Gap
    length shapes the activity signature only — daily volume is bounded by the
    24h open cap regardless of how the gaps are sized.
    """
    median_s = max(60.0, config.LINKEDIN_DORMANT_MEDIAN_MINUTES * 60.0)
    lo_s = max(60.0, config.LINKEDIN_DORMANT_MIN_MINUTES * 60.0)
    hi_s = max(lo_s, config.LINKEDIN_DORMANT_MAX_MINUTES * 60.0)
    mu = math.log(median_s)
    sigma = 0.3
    raw = math.exp(mu + sigma * random.gauss(0, 1))
    return max(lo_s, min(hi_s, raw))


def _sample_error_backoff(attempt: int) -> float:
    """Short escalating backoff after an ABSORBED session error, in seconds.

    Deliberately not the dormant gap, and only reachable when the errored
    session made ZERO profile opens (enforced by _should_retry_session_error,
    not assumed here): with no LinkedIn activity there is no detection
    signature to space out, and the fault was upstream. 90s then
    180s, ±25% jitter, matching the scale of the faults it retries. Bounded by
    LINKEDIN_SESSION_ERROR_RETRIES, so the worst case is a few minutes.
    """
    base = min(90.0 * (2 ** max(0, attempt - 1)), 300.0)
    return base * random.uniform(0.75, 1.25)


# Poll cadence while a persistent campaign sleeps through a closed governor
# window (the 24h profile-open cap; a forced backoff stops instead of
# waiting). 15 minutes: fast enough that a reopened window is picked up
# promptly, slow enough that an overnight sleep logs a handful of lines, not
# hundreds.
_GOVERNOR_WAIT_POLL_SECONDS = 900.0


async def _wait_for_governor_window(
    governor: "SessionGovernor",
    stop_event: "asyncio.Event",
    *,
    session_type: str = "linkedin_sourcing",
) -> bool:
    """Sleep until the governor allows a session again, or the operator stops.

    Returns True when a session may start, False on operator shutdown. This is
    what makes LINKEDIN_CAMPAIGN_PERSIST mean "remain active": a closed
    volume window is retried in place. Waiting spends nothing — no browser,
    no opens, no model calls.
    """
    polls = 0
    while not stop_event.is_set():
        can_start, reason = governor.can_start_session(session_type=session_type)
        if can_start:
            if polls:
                _print_governor("Governor window reopened — resuming campaign.")
            return True
        # A DETECTION refusal is not a closed window — it is a stop.
        # cooldown.record_forced_backoff() persists a 6h block after a
        # rate-limit/blocked signal, and governor.can_start_session() reports
        # it as "Forced backoff active (...)". Waiting that out and then
        # starting automatically is precisely the move persistence must never
        # make (a possibly-flagged seat re-probed unattended). Only VOLUME
        # windows — daily sessions, 24h opens — are waited out.
        try:
            active_backoff = cooldown.get_active_backoff()
        except GovernorStateUnreadable:
            _print_governor(
                "Governor state unreadable — this is a stop, not a closed window. "
                "Persistence ends here; relaunch by hand after checking the seat."
            )
            return False
        if active_backoff is not None:
            _print_governor(
                f"Forced backoff active ({reason}) — this is a DETECTION stop, "
                "not a closed window. Persistence ends here; relaunch by hand "
                "after checking the Recruiter seat."
            )
            return False
        if polls % 4 == 0:  # first refusal, then roughly hourly
            _print_governor(
                f"Governor window closed ({reason}) — campaign persists, "
                f"polling every {_GOVERNOR_WAIT_POLL_SECONDS/60:.0f} min."
            )
        polls += 1
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=_GOVERNOR_WAIT_POLL_SECONDS
            )
        except asyncio.TimeoutError:
            continue
    return False


# Connectivity ground truth: one DNS resolution and TCP handshake to the API
# host, without TLS, HTTP, or LinkedIn interaction. This probe decides whether
# a session error is local connectivity loss; page-derived fault text cannot.
# The API host exercises the same resolver and egress used by campaign calls.
_NETWORK_PROBE_HOST = "api.anthropic.com"
_NETWORK_PROBE_PORT = 443
_NETWORK_PROBE_TIMEOUT_SECONDS = 8.0
_NETWORK_WAIT_POLL_SECONDS = 60.0


def _network_probe_once() -> bool:
    """One blocking DNS+TCP reachability check; never raises."""
    try:
        with socket.create_connection(
            (_NETWORK_PROBE_HOST, _NETWORK_PROBE_PORT),
            timeout=_NETWORK_PROBE_TIMEOUT_SECONDS,
        ):
            return True
    except OSError:
        return False


async def _network_is_reachable() -> bool:
    return await asyncio.to_thread(_network_probe_once)


async def _wait_for_network(stop_event: "asyncio.Event") -> bool:
    """Sleep until the machine's network returns, or the operator stops.

    Unbounded by design: a connectivity outage touches neither LinkedIn nor a
    provider, so waiting spends nothing and risks nothing. Returns True when
    the probe succeeds, False on operator shutdown.
    """
    polls = 0
    while not stop_event.is_set():
        if await _network_is_reachable():
            if polls:
                _print_governor("Network restored — resuming campaign.")
            return True
        if polls % 5 == 0:  # first refusal, then roughly every 5 minutes
            _print_governor(
                "Network still unreachable — waiting "
                f"(probe every {_NETWORK_WAIT_POLL_SECONDS/60:.0f} min)."
            )
        polls += 1
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=_NETWORK_WAIT_POLL_SECONDS
            )
        except asyncio.TimeoutError:
            continue
    return False


# Session-end reasons that mean "a cap closed the window", not "the campaign
# is finished" — under persistence these wait for the window instead of
# exiting. session_duration_cap is the ordinary between-sessions case; the
# governor's own strings cover mid-session cap hits.
#
# Deliberately ABSENT: "blocked_or_rate_limited" (the forced-backoff trip from
# linkedin/recruiter_recovery.py). That is a detection event on the operator's
# real Recruiter seat, and it ends unattended operation, full stop — a
# persistent campaign auto-resuming into a possibly-flagged account is the one
# move this feature must never make. The operator relaunches after looking at
# the seat.
_CAP_SHAPED_SHUTDOWN_PREFIXES = CAP_SHAPED_SHUTDOWN_PREFIXES


def _should_retry_session_error(
    shutdown_reason: object,
    *,
    consecutive_error_resumes: int,
    profile_opens: int = 0,
    error: BaseException | None = None,
) -> bool:
    """Whether a persistent campaign absorbs this session error and resumes.

    Only UNCLASSIFIED errors ("error: <TypeName>") qualify — the deterministic
    regime errors carry their own named reasons (geography_regime_error,
    constraint_manifest_error, preflight_regime_error) and retrying a config
    error just loops; operator interrupts ("interrupted: ...") are a stop, not
    a fault. Bounded by LINKEDIN_SESSION_ERROR_RETRIES absorptions, counted per
    fault family by the day cycle (``consecutive_error_resumes`` carries that
    family's count); activity resets and the zero-progress backstop live at
    the call site.

    Activity does NOT gate absorption. The guards that detect danger carry the
    soft-block boundary:
    a DETECTED block trips "blocked_or_rate_limited", which is deliberately
    absent from every absorb/wait list and ends unattended operation
    outright; the health classifier re-checks on resume; the retry ceiling
    bounds deterministic faults; and an after-activity absorb rests through
    the full dormant gap, never the short backoff (see the day cycle's
    resume_gap selection). ``profile_opens`` and ``error`` stay in the
    signature for call-site compatibility and the gap selection.
    """
    del profile_opens, error  # retained for call-site compatibility
    if not config.LINKEDIN_CAMPAIGN_PERSIST:
        return False
    if consecutive_error_resumes >= config.LINKEDIN_SESSION_ERROR_RETRIES:
        return False
    return str(shutdown_reason or "").startswith("error:")


# ──────────────────────────────────────────────────────────────────────
# Console output
# ──────────────────────────────────────────────────────────────────────

def _print_governor(msg: str):
    print(f"[governor] {msg}", flush=True)


def _print_decoy(msg: str):
    print(f"[decoy] {msg}", flush=True)


_GEOGRAPHY_REGIME_STOP_MESSAGE = (
    "Stopping day cycle: the brief's geography could not be applied "
    "and verified (fail-closed). Fix permanent_filters['Location'] "
    "to exact Recruiter facet names before resuming."
)

_GEOGRAPHY_APPLY_TRANSIENT_RETRY_MESSAGE = (
    "Geography apply flaked on LinkedIn typeahead variance; retrying the "
    "session once."
)


def _classify_session_exception(exc: BaseException) -> str:
    """Normalize orchestrator-level failures for session accounting."""
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted: KeyboardInterrupt"
    if auth_wall_variant_of(exc) is not None:
        return RECRUITER_REAUTH_REQUIRED
    return f"error: {type(exc).__name__}"


def _classify_session_exception_kind(exc: BaseException) -> ShutdownKind:
    """Return the typed shutdown kind paired with the session reason."""
    if isinstance(exc, KeyboardInterrupt):
        return ShutdownKind.INTERRUPTED
    return ShutdownKind.ERROR


def _session_error_shutdown(exc: BaseException) -> tuple[str, ShutdownKind]:
    """Classify a session-loop exception into ``(shutdown_reason, kind)``.

    Geography, constraint-manifest, and preflight-regime failures are stable
    configuration stops with named reasons. Retryable geography apply misses
    use ``geography_apply_transient`` for one same-cycle retry. These classes
    are classified outside the guarded import so their non-retryable reasons
    survive a classifier fault.

    The classifier runs on the error path and must preserve the original
    exception. The heavy orchestrator import remains lazy; an import failure
    falls back to generic classification of the original exception, while an
    operator interrupt still wins. The browser-context-management signature is
    a stable environment stop that requires an explicit operator Chrome restart.

    A LinkedIn sign-in wall (``AuthWallError``, anywhere in the exception
    chain, or a navigation failure quoting an ``/enterprise-authentication/``
    URL) is ``recruiter_reauth_required``: the day cycle ends on first sight
    rather than starting new sessions into the same wall.
    """
    if auth_wall_variant_of(exc) is not None:
        return RECRUITER_REAUTH_REQUIRED, ShutdownKind.ERROR
    if "Browser context management is not supported" in str(exc):
        return "browser_environment_error", ShutdownKind.ERROR
    # The module-scope classes classify OUTSIDE the guarded block: their
    # stable reasons are what stop the day cycle from absorb-retrying a
    # deterministic regime error, and a classifier fault must not demote
    # them to retryable "error: ..." free text.
    if isinstance(exc, ConstraintManifestError):
        return "constraint_manifest_error", ShutdownKind.ERROR
    if isinstance(exc, PreflightRegimeError):
        return "preflight_regime_error", ShutdownKind.ERROR
    try:
        from linkedin.orchestrator import GeographyRegimeError
    except Exception as classify_error:
        try:
            _print_governor(
                "Session-error classifier failed "
                f"({type(classify_error).__name__}); using the generic "
                "classification for the original error."
            )
        except Exception:
            pass
        return f"error: {exc}", ShutdownKind.ERROR
    if isinstance(exc, GeographyRegimeError):
        if getattr(exc, "retryable", False):
            return "geography_apply_transient", ShutdownKind.ERROR
        return "geography_regime_error", ShutdownKind.ERROR
    return f"error: {exc}", ShutdownKind.ERROR


# Reasons a sign-in wall can hide behind: a fail-soft caller swallowed the
# typed wall and a later gate refused generically, or a geography check read
# the wall page as unreadable chips. Every other named reason is kept.
_WALL_MASKABLE_REASONS = frozenset(
    {"geography_regime_error", "geography_apply_transient"}
)
_LIVE_WALL_CHECK_TIMEOUT_SECONDS = 5.0


def _may_hide_an_auth_wall(shutdown_reason: object) -> bool:
    reason = str(shutdown_reason or "")
    return reason.startswith("error:") or reason in _WALL_MASKABLE_REASONS


async def _live_auth_wall(pipeline: object, exc: BaseException) -> AuthWallError | None:
    """The wall on the session's live page, or None. Reads only; never raises."""
    browser = getattr(pipeline, "browser", None)
    if browser is None:
        return None
    try:
        return await asyncio.wait_for(
            live_auth_wall_error(browser, cause=exc),
            timeout=_LIVE_WALL_CHECK_TIMEOUT_SECONDS,
        )
    except Exception:
        return None


def _resume_has_pending_work(brief_path: str, output_dir: str | None = None) -> bool:
    """Map passive unknown state to the active worker's fail-safe resume."""
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path, state_dir=output_dir
    )
    pending = has_pending_work(state_dir)
    return True if pending is None else pending


class ResumeTargetHasNoRun(RuntimeError):
    """`--resume` was given, but the resolved state dir holds no run to resume."""


def _assert_resume_target_exists(brief_path: str, output_dir: str | None = None) -> None:
    """Refuse a `--resume` that would silently become a FRESH run.

    The trap (found in review): `has_pending_work` returns None when a state dir
    has no readable run, and `_resume_has_pending_work` maps None -> True as a
    fail-safe against a *transient* read miss on a dir that DOES have a run. But
    the orchestrator computes `resume_existing_state = resume and (had_runtime_state
    or has_legacy_state)`; against a dir with no run that is False, so `--resume`
    is silently discarded and a brand-new run starts — live searching and live
    physical saves — on the one authorized window, against whichever project the
    tab happens to sit on.

    So the fail-safe and the trap share the same None. Disambiguate by the thing
    the orchestrator actually keys on: does a run EXIST here? If not, refuse
    loudly (nonzero exit) rather than let resume degrade into a fresh run. A dir
    whose DB is present but transiently unreadable still has a run row once it
    reopens, so this only fires on genuinely runless targets.
    """
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path, state_dir=output_dir
    )
    db_path = state_dir / _RUNTIME_DB_FILENAME
    # Scoped to source='linkedin'. `latest_run_in_state_dir` answers with ANY
    # run in the file (`SELECT id FROM runs ORDER BY id DESC`), so a dir holding
    # only another source's runs would satisfy a LinkedIn resume and then start
    # fresh — the same silent-fresh-run defect one layer in.
    if db_path.exists():
        try:
            with sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True
            ) as conn:
                row = conn.execute(
                    "SELECT id FROM runs WHERE source='linkedin' "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if row is not None:
                return
        except sqlite3.Error:
            # Unreadable but PRESENT: a real run almost certainly lives here and
            # this is a transient read failure, so defer to the existing
            # fail-safe rather than refusing a legitimate resume.
            if latest_run_in_state_dir(db_path) is not None:
                return
            return
    legacy_progress = state_dir / "progress.json"
    if legacy_progress.exists():
        return
    raise ResumeTargetHasNoRun(
        f"--resume was requested but no run exists in {state_dir} "
        f"(no readable run in {_RUNTIME_DB_FILENAME}, no legacy progress.json). "
        "Resuming here would silently start a NEW run and save live candidates. "
        "Point --state-dir at the run you meant to resume, or drop --resume to "
        "start a new run deliberately."
    )


def _parse_restart_strings_arg(raw: str | None) -> list[int]:
    """Parse a comma-separated list of string ids from --restart-strings."""
    if not raw:
        return []

    string_ids: list[int] = []
    for chunk in raw.split(","):
        part = chunk.strip()
        if not part:
            continue
        try:
            string_ids.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid string id '{part}' in --restart-strings") from exc
    return string_ids


# ──────────────────────────────────────────────────────────────────────
# Sourcing session runner
# ──────────────────────────────────────────────────────────────────────

async def _run_sourcing_session(
    brief_path: str,
    output_dir: str | None,
    resume: bool,
    input_mode: str,
    governor: SessionGovernor,
    decoy: DecoyAgent | None,
    session_duration: float,
    restart_string_id: int | None = None,
    restart_string_ids: list[int] | None = None,
    shared_browser=None,
    shared_context=None,
    operator_stop_event: asyncio.Event | None = None,
) -> dict:
    """Run one sourcing session with governor limits and decoy interleaving.

    Returns the pipeline's stats dict.
    """
    from linkedin.orchestrator import Pipeline

    # The governor must be in place BEFORE Pipeline constructs its
    # LinkedInBrowser (governance now attaches at browser construction), so
    # it is passed into Pipeline() directly rather than assigned after the
    # fact — a post-hoc `pipeline._governor = governor` would leave
    # pipeline.browser holding whatever governor (or none) it was
    # constructed with, disconnected from this one.
    pipeline = Pipeline(
        brief_path=brief_path,
        output_dir=output_dir,
        input_mode=input_mode,
        governor=governor,
    )
    if shared_browser is not None:
        pipeline.browser.attach_existing_connection(
            shared_browser,
            context=shared_context,
        )

    # A sourcing-only session has no decoy or interleave machinery. The
    # pipeline connects only after its canonical-state lock is held.
    session_start = time.time()
    last_status_print = session_start
    stop_interleave = asyncio.Event()
    interleave_task: asyncio.Task | None = None
    operational_task_error: BaseException | None = None
    if operator_stop_event is None:
        operator_stop_event = asyncio.Event()
    pipeline._operator_stop_event = operator_stop_event

    if decoy is not None:
        pause_requested = asyncio.Event()
        resume_event = asyncio.Event()
        resume_event.set()
        pipeline._pause_requested = pause_requested
        pipeline._resume_event = resume_event
        interleave_scheduler = BurstScheduler(mode="interleave")

        async def _interleave_loop():
            """Periodically request pause, run decoy burst, then resume sourcing."""
            nonlocal operational_task_error
            try:
                while not stop_interleave.is_set():
                    interval = interleave_scheduler.next_interval()
                    waited = 0.0
                    while waited < interval and not stop_interleave.is_set():
                        chunk = min(5.0, interval - waited)
                        await asyncio.sleep(chunk)
                        waited += chunk

                    if stop_interleave.is_set():
                        break

                    resume_event.clear()
                    pause_requested.set()
                    for _ in range(60):
                        if not pause_requested.is_set():
                            break
                        await asyncio.sleep(1)
                    else:
                        pause_requested.clear()
                        resume_event.set()
                        continue

                    _print_decoy("Interleave burst starting...")
                    results = await decoy.execute_burst()
                    summary = ", ".join(
                        f"{result['type']} ({result.get('duration', 0)}s)"
                        for result in results
                    )
                    _print_decoy(f"Activity burst: {summary}")
                    pause_requested.clear()
                    resume_event.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if operational_task_error is None:
                    operational_task_error = exc
                pause_requested.clear()
                resume_event.set()
                operator_stop_event.set()

        interleave_task = asyncio.create_task(_interleave_loop())

    # Status printer
    async def _status_loop():
        nonlocal last_status_print
        while not stop_interleave.is_set():
            await asyncio.sleep(30)
            now = time.time()
            if now - last_status_print >= 300:  # Every 5 minutes
                _print_governor(governor.status_line())
                last_status_print = now

    status_task = asyncio.create_task(_status_loop())

    # Cooperative session duration cap — sets a flag that the pipeline checks
    # at its next safe checkpoint, instead of hard-cancelling mid-operation.
    session_expired = asyncio.Event()
    pipeline._session_expired = session_expired

    async def _session_timer():
        """Sleep for session duration, then signal expiry."""
        nonlocal operational_task_error
        try:
            waited = 0.0
            while waited < session_duration and not stop_interleave.is_set():
                chunk = min(5.0, session_duration - waited)
                await asyncio.sleep(chunk)
                waited += chunk
            session_expired.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if operational_task_error is None:
                operational_task_error = exc
            operator_stop_event.set()

    timer_task = asyncio.create_task(_session_timer())

    # Run the sourcing pipeline
    #
    # shutdown_kind is the typed classification cooldown.record_session_end()
    # uses to decide cap-counting — shutdown_reason stays free text for the
    # console/log. Session-duration caps, governor limits, and API-budget
    # exhaustion are all controlled stops (COMPLETED, counts toward the
    # daily cap); only an unhandled exception is ERROR (does not count).
    shutdown_reason = None
    shutdown_kind: ShutdownKind | None = None
    session_error: BaseException | None = None
    try:
        await pipeline.run_full(
            resume=resume,
            restart_string_id=restart_string_id,
            restart_string_ids=restart_string_ids,
        )
        shutdown_reason = "pipeline_complete"
        shutdown_kind = ShutdownKind.COMPLETED

    except SessionExpired:
        shutdown_reason = "session_duration_cap"
        shutdown_kind = ShutdownKind.COMPLETED
        _print_governor(f"Session duration cap reached ({session_duration/3600:.1f}h)")

    except GovernorLimitReached as e:
        shutdown_reason = e.reason
        shutdown_kind = ShutdownKind.COMPLETED
        _print_governor(f"Governor limit: {e.reason}")

    except ApiBudgetExhaustedError as e:
        shutdown_reason = "api_budget_exhausted"
        shutdown_kind = ShutdownKind.COMPLETED
        session_error = e
        _print_governor(f"API budget exhausted: {e}")

    except OperatorStopRequested:
        shutdown_reason = "operator_stop"
        shutdown_kind = ShutdownKind.INTERRUPTED

    except (KeyboardInterrupt, asyncio.CancelledError):
        shutdown_reason = "operator_stop"
        shutdown_kind = ShutdownKind.INTERRUPTED

    except Exception as e:
        shutdown_reason, shutdown_kind = _session_error_shutdown(e)
        session_error = e
        if _may_hide_an_auth_wall(shutdown_reason):
            wall = await _live_auth_wall(pipeline, e)
            if wall is not None:
                shutdown_reason, session_error = RECRUITER_REAUTH_REQUIRED, wall
        if shutdown_reason == "geography_regime_error":
            _print_governor(
                f"Geography regime failure (terminal until the operator fixes "
                f"the facet values): {e}"
            )
        elif shutdown_reason == "browser_environment_error":
            _print_governor(
                f"Browser environment error (terminal): {e}"
            )
            _print_governor(
                "This is the stale-instance state left when a Chrome update "
                "replaces the app under the running browser (CLO-150). Quit "
                "the CDP Chrome, relaunch it with --remote-debugging-port=9222 "
                "--user-data-dir=~/.chrome-cdp, log back into Recruiter, then "
                "relaunch the campaign. The launcher --check compares the "
                "running browser against the installed app version."
            )
        elif shutdown_reason == RECRUITER_REAUTH_REQUIRED:
            # The day cycle prints the one operator line for this stop; a
            # "Session error" line here would only repeat it.
            pass
        else:
            _print_governor(f"Session error: {e}")

    finally:
        # Stop background tasks
        stop_interleave.set()
        session_expired.set()
        tasks = [status_task, timer_task]
        if interleave_task is not None:
            tasks.append(interleave_task)
        for task in tasks:
            task.cancel()
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in task_results:
            if isinstance(result, BaseException) and not isinstance(
                result,
                asyncio.CancelledError,
            ):
                detail = str(result).splitlines()[0] if str(result) else ""
                _print_governor(
                    "Ancillary session task failed during cleanup: "
                    f"{type(result).__name__}: {detail}"
                )

    if operational_task_error is not None and session_error is None:
        shutdown_reason, shutdown_kind = _session_error_shutdown(
            operational_task_error
        )
        session_error = operational_task_error
        _print_governor(f"Session control task failed: {operational_task_error}")

    return {
        "stats": pipeline.stats,
        "shutdown_reason": shutdown_reason or governor.shutdown_reason or "unknown",
        "shutdown_kind": shutdown_kind,
        "error": session_error,
    }


# ──────────────────────────────────────────────────────────────────────
# Multi-session cycle
# ──────────────────────────────────────────────────────────────────────

async def run_day_cycle(
    brief_path: str,
    output_dir: str | None,
    input_mode: str = "concurrent",
    multi_session: bool = False,
    with_decoy: bool = False,
    resume: bool = False,
    restart_string_id: int | None = None,
    restart_string_ids: list[int] | None = None,
):
    """Hold the user-scoped Recruiter lock for the entire sourcing cycle."""
    if with_decoy and not multi_session:
        raise ValueError("with_decoy requires multi_session")
    browser_lock = _linkedin_browser_lock()
    browser_lock.acquire()
    try:
        return await _run_day_cycle_with_browser_lock(
            brief_path=brief_path,
            output_dir=output_dir,
            input_mode=input_mode,
            multi_session=multi_session,
            with_decoy=with_decoy,
            resume=resume,
            restart_string_id=restart_string_id,
            restart_string_ids=restart_string_ids,
        )
    finally:
        browser_lock.release()


def _linkedin_browser_lock() -> RuntimeStateLock:
    """Return the one process-held lock for this user's shared CDP browser."""
    lock_dir = (
        Path(tempfile.gettempdir())
        / f"cloris-linkedin-{os.getuid()}"
    )
    return RuntimeStateLock(
        lock_dir,
        filename="recruiter-browser.lock",
        resource_name="LinkedIn Recruiter browser",
    )


async def _run_day_cycle_with_browser_lock(
    brief_path: str,
    output_dir: str | None,
    input_mode: str = "concurrent",
    multi_session: bool = False,
    with_decoy: bool = False,
    resume: bool = False,
    restart_string_id: int | None = None,
    restart_string_ids: list[int] | None = None,
):
    """Run one session, or an explicitly requested multi-session cycle."""
    if with_decoy and not multi_session:
        raise ValueError("with_decoy requires multi_session")

    from linkedin.orchestrator import Pipeline

    Pipeline._validate_judgment_runtime_configuration()
    governor = SessionGovernor()
    try:
        from linkedin.posture_report import describe_posture, format_posture
        from shared.storage import log_event

        rows = describe_posture(input_mode=input_mode, with_decoy=with_decoy)
        for line in format_posture(rows):
            _print_governor(line)
        if output_dir is not None:
            output_path = Path(output_dir)
            if output_path.exists():
                try:
                    log_event(
                        output_path / "run_log.jsonl",
                        "posture_report",
                        rows=[list(r) for r in rows],
                    )
                except Exception:
                    pass
    except Exception:
        pass
    stop_event = asyncio.Event()

    # Graceful first Ctrl+C, hard second Ctrl+C
    _shutting_down = False

    def _signal_handler(sig, frame):
        nonlocal _shutting_down
        if _shutting_down:
            _print_governor("Force shutdown. Saving progress...")
            raise KeyboardInterrupt
        _shutting_down = True
        _print_governor("Shutdown signal received. Finishing current activity... (Ctrl+C again to force)")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if resume:
        # Refuse BEFORE the pending-work gate: a runless target passes that gate
        # (None -> fail-safe True) and then degrades to a fresh live run. This
        # raise is the only thing standing between a mis-pointed --state-dir and
        # an unauthorized new run.
        _assert_resume_target_exists(brief_path, output_dir)

    if resume and not _resume_has_pending_work(brief_path, output_dir):
        _print_governor("No canonical sourcing work remains. Nothing to resume.")
        return

    pw = None
    shared_browser = None
    shared_context = None
    decoy: DecoyAgent | None = None
    geography_retry_used = False
    # Absorb budgets are per fault family (exception type name), reset by real
    # activity rather than one shared strike counter. The zero_progress streak
    # bounds churn when an environment repeatedly cycles exception types; each
    # absorbed zero-open session still starts a real LinkedIn session.
    absorbed_error_families: dict[str, int] = {}
    zero_progress_absorptions = 0
    session_num = 0

    try:
        if with_decoy:
            from rebrowser_playwright.async_api import async_playwright

            pw = await async_playwright().start()
            shared_browser = await pw.chromium.connect_over_cdp(config.CDP_URL)
            if not shared_browser.contexts:
                raise RuntimeError("No browser contexts found. Is Chrome open?")
            shared_context = shared_browser.contexts[0]
            decoy = DecoyAgent(shared_context)

        while not stop_event.is_set():
            if resume and not _resume_has_pending_work(brief_path, output_dir):
                _print_governor("No canonical sourcing work remains. Stopping.")
                break

            can_start, reason = governor.can_start_session(
                session_type="linkedin_sourcing"
            )
            if not can_start:
                if config.LINKEDIN_CAMPAIGN_PERSIST and multi_session:
                    if not await _wait_for_governor_window(governor, stop_event):
                        break
                else:
                    _print_governor(f"Cannot start session: {reason}")
                    break

            session_num += 1
            session_id = cooldown.record_session_start(
                session_type="linkedin_sourcing"
            )
            session_slot = cooldown.get_sessions_recorded_today(
                session_type="linkedin_sourcing"
            )
            opens_remaining = (
                MAX_PROFILE_OPENS_PER_24H - cooldown.get_profile_opens_24h()
            )
            session_duration = _sample_session_duration()
            _print_governor(
                f"Session {session_slot} today starting — "
                f"{opens_remaining} profile opens remaining in 24h budget"
            )
            governor.start_session(session_duration_seconds=session_duration)

            result = {"shutdown_reason": "unknown", "stats": {}, "error": None}
            try:
                result = await _run_sourcing_session(
                    brief_path=brief_path,
                    output_dir=output_dir,
                    resume=resume,
                    input_mode=input_mode,
                    governor=governor,
                    decoy=decoy,
                    session_duration=session_duration,
                    restart_string_id=restart_string_id,
                    restart_string_ids=restart_string_ids,
                    shared_browser=shared_browser,
                    shared_context=shared_context,
                    operator_stop_event=stop_event,
                )
            except BaseException as exc:
                result = {
                    "shutdown_reason": _classify_session_exception(exc),
                    "shutdown_kind": _classify_session_exception_kind(exc),
                    "stats": {},
                    "error": exc,
                }
            finally:
                summary = governor.end_session()
                session_profile_opens = int(summary.get("profile_opens_session", 0) or 0)
                cooldown.record_session_end(
                    session_num=session_id,
                    profile_opens=summary["profile_opens_session"],
                    reason=result.get("shutdown_reason", "unknown"),
                    stats=result.get("stats", {}),
                    shutdown_kind=result.get("shutdown_kind"),
                )
                _print_governor(
                    f"Session {session_slot} today ended — "
                    f"{summary['profile_opens_session']} profile opens | "
                    f"Reason: {result.get('shutdown_reason', 'unknown')}"
                )

            error = result.get("error")
            shutdown_reason = result.get("shutdown_reason")

            if shutdown_reason == RECRUITER_REAUTH_REQUIRED:
                # A LinkedIn sign-in wall ends the day cycle on FIRST sight, in
                # either session mode: never absorbed, no backoff, no new
                # session started into the same wall, and not the 6h
                # detection backoff either — the seat needs a human to sign in
                # or pick a session, then a relaunch with resume.
                _print_governor(auth_wall_operator_line(error))
                raise error if error is not None else AuthWallError(
                    "enterprise_reauth"
                )

            if not multi_session:
                if error is not None:
                    raise error
                break

            if shutdown_reason == "geography_apply_transient":
                if geography_retry_used:
                    _print_governor(_GEOGRAPHY_REGIME_STOP_MESSAGE)
                    raise error or RuntimeError(_GEOGRAPHY_REGIME_STOP_MESSAGE)
                geography_retry_used = True
                _print_governor(_GEOGRAPHY_APPLY_TRANSIENT_RETRY_MESSAGE)
                continue

            if session_profile_opens > 0:
                # Real sourcing work resets the error-family streak; faults
                # around an active session are not one degrading streak.
                absorbed_error_families.clear()
                zero_progress_absorptions = 0

            error_absorbed_this_cycle = False
            if error is not None:
                if (
                    config.LINKEDIN_CAMPAIGN_PERSIST
                    and str(shutdown_reason or "").startswith("error:")
                    and not await _network_is_reachable()
                ):
                    # The ground-truth probe decides local connectivity, never
                    # fault text. Waiting touches neither LinkedIn nor a
                    # provider, so it is unbounded and burns no absorb budget;
                    # named dangerous states use their own non-error reason.
                    _print_governor(
                        f"Network unreachable after session error "
                        f"({shutdown_reason}) — connectivity loss is not a "
                        "strike; waiting for the network."
                    )
                    if not await _wait_for_network(stop_event):
                        break
                    error_absorbed_this_cycle = True
                else:
                    error_family = type(error).__name__
                    family_count = absorbed_error_families.get(error_family, 0)
                    zero_progress_cap = max(
                        6, config.LINKEDIN_SESSION_ERROR_RETRIES * 2
                    )
                    if (
                        zero_progress_absorptions < zero_progress_cap
                    ) and _should_retry_session_error(
                        shutdown_reason,
                        consecutive_error_resumes=family_count,
                        profile_opens=session_profile_opens,
                        error=error,
                    ):
                        absorbed_error_families[error_family] = family_count + 1
                        if session_profile_opens <= 0:
                            zero_progress_absorptions += 1
                        error_absorbed_this_cycle = True
                        resume_gap = (
                            "dormant gap"
                            if session_profile_opens > 0
                            else "brief backoff"
                        )
                        _print_governor(
                            f"Session error absorbed ({error_family} "
                            f"{family_count + 1}/"
                            f"{config.LINKEDIN_SESSION_ERROR_RETRIES}): "
                            f"{shutdown_reason} — {resume_gap}, then resuming."
                        )
                    else:
                        raise error
            elif str(shutdown_reason or "").startswith(
                _CAP_SHAPED_SHUTDOWN_PREFIXES
            ) or shutdown_reason == "session_duration_cap":
                # ANY clean cap-ended session proves the campaign is healthy —
                # not just the one spelling. The governor emits
                # "session_duration (4.2h)", "session_profile_cap (...)" and
                # "24h_profile_cap (...)", so keying the reset to a single
                # literal counted a healthy session in between two errors as
                # "consecutive".
                absorbed_error_families.clear()
                zero_progress_absorptions = 0
            if stop_event.is_set():
                break
            # Legacy: only session_duration_cap continues the cycle. Persist
            # widens that to every cap-shaped reason (mid-session governor
            # caps, forced backoff) — the window closed, the campaign is not
            # finished. pipeline_complete and other clean endings still exit
            # under both: persistence keeps a campaign ALIVE, it never
            # resurrects a completed one.
            continuable = shutdown_reason == "session_duration_cap" or (
                config.LINKEDIN_CAMPAIGN_PERSIST
                and str(shutdown_reason or "").startswith(
                    _CAP_SHAPED_SHUTDOWN_PREFIXES
                )
            )
            if error is None and not continuable:
                break
            resume = True
            restart_string_id = None
            restart_string_ids = None
            if not _resume_has_pending_work(brief_path, output_dir):
                _print_governor("No canonical sourcing work remains. Day cycle complete.")
                break

            can_continue, reason = governor.can_start_session(
                session_type="linkedin_sourcing"
            )
            if not can_continue:
                if config.LINKEDIN_CAMPAIGN_PERSIST:
                    if not await _wait_for_governor_window(governor, stop_event):
                        break
                else:
                    _print_governor(f"No more sessions available: {reason}")
                    break

            # Two different sleeps for two different reasons, never conflated.
            # The DORMANT gap is an anti-detection signature between sessions
            # that actually touched LinkedIn — it must stay long and jittered.
            # The ERROR backoff is provider-hygiene after an absorbed session
            # error with zero LinkedIn activity: there is no detection signal
            # to space out, so the short backoff is sufficient.
            if error_absorbed_this_cycle and session_profile_opens <= 0:
                dormant_duration = _sample_error_backoff(zero_progress_absorptions)
                _print_governor(
                    f"Error backoff: ~{dormant_duration/60:.1f} minutes "
                    f"(attempt {zero_progress_absorptions}), then resuming"
                )
            else:
                dormant_duration = _sample_dormant_duration()
                dormant_min = dormant_duration / 60
                next_time = time.strftime(
                    "%I:%M %p", time.localtime(time.time() + dormant_duration)
                )
                _print_governor(
                    f"Dormant period: ~{dormant_min:.0f} minutes "
                    f"(next session ~{next_time})"
                )
            if decoy is not None:
                await decoy.run_dormant_loop(stop_event, dormant_duration)
            else:
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=dormant_duration
                    )
                except asyncio.TimeoutError:
                    pass
    finally:
        if decoy is not None:
            try:
                await decoy.close()
            except Exception as exc:
                _print_governor(f"Decoy cleanup failed: {exc}")
        if pw is not None:
            try:
                await pw.stop()
            except Exception as exc:
                _print_governor(f"Playwright cleanup failed: {exc}")

    _print_governor("Day cycle complete.")


# ──────────────────────────────────────────────────────────────────────
# Decoy-only mode
# ──────────────────────────────────────────────────────────────────────

async def run_decoy_only():
    """Hold the shared-browser lock for an explicit decoy-only session."""
    browser_lock = _linkedin_browser_lock()
    browser_lock.acquire()
    try:
        await _run_decoy_only_with_browser_lock()
    finally:
        browser_lock.release()


async def _run_decoy_only_with_browser_lock():
    """Run only the decoy agent — no sourcing. Useful for cool-down days."""
    _print_governor("Decoy-only mode — passive browsing only, no sourcing.")

    from rebrowser_playwright.async_api import async_playwright
    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp(config.CDP_URL)
    contexts = browser.contexts
    if not contexts:
        _print_governor("No browser contexts found. Is Chrome open?")
        return

    decoy = DecoyAgent(contexts[0])
    stop_event = asyncio.Event()

    def _signal_handler(sig, frame):
        _print_decoy("Shutdown signal received. Finishing current burst...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run until stopped.
    while not stop_event.is_set():
        # Run in 1 hour chunks so shutdown signals are handled promptly.
        await decoy.run_dormant_loop(stop_event, 3600)

    await decoy.close()
    await pw.stop()
    _print_governor("Decoy-only session complete.")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Session Orchestrator — sourcing pipeline with safety governor"
    )
    parser.add_argument("--brief", help="Path to sourcing brief JSON")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Mutable brief-scoped state directory (default: output/state/linkedin/<brief-id>/)",
    )
    parser.add_argument("--output-dir", default=None, help="Deprecated alias for --state-dir")
    session_mode = parser.add_mutually_exclusive_group()
    session_mode.add_argument(
        "--single-session",
        action="store_true",
        help="Run one sourcing session (the default; retained for compatibility)",
    )
    session_mode.add_argument(
        "--multi-session",
        action="store_true",
        help="Cycle through multiple sourcing sessions",
    )
    parser.add_argument(
        "--with-decoy",
        action="store_true",
        help="Add decoy/interleave activity to explicit multi-session mode",
    )
    parser.add_argument("--decoy-only", action="store_true", help="Run decoy agent only (no sourcing)")
    parser.add_argument("--status", action="store_true", help="Print current 24h stats")
    launch_mode = parser.add_mutually_exclusive_group()
    launch_mode.add_argument(
        "--resume", action="store_true", help="Resume canonical nonterminal work"
    )
    launch_mode.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Explicitly discard a resumable run and regenerate the brief + "
            "execution plan (paid strategy-tier calls). Required when the "
            "state dir already carries a generated brief and resumable "
            "progress and --resume is not passed."
        ),
    )
    parser.add_argument("--restart-string", type=int, default=None, help="Reset a specific string to page 1 (use with --resume)")
    parser.add_argument(
        "--restart-strings",
        default=None,
        help="Reset multiple strings to page 1 before resuming, e.g. 4,11,12,16,27",
    )
    parser.add_argument(
        "--input-mode",
        choices=["concurrent", "away"],
        default="concurrent",
        help="Browser input mode for sourcing sessions",
    )

    args = parser.parse_args()

    if args.status:
        cooldown.print_status()
        return

    if args.decoy_only:
        asyncio.run(run_decoy_only())
        return

    if not args.brief:
        parser.error("--brief is required (unless using --decoy-only or --status)")
    if args.with_decoy and not args.multi_session:
        parser.error("--with-decoy requires --multi-session")

    if not Path(args.brief).exists():
        print(f"Error: Brief file not found: {args.brief}")
        sys.exit(1)

    state_dir = str(
        resolve_linkedin_state_dir(
            brief_path=args.brief,
            state_dir=args.state_dir or args.output_dir,
        )
    )
    enable_console_tee(Path(state_dir))

    try:
        restart_string_ids = _parse_restart_strings_arg(args.restart_strings)
    except ValueError as exc:
        parser.error(str(exc))

    if args.restart_string is not None:
        restart_string_ids.append(args.restart_string)

    # Any restart request implies --resume
    if (args.restart_string is not None or restart_string_ids) and args.fresh:
        parser.error("--restart-string/--restart-strings cannot be used with --fresh")
    if (args.restart_string is not None or restart_string_ids) and not args.resume:
        args.resume = True

    # Fresh-regeneration gate: when durable generated artifacts exist and
    # progress is resumable, regeneration must be an explicit --fresh choice.
    if not args.resume and not args.fresh:
        generated_artifacts = (
            Path(state_dir) / "preflight_v2_brief.json",
            Path(state_dir) / "execution_plan.json",
        )
        # _resume_has_pending_work errs toward True when neither canonical
        # state nor its projection is readable — for this gate that errs toward
        # REFUSING, the cheap failure versus the silent reroll it prevents.
        # The generated-artifact existence check keeps genuinely-new projects
        # (no artifacts yet) flowing through untouched.
        resumable = _resume_has_pending_work(
            args.brief, args.state_dir or args.output_dir
        )
        if any(path.exists() for path in generated_artifacts) and resumable:
            parser.error(
                f"state dir {state_dir} carries generated brief/plan artifacts and "
                "resumable progress — refusing to silently regenerate them. "
                "Pass --resume to continue the existing run, or --fresh to "
                "knowingly discard it and re-pay brief + plan generation."
            )

    asyncio.run(run_day_cycle(
        brief_path=args.brief,
        output_dir=state_dir,
        input_mode=args.input_mode,
        multi_session=args.multi_session,
        with_decoy=args.with_decoy,
        resume=args.resume,
        restart_string_id=args.restart_string,
        restart_string_ids=restart_string_ids,
    ))


if __name__ == "__main__":
    main()
