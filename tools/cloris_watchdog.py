#!/usr/bin/env python3
"""Cloris death watchdog: tell the operator when a run dies, stops, or stalls.

One-shot. launchd runs it every 2 minutes (see
``tools/launchd/com.cloris.watchdog.plist.template`` and ``docs/watchdog.md``).
Each scan reads every ``output/state/**/runtime_state.sqlite3`` READ-ONLY plus
the heartbeat files beside it, decides which alarms are due, and sends them
through :func:`shared.notify.notify_sam` (macOS notification always, Slack DM
when configured).

What it alarms on
-----------------

``stopped``
    A run row that ended since the last scan with an alarming outcome:
    status ``error``, ``failed`` or ``abandoned``; ``interrupted`` for any
    reason except the operator's own stop (``operator_stop``/``operator_pause``) and the
    routine 4-5 hour session cap (``session_expired``); or
    ``governor_limit_reached`` only when it carries LinkedIn's
    rate-limit/block signal (a forced backoff recorded during the run, or a
    detail that says so). Routine cap endings are how a healthy campaign
    sleeps, so they stay quiet, unless their detail records a ``cleanup:``
    or ``finalization:`` failure appended after the run ended (the lock or
    Chrome may be left dirty). ``recruiter_reauth_required`` gets
    its own wording, naming the wall variant when the detail carries it,
    whether it sits in ``stop_reason`` or, after the store's
    normalisation, in ``stop_reason_detail``. The alarm says the reason and
    the detail in plain words, and "No detail was recorded" when the detail is
    absent.

    Bursts collapse. A persistent campaign that fails every few minutes would
    otherwise send six DMs in 25 minutes, so one state dir alarms at most
    once per hour per (status, reason, process state); the rows in between
    are counted and the next alarm says how many.

``exited``
    A ``stopped`` alarm went out while the Cloris process was still alive and
    retrying on its own, and that process has since exited without a newer
    clean ending: "Cloris gave up".

``died``
    A row still marked ``running`` whose process is gone and whose last sign
    of work is older than 5 minutes (15 when only one kind of process
    evidence is available).

``stuck``
    A ``running`` row whose process is alive but which has done no recorded
    work for 45 minutes.

``quiet``
    A ``running`` row whose process cannot be checked at all (a run launched
    by code that predates the ``heartbeat`` file, with no heartbeat and no lock
    holder visible) and no recorded work for 90 minutes.

How it decides
--------------

*Process evidence.* First, who holds ``<state>/runtime_state.lock`` (the
pipeline holds it from before ``start_run`` until after ``finish_run``), read
with ``/usr/sbin/lsof``, which only lists open files and takes no lock.
Second, the heartbeat: ``worker.json`` or the ``heartbeat`` file's per-pid
entries. A pid counts as the run's own only if it first beat no later than
two minutes after the row started and beat again after it started; an admin
tool that opens the store later is a visitor and is ignored. An unparseable
heartbeat is "unknown", never an alarm on its own.

*Last sign of work.* The newest of: the run's latest events, attempts and
work-unit timestamps in the DB; the run's own heartbeat pids' last beats; and
the mtime of ``run_log.jsonl``. ``live-console.log`` is deliberately NOT
used: a 5-minute status printer writes to it even when the pipeline is hung.
Healthy runs on 28-29 Sep went up to about 12 minutes between such writes,
hence the 45-minute stuck threshold.

*Governor waits.* Waits for the 24h profile-open cap, dormant gaps, error
backoffs and network waits all happen BETWEEN run rows, while no row is
``running``, so they can never look like a stall. As a second guard, no
``stuck``/``quiet`` alarm fires while ``~/.sourcing-governor/daily_stats.json``
shows the 24h cap reached or a forced backoff active.

*Old rows.* Anything with no sign of work for 24 hours is a corpse from some
earlier crash and is ignored; DB files untouched for 24 hours are not even
opened. The first time the watchdog sees a state dir it only alarms on
endings from the last 30 minutes.

Read-only discipline
--------------------

The DBs are opened with ``mode=ro`` when a ``-wal`` file exists (so WAL pages
are read) and with ``immutable=1`` when none does, because a ``mode=ro``
reader on an idle WAL DB would create ``-wal``/``-shm`` files in the state
dir. A ``-wal`` without its ``-shm`` is skipped for the scan rather than
letting SQLite create one. Nothing here imports ``shared.runtime_state``:
``RuntimeStateStore.__init__`` migrates, i.e. writes. The watchdog writes
only its own ``watchdog-state.json`` under
``~/Library/Application Support/cloris-watchdog/`` (``CLORIS_WATCHDOG_HOME``
overrides), never under ``output/``.

Flags: ``--dry-run`` prints what it would send and saves nothing;
``--test-alert`` sends one test notification.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import notify  # noqa: E402

# Filenames are hard-coded, not imported from shared.runtime_state (see above).
DB_FILENAME = "runtime_state.sqlite3"
LOCK_FILENAME = "runtime_state.lock"
HEARTBEAT_FILENAME = "heartbeat"
WORKER_FILENAME = "worker.json"
RUN_LOG_FILENAME = "run_log.jsonl"
STATE_FILENAME = "watchdog-state.json"
DEFAULT_STATE_ROOT = PROJECT_ROOT / "output" / "state"
GOVERNOR_STATS = Path.home() / ".sourcing-governor" / "daily_stats.json"
LSOF = "/usr/sbin/lsof"

DIED_AFTER = timedelta(minutes=5)
DIED_AFTER_WEAK_EVIDENCE = timedelta(minutes=15)
STUCK_AFTER = timedelta(minutes=45)
QUIET_AFTER = timedelta(minutes=90)
CORPSE_AFTER = timedelta(hours=24)
FIRST_SIGHT_WINDOW = timedelta(minutes=30)
STOP_COOLDOWN = timedelta(hours=1)
OWNER_GRACE = timedelta(minutes=2)
GOVERNOR_24H_CAP = 400
SEEN_KEEP = 500

QUIET_STATUSES = {"completed", "succeeded", "running"}
SAM_STOP_REASONS = {"operator_stop", "operator_pause"}
REAUTH = "recruiter_reauth_required"
BLOCK_MARKERS = ("blocked", "rate_limit", "rate-limit", "forced backoff")

REASON_WORDS = {
    "fatal_runtime_error": "it hit an error it could not handle",
    "browser_disconnect_unrecovered": "it lost the Chrome connection and could not get it back",
    "api_budget_exhausted": "the model API budget ran out (top up credits, then resume)",
    "lock_conflict": "another Cloris process was holding this run's lock",
    "worker_missing": "its worker process disappeared",
    "governor_limit": "LinkedIn showed a rate-limit or block signal, so unattended work is over until you check the Recruiter seat",
    "session_expired": "the session time cap was reached",
    "normal": "it stopped without naming a reason",
    REAUTH: "LinkedIn Recruiter wants you to sign in again. Sort it out in Chrome yourself, then relaunch",
    "cleanup_failed": (
        "it ended for a routine reason, but cleaning up afterwards failed, so the "
        "run lock or the Chrome connection may be left in a bad state"
    ),
}
CLEANUP_FAILED = "cleanup_failed"
# What each sign-in wall variant (linkedin/auth_wall.py) showed, so the alarm
# never claims the one-session page for a login redirect or a checkpoint.
REAUTH_SHOWED = {
    "session_conflict": "it showed the one-session-at-a-time page",
    "enterprise_reauth": "it showed its Recruiter sign-in page",
    "login_redirect": "it sent the Recruiter tab to its sign-in page",
    "checkpoint": "it put a security checkpoint in front of the seat",
}
REAUTH_VARIANT = re.compile(rf"{REAUTH}:\s*([a-z_]+)")
# A failure appended to a run's detail after the run had already ended:
# ``cleanup: ...`` (sidecar, lock, browser teardown) or ``finalization: ...``
# (checkpoint, snapshot). See linkedin/evaluation_loop.py.
TEARDOWN_FAILURE = re.compile(r"\b(?:cleanup|finalization): ")
# The store caps stop_reason_detail at 500 characters; show all of it.
DETAIL_LIMIT = 520


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _local_hm(value: datetime) -> str:
    return value.astimezone().strftime("%H:%M")


def _minutes(delta: timedelta) -> int:
    return int(delta.total_seconds() // 60)


def _clean(value: Any, limit: int = 200) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- probes (injectable for tests) -------------------------------------------


def pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def lock_holders(lock_path: Path) -> list[int] | None:
    """Pids holding ``lock_path`` open, ``[]`` for none, ``None`` if unknown."""

    if not lock_path.exists() or not Path(LSOF).exists():
        return None
    try:
        completed = subprocess.run(
            [LSOF, "-t", "--", str(lock_path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode not in (0, 1):
        return None
    pids = []
    for line in (completed.stdout or "").split():
        if line.strip().isdigit():
            pids.append(int(line))
    if completed.returncode == 1 and (completed.stderr or "").strip():
        return None  # lsof complained; that is not "nobody holds it"
    return sorted(set(pids))


@dataclass
class GovernorState:
    waiting: bool = False
    why: str = ""
    backoff_recorded_at: datetime | None = None
    backoff_until: datetime | None = None
    backoff_reason: str = ""


def read_governor_state(path: Path = GOVERNOR_STATS, *, now: datetime | None = None) -> GovernorState:
    """Read the governor stats file directly (shared.cooldown's readers write)."""

    now = now or _utcnow()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return GovernorState()
    if not isinstance(data, dict):
        return GovernorState()
    state = GovernorState()
    backoff = data.get("forced_backoff")
    if isinstance(backoff, dict):
        try:
            until = datetime.fromtimestamp(float(backoff.get("until", 0)), timezone.utc)
            recorded = datetime.fromtimestamp(float(backoff.get("recorded_at", 0)), timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            until = recorded = None
        if until is not None and until > now:
            state.waiting = True
            state.why = "forced backoff active"
            state.backoff_until = until
            state.backoff_recorded_at = recorded
            state.backoff_reason = _clean(backoff.get("reason"), 120)
    opens = data.get("profile_opens")
    if isinstance(opens, list):
        cutoff = now.timestamp() - 24 * 3600
        recent = sum(1 for ts in opens if isinstance(ts, (int, float)) and ts > cutoff)
        if recent >= GOVERNOR_24H_CAP and not state.waiting:
            state.waiting = True
            state.why = f"24h profile-open cap reached ({recent})"
    return state


@dataclass
class Probe:
    pid_alive: Callable[[int], bool] = pid_alive
    lock_holders: Callable[[Path], "list[int] | None"] = lock_holders
    governor: Callable[[], GovernorState] = read_governor_state


# --- reading a state dir -------------------------------------------------------


@dataclass
class Beat:
    pid: int
    first_at: datetime
    last_at: datetime


@dataclass
class RunRow:
    id: int
    source: str
    brief_id: str
    status: str
    stop_reason: str
    detail: str | None
    started_at: datetime | None
    ended_at: datetime | None
    last_work: datetime | None = None


@dataclass
class Alarm:
    kind: str
    title: str
    body: str
    dedupe_key: str
    on_sent: Callable[[], None] | None = field(default=None, repr=False)
    process: str = ""


def open_readonly(db_path: Path) -> sqlite3.Connection | None:
    """Open the runtime DB without creating or changing any file in the state dir."""

    wal = db_path.with_name(db_path.name + "-wal")
    shm = db_path.with_name(db_path.name + "-shm")
    quoted = urllib.parse.quote(str(db_path))
    if wal.exists():
        if not shm.exists():
            return None
        uri = f"file:{quoted}?mode=ro"
    else:
        uri = f"file:{quoted}?immutable=1"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def read_beats(state_dir: Path) -> list[Beat]:
    """Heartbeat entries from worker.json or the heartbeat file; [] when unknown."""

    beats: list[Beat] = []
    try:
        worker = json.loads((state_dir / WORKER_FILENAME).read_text())
    except (OSError, ValueError):
        worker = None
    if isinstance(worker, dict):
        pid, first, last = worker.get("pid"), _parse_ts(worker.get("started_at")), _parse_ts(worker.get("heartbeat_at"))
        if isinstance(pid, int) and first and last:
            beats.append(Beat(pid, first, last))
    try:
        heartbeat = json.loads((state_dir / HEARTBEAT_FILENAME).read_text())
    except (OSError, ValueError):
        heartbeat = None
    if isinstance(heartbeat, dict):
        pids = heartbeat.get("pids")
        if isinstance(pids, dict):
            for key, entry in pids.items():
                if not (isinstance(entry, dict) and str(key).isdigit()):
                    continue
                first, last = _parse_ts(entry.get("first_at")), _parse_ts(entry.get("last_at"))
                if first and last:
                    beats.append(Beat(int(key), first, last))
    return beats


def owner_beats(beats: Iterable[Beat], run: RunRow) -> list[Beat]:
    """The heartbeat entries that belong to the process running ``run``."""

    if run.started_at is None:
        return []
    return [
        beat
        for beat in beats
        if beat.first_at <= run.started_at + OWNER_GRACE
        and beat.last_at >= run.started_at - timedelta(seconds=1)
    ]


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def read_runs(conn: sqlite3.Connection, *, since: datetime) -> list[RunRow]:
    has_detail = "stop_reason_detail" in _columns(conn, "runs")
    detail_sql = "stop_reason_detail" if has_detail else "NULL AS stop_reason_detail"
    rows = conn.execute(
        f"""
        SELECT id, source, brief_id, status, stop_reason, {detail_sql}, started_at, ended_at
        FROM runs
        WHERE status = 'running' OR ended_at >= ?
        ORDER BY id ASC
        """,
        (_iso(since),),
    ).fetchall()
    return [
        RunRow(
            id=int(row["id"]),
            source=str(row["source"] or ""),
            brief_id=str(row["brief_id"] or ""),
            status=str(row["status"] or ""),
            stop_reason=str(row["stop_reason"] or ""),
            detail=row["stop_reason_detail"],
            started_at=_parse_ts(row["started_at"]),
            ended_at=_parse_ts(row["ended_at"]),
        )
        for row in rows
    ]


def last_db_work(conn: sqlite3.Connection, run_id: int) -> datetime | None:
    """Newest work timestamp the DB holds for ``run_id``."""

    stamps: list[Any] = []
    row = conn.execute(
        "SELECT created_at FROM events WHERE run_id = ? ORDER BY id DESC LIMIT 1", (run_id,)
    ).fetchone()
    if row:
        stamps.append(row["created_at"])
    for row in conn.execute(
        "SELECT started_at, ended_at FROM candidate_attempts WHERE run_id = ? ORDER BY id DESC LIMIT 5",
        (run_id,),
    ).fetchall():
        stamps.extend([row["started_at"], row["ended_at"]])
    row = conn.execute(
        "SELECT MAX(started_at) AS s, MAX(ended_at) AS e FROM work_units WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row:
        stamps.extend([row["s"], row["e"]])
    parsed = [ts for ts in (_parse_ts(s) for s in stamps) if ts is not None]
    return max(parsed) if parsed else None


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return None


def find_state_dirs(state_root: Path) -> list[Path]:
    """State dirs holding a runtime DB, up to three levels under the root."""

    found: set[Path] = set()
    for pattern in (f"{DB_FILENAME}", f"*/{DB_FILENAME}", f"*/*/{DB_FILENAME}", f"*/*/*/{DB_FILENAME}"):
        for db in state_root.glob(pattern):
            if db.is_file():
                found.add(db.parent)
    return sorted(found)


# --- watchdog state ----------------------------------------------------------


def load_state(home: Path) -> dict[str, Any]:
    try:
        data = json.loads((home / STATE_FILENAME).read_text())
    except (OSError, ValueError):
        return {"dirs": {}}
    if not isinstance(data, dict) or not isinstance(data.get("dirs"), dict):
        return {"dirs": {}}
    return data


def save_state(home: Path, state: dict[str, Any]) -> None:
    try:
        home.mkdir(parents=True, exist_ok=True)
        tmp = home / f"{STATE_FILENAME}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        os.replace(tmp, home / STATE_FILENAME)
    except OSError as exc:
        print(f"[watchdog] could not save state ({type(exc).__name__})", file=sys.stderr)


# --- classification ----------------------------------------------------------


def _is_reauth(run: RunRow) -> bool:
    return run.stop_reason == REAUTH or REAUTH in str(run.detail or "")


def _is_block_signal(run: RunRow, governor: GovernorState) -> bool:
    detail = str(run.detail or "").lower()
    if any(marker in detail for marker in BLOCK_MARKERS):
        return True
    recorded = governor.backoff_recorded_at
    if recorded is None or run.started_at is None:
        return False
    ended = run.ended_at or _utcnow()
    return run.started_at - timedelta(minutes=1) <= recorded <= ended + timedelta(minutes=5)


def _teardown_failed(run: RunRow) -> bool:
    return TEARDOWN_FAILURE.search(str(run.detail or "")) is not None


def stop_outcome(run: RunRow, governor: GovernorState) -> str | None:
    """The reason key to alarm on for a finished row, or None when it is routine."""

    if run.status in QUIET_STATUSES:
        return None
    if _is_reauth(run):
        return REAUTH
    if run.stop_reason in SAM_STOP_REASONS:
        return None
    # A routine stop stays quiet unless a teardown failure was appended to its
    # detail after the run ended: that can leave the lock or Chrome dirty, and
    # the routine status no longer says so.
    if run.status == "interrupted" and run.stop_reason == "session_expired":
        return CLEANUP_FAILED if _teardown_failed(run) else None
    if run.status == "governor_limit_reached":
        if _is_block_signal(run, governor):
            return "governor_limit"
        return CLEANUP_FAILED if _teardown_failed(run) else None
    return run.stop_reason or "normal"


def _where(state_dir: Path, run: RunRow) -> str:
    return f"{run.source} brief {run.brief_id}, run {run.id} ({state_dir.name})"


def _detail_line(run: RunRow) -> str:
    detail = _clean(run.detail, DETAIL_LIMIT)
    return f"Detail: {detail}" if detail else "No detail was recorded."


def _reauth_variant(run: RunRow | None) -> str | None:
    match = REAUTH_VARIANT.search(str(run.detail or "")) if run is not None else None
    return match.group(1) if match and match.group(1) in REAUTH_SHOWED else None


def _stop_words(reason: str, run: RunRow | None = None) -> str:
    if reason == REAUTH:
        variant = _reauth_variant(run)
        if variant is not None:
            then = "relaunch once the detection backoff ends" if variant == "checkpoint" else "relaunch"
            return (
                f"LinkedIn Recruiter wants you to sign in again ({REAUTH_SHOWED[variant]}). "
                f"Sort it out in Chrome yourself, then {then}"
            )
    return REASON_WORDS.get(reason, f"it stopped with reason '{_clean(reason, 60)}'")


# --- the scan -----------------------------------------------------------------


def scan(
    state_root: Path,
    state: dict[str, Any],
    *,
    home: Path,
    now: datetime | None = None,
    probe: Probe | None = None,
) -> list[Alarm]:
    """Return the alarms due now; mutate ``state`` only via each alarm's ``on_sent``
    and for bookkeeping that needs no delivery (seen/folded rows)."""

    now = now or _utcnow()
    probe = probe or Probe()
    governor = probe.governor()
    dirs_state: dict[str, Any] = state.setdefault("dirs", {})
    alarms: list[Alarm] = []
    for state_dir in find_state_dirs(state_root):
        try:
            alarms.extend(
                _scan_dir(state_dir, dirs_state, home=home, now=now, probe=probe, governor=governor)
            )
        except sqlite3.Error as exc:
            print(f"[watchdog] {state_dir.name}: DB unreadable this scan ({type(exc).__name__})", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - one bad dir must not blind the rest
            print(f"[watchdog] {state_dir.name}: scan failed ({type(exc).__name__}: {_clean(exc, 120)})", file=sys.stderr)
    return alarms


def _scan_dir(
    state_dir: Path,
    dirs_state: dict[str, Any],
    *,
    home: Path,
    now: datetime,
    probe: Probe,
    governor: GovernorState,
) -> list[Alarm]:
    key = str(state_dir.resolve())
    db = state_dir / DB_FILENAME
    newest_file = max(
        (t for t in (_mtime(db), _mtime(db.with_name(db.name + "-wal"))) if t is not None),
        default=None,
    )
    first_sight = key not in dirs_state
    record = dirs_state.setdefault(key, {"seen": [], "stop_alarms": {}})
    record.setdefault("seen", [])
    record.setdefault("stop_alarms", {})
    if newest_file is None or now - newest_file > CORPSE_AFTER:
        if not record.get("pending_exit"):
            return []

    conn = open_readonly(db)
    if conn is None:
        print(f"[watchdog] {state_dir.name}: -wal without -shm; skipped this scan", file=sys.stderr)
        return []
    try:
        runs = read_runs(conn, since=now - CORPSE_AFTER)
        for run in runs:
            if run.status == "running":
                run.last_work = last_db_work(conn, run.id)
    finally:
        conn.close()

    beats = read_beats(state_dir)
    alarms: list[Alarm] = []
    running = [run for run in runs if run.status == "running"]
    for run in running:
        alarm = _running_alarm(state_dir, run, beats, now=now, probe=probe, governor=governor)
        if alarm is not None:
            alarms.append(alarm)
    stops = _terminal_alarms(
        state_dir, key, record, runs, beats,
        home=home, now=now, probe=probe, governor=governor, first_sight=first_sight,
    )
    alarms.extend(stops)
    # A "stopped, process exited" alarm already says Cloris is gone; it consumes
    # the pending "gave up" alarm instead of doubling it.
    if not any(alarm.process == "exited" for alarm in stops):
        exit_alarm = _exit_alarm(state_dir, key, record, running, probe=probe)
        if exit_alarm is not None:
            alarms.append(exit_alarm)
    return alarms


def _running_alarm(
    state_dir: Path,
    run: RunRow,
    beats: list[Beat],
    *,
    now: datetime,
    probe: Probe,
    governor: GovernorState,
) -> Alarm | None:
    owners = owner_beats(beats, run)
    signs = [run.last_work, run.started_at, _mtime(state_dir / RUN_LOG_FILENAME)]
    signs.extend(beat.last_at for beat in owners)
    last_sign = max((s for s in signs if s is not None), default=None)
    if last_sign is None:
        return None
    quiet_for = now - last_sign
    if quiet_for > CORPSE_AFTER:
        return None

    holders = probe.lock_holders(state_dir / LOCK_FILENAME)
    owner_alive = [beat.pid for beat in owners if probe.pid_alive(beat.pid)]
    if holders:
        alive: bool | None = True
    elif owner_alive:
        alive = True
    elif holders == [] or owners:
        alive = False
    else:
        alive = None
    strong = holders == [] and bool(owners)

    where = _where(state_dir, run)
    since = f"Last sign of work {_minutes(quiet_for)} min ago, at {_local_hm(last_sign)}."
    if alive is False:
        threshold = DIED_AFTER if strong else DIED_AFTER_WEAK_EVIDENCE
        if quiet_for < threshold:
            return None
        return Alarm(
            kind="died",
            title="Cloris died without saying why",
            body=(
                f"{where} is still marked running, but its process is gone. {since} "
                "Check the launch log, then relaunch when you're ready."
            ),
            dedupe_key=f"died:{state_dir.resolve()}:{run.id}",
        )
    if governor.waiting:
        return None
    if alive is True and quiet_for >= STUCK_AFTER:
        return Alarm(
            kind="stuck",
            title="Cloris looks stuck",
            body=(
                f"{where} is running and its process is alive, but it has recorded no work "
                f"for {_minutes(quiet_for)} min. {since} Look at Chrome and the live console."
            ),
            dedupe_key=f"stuck:{state_dir.resolve()}:{run.id}",
        )
    if alive is None and quiet_for >= QUIET_AFTER:
        return Alarm(
            kind="quiet",
            title="Cloris has gone quiet",
            body=(
                f"{where} is marked running but has recorded no work for {_minutes(quiet_for)} min, "
                f"and I can't tell whether its process is alive. {since}"
            ),
            dedupe_key=f"quiet:{state_dir.resolve()}:{run.id}",
        )
    return None


def _process_state(run: RunRow, beats: list[Beat], probe: Probe) -> tuple[str, list[int]]:
    owners = owner_beats(beats, run)
    if not owners:
        return "unknown", []
    alive = sorted({beat.pid for beat in owners if probe.pid_alive(beat.pid)})
    return ("alive", alive) if alive else ("exited", [])


def _terminal_alarms(
    state_dir: Path,
    key: str,
    record: dict[str, Any],
    runs: list[RunRow],
    beats: list[Beat],
    *,
    home: Path,
    now: datetime,
    probe: Probe,
    governor: GovernorState,
    first_sight: bool,
) -> list[Alarm]:
    seen = set(record["seen"])
    groups: dict[str, list[tuple[RunRow, str, list[int]]]] = {}
    for run in runs:
        if run.status == "running" or run.ended_at is None or run.id in seen:
            continue
        outcome = stop_outcome(run, governor)
        if outcome is None:
            seen.add(run.id)
            pending = record.get("pending_exit")
            if pending and run.id > int(pending.get("run_id", 0)):
                record.pop("pending_exit", None)  # a newer clean ending
            continue
        if first_sight and now - run.ended_at > FIRST_SIGHT_WINDOW:
            seen.add(run.id)
            continue
        if notify.already_sent(f"died:{state_dir.resolve()}:{run.id}", state_dir=home):
            seen.add(run.id)  # the died alarm already covered it
            continue
        process, pids = _process_state(run, beats, probe)
        signature = f"{run.status}|{outcome}|{process}"
        groups.setdefault(signature, []).append((run, outcome, pids))
    record["seen"] = sorted(seen)[-SEEN_KEEP:]

    alarms: list[Alarm] = []
    for signature, members in groups.items():
        cooldown = record["stop_alarms"].get(signature) or {}
        last_at = _parse_ts(cooldown.get("at"))
        if last_at is not None and now - last_at < STOP_COOLDOWN:
            cooldown["folded"] = int(cooldown.get("folded") or 0) + len(members)
            record["stop_alarms"][signature] = cooldown
            record["seen"] = sorted(set(record["seen"]) | {run.id for run, _, _ in members})[-SEEN_KEEP:]
            continue
        alarms.append(_stop_alarm(state_dir, record, signature, members, cooldown, now=now))
    return alarms


def _stop_alarm(
    state_dir: Path,
    record: dict[str, Any],
    signature: str,
    members: list[tuple[RunRow, str, list[int]]],
    cooldown: dict[str, Any],
    *,
    now: datetime,
) -> Alarm:
    run, outcome, pids = members[-1]
    process = signature.rsplit("|", 1)[-1]
    if outcome == REAUTH:
        title = "Cloris needs you to sign in to LinkedIn Recruiter"
    elif outcome == "governor_limit":
        title = "Cloris stopped: LinkedIn rate-limit or block signal"
    else:
        title = f"Cloris stopped ({run.status})"
    lines = [f"{_where(state_dir, run)} ended at {_local_hm(run.ended_at or now)}: {_stop_words(outcome, run)}."]
    lines.append(_detail_line(run))
    if len(members) > 1:
        lines.append(f"{len(members)} sessions in this state dir ended this way since the last check.")
    folded = int(cooldown.get("folded") or 0)
    if folded:
        lines.append(f"{folded} more like this were held back since {_local_hm(_parse_ts(cooldown.get('at')) or now)}.")
    if process == "alive":
        lines.append("The Cloris process is still alive and may retry on its own; I'll tell you if it exits.")
    elif process == "exited":
        lines.append("The Cloris process has exited.")
    ids = [m[0].id for m in members]

    def _on_sent() -> None:
        record["stop_alarms"][signature] = {"at": _iso(now), "folded": 0}
        record["seen"] = sorted(set(record["seen"]) | set(ids))[-SEEN_KEEP:]
        if process == "alive":
            record["pending_exit"] = {
                "run_id": run.id,
                "pids": pids,
                "reason": _stop_words(outcome, run),
                "at": _iso(now),
            }
        elif process == "exited":
            record.pop("pending_exit", None)

    return Alarm(
        kind="stopped",
        title=title,
        body=" ".join(lines),
        dedupe_key=f"stopped:{state_dir.resolve()}:{run.id}",
        on_sent=_on_sent,
        process=process,
    )


def _exit_alarm(
    state_dir: Path,
    key: str,
    record: dict[str, Any],
    running: list[RunRow],
    *,
    probe: Probe,
) -> Alarm | None:
    pending = record.get("pending_exit")
    if not pending or running:
        return None
    pids = [pid for pid in pending.get("pids", []) if isinstance(pid, int)]
    if any(probe.pid_alive(pid) for pid in pids):
        return None
    run_id = int(pending.get("run_id", 0))

    def _on_sent() -> None:
        record.pop("pending_exit", None)

    return Alarm(
        kind="exited",
        title="Cloris gave up",
        body=(
            f"The Cloris process for {state_dir.name} has exited after run {run_id} "
            f"ended because {pending.get('reason', 'of an error')}. Nothing is running for this brief now."
        ),
        dedupe_key=f"exited:{state_dir.resolve()}:{run_id}",
        on_sent=_on_sent,
    )


# --- entry point --------------------------------------------------------------


def run_once(
    state_root: Path,
    home: Path,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
    probe: Probe | None = None,
    send: Callable[..., bool] | None = None,
) -> list[Alarm]:
    """One scan: retry pending deliveries, find alarms, send them, save state."""

    send = send or notify.notify_sam
    state = load_state(home)
    if not dry_run:
        notify.retry_pending(state_dir=home)
    alarms = scan(state_root, state, home=home, now=now, probe=probe)
    for alarm in alarms:
        if dry_run:
            print(f"[watchdog] would send {alarm.kind}: {alarm.title} | {alarm.body}")
            continue
        delivered = send(alarm.kind, alarm.title, alarm.body, alarm.dedupe_key, state_dir=home)
        if delivered or notify.already_sent(alarm.dedupe_key, state_dir=home):
            if alarm.on_sent is not None:
                alarm.on_sent()
            if delivered:
                print(f"[watchdog] sent {alarm.kind}: {alarm.title}", file=sys.stderr)
    if not dry_run:
        save_state(home, state)
    return alarms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="print what would be sent; save nothing")
    parser.add_argument("--test-alert", action="store_true", help="send one test notification and exit")
    args = parser.parse_args(argv)

    if args.test_alert:
        stamp = _utcnow().strftime("%Y%m%dT%H%M%S")
        ok = notify.notify_sam(
            "test",
            "Cloris watchdog test",
            "If you can read this, Cloris alarms reach you.",
            f"test:{stamp}",
            state_dir=notify.default_state_dir(),
        )
        print("test alert delivered" if ok else "test alert reached no transport (see log)")
        return 0 if ok else 1

    try:
        run_once(args.state_root, notify.default_state_dir(), dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 - launchd re-runs us; say what broke
        print(f"[watchdog] scan failed ({type(exc).__name__}: {_clean(exc, 160)})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
