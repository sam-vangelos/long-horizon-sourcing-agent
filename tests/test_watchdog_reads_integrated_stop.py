"""CLO-243 integration: the watchdog reads what the integrated store writes.

Slice 3's watchdog tests insert run rows with plain SQL. These build the
runtime DB through the integrated writers instead (``RuntimeStateStore`` /
``RunSafetyCoordinator.finish_run`` with slice 1's ``stop_reason_detail``,
and a real ``run_full`` that meets slice 2's sign-in wall) and check that the
alarm names the wall in plain words.

The store normalises the non-canonical ``recruiter_reauth_required`` to
``fatal_runtime_error`` and keeps the original at the head of the detail, so
the watchdog has to find the wall in ``stop_reason_detail``; this is the
cross-slice seam no single slice could test.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from linkedin.auth_wall import AuthWallError
from linkedin.browser import FacetReconciliationError
from shared import config
from shared.runtime_state.store import RuntimeStateStore
from shared.safety import RunSafetyCoordinator
from tests.test_linkedin_pipeline import _facet_containment_pipeline
from tools import cloris_watchdog as wd

WALL_URL = "https://www.linkedin.com/enterprise-authentication/sessions"
SIGN_IN_TITLE = "Cloris needs you to sign in to LinkedIn Recruiter"
PLAIN_WORDS = "LinkedIn Recruiter wants you to sign in again"
SESSION_SENTENCE_HEAD = (
    "LinkedIn Recruiter says another sign-in to this seat is active and only "
    "one session is allowed"
)


def _probe() -> wd.Probe:
    return wd.Probe(
        pid_alive=lambda _pid: False,
        lock_holders=lambda _path: None,
        governor=lambda: wd.GovernorState(),
    )


def _scan(state_root: Path, home: Path) -> list[wd.Alarm]:
    return wd.scan(
        state_root,
        {"dirs": {}},
        home=home,
        now=datetime.now(timezone.utc),
        probe=_probe(),
    )


@pytest.mark.parametrize("writer", ["store", "coordinator"])
def test_store_written_reauth_stop_alarms_in_plain_words(tmp_path, writer):
    state_dir = tmp_path / "state" / "linkedin" / "9000000003-spl"
    store = RuntimeStateStore(state_dir / wd.DB_FILENAME)
    run_id = store.start_run(
        source="linkedin",
        brief_id="spl-swe",
        output_dir=str(state_dir),
        mode="fresh",
    )
    wall_line = str(AuthWallError("session_conflict", url=WALL_URL)).splitlines()[0]
    if writer == "store":
        store.finish_run(
            run_id,
            "error",
            stop_reason="recruiter_reauth_required",
            stop_reason_detail=wall_line,
        )
    else:
        RunSafetyCoordinator(
            store=store,
            output_dir=state_dir,
            source="linkedin",
            brief_id="spl-swe",
        ).finish_run(
            run_id=run_id,
            status="error",
            stop_reason="recruiter_reauth_required",
            stop_reason_detail=wall_line,
        )

    with store.connect() as conn:
        row = conn.execute(
            "SELECT status, stop_reason, stop_reason_detail FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    # The shape the watchdog has to read: the reason survives only in the detail.
    assert tuple(row) == (
        "error",
        "fatal_runtime_error",
        f"recruiter_reauth_required; {wall_line}",
    )

    home = tmp_path / "watchdog-home"
    files_before = sorted(path.name for path in state_dir.iterdir())
    alarms = _scan(tmp_path / "state", home)
    # Read-only: the scan created nothing in the state dir.
    assert sorted(path.name for path in state_dir.iterdir()) == files_before

    assert [alarm.kind for alarm in alarms] == ["stopped"]
    alarm = alarms[0]
    assert alarm.title == SIGN_IN_TITLE
    assert PLAIN_WORDS in alarm.body
    assert "Detail: recruiter_reauth_required; recruiter_reauth_required: session_conflict: " in alarm.body
    assert SESSION_SENTENCE_HEAD in alarm.body
    assert "No detail was recorded" not in alarm.body
    assert "it hit an error it could not handle" not in alarm.body


def test_a_run_that_meets_the_wall_alarms_as_a_sign_in_not_a_crash(tmp_path):
    """End to end: run_full -> facet strike -> re-anchor lands on the wall ->
    slice 1 records the AuthWallError's first line -> the watchdog scans that
    DB and tells Sam to sign in, not that Cloris crashed."""
    state_root = tmp_path / "state"
    state_dir = state_root / "linkedin" / "test"
    state_dir.mkdir(parents=True)
    p = _facet_containment_pipeline(str(state_dir), string_count=2)
    p.state.context.browser.navigate_to_search = AsyncMock(
        side_effect=AuthWallError("session_conflict", url=WALL_URL)
    )

    async def fake_process(_search_string, _progress):
        raise FacetReconciliationError("search_surface_unready")

    p.state.driver._process_string = fake_process
    with patch.object(
        config, "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True
    ), pytest.raises(AuthWallError):
        asyncio.run(p.run_full(resume=True))

    alarms = _scan(state_root, tmp_path / "watchdog-home")

    assert [alarm.kind for alarm in alarms] == ["stopped"]
    alarm = alarms[0]
    assert alarm.title == SIGN_IN_TITLE
    assert PLAIN_WORDS in alarm.body
    assert (
        "Detail: runtime: AuthWallError: recruiter_reauth_required: session_conflict: "
        in alarm.body
    )
    assert SESSION_SENTENCE_HEAD in alarm.body
    assert "it hit an error it could not handle" not in alarm.body
