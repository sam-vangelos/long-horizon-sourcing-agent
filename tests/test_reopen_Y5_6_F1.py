"""Reopen Y.5.6 (F1) — persisted source_pause + additive-OR spawn gate.

Y.5.2 shipped the env-only launch pause (``CLORIS_PAUSE_LAUNCHES_<SOURCE>``).
F1 adds a DURABLE, server-observable arm: a ``source_pause`` row in the
orchestration DB that the in-process spawn gate reads on its NEXT spawn,
OR'd with the env arm. This pins:

- SERVER-OBSERVABLE: an out-of-process write to ``source_pause`` (simulated by
  writing the store directly, exactly as the CLI arm does) is SEEN by the
  in-process gate — ``_spawn_worker_for_source`` raises ``DomainPausedError``
  with no env flag set and no restart. This is the not-env-theater proof.
- ENV ARM STAYS: the Y.5.2 env arm still raises (monkeypatch.setenv) — the OR's
  first arm is byte-for-byte preserved (its own suite, test_reopen_Y5_2.py,
  also stays green; this re-pins it here at the unit gate).
- ABSENT ROW = NOT PAUSED: a fresh orchestration DB with no ``source_pause``
  row does not block — the gate proceeds past to the brief-path check.
- FAIL-CLOSED: a store-read error blocks the launch with an operator-facing
  HTTP 503 carrying the ``pause_state_unavailable`` error code.
- DISARM: flipping the row's ``paused`` back to 0 stops the gate blocking.
- ISOLATION: a persisted pause on ``researcher`` does not pause ``github``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cloris.api import DomainPausedError, _spawn_worker_for_source
from cloris.api._monolith import BriefPathNotFoundError


@pytest.fixture()
def orch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the orchestration DB resolver to a tmp path.

    ``resolve_orchestration_db_path`` derives from the live ``OUTPUT_ROOT``
    (``OUTPUT_ROOT / "state" / "orchestration" / runtime_state.sqlite3``), so
    patch that. Returns the orchestration state dir for direct store
    construction. Every pause flag is also cleared so the ambient env never
    leaks an arm into a test that's exercising the persisted arm.
    """

    output_root = tmp_path / "output"
    (output_root / "state" / "orchestration").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("shared.output_paths.OUTPUT_ROOT", output_root)
    for src in ("RESEARCHER", "GITHUB", "LINKEDIN"):
        monkeypatch.delenv(f"CLORIS_PAUSE_LAUNCHES_{src}", raising=False)
    return output_root / "state" / "orchestration"


def _store():
    from shared.output_paths import resolve_orchestration_db_path
    from shared.runtime_state.orchestration_store import OrchestrationStateStore

    return OrchestrationStateStore(resolve_orchestration_db_path())


# ---------------------------------------------------------------------------
# 1. SERVER-OBSERVABLE — out-of-process persisted arm is seen by the gate.
# ---------------------------------------------------------------------------


def test_persisted_pause_armed_out_of_process_is_seen_by_gate(orch_root: Path) -> None:
    """Write ``source_pause`` directly (simulating the CLI arm — a DIFFERENT
    process than the gate), then call the gate: it raises ``DomainPausedError``
    with NO env flag set. The in-process gate saw the out-of-process write.

    Discriminator: the brief path does not exist, so absent the persisted arm
    the gate would raise ``BriefPathNotFoundError`` — getting ``DomainPausedError``
    proves the persisted arm fired first."""

    _store().set_source_pause(
        "researcher", paused=True, armed_by="op", reason="vendor outage"
    )

    with pytest.raises(DomainPausedError) as excinfo:
        _spawn_worker_for_source(
            source="researcher",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )
    assert excinfo.value.source == "researcher"


def test_persisted_pause_resume_stops_blocking(orch_root: Path) -> None:
    """Arming then disarming (``paused=False``) leaves the row but stops the
    gate blocking — the gate proceeds to the brief-path check."""

    store = _store()
    store.set_source_pause("researcher", paused=True)
    store.set_source_pause("researcher", paused=False)

    with pytest.raises(BriefPathNotFoundError):
        _spawn_worker_for_source(
            source="researcher",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )


# ---------------------------------------------------------------------------
# 2. ENV ARM STAYS — Y.5.2's first OR arm still raises (re-pinned at the gate).
# ---------------------------------------------------------------------------


def test_env_arm_still_raises_with_no_persisted_row(
    orch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Y.5.2 env arm fires with NO ``source_pause`` row present — proving
    the OR's first arm is intact and short-circuits the persisted read."""

    monkeypatch.setenv("CLORIS_PAUSE_LAUNCHES_RESEARCHER", "1")

    with pytest.raises(DomainPausedError) as excinfo:
        _spawn_worker_for_source(
            source="researcher",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )
    assert excinfo.value.source == "researcher"


# ---------------------------------------------------------------------------
# 3. ABSENT ROW = NOT PAUSED — fresh store, no spurious block.
# ---------------------------------------------------------------------------


def test_absent_row_does_not_pause(orch_root: Path) -> None:
    """A fresh orchestration DB (no source_pause row, no env flag) does not
    block — the gate proceeds past to the brief-path check."""

    # Materialize the orchestration DB so the store/table exists but holds no
    # source_pause row (the gate must read "absent => not paused").
    _store()

    with pytest.raises(BriefPathNotFoundError):
        _spawn_worker_for_source(
            source="researcher",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )


def test_is_source_paused_absent_row_is_false(orch_root: Path) -> None:
    """Unit: ``is_source_paused`` returns False for a source with no row, and
    True only after an arm — the table-level contract under the gate."""

    store = _store()
    assert store.is_source_paused("researcher") is False
    store.set_source_pause("researcher", paused=True)
    assert store.is_source_paused("researcher") is True
    store.set_source_pause("researcher", paused=False)
    assert store.is_source_paused("researcher") is False


# ---------------------------------------------------------------------------
# 4. FAIL-CLOSED — a persisted-read error blocks the launch with a typed 503.
# ---------------------------------------------------------------------------


def test_store_read_error_returns_typed_503_from_launch_endpoint(
    orch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted-read failure blocks the real launch endpoint with a 503."""

    import shared.runtime_state.orchestration_store as orch_mod
    from cloris.app import create_app

    def _boom(*args: object, **kwargs: object):
        raise RuntimeError("simulated orchestration store failure")

    monkeypatch.setattr(orch_mod, "OrchestrationStateStore", _boom)
    monkeypatch.setattr(
        "cloris.api._monolith._resolve_brief_path_or_raise",
        lambda brief_id: Path("/nonexistent/brief.json"),
    )

    response = TestClient(create_app(), raise_server_exceptions=False).post(
        "/api/launch/researcher",
        json={"brief_id": "pause-store-error", "mode": "fresh", "force": True},
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == {
        "error": "pause_state_unavailable",
        "source": "researcher",
        "message": (
            "Pause state could not be read; check the orchestration DB and retry."
        ),
    }


def test_store_read_error_does_not_mask_env_pause(
    orch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env arm still short-circuits before the broken persisted store."""

    import shared.runtime_state.orchestration_store as orch_mod

    def _boom(*args: object, **kwargs: object):
        raise RuntimeError("simulated orchestration store failure")

    monkeypatch.setattr(orch_mod, "OrchestrationStateStore", _boom)
    monkeypatch.setenv("CLORIS_PAUSE_LAUNCHES_RESEARCHER", "1")

    with pytest.raises(DomainPausedError):
        _spawn_worker_for_source(
            source="researcher",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )


# ---------------------------------------------------------------------------
# 5. ISOLATION — a persisted pause on researcher does not pause github.
# ---------------------------------------------------------------------------


def test_persisted_pause_researcher_does_not_pause_github(orch_root: Path) -> None:
    """Arm ``researcher`` in the persisted store; a ``github`` spawn proceeds past
    the gate (reaches the brief-path check) — per-source isolation holds for the
    persisted arm exactly as for the env arm."""

    _store().set_source_pause("researcher", paused=True)

    with pytest.raises(BriefPathNotFoundError):
        _spawn_worker_for_source(
            source="github",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )
