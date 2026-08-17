"""Reopen Y.5.2 — per-domain launch hard-pause kill-switch.

Pins the contract added by Y.5.2:

- ``_spawn_worker_for_source`` consults a ``CLORIS_PAUSE_LAUNCHES_<SOURCE>``
  env gate immediately after the unknown-source check and BEFORE any state
  dir / runs row is created. When the flag is truthy
  (``{"1","true","yes","on"}``, case-insensitive) it raises
  :class:`DomainPausedError` for that source only.
- The pause is a CLEAN REFUSAL, not theater: it propagates through the real
  API launch entry (``POST /api/launch/{source}``) as a first-class HTTP 409
  (``{"error": "domain_paused", ...}``) — never a swallowed success, never a
  raw 500.
- Per-domain isolation: pausing ``researcher`` does not pause ``github``.
- Additive: with the flag unset the spawn behaves exactly as before (no
  ``DomainPausedError``); the brief-keyed candidate path / launcher /
  ``ensure_candidate`` are untouched (grep guard at the bottom of this file).

Gate placement note (verified against ``_spawn_worker_for_source`` source):
the gate fires before ``brief_path.exists()``, so the unit-level gate tests
do NOT need a real brief file on disk — a non-existent path that would
otherwise raise :class:`BriefPathNotFoundError` is the precise discriminator
proving the gate ran *before* that check.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloris import api as cloris_api
from cloris.api import DomainPausedError, _spawn_worker_for_source
from cloris.api._monolith import BriefPathNotFoundError


# ---------------------------------------------------------------------------
# Shared brief seeding (mirrors tests/test_launch_endpoint_generic.py so the
# brief_id the endpoint resolves matches what we compute here).
# ---------------------------------------------------------------------------


def _v2_minimal(role: str) -> dict:
    return {
        "role_title": role,
        "id": role.lower().replace(" ", "_"),
        "linkedin_project_id": role.lower().replace(" ", "_"),
        "capability_areas": [{"name": "Eng", "description": "ships systems."}],
        "depth_distinction": {
            "builder_definition": "owns",
            "user_definition": "uses",
            "edge_case_guidance": "borderline",
        },
    }


def _seed_brief(config_dir: Path, role: str) -> str:
    """Write a real V2 brief under ``config_dir`` and return its brief_id."""

    bdir = config_dir / role.replace(" ", "-")
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "brief.json").write_text(json.dumps(_v2_minimal(role)))

    from shared.output_paths import derive_brief_id

    return derive_brief_id(brief_path=str(bdir / "brief.json"))


class _FakeProcess:
    """Minimal stand-in for ``subprocess.Popen`` — carries only the ``pid``
    attribute ``_spawn_worker_for_source`` reads (``process.pid``)."""

    def __init__(self, pid: int = 99999) -> None:
        self.pid = pid


@pytest.fixture()
def api_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    """Real app + REAL ``_spawn_worker_for_source`` (the GATE is intact).

    The pause gate is exercised for real — the test does NOT stub
    ``_spawn_worker_for_source``, so a paused source raises at the gate and a
    non-paused source flows through the real gate/registry/lock code. Only the
    leaf subprocess machinery is neutralized so a non-paused launch resolves to
    a harmless 201 instead of forking a real worker:

    - ``subprocess.Popen`` → returns a :class:`_FakeProcess` (no real fork).
    - ``wait_for_sidecar`` → stubbed ``True`` (no real worker writes a
      sidecar, so the real function would time out and — post P7.3 — the
      spawn helper now treats that as :class:`WorkerDidNotStartError`;
      the stub simulates "the worker started fine" so this fixture can
      still test the pause gate, not the sidecar-observation gate).

    A paused source never reaches either stub — the gate raises first — so this
    neutralization cannot mask the pause. Readiness blockers are forced empty.
    ``raise_server_exceptions=False`` so an *unhandled* 500 (e.g. a missing 409
    handler regression) surfaces as a response we can assert on rather than
    blowing up the test.
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    monkeypatch.setattr(
        cloris_api._monolith, "_readiness_blockers", lambda source, brief_id: []
    )

    def _fake_popen(*args: Any, **kwargs: Any) -> _FakeProcess:
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        cloris_api._monolith,
        "wait_for_sidecar",
        lambda *args, **kwargs: True,
    )

    return TestClient(_create_app()), config_dir


def _create_app():
    from cloris.app import create_app

    return create_app()


# ---------------------------------------------------------------------------
# 1. Gate raises (unit) — direct _spawn_worker_for_source call.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag_value", ["1", "true", "TRUE", "yes", "on", "On"])
def test_gate_raises_domain_paused_for_researcher(
    monkeypatch: pytest.MonkeyPatch, flag_value: str
) -> None:
    """With ``CLORIS_PAUSE_LAUNCHES_RESEARCHER`` truthy, spawning ``researcher``
    raises :class:`DomainPausedError` — and does so before the
    brief-path-existence check (the path below does not exist, yet we get
    ``DomainPausedError``, not :class:`BriefPathNotFoundError`)."""

    monkeypatch.setenv("CLORIS_PAUSE_LAUNCHES_RESEARCHER", flag_value)

    with pytest.raises(DomainPausedError) as excinfo:
        _spawn_worker_for_source(
            source="researcher",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )

    assert excinfo.value.source == "researcher"
    msg = str(excinfo.value)
    assert "researcher" in msg
    assert "CLORIS_PAUSE_LAUNCHES_RESEARCHER" in msg


# ---------------------------------------------------------------------------
# 2. Propagation (load-bearing) — pause surfaces as a clean 409 from the real
#    API launch entry, NOT a success, NOT a raw 500.
# ---------------------------------------------------------------------------


def test_pause_propagates_as_clean_409_from_real_launch_endpoint(
    api_client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real ``POST /api/launch/{source}`` path, with the REAL spawn helper
    (no stub), surfaces the pause as HTTP 409 ``domain_paused`` — proving the
    raise is caught by ``_launch_for_source_impl``'s typed-allowlist except
    chain rather than swallowed to success or leaking as a raw 500."""

    api, config_dir = api_client
    brief_id = _seed_brief(config_dir, role="Pause Researcher")

    monkeypatch.setenv("CLORIS_PAUSE_LAUNCHES_RESEARCHER", "1")

    # force=True skips readiness so we reach the spawn gate deterministically.
    resp = api.post(
        "/api/launch/researcher",
        json={"brief_id": brief_id, "mode": "fresh", "force": True},
    )

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "domain_paused"
    assert detail["source"] == "researcher"
    assert "paused" in detail["message"].lower()


def test_pause_is_not_swallowed_to_success(
    api_client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit anti-theater assertion: a paused launch must NOT return a 201
    success envelope. (Distinct from the 409 assertion so a future change that
    accidentally maps the pause to 2xx is caught even if the status-code set
    is loosened.)"""

    api, config_dir = api_client
    brief_id = _seed_brief(config_dir, role="Pause No Success")

    monkeypatch.setenv("CLORIS_PAUSE_LAUNCHES_RESEARCHER", "1")

    resp = api.post(
        "/api/launch/researcher",
        json={"brief_id": brief_id, "mode": "fresh", "force": True},
    )

    assert resp.status_code != 201, resp.text
    # And it is not a raw 500 either — it is the intentional refusal.
    assert resp.status_code == 409, resp.text


# ---------------------------------------------------------------------------
# 3. Per-domain isolation — pausing researcher does not pause github.
# ---------------------------------------------------------------------------


def test_pause_researcher_does_not_pause_github(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CLORIS_PAUSE_LAUNCHES_RESEARCHER`` set; a ``github`` spawn proceeds
    PAST the gate. Proven by the github spawn reaching the brief-path check
    (raising :class:`BriefPathNotFoundError` for a non-existent path) rather
    than :class:`DomainPausedError`."""

    monkeypatch.setenv("CLORIS_PAUSE_LAUNCHES_RESEARCHER", "1")
    # Ensure no stray github pause flag is set in the ambient env.
    monkeypatch.delenv("CLORIS_PAUSE_LAUNCHES_GITHUB", raising=False)

    with pytest.raises(BriefPathNotFoundError):
        _spawn_worker_for_source(
            source="github",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )


def test_pause_researcher_does_not_pause_github_at_api_layer(
    api_client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end isolation, strongest form: with only RESEARCHER paused, a
    ``github`` launch flows all the way through the real gate to a 201 success
    (the leaf subprocess machinery is faked by the fixture, so this is a clean
    spawn, not a real worker). Same process env, same request — researcher
    refuses, github proceeds. That contrast is the isolation proof."""

    api, config_dir = api_client
    brief_id = _seed_brief(config_dir, role="Github Not Paused")

    monkeypatch.setenv("CLORIS_PAUSE_LAUNCHES_RESEARCHER", "1")
    monkeypatch.delenv("CLORIS_PAUSE_LAUNCHES_GITHUB", raising=False)

    resp = api.post(
        "/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh", "force": True},
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["source"] == "github"
    assert body["pid"] == 99999  # from _FakeProcess — proves it reached spawn

    # And the same client, same env, refuses researcher — the contrast that
    # makes this an isolation proof rather than a bare success assertion.
    researcher_brief = _seed_brief(config_dir, role="Github Iso Researcher")
    researcher_resp = api.post(
        "/api/launch/researcher",
        json={"brief_id": researcher_brief, "mode": "fresh", "force": True},
    )
    assert researcher_resp.status_code == 409, researcher_resp.text
    assert researcher_resp.json()["detail"]["error"] == "domain_paused"


# ---------------------------------------------------------------------------
# 4. Unset == unchanged — no DomainPausedError when the flag is absent.
# ---------------------------------------------------------------------------


def test_unset_flag_does_not_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``CLORIS_PAUSE_LAUNCHES_RESEARCHER`` unset, the researcher spawn
    proceeds past the gate exactly as before — reaching the brief-path check
    (``BriefPathNotFoundError``), never ``DomainPausedError``."""

    monkeypatch.delenv("CLORIS_PAUSE_LAUNCHES_RESEARCHER", raising=False)

    with pytest.raises(BriefPathNotFoundError):
        _spawn_worker_for_source(
            source="researcher",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", "  "])
def test_falsy_flag_values_do_not_pause(
    monkeypatch: pytest.MonkeyPatch, falsy: str
) -> None:
    """Falsy / empty / whitespace flag values are NOT a pause — they fall
    through to normal behavior (brief-path check)."""

    monkeypatch.setenv("CLORIS_PAUSE_LAUNCHES_RESEARCHER", falsy)

    with pytest.raises(BriefPathNotFoundError):
        _spawn_worker_for_source(
            source="researcher",
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )


# ---------------------------------------------------------------------------
# 5. Additive guard — the brief-keyed candidate path / launcher /
#    ensure_candidate are untouched by the pause feature.
# ---------------------------------------------------------------------------


def test_additive_pause_does_not_leak_into_candidate_path() -> None:
    """Grep guard: the pause env key / DomainPausedError must live ONLY in the
    API monolith + its re-export, never in the launcher registry, the worker,
    or the runtime-state store (where the brief-keyed candidate path +
    ensure_candidate live). A leak here would mean the pause changed candidate
    accretion behavior — it must not."""

    repo_root = Path(__file__).resolve().parents[1]
    forbidden_files = [
        repo_root / "cloris" / "launchers" / "__init__.py",
        repo_root / "shared" / "runtime_state" / "store.py",
    ]
    for path in forbidden_files:
        text = path.read_text()
        assert "CLORIS_PAUSE_LAUNCHES" not in text, (
            f"pause env key leaked into {path}"
        )
        assert "DomainPausedError" not in text, (
            f"DomainPausedError leaked into {path}"
        )

    # ensure_candidate still exists where it always did (sanity: we didn't
    # accidentally move/rename the candidate-accretion seam).
    store_text = (repo_root / "shared" / "runtime_state" / "store.py").read_text()
    assert "def ensure_candidate(" in store_text
