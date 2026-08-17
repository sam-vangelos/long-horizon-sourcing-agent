"""S3 containment tests for raw brief_path API inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cloris import api as cloris_api
from cloris.app import create_app
from cloris.worker import BriefPathNotFoundError


@pytest.fixture()
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    return config_dir


@pytest.fixture()
def client(config_dir: Path) -> TestClient:
    return TestClient(create_app())


@pytest.mark.parametrize(
    "raw",
    ["/etc/hosts", "../../.env", "config/nested/brief.txt"],
)
def test_resolver_rejects_paths_outside_config_or_without_json_suffix(
    raw: str, config_dir: Path
) -> None:
    with pytest.raises(BriefPathNotFoundError) as exc_info:
        cloris_api._paths.resolve_brief_path_contained(raw)

    assert str(exc_info.value) == raw


def test_resolver_follows_symlinks_before_containment_check(
    tmp_path: Path, config_dir: Path
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    symlink = config_dir / "brief.json"
    symlink.symlink_to(outside)

    with pytest.raises(BriefPathNotFoundError):
        cloris_api._paths.resolve_brief_path_contained("config/brief.json")


def test_resolver_accepts_relative_and_absolute_nested_config_briefs(
    config_dir: Path,
) -> None:
    nested = config_dir / "roles" / "backend" / "brief.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}")

    assert (
        cloris_api._paths.resolve_brief_path_contained(
            "config/roles/backend/brief.json"
        )
        == nested.resolve()
    )
    assert (
        cloris_api._paths.resolve_brief_path_contained(str(nested))
        == nested.resolve()
    )


@pytest.mark.parametrize("raw", ["/etc/hosts", "../../.env"])
def test_orchestrator_decide_maps_containment_failures_to_existing_404(
    raw: str, client: TestClient
) -> None:
    response = client.post("/api/orchestrator/decide", json={"brief_path": raw})

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"error": "brief_path_not_found", "path": raw}
    }


@pytest.mark.parametrize("raw", ["/etc/hosts", "../../.env"])
def test_pre_launch_investigation_maps_containment_failures_to_existing_404(
    raw: str, client: TestClient
) -> None:
    response = client.post(
        "/api/exec-search/investigate",
        json={"brief_path": raw, "persist": False},
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert set(detail) == {"error", "detail", "brief_path"}
    assert detail["error"] == "brief_not_found"
    assert detail["brief_path"] == raw


@pytest.mark.parametrize("endpoint", ["/api/launch/linkedin", "/api/resume/linkedin"])
@pytest.mark.parametrize("raw", ["/etc/hosts", "../../.env"])
def test_legacy_worker_routes_keep_brief_not_found_contract(
    endpoint: str,
    raw: str,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_spawn(**kwargs: object) -> None:
        pytest.fail(f"uncontained path reached worker spawn: {kwargs}")

    monkeypatch.setattr(
        cloris_api._monolith, "_spawn_worker_for_source", unexpected_spawn
    )
    response = client.post(endpoint, json={"brief_path": raw})

    assert response.status_code == 400
    assert response.json() == {
        "detail": {"error": "brief_path_not_found", "brief_path": raw}
    }


def test_compat_spawn_ingress_rejects_uncontained_path_before_worker(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_spawn(**kwargs: object) -> None:
        pytest.fail(f"uncontained path reached worker spawn: {kwargs}")

    monkeypatch.setattr(
        cloris_api._monolith, "_spawn_worker_for_source", unexpected_spawn
    )

    with pytest.raises(BriefPathNotFoundError):
        cloris_api._spawn_linkedin_worker(
            cloris_api.LaunchLinkedInRequest(brief_path="../../.env"),
            mode="fresh",
        )


@pytest.mark.parametrize("path_style", ["relative", "absolute"])
def test_resume_accepts_persisted_config_brief_path(
    path_style: str,
    config_dir: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brief_path = config_dir / "persisted" / "nested" / "brief.json"
    brief_path.parent.mkdir(parents=True)
    brief_path.write_text("{}")
    raw = (
        "config/persisted/nested/brief.json"
        if path_style == "relative"
        else str(brief_path)
    )
    captured: dict[str, object] = {}

    def fake_spawn(**kwargs: object) -> cloris_api._SpawnResult:
        captured.update(kwargs)
        state_dir = config_dir.parent / "state"
        return cloris_api._SpawnResult(
            pid=4242,
            state_dir=state_dir,
            worker_json_path=state_dir / "worker.json",
        )

    monkeypatch.setattr(cloris_api._monolith, "_spawn_worker_for_source", fake_spawn)
    response = client.post("/api/resume/linkedin", json={"brief_path": raw})

    assert response.status_code == 201
    assert captured["brief_path"] == brief_path.resolve()
    assert captured["mode"] == "resume"
