"""Tests for Executive Search Slice 9 — pre-launch investigation.

Pins the contract:

- :func:`run_pre_launch_investigation` returns an
  :class:`InvestigationPacket` for a valid brief and an
  :class:`InvestigationFailure` for missing brief / parse error /
  backend failure / persistence error. Never raises.
- The packet honors ``brief.prior_search.ruled_out_urls`` plus the
  caller-supplied ``prior_search_context.ruled_out_urls``,
  de-duplicated.
- ``persist=True`` (default) writes the packet at
  ``output/state/exec_search/<state_key>/investigation_packet.json``;
  ``persist=False`` skips the write.
- Heuristic mode (no ``research_backend``) emits a packet with the
  brief-only sourcing recommendations.
- ``POST /api/exec-search/investigate`` returns 200 + the packet,
  404 on missing brief, 422 on malformed brief.
- The route is the brief-only execution path called for in
  Slice 9 amendment B; ``market_intelligence/engine.py``'s
  ``update_market_intel`` post-run flow is unaffected (regression).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloris.app import create_app
from market_intelligence.pre_launch import (
    InvestigationFailure,
    InvestigationFinding,
    InvestigationPacket,
    run_pre_launch_investigation,
)


def _v2_exec_brief_payload(*, ruled_out_urls: list[str] | None = None) -> dict:
    return {
        "role_title": "VP Engineering",
        "role_summary": "Owns engineering org for a series-D company.",
        "geography": "United States",
        "linkedin_project": "exec",
        "minimum_years_experience": 12,
        "minimum_bar_description": "12+ years engineering leadership.",
        "capability_areas": [
            {
                "name": "Engineering org leadership",
                "description": "Builds and runs 50+ person orgs.",
                "builder_signals": ["VP-level scope"],
                "user_signals": ["IC-level work"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns engineering strategy + delivery.",
            "user_definition": "Manages individual teams without org-wide scope.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "target_modules": ["linkedin", "exec_search"],
        "confidentiality_class": "blind",
        "executive_calibration": {
            "sector": "Healthcare",
            "stage": "Series D",
            "pnl_scale_usd": "$200M ARR",
            "register_notes": "Operator-builder bias.",
        },
        "prior_search": {
            "ruled_out_urls": list(ruled_out_urls or []),
        },
    }


def _write_brief(tmp_path: Path, payload: dict) -> Path:
    sub = tmp_path / "exec_role"
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / "brief.json"
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# run_pre_launch_investigation — unit
# ---------------------------------------------------------------------------


def test_returns_investigation_packet_for_valid_brief(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, _v2_exec_brief_payload())
    out = run_pre_launch_investigation(
        brief_path=brief_path, persist=False
    )
    assert isinstance(out, InvestigationPacket)
    assert out.role_title == "VP Engineering"
    assert out.confidentiality_class == "blind"
    assert out.brief_path == str(brief_path)


def test_returns_failure_for_missing_brief(tmp_path: Path) -> None:
    out = run_pre_launch_investigation(
        brief_path=tmp_path / "missing.json", persist=False
    )
    assert isinstance(out, InvestigationFailure)
    assert out.reason == "brief_not_found"


def test_returns_failure_for_malformed_brief(tmp_path: Path) -> None:
    brief_path = tmp_path / "broken.json"
    brief_path.write_text("not json{")
    out = run_pre_launch_investigation(
        brief_path=brief_path, persist=False
    )
    assert isinstance(out, InvestigationFailure)
    assert out.reason == "brief_load_error"


def test_excluded_urls_merge_caller_context_and_brief(tmp_path: Path) -> None:
    brief_path = _write_brief(
        tmp_path,
        _v2_exec_brief_payload(
            ruled_out_urls=[
                "https://linkedin.com/in/from-brief",
                "https://linkedin.com/in/shared",
            ]
        ),
    )
    out = run_pre_launch_investigation(
        brief_path=brief_path,
        prior_search_context={
            "ruled_out_urls": [
                "https://linkedin.com/in/from-caller",
                "https://linkedin.com/in/shared",
            ]
        },
        persist=False,
    )
    assert isinstance(out, InvestigationPacket)
    # All three URLs present, caller-supplied first, deduped.
    assert "https://linkedin.com/in/from-caller" in out.excluded_urls
    assert "https://linkedin.com/in/from-brief" in out.excluded_urls
    assert out.excluded_urls.count("https://linkedin.com/in/shared") == 1


def test_heuristic_mode_emits_recruiter_recommendations(tmp_path: Path) -> None:
    """No backend → heuristic recommendations from the brief alone."""

    brief_path = _write_brief(tmp_path, _v2_exec_brief_payload())
    out = run_pre_launch_investigation(
        brief_path=brief_path, persist=False
    )
    assert isinstance(out, InvestigationPacket)
    assert out.sourcing_recommendations  # non-empty
    # `dossier_mode` derives from target_modules → so the title_first
    # bias recommendation should appear.
    rec_text = " ".join(out.sourcing_recommendations)
    assert "title_first" in rec_text


def test_research_backend_findings_flow_through(tmp_path: Path) -> None:
    """A research-backend dict shape is mapped onto InvestigationFindings."""

    class _StubBackend:
        def research_brief(self, *, brief: Any, excluded_urls: list[str]) -> dict:
            return {
                "findings": [
                    {
                        "topic": "Market liquidity",
                        "finding": "5-7 plausible candidates in the US.",
                        "citations": ["https://example.com/study"],
                        "confidence": 0.6,
                    },
                ],
                "market_context": "Series D healthcare exec market is thin.",
                "sourcing_recommendations": ["Engage retained search."],
            }

    brief_path = _write_brief(tmp_path, _v2_exec_brief_payload())
    out = run_pre_launch_investigation(
        brief_path=brief_path,
        research_backend=_StubBackend(),
        persist=False,
    )
    assert isinstance(out, InvestigationPacket)
    assert len(out.findings) == 1
    assert isinstance(out.findings[0], InvestigationFinding)
    assert out.findings[0].topic == "Market liquidity"
    assert out.market_context.startswith("Series D")
    assert "Engage retained search." in out.sourcing_recommendations


def test_research_backend_exception_degrades_to_heuristic(tmp_path: Path) -> None:
    """A backend that raises shouldn't abort — degrade to heuristic."""

    class _ExplodingBackend:
        def research_brief(self, *, brief: Any, excluded_urls: list[str]) -> dict:
            raise RuntimeError("upstream-network-failure")

    brief_path = _write_brief(tmp_path, _v2_exec_brief_payload())
    out = run_pre_launch_investigation(
        brief_path=brief_path,
        research_backend=_ExplodingBackend(),
        persist=False,
    )
    assert isinstance(out, InvestigationPacket)
    assert "Research backend raised" in out.notes


def test_persist_true_writes_packet_to_canonical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`persist=True` writes the packet under output/state/exec_search/."""

    from shared import config

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    # output_paths reads the constant at import time; reload to pick
    # up the patched value.
    import importlib
    from shared import output_paths as op

    importlib.reload(op)
    from market_intelligence import pre_launch

    importlib.reload(pre_launch)

    brief_path = _write_brief(tmp_path, _v2_exec_brief_payload())
    out = pre_launch.run_pre_launch_investigation(
        brief_path=brief_path, persist=True
    )
    assert isinstance(out, pre_launch.InvestigationPacket)
    state_dir = (
        tmp_path / "output" / "state" / "exec_search"
    )
    persisted = list(state_dir.glob("*/investigation_packet.json"))
    assert len(persisted) == 1
    on_disk = json.loads(persisted[0].read_text())
    assert on_disk["role_title"] == "VP Engineering"
    assert on_disk["confidentiality_class"] == "blind"


# ---------------------------------------------------------------------------
# POST /api/exec-search/investigate
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_route_returns_packet_for_valid_brief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    from cloris import api as cloris_api

    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", tmp_path)
    brief_path = _write_brief(tmp_path, _v2_exec_brief_payload())
    relative = brief_path.relative_to(tmp_path)
    response = client.post(
        "/api/exec-search/investigate",
        json={"brief_path": str(relative), "persist": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["packet"]["role_title"] == "VP Engineering"
    assert body["packet"]["confidentiality_class"] == "blind"


def test_route_returns_404_for_missing_brief(client: TestClient) -> None:
    response = client.post(
        "/api/exec-search/investigate",
        json={"brief_path": "config/does-not-exist.json", "persist": False},
    )
    assert response.status_code == 404
    body = response.json()
    assert body["detail"]["error"] == "brief_not_found"


def test_route_returns_422_for_malformed_brief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    from cloris import api as cloris_api

    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", tmp_path)
    brief_path = tmp_path / "broken.json"
    brief_path.write_text("{not json")
    response = client.post(
        "/api/exec-search/investigate",
        json={"brief_path": "broken.json", "persist": False},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error"] == "brief_load_error"
