"""Tests for Executive Search Slice 6 — confidentiality enforcement boundary.

Pins the contract:

- :func:`cloris.control_plane.aggregate_briefs` masks ``role_title``
  and zeros ``total_saves`` for ``blind``-class briefs at the
  cross-brief boundary.
- ``referenceable``-class briefs pass through unchanged at the
  brief-aggregator surface (the candidate-name redaction lives in
  the surfaces that emit candidate names — reflection prose / run
  reports / market intel — gated separately by
  :func:`shared.confidentiality.should_emit_candidate_name`).
- ``open``-class briefs (default) are byte-identical to the
  pre-Slice-6 aggregator output.
- :data:`cloris.models.BriefInfo` carries ``confidentiality_class``
  on the wire so the frontend can render the confidentiality pill.
- :func:`cloris.api._scan_authored_briefs` populates
  ``confidentiality_class`` from the V2 raw and defaults to
  ``"open"`` when the field is missing or malformed.

Boundary import-graph guard: every aggregator/emitter named in
the spec's "Confidentiality contract" section MUST import from
``shared.confidentiality`` (or be explicitly excluded with a
TODO). This guards against a future commit that adds a new
aggregator forgetting to route through the helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloris.control_plane import _apply_confidentiality_to_brief_aggregate
from cloris.models import BriefInfo
from shared.confidentiality import (
    BLIND_COUNT_MASK,
    BLIND_TITLE_MASK,
)


def _make_brief_info(
    *,
    confidentiality_class: str = "open",
    role_title: str = "VP Engineering",
    total_saves: int = 12,
) -> BriefInfo:
    return BriefInfo(
        path="config/exec/brief.json",
        role_title=role_title,
        modified_at="2026-04-15T12:00:00Z",
        brief_id="exec-vp-engineering",
        last_run_id=42,
        last_run_at="2026-04-15T11:00:00Z",
        last_run_status="completed",
        last_run_source="exec_search",
        total_runs=1,
        total_saves=total_saves,
        confidentiality_class=confidentiality_class,
    )


# ---------------------------------------------------------------------------
# _apply_confidentiality_to_brief_aggregate
# ---------------------------------------------------------------------------


def test_open_brief_passes_through_unchanged() -> None:
    brief = _make_brief_info(confidentiality_class="open")
    out = _apply_confidentiality_to_brief_aggregate(brief)
    assert out == brief


def test_referenceable_brief_passes_through_at_aggregator_seam() -> None:
    """Referenceable briefs surface title + role + save count to the
    cross-brief library surface; candidate-name gating lives at the
    emission seams (reflection, run reports), not here."""

    brief = _make_brief_info(confidentiality_class="referenceable")
    out = _apply_confidentiality_to_brief_aggregate(brief)
    assert out.role_title == "VP Engineering"
    assert out.total_saves == 12


def test_blind_brief_masks_role_title_and_zeros_save_count() -> None:
    brief = _make_brief_info(confidentiality_class="blind")
    out = _apply_confidentiality_to_brief_aggregate(brief)
    assert out.role_title == BLIND_TITLE_MASK
    assert out.total_saves == 0
    # Other fields stay so the frontend can still link to the brief
    # and surface "blind" pill rendering.
    assert out.brief_id == "exec-vp-engineering"
    assert out.confidentiality_class == "blind"


def test_blind_mask_idempotent() -> None:
    """Re-applying the mask is a no-op — important for any future
    re-aggregation pass that touches the same brief twice."""

    brief = _make_brief_info(confidentiality_class="blind")
    once = _apply_confidentiality_to_brief_aggregate(brief)
    twice = _apply_confidentiality_to_brief_aggregate(once)
    assert once == twice


# ---------------------------------------------------------------------------
# BriefInfo wire contract
# ---------------------------------------------------------------------------


def test_brief_info_carries_confidentiality_class_on_the_wire() -> None:
    brief = BriefInfo(
        path="config/x/brief.json",
        modified_at="2026-04-15T00:00:00Z",
    )
    assert brief.confidentiality_class == "open"
    blind = BriefInfo(
        path="config/x/brief.json",
        modified_at="2026-04-15T00:00:00Z",
        confidentiality_class="blind",
    )
    assert blind.confidentiality_class == "blind"


def test_brief_info_rejects_unknown_confidentiality_class() -> None:
    with pytest.raises(Exception):
        BriefInfo(
            path="config/x/brief.json",
            modified_at="2026-04-15T00:00:00Z",
            confidentiality_class="secret",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# _scan_authored_briefs hydration
# ---------------------------------------------------------------------------


def _v2_brief_payload(confidentiality_class: str | None = None) -> dict:
    payload: dict = {
        "role_title": "VP Engineering",
        "linkedin_project": "exec",
        "capability_areas": [
            {
                "name": "x",
                "description": "y",
                "builder_signals": ["z"],
                "user_signals": [],
            }
        ],
        "depth_distinction": {
            "builder_definition": "a",
            "user_definition": "b",
            "edge_case_guidance": "c",
        },
    }
    if confidentiality_class is not None:
        payload["confidentiality_class"] = confidentiality_class
    return payload


def _write_brief(tmp_path: Path, payload: dict, name: str = "brief.json") -> Path:
    sub = tmp_path / "exec_role"
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / name
    path.write_text(json.dumps(payload))
    return path


def test_scan_authored_briefs_hydrates_explicit_confidentiality_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloris import api as cloris_api

    _write_brief(tmp_path, _v2_brief_payload("blind"))
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    out = cloris_api._scan_authored_briefs(tmp_path)
    assert len(out) == 1
    assert out[0].confidentiality_class == "blind"


def test_scan_authored_briefs_defaults_to_open_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloris import api as cloris_api

    _write_brief(tmp_path, _v2_brief_payload(None))
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    out = cloris_api._scan_authored_briefs(tmp_path)
    assert len(out) == 1
    assert out[0].confidentiality_class == "open"


def test_scan_authored_briefs_defaults_to_open_when_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloris import api as cloris_api

    payload = _v2_brief_payload("blind")
    payload["confidentiality_class"] = 42  # garbage
    _write_brief(tmp_path, payload)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    out = cloris_api._scan_authored_briefs(tmp_path)
    assert len(out) == 1
    assert out[0].confidentiality_class == "open"


# ---------------------------------------------------------------------------
# Boundary import-graph guard
# ---------------------------------------------------------------------------


def test_brief_aggregator_imports_from_shared_confidentiality() -> None:
    """The cross-brief aggregator MUST route through
    `shared.confidentiality`. This guards against a future commit
    that adds a new aggregator branch forgetting the gate."""

    from cloris import control_plane

    source = Path(control_plane.__file__).read_text()
    assert "shared.confidentiality" in source
    assert "_apply_confidentiality_to_brief_aggregate" in source


def test_constants_round_trip_via_module_exports() -> None:
    assert BLIND_TITLE_MASK == "Confidential search"
    # Em-dash, the recruiter-facing mask token for blind save counts.
    assert BLIND_COUNT_MASK == "\u2014"
