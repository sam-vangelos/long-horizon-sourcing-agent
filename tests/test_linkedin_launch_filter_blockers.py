"""LinkedIn launch readiness: brief permanent_filters automation gaps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloris.launchers import linkedin_permanent_filter_automation_blockers


@pytest.fixture()
def brief_path(tmp_path: Path) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps({"role_title": "Example", "permanent_filters": {}}))
    return path


def test_no_blockers_for_location_only_filters(brief_path: Path) -> None:
    brief_path.write_text(
        json.dumps({"role_title": "Example", "permanent_filters": {"Location": "NYC"}}),
    )
    assert linkedin_permanent_filter_automation_blockers(str(brief_path)) == []


def test_blocker_when_seniority_is_set(brief_path: Path) -> None:
    brief_path.write_text(
        json.dumps(
            {
                "role_title": "Example",
                "permanent_filters": {"seniority": "Director"},
            }
        ),
    )
    blockers = linkedin_permanent_filter_automation_blockers(str(brief_path))
    assert len(blockers) == 1
    assert blockers[0].kind == "config"
    assert "seniority" in blockers[0].message.lower()


def test_blocker_lists_unknown_permanent_filter_keys(brief_path: Path) -> None:
    brief_path.write_text(
        json.dumps(
            {
                "role_title": "Example",
                "permanent_filters": {"custom_company_list": ["Acme"]},
            }
        ),
    )
    blockers = linkedin_permanent_filter_automation_blockers(str(brief_path))
    assert len(blockers) == 1
    assert "custom_company_list" in blockers[0].message
