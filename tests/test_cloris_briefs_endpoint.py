"""Tests for the authored-brief picker endpoint (GET /api/briefs).

Pins the contract that drives the homescreen LaunchForm picker:

  - Walks `config/**/brief-*.json` recursively.
  - Excludes `*-draft.json` (AGENTS.md guard pattern — scratch briefs
    must never look runnable by accident).
  - Excludes `.bak-*` backup files.
  - Excludes non-brief JSON files in the same directory.
  - Returns role_title / linkedin_project / linkedin_project_id verbatim
    when present; null otherwise.
  - Sorts most-recently-modified first.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from cloris.api import _scan_authored_briefs


def _write_brief(directory: Path, filename: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(payload))
    return path


# ---- Inclusion / exclusion ------------------------------------------------


class TestScanAuthoredBriefs:
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        # Patch _PROJECT_ROOT so relative_to() doesn't crash on tmp_path.
        # Easier: scan the tmp_path directly; relative_to to itself returns ".".
        from cloris import api as cloris_api

        original_root = cloris_api._paths._PROJECT_ROOT
        original_parent = cloris_api._paths._CONFIG_PARENT
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            assert _scan_authored_briefs(tmp_path) == []
        finally:
            cloris_api._paths._PROJECT_ROOT = original_root
            cloris_api._paths._CONFIG_PARENT = original_parent

    def test_top_level_brief_included(self, tmp_path: Path) -> None:
        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            _write_brief(tmp_path, "brief-fde-v1.json", {
                "role_title": "Forward Deployed Engineer",
                "linkedin_project": "FDE NYC",
                "linkedin_project_id": 3000000007,
            })
            briefs = _scan_authored_briefs(tmp_path)
            assert len(briefs) == 1
            b = briefs[0]
            assert b.role_title == "Forward Deployed Engineer"
            assert b.linkedin_project == "FDE NYC"
            assert b.linkedin_project_id == "3000000007"  # coerced to string
            assert b.path == "brief-fde-v1.json"
            assert b.modified_at != ""
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_nested_brief_included(self, tmp_path: Path) -> None:
        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            _write_brief(tmp_path / "FDE-NYC", "brief-fde-v2.json", {
                "role_title": "FDE",
            })
            briefs = _scan_authored_briefs(tmp_path)
            assert len(briefs) == 1
            assert briefs[0].path == "FDE-NYC/brief-fde-v2.json"
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_draft_excluded(self, tmp_path: Path) -> None:
        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            _write_brief(tmp_path, "brief-foo-draft.json", {"role_title": "x"})
            _write_brief(tmp_path, "brief-foo.json", {"role_title": "Foo"})
            briefs = _scan_authored_briefs(tmp_path)
            paths = [b.path for b in briefs]
            assert "brief-foo.json" in paths
            assert "brief-foo-draft.json" not in paths
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_bak_excluded(self, tmp_path: Path) -> None:
        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            _write_brief(tmp_path, "brief-foo.bak-20260411.json", {"role_title": "x"})
            briefs = _scan_authored_briefs(tmp_path)
            assert briefs == []
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_non_brief_json_excluded(self, tmp_path: Path) -> None:
        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            _write_brief(tmp_path, "evaluation-rubric.json", {"role_title": "x"})
            _write_brief(tmp_path, "brief-foo.json", {"role_title": "Foo"})
            briefs = _scan_authored_briefs(tmp_path)
            paths = [b.path for b in briefs]
            assert "brief-foo.json" in paths
            assert "evaluation-rubric.json" not in paths
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_dev_fixture_briefs_excluded_from_picker(self, tmp_path: Path) -> None:
        """Only ``*-fixture`` dirs are hidden. Legitimate recruiter-authored
        slugs like ``idempotent_role`` round-trip through the catalog (audit
        finding F-1: a hard-coded slug-name set silently invisibilized
        briefs whose titles slugged into one of the fixture names)."""

        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            _write_brief(
                tmp_path / "senior-backend-fintech-fixture",
                "brief.json",
                {"role_title": "Senior Backend Engineer"},
            )
            _write_brief(
                tmp_path / "idempotent_role",
                "brief.json",
                {"role_title": "Idempotent Role"},
            )
            _write_brief(
                tmp_path / "FDL-Colombia",
                "brief-fdl-colombia-v4.json",
                {"role_title": "Junior Frontier Data Lead"},
            )

            briefs = _scan_authored_briefs(tmp_path)

            titles = sorted(b.role_title or "" for b in briefs)
            # ``-fixture`` suffix excluded; recruiter-authored slug stays.
            assert titles == ["Idempotent Role", "Junior Frontier Data Lead"]
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_legitimate_recruiter_slugs_never_silently_excluded(
        self, tmp_path: Path
    ) -> None:
        """Audit finding F-1 regression: every slug previously living in
        ``_HIDDEN_BRIEF_DIRS`` must scan/discover normally now that the
        hard-coded set is gone. Fixture hiding is opt-in via ``*-fixture``."""

        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            slugs = (
                "existing_role",
                "forward_deployed_engineer",
                "idempotent_role",
                "identity_match_role",
                "racy_role",
                "versioned_role",
            )
            for slug in slugs:
                _write_brief(
                    tmp_path / slug,
                    "brief.json",
                    {"role_title": slug.replace("_", " ").title()},
                )
            briefs = _scan_authored_briefs(tmp_path)
            paths = sorted(b.path for b in briefs)
            assert paths == sorted(f"{slug}/brief.json" for slug in slugs)
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_unparseable_json_excluded(self, tmp_path: Path) -> None:
        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            (tmp_path / "brief-broken.json").write_text("{not valid json")
            _write_brief(tmp_path, "brief-good.json", {"role_title": "Good"})
            briefs = _scan_authored_briefs(tmp_path)
            paths = [b.path for b in briefs]
            assert paths == ["brief-good.json"]
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_brief_without_role_title_still_included(self, tmp_path: Path) -> None:
        # Legacy / minimal briefs may lack role_title; the frontend
        # falls back to a humanized filename. The picker still surfaces
        # them so the user can see what's there.
        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            _write_brief(tmp_path, "brief-bare.json", {"some_other_field": 1})
            briefs = _scan_authored_briefs(tmp_path)
            assert len(briefs) == 1
            assert briefs[0].role_title is None
            assert briefs[0].linkedin_project is None
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_sorted_most_recent_first(self, tmp_path: Path) -> None:
        from cloris import api as cloris_api

        repo = Path(__file__).parent.parent
        cloris_api._paths._PROJECT_ROOT = tmp_path
        cloris_api._paths._CONFIG_PARENT = tmp_path
        try:
            old = _write_brief(tmp_path, "brief-old.json", {"role_title": "Old"})
            os.utime(old, (time.time() - 1000, time.time() - 1000))
            recent = _write_brief(tmp_path, "brief-recent.json", {"role_title": "Recent"})
            os.utime(recent, (time.time(), time.time()))
            briefs = _scan_authored_briefs(tmp_path)
            assert [b.path for b in briefs] == ["brief-recent.json", "brief-old.json"]
        finally:
            cloris_api._paths._PROJECT_ROOT = repo
            cloris_api._paths._CONFIG_PARENT = repo

    def test_missing_config_dir_returns_empty(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does-not-exist"
        assert _scan_authored_briefs(nonexistent) == []
