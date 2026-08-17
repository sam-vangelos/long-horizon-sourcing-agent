"""Narrow tests for shared audit-pipeline helpers.

Slice 53a closes two harness bugs:
  - latest_audit_dir() lexicographic sort picks ``states-*`` over real
    timestamped audit dirs because ``s`` sorts after ``2``.
  - audit_report.py renders the audit-dir display path via
    ``Path.relative_to(PROJECT_ROOT)`` and crashes on relative or
    out-of-tree input.

The tests below pin both contracts so the next regression surfaces in
the audit harness rather than in a silent wrong-tree analysis.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import audit_common
from tools.audit_common import (
    PROJECT_ROOT,
    display_audit_path,
    is_audit_dir,
    latest_audit_dir,
)
from tools.audit_surfaces import _safe_slug_fragment, discover_run_targets


@pytest.fixture()
def fake_audit_root(tmp_path, monkeypatch):
    """Point AUDIT_ROOT at a temp tree so tests don't touch the real
    output/audits/ contents."""
    monkeypatch.setattr(audit_common, "AUDIT_ROOT", tmp_path)
    return tmp_path


class TestIsAuditDir:
    def test_timestamp_dir_matches(self, fake_audit_root):
        d = fake_audit_root / "20260516T134954Z"
        d.mkdir()
        assert is_audit_dir(d)

    def test_states_prefix_rejected(self, fake_audit_root):
        d = fake_audit_root / "states-20260505T131715Z"
        d.mkdir()
        assert not is_audit_dir(d)

    def test_arbitrary_name_rejected(self, fake_audit_root):
        d = fake_audit_root / "scratch"
        d.mkdir()
        assert not is_audit_dir(d)

    def test_file_with_audit_name_rejected(self, fake_audit_root):
        f = fake_audit_root / "20260516T134954Z"
        f.write_text("")
        assert not is_audit_dir(f)


class TestLatestAuditDir:
    def test_none_when_root_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit_common, "AUDIT_ROOT", tmp_path / "does-not-exist")
        assert latest_audit_dir() is None

    def test_none_when_root_empty(self, fake_audit_root):
        assert latest_audit_dir() is None

    def test_picks_lexicographically_latest_audit_dir(self, fake_audit_root):
        (fake_audit_root / "20260101T000000Z").mkdir()
        (fake_audit_root / "20260516T134954Z").mkdir()
        (fake_audit_root / "20260509T125841Z").mkdir()
        latest = latest_audit_dir()
        assert latest is not None
        assert latest.name == "20260516T134954Z"

    def test_excludes_states_dirs(self, fake_audit_root):
        # The exact regression: states-* sorted after 2026-* in raw lex
        # order, so the pre-fix code returned the wrong tree here.
        (fake_audit_root / "20260516T134954Z").mkdir()
        (fake_audit_root / "states-20260505T131236Z").mkdir()
        (fake_audit_root / "states-20260505T131715Z").mkdir()
        latest = latest_audit_dir()
        assert latest is not None
        assert latest.name == "20260516T134954Z"

    def test_excludes_unrelated_directories(self, fake_audit_root):
        (fake_audit_root / "20260516T134954Z").mkdir()
        (fake_audit_root / "scratch").mkdir()
        (fake_audit_root / "README.md").write_text("")
        latest = latest_audit_dir()
        assert latest is not None
        assert latest.name == "20260516T134954Z"


class TestDisplayAuditPath:
    def test_in_tree_path_renders_relative(self):
        p = PROJECT_ROOT / "output" / "audits" / "20260516T134954Z"
        assert display_audit_path(p) == "output/audits/20260516T134954Z"

    def test_relative_path_does_not_crash(self, tmp_path, monkeypatch):
        # Pre-fix: audit_report.py:174 called audit_path.relative_to(PROJECT_ROOT)
        # which raised ValueError when the caller passed a relative
        # --audit-dir. display_audit_path resolves first, then guards.
        monkeypatch.chdir(PROJECT_ROOT)
        p = Path("output/audits/20260516T134954Z")
        assert display_audit_path(p) == "output/audits/20260516T134954Z"

    def test_out_of_tree_path_falls_back_to_absolute(self, tmp_path):
        # An audit dir on a different volume / outside PROJECT_ROOT must
        # render the absolute path rather than crash.
        out = tmp_path / "20260516T134954Z"
        out.mkdir()
        rendered = display_audit_path(out)
        assert rendered == str(out.resolve())


# ---------------------------------------------------------------------------
# Slice 53b — run-report slug must include state_key so distinct fixtures
# don't collide. The pre-fix slug was f"run-{source[:2]}-{status[:3]}-{run_id}",
# which produced identical filenames for the two LinkedIn fixtures (same
# source, same status, same run_id).
# ---------------------------------------------------------------------------


class TestSafeSlugFragment:
    def test_replaces_path_separators(self):
        assert _safe_slug_fragment("a/b\\c") == "a-b-c"

    def test_replaces_query_chars(self):
        assert _safe_slug_fragment("brief?id=42&q=x") == "briefid-42-q-x"

    def test_truncates_to_limit(self):
        out = _safe_slug_fragment("x" * 200, limit=24)
        assert len(out) == 24
        assert out == "x" * 24

    def test_empty_input_returns_unknown(self):
        assert _safe_slug_fragment("") == "unknown"


class TestRunReportSlugCollision:
    def test_two_linkedin_fixtures_produce_distinct_slugs(self):
        # Reproduces the collision the harness hit during the original
        # full-surface QA pass: both LinkedIn fixtures, both completed,
        # both run_id=1. Without state_key in the slug, both wrote to
        # the same file and the second overwrote the first.
        api_status = {
            "entries": [
                {
                    "source": "linkedin",
                    "state_key": "head_of_applied_ai_fixture",
                    "latest_run": {"status": "completed", "id": 1},
                },
                {
                    "source": "linkedin",
                    "state_key": "senior_backend_fintech_fixture",
                    "latest_run": {"status": "completed", "id": 1},
                },
            ]
        }
        targets = discover_run_targets(api_status, n=5)
        slugs = [t[0] for t in targets]
        assert len(slugs) == 2
        assert len(set(slugs)) == 2, f"slug collision: {slugs}"

    def test_slug_contains_state_key_fragment(self):
        api_status = {
            "entries": [
                {
                    "source": "linkedin",
                    "state_key": "research_engineer_colombia",
                    "latest_run": {"status": "completed", "id": 7},
                },
            ]
        }
        targets = discover_run_targets(api_status, n=1)
        slug = targets[0][0]
        assert "research_engineer_colombia"[:24] in slug
        assert slug.endswith("-7")
