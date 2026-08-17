"""Phase 3 tests for ``shared.brief_identity``.

Pins the canonical-JSON SHA-256 hash invariants that Run Review and the
brief-drift detector rely on:

- Same logical content → same hash, regardless of key order or whitespace.
- Different content → different hash.
- Hash format includes the ``"sha256:"`` algorithm prefix so a future
  migration to a stronger algorithm is detectable.
- ``compute_brief_identity`` is best-effort: missing/malformed inputs
  return None rather than raising, so the orchestrator's start_run
  path doesn't fail closed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from shared.brief_identity import (
    canonical_brief_hash,
    canonical_brief_snapshot,
    compute_brief_identity,
    hash_current_brief_on_disk,
)


def test_hash_format_has_sha256_prefix() -> None:
    h = canonical_brief_hash({"role_title": "Test"})
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", h)


def test_hash_stable_across_key_order() -> None:
    a = canonical_brief_hash({"role_title": "Test", "geography": "NYC"})
    b = canonical_brief_hash({"geography": "NYC", "role_title": "Test"})
    assert a == b


def test_hash_stable_across_serialization_whitespace(tmp_path: Path) -> None:
    """A brief written with indent=2 and one with indent=None should
    hash identically because we hash the post-load dict."""

    raw = {"role_title": "Test", "minimum_years_experience": 4}

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(raw, indent=2))
    tight = tmp_path / "tight.json"
    tight.write_text(json.dumps(raw, separators=(",", ":")))

    pretty_id = compute_brief_identity(pretty)
    tight_id = compute_brief_identity(tight)
    assert pretty_id is not None
    assert tight_id is not None
    assert pretty_id["brief_content_hash"] == tight_id["brief_content_hash"]


def test_hash_changes_with_content() -> None:
    a = canonical_brief_hash({"role_title": "Engineer"})
    b = canonical_brief_hash({"role_title": "Senior Engineer"})
    assert a != b


def test_compute_brief_identity_returns_all_three_fields(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    raw = {"role_title": "Principal", "linkedin_project": "Test Project"}
    brief_path.write_text(json.dumps(raw))

    identity = compute_brief_identity(brief_path)
    assert identity is not None
    assert identity["brief_path_at_launch"] == str(brief_path)
    assert identity["brief_content_hash"].startswith("sha256:")
    snapshot = json.loads(identity["brief_snapshot_json"])
    assert snapshot == raw


def test_compute_brief_identity_missing_file_returns_none(tmp_path: Path) -> None:
    assert compute_brief_identity(tmp_path / "missing.json") is None


def test_compute_brief_identity_malformed_json_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert compute_brief_identity(bad) is None


def test_compute_brief_identity_non_dict_returns_none(tmp_path: Path) -> None:
    """A JSON array at the top level isn't a brief; identity computation
    must not crash on it."""

    arr = tmp_path / "arr.json"
    arr.write_text(json.dumps(["not", "a", "brief"]))
    assert compute_brief_identity(arr) is None


def test_compute_brief_identity_empty_path_returns_none() -> None:
    assert compute_brief_identity("") is None
    assert compute_brief_identity(None) is None  # type: ignore[arg-type]


def test_hash_current_brief_on_disk_matches_snapshot(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    raw = {"role_title": "Test", "geography": "Remote"}
    brief_path.write_text(json.dumps(raw))

    identity = compute_brief_identity(brief_path)
    current = hash_current_brief_on_disk(brief_path)
    assert identity is not None
    assert current == identity["brief_content_hash"]


def test_hash_current_brief_on_disk_diverges_after_edit(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps({"role_title": "Original"}))
    pinned = canonical_brief_hash({"role_title": "Original"})

    # Recruiter edits the brief mid-run.
    brief_path.write_text(json.dumps({"role_title": "Modified"}))
    current = hash_current_brief_on_disk(brief_path)
    assert current != pinned


def test_canonical_snapshot_is_canonical_form(tmp_path: Path) -> None:
    raw = {"b": 2, "a": 1}
    snap = canonical_brief_snapshot(raw)
    # sort_keys=True means a < b ordering, separators=(",", ":") means tight.
    assert snap == '{"a":1,"b":2}'
