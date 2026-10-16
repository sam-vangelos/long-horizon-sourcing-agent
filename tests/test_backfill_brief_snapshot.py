"""Tests for `tools/backfill_brief_snapshot.py` — Phase F Slice F4.

Pins the F4 contract:
- Dry-run touches no DB rows.
- `--apply` populates `brief_snapshot_json` for legacy rows where the
  brief still exists in `config/`.
- Idempotent: running `--apply` twice is a no-op.
- Orphaned runs (brief no longer in catalog) are skipped, not erased.
- Per-row errors don't abort the batch.
- The pinned snapshot matches what Phase 3's run-start pinning would
  have written (so drift detection works post-backfill).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import backfill_brief_snapshot as backfill


def _v2_minimal(role: str = "F4 Test") -> dict:
    return {
        "role_title": role,
        "id": role.lower().replace(" ", "_"),
        "linkedin_project_id": role.lower().replace(" ", "_"),
        "capability_areas": [
            {"name": "Eng", "description": "ships systems."}
        ],
        "depth_distinction": {
            "builder_definition": "owns",
            "user_definition": "uses",
            "edge_case_guidance": "borderline",
        },
    }


def _seed_brief(config_dir: Path, role: str = "Backfill Role") -> tuple[Path, str]:
    """Write a real V2 brief and return (path, brief_id)."""

    brief_dir = config_dir / role.replace(" ", "-")
    brief_dir.mkdir(parents=True, exist_ok=True)
    brief_path = brief_dir / "brief.json"
    payload = _v2_minimal(role)
    brief_path.write_text(json.dumps(payload))

    from shared.output_paths import derive_brief_id

    brief_id = derive_brief_id(brief_path=str(brief_path))
    return brief_path, brief_id


def _seed_legacy_run(state_dir: Path, *, source: str, brief_id: str) -> int:
    """Insert a legacy run (empty snapshot) and return its run_id."""

    from shared.runtime_state.store import RuntimeStateStore

    state_dir.mkdir(parents=True, exist_ok=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs(source, brief_id, output_dir, mode, status,
                             stop_reason, started_at, brief_snapshot_json)
            VALUES (?, ?, ?, 'fresh', 'completed', 'normal',
                    '2026-01-01T00:00:00Z', '{}')
            """,
            (source, brief_id, str(state_dir)),
        )
        return int(cursor.lastrowid)


def _read_run(state_dir: Path, run_id: int) -> dict[str, Any]:
    from shared.runtime_state.store import RuntimeStateStore

    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    with store.connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row)


@pytest.fixture()
def backfill_env(tmp_path: Path) -> tuple[Path, Path]:
    """Return (state_root, config_dir) under tmp_path."""

    state_root = tmp_path / "output" / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return state_root, config_dir


def test_dry_run_does_not_modify_rows(backfill_env: tuple[Path, Path]) -> None:
    state_root, config_dir = backfill_env
    _, brief_id = _seed_brief(config_dir, "Dry Run")
    state_dir = state_root / "linkedin" / brief_id
    run_id = _seed_legacy_run(state_dir, source="linkedin", brief_id=brief_id)

    rc = backfill.main(
        [
            "--state-root",
            str(state_root),
            "--config-dir",
            str(config_dir),
        ]
    )
    assert rc == 0

    after = _read_run(state_dir, run_id)
    assert after["brief_snapshot_json"] == "{}"


def test_apply_populates_snapshot(
    backfill_env: tuple[Path, Path],
) -> None:
    state_root, config_dir = backfill_env
    _, brief_id = _seed_brief(config_dir, "Apply Run")
    state_dir = state_root / "linkedin" / brief_id
    run_id = _seed_legacy_run(state_dir, source="linkedin", brief_id=brief_id)

    rc = backfill.main(
        [
            "--apply",
            "--state-root",
            str(state_root),
            "--config-dir",
            str(config_dir),
        ]
    )
    assert rc == 0

    after = _read_run(state_dir, run_id)
    assert after["brief_snapshot_json"] != "{}"
    snapshot = json.loads(after["brief_snapshot_json"])
    assert snapshot["role_title"] == "Apply Run"
    # brief_path_at_launch + brief_content_hash also populated.
    assert after["brief_path_at_launch"]
    assert after["brief_content_hash"]
    # Hash matches the canonical-JSON SHA-256 (Phase 3 convention).
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert after["brief_content_hash"] == expected_hash


def test_apply_is_idempotent(backfill_env: tuple[Path, Path]) -> None:
    state_root, config_dir = backfill_env
    _, brief_id = _seed_brief(config_dir, "Idempotent")
    state_dir = state_root / "linkedin" / brief_id
    run_id = _seed_legacy_run(state_dir, source="linkedin", brief_id=brief_id)

    backfill.main(
        [
            "--apply",
            "--state-root",
            str(state_root),
            "--config-dir",
            str(config_dir),
        ]
    )
    first = _read_run(state_dir, run_id)
    assert first["brief_snapshot_json"] != "{}"

    backfill.main(
        [
            "--apply",
            "--state-root",
            str(state_root),
            "--config-dir",
            str(config_dir),
        ]
    )
    second = _read_run(state_dir, run_id)
    # Second run is a no-op — values unchanged.
    assert second["brief_snapshot_json"] == first["brief_snapshot_json"]
    assert second["brief_content_hash"] == first["brief_content_hash"]


def test_orphaned_run_is_skipped_not_erased(
    backfill_env: tuple[Path, Path],
) -> None:
    """A run whose brief_id has no matching brief in the catalog is
    skipped (counted as an orphan); its existing data is never erased."""

    state_root, config_dir = backfill_env
    state_dir = state_root / "linkedin" / "ghost_brief"
    run_id = _seed_legacy_run(
        state_dir, source="linkedin", brief_id="ghost_brief"
    )
    before = _read_run(state_dir, run_id)

    rc = backfill.main(
        [
            "--apply",
            "--state-root",
            str(state_root),
            "--config-dir",
            str(config_dir),
        ]
    )
    assert rc == 0

    after = _read_run(state_dir, run_id)
    # Orphans untouched.
    assert after["brief_snapshot_json"] == before["brief_snapshot_json"]
    assert after["brief_snapshot_json"] == "{}"


def test_already_pinned_run_not_re_pinned(
    backfill_env: tuple[Path, Path],
) -> None:
    """A run that already carries a non-empty snapshot is left alone
    (the SELECT WHERE clause filters it out)."""

    from shared.runtime_state.store import RuntimeStateStore

    state_root, config_dir = backfill_env
    _, brief_id = _seed_brief(config_dir, "Pinned")
    state_dir = state_root / "linkedin" / brief_id
    run_id = _seed_legacy_run(state_dir, source="linkedin", brief_id=brief_id)

    # Pre-populate the snapshot with a canonical-but-stale value.
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET brief_snapshot_json = ? WHERE id = ?",
            ('{"role_title": "Pinned at launch — do not overwrite"}', run_id),
        )

    backfill.main(
        [
            "--apply",
            "--state-root",
            str(state_root),
            "--config-dir",
            str(config_dir),
        ]
    )

    after = _read_run(state_dir, run_id)
    # The pre-pinned value is preserved — backfill skipped this row.
    assert (
        after["brief_snapshot_json"]
        == '{"role_title": "Pinned at launch — do not overwrite"}'
    )


def test_brief_index_includes_both_sources(
    backfill_env: tuple[Path, Path],
) -> None:
    """A single brief on disk indexes under BOTH (linkedin, ...) and
    (github, ...) keys so a GitHub run pointed at the same brief
    gets backfilled too."""

    _, config_dir = backfill_env
    brief_path, _ = _seed_brief(config_dir, "Both Sources")

    index = backfill._build_brief_index(config_dir)
    sources = {source for source, _ in index.keys()}
    assert "linkedin" in sources
    assert "github" in sources


def test_canonical_payload_hash_is_stable(
    backfill_env: tuple[Path, Path],
) -> None:
    """Canonicalization must be insensitive to key order on disk so
    re-saving the brief in a different order doesn't fake drift."""

    _, config_dir = backfill_env
    brief_path, _ = _seed_brief(config_dir, "Stable Hash")
    canonical_a, hash_a = backfill._canonical_brief_payload(brief_path)

    # Rewrite the brief with reordered top-level keys.
    raw = json.loads(brief_path.read_text())
    reordered = {k: raw[k] for k in reversed(list(raw.keys()))}
    brief_path.write_text(json.dumps(reordered))
    canonical_b, hash_b = backfill._canonical_brief_payload(brief_path)

    assert canonical_a == canonical_b
    assert hash_a == hash_b
