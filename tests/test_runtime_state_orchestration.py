"""Tests for the orchestration runtime state SQLite store.

Multi-agent execution Slice 2.3. Pins the schema home for chief-of-staff
runs (brief-grain cross-source) and cross-brief playbook observations
(per-principal calibration log) before Phase 2.5 (the first writer)
lands. Tests exercise:

- Schema bootstrap is idempotent and creates both tables with the
  documented columns.
- Path resolvers (``resolve_orchestration_state_dir``,
  ``resolve_orchestration_db_path``) follow ``OUTPUT_ROOT`` so tests
  can monkeypatch the live root and the resolvers re-derive.
- Round-trip: a writer inserts a CoS run with a dispatch plan; the
  read helper :func:`chief_of_staff_run_by_brief` returns the row and
  the dispatch_plan_json round-trips byte-identical.
- Append + aggregate: multiple cross-brief observations for the same
  principal are appended; :func:`cross_brief_observations_for_principal`
  returns them in reverse-chronological order, narrows by market_key /
  role_shape, and yields empty for unknown principals.
- The reader uses ``mode=ro`` so a missing or unreadable DB collapses
  to ``None`` / empty tuple rather than raising.

The store class only ships ``__init__``/``connect``/``initialize`` per
the slice card constraint ("Do NOT write production code that uses
these tables yet — Phase 2.5 is the first writer"). Tests use raw SQL
through ``store.connect()`` to write, then read through the public
read helpers — exactly the seam Phase 2.5 will plug into.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shared.output_paths as output_paths
from shared.runtime_state.orchestration_store import (
    CURRENT_ORCHESTRATION_SCHEMA_VERSION,
    OrchestrationStateStore,
)
from shared.runtime_state.read_models import (
    ChiefOfStaffRunRecord,
    CrossBriefObservation,
    chief_of_staff_run_by_brief,
    cross_brief_observations_for_principal,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Path to a fresh orchestration SQLite under tmp_path."""

    return tmp_path / "orchestration" / "runtime_state.sqlite3"


def _insert_cos_run(
    store: OrchestrationStateStore,
    *,
    brief_id: str,
    principal_id: str = "",
    status: str = "running",
    dispatch_plan: dict | None = None,
    invocation_order: list | None = None,
    handoff_payloads: dict | None = None,
    synthesis_output: dict | None = None,
    started_at: str = "2026-05-04T17:00:00+00:00",
    ended_at: str | None = None,
) -> int:
    """Insert one chief_of_staff_runs row via raw SQL.

    Phase 2.5 will add a typed writer; until then the test exercises the
    schema directly. Returns the new row id.
    """

    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO chief_of_staff_runs(
                brief_id, principal_id, status,
                dispatch_plan_json, invocation_order_json,
                handoff_payloads_json, synthesis_output_json,
                started_at, ended_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                brief_id,
                principal_id,
                status,
                json.dumps(dispatch_plan or {}, sort_keys=True),
                json.dumps(invocation_order or []),
                json.dumps(handoff_payloads or {}, sort_keys=True),
                json.dumps(synthesis_output or {}, sort_keys=True),
                started_at,
                ended_at,
            ),
        )
        return int(cursor.lastrowid)


def _append_observation(
    store: OrchestrationStateStore,
    *,
    principal_id: str,
    market_key: str = "",
    role_shape: str = "",
    brief_id: str,
    observation: dict | None = None,
    created_at: str,
) -> int:
    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO cross_brief_playbook_observations(
                principal_id, market_key, role_shape, brief_id,
                observation_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                principal_id,
                market_key,
                role_shape,
                brief_id,
                json.dumps(observation or {}, sort_keys=True),
                created_at,
            ),
        )
        return int(cursor.lastrowid)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


def test_initialize_creates_both_tables_and_pins_schema_version(db_path: Path) -> None:
    """Bootstrap is what this slice exists to ship — both tables must
    exist with the documented columns and the schema version pinned in
    meta. Phase 2.5+ readers / writers depend on this shape."""

    store = OrchestrationStateStore(db_path)
    assert db_path.exists()

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "meta",
            "chief_of_staff_runs",
            "cross_brief_playbook_observations",
        }.issubset(tables)

        cos_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(chief_of_staff_runs)").fetchall()
        }
        assert cos_columns == {
            "id",
            "brief_id",
            "principal_id",
            "status",
            "dispatch_plan_json",
            "invocation_order_json",
            "handoff_payloads_json",
            "synthesis_output_json",
            "started_at",
            "ended_at",
        }

        obs_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(cross_brief_playbook_observations)"
            ).fetchall()
        }
        assert obs_columns == {
            "id",
            "principal_id",
            "market_key",
            "role_shape",
            "brief_id",
            "observation_json",
            "created_at",
        }

        version_row = conn.execute(
            "SELECT value FROM meta WHERE key = 'orchestration_schema_version'"
        ).fetchone()
        assert version_row is not None
        assert version_row["value"] == CURRENT_ORCHESTRATION_SCHEMA_VERSION

    # Re-instantiating against the same file is a no-op (CREATE TABLE
    # IF NOT EXISTS, INSERT OR REPLACE on meta) — the writer pattern
    # mirrored from RuntimeStateStore / IdentityStore relies on this.
    OrchestrationStateStore(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        version_row = conn.execute(
            "SELECT value FROM meta WHERE key = 'orchestration_schema_version'"
        ).fetchone()
        assert version_row["value"] == CURRENT_ORCHESTRATION_SCHEMA_VERSION


def test_reinitialize_against_populated_db_preserves_rows(db_path: Path) -> None:
    """Phase 2.5 (writer) and Phase 2.6 (reader) will both instantiate
    ``OrchestrationStateStore`` to access this DB — a second
    instantiation against a populated file must not wipe rows. This
    pins the ``CREATE TABLE IF NOT EXISTS`` contract end-to-end."""

    store = OrchestrationStateStore(db_path)
    cos_id = _insert_cos_run(
        store,
        brief_id="brief-survival",
        principal_id="principal-x",
        status="succeeded",
        started_at="2026-05-04T17:00:00+00:00",
    )
    obs_id = _append_observation(
        store,
        principal_id="principal-x",
        brief_id="brief-survival",
        observation={"k": "v"},
        created_at="2026-05-04T17:01:00+00:00",
    )

    # Second writer enters the room. CREATE TABLE IF NOT EXISTS is the
    # only DDL on the path; no DROP, no DELETE — rows must survive.
    OrchestrationStateStore(db_path)

    assert (
        chief_of_staff_run_by_brief(db_path, brief_id="brief-survival") is not None
    )
    rows = cross_brief_observations_for_principal(
        db_path, principal_id="principal-x"
    )
    assert len(rows) == 1
    assert rows[0].id == obs_id
    # And the original CoS row is still id=cos_id (no AUTOINCREMENT
    # collision; the row never moved).
    record = chief_of_staff_run_by_brief(db_path, brief_id="brief-survival")
    assert record is not None and record.id == cos_id


def test_resolve_orchestration_paths_follow_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path resolvers re-derive from the live ``OUTPUT_ROOT`` so tests
    monkeypatching the root propagate. Mirrors the per-source resolver
    contract pinned by tests/test_output_paths.py."""

    fake_output = tmp_path / "output"
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", fake_output)

    state_dir = output_paths.resolve_orchestration_state_dir()
    assert state_dir == fake_output / "state" / "orchestration"
    assert state_dir.exists() and state_dir.is_dir()

    db = output_paths.resolve_orchestration_db_path()
    assert db == fake_output / "state" / "orchestration" / "runtime_state.sqlite3"
    assert db.parent.exists()


# ---------------------------------------------------------------------------
# Chief-of-staff round-trip
# ---------------------------------------------------------------------------


def test_dispatch_plan_round_trip(db_path: Path) -> None:
    """A dispatch plan written via the schema's column survives a read
    through the public helper byte-identical. Phase 2.5's writer will
    take this seam — the test here protects against accidental shape
    drift in the read primitive."""

    store = OrchestrationStateStore(db_path)
    dispatch_plan = {
        "rationale": "frontier-AI brief; LinkedIn first for warm-intro proximity",
        "specialists": [
            {"source": "linkedin", "reason": "warm-intro proximity"},
            {"source": "github", "reason": "code-evidence anchor"},
            {"source": "researcher", "reason": "publication trail"},
        ],
    }
    invocation_order = ["linkedin", "github", "researcher"]
    handoff_payloads = {
        "linkedin": {"top_saves": ["123", "456"], "summary": "3 strong signals"},
    }
    synthesis_output = {
        "headline": "5 candidates worth a first call",
        "confidence": 0.78,
    }

    run_id = _insert_cos_run(
        store,
        brief_id="brief-frontier-ai-nyc",
        principal_id="principal-anthropic",
        status="succeeded",
        dispatch_plan=dispatch_plan,
        invocation_order=invocation_order,
        handoff_payloads=handoff_payloads,
        synthesis_output=synthesis_output,
        started_at="2026-05-04T17:00:00+00:00",
        ended_at="2026-05-04T17:32:00+00:00",
    )

    record = chief_of_staff_run_by_brief(db_path, brief_id="brief-frontier-ai-nyc")
    assert record is not None
    assert isinstance(record, ChiefOfStaffRunRecord)
    assert record.id == run_id
    assert record.brief_id == "brief-frontier-ai-nyc"
    assert record.principal_id == "principal-anthropic"
    assert record.status == "succeeded"
    assert record.started_at == "2026-05-04T17:00:00+00:00"
    assert record.ended_at == "2026-05-04T17:32:00+00:00"

    # JSON columns surface as raw strings; the reader stays decoupled
    # from the writer's evolving payload shape (Phase 2.5 / 2.6 will
    # iterate). Round-trip equality after json.loads.
    assert json.loads(record.dispatch_plan_json) == dispatch_plan
    assert json.loads(record.invocation_order_json) == invocation_order
    assert json.loads(record.handoff_payloads_json) == handoff_payloads
    assert json.loads(record.synthesis_output_json) == synthesis_output


def test_chief_of_staff_run_by_brief_returns_latest_attempt(db_path: Path) -> None:
    """Phase 2.5's writer is expected to insert one row per CoS run
    rather than mutate-in-place, so a second attempt for the same
    brief must surface as the read helper's return — the older row
    is retained for provenance but the latest is what callers want."""

    store = OrchestrationStateStore(db_path)
    older_id = _insert_cos_run(
        store,
        brief_id="brief-iter",
        status="failed",
        started_at="2026-05-01T10:00:00+00:00",
    )
    newer_id = _insert_cos_run(
        store,
        brief_id="brief-iter",
        status="succeeded",
        started_at="2026-05-04T10:00:00+00:00",
    )
    assert newer_id > older_id

    record = chief_of_staff_run_by_brief(db_path, brief_id="brief-iter")
    assert record is not None
    assert record.id == newer_id
    assert record.status == "succeeded"


def test_chief_of_staff_run_by_brief_returns_none_for_missing_brief(
    db_path: Path,
) -> None:
    OrchestrationStateStore(db_path)
    assert chief_of_staff_run_by_brief(db_path, brief_id="brief-ghost") is None


def test_chief_of_staff_run_by_brief_returns_none_for_missing_db(
    tmp_path: Path,
) -> None:
    """``mode=ro`` URI on a missing file yields ``None`` rather than
    raising — the same total-function contract the per-source read
    primitives ship."""

    missing = tmp_path / "nope" / "runtime_state.sqlite3"
    assert chief_of_staff_run_by_brief(missing, brief_id="anything") is None


# ---------------------------------------------------------------------------
# Cross-brief observation append + aggregate
# ---------------------------------------------------------------------------


def test_cross_brief_observations_append_and_reverse_chronological(
    db_path: Path,
) -> None:
    """Append-only log: multiple observations for the same principal
    accumulate, and the read helper returns them newest-first so the
    aggregator (Phase 3.6) can lean on the order."""

    store = OrchestrationStateStore(db_path)
    _append_observation(
        store,
        principal_id="principal-northwind",
        market_key="film_finance__nyc__head",
        role_shape="head_of_finance",
        brief_id="brief-001",
        observation={"signal": "linkedin-first-was-better", "delta": 0.12},
        created_at="2026-04-01T10:00:00+00:00",
    )
    _append_observation(
        store,
        principal_id="principal-northwind",
        market_key="film_finance__nyc__head",
        role_shape="head_of_finance",
        brief_id="brief-002",
        observation={"signal": "github-first-was-better", "delta": -0.03},
        created_at="2026-04-15T10:00:00+00:00",
    )
    _append_observation(
        store,
        principal_id="principal-northwind",
        market_key="film_finance__nyc__head",
        role_shape="head_of_finance",
        brief_id="brief-003",
        observation={"signal": "linkedin-first-was-better", "delta": 0.21},
        created_at="2026-05-01T10:00:00+00:00",
    )

    rows = cross_brief_observations_for_principal(
        db_path, principal_id="principal-northwind"
    )
    assert len(rows) == 3
    assert all(isinstance(row, CrossBriefObservation) for row in rows)
    # Newest first, by created_at DESC.
    assert [row.brief_id for row in rows] == ["brief-003", "brief-002", "brief-001"]
    # Aggregate-style consumer can json.loads each row's payload and
    # pull the deltas out — what Phase 3.6 calibration will do.
    deltas = [json.loads(row.observation_json)["delta"] for row in rows]
    assert deltas == [0.21, -0.03, 0.12]


def test_cross_brief_observations_filter_by_market_and_role_shape(
    db_path: Path,
) -> None:
    """Optional filters narrow the read to the calibration grain
    (principal × market × role_shape). Same principal across two
    different markets must not bleed across the boundary."""

    store = OrchestrationStateStore(db_path)
    _append_observation(
        store,
        principal_id="principal-shared",
        market_key="film_finance__nyc__head",
        role_shape="head_of_finance",
        brief_id="brief-finance",
        observation={"k": "v_finance"},
        created_at="2026-04-01T10:00:00+00:00",
    )
    _append_observation(
        store,
        principal_id="principal-shared",
        market_key="ai_research__sf__head",
        role_shape="head_of_research",
        brief_id="brief-research",
        observation={"k": "v_research"},
        created_at="2026-04-15T10:00:00+00:00",
    )

    finance_only = cross_brief_observations_for_principal(
        db_path,
        principal_id="principal-shared",
        market_key="film_finance__nyc__head",
    )
    assert len(finance_only) == 1
    assert finance_only[0].brief_id == "brief-finance"

    research_only = cross_brief_observations_for_principal(
        db_path,
        principal_id="principal-shared",
        role_shape="head_of_research",
    )
    assert len(research_only) == 1
    assert research_only[0].brief_id == "brief-research"

    both = cross_brief_observations_for_principal(
        db_path,
        principal_id="principal-shared",
        market_key="film_finance__nyc__head",
        role_shape="head_of_finance",
    )
    assert len(both) == 1
    assert both[0].brief_id == "brief-finance"

    # No match: market + wrong role_shape.
    miss = cross_brief_observations_for_principal(
        db_path,
        principal_id="principal-shared",
        market_key="film_finance__nyc__head",
        role_shape="head_of_research",
    )
    assert miss == tuple()


def test_cross_brief_observations_empty_for_unknown_principal(db_path: Path) -> None:
    OrchestrationStateStore(db_path)
    assert (
        cross_brief_observations_for_principal(db_path, principal_id="ghost") == tuple()
    )


def test_cross_brief_observations_limit_caps_returned_rows(db_path: Path) -> None:
    store = OrchestrationStateStore(db_path)
    for i in range(5):
        _append_observation(
            store,
            principal_id="principal-many",
            brief_id=f"brief-{i:03d}",
            observation={"i": i},
            created_at=f"2026-04-{i+1:02d}T10:00:00+00:00",
        )

    capped = cross_brief_observations_for_principal(
        db_path, principal_id="principal-many", limit=2
    )
    assert len(capped) == 2
    # Newest first, so the two most recent (i=4, i=3).
    assert [json.loads(row.observation_json)["i"] for row in capped] == [4, 3]


def test_cross_brief_observations_returns_empty_for_missing_db(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "runtime_state.sqlite3"
    assert (
        cross_brief_observations_for_principal(missing, principal_id="anyone")
        == tuple()
    )


# ---------------------------------------------------------------------------
# Audit Move #1: merge_handoff_payload round-trip
# ---------------------------------------------------------------------------


def test_merge_handoff_payload_writes_first_source_into_empty_payload(
    db_path: Path,
) -> None:
    """A fresh dispatch row carries handoff_payloads_json = "{}".
    First merge writes the source key with the supplied payload."""

    store = OrchestrationStateStore(db_path)
    _insert_cos_run(store, brief_id="brief-1")

    payload = {
        "source": "linkedin",
        "candidate_count": 100,
        "save_count": 12,
        "confidence": 0.78,
        "per_source_signal_summary": (
            "LinkedIn surfaced 100 candidates and saved 12; "
            "top 5 carry the strongest reads."
        ),
        "top_saves": [
            {
                "candidate_id": "li-A",
                "role_fit_narrative": "Senior FDE shipped at Anthropic.",
                "confidence": 0.88,
            }
        ],
    }
    merged = store.merge_handoff_payload(
        brief_id="brief-1", source="linkedin", payload=payload
    )
    assert merged is True

    record = chief_of_staff_run_by_brief(db_path, brief_id="brief-1")
    assert record is not None
    persisted = json.loads(record.handoff_payloads_json)
    assert persisted == {"linkedin": payload}


def test_merge_handoff_payload_overwrites_existing_source_key(
    db_path: Path,
) -> None:
    """Second write to the same source key replaces the prior payload
    (last-write-wins for re-runs / iteration)."""

    store = OrchestrationStateStore(db_path)
    _insert_cos_run(
        store,
        brief_id="brief-iter",
        handoff_payloads={
            "github": {
                "source": "github",
                "candidate_count": 22,
                "save_count": 0,
                "confidence": 0.33,
                "per_source_signal_summary": (
                    "GitHub surfaced 22 candidates this run; "
                    "none cleared the bar."
                ),
                "top_saves": [],
            }
        },
    )

    new_payload = {
        "source": "github",
        "candidate_count": 30,
        "save_count": 4,
        "confidence": 0.66,
        "per_source_signal_summary": (
            "GitHub surfaced 30 candidates and saved 4."
        ),
        "top_saves": [
            {
                "candidate_id": "gh-1",
                "role_fit_narrative": "kubernetes maintainer.",
                "confidence": 0.9,
            }
        ],
    }
    merged = store.merge_handoff_payload(
        brief_id="brief-iter", source="github", payload=new_payload
    )
    assert merged is True

    record = chief_of_staff_run_by_brief(db_path, brief_id="brief-iter")
    assert record is not None
    persisted = json.loads(record.handoff_payloads_json)
    assert persisted == {"github": new_payload}


def test_merge_handoff_payload_keeps_other_sources_intact(
    db_path: Path,
) -> None:
    """Merging a new source key does not touch the existing source
    keys — multi-module runs accumulate one payload per module across
    sequential per-module merges."""

    store = OrchestrationStateStore(db_path)
    li_payload = {
        "source": "linkedin",
        "candidate_count": 10,
        "save_count": 2,
        "confidence": 0.6,
        "per_source_signal_summary": "LinkedIn read.",
        "top_saves": [],
    }
    _insert_cos_run(
        store,
        brief_id="brief-multi",
        handoff_payloads={"linkedin": li_payload},
    )

    gh_payload = {
        "source": "github",
        "candidate_count": 7,
        "save_count": 1,
        "confidence": 0.5,
        "per_source_signal_summary": "GitHub read.",
        "top_saves": [],
    }
    merged = store.merge_handoff_payload(
        brief_id="brief-multi", source="github", payload=gh_payload
    )
    assert merged is True

    record = chief_of_staff_run_by_brief(db_path, brief_id="brief-multi")
    assert record is not None
    persisted = json.loads(record.handoff_payloads_json)
    assert persisted == {"linkedin": li_payload, "github": gh_payload}


def test_merge_handoff_payload_targets_latest_row_when_multiple_exist(
    db_path: Path,
) -> None:
    """When the brief has multiple chief_of_staff_runs rows (re-runs
    over the brief's lifetime), merge_handoff_payload writes into the
    LATEST one — matching chief_of_staff_run_by_brief's reader
    ordering."""

    store = OrchestrationStateStore(db_path)
    older_id = _insert_cos_run(
        store,
        brief_id="brief-rerun",
        started_at="2026-04-01T10:00:00+00:00",
    )
    newer_id = _insert_cos_run(
        store,
        brief_id="brief-rerun",
        started_at="2026-05-01T10:00:00+00:00",
    )
    assert newer_id > older_id

    payload = {"source": "linkedin", "candidate_count": 1, "save_count": 0}
    merged = store.merge_handoff_payload(
        brief_id="brief-rerun", source="linkedin", payload=payload
    )
    assert merged is True

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, handoff_payloads_json FROM chief_of_staff_runs "
            "WHERE brief_id = ? ORDER BY id ASC",
            ("brief-rerun",),
        ).fetchall()

    assert len(rows) == 2
    older_payload = json.loads(rows[0]["handoff_payloads_json"])
    newer_payload = json.loads(rows[1]["handoff_payloads_json"])
    assert older_payload == {}
    assert newer_payload == {"linkedin": payload}


def test_merge_handoff_payload_returns_false_when_no_row_exists(
    db_path: Path,
) -> None:
    """Single-module briefs that didn't go through dispatch don't have
    a chief_of_staff_runs row. The merge returns False so callers can
    log + ignore without raising."""

    store = OrchestrationStateStore(db_path)
    merged = store.merge_handoff_payload(
        brief_id="brief-no-cos-row",
        source="linkedin",
        payload={"source": "linkedin"},
    )
    assert merged is False


def test_merge_handoff_payload_returns_false_for_empty_source(
    db_path: Path,
) -> None:
    """Empty source key is a defensive guard — never write a key like
    '' that would conflict with the contributing-sources contract."""

    store = OrchestrationStateStore(db_path)
    _insert_cos_run(store, brief_id="brief-1")
    merged = store.merge_handoff_payload(
        brief_id="brief-1", source="", payload={}
    )
    assert merged is False


def test_merge_handoff_payload_normalizes_source_to_lowercase(
    db_path: Path,
) -> None:
    """Source keys are case-insensitive at the protocol level
    (LinkedIn / linkedin / LINKEDIN all map to the same key). The
    merge normalizes so downstream readers don't have to."""

    store = OrchestrationStateStore(db_path)
    _insert_cos_run(store, brief_id="brief-case")
    merged = store.merge_handoff_payload(
        brief_id="brief-case",
        source="LinkedIn",
        payload={"source": "LinkedIn", "candidate_count": 1},
    )
    assert merged is True

    record = chief_of_staff_run_by_brief(db_path, brief_id="brief-case")
    assert record is not None
    persisted = json.loads(record.handoff_payloads_json)
    assert "linkedin" in persisted
    assert "LinkedIn" not in persisted


def test_merge_handoff_payload_recovers_from_malformed_existing_json(
    db_path: Path,
) -> None:
    """A pre-existing malformed handoff_payloads_json (corrupted by a
    prior bug or legacy migration) shouldn't poison the merge — the
    helper falls back to {} and writes the new payload."""

    store = OrchestrationStateStore(db_path)
    _insert_cos_run(store, brief_id="brief-corrupt")
    # Manually corrupt the handoff_payloads_json:
    with store.connect() as conn:
        conn.execute(
            "UPDATE chief_of_staff_runs SET handoff_payloads_json = ? "
            "WHERE brief_id = ?",
            ("not-json-at-all", "brief-corrupt"),
        )

    merged = store.merge_handoff_payload(
        brief_id="brief-corrupt",
        source="linkedin",
        payload={"source": "linkedin", "candidate_count": 5},
    )
    assert merged is True

    record = chief_of_staff_run_by_brief(db_path, brief_id="brief-corrupt")
    assert record is not None
    persisted = json.loads(record.handoff_payloads_json)
    assert persisted == {"linkedin": {"source": "linkedin", "candidate_count": 5}}
