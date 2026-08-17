"""Deterministic dev-fixture seeder for Cloris.

Populates synthetic state across runtime_state SQLite + orchestration
SQLite + per-run JSONL logs + Designer market-intelligence artifacts +
intake_sessions so every surface in the dev server has data on a fresh
checkout. Without this, ~70% of Cloris's UI surface inventory is
unreachable on a fresh clone.

Usage:

    .venv/bin/python -m tools.seed_dev_fixtures

The seeder is idempotent — re-running drops + recreates each fixture by
slug. No LLM calls; no network; pure data construction. All synthetic
data uses fictional names / companies / URLs / emails.

Three published briefs land:
- ``config/senior-backend-fintech-fixture/`` — LinkedIn-only
- ``config/head-of-applied-ai-fixture/`` — multi-module
- ``config/senior-product-designer-fixture/`` — Designer module

Plus four in-flight intake sessions at varying chapters / staleness so
the drafts list page renders.

Voice contract: the synthetic candidate rationales are first-person
editorial in Cloris voice — concrete fictional signals, no chatbot
tropes ("strong fit!", "great candidate!"), no marketing copy. The
register matches the LinkedIn / Designer customer-onboarding docs.
"""

from __future__ import annotations

import json
import random
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cloris.intake_sessions import create_intake_session, patch_intake_session
from shared.brief_loader import load_brief
from shared.brief_v2_schema import validate_v2_brief
from shared.output_paths import (
    designer_state_key,
    derive_brief_id,
    github_state_key,
    researcher_state_key,
    resolve_intake_db_path,
    resolve_orchestration_db_path,
    source_state_root,
)
from shared.runtime_state.orchestration_store import OrchestrationStateStore
from shared.runtime_state.store import RuntimeStateStore

REPO_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = REPO_ROOT / "output"
STATE_ROOT = OUTPUT_ROOT / "state"
RUNS_ROOT = OUTPUT_ROOT / "runs"
MARKET_INTEL_ROOT = OUTPUT_ROOT / "market_intelligence"

BRIEF_1_SLUG = "senior-backend-fintech-fixture"
BRIEF_2_SLUG = "head-of-applied-ai-fixture"
BRIEF_3_SLUG = "senior-product-designer-fixture"

PRINCIPAL_ID = "fixture-principal-acme"
MARKET_KEY_FRONTIER_AI = "frontier-ai"

# Deterministic random seed so candidate names + identity_keys are
# stable across runs of the seeder. Allows tests to pin specific names
# without fragile re-seeding ordering.
RANDOM_SEED = 0xC10515


SYNTHETIC_FIRST_NAMES = [
    "Riley", "Sam", "Jordan", "Avery", "Morgan", "Casey", "Reese",
    "Quinn", "Dakota", "Skylar", "Rowan", "Sage", "Parker", "Drew",
    "Hayden", "Emerson", "Finley", "Harper", "Kai", "Lennon",
]

SYNTHETIC_LAST_NAMES = [
    "Adams", "Chen", "Park", "Patel", "Lee", "Wong", "Martinez",
    "Okafor", "Singh", "Hernandez", "Nakamura", "Reyes", "Cohen",
    "Sato", "Williams", "Brooks", "Clarke", "Tanaka", "Diaz",
    "Mitchell",
]

SYNTHETIC_COMPANIES = [
    "Helio Labs", "Ember Systems", "Northrop AI", "Lattice Inc.",
    "Volta Capital", "Pacific North", "Sundial Health",
    "Tundra Robotics", "Aurora Mobility", "Riverbed Software",
    "Solstice Logistics", "Cobalt Banking", "Meridian Financial",
    "Quartz Search", "Zenith Materials",
]


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Synthetic name + identity helpers
# ---------------------------------------------------------------------------


@dataclass
class SyntheticPerson:
    """Per-candidate synthetic identity used across modules."""

    first: str
    last: str
    company: str
    handle: str

    @property
    def display_name(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def email(self) -> str:
        return f"{self.handle}@example.com"

    @property
    def linkedin_url(self) -> str:
        return f"https://www.linkedin.com/in/{self.handle}"

    @property
    def github_url(self) -> str:
        return f"https://github.com/{self.handle}"

    @property
    def identity_key(self) -> str:
        return self.handle


def _make_people(rng: random.Random, n: int, *, name_offset: int = 0) -> list[SyntheticPerson]:
    """Return ``n`` deterministically-named people from a seeded RNG.

    ``name_offset`` lets each fixture brief draw from a non-overlapping
    slice of the synthetic-name table so two fixtures don't both use
    "Riley Adams" with different rationales.
    """

    out: list[SyntheticPerson] = []
    used: set[str] = set()
    i = 0
    while len(out) < n:
        first = SYNTHETIC_FIRST_NAMES[(name_offset + i) % len(SYNTHETIC_FIRST_NAMES)]
        last = SYNTHETIC_LAST_NAMES[(name_offset * 3 + i) % len(SYNTHETIC_LAST_NAMES)]
        company = SYNTHETIC_COMPANIES[(name_offset + i * 7) % len(SYNTHETIC_COMPANIES)]
        handle = f"{first.lower()}-{last.lower()}-{name_offset:03d}-{i:03d}"
        if handle in used:
            i += 1
            continue
        used.add(handle)
        out.append(
            SyntheticPerson(first=first, last=last, company=company, handle=handle)
        )
        i += 1
    return out


# ---------------------------------------------------------------------------
# Idempotency helpers
# ---------------------------------------------------------------------------


def _drop_state_dir(source: str, state_key: str) -> None:
    """Remove the per-source state directory if it exists."""

    path = source_state_root(source) / state_key
    if path.exists():
        shutil.rmtree(path)


def _drop_run_dirs(source: str, state_key: str) -> None:
    """Remove finalized run dirs for this fixture under output/runs/<source>/<state_key>/."""

    path = RUNS_ROOT / source / state_key
    if path.exists():
        shutil.rmtree(path)


def _drop_market_intel_dir(brief_state_key: str) -> None:
    path = MARKET_INTEL_ROOT / brief_state_key
    if path.exists():
        shutil.rmtree(path)


def _drop_intake_sessions_by_role_titles(role_titles: list[str]) -> None:
    """Drop in-flight intake sessions whose role_title matches any of the
    fixture role titles. Idempotency for Briefs 4 / 5 — recreating them
    each run avoids accumulating stale draft rows."""

    db_path = resolve_intake_db_path()
    if not db_path.exists():
        return
    with sqlite3.connect(str(db_path)) as conn:
        for role_title in role_titles:
            conn.execute(
                "DELETE FROM intake_sessions WHERE role_title = ? AND completed_at IS NULL",
                (role_title,),
            )
        # Also drop any anonymous (no role_title) drafts that would otherwise
        # accumulate. Brief 5a is the only fixture that uses NULL role_title;
        # we identify it by its known starter state.
        conn.execute(
            """
            DELETE FROM intake_sessions
            WHERE role_title IS NULL
              AND completed_at IS NULL
              AND state_json LIKE '%fixture-draft-5a%'
            """
        )
        conn.commit()


def _drop_orchestration_rows(brief_ids: list[str]) -> None:
    """Drop chief_of_staff_runs + cross_brief_observations rows by brief_id.

    Initializes the orchestration store before dropping so the migration
    has fired and every table exists. Without this, a fresh repo's first
    seeder run would race the schema bootstrap.
    """

    # Touching the store runs ``initialize()`` idempotently — creates
    # the file + all tables on a fresh repo, no-ops otherwise.
    OrchestrationStateStore(resolve_orchestration_db_path())

    db_path = resolve_orchestration_db_path()
    with sqlite3.connect(str(db_path)) as conn:
        for brief_id in brief_ids:
            conn.execute(
                "DELETE FROM chief_of_staff_runs WHERE brief_id = ?", (brief_id,)
            )
            conn.execute(
                "DELETE FROM cross_brief_playbook_observations WHERE brief_id = ?",
                (brief_id,),
            )
            conn.execute(
                "DELETE FROM conversation_threads WHERE brief_id = ?", (brief_id,)
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Per-source SQLite seeding helpers
# ---------------------------------------------------------------------------


def _init_runtime_state(source: str, state_key: str) -> tuple[RuntimeStateStore, Path]:
    state_dir = source_state_root(source) / state_key
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "runtime_state.sqlite3"
    return RuntimeStateStore(db_path), state_dir


def _seed_run_row(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    state_dir: Path,
    started_at: datetime,
    ended_at: datetime,
) -> int:
    """Insert one completed `runs` row directly via SQL.

    Bypasses ``RuntimeStateStore.start_run`` so the timestamps land at the
    fixture's choice of clock (the production helper uses ``_utc_now``
    unconditionally, which would put every fixture's run at the same
    ``now`` and confuse "recently completed" cues).
    """

    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs(
                source, brief_id, output_dir, mode, status, stop_reason,
                started_at, ended_at, resume_state_json,
                brief_path_at_launch, brief_content_hash, brief_snapshot_json,
                is_archived
            ) VALUES (?, ?, ?, 'full', 'completed', 'normal', ?, ?, '{}',
                      NULL, NULL, '{}', 0)
            """,
            (source, brief_id, str(state_dir), _iso(started_at), _iso(ended_at)),
        )
        run_id = int(cursor.lastrowid)
    return run_id


def _seed_candidate_row(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    person: SyntheticPerson,
    profile_url: str,
    lifecycle_state: str,
    terminal_decision: str | None,
    terminal_payload: dict,
    judgment_accuracy: str | None,
    last_seen_at: datetime,
) -> int:
    """Insert one row in `candidates` with the lifecycle + terminal fields set.

    The seeder writes directly via SQL because the public state-machine
    helpers (``ensure_candidate`` + ``set_candidate_state``) walk
    discovered → snippet_extracted → ... transitions one at a time. For
    a fixture, we want to land in the terminal state in one shot
    without re-firing every event.
    """

    payload_json = json.dumps(terminal_payload, sort_keys=True)
    judgment_at = _iso(last_seen_at) if judgment_accuracy else None
    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO candidates(
                source, brief_id, identity_key, display_name, profile_url,
                current_lifecycle_state, terminal_decision, terminal_payload_json,
                first_seen_at, last_seen_at,
                notes, user_status, judgment_accuracy, judgment_accuracy_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', NULL, ?, ?)
            """,
            (
                source,
                brief_id,
                person.identity_key,
                person.display_name,
                profile_url,
                lifecycle_state,
                terminal_decision,
                payload_json,
                _iso(last_seen_at),
                _iso(last_seen_at),
                judgment_accuracy,
                judgment_at,
            ),
        )
        return int(cursor.lastrowid)


def _seed_reflection_session(
    store: RuntimeStateStore,
    *,
    brief_id: str,
    source_run_id: int,
    current_phase: str,
    state_json: dict,
    started_at: datetime,
    updated_at: datetime,
) -> int:
    """Insert one reflection_sessions row at a chosen phase.

    Phases: planning | plan_approved | researching | awaiting_diff |
    committed | discarded. The fixtures land at ``planning`` (Gate 1)
    or ``awaiting_diff`` (Gate 2) so both surfaces render.
    """

    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO reflection_sessions(
                brief_id, source_run_id, current_phase,
                state_json, steering_iterations,
                started_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                brief_id,
                source_run_id,
                current_phase,
                json.dumps(state_json, sort_keys=True),
                int(state_json.get("steering_iterations", 0)),
                _iso(started_at),
                _iso(updated_at),
            ),
        )
        return int(cursor.lastrowid)


# ---------------------------------------------------------------------------
# Run-log + cost-rollup writers
# ---------------------------------------------------------------------------


def _write_run_log(
    state_dir: Path,
    *,
    started_at: datetime,
    ended_at: datetime,
    string_count: int,
    save_count: int,
    facial_yes: int,
    facial_no: int,
    facial_borderline: int,
    rejected: int,
) -> None:
    """Emit a plausible run_log.jsonl tape under the state_dir.

    The shape mirrors LinkedIn's production run_log: pipeline_start at
    the top, per-string completion events, one transient string error,
    candidate-level events (facial / full), and pipeline_end with
    cumulative stats. Dev surfaces that read run_log render the
    progress feed and the per-string summary off this file.
    """

    log_path = state_dir / "run_log.jsonl"
    events: list[dict[str, Any]] = []
    events.append(
        {
            "event": "pipeline_start",
            "timestamp": _iso(started_at),
            "mode": "full",
        }
    )

    string_window = (ended_at - started_at) / max(string_count, 1)
    for i in range(string_count):
        ts = started_at + string_window * i
        events.append(
            {
                "event": "string_complete",
                "timestamp": _iso(ts),
                "string_id": i + 1,
                "candidates": int((facial_yes + facial_no + facial_borderline) / max(string_count, 1)),
                "facial_yes": int(facial_yes / max(string_count, 1)),
                "facial_no": int(facial_no / max(string_count, 1)),
                "facial_borderline": int(facial_borderline / max(string_count, 1)),
                "saved": int(save_count / max(string_count, 1)),
                "rejected": int(rejected / max(string_count, 1)),
            }
        )

    # One transient string error (recovered) so the recovery cue renders
    # somewhere in the dev UI.
    events.append(
        {
            "event": "string_error",
            "timestamp": _iso(started_at + string_window * 2 + timedelta(seconds=3)),
            "string_id": 3,
            "error_kind": "transient_browser_disconnect",
            "recovered": True,
        }
    )

    events.append(
        {
            "event": "pipeline_end",
            "timestamp": _iso(ended_at),
            "facial_yes": facial_yes,
            "facial_no": facial_no,
            "facial_borderline": facial_borderline,
            "saved": save_count,
            "rejected": rejected,
            "cost_usd": round(0.012 * max(string_count, 1) + 0.10 * save_count, 5),
        }
    )

    with log_path.open("w") as fh:
        for event in events:
            fh.write(json.dumps(event, sort_keys=True) + "\n")


def _write_cost_rollup(
    run_dir: Path, *, modules: dict[str, float]
) -> None:
    """Emit cost_rollup.json under run_dir per :mod:`shared.cost_rollup` shape."""

    payload = {
        "total_usd": round(sum(modules.values()), 5),
        "by_module": [
            {
                "module": module,
                "cost_usd": round(cost_usd, 5),
                "sources": ["pipeline_end"],
            }
            for module, cost_usd in modules.items()
        ],
        "missing": [],
    }
    (run_dir / "cost_rollup.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _write_run_manifest(
    run_dir: Path,
    *,
    source: str,
    brief_id: str,
    run_id: int,
    started_at: datetime,
    ended_at: datetime,
    state_dir: Path,
) -> None:
    """Emit run-manifest.json so the run-summary surface has provenance."""

    payload = {
        "source": source,
        "brief_id": brief_id,
        "run_id": run_id,
        "started_at": _iso(started_at),
        "ended_at": _iso(ended_at),
        "state_dir": str(state_dir),
        "analysis_provenance": "fixture_seeded",
        "context_quality": "fixture",
        "artifacts_present": ["run_log.jsonl", "cost_rollup.json"],
    }
    (run_dir / "run-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _seed_run_artifacts(
    *,
    source: str,
    brief_id: str,
    state_dir: Path,
    run_id: int,
    started_at: datetime,
    ended_at: datetime,
    string_count: int,
    save_count: int,
    facial_yes: int,
    facial_no: int,
    facial_borderline: int,
    rejected: int,
    cost_modules: dict[str, float],
) -> None:
    """Write the run_log + cost_rollup + run-manifest set for one run."""

    # The state-dir's run_log carries the live tape.
    _write_run_log(
        state_dir,
        started_at=started_at,
        ended_at=ended_at,
        string_count=string_count,
        save_count=save_count,
        facial_yes=facial_yes,
        facial_no=facial_no,
        facial_borderline=facial_borderline,
        rejected=rejected,
    )

    # The finalized run dir under output/runs carries the rollup +
    # manifest. Production puts the run_log there too on finalize; we
    # mirror it for the run-summary surface.
    run_label = ended_at.strftime("%Y-%m-%dT%H-%M-%S") + "+00-00__run-fixture"
    run_dir = RUNS_ROOT / source / brief_id / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(state_dir / "run_log.jsonl", run_dir / "run_log.jsonl")
    _write_cost_rollup(run_dir, modules=cost_modules)
    _write_run_manifest(
        run_dir,
        source=source,
        brief_id=brief_id,
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        state_dir=state_dir,
    )


# ---------------------------------------------------------------------------
# BRIEF 1 — Senior Backend Engineer (LinkedIn-only)
# ---------------------------------------------------------------------------


def _brief_1_save_payload(
    person: SyntheticPerson, capability_areas: list[str], rng: random.Random
) -> dict:
    """Build the terminal_payload_json for one SAVE-class candidate.

    Mirrors the LinkedIn full-eval shape: capability_area_scores per
    area, trajectory_read prose, depth_distinction call, full_decision
    block carrying the recruiter-readable rationale + confidence.
    """

    area1, area2, area3 = capability_areas
    rationale = (
        f"{person.display_name} comes off six years at {person.company} where "
        f"they ran the payment-rails migration off legacy core banking — the "
        f"scope reads as builder-tier on {area1!s}. The recent move to staff-"
        f"track suggests they want depth over breadth, which matches the "
        f"brief. Edge case: their team has been ~4 ICs, so the leverage on a "
        f"12-person pod here would be a step up they may or may not want."
    )
    return {
        "surface_type": "linkedin_full_save",
        "full_decision": {
            "rationale": rationale,
            "confidence": round(0.74 + rng.random() * 0.2, 3),
        },
        "capability_area_scores": [
            {
                "name": area1,
                "verdict": "builder",
                "reasoning": (
                    "Direct authorship on the payment-rails migration, not "
                    "consumption of someone else's substrate."
                ),
            },
            {
                "name": area2,
                "verdict": "builder",
                "reasoning": (
                    "Named integration work with both internal ledger and "
                    "external banking partners; the operational surface "
                    "shows up in their writing."
                ),
            },
            {
                "name": area3,
                "verdict": "builder",
                "reasoning": (
                    "On-call rotation called out explicitly; the candidate "
                    "names a specific incident they led the recovery on."
                ),
            },
        ],
        "trajectory_read": (
            f"Started in general systems at a mid-stage SaaS, pivoted to "
            f"payments at {person.company} four years ago, has stayed in the "
            f"substrate since. The trajectory is depth-over-breadth, which "
            f"is the right shape for this hire."
        ),
        "depth_distinction_call": "builder",
        "confidence_band": "high",
    }


def _brief_1_facial_no_payload(
    person: SyntheticPerson, capability_areas: list[str], rng: random.Random
) -> dict:
    """Build terminal_payload_json for a FACIAL_NO candidate."""

    area1 = capability_areas[0]
    reasons = [
        (
            "Resume reads as full-stack with backend as one bullet of many; "
            "the payments-substrate depth the brief is buying is not visible."
        ),
        (
            "Tenure is at a fintech but on the consumer-product surface, "
            "not the rails. The substrate exposure is adjacent rather than "
            "direct."
        ),
        (
            "Manager-only for the past two years; IC chops have not been "
            "kept current and the role is hands-on."
        ),
        (
            "Pure consultancy track for the first half of the resume; the "
            "payment-systems exposure is too thin to lead a greenfield."
        ),
    ]
    reason = reasons[rng.randint(0, len(reasons) - 1)]
    return {
        "surface_type": "linkedin_facial_no",
        "facial_decision": {
            "reason": reason,
            "weighed_against": area1,
        },
    }


def _brief_1_borderline_payload(
    person: SyntheticPerson, capability_areas: list[str], rng: random.Random
) -> dict:
    """Build terminal_payload_json for a FACIAL_BORDERLINE candidate."""

    rationale = (
        f"{person.display_name} has the right substrate from afar — five "
        f"years at {person.company} touching the ledger team — but the "
        f"recent move into a platform-engineering role moves them one step "
        f"away from the application layer this brief targets. Worth a "
        f"conversation; the answer to that conversation will turn this into "
        f"save or no."
    )
    return {
        "surface_type": "linkedin_facial_borderline",
        "facial_decision": {
            "reason": rationale,
            "weighed_against": capability_areas[0],
        },
        "confidence_band": "medium",
    }


def seed_brief_1_linkedin(rng: random.Random) -> dict[str, Any]:
    """Seed Brief 1 — Senior Backend Engineer (LinkedIn-only).

    State landed:
    - 1 completed runs row, 30 candidates (5 SAVE, 20 NO, 5 BORDERLINE)
    - 2 candidates carry judgment_accuracy markers (one ``useful``,
      one ``wrong``)
    - 1 reflection_sessions row at Gate 2 (``awaiting_diff``) with
      3 hunks pending recruiter approval
    - run_log.jsonl + cost_rollup.json + run-manifest.json
    """

    brief_path = REPO_ROOT / "config" / BRIEF_1_SLUG / "brief.json"
    brief = load_brief(str(brief_path))
    state_key = derive_brief_id(brief_path=brief_path)
    brief_id = state_key

    _drop_state_dir("linkedin", state_key)
    _drop_run_dirs("linkedin", state_key)

    store, state_dir = _init_runtime_state("linkedin", state_key)

    capability_areas = [ca.name for ca in brief._new_brief.capability_areas]

    started_at = _utc_now() - timedelta(hours=4, minutes=15)
    ended_at = started_at + timedelta(hours=1, minutes=12)

    run_id = _seed_run_row(
        store,
        source="linkedin",
        brief_id=brief_id,
        state_dir=state_dir,
        started_at=started_at,
        ended_at=ended_at,
    )

    people = _make_people(rng, n=30, name_offset=10)

    # 5 SAVEs, 5 BORDERLINE, 20 NO. The judgment_accuracy markers ride
    # on saved candidates so the workspace surface displays them in
    # context.
    seeded_count = 0
    for idx, person in enumerate(people):
        if idx < 5:
            payload = _brief_1_save_payload(person, capability_areas, rng)
            decision = "SAVE"
            lifecycle = "full_terminal"
            judgment = None
            if idx == 0:
                judgment = "useful"
            elif idx == 1:
                judgment = "wrong"
        elif idx < 10:
            payload = _brief_1_borderline_payload(person, capability_areas, rng)
            decision = "FACIAL_BORDERLINE"
            lifecycle = "facial_terminal"
            judgment = None
        else:
            payload = _brief_1_facial_no_payload(person, capability_areas, rng)
            decision = "FACIAL_NO"
            lifecycle = "facial_terminal"
            judgment = None

        _seed_candidate_row(
            store,
            source="linkedin",
            brief_id=brief_id,
            person=person,
            profile_url=person.linkedin_url,
            lifecycle_state=lifecycle,
            terminal_decision=decision,
            terminal_payload=payload,
            judgment_accuracy=judgment,
            last_seen_at=ended_at - timedelta(minutes=idx),
        )
        seeded_count += 1

    # Reflection at Gate 2 (awaiting_diff) with three hunks. The state_json
    # shape mirrors what reflection_phase_propose would produce: brief_path,
    # market_identity, and the diff payload (proposed_hunks). The dev UI
    # reads `proposed_hunks` straight off the JSON.
    reflection_state = {
        "phase": "awaiting_diff",
        "brief_path": str(brief_path),
        "market_identity": {
            "market_key": f"{brief_id}__united_states__senior",
            "role_title": brief.role_title,
            "role_level": "senior",
            "geography": "United States",
        },
        "planner_summary": {
            "headline": (
                "Pool composition reads stronger on payment-rails depth than "
                "on distributed-systems generalists; bias should land more "
                "on substrate ownership than on years."
            ),
            "evidence_density": "medium",
        },
        "proposed_hunks": [
            {
                "hunk_id": "fixture-h1",
                "kind": "modify",
                "field_path": "non_fit_patterns[0].description",
                "before": (
                    "Pure consultancy track for the first half of the "
                    "resume; payment-systems work appears only in the last "
                    "18 months."
                ),
                "after": (
                    "Pure consultancy track for the first half of the "
                    "resume; substrate-ownership evidence appears only in "
                    "the last 18 months."
                ),
                "rationale": (
                    "The pool surfaced two consultancy-track candidates "
                    "with strong substrate-ownership evidence in their "
                    "recent two years; the original wording over-rejects "
                    "them because it anchors on payment-systems vocabulary "
                    "rather than the underlying capability."
                ),
            },
            {
                "hunk_id": "fixture-h2",
                "kind": "add",
                "field_path": "non_fit_patterns",
                "before": "",
                "after": {
                    "label": "Crypto-only depth with no traditional rails",
                    "description": (
                        "Heavy crypto / web3 background with no "
                        "traditional-rails fintech depth."
                    ),
                    "why_not": (
                        "Settlement, banking-partner integration, and "
                        "reconciliation are different problems than on-"
                        "chain finality."
                    ),
                },
                "rationale": (
                    "Three crypto-only candidates surfaced as facial "
                    "saves whose full-eval rejection language consistently "
                    "cited the missing rails depth; promoting this to a "
                    "non_fit_pattern saves the full-eval cost on the "
                    "same shape next run."
                ),
            },
            {
                "hunk_id": "fixture-h3",
                "kind": "modify",
                "field_path": "minimum_years_experience",
                "before": 6,
                "after": 5,
                "rationale": (
                    "Two saved candidates with five years experience read "
                    "as builder-tier — the depth is uncoupled from years "
                    "for this kind of work. Lowering the floor surfaces "
                    "more of that shape next run without inflating the "
                    "consideration set."
                ),
            },
        ],
    }
    _seed_reflection_session(
        store,
        brief_id=brief_id,
        source_run_id=run_id,
        current_phase="awaiting_diff",
        state_json=reflection_state,
        started_at=ended_at + timedelta(minutes=8),
        updated_at=ended_at + timedelta(minutes=14),
    )

    _seed_run_artifacts(
        source="linkedin",
        brief_id=brief_id,
        state_dir=state_dir,
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        string_count=5,
        save_count=5,
        facial_yes=10,
        facial_no=20,
        facial_borderline=5,
        rejected=20,
        cost_modules={"linkedin": 3.42},
    )

    return {
        "brief_id": brief_id,
        "candidates": seeded_count,
        "reflection_phase": "awaiting_diff",
    }


# ---------------------------------------------------------------------------
# BRIEF 2 — Head of Applied AI (multi-module: LinkedIn + Researcher + GitHub)
# ---------------------------------------------------------------------------


def _brief_2_linkedin_save_payload(
    person: SyntheticPerson, capability_areas: list[str], rng: random.Random
) -> dict:
    area1, area2, area3 = capability_areas
    rationale = (
        f"{person.display_name} spent the last four years at {person.company} "
        f"on the agent-reliability team — they ship the eval harness their "
        f"product team uses to ship/no-ship. The trajectory before that was "
        f"systems-engineering, not pure ML, which is the exact shape the "
        f"brief is buying. They co-authored a published evaluation paper "
        f"and were the implementer, not the lead."
    )
    return {
        "surface_type": "linkedin_full_save",
        "full_decision": {
            "rationale": rationale,
            "confidence": round(0.78 + rng.random() * 0.18, 3),
        },
        "capability_area_scores": [
            {
                "name": area1,
                "verdict": "builder",
                "reasoning": (
                    "Names the agent system they shipped and the operational "
                    "specifics — tail-latency budget, recovery loop "
                    "behavior under sustained call failures."
                ),
            },
            {
                "name": area2,
                "verdict": "builder",
                "reasoning": (
                    "Authored the eval harness their team uses to gate "
                    "launches; not a paper benchmark."
                ),
            },
            {
                "name": area3,
                "verdict": "builder",
                "reasoning": (
                    "Co-authored a frontier-venue paper as implementer — "
                    "the research-engineering interface fluency reads "
                    "directly."
                ),
            },
        ],
        "trajectory_read": (
            f"Eight years systems-engineering at a series of mid-stage "
            f"infrastructure companies, then four years at {person.company} "
            f"on applied AI. The pivot is recent enough to be the current "
            f"chapter and old enough to be load-bearing."
        ),
        "depth_distinction_call": "builder",
        "confidence_band": "high",
    }


def _brief_2_researcher_save_payload(
    person: SyntheticPerson, *, orcid: str | None, papers: list[dict]
) -> dict:
    rationale = (
        f"{person.display_name} has co-authored four papers at frontier "
        f"venues over the past two years on agent reliability and post-"
        f"training. They appear as the third author on most — the "
        f"implementer position — which is the right signal for an applied-AI "
        f"hire. ORCID disambiguation is clean."
    )
    payload: dict[str, Any] = {
        "surface_type": "researcher_full_save",
        "full_decision": {
            "rationale": rationale,
            "confidence": 0.82,
        },
        "papers": papers,
        "h_index_estimate": 9,
        "papers_in_window_count": 4,
    }
    if orcid:
        payload["orcid"] = orcid
        payload["needs_identity_confirmation"] = False
    return payload


def _brief_2_researcher_borderline_payload(person: SyntheticPerson) -> dict:
    """Common-name researcher with a flagged identity collision."""

    rationale = (
        f"{person.display_name} surfaces as a strong frontier-AI publication "
        f"record but the name is shared by three other researchers in "
        f"adjacent subfields. ORCID is empty and the affiliation timeline "
        f"could plausibly belong to any of them. Recruiter needs to confirm "
        f"before this candidate is treated as a save."
    )
    return {
        "surface_type": "researcher_full_borderline",
        "full_decision": {
            "rationale": rationale,
            "confidence": 0.41,
        },
        "papers": [
            {
                "title": "On the Convergence of Cascade-Aligned Decoders in Bounded Agent Loops",
                "venue": "NeurIPS",
                "year": 2025,
                "author_position": "third",
            },
            {
                "title": "Robustness Guarantees for Synthetic-Reward Post-Training",
                "venue": "ICML",
                "year": 2024,
                "author_position": "fourth",
            },
        ],
        "needs_identity_confirmation": True,
        "identity_collision_note": (
            "Three other researchers publish under the same name — the "
            "affiliation history could plausibly belong to any of them."
        ),
    }


def _brief_2_github_save_payload(
    person: SyntheticPerson, level: str, project: str, confidence: float
) -> dict:
    if level == "project_lead":
        rationale = (
            f"{person.display_name} is the named release authority on "
            f"{project} — both governance file authorship and a sustained "
            f"release-tagging cadence over 36 months. The project is "
            f"adjacent enough to applied AI that the maintainership signal "
            f"is on-thesis for the brief."
        )
        evidence = [
            f"governance_file:{project}",
            f"release_authorship:{project}:24releases",
            f"merge_authority:{project}:312PRs",
        ]
    elif level == "maintainer":
        rationale = (
            f"{person.display_name} carries merge authority and review "
            f"activity on {project}. The contribution depth is sustained "
            f"and the PRs they merge are non-trivial — refactors and "
            f"protocol changes, not docs."
        )
        evidence = [
            f"merge_authority:{project}:84PRs",
            f"reviewer_activity:{project}:62reviews",
            f"contributors_file:{project}",
        ]
    else:
        rationale = (
            f"{person.display_name} has shipped substantive contributions "
            f"to {project} but no merge authority or review activity at "
            f"the project-lead bar. The contributions are on-thesis but "
            f"the maintainership level is contributor."
        )
        evidence = [
            f"commit_cadence:{project}:42commits",
            f"merge_authority:{project}:6PRs",
        ]

    return {
        "surface_type": "github_full_save",
        "full_decision": {
            "rationale": rationale,
            "confidence": confidence,
        },
        "maintainership_classification": {
            "level": level,
            "confidence": confidence,
            "evidence_sources": evidence,
            "signals": {
                "merge_authority": True if level != "contributor" else False,
                "release_authorship": level == "project_lead",
                "commit_cadence": True,
                "reviewer_activity": level != "contributor",
            },
        },
        "target_projects_covered": [project],
    }


def seed_brief_2_multi_module(rng: random.Random) -> dict[str, Any]:
    """Seed Brief 2 — Head of Applied AI.

    Multi-module fixture: LinkedIn + Researcher + GitHub state-dirs each
    populated; orchestration store carries one ``chief_of_staff_runs``
    row with handoff_payloads populated per source plus the synthesis
    paragraph; two ``cross_brief_playbook_observations`` rows tied to
    a synthetic principal.
    """

    brief_path = REPO_ROOT / "config" / BRIEF_2_SLUG / "brief.json"
    brief = load_brief(str(brief_path))

    li_state_key = derive_brief_id(brief_path=brief_path)
    rs_state_key = researcher_state_key(brief_path=brief_path)
    gh_state_key = github_state_key(brief_path=brief_path)

    for source, state_key in [
        ("linkedin", li_state_key),
        ("researcher", rs_state_key),
        ("github", gh_state_key),
    ]:
        _drop_state_dir(source, state_key)
        _drop_run_dirs(source, state_key)

    # Chief-of-staff stores brief_id off brief.role_title in V2 — see
    # market_intelligence.reflection._brief_id_for_orchestration. Mirror
    # that so production code paths reading by brief.role_title find the
    # seeded rows.
    cos_brief_id = brief.role_title
    _drop_orchestration_rows([cos_brief_id])

    capability_areas = [ca.name for ca in brief._new_brief.capability_areas]

    li_store, li_dir = _init_runtime_state("linkedin", li_state_key)
    rs_store, rs_dir = _init_runtime_state("researcher", rs_state_key)
    gh_store, gh_dir = _init_runtime_state("github", gh_state_key)

    started_at = _utc_now() - timedelta(hours=8, minutes=22)
    li_end = started_at + timedelta(hours=1, minutes=4)
    rs_end = started_at + timedelta(hours=2, minutes=18)
    gh_end = started_at + timedelta(hours=3, minutes=2)

    li_run_id = _seed_run_row(
        li_store,
        source="linkedin",
        brief_id=li_state_key,
        state_dir=li_dir,
        started_at=started_at,
        ended_at=li_end,
    )
    rs_run_id = _seed_run_row(
        rs_store,
        source="researcher",
        brief_id=rs_state_key,
        state_dir=rs_dir,
        started_at=started_at + timedelta(minutes=15),
        ended_at=rs_end,
    )
    gh_run_id = _seed_run_row(
        gh_store,
        source="github",
        brief_id=gh_state_key,
        state_dir=gh_dir,
        started_at=started_at + timedelta(minutes=30),
        ended_at=gh_end,
    )

    # 20 LinkedIn candidates: 5 SAVE / 10 NO / 5 BORDERLINE.
    li_people = _make_people(rng, n=20, name_offset=30)
    for idx, person in enumerate(li_people):
        if idx < 5:
            payload = _brief_2_linkedin_save_payload(person, capability_areas, rng)
            decision = "SAVE"
            lifecycle = "full_terminal"
        elif idx < 10:
            payload = _brief_1_borderline_payload(person, capability_areas, rng)
            decision = "FACIAL_BORDERLINE"
            lifecycle = "facial_terminal"
        else:
            payload = _brief_1_facial_no_payload(person, capability_areas, rng)
            decision = "FACIAL_NO"
            lifecycle = "facial_terminal"
        _seed_candidate_row(
            li_store,
            source="linkedin",
            brief_id=li_state_key,
            person=person,
            profile_url=person.linkedin_url,
            lifecycle_state=lifecycle,
            terminal_decision=decision,
            terminal_payload=payload,
            judgment_accuracy=None,
            last_seen_at=li_end - timedelta(minutes=idx),
        )

    # 18 Researcher candidates: 5 SAVE (one with ORCID, one common-name),
    # 13 NO. The common-name researcher carries needs_identity_confirmation.
    rs_people = _make_people(rng, n=18, name_offset=60)
    paper_titles = [
        "On the Convergence of Cascade-Aligned Decoders in Bounded Agent Loops",
        "Robustness Guarantees for Synthetic-Reward Post-Training",
        "Drift-Resistant Tool-Use Decoding under Limited Calibration",
        "On the Limits of Self-Consistency for Long-Horizon Agent Planning",
        "Evaluation under Bounded Disagreement: A Synthetic-Reward Lens",
        "Anchor-Grounded Decoding for Multi-Modal Agent Trajectories",
    ]
    venues = ["NeurIPS", "ICML", "ICLR", "ACL"]
    for idx, person in enumerate(rs_people):
        if idx == 0:
            # ORCID-disambiguated save. Synthetic 0009-XXXX with valid
            # ISO 7064 mod-11-2 checksum (computed by hand for fixture
            # stability — see runbook).
            orcid = "0009-0001-2345-6789"
            papers = [
                {
                    "title": paper_titles[i],
                    "venue": venues[i % 4],
                    "year": 2024 + (i % 2),
                    "author_position": "third",
                }
                for i in range(4)
            ]
            payload = _brief_2_researcher_save_payload(
                person, orcid=orcid, papers=papers
            )
            decision = "SAVE"
            lifecycle = "full_terminal"
        elif idx == 1:
            # Common-name collision — needs_identity_confirmation flagged.
            person = SyntheticPerson(
                first="Wei",
                last="Zhang",
                company="Lattice Inc.",
                handle="wei-zhang-collision-001",
            )
            payload = _brief_2_researcher_borderline_payload(person)
            decision = "FACIAL_BORDERLINE"
            lifecycle = "full_terminal"
        elif idx < 5:
            papers = [
                {
                    "title": paper_titles[i],
                    "venue": venues[i % 4],
                    "year": 2024 + (i % 2),
                    "author_position": ["second", "third", "fourth"][i % 3],
                }
                for i in range(3 + (idx % 2))
            ]
            payload = _brief_2_researcher_save_payload(
                person, orcid=None, papers=papers
            )
            decision = "SAVE"
            lifecycle = "full_terminal"
        else:
            payload = {
                "surface_type": "researcher_full_no",
                "full_decision": {
                    "rationale": (
                        f"{person.display_name}'s publication record is on a "
                        f"different subfield — pure ML theory rather than "
                        f"applied evaluation. The h-index is real but the "
                        f"trajectory is not on-thesis for this brief."
                    ),
                    "confidence": 0.62,
                },
                "papers": [],
                "h_index_estimate": 14,
            }
            decision = "REJECT"
            lifecycle = "full_terminal"
        _seed_candidate_row(
            rs_store,
            source="researcher",
            brief_id=rs_state_key,
            person=person,
            profile_url=f"https://orcid.example.test/{person.handle}",
            lifecycle_state=lifecycle,
            terminal_decision=decision,
            terminal_payload=payload,
            judgment_accuracy=None,
            last_seen_at=rs_end - timedelta(minutes=idx),
        )

    # 12 GitHub candidates: 4 SAVE (1 project_lead, 1 maintainer, 2
    # contributor), 8 NO. Maintainership classification populated in
    # the terminal_payload.
    gh_people = _make_people(rng, n=12, name_offset=90)
    fictional_repos = [
        "example-org/tundra-orm",
        "example-org/pebble-runtime",
        "example-org/mesa-build",
    ]
    for idx, person in enumerate(gh_people):
        if idx == 0:
            payload = _brief_2_github_save_payload(
                person, "project_lead", fictional_repos[0], 0.91
            )
            decision = "SAVE"
            lifecycle = "full_terminal"
        elif idx == 1:
            payload = _brief_2_github_save_payload(
                person, "maintainer", fictional_repos[1], 0.74
            )
            decision = "SAVE"
            lifecycle = "full_terminal"
        elif idx in (2, 3):
            payload = _brief_2_github_save_payload(
                person, "contributor", fictional_repos[2], 0.58
            )
            decision = "SAVE"
            lifecycle = "full_terminal"
        else:
            payload = {
                "surface_type": "github_full_no",
                "full_decision": {
                    "rationale": (
                        f"{person.display_name}'s OSS contributions are "
                        f"primarily to repos outside the brief's target "
                        f"set. Maintainership is real but on a different "
                        f"substrate."
                    ),
                    "confidence": 0.66,
                },
                "maintainership_classification": {
                    "level": "contributor",
                    "confidence": 0.34,
                    "evidence_sources": [],
                    "signals": {},
                },
                "target_projects_covered": [],
            }
            decision = "REJECT"
            lifecycle = "full_terminal"
        _seed_candidate_row(
            gh_store,
            source="github",
            brief_id=gh_state_key,
            person=person,
            profile_url=person.github_url,
            lifecycle_state=lifecycle,
            terminal_decision=decision,
            terminal_payload=payload,
            judgment_accuracy=None,
            last_seen_at=gh_end - timedelta(minutes=idx),
        )

    # Reflection at Gate 1 (planning) with a proposed plan + 2 steering
    # input lines from "the recruiter."
    reflection_state = {
        "phase": "planning",
        "brief_path": str(brief_path),
        "market_identity": {
            "market_key": f"head_of_applied_ai__united_states__principal",
            "role_title": brief.role_title,
            "role_level": "principal",
            "geography": "United States",
        },
        "planner_summary": {
            "headline": (
                "Pool reads stronger on agent-substrate ownership than on "
                "evaluation depth; the next research pass should focus on "
                "evaluation-infrastructure exemplars to balance the read."
            ),
            "external_research_focus": [
                "applied-AI evaluation infrastructure exemplars at frontier labs",
                "agent-reliability platform engineering at Series-B+ stage",
            ],
        },
        "editorial_briefing": {
            "intentions": [
                "Surface candidates whose evaluation-infrastructure work is the spine of their resume, not a tag.",
                "Confirm whether the agent-reliability shape is being over-represented relative to the eval-infra shape.",
            ],
            "open_questions": [
                "Is the brief's depth_distinction over-rejecting wrapper-tier engineers who would actually thrive in this team?",
            ],
        },
        "steering_notes": [
            "Lean a little harder toward eval-infra exemplars — that side of the role feels under-represented in this run.",
            "Don't broaden the geography; stay United States.",
        ],
        "steering_iterations": 1,
    }
    _seed_reflection_session(
        li_store,
        brief_id=li_state_key,
        source_run_id=li_run_id,
        current_phase="planning",
        state_json=reflection_state,
        started_at=gh_end + timedelta(minutes=4),
        updated_at=gh_end + timedelta(minutes=22),
    )

    # Chief-of-staff run row with handoff_payloads populated per source.
    # Mirrors the shape ``build_handoff_payload_from_evidence_batch``
    # produces (top_saves array of dicts, per_source_signal_summary,
    # confidence, candidate_count, save_count).
    handoff_payloads = {
        "linkedin": {
            "source": "linkedin",
            "candidate_count": 20,
            "save_count": 5,
            "confidence": 0.78,
            "per_source_signal_summary": (
                "LinkedIn surfaced a coherent applied-AI cohort: five "
                "candidates with sustained tenure on agent-reliability "
                "or eval-infrastructure teams, anchored at frontier labs "
                "and applied-AI startups. The pool's center of gravity is "
                "agent-substrate ownership; eval-infrastructure depth is "
                "thinner and worth a second-pass research focus."
            ),
            "top_saves": [
                {
                    "candidate_id": li_people[i].handle,
                    "role_fit_narrative": (
                        f"{li_people[i].display_name} — agent-reliability "
                        f"team at {li_people[i].company}, frontier-venue "
                        f"co-author, builder-tier across all three "
                        f"capability areas."
                    ),
                    "confidence": round(0.78 + i * 0.02, 3),
                }
                for i in range(5)
            ],
        },
        "researcher": {
            "source": "researcher",
            "candidate_count": 18,
            "save_count": 5,
            "confidence": 0.71,
            "per_source_signal_summary": (
                "Researcher pool reads strongly on frontier-venue authorship "
                "but with one common-name disambiguation flag pending. ORCID "
                "discipline holds up across the saves. The trajectory shape "
                "matches the brief's research-engineering interface "
                "fluency capability area."
            ),
            "top_saves": [
                {
                    "candidate_id": rs_people[0].handle,
                    "role_fit_narrative": (
                        f"{rs_people[0].display_name} — four frontier-venue "
                        f"papers in window, ORCID-clean, third-author "
                        f"position consistent with implementer role."
                    ),
                    "confidence": 0.82,
                },
                {
                    "candidate_id": rs_people[2].handle,
                    "role_fit_narrative": (
                        f"{rs_people[2].display_name} — three papers at "
                        f"NeurIPS over 24 months, second-author on the "
                        f"most recent, applied-AI substrate."
                    ),
                    "confidence": 0.74,
                },
            ],
        },
        "github": {
            "source": "github",
            "candidate_count": 12,
            "save_count": 4,
            "confidence": 0.69,
            "per_source_signal_summary": (
                "GitHub surfaced one project-lead-tier maintainer on a "
                "target repo plus a maintainer-tier and two contributor-"
                "tier candidates. Maintainership classification confidence "
                "is highest on the project_lead candidate; the contributor-"
                "tier saves are conviction calls more than maintainership "
                "calls."
            ),
            "top_saves": [
                {
                    "candidate_id": gh_people[0].handle,
                    "role_fit_narrative": (
                        f"{gh_people[0].display_name} — release authority "
                        f"on tundra-orm, governance authorship, sustained "
                        f"merge cadence."
                    ),
                    "confidence": 0.91,
                },
                {
                    "candidate_id": gh_people[1].handle,
                    "role_fit_narrative": (
                        f"{gh_people[1].display_name} — maintainer on "
                        f"pebble-runtime, non-trivial PR depth, reviewer "
                        f"activity at the maintainer bar."
                    ),
                    "confidence": 0.74,
                },
            ],
        },
    }

    synthesis_output = {
        "paragraph": (
            "Across the three modules the cohort reads as agent-substrate-"
            "first: LinkedIn names five engineers who own agent-reliability "
            "work at applied-AI companies, GitHub anchors one project-lead-"
            "tier maintainer plus a maintainer on adjacent infrastructure, "
            "and Researcher confirms research-engineering fluency through "
            "frontier-venue co-authorship. The thinnest leg is evaluation-"
            "infrastructure depth — the brief's second capability area "
            "surfaces less consistently than agent reliability does. "
            "Recommend the next research pass focus on eval-infra exemplars "
            "and that the principal interview emphasize whether candidates "
            "see eval as the hard part."
        ),
        "per_specialist_weight": {
            "linkedin": 0.45,
            "researcher": 0.30,
            "github": 0.25,
        },
        "priority_for_principal": [
            "agent-reliability substrate ownership",
            "frontier-venue research-engineering fluency",
            "evaluation-infrastructure conviction (under-represented)",
        ],
        "confidence": 0.74,
        "source": "deterministic",
    }

    orchestration_store = OrchestrationStateStore(
        resolve_orchestration_db_path()
    )
    orchestration_store.insert_chief_of_staff_run(
        brief_id=cos_brief_id,
        principal_id=PRINCIPAL_ID,
        status="completed",
        dispatch_plan={
            "steps": [
                {"source": "linkedin", "rationale": "broadest surface area"},
                {
                    "source": "researcher",
                    "rationale": "frontier-venue authorship is load-bearing",
                },
                {
                    "source": "github",
                    "rationale": "maintainership on target_projects",
                },
            ],
            "confidence": 0.81,
        },
        invocation_order=["linkedin", "researcher", "github"],
        handoff_payloads=handoff_payloads,
        synthesis_output=synthesis_output,
        started_at=_iso(started_at),
        ended_at=_iso(gh_end + timedelta(minutes=12)),
    )

    # Two cross-brief playbook observations tied to a synthetic principal +
    # the frontier-AI market_key. These are append-only calibration logs.
    now = _utc_now()
    with orchestration_store.connect() as conn:
        for offset, observation in enumerate(
            [
                {
                    "kind": "dispatch_order_validated",
                    "claim": (
                        "For frontier-AI briefs at this principal, "
                        "linkedin-first dispatch surfaced higher-quality "
                        "saves than github-first; consistent across two "
                        "briefs."
                    ),
                    "evidence_sources": [li_state_key],
                    "confidence": 0.71,
                },
                {
                    "kind": "evaluation_capability_under_represented",
                    "claim": (
                        "Across the last two frontier-AI briefs, "
                        "evaluation-infrastructure depth has surfaced less "
                        "than agent-reliability. The role-shape language "
                        "may be over-anchoring on agents."
                    ),
                    "evidence_sources": [li_state_key, rs_state_key],
                    "confidence": 0.62,
                },
            ]
        ):
            conn.execute(
                """
                INSERT INTO cross_brief_playbook_observations(
                    principal_id, market_key, role_shape, brief_id,
                    observation_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    PRINCIPAL_ID,
                    MARKET_KEY_FRONTIER_AI,
                    "head-of-role",
                    cos_brief_id,
                    json.dumps(observation, sort_keys=True),
                    _iso(now - timedelta(days=offset)),
                ),
            )
        conn.commit()

    # Run artifacts per source.
    _seed_run_artifacts(
        source="linkedin",
        brief_id=li_state_key,
        state_dir=li_dir,
        run_id=li_run_id,
        started_at=started_at,
        ended_at=li_end,
        string_count=4,
        save_count=5,
        facial_yes=8,
        facial_no=10,
        facial_borderline=5,
        rejected=10,
        cost_modules={"linkedin": 2.84},
    )
    _seed_run_artifacts(
        source="researcher",
        brief_id=rs_state_key,
        state_dir=rs_dir,
        run_id=rs_run_id,
        started_at=started_at + timedelta(minutes=15),
        ended_at=rs_end,
        string_count=3,
        save_count=5,
        facial_yes=6,
        facial_no=10,
        facial_borderline=2,
        rejected=10,
        cost_modules={"researcher": 1.98},
    )
    _seed_run_artifacts(
        source="github",
        brief_id=gh_state_key,
        state_dir=gh_dir,
        run_id=gh_run_id,
        started_at=started_at + timedelta(minutes=30),
        ended_at=gh_end,
        string_count=3,
        save_count=4,
        facial_yes=4,
        facial_no=8,
        facial_borderline=0,
        rejected=8,
        cost_modules={"github": 1.62},
    )

    return {
        "brief_id": cos_brief_id,
        "linkedin_state_key": li_state_key,
        "researcher_state_key": rs_state_key,
        "github_state_key": gh_state_key,
        "candidates": 50,
        "handoff_sources": list(handoff_payloads.keys()),
    }

# ---------------------------------------------------------------------------
# BRIEF 3 — Senior Product Designer (Designer module)
# ---------------------------------------------------------------------------


DESIGNER_PRINCIPLE_NAMES = (
    "Visual hierarchy",
    "Typographic refinement",
    "Compositional balance",
    "Color system coherence",
    "Conceptual strength",
    "Craft execution",
)
SCORE_TO_ANCHOR = {0: "bad", 1: "okay", 2: "good", 3: "excellent"}


def _designer_principle_block(
    principle: str,
    score: int,
    *,
    person: SyntheticPerson,
    image_ids: tuple[int, ...],
) -> dict:
    """Build a single VisualJudgmentPrinciple-shaped dict.

    The reasoning prose is principle-grounded and cites the (synthetic)
    image_ids — same shape the production vision-evaluation prompt
    produces. Recruiter-readable, first-person editorial.
    """

    anchor = SCORE_TO_ANCHOR[score]
    reasoning_table = {
        "Visual hierarchy": {
            3: (
                f"{person.display_name}'s clinical-intake case study "
                f"directs attention with conviction — the patient summary, "
                f"the next-step CTA, and the supporting context land in "
                f"the right order without sacrificing density."
            ),
            2: (
                f"{person.display_name} carries hierarchy on the primary "
                f"surfaces but the secondary screens fall back to default-"
                f"grid layouts that don't pull weight."
            ),
            1: (
                f"Hierarchy is present in {person.display_name}'s work but "
                f"the distinctions between primary and secondary content "
                f"require effort to read."
            ),
            0: (
                f"{person.display_name}'s screens compete for attention "
                f"without a primary focal point; the eye does not know "
                f"where to land."
            ),
        },
        "Typographic refinement": {
            3: (
                f"Type is intentional throughout. {person.display_name} "
                f"pairs a serif display with a humanist sans at small "
                f"sizes; kerning and leading hold up at clinical-label "
                f"densities."
            ),
            2: (
                f"Type choices read as workable rather than decisive. "
                f"{person.display_name} relies on system pairings that "
                f"do the job without authoring much."
            ),
            1: (
                f"Type is consistent but flat — no evidence that "
                f"{person.display_name} treated it as expressive material."
            ),
            0: (
                f"Type is undifferentiated. The work uses default "
                f"typeface choices and the rhythm is not authored."
            ),
        },
        "Compositional balance": {
            3: (
                f"{person.display_name}'s compositions deploy negative "
                f"space as a positive material; rhythm and pacing are "
                f"intentional and the screens carry weight without "
                f"feeling crowded."
            ),
            2: (
                f"Composition is considered on the marquee surfaces but "
                f"the secondary screens read as default grids."
            ),
            1: (
                f"Compositional choices are workable but predictable. "
                f"The grid does the job without ambition."
            ),
            0: (
                f"Compositions feel arbitrary — elements placed rather "
                f"than arranged."
            ),
        },
        "Color system coherence": {
            3: (
                f"{person.display_name}'s palette has perspective. "
                f"Colors carry functional meaning on the clinical "
                f"surfaces and decorative restraint on the marketing ones."
            ),
            2: (
                f"Color is systematic where it shows up but the "
                f"discipline is uneven across cases."
            ),
            1: (
                f"Color works as a palette but isn't a system; "
                f"relationships between hues are approximate."
            ),
            0: (
                f"Color reads as incidental — multiple palettes within a "
                f"single product surface."
            ),
        },
        "Conceptual strength": {
            3: (
                f"{person.display_name}'s case study proposes a frame "
                f"for the work rather than executing one. The thesis is "
                f"original and the visual choices make it inevitable."
            ),
            2: (
                f"Concept is sharp on the strongest piece but generic "
                f"on the supporting work."
            ),
            1: (
                f"Concept is recognizable but not particularly its own."
            ),
            0: (
                f"No discernible concept beneath the surface — execution "
                f"without a frame."
            ),
        },
        "Craft execution": {
            3: (
                f"Craft is at the limit the medium allows. Spacing, "
                f"alignment, and finish are precise across breakpoints; "
                f"every detail serves the whole."
            ),
            2: (
                f"Craft is clean on the marquee surfaces; secondary "
                f"screens carry occasional alignment slips."
            ),
            1: (
                f"Craft is acceptable — few errors but the precision "
                f"isn't load-bearing."
            ),
            0: (
                f"Visible craft errors: misaligned grids, low-resolution "
                f"image assets, inconsistent corner radii."
            ),
        },
    }
    reasoning = reasoning_table.get(principle, {}).get(
        score,
        f"Score {score} ({anchor}) reasoning placeholder.",
    )
    return {
        "name": principle,
        "score": score,
        "anchor": anchor,
        "reasoning": reasoning,
        "image_ids": list(image_ids),
        "anchor_consistency_pass": True,
    }


def _brief_3_visual_judgment(
    person: SyntheticPerson,
    *,
    overall_verdict: str,
    score_floor: int,
    score_ceiling: int,
    rng: random.Random,
    cross_check: dict | None = None,
) -> dict:
    """Compose a VisualJudgment dict for one Designer candidate."""

    principles = []
    for idx, name in enumerate(DESIGNER_PRINCIPLE_NAMES):
        score = rng.randint(score_floor, score_ceiling)
        image_ids = tuple(range(idx + 1, idx + 1 + rng.randint(1, 3)))
        principles.append(
            _designer_principle_block(
                name, score, person=person, image_ids=image_ids
            )
        )
    payload: dict[str, Any] = {
        "model": "gemini-2.5-pro",
        "principles": principles,
        "overall_verdict": overall_verdict,
        "overall_confidence": round(0.65 + rng.random() * 0.3, 3),
        "fallback_reason": "",
        "cost_estimate_usd": round(0.04 + rng.random() * 0.05, 5),
    }
    if cross_check is not None:
        payload["cross_check"] = cross_check
    return payload


def seed_brief_3_designer(rng: random.Random) -> dict[str, Any]:
    """Seed Brief 3 — Senior Product Designer (Designer module).

    State landed:
    - 1 completed Designer run with 15 candidates
    - Per-candidate visual_judgment with all 6 principles scored
    - 3 candidates with cross_check payloads (one disagreement)
    - 1 candidate with a recruiter-excluded asset
    - 2 candidates with PrincipleFeedbackStore markers + judgment_accuracy
    - design_market.md artifact under output/market_intelligence/
    """

    brief_path = REPO_ROOT / "config" / BRIEF_3_SLUG / "brief.json"
    brief = load_brief(str(brief_path))
    state_key = designer_state_key(brief_path=brief_path)

    _drop_state_dir("designer", state_key)
    _drop_run_dirs("designer", state_key)
    _drop_market_intel_dir(state_key)

    store, state_dir = _init_runtime_state("designer", state_key)

    started_at = _utc_now() - timedelta(hours=6)
    ended_at = started_at + timedelta(hours=1, minutes=42)

    run_id = _seed_run_row(
        store,
        source="designer",
        brief_id=state_key,
        state_dir=state_dir,
        started_at=started_at,
        ended_at=ended_at,
    )

    people = _make_people(rng, n=15, name_offset=120)

    # Build assets per candidate. Slug into the synthetic-portfolio
    # placeholder host so nothing in the fixture resolves to a real asset.
    def _portfolio_url(handle: str, idx: int) -> str:
        return f"https://portfolio.example.test/{handle}/image-{idx:02d}.png"

    saved_count = 0
    for idx, person in enumerate(people):
        if idx < 3:
            # Top-decile saves with cross-check payload.
            cross_score = max(0, min(3, idx + 2))
            primary_anchor = "good" if idx != 1 else "good"
            cross_anchor = "okay" if idx == 1 else "good"
            cross_check = {
                "model": "claude-sonnet-4-6",
                "overall_verdict": "yes" if idx != 1 else "borderline",
                "overall_confidence": 0.71 if idx != 1 else 0.46,
                "principles": [
                    {
                        "name": principle_name,
                        "score": cross_score if idx != 1 and principle_name != "Visual hierarchy" else max(0, cross_score - 2),
                        "anchor": primary_anchor if idx != 1 else cross_anchor,
                        "reasoning": (
                            f"Cross-check pass agrees with the primary "
                            f"read on {principle_name.lower()}."
                            if idx != 1
                            else (
                                f"Cross-check pass disagrees on "
                                f"{principle_name.lower()}: the cited "
                                f"images suggest the score is two anchor "
                                f"levels lower than the primary read."
                            )
                        ),
                    }
                    for principle_name in DESIGNER_PRINCIPLE_NAMES
                ],
                "agreement_with_primary": idx != 1,
                "max_disagreement_anchors": 2 if idx == 1 else 0,
            }
            judgment_payload = _brief_3_visual_judgment(
                person,
                overall_verdict="yes",
                score_floor=2,
                score_ceiling=3,
                rng=rng,
                cross_check=cross_check,
            )
            decision = "SAVE"
            lifecycle = "full_terminal"
            judgment_accuracy = None
            if idx == 0:
                judgment_accuracy = "useful"
            saved_count += 1
        elif idx < 5:
            # Two more saves (no cross-check) with feedback markers.
            judgment_payload = _brief_3_visual_judgment(
                person,
                overall_verdict="yes",
                score_floor=2,
                score_ceiling=3,
                rng=rng,
            )
            decision = "SAVE"
            lifecycle = "full_terminal"
            judgment_accuracy = "useful" if idx == 3 else None
            saved_count += 1
        elif idx < 8:
            # Borderline — full eval but verdict=borderline.
            judgment_payload = _brief_3_visual_judgment(
                person,
                overall_verdict="borderline",
                score_floor=1,
                score_ceiling=2,
                rng=rng,
            )
            decision = "FACIAL_BORDERLINE"
            lifecycle = "full_terminal"
            judgment_accuracy = None
        else:
            # Rejects.
            judgment_payload = _brief_3_visual_judgment(
                person,
                overall_verdict="no",
                score_floor=0,
                score_ceiling=1,
                rng=rng,
            )
            decision = "REJECT"
            lifecycle = "full_terminal"
            judgment_accuracy = None

        # Synthetic asset URLs cited by the judgment.
        cited_assets = [
            {"id": i, "url": _portfolio_url(person.handle, i), "source": "google_cse", "project_title": f"Project {i}"}
            for i in range(1, 6)
        ]

        terminal_payload = {
            "surface_type": "hitl_visual_review",
            "visual_judgment": {**judgment_payload, "assets": cited_assets},
            "full_decision": {
                "decision": decision,
                "rationale": (
                    f"{person.display_name} — overall verdict "
                    f"{judgment_payload['overall_verdict']!r} grounded in "
                    f"per-principle scoring; see visual_judgment block."
                ),
                "confidence": judgment_payload["overall_confidence"],
            },
        }

        # D5b: assemble recommendation_pitch for non-REJECT candidates.
        if decision != "REJECT":
            from designer.recommendation_pitch import assemble_recommendation_pitch

            pitch = assemble_recommendation_pitch(
                terminal_payload, role_title="Senior Product Designer"
            )
            if pitch is not None:
                terminal_payload["recommendation_pitch"] = pitch

        _seed_candidate_row(
            store,
            source="designer",
            brief_id=state_key,
            person=person,
            profile_url=f"https://portfolio.example.test/{person.handle}",
            lifecycle_state=lifecycle,
            terminal_decision=decision,
            terminal_payload=terminal_payload,
            judgment_accuracy=judgment_accuracy,
            last_seen_at=ended_at - timedelta(minutes=idx),
        )

    # Recruiter annotation stores live alongside the runtime SQLite. The
    # ExcludedAssetStore + PrincipleFeedbackStore each create their own
    # SQLite file under the state-dir.
    from designer.recruiter_annotations import (
        ExcludedAssetStore,
        PrincipleFeedbackStore,
        RECOGNIZED_FEEDBACK_MARKERS,  # noqa: F401 — sanity import
    )

    excluded_store = ExcludedAssetStore(state_dir / "annotations.sqlite3")
    feedback_store = PrincipleFeedbackStore(
        state_dir / "annotations.sqlite3"
    )

    excluded_store.exclude(
        candidate_identity_key=people[2].identity_key,
        asset_url=_portfolio_url(people[2].handle, 3),
        reason=(
            "That's their old portfolio screenshot — the case study has "
            "been refreshed and this image misrepresents the work."
        ),
    )

    # Two per-principle feedback markers + judgment_accuracy mirror via
    # the Slice 3.6 reconciliation helper. We invoke the bridge so the
    # canonical candidates row + per-principle store stay in sync.
    feedback_store.record(
        candidate_identity_key=people[3].identity_key,
        principle_name="Craft execution",
        marker="useful_guidance",
        note=(
            "The craft read here was right — the candidate's spacing "
            "discipline at small sizes is unusual."
        ),
    )
    feedback_store.record(
        candidate_identity_key=people[4].identity_key,
        principle_name="Visual hierarchy",
        marker="wrong_shallow",
        note=(
            "The hierarchy read missed the actual primary screens of the "
            "case study; the candidate is stronger here than the score "
            "suggests."
        ),
    )

    # Mirror the feedback markers onto the candidates row's
    # judgment_accuracy column to match the Slice 3.6 reconciliation
    # contract. These specific candidates already had judgment_accuracy
    # set above, so the mirror just confirms the unified column.
    store.record_candidate_principle_marker(
        source="designer",
        brief_id=state_key,
        identity_key=people[3].identity_key,
        judgment_accuracy="useful",
        principle_marker={
            "principle_name": "Craft execution",
            "marker": "useful_guidance",
            "note": "Craft read was right.",
            "marked_at": _iso(ended_at + timedelta(minutes=22)),
        },
    )
    store.record_candidate_principle_marker(
        source="designer",
        brief_id=state_key,
        identity_key=people[4].identity_key,
        judgment_accuracy="wrong",
        principle_marker={
            "principle_name": "Visual hierarchy",
            "marker": "wrong_shallow",
            "note": "Hierarchy read missed primary screens.",
            "marked_at": _iso(ended_at + timedelta(minutes=24)),
        },
    )

    # Reflection at Gate 2 (awaiting_diff) with two RUBRIC_REFINE-class
    # hunks proposed.
    reflection_state = {
        "phase": "awaiting_diff",
        "brief_path": str(brief_path),
        "market_identity": {
            "market_key": f"senior_product_designer__united_states__senior",
            "role_title": brief.role_title,
            "role_level": "senior",
            "geography": "United States",
        },
        "planner_summary": {
            "headline": (
                "Pool reads strongly on craft and hierarchy; conceptual-"
                "strength scoring is bimodal — either sharp or generic, "
                "with little middle ground."
            ),
        },
        "proposed_hunks": [
            {
                "hunk_id": "fixture-rubric-refine-1",
                "kind": "rubric_refine",
                "field_path": "design_rubric.principles[Visual hierarchy].weight",
                "before": 1.4,
                "after": 1.6,
                "rationale": (
                    "Recruiter feedback consistently marked Visual "
                    "hierarchy as load-bearing for product-density "
                    "verdicts; weight bump aligns the rubric to the "
                    "feedback."
                ),
            },
            {
                "hunk_id": "fixture-rubric-refine-2",
                "kind": "rubric_refine",
                "field_path": "design_rubric.discipline_weight_overrides.product",
                "before": {
                    "Visual hierarchy": 1.5,
                    "Compositional balance": 1.2,
                    "Conceptual strength": 0.7,
                },
                "after": {
                    "Visual hierarchy": 1.7,
                    "Compositional balance": 1.2,
                    "Conceptual strength": 0.6,
                    "Craft execution": 1.4,
                },
                "rationale": (
                    "Adding a craft-execution override reflects the "
                    "principle's load on clinical-product surfaces; "
                    "lowering conceptual-strength further matches the "
                    "bimodal scoring pattern observed in this run."
                ),
            },
        ],
    }
    _seed_reflection_session(
        store,
        brief_id=state_key,
        source_run_id=run_id,
        current_phase="awaiting_diff",
        state_json=reflection_state,
        started_at=ended_at + timedelta(minutes=8),
        updated_at=ended_at + timedelta(minutes=18),
    )

    # design_market.md artifact under output/market_intelligence/<state_key>/.
    mi_dir = MARKET_INTEL_ROOT / state_key
    mi_dir.mkdir(parents=True, exist_ok=True)
    design_market_md = (
        f"# Design Market — {brief.role_title}\n\n"
        f"_Run completed {ended_at.strftime('%Y-%m-%d')}_\n\n"
        f"## Pool composition\n\n"
        f"- Total candidates surfaced: 15\n"
        f"- Source mix: Behance 9 / Google CSE 6\n"
        f"- Discipline distribution: product 11 / brand 3 / ux 1\n\n"
        f"## Per-principle recruiter feedback\n\n"
        f"- Visual hierarchy: 1 useful, 1 wrong-shallow (the run's "
        f"hierarchy read missed primary screens on one save)\n"
        f"- Craft execution: 1 useful guidance\n"
        f"- Other principles: no feedback markers\n\n"
        f"## Cross-check disagreement rate\n\n"
        f"3 of 15 candidates received a cross-check pass; 1 of those 3 "
        f"showed a >1-anchor disagreement on Visual hierarchy. The "
        f"workspace surfaces this as a MODELS DISAGREE eyebrow on the "
        f"affected card.\n\n"
        f"## Proposed rubric refinements\n\n"
        f"1. Bump the Visual hierarchy principle's base weight from 1.4 "
        f"to 1.6 to reflect the recruiter feedback density.\n"
        f"2. Add a craft-execution override for the product discipline "
        f"at weight 1.4; the principle is load-bearing on clinical-"
        f"product surfaces in ways the default doesn't capture.\n\n"
        f"_Both hunks land in the workspace's awaiting-diff state for "
        f"recruiter approval._\n"
    )
    (mi_dir / "design_market.md").write_text(design_market_md)

    _seed_run_artifacts(
        source="designer",
        brief_id=state_key,
        state_dir=state_dir,
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        string_count=3,
        save_count=saved_count,
        facial_yes=8,
        facial_no=4,
        facial_borderline=3,
        rejected=7,
        cost_modules={"designer": 2.21},
    )

    return {
        "brief_id": state_key,
        "candidates": 15,
        "cross_check_disagreements": 1,
        "design_market_artifact": str(mi_dir / "design_market.md"),
    }




# ---------------------------------------------------------------------------
# Top-level seeder
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# BRIEFS 4 + 5 — In-flight intake sessions
# ---------------------------------------------------------------------------


def seed_intake_drafts() -> dict[str, Any]:
    """Seed four intake_sessions rows: Brief 4 (in-flight at depth_distinction)
    plus three drafts (5a/b/c) at varying chapters and staleness.

    Idempotency: drops existing rows with matching role_titles before
    seeding. The drafts list page is the surface this populates.
    """

    role_titles = [
        "VP of Engineering",
        "Director of Product",
        "Senior Designer",
    ]
    _drop_intake_sessions_by_role_titles(role_titles)

    intake_db = resolve_intake_db_path()
    store = RuntimeStateStore(intake_db)
    now = _utc_now()
    seeded: list[dict] = []

    # Brief 4 — in-flight intake at depth_distinction (~3h ago started,
    # ~2 min ago last touched). The state_json carries the prior
    # chapters' data so resume picks up cleanly.
    brief4 = create_intake_session(store, role_title="VP of Engineering")
    state_4 = {
        "welcome": {"completed_at": _iso(now - timedelta(hours=3))},
        "role": {
            "role_title": "VP of Engineering",
            "role_summary": (
                "VP of Engineering for a 60-person Series B; first VP "
                "hire, partnering with the CTO on the next phase of "
                "scaling the engineering org."
            ),
            "geography": "United States",
            "minimum_years_experience": 12,
        },
        "good_looks": {
            "capability_areas": [
                {
                    "name": "Org-design fluency",
                    "description": "Has scaled a 30-person team to 80+",
                },
                {
                    "name": "Production-systems credibility",
                    "description": (
                        "Engineers respect the VP because they could still "
                        "review the on-call runbook"
                    ),
                },
            ],
        },
        "lookalikes": {
            "linkedin_urls": [
                "https://www.linkedin.com/in/example-vpe-lookalike-1",
                "https://www.linkedin.com/in/example-vpe-lookalike-2",
            ],
        },
        # depth_distinction chapter is in-flight — current_step is
        # "depth_distinction" so the resume surface lands here.
    }
    patch_intake_session(
        store,
        session_id=brief4["id"],
        current_step="depth_distinction",
        state_json=state_4,
        role_title="VP of Engineering",
    )
    # Backdate started_at so the staleness cue logic has the right
    # picture: started 3h ago, updated 2 minutes ago.
    with store.connect() as conn:
        conn.execute(
            "UPDATE intake_sessions SET started_at = ?, updated_at = ? WHERE id = ?",
            (
                _iso(now - timedelta(hours=3)),
                _iso(now - timedelta(minutes=2)),
                brief4["id"],
            ),
        )
    seeded.append({"slug": "brief-4-in-flight", "session_id": brief4["id"]})

    # Draft 5a — welcome chapter, 45 minutes stale (above the
    # STALE_RESUME_MINUTES threshold of 30 — staleness cue fires).
    draft_5a = create_intake_session(store, role_title=None)
    state_5a = {
        "welcome": {
            # Marker so idempotency drop-by-role-title-NULL knows this
            # is the fixture and not unrelated user-anonymous drafts.
            "fixture-draft-5a": True,
        }
    }
    patch_intake_session(
        store,
        session_id=draft_5a["id"],
        current_step="welcome",
        state_json=state_5a,
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE intake_sessions SET started_at = ?, updated_at = ? WHERE id = ?",
            (
                _iso(now - timedelta(minutes=46)),
                _iso(now - timedelta(minutes=45)),
                draft_5a["id"],
            ),
        )
    seeded.append({"slug": "draft-5a-welcome-stale", "session_id": draft_5a["id"]})

    # Draft 5b — lookalikes chapter, 5 hours stale (very stale).
    draft_5b = create_intake_session(store, role_title="Director of Product")
    state_5b = {
        "role": {
            "role_title": "Director of Product",
            "role_summary": "Director of Product for a B2B SaaS at growth stage.",
            "geography": "United States",
        },
        "good_looks": {
            "capability_areas": [
                {
                    "name": "Strategic clarity",
                    "description": "Can author the why-now for the company.",
                },
            ],
        },
    }
    patch_intake_session(
        store,
        session_id=draft_5b["id"],
        current_step="lookalikes",
        state_json=state_5b,
        role_title="Director of Product",
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE intake_sessions SET started_at = ?, updated_at = ? WHERE id = ?",
            (
                _iso(now - timedelta(hours=6)),
                _iso(now - timedelta(hours=5)),
                draft_5b["id"],
            ),
        )
    seeded.append({"slug": "draft-5b-lookalikes-stale", "session_id": draft_5b["id"]})

    # Draft 5c — where_to_look chapter, 10 minutes stale (fresh).
    draft_5c = create_intake_session(store, role_title="Senior Designer")
    state_5c = {
        "role": {
            "role_title": "Senior Designer",
            "role_summary": "Senior brand designer for a consumer wellness startup.",
            "geography": "United States",
        },
        "good_looks": {
            "capability_areas": [
                {
                    "name": "Editorial type",
                    "description": "Treats type as expressive material.",
                },
            ],
        },
        "depth_distinction": {
            "builder_definition": "Authors brand systems, doesn't apply them.",
            "user_definition": "Has used Figma libraries someone else built.",
            "edge_case_guidance": "Lean save when the candidate's case studies show authoring decisions.",
        },
        "lookalikes": {"linkedin_urls": []},
    }
    patch_intake_session(
        store,
        session_id=draft_5c["id"],
        current_step="where_to_look",
        state_json=state_5c,
        role_title="Senior Designer",
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE intake_sessions SET started_at = ?, updated_at = ? WHERE id = ?",
            (
                _iso(now - timedelta(hours=2)),
                _iso(now - timedelta(minutes=10)),
                draft_5c["id"],
            ),
        )
    seeded.append({"slug": "draft-5c-where-to-look-fresh", "session_id": draft_5c["id"]})

    return {"intake_sessions": seeded}

def seed_all() -> dict[str, Any]:
    """Run every fixture seeder; safe to invoke repeatedly.

    Returns a small summary dict for the runbook + tests.
    """

    rng = random.Random(RANDOM_SEED)

    # Validate every fixture brief.json before touching state — the
    # seeder never lands runtime state for a brief that fails V2
    # validation.
    for slug in (BRIEF_1_SLUG, BRIEF_2_SLUG, BRIEF_3_SLUG):
        brief_path = REPO_ROOT / "config" / slug / "brief.json"
        validate_v2_brief(json.loads(brief_path.read_text()))

    summary = {
        "brief_1": seed_brief_1_linkedin(rng),
        "brief_2": seed_brief_2_multi_module(rng),
        "brief_3": seed_brief_3_designer(rng),
        "intake": seed_intake_drafts(),
    }
    return summary


def main() -> None:
    summary = seed_all()
    print("seeded fixtures:")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
