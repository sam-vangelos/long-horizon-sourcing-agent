"""Designer module — recruiter annotation primitives.

Designer Slice 7. The load-bearing HITL distinction (per spec §5.4):
the recruiter sees Cloris's vision judgment alongside the actual
images Cloris evaluated, can flag any image as misrepresentative,
and can mark per-principle feedback ("Useful guidance" /
"Wrong / shallow" / "Off-rubric") that feeds into Slice 9's
reflection polish.

This module owns the data layer for those annotations:

- :class:`ExcludedAssetStore` — append-only log of recruiter-excluded
  ``(candidate_identity_key, asset_url)`` pairs with optional reason.
  Stored alongside :mod:`designer.image_acquisition`'s asset cache
  in the same per-state-dir SQLite file.
- :class:`PrincipleFeedbackStore` — append-only log of per-principle
  feedback markers (one row per ``(candidate, principle, marker)``
  tuple).
- ``compute_re_eval_asset_set(candidate, original_asset_set,
  excluded_asset_urls) -> list[asset_id]`` — pure function: returns
  the asset_ids the re-evaluation pass should consume after
  exclusions are applied.
- ``feedback_marker_distribution(brief_id) -> dict[str, int]`` —
  recruiter-feedback rollup for the workspace surface and Slice 9's
  reflection polish prompt input.
- ``record_designer_principle_feedback(...)`` — Slice 3.6 bridge: writes
  the per-principle marker to the per-state-dir
  :class:`PrincipleFeedbackStore` AND mirrors a unified
  ``judgment_accuracy`` value onto the canonical ``runtime_state.sqlite3``
  ``candidates`` row (with the per-principle detail as
  ``terminal_payload_json`` metadata). Same column, additional detail —
  the calibration aggregator at
  ``shared/runtime_state/calibration.py`` reads ``judgment_accuracy``
  uniformly across modules.

The HTTP endpoint that dispatches into these primitives lives in
``cloris/api.py`` (Slice 7's wire-layer addition). The orchestrator
that re-runs the vision evaluation against the reduced asset set
is wired in Slice 7 as well; this module is the pure-data substrate.

Design choice: append-only stores with a ``revoked_at`` soft-delete
column rather than mutable rows. Recruiter annotations are recruiter
intent over time — a recruiter who excludes then un-excludes an
asset has a different signal than a recruiter who never excluded it,
and Slice 9's reflection polish needs that history.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.runtime_state.store import RuntimeStateStore


# Recognized per-principle feedback markers. Slice 7 ships three
# (positive / neutral-skeptical / negative); Slice 9's reflection
# polish maps these to rubric weight refinements (consistently
# "Off-rubric" → propose lower weight; consistently "Useful guidance"
# → propose higher weight or new exemplar).
RECOGNIZED_FEEDBACK_MARKERS: frozenset[str] = frozenset(
    {"useful_guidance", "wrong_shallow", "off_rubric"}
)


@dataclass(frozen=True)
class ExcludedAsset:
    excluded_id: int
    candidate_identity_key: str
    asset_url: str
    reason: str
    excluded_at: str
    revoked_at: str | None


@dataclass(frozen=True)
class PrincipleFeedbackMarker:
    marker_id: int
    candidate_identity_key: str
    principle_name: str
    marker: str
    note: str
    marked_at: str


class ExcludedAssetStore:
    """Append-only store for recruiter-excluded assets.

    Schema:
    - ``excluded_id`` PK
    - ``(candidate_identity_key, asset_url)`` is NOT unique — exclude
      → revoke → re-exclude is a valid sequence and each row
      preserves the recruiter's intent at that point in time.
    - ``revoked_at`` is non-null when the recruiter un-excluded; the
      "is this asset currently excluded?" query is "exists at least
      one row with revoked_at IS NULL for this (candidate, url)".
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS excluded_assets (
        excluded_id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_identity_key TEXT NOT NULL,
        asset_url TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        excluded_at TEXT NOT NULL,
        revoked_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_excluded_candidate
        ON excluded_assets(candidate_identity_key);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self.SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def exclude(
        self,
        *,
        candidate_identity_key: str,
        asset_url: str,
        reason: str = "",
    ) -> ExcludedAsset:
        """Record a recruiter exclusion.

        Idempotent on currently-active exclusions: if the recruiter
        excludes the same asset twice without revoking, the second
        call returns the existing row rather than inserting a
        duplicate. Re-exclude AFTER revoke produces a new row.
        """

        existing = self.active_exclusion(
            candidate_identity_key=candidate_identity_key, asset_url=asset_url
        )
        if existing is not None:
            return existing

        excluded_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO excluded_assets (
                    candidate_identity_key, asset_url, reason, excluded_at, revoked_at
                ) VALUES (?, ?, ?, ?, NULL)
                """,
                (candidate_identity_key, asset_url, reason, excluded_at),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM excluded_assets WHERE excluded_id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return _row_to_excluded(row)

    def revoke(
        self,
        *,
        candidate_identity_key: str,
        asset_url: str,
    ) -> bool:
        """Mark the active exclusion as revoked. Returns True if
        a row was revoked, False if no active exclusion existed.
        """

        active = self.active_exclusion(
            candidate_identity_key=candidate_identity_key, asset_url=asset_url
        )
        if active is None:
            return False
        revoked_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE excluded_assets SET revoked_at = ? WHERE excluded_id = ?",
                (revoked_at, active.excluded_id),
            )
            conn.commit()
        return True

    def active_exclusion(
        self,
        *,
        candidate_identity_key: str,
        asset_url: str,
    ) -> ExcludedAsset | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM excluded_assets
                WHERE candidate_identity_key = ?
                  AND asset_url = ?
                  AND revoked_at IS NULL
                ORDER BY excluded_id DESC
                LIMIT 1
                """,
                (candidate_identity_key, asset_url),
            ).fetchone()
        return _row_to_excluded(row) if row is not None else None

    def active_exclusions_for_candidate(
        self, candidate_identity_key: str
    ) -> tuple[ExcludedAsset, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM excluded_assets
                WHERE candidate_identity_key = ?
                  AND revoked_at IS NULL
                ORDER BY excluded_id
                """,
                (candidate_identity_key,),
            ).fetchall()
        return tuple(_row_to_excluded(row) for row in rows)


class PrincipleFeedbackStore:
    """Append-only store for per-principle recruiter feedback markers.

    Schema:
    - ``marker_id`` PK
    - ``marker`` MUST be in :data:`RECOGNIZED_FEEDBACK_MARKERS`; the
      :func:`record` validates and raises on unknown markers.
    - ``(candidate, principle, marker)`` is NOT unique — a recruiter
      may mark the same principle differently across runs (their
      taste evolves; this is signal for Slice 9 reflection polish).
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS principle_feedback (
        marker_id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_identity_key TEXT NOT NULL,
        principle_name TEXT NOT NULL,
        marker TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT '',
        marked_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_feedback_candidate
        ON principle_feedback(candidate_identity_key);
    CREATE INDEX IF NOT EXISTS idx_feedback_principle
        ON principle_feedback(principle_name);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self.SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def record(
        self,
        *,
        candidate_identity_key: str,
        principle_name: str,
        marker: str,
        note: str = "",
    ) -> PrincipleFeedbackMarker:
        """Append a feedback marker.

        Raises ``ValueError`` for unknown markers — the wire layer
        catches and maps to HTTP 422 so the recruiter's input never
        silently fails validation.
        """

        if marker not in RECOGNIZED_FEEDBACK_MARKERS:
            raise ValueError(
                f"Unknown feedback marker {marker!r}; "
                f"expected one of {sorted(RECOGNIZED_FEEDBACK_MARKERS)}"
            )
        marked_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO principle_feedback (
                    candidate_identity_key, principle_name, marker, note, marked_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (candidate_identity_key, principle_name, marker, note, marked_at),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM principle_feedback WHERE marker_id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return _row_to_marker(row)

    def markers_for_candidate(
        self, candidate_identity_key: str
    ) -> tuple[PrincipleFeedbackMarker, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM principle_feedback
                WHERE candidate_identity_key = ?
                ORDER BY marker_id
                """,
                (candidate_identity_key,),
            ).fetchall()
        return tuple(_row_to_marker(row) for row in rows)

    def feedback_marker_distribution(self) -> dict[str, dict[str, int]]:
        """Roll up marker counts per principle.

        Returns a nested dict ``{principle_name: {marker: count}}``.
        Slice 9's reflection polish prompt grounds itself in this
        distribution to propose rubric weight refinements.
        """

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT principle_name, marker, COUNT(*) AS cnt
                FROM principle_feedback
                GROUP BY principle_name, marker
                """
            ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            out.setdefault(row["principle_name"], {})[row["marker"]] = int(row["cnt"])
        return out


def _row_to_excluded(row: sqlite3.Row) -> ExcludedAsset:
    return ExcludedAsset(
        excluded_id=int(row["excluded_id"]),
        candidate_identity_key=str(row["candidate_identity_key"]),
        asset_url=str(row["asset_url"]),
        reason=str(row["reason"] or ""),
        excluded_at=str(row["excluded_at"]),
        revoked_at=str(row["revoked_at"]) if row["revoked_at"] is not None else None,
    )


def _row_to_marker(row: sqlite3.Row) -> PrincipleFeedbackMarker:
    return PrincipleFeedbackMarker(
        marker_id=int(row["marker_id"]),
        candidate_identity_key=str(row["candidate_identity_key"]),
        principle_name=str(row["principle_name"]),
        marker=str(row["marker"]),
        note=str(row["note"] or ""),
        marked_at=str(row["marked_at"]),
    )


# ---------------------------------------------------------------------------
# Re-evaluation asset-set helper
# ---------------------------------------------------------------------------


def compute_re_eval_asset_set(
    *,
    original_assets: list[tuple[int, str]],
    excluded_asset_urls: set[str],
) -> list[int]:
    """Return the asset_ids the re-evaluation pass should consume.

    Trivially: drop any asset whose URL is in ``excluded_asset_urls``,
    return the rest's ids in original order. Pure function so the
    re-eval orchestrator can compute the asset set without touching
    the SQLite stores.

    Intentionally bare: future scope (e.g., enriching with newly-
    fetched assets to backfill the excluded ones) lands here, but
    Slice 7 ships the minimum-viable filter only.
    """

    return [
        asset_id
        for (asset_id, asset_url) in original_assets
        if asset_url not in excluded_asset_urls
    ]


# ---------------------------------------------------------------------------
# Slice 3.6 reconciliation: Designer per-principle markers ↔ canonical
# ``judgment_accuracy``
# ---------------------------------------------------------------------------


# Slice 3.6 mapping: Designer's three-value per-principle enum projects
# onto the canonical five-value ``judgment_accuracy`` enum at
# ``shared/runtime_state/store.py:660-666``. The two depth-nuance values
# (``overstated_depth`` / ``understated_depth``) are NOT reachable from
# the Designer surface today — Designer's per-principle UI is "useful
# guidance / wrong shallow / off-rubric" only. The aggregator's allowed
# set still accepts them (cross-module), but Designer never writes them
# from this bridge.
DESIGNER_MARKER_TO_JUDGMENT_ACCURACY: dict[str, str] = {
    "useful_guidance": "useful",
    "wrong_shallow": "wrong",
    "off_rubric": "off_rubric",
}

# Reopen Stage 2: the recruiter-store signal-kind for a per-principle
# Designer correction. Mirrors the constant in
# ``shared/runtime_state/recruiter_store.py`` (kept as a string literal
# resolved lazily inside the helper so this module — imported on the
# Designer pipeline's hot path — stays free of an import-time dependency
# on the global recruiter store).
_RECRUITER_SIGNAL_PRINCIPLE_FEEDBACK = "principle_feedback"


def _record_recruiter_taste_signal_fail_soft(
    *,
    runtime_state_store: "RuntimeStateStore",
    signal_kind: str,
    domain: str,
    dedup_key: str,
    payload: dict,
    source_brief_id: str | None = None,
    confidence: float = 0.5,
) -> None:
    """Mirror a brief-scoped correction into the global recruiter store.

    Reopen Stage 2's committed-intent double-write (shared by every
    Designer double-write site, and structurally identical to the
    market-intel archetype-preference write). Three steps:

      1. ``record_write_intention`` on the per-state-dir DB — MUST
         succeed. If even the intention can't be committed, that's a real
         fault on the recruiter's primary substrate; let it raise.
      2. fail-soft ``recruiter_store.record_taste_signal`` — the global
         store may be momentarily unavailable; a failure here leaves the
         intention with ``completed_at IS NULL`` for the backfill to
         replay, rather than failing the recruiter's action.
      3. ``mark_write_intention_complete`` on success.

    The ``recruiter_id`` is resolved through the
    :func:`shared.recruiter_context.get_current_recruiter_id` seam (NOT a
    hardcoded ``1``) so Phase-2 auth swaps one function and every write
    site follows.
    """

    intention_id = runtime_state_store.record_write_intention(
        signal_kind=signal_kind,
        domain=domain,
        dedup_key=dedup_key,
        payload=payload,
        source_brief_id=source_brief_id,
        confidence=confidence,
    )

    try:
        from shared.output_paths import resolve_recruiter_db_path
        from shared.recruiter_context import get_current_recruiter_id
        from shared.runtime_state.recruiter_store import RecruiterStore

        recruiter_id = get_current_recruiter_id()
        recruiter_store = RecruiterStore(resolve_recruiter_db_path())
        recruiter_store.record_taste_signal(
            recruiter_id,
            signal_kind=signal_kind,
            domain=domain,
            payload=payload,
            source_brief_id=source_brief_id,
            confidence=confidence,
        )
    except Exception:  # noqa: BLE001 — fail-soft; intention is the recovery path
        return

    runtime_state_store.mark_write_intention_complete(intention_id)


def _build_langfuse_dataset_row_for_principle_feedback(
    *,
    brief_id: str,
    identity_key: str,
    principle_name: str,
    marker: str,
    note: str,
    marked_at: str,
    judgment_accuracy_value: str,
) -> dict:
    """Build the typed Langfuse dataset row for a Designer per-principle marker.

    Phase 2 of Langfuse adoption. Mirrors the schema pinned in the
    plan body for ``shared.runtime_state.calibration``'s sibling
    export, but at per-principle grain — ``expected_output`` carries
    the Designer-specific marker enum (``useful_guidance`` /
    ``wrong_shallow`` / ``off_rubric``) and ``metadata.principle_name``
    identifies which rubric principle the recruiter marked.

    Pure builder; doesn't push to Langfuse. The caller drives the API
    with the singleton's no-op-when-degraded posture.
    """

    return {
        "input": {
            "brief_id": brief_id,
            "candidate_summary": (
                f"identity_key={identity_key}; principle={principle_name}"
            ),
            "capability_areas": [],  # Designer eval is rubric-driven, not capability-area-driven.
            "depth_distinction": {},  # Designer doesn't surface depth_distinction at the per-principle layer.
            "source": "designer",
        },
        "expected_output": {
            # Per-principle Designer enum (different from the unified
            # judgment_accuracy enum exported by calibration.py — kept
            # distinct so prompt-regression runners can score against
            # the original Designer-specific feedback).
            "judgment_accuracy": marker,
            "full_decision_rationale": note or "",
            "recruiter_marker_set_at": marked_at,
        },
        "metadata": {
            "identity_key": identity_key,
            "principle_name": principle_name,
            # The unified five-value enum the calibration aggregator
            # reads. Surfacing it on the per-principle dataset row
            # lets cross-module regression runners join on the
            # canonical column.
            "unified_judgment_accuracy": judgment_accuracy_value,
            "confidence_at_eval": 0.0,  # Designer markers don't carry per-eval confidence.
            "cascade_route_hit": None,
        },
    }


def record_designer_principle_feedback(
    *,
    runtime_state_store: "RuntimeStateStore",
    principle_feedback_store: PrincipleFeedbackStore,
    source: str,
    brief_id: str,
    identity_key: str,
    principle_name: str,
    marker: str,
    note: str = "",
    langfuse_dataset_name: str = "designer-rubric-feedback",
) -> PrincipleFeedbackMarker:
    """Record a Designer per-principle feedback marker on both substrates.

    Slice 3.6 of ``plans/multi-agent-execution-plan.md``. Same column,
    additional detail:

    1. The per-principle marker (with original Designer-enum nuance and
       free-text note) is appended to :class:`PrincipleFeedbackStore`'s
       per-state-dir SQLite. Designer-specific surfaces (Slice 9
       reflection polish, the workspace recruiter-annotation panel)
       continue to read from there.
    2. The candidate's ``judgment_accuracy`` column on the canonical
       ``runtime_state.sqlite3`` row is set to the unified five-value
       enum (per :data:`DESIGNER_MARKER_TO_JUDGMENT_ACCURACY`).
    3. The per-principle detail (principle_name, original Designer
       marker, note, marked_at) is appended to ``terminal_payload_json``
       under ``principle_markers`` so the per-principle nuance is still
       readable downstream of the unified column.
    4. Phase 2 of Langfuse adoption: the per-principle marker is ALSO
       appended to a Langfuse dataset (``langfuse_dataset_name``,
       defaulting to ``designer-rubric-feedback``) so prompt-regression
       runners can score against the recruiter's Designer-specific
       judgment. Defensive null-stub posture: when the Langfuse
       client is null / disabled / network-degraded, the dataset push
       is a no-op and the two-store write completes unchanged.

    The two-store write is sequenced — :class:`PrincipleFeedbackStore`
    first (so its validation gate fires before any canonical-store
    write), then the canonical store, then the optional Langfuse
    push. A failure on the canonical write does NOT roll back the
    per-state-dir record; that's intentional: the per-state-dir log
    is the Designer surface's primary substrate, and the canonical
    mirror is the calibration aggregator's read path. A divergence
    is recoverable by replaying the per-state-dir log into the
    canonical store; the inverse is not. The Langfuse push is the
    LAST step and never affects either canonical store — fail-soft
    by design.

    Raises ``ValueError`` for unknown markers (via
    :class:`PrincipleFeedbackStore`) or unknown candidates (via the
    canonical store). The caller is expected to translate those into
    HTTP 422 / 404 at the wire layer. Langfuse failures NEVER raise
    into the caller.
    """

    feedback_marker = principle_feedback_store.record(
        candidate_identity_key=identity_key,
        principle_name=principle_name,
        marker=marker,
        note=note,
    )

    judgment_accuracy_value = DESIGNER_MARKER_TO_JUDGMENT_ACCURACY[marker]

    runtime_state_store.record_candidate_principle_marker(
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        judgment_accuracy=judgment_accuracy_value,
        principle_marker={
            "principle_name": principle_name,
            "marker": marker,
            "note": note,
            "marked_at": feedback_marker.marked_at,
        },
    )

    # Reopen Stage 2: third substrate — the GLOBAL recruiter taste-signal
    # store. The recruiter, not the brief, is Cloris's durable entity, so
    # a per-principle correction is also a cross-brief calibration signal
    # for whichever recruiter is acting. The write is fail-soft (the
    # recruiter store can be momentarily locked / read-only without
    # failing the recruiter's primary action), but a silently-dropped
    # signal is unrecoverable from the brief side — so it rides a
    # committed write-intentions ledger on the per-state-dir DB:
    #   (1) record intention (must succeed) → (2) fail-soft recruiter
    #   write → (3) mark complete. An idempotent backfill replays any
    #   intention left incomplete by a transient recruiter-store failure.
    _record_recruiter_taste_signal_fail_soft(
        runtime_state_store=runtime_state_store,
        signal_kind=_RECRUITER_SIGNAL_PRINCIPLE_FEEDBACK,
        domain="designer",
        source_brief_id=brief_id,
        payload={
            "principle_name": principle_name,
            "marker": marker,
            "judgment_accuracy": judgment_accuracy_value,
            "note": note,
            "identity_key": identity_key,
            "marked_at": feedback_marker.marked_at,
        },
        dedup_key=(
            f"designer:principle_feedback:{source}:{brief_id}:{identity_key}:"
            f"{principle_name}:{marker}:{feedback_marker.marked_at}"
        ),
    )

    # Phase 2: optional Langfuse dataset append. Same null-stub
    # posture as the rest of the observability layer — no-op when
    # the client is null / disabled / network-degraded. Failures
    # NEVER raise into the caller; the two canonical stores above
    # are the authoritative substrate.
    try:
        from shared.observability import is_active
        from shared.observability.langfuse_client import get_client

        if is_active():
            row = _build_langfuse_dataset_row_for_principle_feedback(
                brief_id=brief_id,
                identity_key=identity_key,
                principle_name=principle_name,
                marker=marker,
                note=note,
                marked_at=feedback_marker.marked_at,
                judgment_accuracy_value=judgment_accuracy_value,
            )
            client = get_client()
            # The Langfuse SDK's create_dataset_item shape varies
            # across major versions (v2: client.create_dataset_item;
            # v3: client.api.dataset_items.create). Both ride the
            # singleton's sticky-degrade — any per-call exception
            # flips the singleton to no-op for the rest of the
            # process.
            inner = getattr(client, "_inner", None)
            if inner is not None:
                if hasattr(inner, "create_dataset_item"):
                    inner.create_dataset_item(
                        dataset_name=langfuse_dataset_name,
                        input=row["input"],
                        expected_output=row["expected_output"],
                        metadata=row["metadata"],
                    )
                elif hasattr(inner, "api") and hasattr(
                    getattr(inner, "api", None), "dataset_items"
                ):
                    inner.api.dataset_items.create(
                        dataset_name=langfuse_dataset_name,
                        input=row["input"],
                        expected_output=row["expected_output"],
                        metadata=row["metadata"],
                    )
    except Exception:  # noqa: BLE001 — Langfuse path is fail-soft
        pass

    return feedback_marker
