"""Cross-module identity resolution (Phase F Slice F3).

Resolves duplicate candidates across `output/state/linkedin/<key>/`
and `output/state/github/<key>/` into canonical persons stored in the
global identity DB at `output/state/_identity/identity.sqlite3`.

The split between brief-AGNOSTIC ``persons`` and brief-scoped
``brief_persons`` membership is what makes cross-brief calibration
("we already worked this person on a different brief") possible
without forcing a duplicate human row per brief. F6 inherits the
``brief_persons_with_evidence`` read-model and never sees the
cross-DB fan-out.

Match logic is conservative — false-positive (different humans
merged) is worse than false-negative (same human kept apart). The
recruiter can manually merge ambiguous cases via F6's review-merge
affordance, which writes a `recruiter_locked=1` link that
auto-resolution refuses to overwrite.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from shared.identity_resolution import (
    normalize_person_name,
    normalize_public_linkedin_handle,
)
from shared.output_paths import enumerate_state_dirs

log = logging.getLogger(__name__)
from shared.runtime_state.identity_store import (
    IdentityStore,
    LINK_KIND_AUTO_MEDIUM,
    LINK_KIND_AUTO_STRONG,
    LINK_KIND_MANUAL,
)


# ---------------------------------------------------------------------------
# Public dataclasses (read-model shape consumed by F6).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateLink:
    source: str
    state_key: str
    candidate_id: int
    link_kind: str
    match_signal: dict
    recruiter_locked: bool


@dataclass(frozen=True)
class PersonWithEvidence:
    person_id: int
    canonical_name: str
    canonical_handle: str
    sources: tuple[CandidateLink, ...]


@dataclass(frozen=True)
class PendingDecision:
    """One unresolved pending_merge_decisions row enriched with person evidence.

    Phase G G2: the recruiter-facing read-model. The API hands this to
    the frontend; the UI renders side-by-side person evidence and the
    Cloris-voice ``signal_summary`` prose. ``confidence`` is intentionally
    NOT exposed to the wire — the L14-class hygiene rule keeps raw floats
    out of editorial surfaces; the editorial summary carries the meaning.
    """

    decision_id: int
    person_a: PersonWithEvidence
    person_b: PersonWithEvidence
    signal_summary: str
    created_at: str


@dataclass
class ResolveResult:
    persons_total: int = 0
    candidates_linked: int = 0
    pending_merges: int = 0
    auto_strong: int = 0
    auto_medium: int = 0


# ---------------------------------------------------------------------------
# Internal candidate-row representation collected from per-state-dir DBs.
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    source: str
    state_key: str
    candidate_id: int
    display_name: str
    profile_url: str
    terminal_payload: dict
    # Derived signals used for matching. Computed once; reused across passes.
    linkedin_handle: str = ""
    real_name: str = ""
    normalized_name: str = ""
    github_username: str = ""
    # OSS Maintainers Slice 8: provenance label for `linkedin_handle`
    # when the source is `github`. One of "blog" / "bio" / "readme" /
    # "" (no LinkedIn discovered). Used by downstream consumers to
    # band confidence in the cross-source link without inventing a
    # numeric confidence channel. Empty for non-github candidates
    # (the linkedin source's `linkedin_handle` derives from the
    # candidate's own profile URL, which carries no provenance band).
    linkedin_url_source: str = ""
    # B.3 Researcher cross-module identity bridge fields.
    # ``orcid`` is the high-confidence join key (~30-40% coverage of
    # ML researchers per researcher-module-spec Opinion 3); when present
    # it deterministically pins the researcher↔LinkedIn match because
    # ORCID is a single-person-claimed permanent identifier. ``affiliation``
    # is the medium-confidence corroborating signal for name+affiliation
    # matches when ORCID is absent; populated from the researcher's
    # current institution. Both default to empty for non-researcher
    # candidates.
    orcid: str = ""
    affiliation: str = ""
    # C.8 Designer cross-module identity fields.
    # ``portfolio_urls`` collects the designer's Behance profile URL plus
    # any social links (personal site, Dribbble, etc.) extracted from the
    # Behance user response. Used by the portfolio-URL overlap check in
    # ``_has_corroborating_signal()`` to merge designer ↔ LinkedIn
    # candidates when both carry the same portfolio URL.
    portfolio_urls: list[str] = field(default_factory=list)

    @property
    def primary_key(self) -> tuple[str, str, int]:
        return (self.source, self.state_key, self.candidate_id)


# ---------------------------------------------------------------------------
# Signal extraction.
# ---------------------------------------------------------------------------


def _extract_signals(candidate: _Candidate) -> None:
    """Populate ``linkedin_handle``, ``real_name``, ``normalized_name``,
    ``github_username`` from whatever the source-specific terminal
    payload exposes. Mutates ``candidate`` in place.
    """

    payload = candidate.terminal_payload or {}
    cr = payload.get("candidate_record")
    if not isinstance(cr, dict):
        cr = {}

    if candidate.source == "linkedin":
        candidate.linkedin_handle = normalize_public_linkedin_handle(
            candidate.profile_url
        )
        candidate.real_name = candidate.display_name
    elif candidate.source == "github":
        # GitHub's `display_name` is the username, not the human name.
        # The real name (when known) lives in candidate_record.user.name.
        candidate.github_username = candidate.display_name
        user_obj = cr.get("user") if isinstance(cr, dict) else None
        if isinstance(user_obj, dict):
            candidate.real_name = str(user_obj.get("name") or "").strip()
        if not candidate.real_name:
            candidate.real_name = candidate.display_name
        # The cross-source bridge: LinkedIn URL captured via the GitHub
        # reconciliation pipeline (`shared.reconciliation_schemas` /
        # `linkedin.recruiter_identity_resolver`) AND, since OSS
        # Maintainers Slice 8, via bio + profile-README extraction in
        # `shared.contact_discovery.merge_profile_contact`. Provenance
        # ("blog" / "bio" / "readme") rides on
        # ``contact.linkedin_url_source`` so downstream consumers can
        # band confidence — bio/readme matches require a full URL
        # match (no false positive on bare keywords), but a recruiter-
        # set blog field still wins when both are present.
        contact = cr.get("contact") if isinstance(cr, dict) else None
        linkedin_url_hint = ""
        url_source = ""
        if isinstance(contact, dict):
            linkedin_url_hint = str(contact.get("linkedin_url") or "").strip()
            url_source = str(contact.get("linkedin_url_source") or "").strip()
        if not linkedin_url_hint:
            for key in ("linkedin_url", "linkedin_url_hint", "matched_profile_url"):
                value = cr.get(key) if isinstance(cr, dict) else None
                if value:
                    linkedin_url_hint = str(value).strip()
                    break
        candidate.linkedin_handle = normalize_public_linkedin_handle(linkedin_url_hint)
        candidate.linkedin_url_source = url_source if candidate.linkedin_handle else ""
    elif candidate.source == "researcher":
        # Slice B.3 (Multi-Agent Production Plan) — researcher source
        # extraction. ``display_name`` is the human name (the researcher
        # candidate carries it directly); ORCID + current affiliation
        # come from the terminal_payload's `candidate_record` per the
        # researcher pipeline's writer at
        # ``shared/runtime_state/researcher.py``. The full matching
        # logic (ORCID-anchored auto-strong, name+affiliation-anchored
        # auto-medium) lives in
        # ``shared/cross_module_identity/researcher_to_linkedin.py``;
        # this branch just extracts the signals so the existing
        # ``_group_by_handle`` + ``_group_by_name_with_corroboration``
        # passes can fold researcher candidates into the resolution
        # graph alongside LinkedIn / GitHub.
        candidate.real_name = candidate.display_name
        if isinstance(cr, dict):
            orcid_raw = cr.get("orcid")
            if isinstance(orcid_raw, str) and orcid_raw.strip():
                candidate.orcid = orcid_raw.strip()
            affiliation_raw = cr.get("current_affiliation") or cr.get(
                "affiliation"
            )
            if isinstance(affiliation_raw, str) and affiliation_raw.strip():
                candidate.affiliation = affiliation_raw.strip()
    elif candidate.source == "designer":
        candidate.real_name = candidate.display_name
        if isinstance(cr, dict):
            snippet = cr.get("snippet") or {}
            if isinstance(snippet, dict):
                profile_url = snippet.get("profile_url", "")
                candidate.portfolio_urls = [profile_url] if profile_url else []
                raw_links = snippet.get("social_links") or []
                if isinstance(raw_links, list):
                    for pair in raw_links:
                        if isinstance(pair, list) and len(pair) == 2:
                            url = str(pair[1])
                            if url:
                                candidate.portfolio_urls.append(url)
    else:
        candidate.real_name = candidate.display_name

    candidate.normalized_name = normalize_person_name(candidate.real_name)


# ---------------------------------------------------------------------------
# Per-state-dir candidate iteration.
# ---------------------------------------------------------------------------


def _iter_candidates_for_brief(
    brief_id: str,
    state_root: Path | None,
) -> Iterable[_Candidate]:
    """Walk every per-state-dir DB and yield candidates for ``brief_id``."""

    for source, state_dir in enumerate_state_dirs(state_root=state_root):
        db_path = state_dir / "runtime_state.sqlite3"
        if not db_path.exists():
            continue
        state_key = state_dir.name
        # Read-only connection — F3 contract is that the resolver
        # never mutates per-state-dir DBs.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            try:
                rows = conn.execute(
                    """
                    SELECT id, source, display_name, profile_url, terminal_payload_json
                    FROM candidates
                    WHERE brief_id = ?
                    """,
                    (brief_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                # State dir predates the candidates table or is otherwise
                # unreadable; skip silently rather than fail the whole pass.
                continue
        finally:
            conn.close()

        for row in rows:
            try:
                terminal_payload = json.loads(row["terminal_payload_json"] or "{}")
            except json.JSONDecodeError:
                terminal_payload = {}
            candidate = _Candidate(
                source=source,
                state_key=state_key,
                candidate_id=int(row["id"]),
                display_name=str(row["display_name"] or ""),
                profile_url=str(row["profile_url"] or ""),
                terminal_payload=terminal_payload,
            )
            _extract_signals(candidate)
            yield candidate


# ---------------------------------------------------------------------------
# Match logic.
# ---------------------------------------------------------------------------


def _group_by_handle(
    candidates: list[_Candidate],
) -> tuple[
    list[list[_Candidate]],
    list[_Candidate],
    list[_Candidate],
    dict[str, list[_Candidate]],
]:
    """Pass 1: group by exact non-empty LinkedIn handle.

    P3.8: readme-sourced GitHub matches (``linkedin_url_source ==
    "readme"``) are excluded from this auto_strong grouping entirely,
    even when their handle matches another candidate's. Readme is the
    lowest-confidence LinkedIn-discovery provenance (a profile README
    can reference someone else's profile), so a readme handle match must
    not silently merge — the caller routes it to pending review instead.
    Blog/bio-sourced candidates are unaffected and keep today's behavior.

    Returns ``(groups, remainder, readme_candidates, handle_owners)``:
    - ``groups``/``remainder``: as before, computed over all NON-readme
      candidates only.
    - ``readme_candidates``: readme-sourced candidates with a non-empty
      handle, held out of ``groups``/``remainder`` for the caller to
      route.
    - ``handle_owners``: the raw non-readme handle -> candidates index
      (includes singleton entries that ended up in ``remainder``), so
      the caller can detect whether a readme candidate's handle
      collides with a real owner elsewhere in the graph.
    """

    by_handle: dict[str, list[_Candidate]] = {}
    no_handle: list[_Candidate] = []
    readme_candidates: list[_Candidate] = []
    for c in candidates:
        if c.linkedin_handle and c.linkedin_url_source == "readme":
            readme_candidates.append(c)
        elif c.linkedin_handle:
            by_handle.setdefault(c.linkedin_handle, []).append(c)
        else:
            no_handle.append(c)

    groups: list[list[_Candidate]] = []
    remainder: list[_Candidate] = list(no_handle)
    for handle, group in by_handle.items():
        if len(group) >= 2:
            groups.append(group)
        else:
            remainder.extend(group)
    return groups, remainder, readme_candidates, by_handle


def _has_corroborating_signal(group: list[_Candidate]) -> tuple[bool, list[str]]:
    """Pass 2 helper: do these same-name candidates carry corroborating
    signal beyond the name match? Returns ``(has_signal, evidence_list)``.

    Corroborating signals (any one suffices):
    - LinkedIn handle slug == GitHub username for some pair in the group.
    - GitHub candidate carries a non-empty linkedin_url_hint (already
      promoted to ``linkedin_handle`` by ``_extract_signals``) — but
      Pass 1 would have caught those, so this only fires if the handle
      didn't match anyone in Pass 1.
    """

    evidence: list[str] = []
    linkedin_handles = {c.linkedin_handle for c in group if c.linkedin_handle}
    github_usernames = {c.github_username for c in group if c.github_username}
    overlap = linkedin_handles & github_usernames
    if overlap:
        evidence.append(f"LinkedIn handle matches GitHub username: {sorted(overlap)}")
    sources_in_group = {c.source for c in group}
    if len(sources_in_group) >= 2 and any(
        c.linkedin_handle for c in group if c.source == "github"
    ):
        evidence.append("GitHub candidate carries a LinkedIn URL hint")
    # C.8: Portfolio URL overlap (Designer ↔ any other source).
    if any(c.portfolio_urls for c in group):
        from shared.cross_module_identity.designer_to_linkedin import (
            normalize_portfolio_url,
        )

        designer_urls: set[str] = set()
        other_urls: set[str] = set()
        for c in group:
            normalized = {normalize_portfolio_url(u) for u in c.portfolio_urls if u}
            normalized.discard("")
            if c.source == "designer":
                designer_urls |= normalized
            else:
                other_urls |= normalized
        url_overlap = designer_urls & other_urls
        if url_overlap:
            evidence.append(f"Portfolio URL match: {sorted(url_overlap)}")
    return (bool(evidence), evidence)


def _group_by_name_with_corroboration(
    remainder: list[_Candidate],
) -> tuple[
    list[tuple[list[_Candidate], list[str]]],
    list[tuple[list[_Candidate], list[str]]],
    list[_Candidate],
]:
    """Pass 2 + 3.

    Returns:
      - ``confirmed_groups``: name match + corroborating signal → auto_medium.
      - ``ambiguous_groups``: name match only → pending_merge_decisions.
      - ``singletons``: no other candidate with the same normalized name.
    """

    by_name: dict[str, list[_Candidate]] = {}
    no_name: list[_Candidate] = []
    for c in remainder:
        if c.normalized_name:
            by_name.setdefault(c.normalized_name, []).append(c)
        else:
            no_name.append(c)

    confirmed: list[tuple[list[_Candidate], list[str]]] = []
    ambiguous: list[tuple[list[_Candidate], list[str]]] = []
    singletons: list[_Candidate] = list(no_name)
    for _, group in by_name.items():
        if len(group) < 2:
            singletons.extend(group)
            continue
        has_signal, evidence = _has_corroborating_signal(group)
        if has_signal:
            confirmed.append((group, evidence))
        else:
            ambiguous.append((group, [f"Identical normalized name: {group[0].normalized_name!r}"]))

    return confirmed, ambiguous, singletons


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------


def _now_iso(now: Callable[[], datetime] | None) -> str:
    fn = now or (lambda: datetime.now(timezone.utc))
    return fn().isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_person(
    conn: sqlite3.Connection,
    *,
    canonical_name: str,
    canonical_handle: str,
    now_iso: str,
) -> int:
    """Look up an existing person by canonical_handle (when non-empty);
    otherwise insert a fresh row. Updates ``last_seen_at``.
    """

    if canonical_handle:
        row = conn.execute(
            "SELECT id FROM persons WHERE canonical_handle = ?",
            (canonical_handle,),
        ).fetchone()
        if row is not None:
            person_id = int(row["id"])
            conn.execute(
                "UPDATE persons SET last_seen_at = ?, "
                "canonical_name = COALESCE(NULLIF(canonical_name, ''), ?) "
                "WHERE id = ?",
                (now_iso, canonical_name, person_id),
            )
            return person_id

    cursor = conn.execute(
        """
        INSERT INTO persons(canonical_name, canonical_handle, created_at, last_seen_at)
        VALUES (?, ?, ?, ?)
        """,
        (canonical_name, canonical_handle, now_iso, now_iso),
    )
    return int(cursor.lastrowid)


def _existing_candidate_link(
    conn: sqlite3.Connection,
    candidate: _Candidate,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT person_id, link_kind, recruiter_locked
        FROM candidate_persons
        WHERE source = ? AND state_key = ? AND candidate_id = ?
        """,
        (candidate.source, candidate.state_key, candidate.candidate_id),
    ).fetchone()


def _touch_existing_person(
    conn: sqlite3.Connection,
    *,
    person_id: int,
    canonical_name: str,
    now_iso: str,
) -> int:
    conn.execute(
        "UPDATE persons SET last_seen_at = ?, "
        "canonical_name = COALESCE(NULLIF(canonical_name, ''), ?) "
        "WHERE id = ?",
        (now_iso, canonical_name, person_id),
    )
    return person_id


def _ensure_candidate_person(
    conn: sqlite3.Connection,
    *,
    candidate: _Candidate,
    canonical_name: str,
    canonical_handle: str,
    now_iso: str,
) -> int:
    """Return the stable person for one candidate.

    No-handle candidates cannot be recovered through canonical_handle on
    later resolver passes, so candidate_persons is their idempotency key.
    """

    if not canonical_handle:
        existing = _existing_candidate_link(conn, candidate)
        if existing is not None:
            return _touch_existing_person(
                conn,
                person_id=int(existing["person_id"]),
                canonical_name=canonical_name,
                now_iso=now_iso,
            )
    return _ensure_person(
        conn,
        canonical_name=canonical_name,
        canonical_handle=canonical_handle,
        now_iso=now_iso,
    )


def _ensure_group_person(
    conn: sqlite3.Connection,
    *,
    group: list[_Candidate],
    canonical_name: str,
    canonical_handle: str,
    now_iso: str,
) -> int:
    if canonical_handle:
        return _ensure_person(
            conn,
            canonical_name=canonical_name,
            canonical_handle=canonical_handle,
            now_iso=now_iso,
        )

    for candidate in group:
        existing = _existing_candidate_link(conn, candidate)
        if existing is None:
            continue
        if int(existing["recruiter_locked"]) == 1:
            continue
        if existing["link_kind"] == LINK_KIND_MANUAL:
            continue
        return _touch_existing_person(
            conn,
            person_id=int(existing["person_id"]),
            canonical_name=canonical_name,
            now_iso=now_iso,
        )

    return _ensure_person(
        conn,
        canonical_name=canonical_name,
        canonical_handle="",
        now_iso=now_iso,
    )


def _ensure_brief_membership(
    conn: sqlite3.Connection,
    *,
    brief_id: str,
    person_id: int,
    now_iso: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO brief_persons(brief_id, person_id, first_seen_at)
        VALUES (?, ?, ?)
        """,
        (brief_id, person_id, now_iso),
    )


def _upsert_link(
    conn: sqlite3.Connection,
    *,
    candidate: _Candidate,
    person_id: int,
    brief_id: str,
    link_kind: str,
    match_signal: dict,
    now_iso: str,
) -> bool:
    """Insert or update a candidate_persons row.

    Returns True if a row was written or updated; False if a manual link
    already exists and we refused to stomp it.
    """

    existing = _existing_candidate_link(conn, candidate)

    if existing is not None:
        if int(existing["recruiter_locked"]) == 1:
            return False
        if existing["link_kind"] == LINK_KIND_MANUAL:
            return False
        conn.execute(
            """
            UPDATE candidate_persons
            SET person_id = ?, link_kind = ?, match_signal_json = ?,
                brief_id = ?, updated_at = ?
            WHERE source = ? AND state_key = ? AND candidate_id = ?
            """,
            (
                person_id,
                link_kind,
                json.dumps(match_signal, sort_keys=True),
                brief_id,
                now_iso,
                candidate.source,
                candidate.state_key,
                candidate.candidate_id,
            ),
        )
        return True

    conn.execute(
        """
        INSERT INTO candidate_persons(
            source, state_key, candidate_id, person_id, brief_id,
            link_kind, match_signal_json, recruiter_locked,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        """,
        (
            candidate.source,
            candidate.state_key,
            candidate.candidate_id,
            person_id,
            brief_id,
            link_kind,
            json.dumps(match_signal, sort_keys=True),
            now_iso,
            now_iso,
        ),
    )
    return True


def _record_pending_merge(
    conn: sqlite3.Connection,
    *,
    brief_id: str,
    person_a: int,
    person_b: int,
    confidence: float,
    evidence: list[str],
    now_iso: str,
) -> bool:
    """Insert a pending_merge_decisions row.

    Skips when an undecided row already exists for this pair (any
    ordering) OR when a decided row exists (decision is terminal).
    Returns True if a new row was inserted.
    """

    a, b = sorted((person_a, person_b))
    existing = conn.execute(
        """
        SELECT id, recruiter_decision FROM pending_merge_decisions
        WHERE brief_id = ?
          AND ((person_a = ? AND person_b = ?) OR (person_a = ? AND person_b = ?))
        """,
        (brief_id, a, b, b, a),
    ).fetchone()
    if existing is not None:
        return False
    conn.execute(
        """
        INSERT INTO pending_merge_decisions(
            brief_id, person_a, person_b, confidence, evidence_json,
            recruiter_decision, decided_at, created_at
        )
        VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
        """,
        (
            brief_id,
            a,
            b,
            confidence,
            json.dumps({"reasons": evidence}, sort_keys=True),
            now_iso,
        ),
    )
    return True


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def resolve_persons_for_brief(
    brief_id: str,
    *,
    identity_db_path: Path | None = None,
    state_root: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> ResolveResult:
    """Walk per-state-dir DBs read-only and write canonical persons +
    candidate links to the global identity DB.

    Idempotent: re-runs are no-ops on already-linked candidates whose
    signals haven't changed; ``recruiter_locked=1`` rows are never
    auto-overwritten; pending merges with a recorded decision are
    terminal.
    """

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    candidates = list(_iter_candidates_for_brief(brief_id, state_root))
    if not candidates:
        return ResolveResult()

    handle_groups, remainder_after_handle, readme_candidates, handle_owners = (
        _group_by_handle(candidates)
    )

    # P3.8: a readme-sourced candidate whose handle doesn't collide with
    # any other candidate has nothing to auto-merge with — fold it into
    # the ordinary remainder so it still participates in name-based
    # matching (Pass 2/3/4) exactly like any other handle-unique
    # candidate. Only candidates whose handle DOES collide with a real
    # (non-readme) owner are held out for the pending-review path below.
    readme_colliding: list[_Candidate] = []
    for rc in readme_candidates:
        if handle_owners.get(rc.linkedin_handle):
            readme_colliding.append(rc)
        else:
            remainder_after_handle.append(rc)

    confirmed_name_groups, ambiguous_name_groups, singletons = (
        _group_by_name_with_corroboration(remainder_after_handle)
    )

    result = ResolveResult()
    now_iso = _now_iso(now)
    store = IdentityStore(identity_db_path)

    with store.connect() as conn:
        # Tracks every candidate's resolved person_id across all passes
        # so the P3.8 readme-collision pass (after Pass 3+4 below) can
        # look up the person_id of whatever a readme candidate's handle
        # collided with, regardless of which pass resolved it.
        person_id_by_candidate: dict[tuple[str, str, int], int] = {}

        # Pass 1: handle-matched groups → auto_strong, ONE person per group.
        for group in handle_groups:
            handle = group[0].linkedin_handle
            canonical_name = next(
                (c.real_name for c in group if c.real_name), ""
            )
            person_id = _ensure_person(
                conn,
                canonical_name=canonical_name,
                canonical_handle=handle,
                now_iso=now_iso,
            )
            _ensure_brief_membership(
                conn, brief_id=brief_id, person_id=person_id, now_iso=now_iso
            )
            for c in group:
                wrote = _upsert_link(
                    conn,
                    candidate=c,
                    person_id=person_id,
                    brief_id=brief_id,
                    link_kind=LINK_KIND_AUTO_STRONG,
                    match_signal={"kind": "linkedin_handle", "handle": handle},
                    now_iso=now_iso,
                )
                if wrote:
                    result.candidates_linked += 1
                    result.auto_strong += 1
                person_id_by_candidate[c.primary_key] = person_id
            result.persons_total += 1

        # Pass 2: name + corroboration → auto_medium, ONE person per group.
        for group, evidence in confirmed_name_groups:
            canonical_name = group[0].real_name or group[0].normalized_name
            canonical_handle = next(
                (c.linkedin_handle for c in group if c.linkedin_handle), ""
            )
            person_id = _ensure_group_person(
                conn,
                group=group,
                canonical_name=canonical_name,
                canonical_handle=canonical_handle,
                now_iso=now_iso,
            )
            _ensure_brief_membership(
                conn, brief_id=brief_id, person_id=person_id, now_iso=now_iso
            )
            for c in group:
                wrote = _upsert_link(
                    conn,
                    candidate=c,
                    person_id=person_id,
                    brief_id=brief_id,
                    link_kind=LINK_KIND_AUTO_MEDIUM,
                    match_signal={
                        "kind": "name_with_corroboration",
                        "name": group[0].normalized_name,
                        "corroboration": evidence,
                    },
                    now_iso=now_iso,
                )
                if wrote:
                    result.candidates_linked += 1
                    result.auto_medium += 1
                person_id_by_candidate[c.primary_key] = person_id
            result.persons_total += 1

        # Pass 3 + 4: singletons + ambiguous name groups.
        # Each candidate becomes its own person (auto_strong link to a
        # unique person row), then for each ambiguous pair we record a
        # pending_merge_decisions row.

        # Materialize singletons first.
        for c in singletons:
            canonical_name = c.real_name or c.normalized_name
            canonical_handle = c.linkedin_handle
            person_id = _ensure_candidate_person(
                conn,
                candidate=c,
                canonical_name=canonical_name,
                canonical_handle=canonical_handle,
                now_iso=now_iso,
            )
            _ensure_brief_membership(
                conn, brief_id=brief_id, person_id=person_id, now_iso=now_iso
            )
            wrote = _upsert_link(
                conn,
                candidate=c,
                person_id=person_id,
                brief_id=brief_id,
                link_kind=LINK_KIND_AUTO_STRONG,
                match_signal={"kind": "singleton"},
                now_iso=now_iso,
            )
            if wrote:
                result.candidates_linked += 1
                result.auto_strong += 1
            result.persons_total += 1
            person_id_by_candidate[c.primary_key] = person_id

        # Then ambiguous groups: each candidate keeps its own person, but
        # we record pending_merge_decisions for each pair within the group
        # so F6 can surface a "merge?" affordance.
        for group, evidence in ambiguous_name_groups:
            group_person_ids: list[int] = []
            for c in group:
                # G2 fix: when a candidate already carries a
                # link, reuse its existing person_id when it cannot be
                # rediscovered by canonical_handle alone
                # rather than minting a fresh orphan. Without this, every
                # re-run after a no-handle resolution would create
                # duplicate persons + stale pending rows for the same pair.
                existing_link = _existing_candidate_link(conn, c)
                if existing_link is not None and int(existing_link["recruiter_locked"]) == 1:
                    person_id = int(existing_link["person_id"])
                    _ensure_brief_membership(
                        conn, brief_id=brief_id, person_id=person_id, now_iso=now_iso
                    )
                    group_person_ids.append(person_id)
                    person_id_by_candidate[c.primary_key] = person_id
                    continue

                canonical_name = c.real_name or c.normalized_name
                canonical_handle = c.linkedin_handle
                person_id = _ensure_candidate_person(
                    conn,
                    candidate=c,
                    canonical_name=canonical_name,
                    canonical_handle=canonical_handle,
                    now_iso=now_iso,
                )
                _ensure_brief_membership(
                    conn, brief_id=brief_id, person_id=person_id, now_iso=now_iso
                )
                wrote = _upsert_link(
                    conn,
                    candidate=c,
                    person_id=person_id,
                    brief_id=brief_id,
                    link_kind=LINK_KIND_AUTO_STRONG,
                    match_signal={
                        "kind": "name_only_ambiguous",
                        "name": c.normalized_name,
                    },
                    now_iso=now_iso,
                )
                if wrote:
                    result.candidates_linked += 1
                    result.auto_strong += 1
                result.persons_total += 1
                group_person_ids.append(person_id)
                person_id_by_candidate[c.primary_key] = person_id

            # G2 fix: skip pending-pair generation when all candidates in
            # the group already share one person_id (recruiter merged the
            # group). The dedup'd unique-id check below catches this.
            unique_person_ids = sorted(set(group_person_ids))
            for i in range(len(unique_person_ids)):
                for j in range(i + 1, len(unique_person_ids)):
                    a, b = unique_person_ids[i], unique_person_ids[j]
                    inserted = _record_pending_merge(
                        conn,
                        brief_id=brief_id,
                        person_a=a,
                        person_b=b,
                        confidence=0.55,
                        evidence=evidence,
                        now_iso=now_iso,
                    )
                    if inserted:
                        result.pending_merges += 1

        # P3.8: readme-sourced handle matches route to pending review,
        # never a silent auto_strong merge with the handle's real owner.
        # By this point every non-readme candidate (Pass 1/2/3/4) has a
        # resolved person_id in `person_id_by_candidate`, so we can look
        # up exactly who each readme candidate's handle collided with.
        for rc in readme_colliding:
            owners = handle_owners.get(rc.linkedin_handle, [])
            target_person_ids = sorted(
                {
                    person_id_by_candidate[o.primary_key]
                    for o in owners
                    if o.primary_key in person_id_by_candidate
                }
            )
            canonical_name = rc.real_name or rc.normalized_name
            person_id = _ensure_candidate_person(
                conn,
                candidate=rc,
                canonical_name=canonical_name,
                # Deliberately empty, not `rc.linkedin_handle`: that handle
                # is already claimed (as canonical_handle) by the colliding
                # owner's person. Minting this candidate's own person keyed
                # on the same handle would have `_ensure_person` silently
                # hand back the OWNER's person row — exactly the
                # auto_strong merge this pass exists to prevent.
                canonical_handle="",
                now_iso=now_iso,
            )
            _ensure_brief_membership(
                conn, brief_id=brief_id, person_id=person_id, now_iso=now_iso
            )
            wrote = _upsert_link(
                conn,
                candidate=rc,
                person_id=person_id,
                brief_id=brief_id,
                link_kind=LINK_KIND_AUTO_STRONG,
                match_signal={
                    "kind": "readme_handle_pending_review",
                    "handle": rc.linkedin_handle,
                },
                now_iso=now_iso,
            )
            if wrote:
                result.candidates_linked += 1
                result.auto_strong += 1
            result.persons_total += 1
            person_id_by_candidate[rc.primary_key] = person_id

            for target_id in target_person_ids:
                if target_id == person_id:
                    continue
                inserted = _record_pending_merge(
                    conn,
                    brief_id=brief_id,
                    person_a=person_id,
                    person_b=target_id,
                    confidence=0.6,
                    evidence=[
                        "LinkedIn URL discovered via GitHub profile README "
                        f"(handle {rc.linkedin_handle!r}); lower-confidence "
                        "provenance requires recruiter confirmation before "
                        "merging."
                    ],
                    now_iso=now_iso,
                )
                if inserted:
                    result.pending_merges += 1

    return result


def brief_persons_with_evidence(
    brief_id: str,
    *,
    identity_db_path: Path | None = None,
) -> list[PersonWithEvidence]:
    """Read-model: F6 calls this and renders.

    Hides the cross-DB fan-out — F6 never sees per-state-dir SQLite.
    """

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        person_rows = conn.execute(
            """
            SELECT p.id, p.canonical_name, p.canonical_handle
            FROM persons p
            JOIN brief_persons bp ON bp.person_id = p.id
            WHERE bp.brief_id = ?
              AND EXISTS (
                SELECT 1
                FROM candidate_persons cp
                WHERE cp.person_id = p.id
                  AND cp.brief_id = bp.brief_id
              )
            ORDER BY p.canonical_name
            """,
            (brief_id,),
        ).fetchall()

        results: list[PersonWithEvidence] = []
        for prow in person_rows:
            person_id = int(prow["id"])
            link_rows = conn.execute(
                """
                SELECT source, state_key, candidate_id, link_kind,
                       match_signal_json, recruiter_locked
                FROM candidate_persons
                WHERE person_id = ? AND brief_id = ?
                ORDER BY source, state_key, candidate_id
                """,
                (person_id, brief_id),
            ).fetchall()
            sources = tuple(
                CandidateLink(
                    source=str(row["source"]),
                    state_key=str(row["state_key"]),
                    candidate_id=int(row["candidate_id"]),
                    link_kind=str(row["link_kind"]),
                    match_signal=json.loads(row["match_signal_json"] or "{}"),
                    recruiter_locked=bool(int(row["recruiter_locked"])),
                )
                for row in link_rows
            )
            results.append(
                PersonWithEvidence(
                    person_id=person_id,
                    canonical_name=str(prow["canonical_name"] or ""),
                    canonical_handle=str(prow["canonical_handle"] or ""),
                    sources=sources,
                )
            )
        return results


def brief_person_count(
    brief_id: str,
    *,
    identity_db_path: Path | None = None,
) -> int:
    """Count brief-linked persons using the same membership semantics as
    ``brief_persons_with_evidence`` without materializing every source link.
    """

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS person_count
            FROM (
                SELECT DISTINCT bp.person_id
                FROM brief_persons bp
                JOIN candidate_persons cp
                  ON cp.person_id = bp.person_id
                 AND cp.brief_id = bp.brief_id
                JOIN persons p ON p.id = bp.person_id
                WHERE bp.brief_id = ?
            ) linked_persons
            """,
            (brief_id,),
        ).fetchone()
    return int(row["person_count"] if row is not None else 0)


def _person_with_evidence_from_id(
    conn: sqlite3.Connection, *, person_id: int, brief_id: str
) -> PersonWithEvidence | None:
    """Build a PersonWithEvidence from an open connection.

    G2 internal helper. Returns None if the person row was deleted (a
    completed merge can drop person_b after re-pointing links — pending
    decisions referencing the dropped row become stale).
    """

    prow = conn.execute(
        "SELECT id, canonical_name, canonical_handle FROM persons WHERE id = ?",
        (person_id,),
    ).fetchone()
    if prow is None:
        return None
    link_rows = conn.execute(
        """
        SELECT source, state_key, candidate_id, link_kind,
               match_signal_json, recruiter_locked
        FROM candidate_persons
        WHERE person_id = ? AND brief_id = ?
        ORDER BY source, state_key, candidate_id
        """,
        (person_id, brief_id),
    ).fetchall()
    sources = tuple(
        CandidateLink(
            source=str(row["source"]),
            state_key=str(row["state_key"]),
            candidate_id=int(row["candidate_id"]),
            link_kind=str(row["link_kind"]),
            match_signal=json.loads(row["match_signal_json"] or "{}"),
            recruiter_locked=bool(int(row["recruiter_locked"])),
        )
        for row in link_rows
    )
    return PersonWithEvidence(
        person_id=int(prow["id"]),
        canonical_name=str(prow["canonical_name"] or ""),
        canonical_handle=str(prow["canonical_handle"] or ""),
        sources=sources,
    )


def _describe_pending_signal(evidence: dict) -> str:
    """Editorial prose for a pending_merge_decisions row.

    Pending rows carry ``{"reasons": [...]}`` evidence written by F3's
    resolver — the only currently-emitted reason kind is name-only
    ambiguous matches. Returns Cloris-voice prose; fallback when reasons
    are missing.
    """

    reasons = evidence.get("reasons") if isinstance(evidence, dict) else None
    if isinstance(reasons, list) and reasons:
        joined = ", ".join(str(r) for r in reasons if r)
        if "name_only" in joined or any(
            "name" in str(r) and "ambig" in str(r) for r in reasons
        ):
            return "Same name; review before merging."
        # Future: other reason kinds get their own editorial line.
        return "Cloris flagged this pair for your review."
    return "Cloris flagged this pair for your review."


def pending_decisions_for_brief(
    brief_id: str,
    *,
    identity_db_path: Path | None = None,
) -> list[PendingDecision]:
    """G2 read-model: list undecided merge decisions for a brief, enriched
    with person evidence and Cloris-voice ``signal_summary`` prose.

    Sorted by ``confidence DESC`` then ``created_at ASC`` so highest-signal
    pairs surface first. Stale rows where ``person_a`` or ``person_b`` no
    longer exist (a prior merge dropped one of the persons) are silently
    skipped — they should be cleaned up by a future migration but for now
    the read-model is defensive.
    """

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, person_a, person_b, confidence, evidence_json, created_at
            FROM pending_merge_decisions
            WHERE brief_id = ? AND recruiter_decision IS NULL
            ORDER BY confidence DESC, created_at ASC
            """,
            (brief_id,),
        ).fetchall()
        out: list[PendingDecision] = []
        for row in rows:
            person_a = _person_with_evidence_from_id(
                conn, person_id=int(row["person_a"]), brief_id=brief_id
            )
            person_b = _person_with_evidence_from_id(
                conn, person_id=int(row["person_b"]), brief_id=brief_id
            )
            if person_a is None or person_b is None:
                continue
            if not person_a.sources or not person_b.sources:
                continue
            evidence = json.loads(row["evidence_json"] or "{}")
            out.append(
                PendingDecision(
                    decision_id=int(row["id"]),
                    person_a=person_a,
                    person_b=person_b,
                    signal_summary=_describe_pending_signal(evidence),
                    created_at=str(row["created_at"] or ""),
                )
            )
        return out


def describe_merge_signal(link_kind: str, match_signal: dict) -> str:
    """Cloris-voice editorial prose for a merge link.

    Pre-empts the L14-class enum-leak that F6 would otherwise inherit.
    Returns empty string for singletons (no editorial prose needed
    when nothing was merged).
    """

    kind = (match_signal or {}).get("kind") if isinstance(match_signal, dict) else None

    if link_kind == LINK_KIND_MANUAL:
        return "Recruiter merged these candidates."
    if link_kind == LINK_KIND_AUTO_STRONG and kind == "linkedin_handle":
        return "Same LinkedIn handle on both saves."
    if link_kind == LINK_KIND_AUTO_MEDIUM and kind == "name_with_corroboration":
        return "Names match; corroborating GitHub link."
    if link_kind == LINK_KIND_AUTO_STRONG and kind == "name_only_ambiguous":
        return "Same name; recruiter review suggested before merging."
    return ""


def _resync_recruiter_authority_after_identity_change(
    *,
    refresh_ids: Iterable[int],
    delete_ids: Iterable[int],
    identity_db_path: Path | None,
    recruiter_db_path: Path | None,
    state_root: Path | None,
) -> None:
    """Post-commit re-sync of the recruiter CURRENT-STATE authority (Y.5.8 / F4).

    Called by ``record_recruiter_merge`` / ``record_recruiter_unlink`` AFTER
    their identity-DB transaction has COMMITTED, so the re-sync reads a committed
    persons/links snapshot — a re-sync issued INSIDE the ``with store.connect()``
    block would read uncommitted data and silently write a wrong survivor
    authority (the F4 blocker, mirroring ``record_decision_by_id`` calling
    ``record_recruiter_merge`` only after its own ``with`` block closes).

    ``refresh_ids`` are persons whose current-state may have changed and still
    EXIST in identity (the merge survivor ``keep_id``; an unlink's old + new
    person) — each gets ``fill_recruiter_candidate`` (re-derive + upsert the
    authority). ``delete_ids`` are persons HARD-DELETED from identity (the merge
    ``drop_id``) — their authority rows are dangling tombstones (exactly what the
    F3 sweep flags), removed via ``delete_recruiter_candidate``.

    ALL affected recruiters (C2): the re-sync touches every recruiter with an
    authority row for ANY id in ``refresh_ids ∪ delete_ids`` (via
    ``recruiter_ids_for_persons``), not only the brief's recruiter — else a
    tombstone survives in another recruiter's authority and the F3 sweep keeps
    flagging it.

    FAIL-SOFT (load-bearing): a merge / unlink is user-facing and its identity
    transaction is ALREADY committed by the time this runs; a re-sync failure
    must NOT surface to the recruiter or unwind the merge. Every fault — store
    open, the all-recruiters query, a per-(recruiter, person) fill/delete — is
    caught and logged, never raised. Recruiter modules are LAZY-imported (the
    identity service must not import them at module scope; this keeps the import
    graph acyclic — they import nothing from here either)."""

    try:
        from shared.runtime_state.recruiter_candidate_fill import (
            fill_recruiter_candidate,
        )
        from shared.runtime_state.recruiter_store import RecruiterStore

        if recruiter_db_path is None:
            from shared.output_paths import resolve_recruiter_db_path

            recruiter_db_path = resolve_recruiter_db_path()

        refresh = sorted({int(p) for p in refresh_ids})
        delete = sorted({int(p) for p in delete_ids})
        affected_persons = sorted(set(refresh) | set(delete))
        if not affected_persons:
            return

        store = RecruiterStore(recruiter_db_path)
        recruiter_ids = store.recruiter_ids_for_persons(affected_persons)

        for rid in recruiter_ids:
            for person_id in refresh:
                try:
                    fill_recruiter_candidate(
                        rid,
                        person_id,
                        identity_db_path=identity_db_path,
                        recruiter_db_path=recruiter_db_path,
                        state_root=state_root,
                    )
                except Exception as exc:  # noqa: BLE001 — never break the merge
                    log.warning(
                        "F4 authority refresh failed for recruiter %s person %s "
                        "(%s): %s",
                        rid,
                        person_id,
                        type(exc).__name__,
                        exc,
                    )
            for person_id in delete:
                try:
                    store.delete_recruiter_candidate(rid, person_id)
                except Exception as exc:  # noqa: BLE001 — never break the merge
                    log.warning(
                        "F4 authority tombstone-delete failed for recruiter %s "
                        "person %s (%s): %s",
                        rid,
                        person_id,
                        type(exc).__name__,
                        exc,
                    )
    except Exception as exc:  # noqa: BLE001 — the whole re-sync is best-effort
        log.warning(
            "F4 recruiter-authority re-sync skipped after identity change "
            "(%s): %s",
            type(exc).__name__,
            exc,
        )


def record_recruiter_merge(
    *,
    brief_id: str,
    person_a: int,
    person_b: int,
    decision: str,
    identity_db_path: Path | None = None,
    recruiter_db_path: Path | None = None,
    state_root: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> None:
    """F6 calls this when the recruiter resolves a pending merge.

    ``decision`` ∈ {'merge', 'keep_separate'}. When 'merge', all
    ``candidate_persons`` rows currently pointing at ``person_b`` are
    re-pointed at ``person_a`` and locked (``recruiter_locked=1``); the
    now-orphaned ``person_b`` row is deleted.
    """

    if decision not in {"merge", "keep_separate"}:
        raise ValueError(f"unknown decision: {decision!r}")

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    now_iso = _now_iso(now)
    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        a, b = sorted((person_a, person_b))
        conn.execute(
            """
            UPDATE pending_merge_decisions
            SET recruiter_decision = ?, decided_at = ?
            WHERE brief_id = ?
              AND ((person_a = ? AND person_b = ?) OR (person_a = ? AND person_b = ?))
            """,
            (decision, now_iso, brief_id, a, b, b, a),
        )

        if decision == "keep_separate":
            # P3.8: lock BOTH sides of the pair so later automation cannot
            # silently re-merge what a human split. `_upsert_link` refuses
            # to re-point any candidate_persons row with recruiter_locked=1
            # (checked before link_kind), so even a fresh corroborating
            # signal (e.g. a new handle match) arriving on a later
            # resolver run cannot move either person's existing candidates
            # onto a shared person row. This is global (not brief-scoped),
            # mirroring the merge branch below — "these are different
            # humans" is a fact about the persons, not about one brief.
            conn.execute(
                """
                UPDATE candidate_persons
                SET recruiter_locked = 1, updated_at = ?
                WHERE person_id IN (?, ?)
                """,
                (now_iso, a, b),
            )
            return

        keep_id, drop_id = a, b
        # Re-point links from the dropped person AND lock the existing
        # links on keep_id — once a recruiter asserts these are the
        # same human, every candidate joined to either side is a
        # manual link from then on.
        conn.execute(
            """
            UPDATE candidate_persons
            SET person_id = ?, link_kind = ?, recruiter_locked = 1, updated_at = ?
            WHERE person_id IN (?, ?)
            """,
            (keep_id, LINK_KIND_MANUAL, now_iso, keep_id, drop_id),
        )
        conn.execute(
            "DELETE FROM brief_persons WHERE person_id = ?",
            (drop_id,),
        )
        conn.execute("DELETE FROM persons WHERE id = ?", (drop_id,))
        conn.execute(
            """
            UPDATE persons SET last_seen_at = ?,
                canonical_name = COALESCE(NULLIF(canonical_name, ''), '')
            WHERE id = ?
            """,
            (now_iso, keep_id),
        )

    # Y.5.8 (F4) MERGE re-sync — POST-COMMIT (C1). Reached ONLY on
    # decision=='merge': keep_separate returned above, before the with-block ran
    # the merge writes. The with-block has now closed, so the identity DB is
    # COMMITTED — the re-sync reads the survivor's post-merge links and the fact
    # that drop_id is gone from persons (a re-sync INSIDE the block would read
    # uncommitted state). Recompute keep/drop from the same sorted pair the block
    # used so the post-block hook does not depend on in-block locals.
    keep_id, drop_id = sorted((person_a, person_b))
    _resync_recruiter_authority_after_identity_change(
        refresh_ids=(keep_id,),
        delete_ids=(drop_id,),
        identity_db_path=identity_db_path,
        recruiter_db_path=recruiter_db_path,
        state_root=state_root,
    )


def auto_resolve_anonymous_pending(
    *,
    brief_id: str,
    anonymous_name: str = "LinkedIn Member",
    identity_db_path: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> int:
    """Bulk keep_separate for pairs where either person has an anonymous
    name (e.g. "LinkedIn Member"). Returns the number of rows resolved.

    These pairs offer no reconciliation signal so they should never
    surface to the recruiter.
    """

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    now_iso = _now_iso(now)
    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        cur = conn.execute(
            """
            UPDATE pending_merge_decisions
            SET recruiter_decision = 'keep_separate', decided_at = ?
            WHERE brief_id = ?
              AND recruiter_decision IS NULL
              AND (
                person_a IN (SELECT id FROM persons WHERE canonical_name = ?)
                OR person_b IN (SELECT id FROM persons WHERE canonical_name = ?)
              )
            """,
            (now_iso, brief_id, anonymous_name, anonymous_name),
        )
        return cur.rowcount


def record_decision_by_id(
    *,
    brief_id: str,
    decision_id: int,
    decision: str,
    identity_db_path: Path | None = None,
    recruiter_db_path: Path | None = None,
    state_root: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> str:
    """G2 helper: resolve a pending decision by its row id.

    Looks up ``person_a``/``person_b`` for the given ``decision_id`` and
    forwards to ``record_recruiter_merge``. Returns the canonical decision
    string ("merge" or "keep_separate") on success. Raises:

    - ``ValueError`` if ``decision`` is not in {"merge","keep_separate"}.
    - ``LookupError`` if ``decision_id`` doesn't exist for ``brief_id``.
    - ``RuntimeError`` if the decision was already resolved (terminal).
    """

    if decision not in {"merge", "keep_separate"}:
        raise ValueError(f"unknown decision: {decision!r}")

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT person_a, person_b, recruiter_decision
            FROM pending_merge_decisions
            WHERE id = ? AND brief_id = ?
            """,
            (decision_id, brief_id),
        ).fetchone()
        if row is None:
            raise LookupError(f"pending decision {decision_id} not found for brief {brief_id!r}")
        if row["recruiter_decision"] is not None:
            raise RuntimeError(
                f"pending decision {decision_id} already resolved as {row['recruiter_decision']!r}"
            )
        person_a_id = int(row["person_a"])
        person_b_id = int(row["person_b"])

    record_recruiter_merge(
        brief_id=brief_id,
        person_a=person_a_id,
        person_b=person_b_id,
        decision=decision,
        identity_db_path=identity_db_path,
        recruiter_db_path=recruiter_db_path,
        state_root=state_root,
        now=now,
    )
    return decision


def record_recruiter_unlink(
    *,
    source: str,
    state_key: str,
    candidate_id: int,
    identity_db_path: Path | None = None,
    recruiter_db_path: Path | None = None,
    state_root: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> None:
    """F6 calls this when the recruiter says "this candidate isn't this
    person." Splits the candidate off into its own fresh person row;
    locks the new link so auto-resolution doesn't merge it back.
    """

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    now_iso = _now_iso(now)
    store = IdentityStore(identity_db_path)
    old_person_id: int | None = None
    new_person_id: int | None = None
    with store.connect() as conn:
        existing = conn.execute(
            """
            SELECT person_id, brief_id FROM candidate_persons
            WHERE source = ? AND state_key = ? AND candidate_id = ?
            """,
            (source, state_key, candidate_id),
        ).fetchone()
        if existing is None:
            return
        old_person_id = int(existing["person_id"])
        brief_id = str(existing["brief_id"])
        cursor = conn.execute(
            """
            INSERT INTO persons(canonical_name, canonical_handle, created_at, last_seen_at)
            VALUES ('', '', ?, ?)
            """,
            (now_iso, now_iso),
        )
        new_person_id = int(cursor.lastrowid)
        conn.execute(
            """
            INSERT OR IGNORE INTO brief_persons(brief_id, person_id, first_seen_at)
            VALUES (?, ?, ?)
            """,
            (brief_id, new_person_id, now_iso),
        )
        conn.execute(
            """
            UPDATE candidate_persons
            SET person_id = ?, link_kind = ?, recruiter_locked = 1, updated_at = ?
            WHERE source = ? AND state_key = ? AND candidate_id = ?
            """,
            (new_person_id, LINK_KIND_MANUAL, now_iso, source, state_key, candidate_id),
        )

    # Y.5.8 (F4) UNLINK re-sync — POST-COMMIT (C1), analogous to the merge hook.
    # Reached only when the candidate existed (the no-record case returned inside
    # the block). The split moved the candidate from old_person_id onto a fresh
    # new_person_id; the with-block has now closed, so the identity DB is
    # COMMITTED and the re-sync reads the post-split links. BOTH persons are
    # REFRESHED (fill), never tombstoned: unlink deletes NO person row — the old
    # person still exists (it may retain other candidates, or have none, but it
    # is not gone from identity.persons), so its authority is not an F3 dangling
    # reference and must not be deleted; fill re-derives its now-changed
    # current-state, and gives the new person its first authority row.
    if old_person_id is not None and new_person_id is not None:
        _resync_recruiter_authority_after_identity_change(
            refresh_ids=(old_person_id, new_person_id),
            delete_ids=(),
            identity_db_path=identity_db_path,
            recruiter_db_path=recruiter_db_path,
            state_root=state_root,
        )
