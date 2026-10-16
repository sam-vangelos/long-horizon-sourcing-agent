"""Candidate obligations and exactly-once funnel accounting.

This owner has no browser, page, checkpoint callback, or run-state alias.
Canonical hydration takes explicit runtime inputs; callers publish outcomes
only after their canonical writer succeeds.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.runtime_state import RuntimeStateStore
from linkedin.browser import LinkedInBrowser
from shared.schemas import CandidateSnippet, OpusDecision, Progress, SearchString
from shared.contracts import FAILURE_DECISIONS, LEGACY_REVIEW_DECISIONS, OUTREACH_TIERS
from shared.employer_allowlist import EMPLOYER_SCOPE_GATE_PATH
from shared.excluded_locations import TERRITORY_GATE_PATH
from shared.judger import SAVE_FAMILY_DECISIONS, is_failure_decision

import re
_TRANSITIONAL_OUTREACH_TIER_TAG = re.compile(r"\[TIER:\s*(PRIORITY|STANDARD)\s*\]", re.IGNORECASE)

def _persisted_outreach_tier(full_payload: dict[str, Any]) -> str:
    """Read structured currency, falling back only for historical rows."""

    if "outreach_tier" in full_payload:
        tier = str(full_payload.get("outreach_tier") or "").strip().upper()
        return tier if tier in OUTREACH_TIERS else ""
    matches = list(
        _TRANSITIONAL_OUTREACH_TIER_TAG.finditer(
            str(full_payload.get("rationale") or "")
        )
    )
    return matches[-1].group(1).upper() if matches else ""


@dataclass
class CandidateLedger:
    """Process-local candidate state reconstructed from canonical run ancestry."""
    _candidate_funnel_counted: set[str] = field(init=False, repr=False)
    _facial_funnel_counted: set[str] = field(init=False, repr=False)
    _full_funnel_counted: set[str] = field(init=False, repr=False)
    _saved_urls: set[str] = field(init=False, repr=False)
    _in_flight_urls: set[str] = field(init=False, repr=False)
    _outreach_tier_counted: set[str] = field(init=False, repr=False)
    _prior_outcomes: dict[str, str] = field(init=False, repr=False)
    _prior_session_urls: set[str] = field(init=False, repr=False)
    _resume_pending_full_decisions: dict[str, str] = field(init=False, repr=False)
    _resume_pending_full_owner_ids: dict[str, int] = field(init=False, repr=False)
    _resume_pending_full_snippets: dict[str, CandidateSnippet] = field(init=False, repr=False)
    _seen_urls: set[str] = field(init=False, repr=False)

    @staticmethod
    def _runtime_json_object(raw: object) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str) or not raw:
            return {}
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _runtime_attempt_string_id(cls, row: dict[str, Any]) -> int | None:
        payload = cls._runtime_json_object(row.get("payload_json"))
        cursor = cls._runtime_json_object(row.get("source_cursor_json"))
        payload_cursor = payload.get("cursor")
        if not isinstance(payload_cursor, dict):
            payload_cursor = {}
        raw = (
            payload.get("source_string_id")
            or payload_cursor.get("source_string_id")
            or cursor.get("source_string_id")
            or row.get("source_unit_id")
        )
        try:
            return int(raw) if raw is not None and str(raw).strip() else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _runtime_attempt_decision(
        cls,
        row: dict[str, Any],
        *,
        stage: str,
    ) -> str:
        payload = cls._runtime_json_object(row.get("payload_json"))
        decision = payload.get(f"{stage}_decision")
        if not isinstance(decision, dict):
            return ""
        return str(decision.get("decision") or "").strip()

    @classmethod
    def _runtime_attempt_path(
        cls,
        row: dict[str, Any],
        *,
        stage: str,
    ) -> str:
        payload = cls._runtime_json_object(row.get("payload_json"))
        decision = payload.get(f"{stage}_decision")
        if not isinstance(decision, dict):
            return ""
        return str(decision.get("path") or "").strip()

    @classmethod
    def _runtime_attempt_snippet(
        cls,
        row: dict[str, Any],
        *,
        string_id: int | None,
    ) -> CandidateSnippet | None:
        """Rehydrate the immutable snippet captured with a stage attempt."""

        payload = cls._runtime_json_object(row.get("payload_json"))
        raw_snippet = payload.get("snippet")
        if isinstance(raw_snippet, dict):
            try:
                return CandidateSnippet.from_dict(raw_snippet)
            except (KeyError, TypeError, ValueError):
                pass

        profile_url = str(row.get("profile_url") or "").strip()
        if not profile_url or string_id is None:
            return None
        cursor = cls._runtime_json_object(row.get("source_cursor_json"))

        def cursor_int(name: str, default: int) -> int:
            try:
                return int(cursor.get(name, default))
            except (TypeError, ValueError):
                return default

        return CandidateSnippet(
            name=str(row.get("display_name") or "Unknown candidate"),
            headline="",
            current_title="",
            current_company="",
            location="",
            education_snippet="",
            profile_url=profile_url,
            source_string_id=string_id,
            source_string_name=str(cursor.get("source_string_name") or ""),
            page=max(1, cursor_int("page", 1)),
            result_rank=max(0, cursor_int("result_rank", 0)),
        )

    def _hydrate_resume_funnel_from_runtime(
        self,
        progress: Progress,
        *,
        store: RuntimeStateStore,
        run_id: int | None,
        brief_id: str,
        stats: dict[str, Any],
    ) -> None:
        """Rebuild resume-only funnel truth from this run's canonical ancestry.

        Resuming clones work units into a new run, but candidate attempts remain
        on the interrupted ancestor.  Read only that explicit ancestry chain so
        unrelated completed runs for the same brief cannot leak into the new
        process.  Successful stage attempts own semantic outcome truth; cloned
        SearchString counters remain a compatibility fallback only for a string
        that has no attributable attempts in the chain.
        """

        if not run_id:
            return

        chain_cte = """
            WITH RECURSIVE run_chain(id, resumed_from_run_id) AS (
                SELECT id, resumed_from_run_id
                FROM runs
                WHERE id = ? AND source = 'linkedin' AND brief_id = ?
                UNION
                SELECT r.id, r.resumed_from_run_id
                FROM runs r
                JOIN run_chain child ON r.id = child.resumed_from_run_id
                WHERE r.source = 'linkedin' AND r.brief_id = ?
            )
        """
        with store.connect() as conn:
            attempt_rows = [
                dict(row)
                for row in conn.execute(
                    chain_cte
                    + """
                    SELECT ca.id, ca.run_id, ca.stage, ca.status,
                           ca.payload_json, ca.source_cursor_json,
                           c.identity_key, c.display_name, c.profile_url,
                           wu.source_unit_id
                    FROM candidate_attempts ca
                    JOIN run_chain chain ON chain.id = ca.run_id
                    JOIN candidates c ON c.id = ca.candidate_id
                    LEFT JOIN work_units wu ON wu.id = ca.work_unit_id
                    WHERE c.source = 'linkedin' AND c.brief_id = ?
                      AND ca.status = 'succeeded'
                    ORDER BY ca.id ASC
                    """,
                    (
                        run_id,
                        brief_id,
                        brief_id,
                        brief_id,
                    ),
                ).fetchall()
            ]
            side_effect_rows = [
                dict(row)
                for row in conn.execute(
                    chain_cte
                    + """
                    SELECT se.id, se.run_id, se.effect_type, se.status,
                           se.payload_json,
                           c.identity_key, c.display_name, c.profile_url,
                           ca.payload_json AS attempt_payload_json,
                           ca.source_cursor_json, wu.source_unit_id
                    FROM side_effects se
                    JOIN run_chain chain ON chain.id = se.run_id
                    JOIN candidates c ON c.id = se.candidate_id
                    LEFT JOIN candidate_attempts ca ON ca.id = se.attempt_id
                    LEFT JOIN work_units wu ON wu.id = ca.work_unit_id
                    WHERE c.source = 'linkedin' AND c.brief_id = ?
                      AND se.effect_type IN ('linkedin_save', 'linkedin_note')
                    ORDER BY se.id ASC
                    """,
                    (
                        run_id,
                        brief_id,
                        brief_id,
                        brief_id,
                    ),
                ).fetchall()
            ]

        candidate_rows: dict[str, tuple[int | None, str]] = {}
        facial_rows: dict[str, tuple[str, int | None]] = {}
        # Territory and scope skips stay terminal but out of the facial
        # counters, and a profile scope skip out of the full counters, the same
        # rule the live paths apply (candidate_review.NON_VERDICT_GATE_PATHS).
        non_verdict_facial_keys: set[str] = set()
        scope_full_keys: set[str] = set()
        facial_snippets: dict[str, CandidateSnippet] = {}
        full_rows: dict[str, tuple[str, int | None]] = {}
        full_outreach_tiers: dict[str, str] = {}
        attributable_string_ids: set[int] = set()
        for row in attempt_rows:
            key = str(row.get("identity_key") or row.get("profile_url") or "").strip()
            if not key:
                continue
            string_id = self._runtime_attempt_string_id(row)
            if string_id is not None:
                attributable_string_ids.add(string_id)
            stage = str(row.get("stage") or "")
            if stage == "snippet" and key not in candidate_rows:
                candidate_rows[key] = (
                    string_id,
                    str(row.get("display_name") or ""),
                )
            elif key not in candidate_rows:
                # Compatibility for early runtime rows that predate the
                # explicit snippet attempt but still carry a stage attempt.
                candidate_rows[key] = (
                    string_id,
                    str(row.get("display_name") or ""),
                )
            if stage == "facial" and key not in facial_rows:
                decision = self._runtime_attempt_decision(row, stage="facial")
                if decision:
                    facial_rows[key] = (decision, string_id)
                    if self._runtime_attempt_path(row, stage="facial") in {
                        TERRITORY_GATE_PATH,
                        EMPLOYER_SCOPE_GATE_PATH,
                    }:
                        non_verdict_facial_keys.add(key)
                    snippet = self._runtime_attempt_snippet(
                        row,
                        string_id=string_id,
                    )
                    if snippet is not None:
                        facial_snippets[key] = snippet
            elif stage == "full":
                decision = self._runtime_attempt_decision(row, stage="full")
                # Full rows are first-succeeded-wins EXCEPT that a
                # failure-family row yields to a later real verdict. A
                # contained resume skip settles the candidate with a synthetic
                # succeeded JUDGMENT_FAILURE (see
                # _abandon_unrecoverable_pending_full); under plain first-wins
                # that placeholder would shadow the real decision forever if
                # the person is met and evaluated again, hiding the review from
                # the counters and — for a SAVE-family re-meet — from the
                # pending derivation that drives actuation. Two real decisions
                # keep first-wins. Facial stays first-wins unconditionally: no
                # failure decision ever lands in a succeeded facial row, and
                # the SQL twin in shared/runtime_state/read_models.py orders
                # the same way.
                stored = full_rows.get(key)
                displaces = (
                    stored is not None
                    and stored[0] in FAILURE_DECISIONS
                    and decision not in FAILURE_DECISIONS
                )
                if decision and (stored is None or displaces):
                    full_rows[key] = (decision, string_id)
                    if self._runtime_attempt_path(row, stage="full") == EMPLOYER_SCOPE_GATE_PATH:
                        scope_full_keys.add(key)
                    else:
                        scope_full_keys.discard(key)
                    payload = self._runtime_json_object(row.get("payload_json"))
                    full_payload = payload.get("full_decision")
                    tier = (
                        _persisted_outreach_tier(full_payload)
                        if isinstance(full_payload, dict)
                        else ""
                    )
                    if tier:
                        full_outreach_tiers[key] = tier
                    elif displaces:
                        full_outreach_tiers.pop(key, None)

        saved_rows: dict[str, tuple[int | None, str]] = {}
        already_present_keys: set[str] = set()
        save_attempted_keys: set[str] = set()
        for row in side_effect_rows:
            # Notes share the candidate ledger and must survive this read, but
            # only linkedin_save rows hydrate save counters and resume gates.
            if str(row.get("effect_type") or "") != "linkedin_save":
                continue
            key = str(row.get("identity_key") or row.get("profile_url") or "").strip()
            if not key:
                continue
            save_attempted_keys.add(key)
            if str(row.get("status") or "") != "succeeded":
                continue
            payload = self._runtime_json_object(row.get("payload_json"))
            if payload.get("already_present") is True and not payload.get(
                "reconciled_self_save"
            ):
                already_present_keys.add(key)
            attempt_row = dict(row)
            attempt_row["payload_json"] = row.get("attempt_payload_json")
            saved_rows.setdefault(
                key,
                (
                    self._runtime_attempt_string_id(attempt_row),
                    str(row.get("display_name") or ""),
                ),
            )

        self._candidate_funnel_counted = set(candidate_rows)
        self._facial_funnel_counted = set(facial_rows)
        self._full_funnel_counted = set(full_rows)
        self._outreach_tier_counted = set(full_outreach_tiers)
        self._resume_pending_full_decisions = {}
        self._resume_pending_full_snippets = {}
        self._resume_pending_full_owner_ids = {}
        for identity_key, (decision, string_id) in facial_rows.items():
            full_decision = (
                full_rows.get(identity_key, ("", None))[0]
                if identity_key in full_rows
                else ""
            )
            full_is_terminal = (
                identity_key in full_rows
                and (
                    full_decision not in SAVE_FAMILY_DECISIONS
                    or identity_key in saved_rows
                )
            )
            if (
                decision not in {"FACIAL_YES", "FACIAL_BORDERLINE"}
                or full_is_terminal
            ):
                continue
            snippet = facial_snippets.get(identity_key)
            pending_key = (
                self._funnel_candidate_key(snippet)
                if snippet is not None
                else identity_key
            )
            self._resume_pending_full_decisions[pending_key] = decision
            self._resume_pending_full_owner_ids[pending_key] = string_id
            if snippet is not None:
                self._resume_pending_full_snippets[pending_key] = snippet

        per_string: dict[int, dict[str, Any]] = {}

        def counters_for(string_id: int) -> dict[str, Any]:
            counters = per_string.get(string_id)
            if counters is None:
                counters = self._fresh_string_stats()
                counters["saved_names"] = []
                per_string[string_id] = counters
            return counters

        for _key, (string_id, _name) in candidate_rows.items():
            if string_id is not None:
                counters_for(string_id)["candidates"] += 1
        for _key, (decision, string_id) in facial_rows.items():
            if string_id is None or _key in non_verdict_facial_keys:
                continue
            counter = {
                "FACIAL_YES": "facial_yes",
                "FACIAL_BORDERLINE": "facial_borderline",
                "FACIAL_NO": "facial_no",
            }.get(decision)
            if counter:
                counters_for(string_id)[counter] += 1
        for _key, (decision, string_id) in full_rows.items():
            if string_id is None or _key in scope_full_keys or (
                decision not in SAVE_FAMILY_DECISIONS
                and decision not in LEGACY_REVIEW_DECISIONS
                and decision != "REJECT"
            ):
                continue
            counters = counters_for(string_id)
            counters["full_reviewed"] += 1
            if decision in SAVE_FAMILY_DECISIONS:
                counters["full_outreach"] += 1
            elif decision in LEGACY_REVIEW_DECISIONS:
                counters["full_review"] += 1
            else:
                counters["full_reject"] += 1
                counters["rejects"] += 1
        for key, (string_id, name) in saved_rows.items():
            if string_id is None or key in already_present_keys:
                continue
            counters = counters_for(string_id)
            counters["saves"] += 1
            if name:
                counters["saved_names"].append(name)

        for search_string in progress.strings:
            if search_string.id not in attributable_string_ids:
                continue
            counters = counters_for(search_string.id)
            search_string.candidates_count = int(counters["candidates"])
            search_string.facial_yes_count = int(counters["facial_yes"])
            search_string.facial_borderline_count = int(
                counters["facial_borderline"]
            )
            search_string.facial_no_count = int(counters["facial_no"])
            search_string.full_reviewed_count = int(counters["full_reviewed"])
            search_string.full_outreach_count = int(counters["full_outreach"])
            search_string.full_review_count = int(counters["full_review"])
            search_string.full_reject_count = int(counters["full_reject"])
            search_string.saves = list(counters["saved_names"])

        facial_counts = {
            decision: sum(
                1
                for key, (value, _sid) in facial_rows.items()
                if value == decision and key not in non_verdict_facial_keys
            )
            for decision in ("FACIAL_YES", "FACIAL_BORDERLINE", "FACIAL_NO")
        }
        full_decisions = [
            decision
            for key, (decision, _sid) in full_rows.items()
            if key not in scope_full_keys
        ]
        stats["snippets_extracted"] = len(candidate_rows)
        stats["facial_yes"] = facial_counts["FACIAL_YES"]
        stats["facial_borderline"] = facial_counts["FACIAL_BORDERLINE"]
        stats["facial_no"] = facial_counts["FACIAL_NO"]
        stats["full_outreach"] = sum(
            1 for decision in full_decisions if decision in SAVE_FAMILY_DECISIONS
        )
        stats["full_review"] = sum(
            1 for decision in full_decisions if decision in LEGACY_REVIEW_DECISIONS
        )
        stats["full_reject"] = sum(
            1 for decision in full_decisions if decision == "REJECT"
        )
        stats["full_reviewed"] = (
            stats["full_outreach"]
            + stats["full_review"]
            + stats["full_reject"]
        )
        outreach_tier_counts: dict[str, int] = {}
        for identity_key, (decision, _string_id) in full_rows.items():
            if decision not in SAVE_FAMILY_DECISIONS:
                continue
            tier = full_outreach_tiers.get(identity_key, "")
            if tier:
                outreach_tier_counts[tier] = (
                    outreach_tier_counts.get(tier, 0) + 1
                )
        stats["outreach_tier_counts"] = outreach_tier_counts
        stats["saved"] = len(saved_rows) - len(already_present_keys)
        stats["already_present"] = len(already_present_keys)
        stats["save_attempts"] = len(save_attempted_keys)
        # The live settle counts a profile scope skip as a run-level reject
        # (candidate_review._settle_profile_alumni_gate); so does the rebuild.
        stats["rejected"] = stats["full_reject"] + len(scope_full_keys)
        stats["reviewed"] = stats["full_review"]
        stats["reviewed_inferred"] = sum(
            1 for decision in full_decisions if decision == "REVIEW_INFERRED"
        )
        stats["reviewed_flagged"] = sum(
            1 for decision in full_decisions if decision == "REVIEW_FLAGGED"
        )
        progress.candidates_saved = stats["saved"]
        progress.candidates_rejected = stats["rejected"]

    def _mark_terminal(self, url: str):
        """Promote URL from in-flight to permanent dedup."""
        self._in_flight_urls.discard(url)
        if url:
            self._seen_urls.add(url)

    def _track_full_review_obligation(
        self,
        snippet: CandidateSnippet,
        decision: str,
    ) -> None:
        key = self._funnel_candidate_key(snippet)
        self._resume_pending_full_owner_ids[key] = snippet.source_string_id
        self._resume_pending_full_snippets[key] = snippet
        self._resume_pending_full_decisions[key] = decision

    def _record_outreach_tier_outcome(
        self,
        *,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        stats: dict[str, Any],
    ) -> None:
        """Count semantic save currency once, before physical actuation."""

        if (
            decision.stage != "full"
            or decision.decision not in SAVE_FAMILY_DECISIONS
        ):
            return
        tier = str(getattr(decision, "outreach_tier", "") or "").strip().upper()
        if tier not in OUTREACH_TIERS:
            return
        seen = getattr(self, "_outreach_tier_counted", None)
        if not isinstance(seen, set):
            seen = set()
            self._outreach_tier_counted = seen
        key = self._funnel_candidate_key(snippet)
        if key in seen:
            return
        seen.add(key)
        counts = stats.setdefault("outreach_tier_counts", {})
        counts[tier] = int(counts.get(tier, 0)) + 1

    @staticmethod
    def _fresh_string_stats() -> dict[str, int]:
        return {
            "pages": 0,
            "candidates": 0,
            "duplicates": 0,
            "facial_yes": 0,
            "facial_borderline": 0,
            "facial_no": 0,
            "full_reviewed": 0,
            "full_outreach": 0,
            "full_review": 0,
            "full_reject": 0,
            "saves": 0,
            "rejects": 0,
            "high_pressure_candidates_seen": 0,
            "activity_saturated_preview_skips": 0,
            "high_fit_low_novelty_saves": 0,
        }

    @classmethod
    def _string_stats_for_processing(
        cls,
        search_string: SearchString,
        *,
        resuming: bool,
    ) -> dict[str, int]:
        """Return a cumulative checkpoint baseline for an interrupted string."""

        stats = cls._fresh_string_stats()
        if not resuming:
            return stats
        stats.update(
            {
                "pages": int(search_string.pages_reviewed),
                "candidates": int(search_string.candidates_count),
                "duplicates": int(search_string.duplicates_count),
                "suppressed_prior_session": int(
                    search_string.suppressed_prior_session_count
                ),
                "facial_yes": int(search_string.facial_yes_count),
                "facial_borderline": int(
                    search_string.facial_borderline_count
                ),
                "facial_no": int(search_string.facial_no_count),
                "full_reviewed": int(search_string.full_reviewed_count),
                "full_outreach": int(search_string.full_outreach_count),
                "full_review": int(search_string.full_review_count),
                "full_reject": int(search_string.full_reject_count),
                "saves": len(search_string.saves),
                "rejects": int(search_string.full_reject_count),
            }
        )
        return stats

    def _record_candidate_funnel_discovery(
        self,
        *,
        snippet: CandidateSnippet,
        string_stats: dict[str, int],
        stats: dict[str, Any],
    ) -> bool:
        """Count a discovered identity once across a fresh or resumed process."""

        seen = getattr(self, "_candidate_funnel_counted", None)
        if not isinstance(seen, set):
            seen = set()
            self._candidate_funnel_counted = seen
        key = self._funnel_candidate_key(snippet)
        if key in seen:
            return False
        seen.add(key)
        stats["snippets_extracted"] = int(
            stats.get("snippets_extracted", 0)
        ) + 1
        string_stats["candidates"] = int(string_stats.get("candidates", 0)) + 1
        return True

    def _resume_pending_full_decision(
        self,
        snippet: CandidateSnippet,
    ) -> str | None:
        """Return owned decision, ``None`` for foreign work, or blank if absent."""

        pending = getattr(self, "_resume_pending_full_decisions", None)
        if not isinstance(pending, dict):
            return ""
        key = self._funnel_candidate_key(snippet)
        decision = str(pending.get(key) or "")
        if not decision:
            return ""
        owner_ids = getattr(self, "_resume_pending_full_owner_ids", None)
        if not isinstance(owner_ids, dict) or owner_ids.get(key) is None:
            raise RuntimeError("canonical pending full review lacks owning string")
        if owner_ids[key] != snippet.source_string_id:
            return None
        return decision

    def _clear_resume_pending_full_if_settled(
        self,
        *,
        snippet: CandidateSnippet,
        decision: OpusDecision | None,
    ) -> None:
        if (
            decision is None
            or decision.stage != "full"
            or is_failure_decision(decision.decision)
        ):
            return
        pending = getattr(self, "_resume_pending_full_decisions", None)
        if isinstance(pending, dict):
            key = self._funnel_candidate_key(snippet)
            pending.pop(key, None)
            pending_snippets = getattr(self, "_resume_pending_full_snippets", None)
            if isinstance(pending_snippets, dict):
                pending_snippets.pop(key, None)
            pending_owner_ids = getattr(self, "_resume_pending_full_owner_ids",
                None,
            )
            if isinstance(pending_owner_ids, dict):
                pending_owner_ids.pop(key, None)

    def _failed_full_attempt_counts(
        self,
        profile_urls: list[str],
        *,
        store: RuntimeStateStore,
        run_id: int | None,
        brief_id: str,
    ) -> dict[str, int]:
        """Non-succeeded full-stage attempts per profile URL across the run chain.

        Resume hydration deliberately reads succeeded attempts only, so a
        candidate whose full evaluations keep FAILING looks eternally pending
        to it. This is the failure-side complement: how many full attempts have
        already started and not succeeded for each candidate — the number of
        evaluations (each usually a governed profile open) the run chain has
        already burned on them.
        """

        urls = sorted({str(url) for url in profile_urls if url})
        if not urls or not run_id:
            return {}
        placeholders = ",".join("?" for _ in urls)
        # Run-level aborts set force_retryable=True and are excluded: the process's
        # fault must not make crashes during a candidate's evaluation abandon it.
        query = f"""
            WITH RECURSIVE run_chain(id, resumed_from_run_id) AS (
                SELECT id, resumed_from_run_id
                FROM runs
                WHERE id = ? AND source = 'linkedin' AND brief_id = ?
                UNION
                SELECT r.id, r.resumed_from_run_id
                FROM runs r
                JOIN run_chain child ON r.id = child.resumed_from_run_id
                WHERE r.source = 'linkedin' AND r.brief_id = ?
            )
            SELECT c.profile_url, COUNT(*)
            FROM candidate_attempts ca
            JOIN run_chain chain ON chain.id = ca.run_id
            JOIN candidates c ON c.id = ca.candidate_id
            WHERE c.source = 'linkedin' AND c.brief_id = ?
              AND ca.stage = 'full'
              AND ca.status != 'succeeded'
              AND COALESCE(
                  json_extract(ca.payload_json, '$.force_retryable'), 0
              ) != 1
              AND c.profile_url IN ({placeholders})
            GROUP BY c.profile_url
        """
        with store.connect() as conn:
            rows = conn.execute(
                query,
                (
                    run_id,
                    brief_id,
                    brief_id,
                    brief_id,
                    *urls,
                ),
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    @staticmethod
    def _page_stat_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        keys = (
            "candidates",
            "duplicates",
            "facial_yes",
            "facial_borderline",
            "facial_no",
            "full_reviewed",
            "full_outreach",
            "full_review",
            "full_reject",
            "saves",
            "rejects",
            "high_pressure_candidates_seen",
            "activity_saturated_preview_skips",
            "high_fit_low_novelty_saves",
        )
        return {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in keys}

    @staticmethod
    def _funnel_candidate_key(snippet: CandidateSnippet) -> str:
        """Stable identity for exactly-once in-process funnel accounting."""

        return str(snippet.profile_url or "").strip() or (
            f"{snippet.source_string_id}:{snippet.page}:{snippet.result_rank}:"
            f"{snippet.name}"
        )

    def _record_facial_funnel_outcome(
        self,
        *,
        snippet: CandidateSnippet,
        decision: str,
        search_string: SearchString,
        string_stats: dict[str, int] | None,
        stats: dict[str, Any],
    ) -> None:
        """Count one canonical facial verdict per candidate for this run.

        Facial opens are stage-boundary metrics. They are never reconstructed
        from later full outcomes: that double-counts retries and erases the
        distinction between YES and BORDERLINE. The first retrieval string
        owns attribution if the same profile URL resurfaces later.
        """

        counter_by_decision = {
            "FACIAL_YES": "facial_yes",
            "FACIAL_BORDERLINE": "facial_borderline",
            "FACIAL_NO": "facial_no",
        }
        counter = counter_by_decision.get(decision)
        if counter is None:
            return
        seen = getattr(self, "_facial_funnel_counted", None)
        if not isinstance(seen, set):
            seen = set()
            self._facial_funnel_counted = seen
        key = self._funnel_candidate_key(snippet)
        if key in seen:
            return
        seen.add(key)
        stats[counter] = int(stats.get(counter, 0)) + 1
        if string_stats is not None:
            string_stats[counter] = int(string_stats.get(counter, 0)) + 1

    def _record_full_funnel_outcome(
        self,
        *,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        search_string: SearchString,
        string_stats: dict[str, int],
        stats: dict[str, Any],
    ) -> None:
        """Count one settled full-profile disposition per candidate for this run.

        SAVE-family means the evidence is strong enough for outreach even if
        the later LinkedIn click fails. Human REVIEW is weak positive signal;
        REJECT is negative signal. Provider/parse failures are not settled.
        """

        if decision.stage != "full" or is_failure_decision(decision.decision):
            return
        if (
            decision.decision not in SAVE_FAMILY_DECISIONS
            and decision.decision not in LEGACY_REVIEW_DECISIONS
            and decision.decision != "REJECT"
        ):
            return
        seen = getattr(self, "_full_funnel_counted", None)
        if not isinstance(seen, set):
            seen = set()
            self._full_funnel_counted = seen
        key = self._funnel_candidate_key(snippet)
        if key in seen:
            return
        seen.add(key)

        stats["full_reviewed"] = int(stats.get("full_reviewed", 0)) + 1
        string_stats["full_reviewed"] = int(
            string_stats.get("full_reviewed", 0)
        ) + 1
        if decision.decision in SAVE_FAMILY_DECISIONS:
            stats["full_outreach"] = int(stats.get("full_outreach", 0)) + 1
            string_stats["full_outreach"] = int(
                string_stats.get("full_outreach", 0)
            ) + 1
        elif decision.decision in LEGACY_REVIEW_DECISIONS:
            stats["full_review"] = int(stats.get("full_review", 0)) + 1
            string_stats["full_review"] = int(string_stats.get("full_review", 0)) + 1
        else:
            stats["full_reject"] = int(stats.get("full_reject", 0)) + 1
            string_stats["full_reject"] = int(string_stats.get("full_reject", 0)) + 1
            # Compatibility alias for existing reports while readers migrate.
            string_stats["rejects"] = int(string_stats.get("rejects", 0)) + 1

    def resumable_string_ids(self) -> frozenset[int]:
        """Snapshot recorded owners, including entries with incomplete metadata."""
        return frozenset(getattr(self, "_resume_pending_full_owner_ids", {}).values())

    def has_pending(self, key: str) -> bool:
        return key in self._resume_pending_full_decisions

    def pending_keys(self, owner_id: int) -> tuple[str, ...]:
        """Snapshot an owner's outstanding keys in decision insertion order."""
        return tuple(
            key for key in self._resume_pending_full_decisions
            if self._resume_pending_full_owner_ids.get(key) == owner_id
        )

    def describe_pending(self, keys: tuple[str, ...]) -> str:
        """Describe a prior key snapshot using the currently stored snippets."""
        return ", ".join(
            f"{snippet.name} ({snippet.profile_url})" if snippet is not None else key
            for key in keys
            for snippet in [self._resume_pending_full_snippets.get(key)]
        )

    def pending_for_surface_change(
        self, owner_id: int,
    ) -> Iterator[tuple[str, CandidateSnippet | None]]:
        """Iterate tolerant settlement facts in owner insertion order.

        Missing maps prevent containment; missing snippets are returned so the
        driver can forget them without a canonical write. Unlike recovery,
        this path does not validate decision or snippet identity. Snapshot keys
        before callers settle entries, then read each snippet at its turn.
        """
        pending = getattr(self, "_resume_pending_full_decisions", None)
        snippets = getattr(self, "_resume_pending_full_snippets", None)
        owners = getattr(self, "_resume_pending_full_owner_ids", None)
        if not all(isinstance(value, dict) for value in (pending, snippets, owners)):
            return
        keys = [key for key, owner in owners.items() if owner == owner_id and key in pending]
        for key in keys:
            yield key, snippets.get(key)

    def retain_failed_full(self, snippet: CandidateSnippet) -> None:
        key = self._funnel_candidate_key(snippet)
        facial_decision = self._resume_pending_full_decisions.get(key)
        if facial_decision in {"FACIAL_YES", "FACIAL_BORDERLINE"}:
            self._track_full_review_obligation(snippet, facial_decision)

    def validated_pending(
        self, *, owner_id: int, queue_ids: set[int],
    ) -> list[tuple[str, CandidateSnippet]]:
        """Return owned facts; the driver owns checkpointing validation errors."""
        pending = getattr(self, "_resume_pending_full_decisions", None)
        snippets = getattr(self, "_resume_pending_full_snippets", None)
        owner_ids = getattr(self, "_resume_pending_full_owner_ids", None)
        if not isinstance(pending, dict) or not pending:
            return []

        if not isinstance(snippets, dict) or not isinstance(owner_ids, dict):
            raise RuntimeError("canonical pending full review metadata is incomplete")
        if any(owner_ids.get(key) is None for key in pending):
            raise RuntimeError("canonical pending full review lacks owning string")
        if any(owner_ids.get(key) not in queue_ids for key in pending):
            raise RuntimeError("canonical pending full review owner is absent from the queue")

        owned: list[tuple[str, CandidateSnippet]] = []
        for key, decision in pending.items():
            if owner_ids.get(key) != owner_id:
                continue
            snippet = snippets.get(key)
            if snippet is None:
                raise RuntimeError("canonical pending full review lacks stored snippet")
            if snippet.source_string_id != owner_id:
                raise RuntimeError(
                    "canonical pending full review has conflicting string ownership"
                )
            profile_url = str(snippet.profile_url or "").strip()
            if (
                not profile_url
                or self._funnel_candidate_key(snippet) != key
                or not LinkedInBrowser._profile_url_fragment(profile_url)
            ):
                raise RuntimeError("canonical pending full review lacks exact Recruiter identity")
            if str(decision or "") not in {
                "FACIAL_YES",
                "FACIAL_BORDERLINE",
            }:
                raise RuntimeError("canonical pending full review has invalid facial decision")
            owned.append((key, snippet))
        return sorted(
            owned,
            key=lambda item: (
                max(1, int(item[1].page or 1)),
                max(0, int(item[1].result_rank or 0)),
            ),
        )

    def pending_page(self, owner_id: int, page: int) -> tuple[CandidateSnippet, ...]:
        pending = getattr(self, "_resume_pending_full_decisions", None)
        snippets = getattr(self, "_resume_pending_full_snippets", None)
        owners = getattr(self, "_resume_pending_full_owner_ids", None)
        if not all(isinstance(value, dict) for value in (pending, snippets, owners)):
            return ()
        return tuple(snippet for key, snippet in snippets.items()
                     if key in pending and owners.get(key) == owner_id
                     and snippet.source_string_id == owner_id
                     and max(1, int(snippet.page or 1)) == page)

    def forget_pending(self, key: str) -> None:
        for values in (getattr(self, "_resume_pending_full_decisions", None),
                       getattr(self, "_resume_pending_full_snippets", None),
                       getattr(self, "_resume_pending_full_owner_ids", None)):
            if isinstance(values, dict):
                values.pop(key, None)

    def start_review(self, url: str) -> None:
        self._in_flight_urls.add(url)

    def release_review(self, url: str) -> None:
        self._in_flight_urls.discard(url)

    def allow_review(self, url: str) -> None:
        self._seen_urls.discard(url)

    def record_prior_outcome(self, url: str, decision: str) -> None:
        self._prior_outcomes[url] = decision

    def prior_outcome(self, url: str) -> str:
        return self._prior_outcomes.get(url, "")

    def is_terminal(self, url: str) -> bool:
        return url in self._seen_urls

    def is_in_flight(self, url: str) -> bool:
        return url in self._in_flight_urls

    def was_seen_before_run(self, url: str) -> bool:
        return url in self._prior_session_urls

    def _merge_prior_search_exclusion(self, brief) -> None:
        """Merge ``brief.prior_search.ruled_out_urls`` into the dedup set.

        If ``self.brief`` is not yet bound, the field is missing, or
        ``ruled_out_urls`` is malformed, no-op to preserve initialization and
        classic-brief behavior. Re-running is idempotent because URLs enter a
        set.
        """

        if brief is None:
            return
        prior_search = getattr(brief, "prior_search", None)
        if prior_search is None:
            return
        ruled_out_urls = getattr(prior_search, "ruled_out_urls", None) or []
        if not isinstance(ruled_out_urls, list):
            return
        added = 0
        for url in ruled_out_urls:
            if isinstance(url, str) and url:
                if url not in self._seen_urls:
                    self._seen_urls.add(url)
                    added += 1
        if added > 0:
            print(
                f"  [dedup] Prior-search exclusion: merged {added} URL(s) "
                f"from brief.prior_search.ruled_out_urls into _seen_urls"
            )
