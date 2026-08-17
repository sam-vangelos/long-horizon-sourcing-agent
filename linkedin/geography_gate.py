"""Session geography gate for LinkedIn sourcing runs.

Owns the fail-closed geography precondition cluster: applying and verifying
session location facets, chip invariants, recovery re-asserts, and off-geo
save telemetry. ``Pipeline`` delegates to ``GeographyGateService``.
"""

from __future__ import annotations

import json
import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from linkedin.acquisition import (
    _is_browser_disconnect_error as _acquisition_is_browser_disconnect_error,
)
from linkedin.browser import normalize_facet_value_for_compare
from shared.storage import log_event

if TYPE_CHECKING:
    from shared.brief_loader import Brief
    from shared.schemas import CandidateSnippet


def _is_browser_disconnect_error(error: BaseException | str) -> bool:
    from linkedin.orchestrator import PageRenderFailedError

    if isinstance(error, PageRenderFailedError):
        return True
    return _acquisition_is_browser_disconnect_error(
        error,
        include_render_failures=False,
    )


@dataclass(frozen=True)
class GeographyGateDeps:
    get_browser: Callable[[], Any]
    get_brief_obj: Callable[[], "Brief"]
    log_path: Path
    stats: dict[str, Any]


class GeographyGateService:
    """Owns session geography application, verification, and off-geo telemetry."""

    _OFF_GEO_STOP_TOKENS = frozenset(
        {"united", "states", "metropolitan", "area", "greater", "region", "city"}
    )

    def __init__(self, deps: GeographyGateDeps):
        self.deps = deps
        self._session_location_applied: bool = False
        self._session_geography_receipt: dict = {}
        self._resolved_session_geography: list[str] | None = None
        self._session_geography_resolutions: list[dict] = []

    @property
    def session_geography_receipt(self) -> dict:
        return self._session_geography_receipt

    def reset_location_applied(self) -> None:
        self._session_location_applied = False

    def _session_geography_values(self) -> list[str]:
        """The session's geography as exact Recruiter facet values.

        Live-caught on the SPL test run (2026-07-03): a single conjunction
        string ("NYC ... and SF ...") exact-match missed as one facet and the
        whole run proceeded boolean-only — every save came back off-geography.

        P3a Stage B: once a preflight candidate has been model-resolved to a
        real facet name, the RESOLVED list is the session's geography — the
        chip invariant and every re-assert must verify what was actually
        applied, not the raw candidate the resolution replaced.
        """
        if self._resolved_session_geography is not None:
            return list(self._resolved_session_geography)
        geo = str(
            self.deps.get_brief_obj().permanent_filters.get("Location", "") or ""
        ).strip()
        return [v.strip() for v in geo.split(";") if v.strip()]

    async def _apply_session_location_filter(self) -> None:
        """Apply and VERIFY the brief's geography — a fail-closed precondition.

        Location is a session-level fact (one geography per run, from the brief), not a
        per-lane hypothesis, so it is applied directly here rather than threaded through
        the per-lane constraint grammar (which would flip every lane to hybrid and
        re-apply the same geography per variant). Idempotent within a session via
        _session_location_applied; callers reset that flag after a (re)navigation that
        drops the sidebar chip.

        P3a (plans/sourcing-rigor-hardening.md, decided 2026-07-03): fail-CLOSED.
        browser.apply_location_filter is already fail-closed at its layer (exact-match
        typeahead guard + applied-chip confirmation + subset-apply refusal); the rot was
        here — this method used to swallow a False/raise into one console line and
        proceed boolean-only. A miss now raises GeographyRegimeError instead of
        degrading to an unbounded pool.
        """
        from linkedin.orchestrator import GeographyRegimeError

        if self._session_location_applied:
            return
        geo_values = self._session_geography_values()
        if not geo_values:
            return
        try:
            # Scope is always current-or-past; the browser ignores this arg for
            # the dropdown, but pass the honest value so the call site does not
            # imply current-residents-only.
            applied = await self.deps.get_browser().apply_location_filter(
                geo_values, temporal_scope="current_or_past"
            )
        except Exception as exc:
            if _is_browser_disconnect_error(exc):
                raise
            raise GeographyRegimeError(
                f"Session geography {geo_values!r} failed to apply: {exc!r}. "
                "Aborting rather than searching an unbounded pool — each value must "
                "be an exact LinkedIn Recruiter location facet name."
            ) from exc
        if not applied:
            # P3a Stage B: preflight-sourced candidates get ONE model
            # resolution against the real typeahead options; operator-pinned
            # values keep Stage A's strict abort (the operator typed exact
            # facet names — never rewrite them).
            resolved_values: list[str] | None = None
            if getattr(self.deps.get_brief_obj(), "geography_source", "") == "preflight":
                resolved_values = await self._resolve_and_reapply_geography(geo_values)
            if resolved_values is None:
                raise GeographyRegimeError(
                    f"Session geography {geo_values!r} did not apply (no exact typeahead "
                    "match, or the applied chip never confirmed, or only a subset landed). "
                    "Aborting rather than searching an unbounded pool. Check the run log's "
                    "apply_location_filter diagnostics for the facet options LinkedIn "
                    "actually offered and correct permanent_filters['Location'] to exact "
                    "facet names.",
                    retryable=True,
                )
            geo_values = resolved_values
            self._resolved_session_geography = list(resolved_values)
        self._session_location_applied = True
        # P3b: durable receipt for the run report — verified-as-applied is a
        # recorded fact, not a live flag (the flag resets on navigation).
        # Single-owner reassert counting: EVERY successful re-apply after the
        # first (chip-drop re-assert, crash recovery, mid-string resume,
        # browser-health re-navigation) lands here, so counting here — and
        # only here — neither undercounts recovery re-applies nor
        # double-counts the chip-invariant path (correctness lens, slice 8).
        receipt = self._session_geography_receipt or {}
        self._session_geography_receipt = {
            "intended": list(geo_values),
            "verified_applied": True,
            "reasserts": (
                int(receipt.get("reasserts", 0) or 0) + 1
                if receipt.get("verified_applied")
                else 0
            ),
        }
        # P3a Stage B: every candidate→facet resolution is a recorded fact.
        if self._session_geography_resolutions:
            self._session_geography_receipt["resolutions"] = [
                dict(entry) for entry in self._session_geography_resolutions
            ]
        log_event(self.deps.log_path, "session_location_applied", location="; ".join(geo_values))

    async def _resolve_and_reapply_geography(
        self, geo_values: list[str]
    ) -> list[str] | None:
        """P3a Stage B: ONE model resolution against the REAL typeahead options.

        The browser captured, per missed value, the options the typeahead
        actually offered (``last_location_option_misses``). A single cheap_llm
        call maps each missed candidate to one of ITS OWN offered options —
        an answer outside that list is discarded, never applied (operate the
        filter against real DOM options, not blind strings). Values already
        confirmed as chips are not re-typed (re-typing an applied facet can
        false-miss the exact-match gate); only the still-missing resolved
        names re-apply. Returns the session's final value list on a verified
        re-apply, or None — the caller raises GeographyRegimeError, which the
        day cycle classifies as a stable break exactly like Stage A.
        """
        raw_misses = getattr(
            self.deps.get_browser(), "last_location_option_misses", None
        )
        if not isinstance(raw_misses, dict):
            return None
        misses = {
            str(candidate): [str(o).strip() for o in (options or []) if str(o).strip()]
            for candidate, options in raw_misses.items()
            if options
        }
        misses = {c: opts for c, opts in misses.items() if opts}
        if not misses:
            return None

        from shared.llm_clients import cheap_llm

        system = (
            "You operate a LinkedIn Recruiter Location facet. For each intended "
            "geography value, the typeahead offered a list of real facet options. "
            "Pick, per value, the ONE offered option that covers the same "
            "geography the value intends, or null when none does. Respond with "
            'ONLY JSON: {"resolutions": {"<intended value>": "<offered option '
            'or null>"}}. Never answer with an option that is not in that '
            "value's offered list; never invent a facet name."
        )
        try:
            result = cheap_llm(
                system,
                json.dumps({"intended_values": misses}, indent=2),
                expect_json=True,
                usage_context={
                    "stage": "linkedin_geography_facet_resolution",
                    "brief_id": self.deps.get_brief_obj().id,
                },
            )
        except Exception as exc:
            print(f"  [location] Facet resolution call failed: {exc}")
            return None

        raw_resolutions = result.get("resolutions") if isinstance(result, dict) else None
        mapped: dict[str, str] = {}
        for candidate, options in misses.items():
            choice = (raw_resolutions or {}).get(candidate)
            if isinstance(choice, str) and choice.strip() in options:
                mapped[candidate] = choice.strip()
        if not mapped:
            print(
                "  [location] Facet resolution produced no usable mapping "
                f"(offered options: {misses!r})."
            )
            return None

        # Order-preserving dedup: two distinct candidates may correctly
        # resolve to the SAME facet name; re-typing an already-applied facet
        # gate-2-misses and would abort a correct resolution (lens, slice 13).
        final_values = list(dict.fromkeys(mapped.get(value, value) for value in geo_values))
        try:
            chips = await self.deps.get_browser().read_applied_location_chips()
        except Exception:
            chips = []
        applied_chips = {normalize_facet_value_for_compare(chip) for chip in chips}
        to_apply = [
            v
            for v in final_values
            if normalize_facet_value_for_compare(v) not in applied_chips
        ]
        try:
            reapplied = (
                await self.deps.get_browser().apply_location_filter(
                    to_apply, temporal_scope="current_or_past"
                )
                if to_apply
                else True
            )
        except Exception as exc:
            print(f"  [location] Re-apply of resolved facets failed: {exc}")
            return None
        if not reapplied:
            return None

        # Accumulate (merge by candidate), never overwrite — every resolution
        # this session is a recorded fact, including earlier re-assert rounds.
        merged = {
            str(entry.get("candidate", "")): str(entry.get("resolved", ""))
            for entry in self._session_geography_resolutions
        }
        merged.update(mapped)
        self._session_geography_resolutions = [
            {"candidate": candidate, "resolved": resolved}
            for candidate, resolved in sorted(merged.items())
            if candidate
        ]
        for candidate, resolved in sorted(mapped.items()):
            log_event(
                self.deps.log_path,
                "session_location_resolved",
                candidate=candidate,
                resolved=resolved,
            )
        print(
            "  [location] Facet resolution: "
            + "; ".join(f"{c} → {r}" for c, r in sorted(mapped.items()))
        )
        return final_values

    async def _verify_session_geography_chips(self) -> None:
        """P3a pre-string invariant: geography chips must be on the live sidebar
        before ANY opening search is entered.

        A navigation, crash recovery, or LinkedIn-side sidebar reset that silently
        dropped the chips gets ONE re-assert through the fail-closed apply; if the
        chips still are not present, refuse to run a keyword search on an unbounded
        pool. Chip read is subset-based: pills for other facets may coexist.
        """
        from linkedin.orchestrator import GeographyRegimeError

        geo_values = self._session_geography_values()
        if not geo_values:
            return

        async def _missing() -> list[str]:
            try:
                chips = await self.deps.get_browser().read_applied_location_chips()
            except Exception:
                return list(geo_values)
            normalized = {normalize_facet_value_for_compare(c) for c in chips}
            return [
                v
                for v in geo_values
                if normalize_facet_value_for_compare(v) not in normalized
            ]

        missing = await _missing()
        if not missing:
            return
        print(
            f"  [location] Geography chips missing from sidebar ({missing!r}) — re-asserting."
        )
        self._session_location_applied = False
        await self._apply_session_location_filter()  # raises GeographyRegimeError on miss
        missing = await _missing()
        if missing:
            raise GeographyRegimeError(
                f"Geography chips {missing!r} still absent from the sidebar after a "
                "successful re-apply report — refusing to run a keyword search on an "
                "unbounded pool."
            )
        # Receipt accounting for this re-assert already happened inside
        # _apply_session_location_filter (the single owner of the counter).
        log_event(
            self.deps.log_path,
            "session_location_reasserted",
            location="; ".join(geo_values),
        )

    async def _reassert_session_location_after_recovery(self) -> None:
        """Re-apply the brief's session geography after a browser-crash recovery.

        The P6 crash-recovery flow (recover_recruiter_context) re-binds the tab and
        re-navigates the search surface, which drops the sidebar location chip. The
        recovery snapshot cannot carry the session location back — it rides
        apply_location_filter (which zeroes _last_search_snapshot), not
        apply_advanced_search_plan (the only snapshot writer) — so the replay plan
        dims to keywords-only and the resumed search would otherwise lose its
        geography and return an over-broad, location-unbounded result set.

        Mirrors the legacy check_and_recover re-assert in _ensure_browser_healthy
        (reset the idempotency flag, then call the shared apply).

        P3a: fail-CLOSED at this layer too. A recovery that cannot restore the
        session geography must not resume the search — resuming boolean-only is
        exactly the off-geo pool the gate exists to prevent. The
        GeographyRegimeError propagates and stops the run; a resume can pick up
        after the operator fixes the facet values (or the seat state).
        """
        self._session_location_applied = False
        await self._apply_session_location_filter()

    def _warn_if_off_geo_save(self, snippet: "CandidateSnippet") -> None:
        """P3a defense-in-depth telemetry — WARN only, never enforcement.

        The fail-closed geography gate is the enforcement; this catches
        exactly one residual class: LinkedIn applying the facet but returning
        off-facet results. Deliberately loose token heuristic (free-text
        snippet locations vs facet names make exact comparison meaningless):
        warn when the saved candidate's location shares no significant token
        with any session geography value.
        """
        geo_values = self._session_geography_values()
        location = str(getattr(snippet, "location", "") or "").strip()
        if not geo_values or not location:
            return

        def _tokens(text: str) -> set[str]:
            return {
                token
                for token in _re.findall(r"[a-z]{4,}", text.lower())
                if token not in self._OFF_GEO_STOP_TOKENS
            }

        location_tokens = _tokens(location)
        if not location_tokens:
            return
        if any(location_tokens & _tokens(value) for value in geo_values):
            return
        # Containment check (2026-07-04 SPL run): the token heuristic cannot
        # see metro membership — it flagged "Mountain View, California"
        # against "San Francisco Bay Area", a candidate the facet itself had
        # correctly admitted. One cheap model call resolves containment;
        # True suppresses the false positive, None (call failed or
        # non-conforming) degrades to the pre-check behavior with an honest
        # "unverified" marker — over-sensitive under outage, never blind.
        contained = self._candidate_location_contained(location, geo_values)
        if contained is True:
            return
        suffix = "" if contained is False else " (containment unverified)"
        self.deps.stats["off_geo_saves"] = int(self.deps.stats.get("off_geo_saves", 0) or 0) + 1
        print(
            f"    [geo-warn] Saved candidate location {location!r} shares no "
            f"token with session geography {geo_values!r} — defense-in-depth "
            f"telemetry only; the fail-closed gate is the enforcement.{suffix}"
        )

    def _candidate_location_contained(
        self, location: str, geo_values: list[str]
    ) -> bool | None:
        """Is the candidate location geographically inside any session facet?

        One cheap_llm call per distinct (location, facets) pair per session —
        the cache stores failures too, so a provider outage costs one retry
        cycle, not one per off-token save. Returns True/False on a
        conforming verdict, None for anything else (exception, wrong shape),
        which the caller treats as unverified.
        """
        cache = getattr(self, "_geo_containment_cache", None)
        if cache is None:
            cache = self._geo_containment_cache = {}
        key = (location.lower(), tuple(geo_values))
        if key in cache:
            return cache[key]

        verdict: bool | None = None
        try:
            from shared.llm_clients import cheap_llm

            result = cheap_llm(
                "You judge geographic containment for recruiter location "
                "facets. Answer whether the candidate location is inside any "
                "of the listed LinkedIn location facets — a city inside a "
                "metropolitan-area or state facet counts as contained. "
                'Respond with ONLY JSON: {"contained": true} or '
                '{"contained": false}.',
                json.dumps(
                    {"candidate_location": location, "facets": geo_values},
                    indent=2,
                ),
                expect_json=True,
                usage_context={
                    "stage": "linkedin_off_geo_containment",
                    "brief_id": self.deps.get_brief_obj().id,
                },
            )
            if isinstance(result, dict) and isinstance(
                result.get("contained"), bool
            ):
                verdict = result["contained"]
        except Exception as exc:  # noqa: BLE001 — telemetry stays fail-soft
            print(f"    [geo-warn] containment check unavailable: {str(exc)[:120]}")

        cache[key] = verdict
        return verdict
