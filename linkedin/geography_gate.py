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
from linkedin.auth_wall import AuthWallError, auth_wall_variant_of, raise_if_auth_wall
from linkedin.browser import normalize_facet_value_for_compare
from shared import config
from shared.storage import log_event

if TYPE_CHECKING:
    from shared.brief_loader import Brief
    from shared.schemas import CandidateSnippet


# Allow enough polling for the facet rail to settle after each removal.
# The same settle budget applies to every chip cleared by this service.
_FACET_SETTLE_ATTEMPTS = 21
_FACET_SETTLE_INTERVAL_MS = 120


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
        """Return the session geography as exact Recruiter facet values.

        A conjunction string must resolve to real typeahead facet values before
        it can be applied. For preflight-sourced values, the resolved list is
        the session geography; chip checks and re-assertions verify those
        applied values rather than the raw candidates they replaced.
        """
        if self._resolved_session_geography is not None:
            return list(self._resolved_session_geography)
        geo = str(
            self.deps.get_brief_obj().permanent_filters.get("Location", "") or ""
        ).strip()
        return [v.strip() for v in geo.split(";") if v.strip()]

    async def _apply_session_location_filter(self, *, allow_resolution: bool = True) -> None:
        """Apply and verify the brief's geography as a fail-closed precondition.

        Location is a session-level fact from the brief, not a per-lane
        hypothesis, so this service applies it once rather than threading it
        through each lane. ``_session_location_applied`` makes the operation
        idempotent within a session; callers reset it after navigation drops
        the sidebar chip.

        The browser layer verifies exact typeahead matches and applied chips.
        A miss raises ``GeographyRegimeError`` here instead of degrading to
        an unbounded pool. Preflight-sourced values may take the one allowed
        model-resolution path; operator-pinned values retain strict failure.
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
            # A sign-in wall is an environment fault, not a facet-name fault:
            # it is re-raised as itself, never wrapped into the terminal
            # "fix your facet values" stop. A generic failure raised ON a
            # wall page is named from the live page first.
            if isinstance(exc, AuthWallError):
                raise
            await raise_if_auth_wall(self.deps.get_browser(), cause=exc)
            if _is_browser_disconnect_error(exc) or auth_wall_variant_of(exc):
                raise
            raise GeographyRegimeError(
                f"Session geography {geo_values!r} failed to apply: {exc!r}. "
                "Aborting rather than searching an unbounded pool — each value must "
                "be an exact LinkedIn Recruiter location facet name."
            ) from exc
        if not applied:
            # A missing editor or unreadable chips on a page that is no longer
            # Recruiter is a sign-in wall, not a typeahead miss: name it before
            # any resolution types into the page or a retry starts a session.
            await raise_if_auth_wall(self.deps.get_browser())
            # Preflight-sourced candidates get one model resolution against the
            # real typeahead options. Operator-pinned values keep strict abort:
            # the operator supplied exact facet names, so they are never rewritten.
            resolved_values: list[str] | None = None
            if allow_resolution and getattr(self.deps.get_brief_obj(), "geography_source", "") == "preflight":
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
        # Record verified application in the durable run receipt, not the
        # transient idempotence flag, which resets after navigation. This method
        # owns the reassert counter: count every successful re-apply after the
        # first exactly once, including recovery and resume paths.
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
        # Record each candidate-to-facet resolution as a durable receipt fact.
        if self._session_geography_resolutions:
            self._session_geography_receipt["resolutions"] = [
                dict(entry) for entry in self._session_geography_resolutions
            ]
        log_event(self.deps.log_path, "session_location_applied", location="; ".join(geo_values))

    async def _resolve_and_reapply_geography(
        self, geo_values: list[str]
    ) -> list[str] | None:
        """Resolve once against the real typeahead options and re-apply them.

        For each missed value, the browser supplies the options that the
        typeahead actually offered. A single cheap model call may map a value
        only to one of its offered options; an answer outside that list is
        discarded. Confirmed chips are not re-typed, and only missing resolved
        names are applied. Return the final session values after verified
        re-application, or ``None`` so the caller raises
        ``GeographyRegimeError`` and stops the run.
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

        # Preserve input order while deduplicating resolved values. Distinct
        # candidates may map to the same facet; re-typing an already-applied
        # facet can miss the exact-match gate and abort a correct resolution.
        final_values = list(dict.fromkeys(mapped.get(value, value) for value in geo_values))
        try:
            chips = await self.deps.get_browser().read_applied_location_chips()
        except Exception:
            return None
        if chips is None:
            return None
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

        # Merge by candidate and never overwrite: every resolution in this
        # session, including a re-assert, remains a recorded fact.
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

        if config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED:
            geo_values = self._session_geography_values()
            browser = self.deps.get_browser()
            try:
                chips = await browser.read_applied_location_chips()
            except Exception as exc:
                await raise_if_auth_wall(browser, cause=exc)
                raise GeographyRegimeError(
                    "Session geography wrapper is unreadable; refusing to search."
                ) from exc
            if chips is None:
                await raise_if_auth_wall(browser)
                raise GeographyRegimeError(
                    "Session geography wrapper is unreadable; refusing to search."
                )

            normalized = [normalize_facet_value_for_compare(chip) for chip in chips]
            if not all(normalized) or len(normalized) != len(set(normalized)):
                raise GeographyRegimeError(
                    "Session geography contains blank or duplicate-normalized pills; "
                    "refusing to guess which value to clear."
                )
            desired = {
                normalize_facet_value_for_compare(value) for value in geo_values
            }
            if set(normalized) == desired:
                return

            # Each removal re-runs the search and re-renders the rail. Wait for
            # the observed chips to settle to the expected remainder before
            # removing the next one, because the sidebar is stale between removals.
            remaining = list(chips)
            while remaining:
                chip = remaining[0]
                expected = sorted(
                    normalize_facet_value_for_compare(value)
                    for value in remaining[1:]
                )
                settled: list[str] | None = None
                if await browser.remove_facet_value("locations", chip):
                    for attempt in range(_FACET_SETTLE_ATTEMPTS):
                        observed = await browser.read_applied_location_chips()
                        if observed is not None and expected == sorted(
                            normalize_facet_value_for_compare(value)
                            for value in observed
                        ):
                            settled = list(observed)
                            break
                        if attempt < _FACET_SETTLE_ATTEMPTS - 1:
                            await browser.page.wait_for_timeout(
                                _FACET_SETTLE_INTERVAL_MS
                            )
                if settled is None:
                    raise GeographyRegimeError(
                        f"Session geography pill {chip!r} could not be cleared."
                    )
                remaining = settled
            self._session_location_applied = False
            await self._apply_session_location_filter()

            geo_values = self._session_geography_values()
            desired = {
                normalize_facet_value_for_compare(value) for value in geo_values
            }
            try:
                after = await browser.read_applied_location_chips()
            except Exception as exc:
                raise GeographyRegimeError(
                    "Session geography wrapper became unreadable after re-apply."
                ) from exc
            if after is None or {
                normalize_facet_value_for_compare(value) for value in after
            } != desired:
                raise GeographyRegimeError(
                    "Session geography still differs after clear-and-reapply; "
                    "refusing to search."
                )
            log_event(
                self.deps.log_path,
                "session_location_reasserted",
                location="; ".join(geo_values),
            )
            return

        geo_values = self._session_geography_values()
        if not geo_values:
            return

        async def _missing() -> list[str]:
            try:
                chips = await self.deps.get_browser().read_applied_location_chips()
            except Exception:
                return list(geo_values)
            if chips is None:
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
        """Re-apply session geography after browser-crash recovery.

        Recovery re-binds and re-navigates the tab, which drops the sidebar
        location chip. The recovery snapshot cannot restore it: session location
        rides ``apply_location_filter``, which clears ``_last_search_snapshot``,
        and ``apply_advanced_search_plan`` is the only snapshot writer. Reset the
        idempotence flag and use the shared apply path so a resumed search cannot
        fall back to a location-unbounded pool. Failure remains fail-closed:
        ``GeographyRegimeError`` stops the run until the facet or browser state
        is fixed.
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
        # Token overlap cannot represent metro containment. Ask the cheap model
        # for a containment verdict: True suppresses the warning; False keeps
        # the warning; None means failure or non-conforming output and keeps the
        # pre-check warning with an ``unverified`` marker. Outages stay
        # over-sensitive rather than blind.
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
