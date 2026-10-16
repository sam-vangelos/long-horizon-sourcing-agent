"""Unified candidate preparation, judgment, settlement and bounded FIFO review."""
from __future__ import annotations
from linkedin.evaluation_support import _is_browser_disconnect_error
import asyncio
import math
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from linkedin.page_review import PageReviewContext
from shared.execution import SideEffectOutcome
from shared.human_timing import human_delay_correlated
from shared.schemas import (
    CandidateSnippet,
    CandidateProfileSummary,
    OpusDecision,
    SearchString,
    Progress,
)
from linkedin.judgment_tool_contracts import generate_opaque_candidate_ids
from shared.failures import (
    ApiBudgetExhaustedError,
    is_api_budget_exhausted_error,
    judgment_failure_decision,
    parse_failure_decision,
)
from shared.contracts import LEGACY_REVIEW_DECISIONS
from shared.judger import (
    SAVE_FAMILY_DECISIONS,
    facial_judge,
    full_id_mismatch_retry_context,
    full_judge,
    full_judge_with_external_evidence,
    is_failure_decision,
)
from shared.external_evidence import (
    fetch_external_candidate_evidence,
    should_request_external_evidence,
)
from shared.external_evidence.shadow_writer import (
    ShadowFullJudgmentRecord,
    compute_judgment_diff,
    record_shadow_full_judgment,
)
from shared.schemas import ExternalCandidateEvidence
from shared.employer_blacklist import match_any_employer, match_current_employer
from shared.employer_allowlist import (
    EMPLOYER_SCOPE_GATE_PATH,
    current_employer_scope,
    employer_allowlist_from_brief,
    present_entry_company,
    present_entry_title,
)
from shared.excluded_locations import TERRITORY_GATE_PATH, excluded_location_match
from shared.storage import log_event
from shared.bias_controls import DecisionRecord
from shared.identity_resolution import classify_recruiter_activity_pressure, infer_reachout_status
from shared.reconciliation_schemas import RecruiterActivitySnapshot
from linkedin.run_report import _PageReport
from shared import config
from shared.governor import GovernorLimitReached, OperatorStopRequested, SessionExpired
from linkedin.evaluation_errors import (
    GeographyRegimeError,
    PanelRecoveryError,
    SurfaceOwnershipLostError,
    UnconfirmedSaveError,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING
from linkedin.evaluation_context import EvaluationContext
from linkedin.candidate_ledger import CandidateLedger
from linkedin.search_surface import SearchSurface
from linkedin.allocator_state import AllocatorStateService
from linkedin.acquisition import LinkedInAcquisitionService, LinkedInAcquisitionDeps
from linkedin.side_effects import LinkedInSideEffectsService, LinkedInSideEffectsDeps
if TYPE_CHECKING:
    from linkedin.evaluation_state import PageContext
    from linkedin.evaluation_cascade import CascadeDecision, ProfileProbe
    from linkedin.runtime_attempts import RuntimeAttemptService
from linkedin.evaluation_support import (
    _DEFAULT_READ_INTEREST,
    _FACIAL_READ_INTEREST,
    _FULL_ID_MISMATCH_ACTUAL_ID_MAX_CHARS,
    _FULL_ID_MISMATCH_RECOVERY_CEILING,
    _clear_review_evidence,
    _is_containable_facial_page_error,
    is_legacy_review_decision,
)

# How each deterministic facial gate names itself on the console, in the
# rationale, and in run stats. Every gate decides with no model call.
EMPLOYER_GATE_PATH = "employer_blacklist"
ALUMNI_GATE_PATH = "alumni_blacklist"
# The page outcome a territory skip records instead of facial_no. The canonical
# stage row stays FACIAL_NO (terminal, deduped), but the adaptation layer reads
# page outcomes: facial_no and reject become noise anchors, NOT terms for
# precision rescues, and the early-exit rate. A card in another recruiter's
# market says nothing about whether the string's vocabulary finds fits.
TERRITORY_SKIP_OUTCOME = "territory_skip"
# A brief's employer_allowlist (shared/employer_allowlist.py,
# EMPLOYER_SCOPE_GATE_PATH): a person whose CURRENT employer is identified and
# not listed is out of the campaign's scope. Not a verdict on fit either, for
# the territory gate's reason: the Companies facet matches current OR past
# employees, so a competitor string surfaces that competitor's alumni, the
# right seat at the wrong current employer. Counted as facial_no or reject they
# would read a working vocabulary as noise and spend the mutation budget
# rewriting it. The skip keeps its canonical stage row (FACIAL_NO on the card,
# REJECT/HARD_GATE on the profile) and records this page outcome instead.
EMPLOYER_SCOPE_SKIP_OUTCOME = "employer_scope_skip"
# Gate path -> the page outcome a skip on it records, for the gates whose skips
# stay out of the facial and full funnel counters (exhaustion, family yield,
# and the per-page stats the noise and drift triggers read).
_NON_VERDICT_GATE_OUTCOMES = {
    TERRITORY_GATE_PATH: TERRITORY_SKIP_OUTCOME,
    EMPLOYER_SCOPE_GATE_PATH: EMPLOYER_SCOPE_SKIP_OUTCOME,
}
NON_VERDICT_GATE_PATHS = frozenset(_NON_VERDICT_GATE_OUTCOMES)
EMPLOYER_GATE_PATHS = frozenset(
    {EMPLOYER_GATE_PATH, ALUMNI_GATE_PATH, TERRITORY_GATE_PATH, EMPLOYER_SCOPE_GATE_PATH}
)
_FACIAL_GATE_LABELS = {
    EMPLOYER_GATE_PATH: ("BLACKLIST", "Employer blacklist", "blacklist_skips"),
    ALUMNI_GATE_PATH: ("ALUMNI-BLACKLIST", "Alumni blacklist", "alumni_blacklist_skips"),
    TERRITORY_GATE_PATH: ("TERRITORY", "Excluded location", "excluded_location_skips"),
    EMPLOYER_SCOPE_GATE_PATH: (
        "OUT-OF-SCOPE",
        "Outside the competitor scope",
        "employer_scope_skips",
    ),
}


def facial_gate_labels(path: str) -> tuple[str, str, str]:
    """``(console_tag, rationale_label, stats_key)`` for a facial gate path."""

    return _FACIAL_GATE_LABELS[path]


def gate_skip_outcome(path: object) -> str | None:
    """The page outcome a non-verdict gate's skip records, else None."""

    return _NON_VERDICT_GATE_OUTCOMES.get(str(path or ""))


def facial_gate_console_line(
    path: str, name: str, blocked: str, matched_field: str, matched_value: str
) -> str:
    """The one console line a deterministic facial gate prints for a card."""

    tag = facial_gate_labels(path)[0]
    if path == EMPLOYER_SCOPE_GATE_PATH and matched_field == "current_title":
        return (
            f"    [{tag}] {name} — current seat '{matched_value}' is outside the "
            "client-side title families (current_title)"
        )
    if path == EMPLOYER_SCOPE_GATE_PATH:
        return (
            f"    [{tag}] {name} — current employer '{matched_value}' is not a "
            f"listed competitor ({matched_field})"
        )
    return f"    [{tag}] {name} — '{matched_value}' matches '{blocked}' ({matched_field})"


@dataclass
class PreparedCandidate:
    """One acquired full attempt, owned by preparation until judgment starts."""
    snippet: CandidateSnippet
    runtime_search_string: SearchString
    attempt_id: int | None
    attempt_payload: dict[str, Any]
    summary: CandidateProfileSummary
    profile_status_summary: dict[str, Any]
    probe_decision: CascadeDecision
    lane_context: dict[str, Any]
    panel_released: bool

@dataclass(frozen=True)
class JudgedCandidate:
    """A paid worker's result; canonical settlement stays on the loop task."""
    decision: OpusDecision
    elapsed_ms: float


class CandidateReviewer:
    """Candidate state and services; no loop, progress or experiment-map alias."""

    _profile_probe: ProfileProbe
    _profile_read_interest: dict[str, float]
    _full_contract_corruption_call_ids: set[str]
    _full_eval_consecutive_faults: int
    _full_id_mismatch_recovered_count: int
    _triage_tightened: bool
    _tightening_prefix: str

    def __init__(
        self,
        context: EvaluationContext,
        ledger: CandidateLedger,
        surface: SearchSurface,
        allocator: AllocatorStateService,
        page: PageContext,
    ):
        self.context = context
        self.ledger = ledger
        self.surface = surface
        self.allocator = allocator
        self.page = page
        self.acquisition: LinkedInAcquisitionService | None = None
        self.side_effects: LinkedInSideEffectsService | None = None
        self.attempts: RuntimeAttemptService

    def ensure_services(self) -> None:
        self.ensure_acquisition()
        self.ensure_side_effects()

    def ensure_acquisition(self) -> None:
        if self.acquisition is None:
            self.acquisition = LinkedInAcquisitionService(LinkedInAcquisitionDeps(
                browser=self.context.browser, log_path=self.context.log_path,
                ensure_browser_healthy=lambda: self.surface._ensure_browser_healthy(),
            ))

    def ensure_side_effects(self) -> None:
        if self.side_effects is None:
            self.side_effects = LinkedInSideEffectsService(LinkedInSideEffectsDeps(
                browser=self.context.browser, stats=self.context.stats,
                saved_urls=self.ledger._saved_urls, log_path=self.context.log_path,
                get_test_mode=lambda: self.context.test_mode,
                get_runtime_bridge=lambda: self.context._runtime_bridge,
                get_runtime_run_id=lambda: self.context._runtime_run_id,
                before_irreversible_side_effect=self.surface._honor_irreversible_side_effect_boundary,
                assert_project_context=self.surface._assert_brief_project_context,
            ))

    def _note_page_full_review_expected(
        self, snippet: CandidateSnippet, *, page_context: PageReviewContext | None = None,
    ) -> None:
        """Count a page-local facial positive when full review is scheduled."""

        (page_context or self.page.review).expect(snippet, self.ledger._funnel_candidate_key(snippet))


    def _note_page_full_review_settled(
        self,
        *,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        page_context: PageReviewContext | None = None,
    ) -> None:
        """Count canonical terminal currency for the exact open page only."""

        (page_context or self.page.review).settle(snippet, decision, self.ledger._funnel_candidate_key(snippet))


    async def _perform_full_save(
        self, prepared: PreparedCandidate, decision: OpusDecision,
    ) -> SideEffectOutcome:
        """Retain the judgment/attempt; an inconclusive save can only be probed."""
        confirmation_only = False
        while True:
            try:
                await self.surface._reopen_profile_for_full_eval_save(prepared.snippet)
            except PanelRecoveryError as error:
                if await self.surface._try_page_recovery(
                    owner=prepared.runtime_search_string, page_num=prepared.snippet.page,
                    fault="panel", error=error,
                ):
                    continue
                raise
            outcome = await self.side_effects.handle_save_decision(
                snippet=prepared.snippet,
                runtime_search_string=prepared.runtime_search_string,
                attempt_id=prepared.attempt_id, decision=decision,
                **({"confirmation_only": True} if confirmation_only else {}),
            )
            payload = outcome.payload or {}
            confirmed = (
                outcome.status == "succeeded" or payload.get("already_present")
                or (outcome.status == "skipped" and payload.get("skip_reason") == "existing_succeeded")
            )
            if confirmed or payload.get("failure_reason") not in {
                None, "save_not_confirmed", "save_not_persisted", "save_probe_failed",
                "save_card_not_found", "save_card_ambiguous", "save_trigger_not_found",
            } or (outcome.status == "skipped"):
                return outcome
            if not await self.surface._try_page_recovery(
                owner=prepared.runtime_search_string, page_num=prepared.snippet.page,
                fault="save confirmation",
            ):
                return outcome
            confirmation_only = True


    def _brief_signals_for_profile_probe(self) -> dict[str, list[str]]:
        # The shadow probe needs the V2 capability-area NAMES and non-fit LABELS.
        # Those live on self.brief_obj._new_brief (shared/brief_schema.Brief), NOT
        # on the compat Brief — reading them off self.brief_obj directly via
        # getattr silently yields empties and forces every candidate to
        # a no-signal probe verdict. Use the typed, fail-loud accessor
        # instead.
        return self.context.brief_obj.sourcing_signals()


    @staticmethod
    def _employer_blacklist_match(
        snippet: CandidateSnippet,
        employer_blacklist: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, str, str] | None:
        if not employer_blacklist:
            return None
        # headline+current only; past employers are the alumni gate's job.
        return match_current_employer(
            snippet.current_company, snippet.headline, employer_blacklist
        )

    @staticmethod
    def _alumni_blacklist_match(
        snippet: CandidateSnippet,
        alumni_blacklist: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, str, str] | None:
        """Snippet-stage alumni check: current company, headline, card history.

        The card shows only the first few positions, so this catches what the
        snippet can see; :meth:`_profile_alumni_blacklist_match` re-checks the
        full work history before any judgment or save.
        """

        if not alumni_blacklist:
            return None
        return match_any_employer(
            alumni_blacklist,
            current_company=getattr(snippet, "current_company", "") or "",
            headline=getattr(snippet, "headline", "") or "",
            experience_entries=getattr(snippet, "experience_entries", None) or (),
        )

    @staticmethod
    def _profile_alumni_blacklist_match(
        snippet: CandidateSnippet,
        summary: CandidateProfileSummary,
        alumni_blacklist: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, str, str] | None:
        """Post-profile alumni check over the full work history."""

        if not alumni_blacklist:
            return None
        return match_any_employer(
            alumni_blacklist,
            current_company=getattr(snippet, "current_company", "") or "",
            headline=summary.headline or getattr(snippet, "headline", "") or "",
            experience_companies=[
                experience.company for experience in (summary.experiences or [])
            ],
            # Card lines plus position titles, which match only where they
            # name the employer ("Director at X"), never as a bare mention.
            experience_entries=[
                *(getattr(snippet, "experience_entries", None) or ()),
                *(experience.title for experience in (summary.experiences or [])),
            ],
            about=summary.about or "",
        )

    def _facial_employer_gate(
        self, snippet: CandidateSnippet,
    ) -> tuple[str, tuple[str, str, str]] | None:
        """``(path, (blocked, field, value))`` when a deterministic gate fires.

        The current-employer blacklist is checked first so a current employee
        keeps its historical path stamp; the alumni blacklist then catches
        anyone the snippet shows has EVER worked at a listed employer.
        """

        brief = self.context.brief_obj
        hit = self._employer_blacklist_match(snippet, brief.employer_blacklist)
        if hit:
            return EMPLOYER_GATE_PATH, hit
        hit = self._alumni_blacklist_match(
            snippet, getattr(brief, "alumni_blacklist", None)
        )
        if hit:
            return ALUMNI_GATE_PATH, hit
        # The seed field rides in brief.raw on the seed and the generated brief.
        raw = getattr(brief, "raw", None)
        excluded = raw.get("excluded_locations") if isinstance(raw, dict) else None
        location = getattr(snippet, "location", "") or ""
        entry = excluded_location_match(location, excluded if isinstance(excluded, list) else None)
        if entry:
            return TERRITORY_GATE_PATH, (entry, "location", location)
        hit = self._employer_scope_match(snippet, None, brief)
        if hit:
            return EMPLOYER_SCOPE_GATE_PATH, hit
        return None

    @staticmethod
    def _employer_scope_match(
        snippet: CandidateSnippet,
        summary: CandidateProfileSummary | None,
        brief: Any,
    ) -> tuple[str, str, str] | None:
        """``(employer, field, value)`` when the brief's allowlist puts a person out of scope.

        The card reads its current company and every history line marked
        Present; the full profile adds every position whose end is Present.
        Any listed employer among them keeps the person in scope, and with no
        current employer identified the judge decides. A title-restricted
        employer (a client's data side) keeps the person only when a current
        title or the headline is in its title families; the hit then names
        the seat (field ``current_title``). Inert without a well-formed
        ``employer_allowlist`` on ``brief.raw``.
        """

        allowlist = employer_allowlist_from_brief(brief)
        if allowlist is None:
            return None
        companies: list[str] = [str(getattr(snippet, "current_company", "") or "")]
        titles: list[str] = [str(getattr(snippet, "current_title", "") or "")]
        for entry in getattr(snippet, "experience_entries", None) or ():
            company = present_entry_company(entry)
            if company:
                companies.append(company)
                titles.append(present_entry_title(entry) or "")
        headlines = [str(getattr(snippet, "headline", "") or "")]
        field_name = "current_company"
        if summary is not None:
            field_name = "current_position"
            for experience in summary.experiences or []:
                end = str(getattr(experience, "end", "") or "").strip().casefold()
                if end in {"present", "current", "now"} and experience.company:
                    companies.append(str(experience.company))
                    titles.append(str(experience.title or ""))
            headlines.append(str(summary.headline or ""))
        verdict = current_employer_scope(
            companies=companies,
            headlines=headlines,
            allowlist=allowlist,
            company_field=field_name,
            titles=titles,
        )
        if verdict.in_scope is not False:
            return None
        if verdict.field == "current_title":
            return verdict.value, verdict.field, verdict.value
        employer = verdict.value.split("·", 1)[0].strip()
        return employer, verdict.field, employer

    async def _settle_profile_alumni_gate(
        self,
        prepared: PreparedCandidate,
        hit: tuple[str, str, str],
        *,
        page_report: "_PageReport | None",
        page_context: PageReviewContext | None,
        gate_path: str = ALUMNI_GATE_PATH,
    ) -> OpusDecision:
        """Settle a full-profile gate hit as a REJECT with no judgment.

        The alumni blacklist and the employer-scope gate both settle here.
        The same canonical writes a judged REJECT takes (terminal stage row,
        prior outcome, page settlement), then the panel closes. No judge
        call, no save click, no note: the candidate never reaches either.
        """

        snippet = prepared.snippet
        blocked, matched_field, matched_value = hit
        if gate_path == EMPLOYER_SCOPE_GATE_PATH and matched_field == "current_title":
            rationale = (
                f"Outside the competitor scope: current seat {matched_value} is "
                "outside the client-side title families"
            )
            console = (
                f"    [OUT-OF-SCOPE] {snippet.name} — profile current seat "
                f"'{matched_value}' is outside the client-side title families; "
                "rejected before judgment"
            )
            stat_name = "employer_scope_rejects"
        elif gate_path == EMPLOYER_SCOPE_GATE_PATH:
            rationale = (
                f"Outside the competitor scope: current employer {matched_value} "
                f"({matched_field}) is not a listed competitor"
            )
            console = (
                f"    [OUT-OF-SCOPE] {snippet.name} — profile {matched_field} "
                f"'{matched_value}' is not a listed competitor; rejected before judgment"
            )
            stat_name = "employer_scope_rejects"
        else:
            rationale = f"Alumni blacklist: {blocked} ({matched_field}: {matched_value})"
            console = (
                f"    [ALUMNI-BLACKLIST] {snippet.name} — profile "
                f"{matched_field} '{matched_value}' matches '{blocked}'; "
                "rejected before judgment"
            )
            stat_name = "alumni_blacklist_rejects"
        decision = OpusDecision(
            stage="full",
            decision="REJECT",
            path=gate_path,
            confidence=1.0,
            rationale=rationale,
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
            reject_reason="HARD_GATE",
        )
        # Canonical closure first, in the order a judged REJECT takes it, so
        # nothing below can strand the open attempt.
        self.ledger.record_prior_outcome(snippet.profile_url, decision.decision)
        self.ledger._mark_terminal(snippet.profile_url)
        self._finish_runtime_stage_success(
            attempt_id=prepared.attempt_id,
            stage="full",
            snippet=snippet,
            decision=decision,
            profile_summary=prepared.summary,
            extra_payload={
                **prepared.attempt_payload,
                "gate": gate_path,
                "gate_match": {
                    "blocked": blocked,
                    "field": matched_field,
                    "value": matched_value,
                },
            },
            page_context=page_context,
        )
        print(console)
        event_fields = {
            "name": snippet.name,
            "profile_url": snippet.profile_url,
            "string_id": prepared.runtime_search_string.id,
            "blocked": blocked,
            "field": matched_field,
            "value": matched_value,
        }
        if gate_path == EMPLOYER_SCOPE_GATE_PATH:
            log_event(self.context.log_path, "employer_scope_rejected", **event_fields)
        else:
            log_event(self.context.log_path, "alumni_blacklist_rejected", **event_fields)
        self.context.stats.setdefault(stat_name, 0)
        self.context.stats[stat_name] += 1
        self.context.stats["rejected"] = self.context.stats.get("rejected", 0) + 1
        if page_report:
            page_report.add_skipped_opened(snippet, decision)
        if not prepared.panel_released:
            try:
                await self.context.browser.go_back_to_results()
                await asyncio.sleep(
                    human_delay_correlated(
                        config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                        channel="panel_close",
                    )
                )
            except Exception as e:
                if _is_browser_disconnect_error(e):
                    print(f"    [ERROR] Browser session dropped while closing profile: {e}")
                    log_event(self.context.log_path, "panel_close_browser_disconnect", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                    raise
                print("    [WARN] Profile panel close missed; attempting recovery")
                log_event(self.context.log_path, "go_back_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                decision._panel_stuck = True
        return decision


    def _full_judgment_fault_disposition(self, error: Exception) -> str:
        """Describe how the full-judgment handler treats a fault.

        ``contain`` settles this candidate as ``JUDGMENT_FAILURE`` and keeps the
        string alive. ``ceiling`` treats repeated provider faults as an outage
        and ends the session legibly. ``raise`` preserves fail-fast behavior for
        non-containable faults or when containment is disabled. Only containable
        faults advance the counter; a successful judgment resets it, so the
        ceiling measures consecutive faults.
        """
        if not config.LINKEDIN_FULL_EVAL_CONTAINMENT_ENABLED:
            return "raise"
        if not _is_containable_facial_page_error(error):
            return "raise"
        count = getattr(self, "_full_eval_consecutive_faults", 0) + 1
        self._full_eval_consecutive_faults = count
        ceiling = max(1, int(config.LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS))
        return "ceiling" if count >= ceiling else "contain"


    def _finish_runtime_stage_success(
        self,
        *,
        attempt_id: int | None,
        stage: str,
        snippet: CandidateSnippet,
        decision: OpusDecision,
        profile_summary: CandidateProfileSummary | None = None,
        extra_payload: dict | None = None,
        page_context: PageReviewContext | None = None,
    ) -> None:
        self.attempts._finish_runtime_stage_success(
            attempt_id=attempt_id,
            stage=stage,
            snippet=snippet,
            decision=decision,
            profile_summary=profile_summary,
            extra_payload=extra_payload,
        )

        if stage == "full":
            self.ledger._record_outreach_tier_outcome(snippet=snippet, decision=decision, stats=self.context.stats)
            self._note_page_full_review_settled(
                snippet=snippet, decision=decision, page_context=page_context,
            )


    def _finish_runtime_stage_failure(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        error: Exception,
        stage: str = "full",
        payload: dict | None = None,
    ) -> None:
        self.attempts._finish_runtime_stage_failure(
            attempt_id=attempt_id,
            snippet=snippet,
            error=error,
            stage=stage,
            payload=payload,
        )

        if stage == "full":
            self.ledger.retain_failed_full(snippet)


    def _abort_runtime_stage_attempt(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        error: BaseException,
        payload: dict | None = None,
    ) -> None:
        self.attempts._abort_runtime_stage_attempt(
            attempt_id=attempt_id,
            snippet=snippet,
            error=error,
            payload=payload,
        )

        if str((payload or {}).get("stage") or "full") == "full":
            self.ledger.retain_failed_full(snippet)
        self.ledger.release_review(snippet.profile_url)


    def _propagate_profile_activity_abort(
        self,
        *,
        attempt_id: int | None,
        snippet: CandidateSnippet,
        error: Exception,
        payload: dict,
    ) -> None:
        """Propagate run-level controls from optional profile enrichment."""

        normalized_error: Exception = error
        if isinstance(error, OperatorStopRequested):
            abort_reason = "operator_stop"
        elif isinstance(error, SessionExpired):
            abort_reason = "session_expired"
        elif isinstance(error, GovernorLimitReached):
            abort_reason = "governor_limit"
        elif isinstance(error, GeographyRegimeError):
            abort_reason = "geography_regime"
        elif _is_browser_disconnect_error(error):
            abort_reason = "browser_disconnect"
        elif is_api_budget_exhausted_error(error):
            abort_reason = "api_budget_exhausted"
            if not isinstance(error, ApiBudgetExhaustedError):
                normalized_error = ApiBudgetExhaustedError(str(error))
        else:
            return

        self._abort_runtime_stage_attempt(
            attempt_id=attempt_id,
            snippet=snippet,
            error=normalized_error,
            payload={
                **payload,
                "run_abort": abort_reason,
                "stage": "full",
            },
        )
        if normalized_error is not error:
            raise normalized_error from error
        raise error


    def _facial_ternary_enabled(self) -> bool:
        """Whether this run triages on the ternary facial templates.

        The ambiguity posture is brief content
        (``Brief.facial_ambiguity_posture``, preflight-set), with the
        ``LINKEDIN_FACIAL_BORDERLINE_ENABLED`` config flag as the fallback for
        briefs that do not set it — the same resolution the template selector
        uses, so BORDERLINE handling here can never disagree with the prompt
        that was actually assembled.
        """
        from linkedin.judgment_templates import _facial_ternary_selected

        new_brief = getattr(self.context.brief_obj, "_new_brief", None)
        if new_brief is not None:
            return _facial_ternary_selected(new_brief)
        return bool(config.LINKEDIN_FACIAL_BORDERLINE_ENABLED)


    def _facial_prefix_for_string(self, search_string: SearchString) -> str:
        """Compose tightening with the ternary-only retrieval prior."""
        implication = search_string.evidence_implication.strip()
        if not (
            config.LINKEDIN_SNIPER_PRIOR_ENABLED
            and search_string.precision_class == "sniper"
            and implication
            and self._facial_ternary_enabled()
        ):
            return self._tightening_prefix
        if not implication.endswith((".", "!", "?")):
            implication += "."
        return self._tightening_prefix + (
            "RETRIEVAL PRIOR — PRECISION STRING: These candidates were retrieved by a precision search whose required terms only genuine practitioners carry. "
            f"Appearing in these results is itself evidence: {implication} No other search string re-finds these candidates, so the usual economics of skipping an ambiguous snippet — that another string will surface them — do NOT apply here. "
            "A snippet that is merely sparse or uninformative is NOT a FACIAL_NO: floor it at FACIAL_BORDERLINE so full review can resolve it. FACIAL_NO still applies on affirmative disqualifiers — an entire career in a different line of work, a blacklisted employer, or clearly below the experience floor.\n\n"
        )


    def _stamp_read_interest(self, snippet, raw_decision: str) -> None:
        """Record the read-budget hint from the RAW facial verdict."""
        interest = _FACIAL_READ_INTEREST.get(raw_decision)
        if interest is not None and snippet.profile_url:
            self._profile_read_interest[snippet.profile_url] = interest


    def _read_interest_for(self, snippet) -> float:
        """Interest hint for a profile read; medium when unstamped (e.g. resume)."""
        return self._profile_read_interest.get(
            snippet.profile_url, _DEFAULT_READ_INTEREST
        )


    def _normalize_facial_decision_for_persistence(self, facial: OpusDecision) -> OpusDecision:
        """Preserve ternary BORDERLINE; fail loud under a binary posture.

        When the ternary posture is active for this brief, a returned
        ``FACIAL_BORDERLINE`` is canonical evidence that the snippet could
        not resolve fit and therefore demands full-profile review. Keep it
        distinct through persistence, counters, and adaptation telemetry.

        Under the binary posture, a returned ``FACIAL_BORDERLINE`` is a
        structural surprise (the binary prompt should not produce
        BORDERLINE). Convert to ``PARSE_FAILURE`` and let the standard
        non-terminal failure path handle it. Fail loud, do not silently
        coerce to YES or NO.

        For all non-borderline decisions this is the identity function so
        callers can wrap unconditionally.
        """
        if facial.decision != "FACIAL_BORDERLINE":
            return facial

        if self._facial_ternary_enabled():
            return facial

        failure = parse_failure_decision(
            stage=facial.stage,
            candidate_name=facial.candidate_name,
            profile_url=facial.profile_url,
            detail=(
                "Facial parser emitted FACIAL_BORDERLINE under the binary "
                "ambiguity posture. This is a structural surprise (the "
                "binary prompt should not produce BORDERLINE). "
                f"Original rationale: {facial.rationale}"
            ),
            reason="facial_borderline_under_flag_off",
            path=facial.path,
        )
        failure.prompt_capture = dict(facial.prompt_capture or {})
        return failure


    @staticmethod
    def _is_recoverable_provider_failure_decision(
        decision: OpusDecision | None,
    ) -> bool:
        """Recognize standardized retryable provider/network exhaustion."""

        return bool(
            decision is not None
            and decision.decision == "JUDGMENT_FAILURE"
            and "[judgment_failure: recoverable_error/"
            in str(decision.rationale or "").lower()
        )


    @staticmethod
    def _activity_snapshot_from_sources(
        snippet: CandidateSnippet,
        profile_status_summary: dict | None = None,
    ) -> RecruiterActivitySnapshot | None:
        if isinstance(profile_status_summary, dict) and profile_status_summary:
            return RecruiterActivitySnapshot.from_dict(profile_status_summary)
        return snippet.recruiter_activity


    @staticmethod
    def _activity_summary_text(activity: RecruiterActivitySnapshot | None) -> str:
        if not activity:
            return ""
        parts: list[str] = []
        if activity.message_count:
            parts.append(f"{activity.message_count} messages")
        if activity.project_count:
            parts.append(f"{activity.project_count} projects")
        if activity.view_count:
            parts.append(f"{activity.view_count} views")
        if activity.last_outbound_contact:
            parts.append(f"last outbound {activity.last_outbound_contact}")
        if activity.saved_by:
            parts.append(f"saved by {activity.saved_by}")
        return ", ".join(parts)


    def _snippet_is_clearly_compelling(self, snippet: CandidateSnippet) -> bool:
        from linkedin.orchestrator import _snippet_is_clearly_compelling
        return _snippet_is_clearly_compelling(snippet, self.context.brief_obj)


    def _brief_compelling_terms(self) -> tuple[str, ...]:
        from linkedin.orchestrator import _brief_compelling_terms
        return _brief_compelling_terms(self.context.brief_obj)


    def _record_snippet_activity(
        self,
        snippet: CandidateSnippet,
        string_stats: dict[str, int],
    ) -> None:
        pressure = snippet.novelty_pressure or classify_recruiter_activity_pressure(
            snippet.recruiter_activity
        )
        snippet.novelty_pressure = pressure
        if pressure == "high":
            self.context.stats["high_pressure_candidates_seen"] += 1
            string_stats.setdefault("high_pressure_candidates_seen", 0)
            string_stats["high_pressure_candidates_seen"] += 1


    def _should_skip_full_eval_for_activity(
        self,
        snippet: CandidateSnippet,
        facial_decision: OpusDecision,
    ) -> bool:
        """Whether high recruiter activity merits a low-novelty annotation.

        The V2 facial contract has no reliable confidence signal, so a confidence
        threshold cannot gate this path. The decision rests on novelty pressure,
        facial rationale text, and the snippet-compellingness heuristic.
        """
        if (snippet.novelty_pressure or "").lower() != "high":
            return False
        rationale = (facial_decision.rationale or "").lower()
        if "strong" in rationale or "clear" in rationale:
            return False
        return not self._snippet_is_clearly_compelling(snippet)


    def _record_activity_saturation_context(
        self,
        *,
        snippet: CandidateSnippet,
        search_string: SearchString,
        facial_decision: OpusDecision,
    ) -> None:
        """Annotate recruiter saturation without suppressing full review."""

        activity = self._activity_snapshot_from_sources(snippet)
        activity_summary = self._activity_summary_text(activity) or "high recruiter activity"
        facial_decision.novelty_value = "low"
        facial_decision.value_rationale = f"Low novelty due to recruiter saturation: {activity_summary}"
        log_event(
            self.context.log_path,
            "activity_saturation_context",
            string_id=search_string.id,
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
            novelty_pressure=snippet.novelty_pressure,
            activity_summary=activity_summary,
            facial_confidence=facial_decision.confidence,
        )


    def _derive_novelty_value(
        self,
        snippet: CandidateSnippet,
        *,
        profile_status_summary: dict | None = None,
    ) -> tuple[str, str]:
        activity = self._activity_snapshot_from_sources(snippet, profile_status_summary)
        pressure = classify_recruiter_activity_pressure(activity)
        reachout_status = infer_reachout_status(activity)
        summary = self._activity_summary_text(activity)
        if pressure == "high":
            return "low", f"Low novelty due to visible recruiter saturation ({summary or reachout_status or 'high activity'})."
        if pressure == "medium":
            return "medium", f"Moderate novelty because the candidate already shows recruiter activity ({summary or reachout_status or 'some activity'})."
        return "high", "High novelty: limited visible recruiter activity."


    async def _evaluate_snippet(
        self,
        snippet: CandidateSnippet,
        page_report: _PageReport | None = None,
        search_string: SearchString | None = None,
        string_stats: dict[str, int] | None = None,
        page_context: PageReviewContext | None = None,
    ) -> Optional[OpusDecision]:
        """Run snippet through facial judgment -> full profile -> final judgment."""
        runtime_search_string = search_string or SearchString(
            id=snippet.source_string_id,
            name=snippet.source_string_name,
            boolean="",
        )
        facial_attempt_id: int | None = None

        pending_facial = self.ledger._resume_pending_full_decision(snippet)
        if pending_facial is None:
            raise RuntimeError(
                "foreign pending full review reached the active string evaluator"
            )
        if pending_facial:
            print(
                f"    [resume-full] {snippet.name} — {pending_facial} already "
                "settled; completing full review"
            )
            if snippet.profile_url:
                self.ledger.start_review(snippet.profile_url)
            self._note_page_full_review_expected(snippet, page_context=page_context)
            resumed_full = await self._full_evaluate(
                snippet,
                page_report,
                runtime_search_string,
                page_context=page_context,
            )
            self.ledger._clear_resume_pending_full_if_settled(
                snippet=snippet,
                decision=resumed_full,
            )
            return resumed_full

        # --- Employer and alumni blacklist checks (no LLM call) ---
        employer_gate = self._facial_employer_gate(snippet)
        if employer_gate:
            gate_path, (blocked, matched_field, matched_value) = employer_gate
            gate_tag, gate_label, gate_stat = facial_gate_labels(gate_path)
            print(
                facial_gate_console_line(
                    gate_path, snippet.name, blocked, matched_field, matched_value
                )
            )
            self.context.stats.setdefault(gate_stat, 0)
            self.context.stats[gate_stat] += 1
            if page_report:
                page_report.add_skip_preview(snippet.name, f"{gate_tag}: {blocked}")
            blacklist_decision = OpusDecision(
                stage="facial", decision="FACIAL_NO", path=gate_path,
                confidence=1.0, rationale=f"{gate_label}: {blocked} ({matched_field})",
                candidate_name=snippet.name, profile_url=snippet.profile_url,
            )
            # Same registration-before-stage-write requirement as the
            # batch path's blacklist branch (see there for the RCA).
            self.attempts._record_runtime_snippet(runtime_search_string, snippet)
            facial_attempt_id = self.attempts._start_runtime_stage_attempt(
                search_string=runtime_search_string,
                snippet=snippet,
                stage="facial",
            )
            self._finish_runtime_stage_success(
                attempt_id=facial_attempt_id,
                stage="facial",
                snippet=snippet,
                decision=blacklist_decision,
            )
            # A territory or scope skip is not a facial verdict on fit:
            # counting it as facial_no would teach exhaustion and family yield
            # that a string reaching good people in another recruiter's market,
            # or a competitor's alumni, is noise (NON_VERDICT_GATE_PATHS).
            if gate_path not in NON_VERDICT_GATE_PATHS:
                self.ledger._record_facial_funnel_outcome(
                    snippet=snippet,
                    decision=blacklist_decision.decision,
                    search_string=runtime_search_string,
                    string_stats=string_stats,
                    stats=self.context.stats,
                )
            self.ledger.record_prior_outcome(snippet.profile_url, "FACIAL_NO")
            self.ledger._mark_terminal(snippet.profile_url)
            return blacklist_decision

        # --- Facial judgment (facial tier) ---
        print(f"    Facial judgment ({config.FACIAL_MODEL_NAME.rsplit('/', 1)[-1]})...")
        facial_call_id = f"judge-{uuid.uuid4().hex}"
        facial_attempt_payload = {
            "logical_call_id": facial_call_id,
            "stage": "facial",
            "batch_index": 0,
            "batch_count": 1,
            "batch_size": 1,
            "batch_slot": 0,
        }
        facial_attempt_id = self.attempts._start_runtime_stage_attempt(
            search_string=runtime_search_string,
            snippet=snippet,
            stage="facial",
            payload=facial_attempt_payload,
        )
        _lane_ctx = {
            **self.attempts._lane_context_for_stage(runtime_search_string, stage="facial", variant_id=self.surface._variant_id_for_search_string(runtime_search_string)),
            "logical_call_id": facial_call_id,
            "string_id": runtime_search_string.id,
            "page": snippet.page,
            "batch_index": 0,
            "batch_count": 1,
            "batch_size": 1,
            "batch_slot": 0,
        }
        try:
            facial = facial_judge(snippet, brief=self.context.brief_obj, prompt_prefix=self._facial_prefix_for_string(runtime_search_string), lane_context=_lane_ctx)
        except Exception as e:
            if is_api_budget_exhausted_error(e):
                budget_error = ApiBudgetExhaustedError(str(e))
                self._abort_runtime_stage_attempt(
                    attempt_id=facial_attempt_id,
                    snippet=snippet,
                    error=budget_error,
                    payload={
                        **facial_attempt_payload,
                        "api_budget_exhausted": True,
                        "stage": "facial",
                    },
                )
                raise budget_error from e
            print(f"    [ERROR] Facial judgment failed: {e}")
            log_event(self.context.log_path, "facial_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            facial = judgment_failure_decision(
                stage="facial",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                error=e,
                source="judgment",
            )

        # Preserve raw borderline observability while retaining the verdict
        # itself in canonical state below.
        if (
            self._facial_ternary_enabled()
            and self.context._bias_monitor is not None
            and facial.decision == "FACIAL_BORDERLINE"
        ):
            self.context._bias_monitor.record_facial_borderline_seen(
                string_id=str(snippet.source_string_id),
            )

        self._stamp_read_interest(snippet, facial.decision)

        facial = self._normalize_facial_decision_for_persistence(facial)

        # Intercept parse/judgment failures — log but do NOT persist to cross-session history
        if is_failure_decision(facial.decision):
            print(f"    [PARSE_FAILURE] {facial.rationale}")
            self.context.stats.setdefault("parse_failures", 0)
            self.context.stats["parse_failures"] += 1
            if self.context._bias_monitor:
                self.context._bias_monitor.record_decision(DecisionRecord(
                    candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                    string_id=str(snippet.source_string_id),
                    stage="facial",
                    decision=facial.decision,
                    confidence=facial.confidence,
                    capability_area=None,
                ))
            # Non-terminal: allow retry later in this session or on resume
            self.attempts._finish_runtime_failure_decision(
                attempt_id=facial_attempt_id,
                snippet=snippet,
                decision=facial,
                payload=facial_attempt_payload,
            )
            self.ledger.release_review(snippet.profile_url)
            return facial

        self.ledger.record_prior_outcome(snippet.profile_url, facial.decision)
        self.ledger._mark_terminal(snippet.profile_url)
        self._finish_runtime_stage_success(
            attempt_id=facial_attempt_id,
            stage="facial",
            snippet=snippet,
            decision=facial,
            extra_payload=facial_attempt_payload,
        )

        # Record facial decision for bias monitoring
        if self.context._bias_monitor:
            self.context._bias_monitor.record_decision(DecisionRecord(
                candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                string_id=str(snippet.source_string_id),
                stage="facial",
                decision=facial.decision,
                confidence=facial.confidence,
                capability_area=None,
            ))
            # Tightening defaults off because a brief-level guessed band can punish
            # legitimate per-string precision probes. When enabled, the bias monitor's
            # per-string tightening status triggers the intervention.
            if (
                config.LINKEDIN_FACIAL_TIGHTENING_ENABLED
                and not self._triage_tightened
                and self.context._bias_monitor  # V1 briefs leave this None
            ):
                tightening = self.context._bias_monitor.get_tightening_status(str(snippet.source_string_id))
                if tightening:
                    self._triage_tightened = True
                    self._tightening_prefix = (
                        f"⚠ TRIAGE TIGHTENING ACTIVE: The facial YES rate on this search string is running "
                        f"{tightening['actual_rate']:.0%}, which is {tightening['multiplier']:.1f}x above the expected "
                        f"maximum of {tightening['expected_high']:.0%}. Apply stricter filtering: require TWO strong "
                        f"positive signals for FACIAL_YES instead of one. Generic seniority + AI keywords is insufficient. "
                        f"Example of insufficient: 'VP of Digital Transformation at Accenture' — senior title + consulting firm "
                        f"mentioning AI, but no specific ML/data work visible. Example of sufficient: 'ML Engineer at DeepMind' "
                        f"+ 'Research Scientist at Google Brain' — two positions with direct capability area connections.\n\n"
                    )
                    print(f"    [bias] Tightening facial criteria for remaining candidates on this string "
                          f"(YES rate: {tightening['actual_rate']:.0%}, expected max: {tightening['expected_high']:.0%})")

        if facial.decision in ("FACIAL_NO", "FACIAL_SKIP"):
            tag = "FACIAL_NO" if facial.decision == "FACIAL_NO" else "FACIAL_SKIP"
            print(f"    [{tag}] {facial.rationale}")
            if facial.decision == "FACIAL_NO":
                self.ledger._record_facial_funnel_outcome(
                    snippet=snippet,
                    decision=facial.decision,
                    search_string=runtime_search_string,
                    string_stats=string_stats,
                    stats=self.context.stats,
                )
            else:
                self.context.stats.setdefault("facial_skip", 0)
                self.context.stats["facial_skip"] += 1
            if page_report:
                page_report.add_skip_preview(snippet.name, f"{tag}: {facial.rationale}")
            return facial

        print(f"    [{facial.decision}] {facial.rationale}")
        self.ledger._record_facial_funnel_outcome(
            snippet=snippet,
            decision=facial.decision,
            search_string=runtime_search_string,
            string_stats=string_stats,
            stats=self.context.stats,
        )

        if self._should_skip_full_eval_for_activity(snippet, facial):
            self._record_activity_saturation_context(
                snippet=snippet,
                search_string=runtime_search_string,
                facial_decision=facial,
            )

        self.ledger._track_full_review_obligation(snippet, facial.decision)
        self._note_page_full_review_expected(snippet, page_context=page_context)
        full_decision = await self._full_evaluate(
            snippet,
            page_report,
            runtime_search_string,
            page_context=page_context,
        )
        self.ledger._clear_resume_pending_full_if_settled(
            snippet=snippet,
            decision=full_decision,
        )
        return full_decision


    def _full_candidate_id_mismatch_facts(
        self,
        decision: OpusDecision,
    ) -> dict | None:
        """Machine-typed mismatch facts, or None for any other outcome.

        Reads ONLY the structured record ``full_judge`` attaches to the prompt
        capture. The rationale is deliberately not consulted: its detail
        carries the model-returned ID, and profile text is attacker
        controlled, so classifying on prose would hand an injection the very
        channel this check exists to defend.
        """

        if decision.decision != "PARSE_FAILURE":
            return None
        capture = decision.prompt_capture or {}
        failure = capture.get("judgment_contract_failure")
        if not isinstance(failure, dict):
            return None
        if failure.get("reason") != "candidate_id_mismatch":
            return None
        return failure


    @staticmethod
    def _full_id_mismatch_event_payload(
        *,
        facts: dict,
        snippet: CandidateSnippet,
        logical_call_id: str,
        retry_logical_call_id: str,
        parent_logical_call_id: str,
        retry_scheduled: bool,
        is_retry_result: bool,
        recovered_so_far: int,
    ) -> dict:
        """One payload shape for both mismatch receipts.

        A reader joining the two emissions of a double mismatch must not have
        to branch on which one it holds, so the first (parent) and second
        (re-ask) receipts carry identical keys and differ only in values.
        """

        return {
            "candidate_name": snippet.name,
            "profile_url": snippet.profile_url,
            "expected_id": str(facts.get("expected_candidate_id") or ""),
            # Model output: clamped here, and printed nowhere.
            "actual_id": str(facts.get("actual_candidate_id") or "")[
                :_FULL_ID_MISMATCH_ACTUAL_ID_MAX_CHARS
            ],
            "logical_call_id": logical_call_id,
            "retry_logical_call_id": retry_logical_call_id,
            "parent_logical_call_id": parent_logical_call_id,
            "retry_scheduled": retry_scheduled,
            "is_retry_result": is_retry_result,
            "recovered_mismatches_so_far": recovered_so_far,
        }


    def _resolve_full_candidate_id_mismatch(
        self,
        *,
        decision: OpusDecision,
        summary: CandidateProfileSummary,
        snippet: CandidateSnippet,
        search_string: SearchString,
        lane_context: dict,
        logical_call_id: str,
        trace: dict,
    ) -> tuple[OpusDecision, str]:
        """Re-ask ONCE for a full evaluation the model mis-attributed.

        Returns the decision that flows onward and the logical call ID the
        corruption breaker must key on. Every mismatch is recorded and
        classified BEFORE anything is re-asked — including the ones the
        recovery ceiling refuses — because the receipt is the security signal
        and the re-ask is only a throughput recovery layered on top of it.
        The re-ask carries a FRESH opaque ID and its own child call, so a
        second mismatch is a genuinely independent observation rather than a
        replay of the first, and it earns its own receipt: two mis-attributions
        under two unrelated IDs is the loudest injection signal this design
        can produce, so it is classified rather than merely struck. A re-ask
        that fails surfaces exactly as the original would have, keyed to the
        child call, and only a re-ask that produced a verdict spends budget.

        This method NEVER touches the canonical attempt. It stamps ``trace``
        with the phase it is in and lets every failure escape, so the caller
        owns the single close site and no path can double-close or leak.
        """

        facts = self._full_candidate_id_mismatch_facts(decision)
        if facts is None:
            return decision, logical_call_id

        recovered_so_far = int(
            getattr(self, "_full_id_mismatch_recovered_count", 0)
        )
        retry_scheduled = recovered_so_far < _FULL_ID_MISMATCH_RECOVERY_CEILING
        # Mint the child call BEFORE the receipt so the receipt can name the
        # call that will carry the surviving verdict; runtime state is then
        # joinable parent -> child without reading provider logs.
        retry_context = (
            full_id_mismatch_retry_context(
                lane_context,
                parent_logical_call_id=logical_call_id,
            )
            if retry_scheduled
            else None
        )
        retry_call_id = (
            str(retry_context.get("logical_call_id") or "")
            if retry_context is not None
            else ""
        )
        self.attempts._record_runtime_event(
            search_string=search_string,
            event_type="full_candidate_id_mismatch",
            payload=self._full_id_mismatch_event_payload(
                facts=facts,
                snippet=snippet,
                logical_call_id=logical_call_id,
                retry_logical_call_id=retry_call_id,
                parent_logical_call_id="",
                retry_scheduled=retry_scheduled,
                is_retry_result=False,
                recovered_so_far=recovered_so_far,
            ),
        )
        if retry_context is None:
            print(
                f"    [full-id-mismatch] {snippet.name} — recovery ceiling "
                f"{_FULL_ID_MISMATCH_RECOVERY_CEILING} reached; surfacing the "
                "contract failure"
            )
            return decision, logical_call_id

        print(
            f"    [full-id-mismatch] {snippet.name} — re-asking once under a "
            "fresh candidate ID"
        )
        trace["phase"] = "re_ask"
        retried = full_judge(
            summary,
            self.context.brief_obj,
            lane_context=retry_context,
            opaque_candidate_id=generate_opaque_candidate_ids(1)[0],
        )
        trace["phase"] = "settle"

        retried_facts = self._full_candidate_id_mismatch_facts(retried)
        if retried_facts is not None:
            # Never re-ask a re-ask: this receipt is the terminal
            # classification of a candidate that missed two unrelated IDs.
            self.attempts._record_runtime_event(
                search_string=search_string,
                event_type="full_candidate_id_mismatch",
                payload=self._full_id_mismatch_event_payload(
                    facts=retried_facts,
                    snippet=snippet,
                    logical_call_id=retry_call_id,
                    retry_logical_call_id="",
                    parent_logical_call_id=logical_call_id,
                    retry_scheduled=False,
                    is_retry_result=True,
                    recovered_so_far=recovered_so_far,
                ),
            )
        elif not is_failure_decision(retried.decision):
            self._full_id_mismatch_recovered_count = recovered_so_far + 1
        return retried, retry_call_id


    def _raise_if_full_tool_contract_corruption(
        self,
        *,
        decision: OpusDecision,
        logical_call_id: str,
    ) -> None:
        """Stop the optimized run after two distinct malformed full calls.

        Tool mode has no prose parser or automatic legacy fallback: every
        ``PARSE_FAILURE`` at this boundary is a failed forced-tool contract.
        Count logical calls rather than decision objects so future accounting
        refactors cannot accidentally count the same provider call twice.
        Callers invoke this only after the canonical attempt is closed.
        """

        if (
            config.LINKEDIN_V2_FULL_CONTRACT != "tool"
            or decision.decision != "PARSE_FAILURE"
        ):
            return

        call_id = str(logical_call_id or "").strip()
        if not call_id:
            call_id = str(
                (decision.prompt_capture or {}).get("logical_call_id") or ""
            ).strip()
        if not call_id:
            # Every production full call is assigned an ID before its attempt
            # starts. Keep a bounded fail-safe for tests/older imported calls.
            call_id = f"unattributed-{id(decision)}"

        seen = getattr(self, "_full_contract_corruption_call_ids", None)
        if not isinstance(seen, set):
            seen = set()
            self._full_contract_corruption_call_ids = seen
        if call_id in seen:
            return
        seen.add(call_id)
        self.context.stats["full_contract_corruptions"] = len(seen)

        if len(seen) < 2:
            return

        console_line = (
            "    [CIRCUIT BREAKER] two full tool-contract corruptions "
            "— stopping optimized run"
        )
        print(console_line, flush=True)
        log_event(
            self.context.log_path,
            "circuit_breaker",
            stage="full_tool_contract",
            consecutive_errors=len(seen),
            logical_call_id=call_id,
        )
        raise RuntimeError("full tool-contract corruption threshold reached")


    async def _full_evaluate(
        self,
        snippet: CandidateSnippet,
        page_report: "_PageReport | None" = None,
        search_string: SearchString | None = None,
        *,
        prepared: PreparedCandidate | None = None,
        judgment_task: asyncio.Task[JudgedCandidate] | None = None,
        timing_sink: list[dict[str, float]] | None = None,
        cancel_requested: bool = False,
        page_context: PageReviewContext | None = None,
    ) -> OpusDecision:
        """One judgment and settlement procedure for every candidate window.

        A prepared candidate transfers ownership of its open canonical attempt.
        Browser operations and settlement stay on this task; only the baseline
        judgment may already be running in the single lookahead worker.
        """
        if prepared is None:
            prepared = await self._prepare_full_candidate(
                snippet, page_report, search_string, release_panel=False,
                page_context=page_context,
            )
            if isinstance(prepared, OpusDecision):
                return prepared
        snippet = prepared.snippet
        runtime_search_string = prepared.runtime_search_string
        full_attempt_id = prepared.attempt_id
        full_attempt_payload = prepared.attempt_payload
        full_call_id = str(full_attempt_payload["logical_call_id"])
        full_breaker_call_id = full_call_id
        summary = prepared.summary
        profile_status_summary = prepared.profile_status_summary
        probe_decision = prepared.probe_decision
        _lane_ctx_full = prepared.lane_context
        if prepared.panel_released:
            print(f"    Final judgment ({config.FULL_EVAL_MODEL_NAME.rsplit('/', 1)[-1]})...")
        try:
            if judgment_task is None:
                final = full_judge(summary, brief=self.context.brief_obj, lane_context=_lane_ctx_full)
            else:
                final = await self._await_full_judgment(judgment_task, timing_sink)
            # A successful judgment proves the provider healthy; the containment ceiling
            # counts consecutive faults only.
            self._full_eval_consecutive_faults = 0
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError) and judgment_task is not None:
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "full_judgment_cancelled",
                        "pipeline": "lookahead_one",
                        "stage": "full",
                    },
                )
                raise
            if not isinstance(e, Exception):
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "base_exception_during_full_judgment",
                        "stage": "full",
                    },
                )
                raise
            if is_api_budget_exhausted_error(e):
                budget_error = ApiBudgetExhaustedError(str(e))
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=budget_error,
                    payload={
                        **full_attempt_payload,
                        "api_budget_exhausted": True,
                        "stage": "full",
                    },
                )
                raise budget_error from e
            if config.LINKEDIN_V2_FULL_CONTRACT == "tool":
                # Tool-contract fail-fast applies to contract violations, while containable
                # provider/transport faults settle this candidate as JUDGMENT_FAILURE and
                # keep the string alive. The consecutive-fault ceiling keeps a sustained
                # outage legible.
                disposition = self._full_judgment_fault_disposition(e)
                if disposition != "contain":
                    self._abort_runtime_stage_attempt(
                        attempt_id=full_attempt_id,
                        snippet=snippet,
                        error=e,
                        payload={
                            **full_attempt_payload,
                            "full_tool_transport_failed": True,
                            "stage": "full",
                            **(
                                {"full_eval_containment_ceiling": True}
                                if disposition == "ceiling"
                                else {}
                            ),
                        },
                    )
                    if disposition == "ceiling":
                        raise RuntimeError(
                            "full-eval containment ceiling: "
                            f"{config.LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS}"
                            " consecutive judgment provider faults — treating "
                            f"the provider as down (last: {e})"
                        ) from e
                    raise
                print(
                    "    [CONTAINED] full-judgment provider fault "
                    f"({self._full_eval_consecutive_faults}/"
                    f"{config.LINKEDIN_FULL_EVAL_MAX_CONSECUTIVE_FAULTS}): "
                    "settling this candidate as JUDGMENT_FAILURE and "
                    "continuing the string (CLO-177)"
                )
            print(f"    [ERROR] Final judgment failed: {e}")
            log_event(self.context.log_path, "final_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            final = judgment_failure_decision(
                stage="full",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                error=e,
                source="judgment",
            )

        current_task = asyncio.current_task()
        if cancel_requested or (current_task is not None and current_task.cancelling()):
            cancelled = asyncio.CancelledError()
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id, snippet=snippet, error=cancelled,
                payload={**full_attempt_payload, "stage": "full",
                         "run_abort": "full_judgment_cancelled",
                         "paid_judgment": final.to_dict(),
                         "profile_summary": summary.to_dict()},
            )
            raise cancelled

        # A mis-attributed verdict is intercepted here, before any downstream
        # bookkeeping reads `final`, so only the decision that actually
        # survives reaches the shadow record, the canonical attempt, and the
        # corruption breaker.
        if config.LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED:
            mismatch_trace: dict = {"phase": "classify"}
            try:
                final, full_breaker_call_id = (
                    self._resolve_full_candidate_id_mismatch(
                        decision=final,
                        summary=summary,
                        snippet=snippet,
                        search_string=runtime_search_string,
                        lane_context=_lane_ctx_full,
                        logical_call_id=full_call_id,
                        trace=mismatch_trace,
                    )
                )
            except BaseException as e:
                # The single close site for the whole interception. The
                # resolver deliberately closes nothing, so every failure —
                # a receipt write, the re-ask itself, the second receipt —
                # lands here exactly once and the canonical attempt cannot
                # be stranded in `started` or closed twice.
                abort_payload = {**full_attempt_payload, "stage": "full"}
                budget_exhausted = isinstance(
                    e, Exception
                ) and is_api_budget_exhausted_error(e)
                if budget_exhausted:
                    abort_payload["api_budget_exhausted"] = True
                elif mismatch_trace.get("phase") == "re_ask":
                    abort_payload["full_id_mismatch_retry_failed"] = True
                else:
                    abort_payload["run_abort"] = (
                        "full_id_mismatch_interception_failed"
                    )
                abort_error = (
                    ApiBudgetExhaustedError(str(e)) if budget_exhausted else e
                )
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=abort_error,
                    payload=abort_payload,
                )
                if budget_exhausted:
                    raise abort_error from e
                raise
            if full_breaker_call_id != full_call_id:
                # The surviving verdict came from the re-issue. The attempt's
                # opening call ID stays truthful; the child is named beside
                # it so every close from here on can be joined to the call
                # that actually produced the decision.
                full_attempt_payload["verdict_logical_call_id"] = (
                    full_breaker_call_id
                )

        try:
            shadow_record = self._profile_probe.record_shadow_outcome(
                snippet.name,
                snippet.profile_url,
                runtime_search_string.lane_id or "",
                probe_decision,
                final.decision,
                final.confidence,
            )
            # P10 actuate #2: make the shadow evidence real by persisting it to the
            # runtime DB events channel at this checkpoint. No reader is wired yet
            # (cascade activation stays OFF) — same unwired-shadow-evidence class as
            # the "shadow_full_judge_completed" event above; ad-hoc query against
            # runtime_state.sqlite3 is the only consumer until D8 activation lands.
            # Recovery replay re-judges the interrupted candidate and appends a
            # second cascade_shadow_recorded row, so any future activation feed
            # must dedup on profile_url before computing AuditSampler metrics.
            self.attempts._record_runtime_event(
                search_string=runtime_search_string,
                event_type="cascade_shadow_recorded",
                payload=shadow_record.to_dict(),
            )

            novelty_value, value_rationale = self._derive_novelty_value(
                snippet,
                profile_status_summary=profile_status_summary,
            )
            final.novelty_value = novelty_value
            final.value_rationale = value_rationale
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "full_precommit_bookkeeping_failed",
                    "stage": "full",
                },
            )
            raise

        # Intercept parse/judgment failures — do NOT persist to cross-session history
        if is_failure_decision(final.decision):
            print(f"    [{final.decision}] {final.rationale}")
            self.context.stats.setdefault("parse_failures", 0)
            self.context.stats["parse_failures"] += 1
            # Canonical closure precedes optional bias bookkeeping so an
            # observability defect cannot strand a paid full-eval attempt.
            self.attempts._finish_runtime_failure_decision(
                attempt_id=full_attempt_id,
                snippet=snippet,
                decision=final,
                payload={
                    **full_attempt_payload,
                    "profile_summary": summary.to_dict(),
                },
            )
            self.ledger.release_review(snippet.profile_url)
            if self.context._bias_monitor:
                self.context._bias_monitor.record_decision(DecisionRecord(
                    candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                    string_id=str(snippet.source_string_id),
                    stage="full",
                    decision=final.decision,
                    confidence=final.confidence,
                    capability_area=None,
                ))
            if not prepared.panel_released:
                try:
                    await self.context.browser.go_back_to_results()
                    await asyncio.sleep(
                        human_delay_correlated(
                            config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                            channel="panel_close",
                        )
                    )
                except Exception:
                    final._panel_stuck = True
            self._raise_if_full_tool_contract_corruption(
                decision=final,
                logical_call_id=full_breaker_call_id,
            )
            return final

        # RETIRED 2026-09-16 (the agent IS the reviewer). The judgment
        # contract has no review outcome and the parser maps any survivor
        # to REJECT / REVIEW_RETIRED before it reaches here, so this is the
        # backstop for a verdict arriving from somewhere the parser does
        # not own. It must run BEFORE the canonical terminal write so the
        # SQLite row reflects the mapped decision, and it must never let a
        # review verdict reach a save click.
        if is_legacy_review_decision(final):
            retired_decision = final.decision
            log_event(
                self.context.log_path,
                "candidate_review_recorded",
                name=snippet.name,
                profile_url=snippet.profile_url,
                string_id=runtime_search_string.id,
                original_decision=retired_decision,
                decision="REJECT",
                retired=True,
                reason="review_retired",
                lane_id=runtime_search_string.lane_id,
            )
            print(f"    [REVIEW_RETIRED] {retired_decision} -> REJECT")
            _clear_review_evidence(final)
            final.decision = "REJECT"
            final.outreach_tier = ""
            final.reject_reason = "REVIEW_RETIRED"

        # One counter for every route into the mapping — the parser's and
        # the backstop above — so "is anything still speaking the retired
        # vocabulary?" is a number the run report can print.
        if final.decision == "REJECT" and final.reject_reason == "REVIEW_RETIRED":
            self.context.stats["review_retired_mappings"] += 1

        save_decision = final.decision in SAVE_FAMILY_DECISIONS
        if not save_decision:
            self.ledger.record_prior_outcome(snippet.profile_url, final.decision)
            self.ledger._mark_terminal(snippet.profile_url)
            # SAVE terminalization is delayed until its side effect settles.
            finish_extra_payload: dict | None = dict(full_attempt_payload)
            self._finish_runtime_stage_success(
                attempt_id=full_attempt_id,
                stage="full",
                snippet=snippet,
                decision=final,
                profile_summary=summary,
                extra_payload=finish_extra_payload,
                page_context=page_context,
            )

        # Record full eval decision and check bias alerts
        if self.context._bias_monitor:
            self.context._bias_monitor.record_decision(DecisionRecord(
                candidate_id=f"{snippet.source_string_id}_p{snippet.page}_r{snippet.result_rank}",
                string_id=str(snippet.source_string_id),
                stage="full",
                decision=final.decision,
                confidence=final.confidence,
                capability_area=final.path if final.path != "none" else None,
            ))
            alerts = self.context._bias_monitor.check_alerts(str(snippet.source_string_id))
            for alert in alerts:
                # Bias alerts are observations, never control flow. Print and persist every
                # severity; the adaptation model receives per-string context and owns any
                # skip or pivot decision.
                symbol = {"flag": "⚡", "info": "ℹ"}.get(alert.severity, "⚡")
                print(f"    {symbol} BIAS {alert.severity.upper()}: {alert.message}")
                log_event(self.context.log_path, "bias_alert", severity=alert.severity,
                          alert_type=alert.alert_type, message=alert.message,
                          string_id=alert.string_id)

        # --- Shadow: external evidence augmentation (analytical-debug only) ---
        # Placed AFTER baseline canonical persistence (_finish_runtime_stage_success,
        # _mark_terminal, _prior_outcomes write, bias_monitor record) so that any
        # failure here is mechanically incapable of affecting baseline truth.
        # Disabled by default via config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED.
        try:
            if config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED and not is_failure_decision(final.decision):
                trigger = should_request_external_evidence(summary=summary, brief=self.context.brief_obj)
                external_evidence_status = ""
                evidence: ExternalCandidateEvidence | None = None
                enriched: OpusDecision | None = None

                if not trigger.should_run:
                    external_evidence_status = "skipped_no_trigger"
                    log_event(
                        self.context.log_path,
                        "external_evidence_skipped",
                        name=snippet.name,
                        profile_url=snippet.profile_url,
                        skip_reason=trigger.skip_reason,
                        signals=trigger.signals,
                    )
                    self.attempts._record_runtime_event(
                        search_string=runtime_search_string,
                        event_type="external_evidence_skipped",
                        payload={
                            "profile_url": snippet.profile_url,
                            "skip_reason": trigger.skip_reason,
                            "signals": trigger.signals,
                        },
                    )
                else:
                    identity_hints = {
                        "name": snippet.name,
                        "current_company": snippet.current_company,
                        "current_title": snippet.current_title,
                        "headline": snippet.headline,
                        "education_snippet": snippet.education_snippet,
                        "profile_url": snippet.profile_url,
                    }
                    result = fetch_external_candidate_evidence(
                        summary=summary,
                        brief=self.context.brief_obj,
                        trigger=trigger,
                        identity_hints=identity_hints,
                    )
                    if isinstance(result, ExternalCandidateEvidence):
                        evidence = result
                        external_evidence_status = "evidence_present"
                        log_event(
                            self.context.log_path,
                            "external_evidence_fetched",
                            name=snippet.name,
                            profile_url=snippet.profile_url,
                            trigger_reason=trigger.reason,
                            identity_confidence=evidence.identity_confidence,
                            fact_blocks=len(evidence.external_fact_blocks),
                            inferences=len(evidence.external_inferences),
                        )
                        self.attempts._record_runtime_event(
                            search_string=runtime_search_string,
                            event_type="external_evidence_fetched",
                            payload={
                                "profile_url": snippet.profile_url,
                                "trigger_reason": trigger.reason,
                                "identity_confidence": evidence.identity_confidence,
                            },
                        )
                        try:
                            enriched = full_judge_with_external_evidence(
                                summary,
                                evidence,
                                brief=self.context.brief_obj,
                                lane_context=_lane_ctx_full,
                            )
                        except Exception as enrich_exc:
                            enriched = None
                            log_event(
                                self.context.log_path,
                                "external_evidence_enriched_judge_failed",
                                name=snippet.name,
                                error=str(enrich_exc),
                                traceback=traceback.format_exc(),
                            )
                    else:  # ExternalEvidenceFailure
                        external_evidence_status = result.reason
                        log_event(
                            self.context.log_path,
                            "external_evidence_failed",
                            name=snippet.name,
                            profile_url=snippet.profile_url,
                            reason=result.reason,
                            detail=result.detail,
                            http_status=result.http_status,
                        )
                        self.attempts._record_runtime_event(
                            search_string=runtime_search_string,
                            event_type="external_evidence_failed",
                            payload={
                                "profile_url": snippet.profile_url,
                                "reason": result.reason,
                                "http_status": result.http_status,
                            },
                        )

                diff = compute_judgment_diff(
                    final,
                    enriched,
                    skip_reason=external_evidence_status if enriched is None else "",
                )
                evidence_refs_count = 0
                identity_confidence = None
                if evidence is not None:
                    identity_confidence = evidence.identity_confidence
                    evidence_refs_count = sum(len(b.evidence_refs) for b in evidence.external_fact_blocks)
                    evidence_refs_count += sum(len(i.basis_refs) for i in evidence.external_inferences)

                record_shadow_full_judgment(
                    output_dir=self.context.output_dir,
                    record=ShadowFullJudgmentRecord(
                        candidate_name=snippet.name,
                        profile_url=snippet.profile_url,
                        source_string_id=snippet.source_string_id,
                        page=snippet.page,
                        result_rank=snippet.result_rank,
                        trigger_reason=trigger.reason if trigger.should_run else "",
                        external_evidence_status=external_evidence_status,
                        identity_confidence=identity_confidence,
                        evidence_refs_count=evidence_refs_count,
                        baseline=final.to_dict(),
                        enriched=enriched.to_dict() if enriched is not None else None,
                        diff=diff,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                )

                if enriched is not None:
                    self.attempts._record_runtime_event(
                        search_string=runtime_search_string,
                        event_type="shadow_full_judge_completed",
                        payload={
                            "profile_url": snippet.profile_url,
                            "decision_changed": diff.get("decision_changed"),
                            "path_changed": diff.get("path_changed"),
                            "rationale_changed": diff.get("rationale_changed"),
                            "confidence_delta": diff.get("confidence_delta"),
                            "evidence_refs_count": evidence_refs_count,
                        },
                    )
        except Exception as shadow_exc:
            # The whole shadow path is best-effort. Nothing here may affect baseline.
            try:
                print(f"    [WARN] shadow external-evidence eval failed: {shadow_exc}")
                log_event(
                    self.context.log_path,
                    "external_evidence_shadow_unhandled_exception",
                    name=snippet.name,
                    error=str(shadow_exc),
                    traceback=traceback.format_exc(),
                )
            except Exception:
                pass

        if final.decision in SAVE_FAMILY_DECISIONS:
            tag = final.decision if final.decision in ("INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE") else "SAVE"
            print(f"    [{tag}] {final.rationale}")
            self.context.stats["save_attempts"] += 1
            if final.novelty_value == "low":
                self.context.stats["high_fit_low_novelty_saves"] += 1
            self.ledger._record_outreach_tier_outcome(
                snippet=snippet,
                decision=final,
                stats=self.context.stats,
            )
            # Consume the side-effect outcome. ``handle_save_decision`` emits the one
            # authoritative ``candidate_saved`` event; avoid a duplicate flag-less event
            # that could shadow it for live feeds, monitors, or reports.
            try:
                outcome = await self._perform_full_save(prepared, final)
            except BaseException as exc:
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=exc,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "save_side_effect_interrupted",
                        "stage": "full",
                        "paid_judgment": final.to_dict(),
                        "profile_summary": summary.to_dict(),
                    },
                )
                raise
            outcome_payload = dict(outcome.payload or {})
            skip_reason = str(outcome_payload.get("skip_reason") or "")
            persisted = outcome.status == "succeeded"
            already_present = bool(outcome_payload.get("already_present")) or (
                outcome.status == "skipped"
                and skip_reason == "existing_succeeded"
            )
            final.save_outcome = {
                "status": outcome.status,
                "persisted": persisted,
                "already_present": already_present,
                "failure_reason": outcome_payload.get("failure_reason")
                or (skip_reason or None if not persisted and not already_present else None),
            }
            if outcome_payload.get("reconciled_self_save"):
                final.save_outcome["reconciled_self_save"] = True
            if not (persisted or already_present):
                save_error = UnconfirmedSaveError(
                    failure_reason=final.save_outcome["failure_reason"],
                )
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=save_error,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "save_not_confirmed",
                        "save_outcome": final.save_outcome,
                        "stage": "full",
                    },
                )
                raise save_error

            self.ledger.record_prior_outcome(snippet.profile_url, final.decision)
            self.ledger._mark_terminal(snippet.profile_url)
            self._finish_runtime_stage_success(
                attempt_id=full_attempt_id,
                stage="full",
                snippet=snippet,
                decision=final,
                profile_summary=summary,
                extra_payload=dict(full_attempt_payload),
                page_context=page_context,
            )

            if page_report:
                page_report.add_saved(snippet, final)
        else:
            print(f"    [REJECT] {final.rationale}")
            self.context.stats["rejected"] += 1
            if page_report:
                page_report.add_skipped_opened(snippet, final)
            # Quick exit — saw enough, moving on
            if not prepared.panel_released:
                reject_dwell = max(
                    config.LINKEDIN_REJECT_CLOSE_MIN_SECONDS,
                    min(
                        config.LINKEDIN_REJECT_CLOSE_MAX_SECONDS,
                        human_delay_correlated(
                            config.LINKEDIN_REJECT_CLOSE_BASE_SECONDS,
                            channel="reject_close",
                        ),
                    ),
                )
                await asyncio.sleep(reject_dwell)
                print(f"    [profile-read] REJECT verdict → closing in {reject_dwell:.1f}s")

        if not prepared.panel_released or save_decision:
            try:
                await self.context.browser.go_back_to_results()
                await asyncio.sleep(
                    human_delay_correlated(
                        config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                        channel="panel_close",
                    )
                )
            except Exception as e:
                if _is_browser_disconnect_error(e):
                    print(f"    [ERROR] Browser session dropped while closing profile: {e}")
                    log_event(self.context.log_path, "panel_close_browser_disconnect", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                    raise
                print("    [WARN] Profile panel close missed; attempting recovery")
                log_event(self.context.log_path, "go_back_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                # Tag the result so the page loop knows state is corrupted
                final._panel_stuck = True

        return final


    async def _process_resumed_pending_full_evaluations(
        self,
        *,
        snippets: list[CandidateSnippet],
        page_report: "_PageReport | None",
        search_string: SearchString,
        all_candidates: list[dict],
        string_stats: dict[str, int],
        progress: Progress | None,
        page_num: int,
        preserve_progress_cursor: bool = False,
        page_context: PageReviewContext | None = None,
    ) -> bool:
        """Finish canonical facial positives without issuing another facial call."""
        for snippet in snippets:
            self._note_page_full_review_expected(snippet, page_context=page_context)
        await self._run_full_queue(
            snippets=snippets,
            window=2 if config.FULL_EVAL_PIPELINE_ENABLED else 1,
            origin="resume", page_report=page_report,
            search_string=search_string, all_candidates=all_candidates,
            string_stats=string_stats, progress=progress, page_num=page_num,
            preserve_progress_cursor=preserve_progress_cursor,
            page_context=page_context,
        )
        return False


    async def _run_full_queue(
        self,
        *,
        snippets: list[CandidateSnippet],
        window: int,
        origin: str,
        page_report: "_PageReport | None",
        search_string: SearchString,
        all_candidates: list[dict],
        string_stats: dict[str, int],
        progress: Progress | None,
        page_num: int,
        preserve_progress_cursor: bool = False,
        page_context: PageReviewContext | None = None,
    ) -> list[OpusDecision]:
        """One FIFO candidate path with a window of one or two.

        Window two overlaps the next extraction with one paid judgment; it
        never runs browser actions or settlement concurrently. Resume origin
        retains the serial resume caller's controls and cursor semantics.
        """
        if window not in (1, 2) or origin not in ("page", "resume"):
            raise ValueError("full queue requires window 1/2 and page/resume origin")
        pending: tuple[PreparedCandidate, asyncio.Task[JudgedCandidate] | None] | None = None
        decisions: list[OpusDecision] = []
        full_timing_samples: list[dict[str, float]] = []
        consecutive_api_errors = 0
        checkpoint_kwargs = (
            {} if preserve_progress_cursor
            else {"search_string": search_string, "page_num": page_num}
        )
        checkpoint_search_string = None if preserve_progress_cursor else search_string

        async def publish(
            decision: OpusDecision,
            snippet: CandidateSnippet,
            *,
            honor_controls: bool = True,
        ) -> None:
            nonlocal consecutive_api_errors
            decisions.append(decision)
            consecutive_api_errors = (
                consecutive_api_errors + 1
                if self._is_recoverable_provider_failure_decision(decision) else 0
            )
            self._apply_full_eval_page_outcome(
                decision=decision, snippet=snippet, page_num=page_num,
                all_candidates=all_candidates, string_stats=string_stats,
                search_string=search_string, restore_facial=origin == "resume",
                page_context=page_context,
            )
            self.allocator._checkpoint_progress(progress, **checkpoint_kwargs)
            if honor_controls and getattr(decision, "_panel_stuck", False):
                await self.surface._recover_stuck_profile_panel(
                    candidate_name=snippet.name, page_num=page_num,
                    decision=decision, progress=progress,
                    search_string=checkpoint_search_string,
                    recovery_owner=search_string,
                    page_context=page_context,
                )
            if honor_controls and origin == "page":
                await honor_pause()

        async def complete_pending(*, honor_controls: bool = True, cancel_requested: bool = False) -> None:
            nonlocal pending
            assert pending is not None
            prepared, task = pending
            # Transfer before awaiting; only this settlement owns the attempt.
            pending = None
            decision = await self._full_evaluate(
                prepared.snippet, page_report, search_string,
                prepared=prepared, judgment_task=task,
                timing_sink=full_timing_samples if window == 2 else None,
                page_context=page_context,
                **({"cancel_requested": True} if cancel_requested else {}),
            )
            await publish(decision, prepared.snippet, honor_controls=honor_controls)

        def event_is_set(name: str) -> bool:
            return bool(progress and hasattr(self.context, name)
                        and getattr(self.context, name).is_set())

        async def honor_pause() -> None:
            if not event_is_set("_pause_requested"):
                return
            self.context._pause_requested.clear()
            self.allocator._checkpoint_progress(progress)
            try:
                await asyncio.wait_for(self.context._resume_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                print("  [!] Resume timeout (5 min) — continuing without decoy burst.")
                self.context._resume_event.set()

        def honors_stop_authority() -> bool:
            # Baseline's serial resume drain never consulted stop authority;
            # its pipelined drain did (orchestrator.py@17568fa:12969-12973,
            # :13152-13161). Window one keeps the serial recordings identical;
            # window two keeps the pipelined contract.
            return origin == "page" or window == 2

        async def honor_stop() -> None:
            if not honors_stop_authority():
                return
            stop = (
                SessionExpired("session_duration_cap") if event_is_set("_session_expired")
                else OperatorStopRequested() if event_is_set("_operator_stop_event")
                else None
            )
            if stop is not None:
                if pending is not None:
                    await complete_pending(honor_controls=False)
                self.allocator._checkpoint_progress(progress)
                if isinstance(stop, SessionExpired):
                    (page_context or self.page.review).set_break_reason("session_expired")
                else:
                    print("  [!] Operator stop honored at string/page boundary. Progress saved.")
                    (page_context or self.page.review).set_break_reason("operator_stop")
                raise stop

        async def check_admission() -> None:
            nonlocal consecutive_api_errors
            if not honors_stop_authority():
                return
            await honor_stop()
            # Resolve a possible fifth failure before opening candidate six.
            if pending is not None and consecutive_api_errors >= 4:
                await complete_pending()
                await honor_stop()
            if consecutive_api_errors >= 5:
                print(f"    [CIRCUIT BREAKER] {consecutive_api_errors} consecutive full-eval failures — pausing 60s")
                log_event(self.context.log_path, "circuit_breaker", consecutive_errors=consecutive_api_errors)
                await asyncio.sleep(60)
                consecutive_api_errors = 0

        async def cleanup_abort(
            prepared: PreparedCandidate | None,
            cause: BaseException,
        ) -> None:
            errors: list[BaseException] = []
            if prepared is not None:
                try:
                    self._finish_unspawned_full_eval_context(
                        prepared, reason=f"iteration_abort:{type(cause).__name__}",
                    )
                except BaseException as error:
                    errors.append(error)
            if pending is not None:
                try:
                    await complete_pending(
                        honor_controls=False,
                        cancel_requested=isinstance(cause, asyncio.CancelledError),
                    )
                except BaseException as error:
                    errors.append(error)
            for error in errors:
                try:
                    log_event(
                        self.context.log_path, "full_pipeline_abort_cleanup_failed",
                        cause_type=type(cause).__name__, cleanup_error_type=type(error).__name__,
                    )
                except Exception:
                    pass  # Preserve the original abort even if its diagnostic fails.

        for snippet in snippets:
            await check_admission()
            prepared: PreparedCandidate | None = None
            try:
                acquired = await self._prepare_full_candidate(
                    snippet, page_report, search_string, release_panel=window == 2,
                    page_context=page_context,
                )
                if not isinstance(acquired, OpusDecision):
                    prepared = acquired
                if pending is not None:
                    await complete_pending()
                    # A pause can deliver new stop authority while the next
                    # candidate is prepared but has no paid task yet.
                    await honor_stop()
                if isinstance(acquired, OpusDecision):
                    await publish(acquired, snippet)
                    continue
                assert prepared is not None
                task = (
                    asyncio.create_task(asyncio.to_thread(
                        self._judge_full_profile, summary=prepared.summary,
                        lane_context=prepared.lane_context,
                    )) if window == 2 else None
                )
                pending = prepared, task
                prepared = None
                if window == 1 or (origin == "page" and event_is_set("_pause_requested")):
                    await complete_pending()
            except BaseException as error:
                await cleanup_abort(prepared, error)
                raise
        if pending is not None:
            await complete_pending()

        if full_timing_samples:
            logical_call_elapsed_ms = [
                round(sample["logical_call_elapsed_ms"], 3)
                for sample in full_timing_samples
            ]
            full_timing_payload = {
                "string_id": search_string.id,
                "page": page_num,
                "full_call_count": len(full_timing_samples),
                "logical_call_elapsed_ms": logical_call_elapsed_ms,
                "logical_call_elapsed_total_ms": round(
                    sum(logical_call_elapsed_ms), 3
                ),
                "operator_blocking_elapsed_ms": round(
                    sum(
                        sample["operator_blocking_elapsed_ms"]
                        for sample in full_timing_samples
                    ),
                    3,
                ),
                "pipeline": "lookahead_one",
                "browser_extraction_excluded": True,
            }
            # Timing is diagnostic and cannot change a verdict. A bounded
            # canary whose sinks lose this receipt fails postflight instead of
            # turning observability into live control flow.
            try:
                self.attempts._record_runtime_event(
                    search_string=search_string,
                    event_type="full_page_judgment_timing",
                    payload=full_timing_payload,
                )
                log_event(
                    self.context.log_path,
                    "full_page_judgment_timing",
                    **full_timing_payload,
                )
            except Exception:
                pass

        if window == 2:
            # A stop or expiry that arrived during the last paid judgment is
            # honored here, after the drain and the timing receipt, as the
            # pipelined path did before the queue unification.
            await honor_stop()

        return decisions


    async def _prepare_full_candidate(
        self,
        snippet: CandidateSnippet,
        page_report: "_PageReport | None" = None,
        search_string: SearchString | None = None,
        *,
        release_panel: bool = True,
        page_context: PageReviewContext | None = None,
    ) -> PreparedCandidate | OpusDecision:
        """Acquire one full attempt; release its panel only for lookahead."""
        runtime_search_string = search_string or SearchString(
            id=snippet.source_string_id,
            name=snippet.source_string_name,
            boolean="",
        )
        full_call_id = f"judge-{uuid.uuid4().hex}"
        full_attempt_payload = {
            "logical_call_id": full_call_id,
            "stage": "full",
        }
        full_attempt_id = self.attempts._start_runtime_stage_attempt(
            search_string=runtime_search_string,
            snippet=snippet,
            stage="full",
            payload=full_attempt_payload,
        )

        try:
            self.ensure_services()
            log_event(
                self.context.log_path,
                "candidate_opened",
                name=snippet.name,
                profile_url=snippet.profile_url,
                string_id=runtime_search_string.id,
            )
            acquisition = await self.acquisition.extract_profile_summary(
                snippet, interest=self._read_interest_for(snippet)
            )
            summary = acquisition.profile_summary
            self.surface._note_profile_extraction_success()
        except (OperatorStopRequested, SessionExpired) as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": (
                        "operator_stop"
                        if isinstance(e, OperatorStopRequested)
                        else "session_expired"
                    ),
                    "stage": "full",
                },
            )
            raise
        except GovernorLimitReached as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "governor_limit",
                    "stage": "full",
                },
            )
            raise
        except GeographyRegimeError as e:
            # Profile extraction can reassert session geography through the browser
            # health seam, and that check is fail-closed. A geography regime error is
            # a run-level abort; softening it to judgment failure would allow later
            # pages to run and save off-geography.
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "geography_regime",
                    "stage": "full",
                },
            )
            raise
        except SurfaceOwnershipLostError as e:
            # Same wrapper-swallow hazard as GeographyRegimeError above —
            # a mid-profile recovery navigation is a string-level stop, never
            # a soft per-candidate judgment failure that would keep this
            # string paging on an unverified surface.
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "surface_ownership_lost",
                    "stage": "full",
                },
            )
            raise
        except Exception as e:
            if _is_browser_disconnect_error(e):
                print(f"    [ERROR] Browser session dropped during profile extraction: {e}")
                log_event(self.context.log_path, "profile_browser_disconnect", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "browser_disconnect",
                        "stage": "full",
                    },
                )
                raise
            if is_api_budget_exhausted_error(e):
                budget_error = ApiBudgetExhaustedError(str(e))
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=budget_error,
                    payload={
                        **full_attempt_payload,
                        "api_budget_exhausted": True,
                        "stage": "full",
                    },
                )
                raise budget_error from e
            print(f"    [ERROR] Profile extraction failed: {e}")
            await self.surface._note_profile_extraction_failure(snippet, e)
            log_event(self.context.log_path, "profile_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            self._finish_runtime_stage_failure(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "profile_extraction_failed": True,
                },
            )
            self.ledger.release_review(snippet.profile_url)
            try:
                await self.context.browser.go_back_to_results()
            except Exception as cleanup_error:
                if _is_browser_disconnect_error(cleanup_error):
                    raise
            return judgment_failure_decision(
                stage="full",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
                error=e,
                source="profile_extraction",
            )
        except BaseException as e:
            # asyncio cancellation, KeyboardInterrupt, and SystemExit do not
            # inherit from Exception. Close the canonical attempt before the
            # process-level control signal escapes.
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "base_exception_during_profile_extraction",
                    "stage": "full",
                },
            )
            raise

        profile_status_summary: dict = {}
        try:
            profile_status_summary = await self.context.browser.get_profile_status_summary()
        except Exception as e:
            self._propagate_profile_activity_abort(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload=full_attempt_payload,
            )
            log_event(
                self.context.log_path,
                "profile_activity_enrichment_failed",
                name=snippet.name,
                error=str(e),
                traceback=traceback.format_exc(),
            )
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "base_exception_during_profile_enrichment",
                    "stage": "full",
                },
            )
            raise

        # --- Shadow profile probe (P12) — records decision; full eval always runs ---
        try:
            probe_decision = self._profile_probe.evaluate(
                summary.to_dict(),
                self._brief_signals_for_profile_probe(),
            )
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "profile_probe_failed",
                    "stage": "full",
                },
            )
            raise

        # --- Final judgment (full-eval tier) ---
        if not release_panel:
            print(f"    Final judgment ({config.FULL_EVAL_MODEL_NAME.rsplit('/', 1)[-1]})...")
        try:
            _lane_ctx_full = {
                **self.attempts._lane_context_for_stage(
                    runtime_search_string,
                    stage="full_eval",
                 variant_id=self.surface._variant_id_for_search_string(runtime_search_string)),
                "logical_call_id": full_call_id,
                "string_id": runtime_search_string.id,
                "page": snippet.page,
            }
        except BaseException as e:
            self._abort_runtime_stage_attempt(
                attempt_id=full_attempt_id,
                snippet=snippet,
                error=e,
                payload={
                    **full_attempt_payload,
                    "run_abort": "full_lane_context_failed",
                    "stage": "full",
                },
            )
            raise
        if release_panel:
            try:
                await self.context.browser.go_back_to_results()
                await asyncio.sleep(
                    human_delay_correlated(
                        config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS,
                        channel="panel_close",
                    )
                )
            except Exception as e:
                if _is_browser_disconnect_error(e):
                    print(f"    [ERROR] Browser session dropped while closing profile: {e}")
                    log_event(self.context.log_path, "panel_close_browser_disconnect", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                    self._finish_runtime_stage_failure(
                        attempt_id=full_attempt_id,
                        snippet=snippet,
                        error=e,
                        payload={
                            **full_attempt_payload,
                            "panel_close_failed": True,
                        },
                    )
                    self.ledger.release_review(snippet.profile_url)
                    raise
                print("    [WARN] Profile panel close missed; attempting recovery")
                log_event(self.context.log_path, "go_back_error", name=snippet.name, error=str(e), traceback=traceback.format_exc())
                try:
                    await self.surface._recover_stuck_profile_panel(
                        candidate_name=snippet.name,
                        page_num=snippet.page,
                        recovery_owner=runtime_search_string,
                        page_context=page_context,
                    )
                except BaseException as recovery_error:
                    self._abort_runtime_stage_attempt(
                        attempt_id=full_attempt_id,
                        snippet=snippet,
                        error=recovery_error,
                        payload={
                            **full_attempt_payload,
                            "panel_close_failed": True,
                            "run_abort": "panel_recovery_failed",
                            "stage": "full",
                        },
                    )
                    raise
            except BaseException as e:
                self._abort_runtime_stage_attempt(
                    attempt_id=full_attempt_id,
                    snippet=snippet,
                    error=e,
                    payload={
                        **full_attempt_payload,
                        "run_abort": "base_exception_during_panel_close",
                        "stage": "full",
                    },
                )
                raise


        prepared = PreparedCandidate(
            snippet=snippet,
            runtime_search_string=runtime_search_string,
            attempt_id=full_attempt_id,
            attempt_payload=full_attempt_payload,
            summary=summary,
            profile_status_summary=profile_status_summary,
            probe_decision=probe_decision,
            lane_context=_lane_ctx_full,
            panel_released=release_panel,
        )

        # --- Alumni blacklist over the FULL work history (no LLM call) ---
        # Runs before any judgment task can be spawned and before any save or
        # note: a hit settles here as a REJECT and never becomes a
        # PreparedCandidate. The card shows a few positions at most, so this
        # is the gate that sees a stint the facial gate could not.
        try:
            alumni_hit = self._profile_alumni_blacklist_match(
                snippet,
                summary,
                getattr(self.context.brief_obj, "alumni_blacklist", None),
            )
        except Exception as e:  # noqa: BLE001 — the judge still carries the rule
            print(
                f"    [WARN] alumni blacklist check failed for {snippet.name} "
                f"({type(e).__name__}); judging normally"
            )
            log_event(self.context.log_path, "alumni_blacklist_check_failed", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            alumni_hit = None
        if alumni_hit is not None:
            return await self._settle_profile_alumni_gate(
                prepared,
                alumni_hit,
                page_report=page_report,
                page_context=page_context,
            )

        # --- Employer scope over the full profile's current positions ---
        # A brief's employer_allowlist (inert without one). The card gate saw
        # the card's current company; the profile names every position still
        # marked Present, so this catches a person the card left unidentified.
        try:
            scope_hit = self._employer_scope_match(
                snippet, summary, self.context.brief_obj
            )
        except Exception as e:  # noqa: BLE001 — the judge still carries the scope rule
            print(
                f"    [WARN] employer scope check failed for {snippet.name} "
                f"({type(e).__name__}); judging normally"
            )
            log_event(self.context.log_path, "employer_scope_check_failed", name=snippet.name, error=str(e), traceback=traceback.format_exc())
            scope_hit = None
        if scope_hit is not None:
            return await self._settle_profile_alumni_gate(
                prepared,
                scope_hit,
                page_report=page_report,
                page_context=page_context,
                gate_path=EMPLOYER_SCOPE_GATE_PATH,
            )

        return prepared


    def _judge_full_profile(
        self, *, summary: CandidateProfileSummary,
        lane_context: dict | None,
    ) -> JudgedCandidate:
        started = time.monotonic()
        decision = full_judge(summary, brief=self.context.brief_obj, lane_context=lane_context)
        return JudgedCandidate(
            decision=decision,
            elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
        )


    async def _await_full_judgment(
        self, task: asyncio.Task[JudgedCandidate],
        timing_sink: list[dict[str, float]] | None,
    ) -> OpusDecision:
        """Cancellation drains the paid worker before its attempt can close."""
        started = time.monotonic()
        try:
            judged = await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            try:
                judged = task.result()
            except BaseException:
                raise asyncio.CancelledError from None
            # The caller records this paid result before honoring cancellation.
        if timing_sink is not None:
            elapsed = judged.elapsed_ms
            blocking = (time.monotonic() - started) * 1000.0
            if math.isfinite(elapsed) and elapsed > 0 and math.isfinite(blocking) and blocking >= 0:
                timing_sink.append({
                    "logical_call_elapsed_ms": elapsed,
                    "operator_blocking_elapsed_ms": blocking,
                })
        return judged.decision


    def _apply_full_eval_page_outcome(
        self,
        *,
        decision: OpusDecision,
        snippet: CandidateSnippet,
        page_num: int,
        all_candidates: list[dict],
        string_stats: dict[str, int],
        search_string: SearchString,
        restore_facial: bool = False,
        page_context: PageReviewContext | None = None,
    ) -> None:
        self.ledger._clear_resume_pending_full_if_settled(
            snippet=snippet,
            decision=decision,
        )
        if decision is None:
            return
        if restore_facial:
            preview_outcome = next(
                (
                    candidate.get("outcome")
                    for candidate in all_candidates
                    if (
                        candidate.get("profile_url") == snippet.profile_url
                        or (
                            not candidate.get("profile_url")
                            and candidate.get("name") == snippet.name
                        )
                    )
                    and candidate.get("page") == page_num
                ),
                None,
            )
            if preview_outcome in {"facial_yes", "facial_borderline"}:
                self.ledger._record_facial_funnel_outcome(
                    snippet=snippet,
                    decision=(
                        "FACIAL_BORDERLINE"
                        if preview_outcome == "facial_borderline"
                        else "FACIAL_YES"
                    ),
                    search_string=search_string,
                    string_stats=string_stats,
                    stats=self.context.stats,
                )
        # A profile-level scope skip settles as REJECT/HARD_GATE but is not a
        # full verdict on fit: it stays out of the full funnel (full_reject
        # feeds the noise trigger) and records its own page outcome.
        skip_outcome = gate_skip_outcome(decision.path) if decision.stage == "full" else None
        if skip_outcome is None:
            self.ledger._record_full_funnel_outcome(
                snippet=snippet,
                decision=decision,
                search_string=search_string,
                string_stats=string_stats,
                stats=self.context.stats,
            )
        if skip_outcome is not None:
            outcome = skip_outcome
        elif decision.decision in SAVE_FAMILY_DECISIONS:
            save_outcome = getattr(decision, "save_outcome", None) or {}
            if save_outcome and not (
                save_outcome.get("persisted")
                or save_outcome.get("already_present")
            ):
                outcome = "save_failed"
                string_stats["save_failed"] = string_stats.get("save_failed", 0) + 1
            else:
                outcome = "save"
                if not save_outcome.get("already_present") or save_outcome.get(
                    "reconciled_self_save"
                ):
                    string_stats["saves"] += 1
                    search_string.saves.append(snippet.name)
                    if len(search_string.save_exemplars) < 8:
                        search_string.save_exemplars.append({
                            "title": str(
                                getattr(snippet, "current_title", "")
                                or getattr(snippet, "headline", "")
                                or ""
                            ),
                            "company": str(getattr(snippet, "current_company", "") or ""),
                        })
                self.surface.geography._warn_if_off_geo_save(snippet)
        elif decision.decision == "REJECT":
            outcome = "reject"
        elif decision.decision in LEGACY_REVIEW_DECISIONS:
            # Historical rows only — no live verdict can land here.
            outcome = "review"
        elif decision.stage == "full" and is_failure_decision(decision.decision):
            outcome = "error"
            (page_context or self.page.review).note("errored")
        else:
            outcome = "facial_yes"

        for candidate in all_candidates:
            if (
                (
                    candidate.get("profile_url") == snippet.profile_url
                    or (
                        restore_facial
                        and not candidate.get("profile_url")
                        and candidate.get("name") == snippet.name
                    )
                )
                and candidate["page"] == page_num
                and candidate["outcome"] in {"facial_yes", "facial_borderline"}
            ):
                candidate["outcome"] = outcome
                candidate["rationale"] = decision.rationale
                break


    def _finish_unspawned_full_eval_context(
        self,
        eval_context: PreparedCandidate,
        *,
        reason: str,
    ) -> OpusDecision:
        snippet: CandidateSnippet = eval_context.snippet
        summary: CandidateProfileSummary = eval_context.summary
        error = RuntimeError(f"Full evaluation skipped before judgment: {reason}")
        final = judgment_failure_decision(
            stage="full",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
            error=error,
            source="judgment",
        )
        self.attempts._finish_runtime_failure_decision(
            attempt_id=eval_context.attempt_id,
            snippet=snippet,
            decision=final,
            payload={
                **dict(eval_context.attempt_payload),
                "profile_summary": summary.to_dict(),
                "skipped_reason": reason,
            },
        )
        self.ledger.release_review(snippet.profile_url)
        return final
