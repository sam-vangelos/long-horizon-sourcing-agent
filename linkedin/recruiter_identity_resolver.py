"""Recruiter-first reconciliation for GitHub-sourced leads (identity + holistic fit + engagement)."""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass

from github.reconciliation_input import GitHubReconciliationLead
from linkedin.browser import LinkedInBrowser
from shared.brief_loader import Brief
from shared.extractors import extract_profile_from_innertext
from shared.identity_resolution import (
    build_candidate_lookup_queries,
    build_person_lookup_name,
    canonicalize_location_label,
    choose_best_match,
    classify_recruiter_activity_pressure,
    infer_reachout_status,
    normalize_company_name,
    normalize_location_text,
    normalize_person_name,
    score_linkedin_identity_match,
)
from shared.human_timing import human_delay_correlated
from shared.judger import full_judge
from shared.recruiter_ambiguity_resolution import (
    MultiProfileOutcome,
    consolidate_multi_profile_reviews,
    dedupe_plausible_by_profile_url,
    is_plausible_recruiter_candidate,
    is_single_strong_plausible_for_profile_open,
    single_plausible_is_safely_dominant,
)
from shared.reconciliation_schemas import (
    LinkedInIdentityHints,
    LinkedInMatchResult,
    RecruiterActivitySnapshot,
)
from shared.recruiter_identity_schemas import (
    PlausibleProfileReview,
    RecruiterIdentityCandidate,
    RecruiterIdentityResolution,
)
from shared.recruiter_reconciliation_decision import decide_final_reconciliation_action
from shared.schemas import CandidateProfileSummary, OpusDecision


@dataclass(frozen=True)
class RecruiterResolverConfig:
    max_cards: int = 5
    open_profile_on_likely_match: bool = True
    dry_run_save: bool = False
    max_ambiguity_profiles: int = 3
    # Operator workflow mode. "fit_gated_save" preserves the legacy fit+engagement
    # gated save behavior. "identity_collect" collects identity-confirmed Recruiter
    # profiles into the project regardless of brief fit; the resolver must not call
    # full_judge in this mode (Slice 4 wires the behavior; Slice 1 only carries
    # the value through).
    workflow_mode: str = "fit_gated_save"
    # Query expansion policy (Recruiter-Identity-Collection-Cycle-Audit-Fixes §1).
    # Controls whether the resolver issues enriched company/location/title
    # keyword searches in addition to the bare quoted lookup name.
    #   - "name_first": only the bare quoted lookup name is issued; no enriched
    #     fallback. Used in pre-filtered Recruiter searches where adding
    #     keyword constraints would mutate operator-applied filters.
    #   - "enriched": full bounded enriched plan is issued from the first
    #     query (legacy behavior). Used in fit_gated_save and any non-
    #     current-search identity_collect run.
    #   - "auto": derive at lookup time from workflow_mode + use_current_search
    #     + search_location. identity_collect with use_current_search=True or
    #     a fixed search_location → name_first; otherwise → enriched.
    query_expansion_policy: str = "auto"


_NAME_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

DIRECT_HINT_EVIDENCE = "Direct LinkedIn URL hint matches surfaced Recruiter card"
PUBLIC_HINT_EVIDENCE = "Public LinkedIn URL hint matches opened Recruiter profile"

# Public LinkedIn profile URL slug extractor. Matches both bare and protocol/host-
# qualified shapes:
#   https://www.linkedin.com/in/foo/
#   linkedin.com/in/foo
#   /in/foo
# and returns "foo" (lowercased, trailing slash and query stripped).
_PUBLIC_LINKEDIN_SLUG_RE = re.compile(
    r"(?:https?://)?(?:[a-z]+\.)?linkedin\.com/in/([A-Za-z0-9._\-%]+)|^/in/([A-Za-z0-9._\-%]+)",
    re.IGNORECASE,
)


def _is_recruiter_talent_url(url: str) -> bool:
    """True for Recruiter profile URLs (``/talent/profile/...``).

    The surface-time direct-hint anchor (Recruiter-Identity-Collection-Followups
    §5) only fires for these; public ``/in/...`` URLs require post-open
    comparison because the Recruiter card surface never carries them.
    """
    return "/talent/profile/" in (url or "")


def normalize_public_linkedin_handle(value: str) -> str:
    """Return the lowercase slug for a public LinkedIn URL or empty string.

    Examples:
      ``"https://www.linkedin.com/in/Foo-Bar/"`` -> ``"foo-bar"``
      ``"linkedin.com/in/foo"`` -> ``"foo"``
      ``"/in/foo"`` -> ``"foo"``
      ``"/talent/profile/x"`` -> ``""`` (Recruiter URLs do not carry public
      handles)
    """
    if not value:
        return ""
    match = _PUBLIC_LINKEDIN_SLUG_RE.search(str(value))
    if not match:
        return ""
    slug = match.group(1) or match.group(2) or ""
    slug = slug.strip().lower()
    return slug.rstrip("/").split("?", 1)[0].split("#", 1)[0]


def _name_tokens_lower(value: str) -> list[str]:
    return [token.lower() for token in _NAME_TOKEN_RE.findall(str(value or ""))]


def _normalized_names_match(expected: str, actual: str) -> bool:
    expected_norm = normalize_person_name(expected)
    actual_norm = normalize_person_name(actual)
    return bool(expected_norm) and expected_norm == actual_norm


def _names_share_strong_overlap(expected: str, actual: str) -> bool:
    """True for exact normalized match OR strong token overlap (initialized variants).

    Mirrors the looser plausibility floor used at card-surface scoring so that a
    Recruiter card whose surface only carried "Eri B." can still be treated as a
    near-exact card name when the full opened profile name is "Eri Barrett".
    """
    if _normalized_names_match(expected, actual):
        return True
    expected_tokens = set(_name_tokens_lower(expected))
    actual_tokens = set(_name_tokens_lower(actual))
    if not expected_tokens or not actual_tokens:
        return False
    overlap = len(expected_tokens & actual_tokens)
    smaller = min(len(expected_tokens), len(actual_tokens))
    return overlap >= max(1, smaller - 1)


def _structural_overlaps_count(
    *,
    hints: LinkedInIdentityHints,
    profile_summary: CandidateProfileSummary | None,
) -> int:
    """Count of structural fields (company / title / location) that share any token
    between the GitHub identity hints and the opened Recruiter profile summary.

    Used as a tiebreaker when names are an exact/near-exact match but more than
    one Recruiter result could plausibly be the same person; at least one of
    company/title/location must overlap to elevate identity to ``confirmed``.
    """
    if profile_summary is None:
        return 0
    overlaps = 0

    expected_company = set(_name_tokens_lower(normalize_company_name(hints.company)))
    actual_company_tokens: set[str] = set()
    for exp in profile_summary.experiences:
        actual_company_tokens.update(_name_tokens_lower(normalize_company_name(exp.company)))
    if expected_company and actual_company_tokens and (expected_company & actual_company_tokens):
        overlaps += 1

    expected_title = set(_name_tokens_lower(hints.title))
    actual_title_tokens: set[str] = set()
    for exp in profile_summary.experiences:
        actual_title_tokens.update(_name_tokens_lower(exp.title))
    actual_title_tokens.update(_name_tokens_lower(profile_summary.headline))
    if expected_title and actual_title_tokens and (expected_title & actual_title_tokens):
        overlaps += 1

    expected_location = set(_name_tokens_lower(normalize_location_text(hints.location)))
    actual_location_tokens: set[str] = set()
    for exp in profile_summary.experiences:
        actual_location_tokens.update(_name_tokens_lower(normalize_location_text(exp.location)))
    if expected_location and actual_location_tokens and (expected_location & actual_location_tokens):
        overlaps += 1

    return overlaps


def compute_post_open_identity_status(
    *,
    hints: LinkedInIdentityHints,
    candidate: RecruiterIdentityCandidate,
    profile_summary: CandidateProfileSummary | None,
    extraction_failed: bool,
) -> tuple[str, str]:
    """Decide identity status after a Recruiter profile was opened (plan §4).

    Returns a ``(status, subreason)`` tuple where ``status`` is one of:

    - ``"confirmed"`` when one of these holds:
        * The opened profile's normalized name exactly matches the GitHub hint name.
        * A direct LinkedIn URL hint matched the surfaced Recruiter card (the
          surface-time anchor evidence is propagated through the candidate).
        * The card's surface name was an exact/near-exact match for the GitHub
          name AND the opened profile shares ≥1 structural overlap from
          company/title/location with the GitHub hints.
    - ``"ambiguous"`` when the opened profile is plausible but the structural
      anchors are too weak to confirm identity.
    - ``"no_match"`` when the opened profile contradicts the GitHub identity
      (no name overlap and no direct-hint anchor).
    - ``"tool_failure"`` when extraction/open/read failed.

    The function is pure (no I/O) so it can be unit-tested without a browser.
    """
    if extraction_failed or profile_summary is None:
        return "tool_failure", "extraction_failed"

    direct_hint_anchored = DIRECT_HINT_EVIDENCE in candidate.evidence
    profile_name = profile_summary.name or ""
    card_name = candidate.name or ""
    expected_name = hints.candidate_name

    if _normalized_names_match(expected_name, profile_name):
        return "confirmed", "exact_normalized_profile_name_match"

    if direct_hint_anchored:
        return "confirmed", "direct_linkedin_hint_anchor"

    card_name_strong = _names_share_strong_overlap(expected_name, card_name)
    structural_count = _structural_overlaps_count(hints=hints, profile_summary=profile_summary)
    if card_name_strong and structural_count >= 1:
        return "confirmed", "card_name_plus_structural_overlap"

    expected_tokens = set(_name_tokens_lower(expected_name))
    profile_tokens = set(_name_tokens_lower(profile_name))
    if expected_tokens and profile_tokens and not (expected_tokens & profile_tokens):
        return "no_match", "profile_name_contradicts_github_identity"

    return "ambiguous", "post_open_anchors_too_weak"


def _extract_card_identity(snapshot: dict) -> dict:
    innertext = str(snapshot.get("innertext", "") or "")
    lines = [line.strip() for line in innertext.splitlines() if line.strip()]
    name = str(snapshot.get("name", "") or "").strip()
    headline = ""
    location = ""
    current_title = ""
    current_company = ""

    for line in lines:
        lowered = line.lower()
        if (
            not headline
            and line != name
            and "save to pipeline" not in lowered
            and "change stage" not in lowered
        ):
            headline = line
        if not location and " · " in line and " at " not in line:
            location = line.split(" · ", 1)[0].strip()
        if not current_title and " at " in line:
            current_title, current_company = [part.strip() for part in line.split(" at ", 1)]
            break

    return {
        "name": name,
        "headline": headline,
        "current_title": current_title,
        "current_company": current_company,
        "location": location,
        "profile_url": str(snapshot.get("url", "") or "").strip(),
        "already_saved": bool(snapshot.get("already_saved", False)),
        "raw_card_text": innertext,
    }


class RecruiterIdentityResolver:
    """Resolve a GitHub lead inside a prepared Recruiter search (identity + fit + engagement)."""

    def __init__(
        self,
        *,
        browser: LinkedInBrowser,
        project_url: str = "",
        config: RecruiterResolverConfig | None = None,
        linkedin_brief: Brief | None = None,
        linkedin_brief_path: str = "",
    ):
        self.browser = browser
        self.project_url = project_url
        self.config = config or RecruiterResolverConfig()
        self.linkedin_brief = linkedin_brief
        self.linkedin_brief_path = linkedin_brief_path or ""
        self.search_location = ""
        self._prepared = False
        # Last query string the browser actually issued to Recruiter. The resolver
        # uses this to decide whether a replay is needed before
        # opening/saving a candidate that came from a different bounded query
        # (Recruiter-Identity-Collection-Followups §2). Empty means "we have not
        # yet entered any per-lead query string this run".
        self._current_browser_query: str = ""
        # True when the operator attached to a Recruiter search they prepared
        # manually (use_existing_search) rather than letting the resolver
        # navigate + apply filters (prepare_search). The query-expansion
        # policy in "auto" mode treats current-search runs as pre-filtered and
        # avoids enriched location/title keyword fallbacks (Recruiter-Identity-
        # Collection-Cycle-Audit-Fixes §1).
        self._use_current_search: bool = False

    async def prepare_search(self, search_location: str = "") -> None:
        """Open the Recruiter search surface and optionally set a fixed location filter."""
        self.search_location = canonicalize_location_label(search_location)
        if not self.project_url:
            raise ValueError("project_url is required when the resolver is expected to navigate")
        await self.browser.navigate_to_search(self.project_url)
        if self.search_location:
            await self.browser.apply_permanent_filters({"location": self.search_location})
        self._prepared = True
        self._use_current_search = False

    async def use_existing_search(self, search_location: str = "") -> None:
        """Use the Recruiter search page that the operator has already prepared manually."""
        self.search_location = canonicalize_location_label(search_location)
        await self.browser.go_back_to_results()
        self._prepared = True
        self._use_current_search = True

    def _effective_query_expansion_policy(self) -> str:
        """Resolve ``query_expansion_policy`` to a concrete ``"name_first"`` or
        ``"enriched"`` value (Recruiter-Identity-Collection-Cycle-Audit-Fixes §1).

        - ``"name_first"`` / ``"enriched"`` are returned as-is.
        - ``"auto"`` derives:
          * ``identity_collect`` AND (``_use_current_search`` OR a fixed
            ``search_location``) → ``"name_first"`` (avoid mutating operator-
            applied filters with keyword constraints).
          * otherwise → ``"enriched"`` (legacy bounded plan).
        - Any other configured value falls back to ``"enriched"`` so an unknown
          policy never silently turns into a stricter behavior.
        """
        configured = (self.config.query_expansion_policy or "auto").strip().lower()
        if configured == "name_first":
            return "name_first"
        if configured == "enriched":
            return "enriched"
        identity_collect = self.config.workflow_mode == "identity_collect"
        if identity_collect and (self._use_current_search or bool(self.search_location)):
            return "name_first"
        return "enriched"

    @staticmethod
    def _opus_from_review(review: PlausibleProfileReview) -> OpusDecision | None:
        if not (review.holistic_fit_decision or "").strip():
            return None
        return OpusDecision(
            stage="full",
            decision=review.holistic_fit_decision,
            path=review.holistic_fit_path,
            confidence=review.holistic_fit_confidence,
            rationale=review.holistic_fit_rationale,
            candidate_name=str(review.holistic_profile_summary.get("name", "") or ""),
            profile_url=review.profile_url,
        )

    def _apply_terminal_outcome(
        self,
        resolution: RecruiterIdentityResolution,
        *,
        identity_classification: str,
        identity_high_confidence: bool,
        identity_name_mismatch: bool,
        had_plausible_cards: bool,
        holistic_decision: OpusDecision | None,
        profile_summary: CandidateProfileSummary | None,
        already_saved_card: bool,
        profile_status: RecruiterActivitySnapshot | None,
        novelty_pressure: str,
        reachout_status: str,
        extraction_failed: bool,
    ) -> None:
        resolution.identity_classification = identity_classification
        resolution.had_plausible_cards = had_plausible_cards
        if holistic_decision is not None:
            resolution.holistic_fit_decision = holistic_decision.decision
            resolution.holistic_fit_confidence = float(holistic_decision.confidence or 0.0)
            resolution.holistic_fit_rationale = str(holistic_decision.rationale or "")
            resolution.holistic_fit_path = str(holistic_decision.path or "")
        gate = decide_final_reconciliation_action(
            identity_high_confidence=identity_high_confidence,
            identity_name_mismatch=identity_name_mismatch,
            had_plausible_cards=had_plausible_cards,
            holistic_decision=holistic_decision,
            profile_summary=profile_summary,
            already_saved_card=already_saved_card,
            profile_status=profile_status,
            novelty_pressure=novelty_pressure,
            reachout_status=reachout_status,
            extraction_failed=extraction_failed,
        )
        resolution.final_action = gate.final_action
        resolution.final_subreason = gate.final_subreason

        # In identity_collect mode every terminal row must populate the canonical
        # identity-collection fields, including non-open exits like no_results,
        # surface-level ambiguity, and no-confident-match (Recruiter-Identity-
        # Collection-Followups §4). The mapping below covers the non-open paths;
        # opened-profile paths populate these fields directly in
        # _apply_identity_collect_outcome / multi-profile consolidators.
        if self.config.workflow_mode == "identity_collect":
            self._backfill_identity_collect_terminal_fields(
                resolution,
                identity_classification=identity_classification,
                extraction_failed=extraction_failed,
            )

    @staticmethod
    def _backfill_identity_collect_terminal_fields(
        resolution: RecruiterIdentityResolution,
        *,
        identity_classification: str,
        extraction_failed: bool,
    ) -> None:
        """Populate canonical identity_collect fields on non-open terminal paths.

        Mapping (Recruiter-Identity-Collection-Followups §4):
          - extraction_failed → tool_failure / MANUAL_REVIEW / not_attempted
          - no_results        → no_match / REJECT / not_attempted
          - everything else (manual_review, no_confident_match, high_confidence
            without open) → ambiguous / MANUAL_REVIEW / not_attempted
        """
        # Don't overwrite values that an opened-profile path already set.
        if resolution.identity_status or resolution.collection_action:
            if not resolution.project_save_state:
                resolution.project_save_state = "not_attempted"
            return

        if extraction_failed:
            resolution.identity_status = "tool_failure"
            resolution.identity_subreason = "extraction_failed"
            resolution.collection_action = "MANUAL_REVIEW"
            resolution.collection_subreason = "tool_failure"
        elif identity_classification == "no_results":
            resolution.identity_status = "no_match"
            resolution.identity_subreason = "no_recruiter_results_surfaced"
            resolution.collection_action = "REJECT"
            resolution.collection_subreason = "no_recruiter_results"
        else:
            resolution.identity_status = "ambiguous"
            resolution.identity_subreason = identity_classification or "surface_ambiguous"
            resolution.collection_action = "MANUAL_REVIEW"
            resolution.collection_subreason = identity_classification or "surface_ambiguous"
        resolution.project_save_state = "not_attempted"

    def _hydrate_resolution_from_review(
        self,
        resolution: RecruiterIdentityResolution,
        candidate: RecruiterIdentityCandidate,
        review: PlausibleProfileReview,
        had_plausible_cards: bool,
        *,
        identity_classification: str,
    ) -> None:
        resolution.selected_candidate_rank = candidate.rank
        resolution.selected_profile_url = candidate.profile_url
        resolution.already_saved = candidate.already_saved
        resolution.profile_status = RecruiterActivitySnapshot.from_dict(review.profile_status)
        resolution.holistic_profile_summary = dict(review.holistic_profile_summary)
        resolution.novelty_pressure = review.novelty_pressure
        resolution.reachout_status = review.reachout_status
        resolution.extraction_failed = review.extraction_failed

        summary: CandidateProfileSummary | None = None
        if review.holistic_profile_summary:
            try:
                summary = CandidateProfileSummary.from_dict(review.holistic_profile_summary)
            except Exception:
                summary = None

        self._apply_terminal_outcome(
            resolution,
            identity_classification=identity_classification,
            identity_high_confidence=True,
            identity_name_mismatch=False,
            had_plausible_cards=had_plausible_cards,
            holistic_decision=self._opus_from_review(review),
            profile_summary=summary,
            already_saved_card=candidate.already_saved,
            profile_status=resolution.profile_status,
            novelty_pressure=review.novelty_pressure,
            reachout_status=review.reachout_status,
            extraction_failed=review.extraction_failed,
        )

    @staticmethod
    def _clear_ambiguous_resolution_selection_fields(resolution: RecruiterIdentityResolution) -> None:
        """No-winner multi-review: top-level row must not imply a single chosen profile or one review's holistic outcome."""
        resolution.selected_candidate_rank = 0
        resolution.selected_profile_url = ""
        resolution.already_saved = False
        resolution.holistic_fit_decision = ""
        resolution.holistic_fit_confidence = 0.0
        resolution.holistic_fit_rationale = ""
        resolution.holistic_fit_path = ""
        resolution.holistic_profile_summary = {}
        resolution.profile_status = None
        resolution.novelty_pressure = ""
        resolution.reachout_status = ""
        resolution.extraction_failed = False

    @staticmethod
    def _format_multi_profile_ambiguity_note(
        reviews: list[PlausibleProfileReview],
        outcome: MultiProfileOutcome,
    ) -> str:
        parts = [
            f"rank{r.rank} {r.profile_url or '(no url)'}: gate {r.gate_final_action or 'unknown'}"
            + (f"/{r.gate_final_subreason}" if (r.gate_final_subreason or "").strip() else "")
            for r in reviews
        ]
        joined = "; ".join(parts)
        reason = (
            f"No unique SAVE winner after opening {len(reviews)} plausible profile(s). "
            f"Consolidation outcome: {outcome.final_action}"
            + (f" ({outcome.final_subreason}). " if outcome.final_subreason else ". ")
            + f"Per-profile gates: {joined}"
        )
        return reason

    async def _replay_query_for_candidate(
        self,
        candidate: RecruiterIdentityCandidate,
    ) -> int | None:
        """Ensure the browser is on the query that surfaced ``candidate`` and return
        the card's current index in that result list (or ``None`` if the URL was
        not located among the visible top-N cards on the replayed query).

        Required because the multi-query lookup leaves the browser on whichever
        query happened to be processed last, which may not contain the selected
        candidate at all (Recruiter-Identity-Collection-Followups §2).

        Behavior:
          - If the candidate carries no ``surfaced_query`` (single-query lookup
            or already-on-the-right-page) and the candidate's renumbered rank
            >= 1, returns ``rank - 1`` so callers can keep using rank-based
            focus. No browser navigation is issued in that case.
          - Otherwise re-enters ``surfaced_query``, scans up to ``max_cards``
            visible card snapshots, and returns the index whose URL matches
            ``candidate.profile_url``. Returns ``None`` if no visible card matched
            (callers should fall back to URL-based open without rank focus).
        """
        surfaced_query = (candidate.surfaced_query or "").strip()
        if not surfaced_query:
            rank_index = candidate.rank - 1
            return rank_index if rank_index >= 0 else None

        if surfaced_query != self._current_browser_query:
            await self.browser.enter_search_string(surfaced_query)
            self._current_browser_query = surfaced_query

        target_url = (candidate.profile_url or "").strip()
        if not target_url:
            return None

        try:
            slot_count = await self.browser.get_card_slot_count()
            if slot_count == 0:
                slot_count = await self.browser.get_card_count()
        except Exception:
            slot_count = 0
        cards_to_scan = min(int(slot_count or 0), max(self.config.max_cards, 1))
        for card_index in range(cards_to_scan):
            try:
                snapshot = await self.browser.get_card_snapshot(card_index)
            except Exception:
                continue
            snapshot_url = str((snapshot or {}).get("url", "") or "").strip()
            if snapshot_url == target_url:
                return card_index
        return None

    async def _focus_then_open_candidate(
        self,
        candidate: RecruiterIdentityCandidate,
    ) -> None:
        """Replay the surfacing query and open the candidate's profile slide-in.

        Wraps :meth:`_replay_query_for_candidate` plus the focus-card / open-by-url
        click sequence so callers don't repeat the boilerplate.
        """
        replay_index = await self._replay_query_for_candidate(candidate)
        if replay_index is not None:
            await self.browser.focus_card_for_review(replay_index)
        await self.browser.open_profile_by_url(candidate.profile_url)

    async def _read_one_plausible_profile_review(
        self,
        candidate: RecruiterIdentityCandidate,
        *,
        hints: LinkedInIdentityHints | None = None,
    ) -> PlausibleProfileReview:
        """Open one plausible card, read the profile, and emit a review.

        Behavior is mode-aware:
          - ``fit_gated_save``: legacy path. Reads profile innertext, calls
            :func:`full_judge`, and runs :func:`decide_final_reconciliation_action`
            assuming identity-high. Returns a review whose ``identity_status`` is
            also populated for downstream multi-profile ambiguity (Slice 5).
          - ``identity_collect``: skips :func:`full_judge` entirely (plan §4).
            The opened profile is parsed via :func:`extract_profile_from_innertext`
            so identity confirmation can run, but no holistic-fit decision is
            produced and no reconciliation gate is invoked. Holistic fields stay
            empty; novelty/reachout are recorded as annotations only.

        Replays the candidate's surfacing query before opening when needed
        (Recruiter-Identity-Collection-Followups §2).
        """
        identity_collect = self.config.workflow_mode == "identity_collect"
        await self._focus_then_open_candidate(candidate)

        # Phase 1: read profile status + innertext + extracted summary. NEVER
        # call full_judge here, even in fit_gated_save (Recruiter-Identity-
        # Collection-Followups §3). full_judge is deferred until after the
        # post-open identity check in Phase 2.
        extraction_failed = False
        profile_summary: CandidateProfileSummary | None = None
        profile_status: RecruiterActivitySnapshot | None = None
        profile_text: str = ""
        novelty_pressure = ""
        reachout_status = ""

        try:
            profile_status = RecruiterActivitySnapshot.from_dict(
                await self.browser.get_profile_status_summary()
            )
            if profile_status is not None:
                candidate.recruiter_activity = profile_status
            novelty_pressure = classify_recruiter_activity_pressure(profile_status)
            reachout_status = infer_reachout_status(profile_status)

            await self.browser.simulate_profile_read()
            profile_text = await self.browser.get_profile_innertext() or ""
            if not identity_collect and self.linkedin_brief is None:
                raise RuntimeError("linkedin_brief is required for holistic evaluation")
            profile_summary = extract_profile_from_innertext(profile_text, candidate.profile_url)
        except Exception:
            extraction_failed = True
            profile_summary = None

        # Public LinkedIn hint anchoring (Recruiter-Identity-Collection-
        # Followups §5). Surface-time anchoring only fires for Recruiter URL
        # hints; public ``/in/<slug>`` hints are matched here against the
        # opened-profile innertext before identity status is computed so the
        # confirmed-via-direct-hint path can fire.
        if hints is not None and not extraction_failed and profile_text:
            self._apply_post_open_public_hint_anchor(candidate, hints, profile_text)

        # Phase 2: compute post-open identity status from the freshly-read
        # profile summary. Fit_gated_save only runs full_judge AFTER identity
        # confirms; identity_collect never runs full_judge.
        identity_status, identity_subreason = (
            ("", "")
            if hints is None
            else compute_post_open_identity_status(
                hints=hints,
                candidate=candidate,
                profile_summary=profile_summary,
                extraction_failed=extraction_failed,
            )
        )

        holistic_decision: OpusDecision | None = None
        gate_final_action = ""
        gate_final_subreason = ""
        if not identity_collect and identity_status == "confirmed" and not extraction_failed:
            try:
                holistic_decision = full_judge(profile_summary, brief=self.linkedin_brief)
            except Exception:
                # Treat a judge failure on a confirmed identity as a tool
                # failure rather than silently skipping the gate; the caller
                # routes this through the synthetic identity-only terminal in
                # _apply_unconfirmed_identity_terminal.
                holistic_decision = None
                extraction_failed = True
            else:
                gate = decide_final_reconciliation_action(
                    identity_high_confidence=True,
                    identity_name_mismatch=False,
                    had_plausible_cards=True,
                    holistic_decision=holistic_decision,
                    profile_summary=profile_summary,
                    already_saved_card=candidate.already_saved,
                    profile_status=profile_status,
                    novelty_pressure=novelty_pressure,
                    reachout_status=reachout_status,
                    extraction_failed=False,
                )
                gate_final_action = gate.final_action
                gate_final_subreason = gate.final_subreason

        status_dict = profile_status.to_dict() if profile_status else None
        review = PlausibleProfileReview(
            rank=candidate.rank,
            profile_url=candidate.profile_url,
            card_name=candidate.name,
            match_confidence=candidate.match_confidence,
            extraction_failed=extraction_failed,
            holistic_fit_decision=holistic_decision.decision if holistic_decision else "",
            holistic_fit_confidence=float(holistic_decision.confidence or 0.0) if holistic_decision else 0.0,
            holistic_fit_rationale=str(holistic_decision.rationale or "") if holistic_decision else "",
            holistic_fit_path=str(holistic_decision.path or "") if holistic_decision else "",
            holistic_profile_summary=profile_summary.to_dict() if profile_summary else {},
            profile_status=status_dict,
            novelty_pressure=novelty_pressure,
            reachout_status=reachout_status,
            gate_final_action=gate_final_action,
            gate_final_subreason=gate_final_subreason,
            identity_status=identity_status,
            identity_subreason=identity_subreason,
        )

        try:
            await self.browser.go_back_to_results()
        except Exception:
            pass
        return review

    async def _reopen_and_save_if_save(
        self,
        candidate: RecruiterIdentityCandidate,
        resolution: RecruiterIdentityResolution,
    ) -> None:
        if resolution.final_action != "SAVE":
            return
        if self.config.dry_run_save:
            resolution.notes.append("Dry-run: skipped Recruiter save click.")
            return
        resolution.recruiter_save_attempted = True
        try:
            # Replay the candidate's surfacing query so the click targets the
            # correct result-list DOM (Recruiter-Identity-Collection-Followups §2).
            await self._focus_then_open_candidate(candidate)
            ok = await self.browser.save_candidate()
            resolution.recruiter_save_succeeded = bool(ok)
            if not resolution.recruiter_save_succeeded:
                resolution.final_action = "MANUAL_REVIEW"
                resolution.final_subreason = "tool_failure"
                resolution.notes.append("Recruiter save did not persist after click.")
        except Exception as exc:
            resolution.recruiter_save_succeeded = False
            resolution.final_action = "MANUAL_REVIEW"
            resolution.final_subreason = "tool_failure"
            resolution.notes.append(f"Recruiter save failed: {exc}")
        try:
            await self.browser.go_back_to_results()
        except Exception:
            pass

    async def _attempt_identity_collect_save(
        self,
        candidate: RecruiterIdentityCandidate,
        resolution: RecruiterIdentityResolution,
    ) -> None:
        """Attempt the Recruiter save for a confirmed-identity profile in identity_collect mode.

        Sets ``project_save_state`` to one of:
          - ``"already_saved"``: the surfaced card already had the saved badge.
          - ``"dry_run_skipped"``: ``dry_run_save`` is set on the config.
          - ``"saved_now"``: save_candidate returned True.
          - ``"save_failed"``: save_candidate returned False or threw.

        Replays the candidate's surfacing query before clicking save so
        cross-query selections still target the correct result-list DOM
        (Recruiter-Identity-Collection-Followups §2).
        """
        if candidate.already_saved:
            resolution.project_save_state = "already_saved"
            resolution.notes.append("Identity confirmed; profile already saved in this Recruiter project.")
            return
        if self.config.dry_run_save:
            resolution.project_save_state = "dry_run_skipped"
            resolution.notes.append("Dry-run: skipped Recruiter save click on confirmed identity.")
            return
        resolution.recruiter_save_attempted = True
        try:
            await self._focus_then_open_candidate(candidate)
            ok = await self.browser.save_candidate()
            resolution.recruiter_save_succeeded = bool(ok)
            if resolution.recruiter_save_succeeded:
                resolution.project_save_state = "saved_now"
            else:
                resolution.project_save_state = "save_failed"
                resolution.notes.append("Recruiter save did not persist after click.")
        except Exception as exc:
            resolution.recruiter_save_succeeded = False
            resolution.project_save_state = "save_failed"
            resolution.notes.append(f"Recruiter save failed: {exc}")
        try:
            await self.browser.go_back_to_results()
        except Exception:
            pass

    async def _resolve_ambiguity_among_plausible(
        self,
        to_review: list[RecruiterIdentityCandidate],
        resolution: RecruiterIdentityResolution,
        *,
        had_plausible_cards: bool,
        hints: LinkedInIdentityHints,
    ) -> None:
        resolution.ambiguity_multi_review = True
        self._clear_ambiguous_resolution_selection_fields(resolution)
        queued = [c for c in to_review if (c.profile_url or "").strip()]
        reviews: list[PlausibleProfileReview] = []
        resolution.opened_profile = False
        for candidate in queued:
            review = await self._read_one_plausible_profile_review(candidate, hints=hints)
            reviews.append(review)
            resolution.plausible_profile_reviews.append(review)
            resolution.opened_profile = True

        # Identity-first ordering (plan §5). Both modes now treat identity
        # confirmation as the primary axis; the legacy "exactly one SAVE wins"
        # rule is no longer the primary ambiguity resolver.
        if self.config.workflow_mode == "identity_collect":
            await self._consolidate_multi_profile_identity_collect(
                queued=queued,
                reviews=reviews,
                resolution=resolution,
                had_plausible_cards=had_plausible_cards,
            )
            return
        await self._consolidate_multi_profile_fit_gated_save(
            queued=queued,
            reviews=reviews,
            resolution=resolution,
            had_plausible_cards=had_plausible_cards,
        )

    async def _consolidate_multi_profile_fit_gated_save(
        self,
        *,
        queued: list[RecruiterIdentityCandidate],
        reviews: list[PlausibleProfileReview],
        resolution: RecruiterIdentityResolution,
        had_plausible_cards: bool,
    ) -> None:
        """Identity-first multi-profile consolidation for fit_gated_save mode (plan §5).

        Order of operations:
          - Filter to ``identity_status == "confirmed"`` reviews.
          - Exactly one confirmed → run the existing fit/engagement gate ON that
            confirmed profile (not on the whole set), then SAVE if the gate
            authorizes it.
          - Zero confirmed AND any plausible review existed → MANUAL_REVIEW with
            ``identity_ambiguous`` (or REJECT when all reviews say no_match).
          - More than one confirmed → MANUAL_REVIEW with
            ``multiple_confirmed_identities``. The legacy "exactly one SAVE wins"
            rule is intentionally no longer the primary ambiguity resolver.
        """
        confirmed_indices = [
            i for i, review in enumerate(reviews) if review.identity_status == "confirmed"
        ]
        had_plausible_cards_artifact = bool(had_plausible_cards or reviews)

        if len(confirmed_indices) == 1:
            idx = confirmed_indices[0]
            winner = queued[idx]
            winner_review = reviews[idx]
            resolution.rationale = (
                "Multi-profile identity-first review: one opened profile was identity-"
                f"confirmed (rank {winner.rank}); applying fit/engagement gate."
            )
            resolution.identity_status = winner_review.identity_status
            resolution.identity_subreason = winner_review.identity_subreason
            self._hydrate_resolution_from_review(
                resolution,
                winner,
                winner_review,
                had_plausible_cards_artifact,
                identity_classification="ambiguity_multi_review",
            )
            if resolution.final_action == "SAVE":
                await self._reopen_and_save_if_save(winner, resolution)
            return

        if len(confirmed_indices) >= 2:
            resolution.identity_status = "ambiguous"
            resolution.identity_subreason = "multiple_confirmed_identities"
            resolution.final_action = "MANUAL_REVIEW"
            resolution.final_subreason = "multiple_confirmed_identities"
            resolution.identity_classification = "ambiguity_multi_review"
            resolution.had_plausible_cards = had_plausible_cards_artifact
            self._clear_ambiguous_resolution_selection_fields(resolution)
            resolution.rationale = (
                f"Multi-profile review ({len(reviews)} candidates): more than one opened "
                "profile passed post-open identity confirmation; flagged for manual review "
                "without invoking fit (identity-first ordering)."
            )
            return

        statuses = {review.identity_status for review in reviews}
        if statuses == {"no_match"} and reviews:
            resolution.identity_status = "no_match"
            resolution.final_action = "REJECT"
            resolution.final_subreason = "identity_no_match"
        else:
            resolution.identity_status = "ambiguous"
            resolution.final_action = "MANUAL_REVIEW"
            resolution.final_subreason = "identity_ambiguous"
        resolution.identity_classification = "ambiguity_multi_review"
        resolution.had_plausible_cards = had_plausible_cards_artifact
        self._clear_ambiguous_resolution_selection_fields(resolution)
        resolution.rationale = (
            f"Multi-profile review ({len(reviews)} candidates): no opened profile passed "
            "post-open identity confirmation; legacy fit/engagement gate is not invoked."
        )

    async def _consolidate_multi_profile_identity_collect(
        self,
        *,
        queued: list[RecruiterIdentityCandidate],
        reviews: list[PlausibleProfileReview],
        resolution: RecruiterIdentityResolution,
        had_plausible_cards: bool,
    ) -> None:
        """Identity-first multi-profile consolidation for identity_collect mode.

        Order of operations (plan §5):
          - Filter reviews to those with ``identity_status == "confirmed"``.
          - Exactly one confirmed identity → collect that profile.
          - Zero confirmed identities → MANUAL_REVIEW (or REJECT for all-no-match).
          - More than one confirmed identity → MANUAL_REVIEW with an identity
            ambiguity subreason.

        Slice 5 will extend the same shape to fit_gated_save.
        """
        confirmed_indices = [
            i for i, review in enumerate(reviews) if review.identity_status == "confirmed"
        ]
        had_plausible_cards_artifact = bool(had_plausible_cards or reviews)

        if len(confirmed_indices) == 1:
            idx = confirmed_indices[0]
            winner = queued[idx]
            winner_review = reviews[idx]
            resolution.rationale = (
                "Multi-profile identity-first review: exactly one opened profile was "
                f"identity-confirmed (rank {winner.rank})."
            )
            resolution.identity_status = "confirmed"
            resolution.identity_subreason = winner_review.identity_subreason
            await self._apply_identity_collect_outcome(
                selected_candidate=winner,
                review=winner_review,
                resolution=resolution,
                had_plausible_cards=had_plausible_cards_artifact,
                identity_classification="ambiguity_multi_review",
            )
            return

        if len(confirmed_indices) >= 2:
            resolution.identity_status = "ambiguous"
            resolution.identity_subreason = "multiple_confirmed_identities"
            resolution.collection_action = "MANUAL_REVIEW"
            resolution.collection_subreason = "multiple_confirmed_identities"
            resolution.project_save_state = "not_attempted"
            resolution.identity_classification = "ambiguity_multi_review"
            resolution.had_plausible_cards = had_plausible_cards_artifact
            self._clear_ambiguous_resolution_selection_fields(resolution)
            resolution.final_action = "MANUAL_REVIEW"
            resolution.final_subreason = "multiple_confirmed_identities"
            resolution.rationale = (
                f"Multi-profile review ({len(reviews)} candidates): more than one opened "
                "profile passed post-open identity confirmation; flagged for manual review."
            )
            return

        statuses = {review.identity_status for review in reviews}
        if statuses == {"no_match"} and reviews:
            resolution.identity_status = "no_match"
            resolution.collection_action = "REJECT"
            resolution.collection_subreason = "all_profiles_contradicted_identity"
            resolution.project_save_state = "not_attempted"
            resolution.final_action = "REJECT"
            resolution.final_subreason = "identity_no_match"
        else:
            resolution.identity_status = "ambiguous"
            resolution.collection_action = "MANUAL_REVIEW"
            resolution.collection_subreason = "no_confirmed_identity_among_plausible"
            resolution.project_save_state = "not_attempted"
            resolution.final_action = "MANUAL_REVIEW"
            resolution.final_subreason = "identity_ambiguous"
        resolution.identity_classification = "ambiguity_multi_review"
        resolution.had_plausible_cards = had_plausible_cards_artifact
        self._clear_ambiguous_resolution_selection_fields(resolution)
        resolution.rationale = (
            f"Multi-profile review ({len(reviews)} candidates): no opened profile passed "
            "post-open identity confirmation."
        )

    async def _evaluate_confirmed_identity_profile(
        self,
        *,
        selected_candidate: RecruiterIdentityCandidate,
        resolution: RecruiterIdentityResolution,
        had_plausible_cards: bool,
        hints: LinkedInIdentityHints,
        identity_classification: str = "high_confidence_match",
    ) -> None:
        review = await self._read_one_plausible_profile_review(selected_candidate, hints=hints)
        resolution.opened_profile = True
        resolution.plausible_profile_reviews = [review]
        resolution.ambiguity_multi_review = False
        resolution.identity_status = review.identity_status
        resolution.identity_subreason = review.identity_subreason

        if self.config.workflow_mode == "identity_collect":
            await self._apply_identity_collect_outcome(
                selected_candidate=selected_candidate,
                review=review,
                resolution=resolution,
                had_plausible_cards=had_plausible_cards,
                identity_classification=identity_classification,
            )
            return

        # fit_gated_save: only run the legacy fit/engagement gate when identity
        # is confirmed (plan §4). When identity is not confirmed, return the
        # equivalent manual/no-match result without invoking fit.
        if review.identity_status == "confirmed":
            self._hydrate_resolution_from_review(
                resolution,
                selected_candidate,
                review,
                had_plausible_cards,
                identity_classification=identity_classification,
            )
            await self._reopen_and_save_if_save(selected_candidate, resolution)
            return

        self._apply_unconfirmed_identity_terminal(
            selected_candidate=selected_candidate,
            review=review,
            resolution=resolution,
            had_plausible_cards=had_plausible_cards,
            identity_classification=identity_classification,
        )

    async def _apply_identity_collect_outcome(
        self,
        *,
        selected_candidate: RecruiterIdentityCandidate,
        review: PlausibleProfileReview,
        resolution: RecruiterIdentityResolution,
        had_plausible_cards: bool,
        identity_classification: str,
    ) -> None:
        """Translate one opened-profile review into identity_collect row state."""
        resolution.identity_classification = identity_classification
        resolution.had_plausible_cards = bool(had_plausible_cards or review)
        resolution.selected_candidate_rank = selected_candidate.rank
        resolution.selected_profile_url = selected_candidate.profile_url
        resolution.already_saved = selected_candidate.already_saved
        resolution.profile_status = RecruiterActivitySnapshot.from_dict(review.profile_status)
        resolution.holistic_profile_summary = dict(review.holistic_profile_summary)
        resolution.novelty_pressure = review.novelty_pressure
        resolution.reachout_status = review.reachout_status
        resolution.extraction_failed = review.extraction_failed
        if review.identity_status == "confirmed":
            resolution.collection_action = "COLLECT"
            resolution.collection_subreason = review.identity_subreason
            await self._attempt_identity_collect_save(selected_candidate, resolution)
            # Compatibility fields (plan §2): keep final_action/subreason populated
            # for legacy consumers that read them, but they must not drive export
            # filtering or summary interpretation in identity_collect mode.
            resolution.final_action = "SAVE"
            resolution.final_subreason = ""
        elif review.identity_status == "tool_failure":
            resolution.collection_action = "MANUAL_REVIEW"
            resolution.collection_subreason = "tool_failure"
            resolution.project_save_state = "not_attempted"
            resolution.final_action = "MANUAL_REVIEW"
            resolution.final_subreason = "tool_failure"
        elif review.identity_status == "no_match":
            resolution.collection_action = "REJECT"
            resolution.collection_subreason = review.identity_subreason or "identity_no_match"
            resolution.project_save_state = "not_attempted"
            resolution.final_action = "REJECT"
            resolution.final_subreason = "identity_no_match"
        else:  # ambiguous or empty
            resolution.collection_action = "MANUAL_REVIEW"
            resolution.collection_subreason = review.identity_subreason or "identity_ambiguous"
            resolution.project_save_state = "not_attempted"
            resolution.final_action = "MANUAL_REVIEW"
            resolution.final_subreason = "identity_ambiguous"

    def _apply_unconfirmed_identity_terminal(
        self,
        *,
        selected_candidate: RecruiterIdentityCandidate,
        review: PlausibleProfileReview,
        resolution: RecruiterIdentityResolution,
        had_plausible_cards: bool,
        identity_classification: str,
    ) -> None:
        """fit_gated_save fallback when post-open identity is NOT confirmed.

        Returns the equivalent manual/no-match result without invoking fit
        (plan §4). The legacy gate is bypassed in this branch; we synthesize
        identity-shaped final_action/final_subreason values directly.
        """
        resolution.identity_classification = identity_classification
        resolution.had_plausible_cards = had_plausible_cards
        resolution.selected_candidate_rank = selected_candidate.rank
        resolution.selected_profile_url = selected_candidate.profile_url
        resolution.already_saved = selected_candidate.already_saved
        resolution.profile_status = RecruiterActivitySnapshot.from_dict(review.profile_status)
        resolution.novelty_pressure = review.novelty_pressure
        resolution.reachout_status = review.reachout_status
        resolution.extraction_failed = review.extraction_failed
        if review.identity_status == "tool_failure":
            resolution.final_action = "MANUAL_REVIEW"
            resolution.final_subreason = "tool_failure"
        elif review.identity_status == "no_match":
            resolution.final_action = "REJECT"
            resolution.final_subreason = "identity_no_match"
        else:
            resolution.final_action = "MANUAL_REVIEW"
            resolution.final_subreason = "identity_ambiguous"

    @staticmethod
    def _name_tokens(value: str) -> list[str]:
        return [token.lower() for token in _NAME_TOKEN_RE.findall(str(value or ""))]

    @classmethod
    def _has_initialized_surname_anchor(cls, expected_name: str, matched_name: str) -> bool:
        expected = cls._name_tokens(expected_name)
        matched = cls._name_tokens(matched_name)
        if len(expected) < 2 or len(matched) < 2:
            return False
        if expected[0] != matched[0]:
            return False
        expected_last = expected[-1]
        matched_last = matched[-1]
        if len(matched_last) == 1 and expected_last.startswith(matched_last):
            return True
        if len(expected_last) == 1 and matched_last.startswith(expected_last):
            return True
        return False

    @classmethod
    def _sole_candidate_merits_identity_confirmation_open(
        cls,
        *,
        expected_name: str,
        candidate: RecruiterIdentityCandidate,
        surfaced_count: int,
    ) -> bool:
        """Open a sole surfaced card when the name looks like an initialized variant.

        This covers cases like ``Eri Barrett`` -> ``Eri B.`` where the card surface is
        too sparse to clear the normal plausibility floor, but rejecting without opening
        the only surfaced result is too literal and throws away obvious identity leads.
        """
        if surfaced_count != 1:
            return False
        if not (candidate.profile_url or "").strip():
            return False
        if any("name mismatch" in reason.lower() for reason in candidate.ambiguity_reasons):
            return False
        if not cls._has_initialized_surname_anchor(expected_name, candidate.name):
            return False
        return any("strong partial name overlap" in item.lower() for item in candidate.evidence)

    async def resolve_lead(self, lead: GitHubReconciliationLead) -> RecruiterIdentityResolution:
        if not self._prepared:
            await self.prepare_search("")

        hints = lead.linkedin_hints or LinkedInIdentityHints(
            candidate_name=lead.candidate_name,
            github_username=lead.username,
            github_url=lead.github_url,
            company=lead.company,
            location=lead.location,
            title=lead.title,
            source_query=lead.source_query,
            source_channel=lead.source_channel,
        )
        lookup_name = build_person_lookup_name(lead.candidate_name, lead.username)
        if not lookup_name:
            raise ValueError(f"Lead {lead.username} is missing a usable lookup name")

        primary_queries, fallback_queries = self._build_query_plan(hints, lookup_name)
        planned_queries = list(primary_queries) + list(fallback_queries)
        resolution = RecruiterIdentityResolution(
            github_username=lead.username,
            candidate_name=lead.candidate_name,
            lookup_name=lookup_name,
            github_url=lead.github_url,
            github_company=lead.company,
            github_location=lead.location,
            github_title=lead.title,
            search_location=self.search_location,
            query=lookup_name,
            linkedin_brief_path=self.linkedin_brief_path,
            workflow_mode=self.config.workflow_mode,
            # Recruiter-Identity-Collection-Cycle-Audit-Fixes §4: queries_tried
            # is overwritten below with the attempted-only list once
            # _multi_query_lookup returns. Pre-populating with planned_queries
            # here keeps the schema field non-empty for any code path that
            # snapshots the resolution before the loop completes.
            queries_tried=list(planned_queries),
            planned_queries=list(planned_queries),
        )
        if self.search_location:
            expected_location = canonicalize_location_label(lead.location)
            if expected_location and expected_location != self.search_location:
                resolution.notes.append(
                    f"Lead location '{expected_location}' does not match fixed search location '{self.search_location}'"
                )

        scored_candidates, query_by_url, attempted_queries, stop_reason = await self._multi_query_lookup(
            primary_queries=primary_queries,
            fallback_queries=fallback_queries,
            hints=hints,
            expected_name=lead.candidate_name,
        )
        # Replace the placeholder with the honest attempted log so artifacts
        # reflect what was actually issued, not the full planned plan.
        resolution.queries_tried = list(attempted_queries)
        resolution.stop_reason = stop_reason
        self._apply_direct_hint_anchor(scored_candidates, hints)
        candidates = [candidate for candidate, _match in scored_candidates]
        resolution.top_candidates = candidates

        if not candidates:
            resolution.identity_classification = "no_results"
            resolution.rationale = "No Recruiter result cards surfaced for the name search."
            self._apply_terminal_outcome(
                resolution,
                identity_classification="no_results",
                identity_high_confidence=False,
                identity_name_mismatch=False,
                had_plausible_cards=False,
                holistic_decision=None,
                profile_summary=None,
                already_saved_card=False,
                profile_status=None,
                novelty_pressure="",
                reachout_status="",
                extraction_failed=False,
            )
            return resolution

        ranked_matches = [match for _candidate, match in scored_candidates]
        classification, best_match = choose_best_match(ranked_matches)
        selected_candidate = next(
            (
                candidate
                for candidate, match in scored_candidates
                if match.matched_profile_url == (best_match.matched_profile_url if best_match else "")
            ),
            candidates[0],
        )
        resolution.selected_candidate_rank = selected_candidate.rank
        resolution.selected_profile_url = selected_candidate.profile_url
        resolution.already_saved = selected_candidate.already_saved
        # Record which query in the bounded plan surfaced the selected card; falls
        # back to the synthesized lookup_name when the selected card had no URL.
        resolved_query = query_by_url.get(selected_candidate.profile_url, "")
        resolution.resolved_query = resolved_query or lookup_name
        self._apply_activity_summary(
            resolution=resolution,
            activity=selected_candidate.recruiter_activity,
        )

        name_mismatch_selected = any(
            "name mismatch" in reason.lower() for reason in selected_candidate.ambiguity_reasons
        )
        plausible_full = dedupe_plausible_by_profile_url(
            [c for c in candidates if is_plausible_recruiter_candidate(c)]
        )
        cap = max(self.config.max_ambiguity_profiles, 1)
        ambiguity_slice = plausible_full[:cap]
        had_plausible_for_artifact = bool(plausible_full) or classification == "manual_review"

        identity_high = classification == "high_confidence_match" and not name_mismatch_selected

        resolution.rationale = self._build_rationale(
            selected_candidate,
            prefix=(
                "Top Recruiter card looks like the same person as the GitHub lead."
                if identity_high
                else "Recruiter surfaced candidates; identity requires review."
            ),
        )

        # In identity_collect mode the LinkedIn brief is not required and the
        # judger is never invoked; in fit_gated_save the brief is required for
        # holistic evaluation. The guard is mode-aware so identity_collect can
        # still open profiles for identity confirmation.
        identity_collect = self.config.workflow_mode == "identity_collect"
        can_open_profile = self.config.open_profile_on_likely_match and (
            identity_collect or self.linkedin_brief is not None
        )

        if len(plausible_full) >= 2 and can_open_profile:
            resolution.rationale = (
                "Multiple plausible Recruiter matches for this name; reviewing top candidates in rank order."
            )
            await self._resolve_ambiguity_among_plausible(
                ambiguity_slice,
                resolution,
                had_plausible_cards=had_plausible_for_artifact,
                hints=hints,
            )
            return resolution

        if (
            len(plausible_full) == 1
            and can_open_profile
            and not identity_high
            and is_single_strong_plausible_for_profile_open(plausible_full[0])
            and single_plausible_is_safely_dominant(
                lone=plausible_full[0],
                all_scored=scored_candidates,
            )
        ):
            only = plausible_full[0]
            resolution.rationale = (
                "Single plausible match with strong identity anchor (tier-1 score+structure or tier-2 "
                "exact name + two of company/title/location); no competing plausible cards and clear "
                "score gap vs next result — opening profile for holistic evaluation."
            )
            resolution.selected_candidate_rank = only.rank
            resolution.selected_profile_url = only.profile_url
            resolution.already_saved = only.already_saved
            self._apply_activity_summary(
                resolution=resolution,
                activity=only.recruiter_activity,
            )
            await self._evaluate_confirmed_identity_profile(
                selected_candidate=only,
                resolution=resolution,
                had_plausible_cards=True,
                hints=hints,
                identity_classification="single_strong_plausible_profile",
            )
            return resolution

        if (
            identity_high
            and can_open_profile
            and selected_candidate.profile_url
        ):
            await self._evaluate_confirmed_identity_profile(
                selected_candidate=selected_candidate,
                resolution=resolution,
                had_plausible_cards=had_plausible_for_artifact,
                hints=hints,
            )
            return resolution

        if (
            can_open_profile
            and self._sole_candidate_merits_identity_confirmation_open(
                expected_name=lead.candidate_name,
                candidate=selected_candidate,
                surfaced_count=len(candidates),
            )
        ):
            resolution.rationale = (
                "Only one Recruiter result surfaced, and the card name looks like an initialized "
                "surname variant of the GitHub lead (for example, 'Barrett' -> 'B'). Opening the "
                "profile for identity confirmation before rejecting."
            )
            await self._evaluate_confirmed_identity_profile(
                selected_candidate=selected_candidate,
                resolution=resolution,
                had_plausible_cards=True,
                hints=hints,
                identity_classification="single_surface_name_variant_profile",
            )
            return resolution

        if identity_high and not self.config.open_profile_on_likely_match:
            resolution.identity_classification = "high_confidence_match"
            resolution.notes.append("Holistic fit skipped because profile open is disabled for this run.")
            self._apply_terminal_outcome(
                resolution,
                identity_classification="high_confidence_match",
                identity_high_confidence=True,
                identity_name_mismatch=False,
                had_plausible_cards=had_plausible_for_artifact,
                holistic_decision=None,
                profile_summary=None,
                already_saved_card=selected_candidate.already_saved,
                profile_status=None,
                novelty_pressure=resolution.novelty_pressure,
                reachout_status=resolution.reachout_status,
                extraction_failed=False,
            )
            return resolution

        if identity_high and self.linkedin_brief is None:
            resolution.identity_classification = "high_confidence_match"
            resolution.notes.append("Holistic fit skipped: no LinkedIn brief was configured.")
            self._apply_terminal_outcome(
                resolution,
                identity_classification="high_confidence_match",
                identity_high_confidence=True,
                identity_name_mismatch=False,
                had_plausible_cards=had_plausible_for_artifact,
                holistic_decision=None,
                profile_summary=None,
                already_saved_card=selected_candidate.already_saved,
                profile_status=None,
                novelty_pressure=resolution.novelty_pressure,
                reachout_status=resolution.reachout_status,
                extraction_failed=False,
            )
            return resolution

        if classification == "manual_review" or plausible_full:
            resolution.identity_classification = classification
            self._apply_terminal_outcome(
                resolution,
                identity_classification=classification,
                identity_high_confidence=False,
                identity_name_mismatch=name_mismatch_selected,
                had_plausible_cards=had_plausible_for_artifact,
                holistic_decision=None,
                profile_summary=None,
                already_saved_card=False,
                profile_status=None,
                novelty_pressure="",
                reachout_status="",
                extraction_failed=False,
            )
            return resolution

        resolution.identity_classification = "no_confident_match"
        resolution.rationale = "Recruiter results did not surface a plausible same-person candidate."
        self._apply_terminal_outcome(
            resolution,
            identity_classification="no_confident_match",
            identity_high_confidence=False,
            identity_name_mismatch=name_mismatch_selected,
            had_plausible_cards=False,
            holistic_decision=None,
            profile_summary=None,
            already_saved_card=False,
            profile_status=None,
            novelty_pressure="",
            reachout_status="",
            extraction_failed=False,
        )
        return resolution

    async def _read_top_candidates(
        self,
        hints: LinkedInIdentityHints,
    ) -> list[tuple[RecruiterIdentityCandidate, LinkedInMatchResult]]:
        slot_count = await self.browser.get_card_slot_count()
        if slot_count == 0:
            slot_count = await self.browser.get_card_count()
        candidate_count = min(slot_count, max(self.config.max_cards, 1))
        candidates: list[tuple[RecruiterIdentityCandidate, LinkedInMatchResult]] = []
        for card_index in range(candidate_count):
            await self.browser.focus_card_for_review(card_index)
            await asyncio.sleep(
                human_delay_correlated(
                    random.uniform(0.35, 0.9),
                    channel="recruiter_identity_card_glance",
                )
            )
            snapshot = await self.browser.get_card_snapshot(card_index)
            parsed = _extract_card_identity(snapshot)
            activity = RecruiterActivitySnapshot.from_dict(snapshot.get("recruiter_activity"))
            match = score_linkedin_identity_match(
                hints,
                matched_name=parsed["name"],
                matched_company=parsed["current_company"],
                matched_title=parsed["current_title"] or parsed["headline"],
                matched_location=parsed["location"],
                matched_profile_url=parsed["profile_url"],
                recruiter_activity=activity,
                match_method="recruiter_name_search",
            )
            candidate = RecruiterIdentityCandidate(
                rank=card_index + 1,
                profile_url=parsed["profile_url"],
                name=parsed["name"],
                headline=parsed["headline"],
                current_title=parsed["current_title"],
                current_company=parsed["current_company"],
                location=parsed["location"],
                already_saved=parsed["already_saved"],
                match_confidence=match.match_confidence,
                evidence=list(match.evidence),
                ambiguity_reasons=list(match.ambiguity_reasons),
                recruiter_activity=activity,
                raw_card_text=parsed["raw_card_text"],
            )
            candidates.append((candidate, match))
        return candidates

    def _build_query_plan(
        self,
        hints: LinkedInIdentityHints,
        lookup_name: str,
    ) -> tuple[list[str], list[str]]:
        """Return ``(primary_queries, fallback_queries)`` for ``_multi_query_lookup``.

        Recruiter-Identity-Collection-Cycle-Audit-Fixes §1+§2: identity-collection
        runs in a pre-filtered Recruiter search must NOT keep mutating
        operator-applied filters with enriched company/location/title keyword
        queries. The bare quoted lookup name is always the first primary query;
        enriched variants run only as a fallback under ``"enriched"`` policy
        when the primary plan returned zero surfaced cards.

        Behavior by effective policy
          - ``"name_first"`` → primary = ``[quoted lookup_name]``, fallback = ``[]``.
          - ``"enriched"``   → primary = full bounded plan from
            :func:`build_candidate_lookup_queries` with the bare quoted
            lookup name pulled to index 0, plus the synthesized
            ``lookup_name`` fallback. fallback list is empty (the enriched
            primary already exhausts the bounded plan).

        The list length is capped at 6 to keep the bounded character of the
        legacy plan.
        """
        primary_quoted = f'"{lookup_name}"' if lookup_name else ""

        # Always try the bare quoted lookup name first. This is the safest
        # query in a pre-filtered Recruiter search and protects single-token-
        # from-username surname rescues (e.g. "Michael" + username "mldangelo"
        # → "Michael Mldangelo") from being shadowed by enriched fallbacks.
        primary: list[str] = []
        if primary_quoted:
            primary.append(primary_quoted)

        policy = self._effective_query_expansion_policy()
        if policy == "name_first":
            return primary, []

        # Enriched policy: assemble the legacy bounded plan but with the bare
        # quoted name forced to index 0, and the synthesized lookup_name
        # appended as a fallback when it differs from hints.candidate_name.
        enriched_block = list(build_candidate_lookup_queries(hints))
        if lookup_name and lookup_name != hints.candidate_name.strip():
            quoted = f'"{lookup_name}"'
            if quoted not in enriched_block:
                enriched_block.append(quoted)

        seen: set[str] = set()
        if primary_quoted:
            seen.add(primary_quoted)
        ordered: list[str] = list(primary)
        for query in enriched_block:
            cleaned = " ".join(str(query or "").split())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                ordered.append(cleaned)
        if not ordered and lookup_name:
            ordered.append(lookup_name)
        ordered = ordered[:6]
        # In enriched mode the entire bounded plan is the primary list; there
        # is no separate "after no results" fallback beyond what the plan
        # already carries.
        return ordered, []

    async def _scored_candidates_from_query(
        self,
        query: str,
        hints: LinkedInIdentityHints,
    ) -> list[tuple[RecruiterIdentityCandidate, LinkedInMatchResult]]:
        await self.browser.enter_search_string(query)
        self._current_browser_query = query
        return await self._read_top_candidates(hints)

    async def _multi_query_lookup(
        self,
        *,
        primary_queries: list[str],
        fallback_queries: list[str],
        hints: LinkedInIdentityHints,
        expected_name: str = "",
    ) -> tuple[
        list[tuple[RecruiterIdentityCandidate, LinkedInMatchResult]],
        dict[str, str],
        list[str],
        str,
    ]:
        """Run the two-phase bounded query plan; dedupe by profile URL.

        Phase 1 issues every query in ``primary_queries`` (subject to the
        early-exit rules below). Phase 2 issues ``fallback_queries`` ONLY when
        Phase 1 surfaced zero cards (no URL'd matches AND no URL-less results).
        For ``"name_first"`` policy, ``fallback_queries`` is empty so Phase 2
        never runs and operator-applied filters stay untouched.

        Per-query: scores each card against the GitHub identity hints and keeps
        the highest-confidence scoring per profile URL across queries.
        URL-less cards are kept positionally in surfacing order.

        Returns ``(scored_candidates, query_by_url, attempted, stop_reason)``:
          - ``attempted`` records every query string actually issued to the
            browser, in order. Recruiter-Identity-Collection-Cycle-Audit-Fixes
            §4 uses this to make ``RecruiterIdentityResolution.queries_tried``
            an honest log of attempted (not planned) queries.
          - ``stop_reason`` records why the loop stopped (one of
            ``"score_above_threshold"``, ``"high_confidence_match"``,
            ``"no_new_urls"``, ``"single_surface_name_variant_stop"``,
            ``"plan_exhausted"``).

        Early-exit conditions (any one stops further queries):
          - A surfaced card scores >= 0.9 (already-strong identity match).
          - The kept set already classifies as ``high_confidence_match`` via
            :func:`choose_best_match` (no additional query can improve identity).
          - The current query produced zero NEW profile URLs (further queries
            on the same surfaced set are unlikely to add new identities).
          - Mid-loop sole-initialized-surname stop (Recruiter-Identity-
            Collection-Cycle-Audit-Fixes §3): the most recent query surfaced
            exactly one URL'd card AND
            :meth:`_sole_candidate_merits_identity_confirmation_open` returns
            True for that card. Stops BEFORE further keyword mutation so the
            existing post-loop sole-init open branch can confirm identity
            without later queries shadowing the surfaced card.
        """
        best_by_url: dict[str, tuple[RecruiterIdentityCandidate, LinkedInMatchResult]] = {}
        query_by_url: dict[str, str] = {}
        results_without_url: list[tuple[RecruiterIdentityCandidate, LinkedInMatchResult]] = []
        attempted: list[str] = []
        stop_reason = ""

        # Two-phase flow: run Phase 1 (primary). If a stop reason fires inside
        # Phase 1, Phase 2 is skipped. Otherwise Phase 2 (fallback) runs ONLY
        # when Phase 1 surfaced zero cards (no URL'd matches AND no URL-less
        # results) — this preserves the spec's "after no-results" gating so
        # operator-applied filters are not perturbed when bare-name worked.
        for phase_index, phase_queries in enumerate((primary_queries, fallback_queries)):
            if stop_reason:
                break
            if not phase_queries:
                continue
            if phase_index == 1 and (best_by_url or results_without_url):
                # Phase 1 surfaced something; skip the enriched fallback.
                break
            for query in phase_queries:
                urls_before = set(best_by_url.keys())
                attempted.append(query)
                try:
                    scored = await self._scored_candidates_from_query(query, hints)
                except StopAsyncIteration:
                    # Mock-driven tests exhaust their snapshot side_effect lists;
                    # treat an exhausted card stream the same as an empty result
                    # set so we stop probing rather than hard-fail on a benign
                    # exhaustion.
                    stop_reason = stop_reason or "plan_exhausted"
                    break
                surfaced_with_url: list[
                    tuple[RecruiterIdentityCandidate, LinkedInMatchResult]
                ] = []
                for candidate, match in scored:
                    profile_url = (candidate.profile_url or "").strip()
                    if not profile_url:
                        # Stamp the surfacing query even on URL-less surfaces so
                        # operator-facing artifacts can show provenance.
                        # _replay_query_for_candidate skips replay for them.
                        candidate.surfaced_query = query
                        results_without_url.append((candidate, match))
                        continue
                    surfaced_with_url.append((candidate, match))
                    prior = best_by_url.get(profile_url)
                    if prior is None or prior[1].match_confidence < match.match_confidence:
                        candidate.surfaced_query = query
                        best_by_url[profile_url] = (candidate, match)
                        query_by_url[profile_url] = query

                # Mid-loop sole-init stop: when this query produced exactly
                # one URL'd card and that card looks like an initialized-
                # surname variant of the lead, stop now so subsequent queries
                # don't mutate the operator-prepared search away from this
                # surfaced result.
                if (
                    expected_name
                    and len(surfaced_with_url) == 1
                    and self._sole_candidate_merits_identity_confirmation_open(
                        expected_name=expected_name,
                        candidate=surfaced_with_url[0][0],
                        surfaced_count=1,
                    )
                ):
                    stop_reason = "single_surface_name_variant_stop"
                    break

                if any(item[1].match_confidence >= 0.9 for item in best_by_url.values()):
                    stop_reason = "score_above_threshold"
                    break
                kept_matches = [match for _candidate, match in best_by_url.values()]
                classification, _best_match = choose_best_match(kept_matches)
                if classification == "high_confidence_match":
                    stop_reason = "high_confidence_match"
                    break
                urls_after = set(best_by_url.keys())
                if urls_before and urls_after == urls_before:
                    stop_reason = "no_new_urls"
                    break

            # End of one phase. If we exited normally and surfaced nothing,
            # the next phase (if any) is allowed to run; otherwise the outer
            # loop also breaks because stop_reason is set.

        if not stop_reason:
            stop_reason = "plan_exhausted"

        # Re-rank kept cards so the strongest scorer is first in ranking output.
        ordered = sorted(
            best_by_url.values(),
            key=lambda item: item[1].match_confidence,
            reverse=True,
        )
        # Reassign sequential ranks so downstream rank-based logic stays stable.
        renumbered: list[tuple[RecruiterIdentityCandidate, LinkedInMatchResult]] = []
        for new_rank, (candidate, match) in enumerate(ordered, start=1):
            candidate.rank = new_rank
            renumbered.append((candidate, match))
        # Append URL-less surfaced candidates at the tail (rank-extension).
        for offset, (candidate, match) in enumerate(results_without_url, start=len(renumbered) + 1):
            candidate.rank = offset
            renumbered.append((candidate, match))
        return renumbered, query_by_url, attempted, stop_reason

    @staticmethod
    def _apply_direct_hint_anchor(
        scored_candidates: list[tuple[RecruiterIdentityCandidate, LinkedInMatchResult]],
        hints: LinkedInIdentityHints,
    ) -> None:
        """Surface-time direct-hint anchor (Recruiter ``/talent/profile/`` only).

        Only fires when ``hints.linkedin_url_hint`` is itself a Recruiter URL,
        because Recruiter card surfaces never carry the public ``/in/<slug>``
        URL — comparing a public hint to a Recruiter card URL would be
        guaranteed to miss. Public ``/in/<slug>`` hints are handled
        post-open by :meth:`_apply_post_open_public_hint_anchor` instead
        (Recruiter-Identity-Collection-Followups §5).
        """
        hint = (hints.linkedin_url_hint or "").strip()
        if not hint or not _is_recruiter_talent_url(hint):
            return
        for candidate, match in scored_candidates:
            url = (candidate.profile_url or "").strip()
            if not url or url != hint:
                continue
            if DIRECT_HINT_EVIDENCE not in candidate.evidence:
                candidate.evidence.append(DIRECT_HINT_EVIDENCE)
            if DIRECT_HINT_EVIDENCE not in match.evidence:
                match.evidence.append(DIRECT_HINT_EVIDENCE)

    @staticmethod
    def _apply_post_open_public_hint_anchor(
        candidate: RecruiterIdentityCandidate,
        hints: LinkedInIdentityHints,
        profile_text: str,
    ) -> bool:
        """Public LinkedIn hint anchor (currently disabled).

        DISABLED per Recruiter-Identity-Collection-Review-Fixes §3.

        Previously this helper scanned the opened-profile innertext for ANY
        ``/in/<slug>`` substring matching ``hints.linkedin_url_hint`` and used
        a hit to anchor identity confirmation. That heuristic was unsafe:
        :meth:`LinkedInBrowser.get_profile_innertext` returns the broad profile
        body, so an incidental reference (e.g. the subject mentioning a
        co-author) could falsely confirm identity.

        The intended replacement is to compare the hint against a
        SUBJECT-OWNED public profile identifier (an explicit href tied to the
        opened card / a dedicated extractor field). No such source exists in
        the current Recruiter extraction surface, so public ``/in/<slug>`` hint
        anchoring is disabled here. Surface-time anchoring still fires for
        true Recruiter ``/talent/profile/...`` hints in
        :meth:`_apply_direct_hint_anchor`.

        TODO: re-enable after a first-class subject public profile identifier
        is wired into :func:`extract_profile_from_innertext` (or an equivalent
        extractor). At that point, accept ``hints.linkedin_url_hint`` only
        when ``normalize_public_linkedin_handle(...)`` of the SUBJECT-OWNED
        URL matches.
        """
        # Public-hint anchoring is intentionally a no-op until subject-owned
        # public profile URL extraction lands. See docstring above.
        del candidate, hints, profile_text  # explicitly unused
        return False

    @staticmethod
    def _build_rationale(
        candidate: RecruiterIdentityCandidate,
        *,
        prefix: str,
    ) -> str:
        details: list[str] = []
        if candidate.evidence:
            details.append("; ".join(candidate.evidence[:3]))
        if candidate.already_saved:
            details.append("profile is already saved in Recruiter")
        if candidate.ambiguity_reasons:
            details.append("; ".join(candidate.ambiguity_reasons[:2]))
        if not details:
            return prefix
        return f"{prefix} Evidence: {'; '.join(details)}."

    @staticmethod
    def _apply_activity_summary(
        *,
        resolution: RecruiterIdentityResolution,
        activity: RecruiterActivitySnapshot | None,
    ) -> None:
        resolution.novelty_pressure = classify_recruiter_activity_pressure(activity)
        resolution.reachout_status = infer_reachout_status(activity)
