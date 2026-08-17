"""Retrieval-only experiment harness for GitHub→LinkedIn identity resolution."""

from __future__ import annotations

import asyncio
import random
import statistics
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import quote_plus, urljoin

from github.reconciliation_input import build_identity_resolution_experiment_cohort
from linkedin.identity_experiment_browser import IdentityExperimentBrowser
from shared.identity_experiment_schemas import (
    IdentityResolutionExperimentLead,
    IdentityResolutionGoldLabel,
    StrategyExecutionRecord,
    SurfacedProfileCandidate,
)
from shared.identity_resolution import (
    canonicalize_location_label,
    normalize_company_name,
    normalize_person_name,
    normalize_public_linkedin_url,
)

BLOCKER_CAPTCHA = "captcha"
BLOCKER_LOGIN_WALL = "login_wall"
BLOCKER_EMPTY_RESULTS = "empty_results"
BLOCKER_UI_FAILURE = "ui_failure"
BLOCKER_ABORTED = "aborted"
FATAL_ABORT_BLOCKERS = {BLOCKER_CAPTCHA, BLOCKER_LOGIN_WALL}
DEFAULT_STRATEGY_ORDER = (
    "web_exact_city",
    "web_loose_city",
    "linkedin_people_city",
    "recruiter_name_city",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _looks_like_location(line: str) -> bool:
    lowered = str(line or "").lower()
    return any(
        token in lowered
        for token in (
            " area",
            " united states",
            " canada",
            " india",
            " uk",
            " england",
            " new york",
            " san francisco",
            " seattle",
            " boston",
            " remote",
        )
    )


def _safe_text(value: str) -> str:
    return str(value or "").strip()


def _effective_lookup_name(lead: IdentityResolutionExperimentLead) -> str:
    return _safe_text(lead.lookup_name) or _safe_text(lead.candidate_name)


def _token_overlap(left: str, right: str) -> int:
    left_tokens = set(normalize_person_name(left).split())
    right_tokens = set(normalize_person_name(right).split())
    return len(left_tokens & right_tokens)


def _company_overlap(left: str, right: str) -> bool:
    left_tokens = set(normalize_company_name(left).split())
    right_tokens = set(normalize_company_name(right).split())
    return bool(left_tokens and right_tokens and left_tokens & right_tokens)


def _location_overlap(left: str, right: str) -> bool:
    left_normalized = normalize_person_name(canonicalize_location_label(left))
    right_normalized = normalize_person_name(canonicalize_location_label(right))
    if not left_normalized or not right_normalized:
        return False
    return bool(set(left_normalized.split()) & set(right_normalized.split()))


def _parse_bing_query(lead: IdentityResolutionExperimentLead, *, exact_city: bool) -> tuple[str, str]:
    city = canonicalize_location_label(lead.location)
    lookup_name = _effective_lookup_name(lead)
    if exact_city:
        query = f'site:linkedin.com/in "{lookup_name}" "{city}"'.strip()
    else:
        query = f"site:linkedin.com/in {lookup_name} {city}".strip()
    return query, city


def _normalize_surface_url(url: str, *, base_url: str = "") -> str:
    if not url:
        return ""
    absolute = urljoin(base_url, url)
    return absolute.split("#", 1)[0]


def _parse_public_candidate(
    *,
    url: str,
    display_name: str,
    raw_text: str,
    rank: int,
    surface: str,
) -> SurfacedProfileCandidate:
    lines = _clean_lines(raw_text)
    headline = lines[1] if len(lines) > 1 else ""
    company = ""
    location = ""
    for line in lines[2:6]:
        if _looks_like_location(line):
            location = line
            break
    if len(lines) > 2:
        company = lines[2] if lines[2] != location else ""
    return SurfacedProfileCandidate(
        profile_url=url,
        public_profile_url=url if "linkedin.com/in/" in url else "",
        display_name=display_name or (lines[0] if lines else ""),
        headline=headline,
        company=company,
        location=location,
        rank=rank,
        source_surface=surface,
        raw_evidence=raw_text,
    )


def _parse_recruiter_candidate(snapshot: dict, *, rank: int) -> SurfacedProfileCandidate:
    raw_text = str(snapshot.get("innertext", "") or "").strip()
    lines = _clean_lines(raw_text)
    headline = lines[1] if len(lines) > 1 else ""
    location = next((line for line in lines[2:7] if _looks_like_location(line)), "")
    company = ""
    if headline:
        for separator in (" at ", " @ ", " | "):
            if separator in headline.lower():
                parts = headline.split(separator, 1)
                company = parts[1].strip()
                break
    return SurfacedProfileCandidate(
        profile_url=str(snapshot.get("url", "") or "").strip(),
        public_profile_url="",
        display_name=str(snapshot.get("name", "") or "").strip() or (lines[0] if lines else ""),
        headline=headline,
        company=company,
        location=location,
        rank=rank,
        source_surface="recruiter",
        raw_evidence=raw_text,
    )


def _lead_matches_gold_exact(candidate: SurfacedProfileCandidate, gold: IdentityResolutionGoldLabel) -> bool:
    gold_url = normalize_public_linkedin_url(gold.gold_public_linkedin_url)
    candidate_url = normalize_public_linkedin_url(candidate.public_profile_url or candidate.profile_url)
    if gold_url and candidate_url and gold_url == candidate_url:
        return True

    gold_name = gold.gold_display_name or gold.candidate_name
    if normalize_person_name(candidate.display_name) != normalize_person_name(gold_name):
        return False

    company_match = _company_overlap(candidate.company, gold.gold_company)
    location_match = _location_overlap(candidate.location, gold.gold_location)
    title_overlap = _token_overlap(candidate.headline, gold.gold_title)
    return company_match or (location_match and title_overlap > 0)


def _lead_is_obviously_wrong(candidate: SurfacedProfileCandidate, gold: IdentityResolutionGoldLabel) -> bool:
    gold_name = gold.gold_display_name or gold.candidate_name
    if not candidate.display_name:
        return False
    if normalize_person_name(candidate.display_name) != normalize_person_name(gold_name):
        return True
    company_present = bool(candidate.company and gold.gold_company)
    location_present = bool(candidate.location and gold.gold_location)
    if company_present and not _company_overlap(candidate.company, gold.gold_company):
        if location_present and not _location_overlap(candidate.location, gold.gold_location):
            return True
    return False


def score_execution_record(
    record: StrategyExecutionRecord,
    gold: IdentityResolutionGoldLabel,
) -> StrategyExecutionRecord:
    if record.aborted:
        return record
    candidates = sorted(record.surfaced_candidates, key=lambda item: item.rank)
    top1 = candidates[0] if candidates else None
    top3 = candidates[:3]
    top1_profile_url = top1.public_profile_url or top1.profile_url if top1 else ""
    top3_profile_urls = [candidate.public_profile_url or candidate.profile_url for candidate in top3]

    top1_correct = False
    top3_contains_correct = False
    wrong_person_top1 = False
    ambiguous_only = False
    manual_review_required = False

    if gold.gold_outcome == "exact_match":
        top1_correct = bool(top1 and _lead_matches_gold_exact(top1, gold))
        top3_contains_correct = any(_lead_matches_gold_exact(candidate, gold) for candidate in top3)
        wrong_person_top1 = bool(top1 and not top1_correct and _lead_is_obviously_wrong(top1, gold))
        if candidates and not top3_contains_correct and not wrong_person_top1:
            ambiguous_only = True
            manual_review_required = True
    elif gold.gold_outcome == "manual_review_expected":
        ambiguous_only = bool(candidates)
        manual_review_required = True
    elif gold.gold_outcome == "no_public_match":
        manual_review_required = False

    if record.blocker_state == BLOCKER_EMPTY_RESULTS and not candidates:
        record.no_candidate = True
    elif not candidates:
        record.no_candidate = True

    record.top1_profile_url = top1_profile_url
    record.top3_profile_urls = [url for url in top3_profile_urls if url]
    record.top1_correct = top1_correct
    record.top3_contains_correct = top3_contains_correct
    record.wrong_person_top1 = wrong_person_top1
    record.ambiguous_only = ambiguous_only
    record.manual_review_required = manual_review_required
    return record


def finalize_unscored_record(record: StrategyExecutionRecord) -> StrategyExecutionRecord:
    """Populate non-scoring fields for preview-only experiment runs."""
    if record.aborted:
        return record
    candidates = sorted(record.surfaced_candidates, key=lambda item: item.rank)
    top1 = candidates[0] if candidates else None
    record.top1_profile_url = (top1.public_profile_url or top1.profile_url) if top1 else ""
    record.top3_profile_urls = [
        candidate.public_profile_url or candidate.profile_url
        for candidate in candidates[:3]
        if candidate.public_profile_url or candidate.profile_url
    ]
    record.no_candidate = not bool(candidates)
    return record


def _new_record(
    *,
    strategy_name: str,
    lead: IdentityResolutionExperimentLead,
    query: str,
    location_filter: str = "",
) -> StrategyExecutionRecord:
    record = StrategyExecutionRecord(
        strategy_name=strategy_name,
        github_username=lead.github_username,
        candidate_name=lead.candidate_name,
        cohort_kind=lead.cohort_kind,
        cohort_bucket=lead.cohort_bucket,
        query=query,
        location_filter=location_filter,
        started_at=_timestamp(),
    )
    if _effective_lookup_name(lead) != _safe_text(lead.candidate_name):
        record.notes.append(f"Using cleaned lookup name: {_effective_lookup_name(lead)}")
    return record


async def _capture_page_state(page, record: StrategyExecutionRecord) -> None:
    try:
        record.final_url = _safe_text(page.url)
    except Exception:
        record.final_url = ""
    try:
        record.page_title = _safe_text(await page.title())
    except Exception:
        record.page_title = ""
    try:
        body_text = _safe_text(await page.locator("body").inner_text(timeout=1500))
        record.body_excerpt = body_text[:500]
    except Exception:
        record.body_excerpt = ""


async def _wait_for_any_selector(page, selectors: tuple[str, ...], *, timeout: int) -> str:
    deadline = asyncio.get_running_loop().time() + (timeout / 1000.0)
    while asyncio.get_running_loop().time() < deadline:
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() and await locator.is_visible(timeout=750):
                    return selector
            except Exception:
                continue
        await page.wait_for_timeout(250)
    return ""


async def _wait_for_bing_surface_ready(page) -> str:
    return await _wait_for_any_selector(
        page,
        (
            "li.b_algo",
            ".b_no",
            'text="There are no results"',
        ),
        timeout=8000,
    )


async def _wait_for_public_linkedin_surface_ready(page) -> str:
    return await _wait_for_any_selector(
        page,
        (
            'main a[href*="/in/"]',
            'main :text("Search results for")',
            'text="No results found"',
            'text="Try adjusting your search"',
        ),
        timeout=8000,
    )


async def _wait_for_recruiter_surface_ready(page) -> str:
    return await _wait_for_any_selector(
        page,
        (
            "ol.profile-list",
            "ol.profile-list article.profile-list-item",
            'text="No results"',
        ),
        timeout=8000,
    )


@dataclass
class StrategyHealthTracker:
    processed: int = 0
    blocker_counts: Counter = field(default_factory=Counter)
    consecutive_hard_blocker: str = ""
    consecutive_hard_count: int = 0
    aborted_reason: str = ""

    def record(self, blocker_state: str) -> None:
        self.processed += 1
        if blocker_state:
            self.blocker_counts[blocker_state] += 1

        if blocker_state in FATAL_ABORT_BLOCKERS:
            if blocker_state == self.consecutive_hard_blocker:
                self.consecutive_hard_count += 1
            else:
                self.consecutive_hard_blocker = blocker_state
                self.consecutive_hard_count = 1
        else:
            self.consecutive_hard_blocker = ""
            self.consecutive_hard_count = 0

    def should_abort(self) -> bool:
        if self.aborted_reason:
            return True
        if self.consecutive_hard_count >= 2 and self.consecutive_hard_blocker:
            self.aborted_reason = f"two consecutive {self.consecutive_hard_blocker} blockers"
            return True
        fatal_blockers = sum(
            count
            for blocker, count in self.blocker_counts.items()
            if blocker in FATAL_ABORT_BLOCKERS
        )
        if self.processed >= 5 and self.processed > 0 and (fatal_blockers / self.processed) > 0.10:
            self.aborted_reason = "fatal blocker rate exceeded 10%"
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "processed": self.processed,
            "blocker_counts": dict(self.blocker_counts),
            "aborted_reason": self.aborted_reason,
        }


def _aborted_record(
    *,
    strategy_name: str,
    lead: IdentityResolutionExperimentLead,
    tracker: StrategyHealthTracker,
) -> StrategyExecutionRecord:
    query, location_filter = _query_and_location_for_strategy(strategy_name, lead)
    record = _new_record(
        strategy_name=strategy_name,
        lead=lead,
        query=query,
        location_filter=location_filter,
    )
    record.finished_at = record.started_at
    record.aborted = True
    record.abort_reason = tracker.aborted_reason or "strategy aborted before execution"
    record.blocker_state = BLOCKER_ABORTED
    record.notes.append(record.abort_reason)
    return record


def _query_and_location_for_strategy(
    strategy_name: str,
    lead: IdentityResolutionExperimentLead,
) -> tuple[str, str]:
    if strategy_name == "web_exact_city":
        return _parse_bing_query(lead, exact_city=True)
    if strategy_name == "web_loose_city":
        return _parse_bing_query(lead, exact_city=False)
    return _effective_lookup_name(lead), canonicalize_location_label(lead.location)


class IdentityResolutionStrategy(ABC):
    name: str
    surface: str
    max_results: int

    def __init__(self, *, max_results: int = 8) -> None:
        self.max_results = max_results

    @abstractmethod
    async def run(
        self,
        lead: IdentityResolutionExperimentLead,
        browser: IdentityExperimentBrowser,
        *,
        recruiter_search_url: str = "",
    ) -> StrategyExecutionRecord:
        raise NotImplementedError


class WebExactCityStrategy(IdentityResolutionStrategy):
    name = "web_exact_city"
    surface = "web"

    async def run(
        self,
        lead: IdentityResolutionExperimentLead,
        browser: IdentityExperimentBrowser,
        *,
        recruiter_search_url: str = "",
    ) -> StrategyExecutionRecord:
        query, city = _parse_bing_query(lead, exact_city=True)
        start = _timestamp()
        started = asyncio.get_running_loop().time()
        record = _new_record(
            strategy_name=self.name,
            lead=lead,
            query=query,
            location_filter=city,
        )
        record.started_at = start
        page = None
        try:
            async with browser.strategy_page(self.surface) as page:
                await page.goto(f"https://www.bing.com/search?q={quote_plus(query)}", wait_until="domcontentloaded", timeout=30000)
                record.interaction_count += 1
                await _capture_page_state(page, record)
                blocker = await browser.detect_blocker_state(page, surface=self.surface)
                if blocker:
                    record.blocker_state = blocker
                else:
                    ready_selector = await _wait_for_bing_surface_ready(page)
                    if ready_selector:
                        record.notes.append(f"Bing ready selector: {ready_selector}")
                        record.surfaced_candidates = await _extract_bing_candidates(
                            page,
                            max_results=min(self.max_results, 5),
                        )
                        if not record.surfaced_candidates:
                            record.blocker_state = BLOCKER_EMPTY_RESULTS
                    else:
                        record.blocker_state = BLOCKER_UI_FAILURE
                        record.notes.append("Bing results surface never became ready")
        except Exception as exc:
            record.blocker_state = BLOCKER_UI_FAILURE
            record.notes.append(str(exc))
        finally:
            try:
                if page is not None:
                    await _capture_page_state(page, record)
            except Exception:
                pass
        record.duration_seconds = round(asyncio.get_running_loop().time() - started, 3)
        record.finished_at = _timestamp()
        return record


class WebLooseCityStrategy(IdentityResolutionStrategy):
    name = "web_loose_city"
    surface = "web"

    async def run(
        self,
        lead: IdentityResolutionExperimentLead,
        browser: IdentityExperimentBrowser,
        *,
        recruiter_search_url: str = "",
    ) -> StrategyExecutionRecord:
        query, city = _parse_bing_query(lead, exact_city=False)
        started = asyncio.get_running_loop().time()
        record = _new_record(
            strategy_name=self.name,
            lead=lead,
            query=query,
            location_filter=city,
        )
        page = None
        try:
            async with browser.strategy_page(self.surface) as page:
                await page.goto(f"https://www.bing.com/search?q={quote_plus(query)}", wait_until="domcontentloaded", timeout=30000)
                record.interaction_count += 1
                await _capture_page_state(page, record)
                blocker = await browser.detect_blocker_state(page, surface=self.surface)
                if blocker:
                    record.blocker_state = blocker
                else:
                    ready_selector = await _wait_for_bing_surface_ready(page)
                    if ready_selector:
                        record.notes.append(f"Bing ready selector: {ready_selector}")
                        record.surfaced_candidates = await _extract_bing_candidates(
                            page,
                            max_results=self.max_results,
                        )
                        if not record.surfaced_candidates:
                            record.blocker_state = BLOCKER_EMPTY_RESULTS
                    else:
                        record.blocker_state = BLOCKER_UI_FAILURE
                        record.notes.append("Bing results surface never became ready")
        except Exception as exc:
            record.blocker_state = BLOCKER_UI_FAILURE
            record.notes.append(str(exc))
        finally:
            try:
                if page is not None:
                    await _capture_page_state(page, record)
            except Exception:
                pass
        record.duration_seconds = round(asyncio.get_running_loop().time() - started, 3)
        record.finished_at = _timestamp()
        return record


class LinkedInPeopleCityStrategy(IdentityResolutionStrategy):
    name = "linkedin_people_city"
    surface = "linkedin"

    async def run(
        self,
        lead: IdentityResolutionExperimentLead,
        browser: IdentityExperimentBrowser,
        *,
        recruiter_search_url: str = "",
    ) -> StrategyExecutionRecord:
        location_filter = canonicalize_location_label(lead.location)
        lookup_name = _effective_lookup_name(lead)
        started = asyncio.get_running_loop().time()
        record = _new_record(
            strategy_name=self.name,
            lead=lead,
            query=lookup_name,
            location_filter=location_filter,
        )
        page = None
        try:
            async with browser.strategy_page(self.surface) as page:
                await page.goto(
                    f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(lookup_name)}",
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
                record.interaction_count += 1
                await _capture_page_state(page, record)
                blocker = await browser.detect_blocker_state(page, surface=self.surface)
                if blocker:
                    record.blocker_state = blocker
                else:
                    interactions, location_note = await _apply_public_linkedin_location_filter(page, location_filter)
                    record.interaction_count += interactions
                    if location_note:
                        record.notes.append(location_note)
                    ready_selector = await _wait_for_public_linkedin_surface_ready(page)
                    if ready_selector:
                        record.notes.append(f"LinkedIn ready selector: {ready_selector}")
                        record.surfaced_candidates = await _extract_public_linkedin_candidates(
                            page,
                            max_results=self.max_results,
                        )
                        if not record.surfaced_candidates:
                            record.blocker_state = BLOCKER_EMPTY_RESULTS
                    else:
                        record.blocker_state = BLOCKER_UI_FAILURE
                        record.notes.append("LinkedIn people results surface never became ready")
        except Exception as exc:
            record.blocker_state = BLOCKER_UI_FAILURE
            record.notes.append(str(exc))
        finally:
            try:
                if page is not None:
                    await _capture_page_state(page, record)
            except Exception:
                pass
        record.duration_seconds = round(asyncio.get_running_loop().time() - started, 3)
        record.finished_at = _timestamp()
        return record


class RecruiterNameCityStrategy(IdentityResolutionStrategy):
    name = "recruiter_name_city"
    surface = "recruiter"

    async def run(
        self,
        lead: IdentityResolutionExperimentLead,
        browser: IdentityExperimentBrowser,
        *,
        recruiter_search_url: str = "",
    ) -> StrategyExecutionRecord:
        if not recruiter_search_url:
            raise ValueError("recruiter_search_url is required for recruiter_name_city strategy")
        location_filter = canonicalize_location_label(lead.location)
        lookup_name = _effective_lookup_name(lead)
        started = asyncio.get_running_loop().time()
        record = _new_record(
            strategy_name=self.name,
            lead=lead,
            query=lookup_name,
            location_filter=location_filter,
        )
        page = None
        try:
            async with browser.strategy_page(self.surface) as page:
                await page.goto(recruiter_search_url, wait_until="domcontentloaded", timeout=30000)
                record.interaction_count += 1
                await _capture_page_state(page, record)
                blocker = await browser.detect_blocker_state(page, surface="recruiter")
                if blocker:
                    record.blocker_state = blocker
                else:
                    interactions, location_note = await _apply_recruiter_location_filter(page, location_filter)
                    record.interaction_count += interactions
                    if location_note:
                        record.notes.append(location_note)
                    record.interaction_count += await _enter_recruiter_name_query(page, lookup_name)
                    ready_selector = await _wait_for_recruiter_surface_ready(page)
                    if ready_selector:
                        record.notes.append(f"Recruiter ready selector: {ready_selector}")
                        record.surfaced_candidates = await _extract_recruiter_candidates(
                            page,
                            max_results=self.max_results,
                        )
                        if not record.surfaced_candidates:
                            record.blocker_state = BLOCKER_EMPTY_RESULTS
                    else:
                        record.blocker_state = BLOCKER_UI_FAILURE
                        record.notes.append("Recruiter results surface never became ready")
        except Exception as exc:
            record.blocker_state = BLOCKER_UI_FAILURE
            record.notes.append(str(exc))
        finally:
            try:
                if page is not None:
                    await _capture_page_state(page, record)
            except Exception:
                pass
        record.duration_seconds = round(asyncio.get_running_loop().time() - started, 3)
        record.finished_at = _timestamp()
        return record


async def _extract_bing_candidates(page, *, max_results: int) -> list[SurfacedProfileCandidate]:
    cards = page.locator("li.b_algo")
    count = await cards.count()
    candidates: list[SurfacedProfileCandidate] = []
    for index in range(count):
        if len(candidates) >= max_results:
            break
        card = cards.nth(index)
        link = card.locator("h2 a").first
        href = _normalize_surface_url((await link.get_attribute("href")) or "", base_url=page.url)
        if "linkedin.com/in/" not in href:
            continue
        title = _safe_text(await link.inner_text(timeout=2000))
        text = _safe_text(await card.inner_text(timeout=2000))
        candidates.append(
            _parse_public_candidate(
                url=href,
                display_name=title,
                raw_text=text,
                rank=len(candidates) + 1,
                surface="bing",
            )
        )
    return candidates


async def _apply_public_linkedin_location_filter(page, location_filter: str) -> tuple[int, str]:
    if not location_filter:
        return 0, ""
    interactions = 0
    try:
        button = page.locator(
            'button:not([aria-label^="Cancel"]):has-text("Locations"), '
            'button.search-reusables__filter-pill-button:has-text("Locations"), '
            'button.search-reusables__all-filters-pill-button:has-text("All filters")'
        ).first
        await button.click(timeout=4000)
        interactions += 1
        await page.wait_for_timeout(800)
        input_locator = page.locator(
            '[role="combobox"] input, '
            'input[placeholder*="Add a location"], '
            'input[aria-label*="Add a location"], '
            'input[placeholder*="Location"], '
            'input[aria-label*="Location"]'
        ).first
        await input_locator.wait_for(state="visible", timeout=4000)
        await input_locator.fill(location_filter, timeout=4000)
        interactions += 1
        await page.wait_for_timeout(1000)
        option = page.locator(f'[role="option"]:has-text("{location_filter}"), li:has-text("{location_filter}")').first
        await option.click(timeout=4000)
        interactions += 1
        await page.wait_for_timeout(800)
        apply_button = page.locator(
            'button:has-text("Show results"), button:has-text("Apply"), button[aria-label*="Apply"]'
        ).first
        await apply_button.click(timeout=4000)
        interactions += 1
        await page.wait_for_timeout(2000)
    except Exception as exc:
        return interactions, f"LinkedIn location filter failed: {exc}"
    return interactions, ""


async def _extract_public_linkedin_candidates(page, *, max_results: int) -> list[SurfacedProfileCandidate]:
    candidates: list[SurfacedProfileCandidate] = []
    links = page.locator('main a[href*="/in/"]')
    count = await links.count()
    seen_urls: set[str] = set()
    for index in range(count):
        if len(candidates) >= max_results:
            break
        link = links.nth(index)
        href = _normalize_surface_url((await link.get_attribute("href")) or "", base_url=page.url)
        if "linkedin.com/in/" not in href or href in seen_urls:
            continue
        seen_urls.add(href)
        display_name = _safe_text(await link.inner_text(timeout=2000))
        container = link.locator("xpath=ancestor::li[1]").first
        text = ""
        try:
            text = _safe_text(await container.inner_text(timeout=2000))
        except Exception:
            fallback_container = link.locator("xpath=ancestor::div[1]").first
            try:
                text = _safe_text(await fallback_container.inner_text(timeout=2000))
            except Exception:
                text = display_name
        if not display_name:
            lines = _clean_lines(text)
            if not lines:
                continue
            display_name = lines[0]
        candidates.append(
            _parse_public_candidate(
                url=href,
                display_name=display_name,
                raw_text=text,
                rank=len(candidates) + 1,
                surface="linkedin_people",
            )
        )
    return candidates


async def _apply_recruiter_location_filter(page, location_filter: str) -> tuple[int, str]:
    if not location_filter:
        return 0, ""
    try:
        loc_input = page.locator(
            'input[aria-label*="location" i], input[placeholder*="location" i]'
        ).first
        await loc_input.wait_for(state="visible", timeout=5000)
        await loc_input.click()
        await loc_input.fill(location_filter)
        await page.wait_for_timeout(1000)
        option = page.locator(
            f'[role="option"]:has-text("{location_filter}"), li:has-text("{location_filter}")'
        ).first
        await option.click(timeout=4000)
        await page.wait_for_timeout(1500)
        return 3, ""
    except Exception as exc:
        return 0, f"Recruiter location filter failed: {exc}"


async def _enter_recruiter_name_query(page, name: str) -> int:
    textarea = page.locator('textarea[id*="free-text-single-value-input"]').first
    if not await textarea.is_visible(timeout=1000):
        for selector in (
            'button[aria-label*="Profile keywords"]',
            'button[aria-label*="Edit Keywords"]',
            'button:has-text("Profile keywords")',
        ):
            try:
                button = page.locator(selector).first
                if await button.is_visible(timeout=500):
                    await button.click(timeout=2000)
                    break
            except Exception:
                continue
    await textarea.wait_for(state="visible", timeout=5000)
    await textarea.fill(name)
    await page.keyboard.press("Enter")
    await page.wait_for_timeout(2500)
    return 2


async def _extract_recruiter_candidates(page, *, max_results: int) -> list[SurfacedProfileCandidate]:
    cards = page.locator("ol.profile-list article.profile-list-item")
    count = await cards.count()
    candidates: list[SurfacedProfileCandidate] = []
    for index in range(min(count, max_results)):
        article = cards.nth(index)
        text = _safe_text(await article.inner_text(timeout=3000))
        name = ""
        url = ""
        try:
            name_el = article.locator('[class*="lockup__title"] a').first
            name = _safe_text(await name_el.inner_text(timeout=2000))
            url = _safe_text((await name_el.get_attribute("href")) or "")
        except Exception:
            pass
        if not text and not name and not url:
            continue
        candidates.append(
            _parse_recruiter_candidate(
                {
                    "innertext": text,
                    "name": name,
                    "url": url,
                },
                rank=index + 1,
            )
        )
    return candidates


def build_strategy_instances(
    strategy_names: Iterable[str],
    *,
    web_exact_max_results: int = 5,
    max_results: int = 8,
) -> list[IdentityResolutionStrategy]:
    strategies: list[IdentityResolutionStrategy] = []
    for name in strategy_names:
        if name == "web_exact_city":
            strategies.append(WebExactCityStrategy(max_results=web_exact_max_results))
        elif name == "web_loose_city":
            strategies.append(WebLooseCityStrategy(max_results=max_results))
        elif name == "linkedin_people_city":
            strategies.append(LinkedInPeopleCityStrategy(max_results=max_results))
        elif name == "recruiter_name_city":
            strategies.append(RecruiterNameCityStrategy(max_results=max_results))
        else:
            raise ValueError(f"Unknown identity experiment strategy: {name}")
    return strategies


def build_strategy_order(
    lead: IdentityResolutionExperimentLead,
    strategy_names: Iterable[str],
    *,
    seed: int,
) -> list[str]:
    ordered = list(strategy_names)
    rng = random.Random(f"{seed}:{lead.github_username}:{lead.candidate_name}")
    rng.shuffle(ordered)
    return ordered


class IdentityResolutionExperimentRunner:
    """Run multiple retrieval strategies against a frozen gold set."""

    def __init__(
        self,
        *,
        strategies: list[IdentityResolutionStrategy],
        recruiter_search_url: str = "",
        seed: int = 17,
    ) -> None:
        self.strategies = {strategy.name: strategy for strategy in strategies}
        self.recruiter_search_url = recruiter_search_url
        self.seed = seed
        self._browser = IdentityExperimentBrowser()

    async def run(
        self,
        leads: list[IdentityResolutionExperimentLead],
        *,
        gold_labels: dict[str, IdentityResolutionGoldLabel] | None = None,
    ) -> tuple[list[StrategyExecutionRecord], dict]:
        trackers = {name: StrategyHealthTracker() for name in self.strategies}
        rows: list[StrategyExecutionRecord] = []
        await self._browser.connect()
        try:
            for lead in leads:
                ordered_names = build_strategy_order(lead, self.strategies.keys(), seed=self.seed)
                for strategy_name in ordered_names:
                    tracker = trackers[strategy_name]
                    if tracker.should_abort():
                        rows.append(
                            _aborted_record(
                                strategy_name=strategy_name,
                                lead=lead,
                                tracker=tracker,
                            )
                        )
                        continue
                    strategy = self.strategies[strategy_name]
                    record = await strategy.run(
                        lead,
                        self._browser,
                        recruiter_search_url=self.recruiter_search_url,
                    )
                    if gold_labels is None:
                        finalized = finalize_unscored_record(record)
                        rows.append(finalized)
                        tracker.record(finalized.blocker_state)
                        continue
                    gold = gold_labels.get(lead.github_username)
                    if gold is None:
                        raise ValueError(f"Missing gold label for {lead.github_username}")
                    scored = score_execution_record(record, gold)
                    rows.append(scored)
                    tracker.record(scored.blocker_state)
        finally:
            await self._browser.disconnect()
        return rows, {name: tracker.to_dict() for name, tracker in trackers.items()}


def summarize_experiment_rows(
    rows: list[StrategyExecutionRecord],
    *,
    tracker_state: dict[str, dict] | None = None,
) -> dict:
    primary_rows = [row for row in rows if row.cohort_kind == "primary"]
    strategies = sorted({row.strategy_name for row in rows})
    thresholds = {
        "viable": {
            "top3_contains_correct_rate": 0.80,
            "wrong_person_top1_rate": 0.10,
            "no_candidate_rate": 0.20,
            "median_seconds_per_lead": 60.0,
        },
        "default": {
            "top1_correct_rate": 0.65,
            "top3_contains_correct_rate": 0.85,
            "wrong_person_top1_rate": 0.05,
            "manual_review_rate": 0.35,
            "median_seconds_per_lead": 45.0,
        },
        "subgroup": {
            "common_name_ambiguous": {"wrong_person_top1_rate": 0.15},
            "name_variant": {"top3_contains_correct_rate": 0.70},
            "stale_employer_changed_role": {"no_candidate_rate": 0.25},
        },
    }
    strategy_summaries: dict[str, dict] = {}

    for strategy_name in strategies:
        all_strategy_rows = [row for row in primary_rows if row.strategy_name == strategy_name]
        strategy_rows = [row for row in all_strategy_rows if not row.aborted]
        stratified = {
            bucket: _metric_bundle([row for row in strategy_rows if row.cohort_bucket == bucket])
            for bucket in sorted({row.cohort_bucket for row in strategy_rows})
        }
        aggregate = _metric_bundle(strategy_rows)
        viable = _passes_thresholds(aggregate, thresholds["viable"])
        default_eligible = viable and _passes_thresholds(aggregate, thresholds["default"]) and _passes_subgroup_guardrails(
            stratified,
            thresholds["subgroup"],
        )
        strategy_summaries[strategy_name] = {
            "aggregate_metrics": aggregate,
            "stratified_metrics": stratified,
            "blocker_counts": dict(Counter(row.blocker_state for row in strategy_rows if row.blocker_state)),
            "attempted_leads": len(strategy_rows),
            "aborted_leads": sum(1 for row in all_strategy_rows if row.aborted),
            "coverage_count": sum(1 for row in strategy_rows if row.surfaced_candidates),
            "viable": viable,
            "default_eligible": default_eligible,
            "tracker": (tracker_state or {}).get(strategy_name, {}),
        }

    decision = choose_winning_strategy(strategy_summaries)
    return {
        "total_primary_leads": len({row.github_username for row in primary_rows}),
        "strategies": strategy_summaries,
        "decision": decision,
        "thresholds": thresholds,
    }


def _metric_bundle(rows: list[StrategyExecutionRecord]) -> dict:
    if not rows:
        return {
            "top1_correct_rate": 0.0,
            "top3_contains_correct_rate": 0.0,
            "wrong_person_top1_rate": 0.0,
            "manual_review_rate": 0.0,
            "no_candidate_rate": 0.0,
            "median_seconds_per_lead": 0.0,
            "median_interactions_per_lead": 0.0,
            "has_data": False,
            "row_count": 0,
        }
    return {
        "top1_correct_rate": round(sum(1 for row in rows if row.top1_correct) / len(rows), 4),
        "top3_contains_correct_rate": round(sum(1 for row in rows if row.top3_contains_correct) / len(rows), 4),
        "wrong_person_top1_rate": round(sum(1 for row in rows if row.wrong_person_top1) / len(rows), 4),
        "manual_review_rate": round(sum(1 for row in rows if row.manual_review_required) / len(rows), 4),
        "no_candidate_rate": round(sum(1 for row in rows if row.no_candidate) / len(rows), 4),
        "median_seconds_per_lead": round(statistics.median(row.duration_seconds for row in rows), 3),
        "median_interactions_per_lead": round(statistics.median(row.interaction_count for row in rows), 3),
        "has_data": True,
        "row_count": len(rows),
    }


def _passes_thresholds(metrics: dict, thresholds: dict) -> bool:
    for key, threshold in thresholds.items():
        value = float(metrics.get(key, 0.0) or 0.0)
        if key in {"wrong_person_top1_rate", "no_candidate_rate", "manual_review_rate", "median_seconds_per_lead"}:
            if value > threshold:
                return False
        elif value < threshold:
            return False
    return True


def _passes_subgroup_guardrails(stratified_metrics: dict[str, dict], guardrails: dict[str, dict]) -> bool:
    for bucket, thresholds in guardrails.items():
        if bucket not in stratified_metrics or stratified_metrics[bucket]["row_count"] == 0:
            continue
        if not _passes_thresholds(stratified_metrics[bucket], thresholds):
            return False
    return True


def choose_winning_strategy(strategy_summaries: dict[str, dict]) -> dict:
    eligible = [
        (name, summary)
        for name, summary in strategy_summaries.items()
        if summary.get("default_eligible")
    ]
    if not eligible:
        viable = [name for name, summary in strategy_summaries.items() if summary.get("viable")]
        if viable:
            return {
                "winner": "",
                "decision": "run_phase_2_hybrid",
                "reason": "Multiple or partial viable strategies without a clear default winner",
            }
        return {
            "winner": "",
            "decision": "manual_assist_only",
            "reason": "No strategy cleared the viability thresholds",
        }

    ranked = sorted(
        eligible,
        key=lambda item: (
            item[1]["aggregate_metrics"]["top1_correct_rate"],
            -item[1]["aggregate_metrics"]["wrong_person_top1_rate"],
        ),
        reverse=True,
    )
    winner_name, winner_summary = ranked[0]
    if len(ranked) == 1:
        return {
            "winner": winner_name,
            "decision": "default_retrieval_surface",
            "reason": "Single strategy cleared all default thresholds",
        }
    next_name, next_summary = ranked[1]
    top1_gap = (
        winner_summary["aggregate_metrics"]["top1_correct_rate"]
        - next_summary["aggregate_metrics"]["top1_correct_rate"]
    )
    wrong_person_gap = (
        next_summary["aggregate_metrics"]["wrong_person_top1_rate"]
        - winner_summary["aggregate_metrics"]["wrong_person_top1_rate"]
    )
    if top1_gap >= 0.07 or wrong_person_gap >= 0.05:
        return {
            "winner": winner_name,
            "decision": "default_retrieval_surface",
            "reason": f"{winner_name} cleared default thresholds and separated from {next_name}",
        }
    return {
        "winner": "",
        "decision": "run_phase_2_hybrid",
        "reason": "Two strategies passed but remained too close to choose safely",
    }


def load_experiment_inputs(
    output_dir: str,
    *,
    primary_bucket_size: int,
    sanity_size: int,
) -> tuple[list[IdentityResolutionExperimentLead], list[IdentityResolutionExperimentLead]]:
    cohorts = build_identity_resolution_experiment_cohort(
        output_dir,
        primary_bucket_size=primary_bucket_size,
        sanity_size=sanity_size,
    )
    return cohorts["primary"], cohorts["sanity"]
