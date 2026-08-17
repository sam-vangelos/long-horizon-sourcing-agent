"""LinkedIn acquisition service."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from shared.identity_resolution import (
    classify_recruiter_activity_pressure,
    normalize_public_linkedin_url,
)
from shared.execution import AcquisitionResult
from shared.extractors import extract_profile_from_dom, extract_snippet_from_card_innertext
from shared.failures import (
    ApiBudgetExhaustedError,
    classify_runtime_failure,
    is_api_budget_exhausted_error,
)
from shared.governor import (
    GovernorLimitReached,
    OperatorStopRequested,
    SessionExpired,
)
from shared import config
from shared.human_timing import human_delay_correlated
from shared.reconciliation_schemas import RecruiterActivitySnapshot
from shared.storage import log_event

if TYPE_CHECKING:
    from shared.schemas import CandidateSnippet, SearchString


_BROWSER_DISCONNECT_PATTERNS = (
    "page crashed",
    "target crashed",
    "target closed",
    "target page, context or browser has been closed",
    "has been closed",
    "connection closed",
    "session closed",
    "broken pipe",
    "browser has been closed",
    "page closed",
    "context closed",
    "page.createisolatedworld",
    "page.addscripttoevaluateonnewdocument",
    "cannot get world",
)

_LOCAL_BROWSER_PROCESS_FAILURE_MARKERS = (
    "target crashed",
    "page crashed",
    "target page, context or browser has been closed",
    "browser has been closed",
    "page.createisolatedworld",
    "page.addscripttoevaluateonnewdocument",
    "cannot get world",
)


def _is_browser_disconnect_error(
    error: BaseException | str,
    *,
    include_render_failures: bool = False,
) -> bool:
    if include_render_failures:
        # Lazy to avoid acquisition -> orchestrator -> acquisition at module load.
        from linkedin.orchestrator import PageRenderFailedError

        if isinstance(error, PageRenderFailedError):
            return True
    text = str(error).lower()
    return any(pattern in text for pattern in _BROWSER_DISCONNECT_PATTERNS)


def _is_local_browser_process_failure(error: BaseException | str) -> bool:
    from rebrowser_playwright._impl._errors import TargetClosedError

    if isinstance(error, TargetClosedError):
        return True
    text = str(error).lower()
    return any(
        text.find(marker) >= 0
        for marker in _LOCAL_BROWSER_PROCESS_FAILURE_MARKERS
    )


@dataclass
class LinkedInAcquisitionDeps:
    browser: Any
    log_path: Path
    ensure_browser_healthy: Callable[[], Awaitable[None]]


class LinkedInAcquisitionService:
    """Owns LinkedIn candidate snippet and profile acquisition behavior."""

    def __init__(self, deps: LinkedInAcquisitionDeps):
        self.deps = deps
        self.last_card_extraction_error: Exception | None = None

    async def extract_card_snippet(
        self,
        search_string: "SearchString",
        page_num: int,
        card_index: int,
    ) -> AcquisitionResult | None:
        self.last_card_extraction_error = None
        try:
            await self.deps.browser.focus_card_for_review(card_index)
            await asyncio.sleep(
                human_delay_correlated(random.uniform(0.7, 1.8), channel="card_glance")
            )
            snapshot = await self.deps.browser.get_card_snapshot(card_index)
        except (
            ApiBudgetExhaustedError,
            GovernorLimitReached,
            OperatorStopRequested,
            SessionExpired,
        ):
            raise
        except Exception as exc:
            if _is_browser_disconnect_error(exc):
                raise
            if is_api_budget_exhausted_error(exc):
                raise ApiBudgetExhaustedError(str(exc)) from exc
            self.last_card_extraction_error = exc
            print(f"    [warn] Card {card_index + 1} could not be extracted: {exc}")
            log_event(
                self.deps.log_path,
                "card_extract_error",
                string_id=search_string.id,
                page=page_num,
                card_index=card_index + 1,
                error=str(exc),
            )
            try:
                await self.deps.browser.go_back_to_results()
            except Exception as cleanup_error:
                if _is_browser_disconnect_error(cleanup_error):
                    raise
            return None

        innertext = (snapshot.get("innertext") or "").strip()
        if not innertext and not snapshot.get("name"):
            print(f"    [warn] Card {card_index + 1} rendered without readable text")
            return None

        try:
            snippet = extract_snippet_from_card_innertext(
                innertext,
                string_id=search_string.id,
                string_name=search_string.name,
                page=page_num,
                result_rank=card_index + 1,
                dom_name=(snapshot.get("name") or "").strip(),
                dom_url=(snapshot.get("url") or "").strip(),
            )
        except (
            ApiBudgetExhaustedError,
            GovernorLimitReached,
            OperatorStopRequested,
            SessionExpired,
        ):
            raise
        except Exception as exc:
            if isinstance(exc.__cause__, json.JSONDecodeError):
                self.last_card_extraction_error = exc
                print(f"    [warn] Card {card_index + 1} could not be extracted: {exc}")
                log_event(
                    self.deps.log_path,
                    "card_extract_error",
                    string_id=search_string.id,
                    page=page_num,
                    card_index=card_index + 1,
                    candidate_name=(snapshot.get("name") or "").strip(),
                    profile_url=(snapshot.get("url") or "").strip(),
                    error=str(exc),
                )
                return None
            if _is_browser_disconnect_error(exc):
                raise
            if is_api_budget_exhausted_error(exc):
                raise ApiBudgetExhaustedError(str(exc)) from exc
            classification = classify_runtime_failure(exc, source="llm")
            if classification.domain in {"auth", "request", "provider", "network"}:
                raise
            self.last_card_extraction_error = exc
            print(f"    [warn] Card {card_index + 1} could not be extracted: {exc}")
            log_event(
                self.deps.log_path,
                "card_extract_error",
                string_id=search_string.id,
                page=page_num,
                card_index=card_index + 1,
                candidate_name=(snapshot.get("name") or "").strip(),
                profile_url=(snapshot.get("url") or "").strip(),
                error=str(exc),
            )
            return None
        if snippet is None:
            print(f"    [warn] Could not extract snippet from card {card_index + 1}")
            return None

        if snapshot.get("name"):
            snippet.name = snapshot["name"]
        if snapshot.get("url"):
            # Phase C-bis 0.4: strip LinkedIn search-result tracking
            # parameters (miniProfileUrn / trackingId / searchEntityType /
            # position / searchId) before persisting. The browser DOM
            # returns the full URL with these embedded; storing them
            # bloats the candidate-detail UI rendering and pollutes the
            # diagnostic Reference Slip. Normalization is idempotent —
            # already-clean URLs pass through unchanged.
            snippet.profile_url = normalize_public_linkedin_url(snapshot["url"])
        snippet.card_index = card_index
        snippet.already_saved = bool(snapshot.get("already_saved", False))
        snippet.recruiter_activity = RecruiterActivitySnapshot.from_dict(
            snapshot.get("recruiter_activity")
        )
        snippet.novelty_pressure = classify_recruiter_activity_pressure(
            snippet.recruiter_activity
        )
        return AcquisitionResult(snippet=snippet, metadata={"snapshot": snapshot})

    async def extract_profile_summary(
        self, snippet: "CandidateSnippet", *, interest: float = 0.5
    ) -> AcquisitionResult:
        await self.deps.ensure_browser_healthy()

        print("    Opening profile for full evaluation...")
        scan_time = random.uniform(0.8, 2.2)
        await asyncio.sleep(human_delay_correlated(scan_time, channel="snippet_scan"))

        if snippet.card_index >= 0:
            await self.deps.browser.ensure_card_rendered(snippet.card_index)

        # Governance (check + count) lives inside the browser's profile-open
        # methods, so every caller uses the same authority.
        if snippet.profile_url:
            await self.deps.browser.open_profile_by_url(snippet.profile_url)
        else:
            await self.deps.browser.open_profile(snippet.name)

        if config.LINKEDIN_SECTION_DIRECTED_READ_ENABLED:
            await self.deps.browser.expand_profile_sections()

        await self.deps.browser.simulate_profile_read(interest)
        profile_text = await self.deps.browser.get_profile_innertext()
        print(f"    Profile text size: {len(profile_text) / 1024:.0f} KB")
        summary = extract_profile_from_dom(profile_text, snippet.profile_url)
        return AcquisitionResult(
            profile_summary=summary,
            metadata={"profile_text_size": len(profile_text)},
        )
