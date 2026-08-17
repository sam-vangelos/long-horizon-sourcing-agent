"""LinkedIn side-effect service."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from linkedin.browser import LinkedInBrowser
from shared import config
from shared.execution import SideEffectOutcome
from shared.governor import (
    GovernorLimitReached,
    OperatorStopRequested,
    SessionExpired,
)
from shared.human_timing import human_delay_correlated
from shared.storage import log_event

if TYPE_CHECKING:
    from shared.schemas import CandidateSnippet, SearchString


@dataclass
class LinkedInSideEffectsDeps:
    browser: Any
    stats: dict[str, int]
    saved_urls: set[str]
    log_path: Path
    get_test_mode: Callable[[], bool]
    get_runtime_bridge: Callable[[], Any]
    get_runtime_run_id: Callable[[], int | None]
    before_irreversible_side_effect: Callable[[], None] = lambda: None
    # F2: the project assertion alone, without the stop-authority recheck that
    # `before_irreversible_side_effect` also carries. The already-saved branch
    # commits a SUCCEEDED receipt without ever clicking, so it needs the project
    # guard — but a governor cap or a stop request must not turn "this candidate
    # is already in the right pipeline" into a failure.
    assert_project_context: Callable[[], None] = lambda: None


class LinkedInSideEffectsService:
    """Owns LinkedIn save-click behavior and durable side-effect events."""

    def __init__(self, deps: LinkedInSideEffectsDeps):
        self.deps = deps

    async def handle_save_decision(
        self,
        *,
        snippet: "CandidateSnippet",
        runtime_search_string: "SearchString",
        attempt_id: int | None,
    ) -> SideEffectOutcome:
        self.deps.browser._last_save_failure_reason = None
        side_effect_row = None
        runtime_bridge = self.deps.get_runtime_bridge()
        runtime_run_id = self.deps.get_runtime_run_id()

        if (
            runtime_bridge
            and runtime_run_id
            and getattr(runtime_bridge, "begin_candidate_side_effect", None)
            and snippet.profile_url
        ):
            side_effect_start = runtime_bridge.begin_candidate_side_effect(
                run_id=runtime_run_id,
                search_string=runtime_search_string,
                snippet=snippet,
                attempt_id=attempt_id,
                effect_type="linkedin_save",
                idempotency_key="save",
                payload={"search_string_id": runtime_search_string.id},
            )
            side_effect_row = side_effect_start["side_effect"]
            if not side_effect_start["should_execute"]:
                runtime_bridge.record_side_effect_result(
                    run_id=runtime_run_id,
                    search_string=runtime_search_string,
                    snippet=snippet,
                    attempt_id=attempt_id,
                    effect_type="linkedin_save",
                    status="skipped",
                    payload={"skip_reason": f"existing_{side_effect_row['status']}"},
                )
                return SideEffectOutcome(
                    effect_type="linkedin_save",
                    status="skipped",
                    payload={"skip_reason": f"existing_{side_effect_row['status']}"},
                )

        if self.deps.get_test_mode():
            self.deps.stats["saved"] += 1
            if side_effect_row and getattr(runtime_bridge, "complete_candidate_side_effect", None):
                runtime_bridge.complete_candidate_side_effect(
                    side_effect_id=int(side_effect_row["id"]),
                    status="succeeded",
                    payload={"test_mode": True},
                )
            if runtime_bridge and runtime_run_id:
                runtime_bridge.record_side_effect_result(
                    run_id=runtime_run_id,
                    search_string=runtime_search_string,
                    snippet=snippet,
                    attempt_id=attempt_id,
                    effect_type="linkedin_save",
                    status="succeeded",
                    payload={"test_mode": True},
                )
            return SideEffectOutcome(
                effect_type="linkedin_save",
                status="succeeded",
                payload={"test_mode": True},
            )

        profile_url = getattr(snippet, "profile_url", None) or ""
        already_present = False
        saved = False
        save_error: Exception | None = None
        failure_reason: str | None = None
        expected_identity = LinkedInBrowser._profile_url_fragment(profile_url)

        def assert_expected_identity() -> None:
            if (
                not expected_identity
                or self.deps.browser.current_profile_identity_fragment()
                != expected_identity
            ):
                raise RuntimeError("profile identity mismatch during save")

        try:
            if side_effect_row is None and profile_url in self.deps.saved_urls:
                print("    Already saved this session (skipping duplicate)")
                saved = True
                already_present = True
            else:
                # F2: the project is asserted on BOTH sides of the probe, the
                # same bracketing profile identity already gets. The branch
                # below returns a succeeded/already_present receipt without
                # calling save_candidate, so it never reaches
                # `before_save_click` — the only place the project assertion
                # used to live. Being present in some OTHER project does not
                # discharge a save owed to the brief's project, and a page that
                # changes project during the probe must not be believed either.
                assert_expected_identity()
                self.deps.assert_project_context()
                # Wave G: read the candidate's OWN result card, not the panel.
                # The panel answers about whatever it is currently showing, so
                # presence in a DIFFERENT project could terminalize a save owed
                # to the brief's — F2 guarded that with project assertions
                # either side; scoping the read by href removes the question.
                # A None answer means the card could not be read, which is not
                # evidence of anything: fall through to save_candidate, which
                # resolves the same card and refuses if it still cannot.
                already_saved = await self.deps.browser.is_already_saved_on_card(
                    expected_identity
                )
                assert_expected_identity()
                self.deps.assert_project_context()
                if already_saved is True:
                    print("    Already in LinkedIn pipeline (skipping save click)")
                    saved = True
                    already_present = True
                else:
                    def before_save_click() -> None:
                        assert_expected_identity()
                        self.deps.before_irreversible_side_effect()

                    before_save_click()
                    save_confirmed = await self.deps.browser.save_candidate(
                        before_click=before_save_click,
                    )
                    # Record the OUTCOME before any post-hoc assertion can raise.
                    # These assertions used to run first, so a save that had
                    # physically landed in Recruiter was durably written as
                    # "failed" whenever the page drifted during the ~2s
                    # confirmation window — the ledger then disagreed with
                    # reality in the one direction that is unrecoverable, since
                    # nothing downstream knows to go looking for it. The
                    # assertions still fire below and still stop the run; they
                    # just can no longer rewrite what already happened.
                    saved = save_confirmed
                    if saved:
                        print("    Saved to LinkedIn pipeline")
                        self.deps.saved_urls.add(profile_url)
                    else:
                        print("    [warn] LinkedIn save may have failed")
                    assert_expected_identity()
                    if save_confirmed:
                        # F2, closing the rest of the population: the project
                        # assertion inside `before_save_click` only runs on the
                        # branch that CLICKS. `save_candidate` also returns True
                        # from its own already-saved probes (browser.py `_do`:
                        # the opening probe, the missing-trigger re-probe, the
                        # post-exception probe) — none of which reach
                        # `before_click`. A tab that moved to another project
                        # between the service's probe and one of those answers
                        # "already saved" about the WRONG pipeline, and the
                        # success was committed unexamined. Asserted where the
                        # success is consumed, so every return-True path is
                        # covered rather than one branch at a time. Only on a
                        # confirmed save: a failed save has no false success to
                        # prevent, and raising there would overwrite the
                        # browser's save_trigger_not_found / save_not_persisted
                        # classification.
                        self.deps.assert_project_context()
                    linger = max(
                        config.LINKEDIN_SAVE_LINGER_MIN_SECONDS,
                        min(
                            config.LINKEDIN_SAVE_LINGER_MAX_SECONDS,
                            human_delay_correlated(
                                config.LINKEDIN_SAVE_LINGER_BASE_SECONDS,
                                channel="save_linger",
                            ),
                        ),
                    )
                    chunks_back = random.randint(
                        config.LINKEDIN_SAVE_LINGER_MIN_CHUNKS_BACK,
                        config.LINKEDIN_SAVE_LINGER_MAX_CHUNKS_BACK,
                    )
                    px = await self.deps.browser.scroll_for_linger(chunks_back)
                    await asyncio.sleep(linger)
                    print(f"    [profile-read] SAVE verdict → lingering {linger:.1f}s, scrolled back {chunks_back} chunks")
                    await self.deps.browser.scroll_restore(px)
        except (OperatorStopRequested, SessionExpired, GovernorLimitReached):
            raise
        except Exception as exc:
            save_error = exc
            browser_reason = getattr(
                self.deps.browser,
                "_last_save_failure_reason",
                None,
            )
            failure_reason = (
                browser_reason
                if isinstance(browser_reason, str) and browser_reason
                else f"{type(exc).__name__}: "
                f"{str(exc).splitlines()[0] if str(exc) else ''}"
            )

        # P1.2: "already saved this session" / "already in pipeline" no
        # longer inflates stats["saved"] — those saves happened in some
        # earlier attempt or run. They count as already_present; a failed
        # click counts as save_failed. Only a save that physically landed
        # THIS call increments "saved". A reconciled self-save
        # (already_present on a replay whose own earlier attempt landed the
        # click before a disconnect) is OUR save and counts as saved, matching
        # the orchestrator's live guards and ledger hydration — otherwise the
        # global counter under-reports until the next startup rebuild.
        reconciled_self_save = bool(
            already_present
            and side_effect_row
            and int(side_effect_row.get("attempt_count") or 1) > 1
        )
        if saved and already_present and not reconciled_self_save:
            self.deps.stats["already_present"] = (
                self.deps.stats.get("already_present", 0) + 1
            )
        elif saved:
            self.deps.stats["saved"] += 1
        else:
            self.deps.stats["save_failed"] = (
                self.deps.stats.get("save_failed", 0) + 1
            )

        # P1.2: persist the browser's failure classification
        # (save_trigger_not_found = DOM/class rotation vs
        # save_not_persisted = clicked but state unchanged) into the
        # durable ledger row so systemic rotation is distinguishable
        # from flakes after the fact.
        if not saved and failure_reason is None:
            failure_reason = getattr(
                self.deps.browser, "_last_save_failure_reason", None
            )

        result_payload: dict[str, Any] = {"test_mode": False}
        if already_present:
            result_payload["already_present"] = True
            if reconciled_self_save:
                result_payload["reconciled_self_save"] = True
        if failure_reason:
            result_payload["failure_reason"] = failure_reason

        if side_effect_row and getattr(runtime_bridge, "complete_candidate_side_effect", None):
            runtime_bridge.complete_candidate_side_effect(
                side_effect_id=int(side_effect_row["id"]),
                status="succeeded" if saved else "failed",
                payload=result_payload,
            )

        if runtime_bridge and runtime_run_id:
            runtime_bridge.record_side_effect_result(
                run_id=runtime_run_id,
                search_string=runtime_search_string,
                snippet=snippet,
                attempt_id=attempt_id,
                effect_type="linkedin_save",
                status="succeeded" if saved else "failed",
                payload=result_payload,
            )
        log_event(
            self.deps.log_path,
            "candidate_saved",
            name=snippet.name,
            profile_url=snippet.profile_url,
            string_id=runtime_search_string.id,
            linkedin_save=saved,
            already_present=already_present,
            failure_reason=failure_reason,
        )
        if save_error is not None:
            raise save_error
        return SideEffectOutcome(
            effect_type="linkedin_save",
            status="succeeded" if saved else "failed",
            payload=result_payload,
        )
