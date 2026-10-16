"""Offline evaluation-loop recordings; production effects, recorded externals.

No fixture is a complete historical campaign capture. Inputs are synthetic or
artifact-derived, as stated by each fixture. This module never reads output/.
The comparator keeps console bytes and effect ordering, normalizes temporary
paths and diagnostic provenance, and checks the actual final SQLite rows.
"""
from __future__ import annotations

from linkedin.orchestrator import Pipeline as PipelineFacade

import asyncio
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, is_dataclass
import datetime as datetime_module
from datetime import datetime, timezone
import difflib
import importlib
import io
import json
import logging
from pathlib import Path
import random
import re
import sqlite3
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "evaluation_loop"
REAL_SLEEP = asyncio.sleep
REAL_DATETIME = datetime
FIXED_TIME = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


class RecordedClock:
    """The tape supplies wall time; logical waits advance no real clock."""

    class DateTime(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return FIXED_TIME.replace(tzinfo=None) if tz is None else FIXED_TIME.astimezone(tz)

        @classmethod
        def utcnow(cls):
            return FIXED_TIME.replace(tzinfo=None)

    def __init__(self, trace):
        self.trace = trace
        self.ticks = 0

    def time(self):
        return FIXED_TIME.timestamp()

    def monotonic(self):
        self.ticks += 1
        return self.ticks / 100

    async def sleep(self, seconds):
        self.trace.emit("clock.sleep", seconds)
        await REAL_SLEEP(0)


class EffectTrace(io.StringIO):
    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.effects = []
        self.enabled = False
        self.violations = []
        self.page_commit = False
        self.injected = False

    def normalize(self, value):
        if hasattr(value, "to_dict"):
            return self.normalize(value.to_dict())
        if is_dataclass(value):
            return self.normalize(asdict(value))
        if isinstance(value, dict):
            return {
                str(key): (
                    "<diagnostic-hash>"
                    if key in {"receipt_id", "input_hash", "event_hash", "previous_event_hash", "prev_event_hash"}
                    else self.normalize(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.normalize(item) for item in value]
        if isinstance(value, set):
            return sorted(self.normalize(item) for item in value)
        if isinstance(value, Path):
            return self.normalize(str(value))
        if isinstance(value, str):
            value = value.replace(str(self.root), "<RUN>")
            if value.startswith(("{", "[")):
                try:
                    return self.normalize(json.loads(value))
                except ValueError:
                    pass
            return value
        if isinstance(value, sqlite3.Connection):
            return "<sqlite-connection>"
        if isinstance(value, BaseException):
            return {"type": type(value).__name__, "message": str(value)}
        if value is None or isinstance(value, (bool, int, float)):
            return value
        raise TypeError(f"unrecordable value: {type(value).__name__}")

    def emit(self, kind, value):
        if self.enabled:
            self.effects.append({"effect": kind, "value": self.normalize(value)})

    def forbid(self, name):
        self.violations.append(name)
        self.emit("unrecorded.external", name)
        raise AssertionError(f"unrecorded external call: {name}")

    def write(self, value):
        self.emit("console", value)
        return super().write(value)


class RecordedBrowser:
    """A browser-shaped recording adapter; no browser object is constructed."""

    def __init__(self, spec, trace):
        self.spec, self.trace = spec, trace
        self.page_num = 1
        self.boolean = "replay"
        self.identity = ""
        self.saved = set(spec.get("already_saved", []))
        self.calls = {}
        self._last_save_failure_reason = None
        self._project_id = "replay-project"
        self._state_dir = None
        # Plain state the real browser carries from __init__ onward: the facet
        # set the last reconcile PROVED. Read, never called, so it records no
        # effect — but a fake without it turns a state read into a violation.
        self._owned_facet_target = None
        self.page = SimpleNamespace(
            url=self.search_url,
            title=self.health_title,
            locator=self.locator,
            reload=self.reload,
            wait_for_timeout=self.wait_for_timeout,
        )

    def __getattr__(self, name):
        self.trace.forbid(f"browser.{name}")

    def locator(self, selector):
        self.trace.emit("browser.locator", selector)
        if selector == "body":
            return SimpleNamespace(inner_text=self.health_body)
        if selector != 'text="No search results"':
            self.trace.forbid(f"browser locator {selector}")
        return SimpleNamespace(first=SimpleNamespace(is_visible=self.no_results))

    async def health_title(self):
        return self.respond("health.title", result=self.spec.get("health_title", "LinkedIn Recruiter"))

    async def health_body(self, **kwargs):
        return self.respond("health.body", kwargs, self.spec.get("health_body", "Recorded Recruiter results"))

    async def results_page_offset(self):
        default = (self.page_num - 1) * 25
        return self.respond("results_page_offset", result=self.spec.get("page_offset", default))

    def _zero_slots(self, *, upcoming=False):
        index = self.calls.get("get_card_slot_count", 0) + int(upcoming)
        return self.spec.get("zero_slots", False) or index in self.spec.get("zero_slots_calls", [])

    @property
    def search_url(self):
        return "https://www.linkedin.com/talent/hire/replay-project/discover/recruiterSearch"

    @property
    def cards(self):
        return self.spec["pages"][min(self.page_num, len(self.spec["pages"])) - 1]

    def respond(self, method, arguments=None, result=None):
        self.calls[method] = self.calls.get(method, 0) + 1
        self.trace.emit("browser.call", {"method": method, "arguments": arguments})
        faults = self.spec.get("faults", [self.spec.get("fault", {})])
        for fault in faults:
            if method == fault.get("method") and self.calls[method] in fault.get("calls", []):
                from linkedin.evaluation_errors import (
                    GeographyRegimeError, PanelRecoveryError, PageRenderFailedError,
                    SurfaceOwnershipLostError, ProjectContextMismatchError,
                )
                from shared.governor import GovernorLimitReached
                error_type = {
                    "RuntimeError": RuntimeError, "GeographyRegimeError": GeographyRegimeError,
                    "PanelRecoveryError": PanelRecoveryError, "PageRenderFailedError": PageRenderFailedError,
                    "SurfaceOwnershipLostError": SurfaceOwnershipLostError,
                    "ProjectContextMismatchError": ProjectContextMismatchError,
                    "GovernorLimitReached": GovernorLimitReached,
                }[fault.get("type", "RuntimeError")]
                error = error_type(fault.get("message", "Page.evaluate: Target crashed"))
                self.trace.emit("browser.raise", error)
                raise error
        self.trace.emit("browser.return", {"method": method, "result": result})
        return result

    def set_required_project_id(self, project_id):
        assert project_id == "replay-project"

    def set_company_facet_scope(self, scope):
        assert scope is None

    async def connect(self):
        self.respond("connect")

    async def disconnect(self):
        self.respond("disconnect")

    async def check_and_recover(self):
        return self.respond("check_and_recover", result=False)

    async def navigate_to_search(self, url):
        self.respond("navigate_to_search", url)
        self.page.url = url
        self.page_num, self.identity = 1, ""

    async def enter_search_string(self, boolean):
        self.respond("enter_search_string", boolean)
        self.boolean, self.page_num = boolean, 1
        self.page.url = self.search_url

    async def get_results_count_text(self):
        return self.respond("get_results_count_text", result=str(self.spec.get("result_count", 25)))

    async def get_results_count(self):
        return self.respond("get_results_count", result=self.spec.get("result_count", 25))

    async def no_results(self, **kwargs):
        return self.respond("no_results", kwargs, False)

    async def get_card_slot_count(self):
        count = 0 if self._zero_slots(upcoming=True) else len(self.cards)
        return self.respond("get_card_slot_count", result=count)

    async def get_card_count(self):
        return await self.get_card_slot_count()

    async def scroll_to_load_all_results(self):
        return self.respond("scroll_to_load_all_results", result=0 if self._zero_slots() else len(self.cards))

    async def focus_card_for_review(self, index):
        self.respond("focus_card_for_review", index)

    async def get_card_snapshot(self, index):
        card = self.cards[index]
        return self.respond("get_card_snapshot", index, {
            "innertext": card["id"], "name": card["id"],
            "url": f"/talent/profile/{card['id']}",
            "already_saved": card["id"] in self.saved,
        })

    async def ensure_card_rendered(self, index):
        self.respond("ensure_card_rendered", index)

    async def open_profile_by_url(self, url):
        self.respond("open_profile_by_url", url)
        self.identity = url.rsplit("/", 1)[-1]
        self.page.url = f"{self.search_url}/profile/{self.identity}"

    async def expand_profile_sections(self):
        self.respond("expand_profile_sections")

    async def simulate_profile_read(self, interest):
        self.respond("simulate_profile_read", interest)

    async def get_profile_innertext(self):
        return self.respond("get_profile_innertext", result=self.identity)

    async def get_profile_status_summary(self):
        return self.respond("get_profile_status_summary", result={})

    def current_profile_identity_fragment(self):
        return self.identity

    async def go_back_to_results(self):
        self.respond("go_back_to_results")
        self.identity = ""
        self.page.url = self.search_url

    async def is_already_saved_on_card(self, identity):
        return self.respond("is_already_saved_on_card", identity, identity in self.saved)

    async def save_candidate(self, *, before_click=None):
        if before_click:
            before_click()
        self.respond("save_candidate", self.identity)
        unconfirmed = self.calls["save_candidate"] in self.spec.get("unconfirmed_save_calls", [])
        if not unconfirmed or self.spec.get("save_landed", True):
            self.saved.add(self.identity)
        if unconfirmed:
            self._last_save_failure_reason = self.spec.get("save_failure_reason", "save_not_confirmed")
            return False
        return True

    async def scroll_for_linger(self, chunks):
        return self.respond("scroll_for_linger", chunks, 0)

    async def scroll_restore(self, pixels):
        self.respond("scroll_restore", pixels)

    async def go_to_next_page(self):
        missing = (self.spec.get("pagination_missing") or
                   self.calls.get("go_to_next_page", 0) + 1 in self.spec.get("pagination_missing_calls", []))
        result = not missing and self.page_num < len(self.spec["pages"])
        self.respond("go_to_next_page", result=result)
        if result:
            self.page_num += 1
        return result

    async def _peek_top_card_signature(self):
        signature = (self.cards[0]["id"], str(self.page_num)) if self.cards else ("", "")
        if self.spec.get("page_local_recovery") and self.cards:
            signature = (self.cards[0]["id"], f"/talent/profile/{self.cards[0]['id']}")
        if self.spec.get("unreadable_identity"):
            signature = ("", "")
        return self.respond("_peek_top_card_signature", result=signature)

    async def read_applied_location_chips(self):
        return self.respond("read_applied_location_chips", result=self.spec.get("location_chips", []))

    async def reconcile_facets(self, **kwargs):
        return self.respond("reconcile_facets", kwargs)

    async def reload_search_page(self):
        return self.respond("reload_search_page", result=True)

    async def reload(self, **kwargs):
        self.respond("reload", kwargs)

    async def wait_for_timeout(self, milliseconds):
        self.respond("wait_for_timeout", milliseconds)

    async def capture_failure_forensics(self, label):
        self.respond("capture_failure_forensics", label)

    async def find_result_slot_by_profile_url(self, url):
        identity = url.rsplit("/", 1)[-1]
        slot = next((index for index, card in enumerate(self.cards) if card["id"] == identity), None)
        return self.respond("find_result_slot_by_profile_url", url, slot)

    def get_current_search_url(self):
        return self.search_url

    async def snapshot_advanced_search_controls(self):
        return self.respond("snapshot_advanced_search_controls", result={"controls": []})


class RecordedJudgments:
    def __init__(self, spec, trace):
        self.spec, self.trace = spec, trace
        self.cards = {card["id"]: card for page in spec["pages"] for card in page}
        self.full_calls = 0

    def charge(self, stage, context=None):
        from shared.llm_usage import record_llm_usage
        record_llm_usage(
            provider="anthropic", model="claude-opus-4-1-20250805",
            usage={"input_tokens": 1000, "output_tokens": 200},
            request={"stage": stage}, usage_context=context,
        )

    def decision(self, name, stage):
        from shared.schemas import OpusDecision
        card = self.cards[name]
        verdict = card.get(stage, "REJECT" if stage == "full" else "FACIAL_NO")
        return OpusDecision(
            stage=stage, decision=verdict, path="DIRECT:systems" if verdict == "SAVE" else "none",
            confidence=0.8, rationale=f"Recorded {stage} evidence for {name}.",
            candidate_name=name, profile_url=f"/talent/profile/{name}",
            outreach_tier="STANDARD" if verdict == "SAVE" else "",
        )

    def facial(self, snippet, brief=None, **kwargs):
        self.trace.emit("llm.facial", {"snippet": snippet, "brief": brief, "kwargs": kwargs})
        self.charge("facial", kwargs.get("lane_context"))
        return self.decision(snippet.name, "facial")

    def facial_batch(self, snippets, brief=None, **kwargs):
        self.trace.emit("llm.facial_batch", {"snippets": snippets, "brief": brief, "kwargs": kwargs})
        self.charge("facial", kwargs.get("lane_context"))
        return [self.decision(item.name, "facial") for item in snippets]

    def full(self, summary, brief=None, **kwargs):
        self.trace.emit("llm.full", {"summary": summary, "brief": brief, "kwargs": kwargs})
        self.full_calls += 1
        self.charge("full", kwargs.get("lane_context"))
        return self.decision(summary.name, "full")

    def create(self, **kwargs):
        """Actual SDK seam for legacy JSON truncation and parser recovery."""
        self.trace.emit("llm.request", kwargs)
        responses = self.spec["raw_responses"]
        if self.full_calls >= len(responses):
            self.trace.forbid("recorded full responses exhausted")
        response = responses[self.full_calls]
        self.full_calls += 1
        self.trace.emit("llm.response", response)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=response["text"])],
            stop_reason=response.get("stop_reason", "end_turn"),
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )

    def stream(self, **kwargs):
        """Streamed SDK seam: opus_llm_cached streams for every model, and the
        SDK sends the request on ``__enter__``, so the recorded response is
        served there through the same ``create`` tape."""
        judgments = self

        class _StreamManager:
            def __enter__(self):
                message = judgments.create(**kwargs)
                return SimpleNamespace(get_final_message=lambda: message)

            def __exit__(self, *exc):
                return False

        return _StreamManager()


def replay(spec, root):
    """Replay through run_full with real candidate/page/persistence procedures."""
    from shared.brief_loader import Brief
    from shared.schemas import CandidateSnippet, CandidateProfileSummary, Progress, SearchString, AdaptationResponse, ExecutionPlan
    from shared.runtime_state.store import RuntimeStateStore
    from linkedin.search_intelligence import bootstrap_experiment_state
    from linkedin.adaptation_signal_state import AdaptationGateConfig
    import shared.llm_clients as clients
    import shared.judger as judger
    import shared.storage as storage
    import shared.config as config

    orch = importlib.import_module("linkedin.evaluation_loop")
    reviewer_module = importlib.import_module("linkedin.candidate_review")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    trace = EffectTrace(root)
    clock = RecordedClock(trace)
    browser = RecordedBrowser(spec, trace)
    judgments = RecordedJudgments(spec, trace)
    brief = Brief(
        id="evaluation-replay", role_title="Systems Engineer", role_description="Build systems.",
        kit_url="", linkedin_project="Replay", linkedin_project_id="replay-project",
        minimum_bar="Evidence of building systems.", archetypes=[], noise_archetypes=[],
        hard_skips=[], clear_skips_from_review=[], known_noise_patterns=[],
        permanent_filters={}, save_instructions={}, experience_floor={},
    )
    brief_path = root / "brief.json"
    brief_path.write_text('{"id":"evaluation-replay"}')
    sequence = iter(range(1, 10000))

    def snippet_from_record(text, **kwargs):
        return CandidateSnippet(
            name=text, headline="Engineer", current_title="Engineer", current_company="Recorded Company",
            location="", education_snippet="", profile_url=kwargs["dom_url"],
            source_string_id=kwargs["string_id"], source_string_name=kwargs["string_name"],
            page=kwargs["page"], result_rank=kwargs["result_rank"],
        )

    def profile_from_record(text, url):
        return CandidateProfileSummary(name=text, profile_url=url, headline="Engineer")

    def forbidden(*args, **kwargs):
        trace.forbid("network or subprocess")

    def recorded_client(provider, *args, **kwargs):
        if provider != "anthropic" or not spec.get("raw_responses"):
            trace.forbid(f"LLM provider {provider}")
        return SimpleNamespace(
            messages=SimpleNamespace(create=judgments.create, stream=judgments.stream)
        )

    def recorded_delay(*args, **kwargs):
        trace.emit("clock.delay", {"args": args, "kwargs": kwargs})
        return float(args[0])


    async def inline_thread(function, /, *args, **kwargs):
        # Recorded executor ordering; actual thread overlap has its existing
        # independent wall-time test. This tape never creates provider workers.
        return function(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp, ExitStack() as stack:
        # The recordings were made under the config DEFAULTS of a checkout with
        # no .env. A live checkout's .env (real fleet, real caps) reaches the
        # printed judgment_runtime_profile and its fingerprint, so every knob
        # the loop can read is pinned to that frozen baseline first; the
        # explicit values below override it. Secrets are never in the file.
        for name, value in json.loads((FIXTURES.parent / "evaluation_loop_config_baseline.json").read_text()).items():
            if hasattr(config, name):
                mp.setattr(config, name, value)
        # Four flags the tapes CANNOT carry at their production value. Each was
        # promoted to default-on 2026-09-20, and each changes recorded bytes —
        # a note-write effect, a different batch-triage console line, extra
        # timing effects, a different profile-read path — so replaying them on
        # true would demand re-recording all 24 transcripts from the
        # implementation, which is how a tape stops being a lock. They are
        # pinned here, in the open, rather than silently inherited: the harness
        # is HONESTLY narrower than production on these four and mirrors it on
        # the other six.
        for name in ("LINKEDIN_SAVE_NOTE_ENABLED", "LINKEDIN_FACIAL_PAGE_CONTAINMENT_ENABLED",
                     "LINKEDIN_TIMING_TELEMETRY_ENABLED", "LINKEDIN_SECTION_DIRECTED_READ_ENABLED"):
            mp.setattr(config, name, False)
        for name, value in {
            "FULL_EVAL_PIPELINE_ENABLED": spec.get("pipeline", False),
            "FULL_EVAL_MODEL_NAME": "claude-opus-4-1-20250805",
            "FACIAL_MODEL_NAME": "claude-opus-4-1-20250805",
            "LINKEDIN_V2_FACIAL_CONTRACT": "legacy", "LINKEDIN_V2_FULL_CONTRACT": "legacy",
            "FIREWORKS_JUDGMENT_POLICY_ENABLED": False, "FIREWORKS_PROMPT_AFFINITY_ENABLED": False,
            "LINKEDIN_FACIAL_CONCURRENCY_ENABLED": False,
            "LINKEDIN_EXTERNAL_EVIDENCE_ENABLED": False, "SHADOW_FACIAL_MODEL_ENABLED": False,
            "SHADOW_STRATEGY_ENABLED": False, "LINKEDIN_FULL_ID_MISMATCH_RETRY_ENABLED": False,
            "LINKEDIN_PAGE_ALLOCATOR_MODE": "off",
            "LINKEDIN_TOTAL_PAGE_CAP": 0,
            "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED": False, "LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED": False,
            "LINKEDIN_FACIAL_TIGHTENING_ENABLED": False,
            "LINKEDIN_VOCABULARY_ENUMERATION_ENABLED": False, "CADENCE_INTERVAL_MINUTES": 0,
            "GLANCE_MIN_SNIPPETS": 99, "EARLY_EXIT_MIN_CANDIDATES": 99,
            "MAX_PAGES_PER_STRING": spec.get("max_pages", len(spec["pages"])),
            "LINKEDIN_SAVE_LINGER_MIN_SECONDS": 1, "LINKEDIN_SAVE_LINGER_MAX_SECONDS": 1,
            "LINKEDIN_SAVE_LINGER_BASE_SECONDS": 1, "LINKEDIN_SAVE_LINGER_MIN_CHUNKS_BACK": 0,
            "LINKEDIN_SAVE_LINGER_MAX_CHUNKS_BACK": 0, "LINKEDIN_REJECT_CLOSE_MIN_SECONDS": 1,
            "LINKEDIN_REJECT_CLOSE_MAX_SECONDS": 1, "LINKEDIN_REJECT_CLOSE_BASE_SECONDS": 1,
            "LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS": 1,
        }.items():
            mp.setattr(config, name, value)
        mp.setenv("LINKEDIN_PAGE_LOCAL_RECOVERY_ENABLED", "true" if spec.get("page_local_recovery") else "false")
        mp.setenv("LANGFUSE_DISABLE", "1")
        mp.setattr(datetime_module, "datetime", RecordedClock.DateTime)
        for module in list(sys.modules.values()):
            name = getattr(module, "__name__", "")
            if name.startswith(("shared.", "linkedin.")) and getattr(module, "datetime", None) is REAL_DATETIME:
                mp.setattr(module, "datetime", RecordedClock.DateTime)
        mp.setattr(importlib.import_module("linkedin.evaluation_loop"), "time", clock)
        mp.setattr(importlib.import_module("linkedin.page_review"), "time", clock)
        mp.setattr(reviewer_module, "time", clock)
        mp.setattr(importlib.import_module("linkedin.search_surface"), "time", clock)
        mp.setattr(importlib.import_module("linkedin.allocator_state"), "time", clock)
        mp.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(sequence)))
        mp.setattr(random, "uniform", lambda low, high: (low + high) / 2)
        mp.setattr(random, "randint", lambda low, high: low)
        mp.setattr(asyncio, "sleep", clock.sleep)
        mp.setattr(asyncio, "to_thread", inline_thread)
        # Stack coordinates change when the implementation moves. Their raw
        # bytes are an explicit unproven field; exception type/text stay real.
        import traceback
        mp.setattr(traceback, "format_exc", lambda: "".join(traceback.format_exception_only(*sys.exc_info()[:2])))
        mp.setattr(orch, "load_brief", lambda _: brief)
        browser_factory = lambda **_: browser
        browser_factory._profile_url_fragment = orch.LinkedInBrowser._profile_url_fragment
        mp.setattr(orch, "LinkedInBrowser", browser_factory)
        mp.setattr(clients, "get_llm_client", recorded_client)
        mp.setattr(socket.socket, "connect", forbidden)
        mp.setattr(socket.socket, "connect_ex", forbidden)
        mp.setattr(socket, "getaddrinfo", forbidden)
        mp.setattr(subprocess, "Popen", forbidden)
        mp.setattr(reviewer_module, "facial_judge", judgments.facial)
        mp.setattr(judger, "facial_judge_batch", judgments.facial_batch)
        if not spec.get("raw_responses"):
            mp.setattr(reviewer_module, "full_judge", judgments.full)
        mp.setattr(importlib.import_module("linkedin.acquisition"), "extract_snippet_from_card_innertext", snippet_from_record)
        mp.setattr(importlib.import_module("linkedin.acquisition"), "extract_profile_from_dom", profile_from_record)
        for module_name in ("linkedin.acquisition", "linkedin.side_effects", "linkedin.search_surface", "linkedin.candidate_review", "linkedin.page_review", "linkedin.string_driver"):
            mp.setattr(importlib.import_module(module_name), "human_delay_correlated", recorded_delay)
        def force_backoff(reason):
            governor.shutdown_reason = reason
            trace.emit("governor.force_backoff", reason)
        governor = SimpleNamespace(check_limits=lambda: governor.shutdown_reason,
                                   shutdown_reason=None, force_backoff=force_backoff)
        facade = PipelineFacade(str(brief_path), str(root), governor=governor)
        pipeline = facade._loop
        pipeline.state.context._session_expired = asyncio.Event()
        pipeline.state.context._operator_stop_event = asyncio.Event()
        if spec.get("batch"):
            stack.enter_context(patch.object(Brief, "has_v2_schema", property(lambda _: True)))
        pipeline._enrich_run_snapshot = lambda *args, **kwargs: trace.emit("snapshot.enrich", {"args": args, "kwargs": kwargs})
        pipeline.state._run_report_service._generate_run_report = lambda progress: trace.emit("report.generate", progress)
        pipeline._finalize_run_snapshot = lambda: (trace.emit("snapshot.freeze", pipeline.state.context._runtime_run_id) or root / "frozen")
        pipeline.state._run_report_service._run_health_summary = lambda: {"green_but_useless": False}
        pipeline.state.planner._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)
        pipeline.state.surface._capture_recovery_snapshot = lambda *args, **kwargs: _recorded_recovery_snapshot(trace, args, kwargs)
        pipeline.state._recovery_service.recover = lambda **kwargs: _recorded_recovery(trace, browser, kwargs)
        pipeline.state.surface._reassert_session_location_after_recovery = lambda: clock.sleep(0)
        mp.setattr(importlib.import_module("market_intelligence.live_advisory"), "record_block_checkpoint_and_get_context", lambda **_: "")
        def adapt(*args, **kwargs):
            trace.emit("llm.adapt", {"args": args, "kwargs": kwargs})
            judgments.charge("adaptation")
            return AdaptationResponse(no_change=True)
        def strategy(*args, **kwargs):
            trace.emit("llm.strategy", {"args": args, "kwargs": kwargs})
            judgments.charge("strategy")
            return ExecutionPlan(strategy_rationale="Recorded plan", generated_strings=[
                {"boolean": "replay", "rationale": "Recorded string"},
            ])
        strategy_module = importlib.import_module("linkedin.strategy")
        mp.setattr(strategy_module, "adapt_after_block", adapt)
        mp.setattr(strategy_module, "form_strategy", strategy)

        strings = [SearchString(id=1, name="Recorded string", boolean="replay", block="Block A")]
        if spec.get("adaptation"):
            strings.append(SearchString(id=2, name="Next string", boolean="next", block="Block B"))
        progress = Progress(brief_name=brief.id, strings=strings)
        first_page = spec.get("resume_page", 1)
        progress.current_string_id, progress.current_page = 1, first_page
        strings[0].pages_reviewed = first_page - 1
        state = bootstrap_experiment_state(strings[0])
        state.active_variant.allocator_page_cursor = first_page
        if not spec.get("fresh"):
            pipeline.state.context._runtime_run_id, _ = pipeline.state.context._runtime_bridge.start_or_resume_run(resume=False, initial_progress=progress, experiment_states={1: state})

        writer_prefixes = ("start_", "finish_", "record_", "set_", "upsert_", "begin_", "complete_", "delete_", "update_", "reconcile_", "ensure_candidate")
        for name in vars(RuntimeStateStore):
            original = getattr(RuntimeStateStore, name)
            if not name.startswith(writer_prefixes) or not callable(original):
                continue
            def recorded_write(self, *args, _name=name, _original=original, **kwargs):
                trace.emit("runtime.write", {"method": _name, "args": args, "kwargs": kwargs})
                return _original(self, *args, **kwargs)
            mp.setattr(RuntimeStateStore, name, recorded_write)
        original_connect = RuntimeStateStore.connect
        @contextmanager
        def connect(store):
            with original_connect(store) as connection:
                wrote_work_unit = False
                def sql(statement):
                    nonlocal wrote_work_unit
                    command = statement.lstrip().split(maxsplit=1)[0].upper()
                    if command not in {"BEGIN", "COMMIT", "ROLLBACK", "INSERT", "UPDATE", "DELETE", "REPLACE"}:
                        return
                    wrote_work_unit = wrote_work_unit or bool(re.search(r"(?:INTO|UPDATE)\s+work_units\b", statement))
                    # Normalize SQL literals through the same JSON/path rules;
                    # retain the statement, write order, and transaction edges.
                    def literal(match):
                        value = trace.normalize(match.group(1).replace("''", "'"))
                        if not isinstance(value, str):
                            value = json.dumps(value, sort_keys=True)
                        return "'" + value.replace("'", "''") + "'"
                    normalized = re.sub(r"'((?:''|[^'])*)'", literal, statement)
                    trace.emit("runtime.sql", normalized)
                connection.set_trace_callback(sql)
                yield connection
                if (trace.page_commit and wrote_work_unit and not trace.injected
                        and spec.get("checkpoint_fault") == "before_commit"):
                    trace.injected = True
                    connection.rollback()
                    raise RuntimeError("recorded interruption before canonical commit")
        mp.setattr(RuntimeStateStore, "connect", connect)
        original_replace = Path.replace
        def publish(path, target):
            result = original_replace(path, target)
            target = Path(target)
            if target.is_relative_to(root):
                contents = target.read_text()
                if target.suffix == ".jsonl":
                    contents = [json.loads(line) for line in contents.splitlines() if line]
                trace.emit("projection.publish", {"path": target, "contents": contents})
            return result
        mp.setattr(Path, "replace", publish)
        original_write_json = storage.write_json
        def write_json(path, data):
            original_write_json(path, data)
            trace.emit("file.write_json", {"path": path, "contents": data})
        for module in list(sys.modules.values()):
            if (getattr(module, "__name__", "").startswith(("shared.", "linkedin."))
                    and getattr(module, "write_json", None) is original_write_json):
                mp.setattr(module, "write_json", write_json)
        original_checkpoint = pipeline.state._allocator_service._checkpoint_progress
        def checkpoint(*args, **kwargs):
            trace.page_commit = bool(kwargs.get("completed_page"))
            try:
                result = original_checkpoint(*args, **kwargs)
                trace.emit("checkpoint", json.loads(pipeline.state.progress_path.read_text()))
                if (trace.page_commit and not trace.injected
                        and spec.get("checkpoint_fault") == "after_checkpoint"):
                    trace.injected = True
                    raise RuntimeError("recorded interruption after canonical checkpoint")
                return result
            finally:
                trace.page_commit = False
        pipeline.state._allocator_service._checkpoint_progress = checkpoint
        original_append = storage.append_jsonl
        def append(path, record):
            original_append(path, record)
            trace.emit("file.append", {"path": path, "record": record})
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith(("shared.", "linkedin.")) and getattr(module, "append_jsonl", None) is original_append:
                mp.setattr(module, "append_jsonl", append)
        root_logger = logging.getLogger()
        mp.setattr(root_logger, "handlers", [logging.StreamHandler(trace)])
        mp.setattr(root_logger, "level", logging.WARNING)
        trace.enabled = True
        with redirect_stdout(trace), redirect_stderr(trace):
            try:
                asyncio.run(facade.run_full(resume=not spec.get("fresh", False)))
            except BaseException as error:
                trace.emit("escape", error)
            else:
                trace.emit("escape", None)
        trace.emit("stats", pipeline.state.context.stats)
        with pipeline.state.context._runtime_state.connect() as connection:
            for table in ("runs", "work_units", "candidates", "candidate_attempts", "side_effects", "events"):
                rows = [dict(row) for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY id')]
                trace.emit("runtime.rows", {"table": table, "rows": rows})
        for path in sorted(root.glob("*.json")):
            trace.emit("file.final", {"path": path, "contents": json.loads(path.read_text())})
    # A module imported inside the recording may have captured our datetime
    # class before MonkeyPatch knew about the module. Do not leak frozen time
    # into later tests in the same Python process.
    for module in list(sys.modules.values()):
        if getattr(module, "datetime", None) is RecordedClock.DateTime:
            module.datetime = REAL_DATETIME
    assert not trace.violations, f"unrecorded external calls: {trace.violations}"
    if spec.get("checkpoint_fault"):
        assert trace.injected, "checkpoint interruption did not fire"
    return trace.effects


async def _recorded_recovery_snapshot(trace, args, kwargs):
    trace.emit("recovery.snapshot", {"args": args, "kwargs": kwargs})
    return "recorded-snapshot"


async def _recorded_recovery(trace, browser, kwargs):
    trace.emit("recovery.restore", kwargs)
    browser.page.url = browser.search_url
    browser.identity = ""
    return True


def assert_recording(expected, actual, *, name):
    expected_text = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    actual_text = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    difference = list(difflib.unified_diff(expected_text, actual_text, fromfile="recorded", tofile="replayed", lineterm=""))
    print(f"{name}: effect diff = {json.dumps(difference)}")
    assert not difference, "evaluation effect transcript changed:\n" + "\n".join(difference[:120])
