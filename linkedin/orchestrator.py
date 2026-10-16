"""Compatibility entry point for the LinkedIn day cycle and file rejudging.

The evaluation implementation lives in evaluation_loop. This facade owns
construction and the six data/control accesses used by session_orchestrator.
"""
from __future__ import annotations
from linkedin import judgment_runtime as _judgment_runtime

from shared.schemas import CandidateSnippet, Progress
from shared.governor import SessionGovernor
from linkedin.evaluation_state import EvaluationRunState

def _snippet_is_clearly_compelling(snippet: CandidateSnippet, brief_obj) -> bool:
        """Conservative heuristic for candidates worth opening despite high saturation.

        Structural seniority markers stay in code; role-specific vocabulary
        comes from the brief's calibration mirrors (canonical title/framework
        patterns + per-area key terms) via ``_brief_compelling_terms``. A
        brief carrying no mirrors gets structural signals only — fewer forced
        opens, never a built-in vertical prior.
        """
        text = " ".join(
            part
            for part in [
                snippet.current_title,
                snippet.headline,
                " ".join(snippet.experience_entries[:3]),
            ]
            if part
        ).lower()
        if not text:
            return False

        structural_signals = (
            "executive director",
            "chief architect",
            "principal architect",
            "technology fellow",
            "distinguished engineer",
            "principal scientist",
            "research director",
            "research manager",
        )
        if any(signal in text for signal in structural_signals):
            return True

        brief_terms = _brief_compelling_terms(brief_obj)
        if not brief_terms:
            return False
        leadership_terms = ("head", "chief", "director", "vp", "vice president", "principal", "fellow", "architect")
        return any(term in text for term in brief_terms) and any(
            term in text for term in leadership_terms
        )


def _brief_compelling_terms(brief_obj) -> tuple[str, ...]:
        """Brief-supplied vocabulary the compellingness heuristic may consume.

        Mock-safe by design: non-list attributes (MagicMock briefs in tests,
        legacy briefs without mirrors) contribute nothing.
        """
        terms: list[str] = []
        brief = brief_obj
        for attr in (
            "canonical_title_patterns",
            "canonical_framework_patterns",
            "canonical_broad_patterns",
        ):
            values = getattr(brief, attr, None)
            if isinstance(values, (list, tuple)):
                terms.extend(
                    str(v).strip().lower() for v in values if str(v or "").strip()
                )
        key_terms = getattr(brief, "key_terms_by_area", None)
        if isinstance(key_terms, dict):
            for area_terms in key_terms.values():
                if isinstance(area_terms, (list, tuple)):
                    terms.extend(
                        str(v).strip().lower()
                        for v in area_terms
                        if str(v or "").strip()
                    )
        return tuple(dict.fromkeys(terms))


def _print_session_summary(progress: Progress, stats: dict) -> None:
        """Print running session summary after each string completes."""
        done = sum(1 for s in progress.strings if s.status == "done")
        total = len(progress.strings)
        refined = sum(1 for s in progress.strings if s.notes and "Refined" in s.notes)
        print(f"\n  Session: {done}/{total} strings | "
              f"{stats['snippets_extracted']} evaluated | "
              f"{stats['facial_yes']} facial YES | "
              f"{stats['saved']} saves | "
              f"{stats.get('activity_saturated_preview_skips', 0)} activity-sat skips | "
              f"{refined} refined")


from linkedin.evaluation_loop import (
    EvaluationLoop,
    load_brief,
    GeographyRegimeError,
    PageRenderFailedError,
    PanelRecoveryError,
    TransientPaginationError,
    SurfaceOwnershipLostError,
    ProjectContextMismatchError,
    _PANEL_DOWN_COOLDOWN_SECONDS,
    _PANEL_MOUNT_RELOAD_AT,
    _PANEL_MOUNT_TRIP_AT,
    _STRATEGY_COMPLETE_WORK_UNIT_STATUSES,
    _clear_review_evidence,
    _cost_per_save_usd,
    _is_browser_disconnect_error,
    _is_containable_facial_page_error,
    _is_foreign_project_page,
    _is_recruiter_search_page,
    _persisted_outreach_tier,
    _recruiter_page_project_id,
    _sum_token_cost_log_usd,
    _normalize_title_family,
    _normalize_candidate_name_key,
    _PageReport,
)


class Pipeline:
    """Preserve the day-cycle contract while the loop owns execution."""

    def __init__(self, brief_path: str, output_dir: str | None = None,
                 test_mode: bool = False, input_mode: str = "concurrent",
                 governor: SessionGovernor | None = None):
        import os
        from linkedin.evaluation_loop import EvaluationLoop
        self._loop = EvaluationLoop(EvaluationRunState())
        self._loop.state.surface.budget.enabled = os.getenv(
            "LINKEDIN_PAGE_LOCAL_RECOVERY_ENABLED", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._loop._initialize(brief_path, output_dir, test_mode, input_mode, governor)

    async def run_full(self, resume: bool = False,
                       restart_string_id: int | None = None,
                       restart_string_ids: list[int] | None = None) -> None:
        await self._loop.run_full(resume, restart_string_id, restart_string_ids)

    async def rejudge_from_file(self, snippets_path: str) -> None:
        await self._loop.rejudge_from_file(snippets_path)

    @staticmethod
    def _validate_judgment_runtime_configuration() -> None:
        from linkedin.evaluation_loop import EvaluationLoop
        _judgment_runtime._validate_judgment_runtime_configuration()

    @property
    def stats(self):
        return self._loop.state.context.stats

    @property
    def browser(self):
        return self._loop.state.context.browser

    @property
    def _operator_stop_event(self):
        return self._loop.state.context._operator_stop_event

    @_operator_stop_event.setter
    def _operator_stop_event(self, event):
        self._loop.state.context._operator_stop_event = event

    @property
    def _session_expired(self):
        return self._loop.state.context._session_expired

    @_session_expired.setter
    def _session_expired(self, event):
        self._loop.state.context._session_expired = event

    @property
    def _pause_requested(self):
        return self._loop.state.context._pause_requested

    @_pause_requested.setter
    def _pause_requested(self, event):
        self._loop.state.context._pause_requested = event

    @property
    def _resume_event(self):
        return self._loop.state.context._resume_event

    @_resume_event.setter
    def _resume_event(self, event):
        self._loop.state.context._resume_event = event
