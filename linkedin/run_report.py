"""End-of-run reporting and finalization for LinkedIn runs.

Owns the code that turns a finished `Pipeline` run into:
- `run-report-input.json` (deterministic snapshot fed to the debrief model)
- `run-report.json` / `run-report.md` (structured debrief + markdown render)
- the immutable per-run snapshot under `output/runs/linkedin/...`
- the post-run market-intelligence update

The snapshot builder (`RunReportService._build_run_report_snapshot`) and related
reporting cluster live here; `Pipeline` delegates to `RunReportService`.
"""

from __future__ import annotations

import json
from pathlib import Path
import logging
import re as _re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from typing import TYPE_CHECKING

from shared.run_report_schema import (
    RunDebriefAnalysis,
    StructuredRunReport,
    render_run_report_markdown,
)
from shared.storage import log_event, read_jsonl, write_json


from shared.cost_rollup import (
    _cost_per_save_usd,
    _sum_token_cost_log_usd,
)
from shared.judger import SAVE_FAMILY_DECISIONS
from shared.llm_usage import resolve_cost_log_run_id
from shared.schemas import CandidateSnippet, OpusDecision, Progress, SearchString
from shared.search_memory import build_search_memory_summary
from shared.storage import read_json, read_jsonl

if TYPE_CHECKING:
    from shared.bias_controls import BiasMonitor
    from shared.brief_loader import Brief
    from shared.runtime_state import LinkedInRuntimeStateBridge
    from linkedin.search_intelligence import LinkedInExperimentState


RUN_REPORT_ANALYSIS_SYSTEM = """You are a senior sourcing strategist analyzing an end-of-run sourcing snapshot.

Return valid JSON only with these exact top-level keys:
- winning_lanes
- underperforming_lanes
- coverage_gaps
- noise_patterns
- saved_candidate_patterns
- adaptation_assessment
- recommendations
- brief_iteration_hints

Rules:
- Do NOT re-state run_metadata, metrics_summary, or string_performance; those are deterministic and already captured.
- Use only evidence available in the snapshot.
- Cite concrete strings, candidates, and patterns when possible.
- Keep lists concise and high-signal.
- ACTUATOR CONDITIONING (critical): string_performance entries carry
  "ran_as_keyword_fallback" and a "surface_receipt". A string whose structured
  filters did not land physically ran as keyword-only — its performance is NOT
  evidence about the filter-bounded lane it was designed for. When discussing any
  such string or its lane, label it "ran as keyword-only fallback" and do not
  recommend retiring or promoting a filter-bounded lane based on strings whose
  facets never applied.
- string_performance entries may include nested search_intelligence summaries; use them when assessing whether rescues, experiments, and exploitation decisions actually helped.
- brief_iteration_hints may only suggest mutable brief fields:
  instructions, search_priorities, additional_search_terms, intake_notes, depth_distinction,
  non_fit_patterns, minimum_bar_description, facial_calibration, employer_signal_rules,
  calibration_examples, notes, version.
- Do NOT suggest changes to geography, minimum_years_experience, role identity, LinkedIn project mapping, capability areas, or market density.
- If suggesting employer signal rules, keep save_on_employer_alone false.
- If suggesting facial calibration changes, keep them modest and explicitly evidence-based.

Expected inner shapes:
- winning_lanes: [{"lane","string_ids","candidate_examples","evidence","why_it_worked","recommended_action"}]
- underperforming_lanes: [{"lane","string_ids","issue","evidence","recommended_action"}]
- coverage_gaps: [{"gap","why_it_matters","suggested_search_strategy"}]
- noise_patterns: [{"pattern","evidence","mitigation"}]
- saved_candidate_patterns: {
    "standout_candidates": [{"name","why"}],
    "common_employers": [{"employer","count","note"}],
    "common_titles": [{"title_family","count","note"}],
    "archetype_distribution": [{"archetype","count","note"}],
    "seniority_notes": ["..."]
  }
- adaptation_assessment: {
    "summary": "string",
    "effective_refinements": ["..."],
    "questionable_or_skipped": ["..."],
    "operational_notes": ["..."]
  }
- recommendations: {
    "try_next": ["..."],
    "avoid_next": ["..."],
    "prioritize_pipeline": ["..."]
  }
- brief_iteration_hints: {
    "instructions": ["..."],
    "search_priorities": ["..."],
    "additional_search_terms": ["..."],
    "intake_notes": "string",
    "depth_distinction": {"builder_definition","user_definition","edge_case_guidance"},
    "non_fit_patterns": [{"label","description","why_not","examples"}],
    "minimum_bar_description": "string",
    "facial_calibration": {
      "expected_yes_rate_low": 0.0,
      "expected_yes_rate_high": 0.0,
      "fast_exit_patterns": ["..."],
      "trajectory_yes_patterns": ["..."],
      "trajectory_ambiguous_patterns": ["..."],
      "trajectory_no_patterns": ["..."]
    },
    "employer_signal_rules": [{"tier","employer_patterns","evidence_required","save_on_employer_alone"}],
    "calibration_examples": {
      "strong_saves": [{"name","why"}],
      "incorrect_saves": [{"name","why"}],
      "borderline_verify": [{"name","why"}]
    },
    "notes": "string",
    "locked_field_cautions": ["..."]
  }"""


def normalize_text_for_report(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def bias_summary_for_report(bias_monitor: "BiasMonitor | None") -> str:
    """Format bias monitor summary for injection into the run report prompt."""
    if not bias_monitor:
        return ""
    summary = bias_monitor.session_summary()
    if summary.get("total_decisions", 0) == 0:
        return ""
    lines = [
        "",
        "## Bias Monitor Metrics",
        f"- Facial YES rate: {summary.get('facial_yes_rate', 0):.1%}",
        f"- Full save rate: {summary.get('save_rate', 0):.1%}",
        f"- Parse failures: {summary.get('parse_failures', 0)} ({summary.get('parse_failure_rate', 0):.1%})",
        f"- Alerts fired: {len(summary.get('alerts_fired', []))}",
    ]
    # Telemetry demotion (2026-07-04): render the actual fired signals from
    # the monitor's persisted Alert payloads — ONE definition of "high save
    # density", the check's own — instead of this renderer's former parallel
    # >0.5/≥5 heuristic that could disagree with the alarms it summarized.
    fired = getattr(bias_monitor, "fired_alert_records", None) or []
    if fired:
        lines.append("- Bias signals fired:")
        for record in fired:
            lines.append(
                f"  - [{record.get('severity', '?')}] {record.get('alert_type', '?')}"
                f" (string {record.get('string_id', '?')}): "
                f"{normalize_text_for_report(record.get('message', ''))}"
            )
    return "\n".join(lines) + "\n"


def load_run_report_decisions(
    final_path: Path,
    decision_filter: set[str],
    limit: int = 20,
) -> list[dict]:
    """Load a compact set of final-judgment examples for debrief generation."""
    if not final_path.exists():
        return []
    records: list[dict] = []
    try:
        for row in read_jsonl(final_path):
            if not isinstance(row, dict):
                continue
            if row.get("decision") not in decision_filter:
                continue
            records.append(
                {
                    "candidate_name": row.get("candidate_name", ""),
                    "decision": row.get("decision", ""),
                    "path": row.get("path", ""),
                    "confidence": row.get("confidence", 0.0),
                    "rationale": normalize_text_for_report(row.get("rationale", ""))[:280],
                }
            )
            if len(records) >= limit:
                break
    except Exception:
        return []
    return records


def generate_run_report(
    *,
    snapshot: dict,
    output_dir: Path,
    log_path: Path,
) -> None:
    """Run the model-backed debrief and write run-report artifacts.

    Errors are logged to stdout as warnings and swallowed — report generation
    must never fail a run.
    """
    from shared.llm_clients import opus_llm
    from shared import config

    report_input_path = output_dir / "run-report-input.json"
    report_json_path = output_dir / "run-report.json"
    report_md_path = output_dir / "run-report.md"

    try:
        write_json(report_input_path, snapshot)
        print(f"\n{'=' * 60}")
        print(
            "  Generating end-of-run debrief report "
            f"({config.FULL_EVAL_MODEL_NAME.rsplit('/', 1)[-1]})..."
        )
        print(f"{'=' * 60}")
        run_metadata = snapshot.get("run_metadata") or {}
        usage_context = {
            "stage": "linkedin_run_report_debrief",
            "source": "linkedin",
            "brief_id": run_metadata.get("brief_id"),
            "run_id": run_metadata.get("run_id"),
        }
        analysis_raw = opus_llm(
            RUN_REPORT_ANALYSIS_SYSTEM,
            json.dumps(snapshot, indent=2),
            expect_json=True,
            # 24576 ~= 12K output + reasoning headroom; 420s ~= 24576/70 tok/s + margin.
            max_tokens=24576,
            timeout_seconds=420,
            usage_context=usage_context,
            model_name=config.FULL_EVAL_MODEL_NAME,
        )
        analysis = RunDebriefAnalysis.from_dict(analysis_raw)
        report = StructuredRunReport.from_parts(snapshot, analysis)
        write_json(report_json_path, report.to_dict())
        markdown = render_run_report_markdown(report)
        report_md_path.write_text(markdown)
        print(f"\n{markdown}")
        print(f"\n  Report input saved to: {report_input_path}")
        print(f"  Report JSON saved to:  {report_json_path}")
        print(f"  Report saved to:       {report_md_path}")
        log_event(
            log_path,
            "run_report_generated",
            report_input_path=str(report_input_path),
            report_json_path=str(report_json_path),
            report_path=str(report_md_path),
        )
    except Exception as e:
        print(f"  [warn] Report generation failed: {e}")


def freeze_linkedin_run_snapshot(
    *,
    runtime_run_id: int | None,
    brief_path: str,
    state_dir: Path,
    log_path: Path,
) -> Path | None:
    """Freeze the current state directory into an immutable run snapshot."""
    if not runtime_run_id:
        return None
    try:
        from market_intelligence.run_snapshots import finalize_run_snapshot

        run_dir = finalize_run_snapshot(
            source="linkedin",
            brief_path=brief_path,
            state_dir=state_dir,
            run_id=int(runtime_run_id),
        )
        print(f"  Run snapshot saved to:  {run_dir}")
        log_event(
            log_path,
            "run_snapshot_finalized",
            run_id=int(runtime_run_id),
            run_dir=str(run_dir),
        )
        return run_dir
    except Exception as exc:
        print(f"  [warn] Run snapshot finalization failed: {exc}")
        return None


def enrich_linkedin_run_snapshot(
    *,
    runtime_run_id: int | None,
    brief_path: str,
    run_dir: Path,
    log_path: Path,
) -> None:
    """Update market intelligence from an already-frozen run snapshot."""
    if not runtime_run_id:
        return
    try:
        from market_intelligence.engine import (
            resolve_market_intel_artifact_path,
            update_market_intel,
        )
        from market_intelligence import HeuristicPlannerBackend

        external_backend = None
        try:
            from market_intelligence.research_agent import (
                build_external_research_backend,
            )

            external_backend = build_external_research_backend()
        except Exception:
            pass

        artifact = update_market_intel(
            brief_path=brief_path,
            run_dir=run_dir,
            mode="post_run",
            external_research_backend=external_backend,
            planner_backend=HeuristicPlannerBackend(),
            with_external_research=external_backend is not None,
        )
        artifact_path = resolve_market_intel_artifact_path(
            brief_path,
            output_dir=run_dir,
        )
        stages_degraded = sorted(set(artifact.stages_degraded or []))
        external_sources_count = len(
            (artifact.evidence_index or {}).get("external_sources", []) or []
        )
        status = "degraded" if stages_degraded else "ok"
        status_detail = (
            f"degraded={','.join(stages_degraded)}"
            if stages_degraded
            else "status=ok"
        )
        print(
            "  Market intel updated "
            f"({status_detail}; external_sources={external_sources_count}): "
            f"{artifact_path}"
        )
        log_event(
            log_path,
            "market_intel_updated",
            run_id=int(runtime_run_id),
            run_dir=str(run_dir),
            market_key=artifact.market_identity.market_key,
            status=status,
            stages_degraded=stages_degraded,
            external_sources_count=external_sources_count,
        )
    except Exception as exc:
        print(f"  [warn] Market intel update failed: {exc}")
        try:
            log_event(
                log_path,
                "market_intel_update_failed",
                run_id=int(runtime_run_id),
                run_dir=str(run_dir),
                error=str(exc),
            )
        except Exception:
            pass

logger = logging.getLogger(__name__)

_PARENS_SUFFIX = _re.compile(r"\s*\([^)]*\)")
_NON_ALNUM = _re.compile(r"[^a-z0-9]+")


def _normalize_candidate_name_key(name: str) -> str:
    """Collapse minor punctuation/parenthetical variants when matching saved profiles."""
    t = _PARENS_SUFFIX.sub("", (name or "").lower())
    t = t.replace(".", " ")
    t = _NON_ALNUM.sub(" ", t)
    return " ".join(t.split())

@dataclass(frozen=True)
class RunReportDeps:
    get_brief_obj: Callable[[], "Brief"]
    brief_path: str
    output_dir: Path
    final_path: Path
    log_path: Path
    profiles_path: Path
    get_runtime_db_path: Callable[[], Path]
    stats: dict[str, Any]
    get_search_memory: Callable[[], Any]
    get_constraint_manifest: Callable[[], Any]
    get_experiment_states: Callable[[], dict[int, "LinkedInExperimentState"]]
    get_runtime_bridge: Callable[[], "LinkedInRuntimeStateBridge"]
    get_runtime_run_id: Callable[[], int | None]
    get_session_geography_receipt: Callable[[], Any]
    get_bias_monitor: Callable[[], "BiasMonitor | None"]
    get_lint_blocked_strings: Callable[[], list[dict]]
    _adaptation_roi_summary: Callable[..., dict]
    _shadow_cache_hit_rate: Callable[..., float | None]
    _string_has_seniority_contamination: Callable[..., bool]


class RunReportService:
    """Owns LinkedIn end-of-run reporting snapshot and summary behavior."""

    def __init__(self, deps: RunReportDeps):
        self.deps = deps

    def _load_profile_index_for_adaptation(self) -> dict[str, dict]:
        """Index saved profile summaries by normalized candidate name."""
        if not self.deps.profiles_path.exists():
            return {}

        index: dict[str, dict] = {}
        for entry in read_jsonl(self.deps.profiles_path):
            key = _normalize_candidate_name_key(entry.get("name", ""))
            if key and key not in index:
                index[key] = entry
        return index

    def _search_intelligence_detail_for_string(self, search_string: SearchString) -> dict[str, Any]:
        state = self.deps.get_experiment_states().get(search_string.id)
        if state is None:
            return {}

        summary = state.metrics_summary()
        best_variant = state.best_variant()
        return {
            "mode": summary.get("mode", ""),
            "mutations_used": int(summary.get("mutations_used", 0) or 0),
            "precommit_recovery_attempts_used": int(
                summary.get("precommit_recovery_attempts_used", 0) or 0
            ),
            "drift_attempt_count": int(summary.get("drift_attempt_count", 0) or 0),
            "family_pages_reviewed_total": int(summary.get("family_pages_reviewed_total", 0) or 0),
            "family_signal_total": int(summary.get("family_signal_total", 0) or 0),
            "family_saves_total": int(summary.get("family_saves_total", 0) or 0),
            "family_reviewed_total": int(
                summary.get("family_reviewed_total", 0) or 0
            ),
            "family_outreach_total": int(
                summary.get("family_outreach_total", 0) or 0
            ),
            "family_review_total": int(
                summary.get("family_review_total", 0) or 0
            ),
            "family_reject_total": int(
                summary.get("family_reject_total", 0) or 0
            ),
            "committed_variant_id": summary.get("committed_variant_id"),
            "active_variant_id": summary.get("active_variant_id"),
            "family_outcome_summary": dict(summary.get("family_outcome_summary", {})),
            "drift_rescue_summary": dict(summary.get("drift_rescue_summary", {})),
            "best_variant": {
                "variant_id": best_variant.variant_id,
                "variant_kind": best_variant.variant_kind,
                "score": round(best_variant.score(), 2),
                "result_count": best_variant.result_count,
                "within_target_window": best_variant.within_target_window(),
                "full_reviewed": best_variant.full_reviewed,
                "full_outreach": best_variant.full_outreach,
                "full_review": best_variant.full_review,
                "full_reject": best_variant.full_reject,
            },
        }

    def _search_intelligence_aggregate(
        self,
        strings: list[SearchString],
        *,
        profile_index: dict[str, dict] | None = None,
    ) -> dict[str, Any]:
        family_scores: dict[str, float] = {}
        lane_scores: dict[str, float] = {}
        strings_with_precommit_experiments: list[int] = []
        strings_with_drift_rescue_attempts: list[int] = []
        strings_rescued_by_drift: list[int] = []
        productive_string_ids: list[int] = []
        dead_family_candidates: list[str] = []
        contaminated_string_ids: list[int] = []
        contaminated_family_keys: list[str] = []
        contaminated_domain_lanes: list[str] = []

        for search_string in strings:
            detail = self._search_intelligence_detail_for_string(search_string)
            if not detail:
                continue

            precommit_recovery_attempts_used = int(
                detail.get("precommit_recovery_attempts_used", 0) or 0
            )
            if precommit_recovery_attempts_used > 0:
                strings_with_precommit_experiments.append(search_string.id)

            drift_attempt_count = int(detail.get("drift_attempt_count", 0) or 0)
            if drift_attempt_count > 0:
                strings_with_drift_rescue_attempts.append(search_string.id)

            drift_summary = detail.get("drift_rescue_summary", {}) or {}
            rescued = drift_summary.get("outcome") in {"rescued", "signal_returned"}
            contaminated = self.deps._string_has_seniority_contamination(
                search_string,
                profile_index=profile_index,
            )
            if contaminated:
                contaminated_string_ids.append(search_string.id)
                if search_string.family_key and search_string.family_key not in contaminated_family_keys:
                    contaminated_family_keys.append(search_string.family_key)
                if search_string.domain_lane and search_string.domain_lane not in contaminated_domain_lanes:
                    contaminated_domain_lanes.append(search_string.domain_lane)
            if rescued:
                strings_rescued_by_drift.append(search_string.id)

            family_outreach_total = int(
                detail.get("family_outreach_total", 0) or 0
            )
            family_review_total = int(detail.get("family_review_total", 0) or 0)
            proven = family_outreach_total > 0 and not contaminated
            if proven:
                productive_string_ids.append(search_string.id)
                score = float(
                    family_outreach_total * 20
                    + family_review_total * 2
                    + (6 if rescued else 0)
                )
                if search_string.family_key:
                    family_scores[search_string.family_key] = max(
                        family_scores.get(search_string.family_key, 0.0),
                        score,
                    )
                if search_string.domain_lane:
                    lane_scores[search_string.domain_lane] = max(
                        lane_scores.get(search_string.domain_lane, 0.0),
                        score,
                    )
            elif (
                family_outreach_total == 0
                and family_review_total == 0
                and search_string.pages_reviewed >= 1
                and search_string.family_key
                and search_string.family_key not in dead_family_candidates
            ):
                dead_family_candidates.append(search_string.family_key)

        proven_family_keys = [
            family_key
            for family_key, _score in sorted(
                family_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
        ]
        proven_domain_lanes = [
            lane
            for lane, _score in sorted(
                lane_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
        ]
        dead_family_keys = [
            family_key
            for family_key in dead_family_candidates
            if family_key not in family_scores
        ]

        return {
            "strings_with_precommit_experiments": strings_with_precommit_experiments,
            "strings_with_drift_rescue_attempts": strings_with_drift_rescue_attempts,
            "strings_rescued_by_drift": strings_rescued_by_drift,
            "productive_string_ids": productive_string_ids,
            "proven_family_keys": proven_family_keys,
            "proven_domain_lanes": proven_domain_lanes,
            "dead_family_keys": dead_family_keys,
            "contaminated_string_ids": contaminated_string_ids,
            "contaminated_family_keys": contaminated_family_keys,
            "contaminated_domain_lanes": contaminated_domain_lanes,
        }

    @staticmethod
    def _facial_rate_metrics(stats: dict[str, Any]) -> dict[str, Any]:
        """Return one non-overlapping run-level facial funnel denominator.

        ``facial_yes_rate`` historically meant the rate opened for full
        review, because BORDERLINE was aliased to YES.  Keep that field as a
        compatibility alias while exposing the literal YES and BORDERLINE
        classes separately.
        """
        strict_yes = max(0, int(stats.get("facial_yes", 0) or 0))
        borderline = max(0, int(stats.get("facial_borderline", 0) or 0))
        facial_no = max(0, int(stats.get("facial_no", 0) or 0))
        open_count = strict_yes + borderline
        denominator = open_count + facial_no

        def _rate(count: int) -> float:
            return round(count / denominator, 4) if denominator else 0.0

        open_rate = _rate(open_count)
        return {
            "facial_strict_yes_count": strict_yes,
            "facial_borderline_count": borderline,
            "facial_no_count": facial_no,
            "facial_open_count": open_count,
            "facial_rate_denominator_count": denominator,
            "facial_rate_denominator_semantic": (
                "FACIAL_YES+FACIAL_BORDERLINE+FACIAL_NO"
            ),
            "facial_strict_yes_rate": _rate(strict_yes),
            "facial_borderline_rate": _rate(borderline),
            "facial_open_rate": open_rate,
            # Compatibility alias: authored calibration bands and older
            # consumers treat this as opens-for-full-review, not literal YES.
            "facial_yes_rate": open_rate,
        }

    def _bias_summary_for_report(self) -> str:
        return bias_summary_for_report(self.deps.get_bias_monitor())

    def _load_run_report_decisions(
        self,
        decision_filter: set[str],
        limit: int = 20,
    ) -> list[dict]:
        return load_run_report_decisions(self.deps.final_path, decision_filter, limit=limit)

    def _cost_summary_for_report(self) -> dict:
        """P4.2: run-level cost + cost-per-save for the run report.

        Excludes shadow-judge spend by ``shadow_stage`` presence — the shadow
        is an evaluation instrument, and folding its cost into the primary
        number would contaminate the baseline the A/B exists to measure.
        Provider identity is not the discriminator after the GLM promotion:
        Fireworks rows can now be primary spend.
        Shadow spend reports separately, per tier, in
        metrics_summary.shadow_facial / metrics_summary.shadow_full.
        """
        log_path = self.deps.output_dir / "token-cost-log.jsonl"
        cost_usd = _sum_token_cost_log_usd(
            log_path,
            run_id=resolve_cost_log_run_id(log_path, self.deps.get_runtime_run_id()),
            exclude_rows_with=("shadow_stage",),
        )
        if cost_usd is None:
            return {"status": "no_cost_data"}
        return {
            "status": "ok",
            "cost_usd": cost_usd,
            # None (not 0.0) when there are no saves to divide by — a rate
            # with an undefined denominator is not "free", it's unknown.
            "cost_per_save_usd": _cost_per_save_usd(cost_usd, self.deps.stats.get("saved", 0)),
        }

    def _run_health_summary(self) -> dict:
        """P4.3.1: wire shared.observability_monitors (green_but_useless,
        judge parse-failure baseline) into run finalization. Fail-soft —
        this is observability, its failure must not affect the report."""
        if not self.deps.get_runtime_run_id() or not Path(self.deps.get_runtime_db_path()).exists():
            return {"status": "no_runtime_state"}
        try:
            from shared.observability_monitors import compute_run_health

            health = compute_run_health(self.deps.get_runtime_db_path(), self.deps.get_runtime_run_id())
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        if health is None:
            return {"status": "run_not_found"}
        return {"status": "ok", **health.to_dict()}

    def _shadow_facial_summary(self) -> dict | None:
        """GLM-5.2 (Fireworks) shadow-judge aggregation for the run report.

        Reads every ``facial_shadow_comparison`` event this run logged
        (shared/judger.py's ``_run_facial_shadow_single`` /
        ``_run_facial_shadow_batch``) and reduces them to the instruments
        the shadow-judge evaluation needs: agreement rate, parse-failure
        rate, yes-rate delta (via the two yes-rate fields), and latency.
        Cost: every shadow call lands its own typed receipt in
        token-cost-log.jsonl (provider="fireworks"). Those rows are
        EXCLUDED from the primary cost_summary (baseline honesty) and
        reported here as ``shadow_cost_usd`` instead.

        Returns ``None`` when there are zero comparisons (flag off for
        this run, or the run predates the shadow seam) so the caller can
        omit the whole ``shadow_facial`` key from ``metrics_summary``
        rather than emit an affirmative-zero block.

        Both event shapes (single: scalar fields, batch: per-candidate
        arrays — see the shadow-hook module docstring in shared/judger.py)
        are flattened to the same per-candidate accounting here, so a run
        that mixes sequential and batch facial judging is counted
        uniformly.

        Rate denominators are deliberately NOT all "comparisons":
        - ``primary_yes_rate`` is over all comparisons (the primary
          verdict always exists by the time the shadow hook runs).
        - ``shadow_parse_failure_rate`` is over comparisons where the
          shadow model actually returned a response (excludes shadow
          transport/timeout errors — a distinct failure class from a
          malformed response).
        - ``shadow_yes_rate`` is over comparisons where the shadow
          produced a genuine (non-parse-failed) decision.
        - ``agreement_rate`` is over comparisons where both sides produced
          a genuine decision (``agrees`` is not None).
        Any rate whose denominator is 0 is None, not a fabricated 0.0.
        """
        total = 0
        error_count = 0
        parse_failure_count = 0
        agree_count = 0
        comparable_count = 0
        primary_yes_count = 0
        shadow_yes_count = 0
        shadow_valid_count = 0
        latencies: list[float] = []
        model = ""

        for record in read_jsonl(self.deps.log_path):
            if not isinstance(record, dict) or record.get("event") != "facial_shadow_comparison":
                continue
            count = record.get("candidate_count")
            count = int(count) if isinstance(count, (int, float)) else 0
            if count <= 0:
                continue
            total += count
            model = str(record.get("shadow_model") or model)
            latency = record.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))

            is_batch = bool(record.get("batch"))
            if is_batch:
                primary_decisions = list(record.get("primary_decisions") or [])
                shadow_decisions = list(record.get("shadow_decisions") or [])
                agrees_list = list(record.get("agrees") or [])
                parse_failed_list = list(record.get("shadow_parse_failed") or [])
            else:
                primary_decisions = [record.get("primary_decision")]
                shadow_decisions = [record.get("shadow_decision")]
                agrees_list = [record.get("agrees")]
                parse_failed_list = [record.get("shadow_parse_failed")]

            shadow_errored = bool(record.get("shadow_error"))
            if shadow_errored:
                error_count += count

            for idx in range(count):
                primary_decision = primary_decisions[idx] if idx < len(primary_decisions) else None
                if primary_decision == "FACIAL_YES":
                    primary_yes_count += 1

                if shadow_errored:
                    continue  # no shadow response at all for this candidate

                parse_failed = bool(parse_failed_list[idx]) if idx < len(parse_failed_list) else False
                if parse_failed:
                    parse_failure_count += 1
                else:
                    shadow_decision = shadow_decisions[idx] if idx < len(shadow_decisions) else None
                    if shadow_decision is not None:
                        shadow_valid_count += 1
                        if shadow_decision == "FACIAL_YES":
                            shadow_yes_count += 1

                agrees = agrees_list[idx] if idx < len(agrees_list) else None
                if agrees is not None:
                    comparable_count += 1
                    if agrees:
                        agree_count += 1

        if total == 0:
            return None

        responded = total - error_count
        summary = {
            "model": model,
            "comparisons": total,
            "agreement_rate": (
                round(agree_count / comparable_count, 4) if comparable_count else None
            ),
            "shadow_parse_failure_rate": (
                round(parse_failure_count / responded, 4) if responded else None
            ),
            "primary_yes_rate": round(primary_yes_count / total, 4),
            "shadow_yes_rate": (
                round(shadow_yes_count / shadow_valid_count, 4) if shadow_valid_count else None
            ),
            "mean_latency_ms": (
                round(sum(latencies) / len(latencies), 1) if latencies else None
            ),
            # Shadow spend lives HERE, not in cost_summary — the primary run
            # cost excludes provider="fireworks" rows so the A/B baseline
            # stays clean (Opus-review finding). None when no shadow row
            # priced (never an affirmative 0.0). field_equals scopes this
            # to facial-tier rows only — full-eval shadow rows share
            # provider="fireworks" and report their own spend separately
            # in metrics_summary.shadow_full (see _shadow_full_summary).
            "shadow_cost_usd": _sum_token_cost_log_usd(
                self.deps.output_dir / "token-cost-log.jsonl",
                run_id=resolve_cost_log_run_id(
                    self.deps.output_dir / "token-cost-log.jsonl",
                    self.deps.get_runtime_run_id(),
                ),
                provider_filter="fireworks",
                field_equals={"shadow_stage": "facial_shadow"},
            ),
        }
        cache_hit_rate = self.deps._shadow_cache_hit_rate(shadow_stage="facial_shadow")
        if cache_hit_rate is not None:
            summary["mean_cache_hit_rate"] = cache_hit_rate
        return summary

    def _shadow_full_summary(self) -> dict | None:
        """GLM-5.2 (Fireworks) shadow-judge aggregation for FULL-EVAL.

        Sibling to ``_shadow_facial_summary`` — reads every
        ``full_shadow_comparison`` event (shared/judger.py's
        ``_run_full_shadow_single``, LinkedIn V2-structural ``full_judge``
        only). Unlike facial, there is no batch event shape: ``full_judge``
        has no batch call path, so every event is a singleton (one
        candidate each).

        SAMPLING-BIAS CAVEAT (Opus-review finding — read the agreement rate
        with this in mind): ``full_judge_with_external_evidence`` — the
        enriched RE-judgment fired for triggered/borderline candidates —
        carries no shadow hook. The comparison sample therefore covers
        every candidate's INITIAL full verdict but systematically excludes
        the hardest re-judged sub-population, biasing agreement upward.
        A perfect agreement rate here is necessary, not sufficient, for a
        promote decision; hook the enriched path before trusting it on
        borderline-heavy briefs.

        ``agreement_rate`` is on the SAVE-family-vs-REJECT axis
        (``shared.judger._full_decision_axis``): a SAVE vs
        INFERENTIAL_SAVE mismatch on the raw decision strings still counts
        as agreement on this axis. A comparison where either side lands on
        REVIEW_INFERRED / REVIEW_FLAGGED / a parse failure is not
        classifiable on the axis (``agrees`` is None there) and does not
        enter ``agreement_rate``'s denominator, same "not comparable"
        semantics as a facial shadow parse failure.

        ``primary_save_rate`` / ``shadow_save_rate`` are the save-family
        rate (SAVE_FAMILY_DECISIONS) over, respectively, all comparisons
        and all non-parse-failed/non-errored shadow responses.

        Returns ``None`` when there are zero comparisons (flag off for
        this run, or the run predates the full-eval shadow extension).
        """
        total = 0
        error_count = 0
        parse_failure_count = 0
        agree_count = 0
        comparable_count = 0
        primary_save_count = 0
        shadow_save_count = 0
        shadow_valid_count = 0
        latencies: list[float] = []
        model = ""

        for record in read_jsonl(self.deps.log_path):
            if not isinstance(record, dict) or record.get("event") != "full_shadow_comparison":
                continue
            total += 1
            model = str(record.get("shadow_model") or model)
            latency = record.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))

            primary_decision = record.get("primary_decision")
            if primary_decision in SAVE_FAMILY_DECISIONS:
                primary_save_count += 1

            shadow_errored = bool(record.get("shadow_error"))
            if shadow_errored:
                error_count += 1
            else:
                shadow_parse_failed = bool(record.get("shadow_parse_failed"))
                if shadow_parse_failed:
                    parse_failure_count += 1
                else:
                    shadow_decision = record.get("shadow_decision")
                    if shadow_decision is not None:
                        shadow_valid_count += 1
                        if shadow_decision in SAVE_FAMILY_DECISIONS:
                            shadow_save_count += 1

            agrees = record.get("agrees")
            if agrees is not None:
                comparable_count += 1
                if agrees:
                    agree_count += 1

        if total == 0:
            return None

        responded = total - error_count
        summary = {
            "model": model,
            "comparisons": total,
            "agreement_rate": (
                round(agree_count / comparable_count, 4) if comparable_count else None
            ),
            "shadow_parse_failure_rate": (
                round(parse_failure_count / responded, 4) if responded else None
            ),
            "primary_save_rate": round(primary_save_count / total, 4),
            "shadow_save_rate": (
                round(shadow_save_count / shadow_valid_count, 4) if shadow_valid_count else None
            ),
            "mean_latency_ms": (
                round(sum(latencies) / len(latencies), 1) if latencies else None
            ),
            "shadow_cost_usd": _sum_token_cost_log_usd(
                self.deps.output_dir / "token-cost-log.jsonl",
                run_id=resolve_cost_log_run_id(
                    self.deps.output_dir / "token-cost-log.jsonl",
                    self.deps.get_runtime_run_id(),
                ),
                provider_filter="fireworks",
                field_equals={"shadow_stage": "full_shadow"},
            ),
        }
        cache_hit_rate = self.deps._shadow_cache_hit_rate(shadow_stage="full_shadow")
        if cache_hit_rate is not None:
            summary["mean_cache_hit_rate"] = cache_hit_rate
        return summary

    def _build_run_report_snapshot(self, progress: Progress) -> dict:
        """Build a deterministic raw snapshot for structured debrief generation."""
        done_count = sum(1 for s in progress.strings if s.status == "done")
        skipped_count = sum(1 for s in progress.strings if s.status == "skipped")
        total_results = sum(s.result_count for s in progress.strings if s.result_count > 0)
        total_pages = sum(s.pages_reviewed for s in progress.strings)
        candidates_evaluated = self.deps.stats["snippets_extracted"]
        overall_save_rate = self.deps.stats["saved"] / max(candidates_evaluated, 1)
        facial_rate_metrics = self._facial_rate_metrics(self.deps.stats)
        search_intelligence_summary = self._search_intelligence_aggregate(
            progress.strings,
            profile_index=self._load_profile_index_for_adaptation(),
        )

        string_performance = []
        lane_buckets: dict[str, dict] = {}
        for s in progress.strings:
            if s.status not in {"done", "skipped"}:
                continue
            save_rate = len(s.saves) / max(s.candidates_count or (s.pages_reviewed * 25), 1)
            # P2.2: a string that requested structured filters but ran
            # keyword-only must be labeled as such wherever its performance
            # is discussed — a lane verdict computed over strings whose
            # facets never landed is contaminated feedback (FM12).
            ran_as_keyword_fallback = bool(
                (s.surface_receipt or {}).get("fell_back_to_keyword")
            )
            perf_entry = {
                "string_id": s.id,
                "name": s.name,
                "status": s.status,
                "result_count": s.result_count,
                "pages_reviewed": s.pages_reviewed,
                "saves": len(s.saves),
                "save_rate": round(save_rate, 4),
                "saved_candidates": s.saves[:10],
                "notes": s.notes or "",
                "facial_yes_count": s.facial_yes_count,
                "facial_no_count": s.facial_no_count,
                "candidates_count": s.candidates_count,
                "duplicates_count": s.duplicates_count,
                "family_key": s.family_key,
                "novelty_bucket": s.novelty_bucket,
                "domain_lane": s.domain_lane,
                "domain_lane_raw": s.domain_lane_raw,
                "undeclared_lane": s.undeclared_lane,
                "seniority_risk": s.seniority_risk,
                "title_bucket_risk": s.title_bucket_risk,
                "opening_eligible": s.opening_eligible,
                "surface_receipt": s.surface_receipt,
                "ran_as_keyword_fallback": ran_as_keyword_fallback,
                "search_intelligence": self._search_intelligence_detail_for_string(s),
            }
            string_performance.append(perf_entry)

            # Accumulate lane execution summary
            lane_key = str(s.lane_id or s.domain_lane or s.family_key or "legacy").strip() or "legacy"
            if lane_key not in lane_buckets:
                lane_buckets[lane_key] = {
                    "lane_id": lane_key,
                    "lane_name": str(s.lane_name or lane_key),
                    "family_keys": [],
                    "acquisition_modes": [],
                    "string_count": 0,
                    "result_count": 0,
                    "pages_reviewed": 0,
                    "candidates_evaluated": 0,
                    "facial_yes": 0,
                    "facial_no": 0,
                    "saves": 0,
                    "save_rate": 0.0,
                    "string_ids": [],
                }
            bucket = lane_buckets[lane_key]
            bucket["string_count"] += 1
            bucket["result_count"] += s.result_count
            bucket["pages_reviewed"] += s.pages_reviewed
            bucket["candidates_evaluated"] += s.candidates_count or 0
            bucket["facial_yes"] += s.facial_yes_count
            bucket["facial_no"] += s.facial_no_count
            bucket["saves"] += len(s.saves)
            bucket["string_ids"].append(s.id)
            fk = str(s.family_key or "").strip()
            if fk and fk not in bucket["family_keys"]:
                bucket["family_keys"].append(fk)
            am = str(s.acquisition_mode or "").strip()
            if am and am not in bucket["acquisition_modes"]:
                bucket["acquisition_modes"].append(am)

        # Finalize lane execution summary
        lane_execution_summary = []
        for bucket in lane_buckets.values():
            total_eval = bucket["candidates_evaluated"]
            bucket["save_rate"] = round(bucket["saves"] / max(total_eval, 1), 4)
            lane_execution_summary.append(bucket)

        # P2.2: structured-filter actuator health, aggregated from the
        # per-string surface receipts (per-dimension value apply rates).
        actuator_requested: dict[str, int] = {}
        actuator_applied: dict[str, int] = {}
        actuator_fallback_strings: list[int] = []
        for s in progress.strings:
            receipt = s.surface_receipt or {}
            if not receipt:
                continue
            for dim, count in (receipt.get("requested_value_counts") or {}).items():
                actuator_requested[str(dim)] = actuator_requested.get(str(dim), 0) + int(count or 0)
            for dim, count in (receipt.get("applied_value_counts") or {}).items():
                actuator_applied[str(dim)] = actuator_applied.get(str(dim), 0) + int(count or 0)
            if receipt.get("fell_back_to_keyword"):
                actuator_fallback_strings.append(s.id)
        structured_filter_actuator = {
            "per_dimension": {
                dim: {
                    "requested": requested,
                    "applied": actuator_applied.get(dim, 0),
                    "apply_rate": round(
                        actuator_applied.get(dim, 0) / requested, 4
                    )
                    if requested
                    else 0.0,
                }
                for dim, requested in sorted(actuator_requested.items())
            },
            "strings_fell_back_to_keyword": actuator_fallback_strings,
        }

        # P1.3: pipeline-save health. Ledger-derived when runtime state is
        # available (authoritative — carries per-reason failures, permanent
        # failures, and retries); stats-derived fallback otherwise.
        attempted_stats = self.deps.stats.get("save_attempts", 0)
        failed_stats = self.deps.stats.get("save_failed", 0)
        pipeline_save_health: dict = {
            "attempted": attempted_stats,
            "succeeded": self.deps.stats.get("saved", 0),
            "already_present": self.deps.stats.get("already_present", 0),
            "failed": failed_stats,
            "failed_permanent": 0,
            "interrupted": 0,
            "retried_from_prior": 0,
            "failed_by_reason": {},
            "failure_rate": round(failed_stats / attempted_stats, 4)
            if attempted_stats
            else 0.0,
            "source": "run_stats",
        }
        if self.deps.get_runtime_bridge() and self.deps.get_runtime_run_id():
            try:
                ledger_health = self.deps.get_runtime_bridge().save_side_effect_health(
                    self.deps.get_runtime_run_id()
                )
                if ledger_health.get("attempted"):
                    ledger_health["source"] = "side_effect_ledger"
                    pipeline_save_health = ledger_health
            except Exception as health_exc:
                print(f"  [warn] save-health aggregation failed: {health_exc}")

        # P3.6: facial calibration closes the loop. The brief authors an
        # expected_yes_rate_low/high band at preflight time
        # (shared/preflight_v2.py:87-88) and nothing ever compared it to
        # what actually happened. Denominator is the three non-overlapping
        # facial verdicts (YES+BORDERLINE+NO), not candidates_evaluated.
        # The authored yes-rate band predates distinct BORDERLINE persistence,
        # so ``actual_yes_rate`` remains a compatibility alias of open rate
        # (YES+BORDERLINE) while literal YES and BORDERLINE are reported too.
        # No affirmative zero when there are no verdicts to rate.
        facial_yes_n = facial_rate_metrics["facial_strict_yes_count"]
        facial_borderline_n = facial_rate_metrics["facial_borderline_count"]
        facial_no_n = facial_rate_metrics["facial_no_count"]
        facial_open_n = facial_rate_metrics["facial_open_count"]
        facial_verdicts = facial_rate_metrics["facial_rate_denominator_count"]
        if facial_verdicts == 0:
            facial_calibration: dict = {"status": "no_facial_verdicts"}
        else:
            actual_yes_rate = facial_rate_metrics["facial_open_rate"]
            calibration_rates = {
                "facial_strict_yes_count": facial_yes_n,
                "facial_borderline_count": facial_borderline_n,
                "facial_no_count": facial_no_n,
                "facial_open_count": facial_open_n,
                "denominator_count": facial_verdicts,
                "denominator_semantic": facial_rate_metrics[
                    "facial_rate_denominator_semantic"
                ],
                "actual_strict_yes_rate": facial_rate_metrics[
                    "facial_strict_yes_rate"
                ],
                "actual_borderline_rate": facial_rate_metrics[
                    "facial_borderline_rate"
                ],
                "actual_open_rate": actual_yes_rate,
                # Compatibility alias for the pre-ternary calibration band.
                "actual_yes_rate": actual_yes_rate,
            }
            fc = None
            if (
                self.deps.get_brief_obj()
                and self.deps.get_brief_obj().has_v2_schema
                and self.deps.get_brief_obj()._new_brief.facial_calibration
            ):
                fc = self.deps.get_brief_obj()._new_brief.facial_calibration
            if fc is None:
                facial_calibration = {
                    "status": "band_not_authored",
                    **calibration_rates,
                }
            else:
                authored_low = fc.expected_yes_rate_low
                authored_high = fc.expected_yes_rate_high
                if authored_low <= actual_yes_rate <= authored_high:
                    deviation_from_band = 0.0
                else:
                    deviation_from_band = round(
                        min(
                            abs(actual_yes_rate - authored_low),
                            abs(actual_yes_rate - authored_high),
                        ),
                        4,
                    )
                out_of_band = deviation_from_band > 0.15
                calibration_drift_warning = False
                if out_of_band:
                    try:
                        from market_intelligence.engine import (
                            resolve_market_intel_artifact_path,
                        )

                        artifact_path = resolve_market_intel_artifact_path(
                            self.deps.brief_path
                        )
                        if artifact_path.exists():
                            prior_artifact = read_json(artifact_path) or {}
                            prior_observed = (
                                prior_artifact.get("facial_calibration_observed") or {}
                            )
                            if (
                                int(
                                    prior_observed.get(
                                        "consecutive_out_of_band_runs", 0
                                    )
                                    or 0
                                )
                                >= 1
                            ):
                                calibration_drift_warning = True
                    except Exception as calib_exc:
                        print(
                            f"  [warn] facial-calibration drift lookup failed: {calib_exc}"
                        )
                facial_calibration = {
                    "status": "ok",
                    **calibration_rates,
                    "authored_low": authored_low,
                    "authored_high": authored_high,
                    # P6 (Wave 3 reader): the band's provenance — "preflight"
                    # (model-derived), "loader_default"/"synthesis_default"
                    # (template rode along), "operator", or "unknown" for
                    # briefs authored before provenance stamping. A default
                    # band mis-calibrates the out-of-band judgment below;
                    # the report must say which kind this was.
                    "band_source": str(getattr(fc, "band_source", "") or "") or "unknown",
                    "deviation_from_band": deviation_from_band,
                    "out_of_band": out_of_band,
                    "calibration_drift_warning": calibration_drift_warning,
                }

        run_metadata = {
            "role_title": self.deps.get_brief_obj().role_title,
            "brief_name": progress.brief_name,
            "brief_version": self.deps.get_brief_obj().raw.get("version", ""),
            "linkedin_project": self.deps.get_brief_obj().linkedin_project,
            "linkedin_project_id": self.deps.get_brief_obj().linkedin_project_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_summary": (
                f"Run covered {done_count} executed strings over {total_pages} pages, "
                f"evaluated {candidates_evaluated} candidates, and saved {self.deps.stats['saved']}."
            ),
        }
        # P9.3: surface machine-authored-and-unreviewed provenance in the
        # report header, only when the brief actually carries the stamp
        # (preflight_v2-generated briefs) — hand-authored / legacy briefs
        # have no "provenance" key at all, so this follows the same
        # only-when-present discipline as the rest of the snapshot's
        # optional blocks (P9.5's brief_loader detects intake-born briefs
        # by absence-en-bloc of calibration fields; this key is the FUTURE
        # positive detector for the preflight_v2 case specifically).
        provenance = self.deps.get_brief_obj().raw.get("provenance")
        if isinstance(provenance, dict):
            run_metadata["provenance"] = provenance
        # The generated brief's open questions ride the report header next
        # to the unreviewed-brief stamp — same only-when-present,
        # isinstance-guarded discipline as the provenance block above.
        confidence_notes = self.deps.get_brief_obj().raw.get("preflight_confidence_notes")
        if isinstance(confidence_notes, str) and confidence_notes.strip():
            run_metadata["preflight_confidence_notes"] = confidence_notes.strip()

        # P3b: the session-geography receipt — "what pool did this run
        # search" as a recorded fact (intended facets, verified applied,
        # reassert count). Only-when-present: briefs without geography and
        # runs that never applied it render nothing.
        if self.deps.get_session_geography_receipt():
            run_metadata["session_geography"] = dict(self.deps.get_session_geography_receipt())

        metrics_summary = {
            "strings_executed": done_count,
            "strings_skipped": skipped_count,
            "total_results": total_results,
            "total_pages_reviewed": total_pages,
            "candidates_evaluated": candidates_evaluated,
            "facial_yes": self.deps.stats["facial_yes"],
            "facial_borderline": self.deps.stats.get("facial_borderline", 0),
            "facial_no": self.deps.stats["facial_no"],
            "saved": self.deps.stats["saved"],
            "rejected": self.deps.stats["rejected"],
            "high_pressure_candidates_seen": self.deps.stats.get("high_pressure_candidates_seen", 0),
            "activity_saturated_preview_skips": self.deps.stats.get("activity_saturated_preview_skips", 0),
            "high_fit_low_novelty_saves": self.deps.stats.get("high_fit_low_novelty_saves", 0),
            # P3a defense-in-depth: saves whose snippet location shared no
            # token with the session geography (WARN telemetry; the
            # fail-closed gate is the enforcement).
            "off_geo_saves": self.deps.stats.get("off_geo_saves", 0),
            "overall_save_rate": round(overall_save_rate, 4),
            **facial_rate_metrics,
            "strings_with_precommit_experiments": len(
                search_intelligence_summary.get("strings_with_precommit_experiments", [])
            ),
            "strings_with_drift_rescue_attempts": len(
                search_intelligence_summary.get("strings_with_drift_rescue_attempts", [])
            ),
            "strings_rescued_by_drift": len(
                search_intelligence_summary.get("strings_rescued_by_drift", [])
            ),
            "proven_family_keys": search_intelligence_summary.get("proven_family_keys", []),
            "proven_domain_lanes": search_intelligence_summary.get("proven_domain_lanes", []),
            "pipeline_save_health": pipeline_save_health,
            "structured_filter_actuator": structured_filter_actuator,
            "facial_calibration": facial_calibration,
            "cost_summary": self._cost_summary_for_report(),
            "run_health": self._run_health_summary(),
            "adaptation_roi": self.deps._adaptation_roi_summary(progress),
        }
        # Shadow comparisons now dispatch async off the judge path; drain the
        # queue before the summaries below read the run log, or trailing
        # comparisons land after the read and undercount the report. Bounded
        # wait: a wedged shadow call must not hold the report hostage. 360s,
        # not 120: shadow calls are allowed SHADOW_LLM_TIMEOUT_SECONDS=300
        # per attempt (GLM legitimately needs ~234s for a full 16K-token
        # generation), so a last-candidate comparison still in flight at
        # report time needs more than its own ceiling to land.
        try:
            from shared.judger import drain_shadow_comparisons

            drain_shadow_comparisons(timeout=360.0)
        except Exception:
            pass
        # Shadow-strategist sibling (item 19): its artifacts are file-only —
        # nothing below reads them — but draining here bounds how far a
        # trailing shadow call can outlive the run. Same bounded-wait
        # posture: a wedged shadow must not hold the report hostage.
        try:
            from shared.strategy_shadow import drain_strategy_shadows

            drain_strategy_shadows(timeout=120.0)
        except Exception:
            pass
        # GLM-5.2 shadow-judge instrumentation: only present when this run
        # actually logged >=1 facial_shadow_comparison event (flag on) —
        # absent otherwise, no affirmative zeros (mirrors the
        # "no_adaptation_events"-style guard used for adaptation_roi, but
        # via key omission since the task shape here has no "status" field).
        shadow_facial_summary = self._shadow_facial_summary()
        if shadow_facial_summary is not None:
            metrics_summary["shadow_facial"] = shadow_facial_summary
        # Full-eval sibling — same only-when-present discipline (>=1
        # full_shadow_comparison event this run), independent of whether
        # the facial block is present (a run could have SHADOW_FACIAL_
        # MODEL_ENABLED on but reach full_judge for zero candidates).
        shadow_full_summary = self._shadow_full_summary()
        if shadow_full_summary is not None:
            metrics_summary["shadow_full"] = shadow_full_summary
        # P5 (Wave 2): strings refused at queue build (error-severity lint
        # findings / ubiquity gate). Only-when-present; in-memory for this
        # session — resumed sessions rebuilt no queue, and the durable trace
        # is the search_string_lint_blocked run-log events.
        if self.deps.get_lint_blocked_strings():
            metrics_summary["lint_blocked"] = list(self.deps.get_lint_blocked_strings())
        # P3b: constraint-ownership manifest + the report-time
        # defer-dimension aggregate (structured controls lanes requested that
        # no LinkedIn actuator supports, from per-string surface receipts) —
        # closing the per-event-visible/pattern-invisible gap (audit R5-F2).
        constraint_manifest = self.deps.get_constraint_manifest()
        if constraint_manifest:
            from shared.constraint_manifest import (
                MANIFEST_FILENAME,
                aggregate_unsupported_dimensions,
            )

            manifest = dict(constraint_manifest)
            manifest["requested_but_unsupported"] = aggregate_unsupported_dimensions(
                progress.strings
            )
            constraint_manifest.clear()
            constraint_manifest.update(manifest)
            try:
                write_json(self.deps.output_dir / MANIFEST_FILENAME, manifest)
            except Exception:
                pass  # report rendering must never fail the run
            metrics_summary["constraint_manifest"] = manifest
        # P7 Stage C (Wave 3): run-over-run family-key stability, computed by
        # update_search_memory in the memory artifact. Only-when-present —
        # first runs and epoch changes carry no comparable prior.
        search_memory = self.deps.get_search_memory()
        key_stability = (search_memory or {}).get("key_stability")
        if key_stability:
            metrics_summary["key_stability"] = dict(key_stability)

        return {
            "schema_version": 1,
            "run_metadata": run_metadata,
            "metrics_summary": metrics_summary,
            "string_performance": string_performance,
            "lane_execution_summary": lane_execution_summary,
            "saved_candidate_summaries": self._load_run_report_decisions(
                SAVE_FAMILY_DECISIONS,
                limit=24,
            ),
            "rejected_candidate_summaries": self._load_run_report_decisions({"REJECT"}, limit=24),
            "bias_monitor_summary": self._bias_summary_for_report().strip(),
            "search_memory_summary": build_search_memory_summary(search_memory)
            if search_memory
            else None,
        }

    def _run_report_analysis_system(self) -> str:
        return RUN_REPORT_ANALYSIS_SYSTEM

# ---------------------------------------------------------------------------
# Page report (structured console output per protocol.md format)
# ---------------------------------------------------------------------------

class _PageReport:
    """Collects per-page data and prints a structured report."""

    def __init__(self, string_id: int, string_name: str, page: int, result_count: int):
        self.string_id = string_id
        self.string_name = string_name
        self.page = page
        self.result_count = result_count
        self.saved: list[tuple[CandidateSnippet, OpusDecision]] = []
        self.skipped_opened: list[tuple[CandidateSnippet, OpusDecision]] = []
        self.skipped_preview: list[tuple[str, str]] = []  # (name, reason)
        # P1.2: judge said SAVE but the pipeline save physically failed.
        self.save_failed: list[tuple[CandidateSnippet, OpusDecision, str]] = []

    def add_saved(self, snippet: CandidateSnippet, decision: OpusDecision):
        self.saved.append((snippet, decision))

    def add_save_failed(
        self, snippet: CandidateSnippet, decision: OpusDecision, reason: str
    ):
        self.save_failed.append((snippet, decision, reason))

    def add_skipped_opened(self, snippet: CandidateSnippet, decision: OpusDecision):
        self.skipped_opened.append((snippet, decision))

    def add_skip_preview(self, name: str, reason: str):
        self.skipped_preview.append((name, reason))

    def print_report(self, running_stats: dict) -> None:
        rc = self.result_count if self.result_count >= 0 else "?"
        print(f"\n  {'─' * 50}")
        print(f"  PAGE REPORT: String #{self.string_id} | {self.string_name} | "
              f"Page {self.page} | {rc} results")
        print(f"  {'─' * 50}")

        if self.saved:
            print(f"\n  SAVED ({len(self.saved)}):")
            for snippet, decision in self.saved:
                print(f"    + {snippet.name} — {snippet.current_title} at {snippet.current_company}")
                confidence_text = (
                    f"{decision.confidence:.2f}"
                    if decision.confidence is not None
                    else "—"
                )
                print(f"      Path: {decision.path} | Confidence: {confidence_text}")
                if decision.novelty_value:
                    print(f"      Novelty: {decision.novelty_value}")
                print(f"      {decision.rationale}")
                if decision.value_rationale:
                    print(f"      {decision.value_rationale}")

        if self.save_failed:
            print(f"\n  SAVE FAILED — judged SAVE but not in pipeline ({len(self.save_failed)}):")
            for snippet, decision, reason in self.save_failed:
                print(f"    ! {snippet.name} — {snippet.current_title} at {snippet.current_company}")
                print(f"      Reason: {reason} (will retry on rediscovery/resume)")

        if self.skipped_opened:
            print(f"\n  SKIPPED — profiles opened ({len(self.skipped_opened)}):")
            for snippet, decision in self.skipped_opened:
                print(f"    - {snippet.name} — {snippet.current_title} at {snippet.current_company}")
                print(f"      {decision.rationale}")
                if decision.value_rationale:
                    print(f"      {decision.value_rationale}")

        if self.skipped_preview:
            # Group by reason category
            groups: dict[str, list[str]] = {}
            for name, reason in self.skipped_preview:
                key = reason.split(":")[0] if ":" in reason else reason
                groups.setdefault(key, []).append(name)

            print(f"\n  Skipped from preview ({len(self.skipped_preview)}):")
            for category, names in groups.items():
                if len(names) <= 3:
                    print(f"    {category}: {', '.join(names)}")
                else:
                    print(f"    {category}: {len(names)} candidates")

        print(f"\n  Running totals — Saved: {running_stats['saved']} | "
              f"Save failed: {running_stats.get('save_failed', 0)} | "
              f"Facial YES: {running_stats['facial_yes']} | "
              f"Activity skips: {running_stats.get('activity_saturated_preview_skips', 0)}")
