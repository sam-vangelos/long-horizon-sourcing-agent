"""Chief-of-staff agent — Cloris-as-coordinator post-run synthesis layer.

The orchestrator-agent layer described in
``Cloris-Multi-Agent-Thesis.md``. v1 ships ONE responsibility:
cross-source reflection synthesis. After ≥2 sources contribute
candidates in a multi-module run, the chief-of-staff agent reads the
per-source structured signals (candidate counts, save counts, top
lanes per source) plus the existing single editorial briefing and
produces a team-level read for the principal: a synthesis paragraph
in Cloris voice, an independent 0.0..1.0 trust weight per
contributing specialist, and a one-sentence priority for what the
principal should look at first.

Bounded: one extra LLM call per multi-source reflection-session
creation, only when the env var ``CLORIS_CHIEF_OF_STAFF_ENABLED`` is
set and ≥2 sources actually produced candidates this run.
Single-source briefs no-op. Lands inside the existing
``llm_usage_session`` cost log so token usage is captured.

Cascade discipline mirrors
:class:`market_intelligence.briefing_polish.BriefingPolishBackend`
byte-equivalent (six routes converging on a deterministic heuristic
fallback: ``llm_raise``, ``schema_invalid``, ``banned_token``,
``snake_case_token``, ``specialist_weight_invalid``, ``containment_failed``).
When any cascade route fires, today's behavior is preserved and the
synthesis section is omitted from the principal's view.

v1 explicitly defers the dispatch responsibility (slice 3 of the
original plan) and the calibration-consumption responsibility (slice
4) — both are independent, larger architectural surfaces and ship as
their own theses.
"""

from cloris.chief_of_staff.agent import (
    BANNED_BRIEFING_TOKENS,
    ChiefOfStaffAgent,
    ChiefOfStaffSynthesis,
    HeuristicChiefOfStaffSynthesizer,
    SNAKE_CASE_IDENTIFIER_RE,
)
from cloris.chief_of_staff.decision import DispatchPlan, DispatchStep

__all__ = [
    "BANNED_BRIEFING_TOKENS",
    "ChiefOfStaffAgent",
    "ChiefOfStaffSynthesis",
    "DispatchPlan",
    "DispatchStep",
    "HeuristicChiefOfStaffSynthesizer",
    "SNAKE_CASE_IDENTIFIER_RE",
]
