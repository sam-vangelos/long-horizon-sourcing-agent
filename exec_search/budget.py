"""Per-search dossier-spend circuit-breaker for executive search.

Slice 5 of the executive-search module ships the budget tracker that
gates every dossier-cost-bearing call across the run. Tracks ALL
costs (Perplexity, News API, Crunchbase, PitchBook, per-candidate
Opus exec-register full eval) against a single per-run cap declared
on the brief (``brief.dossier_spend_cap_usd``, default $200).

Two distinct alarms (NOT the same mechanism — see the spec's "Risks"
section for the three-mechanism volume protection):

1. **Hard stop (cost cap):** when accumulated spend across all
   sources + LLM exceeds ``dossier_spend_cap_usd``, every subsequent
   :meth:`reserve` call returns :class:`BudgetExhausted`. The
   orchestrator surfaces a ``BudgetExhausted`` event and returns
   control to the recruiter with the cost-by-source breakdown.

2. **Soft eval-count alarm:** at 50 candidates dossier-evaluated
   (regardless of save outcome), :meth:`reserve` flags
   ``soft_alarm_fired=True`` once. The orchestrator surfaces the
   "this pool is unusually large for an exec search; refine the
   brief?" banner without stopping the run.

Note: the saves-shape alarm (>25 saves) lives in Slice 7's shortlist
surface — different mechanism, different failure mode. Don't
conflate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping


# Default per-run cost cap. The spec's framing: "$200K-$500K per-
# placement value; per-search dossier cost of $50-150 is rounding
# error." $200 is the conservative default; recruiter-overridable
# via ``brief.dossier_spend_cap_usd`` (Slice 5 schema addition).
DEFAULT_DOSSIER_SPEND_CAP_USD: float = 200.0

# Soft alarm threshold: this many candidates dossier-evaluated
# fires the "unusually large pool" banner. Independent of the cost
# cap (cost cap fires on cost; this fires on count).
SOFT_EVAL_COUNT_ALARM_THRESHOLD: int = 50


@dataclass
class BudgetReservation:
    """Successful budget reservation. Caller may proceed with the call."""

    source: str
    cost_usd: float
    accumulated_usd: float
    accumulated_evals: int
    soft_alarm_fired: bool = False


@dataclass
class BudgetExhausted:
    """Hard-stop result. Caller must NOT make the call.

    The orchestrator handles this by surfacing a ``BudgetExhausted``
    event and returning control to the recruiter. The breakdown
    helps the recruiter decide whether to raise the cap or refine
    the brief.
    """

    source: str
    requested_cost_usd: float
    accumulated_usd: float
    cap_usd: float
    by_source: Mapping[str, float] = field(default_factory=dict)


class DossierSpendTracker:
    """Per-run accumulator + circuit-breaker.

    Single-process, single-run scope. Not thread-safe (the dossier
    eval loop is the sole writer; the recruiter resumes via a fresh
    tracker after a ``BudgetExhausted`` event is acknowledged).

    Reserve before the call, not after. The reservation IS the
    permission token. If reserve returns :class:`BudgetExhausted`,
    the caller MUST NOT make the call — it must surface the failure
    upward.
    """

    def __init__(
        self,
        *,
        cap_usd: float = DEFAULT_DOSSIER_SPEND_CAP_USD,
        soft_eval_count_threshold: int = SOFT_EVAL_COUNT_ALARM_THRESHOLD,
    ) -> None:
        self._cap_usd = float(cap_usd)
        self._soft_eval_count_threshold = int(soft_eval_count_threshold)
        self._accumulated_usd: float = 0.0
        self._accumulated_evals: int = 0
        self._by_source: dict[str, float] = {}
        self._soft_alarm_fired_once: bool = False

    @property
    def cap_usd(self) -> float:
        return self._cap_usd

    @property
    def accumulated_usd(self) -> float:
        return self._accumulated_usd

    @property
    def accumulated_evals(self) -> int:
        return self._accumulated_evals

    @property
    def by_source(self) -> Mapping[str, float]:
        return dict(self._by_source)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self._cap_usd - self._accumulated_usd)

    def reserve(
        self,
        *,
        source: str,
        cost_usd: float,
        is_full_eval: bool = False,
    ) -> BudgetReservation | BudgetExhausted:
        """Reserve budget for the next dossier-cost-bearing call.

        ``source`` is the cost source (e.g., ``"perplexity"``,
        ``"news"``, ``"crunchbase"``, ``"pitchbook"``,
        ``"opus_full_eval"``). ``cost_usd`` is the predicted cost of
        the call (callers can refine post-hoc via :meth:`adjust`).

        ``is_full_eval=True`` increments the evaluated-candidate
        counter — used by the soft alarm at the 50-candidate
        threshold. Per-source signal calls do NOT increment this
        counter; only the per-candidate full-eval does.
        """

        cost_usd = max(0.0, float(cost_usd))
        projected = self._accumulated_usd + cost_usd
        if projected > self._cap_usd:
            return BudgetExhausted(
                source=source,
                requested_cost_usd=cost_usd,
                accumulated_usd=self._accumulated_usd,
                cap_usd=self._cap_usd,
                by_source=dict(self._by_source),
            )

        self._accumulated_usd = projected
        self._by_source[source] = self._by_source.get(source, 0.0) + cost_usd
        soft_alarm_fired = False
        if is_full_eval:
            self._accumulated_evals += 1
            if (
                not self._soft_alarm_fired_once
                and self._accumulated_evals >= self._soft_eval_count_threshold
            ):
                self._soft_alarm_fired_once = True
                soft_alarm_fired = True
        return BudgetReservation(
            source=source,
            cost_usd=cost_usd,
            accumulated_usd=self._accumulated_usd,
            accumulated_evals=self._accumulated_evals,
            soft_alarm_fired=soft_alarm_fired,
        )

    def adjust(self, *, source: str, delta_usd: float) -> None:
        """Apply a post-hoc cost correction (e.g., when a call's
        actual cost differs from the predicted reservation).

        Negative ``delta_usd`` reduces accumulated cost. Does not
        un-fire a previously-fired soft alarm — alarms are
        once-per-run and adjusting cost backwards doesn't undo the
        fact that the alarm point was crossed.
        """

        delta_usd = float(delta_usd)
        self._accumulated_usd = max(0.0, self._accumulated_usd + delta_usd)
        self._by_source[source] = max(
            0.0, self._by_source.get(source, 0.0) + delta_usd
        )


# --------------------------------------------------------------------------
# Per-source predicted costs (rough; calibrated from public price
# sheets). The values below feed `reserve()`'s `cost_usd`. Exact
# costs are call-shape-dependent; these are conservative mid-range
# estimates. Slice 5 ships these as constants; later slices may
# replace with usage-API readouts.
# --------------------------------------------------------------------------

# Perplexity: ~$0.005-0.02 per call depending on model + tokens.
PREDICTED_COST_USD_PERPLEXITY: float = 0.02

# News API: free tier exhausts quickly; paid is cheap (~$0.01/call).
PREDICTED_COST_USD_NEWS: float = 0.01

# Crunchbase: enterprise tier; per-call cost depends on contract.
# Using $0.10/call as a placeholder — the real number is contract-
# specific and the recruiter overrides via brief if needed.
PREDICTED_COST_USD_CRUNCHBASE: float = 0.10

# PitchBook: same posture as Crunchbase, often more expensive.
PREDICTED_COST_USD_PITCHBOOK: float = 0.20

# Opus exec-register full eval: ~$1.00-1.50 per candidate (cached
# system prompt + ~3-5K input tokens + ~1-2K output tokens).
PREDICTED_COST_USD_OPUS_FULL_EVAL: float = 1.20


def predicted_cost_for(source: str) -> float:
    """Look up the predicted cost for a known cost source.

    Returns ``0.0`` for unknown sources (callers can pass any name;
    cost-tracking is best-effort against the cap).
    """

    return {
        "perplexity": PREDICTED_COST_USD_PERPLEXITY,
        "news": PREDICTED_COST_USD_NEWS,
        "crunchbase": PREDICTED_COST_USD_CRUNCHBASE,
        "pitchbook": PREDICTED_COST_USD_PITCHBOOK,
        "opus_full_eval": PREDICTED_COST_USD_OPUS_FULL_EVAL,
    }.get(source, 0.0)


__all__ = (
    "BudgetExhausted",
    "BudgetReservation",
    "DEFAULT_DOSSIER_SPEND_CAP_USD",
    "DossierSpendTracker",
    "PREDICTED_COST_USD_CRUNCHBASE",
    "PREDICTED_COST_USD_NEWS",
    "PREDICTED_COST_USD_OPUS_FULL_EVAL",
    "PREDICTED_COST_USD_PERPLEXITY",
    "PREDICTED_COST_USD_PITCHBOOK",
    "SOFT_EVAL_COUNT_ALARM_THRESHOLD",
    "predicted_cost_for",
)
