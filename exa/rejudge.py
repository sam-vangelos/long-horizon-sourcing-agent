"""Which stored Exa verdicts a resume judges again.

A verdict is made under the criteria the judge was given at the time. When
the brief's criteria change mid-campaign, the people already on the slate
(the short list and the worth-a-look list) still carry verdicts the current
brief may not make, and the slate is what the recruiter opens. So a resume
that opts in (``exa.run --rejudge-stale``, which the SPL launcher passes)
judges those verdicts again before it acquires anyone new, from the stored
record and with no Exa call. Without the opt-in only a row a dead re-judge
left cleared is judged.

"The criteria changed" is read from the judgment fingerprint stamped on the
verdict (``exa.judge.judgment_fingerprint``): it moves when the judge's
prompt moves, and stays put for an edit to the search plan. A verdict
written before the fingerprint existed falls back to the brief file's
content hash, so any edit at all re-judges it once.

A re-judge never moves anyone on one noisy sample. A slate verdict is
judged twice: it leaves the slate only when both samples reject, a split
(one save-type sample, one reject) lands on worth-a-look, and a sample that
failed leaves the stored verdict exactly as it was, to be tried again next
resume (``resolve_samples``). Rejected people are judged again only when
the campaign opts in (the brief loosened its bar in places, so a reject can
become a save); one save-type sample promotes a reject, because a promotion
puts a person in front of a recruiter, who is the second opinion. The
preflight states how many of each there are, so the choice is visible.

Never judged again: a verdict a recruiter recorded (``user_status``, the
same rule the store's suppression clause applies), and a deterministic
gate's rejection, which is not a judgment about criteria (the gates run
again on every judgment and on the slate). A failed judgment never reaches
this pass: it is ``failed_retryable``, and the pending pass retries it.

Pure functions over primitive values, so the pipeline and the read-only
preflight apply one rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple, Sequence

from exa.employer_gate import is_gate_decision
from exa.judge import (
    BUCKET_FAILURE,
    BUCKET_REJECT,
    BUCKET_REVIEW,
    BUCKET_SAVE,
    JUDGMENT_FINGERPRINT_KEY,
    JudgmentOutcome,
    classify_verdict,
)
from exa.location_gate import is_location_gate_decision
from shared.schemas import OpusDecision


# The verdict stands under the current criteria, or is not the pass's to touch.
CURRENT = ""
# A short-list or worth-a-look verdict made under other criteria: re-judged.
STALE = "stale"
# A verdict cleared for a re-judge that never ran (the process died between
# the clear and the new verdict). Not terminal and not pending, so nothing
# else would ever pick it up: re-judged.
CLEARED = "cleared"
# A rejection made under other criteria: re-judged only when the campaign
# opts in, and counted for the preflight either way.
STALE_REJECT = "stale_reject"

# The statuses a resume re-judges with no opt-in beyond the slate pass.
REJUDGE_STATUSES = frozenset({STALE, CLEARED})

# The durable record of a re-judged verdict: the old verdict and the samples,
# written before the store clears the row (the clear empties the payload).
REJUDGE_EVENT = "exa_stale_verdict_rejudged"
# A re-judge that left the stored verdict as it was, because a sample failed.
REJUDGE_KEPT_EVENT = "exa_stale_verdict_rejudge_kept"

# The terminal-payload key a split verdict carries its samples under. The
# stored decision is the save-type sample's (the judge's own words); this
# key is what moves it to worth-a-look, the way the scope check does.
REJUDGE_SPLIT_KEY = "rejudge_split"

# Samples per re-judged verdict, by the bucket it was stored in.
SLATE_SAMPLES = 2
REJECT_SAMPLES = 1


def selected_statuses(*, slate: bool, rejects: bool) -> frozenset[str]:
    """The statuses a resume re-judges under the campaign's opt-ins.

    ``CLEARED`` is always in: a cleared row is a re-judge a dead process
    left half done, holding no verdict at all, and nothing else would ever
    pick it up. ``slate`` adds the short-list and worth-a-look verdicts
    made under earlier criteria; ``rejects`` adds the earlier rejects.
    """

    statuses = {CLEARED}
    if slate:
        statuses.add(STALE)
    if rejects:
        statuses.add(STALE_REJECT)
    return frozenset(statuses)


def samples_for(old_bucket: str) -> int:
    """How many samples a verdict stored in ``old_bucket`` is judged on.

    A reject is judged once: one save-type sample promotes it. Anything
    else (short list, worth-a-look, or a cleared row whose old verdict is
    unknown) is judged twice, because leaving the slate takes two rejects.
    """

    return REJECT_SAMPLES if old_bucket == BUCKET_REJECT else SLATE_SAMPLES


@dataclass(frozen=True)
class Resolution:
    """What one re-judged verdict becomes.

    ``outcome`` is None when the stored verdict stays exactly as it is (a
    sample failed); ``reason`` then says why. ``split`` marks a slate verdict
    whose two samples disagreed: ``outcome`` is the save-type sample, bucketed
    as worth-a-look, and ``samples`` carries both for the payload.
    """

    outcome: JudgmentOutcome | None
    split: bool = False
    reason: str = ""
    samples: tuple[dict, ...] = field(default_factory=tuple)


def resolve_samples(
    old_bucket: str, outcomes: Sequence[JudgmentOutcome]
) -> Resolution:
    """Combine a re-judge's samples into the verdict the store settles.

    * Any failed sample: keep the stored verdict; the next resume tries again.
    * A reject (one sample): that sample, whatever it says.
    * A slate verdict (two samples): two rejects take it off the slate; one
      reject and one save-type sample is a split, worth-a-look; two save-type
      samples settle on the stronger (short list over worth-a-look, then the
      first).
    """

    summary = tuple(sample_summary(outcome) for outcome in outcomes)
    if not outcomes or any(o.bucket == BUCKET_FAILURE for o in outcomes):
        return Resolution(outcome=None, reason="sample_failed", samples=summary)
    if len(outcomes) < samples_for(old_bucket):
        return Resolution(outcome=None, reason="too_few_samples", samples=summary)
    rejects = [o for o in outcomes if o.bucket == BUCKET_REJECT]
    others = [o for o in outcomes if o.bucket != BUCKET_REJECT]
    if not others:
        return Resolution(outcome=rejects[0], samples=summary)
    best = next((o for o in others if o.bucket == BUCKET_SAVE), others[0])
    if rejects:
        return Resolution(
            outcome=JudgmentOutcome(full=best.full, bucket=BUCKET_REVIEW),
            split=True,
            samples=summary,
        )
    return Resolution(outcome=best, samples=summary)


def sample_summary(outcome: JudgmentOutcome) -> dict:
    decision = outcome.full
    return {
        "decision": str(getattr(decision, "decision", "") or ""),
        "bucket": outcome.bucket,
        "confidence": getattr(decision, "confidence", None),
        "reject_reason": str(getattr(decision, "reject_reason", "") or ""),
        "rationale": " ".join(str(getattr(decision, "rationale", "") or "").split()),
    }


def rejudge_status(
    *,
    state: str,
    terminal_decision: str,
    decision: dict | None,
    judgment_fingerprint: str,
    user_status: str,
    brief_content_hash: str,
    current_fingerprint: str,
    current_brief_hash: str,
) -> str:
    """Classify one stored candidate row for the re-judge pass."""

    if user_status:
        return CURRENT
    if state != "full_terminal":
        return CURRENT
    if not terminal_decision:
        return CLEARED
    verdict = _decision(terminal_decision, decision)
    if is_gate_decision(verdict) or is_location_gate_decision(verdict):
        return CURRENT
    bucket = classify_verdict(verdict)
    if judgment_fingerprint and current_fingerprint:
        moved = judgment_fingerprint != current_fingerprint
    else:
        moved = bool(
            brief_content_hash
            and current_brief_hash
            and brief_content_hash != current_brief_hash
        )
    if not moved:
        return CURRENT
    return STALE if bucket in (BUCKET_SAVE, BUCKET_REVIEW) else STALE_REJECT


def fingerprint_from_payload(terminal_payload: Any) -> str:
    if not isinstance(terminal_payload, dict):
        return ""
    return str(terminal_payload.get(JUDGMENT_FINGERPRINT_KEY) or "")


class RejudgeWork(NamedTuple):
    """What a resume could re-judge: slate verdicts, cleared rows, rejects."""

    stale: int
    cleared: int
    stale_rejects: int


def count_rejudge_work(
    rows: Iterable[dict],
    *,
    current_fingerprint: str,
    current_brief_hash: str,
) -> RejudgeWork:
    """Stale slate verdicts, cleared rows and stale rejects over stored rows.

    Each row carries the ``candidates`` columns ``current_lifecycle_state``,
    ``terminal_decision``, ``user_status``, ``brief_content_hash`` and the
    parsed ``terminal_payload`` dict.
    """

    stale = cleared = stale_rejects = 0
    for row in rows:
        payload = row.get("terminal_payload") or {}
        status = rejudge_status(
            state=str(row.get("current_lifecycle_state") or ""),
            terminal_decision=str(row.get("terminal_decision") or ""),
            decision=payload.get("full_decision") if isinstance(payload, dict) else None,
            judgment_fingerprint=fingerprint_from_payload(payload),
            user_status=str(row.get("user_status") or ""),
            brief_content_hash=str(row.get("brief_content_hash") or ""),
            current_fingerprint=current_fingerprint,
            current_brief_hash=current_brief_hash,
        )
        if status == STALE:
            stale += 1
        elif status == CLEARED:
            cleared += 1
        elif status == STALE_REJECT:
            stale_rejects += 1
    return RejudgeWork(stale=stale, cleared=cleared, stale_rejects=stale_rejects)


def _decision(terminal_decision: str, decision: dict | None) -> OpusDecision:
    """The stored decision, or the minimum the terminal column carries.

    Mirrors ``exa.orchestrator._decision_from_entry``: a row written by an
    older path may hold the column without the full payload.
    """

    if isinstance(decision, dict) and decision:
        try:
            return OpusDecision.from_dict(decision)
        except (TypeError, ValueError):
            pass
    return OpusDecision(
        stage="full",
        decision=terminal_decision,
        path="",
        confidence=None,
        rationale="",
        candidate_name="",
        profile_url="",
    )


__all__ = [
    "CLEARED",
    "CURRENT",
    "REJECT_SAMPLES",
    "REJUDGE_EVENT",
    "REJUDGE_KEPT_EVENT",
    "REJUDGE_SPLIT_KEY",
    "REJUDGE_STATUSES",
    "SLATE_SAMPLES",
    "STALE",
    "STALE_REJECT",
    "RejudgeWork",
    "Resolution",
    "count_rejudge_work",
    "fingerprint_from_payload",
    "rejudge_status",
    "resolve_samples",
    "sample_summary",
    "samples_for",
    "selected_statuses",
]
