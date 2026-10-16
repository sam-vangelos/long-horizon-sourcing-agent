"""Source-neutral V2 full judgment for Exa candidates.

Evaluate each candidate through ``full_judge_with_external_evidence``. The V2 path
already carries the structured decision contract, review family, and brief
calibration blocks; Exa-specific evidence is built by :mod:`exa.evidence`.

There is no facial stage: Exa's retrieved pages contain the career evidence,
while its upstream lane compiler and namesake guard choose and filter results.
This module therefore evaluates the candidate with the full evidence path.
Judgment is serial by construction and nothing here prints.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Callable

from shared.brief_loader import Brief
from shared.contracts import LEGACY_REVIEW_DECISIONS
from shared.judger import full_judge_with_external_evidence, is_failure_decision
from shared.save_destination import SAVE_DECISION_FAMILY
from shared.schemas import OpusDecision

from exa.evidence import build_external_evidence
from exa.schemas import ExaCandidate, ExaSearchQuery


# The short list includes plain SAVE and INFERENTIAL_SAVE. Exa's web evidence
# is sparse in the LinkedIn profile projection, so INFERENTIAL_SAVE remains a
# distinct save classification while TRANSFERABLE_SAVE, SIGNAL_SAVE, and REVIEW stay
# in the "worth a look" bucket. Keep this classification deterministic rather
# than relying on prompt wording.
SHORT_LIST_DECISIONS = frozenset({"SAVE", "INFERENTIAL_SAVE"})

# The three buckets the slate report renders, plus the non-verdict.
BUCKET_SAVE = "save"
BUCKET_REVIEW = "review"
BUCKET_REJECT = "reject"
BUCKET_FAILURE = "failure"

# The key a verdict's terminal payload carries its criteria fingerprint under.
JUDGMENT_FINGERPRINT_KEY = "judgment_fingerprint"

# The validator's reason token rides in PARSE_FAILURE rationale, so it is
# machine text and safe to branch on.
TRANSFERABILITY_CONTRACT_REASON = "missing_non_direct_transferability"
# A validator slip is a dropped or misplaced output field, not a verdict about
# the person. Treat it as a retryable format failure.
CONTRACT_SLIP_MARKER = "JudgmentToolContractError"


@dataclass
class JudgmentOutcome:
    """One candidate's judgment.

    One stage, one decision. ``bucket`` is the report's currency; the
    decision itself carries the rationale, the confidence, and the
    review reason code when there is one.
    """

    full: OpusDecision
    bucket: str


# ---------------------------------------------------------------------------
# Usage context
# ---------------------------------------------------------------------------


def exa_lane_context(
    candidate: ExaCandidate, query: ExaSearchQuery | None = None
) -> dict[str, str]:
    """Attribution keys for the cost/attempt logs.

    Only keys in ``shared.llm_usage._ATTEMPT_CONTEXT_KEYS`` survive the
    write, so the lane and query identifiers ride the canonical names
    (``lane_id``, ``string_id``) rather than Exa-flavored ones. A key
    named ``lane`` or ``query_id`` would be dropped on the floor and the
    spend would be unattributable — the failure mode is silent, which is
    why this is a named function with a test rather than a dict literal
    at the call site.
    """

    return {
        "source": "exa",
        "module": "exa",
        "lane_id": candidate.source_lane or (query.lane if query else "") or "",
        "string_id": str(query.id if query else candidate.source_query_id),
    }


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


def _stamp(decision: OpusDecision, candidate: ExaCandidate) -> OpusDecision:
    """Stamp the record's name and URL onto the returned decision.

    This matters on failure paths, where the decision is manufactured locally
    and otherwise lacks the candidate's identity.
    """

    decision.candidate_name = candidate.name
    decision.profile_url = candidate.linkedin_url or (
        candidate.evidence_urls() or [""]
    )[0]
    return decision


def _is_contract_slip(decision: OpusDecision) -> bool:
    """Return whether a PARSE_FAILURE contains a validator contract slip.

    Contract slips are format errors, not judgments: the validator may reject
    missing or misplaced fields and place its reason token in the rationale.
    Match the machine markers rather than guessing from prose.
    """

    if not is_failure_decision(decision.decision):
        return False
    rationale = decision.rationale or ""
    return CONTRACT_SLIP_MARKER in rationale or TRANSFERABILITY_CONTRACT_REASON in rationale


def exa_full_judge(
    candidate: ExaCandidate,
    brief: Brief,
    *,
    query: ExaSearchQuery | None = None,
) -> OpusDecision:
    """Run full evaluation with retrieved pages as external evidence.

    Retry once only for a validator contract slip. A format failure is a
    missing or misplaced output field, so a second sample can repair it; a
    second failure is returned without another retry. Real parse or judgment
    failures return on the first call.
    """

    summary = candidate.to_profile_summary()
    evidence = build_external_evidence(
        candidate,
        query_firm=query.firm if query else "",
        exclude_terms=query.exclude_terms if query else (),
        firm_domains=query.include_domains if query else (),
    )
    context = exa_lane_context(candidate, query)

    decision = full_judge_with_external_evidence(
        summary, evidence, brief, lane_context=context
    )
    if _is_contract_slip(decision):
        # Same lane context, deliberately. A ``retry_reason`` key would
        # read well here and be dropped on the floor — it is not in
        # ``shared.llm_usage._ATTEMPT_CONTEXT_KEYS`` — and a label that
        # silently does not persist is worse than none. The retry shows
        # up in the cost log as a second call on the same lane and
        # string id, which is true and attributable.
        decision = full_judge_with_external_evidence(
            summary, evidence, brief, lane_context=context
        )
    return _stamp(decision, candidate)


def judgment_fingerprint(brief: Brief) -> str:
    """A hash of what the full judge is told about the role, or "".

    Taken over both full-evaluation system prompts the brief renders
    (the tool contract and the legacy one), so it changes exactly when an
    edit changes what the judge reads: the instructions, the bar, the
    level, the years. An edit to the search plan (firms, lanes, title
    terms, archetypes) leaves it alone. So does an edit to the employer
    and alumni blacklists, which the prompt also renders: both are
    deterministic gates that run before every judgment and again over
    every stored save when the slate is built, so a blacklist edit moves
    the slate without a model call. A resume compares the fingerprint
    with the one stamped on each stored verdict to find the verdicts the
    criteria have moved past.

    "" for a brief without the V2 schema, or when assembly fails; the
    caller then falls back to the brief file's content hash.
    """

    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is None:
        return ""
    try:
        from shared.judgment.templates import (
            assemble_full_evaluation_system,
            assemble_full_evaluation_tool_system,
        )

        criteria = dataclasses.replace(
            new_brief, employer_blacklist=[], alumni_blacklist=[]
        )
        prompts = (
            assemble_full_evaluation_tool_system(criteria),
            assemble_full_evaluation_system(criteria),
        )
    except Exception:  # noqa: BLE001 — a missing fingerprint degrades to the hash
        return ""
    digest = hashlib.sha256("\x00".join(prompts).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------


def classify_verdict(decision: OpusDecision) -> str:
    """Map a decision onto the slate's buckets.

    Short list, worth a look, dropped — plus ``failure``, which is not a
    verdict about the person at all and must never be counted as one.
    Failure is tested first for exactly that reason: a decision that did
    not happen cannot be a rejection.
    """

    verdict = decision.decision
    if is_failure_decision(verdict):
        return BUCKET_FAILURE
    if verdict in SHORT_LIST_DECISIONS:
        return BUCKET_SAVE
    # The rest of the save family is a hedged save; on Exa evidence it is
    # a lead to look at, not one to put on the short list.
    if verdict in SAVE_DECISION_FAMILY or verdict in LEGACY_REVIEW_DECISIONS:
        return BUCKET_REVIEW
    return BUCKET_REJECT


def judge_candidate(
    candidate: ExaCandidate,
    brief: Brief,
    *,
    query: ExaSearchQuery | None = None,
    full: Callable[..., OpusDecision] = exa_full_judge,
) -> JudgmentOutcome:
    """Judge one candidate and bucket the verdict.

    Let budget-exhaustion errors propagate so the caller pauses the run instead
    of converting a dead-key condition into a candidate verdict.
    """

    decision = full(candidate, brief, query=query)
    return JudgmentOutcome(full=decision, bucket=classify_verdict(decision))


__all__ = [
    "BUCKET_FAILURE",
    "BUCKET_REJECT",
    "BUCKET_REVIEW",
    "BUCKET_SAVE",
    "JUDGMENT_FINGERPRINT_KEY",
    "SHORT_LIST_DECISIONS",
    "TRANSFERABILITY_CONTRACT_REASON",
    "JudgmentOutcome",
    "classify_verdict",
    "exa_full_judge",
    "exa_lane_context",
    "judge_candidate",
    "judgment_fingerprint",
]
