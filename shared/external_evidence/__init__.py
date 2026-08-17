"""Public surface for the candidate-level external evidence augmentation step.

Slice 1 of the perplexity-evidence-augmentation feature: types, provider,
normalizer, and trigger gate, with zero callers.

Slice 2 wires that machinery into the LinkedIn full-eval path as shadow /
analytical-debug only. Slice 2 adds ``shadow_writer`` (the on-disk schema +
writer for the shadow comparison record) to this public surface; the
orchestrator imports those names from here so the package boundary stays
explicit.
"""

from shared.external_evidence.gate import (
    should_request_external_evidence,
    should_request_external_evidence_for_researcher,
)
from shared.external_evidence.normalizer import normalize_perplexity_response
from shared.external_evidence.provider import fetch_external_candidate_evidence
from shared.external_evidence.shadow_writer import (
    ShadowFullJudgmentRecord,
    compute_judgment_diff,
    record_shadow_full_judgment,
)

__all__ = [
    "fetch_external_candidate_evidence",
    "normalize_perplexity_response",
    "should_request_external_evidence",
    "should_request_external_evidence_for_researcher",
    "ShadowFullJudgmentRecord",
    "compute_judgment_diff",
    "record_shadow_full_judgment",
]
