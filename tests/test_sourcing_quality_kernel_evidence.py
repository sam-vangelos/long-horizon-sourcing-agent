from __future__ import annotations

import re
from pathlib import Path

from linkedin.empirical_register import (
    final_linkedin_live_bucket_payload_template,
    validate_final_linkedin_live_bucket,
)


EVIDENCE_PATH = Path("plans/sourcing-quality-kernel-evidence.md")


def _evidence_text() -> str:
    assert EVIDENCE_PATH.exists(), "Sourcing Quality Kernel evidence artifact is missing"
    return EVIDENCE_PATH.read_text()


def test_sourcing_quality_kernel_evidence_artifact_covers_all_milestones() -> None:
    text = _evidence_text()

    assert "Status: complete; final LinkedIn live bucket supplied and verified" in text
    for section in (
        "## M0 Evidence",
        "## M1A Evidence",
        "## M1B Evidence",
        "## M1C Evidence",
        "## M2 Evidence",
        "## M3 Evidence",
        "## M4 Evidence",
        "## Final LinkedIn Live Bucket",
    ):
        assert section in text

    assert "`make validate` -> passed" in text
    assert "Final LinkedIn live-bucket completion on 2026-06-18" in text
    assert "status complete, pending_gates []" in text


def test_sourcing_quality_kernel_evidence_tracks_named_verifiers() -> None:
    text = _evidence_text()

    required_fragments = (
        "pytest tests/test_receipts.py tests/test_runtime_state_*.py -q -ra",
        "pytest tests/test_designer_judging.py -q",
        "pytest tests/test_matching_contract.py -q",
        "pytest tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py -q",
        "pytest tests/test_observability_monitors.py -q",
        "pytest tests/test_adaptation_*.py -q",
        "tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py",
        "backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests",
    )
    for fragment in required_fragments:
        assert fragment in text


def test_sourcing_quality_kernel_evidence_tracks_global_definition_of_done() -> None:
    text = _evidence_text()

    required_fragments = (
        "## Global Definition-of-Done Evidence",
        "`make validate` -> passed",
        "tests/test_search_string_lane_fields.py::test_work_unit_without_structured_filters_stays_boolean_byte_identical",
        "tests/test_seam_strategy_execution.py::test_all_keyword_lane_plan_keeps_legacy_queue_shape",
        "tests/test_phase0_contracts.py::test_run_log_event_vocabulary_matches_current_emitters",
        "missing receipt mirrors, hash tampering, and append-only triggers",
        "green_but_useless_rate",
        "judge_parse_failure_rate",
        "Adapted-string firewall trace:",
        "Sample groundedness verdict:",
        "Sample accessibility drift report:",
        "Sample cold-path fixture-runner report:",
    )
    for fragment in required_fragments:
        assert fragment in text


def test_sourcing_quality_kernel_evidence_tracks_dependency_order() -> None:
    text = _evidence_text()

    assert "## Dependency Order Evidence" in text
    required_fragments = (
        "M0 -> everything",
        "M1A -> M3",
        "M1B -> M1C",
        "M4 remains last",
        "M1B/M1C/M3 remain gated on the Empirical Register",
        "tests/test_receipts.py tests/test_runtime_state_*.py -q -ra",
        "pytest tests/test_designer_judging.py -q",
        "pytest tests/test_matching_contract.py -q",
        "pytest tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py -q",
        "pytest tests/test_observability_monitors.py -q",
        "pytest tests/test_adaptation_*.py -q",
        "tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py",
    )
    for fragment in required_fragments:
        assert fragment in text


def test_sourcing_quality_kernel_evidence_tracks_invariants() -> None:
    text = _evidence_text()

    assert "## Invariant Coverage Evidence" in text
    required_fragments = (
        "INV-1",
        "tests/test_boolean_normalizer.py",
        "INV-2",
        "validate_final_linkedin_live_bucket({})",
        "INV-3",
        "tests/test_matching_contract.py",
        "INV-4",
        "tests/test_seam_strategy_execution.py",
        "INV-5",
        "tests/test_observability_monitors.py",
        "INV-6",
        "test_all_keyword_lane_plan_keeps_legacy_queue_shape",
        "INV-7",
        "tests/test_sourcing_quality_kernel_evidence.py",
        "INV-8",
        "shared/receipts.py rejects boolean statuses",
        "INV-9",
        "test_adapt_after_block_uses_typed_signal_state_and_validates_actions",
    )
    for fragment in required_fragments:
        assert fragment in text


def test_sourcing_quality_kernel_evidence_tracks_stop_rules_and_empirical_gates() -> None:
    text = _evidence_text()

    assert "## Stop-Rule and Empirical Gate Evidence" in text
    required_fragments = (
        "product-judgment decision",
        "external-system fact not yet verified",
        "high-risk files",
        "test weakening or deletion",
        "verification-command output",
        "Keywords field stems / plural-collapses",
        "Hyphen/space tokenization",
        "Skills-facet expansion fires on Keywords field",
        "Recruiter API returns per-query profile IDs",
        "Current green-but-useless run rate",
        "Current judge parse-failure rate",
        "stale matching contracts",
        "live-seat cold paths",
        "validate_final_linkedin_live_bucket({})",
    )
    for fragment in required_fragments:
        assert fragment in text


def test_sourcing_quality_kernel_evidence_tracks_final_live_bucket_items() -> None:
    text = _evidence_text()

    required_items = (
        "Recruiter seat-test counts for `benchmark`, `benchmarks`, `benchmark OR benchmarks`.",
        'Recruiter seat-test counts for `fine-tuning`, `"fine tuning"`, `finetuning`, and their union.',
        "Recruiter seat-test counts for `SaaS` vs `SaaS OR \"Software as a Service\"`.",
        "Live evidence for stable per-result profile IDs before capture-recapture/overlap uses that signal.",
        "A policy decision on stale matching contracts: keep alert-only or make deploy-blocking.",
        "Any cold-path probe that requires a live LinkedIn seat.",
    )
    for item in required_items:
        assert item in text


def test_final_live_bucket_evidence_matches_empty_payload_verifier() -> None:
    text = _evidence_text()
    report = validate_final_linkedin_live_bucket({})
    expected_gates = ",".join(report.pending_gates)

    assert report.status == "pending"
    assert "status: pending" in text

    match = re.search(r"^pending_gates:\s*(?P<gates>[^\n]+)$", text, re.MULTILINE)
    assert match is not None
    assert match.group("gates") == expected_gates


def test_final_live_bucket_payload_template_tracks_canonical_requirements() -> None:
    text = _evidence_text()
    template = final_linkedin_live_bucket_payload_template()

    assert "tools/validate_linkedin_final_live_bucket.py --template" in text
    assert "tools/validate_linkedin_final_live_bucket.py <live-bucket.json>" in text
    assert f"schema_version: {template['schema_version']}" in text
    assert f"verified_at: {template['verified_at']}" in text
    assert (
        f"matching_counts_evidence_ref: {template['matching_counts_evidence_ref']}"
        in text
    )
    assert "stable_profile_id_seen: true" in text
    assert f"sample_size: {template['profile_id_probe']['sample_size']}" in text

    for key, placeholder in template["matching_counts"].items():
        assert f"  {key}: {placeholder}" in text

    assert "matching_queries:" in text
    for key, query in template["matching_queries"].items():
        assert f"  {key}: {query}" in text

    for cold_path in template["cold_path_results"]:
        assert f"  - name: {cold_path['name']}" in text

    policy_options = template["stale_matching_contract_policy"]
    assert f"stale_matching_contract_policy: {policy_options}" in text
