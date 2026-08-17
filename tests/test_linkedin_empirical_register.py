from __future__ import annotations

import pytest

from linkedin.empirical_register import (
    FinalLiveBucketError,
    FINAL_LIVE_BUCKET_SCHEMA_VERSION,
    REQUIRED_COLD_PATHS,
    STALE_MATCHING_CONTRACT_POLICIES,
    final_linkedin_live_bucket_payload_template,
    require_final_linkedin_live_bucket_complete,
    validate_final_linkedin_live_bucket,
)
from linkedin.matching_contract import required_linkedin_matching_seat_tests


class _StringifiesTo:
    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


def _complete_payload() -> dict:
    return {
        "schema_version": FINAL_LIVE_BUCKET_SCHEMA_VERSION,
        "verified_at": "2026-06-17T00:00:00Z",
        "matching_counts": {
            "benchmark": 100,
            "benchmarks": 80,
            "benchmark_or_benchmarks": 170,
            "fine_tuning_hyphenated": 40,
            "fine_tuning_spaced": 55,
            "finetuning_closed": 35,
            "fine_tuning_union": 120,
            "saas_keyword": 25,
            "saas_or_software_as_a_service": 40,
        },
        "matching_queries": {
            spec.query_key: spec.query
            for spec in required_linkedin_matching_seat_tests()
        },
        "matching_counts_evidence_ref": "linkedin-live-matching-counts-20260617",
        "profile_id_probe": {
            "stable_profile_id_seen": True,
            "sample_size": 5,
            "evidence_ref": "linkedin-live-profile-id-probe-20260617",
        },
        "cold_path_results": [
            {
                "name": "save",
                "status": "ok",
                "ran_at": "2026-06-17T00:00:00Z",
                "evidence_ref": "linkedin-live-save-probe-20260617",
            },
            {
                "name": "drawer",
                "status": "ok",
                "ran_at": "2026-06-17T00:00:00Z",
                "evidence_ref": "linkedin-live-drawer-probe-20260617",
            },
            {
                "name": "pagination",
                "status": "ok",
                "ran_at": "2026-06-17T00:00:00Z",
                "evidence_ref": "linkedin-live-pagination-probe-20260617",
            },
            {
                "name": "fallback",
                "status": "ok",
                "ran_at": "2026-06-17T00:00:00Z",
                "evidence_ref": "linkedin-live-fallback-probe-20260617",
            },
        ],
        "stale_matching_contract_policy": "alert_only",
    }


def test_final_live_bucket_reports_pending_gates_without_guessing() -> None:
    report = validate_final_linkedin_live_bucket({})

    assert report.status == "pending"
    assert report.is_complete is False
    assert set(report.pending_gates) == {
        "matching_counts",
        "profile_id_probe",
        "cold_path_results",
        "stale_matching_contract_policy",
    }
    assert report.errors == ()

    with pytest.raises(FinalLiveBucketError, match="matching_counts"):
        require_final_linkedin_live_bucket_complete({})


def test_final_live_bucket_template_tracks_canonical_registries() -> None:
    template = final_linkedin_live_bucket_payload_template()

    assert template["schema_version"] == FINAL_LIVE_BUCKET_SCHEMA_VERSION
    assert set(template["matching_counts"]) == {
        spec.query_key for spec in required_linkedin_matching_seat_tests()
    }
    assert template["matching_queries"] == {
        spec.query_key: spec.query for spec in required_linkedin_matching_seat_tests()
    }
    assert [row["name"] for row in template["cold_path_results"]] == list(
        REQUIRED_COLD_PATHS
    )
    assert template["stale_matching_contract_policy"] == " | ".join(
        STALE_MATCHING_CONTRACT_POLICIES
    )

    report = validate_final_linkedin_live_bucket(template)
    assert report.status == "invalid"
    assert report.is_complete is False


def test_final_live_bucket_rejects_non_object_payload() -> None:
    report = validate_final_linkedin_live_bucket(["not", "an", "object"])

    assert report.status == "invalid"
    assert report.pending_gates == ()
    assert report.errors == ("payload must be an object",)
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None

    with pytest.raises(FinalLiveBucketError, match="payload must be an object"):
        require_final_linkedin_live_bucket_complete(["not", "an", "object"])


def test_final_live_bucket_rejects_wrong_schema_version_when_supplied() -> None:
    payload = _complete_payload()
    payload["schema_version"] = "linkedin.final_live_bucket.v0"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "schema_version must be linkedin.final_live_bucket.v1" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_rejects_non_string_schema_version() -> None:
    payload = _complete_payload()
    payload["schema_version"] = 1

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "schema_version must be a string" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_rejects_unexpected_top_level_keys() -> None:
    payload = _complete_payload()
    payload["operator_notes"] = "hand-entered side channel"
    payload[("tuple", "key")] = "not JSON-shaped"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "payload has unexpected keys: operator_notes" in report.errors
    assert "payload key ('tuple', 'key') must be a string" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_builds_verified_report_from_supplied_evidence() -> None:
    report = require_final_linkedin_live_bucket_complete(_complete_payload())

    assert report.status == "complete"
    assert report.pending_gates == ()
    assert report.errors == ()
    assert report.schema_version == FINAL_LIVE_BUCKET_SCHEMA_VERSION
    assert report.matching_contract is not None
    assert (
        report.matching_contract["evidence_ref"]
        == "linkedin-live-matching-counts-20260617"
    )
    assert report.matching_contract["keyword_stemming"]["value"] == "no_stemming"
    assert report.matching_contract["hyphen_space_tokenization"]["value"] == (
        "distinct"
    )
    assert report.matching_contract["matching_queries"] == {
        spec.query_key: spec.query for spec in required_linkedin_matching_seat_tests()
    }
    assert report.profile_id_availability is not None
    assert report.profile_id_availability["status"] == "verified"
    assert len(report.cold_path_results) == 4
    assert report.stale_matching_contract_policy == "alert_only"


def test_final_live_bucket_invalid_reports_do_not_emit_partial_proof() -> None:
    payload = _complete_payload()
    payload["stale_matching_contract_policy"] = {"policy": "alert_only"}

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert report.errors == ("stale_matching_contract_policy must be a string",)
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_requires_schema_version_when_evidence_supplied() -> None:
    payload = _complete_payload()
    payload.pop("schema_version")

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "pending"
    assert "schema_version" in report.pending_gates
    assert report.errors == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None

    with pytest.raises(FinalLiveBucketError, match="schema_version"):
        require_final_linkedin_live_bucket_complete(payload)


def test_final_live_bucket_normalizes_completed_profile_probe_proof() -> None:
    payload = _complete_payload()
    payload["profile_id_probe"] = {
        "stable_profile_id_seen": True,
        "sample_size": 5,
        "evidence_ref": " linkedin-live-profile-id-probe-20260617 ",
        "verified_at": " 2026-06-17T01:02:03Z ",
    }

    report = require_final_linkedin_live_bucket_complete(payload)

    assert report.profile_id_availability == {
        "status": "verified",
        "evidence": {
            "stable_profile_id_seen": True,
            "sample_size": 5,
            "evidence_ref": "linkedin-live-profile-id-probe-20260617",
            "verified_at": "2026-06-17T01:02:03Z",
        },
        "verified_at": "2026-06-17T01:02:03Z",
    }


def test_final_live_bucket_keeps_negative_profile_probe_pending() -> None:
    payload = _complete_payload()
    payload["profile_id_probe"] = {
        "stable_profile_id_seen": False,
        "sample_size": 5,
        "evidence_ref": "linkedin-live-profile-id-probe-20260617",
    }

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "pending"
    assert "profile_id_probe.stable_profile_id_seen" in report.pending_gates
    assert report.profile_id_availability is None


def test_final_live_bucket_rejects_non_boolean_profile_probe_result() -> None:
    payload = _complete_payload()
    payload["profile_id_probe"]["stable_profile_id_seen"] = "true"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert (
        "profile_id_probe.stable_profile_id_seen must be a boolean"
        in report.errors
    )
    assert "profile_id_probe.stable_profile_id_seen" not in report.pending_gates
    assert report.profile_id_availability is None


def test_final_live_bucket_rejects_unexpected_profile_probe_keys() -> None:
    payload = _complete_payload()
    payload["profile_id_probe"]["screenshot_path"] = "profile-id-proof.png"
    payload["profile_id_probe"][("raw", "key")] = "not JSON-shaped"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "profile_id_probe has unexpected keys: screenshot_path" in report.errors
    assert "profile_id_probe key ('raw', 'key') must be a string" in report.errors
    assert report.profile_id_availability is None


def test_final_live_bucket_requires_matching_counts_evidence_ref() -> None:
    payload = _complete_payload()
    payload.pop("matching_counts_evidence_ref")

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "pending"
    assert "matching_counts_evidence_ref" in report.pending_gates
    assert report.matching_contract is None


def test_final_live_bucket_rejects_bad_counts_before_missing_supporting_fields() -> None:
    payload = _complete_payload()
    payload.pop("matching_counts_evidence_ref")
    payload["matching_counts"]["benchmark"] = "not-an-int"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert any(
        "Seat-test count 'benchmark' must be an integer" in error
        for error in report.errors
    )
    assert "matching_counts_evidence_ref" not in report.pending_gates
    assert report.matching_contract is None

    payload = _complete_payload()
    payload.pop("verified_at")
    payload["matching_counts"]["benchmark_or_benchmarks"] = 1

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert any(
        "benchmark OR benchmarks count cannot be lower" in error
        for error in report.errors
    )
    assert report.matching_contract is None


def test_final_live_bucket_requires_canonical_matching_queries() -> None:
    payload = _complete_payload()
    payload.pop("matching_queries")

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "pending"
    assert report.errors == ()
    assert "matching_queries" in report.pending_gates
    assert report.matching_contract is None

    payload = _complete_payload()
    payload["matching_queries"]["benchmark_or_benchmarks"] = "benchmark AND benchmarks"
    payload["matching_queries"]["saas_keyword"] = 123
    payload["matching_queries"]["extra_query"] = "extra"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "Unexpected matching query keys: extra_query" in report.errors
    assert report.matching_contract is None

    payload = _complete_payload()
    payload["matching_queries"]["benchmark_or_benchmarks"] = "benchmark AND benchmarks"
    payload["matching_queries"]["saas_keyword"] = 123

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert (
        "matching_queries.benchmark_or_benchmarks must be 'benchmark OR benchmarks'"
        in report.errors
    )
    assert "matching_queries.saas_keyword must be a string" in report.errors
    assert report.matching_contract is None


def test_final_live_bucket_rejects_bad_matching_queries_before_missing_evidence_ref() -> None:
    payload = _complete_payload()
    payload.pop("matching_counts_evidence_ref")
    payload["matching_queries"]["benchmark_or_benchmarks"] = "benchmark AND benchmarks"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert (
        "matching_queries.benchmark_or_benchmarks must be 'benchmark OR benchmarks'"
        in report.errors
    )
    assert "matching_counts_evidence_ref" not in report.pending_gates
    assert report.matching_contract is None


def test_final_live_bucket_rejects_incoherent_counts_or_policy_values() -> None:
    payload = _complete_payload()
    payload["matching_counts"] = {
        **payload["matching_counts"],
        "benchmark_or_benchmarks": 90,
    }
    payload["stale_matching_contract_policy"] = "maybe_block"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert any("matching_counts invalid" in error for error in report.errors)
    assert any("stale_matching_contract_policy" in error for error in report.errors)


def test_final_live_bucket_rejects_non_iso_timestamps() -> None:
    payload = _complete_payload()
    payload["profile_id_probe"]["verified_at"] = "2026-06-17Tnot-a-timeZ"
    payload["cold_path_results"][0]["ran_at"] = "2026-06-17Talso-not-a-timeZ"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "profile_id_probe.verified_at must be an ISO timestamp" in report.errors
    assert "cold_path_results.save.ran_at must be an ISO timestamp" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_rejects_future_timestamps() -> None:
    payload = _complete_payload()
    payload["verified_at"] = "2999-01-01T00:00:00Z"
    payload["profile_id_probe"]["verified_at"] = "2999-01-01T00:00:00Z"
    payload["cold_path_results"][0]["ran_at"] = "2999-01-01T00:00:00Z"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "verified_at must not be in the future" in report.errors
    assert "profile_id_probe.verified_at must not be in the future" in report.errors
    assert "cold_path_results.save.ran_at must not be in the future" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_rejects_non_string_timestamps() -> None:
    payload = _complete_payload()
    payload["verified_at"] = 20260617
    payload["profile_id_probe"]["verified_at"] = {"at": "2026-06-17T00:00:00Z"}
    payload["cold_path_results"][0]["ran_at"] = ["2026-06-17T00:00:00Z"]

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "verified_at must be a string ISO timestamp" in report.errors
    assert "profile_id_probe.verified_at must be a string ISO timestamp" in report.errors
    assert "cold_path_results.save.ran_at must be a string ISO timestamp" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_rejects_ambiguous_timestamps() -> None:
    payload = _complete_payload()
    payload["verified_at"] = "2026-06-17"
    payload["profile_id_probe"]["verified_at"] = "2026-06-17T01:02:03"
    payload["cold_path_results"][0]["ran_at"] = "2026-06-17"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "verified_at must include date and time" in report.errors
    assert "profile_id_probe.verified_at must include timezone" in report.errors
    assert "cold_path_results.save.ran_at must include date and time" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_rejects_duplicate_cold_path_results() -> None:
    payload = _complete_payload()
    payload["cold_path_results"].append(
        {
            "name": "save",
            "status": "ok",
            "ran_at": "2026-06-18T00:00:00Z",
            "evidence_ref": "duplicate-save-evidence",
        }
    )

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "cold_path_results.save appears more than once" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_rejects_unknown_cold_path_results() -> None:
    payload = _complete_payload()
    payload["cold_path_results"].append(
        {
            "name": "unknown_path",
            "status": "ok",
            "ran_at": "2026-06-18T00:00:00Z",
            "evidence_ref": "unknown-path-evidence",
        }
    )

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "cold_path_results.unknown_path is not a required cold path" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_rejects_non_list_cold_path_results() -> None:
    payload = _complete_payload()
    payload["cold_path_results"] = tuple(payload["cold_path_results"])

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert report.errors == ("cold_path_results must be a list",)
    assert report.cold_path_results == ()


def test_final_live_bucket_rejects_unknown_cold_path_status() -> None:
    payload = _complete_payload()
    payload["cold_path_results"][0]["status"] = "done"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert any(
        error.startswith("cold_path_results.save.status must be one of:")
        for error in report.errors
    )
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_keeps_known_failed_cold_path_status_pending() -> None:
    payload = _complete_payload()
    payload["cold_path_results"][0]["status"] = "error"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "pending"
    assert report.errors == ()
    assert "cold_path_results.save.status" in report.pending_gates
    assert {result["name"] for result in report.cold_path_results} == {
        "drawer",
        "pagination",
        "fallback",
    }


def test_final_live_bucket_normalizes_completed_cold_path_proof_rows() -> None:
    payload = _complete_payload()
    payload["cold_path_results"][0] = {
        "name": " save ",
        "status": " ok ",
        "ran_at": " 2026-06-17T00:00:00Z ",
        "evidence_ref": " linkedin-live-save-probe-20260617 ",
    }

    report = require_final_linkedin_live_bucket_complete(payload)

    by_name = {result["name"]: result for result in report.cold_path_results}
    assert by_name["save"] == {
        "name": "save",
        "status": "ok",
        "ran_at": "2026-06-17T00:00:00Z",
        "evidence_ref": "linkedin-live-save-probe-20260617",
    }


def test_final_live_bucket_rejects_unexpected_cold_path_result_keys() -> None:
    payload = _complete_payload()
    payload["cold_path_results"][0]["screenshot_path"] = "proof.png"
    payload["cold_path_results"][0][("raw", "key")] = "not JSON-shaped"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert (
        "cold_path_results[0] has unexpected keys: screenshot_path"
        in report.errors
    )
    assert "cold_path_results[0] key ('raw', 'key') must be a string" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_rejects_unknown_matching_count_keys() -> None:
    payload = _complete_payload()
    payload["matching_counts"]["benchmark_near_operator"] = 12

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert any(
        "Unexpected Recruiter seat-test counts: benchmark_near_operator" in error
        for error in report.errors
    )
    assert report.matching_contract is None


def test_final_live_bucket_rejects_non_string_matching_count_keys() -> None:
    payload = _complete_payload()
    payload["matching_counts"][12] = 12

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert any(
        "matching_counts invalid: Seat-test count key 12 must be a string" in error
        for error in report.errors
    )
    assert report.matching_contract is None


def test_final_live_bucket_rejects_non_string_evidence_refs() -> None:
    payload = _complete_payload()
    payload["matching_counts_evidence_ref"] = {"id": "matching-counts"}
    payload["profile_id_probe"]["evidence_ref"] = ["profile-id-probe"]
    payload["cold_path_results"][0]["evidence_ref"] = 123

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "matching_counts_evidence_ref must be a string" in report.errors
    assert "profile_id_probe.evidence_ref must be a string" in report.errors
    assert "cold_path_results.save.evidence_ref must be a string" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_keeps_placeholder_evidence_refs_pending() -> None:
    payload = _complete_payload()
    payload["matching_counts_evidence_ref"] = "<live evidence artifact id>"
    payload["profile_id_probe"]["evidence_ref"] = "<live evidence artifact id>"
    for result in payload["cold_path_results"]:
        result["evidence_ref"] = "<live evidence artifact id>"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "pending"
    assert report.errors == ()
    assert set(report.pending_gates) == {
        "matching_counts_evidence_ref",
        "profile_id_probe.evidence_ref",
        "cold_path_results.save.evidence_ref",
        "cold_path_results.drawer.evidence_ref",
        "cold_path_results.pagination.evidence_ref",
        "cold_path_results.fallback.evidence_ref",
    }
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()


def test_final_live_bucket_keeps_prose_placeholder_refs_pending() -> None:
    payload = _complete_payload()
    payload["matching_counts_evidence_ref"] = "TODO"
    payload["profile_id_probe"]["evidence_ref"] = "tbd - profile proof"
    for result in payload["cold_path_results"]:
        result["evidence_ref"] = "placeholder evidence"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "pending"
    assert report.errors == ()
    assert set(report.pending_gates) == {
        "matching_counts_evidence_ref",
        "profile_id_probe.evidence_ref",
        "cold_path_results.save.evidence_ref",
        "cold_path_results.drawer.evidence_ref",
        "cold_path_results.pagination.evidence_ref",
        "cold_path_results.fallback.evidence_ref",
    }
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()


def test_final_live_bucket_keeps_non_evidence_ref_tokens_pending() -> None:
    payload = _complete_payload()
    payload["matching_counts_evidence_ref"] = "unknown"
    payload["profile_id_probe"]["evidence_ref"] = "example-proof"
    evidence_refs_by_name = {
        "save": "dummy_evidence",
        "drawer": "sample/ref",
        "pagination": "missing evidence",
        "fallback": "fake:evidence",
    }
    for result in payload["cold_path_results"]:
        result["evidence_ref"] = evidence_refs_by_name[result["name"]]

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "pending"
    assert report.errors == ()
    assert set(report.pending_gates) == {
        "matching_counts_evidence_ref",
        "profile_id_probe.evidence_ref",
        "cold_path_results.save.evidence_ref",
        "cold_path_results.drawer.evidence_ref",
        "cold_path_results.pagination.evidence_ref",
        "cold_path_results.fallback.evidence_ref",
    }
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()


def test_final_live_bucket_rejects_supplied_bad_queries_even_without_counts() -> None:
    payload = _complete_payload()
    payload.pop("matching_counts")
    payload["matching_queries"]["benchmark_or_benchmarks"] = "benchmark AND benchmarks"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert (
        "matching_queries.benchmark_or_benchmarks must be 'benchmark OR benchmarks'"
        in report.errors
    )
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None


def test_final_live_bucket_validates_supplied_count_ref_even_without_counts() -> None:
    payload = _complete_payload()
    payload.pop("matching_counts")
    payload["matching_counts_evidence_ref"] = {"id": "counts"}

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "matching_counts_evidence_ref must be a string" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None

    payload = _complete_payload()
    payload.pop("matching_counts")
    payload["matching_counts_evidence_ref"] = "placeholder-count-proof"

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "pending"
    assert report.errors == ()
    assert "matching_counts" in report.pending_gates
    assert "matching_counts_evidence_ref" in report.pending_gates
    assert report.matching_contract is None


def test_final_live_bucket_rejects_non_string_schema_labels() -> None:
    payload = _complete_payload()
    payload["cold_path_results"][0]["name"] = _StringifiesTo("save")
    payload["cold_path_results"][1]["status"] = _StringifiesTo("ok")
    payload["stale_matching_contract_policy"] = _StringifiesTo("alert_only")

    report = validate_final_linkedin_live_bucket(payload)

    assert report.status == "invalid"
    assert "cold_path_results[0].name must be a string" in report.errors
    assert "cold_path_results.drawer.status must be a string" in report.errors
    assert "stale_matching_contract_policy must be a string" in report.errors
    assert report.pending_gates == ()
    assert report.matching_contract is None
    assert report.profile_id_availability is None
    assert report.cold_path_results == ()
    assert report.stale_matching_contract_policy is None
