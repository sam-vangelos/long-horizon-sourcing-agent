from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from linkedin.empirical_register import FINAL_LIVE_BUCKET_SCHEMA_VERSION
from linkedin.matching_contract import required_linkedin_matching_seat_tests


TOOL = Path("tools/validate_linkedin_final_live_bucket.py")
MAKEFILE = Path("Makefile")


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


def _run_tool(
    *args: str, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        check=False,
        capture_output=True,
        input=input_text,
        text=True,
    )


def test_final_live_bucket_tool_prints_canonical_template() -> None:
    result = _run_tool("--template")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == FINAL_LIVE_BUCKET_SCHEMA_VERSION
    assert "benchmark_or_benchmarks" in payload["matching_counts"]
    assert payload["matching_queries"]["benchmark_or_benchmarks"] == (
        "benchmark OR benchmarks"
    )
    assert [row["name"] for row in payload["cold_path_results"]] == [
        "save",
        "drawer",
        "pagination",
        "fallback",
    ]


def test_final_live_bucket_tool_blocks_pending_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "pending.json"
    payload_path.write_text("{}", encoding="utf-8")

    result = _run_tool(str(payload_path))

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "pending"
    assert report["pending_gates"] == [
        "cold_path_results",
        "matching_counts",
        "profile_id_probe",
        "stale_matching_contract_policy",
    ]


def test_final_live_bucket_tool_blocks_invalid_json(tmp_path: Path) -> None:
    payload_path = tmp_path / "invalid.json"
    payload_path.write_text("{not-json", encoding="utf-8")

    result = _run_tool(str(payload_path))

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["status"] == "invalid"
    assert report["errors"]


def test_final_live_bucket_tool_allows_complete_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "complete.json"
    payload_path.write_text(json.dumps(_complete_payload()), encoding="utf-8")

    result = _run_tool(str(payload_path))

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["status"] == "complete"
    assert report["pending_gates"] == []


def test_final_live_bucket_tool_accepts_complete_payload_from_stdin() -> None:
    result = _run_tool("-", input_text=json.dumps(_complete_payload()))

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["status"] == "complete"
    assert report["pending_gates"] == []


def test_final_live_bucket_tool_does_not_persist_contract_by_default(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "complete.json"
    payload_path.write_text(json.dumps(_complete_payload()), encoding="utf-8")
    contract_out = tmp_path / "linkedin-matching-contract.json"

    result = _run_tool(str(payload_path), "--contract-out", str(contract_out))

    assert result.returncode == 0
    assert not contract_out.exists()


def test_final_live_bucket_tool_persists_contract_when_flag_set(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "complete.json"
    payload_path.write_text(json.dumps(_complete_payload()), encoding="utf-8")
    contract_out = tmp_path / "nested" / "linkedin-matching-contract.json"

    result = _run_tool(
        str(payload_path),
        "--persist-contract",
        "--contract-out",
        str(contract_out),
    )

    assert result.returncode == 0
    written = json.loads(contract_out.read_text(encoding="utf-8"))
    report = json.loads(result.stdout)
    assert written == report["matching_contract"]
    assert written["keyword_stemming"]["status"] == "verified"
    assert written["last_empirically_verified"] == "2026-06-17T00:00:00Z"


def test_final_live_bucket_tool_does_not_persist_contract_for_pending_payload(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "pending.json"
    payload_path.write_text("{}", encoding="utf-8")
    contract_out = tmp_path / "linkedin-matching-contract.json"

    result = _run_tool(
        str(payload_path),
        "--persist-contract",
        "--contract-out",
        str(contract_out),
    )

    assert result.returncode == 2
    assert not contract_out.exists()


def test_makefile_exposes_final_live_bucket_verifier_targets() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")

    assert "validate-sqk-evidence:" in text
    assert "sqk-live-bucket-template:" in text
    assert "validate-sqk-live-bucket:" in text
    assert "tests/test_sourcing_quality_kernel_evidence.py" in text
    assert "tests/test_validate_linkedin_final_live_bucket_tool.py" in text
    assert "tools/validate_linkedin_final_live_bucket.py --template" in text
    assert 'tools/validate_linkedin_final_live_bucket.py "$(SQK_LIVE_BUCKET)"' in text
