from __future__ import annotations

import pytest

from linkedin.continuous_verification import (
    AccessibilityNodeSnapshot,
    ColdPathProbe,
    default_cold_path_probe_schedule,
    diff_accessibility_tree,
    evaluate_cold_path_registry,
    evaluate_matching_contract_freshness,
    run_due_cold_path_probes,
)
from linkedin.matching_contract import (
    UNVERIFIED_LINKEDIN_MATCHING_CONTRACT,
    build_verified_contract_from_seat_test_counts,
)


def test_accessibility_tree_diff_alerts_on_selector_drift() -> None:
    report = diff_accessibility_tree(
        {
            "button[data-test-save]": {"role": "button", "name": "Save"},
            "button[data-test-next]": {"role": "button", "name": "Next"},
            "div[data-test-drawer]": {"role": "dialog", "name": "Candidate"},
        },
        {
            "button[data-test-save]": {"role": "button", "name": "Save candidate"},
            "div[data-test-drawer]": {"role": "region", "name": "Candidate"},
        },
        generated_at="2026-06-17T00:00:00Z",
    )

    assert report.status == "alert"
    by_selector = {drift.selector: drift.status for drift in report.drifts}
    assert by_selector == {
        "button[data-test-save]": "name_changed",
        "button[data-test-next]": "missing",
        "div[data-test-drawer]": "role_changed",
    }
    payload = report.to_dict()
    assert payload["alert_only"] is True
    assert payload["cadence_days"] == 7


def test_accessibility_tree_diff_rejects_invalid_snapshot_shapes() -> None:
    with pytest.raises(
        ValueError,
        match="generated_at must be a string ISO timestamp or datetime",
    ):
        diff_accessibility_tree(
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
            generated_at={"at": "2026-06-17T00:00:00Z"},
        )

    with pytest.raises(ValueError, match="cadence_days must be a non-negative integer"):
        diff_accessibility_tree(
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
            cadence_days=True,
        )

    with pytest.raises(
        ValueError,
        match="accessibility baseline snapshot map must be an object",
    ):
        diff_accessibility_tree(
            ["not", "a", "mapping"],
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
        )

    with pytest.raises(ValueError, match="accessibility baseline selector must be a string"):
        diff_accessibility_tree(
            {("button", "save"): {"role": "button", "name": "Save"}},
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
        )

    with pytest.raises(ValueError, match="accessibility current selector must be non-empty"):
        diff_accessibility_tree(
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
            {" ": {"role": "button", "name": "Save"}},
        )

    with pytest.raises(ValueError, match="accessibility snapshot must be an object"):
        diff_accessibility_tree(
            {"button[data-test-save]": 123},
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
        )

    with pytest.raises(ValueError, match="accessibility snapshot role must be a string"):
        diff_accessibility_tree(
            {"button[data-test-save]": {"role": 123, "name": "Save"}},
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
        )

    with pytest.raises(ValueError, match="accessibility snapshot name must be a string"):
        diff_accessibility_tree(
            {"button[data-test-save]": {"role": "button", "name": "Save"}},
            {"button[data-test-save]": {"role": "button", "name": ["Save"]}},
        )


def test_accessibility_tree_diff_normalizes_snapshot_objects_to_map_selector() -> None:
    report = diff_accessibility_tree(
        {
            "button[data-test-save]": AccessibilityNodeSnapshot(
                selector="stale-selector",
                role="button",
                name="Save",
            )
        },
        {},
        generated_at="2026-06-17T00:00:00Z",
    )

    drift = report.drifts[0]
    assert drift.selector == "button[data-test-save]"
    assert drift.baseline is not None
    assert drift.baseline.selector == "button[data-test-save]"

    with pytest.raises(ValueError, match="accessibility snapshot role must be a string"):
        diff_accessibility_tree(
            {
                "button[data-test-save]": AccessibilityNodeSnapshot(
                    selector="button[data-test-save]",
                    role=123,
                    name="Save",
                )
            },
            {},
        )


def test_cold_path_registry_flags_silence_and_buckets_live_seat_paths() -> None:
    report = evaluate_cold_path_registry(
        [
            ColdPathProbe(
                name="save",
                max_silence_days=7,
                last_ran_at="2026-06-01T00:00:00Z",
            ),
            ColdPathProbe(
                name="drawer",
                max_silence_days=7,
                last_ran_at="2026-06-15T00:00:00Z",
            ),
            ColdPathProbe(
                name="pagination",
                max_silence_days=7,
                last_ran_at=None,
            ),
            ColdPathProbe(
                name="fallback",
                max_silence_days=7,
                last_ran_at="2026-06-15T00:00:00Z",
                requires_live_seat=True,
            ),
        ],
        now="2026-06-17T00:00:00Z",
    )

    assert report.status == "alert"
    by_name = {result.name: result for result in report.results}
    assert by_name["save"].status == "stale"
    assert by_name["drawer"].status == "ok"
    assert by_name["pagination"].status == "never_run"
    assert by_name["fallback"].status == "pending_live_seat"
    assert report.to_dict()["alert_only"] is True


def test_cold_path_registry_reports_invalid_probe_metadata_without_crashing() -> None:
    with pytest.raises(ValueError, match="now must be a string ISO timestamp or datetime"):
        evaluate_cold_path_registry(
            [ColdPathProbe(name="save", max_silence_days=7)],
            now={"at": "2026-06-17T00:00:00Z"},
        )

    with pytest.raises(ValueError, match="alert_only must be a boolean"):
        evaluate_cold_path_registry(
            [ColdPathProbe(name="save", max_silence_days=7)],
            alert_only="yes",
        )

    report = evaluate_cold_path_registry(
        [
            ColdPathProbe(
                name="save",
                max_silence_days=True,
                last_ran_at={"at": "2026-06-01T00:00:00Z"},
                requires_live_seat="yes",
            ),
            ColdPathProbe(
                name=123,
                max_silence_days=-1,
                last_ran_at="not-a-timestamp",
            ),
        ],
        now="2026-06-17T00:00:00Z",
    )

    assert report.status == "alert"
    assert [result.status for result in report.results] == [
        "invalid_probe",
        "invalid_probe",
    ]
    assert report.results[0].name == "save"
    assert report.results[0].error == (
        "max_silence_days must be a non-negative integer; "
        "requires_live_seat must be a boolean; "
        "last_ran_at must be a string ISO timestamp"
    )
    assert report.results[1].name == "<invalid>"
    assert report.results[1].error == (
        "name must be a non-empty string; "
        "max_silence_days must be a non-negative integer; "
        "last_ran_at must be an ISO timestamp"
    )


def test_default_cold_path_probe_schedule_covers_m4_paths() -> None:
    probes = default_cold_path_probe_schedule(
        last_ran_at="2026-06-17T00:00:00Z",
        max_silence_days=7,
    )

    assert [probe.name for probe in probes] == [
        "save",
        "drawer",
        "pagination",
        "fallback",
    ]
    assert all(probe.max_silence_days == 7 for probe in probes)


def test_cold_path_runner_executes_due_fixture_paths_and_buckets_live_seat() -> None:
    called: list[str] = []

    def _ok_fixture(probe: ColdPathProbe) -> dict[str, str]:
        called.append(probe.name)
        return {
            "status": "ok",
            "evidence_ref": f"fixture:{probe.name}:20260617",
        }

    report = run_due_cold_path_probes(
        [
            ColdPathProbe(
                name="save",
                max_silence_days=7,
                last_ran_at="2026-06-01T00:00:00Z",
            ),
            ColdPathProbe(
                name="drawer",
                max_silence_days=7,
                last_ran_at="2026-06-15T00:00:00Z",
            ),
            ColdPathProbe(
                name="pagination",
                max_silence_days=7,
                last_ran_at=None,
            ),
            ColdPathProbe(
                name="fallback",
                max_silence_days=7,
                last_ran_at=None,
                requires_live_seat=True,
            ),
        ],
        {"save": _ok_fixture, "pagination": _ok_fixture},
        now="2026-06-17T00:00:00Z",
    )

    assert called == ["save", "pagination"]
    by_name = {result.name: result for result in report.results}
    assert by_name["save"].status == "ok"
    assert by_name["save"].was_due is True
    assert by_name["save"].ran_at == "2026-06-17T00:00:00+00:00"
    assert by_name["save"].evidence_ref == "fixture:save:20260617"
    assert by_name["drawer"].status == "ok"
    assert by_name["drawer"].was_due is False
    assert by_name["pagination"].status == "ok"
    assert by_name["pagination"].evidence_ref == "fixture:pagination:20260617"
    assert by_name["fallback"].status == "pending_live_seat"
    assert report.status == "alert"


def test_cold_path_runner_requires_fixture_evidence_for_ok_status() -> None:
    with pytest.raises(ValueError, match="now must be a string ISO timestamp or datetime"):
        run_due_cold_path_probes(
            [ColdPathProbe(name="save", max_silence_days=7)],
            {"save": lambda _probe: {"status": "ok", "evidence_ref": "fixture:save"}},
            now=["2026-06-17T00:00:00Z"],
        )

    with pytest.raises(ValueError, match="alert_only must be a boolean"):
        run_due_cold_path_probes(
            [ColdPathProbe(name="save", max_silence_days=7)],
            {"save": lambda _probe: {"status": "ok", "evidence_ref": "fixture:save"}},
            alert_only="yes",
        )

    report = run_due_cold_path_probes(
        [
            ColdPathProbe(name="save", max_silence_days=7),
            ColdPathProbe(name="drawer", max_silence_days=7),
        ],
        {"save": lambda _probe: {"status": "ok"}},
        now="2026-06-17T00:00:00Z",
    )

    by_name = {result.name: result for result in report.results}
    assert by_name["save"].status == "missing_evidence"
    assert by_name["save"].ran_at == "2026-06-17T00:00:00+00:00"
    assert by_name["drawer"].status == "missing_runner"
    assert report.status == "alert"


def test_cold_path_runner_reports_fixture_exceptions_without_guessing() -> None:
    def _broken_fixture(_probe: ColdPathProbe) -> None:
        raise RuntimeError("fixture failed")

    report = run_due_cold_path_probes(
        [ColdPathProbe(name="save", max_silence_days=7)],
        {"save": _broken_fixture},
        now="2026-06-17T00:00:00Z",
    )

    result = report.results[0]
    assert result.status == "error"
    assert result.error == "RuntimeError"
    assert result.ran_at == "2026-06-17T00:00:00+00:00"


def test_cold_path_runner_rejects_non_string_fixture_outcome_fields() -> None:
    report = run_due_cold_path_probes(
        [
            ColdPathProbe(name="save", max_silence_days=7),
            ColdPathProbe(name="drawer", max_silence_days=7),
            ColdPathProbe(name="pagination", max_silence_days=7),
        ],
        {
            "save": lambda _probe: {"status": "ok", "evidence_ref": 123},
            "drawer": lambda _probe: {"status": True, "evidence_ref": "fixture:drawer"},
            "pagination": lambda _probe: {
                "status": "error",
                "error": {"message": "not a string"},
            },
        },
        now="2026-06-17T00:00:00Z",
    )

    by_name = {result.name: result for result in report.results}
    assert by_name["save"].status == "invalid_outcome"
    assert by_name["save"].error == "evidence_ref must be a string"
    assert by_name["drawer"].status == "invalid_outcome"
    assert by_name["drawer"].error == "status must be a string"
    assert by_name["pagination"].status == "invalid_outcome"
    assert by_name["pagination"].error == "error must be a string"
    assert report.status == "alert"


def test_cold_path_runner_rejects_shorthand_or_missing_fixture_status() -> None:
    report = run_due_cold_path_probes(
        [
            ColdPathProbe(name="save", max_silence_days=7),
            ColdPathProbe(name="drawer", max_silence_days=7),
            ColdPathProbe(name="pagination", max_silence_days=7),
            ColdPathProbe(name="fallback", max_silence_days=7),
            ColdPathProbe(name="search", max_silence_days=7),
            ColdPathProbe(name="message", max_silence_days=7),
        ],
        {
            "save": lambda _probe: True,
            "drawer": lambda _probe: "ok",
            "pagination": lambda _probe: None,
            "fallback": lambda _probe: {"evidence_ref": "fixture:fallback"},
            "search": lambda _probe: {"status": ""},
            "message": lambda _probe: {
                "status": "done",
                "evidence_ref": "fixture:message",
            },
        },
        now="2026-06-17T00:00:00Z",
    )

    by_name = {result.name: result for result in report.results}
    assert by_name["save"].status == "invalid_outcome"
    assert by_name["save"].error == "outcome must be an object"
    assert by_name["drawer"].status == "invalid_outcome"
    assert by_name["drawer"].error == "outcome must be an object"
    assert by_name["pagination"].status == "invalid_outcome"
    assert by_name["pagination"].error == "outcome must be an object"
    assert by_name["fallback"].status == "invalid_outcome"
    assert by_name["fallback"].error == "status is required"
    assert by_name["search"].status == "invalid_outcome"
    assert by_name["search"].error == "status must be a non-empty string"
    assert by_name["message"].status == "invalid_outcome"
    assert by_name["message"].error == "status must be one of: error, ok"
    assert report.status == "alert"


def test_cold_path_runner_rejects_unexpected_fixture_outcome_keys() -> None:
    report = run_due_cold_path_probes(
        [
            ColdPathProbe(name="save", max_silence_days=7),
            ColdPathProbe(name="drawer", max_silence_days=7),
        ],
        {
            "save": lambda _probe: {
                "status": "ok",
                "evidence_ref": "fixture:save",
                "screenshot_path": "proof.png",
            },
            "drawer": lambda _probe: {
                "status": "ok",
                "evidence_ref": "fixture:drawer",
                ("raw", "key"): "not JSON-shaped",
            },
        },
        now="2026-06-17T00:00:00Z",
    )

    by_name = {result.name: result for result in report.results}
    assert by_name["save"].status == "invalid_outcome"
    assert by_name["save"].error == "outcome has unexpected keys: screenshot_path"
    assert by_name["drawer"].status == "invalid_outcome"
    assert by_name["drawer"].error == "outcome key ('raw', 'key') must be a string"
    assert report.status == "alert"


def test_cold_path_runner_does_not_execute_invalid_probe_metadata() -> None:
    called: list[str] = []

    report = run_due_cold_path_probes(
        [
            ColdPathProbe(
                name="save",
                max_silence_days=True,
                last_ran_at=["2026-06-01T00:00:00Z"],
            ),
        ],
        {"save": lambda probe: called.append(probe.name) or {"status": "ok"}},
        now="2026-06-17T00:00:00Z",
    )

    assert called == []
    assert report.status == "alert"
    result = report.results[0]
    assert result.name == "save"
    assert result.status == "invalid_probe"
    assert result.ran_at == "2026-06-17T00:00:00+00:00"
    assert result.error == (
        "max_silence_days must be a non-negative integer; "
        "last_ran_at must be a string ISO timestamp"
    )


def test_matching_contract_freshness_alerts_when_unverified_or_stale() -> None:
    with pytest.raises(ValueError, match="now must be a string ISO timestamp or datetime"):
        evaluate_matching_contract_freshness(
            UNVERIFIED_LINKEDIN_MATCHING_CONTRACT,
            now=20260617,
        )

    with pytest.raises(ValueError, match="max_age_days must be a non-negative integer"):
        evaluate_matching_contract_freshness(
            UNVERIFIED_LINKEDIN_MATCHING_CONTRACT,
            max_age_days=True,
        )

    with pytest.raises(ValueError, match="alert_only must be a boolean"):
        evaluate_matching_contract_freshness(
            UNVERIFIED_LINKEDIN_MATCHING_CONTRACT,
            alert_only="yes",
        )

    unverified = evaluate_matching_contract_freshness(
        UNVERIFIED_LINKEDIN_MATCHING_CONTRACT,
        now="2026-06-17T00:00:00Z",
        max_age_days=30,
    )
    assert unverified.status == "unverified"
    assert unverified.alert_only is True

    contract = build_verified_contract_from_seat_test_counts(
        {
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
        verified_at="2026-04-01T00:00:00Z",
    )
    stale = evaluate_matching_contract_freshness(
        contract,
        now="2026-06-17T00:00:00Z",
        max_age_days=30,
    )

    assert stale.status == "stale"
    assert stale.age_days == 77
    assert stale.last_empirically_verified == "2026-04-01T00:00:00Z"


def test_matching_contract_freshness_rejects_malformed_mapping_timestamp() -> None:
    non_string = evaluate_matching_contract_freshness(
        {"last_empirically_verified": {"at": "2026-04-01T00:00:00Z"}},
        now="2026-06-17T00:00:00Z",
        max_age_days=30,
    )
    assert non_string.status == "unverified"
    assert non_string.last_empirically_verified is None
    assert non_string.age_days is None

    invalid_string = evaluate_matching_contract_freshness(
        {"last_empirically_verified": "not-a-timestamp"},
        now="2026-06-17T00:00:00Z",
        max_age_days=30,
    )
    assert invalid_string.status == "invalid"
    assert invalid_string.last_empirically_verified == "not-a-timestamp"
    assert invalid_string.age_days is None
