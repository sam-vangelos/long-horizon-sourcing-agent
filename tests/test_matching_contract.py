from __future__ import annotations

import json
from pathlib import Path

import pytest

from linkedin import matching_contract as matching_contract_module
from linkedin.matching_contract import (
    EmpiricalValue,
    HyphenSpaceTokenizationModel,
    KeywordStemmingModel,
    KeywordsSkillsFacetExpansionModel,
    LinkedInMatchingContract,
    MatchingFact,
    SeatTestEvidenceError,
    UnverifiedMatchingModelError,
    UNVERIFIED_LINKEDIN_MATCHING_CONTRACT,
    build_verified_contract_from_seat_test_counts,
    conservative_morphology_repair_hint,
    load_persisted_matching_contract,
    render_adaptation_matching_guidance,
    render_strategy_matching_guidance,
    required_linkedin_matching_seat_tests,
    required_m1b_seat_tests,
    required_m1c_seat_tests,
)


def test_unverified_matching_contract_refuses_external_behavior_values() -> None:
    with pytest.raises(UnverifiedMatchingModelError) as exc_info:
        UNVERIFIED_LINKEDIN_MATCHING_CONTRACT.require_verified()

    message = str(exc_info.value)
    assert "keyword_stemming" in message
    assert "hyphen_space_tokenization" in message
    assert "keywords_skills_facet_expansion" in message
    assert "Recruiter seat-test evidence" in message
    assert "benchmark OR benchmarks" in message
    assert 'fine-tuning OR "fine tuning" OR finetuning' in message
    assert 'SaaS OR "Software as a Service"' in message


def test_required_m1b_seat_tests_are_explicit() -> None:
    specs = required_m1b_seat_tests()
    by_key = {spec.query_key: spec for spec in specs}

    assert by_key["benchmark"].query == "benchmark"
    assert by_key["benchmarks"].query == "benchmarks"
    assert by_key["benchmark_or_benchmarks"].query == "benchmark OR benchmarks"
    assert by_key["fine_tuning_hyphenated"].query == "fine-tuning"
    assert by_key["fine_tuning_spaced"].query == '"fine tuning"'
    assert by_key["finetuning_closed"].query == "finetuning"
    assert by_key["fine_tuning_union"].query == (
        'fine-tuning OR "fine tuning" OR finetuning'
    )

    assert {spec.fact for spec in specs} == {
        MatchingFact.KEYWORD_STEMMING,
        MatchingFact.HYPHEN_SPACE_TOKENIZATION,
    }


def test_required_m1c_seat_tests_are_explicit() -> None:
    specs = required_m1c_seat_tests()
    by_key = {spec.query_key: spec for spec in specs}

    assert by_key["saas_keyword"].query == "SaaS"
    assert by_key["saas_or_software_as_a_service"].query == (
        'SaaS OR "Software as a Service"'
    )
    assert {spec.fact for spec in specs} == {
        MatchingFact.KEYWORDS_SKILLS_FACET_EXPANSION,
    }

    all_specs = required_linkedin_matching_seat_tests()
    assert len(all_specs) == len(required_m1b_seat_tests()) + len(specs)


def test_verified_contract_requires_evidence_and_timestamp() -> None:
    with pytest.raises(ValueError):
        EmpiricalValue.verified(
            fact=MatchingFact.KEYWORD_STEMMING,
            value=KeywordStemmingModel.NO_STEMMING,
            evidence={},
            verified_at="2026-06-17T00:00:00Z",
        )

    with pytest.raises(ValueError):
        EmpiricalValue.verified(
            fact=MatchingFact.KEYWORD_STEMMING,
            value=KeywordStemmingModel.NO_STEMMING,
            evidence={"benchmark": 10},
            verified_at="",
        )

    with pytest.raises(ValueError, match="ISO timestamp"):
        EmpiricalValue.verified(
            fact=MatchingFact.KEYWORD_STEMMING,
            value=KeywordStemmingModel.NO_STEMMING,
            evidence={"benchmark": 10},
            verified_at="not-a-timestamp",
        )


def test_empirical_values_reject_stringified_contract_fields() -> None:
    with pytest.raises(ValueError, match="empirical fact must be a MatchingFact"):
        EmpiricalValue.unverified("keyword_stemming")

    with pytest.raises(ValueError, match="empirical fact must be a MatchingFact"):
        EmpiricalValue.verified(
            fact="keyword_stemming",
            value=KeywordStemmingModel.NO_STEMMING,
            evidence={"benchmark": 10},
            verified_at="2026-06-17T00:00:00Z",
        )

    with pytest.raises(ValueError, match="keyword_stemming value must be"):
        EmpiricalValue.verified(
            fact=MatchingFact.KEYWORD_STEMMING,
            value="no_stemming",
            evidence={"benchmark": 10},
            verified_at="2026-06-17T00:00:00Z",
        )

    with pytest.raises(ValueError, match="evidence object"):
        EmpiricalValue.verified(
            fact=MatchingFact.KEYWORD_STEMMING,
            value=KeywordStemmingModel.NO_STEMMING,
            evidence=["benchmark", 10],
            verified_at="2026-06-17T00:00:00Z",
        )


def test_verified_contract_serializes_typed_empirical_values() -> None:
    contract = LinkedInMatchingContract(
        keyword_stemming=EmpiricalValue.verified(
            fact=MatchingFact.KEYWORD_STEMMING,
            value=KeywordStemmingModel.NO_STEMMING,
            evidence={
                "benchmark": 10,
                "benchmarks": 11,
                "benchmark_or_benchmarks": 21,
            },
            verified_at="2026-06-17T00:00:00Z",
        ),
        hyphen_space_tokenization=EmpiricalValue.verified(
            fact=MatchingFact.HYPHEN_SPACE_TOKENIZATION,
            value=HyphenSpaceTokenizationModel.DISTINCT,
            evidence={
                "fine_tuning_hyphenated": 7,
                "fine_tuning_spaced": 9,
                "finetuning_closed": 5,
                "fine_tuning_union": 20,
            },
            verified_at="2026-06-17T00:00:00Z",
        ),
        keywords_skills_facet_expansion=EmpiricalValue.verified(
            fact=MatchingFact.KEYWORDS_SKILLS_FACET_EXPANSION,
            value=KeywordsSkillsFacetExpansionModel.NO_KEYWORDS_EXPANSION,
            evidence={
                "saas_keyword": 25,
                "saas_or_software_as_a_service": 40,
            },
            verified_at="2026-06-17T00:00:00Z",
        ),
        boolean_operators=("AND", "OR", "NOT"),
    )

    assert contract.require_verified() is contract
    payload = contract.to_dict()
    assert payload["schema_version"] == "linkedin.matching_contract.v1"
    assert payload["last_empirically_verified"] == "2026-06-17T00:00:00Z"
    assert payload["keyword_stemming"]["status"] == "verified"
    assert payload["keyword_stemming"]["value"] == "no_stemming"
    assert payload["hyphen_space_tokenization"]["value"] == "distinct"
    assert payload["keywords_skills_facet_expansion"]["value"] == (
        "no_keywords_expansion"
    )
    assert payload["boolean_operators"] == ["AND", "OR", "NOT"]


def test_verified_empirical_values_normalize_verified_at_timestamp() -> None:
    value = EmpiricalValue.verified(
        fact=MatchingFact.KEYWORD_STEMMING,
        value=KeywordStemmingModel.NO_STEMMING,
        evidence={"benchmark": 10},
        verified_at=" 2026-06-17T00:00:00Z ",
    )

    assert value.verified_at == "2026-06-17T00:00:00Z"
    assert value.to_dict()["verified_at"] == "2026-06-17T00:00:00Z"


def test_build_verified_contract_from_conclusive_seat_counts() -> None:
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
        verified_at="2026-06-17T00:00:00Z",
    )

    assert contract.require_verified() is contract
    payload = contract.to_dict()
    assert payload["keyword_stemming"]["value"] == "no_stemming"
    assert payload["hyphen_space_tokenization"]["value"] == "distinct"
    assert payload["keywords_skills_facet_expansion"]["value"] == (
        "no_keywords_expansion"
    )
    assert payload["keyword_stemming"]["evidence"]["counts"] == {
        "benchmark": 100,
        "benchmarks": 80,
        "benchmark_or_benchmarks": 170,
    }


def test_build_verified_contract_can_represent_collapsed_evidence() -> None:
    contract = build_verified_contract_from_seat_test_counts(
        {
            "benchmark": 100,
            "benchmarks": 100,
            "benchmark_or_benchmarks": 100,
            "fine_tuning_hyphenated": 55,
            "fine_tuning_spaced": 55,
            "finetuning_closed": 55,
            "fine_tuning_union": 55,
            "saas_keyword": 25,
            "saas_or_software_as_a_service": 25,
        },
        verified_at="2026-06-17T00:00:00Z",
    )

    payload = contract.to_dict()
    assert payload["keyword_stemming"]["value"] == "stems_or_collapses"
    assert payload["hyphen_space_tokenization"]["value"] == "collapsed"
    assert payload["keywords_skills_facet_expansion"]["value"] == (
        "expands_or_collapses"
    )


def test_build_verified_contract_rejects_missing_or_incoherent_counts() -> None:
    with pytest.raises(SeatTestEvidenceError, match="counts must be an object"):
        build_verified_contract_from_seat_test_counts(
            ["benchmark", "benchmarks"],
            verified_at="2026-06-17T00:00:00Z",
        )

    with pytest.raises(SeatTestEvidenceError, match="fine_tuning_union"):
        build_verified_contract_from_seat_test_counts(
            {
                "benchmark": 100,
                "benchmarks": 80,
                "benchmark_or_benchmarks": 170,
            },
            verified_at="2026-06-17T00:00:00Z",
        )

    with pytest.raises(SeatTestEvidenceError, match="must be an integer"):
        build_verified_contract_from_seat_test_counts(
            {
                "benchmark": 100,
                "benchmarks": 80,
                "benchmark_or_benchmarks": 170,
                "fine_tuning_hyphenated": 40,
                "fine_tuning_spaced": True,
                "finetuning_closed": 35,
                "fine_tuning_union": 120,
                "saas_keyword": 25,
                "saas_or_software_as_a_service": 40,
            },
            verified_at="2026-06-17T00:00:00Z",
        )

    with pytest.raises(SeatTestEvidenceError, match="SaaS"):
        build_verified_contract_from_seat_test_counts(
            {
                "benchmark": 100,
                "benchmarks": 80,
                "benchmark_or_benchmarks": 170,
                "fine_tuning_hyphenated": 40,
                "fine_tuning_spaced": 55,
                "finetuning_closed": 35,
                "fine_tuning_union": 120,
                "saas_keyword": 40,
                "saas_or_software_as_a_service": 25,
            },
            verified_at="2026-06-17T00:00:00Z",
        )

    with pytest.raises(
        SeatTestEvidenceError,
        match="Unexpected Recruiter seat-test counts: benchmark_near_operator",
    ):
        build_verified_contract_from_seat_test_counts(
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
                "benchmark_near_operator": 12,
            },
            verified_at="2026-06-17T00:00:00Z",
        )

    with pytest.raises(
        SeatTestEvidenceError,
        match="Seat-test count key 12 must be a string",
    ):
        build_verified_contract_from_seat_test_counts(
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
                12: 12,
            },
            verified_at="2026-06-17T00:00:00Z",
        )

    with pytest.raises(ValueError, match="ISO timestamp"):
        build_verified_contract_from_seat_test_counts(
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
            verified_at="not-a-timestamp",
        )

    with pytest.raises(ValueError, match="verified_at must be a string ISO timestamp"):
        build_verified_contract_from_seat_test_counts(
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
            verified_at=20260617,
        )


def test_build_verified_contract_normalizes_verified_at_timestamp() -> None:
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
        verified_at=" 2026-06-17T00:00:00Z ",
    )

    payload = contract.to_dict()
    assert payload["last_empirically_verified"] == "2026-06-17T00:00:00Z"
    assert payload["keyword_stemming"]["verified_at"] == "2026-06-17T00:00:00Z"
    assert payload["hyphen_space_tokenization"]["verified_at"] == (
        "2026-06-17T00:00:00Z"
    )
    assert payload["keywords_skills_facet_expansion"]["verified_at"] == (
        "2026-06-17T00:00:00Z"
    )


def test_matching_guidance_is_contract_owned_and_labels_unverified_facts() -> None:
    strategy_guidance = render_strategy_matching_guidance()
    adaptation_guidance = render_adaptation_matching_guidance()
    repair_hint = conservative_morphology_repair_hint()

    assert "Contract status: unverified" in strategy_guidance
    assert "pending Recruiter seat tests" in strategy_guidance
    assert "Substring behavior is not a verified contract" in strategy_guidance
    assert "contract-owned, pending live verification" in adaptation_guidance
    assert "do not rely on stemming until the matching contract is verified" in repair_hint


def _write_verified_contract_artifact(path: Path) -> LinkedInMatchingContract:
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
        verified_at="2026-06-17T00:00:00Z",
    )
    path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    return contract


def test_load_persisted_matching_contract_reads_verified_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "linkedin-matching-contract.json"
    _write_verified_contract_artifact(artifact_path)

    loaded = load_persisted_matching_contract(artifact_path)

    assert loaded is not None
    assert loaded.require_verified() is loaded
    assert loaded.keyword_stemming.value == KeywordStemmingModel.NO_STEMMING
    assert (
        loaded.hyphen_space_tokenization.value
        == HyphenSpaceTokenizationModel.DISTINCT
    )
    assert loaded.last_empirically_verified == "2026-06-17T00:00:00Z"


def test_load_persisted_matching_contract_returns_none_when_missing_or_malformed(
    tmp_path: Path,
) -> None:
    assert load_persisted_matching_contract(tmp_path / "does-not-exist.json") is None

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{not-json", encoding="utf-8")
    assert load_persisted_matching_contract(malformed_path) is None

    incoherent_path = tmp_path / "incoherent.json"
    incoherent_path.write_text(json.dumps({"keyword_stemming": {}}), encoding="utf-8")
    assert load_persisted_matching_contract(incoherent_path) is None


def test_render_functions_load_persisted_contract_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "linkedin-matching-contract.json"
    _write_verified_contract_artifact(artifact_path)
    monkeypatch.setattr(
        matching_contract_module, "MATCHING_CONTRACT_ARTIFACT_PATH", artifact_path
    )

    strategy_guidance = render_strategy_matching_guidance()
    adaptation_guidance = render_adaptation_matching_guidance()

    assert "Contract status: verified" in strategy_guidance
    assert "lastEmpiricallyVerified: 2026-06-17T00:00:00Z" in strategy_guidance
    assert "Contract status: verified" in adaptation_guidance


def test_render_functions_fall_back_to_unverified_when_artifact_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        matching_contract_module,
        "MATCHING_CONTRACT_ARTIFACT_PATH",
        tmp_path / "does-not-exist.json",
    )

    assert "Contract status: unverified" in render_strategy_matching_guidance()
    assert "Contract status: unverified" in render_adaptation_matching_guidance()


def test_m1b_linkedin_production_code_does_not_hardcode_matching_facts() -> None:
    production_files = sorted(
        path
        for path in Path("linkedin").rglob("*.py")
        if path.name not in {"matching_contract.py"}
    )
    forbidden_fragments = (
        "LinkedIn does NOT stem",
        "LinkedIn does not stem",
        "LinkedIn IS substring-embedded",
        "Substring-embedded.",
        "never add superstrings",
        "hyphen/space tokenization",
        "keywords field stems",
        "skills-facet expansion",
    )

    for path in production_files:
        source = path.read_text()
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{path} hardcodes {fragment!r}"

    with pytest.raises(SeatTestEvidenceError, match="cannot be lower"):
        build_verified_contract_from_seat_test_counts(
            {
                "benchmark": 100,
                "benchmarks": 80,
                "benchmark_or_benchmarks": 99,
                "fine_tuning_hyphenated": 40,
                "fine_tuning_spaced": 55,
                "finetuning_closed": 35,
                "fine_tuning_union": 120,
                "saas_keyword": 25,
                "saas_or_software_as_a_service": 40,
            },
            verified_at="2026-06-17T00:00:00Z",
        )
