from __future__ import annotations

import pytest

from market_intelligence.provenance import (
    EvidenceRef,
    GroundednessStatus,
    MarketClaim,
    evaluate_claim_groundedness,
    ground_market_claims,
)


def test_grounded_market_claim_carries_typed_evidence_and_verdict() -> None:
    report = ground_market_claims(
        [
            MarketClaim(
                claim_id="claim-asset-management",
                text="Asset management research copilot candidates convert strongly",
                evidence_refs=(
                    EvidenceRef(
                        source_id="run:linkedin:123",
                        source_type="run_snapshot",
                        locator="/output/runs/linkedin/123/candidates.jsonl",
                        quote=(
                            "Asset management research copilot candidates convert "
                            "strongly in saved outcomes."
                        ),
                    ),
                ),
            )
        ]
    )

    assert report.status == "ok"
    claim = report.grounded_claims[0]
    assert claim.groundedness is not None
    assert claim.groundedness.status == GroundednessStatus.GROUNDED
    assert claim.evidence_refs[0].source_type == "run_snapshot"

    payload = report.to_dict()
    assert payload["claims"][0]["groundedness"]["status"] == "grounded"
    assert payload["claims"][0]["evidence_refs"][0]["source_id"] == "run:linkedin:123"


def test_ungrounded_market_claim_is_quarantined_not_dropped() -> None:
    report = ground_market_claims(
        [
            {
                "claim_id": "claim-web3",
                "claim": "Web3 gaming profiles dominate the talent pool",
                "evidence_refs": [
                    {
                        "source_id": "run:linkedin:123",
                        "source_type": "run_snapshot",
                        "locator": "/output/runs/linkedin/123/candidates.jsonl",
                        "quote": "Saved candidates repeatedly mention research copilots.",
                    }
                ],
            }
        ]
    )

    assert report.status == "quarantine"
    assert len(report.claims) == 1
    assert len(report.quarantined_claims) == 1
    quarantined = report.quarantined_claims[0]
    assert quarantined.groundedness is not None
    assert quarantined.groundedness.status == GroundednessStatus.UNGROUNDED
    assert quarantined.claim_id == "claim-web3"


def test_partial_grounding_is_quarantined_with_missing_terms() -> None:
    report = ground_market_claims(
        [
            {
                "claim_id": "claim-payments",
                "claim": "Payments research copilots are concentrated in London",
                "evidence_refs": [
                    {
                        "source_id": "web:https://example.com/jobs",
                        "source_type": "external_web",
                        "locator": "https://example.com/jobs",
                        "quote": "Payments research appears in current postings.",
                    }
                ],
            }
        ]
    )

    claim = report.quarantined_claims[0]
    assert claim.groundedness is not None
    assert claim.groundedness.status == GroundednessStatus.PARTIAL
    assert "london" in claim.groundedness.missing_terms


def test_string_evidence_refs_are_typed_but_need_support_text() -> None:
    report = ground_market_claims(
        [
            {
                "claim": "Example employer is hiring applied AI leaders",
                "evidence_refs": ["web:https://example.com/jobs"],
            }
        ]
    )

    claim = report.quarantined_claims[0]
    assert claim.evidence_refs[0].source_type == "web"
    assert claim.evidence_refs[0].locator == "https://example.com/jobs"
    assert claim.groundedness is not None
    assert claim.groundedness.status == GroundednessStatus.UNGROUNDED


def test_mapping_evidence_refs_reject_non_string_typed_fields() -> None:
    with pytest.raises(ValueError, match="evidence ref source_id must be a string"):
        EvidenceRef.from_value(
            {
                "source_id": 123,
                "source_type": "run_snapshot",
                "locator": "/output/runs/linkedin/123/candidates.jsonl",
            }
        )

    with pytest.raises(ValueError, match="evidence ref source_type must be a string"):
        EvidenceRef.from_value(
            {
                "source_id": "run:linkedin:123",
                "source_type": ["run_snapshot"],
                "locator": "/output/runs/linkedin/123/candidates.jsonl",
            }
        )

    with pytest.raises(ValueError, match="evidence ref locator must be a string"):
        EvidenceRef.from_value(
            {
                "source_id": "run:linkedin:123",
                "source_type": "run_snapshot",
                "locator": {"path": "/output/runs/linkedin/123/candidates.jsonl"},
            }
        )

    with pytest.raises(ValueError, match="evidence ref quote must be a string"):
        EvidenceRef.from_value(
            {
                "source_id": "run:linkedin:123",
                "source_type": "run_snapshot",
                "locator": "/output/runs/linkedin/123/candidates.jsonl",
                "quote": ["not", "a", "string"],
            }
        )

    with pytest.raises(ValueError, match="evidence ref metadata must be an object"):
        EvidenceRef.from_value(
            {
                "source_id": "run:linkedin:123",
                "source_type": "run_snapshot",
                "locator": "/output/runs/linkedin/123/candidates.jsonl",
                "metadata": ["not", "an", "object"],
            }
        )


def test_mapping_evidence_refs_reject_non_mapping_entries() -> None:
    with pytest.raises(ValueError, match="evidence ref string must be non-empty"):
        EvidenceRef.from_value("   ")

    with pytest.raises(ValueError, match="evidence ref must be a string or object"):
        EvidenceRef.from_value(123)


def test_market_claim_mapping_rejects_non_string_typed_fields() -> None:
    with pytest.raises(ValueError, match="market claim must be an object"):
        MarketClaim.from_mapping(["not", "an", "object"], default_id="claim-default")

    with pytest.raises(ValueError, match="claims must be a sequence"):
        ground_market_claims("not-a-claim-list")

    with pytest.raises(ValueError, match=r"claims\[0\] must be an object"):
        ground_market_claims([123])

    with pytest.raises(ValueError, match="claim_id must be a string"):
        MarketClaim.from_mapping(
            {
                "claim_id": 123,
                "claim": "Applied AI leaders are concentrated in London",
                "evidence_refs": [],
            },
            default_id="claim-default",
        )


def test_groundedness_rejects_invalid_minimum_term_coverage() -> None:
    claim = MarketClaim(
        claim_id="claim-unsupported",
        text="Payments research copilots cluster in London",
        evidence_refs=(
            EvidenceRef(
                source_id="web:https://example.com/jobs",
                source_type="external_web",
                locator="https://example.com/jobs",
                quote="Unrelated evidence about compiler tooling.",
            ),
        ),
    )

    for value in (-0.1, 1.1, True, "0.6", float("inf")):
        with pytest.raises(
            ValueError,
            match="minimum_term_coverage must be a number between 0 and 1",
        ):
            evaluate_claim_groundedness(claim, minimum_term_coverage=value)

    with pytest.raises(
        ValueError,
        match="minimum_term_coverage must be a number between 0 and 1",
    ):
        ground_market_claims([claim], minimum_term_coverage=-0.1)


def test_evidence_ref_support_text_uses_only_textual_metadata() -> None:
    report = ground_market_claims(
        [
            MarketClaim(
                claim_id="claim-2026",
                text="2026 hiring surge",
                evidence_refs=(
                    EvidenceRef(
                        source_id="web:https://example.com/jobs",
                        source_type="external_web",
                        locator="https://example.com/jobs",
                        metadata={"year": 2026, "note": ["hiring surge"]},
                    ),
                ),
            )
        ]
    )

    claim = report.quarantined_claims[0]
    assert claim.groundedness is not None
    assert claim.groundedness.status == GroundednessStatus.UNGROUNDED
    assert set(claim.groundedness.missing_terms) == {"2026", "hiring", "surge"}

    with pytest.raises(ValueError, match="claim text must be a string"):
        MarketClaim.from_mapping(
            {
                "claim_id": "claim-ai-leaders",
                "claim": ["Applied AI leaders are concentrated in London"],
                "evidence_refs": [],
            },
            default_id="claim-default",
        )

    with pytest.raises(ValueError, match="metadata must be an object"):
        MarketClaim.from_mapping(
            {
                "claim_id": "claim-ai-leaders",
                "claim": "Applied AI leaders are concentrated in London",
                "evidence_refs": [],
                "metadata": ["not", "an", "object"],
            },
            default_id="claim-default",
        )
