from github.reconciliation_report import build_reconciliation_summary


def test_build_reconciliation_summary_uses_confidence_bucket_names_not_actions():
    rows = [
        {"action": "manual_review", "match_confidence": 0.91},
        {"action": "promote", "match_confidence": 0.72},
        {"action": "drop_wrong_person", "match_confidence": 0.21},
    ]

    summary = build_reconciliation_summary(
        rows,
        input_stats={"skipped_ambiguous_name": 1},
    )

    assert summary["match_confidence_distribution"] == {
        "high_confidence": 1,
        "medium_confidence": 1,
        "low_confidence": 1,
    }
    assert summary["manual_review_rate"] == round(1 / 3, 4)
    assert summary["low_confidence_rate"] == round(1 / 3, 4)
    assert summary["input_stats"] == {"skipped_ambiguous_name": 1}
