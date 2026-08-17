"""Tests for the structured-filter surface receipt (production-truth instrument)."""

from __future__ import annotations

from types import SimpleNamespace

from linkedin import surface_receipt as sr


def _ss(**kwargs):
    base = dict(id=1, name="s", acquisition_mode="", surface="", structured_filters={})
    base.update(kwargs)
    return SimpleNamespace(**base)


def _result(**kwargs):
    base = dict(
        applied_controls=[],
        failed_controls=[],
        unsupported_controls=[],
        fallback_to_boolean=False,
        plan_fully_applied=True,
        reason="",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_keyword_string_is_not_hybrid():
    row = sr.intended_surface_for_string(_ss(acquisition_mode="linkedin_boolean"))
    assert row["is_hybrid"] is False
    assert row["structured_dimensions"] == {}


def test_company_filter_string_is_hybrid_with_dimension():
    ss = _ss(
        acquisition_mode="linkedin_hybrid",
        structured_filters={"companies": ["Nubank", "Bancolombia"], "titles": []},
    )
    row = sr.intended_surface_for_string(ss)
    assert row["is_hybrid"] is True
    assert row["structured_dimensions"]["companies"] == ["Nubank", "Bancolombia"]
    assert "titles" not in row["structured_dimensions"]


def test_location_under_sidebar_is_surfaced():
    ss = _ss(
        acquisition_mode="linkedin_hybrid",
        structured_filters={"sidebar_filters": {"locations": ["Colombia"]}},
    )
    row = sr.intended_surface_for_string(ss)
    assert row["structured_dimensions"]["locations"] == ["Colombia"]
    assert row["is_hybrid"] is True


def test_hybrid_mode_without_dimensions_is_not_hybrid():
    # Mode can lie (Fix 2 honesty): dimensions are the ground truth.
    ss = _ss(acquisition_mode="linkedin_hybrid", structured_filters={})
    assert sr.intended_surface_for_string(ss)["is_hybrid"] is False


def test_summary_counts_hybrid_and_dimensions():
    strings = [
        _ss(id=1, acquisition_mode="linkedin_boolean"),
        _ss(id=2, acquisition_mode="linkedin_hybrid", structured_filters={"companies": ["X"]}),
        _ss(id=3, acquisition_mode="linkedin_hybrid", structured_filters={"titles": ["Y"]}),
    ]
    summary = sr.summarize_intended_surfaces(strings)
    assert summary["total_strings"] == 3
    assert summary["hybrid_strings"] == 2
    assert summary["dimension_counts"] == {"companies": 1, "titles": 1}
    assert summary["normalization_guard_counts"] == {
        "ubiquitous_and_gate": 0,
        "token_subset_superstring_pruned": 0,
    }
    # format is total-first and lists only hybrid rows
    text = sr.format_intended_summary(summary)
    assert "2/3" in text
    assert (
        "normalization guards: ubiquitous_and_gate=0, "
        "token_subset_superstring_pruned=0"
    ) in text


def test_apply_receipt_company_applied_is_not_fallback():
    fields = sr.apply_receipt_fields(
        _ss(id=7, acquisition_mode="linkedin_hybrid"),
        _result(applied_controls=["companies", "keywords"], plan_fully_applied=True),
    )
    assert fields["structured_applied"] == ["companies"]
    assert fields["fell_back_to_keyword"] is False
    assert "APPLIED" in sr.format_apply_receipt(fields)


def test_boolean_filter_overlap_flags_duplicated_company():
    # Kernel harmony violation: a company on a filter AND in the keyword Boolean.
    overlap = sr.boolean_filter_overlap(
        '("Nubank" OR "Bancolombia") AND ("ML")', {"companies": ["Nubank"]}
    )
    assert overlap == ["Nubank"]
    # Location overlap is intentionally ignored here (handled in Slice 3).
    assert sr.boolean_filter_overlap(
        "(Colombia OR Bogota)", {"locations": ["Colombia"]}
    ) == []


def test_intended_row_carries_boolean_overlap():
    ss = _ss(
        acquisition_mode="linkedin_hybrid",
        boolean='("Nubank" OR "ML")',
        structured_filters={"companies": ["Nubank"]},
    )
    row = sr.intended_surface_for_string(ss)
    assert row["boolean_filter_overlap"] == ["Nubank"]


def test_summary_reports_boolean_normalization_guard_findings():
    strings = [
        _ss(
            id=1,
            boolean_normalization={
                "changed": False,
                "original_boolean": '("Python") AND ("PyTorch")',
                "normalized_boolean": '("Python") AND ("PyTorch")',
                "findings": [
                    {
                        "code": "ubiquitous_and_gate",
                        "message": "AND clause is composed entirely of ubiquitous terms.",
                        "terms": ["python", "pytorch"],
                    }
                ],
            },
        ),
        _ss(
            id=2,
            boolean_normalization={
                "changed": True,
                "original_boolean": '("reward model" OR "reward model development")',
                "normalized_boolean": '("reward model")',
                "findings": [
                    {
                        "code": "token_subset_superstring_pruned",
                        "message": "Superstring terms were pruned.",
                        "terms": ["reward model development"],
                    }
                ],
            },
        ),
    ]

    summary = sr.summarize_intended_surfaces(strings)

    assert summary["normalization_strings_with_findings"] == 2
    assert summary["normalization_guard_counts"] == {
        "ubiquitous_and_gate": 1,
        "token_subset_superstring_pruned": 1,
    }
    text = sr.format_intended_summary(summary)
    assert (
        "normalization guards: ubiquitous_and_gate=1, "
        "token_subset_superstring_pruned=1"
    ) in text
    assert summary["rows"][1]["boolean_normalization"]["finding_terms"] == {
        "token_subset_superstring_pruned": ["reward model development"]
    }


def test_apply_receipt_dropped_structured_is_fallback():
    fields = sr.apply_receipt_fields(
        _ss(id=8, acquisition_mode="linkedin_hybrid"),
        _result(
            applied_controls=["keywords"],
            unsupported_controls=["companies"],
            fallback_to_boolean=True,
            plan_fully_applied=False,
        ),
    )
    assert fields["structured_applied"] == []
    assert fields["fell_back_to_keyword"] is True
    assert "FELL BACK" in sr.format_apply_receipt(fields)


def test_apply_receipt_carries_boolean_normalization_summary():
    fields = sr.apply_receipt_fields(
        _ss(
            id=9,
            acquisition_mode="linkedin_hybrid",
            boolean_normalization={
                "changed": True,
                "findings": [
                    {
                        "code": "token_subset_superstring_pruned",
                        "message": "Superstring terms were pruned.",
                        "terms": ["reward model development"],
                    }
                ],
            },
        ),
        _result(applied_controls=["keywords"], plan_fully_applied=True),
    )

    assert fields["boolean_normalization"]["guard_counts"] == {
        "ubiquitous_and_gate": 0,
        "token_subset_superstring_pruned": 1,
    }


def test_apply_receipt_counts_values_per_dimension_when_applied():
    """P2.2: per-dimension value counts — applied dimension counts all values."""
    fields = sr.apply_receipt_fields(
        _ss(
            id=10,
            acquisition_mode="linkedin_hybrid",
            structured_filters={"companies": ["Nubank", "Bancolombia"], "titles": ["Staff Engineer"]},
        ),
        _result(
            applied_controls=["companies", "keywords"],
            unsupported_controls=["titles"],
            plan_fully_applied=False,
        ),
    )
    assert fields["requested_value_counts"] == {"companies": 2, "titles": 1}
    assert fields["applied_value_counts"] == {"companies": 2, "titles": 0}
    assert fields["fell_back_to_keyword"] is False


def test_apply_receipt_counts_zero_applied_on_fallback():
    """P2.2: a keyword-only fallback applies zero values on every dimension."""
    fields = sr.apply_receipt_fields(
        _ss(
            id=11,
            acquisition_mode="linkedin_hybrid",
            structured_filters={"companies": ["Nubank", "Rappi", "Mercado Libre"]},
        ),
        _result(
            applied_controls=["keywords"],
            unsupported_controls=["companies"],
            fallback_to_boolean=True,
            plan_fully_applied=False,
        ),
    )
    assert fields["requested_value_counts"] == {"companies": 3}
    assert fields["applied_value_counts"] == {"companies": 0}
    assert fields["fell_back_to_keyword"] is True
