"""Prompt regression runner — Phase 2 tests.

Pins the contract for :mod:`tools.run_prompt_regression`'s scoring
functions. Tests use ``score_predictions`` + ``parse_predicted_marker``
directly with fixture data so they don't need a Langfuse client.
"""

from __future__ import annotations

import pytest

from tools.run_prompt_regression import (
    RECOGNIZED_MARKER_VALUES,
    RESPONSE_EXTRACTORS,
    RegressionReport,
    extract_predicted_marker,
    parse_predicted_marker,
    run_regression_against_dataset,
    score_predictions,
)


# ---------------------------------------------------------------------------
# parse_predicted_marker — tolerant LLM output parser
# ---------------------------------------------------------------------------


class TestParsePredictedMarker:
    def test_extracts_marker_from_dict_with_judgment_accuracy_key(self) -> None:
        assert (
            parse_predicted_marker({"judgment_accuracy": "useful"}) == "useful"
        )

    def test_extracts_marker_from_dict_with_marker_alias(self) -> None:
        assert parse_predicted_marker({"marker": "wrong"}) == "wrong"

    def test_extracts_marker_from_dict_with_prediction_alias(self) -> None:
        assert parse_predicted_marker({"prediction": "off_rubric"}) == (
            "off_rubric"
        )

    def test_normalizes_case_and_whitespace(self) -> None:
        assert parse_predicted_marker({"judgment_accuracy": " USEFUL "}) == (
            "useful"
        )

    def test_returns_none_for_dict_with_unrecognized_value(self) -> None:
        assert (
            parse_predicted_marker({"judgment_accuracy": "completely_off"})
            is None
        )

    def test_returns_none_for_dict_without_marker_key(self) -> None:
        assert parse_predicted_marker({"some_other_key": "useful"}) is None

    def test_extracts_marker_from_bare_string(self) -> None:
        assert parse_predicted_marker("useful") == "useful"
        assert parse_predicted_marker("WRONG") == "wrong"

    def test_extracts_marker_from_string_containing_token(self) -> None:
        """Real LLM outputs sometimes wrap the marker in narrative
        ('the recruiter would call this useful'). The parser pulls
        out a recognized token rather than failing strict-match."""

        out = parse_predicted_marker(
            "After review, the recruiter's read here is wrong."
        )
        assert out == "wrong"

    def test_returns_none_for_unknown_string(self) -> None:
        assert parse_predicted_marker("just a generic response") is None

    def test_returns_none_for_unsupported_types(self) -> None:
        assert parse_predicted_marker(None) is None
        assert parse_predicted_marker(42) is None
        assert parse_predicted_marker(["useful"]) is None


class TestExtractorModes:
    def test_json_extractor_accepts_structured_output_only(self) -> None:
        assert (
            extract_predicted_marker(
                {"judgment_accuracy": "useful"},
                response_extractor="judgment_accuracy_json",
            )
            == "useful"
        )
        assert (
            extract_predicted_marker(
                '{"judgment_accuracy": "wrong"}',
                response_extractor="judgment_accuracy_json",
            )
            == "wrong"
        )
        assert (
            extract_predicted_marker(
                "the recruiter would call this wrong",
                response_extractor="judgment_accuracy_json",
            )
            is None
        )

    def test_text_extractor_accepts_free_text_marker(self) -> None:
        assert (
            extract_predicted_marker(
                "After review, this looks off_rubric.",
                response_extractor="judgment_accuracy_text",
            )
            == "off_rubric"
        )


# ---------------------------------------------------------------------------
# score_predictions — aggregation math
# ---------------------------------------------------------------------------


class TestScorePredictions:
    def test_empty_pairs_returns_zero_aggregate(self) -> None:
        report = score_predictions(paired_outcomes=[])
        assert report.rows_evaluated == 0
        assert report.aggregate_agreement_rate == 0.0

    def test_perfect_agreement_yields_1_0_aggregate(self) -> None:
        pairs = [
            ("useful", "useful"),
            ("wrong", "wrong"),
            ("off_rubric", "off_rubric"),
        ]
        report = score_predictions(paired_outcomes=pairs)
        assert report.rows_evaluated == 3
        assert report.aggregate_agreement_rate == 1.0

    def test_zero_agreement_yields_0_0_aggregate(self) -> None:
        pairs = [
            ("useful", "wrong"),
            ("wrong", "useful"),
        ]
        report = score_predictions(paired_outcomes=pairs)
        assert report.aggregate_agreement_rate == 0.0

    def test_partial_agreement(self) -> None:
        pairs = [
            ("useful", "useful"),  # match
            ("wrong", "useful"),  # miss
            ("useful", "useful"),  # match
            ("wrong", "wrong"),  # match
        ]
        report = score_predictions(paired_outcomes=pairs)
        assert report.rows_evaluated == 4
        assert report.aggregate_agreement_rate == 0.75

    def test_weighted_agreement_grants_partial_credit_for_pinned_pairs(self) -> None:
        pairs = [
            ("wrong", "off_rubric"),  # 0.5
            ("overstated_depth", "understated_depth"),  # 0.5
            ("useful", "useful"),  # 1.0
            ("useful", "wrong"),  # 0.0
        ]
        report = score_predictions(paired_outcomes=pairs)
        assert report.aggregate_agreement_rate == 0.25
        assert report.aggregate_weighted_agreement_rate == 0.5

    def test_per_marker_precision_and_recall(self) -> None:
        """Precision = TP / (TP + FP); recall = TP / (TP + FN).

        Setup:
        - "useful" — 2 expected, 2 predicted, 2 match (P=1, R=1)
        - "wrong" — 2 expected, 1 predicted (matched), 1 miss
          (predicted "off_rubric" instead) → P=1 (1/(1+0)), R=0.5
          (1/(1+1))
        - "off_rubric" — 0 expected, 1 predicted (the one that
          should have been "wrong") → P=0, R undefined → 0.
        """

        pairs = [
            ("useful", "useful"),
            ("useful", "useful"),
            ("wrong", "wrong"),
            ("wrong", "off_rubric"),
        ]
        report = score_predictions(paired_outcomes=pairs)

        assert report.per_marker_precision["useful"] == 1.0
        assert report.per_marker_recall["useful"] == 1.0
        assert report.per_marker_precision["wrong"] == 1.0
        assert report.per_marker_recall["wrong"] == 0.5
        assert report.per_marker_precision["off_rubric"] == 0.0
        # off_rubric had no expected support → recall denominator is
        # zero → recall == 0 by the helper's convention.
        assert report.per_marker_recall["off_rubric"] == 0.0

        assert report.per_marker_support["useful"] == 2
        assert report.per_marker_support["wrong"] == 2
        # off_rubric never appears in expected → not in support.
        assert "off_rubric" not in report.per_marker_support

    def test_confusion_matrix_records_per_cell_counts(self) -> None:
        pairs = [
            ("useful", "useful"),
            ("useful", "useful"),
            ("wrong", "useful"),
            ("wrong", "off_rubric"),
        ]
        report = score_predictions(paired_outcomes=pairs)

        assert report.confusion_matrix["useful"]["useful"] == 2
        assert report.confusion_matrix["wrong"]["useful"] == 1
        assert report.confusion_matrix["wrong"]["off_rubric"] == 1

    def test_metadata_propagates_to_report(self) -> None:
        report = score_predictions(
            paired_outcomes=[("useful", "useful")],
            cost_usd_total=12.34567,
            prompt_id="chief-of-staff-synthesis-v1",
            prompt_label="experimental",
            dataset_name="judgment-accuracy-linkedin-brief-x",
            rows_skipped_unparseable_output=3,
        )
        assert report.prompt_id == "chief-of-staff-synthesis-v1"
        assert report.prompt_label == "experimental"
        assert report.dataset_name == "judgment-accuracy-linkedin-brief-x"
        assert report.rows_skipped_unparseable_output == 3
        assert report.aggregate_cost_usd == 12.3457  # rounded to 4 places

    def test_breakdowns_by_capture_mode_and_cascade_route(self) -> None:
        report = score_predictions(
            paired_outcomes=[
                ("useful", "useful"),
                ("wrong", "off_rubric"),
                ("useful", "wrong"),
            ],
            row_metadata=[
                {
                    "capture_mode": "captured_prompt",
                    "cascade_route_hit": "clean",
                },
                {
                    "capture_mode": "captured_prompt",
                    "cascade_route_hit": "schema_invalid",
                },
                {
                    "capture_mode": "legacy_summary_fallback",
                    "cascade_route_hit": None,
                },
            ],
        )

        assert report.rows_by_capture_mode == {
            "captured_prompt": 2,
            "legacy_summary_fallback": 1,
        }
        assert report.rows_by_cascade_route == {
            "clean": 1,
            "schema_invalid": 1,
            "unlinked": 1,
        }
        assert report.agreement_rate_by_cascade_route["clean"] == 1.0
        assert report.weighted_agreement_rate_by_cascade_route["schema_invalid"] == 0.5
        assert report.agreement_rate_by_cascade_route["unlinked"] == 0.0

    def test_to_dict_round_trips(self) -> None:
        """The CLI emits the report as JSON; ``to_dict`` must produce
        a JSON-serializable shape."""

        import json

        report = score_predictions(
            paired_outcomes=[
                ("useful", "useful"),
                ("wrong", "useful"),
            ],
            prompt_id="p",
            dataset_name="d",
        )
        payload = report.to_dict()
        # All keys present + JSON-serializable.
        json.dumps(payload, sort_keys=True)
        assert payload["aggregate_agreement_rate"] == 0.5
        assert "per_marker_precision" in payload
        assert "confusion_matrix" in payload


# ---------------------------------------------------------------------------
# Recognized marker enum stays stable
# ---------------------------------------------------------------------------


def test_recognized_marker_values_match_writer_validated_set() -> None:
    """Pinned in shared/runtime_state/store.py:767-773. If the writer
    bumps the enum, the regression runner needs an explicit update."""

    assert RECOGNIZED_MARKER_VALUES == frozenset(
        {"useful", "wrong", "off_rubric", "overstated_depth", "understated_depth"}
    )


def test_response_extractors_enum_stays_stable() -> None:
    assert RESPONSE_EXTRACTORS == frozenset(
        {"judgment_accuracy_json", "judgment_accuracy_text"}
    )


# ---------------------------------------------------------------------------
# RegressionReport dataclass
# ---------------------------------------------------------------------------


def test_regression_report_default_fields() -> None:
    report = RegressionReport(
        prompt_id="p",
        prompt_label="production",
        dataset_name="d",
        rows_evaluated=0,
        rows_skipped_unparseable_output=0,
        aggregate_agreement_rate=0.0,
    )
    # Default factories produce empty dicts.
    assert report.per_marker_precision == {}
    assert report.per_marker_recall == {}
    assert report.per_marker_support == {}
    assert report.confusion_matrix == {}
    assert report.aggregate_cost_usd == 0.0
    assert report.aggregate_weighted_agreement_rate == 0.0
    assert report.rows_by_capture_mode == {}
    assert report.rows_by_cascade_route == {}


class _FakeDataset:
    def __init__(self, items):
        self.items = items


class _FakeItem:
    def __init__(self, item_id: str, input: dict, expected_output: dict, metadata: dict):
        self.id = item_id
        self.input = input
        self.expected_output = expected_output
        self.metadata = metadata


class _FakeTextPrompt:
    def compile(self, **kwargs):
        return f"Judge this candidate:\n{kwargs['candidate_text']}"


class _FakeChatPrompt:
    def compile(self, **kwargs):
        return [{"role": "user", "content": kwargs["candidate_text"]}]


class _FakeInner:
    def __init__(self, prompt, items):
        self._prompt = prompt
        self._dataset = _FakeDataset(items)

    def get_prompt(self, prompt_id: str, label: str = "production"):
        assert prompt_id == "prompt-x"
        assert label == "production"
        return self._prompt

    def get_dataset(self, name: str):
        assert name == "dataset-x"
        return self._dataset


def test_run_regression_compiles_text_prompt_and_uses_json_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_prompts: list[tuple[str, str]] = []

    def _llm_caller(system_prompt: str, user_prompt: str):
        captured_prompts.append((system_prompt, user_prompt))
        return {"judgment_accuracy": "useful"}

    fake_client = type("FakeClient", (), {"_inner": _FakeInner(
        _FakeTextPrompt(),
        [
            _FakeItem(
                "row-1",
                {"candidate_text": "Compiled candidate", "candidate_summary": "Summary"},
                {"judgment_accuracy": "useful"},
                {"capture_mode": "captured_prompt", "cascade_route_hit": "clean"},
            )
        ],
    )})()

    import shared.observability
    import shared.observability.langfuse_client

    monkeypatch.setattr(shared.observability, "is_active", lambda: True)
    monkeypatch.setattr(
        shared.observability.langfuse_client,
        "get_client",
        lambda: fake_client,
    )

    report = run_regression_against_dataset(
        prompt_id="prompt-x",
        dataset_name="dataset-x",
        prompt_label="production",
        response_extractor="judgment_accuracy_json",
        llm_caller=_llm_caller,
    )

    assert captured_prompts == [("", "Judge this candidate:\nCompiled candidate")]
    assert report.rows_evaluated == 1
    assert report.aggregate_agreement_rate == 1.0
    assert report.rows_by_capture_mode == {"captured_prompt": 1}


def test_run_regression_rejects_chat_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = type("FakeClient", (), {"_inner": _FakeInner(
        _FakeChatPrompt(),
        [
            _FakeItem(
                "row-1",
                {"candidate_text": "Compiled candidate"},
                {"judgment_accuracy": "useful"},
                {},
            )
        ],
    )})()

    import shared.observability
    import shared.observability.langfuse_client

    monkeypatch.setattr(shared.observability, "is_active", lambda: True)
    monkeypatch.setattr(
        shared.observability.langfuse_client,
        "get_client",
        lambda: fake_client,
    )

    with pytest.raises(RuntimeError, match="chat prompts are not supported"):
        run_regression_against_dataset(
            prompt_id="prompt-x",
            dataset_name="dataset-x",
            prompt_label="production",
        )
