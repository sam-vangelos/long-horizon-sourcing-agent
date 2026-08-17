#!/usr/bin/env python3
"""Run a Langfuse prompt against a Langfuse dataset; score vs recruiter markers.

Phase 2 of Langfuse adoption. Closes the eval-loop: dataset rows
(synced via :mod:`tools.sync_judgment_datasets`) carry the recruiter's
``judgment_accuracy`` ground truth in ``expected_output``; this CLI
runs a prompt (the production version OR a candidate experimental
version) against every row, scores each output against the
recruiter's marker, and emits a regression report (precision /
recall / agreement-rate per marker, plus an aggregate accuracy).

## Usage

    python -m tools.run_prompt_regression \\
        --prompt-id chief-of-staff-synthesis-v1 \\
        --dataset judgment-accuracy-linkedin-frontier-ai-fde \\
        [--prompt-label production] \\
        [--max-rows 100] \\
        [--report-path output/regression_report.json]

The CLI reads Langfuse credentials from the same env vars
:mod:`shared.observability.langfuse_client` does. When credentials
are absent OR ``LANGFUSE_DISABLE=1`` the tool exits 2 with a clear
message — running without the dataset push doesn't make sense.

## Scoring contract

For each dataset row:

1. The prompt is fetched from Langfuse (versioned by ``prompt-label``,
   defaulting to ``production``).
2. The prompt is rendered with the row's ``input`` as template
   variables.
3. The rendered prompt is executed via the configured LLM caller
   (``opus_llm`` from :mod:`shared.llm_clients` by default; tests
   inject a stub).
4. The LLM's response is parsed and compared to the row's
   ``expected_output.judgment_accuracy``.

The agreement scoring is exact-match for v1: the LLM either produces
the same enum value or it doesn't. False positives + false negatives
are tracked per marker; precision / recall fall out at the aggregate
level.

## v1 limitations

- The LLM execution path assumes the prompt produces a JSON dict
  with a ``judgment_accuracy`` key. Prompts that emit free-text
  rationale need a parser shim — wired as a follow-up Phase 3 slice.
- Agreement is exact-match only. "Useful" vs "wrong" is a strict
  diff; no partial credit for "off_rubric" vs "overstated_depth"
  (both indicate Cloris was off, just on different axes). Weighted
  agreement is a Phase 3 enhancement.
- Cost-per-row is reported separately from agreement so operators
  can decide if a prompt that's 5% more accurate but 20% more
  expensive is worth shipping.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# Mirrors the canonical writer-validated set in
# ``shared/runtime_state/store.py:767-773``. Pinned here so the
# regression scoring contract can't drift if the writer enum bumps.
RECOGNIZED_MARKER_VALUES: frozenset[str] = frozenset(
    {
        "useful",
        "wrong",
        "off_rubric",
        "overstated_depth",
        "understated_depth",
    }
)
RESPONSE_EXTRACTORS: frozenset[str] = frozenset(
    {"judgment_accuracy_json", "judgment_accuracy_text"}
)
_PARTIAL_CREDIT_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"wrong", "off_rubric"}),
        frozenset({"overstated_depth", "understated_depth"}),
    }
)


@dataclass
class RegressionReport:
    """Aggregate regression scores across one prompt × one dataset.

    The CLI emits this as JSON when ``--report-path`` is set. Tests
    construct this directly to assert aggregation correctness.
    """

    prompt_id: str
    prompt_label: str
    dataset_name: str
    rows_evaluated: int
    rows_skipped_unparseable_output: int
    aggregate_agreement_rate: float
    aggregate_weighted_agreement_rate: float = 0.0
    per_marker_precision: dict[str, float] = field(default_factory=dict)
    per_marker_recall: dict[str, float] = field(default_factory=dict)
    per_marker_support: dict[str, int] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    aggregate_cost_usd: float = 0.0
    rows_by_capture_mode: dict[str, int] = field(default_factory=dict)
    rows_by_cascade_route: dict[str, int] = field(default_factory=dict)
    agreement_rate_by_cascade_route: dict[str, float] = field(default_factory=dict)
    weighted_agreement_rate_by_cascade_route: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _weighted_agreement(expected: str, predicted: str) -> float:
    if expected == predicted:
        return 1.0
    if frozenset({expected, predicted}) in _PARTIAL_CREDIT_PAIRS:
        return 0.5
    return 0.0


def _normalize_cascade_route(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return "unlinked"


def _parse_marker_from_dict(payload: dict[str, Any]) -> str | None:
    for key in ("judgment_accuracy", "marker", "prediction"):
        value = payload.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in RECOGNIZED_MARKER_VALUES:
                return normalized
    return None


def _parse_marker_from_json_output(llm_output: Any) -> str | None:
    if isinstance(llm_output, dict):
        return _parse_marker_from_dict(llm_output)
    if not isinstance(llm_output, str):
        return None
    try:
        parsed = json.loads(llm_output)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(parsed, dict):
        return _parse_marker_from_dict(parsed)
    return None


def _parse_marker_from_text_output(llm_output: Any) -> str | None:
    if isinstance(llm_output, dict):
        return _parse_marker_from_dict(llm_output)
    if isinstance(llm_output, str):
        normalized = llm_output.strip().lower()
        if normalized in RECOGNIZED_MARKER_VALUES:
            return normalized
        for marker in RECOGNIZED_MARKER_VALUES:
            if marker in normalized:
                return marker
    return None


def score_predictions(
    *,
    paired_outcomes: list[tuple[str, str]],
    row_metadata: list[dict[str, Any]] | None = None,
    cost_usd_total: float = 0.0,
    prompt_id: str = "",
    prompt_label: str = "",
    dataset_name: str = "",
    rows_skipped_unparseable_output: int = 0,
) -> RegressionReport:
    """Aggregate paired (expected, predicted) marker tuples into a report.

    Pure function; no LLM coupling. Tests call this directly with a
    fixture list of pairs; the CLI calls it after running the LLM
    over the full dataset.

    - ``aggregate_agreement_rate`` — fraction of rows where
      predicted == expected.
    - ``per_marker_precision[marker]`` — TP / (TP + FP) where TP is
      "predicted == marker AND expected == marker", FP is "predicted
      == marker AND expected != marker".
    - ``per_marker_recall[marker]`` — TP / (TP + FN) where FN is
      "expected == marker AND predicted != marker".
    - ``per_marker_support[marker]`` — count of rows where
      ``expected == marker``. Surfaces low-N markers where
      precision/recall are noisy.
    - ``confusion_matrix[expected][predicted]`` — per-cell count.
    """

    rows = len(paired_outcomes)
    if rows == 0:
        return RegressionReport(
            prompt_id=prompt_id,
            prompt_label=prompt_label,
            dataset_name=dataset_name,
            rows_evaluated=0,
            rows_skipped_unparseable_output=rows_skipped_unparseable_output,
            aggregate_agreement_rate=0.0,
            aggregate_weighted_agreement_rate=0.0,
            aggregate_cost_usd=round(cost_usd_total, 4),
        )

    agreement = sum(1 for exp, pred in paired_outcomes if exp == pred)
    aggregate = agreement / rows
    weighted_agreement_total = sum(
        _weighted_agreement(expected, predicted)
        for expected, predicted in paired_outcomes
    )
    aggregate_weighted = weighted_agreement_total / rows

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    support: Counter[str] = Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for expected, predicted in paired_outcomes:
        support[expected] += 1
        confusion[expected][predicted] += 1
        if expected == predicted:
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1

    per_marker_precision: dict[str, float] = {}
    per_marker_recall: dict[str, float] = {}
    for marker in RECOGNIZED_MARKER_VALUES | set(support) | set(fp):
        tp_n = tp[marker]
        fp_n = fp[marker]
        fn_n = fn[marker]
        precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
        recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
        per_marker_precision[marker] = round(precision, 3)
        per_marker_recall[marker] = round(recall, 3)

    confusion_serialized = {
        expected: dict(predicted_counts)
        for expected, predicted_counts in confusion.items()
    }

    rows_by_capture_mode: Counter[str] = Counter()
    rows_by_cascade_route: Counter[str] = Counter()
    agreement_by_cascade_route: Counter[str] = Counter()
    weighted_agreement_by_cascade_route: defaultdict[str, float] = defaultdict(float)
    if row_metadata:
        for (expected, predicted), metadata in zip(paired_outcomes, row_metadata):
            capture_mode = str(metadata.get("capture_mode") or "unknown")
            cascade_route = _normalize_cascade_route(
                metadata.get("cascade_route_hit")
            )
            rows_by_capture_mode[capture_mode] += 1
            rows_by_cascade_route[cascade_route] += 1
            if expected == predicted:
                agreement_by_cascade_route[cascade_route] += 1
            weighted_agreement_by_cascade_route[cascade_route] += _weighted_agreement(
                expected,
                predicted,
            )

    agreement_rate_by_cascade_route = {
        route: round(agreement_by_cascade_route[route] / count, 3)
        for route, count in rows_by_cascade_route.items()
        if count > 0
    }
    weighted_agreement_rate_by_cascade_route = {
        route: round(weighted_agreement_by_cascade_route[route] / count, 3)
        for route, count in rows_by_cascade_route.items()
        if count > 0
    }

    return RegressionReport(
        prompt_id=prompt_id,
        prompt_label=prompt_label,
        dataset_name=dataset_name,
        rows_evaluated=rows,
        rows_skipped_unparseable_output=rows_skipped_unparseable_output,
        aggregate_agreement_rate=round(aggregate, 3),
        aggregate_weighted_agreement_rate=round(aggregate_weighted, 3),
        per_marker_precision=per_marker_precision,
        per_marker_recall=per_marker_recall,
        per_marker_support=dict(support),
        confusion_matrix=confusion_serialized,
        aggregate_cost_usd=round(cost_usd_total, 4),
        rows_by_capture_mode=dict(rows_by_capture_mode),
        rows_by_cascade_route=dict(rows_by_cascade_route),
        agreement_rate_by_cascade_route=agreement_rate_by_cascade_route,
        weighted_agreement_rate_by_cascade_route=weighted_agreement_rate_by_cascade_route,
    )


def parse_predicted_marker(llm_output: Any) -> str | None:
    """Extract a recognized marker from an LLM response.

    Tolerant parser: accepts dicts (with ``judgment_accuracy`` /
    ``marker`` / ``prediction`` key), strings (case-insensitive
    match against the marker enum), or returns None on anything
    else. The score function treats None as "skipped row" — the
    regression report counts those separately so a prompt that
    fails to parse half its outputs is loudly visible.
    """

    return _parse_marker_from_text_output(llm_output)


def extract_predicted_marker(
    llm_output: Any,
    *,
    response_extractor: str,
) -> str | None:
    if response_extractor == "judgment_accuracy_json":
        return _parse_marker_from_json_output(llm_output)
    if response_extractor == "judgment_accuracy_text":
        return _parse_marker_from_text_output(llm_output)
    raise ValueError(f"unknown response_extractor: {response_extractor}")


def run_regression_against_dataset(
    *,
    prompt_id: str,
    dataset_name: str,
    prompt_label: str = "production",
    response_extractor: str = "judgment_accuracy_json",
    llm_caller: Callable[[str, str], Any] | None = None,
    max_rows: int | None = None,
) -> RegressionReport:
    """Walk a Langfuse dataset; run the prompt against each row; score.

    Live path used by the CLI. Tests use ``score_predictions`` +
    ``parse_predicted_marker`` directly with fixture data so they
    don't need a Langfuse client.
    """

    from shared.observability import is_active
    from shared.observability.langfuse_client import get_client

    if not is_active():
        raise RuntimeError(
            "Langfuse client is null / disabled / network-degraded. "
            "Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and clear "
            "LANGFUSE_DISABLE) before running the regression."
        )

    client = get_client()
    inner = getattr(client, "_inner", None)
    if inner is None:
        raise RuntimeError(
            "Langfuse client has no inner SDK reference — regression "
            "cannot proceed."
        )
    if response_extractor not in RESPONSE_EXTRACTORS:
        raise RuntimeError(
            "response_extractor must be one of "
            f"{sorted(RESPONSE_EXTRACTORS)}"
        )

    # Resolve the prompt + dataset via the SDK. Both have v2/v3 surface
    # divergence; try v3 first.
    prompt = None
    if hasattr(inner, "get_prompt"):
        prompt = inner.get_prompt(prompt_id, label=prompt_label)
    elif hasattr(inner, "api") and hasattr(
        getattr(inner, "api", None), "prompts"
    ):
        prompt = inner.api.prompts.get(name=prompt_id, label=prompt_label)
    else:
        raise RuntimeError(
            "Langfuse SDK has no recognized prompt-fetch surface."
        )
    if prompt is None or not hasattr(prompt, "compile"):
        raise RuntimeError("Langfuse prompt object does not expose compile().")

    dataset_items: list[Any] = []
    if hasattr(inner, "get_dataset"):
        dataset = inner.get_dataset(name=dataset_name)
        dataset_items = list(getattr(dataset, "items", []) or [])
    elif hasattr(inner, "api") and hasattr(
        getattr(inner, "api", None), "datasets"
    ):
        dataset = inner.api.datasets.get(name=dataset_name)
        dataset_items = list(getattr(dataset, "items", []) or [])
    else:
        raise RuntimeError(
            "Langfuse SDK has no recognized dataset-fetch surface."
        )

    if max_rows is not None:
        dataset_items = dataset_items[:max_rows]

    if llm_caller is None:
        # Default to opus_llm. Phase 3 replays the compiled Langfuse
        # text prompt as the user prompt and leaves the system empty.
        from shared.llm_clients import opus_llm

        def _default_caller(system_prompt: str, user_prompt: str) -> Any:
            return opus_llm(
                system_prompt,
                user_prompt,
                expect_json=response_extractor == "judgment_accuracy_json",
            )

        llm_caller = _default_caller

    paired: list[tuple[str, str]] = []
    row_metadata: list[dict[str, Any]] = []
    skipped = 0
    for item in dataset_items:
        input_dict = getattr(item, "input", None) or {}
        expected = (
            getattr(item, "expected_output", None) or {}
        ).get("judgment_accuracy")
        if expected not in RECOGNIZED_MARKER_VALUES:
            skipped += 1
            continue

        try:
            compiled_prompt = prompt.compile(**input_dict)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"prompt.compile failed for dataset row {getattr(item, 'id', '<no-id>')}: "
                f"{exc}"
            ) from exc
        if not isinstance(compiled_prompt, str):
            raise RuntimeError(
                "Langfuse chat prompts are not supported by this regression "
                "runner yet; use a text prompt."
            )
        try:
            llm_output = llm_caller("", compiled_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM call raised on dataset row %s: %s",
                getattr(item, "id", "<no-id>"),
                exc.__class__.__name__,
            )
            skipped += 1
            continue

        predicted = extract_predicted_marker(
            llm_output,
            response_extractor=response_extractor,
        )
        if predicted is None:
            skipped += 1
            continue
        paired.append((expected, predicted))
        row_metadata.append(dict(getattr(item, "metadata", None) or {}))

    return score_predictions(
        paired_outcomes=paired,
        row_metadata=row_metadata,
        prompt_id=prompt_id,
        prompt_label=prompt_label,
        dataset_name=dataset_name,
        rows_skipped_unparseable_output=skipped,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_prompt_regression",
        description=(
            "Run a Langfuse prompt against a Langfuse dataset; score "
            "predictions vs recruiter markers; emit precision / recall / "
            "agreement rate."
        ),
    )
    parser.add_argument(
        "--prompt-id",
        required=True,
        help="Langfuse prompt name (e.g., chief-of-staff-synthesis-v1).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Langfuse dataset name (e.g., judgment-accuracy-linkedin-<brief_id>).",
    )
    parser.add_argument(
        "--prompt-label",
        default="production",
        help="Prompt version label. Defaults to 'production'.",
    )
    parser.add_argument(
        "--response-extractor",
        default="judgment_accuracy_json",
        choices=sorted(RESPONSE_EXTRACTORS),
        help=(
            "How to extract the predicted marker from the replay output. "
            "Defaults to structured JSON parsing."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap rows evaluated. Useful for cost-bounded smoke runs.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Write the regression report JSON to this path. Stdout-only when unset.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        report = run_regression_against_dataset(
            prompt_id=args.prompt_id,
            dataset_name=args.dataset,
            prompt_label=args.prompt_label,
            response_extractor=args.response_extractor,
            max_rows=args.max_rows,
        )
    except RuntimeError as exc:
        logger.error("regression failed: %s", exc)
        return 2

    payload = report.to_dict()
    if args.report_path:
        Path(args.report_path).write_text(json.dumps(payload, indent=2))
        logger.info("wrote regression report to %s", args.report_path)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
