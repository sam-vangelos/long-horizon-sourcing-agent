"""On-disk schema + writer for the shadow full-judgment comparison record.

Slice 2 of perplexity-evidence-augmentation. The orchestrator must not know
the on-disk schema; it builds a ``ShadowFullJudgmentRecord`` and hands it to
``record_shadow_full_judgment``, which appends a single JSONL line to
``shadow_final_judgments.jsonl`` (declared as ``ANALYTICAL_DEBUG`` in
``shared/runtime_state/artifacts.py``).

Hard rules:

- The writer never raises. Any I/O error is absorbed and surfaced via ``print``
  so canonical state can never depend on shadow-write success.
- ``compute_judgment_diff`` is pure and deterministic.
- Timestamps are UTC ISO-8601 strings.
- ``feature_version="slice2"`` is hard-coded; future slices bump it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from shared.schemas import OpusDecision
from shared.storage import append_jsonl


_SHADOW_ARTIFACT_NAME = "shadow_final_judgments.jsonl"


@dataclass
class ShadowFullJudgmentRecord:
    candidate_name: str
    profile_url: str
    source_string_id: int
    page: int
    result_rank: int
    trigger_reason: str
    external_evidence_status: str
    identity_confidence: float | None
    evidence_refs_count: int
    baseline: dict
    enriched: dict | None
    diff: dict
    timestamp: str
    feature_version: str = "slice2"

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_rationale(text: str | None) -> str:
    """Whitespace-normalize a rationale string for diff comparison.

    Slice-2 rule: ``.strip()`` only. Internal whitespace is signal, not noise.
    """

    if text is None:
        return ""
    return str(text).strip()


def _normalize_path(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).strip()


def compute_judgment_diff(
    baseline: OpusDecision,
    enriched: OpusDecision | None,
    *,
    skip_reason: str = "",
) -> dict:
    """Compare baseline vs enriched ``OpusDecision`` for shadow analytics.

    When ``enriched`` is ``None`` the diff is not computed; the returned dict
    explains why with a non-empty ``reason``.
    """

    if enriched is None:
        return {
            "computed": False,
            "reason": skip_reason or "no_enriched_decision",
        }

    baseline_decision = str(baseline.decision or "")
    enriched_decision = str(enriched.decision or "")
    baseline_path = _normalize_path(baseline.path)
    enriched_path = _normalize_path(enriched.path)
    baseline_rationale = _normalize_rationale(baseline.rationale)
    enriched_rationale = _normalize_rationale(enriched.rationale)

    try:
        confidence_delta = float(enriched.confidence) - float(baseline.confidence)
    except (TypeError, ValueError):
        confidence_delta = 0.0

    return {
        "computed": True,
        "decision_changed": baseline_decision != enriched_decision,
        "decision_baseline": baseline_decision,
        "decision_enriched": enriched_decision,
        "path_changed": baseline_path != enriched_path,
        "path_baseline": baseline_path,
        "path_enriched": enriched_path,
        "rationale_changed": baseline_rationale != enriched_rationale,
        "confidence_delta": confidence_delta,
    }


def record_shadow_full_judgment(
    *,
    output_dir: Path,
    record: ShadowFullJudgmentRecord,
) -> None:
    """Append a single shadow-judgment record to the analytical-debug file.

    Never raises out: any failure is logged via ``print`` and swallowed so
    canonical state cannot be affected by shadow-write errors.
    """

    try:
        target = Path(output_dir) / _SHADOW_ARTIFACT_NAME
        append_jsonl(str(target), record.to_dict())
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"    [WARN] shadow_final_judgments.jsonl write failed: "
            f"{type(exc).__name__}: {exc}"
        )
