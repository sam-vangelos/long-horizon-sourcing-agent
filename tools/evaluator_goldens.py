#!/usr/bin/env python3
"""Run evaluator golden profiles through the production full-profile judge.

The module deliberately imports only the standard library at import time.
``shared.config`` reads environment variables while importing, so ``main``
must parse ``--env-profile`` and apply it before loading any shared runtime
dependencies.  Keeping the dependency seam lazy also makes dry runs and unit
tests network-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BRIEF = REPO_ROOT / "output/state/linkedin/2085486330/preflight_v2_brief.json"
DEFAULT_FIXTURES = REPO_ROOT / "tests/fixtures/evaluator_goldens/fixtures.json"
DEFAULT_REAL_GOLDENS = (
    REPO_ROOT / "tests/fixtures/evaluator_goldens/real_goldens.json"
)
DEFAULT_STATE_DIR = REPO_ROOT / "output/state/linkedin/2085486330"

ENV_KEYS = (
    "FULL_EVAL_MODEL_NAME",
    "FIREWORKS_JUDGMENT_POLICY_ENABLED",
    "FIREWORKS_FULL_REASONING_EFFORT",
    "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS",
    "FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS",
    "FIREWORKS_FULL_MAX_ATTEMPTS",
    "LINKEDIN_V2_FULL_CONTRACT",
)
TBA_ENV_DEFAULTS = {
    # Was routers/glm-5p2-fast until 2026-08-04. That router is degraded
    # provider-side (CLO-50) — deterministic zero-byte hangs on short prompts,
    # 19.6% failure at production prompt sizes — so a goldens re-run against it
    # would stall for the full attempt timeout rather than produce a verdict.
    "FULL_EVAL_MODEL_NAME": "accounts/fireworks/models/glm-5p2",
    "FIREWORKS_JUDGMENT_POLICY_ENABLED": "true",
    "FIREWORKS_FULL_REASONING_EFFORT": "high",
    "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS": "240",
    "FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS": "360",
    "FIREWORKS_FULL_MAX_ATTEMPTS": "2",
    "LINKEDIN_V2_FULL_CONTRACT": "tool",
}

SAVE_FAMILY = "SAVE_FAMILY"
REVIEW_FAMILY = "REVIEW_FAMILY"
REJECT = "REJECT"
DECISION_FAMILIES = {
    SAVE_FAMILY: frozenset(
        {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}
    ),
    REVIEW_FAMILY: frozenset({"REVIEW_INFERRED", "REVIEW_FLAGGED"}),
    REJECT: frozenset({"REJECT"}),
}

_REJECT_REASON_TAG = re.compile(
    r"\[REJECT_REASON:\s*([A-Z][A-Z0-9_]*)\s*\]",
    re.IGNORECASE,
)
_OUTREACH_TIER_TAG = re.compile(
    r"\[TIER:\s*(PRIORITY|STANDARD)\s*\]",
    re.IGNORECASE,
)

# Public, lazy test seam.  Production resolves this to shared.judger.full_judge
# inside main after environment setup.  Tests can monkeypatch either name.
full_judge: Callable[..., Any] | None = None
_full_judge: Callable[..., Any] | None = None
load_brief: Callable[[str | Path], Any] | None = None
_load_brief: Callable[[str | Path], Any] | None = None


@dataclass(frozen=True)
class PlannedRow:
    """One profile and its optional golden expectation."""

    source: str
    profile: dict[str, Any]
    fixture_id: str | None = None
    expected_new: dict[str, Any] | None = None

    @property
    def profile_url(self) -> str:
        return str(self.profile.get("profile_url", ""))

    @property
    def display_id(self) -> str:
        if self.fixture_id:
            return self.fixture_id
        return _opaque_console_id(
            self.source,
            self.profile_url or str(self.profile.get("name", "")),
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _opaque_console_id(source: str, identity: str) -> str:
    """Return a stable run-console label without emitting candidate PII."""

    if not identity:
        return f"{source}:unknown"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{digest}"


def _default_out_dir() -> Path:
    stamp = _utc_now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / f"output/experiments/evaluator_goldens/run-{stamp}"


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _concurrency(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 6:
        raise argparse.ArgumentTypeError("must be between 1 and 6")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge evaluator goldens through shared.judger.full_judge."
    )
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--no-fixtures", action="store_true")
    parser.add_argument("--real-goldens", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--all-state-profiles", action="store_true")
    parser.add_argument("--select", default="")
    parser.add_argument("--limit", type=_nonnegative_int)
    parser.add_argument("--env-profile", choices=("tba",))
    parser.add_argument("--concurrency", type=_concurrency, default=4)
    parser.add_argument("--max-calls", type=_nonnegative_int, default=130)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--grade-only", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=("current", "new"), default="current")
    return parser


def apply_env_profile(name: str | None) -> None:
    """Apply profile defaults without replacing explicit environment values."""

    if name is None:
        return
    if name != "tba":
        raise ValueError(f"unsupported environment profile: {name}")
    for key, value in TBA_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _load_config() -> Any:
    # This import must remain below apply_env_profile in main.
    import shared.config as config

    return config


def _runtime_dependencies() -> tuple[Callable[[str | Path], Any], type, Callable[..., Any]]:
    """Load profile/brief/judge dependencies after environment resolution."""

    from shared.schemas import CandidateProfileSummary

    global full_judge, _full_judge, load_brief, _load_brief
    brief_loader = load_brief or _load_brief
    if brief_loader is None:
        from shared.brief_loader import load_brief as production_load_brief

        load_brief = production_load_brief
        _load_brief = production_load_brief
        brief_loader = production_load_brief
    judge = full_judge or _full_judge
    if judge is None:
        from shared.judger import full_judge as production_full_judge

        full_judge = production_full_judge
        _full_judge = production_full_judge
        judge = production_full_judge
    return brief_loader, CandidateProfileSummary, judge


def _resolved_environment(config: Any) -> dict[str, str]:
    """Return effective config values in their environment-string form."""

    fallback_values = {
        "FULL_EVAL_MODEL_NAME": config.FULL_EVAL_MODEL_NAME,
        "FIREWORKS_JUDGMENT_POLICY_ENABLED": (
            "true" if config.FIREWORKS_JUDGMENT_POLICY_ENABLED else "false"
        ),
        "FIREWORKS_FULL_REASONING_EFFORT": config.FIREWORKS_FULL_REASONING_EFFORT,
        "FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS": (
            config.FIREWORKS_FULL_ATTEMPT_TIMEOUT_SECONDS
        ),
        "FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS": (
            config.FIREWORKS_FULL_TOTAL_DEADLINE_SECONDS
        ),
        "FIREWORKS_FULL_MAX_ATTEMPTS": config.FIREWORKS_FULL_MAX_ATTEMPTS,
        "LINKEDIN_V2_FULL_CONTRACT": config.LINKEDIN_V2_FULL_CONTRACT,
    }
    return {
        key: os.environ.get(key, str(fallback_values[key]))
        for key in ENV_KEYS
    }


def _load_json_array(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"{label} must be a JSON array of objects: {path}")
    return payload


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    rows: list[dict[str, Any]] = []
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid {label} row {line_number} in {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{label} row {line_number} is not an object: {path}"
                )
            rows.append(row)
    return rows


def _state_profiles(state_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(state_dir / "profile_summaries.jsonl", "state profiles")


def plan_rows(args: argparse.Namespace) -> list[PlannedRow]:
    """Build the deterministic fixture/real/state execution plan."""

    planned: list[PlannedRow] = []
    if not args.no_fixtures:
        for fixture in _load_json_array(args.fixtures, "fixture data"):
            profile = fixture.get("profile")
            fixture_id = fixture.get("fixture_id")
            if not isinstance(profile, dict) or not isinstance(fixture_id, str):
                raise ValueError("each fixture requires string fixture_id and object profile")
            expected = fixture.get("expected_new")
            if expected is not None and not isinstance(expected, dict):
                raise ValueError(f"fixture {fixture_id} expected_new must be an object")
            planned.append(
                PlannedRow(
                    source="fixture",
                    fixture_id=fixture_id,
                    profile=dict(profile),
                    expected_new=dict(expected) if expected is not None else None,
                )
            )

    state_profiles: list[dict[str, Any]] | None = None
    if args.real_goldens or args.all_state_profiles:
        state_profiles = _state_profiles(args.state_dir)

    if args.real_goldens:
        assert state_profiles is not None
        by_url: dict[str, dict[str, Any]] = {}
        for profile in state_profiles:
            profile_url = str(profile.get("profile_url", ""))
            if not profile_url:
                raise ValueError("state profile is missing profile_url")
            if profile_url in by_url:
                raise ValueError(f"duplicate profile_url in state profiles: {profile_url}")
            by_url[profile_url] = profile
        for golden in _load_json_array(DEFAULT_REAL_GOLDENS, "real goldens"):
            profile_url = str(golden.get("profile_url", ""))
            if profile_url not in by_url:
                raise ValueError(
                    f"real golden profile_url is absent from state profiles: {profile_url}"
                )
            expected = golden.get("expected_new")
            if not isinstance(expected, dict):
                raise ValueError(
                    f"real golden {profile_url} expected_new must be an object"
                )
            planned.append(
                PlannedRow(
                    source="real_golden",
                    profile=dict(by_url[profile_url]),
                    expected_new=dict(expected),
                )
            )

    if args.all_state_profiles:
        assert state_profiles is not None
        planned.extend(
            PlannedRow(source="state", profile=dict(profile))
            for profile in state_profiles
        )

    selected = {token.strip() for token in args.select.split(",") if token.strip()}
    if selected:
        planned = [
            row
            for row in planned
            if row.profile_url in selected
            or (row.fixture_id is not None and row.fixture_id in selected)
        ]
    if args.limit is not None:
        planned = planned[: args.limit]
    return planned


def _source_counts(rows: Sequence[PlannedRow]) -> dict[str, int]:
    counts = Counter(row.source for row in rows)
    return {
        "fixture": counts["fixture"],
        "real_golden": counts["real_golden"],
        "state": counts["state"],
    }


def _print_plan(
    rows: Sequence[PlannedRow],
    *,
    env_profile: str | None,
    model: str,
    contract_mode: str,
) -> None:
    counts = _source_counts(rows)
    print(f"Environment profile: {env_profile or 'as-is'}")
    print(f"Resolved model: {model}")
    print(f"Contract mode: {contract_mode}")
    print(
        "Planned rows: "
        f"{len(rows)} (fixture={counts['fixture']}, "
        f"real_golden={counts['real_golden']}, state={counts['state']})"
    )
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row.source} {row.display_id}")
    print(f"Dry run complete: {len(rows)} planned rows; 0 LLM calls.")


def _get_field(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def parse_transitional_tags(row: Mapping[str, Any]) -> dict[str, str]:
    """Extract the last transitional evaluator tag from rationale/summary."""

    parsed: dict[str, str] = {}
    for field_name in ("rationale", "summary"):
        text = str(row.get(field_name, "") or "")
        for match in _REJECT_REASON_TAG.finditer(text):
            parsed["reject_reason"] = match.group(1).upper()
        for match in _OUTREACH_TIER_TAG.finditer(text):
            parsed["outreach_tier"] = match.group(1).upper()
    return parsed


def apply_transitional_tags(row: dict[str, Any]) -> dict[str, Any]:
    """Backfill absent allocator fields from historical prose tags."""

    for key, value in parse_transitional_tags(row).items():
        if key not in row:
            row[key] = value
    return row


def _judgment_failure_row(
    row: PlannedRow,
    *,
    latency_s: float,
    model: str,
    contract_mode: str,
    exc: Exception,
) -> dict[str, Any]:
    result = _base_result_row(row)
    result.update(
        {
            "decision": "JUDGMENT_FAILURE",
            "confidence": None,
            "path": "none",
            "post_save_modifier": "NONE",
            "review_reason_code": "",
            "rationale": f"full_judge raised {type(exc).__name__}",
            "semantic_evidence": {},
            "latency_s": round(latency_s, 3),
            "model": model,
            "contract_mode": contract_mode,
        }
    )
    return apply_transitional_tags(result)


def _base_result_row(row: PlannedRow) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": row.source,
        "candidate_name": str(row.profile.get("name", "")),
        "profile_url": row.profile_url,
    }
    if row.fixture_id is not None:
        result["fixture_id"] = row.fixture_id
    if row.expected_new is not None:
        result["expected_new"] = row.expected_new
    return result


def _judge_row(
    row: PlannedRow,
    *,
    brief: Any,
    profile_class: type,
    judge: Callable[..., Any],
    model: str,
    contract_mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        summary = profile_class.from_dict(row.profile)
        decision = judge(
            summary,
            brief=brief,
            lane_context={"lane_id": "goldens", "string_id": 0},
        )
    except Exception as exc:
        return _judgment_failure_row(
            row,
            latency_s=time.perf_counter() - started,
            model=model,
            contract_mode=contract_mode,
            exc=exc,
        )

    result = _base_result_row(row)
    result.update(
        {
            "candidate_name": str(
                _get_field(decision, "candidate_name", result["candidate_name"])
                or result["candidate_name"]
            ),
            "profile_url": str(
                _get_field(decision, "profile_url", result["profile_url"])
                or result["profile_url"]
            ),
            "decision": str(_get_field(decision, "decision", "JUDGMENT_FAILURE")),
            "confidence": _get_field(decision, "confidence", None),
            "path": str(_get_field(decision, "path", "none")),
            "post_save_modifier": str(
                _get_field(decision, "post_save_modifier", "NONE") or "NONE"
            ),
            "review_reason_code": str(
                _get_field(decision, "review_reason_code", "") or ""
            ),
            "rationale": str(_get_field(decision, "rationale", "") or ""),
            "semantic_evidence": _get_field(decision, "semantic_evidence", {}) or {},
            "outreach_tier": _get_field(decision, "outreach_tier", None) or None,
            "reject_reason": _get_field(decision, "reject_reason", None) or None,
            "latency_s": round(time.perf_counter() - started, 3),
            "model": model,
            "contract_mode": contract_mode,
        }
    )
    # Present structured fields, including null, are authoritative. Tags only
    # backfill historical rows where those keys do not exist.
    return apply_transitional_tags(result)


def _format_confidence(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _print_completion(index: int, total: int, result: Mapping[str, Any]) -> None:
    display_id = str(result.get("fixture_id") or "")
    if not display_id:
        identity = str(result.get("profile_url") or result.get("candidate_name") or "")
        display_id = _opaque_console_id(str(result.get("source") or "row"), identity)
    print(
        f"[{index}/{total}] {display_id} \u2192 {result.get('decision', 'UNKNOWN')} "
        f"{_format_confidence(result.get('confidence'))} "
        f"{float(result.get('latency_s', 0.0)):.1f}s"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_rows(
    rows: Sequence[PlannedRow],
    *,
    args: argparse.Namespace,
    argv: Sequence[str],
    model: str,
    contract_mode: str,
    resolved_env: Mapping[str, str],
) -> list[dict[str, Any]]:
    brief_loader, profile_class, judge = _runtime_dependencies()
    brief = brief_loader(args.brief)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    started = _utc_now()
    histogram: Counter[str] = Counter()
    completed_rows: list[dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    _judge_row,
                    row,
                    brief=brief,
                    profile_class=profile_class,
                    judge=judge,
                    model=model,
                    contract_mode=contract_mode,
                )
                for row in rows
            ]
            for completed_index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                completed_rows.append(result)
                histogram[str(result["decision"])] += 1
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                _print_completion(completed_index, len(rows), result)

    finished = _utc_now()
    meta = {
        "argv": list(argv),
        "env_profile": args.env_profile,
        "resolved_env": dict(resolved_env),
        "brief_path": str(args.brief),
        "row_counts": _source_counts(rows),
        "total_rows": len(rows),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "decision_histogram": dict(sorted(histogram.items())),
    }
    _write_json(out_dir / "run_meta.json", meta)
    return completed_rows


def decision_family(decision: str) -> str | None:
    """Return the symbolic evaluator family for a terminal decision."""

    for family, members in DECISION_FAMILIES.items():
        if decision in members:
            return family
    return None


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _check(status: str, **detail: Any) -> dict[str, Any]:
    return {"status": status, **detail}


def grade_row(row: Mapping[str, Any], mode: str = "current") -> dict[str, Any]:
    """Grade one result row against its ``expected_new`` contract."""

    if mode not in {"current", "new"}:
        raise ValueError(f"unsupported evaluator mode: {mode}")
    expected = row.get("expected_new")
    identity = {
        "source": row.get("source", ""),
        "fixture_id": row.get("fixture_id"),
        "candidate_name": row.get("candidate_name", ""),
        "profile_url": row.get("profile_url", ""),
        "decision": str(row.get("decision", "")),
    }
    family = decision_family(identity["decision"])
    identity["decision_family"] = family
    if not isinstance(expected, Mapping):
        return {
            **identity,
            "passed": None,
            "verdict": "NOT_GRADED",
            "checks": {},
        }

    failures: list[str] = []
    checks: dict[str, Any] = {}

    allowed_families = [str(value) for value in _values(expected.get("families"))]
    family_ok = family in allowed_families
    checks["families"] = _check(
        "pass" if family_ok else "fail",
        expected=allowed_families,
        actual=family,
    )
    if not family_ok:
        failures.append("families")

    if "decisions" in expected:
        allowed_decisions = [
            str(value) for value in _values(expected.get("decisions"))
        ]
        decision_ok = identity["decision"] in allowed_decisions
        checks["decisions"] = _check(
            "pass" if decision_ok else "fail",
            expected=allowed_decisions,
            actual=identity["decision"],
        )
        if not decision_ok:
            failures.append("decisions")
    else:
        checks["decisions"] = _check("not_specified")

    must_not_raw = expected.get("must_not", {})
    must_not = must_not_raw if isinstance(must_not_raw, Mapping) else {}
    must_not_checks: dict[str, Any] = {}
    for clause in ("families", "decisions", "tier", "reject_reasons"):
        if clause not in must_not:
            continue
        forbidden = [str(value) for value in _values(must_not.get(clause))]
        if clause == "families":
            actual_values = [] if family is None else [family]
        elif clause == "decisions":
            actual_values = [identity["decision"]]
        elif mode == "new":
            result_field = {
                "tier": "outreach_tier",
                "reject_reasons": "reject_reason",
            }[clause]
            actual_values = [
                str(value) for value in _values(row.get(result_field))
            ]
        elif clause not in row:
            must_not_checks[clause] = _check(
                "n/a_current_mode", forbidden=forbidden
            )
            continue
        else:
            actual_values = [str(value) for value in _values(row.get(clause))]
        violations = sorted(set(actual_values) & set(forbidden))
        clause_ok = not violations
        must_not_checks[clause] = _check(
            "pass" if clause_ok else "fail",
            forbidden=forbidden,
            actual=actual_values,
            violations=violations,
        )
        if not clause_ok:
            failures.append(f"must_not.{clause}")
    checks["must_not"] = must_not_checks

    aspirational: dict[str, Any] = {}
    for clause, result_field in (
        ("tier", "outreach_tier"),
        ("reject_reasons", "reject_reason"),
    ):
        if clause not in expected:
            continue
        allowed = [str(value) for value in _values(expected.get(clause))]
        if mode == "current":
            aspirational[clause] = _check(
                "not_graded_current_mode", expected=expected.get(clause)
            )
            continue
        # A reject_reasons expectation binds only when the decision IS a
        # reject, and a tier expectation only when it IS a save — the golden
        # means "if rejected, for this reason", not "must be rejected".
        # (2026-07-14: literal grading failed REVIEW rows for lacking a
        # reject tag they could never carry.)
        applies = (
            family == REJECT
            if clause == "reject_reasons"
            else family == SAVE_FAMILY
        )
        if not applies:
            aspirational[clause] = _check(
                "n/a_decision_family", expected=allowed, actual_family=family
            )
            continue
        actual = row.get(result_field)
        clause_ok = actual is not None and str(actual) in allowed
        checks[clause] = _check(
            "pass" if clause_ok else "fail",
            expected=allowed,
            actual=actual,
        )
        if not clause_ok:
            failures.append(clause)
    checks["aspirational"] = aspirational

    passed = not failures
    return {
        **identity,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "failures": failures,
        "checks": checks,
    }


# Backward-friendly singular name for tests and small callers.
grade_result = grade_row


def _fixture_class(verdict: Mapping[str, Any]) -> str | None:
    if verdict.get("source") == "real_golden":
        return "real"
    fixture_id = str(verdict.get("fixture_id") or "")
    if fixture_id.startswith("adv_"):
        return "adv_"
    if fixture_id.startswith("prot_"):
        return "prot_"
    return None


def _new_totals() -> dict[str, int]:
    return {"total": 0, "passed": 0, "failed": 0, "not_graded": 0}


def _add_total(total: dict[str, int], passed: bool | None) -> None:
    total["total"] += 1
    if passed is True:
        total["passed"] += 1
    elif passed is False:
        total["failed"] += 1
    else:
        total["not_graded"] += 1


def grade_results(
    rows: Sequence[Mapping[str, Any]], mode: str = "current"
) -> dict[str, Any]:
    """Build the complete per-row report and fixture-class totals."""

    verdicts = [grade_row(row, mode=mode) for row in rows]
    overall = _new_totals()
    by_class = {name: _new_totals() for name in ("adv_", "prot_", "real")}
    for verdict in verdicts:
        passed = verdict.get("passed")
        _add_total(overall, passed if isinstance(passed, bool) else None)
        fixture_class = _fixture_class(verdict)
        if fixture_class is not None:
            _add_total(
                by_class[fixture_class],
                passed if isinstance(passed, bool) else None,
            )
    return {
        "mode": mode,
        "rows": verdicts,
        "totals": {"overall": overall, "by_class": by_class},
    }


def _short_label(verdict: Mapping[str, Any]) -> str:
    fixture_id = str(verdict.get("fixture_id") or "")
    if fixture_id:
        return fixture_id
    identity = str(verdict.get("profile_url") or verdict.get("candidate_name") or "")
    return _opaque_console_id(str(verdict.get("source") or "row"), identity)


def print_grade_table(report: Mapping[str, Any]) -> None:
    rows = report.get("rows", [])
    display_rows = [
        (
            _short_label(row),
            str(row.get("decision", "")),
            str(row.get("decision_family") or "unmapped"),
            str(row.get("verdict", "")),
        )
        for row in rows
        if isinstance(row, Mapping)
    ]
    headers = ("ROW", "DECISION", "FAMILY", "VERDICT")
    widths = [len(value) for value in headers]
    for values in display_rows:
        for index, value in enumerate(values):
            widths[index] = max(widths[index], len(value))
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    for values in display_rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(values)))
    totals = report.get("totals", {}).get("overall", {})
    print(
        "Totals: "
        f"{totals.get('passed', 0)} pass, {totals.get('failed', 0)} fail, "
        f"{totals.get('not_graded', 0)} not graded"
    )


def _read_results(path: Path) -> list[dict[str, Any]]:
    # Rows written before the judge-time enrichment landed carry tags only in
    # prose; re-derive on read so --grade-only grades them identically.
    return [
        apply_transitional_tags(row)
        for row in _load_jsonl(path, "evaluator results")
    ]


def refresh_expectations(
    rows: Sequence[dict[str, Any]],
    *,
    fixtures_path: Path = DEFAULT_FIXTURES,
    real_goldens_path: Path = DEFAULT_REAL_GOLDENS,
) -> list[dict[str, Any]]:
    """Re-attach ``expected_new`` from the CURRENT golden files.

    ``expected_new`` is snapshotted into each result row at judge time, so a
    later golden-spec calibration would otherwise never reach --grade-only —
    expectations are spec, results are measurements, and grading must always
    use the current spec (2026-07-14 Phase 1 gate).  Rows the golden files do
    not identify (plain state rows) pass through unchanged.
    """

    by_fixture: dict[str, Any] = {}
    try:
        for fixture in _load_json_array(fixtures_path, "fixture data"):
            fixture_id = fixture.get("fixture_id")
            if isinstance(fixture_id, str) and isinstance(
                fixture.get("expected_new"), Mapping
            ):
                by_fixture[fixture_id] = dict(fixture["expected_new"])
    except ValueError:
        pass
    by_url: dict[str, Any] = {}
    try:
        for golden in _load_json_array(real_goldens_path, "real goldens"):
            url = str(golden.get("profile_url", ""))
            if url and isinstance(golden.get("expected_new"), Mapping):
                by_url[url] = dict(golden["expected_new"])
    except ValueError:
        pass

    refreshed: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        fixture_id = row.get("fixture_id")
        if isinstance(fixture_id, str) and fixture_id in by_fixture:
            row["expected_new"] = by_fixture[fixture_id]
        elif row.get("source") == "real_golden" and row.get("profile_url") in by_url:
            row["expected_new"] = by_url[str(row.get("profile_url"))]
        refreshed.append(row)
    return refreshed


def emit_grade_report(
    rows: Sequence[Mapping[str, Any]], *, out_dir: Path, mode: str
) -> dict[str, Any]:
    report = grade_results(rows, mode=mode)
    _write_json(out_dir / "grade_report.json", report)
    print_grade_table(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    args.brief = args.brief.expanduser().resolve()
    args.fixtures = args.fixtures.expanduser().resolve()
    args.state_dir = args.state_dir.expanduser().resolve()
    args.out = (args.out or _default_out_dir()).expanduser().resolve()

    if args.grade_only and args.dry_run:
        parser.error("--grade-only cannot be combined with --dry-run")
    if args.grade and args.dry_run:
        parser.error("--grade cannot be combined with --dry-run")

    # This ordering is the harness's critical import-time contract.
    apply_env_profile(args.env_profile)

    if args.grade_only:
        try:
            rows = _read_results(args.out / "results.jsonl")
            rows = refresh_expectations(rows, fixtures_path=args.fixtures)
            emit_grade_report(rows, out_dir=args.out, mode=args.mode)
        except (OSError, ValueError) as exc:
            print(f"evaluator-goldens: {exc}", file=sys.stderr)
            return 2
        return 0

    # Loading config after setdefault also applies the repo's normal .env
    # layering.  Dry-run intentionally stops before importing the judge.
    config = _load_config()
    resolved_env = _resolved_environment(config)
    model = str(config.FULL_EVAL_MODEL_NAME)
    contract_mode = str(config.LINKEDIN_V2_FULL_CONTRACT)

    try:
        rows = plan_rows(args)
    except (OSError, ValueError) as exc:
        print(f"evaluator-goldens: {exc}", file=sys.stderr)
        return 2
    if len(rows) > args.max_calls:
        print(
            f"evaluator-goldens: planned {len(rows)} calls exceeds "
            f"--max-calls {args.max_calls}",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        _print_plan(
            rows,
            env_profile=args.env_profile,
            model=model,
            contract_mode=contract_mode,
        )
        return 0

    try:
        completed_rows = _run_rows(
            rows,
            args=args,
            argv=raw_argv,
            model=model,
            contract_mode=contract_mode,
            resolved_env=resolved_env,
        )
        if args.grade:
            emit_grade_report(completed_rows, out_dir=args.out, mode=args.mode)
    except (OSError, ValueError) as exc:
        print(f"evaluator-goldens: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
