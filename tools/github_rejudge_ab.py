#!/usr/bin/env python3
"""Golden re-judge A/B harness for GitHub full-evaluation prompt revisions.

Replays stored ``prompt_capture.candidate_text`` evidence from a completed run
against the *current* full-evaluation system prompt assembler and compares new
decisions to the persisted run-1 outcomes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from github.judgment_templates import (  # noqa: E402
    assemble_github_full_evaluation_system,
    parse_full_evaluation_response,
)
from shared.brief_loader import load_brief  # noqa: E402
from shared.contracts import SAVE_DECISIONS  # noqa: E402
from shared.failures import ApiBudgetExhaustedError  # noqa: E402
from shared.llm_clients import opus_llm_cached  # noqa: E402

_USERNAME_LINE_RE = re.compile(r"^Username:\s*(.+?)\s*$", re.MULTILINE)


@dataclass
class CandidateRow:
    username: str
    candidate_text: str
    old_decision: str
    old_confidence: float | None
    system_prompt_sha256: str | None = None


@dataclass
class RejudgeRow:
    username: str
    old_decision: str
    old_confidence: float | None
    new_decision: str | None = None
    new_confidence: float | None = None
    changed: str = "no"
    error: str | None = None


@dataclass
class LoadResult:
    candidates: list[CandidateRow]
    load_errors: list[RejudgeRow]
    skipped_foreign: int
    skipped_duplicate: int


def build_state_db_uri(db_path: Path) -> str:
    """Return a read-only SQLite URI for ``db_path`` (no ``immutable`` flag)."""
    return f"file:{db_path}?mode=ro"


def _open_state_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(build_state_db_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _validate_out_path(out_path: Path, state_dir: Path) -> None:
    out_resolved = out_path.resolve()
    state_resolved = state_dir.resolve()
    out_norm = os.path.normcase(str(out_resolved))
    state_norm = os.path.normcase(str(state_resolved))
    parent_norms = {os.path.normcase(str(parent)) for parent in out_resolved.parents}
    if out_norm == state_norm or state_norm in parent_norms:
        raise SystemExit("--out must not be inside --state-dir")
    out_parent = out_resolved.parent
    if out_parent.exists() and os.path.samefile(out_parent, state_resolved):
        raise SystemExit("--out must not be inside --state-dir")


def _username_from_candidate_text(candidate_text: str) -> str | None:
    match = _USERNAME_LINE_RE.search(candidate_text)
    if not match:
        return None
    return match.group(1).strip() or None


def _extract_username(payload: dict[str, Any], candidate_text: str) -> str:
    for key_path in (
        ("candidate_record", "username"),
        ("candidate", "username"),
        ("profile", "username"),
    ):
        block = payload.get(key_path[0])
        if isinstance(block, dict):
            username = block.get(key_path[1])
            if isinstance(username, str) and username.strip():
                return username.strip()
    parsed = _username_from_candidate_text(candidate_text)
    if parsed:
        return parsed
    return "unknown"


def _extract_old_verdict(payload: dict[str, Any]) -> tuple[str, float | None]:
    decision_block = payload.get("full_decision")
    if not isinstance(decision_block, dict):
        nested = payload.get("decision")
        decision_block = nested if isinstance(nested, dict) else {}

    decision = decision_block.get("decision")
    if not isinstance(decision, str) or not decision.strip():
        raise ValueError("payload missing full_decision.decision")

    confidence = decision_block.get("confidence")
    if confidence is None:
        return decision.strip(), None
    try:
        return decision.strip(), float(confidence)
    except (TypeError, ValueError):
        return decision.strip(), None


def _is_github_capture(prompt_capture: dict[str, Any]) -> bool:
    source = prompt_capture.get("source")
    if source == "github":
        return True
    render_route = prompt_capture.get("render_route")
    return isinstance(render_route, str) and render_route.startswith("github.full")


def _copy_state_db_snapshot(state_dir: Path, tmp_path: Path) -> Path:
    db_name = "runtime_state.sqlite3"
    src_db = state_dir / db_name
    if not src_db.is_file():
        raise SystemExit(f"runtime_state.sqlite3 not found under {state_dir}")

    dst_db = tmp_path / db_name
    shutil.copy2(src_db, dst_db)
    for suffix in ("-wal", "-shm"):
        sidecar = state_dir / f"{db_name}{suffix}"
        if sidecar.is_file():
            shutil.copy2(sidecar, tmp_path / f"{db_name}{suffix}")
    return dst_db


def _load_candidates(state_dir: Path, *, limit: int | None) -> LoadResult:
    candidates: list[CandidateRow] = []
    load_errors: list[RejudgeRow] = []
    skipped_foreign = 0
    skipped_duplicate = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = _copy_state_db_snapshot(state_dir, Path(tmp_dir))
        qualifying: list[tuple[int, int, str, CandidateRow]] = []

        with contextlib.closing(_open_state_db(db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT id, attempt_number, payload_json
                FROM candidate_attempts
                WHERE stage = 'full'
                ORDER BY id
                """
            )
            for record in cursor:
                payload_raw = record["payload_json"]
                if not payload_raw:
                    continue
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError as exc:
                    load_errors.append(
                        RejudgeRow(
                            username="unknown",
                            old_decision="",
                            old_confidence=None,
                            error=f"invalid payload_json: {exc}",
                        )
                    )
                    continue
                if not isinstance(payload, dict):
                    continue

                prompt_capture = payload.get("prompt_capture")
                if not isinstance(prompt_capture, dict):
                    continue
                candidate_text = prompt_capture.get("candidate_text")
                if not isinstance(candidate_text, str) or not candidate_text.strip():
                    continue

                if not _is_github_capture(prompt_capture):
                    skipped_foreign += 1
                    continue

                username = _extract_username(payload, candidate_text)
                try:
                    old_decision, old_confidence = _extract_old_verdict(payload)
                except (ValueError, TypeError) as exc:
                    load_errors.append(
                        RejudgeRow(
                            username=username,
                            old_decision="",
                            old_confidence=None,
                            error=str(exc),
                        )
                    )
                    continue

                system_prompt_sha256 = prompt_capture.get("system_prompt_sha256")
                sha_value = (
                    system_prompt_sha256.strip()
                    if isinstance(system_prompt_sha256, str) and system_prompt_sha256.strip()
                    else None
                )
                row = CandidateRow(
                    username=username,
                    candidate_text=candidate_text,
                    old_decision=old_decision,
                    old_confidence=old_confidence,
                    system_prompt_sha256=sha_value,
                )
                qualifying.append(
                    (int(record["attempt_number"]), int(record["id"]), username, row)
                )

        best_by_username: dict[str, tuple[int, int, CandidateRow]] = {}
        for attempt_number, row_id, username, row in qualifying:
            current = best_by_username.get(username)
            if current is None or (attempt_number, row_id) > (current[0], current[1]):
                best_by_username[username] = (attempt_number, row_id, row)

        skipped_duplicate = len(qualifying) - len(best_by_username)
        ordered = sorted(best_by_username.values(), key=lambda item: item[1])
        for _, _, row in ordered:
            candidates.append(row)
            if limit is not None and len(candidates) >= limit:
                break

    return LoadResult(
        candidates=candidates,
        load_errors=load_errors,
        skipped_foreign=skipped_foreign,
        skipped_duplicate=skipped_duplicate,
    )


def _flip_category(old_decision: str, new_decision: str) -> str | None:
    if old_decision == new_decision:
        return None
    if old_decision in SAVE_DECISIONS and new_decision == "REJECT":
        return "save_to_reject"
    if old_decision == "REJECT" and new_decision in SAVE_DECISIONS:
        return "reject_to_save"
    return "other_decision_change"


def _format_confidence(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_confidence_delta(
    old_confidence: float | None,
    new_confidence: float | None,
) -> str:
    if old_confidence is None or new_confidence is None:
        return ""
    delta = new_confidence - old_confidence
    if abs(delta) < 1e-9:
        return "0"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f}".rstrip("0").rstrip(".")


def _render_report(
    *,
    state_dir: Path,
    brief_path: Path,
    model_name: str,
    brief_id: str,
    results: list[RejudgeRow],
    selected_count: int,
    skipped_foreign: int,
    skipped_duplicate: int,
    old_prompt_sha256s: list[str],
    aborted: bool,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# GitHub full-evaluation re-judge A/B",
        "",
        f"- **State dir:** `{state_dir}`",
        f"- **Brief:** `{brief_path}` (`{brief_id}`)",
        f"- **Model:** `{model_name}`",
        f"- **Timestamp:** {timestamp}",
        f"- **Selected:** {selected_count}",
        f"- **Skipped foreign:** {skipped_foreign}",
        f"- **Skipped duplicate:** {skipped_duplicate}",
    ]
    if old_prompt_sha256s:
        lines.append(
            "- **Old-prompt sha256 (stored):** "
            + ", ".join(f"`{value}`" for value in old_prompt_sha256s)
        )
    lines.append(
        "- **Caveat:** Old captures may predate evidence-format changes; "
        "sections absent from old captures read as absent evidence."
    )
    if aborted:
        lines.append("- **Status:** ABORTED (API budget exhausted)")
    lines.extend(
        [
            "",
            "| username | old decision | old conf | new decision | new conf | Δconf | changed |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    unchanged = 0
    save_to_reject = 0
    reject_to_save = 0
    other_decision_changes = 0
    confidence_moved = 0
    errors = 0

    for row in results:
        if row.error:
            errors += 1
            lines.append(
                "| {username} | {old_decision} | {old_conf} | | | | ERROR |".format(
                    username=row.username,
                    old_decision=row.old_decision,
                    old_conf=_format_confidence(row.old_confidence),
                )
            )
            continue

        new_decision = row.new_decision or ""
        new_conf = _format_confidence(row.new_confidence)
        delta_conf = _format_confidence_delta(row.old_confidence, row.new_confidence)
        lines.append(
            "| {username} | {old_decision} | {old_conf} | {new_decision} | {new_conf} | {delta_conf} | {changed} |".format(
                username=row.username,
                old_decision=row.old_decision,
                old_conf=_format_confidence(row.old_confidence),
                new_decision=new_decision,
                new_conf=new_conf,
                delta_conf=delta_conf,
                changed=row.changed,
            )
        )

        if row.changed == "no":
            unchanged += 1
            if (
                row.new_confidence is not None
                and row.old_confidence is not None
                and row.new_decision == row.old_decision
                and abs(row.new_confidence - row.old_confidence) >= 0.10
            ):
                confidence_moved += 1
            continue
        if row.changed == "(dry)":
            continue
        category = _flip_category(row.old_decision, new_decision)
        if category == "save_to_reject":
            save_to_reject += 1
        elif category == "reject_to_save":
            reject_to_save += 1
        elif category == "other_decision_change":
            other_decision_changes += 1

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Unchanged: {unchanged}",
            f"- SAVE→REJECT: {save_to_reject}",
            f"- REJECT→SAVE: {reject_to_save}",
            f"- Other decision changes: {other_decision_changes}",
            f"- Confidence moved ≥ 0.10 (same decision): {confidence_moved}",
            f"- Errors: {errors}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_rejudge(
    *,
    state_dir: Path,
    brief_path: Path,
    model_name: str,
    out_path: Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    state_dir = state_dir.resolve()
    brief_path = brief_path.resolve()
    out_path = out_path.resolve()
    _validate_out_path(out_path, state_dir)

    load_result = _load_candidates(state_dir, limit=limit)
    if not load_result.candidates and not load_result.load_errors:
        raise SystemExit("no full-stage rows with prompt_capture.candidate_text found")

    brief = load_brief(brief_path)
    if not brief.has_v2_schema:
        raise SystemExit("brief must be V2 schema")
    system_prompt = assemble_github_full_evaluation_system(brief._new_brief)

    with brief_path.open(encoding="utf-8") as brief_file:
        brief_json = json.load(brief_file)
    brief_id = brief_json.get("id") if isinstance(brief_json, dict) else None
    if not isinstance(brief_id, str) or not brief_id.strip():
        brief_id = brief_path.stem

    old_prompt_sha256s = sorted(
        {
            row.system_prompt_sha256
            for row in load_result.candidates
            if row.system_prompt_sha256
        }
    )

    results: list[RejudgeRow] = list(load_result.load_errors)
    aborted = False

    for candidate in load_result.candidates:
        row = RejudgeRow(
            username=candidate.username,
            old_decision=candidate.old_decision,
            old_confidence=candidate.old_confidence,
        )
        if dry_run:
            row.new_decision = "(dry)"
            row.new_confidence = None
            row.changed = "(dry)"
            results.append(row)
            continue

        try:
            raw = opus_llm_cached(
                system_prompt,
                candidate.candidate_text,
                expect_json=False,
                usage_context={
                    "tool": "github_rejudge_ab",
                    "username": candidate.username,
                },
                model_name=model_name,
            )
            parsed = parse_full_evaluation_response(raw)
            row.new_decision = parsed.decision
            row.new_confidence = parsed.confidence
            row.changed = "YES" if parsed.decision != candidate.old_decision else "no"
            results.append(row)
        except ApiBudgetExhaustedError as exc:
            row.error = str(exc)
            results.append(row)
            aborted = True
            break
        except Exception as exc:  # noqa: BLE001 — per-candidate isolation
            row.error = str(exc)
            row.changed = "ERROR"
            results.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render_report(
            state_dir=state_dir,
            brief_path=brief_path,
            model_name=model_name,
            brief_id=str(brief_id),
            results=results,
            selected_count=len(load_result.candidates),
            skipped_foreign=load_result.skipped_foreign,
            skipped_duplicate=load_result.skipped_duplicate,
            old_prompt_sha256s=old_prompt_sha256s,
            aborted=aborted,
        ),
        encoding="utf-8",
    )

    if aborted:
        return 1
    if any(r.error for r in results):
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-judge stored GitHub full-evaluation evidence against the current prompt.",
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    return run_rejudge(
        state_dir=args.state_dir,
        brief_path=args.brief,
        model_name=args.model,
        out_path=args.out,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
