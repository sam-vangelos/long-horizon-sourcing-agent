#!/usr/bin/env python3
"""Manifest-driven Langfuse eval-loop orchestration.

Validates a list of eval targets, resolves the corresponding state-dir,
runs dataset sync plus prompt regression in real mode, and emits one
timestamped JSON/Markdown report bundle under ``output/langfuse/``.

Dry-run mode validates the manifest and discovery wiring without
requiring Langfuse or LLM credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_prompt_regression import (
    RESPONSE_EXTRACTORS,
    run_regression_against_dataset,
)
from tools.sync_judgment_datasets import (
    dataset_name_for,
    sync_one,
    _brief_ids_in_state_dir,
    _discover_state_dirs,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalTarget:
    name: str
    output_root: str
    source: str
    brief_id: str
    prompt_id: str
    prompt_label: str
    response_extractor: str
    max_rows: int | None


@dataclass
class TargetRunResult:
    target: EvalTarget
    state_dir: str | None
    dataset_name: str
    dry_run: bool
    sync: dict[str, Any] | None = None
    regression: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": asdict(self.target),
            "state_dir": self.state_dir,
            "dataset_name": self.dataset_name,
            "dry_run": self.dry_run,
            "sync": self.sync,
            "regression": self.regression,
        }


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_manifest(manifest_path: Path) -> list[EvalTarget]:
    if not manifest_path.exists():
        raise RuntimeError(f"manifest does not exist: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not parse manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise RuntimeError("manifest must be a JSON array of eval targets")
    return [_validate_target(item, index=index) for index, item in enumerate(raw)]


def _validate_target(item: Any, *, index: int) -> EvalTarget:
    if not isinstance(item, dict):
        raise RuntimeError(f"manifest target #{index} must be an object")

    required_fields = (
        "name",
        "output_root",
        "source",
        "brief_id",
        "prompt_id",
        "prompt_label",
        "response_extractor",
        "max_rows",
    )
    missing = [field for field in required_fields if field not in item]
    if missing:
        raise RuntimeError(
            f"manifest target #{index} is missing required field(s): {', '.join(missing)}"
        )

    response_extractor = item["response_extractor"]
    if response_extractor not in RESPONSE_EXTRACTORS:
        raise RuntimeError(
            f"manifest target #{index} has invalid response_extractor={response_extractor!r}"
        )

    max_rows = item["max_rows"]
    if max_rows is not None:
        try:
            max_rows = int(max_rows)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"manifest target #{index} has non-integer max_rows={item['max_rows']!r}"
            ) from exc
        if max_rows <= 0:
            raise RuntimeError(
                f"manifest target #{index} must set max_rows > 0 when provided"
            )

    text_fields = {
        "name": item["name"],
        "output_root": item["output_root"],
        "source": item["source"],
        "brief_id": item["brief_id"],
        "prompt_id": item["prompt_id"],
        "prompt_label": item["prompt_label"],
    }
    normalized: dict[str, str] = {}
    for field, value in text_fields.items():
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"manifest target #{index} has invalid {field}={value!r}")
        normalized[field] = value.strip()

    return EvalTarget(
        name=normalized["name"],
        output_root=normalized["output_root"],
        source=normalized["source"],
        brief_id=normalized["brief_id"],
        prompt_id=normalized["prompt_id"],
        prompt_label=normalized["prompt_label"],
        response_extractor=response_extractor,
        max_rows=max_rows,
    )


def _resolve_state_dir(target: EvalTarget) -> Path | None:
    state_dirs = _discover_state_dirs(
        output_root=Path(target.output_root),
        sources=[target.source],
    )
    matches = [
        state_dir
        for source, state_dir in state_dirs
        if source == target.source
        and target.brief_id in _brief_ids_in_state_dir(state_dir)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple state directories matched {target.source}/{target.brief_id}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _run_target(target: EvalTarget, *, dry_run: bool) -> TargetRunResult:
    dataset_name = dataset_name_for(source=target.source, brief_id=target.brief_id)
    state_dir = _resolve_state_dir(target)
    if dry_run:
        return TargetRunResult(
            target=target,
            state_dir=str(state_dir) if state_dir is not None else None,
            dataset_name=dataset_name,
            dry_run=True,
        )
    if state_dir is None:
        raise RuntimeError(
            f"no state directory found for {target.source}/{target.brief_id} under {target.output_root}"
        )

    sync_result = sync_one(
        state_dir=state_dir,
        source=target.source,
        brief_id=target.brief_id,
        dry_run=False,
    )
    if sync_result.failed_count > 0:
        raise RuntimeError(
            f"sync failed for {target.source}/{target.brief_id}: "
            f"{sync_result.failed_count} row(s) failed to push"
        )

    regression = run_regression_against_dataset(
        prompt_id=target.prompt_id,
        dataset_name=dataset_name,
        prompt_label=target.prompt_label,
        response_extractor=target.response_extractor,
        max_rows=target.max_rows,
    )
    return TargetRunResult(
        target=target,
        state_dir=str(state_dir),
        dataset_name=dataset_name,
        dry_run=False,
        sync=asdict(sync_result),
        regression=regression.to_dict(),
    )


def _write_reports(
    *,
    reports_root: Path,
    results: list[TargetRunResult],
    manifest_path: Path,
    dry_run: bool,
) -> tuple[Path, Path]:
    run_dir = reports_root / _timestamp_slug()
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "dry_run": dry_run,
        "targets": [result.to_dict() for result in results],
    }
    json_path = run_dir / "summary.json"
    md_path = run_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown_summary(payload))
    return json_path, md_path


def _render_markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Langfuse Eval Loop",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Dry run: {payload['dry_run']}",
        f"- Manifest: `{payload['manifest_path']}`",
        "",
    ]
    targets = payload.get("targets") or []
    if not targets:
        lines.append("No eval targets were configured.")
        return "\n".join(lines)

    for target_payload in targets:
        target = target_payload["target"]
        lines.append(f"## {target['name']}")
        lines.append("")
        lines.append(f"- Source / brief: `{target['source']}` / `{target['brief_id']}`")
        lines.append(f"- Dataset: `{target_payload['dataset_name']}`")
        lines.append(f"- State dir: `{target_payload.get('state_dir')}`")
        lines.append(f"- Dry run: `{target_payload['dry_run']}`")
        sync = target_payload.get("sync") or {}
        if sync:
            lines.append(
                "- Sync: "
                f"built={sync.get('rows_built', 0)} "
                f"pushed={sync.get('rows_pushed', 0)} "
                f"skipped={sync.get('rows_skipped_idempotent', 0)} "
                f"failed={sync.get('failed_count', 0)}"
            )
        regression = target_payload.get("regression") or {}
        if regression:
            lines.append(
                "- Regression: "
                f"rows={regression.get('rows_evaluated', 0)} "
                f"agreement={regression.get('aggregate_agreement_rate', 0.0)} "
                f"weighted={regression.get('aggregate_weighted_agreement_rate', 0.0)}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_langfuse_eval_loop",
        description=(
            "Validate the Langfuse eval manifest, resolve state directories, "
            "run dataset sync plus prompt regression, and emit report bundles."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="config/langfuse/eval_targets.json",
        help="Path to the eval target manifest JSON.",
    )
    parser.add_argument(
        "--reports-root",
        default="output/langfuse",
        help="Directory to write timestamped summary bundles into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest + discovery wiring only; skip Langfuse calls.",
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

    manifest_path = Path(args.manifest)
    reports_root = Path(args.reports_root)

    try:
        targets = _load_manifest(manifest_path)
        results = [_run_target(target, dry_run=args.dry_run) for target in targets]
        json_path, md_path = _write_reports(
            reports_root=reports_root,
            results=results,
            manifest_path=manifest_path,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        logger.error("eval loop failed: %s", exc)
        return 2

    logger.info("wrote eval-loop reports: %s %s", json_path, md_path)
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
