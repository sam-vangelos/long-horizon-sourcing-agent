from __future__ import annotations

import json
from pathlib import Path

from shared.runtime_state.store import RuntimeStateStore
from tools.run_langfuse_eval_loop import main


def test_eval_loop_manifest_validation_requires_response_extractor(tmp_path: Path) -> None:
    manifest_path = tmp_path / "eval_targets.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "name": "missing-field",
                    "output_root": str(tmp_path / "output"),
                    "source": "linkedin",
                    "brief_id": "brief-x",
                    "prompt_id": "prompt-x",
                    "prompt_label": "production",
                    "max_rows": 10,
                }
            ]
        )
    )

    assert main(["--manifest", str(manifest_path), "--dry-run"]) == 2


def test_eval_loop_dry_run_writes_summary_without_langfuse_credentials(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    state_dir = output_root / "state" / "linkedin" / "brief-x"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-x",
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-x"},
    )
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-x",
        identity_key="li-1",
        display_name="Candidate 1",
        profile_url="https://example.test/li-1",
    )
    candidate = store.get_candidate(
        source="linkedin",
        brief_id="brief-x",
        identity_key="li-1",
    )
    store.set_candidate_judgment_accuracy(int(candidate["id"]), "useful")

    manifest_path = tmp_path / "eval_targets.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "name": "linkedin-brief-x",
                    "output_root": str(output_root),
                    "source": "linkedin",
                    "brief_id": "brief-x",
                    "prompt_id": "prompt-x",
                    "prompt_label": "production",
                    "response_extractor": "judgment_accuracy_json",
                    "max_rows": 25,
                }
            ]
        )
    )
    reports_root = tmp_path / "reports"

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--reports-root",
            str(reports_root),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    summary_files = list(reports_root.glob("*/summary.json"))
    assert len(summary_files) == 1
    payload = json.loads(summary_files[0].read_text())
    assert payload["dry_run"] is True
    assert payload["targets"][0]["dataset_name"] == "judgment-accuracy-linkedin-brief-x"
    assert payload["targets"][0]["state_dir"] == str(state_dir)
