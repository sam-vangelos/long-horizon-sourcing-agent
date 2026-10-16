from __future__ import annotations

import csv
import json
from pathlib import Path

from github.export import export_saved_candidates_csv
from shared.runtime_state.store import RuntimeStateStore


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _candidate(
    *,
    username: str,
    name: str,
    confidence_level_payload: dict | None = None,
) -> dict:
    return {
        "user": {
            "username": username,
            "name": name,
            "bio": "Open source maintainer",
            "company": "Example OSS",
            "location": "Remote",
            "profile_url": f"https://github.com/{username}",
        },
        "contact": {
            "emails": [f"{username}@example.com"],
            "linkedin_url": "",
            "website": "",
        },
        "top_repos": [
            {"name": "scheduler", "stars": 1200},
        ],
        "portfolio_summary": {},
        "maintainership": confidence_level_payload,
        "source_query": "repo:kubernetes/kubernetes",
        "source_strategy": "repo_mining",
    }


def _full_judgment(
    *,
    candidate_name: str,
    decision: str,
    confidence: float,
) -> dict:
    return {
        "candidate_name": candidate_name,
        "stage": "full",
        "decision": decision,
        "confidence": confidence,
        "path": "DIRECT:Infrastructure",
        "rationale": "Strong OSS evidence.",
    }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_export_saved_candidates_csv_preserves_maintainership_columns(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _append_jsonl(
        output_dir / "candidates.jsonl",
        _candidate(
            username="alice",
            name="Alice Doe",
            confidence_level_payload={
                "level": "maintainer",
                "confidence": 0.78,
                "evidence_sources": [
                    "merge_authority:kubernetes/kubernetes:23PRs",
                    "maintainers_file:kubernetes/kubernetes",
                ],
                "signals": {"budget_exhausted": False},
            },
        ),
    )
    _append_jsonl(
        output_dir / "candidates.jsonl",
        _candidate(
            username="bob",
            name="Bob Roe",
            confidence_level_payload=None,
        ),
    )
    _append_jsonl(
        output_dir / "final_judgments.jsonl",
        _full_judgment(candidate_name="Alice Doe", decision="SAVE", confidence=0.91),
    )
    _append_jsonl(
        output_dir / "final_judgments.jsonl",
        _full_judgment(
            candidate_name="Bob Roe",
            decision="INFERENTIAL_SAVE",
            confidence=0.42,
        ),
    )

    csv_path = export_saved_candidates_csv(output_dir)

    fieldnames, rows = _read_csv(csv_path)
    assert "Maintainership Level" in fieldnames
    assert "Maintainership Confidence" in fieldnames
    assert "Maintainership Evidence" in fieldnames
    assert "Maintainership Target Project" in fieldnames

    by_username = {row["GitHub Username"]: row for row in rows}
    alice = by_username["alice"]
    assert alice["Maintainership Level"] == "maintainer"
    assert alice["Maintainership Confidence"] == "0.78"
    assert "merge_authority:kubernetes/kubernetes:23PRs" in alice[
        "Maintainership Evidence"
    ]
    assert "maintainers_file:kubernetes/kubernetes" in alice[
        "Maintainership Evidence"
    ]
    # Inferred evidence tokens are target-project-scoped by construction
    # (classify runs only against brief.target_projects), so they populate
    # the target-project column.
    assert alice["Maintainership Target Project"] == "kubernetes/kubernetes"

    bob = by_username["bob"]
    assert bob["Maintainership Level"] == ""
    assert bob["Maintainership Confidence"] == ""
    assert bob["Maintainership Evidence"] == ""
    assert bob["Maintainership Target Project"] == ""


def test_export_confidence_blank_when_unknown(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _append_jsonl(
        output_dir / "candidates.jsonl",
        _candidate(
            username="carol",
            name="Carol Coe",
            confidence_level_payload={
                "level": "maintainer",
                "confidence": None,
                "role_certainty": "declared",
                "evidence_sources": ["declared:kubernetes/kubernetes:.github/CODEOWNERS"],
                "signals": {},
            },
        ),
    )
    _append_jsonl(
        output_dir / "final_judgments.jsonl",
        _full_judgment(candidate_name="Carol Coe", decision="SAVE", confidence=0.91),
    )

    csv_path = export_saved_candidates_csv(output_dir)
    _, rows = _read_csv(csv_path)
    carol = rows[0]
    assert carol["Maintainership Level"] == "maintainer"
    assert carol["Maintainership Confidence"] == ""


def test_export_target_project_from_declared_evidence_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _append_jsonl(
        output_dir / "candidates.jsonl",
        _candidate(
            username="dana",
            name="Dana Doe",
            confidence_level_payload={
                "level": "maintainer",
                "confidence": 0.78,
                "role_certainty": "declared",
                "evidence_sources": [
                    # Both token families are target-project-scoped upstream:
                    # the classifier only runs against target_projects, and
                    # declared entries filter to target projects at the
                    # classify seam (W3-PX1). Both therefore populate the
                    # Maintainership Target Project column.
                    "declared:kubernetes/kubernetes:.github/CODEOWNERS",
                    "merge_authority:kubernetes/kubernetes:2PRs",
                ],
                "signals": {},
            },
        ),
    )
    _append_jsonl(
        output_dir / "final_judgments.jsonl",
        _full_judgment(candidate_name="Dana Doe", decision="SAVE", confidence=0.91),
    )

    csv_path = export_saved_candidates_csv(output_dir)
    _, rows = _read_csv(csv_path)
    dana = rows[0]
    assert dana["Maintainership Target Project"] == "kubernetes/kubernetes"


def _seed_github_run(state_dir: Path, brief_id: str = "brief-github-1") -> tuple[RuntimeStateStore, int]:
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="github",
        brief_id=brief_id,
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": brief_id},
    )
    return store, run_id


def _github_candidate_record(
    *,
    username: str,
    name: str,
    linkedin_url: str = "",
    maintainership: dict | None = None,
) -> dict:
    return {
        "username": username,
        "user": {
            "username": username,
            "name": name,
            "bio": "Open source maintainer",
            "company": "Example OSS",
            "location": "Remote",
            "profile_url": f"https://github.com/{username}",
        },
        "contact": {
            "emails": [f"{username}@example.com"],
            "linkedin_url": linkedin_url,
            "website": "",
        },
        "top_repos": [{"name": "scheduler", "stars": 1200}],
        "portfolio_summary": {},
        "maintainership": maintainership,
        "source_query": "repo:kubernetes/kubernetes",
        "source_strategy": "repo_mining",
    }


def _save_github_candidate(
    store: RuntimeStateStore,
    *,
    run_id: int,
    brief_id: str,
    identity_key: str,
    display_name: str,
    candidate_record: dict,
    decision: str = "SAVE",
    confidence: float = 0.91,
    person_key: str | None = None,
    outreach_copy: dict | None = None,
) -> None:
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="github",
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=display_name,
        profile_url=f"https://github.com/{identity_key}",
        person_key=person_key,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="github",
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="snippet_extracted",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="github",
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_started",
    )
    full_id = store.start_attempt(
        run_id=run_id,
        source="github",
        brief_id=brief_id,
        identity_key=identity_key,
        stage="full",
    )
    if outreach_copy is not None:
        candidate_record = {
            **candidate_record,
            "outreach_copy": outreach_copy,
        }
    terminal_payload = {
        "full_decision": {
            "decision": decision,
            "confidence": confidence,
            "path": "DIRECT:Infrastructure",
            "rationale": "Strong OSS evidence.",
            "candidate_name": display_name,
            "profile_url": f"https://github.com/{identity_key}",
        },
        "candidate_record": candidate_record,
    }
    store.finish_attempt_success(
        attempt_id=full_id,
        new_state="full_terminal",
        terminal_decision=decision,
        payload=terminal_payload,
        run_id=run_id,
    )


def test_export_reads_canonical_runtime_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "run"
    store, run_id = _seed_github_run(state_dir, brief_id="brief-export-1")

    _save_github_candidate(
        store,
        run_id=run_id,
        brief_id="brief-export-1",
        identity_key="alice-one",
        display_name="Alice Doe",
        candidate_record=_github_candidate_record(username="alice-one", name="Alice Doe"),
        confidence=0.91,
    )
    _save_github_candidate(
        store,
        run_id=run_id,
        brief_id="brief-export-1",
        identity_key="bob-two",
        display_name="Bob Roe",
        candidate_record=_github_candidate_record(username="bob-two", name="Bob Roe"),
        decision="INFERENTIAL_SAVE",
        confidence=0.42,
    )

    csv_path = export_saved_candidates_csv(state_dir)
    fieldnames, rows = _read_csv(csv_path)

    assert "github_username" in fieldnames
    assert len(rows) == 2
    by_username = {row["github_username"]: row for row in rows}
    assert by_username["alice-one"]["Decision"] == "SAVE"
    assert by_username["alice-one"]["Confidence"] == "0.91"
    assert by_username["alice-one"]["Evaluation Summary"] == "Strong OSS evidence."
    assert by_username["bob-two"]["Decision"] == "INFERENTIAL_SAVE"
    assert by_username["bob-two"]["Confidence"] == "0.42"


def test_duplicate_display_names_do_not_collapse(tmp_path: Path) -> None:
    state_dir = tmp_path / "run"
    store, run_id = _seed_github_run(state_dir, brief_id="brief-dup-names")

    shared_name = "Alex Chen"
    _save_github_candidate(
        store,
        run_id=run_id,
        brief_id="brief-dup-names",
        identity_key="alex-alpha",
        display_name=shared_name,
        candidate_record=_github_candidate_record(username="alex-alpha", name=shared_name),
        confidence=0.88,
    )
    _save_github_candidate(
        store,
        run_id=run_id,
        brief_id="brief-dup-names",
        identity_key="alex-beta",
        display_name=shared_name,
        candidate_record=_github_candidate_record(username="alex-beta", name=shared_name),
        confidence=0.77,
    )

    csv_path = export_saved_candidates_csv(state_dir)
    _, rows = _read_csv(csv_path)

    assert len(rows) == 2
    usernames = {row["github_username"] for row in rows}
    assert usernames == {"alex-alpha", "alex-beta"}
    assert all(row["First Name"] == "Alex" and row["Last Name"] == "Chen" for row in rows)


def test_export_carries_identity_handles(tmp_path: Path) -> None:
    state_dir = tmp_path / "run"
    store, run_id = _seed_github_run(state_dir, brief_id="brief-identity")

    linkedin = "https://linkedin.com/in/ada-oss"
    _save_github_candidate(
        store,
        run_id=run_id,
        brief_id="brief-identity",
        identity_key="ada-oss",
        display_name="Ada Lovelace",
        candidate_record=_github_candidate_record(
            username="ada-oss",
            name="Ada Lovelace",
            linkedin_url=linkedin,
        ),
        person_key="gh:ada-oss",
    )

    csv_path = export_saved_candidates_csv(state_dir)
    _, rows = _read_csv(csv_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["github_username"] == "ada-oss"
    assert row["github_profile_url"] == "https://github.com/ada-oss"
    assert row["person_key"] == "gh:ada-oss"
    assert row["linkedin_url"] == linkedin
    assert row["LinkedIn URL"] == linkedin


def test_export_reads_canonical_outreach_copy(tmp_path: Path) -> None:
    state_dir = tmp_path / "run"
    store, run_id = _seed_github_run(state_dir, brief_id="brief-outreach")

    outreach = {
        "subject_line": "Open source infra role",
        "message": "Hi Alice — your scheduler work caught our eye.",
    }
    _save_github_candidate(
        store,
        run_id=run_id,
        brief_id="brief-outreach",
        identity_key="alice-one",
        display_name="Alice Doe",
        candidate_record=_github_candidate_record(username="alice-one", name="Alice Doe"),
        outreach_copy=outreach,
    )

    csv_path = export_saved_candidates_csv(state_dir)
    _, rows = _read_csv(csv_path)

    assert len(rows) == 1
    assert rows[0]["Outreach Subject"] == outreach["subject_line"]
    assert rows[0]["Outreach Message"] == outreach["message"]


def test_export_covers_all_brief_ids(tmp_path: Path) -> None:
    state_dir = tmp_path / "run"
    store, run_id_1 = _seed_github_run(state_dir, brief_id="brief-alpha")
    _save_github_candidate(
        store,
        run_id=run_id_1,
        brief_id="brief-alpha",
        identity_key="alpha-dev",
        display_name="Alpha Dev",
        candidate_record=_github_candidate_record(username="alpha-dev", name="Alpha Dev"),
        confidence=0.88,
    )

    run_id_2 = store.start_run(
        source="github",
        brief_id="brief-beta",
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-beta"},
    )
    _save_github_candidate(
        store,
        run_id=run_id_2,
        brief_id="brief-beta",
        identity_key="beta-dev",
        display_name="Beta Dev",
        candidate_record=_github_candidate_record(username="beta-dev", name="Beta Dev"),
        confidence=0.77,
    )

    csv_path = export_saved_candidates_csv(state_dir)
    _, rows = _read_csv(csv_path)

    assert len(rows) == 2
    usernames = {row["github_username"] for row in rows}
    assert usernames == {"alpha-dev", "beta-dev"}


def test_legacy_profile_url_miss_falls_through_to_unambiguous_name(tmp_path: Path) -> None:
    output_dir = tmp_path / "legacy"
    _append_jsonl(
        output_dir / "candidates.jsonl",
        _candidate(username="alice", name="Alice Doe"),
    )
    _append_jsonl(
        output_dir / "final_judgments.jsonl",
        {
            "candidate_name": "Alice Doe",
            "stage": "full",
            "decision": "SAVE",
            "confidence": 0.91,
            "path": "DIRECT:Infrastructure",
            "rationale": "Strong OSS evidence.",
            "profile_url": "https://github.com/nonexistent-user",
        },
    )

    csv_path = export_saved_candidates_csv(output_dir)
    _, rows = _read_csv(csv_path)

    assert len(rows) == 1
    assert rows[0]["GitHub Username"] == "alice"
    assert rows[0]["First Name"] == "Alice"
    assert rows[0]["Last Name"] == "Doe"
