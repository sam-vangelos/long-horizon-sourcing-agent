import json
from pathlib import Path

from exec_search.budget import DossierSpendTracker
from exec_search.orchestrator import ExecSearchPipeline
from shared.brief_loader import load_brief
from shared.runtime_state.exec_search import ExecSearchRuntimeStateBridge
from shared.runtime_state.store import EXEC_SEARCH_QUERY_KIND, RuntimeStateStore
from shared.schemas import CandidateProfileSummary, Experience


def _write_brief(tmp_path: Path) -> Path:
    payload = {
        "role_title": "Chief Product Officer",
        "role_level": "Executive",
        "role_summary": "Owns product strategy for a growth-stage company.",
        "geography": "United States",
        "minimum_years_experience": 12,
        "minimum_bar_description": "Scaled product orgs at growth-stage companies.",
        "linkedin_project": "exec",
        "capability_areas": [
            {
                "name": "Product org leadership",
                "description": "Owns strategy and product operating cadence.",
                "builder_signals": ["CPO scope", "multi-team product org"],
                "user_signals": ["single product line only"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns company-level product strategy.",
            "user_definition": "Owns one squad or feature area.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "target_modules": ["exec_search"],
        "source_config": {
            "exec_search": {
                "target_companies": ["Acme Health"],
                "target_titles": ["Chief Product Officer"],
                "career_path_hypotheses": ["VP Product at growth-stage healthtech"],
            }
        },
    }
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exec_search_pipeline_mocked_end_to_end_persists_dossier_decision(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path)
    brief = load_brief(brief_path)
    state_dir = tmp_path / "exec_search" / "key"
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    bridge = ExecSearchRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)
    candidate = CandidateProfileSummary(
        name="Casey Operator",
        profile_url="https://linkedin.com/in/casey-operator",
        headline="Chief Product Officer at Acme Health",
        experiences=[
            Experience(
                title="Chief Product Officer",
                company="Acme Health",
                summary_bullets=["Scaled product org from 8 to 55 and owned product strategy."],
            )
        ],
        skills_snippet=["Product strategy", "Executive leadership"],
    )

    def discover(lane: dict) -> list[CandidateProfileSummary]:
        if lane.get("title") == "Chief Product Officer":
            return [candidate]
        return []

    pipeline = ExecSearchPipeline(
        brief=brief,
        bridge=bridge,
        candidate_discoverer=discover,
        signal_sources=(),
        full_llm_caller=lambda _system, _user: (
            "DECISION: SAVE\n"
            "PATH: company_scope\n"
            "CONFIDENCE: 0.9\n"
            "DOSSIER_RATIONALE: Casey has CPO-level scope at the target company and clear org-scaling evidence.\n\n"
            "The dossier is thin on public-web signals in this mocked run, but profile evidence clears the thesis."
        ),
    )

    stats = pipeline.run(run_id=run_id)

    assert stats.candidates_discovered == 1
    assert stats.saves == 1
    work_units = store.list_work_units(run_id, kind=EXEC_SEARCH_QUERY_KIND)
    assert work_units
    assert all(wu["status"] == "done" for wu in work_units)

    with store.connect() as conn:
        candidate_row = conn.execute(
            "SELECT terminal_decision, terminal_payload_json FROM candidates WHERE source='exec_search'"
        ).fetchone()
        event = conn.execute(
            "SELECT payload_json FROM events WHERE run_id = ? AND event_type='adaptation_decision'",
            (run_id,),
        ).fetchone()

    assert candidate_row["terminal_decision"] == "SAVE"
    payload = json.loads(candidate_row["terminal_payload_json"])
    assert payload["surface_type"] == "exec_search_dossier"
    assert "Candidate profile" in payload["dossier_evidence"]
    assert payload["full_decision"]["path"] == "company_scope"
    assert event is not None

    # Exec Search skips facial triage entirely — no facial attempt rows
    # should exist after the run. (Cross-source facial-rate rollups depend
    # on this contract.)
    with store.connect() as conn:
        stage_rows = conn.execute(
            "SELECT stage, COUNT(*) AS n FROM candidate_attempts "
            "WHERE run_id = ? GROUP BY stage",
            (run_id,),
        ).fetchall()
    stages = {row["stage"]: row["n"] for row in stage_rows}
    assert "facial" not in stages
    assert stages.get("snippet", 0) == 1
    assert stages.get("full", 0) == 1


def _write_two_lane_brief(tmp_path: Path) -> Path:
    payload = {
        "role_title": "Chief Product Officer",
        "role_level": "Executive",
        "role_summary": "Owns product strategy.",
        "geography": "United States",
        "minimum_years_experience": 12,
        "minimum_bar_description": "Scaled product orgs.",
        "linkedin_project": "exec",
        "capability_areas": [
            {"name": "Product org leadership", "description": "Strategy and operating cadence."}
        ],
        "depth_distinction": {
            "builder_definition": "Owns company-level product strategy.",
            "user_definition": "Owns one squad.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "target_modules": ["exec_search"],
        "source_config": {
            "exec_search": {
                "target_companies": ["Acme Health", "Beta Robotics"],
                "target_titles": ["Chief Product Officer"],
            }
        },
    }
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_budget_not_burned_by_disabled_signals(tmp_path: Path, monkeypatch) -> None:
    # When provider env vars are unset, the orchestrator must NOT
    # reserve cost for those sources. Previously every reservation
    # debited the cap even though the disabled adapters returned
    # SignalFailure — silently burning the recruiter's cap.
    for env_var in ("CRUNCHBASE_API_KEY", "PITCHBOOK_API_KEY", "NEWSAPI_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    brief_path = _write_brief(tmp_path)
    brief = load_brief(brief_path)
    state_dir = tmp_path / "exec_search" / "no-signals"
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    bridge = ExecSearchRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)

    cand = CandidateProfileSummary(
        name="Casey Operator",
        profile_url="https://linkedin.com/in/casey-operator",
        headline="CPO at Acme",
        experiences=[
            Experience(title="CPO", company="Acme Health", summary_bullets=["Scaled."])
        ],
        skills_snippet=["Strategy"],
    )

    pipeline = ExecSearchPipeline(
        brief=brief,
        bridge=bridge,
        candidate_discoverer=lambda lane: (
            [cand] if lane.get("title") == "Chief Product Officer" else []
        ),
        # Explicit signal_sources includes the env-gated ones so the
        # availability probe gets exercised.
        signal_sources=("crunchbase", "pitchbook", "news"),
        full_llm_caller=lambda _s, _u: "DECISION: SAVE\nPATH: x\nCONFIDENCE: 0.9\nSUMMARY: ok",
    )

    pipeline.run(run_id=run_id)

    # None of the disabled signal sources should have reserved cost
    # against the tracker. The Opus full-eval reservation is fine
    # (that's the only thing that genuinely costs money in this test).
    by_source = dict(pipeline.spend_tracker.by_source)
    for env_gated_source in ("crunchbase", "pitchbook", "news"):
        assert env_gated_source not in by_source, (
            f"{env_gated_source} burned cap despite missing env var"
        )

    # The run completed normally — no budget_exhausted, no error.
    with store.connect() as conn:
        run_row = conn.execute(
            "SELECT status, stop_reason FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    assert run_row["status"] == "completed"
    assert run_row["stop_reason"] == "normal"


def test_budget_exhausted_stops_run_and_skips_remaining_lanes(tmp_path: Path) -> None:
    brief_path = _write_two_lane_brief(tmp_path)
    brief = load_brief(brief_path)
    state_dir = tmp_path / "exec_search" / "budget"
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    bridge = ExecSearchRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)

    def candidate(name: str, company: str) -> CandidateProfileSummary:
        return CandidateProfileSummary(
            name=name,
            profile_url=f"https://linkedin.com/in/{name.lower().replace(' ', '-')}",
            headline=f"Chief Product Officer at {company}",
            experiences=[
                Experience(
                    title="Chief Product Officer",
                    company=company,
                    summary_bullets=["Scaled product org."],
                )
            ],
            skills_snippet=["Product strategy"],
        )

    discovered_lanes: list[str] = []

    def discover(lane: dict) -> list[CandidateProfileSummary]:
        discovered_lanes.append(lane.get("company") or "")
        return [candidate("Alex Operator", lane.get("company") or "")]

    pipeline = ExecSearchPipeline(
        brief=brief,
        bridge=bridge,
        candidate_discoverer=discover,
        signal_sources=(),
        full_llm_caller=lambda _s, _u: "DECISION: SAVE\nPATH: x\nCONFIDENCE: 0.9\nSUMMARY: ok",
        spend_tracker=DossierSpendTracker(cap_usd=0.0),
    )

    stats = pipeline.run(run_id=run_id)

    work_units = sorted(
        store.list_work_units(run_id, kind=EXEC_SEARCH_QUERY_KIND),
        key=lambda wu: wu["ordering_index"],
    )
    # The exact lane count depends on the strategy's title defaults; what
    # matters is that exactly one lane completes (the one that consumed
    # the cap) and every other lane lands at status="skipped" — no
    # lingering "in_progress" / "queued" rows.
    statuses = [wu["status"] for wu in work_units]
    assert statuses.count("done") == 1
    assert statuses.count("skipped") == len(statuses) - 1
    assert work_units[0]["status"] == "done"
    # Discovery only ran for the first lane — every other lane was skipped
    # before its discoverer was invoked.
    assert len(discovered_lanes) == 1

    with store.connect() as conn:
        budget_events = conn.execute(
            "SELECT payload_json FROM events WHERE run_id = ? AND event_type = 'budget_exhausted'",
            (run_id,),
        ).fetchall()
        run_row = conn.execute(
            "SELECT status, stop_reason FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    assert len(budget_events) == 1
    assert run_row["status"] == "interrupted"
    assert run_row["stop_reason"] == "api_budget_exhausted"

    log_path = state_dir / "run_log.jsonl"
    log_lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    end_entry = next(entry for entry in reversed(log_lines) if entry.get("event") == "pipeline_end")
    assert end_entry["status"] == "budget_exhausted"

    # Stats reflect the partial completion.
    assert stats.lanes_completed == 1
    assert stats.saves == 0  # cap fired before the SAVE was recorded
