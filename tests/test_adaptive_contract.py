import json

from shared.adaptive import (
    ADAPTATION_EVENT_TYPE,
    AdaptiveAction,
    AdaptationDecision,
    ChannelExhaustion,
    NoiseMarker,
    ScoutMetrics,
    SignalMarker,
    record_adaptation_decision,
)
from shared.runtime_state.store import RuntimeStateStore


def test_adaptation_decision_round_trips_source_native_payload() -> None:
    decision = AdaptationDecision(
        source="github",
        action=AdaptiveAction.EXPERIMENT,
        lane="repo_mining",
        rationale="Repo mining produced maintainer saves while broad user search decayed.",
        work_unit_kind="github_query",
        work_unit_family="oss-maintainers",
        inserted_work_units=["q-101", "q-102"],
        skipped_work_units=["q-7"],
        reordered_work_units=["q-101", "q-8"],
        metrics=ScoutMetrics(
            work_units_run=3,
            candidates_discovered=42,
            candidates_enriched=21,
            facial_yes=8,
            facial_no=11,
            facial_borderline=2,
            saves=3,
            rejects=5,
            insufficient=4,
            signal_markers=[
                SignalMarker(
                    kind="maintainer_signal",
                    label="maintainers in target repos",
                    count=3,
                    examples=["owner/repo"],
                )
            ],
            noise_markers=[
                NoiseMarker(
                    kind="toolchain_noise",
                    label="tutorial forks",
                    count=9,
                    examples=["hello-world-llm"],
                )
            ],
            exhaustion=[
                ChannelExhaustion(
                    channel="code_search",
                    reason="rate_limited",
                    retry_after_seconds=60,
                )
            ],
        ),
        source_payload={
            "native_action": "insert_queries",
            "queries": [{"channel": "repo_mining", "target_repo": "owner/repo"}],
        },
    )

    restored = AdaptationDecision.from_dict(decision.to_dict())

    assert restored == decision
    assert restored.source_payload["queries"][0]["channel"] == "repo_mining"


def test_record_adaptation_decision_writes_runtime_event(tmp_path) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="researcher",
        brief_id="brief-1",
        output_dir=str(tmp_path),
        mode="full",
    )
    decision = AdaptationDecision(
        source="researcher",
        action=AdaptiveAction.BROADEN,
        lane="topic_concepts",
        rationale="Initial venue-filtered scout was sparse.",
        metrics=ScoutMetrics(work_units_run=1, candidates_discovered=0),
        source_payload={"add_topic_concepts": ["machine learning systems"]},
    )

    record_adaptation_decision(store, run_id=run_id, decision=decision)

    with store.connect() as conn:
        row = conn.execute(
            "SELECT event_type, payload_json FROM events WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (run_id,),
        ).fetchone()

    assert row["event_type"] == ADAPTATION_EVENT_TYPE
    payload = json.loads(row["payload_json"])
    assert payload["action"] == "broaden"
    assert payload["source_payload"]["add_topic_concepts"] == ["machine learning systems"]


def test_inserted_work_units_are_stable_ids_across_sources() -> None:
    # Cross-source telemetry joins on inserted_work_units against the
    # work_units table — every source must emit integer-ID strings. Locks
    # in the convention regression-tested across the four adaptive
    # sources (github, researcher, designer, exec_search).
    examples = [
        AdaptationDecision(
            source="github",
            action=AdaptiveAction.EXPERIMENT,
            lane="repo_mining",
            rationale="r",
            inserted_work_units=["99"],
        ),
        AdaptationDecision(
            source="researcher",
            action=AdaptiveAction.BROADEN,
            lane="academic_search",
            rationale="r",
            inserted_work_units=["2", "3"],
        ),
        AdaptationDecision(
            source="designer",
            action=AdaptiveAction.BROADEN,
            lane="google_cse",
            rationale="r",
            inserted_work_units=["17"],
        ),
        AdaptationDecision(
            source="exec_search",
            action=AdaptiveAction.BROADEN,
            lane="acme-health",
            rationale="r",
            inserted_work_units=["1001"],
        ),
    ]
    for decision in examples:
        assert decision.inserted_work_units, decision.source
        for item in decision.inserted_work_units:
            assert item.isdigit(), (
                f"{decision.source} put non-integer {item!r} in inserted_work_units"
            )


def test_lane_is_single_token_across_sources() -> None:
    # The market-intel summarizer pipes ``lane`` straight into the
    # recruiter-facing report. CSV lanes from one source next to single-
    # word lanes from another look incoherent — freeze the contract.
    examples = [
        ("github", "repo_mining"),
        ("researcher", "academic_search"),
        ("designer", "google_cse"),
        ("exec_search", "acme-health"),
    ]
    for source, lane in examples:
        decision = AdaptationDecision(
            source=source,
            action=AdaptiveAction.CONTINUE,
            lane=lane,
            rationale="r",
        )
        assert "," not in decision.lane, f"{source} emitted CSV lane {lane!r}"
