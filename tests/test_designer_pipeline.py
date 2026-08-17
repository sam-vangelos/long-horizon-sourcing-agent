import asyncio
import json
from pathlib import Path

import pytest

from designer.image_acquisition import AssetAcquisitionResult, AssetCache
from designer.orchestrator import DesignerPipeline, _hydrate_default_rubric
from designer.schemas import DesignerCandidate, DesignerSearchQuery, DesignerSnippet
from shared.runtime_state.designer import DesignerRuntimeStateBridge
from shared.runtime_state.store import DESIGNER_CSE_QUERY_KIND, RuntimeStateStore


def _brief() -> dict:
    return {
        "id": "designer-pipeline-test",
        "role_title": "Senior Product Designer",
        "role_summary": "Design systems and high-craft product surfaces.",
        "capability_areas": [
            {"name": "Design systems", "description": "Reusable UI systems."}
        ],
        "design_rubric": {
            "principles": [
                {
                    "name": "Systems craft",
                    "description": "Reusable product systems.",
                    "anchors": {
                        "bad": "No system evidence.",
                        "okay": "Some consistency.",
                        "good": "Clear reusable interface system.",
                        "excellent": "Exceptional reusable interface system with strong craft.",
                    },
                }
            ],
            "hard_reject_patterns": [],
        },
    }


def test_designer_pipeline_mocked_end_to_end_persists_terminal_decision(tmp_path: Path) -> None:
    state_dir = tmp_path / "designer" / "key"
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _brief()
    bridge = DesignerRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief["id"],
        brief_name=brief["role_title"],
    )
    run_id = bridge.start_or_resume_run(resume=False)
    query = DesignerSearchQuery(
        source="google_cse",
        query_text="design systems portfolio",
        capability_area_name="Design systems",
        discipline="product",
    )
    snippet = DesignerSnippet(
        source="google_cse",
        identity_key="cse:example.com/portfolio",
        display_name="Alex Designer",
        profile_url="https://example.com/portfolio",
        headline="Product designer focused on systems",
        fields=("Product Design",),
        tools=("Figma",),
        top_project_titles=("Design System Revamp",),
    )

    def candidate_acquirer(_query: DesignerSearchQuery) -> list[DesignerCandidate]:
        return [DesignerCandidate(snippet=snippet)]

    def asset_acquirer(candidate: DesignerCandidate) -> AssetAcquisitionResult:
        cache = AssetCache(state_dir / "assets.sqlite3")
        asset = cache.write_asset(
            candidate_identity_key=candidate.snippet.identity_key,
            asset_url="https://example.com/thumb.jpg",
            source="google_cse",
            image_bytes=b"fake-jpeg-bytes",
            tos_source="test",
            project_title="Design System Revamp",
        )
        return AssetAcquisitionResult(cached_assets=(asset,), failed_urls=())

    pipeline = DesignerPipeline(
        brief=brief,
        bridge=bridge,
        queries=[query],
        candidate_acquirer=candidate_acquirer,
        asset_acquirer=asset_acquirer,
        facial_llm_caller=lambda _s, _u: (
            "DECISION: FACIAL_YES\n"
            "REASON: Portfolio text shows design systems work."
        ),
        full_llm_caller=lambda _s, _u: (
            "DECISION: SAVE\n"
            "PATH: design_systems\n"
            "CONFIDENCE: 0.86\n"
            "SUMMARY: Strong product systems portfolio with relevant project context."
        ),
        vision_llm_call=lambda _model, _system, _user, _images: {
            "principles": [
                {
                    "name": "Systems craft",
                    "score": 3,
                    "reasoning": "image_id 0 shows an exceptional reusable interface system with strong craft.",
                    "image_ids": [0],
                }
            ],
            "overall_verdict": "yes",
            "overall_confidence": 0.91,
            "overall_reasoning": "Strong visual evidence.",
        },
    )

    stats = asyncio.run(pipeline.run(run_id=run_id))

    assert stats.candidates_discovered == 1
    assert stats.saves == 1
    work_units = store.list_work_units(run_id, kind=DESIGNER_CSE_QUERY_KIND)
    assert len(work_units) == 1
    assert work_units[0]["status"] == "done"

    with store.connect() as conn:
        candidate = conn.execute(
            "SELECT terminal_decision, terminal_payload_json FROM candidates WHERE source='designer'"
        ).fetchone()
        event = conn.execute(
            "SELECT payload_json FROM events WHERE run_id = ? AND event_type='adaptation_decision'",
            (run_id,),
        ).fetchone()

    assert candidate["terminal_decision"] == "SAVE"
    payload = json.loads(candidate["terminal_payload_json"])
    assert payload["full_decision"]["decision"] == "SAVE"
    assert payload["visual_judgment"]["overall_verdict"] == "yes"
    assert payload["visual_judgment"]["principles"][0]["name"] == "Systems craft"
    assert event is not None
    # Vision cost rolled up into pipeline_end.cost_usd (previously hardcoded
    # to 0.0). Cross-module cost rollup depends on this.
    log_path = state_dir / "run_log.jsonl"
    log_lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    end_entry = next(entry for entry in reversed(log_lines) if entry.get("event") == "pipeline_end")
    assert "cost_usd" in end_entry
    assert end_entry["cost_usd"] > 0.0
    assert stats.cost_usd == end_entry["cost_usd"]


def test_sniff_image_mime_recognizes_common_formats() -> None:
    from designer.vision_evaluation import _sniff_image_mime

    assert _sniff_image_mime(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert _sniff_image_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    # WebP requires the RIFF/WEBP frame header.
    assert (
        _sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 rest")
        == "image/webp"
    )
    assert _sniff_image_mime(b"GIF89a-rest") == "image/gif"
    # Unknown magic falls back to JPEG (Gemini's most permissive default).
    assert _sniff_image_mime(b"unknown bytes") == "image/jpeg"


def test_vision_provider_unavailable_emits_one_event_and_disables_vision(
    tmp_path: Path, monkeypatch
) -> None:
    # Strip the env so the upfront probe disables vision. The orchestrator
    # should emit exactly one ``provider_unavailable`` event and short-
    # circuit every candidate to INFERENTIAL_SAVE without invoking the
    # vision LLM caller.
    from shared import config

    monkeypatch.setattr(config, "GOOGLE_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "DESIGNER_VISION_FALLBACK_MODEL_NAME", "", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("DESIGNER_VISION_FALLBACK_MODEL_NAME", raising=False)

    state_dir = tmp_path / "designer" / "no-vision"
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _brief()
    bridge = DesignerRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief["id"],
        brief_name=brief["role_title"],
    )
    run_id = bridge.start_or_resume_run(resume=False)
    query = DesignerSearchQuery(
        source="google_cse",
        query_text="design systems portfolio",
        capability_area_name="Design systems",
        discipline="product",
    )
    snippet = DesignerSnippet(
        source="google_cse",
        identity_key="cse:example.com/portfolio",
        display_name="Alex Designer",
        profile_url="https://example.com/portfolio",
        headline="Product designer focused on systems",
        fields=("Product Design",),
        tools=("Figma",),
        top_project_titles=("Design System Revamp",),
    )

    def candidate_acquirer(_query: DesignerSearchQuery) -> list[DesignerCandidate]:
        return [DesignerCandidate(snippet=snippet)]

    pipeline = DesignerPipeline(
        brief=brief,
        bridge=bridge,
        queries=[query],
        candidate_acquirer=candidate_acquirer,
        facial_llm_caller=lambda _s, _u: "DECISION: FACIAL_YES\nREASON: ok",
        full_llm_caller=lambda _s, _u: (
            "DECISION: SAVE\nPATH: design_systems\nCONFIDENCE: 0.8\nSUMMARY: ok"
        ),
        # vision_llm_call left at default so the probe takes the
        # provider-unavailable path.
    )

    asyncio.run(pipeline.run(run_id=run_id))

    log_path = state_dir / "run_log.jsonl"
    log_lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    vision_events = [
        entry for entry in log_lines
        if entry.get("event") == "provider_unavailable"
        and entry.get("provider") == "vision_primary"
    ]
    assert len(vision_events) == 1
    assert vision_events[0]["fallback_available"] is False
    assert pipeline.vision_disabled is True

    with store.connect() as conn:
        candidate_row = conn.execute(
            "SELECT terminal_decision, terminal_payload_json FROM candidates WHERE source='designer'"
        ).fetchone()
    payload = json.loads(candidate_row["terminal_payload_json"])
    assert candidate_row["terminal_decision"] == "INFERENTIAL_SAVE"
    assert payload["visual_judgment"]["fallback_reason"] == "provider_unavailable_pre_check"
    assert payload["visual_judgment"]["skipped"] is True


def test_designer_pipeline_error_path_closes_run_and_emits_pipeline_error(tmp_path: Path) -> None:
    state_dir = tmp_path / "designer" / "err"
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _brief()
    bridge = DesignerRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief["id"],
        brief_name=brief["role_title"],
    )
    run_id = bridge.start_or_resume_run(resume=False)
    query = DesignerSearchQuery(
        source="google_cse",
        query_text="design systems portfolio",
        capability_area_name="Design systems",
        discipline="product",
    )

    def boom_acquirer(_query: DesignerSearchQuery) -> list[DesignerCandidate]:
        raise RuntimeError("acquirer exploded")

    pipeline = DesignerPipeline(
        brief=brief,
        bridge=bridge,
        queries=[query],
        candidate_acquirer=boom_acquirer,
    )

    with pytest.raises(RuntimeError, match="acquirer exploded"):
        asyncio.run(pipeline.run(run_id=run_id))

    with store.connect() as conn:
        run_row = conn.execute(
            "SELECT status, stop_reason, stop_reason_detail FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    assert run_row["status"] == "error"
    assert run_row["stop_reason"] == "fatal_runtime_error"
    assert run_row["stop_reason_detail"] == "error: RuntimeError"

    log_path = state_dir / "run_log.jsonl"
    log_lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    log_names = [entry.get("event") for entry in log_lines]
    assert "pipeline_error" in log_names
    assert log_names[-1] == "pipeline_end"
    end_entry = next(entry for entry in reversed(log_lines) if entry.get("event") == "pipeline_end")
    assert end_entry["status"] == "error"


# ---------------------------------------------------------------------------
# D4: rubric hydration
# ---------------------------------------------------------------------------


def test_hydrate_default_rubric_injects_when_missing() -> None:
    brief = {"target_modules": ["designer"]}
    _hydrate_default_rubric(brief)
    rubric = brief.get("design_rubric")
    assert isinstance(rubric, dict)
    assert len(rubric["principles"]) == 6


def test_hydrate_default_rubric_preserves_recruiter_authored() -> None:
    custom_rubric = {
        "principles": [{"name": "Custom", "description": "test"}],
    }
    brief = {"target_modules": ["designer"], "design_rubric": custom_rubric}
    _hydrate_default_rubric(brief)
    assert brief["design_rubric"] is custom_rubric
    assert len(brief["design_rubric"]["principles"]) == 1


def test_hydrate_default_rubric_skips_non_designer_brief() -> None:
    brief = {"target_modules": ["linkedin"]}
    _hydrate_default_rubric(brief)
    assert "design_rubric" not in brief


# ---------------------------------------------------------------------------
# D5b: recommendation pitch in pipeline output
# ---------------------------------------------------------------------------


def test_pipeline_output_includes_recommendation_pitch(tmp_path: Path) -> None:
    state_dir = tmp_path / "designer" / "pitch"
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _brief()
    bridge = DesignerRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief["id"],
        brief_name=brief["role_title"],
    )
    run_id = bridge.start_or_resume_run(resume=False)
    query = DesignerSearchQuery(
        source="google_cse",
        query_text="design systems portfolio",
        capability_area_name="Design systems",
        discipline="product",
    )
    snippet = DesignerSnippet(
        source="google_cse",
        identity_key="cse:pitch-test/portfolio",
        display_name="Pitch Test Designer",
        profile_url="https://example.com/pitch-portfolio",
        headline="Product designer",
        fields=("Product Design",),
        tools=("Figma",),
        top_project_titles=("Case Study",),
    )

    def candidate_acquirer(_query: DesignerSearchQuery) -> list[DesignerCandidate]:
        return [DesignerCandidate(snippet=snippet)]

    def asset_acquirer(candidate: DesignerCandidate) -> AssetAcquisitionResult:
        cache = AssetCache(state_dir / "assets.sqlite3")
        asset = cache.write_asset(
            candidate_identity_key=candidate.snippet.identity_key,
            asset_url="https://example.com/thumb.jpg",
            source="google_cse",
            image_bytes=b"fake-jpeg-bytes",
            tos_source="test",
            project_title="Case Study",
        )
        return AssetAcquisitionResult(cached_assets=(asset,), failed_urls=())

    pipeline = DesignerPipeline(
        brief=brief,
        bridge=bridge,
        queries=[query],
        candidate_acquirer=candidate_acquirer,
        asset_acquirer=asset_acquirer,
        facial_llm_caller=lambda _s, _u: (
            "DECISION: FACIAL_YES\nREASON: ok"
        ),
        full_llm_caller=lambda _s, _u: (
            "DECISION: SAVE\nPATH: design_systems\n"
            "CONFIDENCE: 0.86\nSUMMARY: Strong portfolio."
        ),
        vision_llm_call=lambda _model, _system, _user, _images: {
            "principles": [
                {
                    "name": "Systems craft",
                    "score": 3,
                    "reasoning": "Excellent reusable system.",
                    "image_ids": [0],
                }
            ],
            "overall_verdict": "yes",
            "overall_confidence": 0.91,
            "overall_reasoning": "Strong visual evidence.",
        },
    )

    asyncio.run(pipeline.run(run_id=run_id))

    with store.connect() as conn:
        candidate = conn.execute(
            "SELECT terminal_payload_json FROM candidates WHERE source='designer'"
        ).fetchone()

    payload = json.loads(candidate["terminal_payload_json"])
    assert "recommendation_pitch" in payload
    pitch = payload["recommendation_pitch"]
    assert "headline" in pitch
    assert "summary" in pitch
    assert isinstance(pitch["evidence_bullets"], list)
    assert isinstance(pitch["caveats"], list)
