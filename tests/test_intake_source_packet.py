from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import FormData


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from cloris import api as cloris_api
    from cloris.app import create_app

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLORIS_OUTPUT_ROOT", str(output_dir))
    from shared import output_paths

    intake_root = output_dir / "intake"
    intake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(output_paths, "INTAKE_ROOT", intake_root)
    return TestClient(create_app())


def _create_session(client: TestClient) -> int:
    response = client.post("/api/intake/sessions", json={})
    assert response.status_code == 201, response.text
    return int(response.json()["session"]["id"])


def _upload_request(content: bytes, *, content_length: str | None = None):
    upload = SimpleNamespace(
        filename="jd.txt",
        content_type="text/plain",
        read=AsyncMock(return_value=content),
    )
    form = FormData([("kind", "job_description"), ("files", upload)])
    headers = {} if content_length is None else {"content-length": content_length}
    request = SimpleNamespace(
        headers=headers,
        form=AsyncMock(return_value=form),
    )
    return request, upload


def _fallback_upload_request(
    content: bytes,
    *,
    content_length: str | None = None,
):
    boundary = "test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="kind"\r\n\r\n'
        "job_description\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="files"; filename="jd.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    headers = {"content-type": f"multipart/form-data; boundary={boundary}"}
    if content_length is not None:
        headers["content-length"] = content_length
    return SimpleNamespace(
        headers=headers,
        form=AsyncMock(side_effect=RuntimeError("python-multipart unavailable")),
        body=AsyncMock(return_value=body),
    )


def _pdf_with_text(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 24 Tf 72 720 Td ({escaped}) Tj ET\n".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def test_read_source_uploads_rejects_content_length_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cloris.api import intake

    monkeypatch.setattr(intake, "MAX_UPLOAD_BYTES", 4)
    request, upload = _upload_request(b"okay", content_length="5")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(intake._read_source_uploads(request))

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["error"] == "source_file_too_large"
    assert exc_info.value.detail["size_bytes"] == 5
    assert exc_info.value.detail["max_bytes"] == 4
    request.form.assert_not_awaited()
    upload.read.assert_not_awaited()


def test_read_source_uploads_rejects_oversized_materialized_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cloris.api import intake

    monkeypatch.setattr(intake, "MAX_UPLOAD_BYTES", 4)
    request, upload = _upload_request(b"12345")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(intake._read_source_uploads(request))

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["error"] == "source_file_too_large"
    assert exc_info.value.detail["size_bytes"] == 5
    assert exc_info.value.detail["max_bytes"] == 4
    upload.read.assert_awaited_once_with()


def test_read_source_uploads_fallback_rejects_oversized_file_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cloris.api import intake

    monkeypatch.setattr(intake, "MAX_UPLOAD_BYTES", 4)
    request = _fallback_upload_request(b"12345", content_length="4")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(intake._read_source_uploads_fallback(request))

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["error"] == "source_file_too_large"
    assert exc_info.value.detail["size_bytes"] == 5
    assert exc_info.value.detail["max_bytes"] == 4
    request.body.assert_awaited_once_with()


def test_read_source_uploads_fallback_without_content_length_rejects_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cloris.api import intake

    monkeypatch.setattr(intake, "MAX_UPLOAD_BYTES", 4)
    request = _fallback_upload_request(b"12345")
    assert "content-length" not in request.headers

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(intake._read_source_uploads(request))

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["error"] == "source_file_too_large"
    assert exc_info.value.detail["size_bytes"] == 5
    assert exc_info.value.detail["max_bytes"] == 4
    request.form.assert_awaited_once_with()
    request.body.assert_awaited_once_with()


def test_read_source_uploads_accepts_under_cap_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cloris.api import intake

    monkeypatch.setattr(intake, "MAX_UPLOAD_BYTES", 5)
    request, upload = _upload_request(b"okay")

    kind, uploads = asyncio.run(intake._read_source_uploads(request))

    assert kind == "job_description"
    assert uploads == [
        {
            "filename": "jd.txt",
            "content_type": "text/plain",
            "content": b"okay",
        }
    ]
    upload.read.assert_awaited_once_with()


def test_source_packet_endpoint_writes_draft_gaps_and_readback(
    client: TestClient,
) -> None:
    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet",
        json={
            "job_description_text": (
                "Staff Platform Engineer\n\nOwn the developer platform, "
                "ship internal systems, and set reliability patterns."
            ),
            "intake_notes_text": "Needs hands-on infra ownership.",
            "geography": "US remote",
        },
    )
    assert response.status_code == 200, response.text
    state = response.json()["session"]["state_json"]
    assert state["v2_draft"]["capability_areas"]
    assert state["field_provenance"]
    assert "gap_questions" in state
    assert state["distillation"]["prose"]


def _synthesis_stub_v2_draft(**overrides) -> dict:
    draft = {
        "role_title": "Staff Platform Engineer",
        "role_level": "IC4",
        "role_summary": "Own the developer platform and ship internal systems.",
        "geography": "US remote",
        "minimum_years_experience": 5,
        "minimum_bar_description": "Strong platform builder.",
        "linkedin_project": "test-project",
        "capability_areas": [
            {
                "name": "Platform engineering",
                "description": "Owns developer tooling.",
                "builder_signals": ["internal platforms"],
                "user_signals": ["ticket routing only"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns architecture.",
            "user_definition": "Uses existing services.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }
    draft.update(overrides)
    return draft


def test_distillation_skipped_when_draft_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloris.api import intake as intake_mod
    from market_intelligence.brief_distillation import (
        DistillationResult,
        FaithfulnessReport,
    )
    from shared.runtime_state.store import RuntimeStateStore
    from shared.source_packet_synthesis import SourcePacketSynthesisResult

    store = RuntimeStateStore(tmp_path / "intake.sqlite3")
    monkeypatch.setattr(intake_mod, "_intake_store", lambda: store)

    v2_draft = _synthesis_stub_v2_draft()
    field_provenance = {"role_title": {"source": "synthesis"}}
    synthesis_result = SourcePacketSynthesisResult(
        v2_draft=v2_draft,
        field_provenance=field_provenance,
        confidence_overall=0.9,
        synthesized_at="2026-01-01T00:00:00+00:00",
        source="deterministic",
    )
    monkeypatch.setattr(
        "shared.source_packet_synthesis.synthesize_v2_from_source_packet",
        lambda **_kwargs: synthesis_result,
    )

    distill_calls: list[dict] = []

    def _distill_brief(**kwargs):
        distill_calls.append(kwargs)
        return DistillationResult(
            prose="Recruiter-facing read-back.",
            structure_map={},
            faithfulness=FaithfulnessReport(True, [], [], 1.0),
            source="deterministic",
            generated_at="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(
        "market_intelligence.brief_distillation.distill_brief",
        _distill_brief,
    )

    state = {
        "source_packet": {
            "job_description_text": (
                "Staff Platform Engineer\n\nOwn the developer platform."
            )
        }
    }
    intake_mod._refresh_source_packet_artifacts(state=state, session_id=1)
    first_distillation = dict(state["distillation"])

    # Re-answer a gap question: source_text grows, synthesis output unchanged.
    state["gap_answer_history"] = [
        {"question_id": "geography", "answer": "US remote"},
    ]
    intake_mod._refresh_source_packet_artifacts(state=state, session_id=1)

    assert len(distill_calls) == 1
    assert state["distillation"] == first_distillation
    assert state.get("distillation_input_hash")


def test_distillation_recomputed_when_draft_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloris.api import intake as intake_mod
    from market_intelligence.brief_distillation import (
        DistillationResult,
        FaithfulnessReport,
    )
    from shared.runtime_state.store import RuntimeStateStore
    from shared.source_packet_synthesis import SourcePacketSynthesisResult

    store = RuntimeStateStore(tmp_path / "intake.sqlite3")
    monkeypatch.setattr(intake_mod, "_intake_store", lambda: store)

    field_provenance = {"role_title": {"source": "synthesis"}}
    synthesis_passes = [
        _synthesis_stub_v2_draft(role_title="Staff Platform Engineer"),
        _synthesis_stub_v2_draft(role_title="Principal Platform Engineer"),
    ]
    pass_index = {"value": 0}

    def _synthesize(**_kwargs):
        result = SourcePacketSynthesisResult(
            v2_draft=synthesis_passes[pass_index["value"]],
            field_provenance=field_provenance,
            confidence_overall=0.9,
            synthesized_at="2026-01-01T00:00:00+00:00",
            source="deterministic",
        )
        pass_index["value"] += 1
        return result

    monkeypatch.setattr(
        "shared.source_packet_synthesis.synthesize_v2_from_source_packet",
        _synthesize,
    )

    distill_calls: list[dict] = []

    def _distill_brief(**kwargs):
        distill_calls.append(kwargs)
        return DistillationResult(
            prose=f"Read-back for {kwargs['v2_draft']['role_title']}.",
            structure_map={},
            faithfulness=FaithfulnessReport(True, [], [], 1.0),
            source="deterministic",
            generated_at="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(
        "market_intelligence.brief_distillation.distill_brief",
        _distill_brief,
    )

    state = {
        "source_packet": {
            "job_description_text": (
                "Staff Platform Engineer\n\nOwn the developer platform."
            )
        }
    }
    intake_mod._refresh_source_packet_artifacts(state=state, session_id=1)
    intake_mod._refresh_source_packet_artifacts(state=state, session_id=1)

    assert len(distill_calls) == 2
    assert distill_calls[0]["v2_draft"]["role_title"] == "Staff Platform Engineer"
    assert distill_calls[1]["v2_draft"]["role_title"] == "Principal Platform Engineer"


def test_source_packet_file_upload_persists_kind(
    client: TestClient,
) -> None:
    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "job_description"},
        files={
            "files": (
                "jd.txt",
                b"Staff Platform Engineer\n\nOwn developer systems and reliability.",
                "text/plain",
            )
        },
    )
    assert response.status_code == 200, response.text
    record = response.json()["session"]["state_json"]["source_packet"]["files"][0]
    assert record["kind"] == "job_description"
    assert record["filename"] == "jd.txt"
    assert "Own developer systems and reliability" in record["text"]
    assert record["char_count"] == len(record["text"])


def test_source_packet_file_upload_rejects_aggregate_oversize(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloris.api import intake

    monkeypatch.setattr(intake, "MAX_PACKET_UPLOAD_BYTES", 5)
    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "job_description"},
        files=[
            ("files", ("first.txt", b"abc", "text/plain")),
            ("files", ("second.txt", b"def", "text/plain")),
        ],
    )

    assert response.status_code == 413, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "source_packet_too_large"
    assert detail["size_bytes"] == 6
    assert detail["max_bytes"] == 5


def test_source_packet_markdown_upload_persists_kind(
    client: TestClient,
) -> None:
    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "intake_notes"},
        files={
            "files": (
                "head-applied-ai-bfs-intake-notes.md",
                (
                    b"# Head of Applied AI Lab\n\n"
                    b"Quasi-CTO for the BFS applied AI group.\n\n"
                    b"Capital markets, asset management, and wealth management matter."
                ),
                "text/markdown",
            )
        },
    )
    assert response.status_code == 200, response.text
    record = response.json()["session"]["state_json"]["source_packet"]["files"][0]
    assert record["kind"] == "intake_notes"
    assert record["filename"] == "head-applied-ai-bfs-intake-notes.md"
    assert record["content_type"] == "text/markdown"
    assert "Quasi-CTO for the BFS applied AI group" in record["text"]


def test_source_packet_pdf_upload_drives_synthesis_and_readback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shared.source_packet_synthesis._has_llm_access", lambda: False
    )

    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "job_description"},
        files={
            "files": (
                "head-of-applied-ai.pdf",
                _pdf_with_text("Head of Applied AI Lab Banking Financial Services"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200, response.text
    state = response.json()["session"]["state_json"]
    record = state["source_packet"]["files"][0]
    assert record["kind"] == "job_description"
    assert record["filename"] == "head-of-applied-ai.pdf"
    assert "Head of Applied AI Lab" in record["text"]
    # Upload returns immediately with synthesis flagged ``running``; the
    # worker writes ``v2_draft`` on completion. Wait for ready before
    # asserting on synthesis-owned fields.
    from cloris.api.intake_synthesis import wait_for_synthesis

    assert wait_for_synthesis(session_id, timeout=5.0)
    final_state = client.get(
        f"/api/intake/sessions/{session_id}"
    ).json()["session"]["state_json"]
    assert "Applied AI Lab" in final_state["v2_draft"]["role_title"]


def test_source_packet_file_upload_persists_intake_notes_kind(
    client: TestClient,
) -> None:
    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "intake_notes"},
        files={
            "files": (
                "intake.txt",
                b"Hiring manager wants a builder, not an advisor.",
                "text/plain",
            )
        },
    )
    assert response.status_code == 200, response.text
    record = response.json()["session"]["state_json"]["source_packet"]["files"][0]
    assert record["kind"] == "intake_notes"
    assert record["filename"] == "intake.txt"
    assert "builder" in record["text"]
    assert record["char_count"] == len(record["text"])


def test_source_packet_file_upload_rejects_missing_files(
    client: TestClient,
) -> None:
    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "job_description"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["error"] == "no_source_files"


def test_source_packet_file_upload_rejects_unsupported_extension(
    client: TestClient,
) -> None:
    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "job_description"},
        files={
            "files": (
                "jd.exe",
                b"binary blob",
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["error"] == "unsupported_source_file"


def test_source_packet_file_upload_rejects_empty_file(
    client: TestClient,
) -> None:
    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "job_description"},
        files={
            "files": (
                "empty.txt",
                b"   \n  \n",
                "text/plain",
            )
        },
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["error"] == "empty_source_file"


_REALISTIC_JD = (
    "Head of Applied AI Lab, Banking & Financial Services\n\n"
    "Owns applied AI strategy, lab buildout, executive stakeholder "
    "alignment, regulated financial-services AI delivery, and production "
    "GenAI evaluation.\n\n"
    "Needs someone who has actually built and led applied AI teams in "
    "banking or financial services, not just advised on AI strategy."
)


def test_realistic_jd_upload_drives_synthesis_and_groups_by_kind(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Heuristic synthesis only — keeps CI deterministic without provider keys.
    monkeypatch.setattr(
        "shared.source_packet_synthesis._has_llm_access", lambda: False
    )

    session_id = _create_session(client)
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "job_description"},
        files={"files": ("head-of-applied-ai.txt", _REALISTIC_JD.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200, response.text

    state = response.json()["session"]["state_json"]
    files = state["source_packet"]["files"]
    assert len(files) == 1
    assert files[0]["kind"] == "job_description"
    assert files[0]["filename"] == "head-of-applied-ai.txt"
    assert "Applied AI Lab" in files[0]["text"]

    # Synthesis now runs in a background worker. Wait for it before
    # asserting on the recruiter-facing draft + distillation.
    from cloris.api.intake_synthesis import wait_for_synthesis

    assert wait_for_synthesis(session_id, timeout=5.0)
    final_state = client.get(
        f"/api/intake/sessions/{session_id}"
    ).json()["session"]["state_json"]

    draft = final_state["v2_draft"]
    assert draft["role_title"], draft
    assert "Applied AI Lab" in draft["role_title"]
    assert isinstance(draft["role_summary"], str) and draft["role_summary"].strip()
    assert isinstance(draft["capability_areas"], list) and draft["capability_areas"]
    assert isinstance(draft["target_modules"], list) and draft["target_modules"]
    assert isinstance(draft["source_strategy"], list) and draft["source_strategy"]
    # The fixture's distinctive phrase must reach the read-back, proving the
    # uploaded JD (not a generic stub) drove synthesis.
    assert "Applied AI" in final_state["distillation"]["prose"]




def test_critique_commit_applies_structured_edit_and_records_history(
    client: TestClient,
) -> None:
    session_id = _create_session(client)
    client.post(
        f"/api/intake/sessions/{session_id}/source_packet",
        json={
            "job_description_text": (
                "Staff Platform Engineer\n\nOwn the developer platform and "
                "ship internal systems."
            )
        },
    )
    critique = client.post(
        f"/api/intake/sessions/{session_id}/critique",
        json={
            "critique_text": (
                '{"edits":[{"field":"role_title","op":"set","value":"Principal Platform Engineer"}]}'
            )
        },
    )
    assert critique.status_code == 200, critique.text
    commit = client.post(
        f"/api/intake/sessions/{session_id}/critique/commit",
        json={"approved_edit_indices": [0]},
    )
    assert commit.status_code == 200, commit.text
    state = commit.json()["session"]["state_json"]
    assert state["v2_draft"]["role_title"] == "Principal Platform Engineer"
    assert state["critique_history"][0]["before_values"]["role_title"]
    prefs = client.get("/api/recruiter/preferences")
    assert prefs.status_code == 200
    assert prefs.json()["preferences"]["override_counts"]["role_title"] == 1


def test_complete_indexes_accepted_brief_into_corpus(client: TestClient) -> None:
    session_id = _create_session(client)
    synth = client.post(
        f"/api/intake/sessions/{session_id}/source_packet",
        json={
            "job_description_text": (
                "Staff Runtime Engineer\n\nOwn distributed runtime systems "
                "and reliability for product engineering teams."
            )
        },
    )
    assert synth.status_code == 200, synth.text
    complete = client.post(f"/api/intake/sessions/{session_id}/complete")
    assert complete.status_code == 200, complete.text
    brief_id = complete.json()["brief_id"]

    from cloris.api.intake import _intake_store

    with _intake_store().connect() as conn:
        row = conn.execute(
            "SELECT title FROM brief_corpus WHERE brief_key = ?",
            (brief_id,),
        ).fetchone()
    assert row is not None
    assert row["title"] == "Staff Runtime Engineer"
