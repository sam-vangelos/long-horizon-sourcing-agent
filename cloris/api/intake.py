"""Intake session HTTP routes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from cloris import intake_sessions
from cloris.models import (
    IntakeCritiqueCommitRequest,
    IntakeCritiqueRequest,
    IntakeGapAnswerRequest,
    ComposeJobResult,
    ConversationComposeJob,
    IntakeComposeJobResponse,
    IntakeSession,
    IntakeSessionCompleteResponse,
    IntakeSessionCreateRequest,
    IntakeSessionDeleteResponse,
    IntakeSessionListResponse,
    IntakeSessionPatchRequest,
    IntakeSessionResponse,
    IntakeSourcePacketRequest,
    RecruiterPreferencesRequest,
    RecruiterPreferencesResponse,
)
from shared.runtime_state.store import RuntimeStateStore

from . import _paths
from ._sse import sse_pack
from .routing import router

log = logging.getLogger("cloris.api")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PACKET_UPLOAD_BYTES = 25 * 1024 * 1024


def _intake_db_path() -> Path:
    """Path to the intake-sessions SQLite store."""

    from shared.output_paths import resolve_intake_db_path

    return resolve_intake_db_path()


def _intake_session_wire(row: dict[str, Any]) -> IntakeSession:
    """Build the intake session wire shape with filing readiness attached."""

    from shared.intake_filing import filing_readiness_wire

    payload = dict(row)
    payload["filing_readiness"] = filing_readiness_wire(payload)
    return IntakeSession.model_validate(payload)


def _intake_store() -> RuntimeStateStore:
    """Resolve the canonical RuntimeStateStore for intake-session writes."""

    return RuntimeStateStore(_intake_db_path())


def _session_state(session: dict, *, session_id: int) -> dict[str, Any]:
    state = session.get("state_json") or {}
    if not isinstance(state, dict):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_state_json",
                "message": f"Intake session {session_id} state_json is not an object.",
            },
        )
    return state


def _source_files_from_state(source_packet: dict[str, Any]) -> list[Any]:
    from shared.source_packet import ExtractedSourceFile

    raw_files = source_packet.get("files")
    if not isinstance(raw_files, list):
        return []
    files: list[ExtractedSourceFile] = []
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        filename = item.get("filename")
        if not isinstance(text, str) or not isinstance(filename, str):
            continue
        files.append(
            ExtractedSourceFile(
                filename=filename,
                content_type=item.get("content_type")
                if isinstance(item.get("content_type"), str)
                else None,
                char_count=int(item.get("char_count") or len(text)),
                text=text,
                kind=item.get("kind")
                if item.get("kind") in {"job_description", "intake_notes", "general"}
                else "general",
            )
        )
    return files


def _normalize_source_file_kind(raw_kind: Any) -> str:
    if raw_kind in {"job_description", "intake_notes", "general"}:
        return str(raw_kind)
    return "general"


def _multipart_header_param(header_value: str, key: str) -> str | None:
    quoted = re.search(rf'{re.escape(key)}="([^"]*)"', header_value)
    if quoted:
        return quoted.group(1)
    bare = re.search(rf"{re.escape(key)}=([^;]+)", header_value)
    if bare:
        return bare.group(1).strip()
    return None


def _multipart_boundary(content_type: str) -> str | None:
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            return part.split("=", 1)[1].strip().strip('"')
    return None


def _check_upload_content_length(request: Request) -> None:
    raw_size = request.headers.get("content-length")
    try:
        size = int(raw_size) if raw_size is not None else None
    except (TypeError, ValueError):
        return
    if size is not None and size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "source_file_too_large",
                "message": "The upload request exceeds the per-file size limit.",
                "size_bytes": size,
                "max_bytes": MAX_UPLOAD_BYTES,
            },
        )


async def _read_source_uploads(
    request: Request,
) -> tuple[str, list[dict[str, Any]]]:
    """Read source uploads even when python-multipart is unavailable."""

    _check_upload_content_length(request)
    try:
        form = await request.form()
    except Exception:
        return await _read_source_uploads_fallback(request)

    raw_kind = form.get("kind")
    uploads: list[dict[str, Any]] = []
    for upload in form.getlist("files"):
        if not hasattr(upload, "read"):
            continue
        filename = getattr(upload, "filename", None) or "upload"
        content = await upload.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "source_file_too_large",
                    "message": f"{filename} exceeds the per-file size limit.",
                    "size_bytes": len(content),
                    "max_bytes": MAX_UPLOAD_BYTES,
                },
            )
        uploads.append(
            {
                "filename": filename,
                "content_type": getattr(upload, "content_type", None),
                "content": content,
            }
        )
    return _normalize_source_file_kind(raw_kind), uploads


async def _read_source_uploads_fallback(
    request: Request,
) -> tuple[str, list[dict[str, Any]]]:
    _check_upload_content_length(request)
    content_type = request.headers.get("content-type", "")
    boundary = _multipart_boundary(content_type)
    if not boundary:
        # When the request isn't multipart at all (e.g. urlencoded with kind
        # but no files), fall through to "no uploads" so the route can raise
        # the structured no_source_files error rather than a parser failure.
        if "application/x-www-form-urlencoded" in content_type.lower():
            from urllib.parse import parse_qs

            body = await request.body()
            parsed = parse_qs(body.decode("utf-8", errors="replace"))
            raw_kind = parsed.get("kind", [None])[0]
            return _normalize_source_file_kind(raw_kind), []
        raise HTTPException(
            status_code=422,
            detail={
                "error": "source_multipart_parse_failed",
                "message": "Could not read uploaded files.",
            },
        )

    body = await request.body()
    delimiter = b"--" + boundary.encode("utf-8")
    raw_kind: str | None = None
    uploads: list[dict[str, Any]] = []
    for raw_part in body.split(delimiter):
        # Only trim the boundary's leading CRLFs; bare ``.strip()`` would eat
        # whitespace-only file payloads ("\n\n") and drop the part entirely.
        part = raw_part.lstrip(b"\r\n")
        if not part:
            continue
        # Final delimiter is ``--``; ignore it and any trailing CRLF.
        if part.startswith(b"--"):
            continue
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if part.endswith(b"--"):
            part = part[:-2].rstrip(b"\r\n")
        header_blob, separator, content = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers: dict[str, str] = {}
        for raw_line in header_blob.decode("latin-1", errors="replace").split("\r\n"):
            name, sep, value = raw_line.partition(":")
            if sep:
                headers[name.strip().lower()] = value.strip()
        disposition = headers.get("content-disposition", "")
        field_name = _multipart_header_param(disposition, "name")
        filename = _multipart_header_param(disposition, "filename")
        payload = content
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        if field_name == "kind":
            raw_kind = payload.decode("utf-8", errors="replace").strip()
        elif field_name == "files" and filename:
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "source_file_too_large",
                        "message": f"{filename} exceeds the per-file size limit.",
                        "size_bytes": len(payload),
                        "max_bytes": MAX_UPLOAD_BYTES,
                    },
                )
            uploads.append(
                {
                    "filename": filename or "upload",
                    "content_type": headers.get("content-type"),
                    "content": payload,
                }
            )
    return _normalize_source_file_kind(raw_kind), uploads


def _distilled_state_with_cache(
    state: dict[str, Any],
    *,
    v2_draft: dict[str, Any],
    field_provenance: dict[str, Any] | None,
    source_text: str | None = None,
    session_id: int | None = None,
) -> dict[str, Any]:
    key = hashlib.sha256(
        json.dumps(
            {"v2_draft": v2_draft, "field_provenance": field_provenance or {}},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    distillation = state.get("distillation")
    if isinstance(distillation, dict) and state.get("distillation_input_hash") == key:
        return distillation
    from market_intelligence.brief_distillation import distill_brief

    distilled = distill_brief(
        v2_draft=v2_draft,
        field_provenance=field_provenance,
        source_text=source_text,
        session_id=session_id,
    ).to_state_dict()
    state["distillation_input_hash"] = key
    return distilled


def _refresh_source_packet_artifacts(
    *,
    state: dict[str, Any],
    session_id: int,
) -> None:
    """Run synthesis, gap detection, and distillation into ``state``.

    Synchronous helper used by the JSON-paste ``source_packet`` route
    and the ``answer_questions`` route. The async ``source_packet/files``
    upload route does NOT call this helper directly — it schedules
    background synthesis via :mod:`cloris.api.intake_synthesis` so the
    HTTP response can return immediately.

    Stale-write protection: bumps
    ``state_json.source_packet_synthesis.revision`` so any concurrently
    in-flight background worker spawned by an earlier upload drops its
    commit on revision mismatch.
    """

    from shared.brief_corpus import build_exemplar_block
    from shared.gap_questions import generate_gap_questions
    from shared.output_paths import resolve_recruiter_db_path
    from shared.recruiter_context import get_current_recruiter_id
    from shared.recruiter_overrides import (
        recruiter_voice_line_for_extract,
        resolve_intake_preferences,
    )
    from shared.runtime_state.recruiter_store import RecruiterStore
    from shared.source_packet import compose_source_packet_text
    from shared.source_packet_synthesis import (
        SYNTHESIS_SOURCE_CHAR_BUDGET,
        synthesize_v2_from_source_packet,
    )

    from .intake_synthesis import (
        SYNTHESIS_STATUS_READY,
        bump_synthesis_revision,
        ensure_synthesis_state,
    )

    source_packet = state.get("source_packet")
    if not isinstance(source_packet, dict):
        source_packet = {}
        state["source_packet"] = source_packet

    files = _source_files_from_state(source_packet)
    gap_answer_history = state.get("gap_answer_history")
    if not isinstance(gap_answer_history, list):
        gap_answer_history = []

    jd_text = (
        source_packet.get("job_description_text")
        if isinstance(source_packet.get("job_description_text"), str)
        else ""
    )
    intake_notes = (
        source_packet.get("intake_notes_text")
        if isinstance(source_packet.get("intake_notes_text"), str)
        else ""
    )
    source_text = compose_source_packet_text(
        job_description_text=jd_text,
        intake_notes_text=intake_notes,
        files=files,
        gap_answer_history=gap_answer_history,
    )
    if not source_text.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "empty_source_packet",
                "message": "Paste or upload source material before synthesis.",
            },
        )

    store = _intake_store()
    exemplar_block, used_ids = build_exemplar_block(store, source_text)
    # Reopen recruiter learns-half, R6.3 (THE FLIP). Read recruiter calibration
    # from the durable cross-brief SPINE, fail-closed to the legacy per-intake-DB
    # blob (resolve_intake_preferences). ``store`` is the intake DB; the recruiter
    # store + acting recruiter resolve through the same seams the forward writes
    # use, so a correction on an earlier brief reaches synthesis here.
    # Fail-soft to the legacy reader if recruiter-store resolution/construction
    # raises (RecruiterStore.__init__ runs mkdir + DDL and can raise) — the flip
    # must never 500 synthesis, matching the forward-write's posture and
    # extending the locked fail-closed intent to its real boundary.
    try:
        rid = get_current_recruiter_id()
        recruiter_store = RecruiterStore(resolve_recruiter_db_path())
        preferences = resolve_intake_preferences(store, recruiter_store, rid, state)
    except Exception:  # noqa: BLE001 — degrade to legacy, never break synthesis
        log.warning(
            "R6.3: recruiter-store resolution failed; falling back to the "
            "legacy intake-blob reader for synthesis preferences",
            exc_info=True,
        )
        preferences = recruiter_voice_line_for_extract(store, state)
    current_v2 = state.get("v2_draft") if isinstance(state.get("v2_draft"), dict) else None
    current_provenance = (
        state.get("field_provenance")
        if isinstance(state.get("field_provenance"), dict)
        else None
    )
    result = synthesize_v2_from_source_packet(
        source_text=source_text,
        job_description_text=jd_text,
        intake_notes_text=intake_notes,
        current_v2_draft=current_v2,
        field_provenance=current_provenance,
        geography=source_packet.get("geography")
        if isinstance(source_packet.get("geography"), str)
        else None,
        exemplar_block=exemplar_block,
        recruiter_preferences=preferences,
        session_id=session_id,
    )
    state["source_truncation"] = {
        "truncated": bool(result.source_truncated),
        "source_chars": int(result.source_char_count),
        "budget": SYNTHESIS_SOURCE_CHAR_BUDGET,
    }
    state["v2_draft"] = result.v2_draft
    state["v2_draft_polish_meta"] = result.to_polish_meta_dict()
    state["field_provenance"] = result.field_provenance
    state["gap_questions"] = generate_gap_questions(
        v2_draft=result.v2_draft,
        field_provenance=result.field_provenance,
    )
    state["retrieval_meta"] = {
        "source": "brief_corpus",
        "used_brief_ids": used_ids,
        "exemplar_count": len(used_ids),
    }
    state["distillation"] = _distilled_state_with_cache(
        state,
        v2_draft=result.v2_draft,
        field_provenance=result.field_provenance,
        source_text=source_text,
        session_id=session_id,
    )

    bump_synthesis_revision(state)
    block = ensure_synthesis_state(state)
    block["status"] = SYNTHESIS_STATUS_READY
    block["error"] = None
    block["started_at"] = None
    block["completed_at"] = datetime.now(timezone.utc).isoformat()


def _patch_state_response(
    *,
    session_id: int,
    state: dict[str, Any],
    role_title: str | None = None,
) -> IntakeSessionResponse:
    updated = intake_sessions.patch_intake_session(
        store=_intake_store(),
        session_id=session_id,
        state_json=state,
        role_title=role_title,
    )
    if updated is None:
        raise HTTPException(
            status_code=410,
            detail={"error": "intake_session_gone_after_update", "id": session_id},
        )
    return IntakeSessionResponse(
        slice="v0-onboarding-slice-1",
        session=_intake_session_wire(updated),
    )


@router.post(
    "/api/intake/sessions",
    status_code=201,
    response_model=IntakeSessionResponse,
)
def create_intake_session_endpoint(
    req: IntakeSessionCreateRequest,
) -> IntakeSessionResponse:
    """Create a new intake session."""

    session = intake_sessions.create_intake_session(
        store=_intake_store(), role_title=req.role_title
    )
    log.info(
        "conversational_intake_trace create_session id=%s",
        session["id"],
    )
    return IntakeSessionResponse(
        slice="v0-onboarding-slice-1",
        session=_intake_session_wire(session),
    )


@router.get(
    "/api/intake/sessions",
    response_model=IntakeSessionListResponse,
)
def list_intake_sessions_endpoint() -> IntakeSessionListResponse:
    """List active (non-archived) intake sessions, newest first."""

    from shared.runtime_state import read_models

    sessions = read_models.list_intake_sessions(_intake_db_path())
    log.info(
        "conversational_intake_trace list_sessions count=%s",
        len(sessions),
    )
    return IntakeSessionListResponse(
        slice="v0-onboarding-slice-1",
        sessions=[_intake_session_wire(row) for row in sessions],
    )


@router.get(
    "/api/intake/sessions/{session_id}",
    response_model=IntakeSessionResponse,
)
def get_intake_session_endpoint(session_id: int) -> IntakeSessionResponse:
    """Return one intake session by id, or 404 if missing."""

    from shared.runtime_state import read_models

    session = read_models.get_intake_session(
        _intake_db_path(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    log.info(
        "conversational_intake_trace get_session id=%s step=%s",
        session_id,
        session.get("current_step"),
    )
    return IntakeSessionResponse(
        slice="v0-onboarding-slice-1",
        session=_intake_session_wire(session),
    )


@router.patch(
    "/api/intake/sessions/{session_id}",
    response_model=IntakeSessionResponse,
)
def patch_intake_session_endpoint(
    session_id: int, req: IntakeSessionPatchRequest
) -> IntakeSessionResponse:
    """Partial-update an intake session."""

    session = intake_sessions.patch_intake_session(
        store=_intake_store(),
        session_id=session_id,
        current_step=req.current_step,
        state_json=req.state_json,
        role_title=req.role_title,
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    return IntakeSessionResponse(
        slice="v0-onboarding-slice-1",
        session=_intake_session_wire(session),
    )


@router.patch(
    "/api/intake/sessions/{session_id}/archive",
    response_model=IntakeSessionResponse,
)
def archive_intake_session_endpoint(session_id: int) -> IntakeSessionResponse:
    """Archive (soft-delete) an intake session; 404 if missing.

    Idempotent — archiving an already-archived session is a no-op on
    ``archived_at`` (see :func:`cloris.intake_sessions.archive_intake_session`).
    """

    session = intake_sessions.archive_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    return IntakeSessionResponse(
        slice="v0-onboarding-slice-1",
        session=_intake_session_wire(session),
    )


@router.post(
    "/api/intake/sessions/{session_id}/source_packet",
    response_model=IntakeSessionResponse,
)
def source_packet_intake_session_endpoint(
    session_id: int,
    req: IntakeSourcePacketRequest,
) -> IntakeSessionResponse:
    """Synthesize a brief draft from pasted JD and intake notes."""

    from shared.source_packet import normalize_source_text
    from shared.v2_draft_undo import push_v2_undo

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    state = _session_state(session, session_id=session_id)
    if isinstance(state.get("v2_draft"), dict):
        push_v2_undo(state)

    existing_packet = state.get("source_packet")
    if not isinstance(existing_packet, dict):
        existing_packet = {}
    files = existing_packet.get("files") if isinstance(existing_packet.get("files"), list) else []
    state["source_packet"] = {
        "job_description_text": normalize_source_text(req.job_description_text),
        "intake_notes_text": normalize_source_text(req.intake_notes_text),
        "geography": normalize_source_text(req.geography),
        "files": files,
    }
    _refresh_source_packet_artifacts(state=state, session_id=session_id)
    draft = state.get("v2_draft") if isinstance(state.get("v2_draft"), dict) else {}
    role_title = draft.get("role_title") if isinstance(draft.get("role_title"), str) else None
    return _patch_state_response(
        session_id=session_id,
        state=state,
        role_title=role_title,
    )


@router.post(
    "/api/intake/sessions/{session_id}/source_packet/files",
    response_model=IntakeSessionResponse,
)
async def upload_source_packet_files_endpoint(
    session_id: int,
    request: Request,
) -> IntakeSessionResponse:
    """Attach source-packet documents and kick off background synthesis.

    Returns quickly after persisting the extracted file bytes; long-running
    synthesis runs in a background worker (see :mod:`cloris.api.intake_synthesis`).
    Recruiters see the file row immediately; the intake draft sidebar polls
    ``state_json.source_packet_synthesis.status`` until it flips to
    ``ready`` or ``failed``.

    Blocking work (PDF parsing inside ``extract_source_file_text`` and the
    LLM-driven synthesis) is deliberately taken off the asyncio event loop
    so concurrent ``/api/status`` polls keep answering and the masthead
    never lies "Cloris isn't responding…" during a slow upload.
    """

    from shared.source_packet import SourcePacketError, extract_source_file_text
    from shared.v2_draft_undo import push_v2_undo

    from .intake_synthesis import (
        SYNTHESIS_STATUS_RUNNING,
        ensure_synthesis_state,
        schedule_source_packet_synthesis,
    )

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    state = _session_state(session, session_id=session_id)
    packet = state.get("source_packet")
    if not isinstance(packet, dict):
        packet = {}
        state["source_packet"] = packet
    if isinstance(state.get("v2_draft"), dict):
        push_v2_undo(state)

    kind, uploads = await _read_source_uploads(request)
    if not uploads:
        raise HTTPException(
            status_code=422,
            detail={"error": "no_source_files", "message": "No files were uploaded."},
        )

    existing_files = packet.get("files") if isinstance(packet.get("files"), list) else []
    next_files = list(existing_files)
    packet_upload_bytes = sum(len(upload.get("content") or b"") for upload in uploads)
    if packet_upload_bytes > MAX_PACKET_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "source_packet_too_large",
                "message": "The uploaded files exceed the packet size limit.",
                "size_bytes": packet_upload_bytes,
                "max_bytes": MAX_PACKET_UPLOAD_BYTES,
            },
        )
    for upload in uploads:
        try:
            extracted = await asyncio.to_thread(
                extract_source_file_text,
                filename=str(upload.get("filename") or "upload"),
                content=upload.get("content") or b"",
                content_type=upload.get("content_type")
                if isinstance(upload.get("content_type"), str)
                else None,
                kind=kind,
            )
        except SourcePacketError as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": exc.code, "message": str(exc)},
            ) from exc
        next_files.append(
            {
                "filename": extracted.filename,
                "content_type": extracted.content_type,
                "char_count": extracted.char_count,
                "text": extracted.text,
                "kind": extracted.kind,
            }
        )
    packet["files"] = next_files

    block = ensure_synthesis_state(state)
    next_revision = int(block.get("revision") or 0) + 1
    block["revision"] = next_revision
    block["status"] = SYNTHESIS_STATUS_RUNNING
    block["error"] = None
    block["started_at"] = datetime.now(timezone.utc).isoformat()
    block["completed_at"] = None

    response = _patch_state_response(session_id=session_id, state=state)

    schedule_source_packet_synthesis(
        session_id=session_id,
        expected_revision=next_revision,
    )

    return response


def _conversation_compose_job_from_state(state: dict[str, Any]) -> ConversationComposeJob:
    block = state.get("conversation_compose")
    if not isinstance(block, dict):
        return ConversationComposeJob()
    raw_result = block.get("result")
    result: ComposeJobResult | None = None
    if isinstance(raw_result, dict):
        compose_status = raw_result.get("compose_status")
        if compose_status in {"composed", "deficits"}:
            result = ComposeJobResult(
                compose_status=compose_status,
                deficits=[
                    x
                    for x in (raw_result.get("deficits") or [])
                    if isinstance(x, dict)
                ],
                missing_keys=[
                    str(x)
                    for x in (raw_result.get("missing_keys") or [])
                    if isinstance(x, str)
                ],
                invalid_keys=[
                    str(x)
                    for x in (raw_result.get("invalid_keys") or [])
                    if isinstance(x, str)
                ],
                insight_deficits=[
                    x
                    for x in (raw_result.get("insight_deficits") or [])
                    if isinstance(x, dict)
                ],
            )
    status = block.get("status")
    if status not in {"idle", "composing", "ready", "failed"}:
        status = "idle"
    return ConversationComposeJob(
        status=status,
        revision=int(block.get("revision") or 0),
        error=block.get("error") if isinstance(block.get("error"), str) else None,
        started_at=block.get("started_at")
        if isinstance(block.get("started_at"), str)
        else None,
        completed_at=block.get("completed_at")
        if isinstance(block.get("completed_at"), str)
        else None,
        result=result,
    )


def _compose_job_response(session: dict[str, Any]) -> IntakeComposeJobResponse:
    state = session.get("state_json") or {}
    if not isinstance(state, dict):
        state = {}
    return IntakeComposeJobResponse(
        session=_intake_session_wire(session),
        job=_conversation_compose_job_from_state(state),
    )


def _start_compose_job(session_id: int) -> IntakeComposeJobResponse:
    from .intake_compose import (
        COMPOSE_STATUS_COMPOSING,
        ensure_compose_state,
        run_compose_worker_inline,
        schedule_compose_from_conversation,
        should_run_compose_synchronously,
    )

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    state = _session_state(session, session_id=session_id)
    block = ensure_compose_state(state)
    next_revision = int(block.get("revision") or 0) + 1
    block["revision"] = next_revision
    block["status"] = COMPOSE_STATUS_COMPOSING
    block["error"] = None
    block["started_at"] = datetime.now(timezone.utc).isoformat()
    block["completed_at"] = None
    block["result"] = None

    patched = intake_sessions.patch_intake_session(
        store=_intake_store(),
        session_id=session_id,
        state_json=state,
    )
    if patched is None:
        raise HTTPException(
            status_code=410,
            detail={"error": "intake_session_gone_after_compose_schedule", "id": session_id},
        )

    if should_run_compose_synchronously():
        run_compose_worker_inline(session_id=session_id, expected_revision=next_revision)
        latest = intake_sessions.get_intake_session(
            store=_intake_store(), session_id=session_id
        )
        if latest is None:
            raise HTTPException(
                status_code=410,
                detail={"error": "intake_session_gone_after_compose", "id": session_id},
            )
        return _compose_job_response(latest)

    schedule_compose_from_conversation(
        session_id=session_id,
        expected_revision=next_revision,
    )
    return _compose_job_response(patched)


@router.post(
    "/api/intake/sessions/{session_id}/compose_jobs",
    response_model=IntakeComposeJobResponse,
)
def create_compose_job_endpoint(session_id: int) -> IntakeComposeJobResponse:
    """Schedule transcript-to-brief composition as a background job."""

    return _start_compose_job(session_id)


@router.get(
    "/api/intake/sessions/{session_id}/compose_jobs/current",
    response_model=IntakeComposeJobResponse,
)
def get_current_compose_job_endpoint(session_id: int) -> IntakeComposeJobResponse:
    """Return the current ``conversation_compose`` job for polling."""

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    return _compose_job_response(session)


@router.post(
    "/api/intake/sessions/{session_id}/compose_from_conversation",
    response_model=IntakeComposeJobResponse,
)
def compose_from_conversation_endpoint(session_id: int) -> IntakeComposeJobResponse:
    """Recover a reviewable V2 draft from the persisted conversation (async job)."""

    return _start_compose_job(session_id)


@router.post(
    "/api/intake/sessions/{session_id}/answer_questions",
    response_model=IntakeSessionResponse,
)
def answer_intake_gap_questions_endpoint(
    session_id: int,
    req: IntakeGapAnswerRequest,
) -> IntakeSessionResponse:
    """Apply natural-language answers to source-packet gaps."""

    from shared.v2_draft_undo import push_v2_undo

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    state = _session_state(session, session_id=session_id)
    if isinstance(state.get("v2_draft"), dict):
        push_v2_undo(state)
    history = state.get("gap_answer_history")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "answer_text": req.answer_text.strip(),
            "answered_question_ids": req.answered_question_ids,
        }
    )
    state["gap_answer_history"] = history
    _refresh_source_packet_artifacts(state=state, session_id=session_id)
    return _patch_state_response(session_id=session_id, state=state)


@router.delete(
    "/api/intake/sessions/{session_id}",
    response_model=IntakeSessionDeleteResponse,
)
def delete_intake_session_endpoint(
    session_id: int,
) -> IntakeSessionDeleteResponse:
    """Hard-delete an intake session by id; 404 if missing."""

    deleted = intake_sessions.delete_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    return IntakeSessionDeleteResponse(
        slice="v0-onboarding-slice-1", deleted=True, id=session_id
    )


@router.post(
    "/api/intake/sessions/{session_id}/complete",
    response_model=IntakeSessionCompleteResponse,
)
def complete_intake_session_endpoint(
    session_id: int,
) -> IntakeSessionCompleteResponse:
    """Finalize an intake session and write the V2 brief."""

    from shared.brief_v2_schema import (
        BriefSchemaError,
        normalize_generated_engagement_context,
        validate_v2_brief,
    )
    from shared.brief_writer import write_brief_atomic
    from shared.output_paths import derive_brief_id, slugify_output_component

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )

    from shared.intake_filing import http_status_for_blocker, primary_filing_blocker

    blocker = primary_filing_blocker(session)
    if blocker is not None:
        raise HTTPException(
            status_code=http_status_for_blocker(blocker),
            detail={"error": blocker.code, "message": blocker.message},
        )

    state = session.get("state_json") or {}
    v2_draft = state.get("v2_draft")

    if not isinstance(v2_draft, dict):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_v2_brief",
                "message": (
                    "Session has no v2_draft on state_json — the review "
                    "chapter must populate it before completion."
                ),
                "missing_keys": ["v2_draft"],
                "invalid_keys": [],
            },
        )

    v2_draft = dict(v2_draft)
    normalize_generated_engagement_context(v2_draft)
    try:
        validate_v2_brief(v2_draft)
    except BriefSchemaError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_v2_brief",
                "message": str(exc),
                "missing_keys": list(exc.missing_keys),
                "invalid_keys": list(exc.invalid_keys),
            },
        ) from exc

    role_title = (
        v2_draft.get("role_title")
        if isinstance(v2_draft.get("role_title"), str)
        else session.get("role_title")
    )
    slug_source = role_title or f"intake-{session_id}"
    slug = slugify_output_component(slug_source)
    target_dir = _paths._CONFIG_DIR / slug
    target_path = target_dir / "brief.json"
    if target_path.exists():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "brief_already_exists",
                "message": (
                    f"A brief already exists at config/{slug}/brief.json. "
                    f"Pick a different role title or edit the existing brief."
                ),
                "slug": slug,
            },
        )

    try:
        write_brief_atomic(abs_path=target_path, payload=v2_draft)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "brief_write_failed", "reason": str(exc)},
        ) from exc

    try:
        brief_id = derive_brief_id(brief_path=str(target_path))
    except Exception as exc:
        log.error(
            "brief_id computation failed post-write at %s: %s",
            target_path,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "brief_id_computation_failed",
                "reason": str(exc),
                "brief_path": str(target_path.relative_to(_paths._CONFIG_PARENT)),
            },
        ) from exc

    try:
        from shared.brief_corpus import index_v2_brief

        index_v2_brief(
            _intake_store(),
            brief_key=brief_id,
            v2_json=v2_draft,
            title=role_title if isinstance(role_title, str) else None,
        )
    except Exception as exc:  # noqa: BLE001 - completion must not fail on learning
        log.warning("Failed to index accepted brief %s into corpus: %s", brief_id, exc)

    normalized_state = dict(state)
    normalized_state["v2_draft"] = v2_draft
    intake_sessions.patch_intake_session(
        store=_intake_store(),
        session_id=session_id,
        state_json=normalized_state,
    )

    completed = intake_sessions.complete_intake_session(
        store=_intake_store(), session_id=session_id, brief_id=brief_id
    )
    if completed is None:
        log.warning(
            "Intake session %d disappeared during completion; "
            "brief written at %s (brief_id=%s) but the session row "
            "was removed mid-flight",
            session_id,
            target_path,
            brief_id,
        )
        raise HTTPException(
            status_code=410,
            detail={
                "error": "intake_session_gone_after_complete",
                "message": (
                    "The brief was written but the intake draft was "
                    "removed before completion stamped through. "
                    "Refresh the brief library to find it."
                ),
                "brief_id": brief_id,
                "brief_path": str(target_path.relative_to(_paths._CONFIG_PARENT)),
            },
        )

    log.info(
        "Intake session %d completed → brief_id=%s at %s "
        "(Phase D Slice D3)",
        session_id,
        brief_id,
        target_path,
    )

    return IntakeSessionCompleteResponse(
        slice="v0-onboarding-slice-1",
        session=_intake_session_wire(completed),
        brief_id=brief_id,
        brief_path=str(target_path.relative_to(_paths._CONFIG_PARENT)),
    )


@router.post(
    "/api/intake/sessions/{session_id}/distill",
    response_model=IntakeSessionResponse,
)
def distill_intake_session_endpoint(session_id: int) -> IntakeSessionResponse:
    """Refresh the recruiter-facing brief read-back."""

    from shared.source_packet import compose_source_packet_text

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    state = _session_state(session, session_id=session_id)
    v2_draft = state.get("v2_draft")
    if not isinstance(v2_draft, dict):
        raise HTTPException(
            status_code=422,
            detail={"error": "missing_v2_draft", "message": "No draft to distill."},
        )
    packet = state.get("source_packet") if isinstance(state.get("source_packet"), dict) else {}
    source_text = compose_source_packet_text(
        job_description_text=str(packet.get("job_description_text") or ""),
        intake_notes_text=str(packet.get("intake_notes_text") or ""),
        files=_source_files_from_state(packet),
        gap_answer_history=state.get("gap_answer_history")
        if isinstance(state.get("gap_answer_history"), list)
        else [],
    )
    provenance = state.get("field_provenance") if isinstance(state.get("field_provenance"), dict) else {}
    state["distillation"] = _distilled_state_with_cache(
        state,
        v2_draft=v2_draft,
        field_provenance=provenance,
        source_text=source_text,
        session_id=session_id,
    )
    return _patch_state_response(session_id=session_id, state=state)


@router.post(
    "/api/intake/sessions/{session_id}/critique",
    response_model=IntakeSessionResponse,
)
def critique_intake_session_endpoint(
    session_id: int,
    req: IntakeCritiqueRequest,
) -> IntakeSessionResponse:
    """Parse natural-language critique into proposed field edits."""

    from market_intelligence.brief_critique import BriefCritiqueBackend
    from shared.critique_locks import filter_locked_edits

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    state = _session_state(session, session_id=session_id)
    v2_draft = state.get("v2_draft")
    if not isinstance(v2_draft, dict):
        raise HTTPException(
            status_code=422,
            detail={"error": "missing_v2_draft", "message": "No draft to critique."},
        )
    affirmed_fields = state.get("affirmed_fields")
    if not isinstance(affirmed_fields, list):
        affirmed_fields = []
    result = BriefCritiqueBackend().parse(
        critique_text=req.critique_text,
        v2_draft=v2_draft,
        affirmed_fields=[str(x) for x in affirmed_fields],
        field_provenance=state.get("field_provenance")
        if isinstance(state.get("field_provenance"), dict)
        else {},
        session_id=session_id,
    )
    allowed, blocked = filter_locked_edits(
        result.edits,
        [str(x) for x in affirmed_fields],
    )
    state["pending_critique"] = {
        "critique_text": req.critique_text,
        "modality": req.modality,
        "edits": allowed,
        "blocked": blocked + result.blocked,
        "alternatives": result.alternatives,
        "stage_errors": result.stage_errors,
        "source": result.source,
    }
    return _patch_state_response(session_id=session_id, state=state)


@router.post(
    "/api/intake/sessions/{session_id}/critique/commit",
    response_model=IntakeSessionResponse,
)
def commit_intake_critique_endpoint(
    session_id: int,
    req: IntakeCritiqueCommitRequest,
) -> IntakeSessionResponse:
    """Apply selected pending critique edits to the draft."""

    from shared.brief_v2_schema import BriefSchemaError, validate_v2_brief
    from shared.critique_apply import apply_critique_edits, get_field_value
    from shared.gap_questions import generate_gap_questions
    from shared.recruiter_overrides import record_override_for_field_path
    from shared.v2_draft_undo import push_v2_undo

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )
    state = _session_state(session, session_id=session_id)
    v2_draft = state.get("v2_draft")
    pending = state.get("pending_critique")
    if not isinstance(v2_draft, dict) or not isinstance(pending, dict):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_pending_critique",
                "message": "No pending critique edits to apply.",
            },
        )
    edits = pending.get("edits") if isinstance(pending.get("edits"), list) else []
    normalized_edits = [e for e in edits if isinstance(e, dict)]
    if req.approved_edit_indices is not None:
        approved = []
        for idx in req.approved_edit_indices:
            if 0 <= idx < len(normalized_edits):
                approved.append(normalized_edits[idx])
    else:
        approved = normalized_edits

    before_values = {
        str(edit.get("field")): get_field_value(v2_draft, str(edit.get("field")))
        for edit in approved
        if edit.get("field")
    }
    try:
        next_v2 = apply_critique_edits(v2_draft, approved)
        validate_v2_brief(next_v2)
    except BriefSchemaError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_v2_brief_after_critique",
                "message": str(exc),
                "missing_keys": list(exc.missing_keys),
                "invalid_keys": list(exc.invalid_keys),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_critique_edit", "message": str(exc)},
        ) from exc

    push_v2_undo(state)
    state["v2_draft"] = next_v2
    provenance = state.get("field_provenance") if isinstance(state.get("field_provenance"), dict) else {}
    state["gap_questions"] = generate_gap_questions(
        v2_draft=next_v2,
        field_provenance=provenance,
    )
    history = state.get("critique_history")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "critique_text": pending.get("critique_text"),
            "approved_edits": approved,
            "before_values": before_values,
            "blocked": pending.get("blocked") if isinstance(pending.get("blocked"), list) else [],
        }
    )
    state["critique_history"] = history
    affirmed = [str(x) for x in state.get("affirmed_fields", [])] if isinstance(state.get("affirmed_fields"), list) else []
    for field in req.released_locks:
        affirmed = [x for x in affirmed if x != field]
    for field in req.newly_affirmed_fields:
        if field not in affirmed:
            affirmed.append(field)
    state["affirmed_fields"] = affirmed
    state.pop("pending_critique", None)
    for edit in approved:
        field = edit.get("field")
        if isinstance(field, str):
            record_override_for_field_path(_intake_store(), field)
    state["distillation"] = _distilled_state_with_cache(
        state,
        v2_draft=next_v2,
        field_provenance=provenance,
        source_text=None,
        session_id=session_id,
    )
    return _patch_state_response(session_id=session_id, state=state)


@router.get(
    "/api/recruiter/preferences",
    response_model=RecruiterPreferencesResponse,
)
def get_recruiter_preferences_endpoint() -> RecruiterPreferencesResponse:
    from shared.recruiter_overrides import get_recruiter_preferences

    return RecruiterPreferencesResponse(
        preferences=get_recruiter_preferences(_intake_store())
    )


@router.put(
    "/api/recruiter/preferences",
    response_model=RecruiterPreferencesResponse,
)
def put_recruiter_preferences_endpoint(
    req: RecruiterPreferencesRequest,
) -> RecruiterPreferencesResponse:
    from shared.recruiter_overrides import (
        get_recruiter_preferences,
        put_recruiter_preferences,
    )

    existing = get_recruiter_preferences(_intake_store())
    patch = req.model_dump(exclude_unset=True)
    merged = {**existing, **patch}
    # HOIST the meta write out of the return so the dark spine write can run
    # between it and the response. ``saved`` is the live behavior, unchanged.
    saved = put_recruiter_preferences(_intake_store(), merged)

    # Reopen recruiter learns-half, R6.2prime (DARK forward write). When the
    # recruiter sets a bare top-level ``summary`` here, mirror it into the
    # durable recruiter primitive as a BUCKETLESS ``{"summary": text}``
    # principle_feedback signal — the shape R6.1prime's reader projects into the
    # bare top-level line. Fail-soft BY CONSTRUCTION (H2): the try/except wraps
    # the FULL RecruiterStore construction (__init__ runs mkdir + DDL and can
    # raise) plus the write, so a recruiter-store problem can never turn this
    # 200 into a 500 — the meta write above already landed. APPEND-ONLY (H3):
    # a bare INSERT, no dedup; signals are append-only + soft-superseded.
    summary = patch.get("summary")
    if isinstance(summary, str) and summary.strip():
        try:
            from shared.output_paths import resolve_recruiter_db_path
            from shared.recruiter_context import get_current_recruiter_id
            from shared.recruiter_overrides import INTAKE_SYNTHESIS_DOMAIN
            from shared.runtime_state.recruiter_store import (
                SIGNAL_PRINCIPLE_FEEDBACK,
                RecruiterStore,
            )

            RecruiterStore(resolve_recruiter_db_path()).record_taste_signal(
                get_current_recruiter_id(),
                signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
                domain=INTAKE_SYNTHESIS_DOMAIN,
                payload={"summary": summary.strip()},
            )
        except Exception:  # noqa: BLE001 — fail-soft; the PUT already succeeded
            log.debug(
                "R6.2prime bare-summary spine write failed (fail-soft)",
                exc_info=True,
            )

    return RecruiterPreferencesResponse(preferences=saved)


@router.post(
    "/api/intake/sessions/{session_id}/polish",
    response_model=IntakeSessionResponse,
)
def polish_intake_session_endpoint(
    session_id: int,
) -> IntakeSessionResponse:
    """Polish the in-flight v2_draft via the LLM cascade."""

    from market_intelligence.brief_polish import BriefPolishBackend
    from shared.v2_draft_undo import push_v2_undo

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )

    state = session.get("state_json") or {}
    if not isinstance(state, dict):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_state_json",
                "message": "Session state_json is not an object.",
            },
        )

    chapter_captures: dict[str, object] = {}
    for chapter_id in ("role", "good_looks", "lookalikes", "where_to_look"):
        sub = state.get(chapter_id)
        if isinstance(sub, dict):
            chapter_captures[chapter_id] = sub

    push_v2_undo(state)

    backend = BriefPolishBackend()
    result = backend.polish(
        chapter_captures=chapter_captures,
        role_title=session.get("role_title"),
        session_id=session_id,
    )

    # Suspect-output gate: if polish produced placeholder-shaped values
    # (LLM hallucination or fallback regression), do NOT overwrite the
    # user's existing v2_draft. Surface the meta only so the UI can
    # show a "polish produced suspect output — keeping your previous
    # draft" message and let the user try again.
    from market_intelligence.brief_distillation import _looks_like_placeholder

    suspect_fields: list[str] = []
    polished_role_title = (result.v2_draft or {}).get("role_title") if isinstance(result.v2_draft, dict) else None
    if isinstance(polished_role_title, str) and _looks_like_placeholder(
        polished_role_title, kind="role_title"
    ):
        suspect_fields.append("role_title")
    polished_caps = (result.v2_draft or {}).get("capability_areas") if isinstance(result.v2_draft, dict) else None
    if isinstance(polished_caps, list):
        for i, area in enumerate(polished_caps):
            if not isinstance(area, dict):
                continue
            for key in ("name", "description"):
                val = area.get(key)
                if isinstance(val, str) and _looks_like_placeholder(val):
                    suspect_fields.append(f"capability_areas[{i}].{key}")
    polished_depth = (result.v2_draft or {}).get("depth_distinction") if isinstance(result.v2_draft, dict) else None
    if isinstance(polished_depth, dict):
        for key in ("builder_definition", "user_definition", "edge_case_guidance"):
            val = polished_depth.get(key)
            if isinstance(val, str) and _looks_like_placeholder(val):
                suspect_fields.append(f"depth_distinction.{key}")

    if suspect_fields:
        meta = result.to_meta_dict()
        meta["source"] = "suspect"
        meta["suspect_fields"] = suspect_fields
        state["v2_draft_polish_meta"] = meta
        # v2_draft intentionally unchanged.
    else:
        state["v2_draft"] = result.v2_draft
        state["v2_draft_polish_meta"] = result.to_meta_dict()

    updated = intake_sessions.patch_intake_session(
        store=_intake_store(),
        session_id=session_id,
        state_json=state,
    )
    if updated is None:
        raise HTTPException(
            status_code=410,
            detail={
                "error": "intake_session_gone_after_polish",
                "id": session_id,
            },
        )
    return IntakeSessionResponse(
        slice="v0-onboarding-slice-1",
        session=_intake_session_wire(updated),
    )


@router.post(
    "/api/intake/sessions/{session_id}/restore_prev_draft",
    response_model=IntakeSessionResponse,
)
def restore_prev_draft_endpoint(
    session_id: int,
) -> IntakeSessionResponse:
    """Restore the pre-polish v2_draft from the one-deep undo buffer."""

    from shared.v2_draft_undo import pop_v2_undo_into_state

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )

    state = session.get("state_json") or {}
    if not isinstance(state, dict):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_state_json",
                "message": "Session state_json is not an object.",
            },
        )

    if not pop_v2_undo_into_state(state):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_prev_draft",
                "id": session_id,
                "message": (
                    "No prior draft to restore. Polish the brief at least "
                    "once to populate the undo buffer."
                ),
            },
        )

    updated = intake_sessions.patch_intake_session(
        store=_intake_store(),
        session_id=session_id,
        state_json=state,
    )
    if updated is None:
        raise HTTPException(
            status_code=410,
            detail={
                "error": "intake_session_gone_after_restore",
                "id": session_id,
            },
        )
    return IntakeSessionResponse(
        slice="v0-onboarding-slice-1",
        session=_intake_session_wire(updated),
    )


# =====================================================================
# Conversational intake (Phase C5).
#
# `POST /api/intake/sessions/{id}/start_conversation` writes the opener
# Cloris message to the transcript (mode init).
#
# `POST /api/intake/sessions/{id}/converse/stream` is the SSE turn loop —
# accepts a recruiter message, streams Cloris's reply via Anthropic's
# prompt-cached streaming API, runs slot extraction + sufficiency on the
# updated v2_draft, and emits message_chunk / slot_update / ready /
# done / error events. One in-flight turn per session, enforced by an
# asyncio.Lock with finally-eviction so the dict doesn't leak.
#
# See plans/conversational-intake.md and the C5 phase block in
# ~/.cursor/plans/conversational_intake_implementation_*.plan.md.
# =====================================================================


# Process-local lock dict. One entry per in-flight session_id, evicted
# in `finally` after every turn (hit, miss, or error) so the dict size
# is bounded by concurrent in-flight turns rather than session lifetime.
# Without eviction this would slow-leak across the Northwind trial.
_CONVERSATION_LOCKS: dict[int, asyncio.Lock] = {}


class ConverseRequest(BaseModel):
    """Body for ``POST /api/intake/sessions/{id}/converse/stream``."""

    model_config = ConfigDict(extra="forbid")

    recruiter_message: str


class StartConversationResponse(BaseModel):
    """Body returned by ``POST .../start_conversation``."""

    model_config = ConfigDict(extra="forbid")

    slice: str = "v0-conversational-intake"
    session: IntakeSession


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conversation_log_path(session_id: int) -> Path:
    """Per-session llm_usage.jsonl path under output/intake/conversation/."""

    base = _intake_db_path().parent / "conversation" / f"session_{session_id}"
    base.mkdir(parents=True, exist_ok=True)
    return base / "llm_usage.jsonl"


@router.post(
    "/api/intake/sessions/{session_id}/start_conversation",
    response_model=StartConversationResponse,
)
def start_conversation_endpoint(session_id: int) -> StartConversationResponse:
    """Initialize a session for conversational intake.

    Sets ``current_step = "conversation"`` and writes Cloris's opener
    message to ``state_json.messages`` so the frontend can hydrate
    without sending an empty user message on mount (Anthropic rejects
    empty user content).

    Idempotent on existing conversational sessions: if ``messages`` is
    already non-empty, the session is returned unchanged. The opener
    string varies by ``source_packet`` presence, mirroring the v0
    plan's two openers.
    """

    from shared.intake_conversation import ConversationMessage
    from shared.intake_conversation.prompts import (
        OPENER_NO_PACKET,
        OPENER_WITH_PACKET,
    )
    from shared.intake_conversation.state import append_message

    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "intake_session_not_found", "id": session_id},
        )

    state = _session_state(session, session_id=session_id)

    existing_messages = state.get("messages")
    log.info(
        "conversational_intake_trace start_conversation id=%s "
        "existing_messages=%s current_step=%s",
        session_id,
        len(existing_messages)
        if isinstance(existing_messages, list)
        else None,
        session.get("current_step"),
    )
    if isinstance(existing_messages, list) and existing_messages:
        # Idempotent — already initialized.
        return StartConversationResponse(
            session=_intake_session_wire(session)
        )

    has_packet = isinstance(state.get("source_packet"), dict) and (
        bool(state["source_packet"].get("job_description_text"))
        or bool(state["source_packet"].get("intake_notes_text"))
        or bool(state["source_packet"].get("files"))
    )
    opener = OPENER_WITH_PACKET if has_packet else OPENER_NO_PACKET

    msg: ConversationMessage = {  # type: ignore[typeddict-item]
        "role": "cloris",
        "content": opener,
        "ts": _utc_iso_now(),
        "meta": {"opener": True},
    }
    state = append_message(state, msg)

    updated = intake_sessions.patch_intake_session(
        store=_intake_store(),
        session_id=session_id,
        current_step="conversation",
        state_json=state,
    )
    if updated is None:
        raise HTTPException(
            status_code=410,
            detail={
                "error": "intake_session_gone_after_start_conversation",
                "id": session_id,
            },
        )
    return StartConversationResponse(
        session=_intake_session_wire(updated)
    )


@router.post("/api/intake/sessions/{session_id}/converse/stream")
async def converse_stream_endpoint(
    session_id: int, body: ConverseRequest, request: Request
) -> StreamingResponse:
    """Stream one orchestrator turn as SSE events.

    Wire shape (see ``cloris/frontend/src/lib/intake_conversation/api.ts``):

    - ``event: message_chunk`` ``data: {"text": "<token>"}``
    - ``event: slot_update`` ``data: {"v2_draft": {...}, "missing": [...]}``
    - ``event: extraction_partial`` ``data: {"dropped_keys": [...]}`` —
      emitted right after ``slot_update``, only when the extractor dropped
      one or more structurally-invalid top-level keys this round (P9.4).
    - ``event: ready_to_compose`` ``data: {"missing": []}``
    - ``event: done`` ``data: {"turn_count": N, "cost_usd_running_total": F}``
    - ``event: error`` ``data: {"detail": "...", "fallback_message": "..."}``

    Concurrency: at most one turn in flight per ``session_id``. Second
    concurrent request returns 409. Lock is evicted in ``finally`` so
    the per-session entry doesn't leak across many turns.

    Auth: bearer token via ``?token=`` query param (this path is
    SSE-exempt in :mod:`cloris.api.auth`).
    """

    lock = _CONVERSATION_LOCKS.setdefault(session_id, asyncio.Lock())
    if lock.locked():
        # Single-threaded asyncio: no race between lock.locked() and the
        # subsequent acquire because there is no await between them.
        raise HTTPException(
            status_code=409,
            detail={"error": "turn_in_flight", "id": session_id},
        )

    async def _gen():
        try:
            async with lock:
                async for chunk in _converse_turn(session_id, body, request):
                    yield chunk
        finally:
            # Evict the lock entry so the dict stays bounded by
            # concurrent in-flight turns, not session lifetime.
            _CONVERSATION_LOCKS.pop(session_id, None)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _converse_turn(
    session_id: int, body: ConverseRequest, request: Request
):
    """Drive one turn end-to-end: append → stream → extract → persist.

    Yields raw SSE-packed strings. All exceptions caught and re-emitted
    as ``error`` events; underlying state is persisted with the
    recruiter message at minimum so the next turn detects the dropped
    turn correctly.
    """

    from shared.intake_conversation import (
        ConversationMessage,
        cap_state_for,
    )
    from shared.intake_conversation.extractor import (
        ExtractionResult,
        extract_slots,
    )
    from shared.intake_conversation.insights import (
        HIRING_MANAGER_PICTURE_KEY,
        HIRING_MANAGER_PICTURE_LOCK_PATH,
        merge_intake_insights,
    )
    from shared.intake_conversation.orchestrator import (
        DEGRADED_REASON_PROVIDER_FAILED,
        stream_next_turn,
    )
    from shared.intake_conversation.sse_bridge import (
        bridge_sync_stream_to_async,
    )
    from shared.intake_conversation.state import (
        append_message,
        bump_conversation_meta,
        detect_dropped_turn,
        merge_extracted,
    )
    from shared.intake_conversation.sufficiency import is_ready_to_compose
    from shared.llm_usage import estimate_usage_cost_usd, llm_usage_session
    from shared import config as shared_config

    log.info(
        "conversational_intake_trace converse_turn id=%s message_chars=%s",
        session_id,
        len(body.recruiter_message or ""),
    )

    # ---------------------------------------------------------
    # Load + validate session.
    # ---------------------------------------------------------
    session = intake_sessions.get_intake_session(
        store=_intake_store(), session_id=session_id
    )
    if session is None:
        yield sse_pack(
            "error",
            {
                "detail": "intake_session_not_found",
                "fallback_message": "",
            },
        )
        return

    state = session.get("state_json")
    if not isinstance(state, dict):
        yield sse_pack(
            "error",
            {
                "detail": "invalid_state_json",
                "fallback_message": "",
            },
        )
        return

    # Detect dropped-turn BEFORE we append the recruiter message —
    # otherwise we'd always think the latest turn was dropped.
    dropped_turn = detect_dropped_turn(state)

    manually_edited_keys = set(
        state.get("conversation_meta", {}).get("manually_edited_keys", [])
        if isinstance(state.get("conversation_meta"), dict)
        else []
    )

    # ---------------------------------------------------------
    # Append recruiter message (persist immediately so dropped-turn
    # detection works on the next call if we crash mid-stream).
    # ---------------------------------------------------------
    recruiter_msg: ConversationMessage = {  # type: ignore[typeddict-item]
        "role": "recruiter",
        "content": body.recruiter_message,
        "ts": _utc_iso_now(),
    }
    state = append_message(state, recruiter_msg)
    intake_sessions.patch_intake_session(
        store=_intake_store(),
        session_id=session_id,
        state_json=state,
    )

    # Sufficiency BEFORE the orchestrator turn — drives the prompt's
    # volunteer behavior. The wire event after extraction is what flips
    # ``ready_to_compose`` for the frontend.
    pre_v2 = state.get("v2_draft") if isinstance(state.get("v2_draft"), dict) else {}
    sufficiency_state = is_ready_to_compose(pre_v2)
    pre_ready = sufficiency_state[0]

    # Cap state. Computed from the PRE-turn meta so a turn that lands
    # ON the cap line gets the cap-aware prompt. Hard cap overrides
    # the post-extraction sufficiency check below to force the
    # ready_to_compose event regardless of v2_draft state.
    pre_meta = (
        state.get("conversation_meta")
        if isinstance(state.get("conversation_meta"), dict)
        else {}
    )
    cap_state = cap_state_for(
        turn_count=int(pre_meta.get("turn_count", 0)),
        cost_usd_running_total=float(
            pre_meta.get("cost_usd_running_total", 0.0)
        ),
    )

    # ---------------------------------------------------------
    # Stream the orchestrator's reply via the bridge.
    # ---------------------------------------------------------
    cost_log = _conversation_log_path(session_id)
    cloris_text_buffer: list[str] = []
    usage_payload: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    # Set when the orchestrator emits a ``("degraded", ...)`` marker, OR
    # when the bridge itself raises. Carries the recruiter-safe wire
    # payload the post-loop section flushes as a ``degraded`` SSE event.
    # The recruiter copy is owned here, not by the orchestrator, because
    # the wire boundary is what the frontend banner reads.
    degraded_payload: dict[str, Any] | None = None

    def _orchestrator_factory():
        return stream_next_turn(
            messages=state.get("messages") or [],
            v2_draft=pre_v2,
            source_packet=state.get("source_packet")
            if isinstance(state.get("source_packet"), dict)
            else None,
            sufficiency_state=sufficiency_state,
            dropped_turn=dropped_turn,
            cap_state=cap_state,
            session_id=session_id,
        )

    try:
        with llm_usage_session(
            cost_log,
            stage="intake_conversation_orchestrator",
            session_id=session_id,
        ):
            async for kind, payload in bridge_sync_stream_to_async(
                _orchestrator_factory
            ):
                if await request.is_disconnected():
                    return
                if kind == "delta":
                    # Buffer-then-emit (Slice 3): accumulate the full turn
                    # before any message_chunk reaches the recruiter so
                    # pre-emit guards can rewrite or replace bad text.
                    cloris_text_buffer.append(payload)
                elif kind == "usage":
                    usage_payload = payload or usage_payload
                elif kind == "degraded":
                    # Audit finding F-2: orchestrator-side provider
                    # failure. Capture the marker; the post-loop
                    # section emits the wire ``degraded`` event so
                    # ordering stays after extraction/slot_update.
                    if isinstance(payload, dict):
                        degraded_payload = {
                            "reason": str(
                                payload.get("reason")
                                or DEGRADED_REASON_PROVIDER_FAILED
                            ),
                            "any_delta": bool(payload.get("any_delta")),
                        }
                    else:
                        degraded_payload = {
                            "reason": DEGRADED_REASON_PROVIDER_FAILED,
                            "any_delta": bool(cloris_text_buffer),
                        }
    except Exception as exc:  # noqa: BLE001 — recover gracefully
        # Bridge-level failure (the orchestrator itself is supposed to
        # swallow). Treat as ``provider_failed`` for wire purposes; the
        # cause is in cloris.log via the orchestrator's own warning
        # plus this entry.
        log.warning(
            "converse_stream bridge failed session_id=%s error=%s",
            session_id,
            exc,
            exc_info=True,
        )
        degraded_payload = {
            "reason": DEGRADED_REASON_PROVIDER_FAILED,
            "any_delta": bool(cloris_text_buffer),
        }

    # ---------------------------------------------------------
    # Compute cost from in-stream usage payload.
    # ---------------------------------------------------------
    cost_estimate, _rate_source = estimate_usage_cost_usd(
        model=shared_config.OPUS_MODEL_NAME,
        input_tokens=int(usage_payload.get("input_tokens") or 0),
        output_tokens=int(usage_payload.get("output_tokens") or 0),
        cache_read_input_tokens=int(
            usage_payload.get("cache_read_input_tokens") or 0
        ),
        cache_creation_input_tokens=int(
            usage_payload.get("cache_creation_input_tokens") or 0
        ),
    )
    cost_delta_usd = float(cost_estimate or 0.0)

    # ---------------------------------------------------------
    # Pre-emit guards on the buffered turn, then emit a single chunk.
    # Brief-dump shapes and question-economy violations are enforced
    # here — not log-only — before the recruiter sees Cloris's text.
    # ---------------------------------------------------------
    cloris_reply_raw = "".join(cloris_text_buffer)
    pre_insights = (
        state.get("intake_insights")
        if isinstance(state.get("intake_insights"), dict)
        else None
    )
    turn_count_for_guards = int(pre_meta.get("turn_count", 0) or 0)
    if cloris_reply_raw.strip():
        from shared.intake_conversation.question_economy import apply_pre_emit_guards

        cloris_reply, _guard_reasons = apply_pre_emit_guards(
            cloris_reply_raw,
            v2_draft=pre_v2,
            source_packet=state.get("source_packet")
            if isinstance(state.get("source_packet"), dict)
            else None,
            messages=state.get("messages") if isinstance(state.get("messages"), list) else [],
            intake_insights=pre_insights,
            turn_count=turn_count_for_guards,
            sufficiency_state=sufficiency_state,
        )
        if _guard_reasons:
            log.warning(
                "converse_stream pre_emit_guards session_id=%s reasons=%s",
                session_id,
                _guard_reasons,
            )
        if cloris_reply:
            yield sse_pack("message_chunk", {"text": cloris_reply})
    else:
        cloris_reply = cloris_reply_raw

    # ---------------------------------------------------------
    # Persist Cloris's reply + bump conversation_meta.
    #
    # Audit finding F-2: a degraded turn with no streamed content must
    # NOT commit a Cloris message — the recruiter sees a banner instead
    # of a normal-shaped Cloris turn that says "Lost my train of
    # thought". A degraded turn WITH partial content keeps the partial
    # transcript so the conversation isn't silently truncated.
    # ---------------------------------------------------------
    is_degraded = degraded_payload is not None
    commit_cloris_turn = bool(cloris_reply) and (
        not is_degraded or degraded_payload.get("any_delta", False)
    )
    if commit_cloris_turn:
        cloris_msg: ConversationMessage = {  # type: ignore[typeddict-item]
            "role": "cloris",
            "content": cloris_reply,
            "ts": _utc_iso_now(),
            "meta": {
                "cost_usd": cost_delta_usd,
                "model": shared_config.OPUS_MODEL_NAME,
                **({"degraded": True} if is_degraded else {}),
            },
        }
        state = append_message(state, cloris_msg)
    state = bump_conversation_meta(
        state, turn_delta=1, cost_delta_usd=cost_delta_usd
    )

    # ---------------------------------------------------------
    # Run extraction + merge + sufficiency. Skip extraction on a
    # degraded turn with no committed Cloris text — there's no new
    # signal to extract from beyond the recruiter message, and the
    # extractor would just burn a cheap-LLM call to confirm that.
    # ---------------------------------------------------------
    extraction: ExtractionResult = ExtractionResult({}, {})
    if commit_cloris_turn or not is_degraded:
        try:
            extraction = await asyncio.to_thread(
                extract_slots,
                messages=state.get("messages") or [],
                current_v2_draft=pre_v2,
                source_packet=state.get("source_packet")
                if isinstance(state.get("source_packet"), dict)
                else None,
                manually_edited_keys=manually_edited_keys,
                session_id=session_id,
                current_intake_insights=state.get("intake_insights")
                if isinstance(state.get("intake_insights"), dict)
                else None,
            )
        except Exception as exc:  # noqa: BLE001 — extraction is non-fatal
            log.warning(
                "converse_stream extraction failed session_id=%s error=%s",
                session_id,
                exc,
                exc_info=True,
            )
            extraction = ExtractionResult({}, {})

    new_v2 = merge_extracted(
        pre_v2, extraction.v2_updates, manually_edited_keys=manually_edited_keys
    )
    synthesis_block = state.get("source_packet_synthesis")
    synthesis_running = (
        isinstance(synthesis_block, dict)
        and synthesis_block.get("status") == "running"
    )
    # While source-packet synthesis owns v2_draft, chat extraction must not
    # clobber the in-flight worker draft. Insights and messages still merge.
    if synthesis_running:
        wire_v2 = pre_v2
    else:
        state["v2_draft"] = new_v2
        wire_v2 = new_v2

    # Merge insights via the parallel ``intake_insights`` bag. Insights
    # never enter ``v2_draft``: separate merge primitive, separate lock
    # paths (``intake_insights.<key>``), separate validation surface.
    pre_insights = (
        state.get("intake_insights")
        if isinstance(state.get("intake_insights"), dict)
        else {}
    )
    new_insights = merge_intake_insights(
        pre_insights,
        extraction.insight_updates,
        manually_edited_keys=manually_edited_keys,
    )
    state["intake_insights"] = new_insights

    # Recruiter-correction propagation: when the extractor surfaces a
    # ``corrected_by_recruiter: true`` insight, append the lock path so
    # subsequent extractor / synthesis writes are gated. ``manually_edited_keys``
    # remains a sorted list of unique strings to keep prompt-rendering
    # stable across turns.
    picture_update = extraction.insight_updates.get(HIRING_MANAGER_PICTURE_KEY)
    if (
        isinstance(picture_update, dict)
        and bool(picture_update.get("corrected_by_recruiter"))
        and HIRING_MANAGER_PICTURE_LOCK_PATH not in manually_edited_keys
    ):
        manually_edited_keys = set(manually_edited_keys) | {
            HIRING_MANAGER_PICTURE_LOCK_PATH
        }

    conv_meta = (
        state.get("conversation_meta")
        if isinstance(state.get("conversation_meta"), dict)
        else {}
    )
    updated_keys = sorted(extraction.v2_updates.keys()) + sorted(
        f"intake_insights.{k}" for k in extraction.insight_updates.keys()
    )
    dropped_keys = list(extraction.dropped_keys)
    if dropped_keys:
        last_extraction_status = "partial"
    elif extraction.v2_updates or extraction.insight_updates:
        last_extraction_status = "updated"
    else:
        last_extraction_status = "empty"
    conv_meta["last_extraction"] = {
        "turn_count": int(conv_meta.get("turn_count", 0) or 0),
        "status": last_extraction_status,
        "updated_keys": updated_keys,
        # P9.4: names the top-level v2_updates keys this round proposed
        # but that failed structural validation and were dropped — never
        # silent. Empty list on a clean round.
        "dropped_keys": dropped_keys,
        "ran_at": _utc_iso_now(),
    }
    conv_meta["last_extraction_at_turn"] = int(conv_meta.get("turn_count", 0) or 0)
    # Persist the locks list so future turns see the correction lock.
    conv_meta["manually_edited_keys"] = sorted(manually_edited_keys)

    post_ready, post_missing = is_ready_to_compose(wire_v2)
    if cap_state == "hard":
        # Hard cap forces composition regardless of v2_draft state —
        # the recruiter sees the draft after this turn whether or not
        # it crossed the deterministic threshold. C11 contract.
        post_ready = True
        post_missing = []

    conv_meta["ready_to_compose"] = post_ready
    state["conversation_meta"] = conv_meta

    compose_revision: int | None = None
    if cap_state == "hard" and post_ready:
        from .intake_compose import (
            COMPOSE_STATUS_COMPOSING,
            ensure_compose_state,
            schedule_compose_from_conversation,
            should_run_compose_synchronously,
            run_compose_worker_inline,
        )

        compose_block = ensure_compose_state(state)
        compose_status = compose_block.get("status")
        if compose_status not in {"composing", "ready"}:
            compose_revision = int(compose_block.get("revision") or 0) + 1
            compose_block["revision"] = compose_revision
            compose_block["status"] = COMPOSE_STATUS_COMPOSING
            compose_block["error"] = None
            compose_block["started_at"] = _utc_iso_now()
            compose_block["completed_at"] = None
            compose_block["result"] = None

    intake_sessions.patch_intake_session(
        store=_intake_store(),
        session_id=session_id,
        state_json=state,
    )

    if compose_revision is not None:
        if should_run_compose_synchronously():
            run_compose_worker_inline(
                session_id=session_id,
                expected_revision=compose_revision,
            )
        else:
            schedule_compose_from_conversation(
                session_id=session_id,
                expected_revision=compose_revision,
            )

    yield sse_pack(
        "slot_update",
        {"v2_draft": wire_v2, "missing": post_missing},
    )

    if dropped_keys:
        # P9.4: a structurally-invalid extractor key was dropped this round
        # — the valid keys still landed (see conv_meta["last_extraction"]
        # above), but the loss must be visible to the recruiter-facing UI,
        # never silent. Only emitted on a partial round.
        yield sse_pack("extraction_partial", {"dropped_keys": dropped_keys})

    if post_ready and not pre_ready:
        yield sse_pack("ready_to_compose", {"missing": []})

    if is_degraded:
        # Wire shape consumed by ``cloris/frontend/src/lib/intake_conversation/api.ts``.
        # ``message`` is recruiter-safe copy; ``reason`` is the
        # internal classifier; ``recoverable`` tells the UI whether
        # to surface a retry affordance.
        assert degraded_payload is not None
        yield sse_pack(
            "degraded",
            {
                "reason": degraded_payload.get(
                    "reason", DEGRADED_REASON_PROVIDER_FAILED
                ),
                "recoverable": True,
                "message": (
                    "Cloris hit a snag answering. Try sending again."
                ),
            },
        )

    yield sse_pack(
        "done",
        {
            "turn_count": int(
                state.get("conversation_meta", {}).get("turn_count", 0)
                if isinstance(state.get("conversation_meta"), dict)
                else 0
            ),
            "cost_usd_running_total": float(
                state.get("conversation_meta", {}).get(
                    "cost_usd_running_total", 0.0
                )
                if isinstance(state.get("conversation_meta"), dict)
                else 0.0
            ),
        },
    )
