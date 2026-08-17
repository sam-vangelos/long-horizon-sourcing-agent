"""Status and brief catalog HTTP routes."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException

from cloris.control_plane import aggregate_briefs, aggregate_status
from cloris.models import (
    BriefDetailResponse,
    BriefEditRequest,
    BriefInfo,
    BriefsListResponse,
    BriefVersionEntry,
    BriefVersionsResponse,
    StatusResponse,
)

from . import _paths
from .routing import router

log = logging.getLogger("cloris.api")

# Test/dev fixture marker: any directory ending in ``-fixture`` is hidden
# from the recruiter-facing catalog. We deliberately do NOT maintain a
# hard-coded set of slug names, because legitimate recruiter-authored
# titles (e.g. "Forward Deployed Engineer" → ``forward_deployed_engineer``)
# would otherwise be silently invisibilized after filing.
#
# History: an earlier ``_HIDDEN_BRIEF_DIRS`` constant carried fixture-only
# slugs like ``existing_role``, ``forward_deployed_engineer``, and
# ``idempotent_role``. A recruiter who filed a brief with one of those
# titles would land on a 404 because both ``/api/briefs`` and
# ``/api/brief/{brief_id}`` route through this scanner. Filed away as
# F-1 in the audit ledger; the fix is a single marker convention
# (``*-fixture`` suffix) that fixtures opt into explicitly.
_FIXTURE_DIR_SUFFIX = "-fixture"


@router.get("/api/status")
def api_status() -> StatusResponse:
    """Aggregate read-only status across LinkedIn and GitHub state dirs."""

    return aggregate_status()


def _scan_authored_briefs(config_dir: Path) -> list[BriefInfo]:
    """Walk authored brief files under ``config_dir`` for the picker/catalog."""

    out: list[BriefInfo] = []
    if not config_dir.exists() or not config_dir.is_dir():
        return out

    seen: set[Path] = set()

    def _candidates() -> Iterator[Path]:
        for p in config_dir.rglob("brief-*.json"):
            if p.is_file():
                yield p
        for p in config_dir.rglob("brief.json"):
            if p.is_file():
                yield p

    for path in _candidates():
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        name = path.name
        if name.endswith("-draft.json"):
            continue
        if ".bak-" in name:
            continue
        if path.parent.name.endswith(_FIXTURE_DIR_SUFFIX):
            continue
        try:
            raw = path.read_text()
            data = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue

        linkedin_project_id = data.get("linkedin_project_id")
        if linkedin_project_id is not None and not isinstance(linkedin_project_id, str):
            linkedin_project_id = str(linkedin_project_id)
        try:
            modified_at = datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat()
        except OSError:
            modified_at = ""

        raw_target_modules = data.get("target_modules")
        target_modules: list[str] | None = None
        if isinstance(raw_target_modules, list):
            target_modules = [str(m) for m in raw_target_modules if isinstance(m, str)]

        confidentiality_class = data.get("confidentiality_class")
        if not isinstance(confidentiality_class, str) or confidentiality_class not in {
            "open",
            "referenceable",
            "blind",
        }:
            confidentiality_class = "open"

        out.append(
            BriefInfo(
                path=str(path.relative_to(_paths._CONFIG_PARENT)),
                role_title=data.get("role_title")
                if isinstance(data.get("role_title"), str)
                else None,
                linkedin_project=data.get("linkedin_project")
                if isinstance(data.get("linkedin_project"), str)
                else None,
                linkedin_project_id=linkedin_project_id,
                modified_at=modified_at,
                target_modules=target_modules,
                confidentiality_class=confidentiality_class,
            )
        )
    out.sort(key=lambda b: b.modified_at, reverse=True)
    return out


@router.get("/api/briefs", response_model=BriefsListResponse)
def api_briefs(decorate_runs: bool = True) -> BriefsListResponse:
    """List authored briefs from ``config/`` with optional run metadata."""

    return BriefsListResponse(
        briefs=aggregate_briefs(decorate_runs=decorate_runs)
    )


def _resolve_brief_by_id(brief_id: str) -> tuple[Path, bool] | None:
    """Find the catalog file for a brief id."""

    from shared.output_paths import derive_brief_id

    raw_briefs = _scan_authored_briefs(_paths._CONFIG_DIR)
    for brief in raw_briefs:
        abs_path = _paths._CONFIG_PARENT / brief.path
        try:
            computed = derive_brief_id(brief_path=str(abs_path))
        except Exception:
            continue
        if computed == brief_id:
            was_flat = abs_path.parent.resolve() == _paths._CONFIG_DIR.resolve()
            return abs_path, was_flat
    return None


@router.get("/api/brief/{brief_id}", response_model=BriefDetailResponse)
def api_brief_detail(brief_id: str) -> BriefDetailResponse:
    """Read a brief's V2 and legacy partition for editing."""

    resolved = _resolve_brief_by_id(brief_id)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "brief_not_found", "brief_id": brief_id},
        )
    abs_path, was_flat = resolved
    return _brief_detail_from_path(
        abs_path,
        was_flat=was_flat,
        brief_id=brief_id,
    )


def _brief_detail_from_path(
    abs_path: Path,
    *,
    was_flat: bool,
    brief_id: str,
) -> BriefDetailResponse:
    from shared.brief_v2_schema import BriefSchemaError, merge_legacy_brief

    try:
        raw = json.loads(abs_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "brief_unreadable", "reason": str(exc)},
        ) from exc

    try:
        merged = merge_legacy_brief(raw)
    except BriefSchemaError as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "brief_unparseable", "reason": str(exc)},
        ) from exc

    try:
        modified_at = datetime.fromtimestamp(
            abs_path.stat().st_mtime, timezone.utc
        ).isoformat()
    except OSError:
        modified_at = ""

    versions_dir = abs_path.parent / "versions"
    version_count = 0
    if versions_dir.is_dir():
        version_count = sum(
            1 for p in versions_dir.glob("*.json") if p.is_file()
        )

    return BriefDetailResponse(
        brief_id=brief_id,
        path=str(abs_path.relative_to(_paths._CONFIG_PARENT)),
        role_title=raw.get("role_title")
        if isinstance(raw.get("role_title"), str)
        else None,
        v2_data=merged.v2_data,
        preserved_legacy=merged.preserved_legacy,
        deprecated_keys=list(merged.deprecated_keys),
        unknown_keys=list(merged.unknown_keys),
        last_modified=modified_at,
        version_count=version_count,
        was_flat=was_flat,
    )


@router.put("/api/brief/{brief_id}", response_model=BriefDetailResponse)
def api_brief_edit(brief_id: str, request: BriefEditRequest) -> BriefDetailResponse:
    """Write a new version of a brief."""

    from shared.brief_v2_schema import (
        BriefSchemaError,
        validate_v2_brief,
    )
    from shared.brief_writer import write_brief_atomic

    resolved = _resolve_brief_by_id(brief_id)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "brief_not_found", "brief_id": brief_id},
        )
    abs_path, was_flat = resolved

    if request.last_modified is not None:
        from datetime import datetime, timezone

        try:
            current_mtime = datetime.fromtimestamp(
                abs_path.stat().st_mtime, timezone.utc
            ).isoformat()
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": "brief_stat_failed", "reason": str(exc)},
            ) from exc
        if current_mtime != request.last_modified:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "stale_edit",
                    "message": (
                        "The brief was modified by another edit since you opened it. "
                        "Reload and reapply your changes."
                    ),
                    "client_last_modified": request.last_modified,
                    "server_last_modified": current_mtime,
                },
            )

    try:
        validate_v2_brief(request.v2_data)
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

    full_payload = dict(request.preserved_legacy)
    full_payload.update(request.v2_data)

    if was_flat:
        stem = abs_path.stem
        nested_dir = abs_path.parent / stem
        nested_dir.mkdir(parents=True, exist_ok=True)
        nested_path = nested_dir / "brief.json"
        shutil.move(str(abs_path), str(nested_path))
        log.info(
            "Flat brief promoted to nested layout: %s -> %s "
            "(brief_id=%s; Phase D Slice D2)",
            abs_path,
            nested_path,
            brief_id,
        )
        abs_path = nested_path

    try:
        write_brief_atomic(abs_path=abs_path, payload=full_payload)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "brief_write_failed", "reason": str(exc)},
        ) from exc

    from shared.output_paths import derive_brief_id

    new_id = derive_brief_id(brief_path=str(abs_path))
    return _brief_detail_from_path(
        abs_path,
        was_flat=False,
        brief_id=new_id,
    )


@router.get(
    "/api/brief/{brief_id}/versions",
    response_model=BriefVersionsResponse,
)
def api_brief_versions(brief_id: str) -> BriefVersionsResponse:
    """List the snapshot history under ``versions/`` for a brief."""

    resolved = _resolve_brief_by_id(brief_id)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "brief_not_found", "brief_id": brief_id},
        )
    abs_path, _ = resolved

    versions_dir = abs_path.parent / "versions"
    versions: list[BriefVersionEntry] = []
    if versions_dir.is_dir():
        for snapshot in sorted(versions_dir.glob("*.json"), reverse=True):
            if not snapshot.is_file():
                continue
            stem = snapshot.stem
            created_at = stem
            if "T" in stem:
                date_part, _, time_part = stem.partition("T")
                created_at = (
                    date_part
                    + "T"
                    + time_part.replace("-", ":", 2)
                )
            try:
                size = snapshot.stat().st_size
            except OSError:
                size = 0
            versions.append(
                BriefVersionEntry(
                    version_id=stem,
                    created_at=created_at,
                    size_bytes=size,
                )
            )

    return BriefVersionsResponse(brief_id=brief_id, versions=versions)
