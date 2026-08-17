"""Shared output-directory contract helpers."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from shared import config
from shared.brief_loader import Brief, load_brief
from shared.storage import read_json


OUTPUT_ROOT = config.OUTPUT_DIR
STATE_ROOT = OUTPUT_ROOT / "state"
RUNS_ROOT = OUTPUT_ROOT / "runs"
MARKET_INTELLIGENCE_ROOT = OUTPUT_ROOT / "market_intelligence"
EXPORTS_ROOT = OUTPUT_ROOT / "exports"
ARCHIVE_ROOT = OUTPUT_ROOT / "archive"
CACHE_ROOT = OUTPUT_ROOT / "cache"
DEBUG_ROOT = OUTPUT_ROOT / "debug"

# Northwind trial plan, Slice 1B: intake sessions are global (not per-state-dir
# and not per-source) because the brief-authoring conversation pre-dates
# any commitment to source or state_key.
INTAKE_ROOT = OUTPUT_ROOT / "intake"
INTAKE_DB_FILENAME = "intake_sessions.sqlite3"

# Phase F Slice F3: cross-module identity lives outside per-state-dir DBs
# because cross-source identity is, by definition, structurally incapable
# of being seen from within a single source's state_dir. The `_identity`
# prefix sits alongside the per-source dirs under `state/` but starts
# with an underscore so `enumerate_state_dirs()` (which skips
# underscore-prefixed siblings and only yields dirs whose name is in
# :data:`KNOWN_STATE_SOURCES`) never treats the identity DB's parent
# dir as a per-source state-dir root.
IDENTITY_ROOT = STATE_ROOT / "_identity"
IDENTITY_DB_FILENAME = "identity.sqlite3"

# Multi-agent execution Slice 2.3: orchestration SQLite is parallel to
# per-source state-dirs and lives at
# ``output/state/orchestration/runtime_state.sqlite3``. Cross-source
# (chief-of-staff) state is brief-grain not per-source-per-brief grain,
# so it cannot live on the per-source ``runs`` table; a new table inside
# any per-source SQLite would also violate the §1 read-only invariant
# preserved for ``shared.runtime_state.read_models`` callers. The
# directory name is not in :data:`KNOWN_STATE_SOURCES`, so
# ``enumerate_state_dirs`` will not iterate it as a per-source state-dir
# (no underscore-prefix dance needed).
ORCHESTRATION_ROOT = STATE_ROOT / "orchestration"
ORCHESTRATION_DB_FILENAME = "runtime_state.sqlite3"

# Reopen Stage 1/2: the durable RECRUITER primitive lives outside any
# per-source / per-state-dir DB for the same structural reason
# ``_identity`` does — a recruiter spans every brief and every source,
# so a recruiter table inside one source's state_dir cannot reference
# the others. The ``_recruiter`` prefix sits alongside the per-source
# dirs under ``state/`` but starts with an underscore so
# ``enumerate_state_dirs()`` (which skips underscore-prefixed siblings
# and only yields dirs whose name is in :data:`KNOWN_STATE_SOURCES`)
# never treats it as a per-source state-dir root.
# Sibling to ``_identity/identity.sqlite3``.
RECRUITER_ROOT = STATE_ROOT / "_recruiter"
RECRUITER_DB_FILENAME = "recruiter.sqlite3"

for _root in (
    STATE_ROOT,
    RUNS_ROOT,
    MARKET_INTELLIGENCE_ROOT,
    EXPORTS_ROOT,
    ARCHIVE_ROOT,
    CACHE_ROOT,
    DEBUG_ROOT,
    INTAKE_ROOT,
    IDENTITY_ROOT,
    ORCHESTRATION_ROOT,
    RECRUITER_ROOT,
):
    _root.mkdir(parents=True, exist_ok=True)

# Shared-side source registry for canonical state enumeration.
# ``cloris.launchers.known_sources()`` is the launcher registry (launch
# behavior, pipeline state, module picker); this tuple is the allowlist
# for ``enumerate_state_dirs``. Duplicated here rather than imported
# because ``shared/`` must not depend on ``cloris/`` (spec §A2), and the
# launcher registry carries launch behavior this module has no business
# importing. A test pins the two in sync so they cannot drift.
KNOWN_STATE_SOURCES: tuple[str, ...] = (
    "designer",
    "exec_search",
    "github",
    "linkedin",
    "researcher",
)

# P4.4: state dirs left behind by tests that call resolve_linkedin_state_dir()
# (or similar per-source resolvers) without an explicit `state_dir` override.
# Those helpers derive the on-disk key from brief content; several tests build
# that content around a ``tempfile.TemporaryDirectory()``'s generated name
# (e.g. ``f"exhausted-{Path(td).name}"``, which slugify_output_component()
# turns into ``exhausted_tmpXXXXXXXX``) and never pass `state_dir=`, so each
# run derives a *new* key and leaves a fresh, permanent directory under the
# real STATE_ROOT. Confirmed against the actual debris under
# output/state/linkedin/ (74 dirs, all matching this exact pattern — see
# tests/test_linkedin_session_orchestrator.py) before hardcoding it. The
# pattern intentionally does not exclude "unknown", "<numeric-id>", or
# "*_fixture" dirs — none of those observed real/legit state dirs contain
# "_tmp".
#
# Known false-positive class (accepted): a recruiter-authored brief id or
# role title containing the token "tmp" would slug to a key matching this
# glob and be silently hidden from every status/reconciler surface.
# LinkedIn keys are numeric project ids, so the exposure is the
# github/brief-derived path only; judged remote enough to accept — tighten
# to tempfile's suffix shape (_tmp + 8 alphanumerics) if it ever bites.
_TEST_DEBRIS_GLOB = "*_tmp*"


def _is_test_debris_dir(name: str) -> bool:
    return fnmatch.fnmatch(name, _TEST_DEBRIS_GLOB)


def enumerate_state_dirs(
    state_root: Path | None = None,
) -> Iterator[tuple[str, Path]]:
    """Yield ``(source, state_dir)`` pairs across registered source roots.

    ``state_root`` defaults to :data:`STATE_ROOT`. Callers in tests pass an
    explicit ``tmp_path`` and never touch the real ``output/state/`` tree.

    Discovery walks siblings of ``state_root`` whose name is in
    :data:`KNOWN_STATE_SOURCES` (the shared-side source allowlist), skipping
    underscore-prefixed dirs (``_identity``, ``_recruiter``) and any
    unregistered top-level dir (``scratch``, ``orchestration``, …). The
    allowlist mirrors ``cloris.launchers.known_sources()`` on a normal tree
    without importing ``cloris.launchers``.

    Per-source roots that don't exist (or aren't directories) are skipped
    silently; an empty source root yields no entries. Within each source root
    we iterate ``iterdir()`` filtered to directories, sorted by name, so test
    output is deterministic. Directories matching :data:`_TEST_DEBRIS_GLOB`
    (``*_tmp*``) are skipped — see the P4.4 comment above for why they exist
    and why the exclusion is safe.
    """

    root = state_root if state_root is not None else STATE_ROOT
    if not root.exists() or not root.is_dir():
        return

    for source_dir in sorted(root.iterdir()):
        if not source_dir.is_dir():
            continue
        source_name = source_dir.name
        if source_name.startswith("_") or source_name not in KNOWN_STATE_SOURCES:
            continue
        for child in sorted(source_dir.iterdir()):
            if child.is_dir() and not _is_test_debris_dir(child.name):
                yield source_name, child


def resolve_identity_db_path() -> Path:
    """Path to the global cross-module identity SQLite store.

    Cross-source identity (Phase F Slice F3) is fundamentally global —
    it merges duplicates across `output/state/linkedin/<key>/` and
    `output/state/github/<key>/`. Storing the persons table inside any
    one of those state-dirs would be structurally incapable of seeing
    the others, so F3 ships a separate global DB.

    Resolved against the live :data:`IDENTITY_ROOT` so tests that
    monkeypatch :data:`OUTPUT_ROOT` must also monkeypatch
    :data:`IDENTITY_ROOT` (mirrors the
    :func:`resolve_intake_db_path` pattern).
    """

    return IDENTITY_ROOT / IDENTITY_DB_FILENAME


def resolve_recruiter_db_path() -> Path:
    """Path to the global recruiter SQLite store (reopen Stage 1/2).

    The recruiter is Cloris's durable cross-brief, cross-source entity
    (see ``shared/runtime_state/recruiter_store.py``). Like cross-source
    identity, it is fundamentally global — a recruiter owns briefs across
    ``output/state/linkedin/<key>/`` and ``output/state/github/<key>/``,
    so storing the recruiters table inside any one of those state-dirs
    would be structurally incapable of seeing the others. Resolved at
    ``output/state/_recruiter/recruiter.sqlite3``, sibling to
    ``_identity/identity.sqlite3``.

    Resolved against the live :data:`RECRUITER_ROOT` so tests that
    monkeypatch :data:`OUTPUT_ROOT` must also monkeypatch
    :data:`RECRUITER_ROOT` (mirrors the :func:`resolve_identity_db_path`
    pattern — derived constants are not auto-recomputed when the root is
    patched).
    """

    return RECRUITER_ROOT / RECRUITER_DB_FILENAME


def resolve_orchestration_state_dir() -> Path:
    """Path to the orchestration state directory.

    Multi-agent execution Slice 2.3. Mirrors :func:`resolve_identity_db_path`'s
    posture: this is a *global* (cross-source) directory rather than a
    per-(source, state_key) directory, so the resolver takes no brief or
    source argument. Resolved against the live :data:`OUTPUT_ROOT` so
    tests that monkeypatch :data:`OUTPUT_ROOT` propagate into this
    helper without also having to monkeypatch :data:`ORCHESTRATION_ROOT`
    (the derived constant set at import time is only used by the eager
    ``mkdir`` loop above; this resolver re-derives from the live
    :data:`OUTPUT_ROOT` on every call).
    """

    path = OUTPUT_ROOT / "state" / "orchestration"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_orchestration_db_path() -> Path:
    """Path to the orchestration SQLite store.

    Houses ``chief_of_staff_runs`` (brief-grain cross-source) and
    ``cross_brief_playbook_observations`` (per-principal calibration log).
    Distinct from per-state-dir ``runtime_state.sqlite3`` because the
    grain is wrong (per-source per-brief vs cross-source per-brief) and
    because reaching into a per-source DB from a cross-source writer
    would violate the §1 read-only invariant preserved for
    ``shared.runtime_state.read_models`` callers.
    """

    return resolve_orchestration_state_dir() / ORCHESTRATION_DB_FILENAME


def resolve_intake_db_path() -> Path:
    """Path to the global intake-sessions SQLite store.

    Distinct from per-state-dir ``runtime_state.sqlite3`` files: intake
    sessions are authored before any (source, state_key) commitment is
    made. The DB lives at ``output/intake/intake_sessions.sqlite3``.

    Resolved against the live :data:`INTAKE_ROOT` so tests that
    monkeypatch :data:`OUTPUT_ROOT` still need to write through this
    helper after also monkeypatching :data:`INTAKE_ROOT` (mirrors the
    pattern used by other output-path helpers — derived constants are
    not auto-recomputed when the root is patched).
    """

    return INTAKE_ROOT / INTAKE_DB_FILENAME


def slugify_output_component(value: str) -> str:
    lowered = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    while "__" in lowered:
        lowered = lowered.replace("__", "_")
    return lowered.strip("_") or "unknown"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _parts_after_output(path: str | Path) -> tuple[Path, tuple[str, ...]] | None:
    resolved = Path(path).resolve()
    parts = resolved.parts
    try:
        index = len(parts) - 1 - list(reversed(parts)).index("output")
    except ValueError:
        return None
    return resolved, parts[index + 1 :]


def output_root_for_path(path: str | Path | None) -> Path:
    if path is None:
        return OUTPUT_ROOT
    parsed = _parts_after_output(path)
    if parsed is None:
        return OUTPUT_ROOT
    resolved, tail = parsed
    return resolved if not tail else resolved.parents[len(tail) - 1]


def classify_output_location(path: str | Path | None) -> str:
    if path is None:
        return "unknown"
    parsed = _parts_after_output(path)
    if parsed is None:
        return "external"
    _, tail = parsed
    if not tail:
        return "output_root"
    if tail[0] == "state":
        return "state_dir" if len(tail) >= 3 else "state_root"
    if tail[0] == "runs":
        return "run_dir" if len(tail) >= 4 else "runs_root"
    if tail[0] == "market_intelligence":
        return "market_dir" if len(tail) >= 2 else "market_root"
    if tail[0] == "exports":
        return "exports_dir"
    if tail[0] == "archive":
        return "archive_dir"
    if tail[0] == "cache":
        return "cache_dir"
    if tail[0] == "debug":
        return "debug_dir"
    return "legacy_output"


def is_output_root(path: str | Path | None) -> bool:
    return classify_output_location(path) == "output_root"


def is_state_dir(path: str | Path | None) -> bool:
    return classify_output_location(path) == "state_dir"


def is_run_dir(path: str | Path | None) -> bool:
    return classify_output_location(path) == "run_dir"


def source_state_root(source: str, *, output_root: str | Path | None = None) -> Path:
    root = output_root_for_path(output_root)
    path = root / "state" / slugify_output_component(source)
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_runs_root(
    source: str,
    brief_id: str,
    *,
    output_root: str | Path | None = None,
) -> Path:
    root = output_root_for_path(output_root)
    path = root / "runs" / slugify_output_component(source) / slugify_output_component(brief_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_exports_root(
    source: str,
    brief_id: str,
    *,
    output_root: str | Path | None = None,
) -> Path:
    root = output_root_for_path(output_root)
    path = root / "exports" / slugify_output_component(source) / slugify_output_component(brief_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_archive_root(
    source: str,
    brief_id: str,
    *,
    output_root: str | Path | None = None,
) -> Path:
    root = output_root_for_path(output_root)
    path = root / "archive" / slugify_output_component(source) / slugify_output_component(brief_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def derive_brief_id(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
    raw: dict | None = None,
) -> str:
    """Derive the canonical brief id (stable state-key) from brief content.

    Universal resolver: not LinkedIn-specific despite historic naming. The
    canonical id is derived from the brief's primary source identifier
    (LinkedIn ``project_id`` / flat fallback / ``brief.id`` / path stem).

    Phase F Slice F2 added the `source_config.linkedin.project_id`
    lookup ahead of the flat `linkedin_project_id` fallback so the
    same hash holds across the migration: a brief that's been edited
    via F2's UI (writing the nested path) and one that still carries
    only the flat field produce the same state-key. Without the
    fallback, every existing state_dir would be orphaned the moment
    F2 introduced the new path.

    Resolution order:
      1. ``source_config.linkedin.project_id`` (V2 shape, F2 onward)
      2. ``linkedin_project_id`` flat field (Phase D and earlier)
      3. ``brief.linkedin_project_id`` from the parsed Brief
      4. ``brief.id``
      5. ``brief_path.stem`` (last-resort)
    """

    from shared.brief_v2_schema import linkedin_project_id_from_brief

    brief_path = Path(brief_path)
    raw = raw or read_json(brief_path)
    brief = brief or load_brief(str(brief_path))
    candidate = (
        linkedin_project_id_from_brief(raw)
        or brief.linkedin_project_id
        or brief.id
        or brief_path.stem
    )
    return slugify_output_component(str(candidate))


def github_state_key(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
) -> str:
    brief_path = Path(brief_path)
    brief = brief or load_brief(str(brief_path))
    return slugify_output_component(brief.id or brief.role_title or brief_path.stem)


def designer_state_key(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
) -> str:
    """Derive the canonical Designer state-key from brief content.

    Designer briefs don't carry a per-source identifier (no
    LinkedIn-style project_id) — same posture as GitHub. State-key
    falls back through brief.id → role_title → brief filename stem.
    """

    brief_path = Path(brief_path)
    brief = brief or load_brief(str(brief_path))
    return slugify_output_component(brief.id or brief.role_title or brief_path.stem)


def exec_search_state_key(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
) -> str:
    """Derive the canonical Executive Search state-key from brief content.

    Executive Search briefs reuse the LinkedIn evaluation pipeline but
    carry their own state directory under ``output/state/exec_search/``
    (separate from LinkedIn's so confidential briefs don't aggregate
    into the LinkedIn home view). Mirrors GitHub's posture: state-key
    falls back through brief.id → role_title → brief filename stem.
    No project_id concept — saves land in the Cloris-native shortlist
    destination shipping in Slice 7.
    """

    brief_path = Path(brief_path)
    brief = brief or load_brief(str(brief_path))
    return slugify_output_component(brief.id or brief.role_title or brief_path.stem)


def derive_market_key_from_brief(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
    raw: dict | None = None,
) -> str:
    brief_path = Path(brief_path)
    raw = raw or read_json(brief_path)
    brief = brief or load_brief(str(brief_path))
    role_level = str(
        raw.get("role_level")
        or getattr(getattr(brief, "_new_brief", None), "role_level", "")
        or ""
    ).strip()
    geography = str(
        raw.get("geography")
        or brief.permanent_filters.get("Location")
        or ""
    ).strip()
    return "__".join(
        [
            slugify_output_component(brief.role_title),
            slugify_output_component(geography),
            slugify_output_component(role_level),
        ]
    )


def resolve_linkedin_state_dir(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
    raw: dict | None = None,
    state_dir: str | Path | None = None,
) -> Path:
    if state_dir:
        path = Path(state_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    key = derive_brief_id(brief_path=brief_path, brief=brief, raw=raw)
    path = source_state_root("linkedin") / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_github_state_dir(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
    state_dir: str | Path | None = None,
) -> Path:
    if state_dir:
        path = Path(state_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    key = github_state_key(brief_path=brief_path, brief=brief)
    path = source_state_root("github") / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_designer_state_dir(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
    state_dir: str | Path | None = None,
) -> Path:
    """Resolve the per-brief state directory for a Designer run.

    Mirrors :func:`resolve_github_state_dir`. The Designer state-key
    derives from brief content (no per-source identifier), and the
    state-dir lives under ``output/state/designer/<state_key>/``.
    Caches the SQLite asset blob, the canonical
    ``runtime_state.sqlite3``, and the JSONL projection files for the
    Designer module's lifetime + 30 days.
    """

    if state_dir:
        path = Path(state_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    key = designer_state_key(brief_path=brief_path, brief=brief)
    path = source_state_root("designer") / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_exec_search_state_dir(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
    state_dir: str | Path | None = None,
) -> Path:
    """Resolve the per-brief state directory for an Executive Search run.

    Mirrors :func:`resolve_github_state_dir`. State-dir lives under
    ``output/state/exec_search/<state_key>/``, separate from
    ``output/state/linkedin/`` so confidential briefs don't aggregate
    into LinkedIn's shared per-source views.
    """

    if state_dir:
        path = Path(state_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    key = exec_search_state_key(brief_path=brief_path, brief=brief)
    path = source_state_root("exec_search") / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def researcher_state_key(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
) -> str:
    """Derive the canonical Researcher state-key from brief content.

    Researcher has no per-brief project_id concept (workspace is the
    only save destination per Researcher Module Spec Opinion 4), so the
    key resolves from ``brief.id`` → ``brief.role_title`` → file stem,
    mirroring :func:`github_state_key`.
    """

    brief_path = Path(brief_path)
    brief = brief or load_brief(str(brief_path))
    return slugify_output_component(brief.id or brief.role_title or brief_path.stem)


def resolve_researcher_state_dir(
    *,
    brief_path: str | Path,
    brief: Brief | None = None,
    state_dir: str | Path | None = None,
) -> Path:
    if state_dir:
        path = Path(state_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    key = researcher_state_key(brief_path=brief_path, brief=brief)
    path = source_state_root("researcher") / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_run_dir(
    *,
    source: str,
    brief_id: str,
    run_stamp: str,
    run_id: int | str | None,
    imported: bool = False,
    legacy_index: int | None = None,
    output_root: str | Path | None = None,
) -> Path:
    parent = source_runs_root(source, brief_id, output_root=output_root)
    if imported:
        suffix = f"__legacy-{int(legacy_index or 1)}"
        name = f"imported-{run_stamp}{suffix}"
    else:
        suffix = f"__run-{run_id}" if run_id is not None else ""
        name = f"{run_stamp}{suffix}"
    path = parent / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def looks_like_finalized_run_dir(path: str | Path | None) -> bool:
    if not is_run_dir(path):
        return False
    if path is None:
        return False
    candidate = Path(path)
    if (candidate / "run-manifest.json").exists():
        return True
    required = (
        candidate / "final_judgments.jsonl",
        candidate / "runtime_state.sqlite3",
        candidate / "run-report.json",
    )
    return any(item.exists() for item in required)
