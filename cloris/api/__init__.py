"""Cloris HTTP API surface (package).

Route implementation is split across small package modules plus the remaining
legacy :mod:`cloris.api._monolith` surface. This package exports only the
supported compatibility names used by app bootstrap, control-plane code, and
tests; new callers should import from the owning module directly.

Path sentinels live in :mod:`cloris.api._paths`. Monkeypatch
``_PROJECT_ROOT`` / ``_CONFIG_DIR`` / ``_CONFIG_PARENT`` there so all API
submodules observe the override.
"""

from __future__ import annotations

from typing import Any

# Import order: split modules register routes on the shared ``router`` before
# ``_monolith`` attaches the remaining legacy endpoints.
from . import _paths
from . import health  # noqa: F401
from . import conversation  # noqa: F401
from . import static_ui  # noqa: F401
from . import briefs  # noqa: F401
from . import intake  # noqa: F401
from . import candidate_routes  # noqa: F401
from . import _monolith
from .briefs import _scan_authored_briefs
from .conversation import _CONVERSATION_QUERY_BUCKETS
from .intake import _intake_db_path, _intake_store
from ._monolith import (
    BriefIdNotFoundError,
    DomainPausedError,
    LaunchLinkedInRequest,
    LaunchNotReadyError,
    NoPendingWorkError,
    StateDirNotFoundError,
    UnknownSourceError,
    WorkerAlreadyRunningError,
    _SpawnResult,
    _build_worker_argv,
    _readiness_blockers,
    _reflection_store_factory,
    _resolve_brief_path_or_raise,
    _spawn_linkedin_worker,
    _spawn_worker_for_source,
    stop_worker,
)
from .static_ui import _warn_if_dist_stale, mount_static
from .routing import router

_PATH_EXPORTS = frozenset(
    {
        "_CONFIG_DIR",
        "_CONFIG_PARENT",
        "_PROJECT_ROOT",
        "_DIST_DIR",
        "_FRONTEND_SRC_DIR",
    }
)


def __getattr__(name: str) -> Any:
    if name in _PATH_EXPORTS:
        return getattr(_paths, name)
    raise AttributeError(f"module 'cloris.api' has no exported attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PATH_EXPORTS)


__all__ = [
    "BriefIdNotFoundError",
    "DomainPausedError",
    "LaunchLinkedInRequest",
    "LaunchNotReadyError",
    "NoPendingWorkError",
    "StateDirNotFoundError",
    "UnknownSourceError",
    "WorkerAlreadyRunningError",
    "_CONFIG_DIR",
    "_CONFIG_PARENT",
    "_CONVERSATION_QUERY_BUCKETS",
    "_DIST_DIR",
    "_FRONTEND_SRC_DIR",
    "_PROJECT_ROOT",
    "_SpawnResult",
    "_build_worker_argv",
    "_intake_db_path",
    "_intake_store",
    "_monolith",
    "_paths",
    "_readiness_blockers",
    "_reflection_store_factory",
    "_resolve_brief_path_or_raise",
    "_scan_authored_briefs",
    "_spawn_linkedin_worker",
    "_spawn_worker_for_source",
    "_warn_if_dist_stale",
    "mount_static",
    "router",
    "stop_worker",
]
