#!/usr/bin/env python3
"""Shared filesystem and source-identity contract for paid GLM evidence."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHROME_PROFILE_ROOT = (Path.home() / ".chrome-cdp").resolve()

# The offline matrix does not execute all of these files, but each one can
# change the safety or attribution of the live one-page canary. The paid smoke
# receipt pins them so readiness can prove they have not changed since evidence
# collection.
LIVE_CANARY_RUNTIME_FILES = (
    "linkedin/orchestrator.py",
    "linkedin/session_orchestrator.py",
    "shared/runtime_state/__init__.py",
    "shared/runtime_state/linkedin.py",
    "shared/runtime_state/store.py",
    "tools/glm_artifact_contract.py",
)


class ArtifactContractError(ValueError):
    """A paid-evidence artifact path crosses a protected local boundary."""


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_external_artifact_path(
    path: Path,
    *,
    label: str,
    must_exist: bool = False,
) -> Path:
    """Resolve one paid-artifact path outside the repo and Chrome profile."""

    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ArtifactContractError(f"{label} must be an absolute path")
    if raw.is_symlink():
        raise ArtifactContractError(f"{label} must not be a symlink")
    resolved = raw.resolve()
    protected_roots = (REPO_ROOT.resolve(), CHROME_PROFILE_ROOT)
    if any(_is_relative_to(resolved, root) for root in protected_roots):
        raise ArtifactContractError(
            f"{label} is inside the protected repository or browser profile"
        )
    if must_exist and not resolved.exists():
        raise ArtifactContractError(f"{label} does not exist: {resolved}")
    return resolved


__all__ = [
    "ArtifactContractError",
    "LIVE_CANARY_RUNTIME_FILES",
    "resolve_external_artifact_path",
]
