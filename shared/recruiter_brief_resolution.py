"""Resolve the canonical LinkedIn brief for GitHub-run → Recruiter reconciliation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from shared.brief_loader import load_brief
from shared.output_paths import derive_market_key_from_brief
from shared.storage import read_json

# Match a trailing version suffix on a brief filename, e.g. "...-v1.4.json".
# The suffix must be the LAST hyphen-separated token before .json so we don't
# accidentally pick up role-name fragments that happen to contain "v1".
_BRIEF_VERSION_RE = re.compile(r"-v(\d+(?:\.\d+)*)$")


def _read_github_brief_path_from_manifest(github_output_dir: Path) -> Path | None:
    manifest_path = github_output_dir / "run-manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_path = str(data.get("brief_path") or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_file() else None


def _parse_brief_version(stem: str) -> tuple[int, ...] | None:
    """Return the numeric version tuple parsed from a brief filename stem.

    Examples:
      ``"brief-foo-v1"`` -> ``(1,)``
      ``"brief-foo-v1.4"`` -> ``(1, 4)``
      ``"brief-foo-v2.10.3"`` -> ``(2, 10, 3)``
      ``"brief-foo"`` -> ``None`` (no parseable version suffix)

    Returning ``None`` is the signal to the caller that the sibling cannot be
    safely version-compared and the run should fail closed (plan §6).
    """
    match = _BRIEF_VERSION_RE.search(stem)
    if not match:
        return None
    parts = match.group(1).split(".")
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def _select_highest_version_brief(matches: list[Path]) -> Path:
    """Choose the matching sibling brief with the highest numeric version suffix.

    Plan §6 fail-closed rules:
      - If any matching sibling does not carry a parseable ``-vN[.N]*`` suffix,
        raise ValueError so the operator must pass ``--linkedin-brief``.
      - If two or more siblings share the highest version, raise ValueError for
        the same reason.
      - Otherwise return the unique highest-version sibling.
    """
    versioned: list[tuple[tuple[int, ...], Path]] = []
    unparseable: list[Path] = []
    for path in matches:
        version = _parse_brief_version(path.stem)
        if version is None:
            unparseable.append(path)
        else:
            versioned.append((version, path))

    if unparseable:
        listing = ", ".join(sorted(p.name for p in unparseable))
        raise ValueError(
            "Ambiguous LinkedIn brief sibling set: at least one matching sibling has no "
            f"parseable -vN[.N]* version suffix ({listing}). "
            "Pass --linkedin-brief with the canonical LinkedIn brief path."
        )

    versioned.sort(key=lambda item: item[0], reverse=True)
    top_version = versioned[0][0]
    top_paths = [path for version, path in versioned if version == top_version]
    if len(top_paths) > 1:
        listing = ", ".join(sorted(p.name for p in top_paths))
        raise ValueError(
            "Ambiguous LinkedIn brief sibling set: multiple siblings share the highest "
            f"version v{'.'.join(str(part) for part in top_version)} ({listing}). "
            "Pass --linkedin-brief with the canonical LinkedIn brief path."
        )
    return top_paths[0]


def resolve_linkedin_brief_path_for_github_run(
    github_output_dir: str | Path,
    *,
    explicit_linkedin_brief: str | Path | None = None,
    github_brief_path: str | Path | None = None,
) -> Path:
    """Pick the LinkedIn (non-GitHub) brief that matches the GitHub run's role/market context.

    Resolution order:
    1. ``explicit_linkedin_brief`` when provided (must exist).
    2. Otherwise, read ``run-manifest.json`` → ``brief_path`` (GitHub brief used for the run).
    3. In the same directory as that GitHub brief, choose a sibling ``*.json`` that:
       - is not the GitHub brief file
       - does not have ``github`` in the filename (case-insensitive)
       - yields the same ``derive_market_key_from_brief`` as the GitHub brief
       - shares the same ``linkedin_project_id`` in raw JSON

    Sibling selection (plan §6, version-aware fail-closed):
      - Each sibling filename must carry a trailing ``-vN[.N]*`` numeric version
        suffix (e.g. ``-v1``, ``-v1.4``).
      - The unique highest numeric version wins.
      - If any sibling lacks a parseable suffix, or two or more siblings tie at
        the highest version, this function raises ValueError so the operator
        must pass ``--linkedin-brief`` explicitly.
    """
    github_output_dir = Path(github_output_dir)
    if explicit_linkedin_brief:
        path = Path(explicit_linkedin_brief).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LinkedIn brief not found: {path}")
        return path

    gh_brief_path: Path | None = None
    if github_brief_path:
        gh_brief_path = Path(github_brief_path).expanduser().resolve()
        if not gh_brief_path.is_file():
            raise FileNotFoundError(f"GitHub brief not found: {gh_brief_path}")
    else:
        gh_brief_path = _read_github_brief_path_from_manifest(github_output_dir)

    if gh_brief_path is None:
        raise ValueError(
            "Could not resolve GitHub brief path (missing run-manifest.json or brief_path). "
            "Pass --github-brief or --linkedin-brief explicitly."
        )

    raw_gh = read_json(gh_brief_path)
    project_id = str(raw_gh.get("linkedin_project_id") or "").strip()
    market_key = derive_market_key_from_brief(brief_path=gh_brief_path)

    matches: list[Path] = []
    for candidate in sorted(gh_brief_path.parent.glob("*.json")):
        if candidate.resolve() == gh_brief_path.resolve():
            continue
        if "github" in candidate.name.lower():
            continue
        try:
            raw_c = read_json(candidate)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if project_id and str(raw_c.get("linkedin_project_id") or "").strip() != project_id:
            continue
        try:
            if derive_market_key_from_brief(brief_path=candidate) != market_key:
                continue
        except Exception:
            continue
        # Must load as a sourcing brief (LinkedIn briefs use V2 or legacy schema).
        try:
            load_brief(str(candidate))
        except Exception:
            continue
        matches.append(candidate)

    if not matches:
        raise ValueError(
            f"No LinkedIn brief sibling found for {gh_brief_path} "
            f"(market_key={market_key!r}, linkedin_project_id={project_id!r}). "
            "Pass --linkedin-brief with the canonical LinkedIn brief path."
        )

    return _select_highest_version_brief(matches).resolve()
