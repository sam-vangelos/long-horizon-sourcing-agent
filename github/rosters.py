"""Governance roster reader for OSS Maintainers Wave 3.

Structural extraction of declared authority from CODEOWNERS,
MAINTAINERS, GOVERNANCE, and conda-forge recipe-maintainers files.
Identity doctrine: only ``@handle`` tokens, ``github.com/<login>``
profile URLs, and YAML list entries identify people — bare names and
emails are never extracted or stored.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from github import maintainer_signal_cache as mcache
from github.maintainership import GOVERNANCE_PATHS, MAINTAINERS_PATHS

logger = logging.getLogger(__name__)

CODEOWNERS_PATHS: tuple[str, ...] = (
    ".github/CODEOWNERS",
    "CODEOWNERS",
    "docs/CODEOWNERS",
)

RECIPE_META_PATH = "recipe/meta.yaml"

_GITHUB_HANDLE = re.compile(
    r"(?<![\w.@-])@([a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)"
)
_GITHUB_PROFILE_URL = re.compile(
    r"https://github\.com/([a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)(?:\s|#|\?|$)"
)
_EMAIL_TOKEN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PYTHON_IMPORT = re.compile(r"\bimport\s+([a-zA-Z_][\w.]*)")
_PYTHON_FROM_IMPORT = re.compile(r"\bfrom\s+([a-zA-Z_][\w.]*)\s+import")


class _RepoContentsClient(Protocol):
    async def get_repo_contents(self, owner_repo: str, path: str) -> Optional[str]: ...


@dataclass(frozen=True)
class RosterEntry:
    handle: str
    role: str
    source_file: str
    repo: str


@dataclass(frozen=True)
class RosterResult:
    repo: str
    entries: list[RosterEntry]
    team_entries: list[str]
    files_found: list[str]


def parse_codeowners(
    text: str,
    *,
    source_file: str,
    repo: str,
) -> tuple[list[RosterEntry], list[str]]:
    """Parse a CODEOWNERS file into handle entries and team refs."""
    entries: list[RosterEntry] = []
    teams: list[str] = []
    seen_handles: set[str] = set()
    seen_teams: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        for token in tokens[1:]:
            if token.startswith("@"):
                ref = token[1:]
                if "/" in ref:
                    if ref not in seen_teams:
                        seen_teams.add(ref)
                        teams.append(ref)
                elif ref not in seen_handles:
                    seen_handles.add(ref)
                    entries.append(
                        RosterEntry(
                            handle=ref,
                            role="code_owner",
                            source_file=source_file,
                            repo=repo,
                        )
                    )
            elif _EMAIL_TOKEN.match(token):
                continue

    return entries, teams


def parse_maintainers_text(
    text: str,
    *,
    source_file: str,
    repo: str,
    role: str,
) -> list[RosterEntry]:
    """Extract ``@handle`` tokens and GitHub profile URLs from roster text."""
    entries: list[RosterEntry] = []
    seen_handles: set[str] = set()

    def _add_handle(handle: str) -> None:
        if handle and handle not in seen_handles:
            seen_handles.add(handle)
            entries.append(
                RosterEntry(
                    handle=handle,
                    role=role,
                    source_file=source_file,
                    repo=repo,
                )
            )

    for match in _GITHUB_HANDLE.finditer(text):
        if match.end() < len(text) and text[match.end()] == "/":
            continue
        _add_handle(match.group(1))

    for match in _GITHUB_PROFILE_URL.finditer(text):
        _add_handle(match.group(1))

    return entries


def parse_recipe_maintainers(meta_yaml_text: str, *, repo: str) -> list[RosterEntry]:
    """Parse conda-forge ``recipe-maintainers`` list from ``meta.yaml`` text."""
    try:
        lines = meta_yaml_text.splitlines()
        in_block = False
        entries: list[RosterEntry] = []
        seen_handles: set[str] = set()

        for line in lines:
            if re.match(r"^\s*recipe-maintainers:\s*$", line):
                in_block = True
                continue
            if not in_block:
                continue
            item_match = re.match(r"^\s*-\s+(.+)$", line)
            if not item_match:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    continue
                if not line.startswith((" ", "\t")):
                    break
                return []

            value = item_match.group(1).strip()
            if not value:
                return []
            if value[0] in "\"'":
                quote = value[0]
                close_idx = value.find(quote, 1)
                if close_idx == -1:
                    return []
                value = value[1:close_idx]
            else:
                value = re.sub(r"\s+#.*$", "", value).strip()

            if not value or value in seen_handles:
                continue
            seen_handles.add(value)
            entries.append(
                RosterEntry(
                    handle=value,
                    role="recipe_maintainer",
                    source_file=RECIPE_META_PATH,
                    repo=repo,
                )
            )

        return entries
    except Exception:  # noqa: BLE001 — fail-soft per spec
        return []


def _code_signals_from_brief(brief: Any) -> list[str]:
    signals: list[str] = []
    if isinstance(brief, dict):
        areas = brief.get("capability_areas") or []
    else:
        new_brief = getattr(brief, "_new_brief", None)
        if new_brief is not None:
            areas = getattr(new_brief, "capability_areas", None) or []
        else:
            areas = getattr(brief, "capability_areas", None) or []
    for area in areas:
        if isinstance(area, dict):
            area_signals = area.get("github_code_signals") or []
        else:
            area_signals = getattr(area, "github_code_signals", None) or []
        for signal in area_signals:
            if isinstance(signal, str) and signal.strip():
                signals.append(signal.strip())
    return signals


def _python_packages_from_signals(signals: list[str]) -> list[str]:
    packages: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        for match in _PYTHON_FROM_IMPORT.finditer(signal):
            package = match.group(1).split(".")[0].lower()
            if package and package not in seen:
                seen.add(package)
                packages.append(package)
        without_from_imports = re.sub(
            r"\bfrom\s+[\w.]+\s+import\b.*",
            "",
            signal,
        )
        for match in _PYTHON_IMPORT.finditer(without_from_imports):
            package = match.group(1).split(".")[0].lower()
            if package and package not in seen:
                seen.add(package)
                packages.append(package)
    return packages


def derive_feedstock_repos(brief: Any) -> list[str]:
    """Derive conda-forge feedstock repos from brief python import signals."""
    packages = _python_packages_from_signals(_code_signals_from_brief(brief))
    return [f"conda-forge/{package}-feedstock" for package in packages]


def _is_feedstock_repo(owner_repo: str) -> bool:
    if "/" not in owner_repo:
        return False
    owner, repo = owner_repo.split("/", 1)
    return owner.lower() == "conda-forge" and repo.lower().endswith("-feedstock")


_ROSTER_PATH_SEP = "\n"

_CLASSIFIER_SHARED_CACHE_KINDS: frozenset[str] = frozenset(
    {"maintainers_file", "governance"}
)


def _unpack_cached_roster_payload(
    cached_text: str,
    paths: tuple[str, ...],
    cache_kind: str,
) -> tuple[str, str]:
    if cache_kind in _CLASSIFIER_SHARED_CACHE_KINDS:
        return cached_text, paths[0]
    if cached_text.startswith("PATH:") and _ROSTER_PATH_SEP in cached_text:
        header, text = cached_text.split(_ROSTER_PATH_SEP, 1)
        return text, header[5:]
    return cached_text, paths[0]


def _pack_cached_roster_payload(path: str, text: str, cache_kind: str) -> str:
    if cache_kind in _CLASSIFIER_SHARED_CACHE_KINDS:
        return text
    return f"PATH:{path}{_ROSTER_PATH_SEP}{text}"


async def _fetch_roster_text(
    client: _RepoContentsClient,
    owner: str,
    repo: str,
    paths: tuple[str, ...],
    cache_kind: str,
) -> tuple[Optional[str], Optional[str]]:
    """Fetch roster file text with mcache; return (text, path_found)."""
    cached = mcache.get(owner, repo, cache_kind)
    if cached is not None and isinstance(cached.data, str):
        if not cached.data:
            return None, None
        text, path = _unpack_cached_roster_payload(cached.data, paths, cache_kind)
        return text, path

    saw_exception = False
    for path in paths:
        try:
            text = await client.get_repo_contents(f"{owner}/{repo}", path)
        except Exception as exc:  # noqa: BLE001 — fail-soft per spec §12
            saw_exception = True
            logger.warning(
                "roster contents fetch failed for %s/%s/%s: %s",
                owner,
                repo,
                path,
                exc,
            )
            continue
        if text:
            mcache.put(
                owner,
                repo,
                cache_kind,
                _pack_cached_roster_payload(path, text, cache_kind),
            )
            return text, path

    if not saw_exception:
        mcache.put(owner, repo, cache_kind, "")
    return None, None


async def fetch_repo_roster(
    client: _RepoContentsClient,
    owner_repo: str,
) -> RosterResult:
    """Fetch and parse all governance roster files for a repository."""
    if "/" not in owner_repo:
        return RosterResult(
            repo=owner_repo,
            entries=[],
            team_entries=[],
            files_found=[],
        )

    owner, repo = owner_repo.split("/", 1)
    entries: list[RosterEntry] = []
    team_entries: list[str] = []
    files_found: list[str] = []

    codeowners_text, codeowners_path = await _fetch_roster_text(
        client,
        owner,
        repo,
        CODEOWNERS_PATHS,
        "roster_codeowners",
    )
    if codeowners_text and codeowners_path:
        co_entries, co_teams = parse_codeowners(
            codeowners_text,
            source_file=codeowners_path,
            repo=owner_repo,
        )
        entries.extend(co_entries)
        team_entries.extend(co_teams)
        files_found.append(codeowners_path)

    maintainers_text, maintainers_path = await _fetch_roster_text(
        client,
        owner,
        repo,
        MAINTAINERS_PATHS,
        "maintainers_file",
    )
    if maintainers_text and maintainers_path:
        entries.extend(
            parse_maintainers_text(
                maintainers_text,
                source_file=maintainers_path,
                repo=owner_repo,
                role="maintainer",
            )
        )
        files_found.append(maintainers_path)

    governance_text, governance_path = await _fetch_roster_text(
        client,
        owner,
        repo,
        GOVERNANCE_PATHS,
        "governance",
    )
    if governance_text and governance_path:
        entries.extend(
            parse_maintainers_text(
                governance_text,
                source_file=governance_path,
                repo=owner_repo,
                role="governance_listed",
            )
        )
        files_found.append(governance_path)

    if _is_feedstock_repo(owner_repo):
        recipe_text, recipe_path = await _fetch_roster_text(
            client,
            owner,
            repo,
            (RECIPE_META_PATH,),
            "roster_recipe",
        )
        if recipe_text and recipe_path:
            entries.extend(parse_recipe_maintainers(recipe_text, repo=owner_repo))
            files_found.append(recipe_path)

    return RosterResult(
        repo=owner_repo,
        entries=entries,
        team_entries=team_entries,
        files_found=files_found,
    )
