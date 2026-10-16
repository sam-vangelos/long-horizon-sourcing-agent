"""Brief-derived registry targeting for npm, crates.io, and PyPI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from shared.resolvers.ecosystems import REGISTRY_ALIASES

SUPPORTED_ECOSYSTEMS: frozenset[str] = frozenset(
    {"npmjs.org", "crates.io", "pypi.org", "huggingface.co"}
)
# Stack selection excludes huggingface.co — strategy adds HF via bare-token evidence only.
_STACK_SELECTABLE_ECOSYSTEMS: frozenset[str] = frozenset(
    {"npmjs.org", "crates.io", "pypi.org"}
)
W2_ECOSYSTEMS = _STACK_SELECTABLE_ECOSYSTEMS

_NPM_REQUIRE = re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)""")
_NPM_IMPORT_FROM = re.compile(r"""from\s+['"]([^'"]+)['"]""")
_NPM_PACKAGE_JSON = re.compile(
    r"""['"]([^'"]+)['"]\s+in\s+package\.json\s+dependencies""",
    re.IGNORECASE,
)
_CRATES_USE = re.compile(r"""use\s+([a-zA-Z0-9_-]+)::""")
_CRATES_CARGO_ADD = re.compile(r"""cargo\s+add\s+([a-zA-Z0-9_-]+)""")

_BARE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]{2,}$")
_NPM_SCOPED_TOKEN_RE = re.compile(r"^@[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_PEP503_RUN_RE = re.compile(r"[-_.]+")


@dataclass(frozen=True)
class RegistryTarget:
    ecosystem: str
    seed_packages: list[str]


def _normalize_stack(stack: str) -> str | None:
    key = str(stack or "").strip().lower()
    if not key:
        return None
    return REGISTRY_ALIASES.get(key)


def _target_stacks_from_brief(brief: Any) -> list[str]:
    if isinstance(brief, dict):
        stacks = brief.get("target_stacks") or []
    else:
        stacks = getattr(brief, "target_stacks", None) or []
    return [stack for stack in stacks if isinstance(stack, str)]


def _ecosystems_from_target_stacks(brief: Any) -> set[str]:
    ecosystems: set[str] = set()
    for stack in _target_stacks_from_brief(brief):
        normalized = _normalize_stack(stack)
        if normalized in _STACK_SELECTABLE_ECOSYSTEMS:
            ecosystems.add(normalized)
    return ecosystems


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


def _is_bare_token(signal: str, ecosystem: str) -> bool:
    token = signal.strip()
    if not token or any(char.isspace() for char in token):
        return False
    if len(token) < 2:
        return False
    if ecosystem == "npmjs.org":
        if _NPM_SCOPED_TOKEN_RE.fullmatch(token):
            return True
        return _BARE_TOKEN_RE.fullmatch(token) is not None
    return _BARE_TOKEN_RE.fullmatch(token) is not None


def _normalize_bare_token(token: str, ecosystem: str) -> str:
    if ecosystem == "pypi.org":
        normalized = token.lower()
        return _PEP503_RUN_RE.sub("-", normalized)
    if ecosystem in {"npmjs.org", "crates.io"}:
        return token.lower()
    return token


def bare_token_seeds(brief: Any, ecosystem: str) -> list[str]:
    """Collect grammar-passing ``github_code_signals`` as registry seed tokens."""
    if ecosystem not in SUPPORTED_ECOSYSTEMS:
        return []

    seeds: list[str] = []
    seen: set[str] = set()
    for signal in _code_signals_from_brief(brief):
        if not _is_bare_token(signal, ecosystem):
            continue
        normalized = _normalize_bare_token(signal.strip(), ecosystem)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        seeds.append(normalized)
    return seeds


def _parse_npm_seeds(signals: list[str]) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        for pattern in (_NPM_REQUIRE, _NPM_IMPORT_FROM, _NPM_PACKAGE_JSON):
            for match in pattern.finditer(signal):
                package = match.group(1).strip()
                if package and package not in seen:
                    seen.add(package)
                    seeds.append(package)
    return seeds


def _parse_crates_seeds(signals: list[str]) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        for pattern in (_CRATES_USE, _CRATES_CARGO_ADD):
            for match in pattern.finditer(signal):
                package = match.group(1).strip()
                if package and package not in seen:
                    seen.add(package)
                    seeds.append(package)
    return seeds


def derive_registry_targets(brief: Any) -> list[RegistryTarget]:
    """Derive npm, crates.io, and PyPI registry targets from a brief."""
    ecosystems = _ecosystems_from_target_stacks(brief)
    signals = _code_signals_from_brief(brief)
    npm_seeds = _parse_npm_seeds(signals)
    crates_seeds = _parse_crates_seeds(signals)

    if npm_seeds:
        ecosystems.add("npmjs.org")
    if crates_seeds:
        ecosystems.add("crates.io")

    if not ecosystems:
        return []

    targets: list[RegistryTarget] = []
    if "npmjs.org" in ecosystems:
        targets.append(RegistryTarget("npmjs.org", npm_seeds))
    if "crates.io" in ecosystems:
        targets.append(RegistryTarget("crates.io", crates_seeds))
    if "pypi.org" in ecosystems:
        targets.append(RegistryTarget("pypi.org", []))
    return targets
