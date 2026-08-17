"""Brief-derived registry targeting for npm and crates.io."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from shared.resolvers.ecosystems import REGISTRY_ALIASES

W2_ECOSYSTEMS: frozenset[str] = frozenset({"npmjs.org", "crates.io"})

_NPM_REQUIRE = re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)""")
_NPM_IMPORT_FROM = re.compile(r"""from\s+['"]([^'"]+)['"]""")
_NPM_PACKAGE_JSON = re.compile(
    r"""['"]([^'"]+)['"]\s+in\s+package\.json\s+dependencies""",
    re.IGNORECASE,
)
_CRATES_USE = re.compile(r"""use\s+([a-zA-Z0-9_-]+)::""")
_CRATES_CARGO_ADD = re.compile(r"""cargo\s+add\s+([a-zA-Z0-9_-]+)""")


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
        if normalized in W2_ECOSYSTEMS:
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
    """Derive npm and crates.io registry targets from a brief."""
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
    return targets
