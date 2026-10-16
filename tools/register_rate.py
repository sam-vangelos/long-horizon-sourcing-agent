#!/usr/bin/env python3
"""Measure generated-string register against brief vocabulary channels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from shared.boolean_metrics import _QUOTED
except ImportError:  # pragma: no cover - fallback for standalone use.
    _QUOTED = re.compile(r'"([^"]*)"')


def _norm(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _read_text(path_or_raw: str) -> str:
    if "\n" not in path_or_raw and len(path_or_raw) < 1024:
        try:
            path = Path(path_or_raw)
            if path.exists():
                return path.read_text()
        except OSError:
            pass
    return path_or_raw


def _parse_jsonish(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def _generated_strings(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        generated = payload.get("generated_strings")
        if isinstance(generated, list):
            return generated
        for value in payload.values():
            found = _generated_strings(value)
            if found:
                return found
    return []


def _booleans(plan_payload: Any) -> list[str]:
    booleans: list[str] = []
    for item in _generated_strings(plan_payload):
        if isinstance(item, dict):
            boolean = item.get("boolean")
        else:
            boolean = item
        if isinstance(boolean, str) and boolean.strip():
            booleans.append(boolean)
    return booleans


def _brief_reference_sets(brief_payload: Any) -> tuple[set[str], set[str]]:
    key_refs: set[str] = set()
    candidate_refs: set[str] = set()
    if not isinstance(brief_payload, dict):
        return key_refs, candidate_refs
    for area in brief_payload.get("capability_areas") or []:
        if not isinstance(area, dict):
            continue
        area_name = _norm(area.get("name"))
        if area_name:
            key_refs.add(area_name)
        for term in area.get("key_terms") or []:
            normalized = _norm(term)
            if normalized:
                key_refs.add(normalized)
        for term in area.get("candidate_register_terms") or []:
            normalized = _norm(term)
            if normalized:
                candidate_refs.add(normalized)
    return key_refs, candidate_refs


def _quoted_terms(booleans: list[str]) -> list[str]:
    terms: list[str] = []
    for boolean in booleans:
        terms.extend(_norm(match) for match in _QUOTED.findall(boolean) if _norm(match))
    return terms


def _row(label: str, hits: int, total: int) -> str:
    rate = (hits / total) if total else 0.0
    return f"{label:<34} {hits:>5} {total:>5} {rate:>8.1%}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare quoted terms in generated_strings booleans against a "
            "brief's key_terms+area-names and candidate_register_terms sets."
        )
    )
    parser.add_argument("plan", help="Execution-plan JSON file or raw formation response")
    parser.add_argument("brief", help="V2 brief JSON file or raw brief JSON")
    args = parser.parse_args(argv)

    plan_payload = _parse_jsonish(_read_text(args.plan))
    brief_payload = _parse_jsonish(_read_text(args.brief))
    terms = _quoted_terms(_booleans(plan_payload))
    key_refs, candidate_refs = _brief_reference_sets(brief_payload)

    key_hits = sum(1 for term in terms if term in key_refs)
    candidate_hits = sum(1 for term in terms if term in candidate_refs)

    print(f"{'reference set':<34} {'hits':>5} {'total':>5} {'rate':>8}")
    print("-" * 56)
    print(_row("key_terms + area names", key_hits, len(terms)))
    print(_row("candidate_register_terms", candidate_hits, len(terms)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
