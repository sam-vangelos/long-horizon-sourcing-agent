#!/usr/bin/env python3
"""Role-agnosticism ratchet (plans/sourcing-generality-hardening.md item 21).

Structure-level sibling of tools/lint_vertical_vocab.py. That linter catches
vertical VOCABULARY carried as string literals in deterministic code; this one
catches role-SHAPED code that carries no lexicon nouns at all — the structures
a vertical grows back through even after its words are gone.

Three checks, one finding stream:

1. ``renderer`` — NULL-BRIEF RENDER RESIDUE. The assembled prompts
   (linkedin.judgment_templates.assemble_facial_system /
   assemble_facial_batch_system / assemble_full_evaluation_system and
   linkedin.strategy._build_strategy_system) are rendered against a synthetic
   neutral brief whose every string field is a "neutral-term-*" placeholder,
   built through the real production path
   (shared.preflight_v2.preflight_to_brief_json →
   shared.brief_loader._load_v2_brief). The rendered text is scanned against
   the vertical-vocab lexicon. The brief contributes zero lexicon terms, so
   ANY hit came from CODE — templates, fallback prose, defaults — which is
   exactly the residue the literal scan misses when it hides behind a
   grandfather marker or a computed string. Both facial ambiguity postures
   ("binary" and "ternary") are rendered so the finding set does not depend on
   the LINKEDIN_FACIAL_BORDERLINE_ENABLED environment default, and the
   full-evaluation prompt is rendered at both a neutral level and a senior
   ("L7") level so the seniority-gated blocks are scanned too.
   Reported as ``renderer:<assembly_fn_name>:<term>``.

2. ``keyword_collection`` — a module-level assignment whose value is a
   tuple/list/set literal (or a tuple/list/set/frozenset(...) call wrapping
   one) containing >= 6 string constants of <= 4 words each: the
   hard-coded-pattern-list shape (``_BUY_SIDE_PATTERNS`` in
   shared/strict_seniority.py is the canonical instance). Assignments whose
   target name ends in _KEYS/_FIELDS/_CODES/_DECISIONS/_MARKERS are skipped —
   wire-format/schema enums, not market vocabulary.

3. ``brief_text_gate`` — ``ast.Compare`` with ``ops=[ast.In]`` whose left
   operand is a string constant (or a loop/comprehension variable bound over a
   string-literal collection) and whose comparator's source contains one of
   "text" / "haystack" / "brief" / "prose" / "notes": the magic-word-gate
   shape (``_brief_targets_edge_case_opening`` in linkedin/strategy.py is the
   canonical instance).

Structural checks scan the same roots/exclusions as lint_vertical_vocab
(shared/ + linkedin/ production code; shared/role_strategy_profiles.py is
data-by-declaration and excluded; files under a tests directory or named
test_* are skipped). The docstring exemption is inherited by construction:
neither structural rule inspects raw string literals inside docstrings
(Assign/Compare nodes cannot live in one), and check 1 scans rendered prompt
text, where docstrings never appear.

RATCHET SEMANTICS — the exemption mechanism is the plain-text baseline at
tools/role_agnosticism_baseline.txt (NOT source markers): every current
finding must be listed there, and every listed entry must still fire. The
pytest wrapper (tests/test_role_agnosticism_ratchet.py) fails new findings
with "role-shaped structure belongs in the brief" and fails stale entries so
the baseline only shrinks.

Usage:
    python tools/lint_role_agnosticism.py                  # exit 1 on new findings
    python tools/lint_role_agnosticism.py --update-baseline
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tools" / "role_agnosticism_baseline.txt"
LEXICON_PATH = ROOT / "tools" / "vertical_vocab_lexicon.txt"
SCAN_ROOTS = ("shared", "linkedin", "github")
EXCLUDED_FILES = {
    ROOT / "shared" / "role_strategy_profiles.py",
}

# Comparator-source substrings that mark the right-hand side of an ``in`` test
# as brief/prose text — the haystack a magic-word gate greps.
_HAYSTACK_NAME_HINTS = ("text", "haystack", "brief", "prose", "notes")

# Target-name suffixes exempt from the keyword_collection rule: wire-format /
# schema enums (decision tokens, JSON keys, marker strings), not market
# vocabulary. Dunder targets (``__all__`` & co.) are module machinery and are
# skipped for the same reason.
_SCHEMA_ENUM_SUFFIXES = ("_KEYS", "_FIELDS", "_CODES", "_DECISIONS", "_MARKERS", "_COLUMNS", "_DIMENSIONS", "_KINDS", "_CLASS_NAMES")

# Strip string-literal contents from a comparator's source before hint
# matching: `field in {"intake_notes", ...}` must not match on "notes" INSIDE
# a literal — the rule targets the comparator's NAME (text/haystack/brief/...),
# not data it happens to hold.
_QUOTED_SEGMENT = re.compile(r"'[^']*'|\"[^\"]*\"")

_KEYWORD_COLLECTION_MIN_STRINGS = 6
_KEYWORD_COLLECTION_MAX_WORDS = 4

_RATCHET_MESSAGE = (
    "role-shaped structure belongs in the brief — "
    "see plans/sourcing-generality-hardening.md item 21"
)


# ---------------------------------------------------------------------------
# Lexicon (shared with tools/lint_vertical_vocab.py)
# ---------------------------------------------------------------------------


def _parse_lexicon_terms(path: Path) -> list[str]:
    """The lexicon's parse rules, mirrored from lint_vertical_vocab.load_lexicon
    so terms and compiled patterns stay index-aligned."""
    terms: list[str] = []
    for raw_line in path.read_text().splitlines():
        term = raw_line.strip()
        if not term or term.startswith("#"):
            continue
        terms.append(term)
    return terms


def load_lexicon_terms() -> list[tuple[str, re.Pattern[str]]]:
    """(term, compiled pattern) pairs. Reuses lint_vertical_vocab.load_lexicon
    when importable (the compiled patterns stay identical by construction);
    re-implements the loader otherwise."""
    terms = _parse_lexicon_terms(LEXICON_PATH)
    patterns: list[re.Pattern[str]] | None = None
    try:
        from lint_vertical_vocab import load_lexicon  # tools/ on sys.path
    except ImportError:
        try:
            from tools.lint_vertical_vocab import load_lexicon  # repo root on sys.path
        except ImportError:
            load_lexicon = None  # type: ignore[assignment]
    if load_lexicon is not None:
        candidate = load_lexicon()
        if len(candidate) == len(terms):
            patterns = candidate
    if patterns is None:
        patterns = [
            re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in terms
        ]
    return list(zip(terms, patterns))


# ---------------------------------------------------------------------------
# Check 1 — null-brief render residue
# ---------------------------------------------------------------------------


def _neutral_preflight_dict() -> dict:
    """A minimal VALID preflight dict (the shape of
    tests/test_orchestrator_preflight.py::_valid_preflight_dict) whose every
    string field is a neutral placeholder. Enum-constrained fields keep valid
    enum values ("moderate", tier "neutral") — those are wire format, not
    vocabulary. Nothing in this dict matches any lexicon term, so every
    lexicon hit in a prompt rendered from it is code-carried."""
    return {
        "role_title": "Neutral-Term-Title",
        "role_level": "neutral-term-level",
        "role_summary": "neutral-term-summary sentence one.",
        "hiring_company": "Neutral-Term-Hiring-Company",
        "employer_blacklist": ["Neutral-Term-Hiring-Company"],
        "capability_areas": [
            {
                "name": "Neutral-term-capability-area",
                "description": "neutral-term-description of the area.",
                "builder_signals": ["neutral-term-builder-signal"],
                "user_signals": ["neutral-term-user-signal"],
                "key_terms": ["neutral-term-key-term"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "neutral-term-builder-definition.",
            "user_definition": "neutral-term-user-definition.",
            "edge_case_guidance": "neutral-term-edge-case-guidance.",
        },
        "non_fit_patterns": [
            {
                "label": "Neutral-term-non-fit",
                "description": "neutral-term-non-fit-description.",
                "why_not": "neutral-term-non-fit-reason.",
                "examples": ["neutral-term-non-fit-example"],
            }
        ],
        "employer_signal_rules": [
            {
                "tier": "neutral",
                "employer_patterns": ["Neutral-Term-Employer"],
                "evidence_required": "neutral-term-evidence-required.",
                "save_on_employer_alone": False,
            }
        ],
        "minimum_years_experience": 5,
        "minimum_bar_description": "neutral-term-minimum-bar-description.",
        "facial_calibration": {
            "expected_yes_rate_low": 0.15,
            "expected_yes_rate_high": 0.35,
            "yes_rate_rationale": "neutral-term-yes-rate-rationale.",
            "fast_exit_patterns": ["neutral-term-fast-exit-pattern"],
            "trajectory_yes_patterns": ["neutral-term-trajectory-yes-pattern"],
            "trajectory_ambiguous_patterns": ["neutral-term-trajectory-ambiguous-pattern"],
            "trajectory_no_patterns": ["neutral-term-trajectory-no-pattern"],
        },
        "domain_lane_hints": [
            {"lane": "neutral_lane_one", "patterns": ["neutral-term-lane-pattern"]}
        ],
        "market_density": "moderate",
    }


def _rendered_prompts() -> dict[str, list[str]]:
    """Render the four prompt assemblies against the neutral brief.

    Returns {assembly_fn_name: [rendered variants]}. Variants pin the
    environment-dependent branches (facial ambiguity posture) and the
    brief-level branches worth scanning (senior vs IC full-eval blocks) so the
    finding set is deterministic and covers the fallback prose on both sides
    of each gate.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from shared.brief_loader import _load_v2_brief
    from shared.preflight_v2 import preflight_to_brief_json
    from linkedin.judgment_templates import (
        assemble_facial_batch_system,
        assemble_facial_system,
        assemble_full_evaluation_system,
    )
    from linkedin.strategy import _build_strategy_system

    compat_brief = _load_v2_brief(preflight_to_brief_json(_neutral_preflight_dict()))
    new_brief = compat_brief._new_brief
    if new_brief is None:  # pragma: no cover — _load_v2_brief always sets it
        raise RuntimeError("neutral preflight dict did not load as a V2 brief")

    rendered: dict[str, list[str]] = {
        "assemble_facial_system": [],
        "assemble_facial_batch_system": [],
        "assemble_full_evaluation_system": [],
        "_build_strategy_system": [],
    }

    # Facial triage: render BOTH postures so the selected template never
    # depends on shared.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED.
    for posture in ("binary", "ternary"):
        new_brief.facial_ambiguity_posture = posture
        rendered["assemble_facial_system"].append(assemble_facial_system(new_brief))
        rendered["assemble_facial_batch_system"].append(
            assemble_facial_batch_system(new_brief)
        )
    new_brief.facial_ambiguity_posture = ""

    # Full evaluation: neutral level exercises the IC fallback blocks; "L7"
    # (pure seniority marker, zero vertical vocabulary) trips is_senior_role
    # so the executive calibration / builder / decision-matrix branches are
    # scanned as well.
    rendered["assemble_full_evaluation_system"].append(
        assemble_full_evaluation_system(new_brief)
    )
    original_level = new_brief.role_level
    new_brief.role_level = "L7"
    rendered["assemble_full_evaluation_system"].append(
        assemble_full_evaluation_system(new_brief)
    )
    new_brief.role_level = original_level

    rendered["_build_strategy_system"].append(
        _build_strategy_system(compat_brief, has_kit=False)
    )
    return rendered


def scan_render_residue() -> list[str]:
    """Lexicon terms present in prompts rendered from the neutral brief.

    The brief carries no lexicon vocabulary, so every hit is code-carried
    (template text, fallback prose, defaults). Reported once per
    (assembly function, term): ``renderer:<assembly_fn_name>:<term>``.
    """
    lexicon = load_lexicon_terms()
    # Self-check the check's own premise: the neutral brief must carry ZERO
    # lexicon terms, or hits would be misattributed to code. Fails loud so an
    # edit to the placeholders can never silently corrupt attribution.
    import json

    neutral_json = json.dumps(_neutral_preflight_dict())
    tainted = [term for term, pattern in lexicon if pattern.search(neutral_json)]
    if tainted:
        raise RuntimeError(
            "neutral preflight dict carries lexicon terms — render-residue "
            f"attribution would be corrupted: {tainted}"
        )
    findings: list[str] = []
    for fn_name, texts in _rendered_prompts().items():
        for term, pattern in lexicon:
            if any(pattern.search(text) for text in texts):
                findings.append(f"renderer:{fn_name}:{term}")
    return sorted(set(findings))


# ---------------------------------------------------------------------------
# Check 2 — AST structural rules
# ---------------------------------------------------------------------------


def _production_files() -> list[Path]:
    files: list[Path] = []
    for scan_root in SCAN_ROOTS:
        for path in sorted((ROOT / scan_root).rglob("*.py")):
            if path in EXCLUDED_FILES:
                continue
            if "tests" in path.parts or path.name.startswith("test_"):
                continue
            files.append(path)
    return files


def _string_literal_elements(node: ast.AST) -> tuple[list[str], bool]:
    """(string constants inside a collection literal, literal-is-pure-strings).

    Accepts tuple/list/set literals directly, and set/frozenset/tuple/list(...)
    calls wrapping one. Returns ([], False) for any other shape.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set", "tuple", "list"}
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [], False
    strings = [
        elt.value
        for elt in node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    ]
    pure = bool(node.elts) and len(strings) == len(node.elts)
    return strings, pure


def _assignment_name_and_value(stmt: ast.stmt) -> tuple[str, ast.AST] | None:
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            return stmt.targets[0].id, stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        if isinstance(stmt.target, ast.Name) and stmt.value is not None:
            return stmt.target.id, stmt.value
    return None


def _scan_keyword_collections(tree: ast.Module, rel_path: str) -> list[str]:
    findings: list[str] = []
    for stmt in tree.body:
        named = _assignment_name_and_value(stmt)
        if named is None:
            continue
        name, value = named
        if name.endswith(_SCHEMA_ENUM_SUFFIXES):
            continue
        if name.startswith("__") and name.endswith("__"):
            continue  # __all__ and friends — module machinery, not vocabulary
        strings, _pure = _string_literal_elements(value)
        short_strings = [
            s
            for s in strings
            if s.strip() and len(s.split()) <= _KEYWORD_COLLECTION_MAX_WORDS
        ]
        if len(short_strings) >= _KEYWORD_COLLECTION_MIN_STRINGS:
            findings.append(f"{rel_path}:{stmt.lineno}:keyword_collection:{name}")
    return findings


def _comparator_haystack_source(comparator: ast.AST) -> str | None:
    """The comparator's source when it NAMES brief/prose text; None otherwise.

    Hint matching runs on the source with string-literal contents stripped, so
    only identifiers (``text``, ``haystack``, ``column_haystack``, ``s.notes``,
    ``_brief_text(payload)``) can match — never data inside a literal.
    """
    try:
        segment = ast.unparse(comparator)
    except Exception:  # pragma: no cover — unparse handles all valid nodes
        return None
    name_source = _QUOTED_SEGMENT.sub("", segment).lower()
    if any(hint in name_source for hint in _HAYSTACK_NAME_HINTS):
        return segment
    return None


def _string_collection_names(tree: ast.Module) -> set[str]:
    """Names (any scope) assigned a pure string-literal collection."""
    names: set[str] = set()
    for node in ast.walk(tree):
        named = _assignment_name_and_value(node) if isinstance(node, ast.stmt) else None
        if named is None:
            continue
        name, value = named
        _strings, pure = _string_literal_elements(value)
        if pure:
            names.add(name)
    return names


def _is_string_collection_expr(node: ast.AST, known_names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in known_names
    _strings, pure = _string_literal_elements(node)
    return pure


def _gate_detail(left_source: str, comparator_source: str) -> str:
    return f"{left_source[:40]} in {comparator_source[:40]}"


def _scan_brief_text_gates(tree: ast.Module, rel_path: str) -> list[str]:
    findings: set[str] = set()

    # Pass 1 — literal left operand: `"magic word" in text`.
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
        ):
            continue
        if not (isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)):
            continue
        if node.left.value in {"/", "\\", "\n"}:
            continue
        comparator = node.comparators[0]
        try:
            comparator_unparsed = ast.unparse(comparator)
        except Exception:
            comparator_unparsed = ""
        if ".text(" in comparator_unparsed:
            continue
        if (
            isinstance(comparator, ast.Call)
            and isinstance(comparator.func, ast.Attribute)
            and comparator.func.attr == "text"
        ):
            continue
        comparator_source = _comparator_haystack_source(comparator)
        if comparator_source is None:
            continue
        detail = _gate_detail(repr(node.left.value), comparator_source)
        findings.add(f"{rel_path}:{node.lineno}:brief_text_gate:{detail}")

    # Pass 2 — bound-name left operand: `any(t in haystack for t in TRIGGERS)`
    # (or a for-loop) where the bound name iterates a string-literal
    # collection. Same gate shape, one indirection deeper.
    known_names = _string_collection_names(tree)
    for scope in ast.walk(tree):
        if isinstance(scope, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            generators = scope.generators
        elif isinstance(scope, ast.For):
            generators = [scope]
        else:
            continue
        bound: set[str] = set()
        for gen in generators:
            if isinstance(gen.target, ast.Name) and _is_string_collection_expr(
                gen.iter, known_names
            ):
                bound.add(gen.target.id)
        if not bound:
            continue
        for node in ast.walk(scope):
            if not (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.In)
            ):
                continue
            if not (isinstance(node.left, ast.Name) and node.left.id in bound):
                continue
            comparator_source = _comparator_haystack_source(node.comparators[0])
            if comparator_source is None:
                continue
            detail = _gate_detail(node.left.id, comparator_source)
            findings.add(f"{rel_path}:{node.lineno}:brief_text_gate:{detail}")

    return sorted(findings)


def scan_structural() -> list[str]:
    findings: list[str] = []
    for path in _production_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        rel_path = str(path.relative_to(ROOT))
        findings.extend(_scan_keyword_collections(tree, rel_path))
        findings.extend(_scan_brief_text_gates(tree, rel_path))
    return sorted(set(findings))


# ---------------------------------------------------------------------------
# Ratchet plumbing
# ---------------------------------------------------------------------------


def scan() -> list[str]:
    """All current findings across the three checks, sorted and unique."""
    return sorted(set(scan_render_residue()) | set(scan_structural()))


def load_baseline() -> list[str]:
    if not BASELINE_PATH.exists():
        return []
    entries: list[str] = []
    for raw_line in BASELINE_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def write_baseline(findings: list[str]) -> None:
    renderer = sorted(f for f in findings if f.startswith("renderer:"))
    collections = sorted(f for f in findings if ":keyword_collection:" in f)
    gates = sorted(f for f in findings if ":brief_text_gate:" in f)
    lines = [
        "# Role-agnosticism ratchet baseline — tools/lint_role_agnosticism.py",
        "# (plans/sourcing-generality-hardening.md item 21).",
        "#",
        "# Grandfathered role-shaped structure. SHRINK ONLY: fixing a finding",
        "# means deleting its line here (the pytest wrapper fails stale",
        "# entries); a NEW finding fails CI — role-shaped structure belongs",
        "# in the brief. Regenerate after a legitimate shrink with:",
        "#   python tools/lint_role_agnosticism.py --update-baseline",
        "",
        "# --- renderer allowlist (null-brief render residue) ---",
        "# Lexicon terms that prompt assembly injects from CODE (templates,",
        "# fallbacks, defaults) even when the brief carries none.",
        *(renderer or ["# (none — the assembled prompts render lexicon-clean)"]),
        "",
        "# --- keyword_collection (hard-coded pattern lists) ---",
        *collections,
        "",
        "# --- brief_text_gate (magic-word gates over brief text) ---",
        *gates,
        "",
    ]
    BASELINE_PATH.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite tools/role_agnosticism_baseline.txt from the current tree",
    )
    args = parser.parse_args(argv)

    findings = scan()
    if args.update_baseline:
        write_baseline(findings)
        print(
            f"[role-agnosticism] baseline rewritten: {len(findings)} finding(s) "
            f"→ {BASELINE_PATH.relative_to(ROOT)}"
        )
        return 0

    baseline = set(load_baseline())
    current = set(findings)
    new = sorted(current - baseline)
    stale = sorted(baseline - current)
    grandfathered = len(current & baseline)

    if grandfathered:
        print(f"[role-agnosticism] {grandfathered} baselined finding(s).")
    if stale:
        noun = "entry" if len(stale) == 1 else "entries"
        print(
            f"[role-agnosticism] {len(stale)} stale baseline {noun} no longer "
            "fire — shrink the baseline (delete the lines or --update-baseline):"
        )
        for record in stale:
            print(f"  {record}")
    if new:
        print(f"[role-agnosticism] FAIL — {_RATCHET_MESSAGE}. New findings:")
        for record in new:
            print(f"  {record}")
        return 1
    print("[role-agnosticism] clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
