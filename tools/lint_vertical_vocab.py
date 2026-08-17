#!/usr/bin/env python3
"""Vertical-vocabulary ratchet (Wave 2 slice 9, plans/sourcing-rigor-hardening.md P1).

Rule: deterministic code may CONSUME brief-supplied vocabulary; it may never
CARRY its own vertical vocabulary. This linter scans string literals in
shared/ and linkedin/ production code for lexicon terms
(tools/vertical_vocab_lexicon.txt) and fails on any hit that is not
explicitly grandfathered with a `# VERTICAL-VOCAB(<audit-ref>)` marker on the
same physical line.

The allowlist may SHRINK, never grow: the pytest wrapper
(tests/test_vertical_vocab_ratchet.py) pins the marked-line baseline. New
vertical vocabulary in deterministic code fails CI with the message
"vocabulary belongs in the brief".

Exclusions (data-by-declaration, not deterministic code):
- shared/role_strategy_profiles.py — role-class profiles are vertical
  templates by definition; their vocabulary is the product content.
- Docstrings — prose ABOUT vocabulary is not vocabulary a matcher consumes.

Usage: python tools/lint_vertical_vocab.py  (exit 1 on unmarked hits)
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEXICON_PATH = ROOT / "tools" / "vertical_vocab_lexicon.txt"
SCAN_ROOTS = ("shared", "linkedin", "github")
EXCLUDED_FILES = {
    ROOT / "shared" / "role_strategy_profiles.py",
}
MARKER = "VERTICAL-VOCAB"


def load_lexicon() -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for raw_line in LEXICON_PATH.read_text().splitlines():
        term = raw_line.strip()
        if not term or term.startswith("#"):
            continue
        patterns.append(re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE))
    return patterns


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Line ranges of docstrings — prose, exempt from the literal scan."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                doc = body[0].value
                end = doc.end_lineno or doc.lineno
                lines.update(range(doc.lineno, end + 1))
    return lines


def scan() -> tuple[list[str], list[str]]:
    """Returns (violations, marked_lines) as "path:lineno:term" strings."""
    lexicon = load_lexicon()
    violations: list[str] = []
    marked: list[str] = []
    for scan_root in SCAN_ROOTS:
        for path in sorted((ROOT / scan_root).rglob("*.py")):
            if path in EXCLUDED_FILES:
                continue
            source = path.read_text()
            source_lines = source.splitlines()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            doc_lines = _docstring_lines(tree)
            # Statement spans, innermost-last, so a literal can be exempted
            # by a marker on its enclosing STATEMENT's first line or the line
            # above it — an inline comment inside a multi-line template would
            # otherwise corrupt the string content.
            statement_spans = [
                (stmt.lineno, stmt.end_lineno or stmt.lineno)
                for stmt in ast.walk(tree)
                if isinstance(stmt, ast.stmt)
            ]
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                    continue
                if node.lineno in doc_lines:
                    continue
                hits = [
                    pattern.pattern
                    for pattern in lexicon
                    if pattern.search(node.value)
                ]
                if not hits:
                    continue
                candidate_lines = {node.lineno, node.lineno - 1}
                enclosing = [
                    span
                    for span in statement_spans
                    if span[0] <= node.lineno <= span[1]
                ]
                if enclosing:
                    stmt_start = max(span[0] for span in enclosing)
                    candidate_lines.update({stmt_start, stmt_start - 1})
                line_text = "".join(
                    source_lines[lineno - 1]
                    for lineno in candidate_lines
                    if 0 < lineno <= len(source_lines)
                )
                record = (
                    f"{path.relative_to(ROOT)}:{node.lineno}:"
                    f"{','.join(sorted(set(hits)))}"
                )
                if MARKER in line_text:
                    marked.append(record)
                else:
                    violations.append(record)
    return violations, sorted(set(marked))


def main() -> int:
    violations, marked = scan()
    if marked:
        print(f"[vertical-vocab] {len(set(marked))} grandfathered marked line(s).")
    if violations:
        print(
            "[vertical-vocab] FAIL — vocabulary belongs in the brief. "
            "Unmarked vertical-vocabulary literals in deterministic code:"
        )
        for violation in sorted(set(violations)):
            print(f"  {violation}")
        return 1
    print("[vertical-vocab] clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
