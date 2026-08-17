"""P5 dead-lever guardrails (Wave 2 slice 7).

The generalized "boolean_lint had zero readers for months" detector: every
detector must have a consumer that can act. Concretely, every public
``lint_*`` / ``attach_*`` symbol in linkedin/boolean_compiler.py must be
CALLED — as an actual ast.Call, not a comment/docstring mention — from
production code (linkedin/, shared/, market_intelligence/), either directly
or through a chain of same-module callers that terminates in a production
call site (e.g. lint_constraint_compile ← attach_constraint_lint_to_plan ←
attach_boolean_lint_to_plan ← linkedin/strategy.py).

Raw text search was rejected here deliberately: the test-honesty lens proved
a severed call site stayed green because a docstring elsewhere mentioned the
symbol name. AST call resolution is the honest instrument.

A symbol failing here means one of two things, both actionable:
- it was wired once and a refactor severed the consumer (regression), or
- it was built speculatively and never wired (delete it — unfireable
  protection reads as protection and provides none).
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "linkedin" / "boolean_compiler.py"

_PRODUCTION_ROOTS = ("linkedin", "shared", "market_intelligence")


def _called_names(tree: ast.AST) -> set[str]:
    """Names invoked as calls: plain ``name(...)`` and ``module.name(...)``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def _module_call_graph(tree: ast.Module) -> dict[str, set[str]]:
    """Top-level function -> names it calls, within boolean_compiler itself."""
    graph: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            graph[node.name] = _called_names(node)
    return graph


def test_every_public_lint_symbol_has_a_production_consumer():
    module_tree = ast.parse(MODULE.read_text())
    public = {
        node.name
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (node.name.startswith("lint_") or node.name.startswith("attach_"))
        and not node.name.startswith("_")
    }
    assert public, "expected lint_/attach_ symbols in boolean_compiler.py"

    outside_calls: set[str] = set()
    for source_root in _PRODUCTION_ROOTS:
        for path in (ROOT / source_root).rglob("*.py"):
            if path == MODULE:
                continue
            try:
                outside_calls |= _called_names(ast.parse(path.read_text()))
            except SyntaxError:  # pragma: no cover — broken file fails elsewhere
                continue

    # Transitive closure over boolean_compiler's own call graph: a symbol is
    # consumed if a production file calls it, or calls a same-module function
    # that (transitively) calls it.
    graph = _module_call_graph(module_tree)
    consumed: set[str] = set()
    frontier = [name for name in graph if name in outside_calls]
    seen: set[str] = set(frontier)
    while frontier:
        caller = frontier.pop()
        consumed.add(caller)
        for callee in graph.get(caller, ()):
            if callee in graph and callee not in seen:
                seen.add(callee)
                frontier.append(callee)

    missing = sorted(
        name for name in public if name not in outside_calls and name not in consumed
    )
    assert not missing, (
        f"dead levers — public lint symbols with no production call site "
        f"(direct or via a consumed same-module chain): {missing}. "
        f"Wire each one to a consumer that can act, or delete it."
    )
