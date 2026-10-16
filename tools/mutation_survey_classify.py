#!/usr/bin/env python3
"""Sort the tests the mutation survey never saw fail into suggested classes.

Usage, after ``tools/mutation_survey.py report``::

    python tools/mutation_survey_classify.py --state-dir DIR            # classify
    python tools/mutation_survey_classify.py --state-dir DIR --draws 20 # plus draws

A suggestion is a lead, not a verdict. It combines what the survey measured
(verdict, kill kinds, production files the test opened, out-of-scope code it
ran) with static facts about the test function (what it asserts on, whether it
reads source text, inspects imports, shells out, or is skipped). The fixed-seed
draws exist so a person can read twenty tests per class and put a measured
precision beside each count before anything is acted on.

Suggested classes. A, B, C and E are the audit's; D (a test that defends a
defect) is never suggested, because such a test kills mutants like any other
lock and only a reader can find it. The rest say why a test is outside A-E:

* ``A`` structural by design: reads source text, inspects imports, pins a
  declared value or a schema shape.
* ``B`` tautological: runs no production body and touches no production
  value, or asserts a literal it assigned itself.
* ``C`` shallow: asserts on a proxy or on nothing, or survived while running a
  function that no test anywhere notices emptied. The second half is measured,
  not guessed (the survey's own function verdicts), and it is a lead: the
  function may be setup the test never claimed.
* ``E`` unreachable: skipped or expected-fail on unmutated code.
* ``N`` no conclusion from these operators, and harmless: the test survived,
  but every function it runs is locked by some other test. Mostly
  negative cases (``is None``, falsy, empty, not called) and invariance checks
  (idempotent, deterministic, input not mutated), which an emptied body also
  satisfies. Legitimate beside the positive sibling the survey found.
* ``K`` crash-shadowed: fails under mutation, but only by crashing. The emptied
  body returns ``None`` and the run breaks before an assertion is reached, so
  these operators cannot say whether the assertions would have noticed.
* ``O`` subject outside the surveyed scope.
* ``S`` subject runs in a subprocess, where coverage contexts cannot follow.
* ``X`` no conclusion: fails unmutated in the hermetic copy, or unstable.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import mutation_survey as ms  # noqa: E402

DRAW_SEED = 20260916
_WITNESS_MAX_COVERING = 20
NOT_LOCKS = ("survivor", "crash_only", "no_body", "no_control", "unstable")

_STRUCTURAL_NAME = re.compile(
    r"does_not_import|doesnt_import|never_imports|import_narrow|no_\w*imports?\b|"
    r"imports_no|import_boundary|does_not_pull|importable|compiles_from_source|"
    r"ratchet|provenance|source_shape|hygiene",
    re.IGNORECASE,
)
_SOURCE_READ_CALLS = {"read_text", "read_bytes", "getsource", "getsourcelines", "getsourcefile"}
_SOURCE_SUFFIXES = (".py", ".sh", ".md", ".toml", "Makefile")
_DATA_SUFFIXES = (".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".html")
_IMPORT_PROBES = {"modules", "import_module", "find_spec", "iter_modules", "walk_packages"}
_SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call", "call", "create_subprocess_exec"}
_CAPTURE_FIXTURES = {"caplog", "capsys", "capfd", "capsysbinary", "capfdbinary"}
_MOCK_DATA = {
    "call_args",
    "call_args_list",
    "call_count",
    "called",
    "mock_calls",
    "method_calls",
    "await_args",
    "await_args_list",
    "await_count",
    "awaited",
}
_NEGATIVE_MOCK_ASSERTS = {"assert_not_called", "assert_not_awaited"}


@dataclasses.dataclass
class TestFacts:
    found: bool = False
    lineno: int = 0
    assertions: int = 0
    negative_only: bool = False
    proxy_only: bool = False
    tautology: bool = False
    pin_only: bool = False
    source_read: bool = False
    import_probe: bool = False
    subprocess_use: bool = False
    structural_name: bool = False
    uses_production_names: bool = False
    uses_test_support_names: bool = False
    uses_foreign_names: bool = False
    subject_excluded: bool = False
    module_imports_excluded: bool = False
    data_read: bool = False
    skip_marker: str = ""


# ---------------------------------------------------------------------------
# Static facts
# ---------------------------------------------------------------------------


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _attrs(node: ast.AST) -> set[str]:
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def _names_file(node: ast.AST, suffixes: tuple[str, ...]) -> bool:
    return any(
        isinstance(const, ast.Constant)
        and isinstance(const.value, str)
        and const.value.endswith(suffixes)
        for const in ast.walk(node)
    )


def _names_source_file(node: ast.AST) -> bool:
    return _names_file(node, _SOURCE_SUFFIXES)


@dataclasses.dataclass(frozen=True)
class Scope:
    """Which project modules the survey mutates, and which it only watches.

    ``roots`` alone (no file lists) treats every root as in scope, which is all
    a unit test of the static facts needs.
    """

    roots: frozenset[str]
    scope_files: frozenset[str] = frozenset()
    project_files: frozenset[str] = frozenset()

    def place(self, module: str, name: str = "") -> str:
        """``"in"`` scope, ``"out"`` of scope but project code, or ``""`` for neither."""
        if module.split(".")[0] not in self.roots:
            return ""
        if not self.project_files:
            return "in"
        base = module.replace(".", "/")
        candidates = [f"{base}.py", f"{base}/__init__.py"]
        if name:
            candidates = [f"{base}/{name}.py", f"{base}/{name}/__init__.py", *candidates]
        for candidate in candidates:
            if candidate in self.project_files:
                return "in" if candidate in self.scope_files else "out"
        return "out"


def _as_scope(production_roots: "set[str] | Scope") -> Scope:
    if isinstance(production_roots, Scope):
        return production_roots
    return Scope(roots=frozenset(production_roots))


def _local_imports(func: ast.AST, production_roots: "set[str] | Scope", place: str = "in") -> set[str]:
    """Project names a test imports inside its own body, in or out of scope."""
    scope = _as_scope(production_roots)
    bound: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if scope.place(node.module or "", alias.name) == place:
                    bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if scope.place(alias.name) == place:
                    bound.add((alias.asname or alias.name).split(".")[0])
    return bound


def _is_source_read(node: ast.AST, source_paths: frozenset[str] = frozenset()) -> bool:
    """A call that reads text, in an expression that also names a source file.

    ``source_paths`` are module-level names bound to a source file's path, so
    ``SCRIPT_PATH.read_text()`` counts even though the suffix is written elsewhere.
    """
    reads = any(
        isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Attribute) and call.func.attr in _SOURCE_READ_CALLS)
            or (isinstance(call.func, ast.Name) and call.func.id == "open")
        )
        for call in ast.walk(node)
    )
    if not reads:
        return False
    if "__file__" in _attrs(node) or "getsource" in _attrs(node) | _names(node):
        return True
    return _names_source_file(node) or bool(_names(node) & source_paths)


def _module_context(
    tree: ast.Module,
    production_roots: set[str],
    excluded_modules: frozenset[str] = frozenset(),
    test_support_modules: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Names a test module binds to production code, test support, source text and helpers.

    ``test_support_modules`` are the helper modules that live beside the tests:
    with ``tests/`` on the path they are imported by bare name.
    """
    scope = _as_scope(production_roots)
    production: set[str] = set()
    foreign: set[str] = set()
    test_support: set[str] = set()
    excluded: set[str] = set()
    source_names: set[str] = set()
    source_paths: set[str] = set()
    data_paths: set[str] = set()
    helpers: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            bound = {alias.asname or alias.name for alias in node.names}
            qualified = {f"{module}.{alias.name}" for alias in node.names}
            if module in excluded_modules or qualified & excluded_modules:
                excluded |= bound
            elif module.split(".")[0] in scope.roots:
                for alias in node.names:
                    target = production if scope.place(module, alias.name) == "in" else foreign
                    target.add(alias.asname or alias.name)
            elif module.split(".")[0] == "tests" or module.split(".")[0] in test_support_modules:
                test_support |= bound
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in excluded_modules:
                    excluded.add((alias.asname or alias.name).split(".")[0])
                elif alias.name.split(".")[0] in scope.roots:
                    target = production if scope.place(alias.name) == "in" else foreign
                    target.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound = {name for target in targets for name in _names(target)}
            if _is_source_read(node.value, frozenset(source_paths)):
                source_names |= bound
            elif _names_source_file(node.value):
                # A repo root derived from __file__ is not a source path: tests
                # read fixtures through it all the time.
                source_paths |= bound
            elif _names_file(node.value, _DATA_SUFFIXES) or _names(node.value) & data_paths:
                # Also a name derived from a data path: ``SEED = A if A.is_file() else B``.
                data_paths |= bound
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            helpers[node.name] = node
    paths = frozenset(source_paths)
    # A helper that reads source makes its callers source readers.
    source_helpers = {name for name, node in helpers.items() if _is_source_read(node, paths)}
    return {
        "production": production,
        "foreign": foreign,
        "test_support": test_support,
        "excluded": excluded,
        "source_names": source_names,
        "source_paths": paths,
        "data_paths": frozenset(data_paths),
        "helpers": helpers,
        "source_helpers": source_helpers,
    }


def _find_test(tree: ast.Module, nodeid: str) -> ast.AST | None:
    parts = [part.split("[", 1)[0] for part in nodeid.split("::")[1:]]
    scope: ast.AST = tree
    for depth, name in enumerate(parts):
        wanted = (ast.ClassDef,) if depth < len(parts) - 1 else (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        )
        matches = [
            node
            for node in ast.iter_child_nodes(scope)
            if isinstance(node, wanted) and node.name == name
        ]
        if not matches:
            return None
        scope = matches[-1]
    return scope if parts else None


def _is_negative(test: ast.expr) -> bool:
    """An assertion that ``None`` (or nothing happening) also satisfies."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return True
    if isinstance(test, ast.BoolOp):
        return all(_is_negative(value) for value in test.values)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, right = test.ops[0], test.comparators[0]
        if isinstance(right, ast.Constant) and isinstance(op, (ast.Is, ast.Eq)):
            return right.value in (None, False, 0, "", b"")
        if isinstance(op, ast.Eq) and isinstance(right, (ast.List, ast.Dict, ast.Tuple, ast.Set)):
            return not (getattr(right, "elts", None) or getattr(right, "keys", None))
        return isinstance(op, ast.NotIn)
    return False


def _statement_facts(func: ast.AST, context: dict[str, Any]) -> dict[str, Any]:
    """Assertion points in a test body, with one level of same-module helpers folded in."""
    bodies = [func]
    for call in ast.walk(func):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
            helper = context["helpers"].get(call.func.id)
            if helper is not None and helper is not func:
                bodies.append(helper)

    captured = set(_CAPTURE_FIXTURES)
    literals: dict[str, Any] = {}
    points: list[dict[str, bool]] = []
    tautology = False
    for body in bodies:
        for node in ast.walk(body):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if _names(node.value) & captured:
                    captured |= _names(target)
                key = ast.unparse(target)
                if isinstance(node.value, ast.Constant):
                    literals[key] = node.value.value
                else:
                    literals.pop(key, None)
            elif isinstance(node, ast.Assert):
                test = node.test
                proxy = bool(_names(test) & captured) or bool(_attrs(test) & _MOCK_DATA)
                points.append({"negative": _is_negative(test), "proxy": proxy})
                if isinstance(test, ast.Constant):
                    tautology = True
                if (
                    isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], (ast.Eq, ast.Is))
                    and isinstance(test.comparators[0], ast.Constant)
                ):
                    key = ast.unparse(test.left)
                    if key in literals and literals[key] == test.comparators[0].value:
                        tautology = True
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                on_self = isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                if attr.startswith("assert_"):
                    # Mock call assertions: assert_called_once_with, assert_not_awaited, ...
                    points.append({"negative": attr in _NEGATIVE_MOCK_ASSERTS, "proxy": True})
                elif attr.startswith("assert") and on_self:
                    points.append({"negative": False, "proxy": False})
                elif attr in {"raises", "warns", "fail", "deprecated_call"} and (
                    isinstance(node.func.value, ast.Name) and node.func.value.id == "pytest"
                ):
                    points.append({"negative": False, "proxy": False})
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.lstrip("_").startswith("assert")
                and node.func.id not in context["helpers"]
            ):
                # An imported assertion helper; local ones are folded in above.
                points.append({"negative": False, "proxy": False})
    return {"points": points, "tautology": tautology}


def _skip_marker(func: ast.AST) -> str:
    for decorator in getattr(func, "decorator_list", []):
        text = ast.unparse(decorator)
        for marker in ("skipif", "skip", "xfail"):
            if f"mark.{marker}" in text:
                return marker
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"skip", "importorskip", "xfail"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
        ):
            return f"runtime_{node.func.attr}"
    return ""


def static_facts(
    tree: ast.Module,
    nodeid: str,
    production_roots: set[str],
    excluded_modules: frozenset[str] = frozenset(),
    test_support_modules: frozenset[str] = frozenset(),
) -> TestFacts:
    func = _find_test(tree, nodeid)
    if func is None:
        return TestFacts()
    context = _module_context(tree, production_roots, excluded_modules, test_support_modules)
    statements = _statement_facts(func, context)
    points = statements["points"]
    names, attrs = _names(func), _attrs(func)
    production = context["production"] | _local_imports(func, production_roots)
    foreign = context["foreign"] | _local_imports(func, production_roots, "out")
    # A fixture is named as a parameter, not called: fold those helpers in too.
    arguments = getattr(func, "args", None)
    parameters = {arg.arg for arg in arguments.args} if arguments is not None else set()
    helper_names = (names | parameters) & set(context["helpers"])
    reads_data = any(
        isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Attribute) and call.func.attr in {"read_text", "read_bytes", "load", "loads"})
            or (isinstance(call.func, ast.Name) and call.func.id == "open")
        )
        for body in [func, *(context["helpers"][n] for n in names & set(context["helpers"]))]
        for call in ast.walk(body)
    )
    names_data = bool(names & context["data_paths"]) or any(
        _names_file(body, _DATA_SUFFIXES) or bool(_names(body) & context["data_paths"])
        for body in [func, *(context["helpers"][n] for n in names & set(context["helpers"]))]
    )
    asserts = [node for node in ast.walk(func) if isinstance(node, ast.Assert)]
    pin_only = bool(asserts) and all(
        not any(isinstance(n, ast.Call) for n in ast.walk(node.test))
        and bool(_names(node.test) & production)
        for node in asserts
    )
    subprocess_use = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in _SUBPROCESS_CALLS
        and "subprocess" in ast.unparse(call.func)
        for body in [func, *(context["helpers"][n] for n in helper_names)]
        for call in ast.walk(body)
    )
    return TestFacts(
        found=True,
        lineno=func.lineno,
        assertions=len(points),
        negative_only=bool(points) and all(point["negative"] for point in points),
        proxy_only=bool(points) and all(point["proxy"] for point in points),
        tautology=statements["tautology"],
        pin_only=pin_only,
        source_read=_is_source_read(func, context["source_paths"])
        or bool(names & context["source_names"])
        or bool(names & context["source_helpers"]),
        import_probe=bool(attrs & _IMPORT_PROBES) and ("sys" in names or "importlib" in names),
        subprocess_use=subprocess_use,
        structural_name=bool(_STRUCTURAL_NAME.search(nodeid.split("::", 1)[1])),
        uses_production_names=bool(names & production),
        data_read=reads_data and names_data,
        uses_test_support_names=bool(names & context["test_support"]),
        uses_foreign_names=bool(names & foreign),
        subject_excluded=bool(names & context["excluded"]),
        module_imports_excluded=bool(context["excluded"]),
        skip_marker=_skip_marker(func),
    )


# ---------------------------------------------------------------------------
# Suggestion
# ---------------------------------------------------------------------------


def suggest(facts: TestFacts, row: dict[str, Any], covered_in_scope: int) -> tuple[str, str]:
    verdict = row["verdict"]
    if verdict == "no_control":
        if row["control"] in {"skipped", "xfailed", "xpassed"}:
            return "E", f"{row['control']} on unmutated code: {row.get('reason', '')[:120]}"
        return "X", f"{row['control']} on unmutated code in the hermetic copy"
    if verdict == "unstable":
        return "X", "keeps failing a behavior-preserving edit of the code under it"
    if facts.subject_excluded:
        return "O", "its subject is excluded from the survey's scope"
    if facts.uses_foreign_names and not facts.uses_production_names:
        return "O", "its subject is project code outside the surveyed scope"
    if facts.subprocess_use:
        # Before the structural rule: a test that runs a script and also reads
        # its text is a behavior test the survey cannot follow, not a shape test.
        return "S", "drives its subject through a subprocess"
    # A test that fails under mutation, even by crashing, runs behavior: opening
    # a production file along the way does not make it a shape test.
    structural = verdict != "crash_only" and (
        facts.source_read or bool(row.get("reads")) or facts.import_probe or facts.structural_name
    )
    if structural:
        how = (
            "reads production source text"
            if facts.source_read or row.get("reads")
            else "inspects imports"
            if facts.import_probe
            else "named as a structural guard"
        )
        return "A", how
    if verdict == "no_body":
        if row.get("observed_functions", 0) > 0:
            return "O", f"runs {row['observed_functions']} out-of-scope function bodies, none in scope"
        if facts.uses_test_support_names and not facts.uses_production_names:
            return "O", "its subject is test-support code under tests/"
        if facts.subprocess_use:
            return "S", "drives its subject through a subprocess"
        if facts.pin_only:
            return "A", "pins declared production values without calling anything"
        if facts.data_read:
            return "A", "pins the content of a repo data file"
        if facts.uses_production_names and facts.assertions:
            return "A", "asserts on declared shape: runs no production function body"
        if facts.module_imports_excluded:
            # Fixture-driven tests of a subject the survey leaves out of scope.
            return "O", "its module tests a subject excluded from the survey's scope"
        return "B", "runs no production body and asserts on no production value"
    if facts.tautology:
        return "B", "asserts a literal the test assigned itself"
    if row.get("observed_functions", 0) > covered_in_scope:
        return "O", "most of the code it runs is outside the surveyed scope"
    if facts.subprocess_use:
        return "S", "drives its subject through a subprocess"
    # What the test's own assertions could notice. "Weak" assertions hold for an
    # emptied body too: none at all (a must-not-raise smoke test), only negative
    # ones, or only captured output and mock call data.
    if facts.found and facts.assertions == 0:
        weak = "asserts nothing (it can only fail by crashing)"
    elif facts.negative_only:
        weak = "every assertion is one an emptied body also satisfies"
    elif facts.proxy_only:
        weak = "every assertion is on captured output or mock call data"
    else:
        weak = ""
    if verdict == "crash_only" and not weak:
        return "K", "fails under mutation only by crashing, before an assertion is reached"
    unlocked = list(row.get("unlocked_met") or [])
    if unlocked:
        more = f" (+{len(unlocked) - 2} more)" if len(unlocked) > 2 else ""
        return "C", (
            (weak or "its assertions hold for an emptied body too")
            + "; ran a function no test locks: "
            + ", ".join(unlocked[:2])
            + more
        )
    return "N", (
        (weak or "its assertions hold for an emptied body too")
        + "; every function it runs is locked by another test"
    )


def classify(state: Path) -> dict[str, Any]:
    report = ms._read_json(state / "report.json")
    survey_map = ms._read_json(state / "map" / "map.json")
    cfg = ms.load_config(state)
    roots = {prefix.rstrip("/") for prefix in cfg.production_prefixes()}
    project_files = frozenset(
        str(entry.relative_to(state / "pristine"))
        for root in sorted(roots)
        for entry in (state / "pristine" / root).rglob("*.py")
    )
    production_roots = Scope(
        roots=frozenset(roots),
        scope_files=frozenset(
            name
            for name in project_files
            if ms._glob_match(name, cfg.scope) and not ms._glob_match(name, cfg.scope_exclude)
        ),
        project_files=project_files,
    )
    excluded_modules = frozenset(
        pattern[: -len(".py")].replace("/", ".")
        for pattern in cfg.scope_exclude
        if pattern.endswith(".py") and "*" not in pattern
    )
    tests_dir = state / "pristine" / "tests"
    test_support_modules = frozenset(
        entry.stem
        for entry in (tests_dir.iterdir() if tests_dir.is_dir() else ())
        if (entry.suffix == ".py" or entry.is_dir())
        and not entry.name.startswith(("test_", "_", "."))
        and entry.stem not in roots
        and entry.stem != "conftest"
    )
    # Functions no test distinguishes from an empty body, and which functions each
    # test met a mutant of: a survivor matters when it is a witness to one of them.
    # "Witness" has to mean something: constructors and widely shared setup run
    # under hundreds of tests that never claimed them, so only a function few
    # tests run counts, and never a dunder.
    unlocked_functions = {
        entry["key"]
        for entry in report["function_rows"]
        if entry["verdict"] == "unlocked"
        and entry["covering"] <= _WITNESS_MAX_COVERING
        and not entry["key"].rsplit("@", 1)[0].rsplit(".", 1)[-1].startswith("__")
    }
    met: dict[str, set[str]] = {}
    results_path = state / "results.jsonl"
    if results_path.is_file():
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                result = json.loads(line)
                for test_id in result.get("tests", {}):
                    met.setdefault(ms._test_function_id(test_id), set()).add(result["key"])
    covered: dict[int, int] = {}
    for function in survey_map["functions"]:
        if not function["skip_reason"]:
            for index in ms._covering(function):
                covered[index] = covered.get(index, 0) + 1
    index_of = {test["id"]: index for index, test in enumerate(survey_map["tests"])}

    # One row per test function: its strongest instance speaks for it.
    rank = {"lock": 6, "crash_only": 5, "pending": 4, "unstable": 3, "survivor": 2, "no_body": 1}
    by_function: dict[str, dict[str, Any]] = {}
    for row in report["tests"]:
        name = ms._test_function_id(row["id"])
        best = by_function.get(name)
        if best is None or rank.get(row["verdict"], 0) > rank.get(best["verdict"], 0):
            by_function[name] = row

    trees: dict[str, ast.Module | None] = {}
    rows: list[dict[str, Any]] = []
    for name, row in sorted(by_function.items()):
        if row["verdict"] not in NOT_LOCKS:
            continue
        path = name.split("::", 1)[0]
        if path not in trees:
            try:
                source = (state / "pristine" / path).read_text(encoding="utf-8")
                trees[path] = ast.parse(source)
            except (OSError, SyntaxError):
                trees[path] = None
        tree = trees[path]
        facts = (
            static_facts(tree, name, production_roots, excluded_modules, test_support_modules)
            if tree is not None
            else TestFacts()
        )
        row = {**row, "unlocked_met": sorted(met.get(name, set()) & unlocked_functions)}
        suggested, reason = suggest(facts, row, covered.get(index_of.get(row["id"], -1), 0))
        rows.append(
            {
                "test": name,
                "verdict": row["verdict"],
                "class": suggested,
                "reason": reason,
                "saw": row["saw"],
                "unlocked_met": row["unlocked_met"],
                "kill_kinds": row.get("kill_kinds", []),
                "facts": dataclasses.asdict(facts),
            }
        )
    counts: dict[str, dict[str, int]] = {}
    for entry in rows:
        per_class = counts.setdefault(entry["class"], {})
        per_class[entry["verdict"]] = per_class.get(entry["verdict"], 0) + 1
    return {
        "test_functions": len(by_function),
        "locks": sum(1 for row in by_function.values() if row["verdict"] == "lock"),
        "pending": sum(1 for row in by_function.values() if row["verdict"] == "pending"),
        "not_locks": len(rows),
        "counts": {name: counts[name] for name in sorted(counts)},
        "rows": rows,
    }


def draw(rows: Iterable[dict[str, Any]], size: int, seed: int = DRAW_SEED) -> dict[str, list[str]]:
    """The same ``size`` tests per class for every reader: sorted, then seeded."""
    by_class: dict[str, list[str]] = {}
    for entry in rows:
        by_class.setdefault(entry["class"], []).append(entry["test"])
    draws: dict[str, list[str]] = {}
    for name, tests in sorted(by_class.items()):
        ordered = sorted(tests)
        random.Random(f"{seed}:{name}").shuffle(ordered)
        draws[name] = ordered[:size]
    return draws


def _iter_summary(result: dict[str, Any]) -> Iterator[str]:
    yield (
        f"test functions: {result['test_functions']}; assertion locks: {result['locks']}; "
        f"pending: {result['pending']}; never seen to fail: {result['not_locks']}"
    )
    for name, per_verdict in result["counts"].items():
        total = sum(per_verdict.values())
        yield f"  {name}: {total:>5}  {json.dumps(per_verdict, sort_keys=True)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--state-dir", required=True, help="a finished survey's state directory")
    parser.add_argument("--draws", type=int, default=0, help="fixed-seed draws per class")
    args = parser.parse_args(argv)
    state = Path(args.state_dir).resolve()
    result = classify(state)
    if args.draws:
        result["draws"] = draw(result["rows"], args.draws)
    ms._write_json(state / "classification.json", result)
    for line in _iter_summary(result):
        print(line)
    print(f"[classify] full classification: {state / 'classification.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
