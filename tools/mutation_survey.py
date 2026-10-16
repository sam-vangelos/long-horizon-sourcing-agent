#!/usr/bin/env python3
"""Mutation survey: which tests can fail, and which functions does anything lock.

The default suite is the gate for every merge, so a test that cannot fail is a
false alibi. This tool measures it directly. It maps every test to the
production functions it executes (one coverage run with per-test contexts),
empties each function in a hermetic copy of the repo, reruns only the tests
that execute it, and records which tests notice.

Usage, from the repo root::

    python tools/mutation_survey.py map      # hermetic copy + coverage run
    python tools/mutation_survey.py run      # resumable mutation runs
    python tools/mutation_survey.py report   # verdicts per test and function
    python tools/mutation_survey.py status
    python tools/mutation_survey.py prove --function KEY --tests NODEID ...

Two operators, no more: the empty body (the function returns before its first
statement) and boolean-return inversion (for functions that return bool). The
empty-body stub is inserted, never substituted: the original text stays in the
file as unreachable code, so a test that reads source text sees what it always
saw and only behavior changes.

Verdicts per test:

* ``lock``        an assertion in the test (or test helper) failed under at
                  least one mutation of a function the test executes.
* ``crash_only``  the test went red only because emptied code blew up: inside
                  production code, in a fixture, or on an ordinary line of the
                  test. No assertion fired, so nothing the test checks caught it.
* ``survivor``    the test ran against every mutation of every function it
                  executes and stayed green.
* ``no_body``     the test executes no mutable production function body, so
                  no body mutation can reach it.
* ``no_control``  the test does not pass on unmutated code (skipped, xfail,
                  failing), so nothing can be concluded from a mutant.
* ``unstable``    the test keeps failing a behavior-preserving edit of the
                  same function, so its failures say nothing about behavior.

A test leaves the survey at its first assertion kill; survivors and crash-only
tests see every mutant. No kill is credited on the mutant run alone: the
killers rerun against a null control, the same insertion with the behavior
change taken out, and only a test that passes it keeps its kill. That filters
text pins, ``git diff`` guards and failures that come from running a subset.
Function-level lock status is answered separately by probing already-resolved
tests, most specific first, until one fails.

Bytecode. Subprocesses never write bytecode, and the copy carries checked-hash
pycs only: a pyc is used only while its source file's hash matches, so a
mutated file is always compiled from its mutated source. Every mutant run then
proves it directly: the bytecode of the function object the subprocess holds
is compared with the bytecode the mutated source compiles to.

Safety envelope for every subprocess: secrets stripped from the environment,
no ``.env`` in the copy, a socket guard that refuses non-loopback connections
and the Chrome debugging port, ``CDP_URL`` pointed at a dead port, a dead proxy
for any Chromium that Playwright launches, and a throwaway ``HOME``.

State lives outside the repo, under the directory this tool prints at start.
Nothing here is ever committed.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

TOOL_VERSION = 1
PLUGIN_ENV = "MUTATION_SURVEY_PLUGIN_CONFIG"
PLUGIN_MODULE = "mutation_survey_plugin"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Cloris defaults. Every one is overridable on the command line so the tool
# can survey the planted fixture project in its own tests.
# ---------------------------------------------------------------------------

# Production code the survey mutates.
DEFAULT_SCOPE: tuple[str, ...] = (
    "linkedin/**/*.py",
    "shared/**/*.py",
    "github/**/*.py",
    "exa/**/*.py",
    "market_intelligence/**/*.py",
    "tools/*.py",
)
# Production code the survey measures but never mutates. A test whose real
# subject lives here is out of the program's scope, not theater.
DEFAULT_OBSERVE: tuple[str, ...] = (
    "cloris/**/*.py",
    "designer/**/*.py",
    "exec_search/**/*.py",
    "researcher/**/*.py",
    "decoy/**/*.py",
    "scripts/**/*.py",
    "tools/*/**/*.py",
)
DEFAULT_SCOPE_EXCLUDE: tuple[str, ...] = (
    # The survey does not survey itself: its end-to-end tests run surveys.
    "tools/mutation_survey.py",
)
DEFAULT_PYTEST_ARGS: tuple[str, ...] = (
    "-q",
    "--ignore=tests/test_fde_iteration_dataset.py",
    "--ignore-glob=tests/test_researcher_*.py",
)
# Priority order, so partial results are useful. First match wins.
DEFAULT_TIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("save path", ("linkedin/side_effects.py", "shared/save_destination/**/*.py")),
    ("runtime-state writers", ("shared/runtime_state/store.py",)),
    (
        "evaluation-loop stages",
        (
            "linkedin/candidate_review.py",
            "linkedin/page_review.py",
            "linkedin/string_driver.py",
            "linkedin/queue_planner.py",
            "linkedin/evaluation_loop.py",
        ),
    ),
    ("governor and safety", ("shared/governor.py", "shared/safety/**/*.py")),
    ("geography gate", ("linkedin/geography_gate.py",)),
    ("judgment parsing", ("shared/judgment/**/*.py", "shared/judger.py")),
    ("linkedin", ("linkedin/**/*.py",)),
    ("shared", ("shared/**/*.py",)),
    ("github", ("github/**/*.py",)),
    ("exa", ("exa/**/*.py",)),
    ("market_intelligence", ("market_intelligence/**/*.py",)),
    ("tools", ("tools/*.py",)),
)

_COPY_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
_DEAD_ENDPOINT = "http://127.0.0.1:9"
_LOCK_PROBE_CAP = 120
_UNSTABLE_AFTER = 3


def _glob_regex(pattern: str) -> re.Pattern[str]:
    parts = re.split(r"(\*\*/|\*|\?)", pattern)
    translate = {"**/": r"(?:[^/]+/)*", "*": r"[^/]*", "?": r"[^/]"}
    return re.compile("".join(translate.get(part, re.escape(part)) for part in parts) + r"\Z")


def _glob_match(path: str, patterns: Iterable[str]) -> bool:
    """Posix relpath globs: ``*`` stays inside one directory, ``**/`` spans zero or more."""
    return any(_glob_regex(pattern).match(path) for pattern in patterns)


# ---------------------------------------------------------------------------
# Source analysis
# ---------------------------------------------------------------------------

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_COMPOUND_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Match,
) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
_BOOL_CALLS = {"isinstance", "issubclass", "hasattr", "callable", "any", "all", "bool"}
_BOOL_METHODS = {"startswith", "endswith"}


@dataclasses.dataclass(frozen=True)
class FunctionTarget:
    """One production function, as the survey sees it."""

    path: str
    qualname: str
    lineno: int
    end_lineno: int
    is_generator: bool
    returns_value: bool
    bool_return: str  # "", "annotated" or "inferred"
    skip_reason: str  # why no empty-body mutant exists, or ""
    own_lines: tuple[int, ...]
    def_line_shared: bool  # body starts on the def line: import executes it too
    first_line: int = 0  # where the code object starts: first decorator, else the def

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}@{self.lineno}"


def _own_scope_nodes(func: ast.AST) -> Iterator[ast.AST]:
    """Walk a function's own statements, not the bodies of nested defs or lambdas."""
    stack: list[ast.AST] = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _FUNCTION_NODES + (ast.Lambda,)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _split_docstring(body: Sequence[ast.stmt]) -> tuple[ast.stmt | None, list[ast.stmt]]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0], list(body[1:])
    return None, list(body)


def _is_trivial_statement(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return True  # ``...`` or a stray string
    if isinstance(stmt, ast.Return):
        return stmt.value is None or (
            isinstance(stmt.value, ast.Constant) and stmt.value.value is None
        )
    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
        return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"
    return False


def _decorator_names(func: ast.AST) -> set[str]:
    names: set[str] = set()
    for decorator in getattr(func, "decorator_list", []):
        node = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _is_boolish(expr: ast.expr | None) -> bool:
    if expr is None:
        return False
    if isinstance(expr, ast.Constant):
        return isinstance(expr.value, bool)
    if isinstance(expr, ast.Compare):
        return True
    if isinstance(expr, ast.UnaryOp):
        return isinstance(expr.op, ast.Not)
    if isinstance(expr, ast.BoolOp):
        return all(_is_boolish(value) for value in expr.values)
    if isinstance(expr, ast.IfExp):
        return _is_boolish(expr.body) and _is_boolish(expr.orelse)
    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Name):
            return expr.func.id in _BOOL_CALLS
        if isinstance(expr.func, ast.Attribute):
            return expr.func.attr in _BOOL_METHODS
    return False


def _annotation_is_bool(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id == "bool"
    if isinstance(annotation, ast.Constant):
        return annotation.value == "bool"
    if isinstance(annotation, ast.Subscript):
        head = annotation.value
        name = head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", "")
        return name in {"TypeGuard", "TypeIs"}
    return False


def _value_returns(func: ast.AST) -> list[ast.Return]:
    return [
        node
        for node in _own_scope_nodes(func)
        if isinstance(node, ast.Return)
        and node.value is not None
        and not (isinstance(node.value, ast.Constant) and node.value.value is None)
    ]


def _bool_return_kind(func: ast.AST, is_generator: bool) -> str:
    if is_generator:
        return ""
    value_returns = _value_returns(func)
    if not value_returns:
        return ""
    if _annotation_is_bool(getattr(func, "returns", None)):
        return "annotated"
    if getattr(func, "returns", None) is not None:
        return ""
    all_returns = [n for n in _own_scope_nodes(func) if isinstance(n, ast.Return)]
    if all(_is_boolish(node.value) for node in all_returns):
        return "inferred"
    return ""


def _own_lines(func: ast.AST, rest: Sequence[ast.stmt]) -> tuple[int, ...]:
    if not rest:
        return ()
    lines = set(range(rest[0].lineno, (func.end_lineno or rest[-1].lineno) + 1))
    for node in _own_scope_nodes(func):
        if isinstance(node, _FUNCTION_NODES) and node.body:
            lines -= set(range(node.body[0].lineno, (node.end_lineno or node.lineno) + 1))
    return tuple(sorted(lines))


def discover_functions(source: str, path: str) -> list[FunctionTarget]:
    """Every function in ``source``, nested ones included, with its own body lines."""
    tree = ast.parse(source, filename=path)
    targets: list[FunctionTarget] = []

    def visit(node: ast.AST, prefix: str, type_checking: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.", type_checking)
            elif isinstance(child, _FUNCTION_NODES):
                targets.append(_describe(child, f"{prefix}{child.name}", type_checking))
                visit(child, f"{prefix}{child.name}.<locals>.", type_checking)
            elif isinstance(child, ast.If):
                test = child.test
                name = test.attr if isinstance(test, ast.Attribute) else getattr(test, "id", "")
                visit(child, prefix, type_checking or name == "TYPE_CHECKING")
            else:
                visit(child, prefix, type_checking)

    def _describe(func: ast.AST, qualname: str, type_checking: bool) -> FunctionTarget:
        _, rest = _split_docstring(func.body)
        is_generator = any(
            isinstance(node, (ast.Yield, ast.YieldFrom)) for node in _own_scope_nodes(func)
        )
        decorators = _decorator_names(func)
        if type_checking:
            skip_reason = "type_checking_only"
        elif "overload" in decorators:
            skip_reason = "overload_stub"
        elif "abstractmethod" in decorators:
            skip_reason = "abstract_method"
        elif all(_is_trivial_statement(stmt) for stmt in rest):
            skip_reason = "trivial_body"
        else:
            skip_reason = ""
        return FunctionTarget(
            path=path,
            qualname=qualname,
            lineno=func.lineno,
            end_lineno=func.end_lineno or func.lineno,
            is_generator=is_generator,
            returns_value=bool(_value_returns(func)),
            bool_return="" if skip_reason else _bool_return_kind(func, is_generator),
            skip_reason=skip_reason,
            own_lines=_own_lines(func, rest),
            def_line_shared=bool(rest) and rest[0].lineno == func.lineno,
            first_line=_statement_first_line(func),
        )

    visit(tree, "", False)
    return targets


def _find_function_node(tree: ast.AST, target: FunctionTarget) -> ast.AST:
    for node in ast.walk(tree):
        if (
            isinstance(node, _FUNCTION_NODES)
            and node.lineno == target.lineno
            and node.name == target.qualname.rsplit(".", 1)[-1]
        ):
            return node
    raise LookupError(f"function not found at recorded position: {target.key}")


# ---------------------------------------------------------------------------
# Mutators. Text edits on UTF-8 bytes, because ast column offsets are bytes.
# ---------------------------------------------------------------------------

OPERATORS = ("empty_body", "invert_bool")


class MutantInvalid(Exception):
    """The operator does not apply, or the mutated source does not compile."""


def _apply_insertions(source: str, insertions: list[tuple[int, int, str]]) -> str:
    lines = source.encode("utf-8").splitlines(keepends=True)
    for lineno, col, text in sorted(insertions, reverse=True):
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:col] + text.encode("utf-8") + line[col:]
    return b"".join(lines).decode("utf-8")


def _statement_first_line(stmt: ast.stmt) -> int:
    decorators = getattr(stmt, "decorator_list", [])
    return min([stmt.lineno] + [d.lineno for d in decorators])


def mutate_empty_body(source: str, target: FunctionTarget, *, null: bool = False) -> str:
    """Make the function return before its first statement; keep every character.

    ``null=True`` builds the control for text-sensitive tests: the same
    insertion with ``pass`` in place of the return, so behavior is unchanged
    and only the text differs.
    """
    if target.skip_reason:
        raise MutantInvalid(target.skip_reason)
    func = _find_function_node(ast.parse(source), target)
    docstring, rest = _split_docstring(func.body)
    if not rest:
        raise MutantInvalid("trivial_body")
    # A generator stays a generator: the original yields remain in the text.
    # An async generator may not return a value, so generators get a bare return.
    stub = "return None" if target.returns_value and not target.is_generator else "return"
    if null:
        stub = "pass"
    anchor = rest[0]
    if not isinstance(anchor, _COMPOUND_NODES):
        mutated = _apply_insertions(source, [(anchor.lineno, anchor.col_offset, f"{stub}; ")])
    elif docstring is not None and docstring.end_lineno is not None:
        mutated = _apply_insertions(
            source, [(docstring.end_lineno, docstring.end_col_offset or 0, f"; {stub}")]
        )
    else:
        first_line = _statement_first_line(anchor)
        raw = source.encode("utf-8").splitlines(keepends=True)
        anchor_line = raw[first_line - 1]
        indent = anchor_line[: len(anchor_line) - len(anchor_line.lstrip())]
        raw.insert(first_line - 1, indent + stub.encode("utf-8") + b"\n")
        mutated = b"".join(raw).decode("utf-8")
    _require_compiles(mutated, target)
    return mutated


def mutate_invert_bool(source: str, target: FunctionTarget, *, null: bool = False) -> str:
    """Wrap every returned value in ``not (...)``. The null control only parenthesizes."""
    if not target.bool_return:
        raise MutantInvalid("not_a_bool_function")
    func = _find_function_node(ast.parse(source), target)
    insertions: list[tuple[int, int, str]] = []
    for node in _value_returns(func):
        value = node.value
        assert value is not None and value.end_lineno is not None
        insertions.append((value.lineno, value.col_offset, "(" if null else "not ("))
        insertions.append((value.end_lineno, value.end_col_offset or 0, ")"))
    if not insertions:
        raise MutantInvalid("no_value_returns")
    mutated = _apply_insertions(source, insertions)
    _require_compiles(mutated, target)
    return mutated


def mutate_empty_lines(source: str, start: int, end: int) -> str:
    """Replace whole statements spanning ``start..end`` with ``pass``; keep the line count."""
    tree = ast.parse(source)
    block: list[ast.stmt] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody", "handlers"):
            statements = getattr(node, field, None)
            if not isinstance(statements, list):
                continue
            inside = [
                s
                for s in statements
                if isinstance(s, ast.stmt)
                and _statement_first_line(s) >= start
                and (s.end_lineno or s.lineno) <= end
            ]
            if inside and len(inside) > len(block):
                block = inside
    if not block:
        raise MutantInvalid(f"no whole statements inside lines {start}-{end}")
    first, last = _statement_first_line(block[0]), block[-1].end_lineno or block[-1].lineno
    raw = source.encode("utf-8").splitlines(keepends=True)
    head = raw[first - 1]
    indent = head[: len(head) - len(head.lstrip())]
    raw[first - 1] = indent + b"pass\n"
    for lineno in range(first + 1, last + 1):
        raw[lineno - 1] = b"\n"
    mutated = b"".join(raw).decode("utf-8")
    compile(mutated, "<empty-lines>", "exec")
    return mutated


def mutate_replace_on_line(source: str, lineno: int, old: str, new: str) -> str:
    """Swap ``old`` for ``new`` on one line: a branch condition forced, a call removed."""
    lines = source.splitlines(keepends=True)
    if not 1 <= lineno <= len(lines) or lines[lineno - 1].count(old) != 1:
        raise MutantInvalid(f"line {lineno} does not contain {old!r} exactly once")
    lines[lineno - 1] = lines[lineno - 1].replace(old, new)
    mutated = "".join(lines)
    compile(mutated, "<replace-on-line>", "exec")
    return mutated


def _require_compiles(mutated: str, target: FunctionTarget) -> None:
    try:
        compile(mutated, target.path, "exec")
    except SyntaxError as exc:
        raise MutantInvalid(f"mutant does not compile: {exc.msg} (line {exc.lineno})") from exc


def build_mutant(source: str, target: FunctionTarget, operator: str, *, null: bool = False) -> str:
    if operator == "empty_body":
        return mutate_empty_body(source, target, null=null)
    if operator == "invert_bool":
        return mutate_invert_bool(source, target, null=null)
    raise ValueError(f"unknown operator: {operator}")


# ---------------------------------------------------------------------------
# Hermetic copies and the subprocess envelope
# ---------------------------------------------------------------------------

_SITECUSTOMIZE = '''\
"""Network guard for mutation-survey subprocesses (written by tools/mutation_survey.py)."""
import os
import socket

_BLOCKED_PORTS = {
    int(port)
    for port in os.environ.get("MUTATION_SURVEY_BLOCKED_PORTS", "9222").split(",")
    if port.strip().isdigit()
}


class MutationSurveyNetworkBlocked(ConnectionRefusedError):
    pass


def _allowed(address):
    if not isinstance(address, tuple) or len(address) < 2:
        return True
    host, port = str(address[0]), address[1]
    if port in _BLOCKED_PORTS:
        return False
    return host in ("localhost", "::1", "0.0.0.0", "") or host.startswith("127.")


_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def _guarded_connect(self, address):
    if not _allowed(address):
        raise MutationSurveyNetworkBlocked(f"mutation survey blocks connect to {address!r}")
    return _connect(self, address)


def _guarded_connect_ex(self, address):
    if not _allowed(address):
        raise MutationSurveyNetworkBlocked(f"mutation survey blocks connect to {address!r}")
    return _connect_ex(self, address)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
'''


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def list_project_files(root: Path) -> list[str]:
    """Tracked plus untracked-unignored files when git can say; otherwise a plain walk."""
    try:
        output = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        files = [name for name in output.decode("utf-8").split("\0") if name]
        if files:
            return sorted(name for name in files if (root / name).is_file())
    except (OSError, subprocess.CalledProcessError):
        pass  # not a git checkout (a hermetic copy, say): walk instead
    found: list[str] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _COPY_SKIP_DIRS]
        for filename in filenames:
            if filename.endswith((".pyc", ".pyo")):
                continue
            found.append(Path(directory, filename).relative_to(root).as_posix())
    return sorted(found)


def build_pristine(root: Path, destination: Path) -> list[str]:
    """Copy the project into ``destination``: no git dir, no venv, no bytecode, no .env."""
    if destination.exists():
        shutil.rmtree(destination)
    files = [
        name
        for name in list_project_files(root)
        if not (set(Path(name).parts) & _COPY_SKIP_DIRS)
        and Path(name).name != ".env"
        and not name.endswith((".pyc", ".pyo"))
    ]
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target, follow_symlinks=True)
    _init_throwaway_repository(destination)
    return files


def _init_throwaway_repository(copy_root: Path) -> None:
    """Give the copy a repository of its own, so tests that ask git for HEAD get an answer.

    A fresh ``git init`` under the state directory shares nothing with the real
    repository: no remote, no objects, no worktree link, no user configuration.
    Best effort; without git the tests that need it fail their control and say so.
    """
    env = {
        **os.environ,
        "GIT_CEILING_DIRECTORIES": str(copy_root.parent),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    identity = ["-c", "user.name=mutation-survey", "-c", "user.email=survey@localhost"]
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", *identity, "commit", "-q", "--no-gpg-sign", "-m", "pristine hermetic copy"],
    ):
        try:
            done = subprocess.run(command, cwd=copy_root, env=env, capture_output=True)
        except OSError:
            return
        if done.returncode != 0:
            return
        if command[1] == "init":
            # The survey's own bytecode and marker never show up as changes.
            (copy_root / ".git" / "info").mkdir(parents=True, exist_ok=True)
            (copy_root / ".git" / "info" / "exclude").write_text(
                "__pycache__/\n*.pyc\n.mutation-survey-bytecode\n", encoding="utf-8"
            )


def clone_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        # APFS clone: near-free copies, one per worker.
        done = subprocess.run(["cp", "-cR", str(source), str(destination)], capture_output=True)
        if done.returncode == 0:
            return
        shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination, symlinks=True)


def subprocess_env(state: Path, copy_root: Path, tmp_dir: Path) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if not _SECRET_NAME.search(name)}
    real_home = Path(os.environ.get("HOME", str(Path.home())))
    browsers = Path(
        os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        or real_home / "Library" / "Caches" / "ms-playwright"
    )
    if browsers.exists():
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers)
    home = state / "home"
    home.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(tmp_dir),
            "PYTHONPATH": os.pathsep.join([str(state / "support"), str(copy_root)]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CDP_URL": _DEAD_ENDPOINT,
            "NO_COLOR": "1",
            # A test that shells out to git must never find a repository above the copy.
            "GIT_CEILING_DIRECTORIES": str(state),
        }
    )
    for name in ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG", PLUGIN_ENV):
        env.pop(name, None)
    return env


def write_support_files(state: Path) -> None:
    support = state / "support"
    support.mkdir(parents=True, exist_ok=True)
    (support / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    shutil.copyfile(Path(__file__).resolve(), support / f"{PLUGIN_MODULE}.py")


def purge_pycache(root: Path) -> None:
    for directory, dirnames, _ in os.walk(root):
        for name in list(dirnames):
            if name == "__pycache__":
                shutil.rmtree(Path(directory, name), ignore_errors=True)
                dirnames.remove(name)


def compile_checked_hash(root: Path) -> None:
    """Give the copy bytecode that cannot go stale.

    Without any bytecode every subprocess recompiles the whole import graph
    from source, which costs more than the tests it then runs. A checked-hash
    pyc is valid only while the source file's hash matches, so a mutated file
    is always compiled from its mutated source and a restored file gets its
    cache back. Subprocesses never write bytecode, so no mutant can leave one.
    """
    purge_pycache(root)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "-j",
            "0",
            "--invalidation-mode",
            "checked-hash",
            str(root),
        ],
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": ""},
    )
    (root / ".mutation-survey-bytecode").write_text("checked-hash\n", encoding="utf-8")


def function_code_fingerprint(source: str, name: str, first_line: int) -> str:
    """sha256 of the bytecode of the function named ``name`` that starts at ``first_line``."""
    pending = [compile(source, "<fingerprint>", "exec", dont_inherit=True)]
    while pending:
        code = pending.pop()
        if code.co_name == name and code.co_firstlineno == first_line:
            return _sha256(code.co_code)
        pending.extend(const for const in code.co_consts if isinstance(const, types.CodeType))
    return ""


# ---------------------------------------------------------------------------
# pytest plugin half. Active only inside survey subprocesses, where the module
# is loaded as ``-p mutation_survey_plugin`` from the state dir's support copy.
# ---------------------------------------------------------------------------


class MutationSurveyTimeout(BaseException):
    """BaseException on purpose: absorb-and-resume loops catch Exception."""


def pytest_configure(config: Any) -> None:
    config_path = os.environ.get(PLUGIN_ENV)
    if config_path:
        plugin_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        config.pluginmanager.register(_build_plugin(plugin_config), "mutation-survey-runtime")


def _build_plugin(cfg: dict[str, Any]) -> Any:
    import pytest

    root = os.path.realpath(cfg["root"])
    production_prefixes = tuple(cfg["production_prefixes"])
    mode = cfg["mode"]
    primary = list(cfg.get("primary", []))
    primary_set = set(primary)
    secondary_rank = {nodeid: rank for rank, nodeid in enumerate(cfg.get("secondary", []))}
    timeouts: dict[str, float] = cfg.get("timeouts", {})
    text_sensitive = set(cfg.get("text_sensitive", []))

    def relpath(path: Any) -> str | None:
        try:
            real = os.path.realpath(os.fspath(path))
        except TypeError:
            return None
        if real == root or not real.startswith(root + os.sep):
            return None
        return real[len(root) + 1 :].replace(os.sep, "/")

    class SurveyPlugin:
        def __init__(self) -> None:
            self.records: dict[str, dict[str, Any]] = {}
            self.kinds: dict[tuple[str, str], dict[str, str]] = {}
            self.collection_errors: list[dict[str, str]] = []
            self.reads: dict[str, set[str]] = {}
            self.fixture_owner: dict[str, str] = {}
            self.fixture_users: dict[str, list[str]] = {}
            self.test_trees: dict[str, ast.AST | None] = {}
            self.current: str | None = None
            self.kill_seen = False
            self.stopped_early = False
            self.primary_remaining = 0
            self.progress = open(cfg["progress_path"], "a", encoding="utf-8")
            if mode == "map":
                sys.addaudithook(self._audit)
            if cfg.get("guard_playwright"):
                _guard_playwright()

        # -- map mode: which tests read production source text ---------------
        def _audit(self, event: str, args: tuple[Any, ...]) -> None:
            if event != "open" or self.current is None or not args:
                return
            # Suffix check first: this hook sees every file any test opens.
            if not isinstance(args[0], (str, os.PathLike)) or not str(args[0]).endswith(".py"):
                return
            rel = relpath(args[0])
            if rel is None or not rel.startswith(production_prefixes):
                return
            frame = sys._getframe(1)
            while frame is not None:
                filename = frame.f_code.co_filename
                # Imports and traceback rendering open source too; neither is
                # a test reading production text.
                if filename.startswith("<frozen importlib") or filename.endswith(
                    (os.sep + "traceback.py", os.sep + "_code" + os.sep + "code.py")
                ):
                    return
                frame = frame.f_back
            self.reads.setdefault(self.current, set()).add(rel)

        # -- map mode: who else depends on a shared fixture's work -----------
        # Coverage credits a module- or session-scoped fixture's execution to
        # the first test that asks for it. Every later user depends on that
        # work just as much, so the map hands them the owner's setup coverage.
        def _is_project_shared_fixture(self, fixturedef: Any) -> bool:
            """A fixture wider than one test, defined in this project.

            pytest's own session fixtures (tmp_path_factory and the like) are
            shared by every test and run no project code; counting them would
            hand the first test's setup coverage to the whole suite.
            """
            if fixturedef.scope == "function":
                return False
            code = getattr(getattr(fixturedef, "func", None), "__code__", None)
            return code is not None and relpath(code.co_filename) is not None

        @pytest.hookimpl(hookwrapper=True)
        def pytest_fixture_setup(self, fixturedef: Any, request: Any) -> Iterator[None]:
            if (
                mode == "map"
                and self.current is not None
                and self._is_project_shared_fixture(fixturedef)
            ):
                key = "|".join(
                    [
                        fixturedef.argname,
                        str(fixturedef.baseid),
                        getattr(request.node, "nodeid", ""),
                        str(getattr(request, "param_index", 0)),
                    ]
                )
                self.fixture_owner.setdefault(key, self.current)
            yield

        def _shared_fixture_keys(self, item: Any) -> list[str]:
            info = getattr(item, "_fixtureinfo", None)
            if info is None:
                return []
            indices = getattr(getattr(item, "callspec", None), "indices", {})
            scope_nodes = {
                "class": item.getparent(pytest.Class) or item.getparent(pytest.Module),
                "module": item.getparent(pytest.Module),
                "package": item.getparent(pytest.Package) or item.session,
                "session": item.session,
            }
            keys = []
            for name in item.fixturenames:
                definitions = info.name2fixturedefs.get(name)
                if not definitions or not self._is_project_shared_fixture(definitions[-1]):
                    continue
                fixturedef = definitions[-1]
                node = scope_nodes.get(fixturedef.scope)
                keys.append(
                    "|".join(
                        [
                            name,
                            str(fixturedef.baseid),
                            getattr(node, "nodeid", ""),
                            str(indices.get(name, 0)),
                        ]
                    )
                )
            return keys

        # -- mutant mode: run only the covering tests, primary first ---------
        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, config: Any, items: list[Any]) -> None:
            if mode == "map":
                return
            first = [item for item in items if item.nodeid in primary_set]
            second = sorted(
                (item for item in items if item.nodeid in secondary_rank),
                key=lambda item: secondary_rank[item.nodeid],
            )
            keep = first + second
            kept = {item.nodeid for item in keep}
            dropped = [item for item in items if item.nodeid not in kept]
            items[:] = keep
            if dropped:
                config.hook.pytest_deselected(items=dropped)
            self.primary_remaining = len(first)

        def pytest_collectreport(self, report: Any) -> None:
            if report.failed:
                self.collection_errors.append(
                    {"nodeid": report.nodeid, "message": str(report.longrepr)[-600:]}
                )

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_protocol(self, item: Any, nextitem: Any) -> Iterator[None]:
            import linecache

            self.current = item.nodeid
            linecache.clearcache()
            if mode == "map":
                self.fixture_users[item.nodeid] = self._shared_fixture_keys(item)
            self.progress.write(f"START {item.nodeid}\n")
            self.progress.flush()
            budget = float(timeouts.get(item.nodeid, cfg.get("default_timeout", 0)) or 0)
            armed = budget > 0 and threading.current_thread() is threading.main_thread()
            if armed:
                signal.signal(signal.SIGALRM, _raise_timeout)
                signal.setitimer(signal.ITIMER_REAL, budget, 5.0)
            try:
                yield
            finally:
                if armed:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                self.current = None

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item: Any, call: Any) -> Iterator[None]:
            outcome = yield
            report = outcome.get_result()
            if report.failed and call.excinfo is not None:
                self.kinds[(item.nodeid, report.when)] = self._classify(call.excinfo, report.when)

        def _classify(self, excinfo: Any, when: str) -> dict[str, str]:
            """Name how the test went red.

            ``assert`` is reserved for an assertion mechanism firing in test
            code: an AssertionError, a pytest failure such as DID NOT RAISE, or
            any exception raised while an ``assert`` statement was being
            evaluated. A broken object blowing up on an ordinary test line is a
            ``test_crash``; an exception whose innermost project frame is
            production code is a ``crash``. Neither proves the test checks
            anything, so neither lets the test leave the survey.
            """
            where, frame, asserting = "unknown", "", False
            for entry in reversed(excinfo.traceback):
                rel = relpath(str(entry.path))
                if rel is None:
                    continue
                where = "production" if rel.startswith(production_prefixes) else "test"
                frame = f"{rel}:{entry.lineno + 1}"
                asserting = where == "test" and self._is_assert_line(
                    str(entry.path), entry.lineno + 1
                )
                break
            name = excinfo.type.__name__
            if name == "MutationSurveyTimeout":
                kind = "timeout"
            elif when != "call":
                kind = when
            elif where != "test":
                kind = "crash"
            elif asserting or issubclass(excinfo.type, AssertionError) or name == "Failed":
                kind = "assert"
            else:
                kind = "test_crash"
            return {"kind": kind, "exc": name, "msg": str(excinfo.value)[:240], "frame": frame}

        def _is_assert_line(self, path: str, lineno: int) -> bool:
            """Is the innermost statement at this line of a test file an ``assert``?"""
            if path not in self.test_trees:
                try:
                    self.test_trees[path] = ast.parse(Path(path).read_text(encoding="utf-8"))
                except (OSError, SyntaxError, ValueError):
                    self.test_trees[path] = None
            tree = self.test_trees[path]
            innermost: ast.stmt | None = None
            for node in ast.walk(tree) if tree is not None else ():
                if not isinstance(node, ast.stmt):
                    continue
                if node.lineno <= lineno <= (node.end_lineno or node.lineno):
                    if innermost is None or node.lineno >= innermost.lineno:
                        innermost = node
            return isinstance(innermost, ast.Assert)

        def pytest_runtest_logreport(self, report: Any) -> None:
            record = self.records.setdefault(
                report.nodeid, {"id": report.nodeid, "outcome": "passed", "duration": 0.0}
            )
            record["duration"] = round(record["duration"] + float(report.duration), 4)
            if report.failed and record["outcome"] in {"passed", "skipped"}:
                record["outcome"] = "failed"
                record["phase"] = report.when
                record.update(self.kinds.get((report.nodeid, report.when), {"kind": report.when}))
            elif report.skipped and record["outcome"] == "passed":
                xfail = hasattr(report, "wasxfail")
                record["outcome"] = "xfailed" if xfail else "skipped"
                longrepr = report.longrepr
                reason = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
                record["reason"] = (getattr(report, "wasxfail", "") or reason)[:240]
            elif report.passed and report.when == "call" and hasattr(report, "wasxfail"):
                record["outcome"] = "xpassed"
            if report.when != "teardown":
                return
            self.progress.write(f"END {report.nodeid} {record['outcome']}\n")
            self.progress.flush()
            if mode == "map":
                return
            if record["outcome"] == "failed" and report.nodeid not in text_sensitive:
                self.kill_seen = True
            if report.nodeid in primary_set:
                self.primary_remaining -= 1
            if self.kill_seen and self.primary_remaining <= 0:
                self.stopped_early = True
                self.session.shouldstop = "mutation survey: lock found, primary tests done"

        def pytest_sessionstart(self, session: Any) -> None:
            self.session = session

        def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
            for nodeid, files in self.reads.items():
                self.records.setdefault(nodeid, {"id": nodeid, "outcome": "unknown"})[
                    "reads"
                ] = sorted(files)
            for nodeid, keys in self.fixture_users.items():
                owners = {self.fixture_owner.get(key) for key in keys} - {None, nodeid}
                if owners and nodeid in self.records:
                    self.records[nodeid]["shared_fixture_owners"] = sorted(owners)
            result: dict[str, Any] = {
                "mode": mode,
                "exitstatus": int(exitstatus),
                "stopped_early": self.stopped_early,
                "tests": list(self.records.values()),
                "collection_errors": self.collection_errors,
                "dont_write_bytecode": bool(sys.dont_write_bytecode),
            }
            target = cfg.get("target_path")
            if target:
                result["target"] = _confirm_import(
                    root, target, cfg.get("expected_sha256", ""), cfg.get("expected_code")
                )
            temporary = Path(cfg["result_path"] + ".part")
            temporary.write_text(json.dumps(result), encoding="utf-8")
            os.replace(temporary, cfg["result_path"])
            self.progress.close()

    return SurveyPlugin()


def _raise_timeout(signum: int, frame: Any) -> None:
    raise MutationSurveyTimeout("mutation survey per-test budget exceeded")


def _confirm_import(
    root: str, target: str, expected_sha256: str, expected_code: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Was the mutated file the one this process imported, and is the mutant what it runs?

    ``code_confirmed`` compares the bytecode of the function object this process
    actually holds with the bytecode the mutated source compiles to. True is
    direct proof; False means stale or foreign code ran and the run is void;
    None means the function object could not be reached from the module.
    """
    target_file = os.path.realpath(os.path.join(root, target))
    imported_from = ""
    target_module = None
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if module_file and os.path.realpath(module_file) == target_file:
            imported_from, target_module = module_file, module
            break
    try:
        disk_sha = _sha256(Path(target_file).read_bytes())
    except OSError:
        disk_sha = ""
    code_confirmed: bool | None = None
    if target_module is not None and expected_code and expected_code.get("sha256"):
        loaded = _loaded_function_code(
            target_module, target_file, expected_code["name"], int(expected_code["first_line"])
        )
        if loaded is not None:
            code_confirmed = _sha256(loaded.co_code) == expected_code["sha256"]
    return {
        "imported": bool(imported_from),
        "module_file": imported_from,
        "disk_sha_ok": disk_sha == expected_sha256,
        "code_confirmed": code_confirmed,
    }


def _loaded_function_code(
    module: Any, module_file: str, name: str, first_line: int
) -> types.CodeType | None:
    """The code object of the named function as this process holds it, nested ones included."""
    pending: list[Any] = list(vars(module).values())
    seen: set[int] = set()
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, types.CodeType):
            if os.path.realpath(obj.co_filename) != module_file:
                continue
            if obj.co_name == name and obj.co_firstlineno == first_line:
                return obj
            pending.extend(const for const in obj.co_consts if isinstance(const, types.CodeType))
        elif isinstance(obj, (staticmethod, classmethod)):
            pending.append(obj.__func__)
        elif isinstance(obj, property):
            pending.extend(part for part in (obj.fget, obj.fset, obj.fdel) if part is not None)
        elif isinstance(obj, type):
            if getattr(obj, "__module__", None) == getattr(module, "__name__", None):
                pending.extend(vars(obj).values())
        else:
            for attribute in ("__code__", "__wrapped__", "__func__"):
                try:
                    inner = getattr(obj, attribute, None)
                except Exception:  # a lazy proxy in a module namespace may raise on access
                    inner = None
                if isinstance(inner, types.CodeType) or callable(inner):
                    pending.append(inner)
    return None


def _guard_playwright() -> None:
    """No Chromium launched under the survey can reach the network or a live Chrome."""
    for package in ("rebrowser_playwright", "playwright"):
        try:
            module = __import__(f"{package}.async_api", fromlist=["BrowserType"])
        except Exception:
            continue
        browser_type = module.BrowserType
        if getattr(browser_type, "_mutation_survey_guarded", False):
            continue
        launch = browser_type.launch

        async def guarded_launch(self: Any, *args: Any, _launch: Any = launch, **kwargs: Any) -> Any:
            kwargs.setdefault("proxy", {"server": _DEAD_ENDPOINT})
            return await _launch(self, *args, **kwargs)

        async def refuse_attach(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise ConnectionRefusedError("mutation survey refuses connect_over_cdp")

        browser_type.launch = guarded_launch
        browser_type.connect_over_cdp = refuse_attach
        browser_type._mutation_survey_guarded = True


# ---------------------------------------------------------------------------
# Survey state
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SurveyConfig:
    root: Path
    state: Path
    scope: tuple[str, ...]
    observe: tuple[str, ...]
    scope_exclude: tuple[str, ...]
    pytest_args: tuple[str, ...]
    tiers: tuple[tuple[str, tuple[str, ...]], ...]

    def production_prefixes(self) -> list[str]:
        prefixes = {pattern.split("/", 1)[0] + "/" for pattern in self.scope + self.observe}
        return sorted(prefixes)

    def tier_of(self, path: str) -> int:
        for index, (_, patterns) in enumerate(self.tiers):
            if _glob_match(path, patterns):
                return index
        return len(self.tiers)

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["root"], data["state"] = str(self.root), str(self.state)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SurveyConfig":
        return cls(
            root=Path(data["root"]),
            state=Path(data["state"]),
            scope=tuple(data["scope"]),
            observe=tuple(data["observe"]),
            scope_exclude=tuple(data["scope_exclude"]),
            pytest_args=tuple(data["pytest_args"]),
            tiers=tuple((name, tuple(patterns)) for name, patterns in data["tiers"]),
        )


def default_state_dir(root: Path) -> Path:
    digest = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:10]
    return Path(tempfile.gettempdir()) / "cloris-mutation-survey" / digest


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def load_config(state: Path) -> SurveyConfig:
    meta = state / "meta.json"
    if not meta.exists():
        raise SystemExit(f"[survey] no survey state at {state}; run the map step first")
    return SurveyConfig.from_json(_read_json(meta)["config"])


def _run_pytest(
    cfg: SurveyConfig,
    copy_root: Path,
    run_dir: Path,
    plugin_config: dict[str, Any],
    pytest_args: Sequence[str],
    *,
    budget_seconds: float,
) -> dict[str, Any]:
    """One pytest subprocess inside a hermetic copy. Returns the plugin's result."""
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path, progress_path = run_dir / "result.json", run_dir / "progress.log"
    for stale in (result_path, progress_path):
        stale.unlink(missing_ok=True)
    plugin_config = {
        **plugin_config,
        "root": str(copy_root),
        "production_prefixes": cfg.production_prefixes(),
        "result_path": str(result_path),
        "progress_path": str(progress_path),
    }
    config_path = run_dir / "plugin_config.json"
    _write_json(config_path, plugin_config)
    env = subprocess_env(cfg.state, copy_root, run_dir / "tmp")
    env[PLUGIN_ENV] = str(config_path)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--rootdir",
        str(copy_root),
        "-p",
        PLUGIN_MODULE,
        "-p",
        "no:cacheprovider",
        *pytest_args,
    ]
    started = time.monotonic()
    with open(run_dir / "pytest.out", "wb") as output:
        process = subprocess.Popen(
            command,
            cwd=copy_root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=budget_seconds)
            hard_timeout = False
        except subprocess.TimeoutExpired:
            hard_timeout = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    result: dict[str, Any] = {"tests": [], "collection_errors": [], "exitstatus": None}
    if result_path.exists():
        result = _read_json(result_path)
    result["hard_timeout"] = hard_timeout
    result["seconds"] = round(time.monotonic() - started, 2)
    result["progress"] = (
        progress_path.read_text(encoding="utf-8").splitlines() if progress_path.exists() else []
    )
    shutil.rmtree(run_dir / "tmp", ignore_errors=True)
    return result


# ---------------------------------------------------------------------------
# map: the hermetic copy and the coverage run
# ---------------------------------------------------------------------------


def _scope_files(cfg: SurveyConfig, files: Sequence[str], patterns: Sequence[str]) -> list[str]:
    return [
        name
        for name in files
        if name.endswith(".py")
        and _glob_match(name, patterns)
        and not _glob_match(name, cfg.scope_exclude)
    ]


def _uses_playwright(copy_root: Path, test_files: Iterable[str]) -> bool:
    for name in test_files:
        try:
            if "playwright" in (copy_root / name).read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def command_map(cfg: SurveyConfig) -> int:
    state = cfg.state
    state.mkdir(parents=True, exist_ok=True)
    print(f"[survey] state directory: {state}")
    write_support_files(state)
    results = state / "results.jsonl"
    if results.exists():
        # Results answer to the map they were run under; a new map starts a new log.
        retired = state / f"results-superseded-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
        results.rename(retired)
        print(f"[survey] earlier results set aside: {retired.name}")
    print("[survey] building the pristine hermetic copy ...")
    files = build_pristine(cfg.root, state / "pristine")
    compile_checked_hash(state / "pristine")
    manifest = {
        name: _sha256((state / "pristine" / name).read_bytes())
        for name in files
        if name.endswith(".py")
    }
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cfg.root, capture_output=True, text=True
    ).stdout.strip()
    _write_json(
        state / "meta.json",
        {
            "tool_version": TOOL_VERSION,
            "config": cfg.to_json(),
            "git_head": head,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "manifest": manifest,
        },
    )

    copy_root = state / "copies" / "map"
    clone_tree(state / "pristine", copy_root)
    scope_files = _scope_files(cfg, files, cfg.scope)
    observe_files = _scope_files(cfg, files, cfg.observe)
    measured_dirs = sorted({name.split("/", 1)[0] for name in scope_files + observe_files})
    data_file = state / "map" / "coverage.sqlite"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    for stale in data_file.parent.glob("coverage.sqlite*"):
        stale.unlink()
    rc_path = state / "map" / "coveragerc"
    rc_path.write_text(
        "[run]\n"
        f"data_file = {data_file}\n"
        "source =\n" + "".join(f"    {copy_root / name}\n" for name in measured_dirs),
        encoding="utf-8",
    )
    print(f"[survey] coverage run over {len(scope_files)} in-scope files ...")
    result = _run_pytest(
        cfg,
        copy_root,
        state / "map" / "run",
        {"mode": "map", "guard_playwright": True},
        [
            *cfg.pytest_args,
            "--cov",
            f"--cov-config={rc_path}",
            "--cov-context=test",
            "--cov-report=",
        ],
        budget_seconds=6 * 3600,
    )
    if result.get("exitstatus") is None:
        tail = (state / "map" / "run" / "pytest.out").read_text(errors="replace")[-2000:]
        raise SystemExit(f"[survey] the coverage run produced no result:\n{tail}")
    _write_json(state / "map" / "control.json", result)
    survey_map = build_map(cfg, copy_root, data_file, result, scope_files, observe_files)
    _write_json(state / "map" / "map.json", survey_map)
    counts = survey_map["counts"]
    print(
        f"[survey] tests: {counts['tests']} collected, {counts['tests_passed']} passed control; "
        f"functions in scope: {counts['functions']}, covered: {counts['functions_covered']}, "
        f"mutable and covered: {counts['functions_mutable_covered']}"
    )
    return 0


def build_map(
    cfg: SurveyConfig,
    copy_root: Path,
    data_file: Path,
    control: dict[str, Any],
    scope_files: Sequence[str],
    observe_files: Sequence[str],
) -> dict[str, Any]:
    import coverage

    data = coverage.CoverageData(basename=str(data_file))
    data.read()
    real_root = os.path.realpath(copy_root)
    measured: dict[str, str] = {}
    for filename in data.measured_files():
        real = os.path.realpath(filename)
        if real.startswith(real_root + os.sep):
            measured[real[len(real_root) + 1 :].replace(os.sep, "/")] = filename

    tests = sorted(control["tests"], key=lambda record: record["id"])
    index_of = {record["id"]: index for index, record in enumerate(tests)}
    # owner test index -> the tests that lean on a shared fixture it instantiated
    dependents: dict[int, list[int]] = {}
    for index, record in enumerate(tests):
        for owner in record.get("shared_fixture_owners", []):
            if owner in index_of:
                dependents.setdefault(index_of[owner], []).append(index)

    def covering(path: str, target: FunctionTarget) -> tuple[list[int], list[int], list[int]]:
        by_line = contexts_by_file.get(path)
        if by_line is None:
            by_line = contexts_by_file[path] = (
                data.contexts_by_lineno(measured[path]) if path in measured else {}
            )
        run: set[int] = set()
        setup: set[int] = set()
        for line in target.own_lines:
            for context in by_line.get(line, ()):
                nodeid, _, phase = context.rpartition("|")
                index = index_of.get(nodeid)
                if index is None:
                    continue
                if phase == "run":
                    run.add(index)
                elif phase == "setup":
                    setup.add(index)
        shared = {user for owner in setup for user in dependents.get(owner, [])}
        return sorted(run), sorted(setup - run), sorted(shared - run - setup)

    contexts_by_file: dict[str, dict[int, list[str]]] = {}
    functions: list[dict[str, Any]] = []
    observed_by_test: dict[int, int] = {}
    for path in scope_files:
        source = (copy_root / path).read_text(encoding="utf-8")
        for target in discover_functions(source, path):
            run, setup, shared = covering(path, target)
            functions.append(
                {
                    **{k: v for k, v in dataclasses.asdict(target).items() if k != "own_lines"},
                    "key": target.key,
                    "tier": cfg.tier_of(path),
                    "run": run,
                    "setup": setup,
                    "shared": shared,
                }
            )
        contexts_by_file.pop(path, None)
    for path in observe_files:
        source = (copy_root / path).read_text(encoding="utf-8")
        for target in discover_functions(source, path):
            for index in {i for group in covering(path, target) for i in group}:
                observed_by_test[index] = observed_by_test.get(index, 0) + 1
        contexts_by_file.pop(path, None)

    for index, record in enumerate(tests):
        record["observed_functions"] = observed_by_test.get(index, 0)
    covered = [f for f in functions if _covering(f)]
    return {
        "tests": tests,
        "functions": functions,
        "collection_errors": control.get("collection_errors", []),
        "counts": {
            "tests": len(tests),
            "tests_passed": sum(1 for record in tests if record["outcome"] == "passed"),
            "functions": len(functions),
            "functions_covered": len(covered),
            "functions_mutable_covered": sum(1 for f in covered if not f["skip_reason"]),
        },
    }


# ---------------------------------------------------------------------------
# run: the mutation runs
# ---------------------------------------------------------------------------


def _target_from_map(entry: dict[str, Any], source: str) -> FunctionTarget:
    for target in discover_functions(source, entry["path"]):
        if target.key == entry["key"]:
            return target
    raise LookupError(f"map entry no longer matches the pristine source: {entry['key']}")


def _expected_code(source: str, target: FunctionTarget) -> dict[str, Any]:
    """What the subprocess must find loaded: the target function compiled from this source."""
    name = target.qualname.rsplit(".", 1)[-1]
    return {
        "name": name,
        "first_line": target.first_line,
        "sha256": function_code_fingerprint(source, name, target.first_line),
    }


def _covering(function: dict[str, Any]) -> list[int]:
    """Tests that execute the function: in their body, their setup, or a shared fixture."""
    return function["run"] + function["setup"] + function.get("shared", [])


def _nodeid_file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _test_budget(duration: float) -> float:
    return max(20.0, 8.0 * float(duration) + 10.0)


class Survey:
    """The run phase: shared resolution state, worker copies, the results log."""

    def __init__(self, cfg: SurveyConfig, workers: int) -> None:
        self.cfg = cfg
        self.state = cfg.state
        self.map = _read_json(self.state / "map" / "map.json")
        self.meta = _read_json(self.state / "meta.json")
        self.tests: list[dict[str, Any]] = self.map["tests"]
        self.functions: list[dict[str, Any]] = self.map["functions"]
        self.results_path = self.state / "results.jsonl"
        self.lock = threading.Lock()
        self.resolved: set[int] = set()
        self.null_failures: dict[int, int] = {}
        self.done: set[str] = set()
        self.specificity = [0] * len(self.tests)
        for function in self.functions:
            if function["skip_reason"]:
                continue
            for index in _covering(function):
                self.specificity[index] += 1
        self.index_of = {record["id"]: index for index, record in enumerate(self.tests)}
        self.workers = workers
        self.keep_runs = False
        self.copies: list[Path] = []
        self._load_results()

    def _load_results(self) -> None:
        if not self.results_path.exists():
            return
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            self.done.add(record["mutant"])
            self._absorb(record)

    def _absorb(self, record: dict[str, Any]) -> None:
        for nodeid, outcome in record.get("tests", {}).items():
            index = self.index_of.get(nodeid)
            if index is None:
                continue
            if outcome.get("verdict") == "killed" and outcome.get("kind") == "assert":
                self.resolved.add(index)
            elif outcome.get("note") == "null_control_failed":
                self.null_failures[index] = self.null_failures.get(index, 0) + 1

    def unstable(self) -> set[int]:
        """Tests that keep failing a behavior-preserving edit: nothing to learn by rerunning."""
        return {i for i, count in self.null_failures.items() if count >= _UNSTABLE_AFTER}

    def mutants(self) -> list[tuple[dict[str, Any], str]]:
        ordered = sorted(self.functions, key=lambda f: (f["tier"], f["path"], f["lineno"]))
        plan: list[tuple[dict[str, Any], str]] = []
        for function in ordered:
            if function["skip_reason"] or not _covering(function):
                continue
            plan.append((function, "empty_body"))
            if function["bool_return"]:
                plan.append((function, "invert_bool"))
        return plan

    def prepare_copies(self) -> None:
        write_support_files(self.state)
        self._verify_pristine()
        if not (self.state / "pristine" / ".mutation-survey-bytecode").exists():
            compile_checked_hash(self.state / "pristine")
        for number in range(self.workers):
            copy_root = self.state / "copies" / f"w{number}"
            clone_tree(self.state / "pristine", copy_root)
            self.copies.append(copy_root)

    def _verify_pristine(self) -> None:
        for name, digest in self.meta["manifest"].items():
            if _sha256((self.state / "pristine" / name).read_bytes()) != digest:
                raise SystemExit(f"[survey] pristine copy drifted from its manifest: {name}")

    def select_tests(self, function: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
        """Primary (always run), secondary (probed until one fails), text-sensitive.

        Primary is every unresolved test, plus every test that reads this
        file's text: its failures are never trusted without the null control,
        so it cannot serve as a probe and must simply run.
        """
        path = function["path"]
        with self.lock:
            resolved = set(self.resolved)
            unstable = self.unstable() - resolved
        covering = [
            index
            for index in _covering(function)
            if self.tests[index]["outcome"] == "passed" and index not in unstable
        ]
        reads_this_file = {
            index for index in covering if path in self.tests[index].get("reads", [])
        }
        primary = [index for index in covering if index not in resolved or index in reads_this_file]
        probes = sorted(
            (index for index in covering if index in resolved and index not in reads_this_file),
            key=lambda index: (self.specificity[index], self.tests[index]["duration"]),
        )[:_LOCK_PROBE_CAP]
        ids = lambda indices: [self.tests[index]["id"] for index in indices]  # noqa: E731
        return ids(primary), ids(probes), ids(sorted(reads_this_file))

    def run_mutant(self, function: dict[str, Any], operator: str, copy_root: Path) -> dict[str, Any]:
        mutant_id = f"{function['key']}|{operator}"
        record: dict[str, Any] = {
            "mutant": mutant_id,
            "key": function["key"],
            "operator": operator,
            "tests": {},
        }
        path = function["path"]
        pristine = (self.state / "pristine" / path).read_bytes()
        try:
            target = _target_from_map(function, pristine.decode("utf-8"))
            mutated = build_mutant(pristine.decode("utf-8"), target, operator).encode("utf-8")
        except (MutantInvalid, LookupError) as exc:
            record["status"] = f"invalid: {exc}"
            return record
        primary, secondary, text_sensitive = self.select_tests(function)
        with self.lock:
            resolved_before = {self.tests[index]["id"] for index in self.resolved}
        record["covering"] = sum(
            1 for index in _covering(function) if self.tests[index]["outcome"] == "passed"
        )
        if not primary and not secondary:
            record["status"] = "no_passing_covering_tests"
            return record
        run_dir = self.state / "runs" / hashlib.sha256(mutant_id.encode()).hexdigest()[:16]
        target_file = copy_root / path
        try:
            target_file.write_bytes(mutated)
            expected_code = _expected_code(mutated.decode("utf-8"), target)
            remaining = list(primary)
            for _ in range(3):
                outcome = self._run_selection(
                    copy_root,
                    run_dir,
                    path,
                    mutated,
                    remaining,
                    secondary,
                    text_sensitive,
                    expected_code,
                )
                self._merge(record, outcome, remaining)
                hung = _hung_test(outcome)
                if not outcome["hard_timeout"] or hung is None:
                    break
                record["tests"][hung] = {"verdict": "killed", "kind": "timeout", "exc": "hard"}
                record["hard_timeouts"] = record.get("hard_timeouts", 0) + 1
                remaining = [t for t in remaining if t not in record["tests"]]
                secondary = []
                if not remaining:
                    break
            # No kill is credited on the mutant run alone. A test about to leave
            # the survey on its first assertion kill, and any test that reads
            # this file's text, must also pass a behavior-preserving edit of the
            # same function in the same company; otherwise its failure says
            # nothing about behavior (text pins, `git diff` guards, order
            # dependence inside a subset).
            suspects = [
                nodeid
                for nodeid, result in record["tests"].items()
                if result["verdict"] == "killed"
                and (
                    nodeid in text_sensitive
                    or (result.get("kind") == "assert" and nodeid not in resolved_before)
                )
            ]
            if suspects:
                self._null_control(record, function, operator, target, copy_root, run_dir, suspects)
        finally:
            target_file.write_bytes(pristine)
        if _sha256(target_file.read_bytes()) != _sha256(pristine):
            raise SystemExit(f"[survey] failed to restore {target_file}")
        record["status"] = record.get("status", "ran")
        imported = record.get("import", {})
        clean = (
            record["status"] == "ran"
            and imported.get("imported")
            and imported.get("disk_sha_ok")
            and imported.get("code_confirmed") is not False
        )
        if clean and not record.get("hard_timeouts") and not self.keep_runs:
            shutil.rmtree(run_dir, ignore_errors=True)  # keep the evidence only when odd
        return record

    def _run_selection(
        self,
        copy_root: Path,
        run_dir: Path,
        path: str,
        mutated: bytes,
        primary: list[str],
        secondary: list[str],
        text_sensitive: list[str],
        expected_code: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected = primary + secondary
        files = sorted({_nodeid_file(nodeid) for nodeid in selected})
        timeouts = {
            nodeid: _test_budget(self.tests[self.index_of[nodeid]]["duration"])
            for nodeid in selected
        }
        return _run_pytest(
            self.cfg,
            copy_root,
            run_dir,
            {
                "mode": "mutant",
                "primary": primary,
                "secondary": secondary,
                "text_sensitive": text_sensitive,
                "timeouts": timeouts,
                "target_path": path,
                "expected_sha256": _sha256(mutated),
                "expected_code": expected_code,
                "guard_playwright": _uses_playwright(copy_root, files),
            },
            [*_selection_args(self.cfg.pytest_args), *files],
            budget_seconds=min(3600.0, 90.0 + sum(timeouts[nodeid] for nodeid in primary) + 600.0),
        )

    def _merge(self, record: dict[str, Any], outcome: dict[str, Any], primary: list[str]) -> None:
        """Fold one subprocess result into the mutant's record.

        Every primary test must be accounted for. One that neither ran nor sat
        in a file that failed to collect has vanished under the mutation (its
        parameters are generated by the emptied code); the suite would stay
        green, so it counts as survived, and says so.
        """
        record["seconds"] = round(record.get("seconds", 0.0) + outcome.get("seconds", 0.0), 2)
        target = outcome.get("target")
        if target is not None:
            record["import"] = target
        if outcome.get("exitstatus") is None and not outcome["hard_timeout"]:
            record["status"] = "no_result"
        broken_files = {
            _nodeid_file(error["nodeid"]): error["message"]
            for error in outcome.get("collection_errors", [])
        }
        for test in outcome.get("tests", []):
            nodeid = test["id"]
            if test["outcome"] == "failed":
                record["tests"][nodeid] = {
                    "verdict": "killed",
                    "kind": test.get("kind", "crash"),
                    "exc": test.get("exc", ""),
                    "frame": test.get("frame", ""),
                    "msg": test.get("msg", ""),
                }
            elif test["outcome"] == "passed":
                record["tests"][nodeid] = {"verdict": "survived"}
            else:
                record["tests"][nodeid] = {"verdict": "survived", "note": test["outcome"]}
        if outcome.get("exitstatus") is None:
            return  # killed mid-run; the caller re-queues what never ran
        if outcome["exitstatus"] not in (0, 1) and not outcome.get("stopped_early"):
            # The session died on its own (internal error, usage error, an exit
            # raised inside a test). Unrun tests are unknown, never "survived".
            record["status"] = f"abnormal_exit:{outcome['exitstatus']}"
            return
        for nodeid in primary:
            if nodeid in record["tests"]:
                continue
            if _nodeid_file(nodeid) in broken_files:
                record["tests"][nodeid] = {
                    "verdict": "killed",
                    "kind": "collection",
                    "msg": broken_files[_nodeid_file(nodeid)][-240:],
                }
            else:
                record["tests"][nodeid] = {"verdict": "survived", "note": "vanished"}

    def _null_control(
        self,
        record: dict[str, Any],
        function: dict[str, Any],
        operator: str,
        target: FunctionTarget,
        copy_root: Path,
        run_dir: Path,
        suspects: list[str],
    ) -> None:
        """Rerun the killers against the same edit with its behavior change taken out."""
        path = function["path"]
        pristine = (self.state / "pristine" / path).read_text(encoding="utf-8")
        null_source = build_mutant(pristine, target, operator, null=True).encode("utf-8")
        (copy_root / path).write_bytes(null_source)
        outcome = self._run_selection(
            copy_root,
            run_dir,
            path,
            null_source,
            suspects,
            [],
            [],
            _expected_code(null_source.decode("utf-8"), target),
        )
        record["seconds"] = round(record.get("seconds", 0.0) + outcome.get("seconds", 0.0), 2)
        passed = {test["id"] for test in outcome.get("tests", []) if test["outcome"] == "passed"}
        for nodeid in suspects:
            if nodeid not in passed:
                record["tests"][nodeid] = {
                    "verdict": "survived",
                    "note": "null_control_failed",
                    "claimed": record["tests"][nodeid].get("kind", ""),
                }

    def append(self, record: dict[str, Any]) -> None:
        with self.lock:
            with open(self.results_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            self.done.add(record["mutant"])
            self._absorb(record)


def _selection_args(pytest_args: Sequence[str]) -> list[str]:
    """Project pytest args minus anything that names or ignores paths."""
    kept = [arg for arg in pytest_args if not arg.startswith(("--ignore", "--deselect"))]
    return [*kept, "--tb=no"]


def _hung_test(outcome: dict[str, Any]) -> str | None:
    started, ended = None, set()
    for line in outcome.get("progress", []):
        if line.startswith("START "):
            started = line[6:]
        elif line.startswith("END "):
            ended.add(line[4:].rsplit(" ", 1)[0])
    return started if started and started not in ended else None


def command_run(cfg: SurveyConfig, args: argparse.Namespace) -> int:
    print(f"[survey] state directory: {cfg.state}")
    survey = Survey(cfg, workers=args.workers)
    survey.keep_runs = args.keep_runs
    plan = [
        (function, operator)
        for function, operator in survey.mutants()
        if f"{function['key']}|{operator}" not in survey.done
        and (not args.only or args.only in function["key"])
        and (args.tier is None or function["tier"] == args.tier)
    ]
    if args.limit:
        plan = plan[: args.limit]
    print(
        f"[survey] {len(survey.done)} mutants already recorded, {len(plan)} to run, "
        f"{args.workers} workers"
    )
    if not plan:
        return 0
    survey.prepare_copies()
    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
    free = list(survey.copies)
    pending = iter(plan)
    in_flight: dict[concurrent.futures.Future[dict[str, Any]], Path] = {}
    completed = 0
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            while free and not stop.is_set() and (deadline is None or time.monotonic() < deadline):
                item = next(pending, None)
                if item is None:
                    break
                copy_root = free.pop()
                in_flight[pool.submit(survey.run_mutant, item[0], item[1], copy_root)] = copy_root
            if not in_flight:
                break
            finished, _ = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in finished:
                free.append(in_flight.pop(future))
                record = future.result()
                survey.append(record)
                completed += 1
                killed = sum(1 for t in record["tests"].values() if t["verdict"] == "killed")
                print(
                    f"[survey] {completed}/{len(plan)} {record['mutant']}: "
                    f"{record.get('status')}, {killed}/{len(record['tests'])} killed, "
                    f"{record.get('seconds', 0)}s",
                    flush=True,
                )
    remaining = len(plan) - completed
    print(f"[survey] run complete: {completed} mutants this session, {remaining} left")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _test_function_id(nodeid: str) -> str:
    return nodeid.split("[", 1)[0]


def build_report(cfg: SurveyConfig) -> dict[str, Any]:
    state = cfg.state
    survey_map = _read_json(state / "map" / "map.json")
    tests: list[dict[str, Any]] = survey_map["tests"]
    functions: list[dict[str, Any]] = survey_map["functions"]
    results: dict[str, dict[str, Any]] = {}
    if (state / "results.jsonl").exists():
        for line in (state / "results.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                results[record["mutant"]] = record

    expected: dict[int, int] = {}
    inconclusive: dict[int, int] = {}
    seen: dict[str, list[dict[str, Any]]] = {}
    function_rows: list[dict[str, Any]] = []
    for function in functions:
        covering = _covering(function)
        row = {
            "key": function["key"],
            "tier": function["tier"],
            "covering": len(covering),
            "bool_return": function["bool_return"],
        }
        if function["skip_reason"]:
            row["verdict"] = f"skipped:{function['skip_reason']}"
        elif not covering:
            row["verdict"] = "uncovered"
        else:
            operators = ["empty_body"] + (["invert_bool"] if function["bool_return"] else [])
            row["operators"] = {}
            for operator in operators:
                record = results.get(f"{function['key']}|{operator}")
                row["operators"][operator] = _function_operator_verdict(record, len(covering))
                status = (record or {}).get("status", "")
                if status.startswith("abnormal_exit") or status == "no_passing_covering_tests":
                    # Finished without a per-test answer (an emptied import-time
                    # function breaks collection itself): no test waits on it.
                    for index in covering:
                        inconclusive[index] = inconclusive.get(index, 0) + 1
                elif record is None or not status.startswith("invalid"):
                    # An invalid mutant is owed to no test; a missing one still is.
                    for index in covering:
                        expected[index] = expected.get(index, 0) + 1
                for nodeid, outcome in (record or {}).get("tests", {}).items():
                    seen.setdefault(nodeid, []).append(
                        {"mutant": record["mutant"], **outcome}  # type: ignore[index]
                    )
            row["verdict"] = row["operators"]["empty_body"]["verdict"]
        function_rows.append(row)

    test_rows: list[dict[str, Any]] = []
    for index, test in enumerate(tests):
        saw = seen.get(test["id"], [])
        kills = [entry for entry in saw if entry["verdict"] == "killed"]
        row: dict[str, Any] = {
            "id": test["id"],
            "control": test["outcome"],
            "reads": test.get("reads", []),
            "observed_functions": test.get("observed_functions", 0),
            "saw": len(saw),
            "kills": len(kills),
            "assert_kills": sum(1 for entry in kills if entry.get("kind") == "assert"),
            "null_control_failures": sum(
                1 for entry in saw if entry.get("note") == "null_control_failed"
            ),
        }
        if test["outcome"] != "passed":
            row["verdict"] = "no_control"
            row["reason"] = test.get("reason", test.get("kind", ""))
        elif not expected.get(index):
            row["verdict"] = "no_body"
        elif row["assert_kills"]:
            row["verdict"] = "lock"
            row["first_kill"] = next(e["mutant"] for e in kills if e.get("kind") == "assert")
        elif row["null_control_failures"] >= _UNSTABLE_AFTER:
            row["verdict"] = "unstable"
        elif len(saw) < expected[index]:
            row["verdict"] = "pending"
        elif kills:
            row["verdict"] = "crash_only"
            row["kill_kinds"] = sorted({entry.get("kind", "") for entry in kills})
        else:
            row["verdict"] = "survivor"
        row["expected"] = expected.get(index, 0)
        if inconclusive.get(index):
            row["inconclusive_mutants"] = inconclusive[index]
        test_rows.append(row)

    rank = {
        "lock": 6,
        "crash_only": 5,
        "pending": 4,
        "unstable": 3,
        "survivor": 2,
        "no_body": 1,
        "no_control": 0,
    }
    by_function: dict[str, str] = {}
    for row in test_rows:
        name = _test_function_id(row["id"])
        if name not in by_function or rank[row["verdict"]] > rank[by_function[name]]:
            by_function[name] = row["verdict"]

    def tally(values: Iterable[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "state": str(state),
        "git_head": _read_json(state / "meta.json").get("git_head", ""),
        "mutants_recorded": len(results),
        "test_items": tally(row["verdict"] for row in test_rows),
        "test_functions": tally(by_function.values()),
        "functions": tally(row["verdict"].split(":")[0] for row in function_rows),
        "tests": test_rows,
        "function_rows": function_rows,
    }


def _function_operator_verdict(record: dict[str, Any] | None, covering: int) -> dict[str, Any]:
    if record is None:
        return {"verdict": "pending"}
    if record.get("status", "").startswith("invalid"):
        return {"verdict": "skipped", "status": record["status"]}
    if record.get("status") == "no_passing_covering_tests":
        return {"verdict": "no_control", "ran": 0, "covering": covering}
    outcomes = record.get("tests", {})
    eligible = record.get("covering", covering)
    killers = [nodeid for nodeid, outcome in outcomes.items() if outcome["verdict"] == "killed"]
    imported = record.get("import", {})
    if killers:
        # An assertion kill outranks a crash: name the strongest lock found.
        killers.sort(key=lambda nodeid: outcomes[nodeid].get("kind") != "assert")
        return {
            "verdict": "locked",
            "killer": killers[0],
            "kind": outcomes[killers[0]].get("kind", ""),
            "killers": len(killers),
            "ran": len(outcomes),
        }
    if (
        not imported.get("imported", False)
        or not imported.get("disk_sha_ok", False)
        or imported.get("code_confirmed") is False
    ):
        return {"verdict": "unconfirmed", "import": imported, "ran": len(outcomes)}
    verdict = "unlocked" if len(outcomes) >= eligible else "unlocked_within_cap"
    return {"verdict": verdict, "ran": len(outcomes), "covering": eligible}


def command_report(cfg: SurveyConfig) -> int:
    report = build_report(cfg)
    _write_json(cfg.state / "report.json", report)
    lines = [
        f"mutation survey report — state {report['state']}",
        f"git head {report['git_head']}; mutants recorded: {report['mutants_recorded']}",
        "",
        "test items by verdict:      " + json.dumps(report["test_items"]),
        "test functions by verdict:  " + json.dumps(report["test_functions"]),
        "production functions:       " + json.dumps(report["functions"]),
    ]
    text = "\n".join(lines) + "\n"
    (cfg.state / "report.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[survey] full report: {cfg.state / 'report.json'}")
    return 0


def command_status(cfg: SurveyConfig) -> int:
    survey = Survey(cfg, workers=1)
    plan = survey.mutants()
    print(f"[survey] state directory: {cfg.state}")
    for tier, (name, _) in enumerate(cfg.tiers):
        total = [m for m in plan if m[0]["tier"] == tier]
        done = [m for m in total if f"{m[0]['key']}|{m[1]}" in survey.done]
        if total:
            print(f"  tier {tier} {name}: {len(done)}/{len(total)} mutants")
    print(f"  tests resolved as locks so far: {len(survey.resolved)}/{len(survey.tests)}")
    return 0


# ---------------------------------------------------------------------------
# prove: red under mutation, green under real code, for one named lock
# ---------------------------------------------------------------------------


def command_prove(cfg: SurveyConfig, args: argparse.Namespace) -> int:
    state = cfg.state
    print(f"[prove] state directory: {state}")
    write_support_files(state)
    copy_root = state / "prove" / "copy"
    build_pristine(cfg.root, copy_root)
    compile_checked_hash(copy_root)
    if args.function:
        path = args.function.split("::", 1)[0]
        source = (copy_root / path).read_text(encoding="utf-8")
        matches = [
            target
            for target in discover_functions(source, path)
            if target.key == args.function or target.key.rsplit("@", 1)[0] == args.function
        ]
        if len(matches) != 1:
            raise SystemExit(f"[prove] {len(matches)} functions match {args.function!r}")
        operator = "invert_bool" if args.invert else "empty_body"
        mutated = build_mutant(source, matches[0], operator)
        label = f"{matches[0].key} operator={operator}"
    elif args.empty_lines:
        path, _, span = args.empty_lines.partition(":")
        start, _, end = span.partition("-")
        source = (copy_root / path).read_text(encoding="utf-8")
        mutated = mutate_empty_lines(source, int(start), int(end or start))
        label = f"{path} lines {start}-{end or start} emptied"
    else:
        path, line, old, new = args.replace_line
        source = (copy_root / path).read_text(encoding="utf-8")
        mutated = mutate_replace_on_line(source, int(line), old, new)
        label = f"{path}:{line} {old!r} -> {new!r}"
    files = sorted({_nodeid_file(nodeid) for nodeid in args.tests})
    common = {
        "mode": "mutant",
        "primary": list(args.tests),
        "secondary": [],
        "default_timeout": 300,
        "guard_playwright": _uses_playwright(copy_root, files),
    }
    pytest_args = [*_selection_args(cfg.pytest_args), *files]
    control = _run_pytest(
        cfg, copy_root, state / "prove" / "control", common, pytest_args, budget_seconds=1800
    )
    (copy_root / path).write_text(mutated, encoding="utf-8")
    mutant = _run_pytest(
        cfg,
        copy_root,
        state / "prove" / "mutant",
        {**common, "target_path": path, "expected_sha256": _sha256(mutated.encode("utf-8"))},
        pytest_args,
        budget_seconds=1800,
    )
    print(f"[prove] target: {label}")
    verdict = 0
    for name, outcome in (("real code", control), ("mutant", mutant)):
        by_outcome: dict[str, int] = {}
        for test in outcome["tests"]:
            by_outcome[test["outcome"]] = by_outcome.get(test["outcome"], 0) + 1
        print(f"[prove] {name}: {json.dumps(by_outcome, sort_keys=True)}")
        for test in outcome["tests"]:
            if test["outcome"] == "failed":
                print(
                    f"    FAILED {test['id']} [{test.get('kind')}] "
                    f"{test.get('exc')}: {test.get('msg', '')[:160]}"
                )
    print(f"[prove] mutated file imported by the run: {json.dumps(mutant.get('target'))}")
    control_ids = {t["id"] for t in control["tests"] if t["outcome"] == "passed"}
    mutant_failed = {t["id"] for t in mutant["tests"] if t["outcome"] == "failed"}
    for nodeid in args.tests:
        green, red = nodeid in control_ids, nodeid in mutant_failed
        print(f"[prove] {nodeid}: {'GREEN' if green else 'NOT GREEN'} on real code, "
              f"{'RED' if red else 'NOT RED'} under mutation")
        if not (green and red):
            verdict = 1
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> SurveyConfig:
    root = Path(args.root).resolve()
    state = Path(args.state_dir).resolve() if args.state_dir else default_state_dir(root)
    if args.command != "map" and (state / "meta.json").exists():
        return dataclasses.replace(load_config(state), state=state)
    if args.scope:
        # A custom scope is a different project: none of the Cloris defaults apply.
        return SurveyConfig(
            root=root,
            state=state,
            scope=tuple(args.scope),
            observe=tuple(args.observe or ()),
            scope_exclude=(),
            pytest_args=tuple(args.pytest_arg or ("-q",)),
            tiers=(("all", tuple(args.scope)),),
        )
    return SurveyConfig(
        root=root,
        state=state,
        scope=DEFAULT_SCOPE,
        observe=tuple(args.observe or DEFAULT_OBSERVE),
        scope_exclude=DEFAULT_SCOPE_EXCLUDE,
        pytest_args=tuple(args.pytest_arg or DEFAULT_PYTEST_ARGS),
        tiers=DEFAULT_TIERS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mutation survey: find the tests that cannot fail")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="project to survey")
    parser.add_argument("--state-dir", default="", help="survey state (default: a temp dir)")
    parser.add_argument("--scope", action="append", help="glob of files to mutate (repeatable)")
    parser.add_argument("--observe", action="append", help="glob measured but never mutated")
    parser.add_argument(
        "--pytest-arg", action="append", help="pytest argument for the coverage run (repeatable)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("map", help="hermetic copy plus one coverage run with per-test contexts")
    run = commands.add_parser("run", help="resumable mutation runs in priority order")
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    run.add_argument("--limit", type=int, default=0, help="at most this many mutants")
    run.add_argument("--max-minutes", type=float, default=0.0, help="stop submitting after this")
    run.add_argument("--only", default="", help="only function keys containing this text")
    run.add_argument("--tier", type=int, default=None, help="only this priority tier")
    run.add_argument("--keep-runs", action="store_true", help="keep every run's raw output")
    commands.add_parser("report", help="verdicts per test and per function")
    commands.add_parser("status", help="progress by priority tier")
    prove = commands.add_parser("prove", help="red under mutation, green under real code")
    target = prove.add_mutually_exclusive_group(required=True)
    target.add_argument("--function", help="function key: path::qualname[@lineno]")
    target.add_argument("--empty-lines", help="path:START-END, whole statements to empty")
    target.add_argument(
        "--replace-line",
        nargs=4,
        metavar=("PATH", "LINE", "OLD", "NEW"),
        help="swap OLD for NEW on one line, to force a branch or remove a call",
    )
    prove.add_argument("--invert", action="store_true", help="boolean inversion, not empty body")
    prove.add_argument("--tests", nargs="+", required=True, help="pytest node ids")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config_from_args(args)
    if args.command == "map":
        return command_map(cfg)
    if args.command == "run":
        return command_run(cfg, args)
    if args.command == "report":
        return command_report(cfg)
    if args.command == "status":
        return command_status(cfg)
    return command_prove(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
