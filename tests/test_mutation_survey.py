"""tools/mutation_survey.py: the mutators, the safety envelope, and an
end-to-end survey of the planted project whose verdicts are known in advance.

The planted project (tests/fixtures/mutation_survey_planted) carries one test
per relationship a test can have to its subject. The survey must call the
planted tautology and the planted shallow test survivors and the planted real
lock a kill before any classification of the real suite means anything.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import mutation_survey as ms
from tools import mutation_survey_classify as classify

PLANTED_ROOT = Path(__file__).parent / "fixtures" / "mutation_survey_planted"
SUITE = "planted_suite.py"


def _target(source: str, qualname: str) -> ms.FunctionTarget:
    matches = [t for t in ms.discover_functions(source, "mod.py") if t.qualname == qualname]
    assert len(matches) == 1, [t.qualname for t in ms.discover_functions(source, "mod.py")]
    return matches[0]


def _load(source: str) -> dict:
    namespace: dict = {}
    exec(compile(source, "mod.py", "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------
# Empty-body operator
# ---------------------------------------------------------------------------

SIMPLE = '''\
def total(amounts):
    """Add them."""
    result = sum(amounts)
    return result
'''


def test_empty_body_returns_before_the_first_statement_and_keeps_every_character():
    mutated = ms.mutate_empty_body(SIMPLE, _target(SIMPLE, "total"))

    assert _load(SIMPLE)["total"]([1, 2]) == 3
    assert _load(mutated)["total"]([1, 2]) is None
    assert len(mutated.splitlines()) == len(SIMPLE.splitlines())
    assert mutated.replace("return None; ", "", 1) == SIMPLE


COMPOUND_WITH_DOCSTRING = '''\
def first_even(numbers):
    """Find one."""
    for number in numbers:
        if number % 2 == 0:
            return number
    return -1
'''

COMPOUND_WITHOUT_DOCSTRING = '''\
def first_even(numbers):
    for number in numbers:
        if number % 2 == 0:
            return number
    return -1
'''


def test_empty_body_before_a_compound_statement_rides_the_docstring_line():
    mutated = ms.mutate_empty_body(
        COMPOUND_WITH_DOCSTRING, _target(COMPOUND_WITH_DOCSTRING, "first_even")
    )

    assert _load(mutated)["first_even"]([1, 2]) is None
    assert len(mutated.splitlines()) == len(COMPOUND_WITH_DOCSTRING.splitlines())
    assert '"""Find one."""; return None' in mutated


def test_empty_body_before_a_compound_statement_without_docstring_adds_one_line():
    mutated = ms.mutate_empty_body(
        COMPOUND_WITHOUT_DOCSTRING, _target(COMPOUND_WITHOUT_DOCSTRING, "first_even")
    )

    assert _load(mutated)["first_even"]([1, 2]) is None
    assert len(mutated.splitlines()) == len(COMPOUND_WITHOUT_DOCSTRING.splitlines()) + 1
    for line in COMPOUND_WITHOUT_DOCSTRING.splitlines():
        assert line in mutated.splitlines()


GENERATORS = '''\
def evens(numbers):
    for number in numbers:
        if number % 2 == 0:
            yield number


async def aevens(numbers):
    for number in numbers:
        yield number


async def fetch(value):
    doubled = value * 2
    return doubled
'''


def test_an_emptied_generator_is_still_a_generator_that_yields_nothing():
    mutated = ms.mutate_empty_body(GENERATORS, _target(GENERATORS, "evens"))

    assert list(_load(GENERATORS)["evens"]([1, 2, 4])) == [2, 4]
    assert list(_load(mutated)["evens"]([1, 2, 4])) == []


def test_async_bodies_empty_the_same_way():
    async def drain(agen):
        return [item async for item in agen]

    emptied_agen = ms.mutate_empty_body(GENERATORS, _target(GENERATORS, "aevens"))
    emptied_coro = ms.mutate_empty_body(GENERATORS, _target(GENERATORS, "fetch"))

    assert asyncio.run(drain(_load(GENERATORS)["aevens"]([1, 2]))) == [1, 2]
    assert asyncio.run(drain(_load(emptied_agen)["aevens"]([1, 2]))) == []
    assert asyncio.run(_load(GENERATORS)["fetch"](4)) == 8
    assert asyncio.run(_load(emptied_coro)["fetch"](4)) is None


def test_one_line_functions_are_emptied_in_place():
    source = "def double(x): return x * 2\n"
    mutated = ms.mutate_empty_body(source, _target(source, "double"))

    assert mutated == "def double(x): return None; return x * 2\n"
    assert _load(mutated)["double"](2) is None
    assert _target(source, "double").def_line_shared is True


TRIVIAL = '''\
import abc
from typing import overload


class Base(abc.ABC):
    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    def hook(self):
        """Optional hook."""

    def stub(self): ...

    def nothing(self):
        return None

    @overload
    def typed(self, x: int) -> int:
        return x
'''


@pytest.mark.parametrize(
    "qualname, reason",
    [
        ("Base.run", "abstract_method"),
        ("Base.hook", "trivial_body"),
        ("Base.stub", "trivial_body"),
        ("Base.nothing", "trivial_body"),
        ("Base.typed", "overload_stub"),
    ],
)
def test_functions_with_nothing_to_empty_are_skipped_with_a_reason(qualname, reason):
    target = _target(TRIVIAL, qualname)

    assert target.skip_reason == reason
    with pytest.raises(ms.MutantInvalid):
        ms.mutate_empty_body(TRIVIAL, target)


NESTED = '''\
class Outer:
    def method(self, items):
        def helper(item):
            return item * 2

        doubled = [helper(item) for item in items]
        return doubled
'''


def test_nested_functions_are_their_own_targets_with_their_own_lines():
    targets = {t.qualname: t for t in ms.discover_functions(NESTED, "mod.py")}

    assert set(targets) == {"Outer.method", "Outer.method.<locals>.helper"}
    assert targets["Outer.method.<locals>.helper"].own_lines == (4,)
    # The outer function owns the nested def line, not the nested body.
    assert targets["Outer.method"].own_lines == (3, 5, 6, 7)


# ---------------------------------------------------------------------------
# Boolean-inversion operator
# ---------------------------------------------------------------------------

PREDICATES = '''\
def annotated(value) -> bool:
    if value is None:
        return False
    return (
        value > 0
        and value < 10
    )


def inferred(value):
    return isinstance(value, int) and value > 0


def not_a_predicate(value):
    return value or None


def optional_predicate(value):
    if value is None:
        return
    return value > 0
'''


def test_bool_functions_are_found_by_annotation_or_by_what_they_return():
    kinds = {t.qualname: t.bool_return for t in ms.discover_functions(PREDICATES, "mod.py")}

    assert kinds == {
        "annotated": "annotated",
        "inferred": "inferred",
        "not_a_predicate": "",
        "optional_predicate": "",
    }


def test_inversion_flips_every_returned_value_across_lines():
    mutated = ms.mutate_invert_bool(PREDICATES, _target(PREDICATES, "annotated"))
    real, inverted = _load(PREDICATES)["annotated"], _load(mutated)["annotated"]

    assert [real(v) for v in (None, 5, 50)] == [False, True, False]
    assert [inverted(v) for v in (None, 5, 50)] == [True, False, True]
    assert len(mutated.splitlines()) == len(PREDICATES.splitlines())


def test_the_null_controls_change_text_but_not_behavior():
    null_inversion = ms.mutate_invert_bool(PREDICATES, _target(PREDICATES, "inferred"), null=True)
    null_empty = ms.mutate_empty_body(SIMPLE, _target(SIMPLE, "total"), null=True)

    assert null_inversion != PREDICATES and null_empty != SIMPLE
    assert _load(null_inversion)["inferred"](3) is True
    assert _load(null_empty)["total"]([1, 2]) == 3


def test_inversion_refuses_a_function_that_does_not_return_bool():
    with pytest.raises(ms.MutantInvalid):
        ms.mutate_invert_bool(PREDICATES, _target(PREDICATES, "not_a_predicate"))


BRANCHY = '''\
def settle(state):
    if state.failed:
        state.retries += 1
        state.backoff()
    else:
        state.retries = 0
    return state
'''


def test_empty_lines_replaces_whole_statements_and_keeps_the_line_count():
    mutated = ms.mutate_empty_lines(BRANCHY, 3, 4)

    assert mutated.splitlines()[2].strip() == "pass"
    assert mutated.splitlines()[3] == ""
    assert "state.retries = 0" in mutated
    assert len(mutated.splitlines()) == len(BRANCHY.splitlines())
    with pytest.raises(ms.MutantInvalid):
        ms.mutate_empty_lines(BRANCHY, 40, 41)


def test_replace_on_line_forces_a_branch_without_moving_a_line():
    forced = ms.mutate_replace_on_line(BRANCHY, 2, "state.failed", "False")

    assert forced.splitlines()[1] == "    if False:"
    assert len(forced.splitlines()) == len(BRANCHY.splitlines())
    with pytest.raises(ms.MutantInvalid, match="exactly once"):
        ms.mutate_replace_on_line(BRANCHY, 2, "state.missing", "False")
    with pytest.raises(ms.MutantInvalid, match="exactly once"):
        ms.mutate_replace_on_line(BRANCHY, 99, "state.failed", "False")


# ---------------------------------------------------------------------------
# The subprocess envelope
# ---------------------------------------------------------------------------


def test_subprocess_env_strips_secrets_and_points_chrome_at_a_dead_port(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-not-a-real-token")
    monkeypatch.setenv("COVERAGE_PROCESS_START", "/somewhere/.coveragerc")

    env = ms.subprocess_env(tmp_path / "state", tmp_path / "copy", tmp_path / "tmp")

    assert "ANTHROPIC_API_KEY" not in env and "GITHUB_TOKEN" not in env
    assert "COVERAGE_PROCESS_START" not in env
    assert env["CDP_URL"] == "http://127.0.0.1:9"
    assert env["HOME"] == str(tmp_path / "state" / "home")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONPATH"].split(":")[1] == str(tmp_path / "copy")


def test_the_network_guard_refuses_outside_hosts_and_the_chrome_debug_port(tmp_path):
    ms.write_support_files(tmp_path)
    probe = (
        "import socket\n"
        "def attempt(address):\n"
        "    s = socket.socket(); s.settimeout(0.2)\n"
        "    try:\n"
        "        s.connect(address)\n"
        "    except Exception as exc:\n"
        "        return type(exc).__name__\n"
        "    finally:\n"
        "        s.close()\n"
        "    return 'connected'\n"
        "print(attempt(('192.0.2.1', 80)), attempt(('127.0.0.1', 9222)), attempt(('127.0.0.1', 9)))\n"
    )

    done = subprocess.run(
        [sys.executable, "-c", probe],
        env={"PYTHONPATH": str(tmp_path / "support"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )

    outside, debug_port, loopback = done.stdout.split()
    assert outside == "MutationSurveyNetworkBlocked", done.stderr
    assert debug_port == "MutationSurveyNetworkBlocked"
    # Ordinary loopback traffic reaches the OS, which refuses the closed port itself.
    assert loopback != "MutationSurveyNetworkBlocked"


# ---------------------------------------------------------------------------
# Selection, bookkeeping and verdict logic
# ---------------------------------------------------------------------------


def test_scope_globs_span_zero_or_more_directories():
    assert ms._glob_match("shared/judger.py", ["shared/**/*.py"])
    assert ms._glob_match("shared/judgment/templates.py", ["shared/**/*.py"])
    assert ms._glob_match("tools/mutation_survey.py", ["tools/*.py"])
    assert not ms._glob_match("tools/one_offs/recover_saves.py", ["tools/*.py"])
    assert not ms._glob_match("cloris/app.py", ["shared/**/*.py", "tools/*.py"])


def _survey_with_map(tmp_path, tests, functions):
    state = tmp_path / "state"
    cfg = ms.SurveyConfig(
        root=tmp_path,
        state=state,
        scope=("pkg/**/*.py",),
        observe=(),
        scope_exclude=(),
        pytest_args=("-q",),
        tiers=(("all", ("pkg/**/*.py",)),),
    )
    ms._write_json(state / "meta.json", {"config": cfg.to_json(), "manifest": {}})
    ms._write_json(state / "map" / "map.json", {"tests": tests, "functions": functions})
    return ms.Survey(cfg, workers=1)


def _function(key, *, run=(), setup=(), shared=()):
    return {
        "key": key,
        "path": key.split("::", 1)[0],
        "lineno": 1,
        "tier": 0,
        "skip_reason": "",
        "bool_return": "",
        "run": list(run),
        "setup": list(setup),
        "shared": list(shared),
    }


def test_selection_runs_the_unresolved_and_probes_the_resolved_most_specific_first(tmp_path):
    tests = [
        {"id": "t::broad", "outcome": "passed", "duration": 0.1},
        {"id": "t::narrow", "outcome": "passed", "duration": 0.1},
        {"id": "t::unresolved", "outcome": "passed", "duration": 0.1},
        {"id": "t::skipped", "outcome": "skipped", "duration": 0.0},
        {"id": "t::reader", "outcome": "passed", "duration": 0.1, "reads": ["pkg/a.py"]},
    ]
    target = _function("pkg/a.py::f@1", run=[0, 1, 2, 3], shared=[4])
    noise = [_function(f"pkg/b.py::g{n}@1", run=[0]) for n in range(3)]
    survey = _survey_with_map(tmp_path, tests, [target, *noise])
    survey.resolved = {0, 1, 4}

    primary, secondary, text_sensitive = survey.select_tests(target)

    # The reader is resolved, yet it still runs: its kills need the null control.
    assert primary == ["t::unresolved", "t::reader"]
    assert secondary == ["t::narrow", "t::broad"]
    assert text_sensitive == ["t::reader"]


def test_a_test_resolves_only_on_an_assertion_kill(tmp_path):
    tests = [{"id": f"t::{n}", "outcome": "passed", "duration": 0.1} for n in "abc"]
    survey = _survey_with_map(tmp_path, tests, [_function("pkg/a.py::f@1", run=[0, 1, 2])])

    survey.append(
        {
            "mutant": "pkg/a.py::f@1|empty_body",
            "tests": {
                "t::a": {"verdict": "killed", "kind": "assert"},
                "t::b": {"verdict": "killed", "kind": "crash"},
                "t::c": {"verdict": "survived"},
            },
        }
    )

    assert survey.resolved == {0}
    assert ms.Survey(survey.cfg, workers=1).resolved == {0}, "resolution must survive a resume"


def test_a_test_that_keeps_failing_the_null_control_stops_being_rerun(tmp_path):
    tests = [{"id": f"t::{n}", "outcome": "passed", "duration": 0.1} for n in "ab"]
    functions = [_function(f"pkg/a.py::f{n}@1", run=[0, 1]) for n in range(4)]
    survey = _survey_with_map(tmp_path, tests, functions)

    for function in functions[:3]:
        assert survey.select_tests(function)[0] == ["t::a", "t::b"]
        survey.append(
            {
                "mutant": f"{function['key']}|empty_body",
                "tests": {
                    "t::a": {"verdict": "survived", "note": "null_control_failed"},
                    "t::b": {"verdict": "survived"},
                },
            }
        )

    assert survey.select_tests(functions[3])[0] == ["t::b"]


def test_merge_accounts_for_every_primary_test(tmp_path):
    survey = _survey_with_map(tmp_path, [], [])
    record = {"tests": {}}
    outcome = {
        "exitstatus": 1,
        "hard_timeout": False,
        "seconds": 1.0,
        "tests": [{"id": "tests/test_a.py::ran", "outcome": "passed"}],
        "collection_errors": [{"nodeid": "tests/test_b.py", "message": "ImportError: boom"}],
    }

    survey._merge(
        record, outcome, ["tests/test_a.py::ran", "tests/test_a.py::gone[x]", "tests/test_b.py::t"]
    )

    assert record["tests"]["tests/test_a.py::ran"] == {"verdict": "survived"}
    assert record["tests"]["tests/test_a.py::gone[x]"] == {"verdict": "survived", "note": "vanished"}
    assert record["tests"]["tests/test_b.py::t"]["kind"] == "collection"
    assert record["tests"]["tests/test_b.py::t"]["verdict"] == "killed"


def test_a_session_that_died_on_its_own_leaves_its_unrun_tests_unknown(tmp_path):
    survey = _survey_with_map(tmp_path, [], [])
    died = {"exitstatus": 3, "hard_timeout": False, "seconds": 1.0, "tests": []}
    stopped = {**died, "exitstatus": 2, "stopped_early": True}

    record = {"tests": {}}
    survey._merge(record, died, ["tests/test_a.py::never_ran"])
    assert record["tests"] == {} and record["status"] == "abnormal_exit:3"

    # The plugin's own early stop also exits 2, and there an unrun primary did vanish.
    record = {"tests": {}}
    survey._merge(record, stopped, ["tests/test_a.py::never_ran"])
    assert record["tests"]["tests/test_a.py::never_ran"]["note"] == "vanished"


def test_the_report_counts_a_parametrized_test_once_by_its_strongest_instance(tmp_path):
    tests = [
        {"id": "t.py::test_case[kills]", "outcome": "passed", "duration": 0.1},
        {"id": "t.py::test_case[survives]", "outcome": "passed", "duration": 0.1},
        {"id": "t.py::test_other", "outcome": "skipped", "duration": 0.0, "reason": "no data"},
    ]
    survey = _survey_with_map(tmp_path, tests, [_function("pkg/a.py::f@1", run=[0, 1, 2])])
    survey.append(
        {
            "mutant": "pkg/a.py::f@1|empty_body",
            "status": "ran",
            "covering": 2,
            "import": {"imported": True, "disk_sha_ok": True},
            "tests": {
                "t.py::test_case[kills]": {"verdict": "killed", "kind": "assert"},
                "t.py::test_case[survives]": {"verdict": "survived"},
            },
        }
    )

    report = ms.build_report(survey.cfg)

    assert report["test_items"] == {"lock": 1, "no_control": 1, "survivor": 1}
    assert report["test_functions"] == {"lock": 1, "no_control": 1}
    assert report["functions"] == {"locked": 1}


def test_a_mutant_that_broke_collection_keeps_no_test_waiting(tmp_path):
    tests = [{"id": "t.py::test_case", "outcome": "passed", "duration": 0.1}]
    functions = [
        _function("pkg/a.py::runs_at_import@1", run=[0]),
        _function("pkg/a.py::ordinary@9", run=[0]),
    ]
    survey = _survey_with_map(tmp_path, tests, functions)
    survey.append(
        {
            "mutant": "pkg/a.py::runs_at_import@1|empty_body",
            "status": "abnormal_exit:2",
            "covering": 1,
            "import": {"imported": False, "disk_sha_ok": True},
            "tests": {},
        }
    )
    survey.append(
        {
            "mutant": "pkg/a.py::ordinary@9|empty_body",
            "status": "ran",
            "covering": 1,
            "import": {"imported": True, "disk_sha_ok": True},
            "tests": {"t.py::test_case": {"verdict": "survived"}},
        }
    )

    report = ms.build_report(survey.cfg)

    assert report["test_items"] == {"survivor": 1}
    assert report["tests"][0]["inconclusive_mutants"] == 1
    assert report["functions"] == {"unconfirmed": 1, "unlocked": 1}


def test_default_tiers_put_the_save_path_first_and_tools_last():
    cfg = ms.SurveyConfig(
        root=Path("."),
        state=Path("."),
        scope=ms.DEFAULT_SCOPE,
        observe=ms.DEFAULT_OBSERVE,
        scope_exclude=ms.DEFAULT_SCOPE_EXCLUDE,
        pytest_args=ms.DEFAULT_PYTEST_ARGS,
        tiers=ms.DEFAULT_TIERS,
    )

    order = [
        cfg.tier_of(path)
        for path in (
            "linkedin/side_effects.py",
            "shared/save_destination/cloris_workspace.py",
            "shared/runtime_state/store.py",
            "linkedin/evaluation_loop.py",
            "shared/governor.py",
            "linkedin/geography_gate.py",
            "shared/judger.py",
            "linkedin/browser.py",
            "shared/config.py",
            "tools/iterate_brief.py",
        )
    ]

    assert order == sorted(order) and order[0] == order[1] == 0
    assert len(set(order)) == 9
    assert cfg.tier_of("somewhere/else.py") == len(ms.DEFAULT_TIERS)


def test_purge_pycache_and_the_playwright_sniff(tmp_path):
    cache = tmp_path / "pkg" / "sub" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "mod.cpython-312.pyc").write_bytes(b"stale")
    (tmp_path / "test_browser.py").write_text("from rebrowser_playwright.async_api import x\n")
    (tmp_path / "test_plain.py").write_text("import json\n")

    ms.purge_pycache(tmp_path)

    assert not cache.exists() and (tmp_path / "pkg" / "sub").exists()
    assert ms._uses_playwright(tmp_path, ["test_browser.py", "test_plain.py"]) is True
    assert ms._uses_playwright(tmp_path, ["test_plain.py", "missing.py"]) is False


def test_import_confirmation_names_the_file_that_was_loaded_and_checks_its_bytes(tmp_path):
    import importlib.util

    source = tmp_path / "pkg_under_test.py"
    source.write_text("VALUE = 1\n")
    spec = importlib.util.spec_from_file_location("pkg_under_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pkg_under_test"] = module
    try:
        spec.loader.exec_module(module)
        expected = ms._sha256(source.read_bytes())

        confirmed = ms._confirm_import(str(tmp_path), "pkg_under_test.py", expected)
        wrong_bytes = ms._confirm_import(str(tmp_path), "pkg_under_test.py", "0" * 64)
        never_loaded = ms._confirm_import(str(tmp_path), "absent.py", expected)
    finally:
        del sys.modules["pkg_under_test"]

    assert expected == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert confirmed["imported"] and confirmed["disk_sha_ok"]
    assert confirmed["module_file"] == str(source)
    assert wrong_bytes["imported"] and not wrong_bytes["disk_sha_ok"]
    assert not never_loaded["imported"] and not never_loaded["disk_sha_ok"]


def test_a_mutant_that_does_not_compile_is_refused():
    target = _target(SIMPLE, "total")

    ms._require_compiles(SIMPLE, target)
    with pytest.raises(ms.MutantInvalid, match="does not compile"):
        ms._require_compiles("def total(amounts):\n    return None; )\n", target)


def test_a_drifted_pristine_copy_stops_the_run(tmp_path):
    survey = _survey_with_map(tmp_path, [], [])
    pristine = survey.state / "pristine" / "pkg"
    pristine.mkdir(parents=True)
    (pristine / "a.py").write_text("VALUE = 1\n")
    survey.meta["manifest"] = {"pkg/a.py": hashlib.sha256(b"VALUE = 1\n").hexdigest()}

    survey._verify_pristine()
    (pristine / "a.py").write_text("VALUE = 2\n")
    with pytest.raises(SystemExit, match="drifted"):
        survey._verify_pristine()


DECORATED = '''\
import functools


def traced(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class Book:
    @traced
    def total(self, amounts):
        def keep(amount):
            return amount > 0

        return sum(a for a in amounts if keep(a))

    @staticmethod
    def label():
        return "book"

    @property
    def size(self):
        return 3
'''


@pytest.mark.parametrize(
    "qualname", ["Book.total", "Book.total.<locals>.keep", "Book.label", "Book.size", "traced"]
)
def test_the_loaded_function_is_proven_to_be_the_mutant(tmp_path, qualname):
    import importlib.util

    target = _target(DECORATED, qualname)
    mutated = ms.mutate_empty_body(DECORATED, target)
    module_file = tmp_path / "book_under_test.py"
    module_file.write_text(mutated)
    spec = importlib.util.spec_from_file_location("book_under_test", module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["book_under_test"] = module
    try:
        spec.loader.exec_module(module)
        digest = hashlib.sha256(mutated.encode()).hexdigest()
        holds_mutant = ms._confirm_import(
            str(tmp_path), "book_under_test.py", digest, ms._expected_code(mutated, target)
        )
        # The same process checked against the pristine bytecode: stale code is caught.
        holds_pristine = ms._confirm_import(
            str(tmp_path), "book_under_test.py", digest, ms._expected_code(DECORATED, target)
        )
    finally:
        del sys.modules["book_under_test"]

    assert ms._expected_code(mutated, target)["sha256"] != ms._expected_code(DECORATED, target)["sha256"]
    assert holds_mutant["code_confirmed"] is True
    assert holds_pristine["code_confirmed"] is False


def test_the_copy_carries_only_bytecode_that_cannot_go_stale(tmp_path):
    (tmp_path / "pkg").mkdir()
    source = tmp_path / "pkg" / "mod.py"
    source.write_text("VALUE = 1\n")
    stale = tmp_path / "pkg" / "__pycache__"
    stale.mkdir()
    (stale / "mod.cpython-312.pyc").write_bytes(b"stale timestamp pyc")

    ms.compile_checked_hash(tmp_path)

    pycs = list(stale.glob("mod.*.pyc"))
    assert len(pycs) == 1
    flags = int.from_bytes(pycs[0].read_bytes()[4:8], "little")
    assert flags == 0b11, "hash-based and checked against the source"


def test_a_run_killed_mid_test_names_the_test_that_hung():
    progress = ["START t::a", "END t::a passed", "START t::b"]

    assert ms._hung_test({"progress": progress}) == "t::b"
    assert ms._hung_test({"progress": progress + ["END t::b failed"]}) is None


def test_function_verdicts_prefer_an_assertion_kill_and_admit_what_was_not_run():
    confirmed = {"imported": True, "disk_sha_ok": True}
    killed = {
        "status": "ran",
        "covering": 2,
        "import": confirmed,
        "tests": {
            "t::crashes": {"verdict": "killed", "kind": "crash"},
            "t::asserts": {"verdict": "killed", "kind": "assert"},
        },
    }
    survived = {"status": "ran", "covering": 3, "import": confirmed, "tests": {"t::a": {"verdict": "survived"}}}

    locked = ms._function_operator_verdict(killed, 2)
    assert (locked["verdict"], locked["killer"], locked["kind"]) == ("locked", "t::asserts", "assert")
    assert ms._function_operator_verdict(survived, 3)["verdict"] == "unlocked_within_cap"
    assert ms._function_operator_verdict({**survived, "covering": 1}, 3)["verdict"] == "unlocked"
    assert ms._function_operator_verdict({**survived, "import": {}}, 3)["verdict"] == "unconfirmed"
    assert ms._function_operator_verdict(None, 3)["verdict"] == "pending"
    assert ms._function_operator_verdict({"status": "invalid: x"}, 3)["verdict"] == "skipped"


# ---------------------------------------------------------------------------
# End to end: the planted project
# ---------------------------------------------------------------------------


def _survey(state: Path, *command: str) -> int:
    return ms.main(
        [
            "--root",
            str(PLANTED_ROOT),
            "--state-dir",
            str(state),
            "--scope",
            "planted_pkg/**/*.py",
            "--pytest-arg=-q",
            f"--pytest-arg={SUITE}",
            *command,
        ]
    )


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    state = tmp_path_factory.mktemp("planted-survey")
    assert _survey(state, "map") == 0
    # Two sessions, so the second one proves the survey resumes.
    assert _survey(state, "run", "--workers", "3", "--limit", "4") == 0
    first_session = (state / "results.jsonl").read_text().splitlines()
    assert _survey(state, "run", "--workers", "3") == 0
    assert _survey(state, "report") == 0
    report = json.loads((state / "report.json").read_text())
    records = [json.loads(line) for line in (state / "results.jsonl").read_text().splitlines()]
    assert classify.main(["--state-dir", str(state), "--draws", "2"]) == 0
    classification = json.loads((state / "classification.json").read_text())
    return {
        "state": state,
        "first_session": first_session,
        "records": records,
        "classification": classification,
        "classes": {
            row["test"].split("::", 1)[1]: row["class"] for row in classification["rows"]
        },
        "tests": {row["id"].split("::", 1)[1]: row for row in report["tests"]},
        "functions": {row["key"].split("::", 1)[1]: row for row in report["function_rows"]},
    }


def test_the_planted_tautology_and_the_planted_shallow_test_survive(planted):
    tautology = planted["tests"]["test_planted_tautology_success_resets_the_failures"]
    shallow = planted["tests"]["test_planted_shallow_settle_leaves_status_ok"]

    assert tautology["verdict"] == "survivor" and tautology["saw"] == 1
    assert shallow["verdict"] == "survivor" and shallow["saw"] == 1
    # The reset the tautology claims to test is locked by nothing.
    assert planted["functions"]["Ledger.record_failure@30"]["verdict"] == "unlocked"


def test_the_planted_real_locks_kill_with_their_own_assertions(planted):
    for name in (
        "test_planted_real_lock_settle_totals_the_amounts",
        "test_planted_real_lock_settle_later_returns_the_balance",
        "test_planted_real_lock_iter_amounts_skips_zeroes",
    ):
        row = planted["tests"][name]
        assert row["verdict"] == "lock", name
        assert row["assert_kills"] >= 1, name
    settle = planted["functions"]["Ledger.settle@23"]["operators"]["empty_body"]
    assert settle["verdict"] == "locked" and settle["kind"] == "assert"


def test_a_test_that_only_crashes_is_not_counted_as_a_lock(planted):
    smoke = planted["tests"]["test_planted_crash_only_audit_runs"]
    copycat = planted["tests"]["test_planted_reimplements_its_subject_and_asserts_on_the_copy"]

    # Emptied code blew up inside production code: the test asserts nothing.
    assert smoke["verdict"] == "crash_only"
    assert smoke["kills"] == 1 and smoke["assert_kills"] == 0
    assert smoke["kill_kinds"] == ["crash"]
    # Emptied code broke an ordinary line of the test; its assertions never fired.
    assert copycat["verdict"] == "crash_only"
    assert copycat["kill_kinds"] == ["test_crash"]


def test_a_negative_case_is_locked_only_by_the_inversion_operator(planted):
    row = planted["tests"]["test_planted_negative_case_not_overdrawn"]
    operators = planted["functions"]["Ledger.is_overdrawn@36"]["operators"]

    assert row["verdict"] == "lock"
    assert row["first_kill"].endswith("Ledger.is_overdrawn@36|invert_bool")
    assert operators["empty_body"]["verdict"] == "unlocked"
    assert operators["invert_bool"]["verdict"] == "locked"


def test_source_reading_tests_are_flagged_and_never_credited_with_a_text_kill(planted):
    structural = planted["tests"]["test_planted_structural_settle_keeps_its_rounding"]
    text_pin = planted["tests"]["test_planted_text_pin_record_success_source"]

    assert structural["verdict"] == "no_body"
    assert structural["reads"] == ["planted_pkg/ledger.py"]
    # The pin fails under the mutant and under the behavior-preserving null
    # control alike, so its failure is an artifact of the text, not a kill.
    assert text_pin["verdict"] == "survivor"
    assert text_pin["null_control_failures"] == 1 and text_pin["kills"] == 0


def test_tests_behind_a_shared_fixture_inherit_the_coverage_of_its_work(planted):
    first = planted["tests"]["test_planted_shared_fixture_first_user_checks_a_proxy"]
    second = planted["tests"]["test_planted_shared_fixture_second_user_reads_the_balance"]

    # Coverage alone would call the second user bodiless and never mutate under it.
    assert first["verdict"] == "survivor"
    assert second["verdict"] == "lock"
    assert second["first_kill"].endswith("Ledger.settle@23|empty_body")


def test_skipped_tests_have_no_control_and_uncovered_functions_are_a_separate_count(planted):
    assert planted["tests"]["test_planted_unreachable_skipped"]["verdict"] == "no_control"
    assert planted["functions"]["never_called@58"]["verdict"] == "uncovered"


def test_every_mutant_ran_against_the_file_in_the_hermetic_copy(planted):
    copies = str(planted["state"] / "copies")

    assert len(planted["records"]) == 9
    for record in planted["records"]:
        confirmation = record["import"]
        assert confirmation["imported"] and confirmation["disk_sha_ok"], record["mutant"]
        assert confirmation["module_file"].startswith(copies), record["mutant"]
        assert not confirmation["module_file"].startswith(str(PLANTED_ROOT))
        # Direct proof: the function object the subprocess held was the mutant.
        assert confirmation["code_confirmed"] is True, record["mutant"]
    # The repo's own copy of the planted module was never touched.
    assert "return None; " not in (PLANTED_ROOT / "planted_pkg" / "ledger.py").read_text()


def test_the_hermetic_copy_is_a_repository_of_its_own(planted):
    pristine = planted["state"] / "pristine"

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=pristine, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    # Tests that ask git for HEAD get an answer, and it is never the real repository's.
    assert Path(git("rev-parse", "--show-toplevel")).resolve() == pristine.resolve()
    assert git("rev-list", "--count", "HEAD") == "1"
    assert git("status", "--porcelain") == ""
    assert git("remote") == ""


def test_the_survey_resumes_without_repeating_a_mutant(planted):
    mutants = [record["mutant"] for record in planted["records"]]

    assert len(planted["first_session"]) == 4
    assert len(mutants) == len(set(mutants)) == 9
    assert [json.loads(line)["mutant"] for line in planted["first_session"]] == mutants[:4]


def test_prove_is_red_under_mutation_and_green_under_real_code(planted, capsys):
    lock = f"{SUITE}::test_planted_real_lock_settle_totals_the_amounts"
    tautology = f"{SUITE}::test_planted_tautology_success_resets_the_failures"
    state = planted["state"]

    proven = _survey(state, "prove", "--function", "planted_pkg/ledger.py::Ledger.settle", "--tests", lock)
    refuted = _survey(
        state, "prove", "--function", "planted_pkg/ledger.py::Ledger.record_failure", "--tests", tautology
    )

    output = capsys.readouterr().out
    assert proven == 0 and refuted == 1
    assert f"{lock}: GREEN on real code, RED under mutation" in output
    assert f"{tautology}: GREEN on real code, NOT RED under mutation" in output


# ---------------------------------------------------------------------------
# Classification of the tests never seen to fail
# ---------------------------------------------------------------------------

CLASSIFIED_TESTS = '''\
import subprocess
import sys
from pathlib import Path

from shared import config
from linkedin import session_orchestrator as so
from linkedin.side_effects import save
from tests.intake_conversation.voice_asserts import assert_no_bullets

DAY_CYCLE_SOURCE = Path(so.__file__).read_text()
LAUNCHER = Path("tools") / "launch_exa.sh"
SEED_BRIEF = Path("config") / "seed-brief.json"


def _assert_saved(store):
    assert store.rows == ["x"]


def test_reads_a_module_constant_of_source():
    assert "dormant gap" in DAY_CYCLE_SOURCE


def test_pins_a_default():
    assert config.LINKEDIN_CAMPAIGN_PERSIST is False


def test_negative_only(caplog):
    assert save(None) is None
    assert not save([])


def test_proxy_only(capsys, sink):
    save("x")
    out = capsys.readouterr().out
    assert "saved" in out
    sink.write.assert_called_once_with("x")


def test_asserts_through_a_helper(store):
    save("x")
    _assert_saved(store)


def test_asserts_nothing():
    save("x")


def test_tautology(state):
    state.faults = 0
    assert state.faults == 0


def test_shells_out():
    done = subprocess.run([sys.executable, "-c", "import linkedin"], capture_output=True)
    assert done.returncode == 0


def _run_tool(*args):
    return subprocess.run([sys.executable, "tools/x.py", *args], capture_output=True)


def test_shells_out_through_a_helper():
    assert _run_tool("--check").returncode == 0


def test_reads_a_launcher_through_a_path_constant():
    assert "exa" in LAUNCHER.read_text()


def test_pins_a_data_file():
    assert json.loads(SEED_BRIEF.read_text())["target"] == "designer"


def test_imports_its_subject_inside_the_body():
    from shared.contracts import RUN_LOG_EVENTS

    assert "string_results" in RUN_LOG_EVENTS


def test_checks_with_an_imported_assertion_helper():
    assert_no_bullets(config.GREETING)


def test_a_test_support_helper():
    assert_no_bullets("plain prose")


class TestGrouped:
    def test_inside_a_class(self):
        assert save("x") == "saved"
'''


EXCLUDED_SUBJECT_TESTS = '''\
import pytest

from tools import mutation_survey as ms


@pytest.fixture(scope="module")
def surveyed():
    return ms.main(["report"])


def test_the_excluded_tool():
    assert ms.TOOL_VERSION == 1


def test_the_excluded_tool_through_a_fixture(surveyed):
    assert surveyed == 0
'''


def _facts(name: str, source: str = "") -> classify.TestFacts:
    import ast

    tree = ast.parse(source or CLASSIFIED_TESTS)
    return classify.static_facts(
        tree,
        f"tests/test_x.py::{name}",
        {"shared", "linkedin", "tools"},
        frozenset({"tools.mutation_survey"}),
        frozenset({"replay_support"}),
    )


DERIVED_SUBJECT_TESTS = '''\
from pathlib import Path

from replay_support import assert_recording

ROOT = Path("config")
PRIMARY = ROOT / "series-b" / "brief.json"
FALLBACK = ROOT / "fixture" / "brief.json"
SEED = PRIMARY if PRIMARY.is_file() else FALLBACK


def test_pins_a_data_file_chosen_at_import():
    import json

    assert "designer" in json.loads(SEED.read_text())["target_modules"]


def test_a_support_module_imported_by_bare_name():
    assert_recording([1], [1])
'''


def test_static_facts_follow_derived_paths_and_bare_support_imports():
    pinned = _facts("test_pins_a_data_file_chosen_at_import", DERIVED_SUBJECT_TESTS)
    assert pinned.data_read is True
    support = _facts("test_a_support_module_imported_by_bare_name", DERIVED_SUBJECT_TESTS)
    assert support.uses_test_support_names and not support.uses_production_names
    assert _suggest("test_pins_a_data_file_chosen_at_import", "no_body", DERIVED_SUBJECT_TESTS) == "A"
    assert (
        _suggest("test_a_support_module_imported_by_bare_name", "no_body", DERIVED_SUBJECT_TESTS)
        == "O"
    )


def test_static_facts_see_what_a_test_asserts_on():
    assert _facts("test_reads_a_module_constant_of_source").source_read is True
    assert _facts("test_pins_a_default").pin_only is True
    assert _facts("test_negative_only").negative_only is True
    assert _facts("test_proxy_only").proxy_only is True
    assert _facts("test_asserts_nothing").assertions == 0
    assert _facts("test_tautology").tautology is True
    assert _facts("test_shells_out").subprocess_use is True

    assert _facts("test_shells_out_through_a_helper").subprocess_use is True
    assert _facts("test_reads_a_launcher_through_a_path_constant").source_read is True
    assert _facts("test_pins_a_data_file").data_read is True
    assert _facts("test_pins_a_default").data_read is False
    assert _facts("test_imports_its_subject_inside_the_body").pin_only is True
    helper_checked = _facts("test_checks_with_an_imported_assertion_helper")
    assert helper_checked.assertions == 1 and helper_checked.uses_production_names
    support = _facts("test_a_test_support_helper")
    assert support.uses_test_support_names and not support.uses_production_names
    assert _facts("test_the_excluded_tool", EXCLUDED_SUBJECT_TESTS).subject_excluded is True
    through_fixture = _facts("test_the_excluded_tool_through_a_fixture", EXCLUDED_SUBJECT_TESTS)
    assert through_fixture.subject_excluded is False and through_fixture.module_imports_excluded
    assert _facts("test_pins_a_default").module_imports_excluded is False

    through_helper = _facts("test_asserts_through_a_helper")
    assert through_helper.assertions == 1 and not through_helper.negative_only
    grouped = _facts("TestGrouped::test_inside_a_class[param-1]")
    assert grouped.found and grouped.assertions == 1 and grouped.uses_production_names
    assert _facts("test_that_does_not_exist").found is False


SCOPED_SUBJECT_TESTS = '''\
import subprocess

import pytest

from cloris.models import THRESHOLD
from shared.config import LIMIT
from tools.experiments import gate_experiment


@pytest.fixture
def harness(tmp_path):
    def run(*args):
        return subprocess.run(["bash", "tools/launch.sh", *args], capture_output=True)

    return run


def test_pins_a_relic_constant():
    assert THRESHOLD == 25


def test_pins_an_in_scope_constant():
    assert LIMIT == 3


def test_a_tool_subpackage():
    assert gate_experiment.run([]) == 0


def test_a_script_through_a_fixture(harness):
    assert harness("--check").returncode == 0
'''


def _scoped(name: str, verdict: str, **row) -> str:
    import ast

    scope = classify.Scope(
        roots=frozenset({"shared", "cloris", "tools"}),
        scope_files=frozenset({"shared/config.py", "tools/iterate_brief.py"}),
        project_files=frozenset(
            {
                "shared/config.py",
                "tools/iterate_brief.py",
                "tools/experiments/gate_experiment.py",
                "cloris/models.py",
            }
        ),
    )
    facts = classify.static_facts(ast.parse(SCOPED_SUBJECT_TESTS), f"tests/test_x.py::{name}", scope)
    base = {"verdict": verdict, "control": "passed", "reads": [], "observed_functions": 0}
    return classify.suggest(facts, {**base, **row}, covered_in_scope=3)[0]


def test_suggestions_tell_watched_code_from_surveyed_code():
    # Project code the survey only watches is not "production" for a pin.
    assert _scoped("test_pins_a_relic_constant", "no_body") == "O"
    assert _scoped("test_pins_an_in_scope_constant", "no_body") == "A"
    assert _scoped("test_a_tool_subpackage", "survivor") == "O"
    # The subprocess hides inside a fixture the test only names as a parameter.
    assert _scoped("test_a_script_through_a_fixture", "no_body") == "S"
    # Opening a production file while running behavior is not a shape test.
    opened = {"reads": ["shared/config.py"]}
    assert _scoped("test_pins_an_in_scope_constant", "crash_only", **opened) != "A"
    assert _scoped("test_pins_an_in_scope_constant", "survivor", **opened) == "A"


def _suggest(name: str, verdict: str, source: str = "", **row) -> str:
    base = {"verdict": verdict, "control": "passed", "reads": [], "observed_functions": 0}
    return classify.suggest(_facts(name, source), {**base, **row}, covered_in_scope=3)[0]


def test_suggestions_follow_the_strongest_signal():
    assert _suggest("test_reads_a_module_constant_of_source", "no_body") == "A"
    assert _suggest("test_pins_a_default", "no_body") == "A"
    assert _suggest("test_asserts_nothing", "no_body") == "B"
    assert _suggest("test_asserts_nothing", "no_body", observed_functions=4) == "O"
    assert _suggest("test_shells_out", "no_body") == "S"
    assert _suggest("test_shells_out_through_a_helper", "no_body") == "S"
    assert _suggest("test_reads_a_launcher_through_a_path_constant", "no_body") == "A"
    assert _suggest("test_pins_a_data_file", "no_body") == "A"
    assert _suggest("test_imports_its_subject_inside_the_body", "no_body") == "A"
    assert _suggest("test_checks_with_an_imported_assertion_helper", "no_body") == "A"
    assert _suggest("test_a_test_support_helper", "no_body") == "O"
    assert _suggest("test_the_excluded_tool", "survivor", EXCLUDED_SUBJECT_TESTS) == "O"
    assert (
        _suggest("test_the_excluded_tool_through_a_fixture", "no_body", EXCLUDED_SUBJECT_TESTS)
        == "O"
    )
    assert _suggest("test_tautology", "survivor") == "B"
    # Weak assertions (none, negative-only, proxy-only) are harmless beside a
    # real lock on everything the test runs...
    assert _suggest("test_asserts_nothing", "crash_only") == "N"
    assert _suggest("test_proxy_only", "survivor") == "N"
    assert _suggest("test_negative_only", "survivor") == "N"
    assert _suggest("test_asserts_through_a_helper", "survivor") == "N"
    # ...and shallow where they are a witness to a function nothing locks. That
    # half is measured: the survey's own function verdicts.
    alone = ["linkedin/x.py::helper@10"]
    assert _suggest("test_asserts_nothing", "crash_only", unlocked_met=alone) == "C"
    assert _suggest("test_proxy_only", "survivor", unlocked_met=alone) == "C"
    assert _suggest("test_negative_only", "survivor", unlocked_met=alone) == "C"
    assert _suggest("test_asserts_through_a_helper", "survivor", unlocked_met=alone) == "C"
    # Real assertions that only ever saw crashes: no conclusion, whatever else ran.
    assert _suggest("test_asserts_through_a_helper", "crash_only", unlocked_met=alone) == "K"
    assert _suggest("test_asserts_through_a_helper", "survivor", reads=["linkedin/x.py"]) == "A"
    assert _suggest("test_asserts_through_a_helper", "survivor", observed_functions=9) == "O"
    assert _suggest("test_asserts_through_a_helper", "no_control", control="skipped") == "E"
    assert _suggest("test_asserts_through_a_helper", "no_control", control="failed") == "X"
    assert _suggest("test_asserts_through_a_helper", "unstable") == "X"


def test_the_planted_survey_classifies_as_designed(planted):
    classes = planted["classes"]

    assert classes["test_planted_tautology_success_resets_the_failures"] == "B"
    assert classes["test_planted_structural_settle_keeps_its_rounding"] == "A"
    assert classes["test_planted_text_pin_record_success_source"] == "A"
    assert classes["test_planted_crash_only_audit_runs"] == "C"
    assert classes["test_planted_unreachable_skipped"] == "E"
    # Shallow, but beside a real lock on the same function: the survey found
    # the sibling that makes it harmless.
    assert classes["test_planted_shallow_settle_leaves_status_ok"] == "N"
    assert not any(name.startswith("test_planted_real_lock") for name in classes)
    assert planted["classification"]["locks"] == 5
    assert planted["classification"]["not_locks"] == 8


def test_draws_are_the_same_for_every_reader():
    rows = [{"class": "A", "test": f"t::a{n}"} for n in range(30)]
    rows += [{"class": "B", "test": "t::only"}]

    first, again = classify.draw(rows, 5), classify.draw(list(reversed(rows)), 5)

    assert first == again
    assert len(first["A"]) == 5 and first["B"] == ["t::only"]
    assert first["A"] != sorted(first["A"]), "a draw, not the first five"


def test_the_default_gate_never_collects_the_planted_suite():
    assert not SUITE.startswith("test_") and not SUITE.endswith("_test.py")
