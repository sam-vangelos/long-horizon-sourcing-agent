"""Planted tests with known verdicts, surveyed by tests/test_mutation_survey.py.

The file name deliberately does not match ``test_*.py``: the default gate must
never collect these. The survey collects the file by explicit path inside a
hermetic copy of this directory.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from pathlib import Path

import pytest

from planted_pkg import ledger as ledger_module
from planted_pkg.ledger import Ledger, _checksum, iter_amounts, settle_later


def test_planted_real_lock_settle_totals_the_amounts():
    ledger = Ledger()
    assert ledger.settle([5, 7]) == 12
    assert ledger.balance == 12


def test_planted_real_lock_settle_later_returns_the_balance():
    assert asyncio.run(settle_later(Ledger(), [1, 2])) == 3


def test_planted_real_lock_iter_amounts_skips_zeroes():
    assert list(iter_amounts([1, 0, 2])) == [1, 2]


def test_planted_tautology_success_resets_the_failures():
    ledger = Ledger()
    ledger.record_failure()
    ledger.record_failure()
    # The success path zeroes the counter.
    ledger.failures = 0
    assert ledger.failures == 0


def test_planted_shallow_settle_leaves_status_ok():
    ledger = Ledger()
    ledger.settle([5, 7])
    assert ledger.status == "ok"


def test_planted_crash_only_audit_runs():
    Ledger().audit()


def test_planted_reimplements_its_subject_and_asserts_on_the_copy():
    # Touches the production object on an ordinary line, then checks only its
    # own arithmetic. An emptied _checksum breaks the line, not an assertion.
    produced = _checksum(5).hexdigest()
    expected = hashlib.sha256(b"5").hexdigest()
    assert len(produced) == 64
    assert expected == hashlib.sha256(b"5").hexdigest()


def test_planted_negative_case_not_overdrawn():
    ledger = Ledger()
    ledger.settle([5])
    assert not ledger.is_overdrawn()


def test_planted_structural_settle_keeps_its_rounding():
    source = Path(ledger_module.__file__).read_text(encoding="utf-8")
    assert "round(total, 2)" in source


def test_planted_text_pin_record_success_source():
    ledger = Ledger()
    ledger.record_success()
    last_line = inspect.getsource(Ledger.record_success).splitlines()[-1]
    assert last_line.strip() == "self.failures = 0"


@pytest.mark.skip(reason="planted: permanently skipped")
def test_planted_unreachable_skipped():
    assert Ledger().settle([1]) == 1


@pytest.fixture(scope="module")
def settled_ledger():
    ledger = Ledger()
    ledger.settle([2, 3])
    return ledger


def test_planted_shared_fixture_first_user_checks_a_proxy(settled_ledger):
    assert settled_ledger.status == "ok"


def test_planted_shared_fixture_second_user_reads_the_balance(settled_ledger):
    # Runs no production code of its own: coverage credits the fixture's work
    # to the first user, yet this is the test that locks it.
    assert settled_ledger.balance == 5
