"""Planted production-like module for the mutation survey's own tests.

Every function here exists so that one planted test in ``planted_suite.py``
can stand in a known relationship to it: a real lock, a tautology, a shallow
proxy, a crash-only smoke, a negative case, a structural source check. The
survey is proven against these before any classification of the real suite
is reported. Not imported by Cloris.
"""

from __future__ import annotations

import hashlib
from typing import Iterator


class Ledger:
    # Class-level defaults on purpose: a planted survivor must be able to
    # survive the emptying of every method it touches, construction included.
    status = "ok"
    balance = 0
    failures = 0

    def settle(self, amounts):
        total = 0
        for amount in amounts:
            total += amount
        self.balance = round(total, 2)
        return self.balance

    def record_failure(self):
        self.failures += 1

    def record_success(self):
        self.failures = 0

    def is_overdrawn(self) -> bool:
        return self.balance < 0

    def audit(self):
        digest = _checksum(self.balance)
        return digest.hexdigest()


def _checksum(balance):
    return hashlib.sha256(str(balance).encode())


def iter_amounts(amounts) -> Iterator[int]:
    for amount in amounts:
        if amount:
            yield amount


async def settle_later(ledger, amounts):
    return ledger.settle(amounts)


def never_called(amounts):
    return sum(amounts)
