"""Source-compilation guards for market-intelligence modules."""

from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path(__file__).parent.parent


def test_market_intelligence_engine_compiles_from_source():
    py_compile.compile(
        str(ROOT / "market_intelligence" / "engine.py"),
        doraise=True,
    )
