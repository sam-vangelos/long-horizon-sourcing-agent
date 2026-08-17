"""The vertical-vocabulary ratchet (Wave 2 slice 9).

Rule (plans/sourcing-rigor-hardening.md P1): deterministic code may CONSUME
brief-supplied vocabulary; it may never CARRY its own. tools/lint_vertical_vocab.py
scans string literals in shared/ + linkedin/ for lexicon terms; every
grandfathered constant carries an inline ``# VERTICAL-VOCAB(<ref>)`` marker.

The allowlist may SHRINK, never grow:
- zero UNMARKED lexicon hits (new vertical vocabulary fails CI — it belongs
  in the brief), and
- the marked-line count never exceeds the recorded baseline (discharging a
  grandfathered constant lowers the baseline; adding one is impossible
  without editing this number, which is the point — it makes growth a
  reviewed decision).

shared/role_strategy_profiles.py is excluded by design: role-class profiles
are vertical templates — their vocabulary IS the product content. And
shared/role_strategy.py (the router mechanism) ships with an EMPTY allowlist:
zero markers, zero hits (plan addendum item 6 — no carve-out).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from lint_vertical_vocab import scan  # noqa: E402

# Recorded when the ratchet landed (Wave 2 slice 9). Shrink freely; growing
# it means deterministic code gained vertical vocabulary — put it in the
# brief instead.
_BASELINE_MARKED_LINES = 43  # Wave 3 (2026-07-04): slice 14 discharged the
# preflight non-fit example, the ML title synonyms, and the seniority-band
# detector's AI vocabulary (50→45); the Codex-review fix de-verticalized the
# kernel-spec prompt blocks and discharged both their markers (45→43), with
# the lexicon extended to the variants that had slipped (applied ai,
# palantir, scale ai, frontier company, fdes, head of ai, ai builder(s)).
# Was 50 at the Wave 2 slice 9 landing (2026-07-03).


def test_no_unmarked_vertical_vocabulary_in_deterministic_code():
    violations, _marked = scan()
    assert not violations, (
        "vocabulary belongs in the brief — unmarked vertical-vocabulary "
        f"literals in deterministic code: {violations}"
    )


def test_grandfathered_marker_count_never_grows():
    _violations, marked = scan()
    assert len(marked) <= _BASELINE_MARKED_LINES, (
        f"marked vertical-vocab lines grew: {len(marked)} > baseline "
        f"{_BASELINE_MARKED_LINES}. The allowlist may shrink, never grow — "
        "new vocabulary belongs in the brief."
    )


def test_role_strategy_router_has_empty_allowlist():
    """Plan addendum item 6: the router mechanism carries ZERO vocabulary —
    no markers, no hits. Profiles (the data module) carry their own."""
    violations, marked = scan()
    role_strategy_records = [
        record
        for record in (violations + marked)
        if record.startswith("shared/role_strategy.py:")
    ]
    assert not role_strategy_records, role_strategy_records


def test_ratchet_detects_a_planted_literal(tmp_path, monkeypatch):
    """Red-team the ratchet itself: a planted unmarked literal in a scanned
    root must be flagged (the linter is not a tautology)."""
    import lint_vertical_vocab as linter

    scan_root = tmp_path / "shared"
    scan_root.mkdir()
    (scan_root / "planted.py").write_text(
        'PATTERNS = ("capital markets", "production")\n'
    )
    monkeypatch.setattr(linter, "ROOT", tmp_path)
    monkeypatch.setattr(linter, "EXCLUDED_FILES", set())
    # Lexicon still read from the real repo.
    monkeypatch.setattr(
        linter, "LEXICON_PATH", ROOT / "tools" / "vertical_vocab_lexicon.txt"
    )
    (tmp_path / "linkedin").mkdir()

    violations, marked = linter.scan()

    assert any("planted.py" in violation for violation in violations)
    assert marked == []
