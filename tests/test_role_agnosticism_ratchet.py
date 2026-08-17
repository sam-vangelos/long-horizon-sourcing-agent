"""The role-agnosticism ratchet (plans/sourcing-generality-hardening.md item 21).

Structure-level sibling of tests/test_vertical_vocab_ratchet.py. That ratchet
pins vertical VOCABULARY (lexicon terms in string literals); this one pins
role-SHAPED STRUCTURE that carries no lexicon nouns:

- ``renderer`` — lexicon residue in prompts assembled from a synthetic
  neutral brief (any hit is code-carried: template text, fallbacks, defaults);
- ``keyword_collection`` — module-level hard-coded pattern lists (the
  ``_BUY_SIDE_PATTERNS`` shape in shared/strict_seniority.py);
- ``brief_text_gate`` — ``"magic word" in brief_text`` gates (the
  ``_brief_targets_edge_case_opening`` shape in linkedin/strategy.py).

Ratchet semantics differ from the vocab ratchet's count-pin: the baseline at
tools/role_agnosticism_baseline.txt lists every grandfathered finding
verbatim. It may SHRINK, never grow:

- every CURRENT finding must be baselined — a new finding fails with
  "role-shaped structure belongs in the brief", and
- every BASELINED entry must still fire — a stale entry fails so discharging
  a structure forces the baseline line to be deleted (growth stays a
  reviewed, diff-visible decision; silent re-expansion is impossible).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from lint_role_agnosticism import (  # noqa: E402
    _RATCHET_MESSAGE,
    load_baseline,
    scan,
)


@lru_cache(maxsize=1)
def _current_findings() -> tuple[str, ...]:
    """One scan per test session — the render pass imports and renders the
    real prompt assemblies, so don't repeat it per test."""
    return tuple(scan())


def test_every_current_finding_is_baselined():
    findings = set(_current_findings())
    baseline = set(load_baseline())
    new = sorted(findings - baseline)
    assert not new, f"{_RATCHET_MESSAGE}: {new}"


def test_baseline_carries_no_stale_entries_so_it_only_shrinks():
    findings = set(_current_findings())
    baseline = set(load_baseline())
    stale = sorted(baseline - findings)
    assert not stale, (
        "stale role-agnosticism baseline entries no longer fire — delete "
        "these lines from tools/role_agnosticism_baseline.txt (the baseline "
        f"may shrink, never grow): {stale}"
    )


def test_baseline_is_nonempty_and_anchored_on_the_known_structures():
    """The detectors are not tautologies: the canonical role-shaped
    structures named in the plan must be in the grandfathered baseline —
    strict_seniority's hard-coded pattern lists (keyword_collection) and
    the tapped-market magic-word gate in linkedin/strategy.py
    (brief_text_gate over the brief-text haystack)."""
    baseline = load_baseline()
    assert baseline, "role-agnosticism baseline is empty — the linter found nothing, which is wrong for the current tree"
    assert any(
        entry.startswith("shared/strict_seniority.py:")
        and ":keyword_collection:_BUY_SIDE_PATTERNS" in entry
        for entry in baseline
    ), baseline
    assert any(
        entry.startswith("linkedin/strategy.py:")
        and ":brief_text_gate:" in entry
        and "haystack" in entry
        for entry in baseline
    ), baseline


def test_structural_detector_flags_planted_structures(tmp_path, monkeypatch):
    """Red-team the AST rules: a planted keyword collection and a planted
    brief-text gate in a scanned root must both be flagged, while the
    wire-format suffix and dunder exemptions must hold."""
    import lint_role_agnosticism as linter

    scan_root = tmp_path / "shared"
    scan_root.mkdir()
    (tmp_path / "linkedin").mkdir()
    (scan_root / "planted.py").write_text(
        "PLANTED_PATTERNS = (\n"
        '    "alpha bank", "beta fund", "gamma", "delta shop",\n'
        '    "epsilon", "zeta capital",\n'
        ")\n"
        'PLANTED_DECISIONS = ("SAVE", "REJECT", "SKIP", "HOLD", "RETRY", "ABORT")\n'
        '__all__ = ("a", "b", "c", "d", "e", "f")\n'
        "\n"
        "def direct_gate(brief_text: str) -> bool:\n"
        '    return "magic phrase" in brief_text\n'
        "\n"
        "def indirect_gate(haystack: str) -> bool:\n"
        '    triggers = ("magic one", "magic two")\n'
        "    return any(trigger in haystack for trigger in triggers)\n"
    )
    monkeypatch.setattr(linter, "ROOT", tmp_path)
    monkeypatch.setattr(linter, "EXCLUDED_FILES", set())

    findings = linter.scan_structural()

    assert any(
        "planted.py" in f and ":keyword_collection:PLANTED_PATTERNS" in f
        for f in findings
    ), findings
    # Constant-left gate AND comprehension-bound-name gate both fire.
    assert any(
        "planted.py" in f and ":brief_text_gate:" in f and "'magic phrase'" in f
        for f in findings
    ), findings
    assert any(
        "planted.py" in f and ":brief_text_gate:trigger in haystack" in f
        for f in findings
    ), findings
    # Exemptions hold: schema-enum suffixes and dunders never fire.
    assert not any("PLANTED_DECISIONS" in f for f in findings), findings
    assert not any("__all__" in f for f in findings), findings


def test_render_residue_detector_flags_a_planted_lexicon_term(monkeypatch):
    """Red-team the null-brief render check: it reported zero findings on
    the current tree, so prove that is a true clean — a lexicon term planted
    into a rendered prompt must be reported against its assembly function."""
    import lint_role_agnosticism as linter

    real_rendered = linter._rendered_prompts()
    assert all(
        text and len(text) > 500
        for texts in real_rendered.values()
        for text in texts
    ), "neutral-brief renders came back empty/truncated — the scan would be vacuous"

    planted = {fn: list(texts) for fn, texts in real_rendered.items()}
    planted["assemble_facial_system"] = [
        planted["assemble_facial_system"][0] + "\nplanted machine learning residue"
    ]
    monkeypatch.setattr(linter, "_rendered_prompts", lambda: planted)

    findings = linter.scan_render_residue()

    assert "renderer:assemble_facial_system:machine learning" in findings, findings
    assert not any(
        f.startswith("renderer:assemble_full_evaluation_system:") for f in findings
    ), findings
