#!/usr/bin/env python3
"""Generate edit pairs for the scorer's real acceptance test.

A reward's job in a training rig is to rank several completions of the SAME
prompt against each other, never to rank unrelated strings. Correlating the
composite against per-string ratings measures cross-string ranking, which is
dominated by topic choice and rater taste the enricher never controls. What
matters is preference accuracy on edit pairs: take a real string, apply one
deterministic edit, and check that the scorer prefers the side a human would.

Each operator below names the edit and the direction a rater should prefer.
The generator draws base strings spanning the target range, applies every
operator that fits, randomises which side is the edited one, and writes the
keyed fixture beside a blind copy for the raters.

    python tools/make_boolean_pairs.py [--out <fixture>] [--blind-dir <dir>]

Standard library only; nothing here is imported by the scorer.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linkedin.boolean_score.lexicon import (  # noqa: E402
    LEXICON_DIR,
    Lexicon,
    derived_hyphen_twin,
    load_lexicon,
    normalize,
)
from linkedin.boolean_score.parser import Or, Term, parse  # noqa: E402
from linkedin.boolean_score.score import score  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "boolean_corpus"
LEXICON_PATH = LEXICON_DIR / "research-engineer-post-training.json"
DEFAULT_OUT = FIXTURES / "pairs_v1.json"
DEFAULT_BLIND = ROOT / "output" / "boolean-rl"
# Preference is read in log space: a multiplicative reward makes any
# improvement on a bad completion invisible in absolute terms, and GRPO
# normalises advantages within a group, so a log-scale reward is the natural
# one. 0.01 log units is about a 1% relative difference.
TIE_BAND = 0.01
SEED = 20260921

_GROUP_RE = re.compile(r"\(([^()]*)\)")
_TERM_RE = re.compile(r'"([^"]*)"')

BETTER, WORSE, TIE = "better", "worse", "tie"


# ---------------------------------------------------------------------------
# String surgery
# ---------------------------------------------------------------------------


def _groups(boolean: str) -> list[tuple[int, int, list[str]]]:
    """(start, end, terms) for each flat parenthetical group."""
    found = []
    for match in _GROUP_RE.finditer(boolean):
        terms = [t.strip() for t in _TERM_RE.findall(match.group(1)) if t.strip()]
        if terms:
            found.append((match.start(), match.end(), terms))
    return found


def _rebuild(terms: Sequence[str]) -> str:
    return "(" + " OR ".join(f'"{term}"' for term in terms) + ")"


def _replace_group(boolean: str, index: int, terms: Sequence[str]) -> str:
    start, end, _ = _groups(boolean)[index]
    return boolean[:start] + _rebuild(terms) + boolean[end:]


def _positive_group_indices(boolean: str) -> list[int]:
    """Groups that are not the operand of a NOT."""
    keep = []
    for index, (start, _end, _terms) in enumerate(_groups(boolean)):
        before = boolean[:start].rstrip()
        if not before.endswith("NOT"):
            keep.append(index)
    return keep


def _concepts_in(terms: Sequence[str], lexicon: Lexicon) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for term in terms:
        concept = lexicon.concept_for(term)
        if concept is not None:
            found.setdefault(concept.id, concept)
    return found


# ---------------------------------------------------------------------------
# Edit operators
# ---------------------------------------------------------------------------

Operator = Callable[[str, Lexicon], str | None]
UMBRELLA_INJECTION = "machine learning"
RARE_INJECTIONS = (
    "reinforcement learning from human feedback",
    "reinforcement learning from AI feedback",
    "group relative policy optimization",
)
FRAGMENTS = ("built an LLM", "shipped an LLM")
NOISY_INJECTION = "Magpie"
DISTANT_TERMS = ("vector database", "hybrid search", "reranker")
VERB_GATE = ("led", "managed", "owned")
SELF_EXCLUDING = ("engineer", "researcher")


def add_missing_common_forms(boolean: str, lexicon: Lexicon) -> str | None:
    for index in _positive_group_indices(boolean):
        terms = _groups(boolean)[index][2]
        present = {normalize(term) for term in terms}
        additions = [
            form.text
            for concept in _concepts_in(terms, lexicon).values()
            for form in concept.common_forms
            if form.normalized not in present
        ]
        if additions:
            return _replace_group(boolean, index, terms + additions[:4])
    return None


def drop_forms_to_one_per_concept(boolean: str, lexicon: Lexicon) -> str | None:
    for index in _positive_group_indices(boolean):
        terms = _groups(boolean)[index][2]
        kept, seen = [], set()
        for term in terms:
            concept = lexicon.concept_for(term)
            key = concept.id if concept else term
            if key in seen:
                continue
            seen.add(key)
            kept.append(term)
        if len(kept) >= 2 and len(kept) < len(terms):
            return _replace_group(boolean, index, kept)
    return None


def inject_umbrella_term(boolean: str, lexicon: Lexicon) -> str | None:
    for index in _positive_group_indices(boolean):
        terms = _groups(boolean)[index][2]
        if any(lexicon.is_umbrella(term) for term in terms):
            continue
        if _concepts_in(terms, lexicon):
            return _replace_group(boolean, index, terms + [UMBRELLA_INJECTION])
    return None


def inject_rare_expansions(boolean: str, lexicon: Lexicon) -> str | None:
    indices = _positive_group_indices(boolean)
    if not indices:
        return None
    terms = _groups(boolean)[indices[0]][2]
    return _replace_group(boolean, indices[0], terms + list(RARE_INJECTIONS))


def remove_anchor_group(boolean: str, lexicon: Lexicon) -> str | None:
    tree = parse(boolean).tree
    groups = _groups(boolean)
    indices = _positive_group_indices(boolean)
    if tree is None or len(indices) < 2:
        return None
    start, end, _ = groups[indices[-1]]
    trimmed = (boolean[:start] + boolean[end:]).strip()
    trimmed = re.sub(r"\s+AND\s*$", "", trimmed).strip()
    trimmed = re.sub(r"^\s*AND\s+", "", trimmed).strip()
    return trimmed if parse(trimmed).legal else None


def add_self_excluding_not(boolean: str, _lexicon: Lexicon) -> str | None:
    return f"{boolean} NOT {_rebuild(SELF_EXCLUDING)}"


def add_third_gate(boolean: str, _lexicon: Lexicon) -> str | None:
    return f"{boolean} AND {_rebuild(VERB_GATE)}"


def extend_sweep_with_known_instances(boolean: str, lexicon: Lexicon) -> str | None:
    for index in _positive_group_indices(boolean):
        terms = _groups(boolean)[index][2]
        concepts = _concepts_in(terms, lexicon)
        if len(concepts) < 2 or any(c.names_a_class for c in concepts.values()):
            continue
        shared = set(next(iter(concepts.values())).families)
        for concept in concepts.values():
            shared &= set(concept.families)
        if not shared:
            continue
        family = sorted(shared)[0]
        present = {normalize(term) for term in terms}
        extra = [
            other.common_forms[0].text
            for other in lexicon.concepts
            if not other.names_a_class
            and family in other.families
            and other.id not in concepts
            and other.common_forms
            and other.common_forms[0].normalized not in present
        ]
        if len(extra) >= 6:
            return _replace_group(boolean, index, terms + extra[:6])
    return None


def inject_sentence_fragments(boolean: str, _lexicon: Lexicon) -> str | None:
    indices = _positive_group_indices(boolean)
    if not indices:
        return None
    terms = _groups(boolean)[indices[0]][2]
    return _replace_group(boolean, indices[0], terms + list(FRAGMENTS))


def inject_unrelated_family_terms(boolean: str, lexicon: Lexicon) -> str | None:
    for index in _positive_group_indices(boolean):
        terms = _groups(boolean)[index][2]
        if any(normalize(t) in {normalize(d) for d in DISTANT_TERMS} for t in terms):
            continue
        if _concepts_in(terms, lexicon):
            return _replace_group(boolean, index, terms + list(DISTANT_TERMS))
    return None


def add_hyphen_twins(boolean: str, lexicon: Lexicon) -> str | None:
    for index in _positive_group_indices(boolean):
        terms = _groups(boolean)[index][2]
        present = {normalize(term) for term in terms}
        twins = []
        for term in terms:
            twin = derived_hyphen_twin(term)
            if twin and twin not in present and lexicon.concept_for(term):
                twins.append(twin)
        if twins:
            return _replace_group(boolean, index, terms + twins[:3])
    return None


def inject_noisy_token(boolean: str, lexicon: Lexicon) -> str | None:
    for index in _positive_group_indices(boolean):
        terms = _groups(boolean)[index][2]
        if NOISY_INJECTION in terms:
            continue
        return _replace_group(boolean, index, terms + [NOISY_INJECTION])
    return None


def reorder_terms(boolean: str, _lexicon: Lexicon) -> str | None:
    indices = _positive_group_indices(boolean)
    if not indices:
        return None
    terms = _groups(boolean)[indices[0]][2]
    if len(terms) < 3:
        return None
    return _replace_group(boolean, indices[0], list(reversed(terms)))


def add_more_real_forms_beyond_common(boolean: str, lexicon: Lexicon) -> str | None:
    for index in _positive_group_indices(boolean):
        terms = _groups(boolean)[index][2]
        present = {normalize(term) for term in terms}
        additions = [
            form.text
            for concept in _concepts_in(terms, lexicon).values()
            for form in concept.occasional_forms
            if form.normalized not in present
        ]
        if len(additions) >= 2:
            return _replace_group(boolean, index, terms + additions[:5])
    return None


OPERATORS: tuple[tuple[str, Operator, str], ...] = (
    ("add_missing_common_forms", add_missing_common_forms, BETTER),
    ("drop_forms_to_one_per_concept", drop_forms_to_one_per_concept, WORSE),
    ("inject_umbrella_term", inject_umbrella_term, WORSE),
    ("inject_rare_expansions", inject_rare_expansions, WORSE),
    ("remove_anchor_group", remove_anchor_group, WORSE),
    ("add_self_excluding_NOT", add_self_excluding_not, WORSE),
    ("add_third_gate", add_third_gate, WORSE),
    ("extend_sweep_with_known_instances", extend_sweep_with_known_instances, BETTER),
    ("inject_sentence_fragments", inject_sentence_fragments, WORSE),
    ("inject_unrelated_family_terms", inject_unrelated_family_terms, WORSE),
    ("add_hyphen_twins", add_hyphen_twins, BETTER),
    ("inject_noisy_token", inject_noisy_token, WORSE),
    ("reorder_terms", reorder_terms, TIE),
    ("add_more_real_forms_beyond_common", add_more_real_forms_beyond_common, TIE),
)

TIE_OPERATORS = frozenset({"reorder_terms", "add_more_real_forms_beyond_common"})
PER_OPERATOR_CAP = 6


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_pairs(lexicon: Lexicon) -> list[dict[str, Any]]:
    bases = json.loads((FIXTURES / "rater_targets.json").read_text())["strings"]
    bases = sorted(bases, key=lambda row: row["target"])
    spread = [bases[index] for index in range(0, len(bases), 2)] + [
        bases[index] for index in range(1, len(bases), 2)
    ]
    rng = random.Random(SEED)
    pairs: list[dict[str, Any]] = []
    for name, operator, direction in OPERATORS:
        made = 0
        for base in spread:
            if made >= PER_OPERATOR_CAP:
                break
            original = base["boolean"]
            try:
                edited = operator(original, lexicon)
            except Exception:
                edited = None
            if not edited or edited == original or not parse(edited).legal:
                continue
            flip = rng.random() < 0.5
            left, right = (edited, original) if flip else (original, edited)
            if direction == TIE:
                expected = "tie"
            elif direction == BETTER:
                expected = "A" if flip else "B"
            else:
                expected = "B" if flip else "A"
            pairs.append({
                "pair_id": f"P{len(pairs) + 1:03d}",
                "A": left,
                "B": right,
                "base_id": base["id"],
                "edit": name,
                "edited_side": "A" if flip else "B",
                "expected": expected,
            })
            made += 1
    return pairs


def scorer_preference(pair: dict[str, Any], lexicon: Lexicon) -> str:
    delta = (
        score(pair["B"], lexicon).log_composite
        - score(pair["A"], lexicon).log_composite
    )
    if abs(delta) < TIE_BAND:
        return "tie"
    return "B" if delta > 0 else "A"


def render_markdown(pairs: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Boolean edit pairs, blind",
        "",
        "For each pair, say which side is the better search for a research "
        "engineer working on LLM post-training, or `tie` when the difference "
        "does not matter. Answer with the pair id and `A`, `B` or `tie`.",
        "",
    ]
    for pair in pairs:
        lines += [f"## {pair['pair_id']}", "", "A", "", "```", pair["A"], "```", "",
                  "B", "", "```", pair["B"], "```", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--blind-dir", default=str(DEFAULT_BLIND))
    args = parser.parse_args(argv)

    lexicon = load_lexicon(LEXICON_PATH)
    pairs = build_pairs(lexicon)
    Path(args.out).write_text(
        json.dumps(
            {
                "note": (
                    "Edit pairs for preference accuracy. Each pair is one real "
                    "string and the same string after one deterministic edit; "
                    "which side is edited is randomised. `expected` is the "
                    "direction the edit operator is designed to move quality."
                ),
                "tie_band": TIE_BAND,
                "pairs": pairs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    blind_dir = Path(args.blind_dir)
    blind_dir.mkdir(parents=True, exist_ok=True)
    (blind_dir / "pairs_v1_blind.json").write_text(
        json.dumps(
            {"pairs": [{k: pair[k] for k in ("pair_id", "A", "B")} for pair in pairs]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    (blind_dir / "pairs_v1_blind.md").write_text(render_markdown(pairs) + "\n")

    print(f"{len(pairs)} pairs -> {args.out}")
    print(f"blind copies -> {blind_dir}/pairs_v1_blind.json and .md\n")
    print("operator                            pairs   scorer agrees with expected")
    overall_hit = overall_total = 0
    for name, _operator, _direction in OPERATORS:
        rows = [pair for pair in pairs if pair["edit"] == name]
        if not rows:
            print(f"  {name:34s}     0   -")
            continue
        hits = sum(1 for pair in rows if scorer_preference(pair, lexicon) == pair["expected"])
        overall_hit += hits
        overall_total += len(rows)
        print(f"  {name:34s} {len(rows):5d}   {hits}/{len(rows)} ({hits / len(rows):.0%})")
    print(f"  {'OVERALL':34s} {overall_total:5d}   {overall_hit}/{overall_total} "
          f"({overall_hit / overall_total:.0%})" if overall_total else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
