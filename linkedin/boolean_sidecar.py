"""Ask the enricher sidecar for better OR groups, behind its flag.

The lexicon enricher (``linkedin/boolean_score/enrich.py``) adds only forms
the lexicon already declares. The sidecar is a small trained model served
over local HTTP by the cloris-boolean-rl repo, and it proposes whole groups:
the spellings a sourcer would write for one concept (the ``register`` task)
and the class terms a lone sweep of named things is missing (the ``anchor``
task). This module is Cloris's side of that exchange. It sends group text,
receives group text, and keeps a proposal only when this repo's own scorer
rates the whole string higher; the model's view of its output counts for
nothing.

WHERE IT RUNS. Directly after the lexicon enricher, on the same fields:
``generated_strings[].boolean`` and ``coverage_gaps[].suggested_boolean`` at
strategy formation, ``new_strings[].boolean`` at adaptation.

WHAT IT DECIDES, per string. Register first: every positive, contiguous OR
group whose terms all map to one lexicon concept is sent as authored, and the
best returned group that parses, keeps every other group as it was, fits the
field cap and lifts ``log_composite`` by more than ``MIN_GAIN`` replaces it.
Groups are taken left to right, each judged against the string as it stands
after the ones before.

Then the mixed groups: a positive, contiguous OR group naming several
concepts, or names the lexicon cannot map (companies, titles, tools) beside
the ones it can. Its mapped terms are split by concept into runs, and a run
is sent alone as one OR group of its terms as authored, the shape the model
was trained on; an unmapped term is never sent. From the reply only the
terms new to the whole group are taken, and they are added after the run's
last term. Nothing authored is removed or reordered, because the rest of a
mixed group carries meaning the model never saw: replacing the group with
the model's answer would drop it. The same bar applies (legal, every other
group unchanged, the field cap, a gain above ``MIN_GAIN``), at most one
change is kept per run, and runs are taken left to right. Each request costs
seconds, so a string sends at most ``MAX_CONCEPT_RUNS_PER_STRING`` runs, the
thinnest first (fewest forms of the concept in the group), and never a run
whose group already carries ``RICH_RUN_FORMS`` forms of its concept.

Anchor last, on that result: if the string is still a lone sweep with no
anchor, the sweep group is sent once, and the best returned group that
clears the same bar is AND'd in front. Register goes first because a richer
sweep is what the anchor request should describe, and at most one anchor is
ever inserted.

WHAT IT LEAVES BEHIND, only when something was kept: ``<field>_sidecar`` —
the adapter, the request count, and one entry per kept change with the whole
string and its ``log_composite`` before and after (a run's entry adds its
concept and the terms added); ``<field>_original`` when
nothing earlier set it; and, for the primary boolean, ``boolean_score``
recomputed the way the lexicon enricher computes it. An item that already
carries ``<field>_sidecar`` is skipped, because the adaptation hook runs
twice. With the flag off nothing is read, sent or written.

PADDING EARNS NOTHING. The scorer ignores terms the lexicon cannot map, so a
model can pad a good group with a misspelling ("evalution") at no cost, and
it counts a term as often as it is written. A term a proposal repeats is
dropped before anything is judged, the first spelling kept. A proposal is
refused when a term it adds is unmapped and is within
``NEAR_DUPLICATE_DISTANCE`` edits of a term already there, or is such a
term of several words run into one ("LLMjudging" beside "LLM judge"), or has
any form the lexicon maps, of whatever concept, fused into a longer word
("evaluationsuite", not "evaluation framework", "eval-harness" or the
inflected "benchmarking"). Each test
applies only to terms and forms of at least ``NEAR_DUPLICATE_MIN_LENGTH``
characters. A register
proposal is refused when a term it adds maps to a concept other than the
group's, whatever its spelling shares with the group ("evaluations" beside
"eval harness"), or maps to none and either has no word starting with the
stem of one of the group's words ("scoring rubric" beside "rubric", not
"scorecard") or keeps some but not all of one group term's words ("frontier"
beside "frontier model": a qualifier dropped, and far broader), a run
standing in for the group in these tests; an anchor, when
one of its terms
maps to nothing and has no word starting with the stem of a mapped term's
word ("metrics" beside "evaluation"). A stem is a word less its longest
listed suffix, or the whole word when fewer than ``_MIN_STEM`` letters would
remain, so "bench" is not on the root of "benchmark"; a word shorter than
``_MIN_STEM`` gives no stem, so the "a" of "LLM-as-a-judge" roots nothing and
a group of acronyms admits only terms mapped to its concept. Among proposals
scoring within ``MIN_GAIN`` of the best, the one with the fewest unmapped
terms wins, then the shortest.

FAILURE IS SILENT AND LOCAL. A sidecar that answers badly yields no
candidates and the string stays as it was. One that cannot be reached at all
(a request that times out or is refused twice) is not asked again for the
rest of that plan or adaptation: strings already changed stay changed, the
string in flight and every later one are left exactly as authored. Nothing
prints, because the run console is a fixed surface and this is not on it.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Sequence

import shared.config as _config

# _score_summary is private to the enrichment adapter; importing it keeps
# boolean_score identical in shape whichever adapter wrote it last.
from linkedin.boolean_enrichment import _score_summary, load_role_lexicon
from linkedin.boolean_score.enrich import DEFAULT_MAX_CHARS

# Private helpers of the enricher, imported rather than copied so the sidecar
# finds, bounds and verifies groups exactly as the enricher does.
from linkedin.boolean_score.enrich import (
    _closes_safely,
    _collect,
    _has_and,
    _render_addition,
    _render_authored,
    _signature,
    _terms_are_contiguous,
)
from linkedin.boolean_score.lexicon import Lexicon
from linkedin.boolean_score.parser import Or, ParseResult, Term, mask_quoted_spans, parse
from linkedin.boolean_score.score import ScoreReport, score

REGISTER_TASK = "register"
# The receipt's name for a register request made for one concept run inside
# a mixed group; on the wire it is a register request like any other.
REGISTER_RUN_TASK = "register_run"
ANCHOR_TASK = "anchor"
ANCHOR_SLOT = "anchor"
LONE_SWEEP_SHAPE = "lone_sweep"
DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_SAMPLES = 8
DEFAULT_TEMPERATURE = 0.8
# A proposal must beat the string it would replace by more than this, in
# log-composite units, so scorer noise never rewrites a plan.
MIN_GAIN = 0.01
# An unmapped term this close to another term in the group is a misspelling
# or a fused compound of it, not a new way people write the concept.
NEAR_DUPLICATE_DISTANCE = 2
# Below this length two unrelated acronyms sit an edit or two apart by chance,
# and a short term turns up inside longer words, so neither test applies.
NEAR_DUPLICATE_MIN_LENGTH = 5
# Concept runs sent per string. A request takes seconds on a laptop, and a
# plan carries a few dozen strings, so this bounds what formation waits for.
MAX_CONCEPT_RUNS_PER_STRING = 3
# A run whose group already holds this many forms of its concept has little
# register left to gain, so its request is spent on a thinner run.
RICH_RUN_FORMS = 4
_MAX_CHARS = DEFAULT_MAX_CHARS

_RawGroup = tuple[tuple[Term, ...], bool]
Ask = Callable[[str, str], list[str]]


def sidecar_enabled() -> bool:
    return bool(getattr(_config, "LINKEDIN_ENRICHER_SIDECAR_ENABLED", False))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Candidates(list):
    """The candidate groups, carrying the adapter the sidecar says wrote them
    and whether the sidecar could not be reached at all."""

    def __init__(
        self, groups: Iterable[str] = (), adapter: str = "", failed: bool = False
    ) -> None:
        super().__init__(groups)
        self.adapter = adapter
        self.failed = failed


class _SidecarDown(Exception):
    """Raised to abandon the string in flight once the sidecar is unreachable."""


@dataclasses.dataclass
class _Breaker:
    """Shared by every field of one plan or adaptation."""

    tripped: bool = False


def request_candidates(
    task: str,
    given: str,
    *,
    url: str,
    timeout: float,
    n: int,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[str]:
    """POST one request to ``<url>/enrich``; the non-null groups it returns.

    A refused connection or a timeout is retried once, and a second one
    comes back marked ``failed``. Anything else that is not a 200 with the
    contracted JSON body yields no candidates, unmarked: the sidecar answered.
    """
    body = json.dumps(
        {"task": task, "given": given, "n": n, "temperature": temperature}
    ).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/enrich",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    payload: Any = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    return Candidates()
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError:
            # The sidecar answered and refused; asking again changes nothing.
            return Candidates()
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt:
                return Candidates(failed=True)
        except (ValueError, http.client.HTTPException):
            return Candidates()
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        return Candidates()
    groups = [
        candidate["group"]
        for candidate in payload["candidates"]
        if isinstance(candidate, dict)
        and isinstance(candidate.get("group"), str)
        and candidate["group"].strip()
    ]
    adapter = payload.get("adapter")
    return Candidates(groups, adapter if isinstance(adapter, str) else "")


# ---------------------------------------------------------------------------
# One string
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Outcome:
    boolean: str
    requests: int = 0
    adapter: str = ""
    kept: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # What the string offered, for the replay tool; never written to a plan.
    whole_groups: int = 0
    concept_runs: int = 0


def _groups(result: ParseResult) -> list[_RawGroup]:
    raw: list[_RawGroup] = []
    if result.tree is not None:
        _collect(result.tree, False, raw)
    return raw


def _extent(source: str, terms: Sequence[Term]) -> tuple[int, int]:
    """The group as authored: its terms, plus the parentheses that enclose
    exactly them when there are any.

    The terms of a contiguous group have only ``OR`` between them, so a ``(``
    directly before the first and a ``)`` directly after the last are a
    matched pair holding this group and nothing else.
    """
    start, end = terms[0].span[0], terms[-1].span[1]
    before = source[:start].rstrip()
    after = source[end:].lstrip()
    if before.endswith("(") and after.startswith(")"):
        return len(before) - 1, len(source) - len(after) + 1
    return start, end


def _single_concept(terms: Sequence[Term], lexicon: Lexicon) -> bool:
    ids = set()
    for term in terms:
        concept = lexicon.concept_for(term.text)
        if concept is None:
            return False
        ids.add(concept.id)
    return len(ids) == 1


def _candidate_terms(group: str) -> tuple[Term, ...] | None:
    """The terms of a proposal that is one positive OR group of plain terms."""
    result = parse(group)
    if not result.legal:
        return None
    tree = result.tree
    if isinstance(tree, Term):
        return (tree,)
    if isinstance(tree, Or) and all(isinstance(child, Term) for child in tree.children):
        return tuple(tree.children)
    return None


def _interior(group: str, terms: Sequence[Term]) -> str:
    return " OR ".join(_render_authored(group, term) for term in terms)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char != other))
            )
        previous = current
    return previous[-1]


_SEPARATORS = " -"


def _fused(key: str, inner: str, lexicon: Lexicon) -> bool:
    """True when ``inner`` occurs in ``key`` run into other letters with no
    space or hyphen between, inside a word that is not itself mapped.

    Letters that are only one inflection ending after ``inner``
    ("benchmarking", "reward models") make a form of the word, not a fusion."""
    start = key.find(inner)
    while start >= 0:
        end = start + len(inner)
        left = max(key.rfind(sep, 0, start) for sep in _SEPARATORS) + 1
        ends = [key.find(sep, end) for sep in _SEPARATORS]
        right = min((i for i in ends if i >= 0), default=len(key))
        inflected = left == start and key[end:right] in _STEM_SUFFIX_KEYS
        if (
            (left, right) != (start, end)
            and not inflected
            and lexicon.concept_for(key[left:right]) is None
        ):
            return True
        start = key.find(inner, start + 1)
    return False


def _long_forms(lexicon: Lexicon) -> list[str]:
    return [
        form.normalized
        for concept in lexicon.concepts
        for form in concept.forms
        if len(form.normalized) >= NEAR_DUPLICATE_MIN_LENGTH
    ]


def _run_together(key: str, other: str) -> bool:
    """True when ``key`` is one word that spells ``other``, a term of several
    words, with its spaces and hyphens dropped, give or take an inflection
    on each and a slip or two ("llmjudging" for "llm judge")."""
    if any(sep in key for sep in _SEPARATORS) or not any(sep in other for sep in _SEPARATORS):
        return False
    left = _stem(key)
    right = _stem("".join(char for char in other if char not in _SEPARATORS))
    return min(len(left), len(right)) >= NEAR_DUPLICATE_MIN_LENGTH and (
        _edit_distance(left, right) <= NEAR_DUPLICATE_DISTANCE
    )


def _padded(proposed: Sequence[Term], authored: Sequence[Term], lexicon: Lexicon) -> bool:
    """True when the proposal adds an unmapped near-duplicate: a misspelling
    of a term already in the group or elsewhere in the proposal, that term
    run together into one word, or any form the lexicon maps, of whatever
    concept, fused into a longer word."""
    kept = {term.normalized for term in authored}
    forms = _long_forms(lexicon)
    for index, term in enumerate(proposed):
        key = term.normalized
        if key in kept or lexicon.concept_for(key) is not None:
            continue
        if any(form in key and _fused(key, form, lexicon) for form in forms):
            return True
        others = [t.normalized for i, t in enumerate(proposed) if i != index]
        for other in [*kept, *others]:
            if min(len(key), len(other)) >= NEAR_DUPLICATE_MIN_LENGTH and (
                _edit_distance(key, other) <= NEAR_DUPLICATE_DISTANCE
            ):
                return True
            if _run_together(key, other):
                return True
    return False


# Inflections, tried longest first; only one is ever stripped. Named like the
# scorer's own grammar sets (``_ARTICLE_KEYS``): grammar, not role vocabulary.
_STEM_SUFFIX_KEYS = ("ations", "ation", "ings", "ing", "ers", "er", "es", "ed", "s")
_MIN_STEM = 4
_WORD_BREAK = re.compile(r"[ /-]")


def _words(key: str) -> list[str]:
    return [word for word in _WORD_BREAK.split(key) if word]


def _stem(word: str) -> str:
    """``word`` less its longest listed suffix, or all of it when that would
    leave fewer than ``_MIN_STEM`` letters."""
    for suffix in _STEM_SUFFIX_KEYS:
        if word.endswith(suffix):
            stripped = word[: -len(suffix)]
            return stripped if len(stripped) >= _MIN_STEM else word
    return word


def _stems(terms: Iterable[Term]) -> set[str]:
    """The stems of the terms' words of at least ``_MIN_STEM`` letters. A
    shorter word ("a", "as", "llm") starts too many words to vouch for any."""
    return {
        _stem(word)
        for term in terms
        for word in _words(term.normalized)
        if len(word) >= _MIN_STEM
    }


def _rooted(key: str, stems: set[str]) -> bool:
    """True when a whole word of ``key`` starts with one of ``stems``."""
    return any(word.startswith(stem) for word in _words(key) for stem in stems)


def _concept_id(term: Term, lexicon: Lexicon) -> str | None:
    concept = lexicon.concept_for(term.text)
    return concept.id if concept is not None else None


def _deduped(terms: Sequence[Term]) -> tuple[Term, ...]:
    """Each term once, the first spelling of it kept."""
    seen: set[str] = set()
    kept = []
    for term in terms:
        if term.normalized not in seen:
            seen.add(term.normalized)
            kept.append(term)
    return tuple(kept)


def _off_root(proposed: Sequence[Term], authored: Sequence[Term], lexicon: Lexicon) -> bool:
    """True when a term the proposal adds to the group maps to another
    concept, or maps to none and has no word starting with the stem of one of
    the group's words.

    A mapped term is judged by its concept alone: sharing a stem with the
    group does not make a different concept the same one, and adding one
    widens a specific group into a general one."""
    had = {term.normalized for term in authored}
    concepts = {_concept_id(term, lexicon) for term in authored} - {None}
    stems = _stems(authored)
    for term in proposed:
        if term.normalized in had:
            continue
        concept = _concept_id(term, lexicon)
        if concept is not None:
            if concept not in concepts:
                return True
            continue
        if _drops_a_qualifier(term.normalized, authored):
            return True
        if not _rooted(term.normalized, stems):
            return True
    return False


def _word_stems(key: str) -> set[str]:
    return {_stem(word) for word in _words(key)}


def _drops_a_qualifier(key: str, authored: Sequence[Term]) -> bool:
    """True when ``key``'s words are some but not all of one authored term's
    words, stemmed ("frontier" from "frontier model"): the term it came from
    with a qualifier gone, and so far broader."""
    words = _word_stems(key)
    return bool(words) and any(words < _word_stems(term.normalized) for term in authored)


def _unrooted_anchor(proposed: Sequence[Term], lexicon: Lexicon) -> bool:
    """True when an anchor term maps to no concept and no word of it starts
    with the stem of a mapped term's word in the same proposal."""
    mapped = [term for term in proposed if lexicon.concept_for(term.text) is not None]
    stems = _stems(mapped)
    return any(
        lexicon.concept_for(term.text) is None and not _rooted(term.normalized, stems)
        for term in proposed
    )


def _wrapped(source: str) -> bool:
    """True when one pair of parentheses encloses the whole expression."""
    masked = mask_quoted_spans(source.strip())
    if not (masked.startswith("(") and masked.endswith(")")):
        return False
    depth = 0
    for index, char in enumerate(masked):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index == len(masked) - 1
    return False


def _acceptable(trial: str) -> ParseResult | None:
    after = parse(trial)
    if not after.legal or len(trial) > _MAX_CHARS:
        return None
    return after


def _register_trials(
    source: str,
    index: int,
    groups: list[_RawGroup],
    proposals: Iterable[str],
    lexicon: Lexicon,
) -> Iterable[str]:
    """Each proposal spliced over group ``index``, where it leaves every other
    group as it was."""
    terms = groups[index][0]
    before = parse(source)
    old = _signature(before, lexicon)
    safe = _closes_safely(source, terms, not _has_and(before.tree) and len(groups) == 1)
    start, end = terms[0].span[0], terms[-1].span[1]
    for proposal in proposals:
        proposed = _candidate_terms(proposal)
        if proposed is None:
            continue
        proposed = _deduped(proposed)
        if _padded(proposed, terms, lexicon) or _off_root(proposed, terms, lexicon):
            continue
        interior = _interior(proposal, proposed)
        trial = source[:start] + (interior if safe else f"({interior})") + source[end:]
        after = _acceptable(trial)
        if after is None:
            continue
        new = _signature(after, lexicon)
        if len(new) != len(old) or new[index][0]:
            continue
        if any(new[j] != old[j] for j in range(len(old)) if j != index):
            continue
        yield trial


def _anchor_trials(
    source: str, proposals: Iterable[str], lexicon: Lexicon
) -> Iterable[str]:
    """Each proposal AND'd in front, where the string's own groups survive."""
    before = parse(source)
    old = _signature(before, lexicon)
    body = source.strip()
    if isinstance(before.tree, Or) and not _wrapped(body):
        body = f"({body})"
    for proposal in proposals:
        proposed = _candidate_terms(proposal)
        if proposed is None:
            continue
        proposed = _deduped(proposed)
        if _padded(proposed, (), lexicon) or _unrooted_anchor(proposed, lexicon):
            continue
        trial = f"({_interior(proposal, proposed)}) AND {body}"
        after = _acceptable(trial)
        if after is None:
            continue
        new = _signature(after, lexicon)
        if new[1:] != old or new[0][0]:
            continue
        yield trial


@dataclasses.dataclass(frozen=True)
class _Run:
    """One concept's terms inside a mixed group, found by group and concept
    so it can be found again after earlier edits move it."""

    group_index: int
    concept: str
    forms: int
    position: int


def _run_terms(terms: Sequence[Term], concept: str, lexicon: Lexicon) -> tuple[Term, ...]:
    return tuple(term for term in terms if _concept_id(term, lexicon) == concept)


def _concept_runs(
    source: str, groups: list[_RawGroup], lexicon: Lexicon
) -> list[_Run]:
    """Every run worth a request: in positive, contiguous groups that are not
    one concept, where the group holds fewer than ``RICH_RUN_FORMS`` forms of
    the run's concept. Left to right."""
    runs: list[_Run] = []
    for index, (terms, negated) in enumerate(groups):
        if negated or not _terms_are_contiguous(source, terms):
            continue
        if _single_concept(terms, lexicon):
            continue
        members: dict[str, list[Term]] = {}
        first: dict[str, int] = {}
        for position, term in enumerate(terms):
            concept = _concept_id(term, lexicon)
            if concept is None:
                continue
            members.setdefault(concept, []).append(term)
            first.setdefault(concept, position)
        for concept, found in members.items():
            forms = len({term.normalized for term in found})
            if forms < RICH_RUN_FORMS:
                runs.append(_Run(index, concept, forms, first[concept]))
    return runs


def _chosen_runs(runs: list[_Run]) -> list[_Run]:
    """The thinnest ``MAX_CONCEPT_RUNS_PER_STRING`` runs, ties to the left,
    returned in string order."""
    thinnest = sorted(runs, key=lambda run: (run.forms, run.group_index, run.position))
    chosen = thinnest[:MAX_CONCEPT_RUNS_PER_STRING]
    return sorted(chosen, key=lambda run: (run.group_index, run.position))


def _run_trials(
    source: str,
    index: int,
    concept: str,
    proposals: Iterable[str],
    lexicon: Lexicon,
) -> Iterable[tuple[str, tuple[str, ...]]]:
    """Each proposal's new terms added after the run's last term in group
    ``index``, with the terms added, where every group but this one is as it
    was and this one gained exactly those terms."""
    before = parse(source)
    groups = _groups(before)
    old = _signature(before, lexicon)
    terms = groups[index][0]
    run = _run_terms(terms, concept, lexicon)
    if not run:
        return
    present = {term.normalized for term in terms}
    cut = run[-1].span[1]
    # Between two of the group's terms an added OR cannot re-associate; after
    # the last one it can, unless the group is closed or is the whole string.
    safe = run[-1] is not terms[-1] or _closes_safely(
        source, terms, not _has_and(before.tree) and len(groups) == 1
    )
    open_at = terms[0].span[0]
    for proposal in proposals:
        proposed = _candidate_terms(proposal)
        if proposed is None:
            continue
        new = tuple(term for term in _deduped(proposed) if term.normalized not in present)
        if not new:
            continue
        if _padded(new, run, lexicon) or _off_root(new, run, lexicon):
            continue
        added = " OR ".join(
            _render_addition(term.text, bool(run[0].quoted) or term.quoted) for term in new
        )
        if safe:
            trial = f"{source[:cut]} OR {added}{source[cut:]}"
        else:
            trial = f"{source[:open_at]}({source[open_at:cut]} OR {added}){source[cut:]}"
        after = _acceptable(trial)
        if after is None:
            continue
        signature = _signature(after, lexicon)
        if len(signature) != len(old):
            continue
        grown = (False, old[index][1] | {term.normalized for term in new})
        if signature[index] != grown:
            continue
        if any(signature[j] != old[j] for j in range(len(old)) if j != index):
            continue
        yield trial, tuple(term.text for term in new)


def _best(
    trials: Iterable[str], current: ScoreReport, lexicon: Lexicon
) -> tuple[str, ScoreReport] | None:
    """The proposal to keep, if any beats ``current`` by more than
    ``MIN_GAIN``: of those within ``MIN_GAIN`` of the top score, the one with
    the fewest unmapped terms, then the shortest."""
    bar = current.log_composite + MIN_GAIN
    scored = [(trial, score(trial, lexicon)) for trial in dict.fromkeys(trials)]
    passing = [pair for pair in scored if pair[1].log_composite > bar]
    if not passing:
        return None
    top = max(report.log_composite for _, report in passing)
    band = [pair for pair in passing if pair[1].log_composite >= top - MIN_GAIN]

    def unmapped(report: ScoreReport) -> int:
        return sum(cid is None for group in report.groups for cid in group.concept_ids)

    return min(band, key=lambda pair: (unmapped(pair[1]), len(pair[0])))


def _improve(boolean: str, lexicon: Lexicon, ask: Ask) -> _Outcome:
    outcome = _Outcome(boolean)
    current = score(boolean, lexicon)
    if not current.legal:
        return outcome

    def _ask(task: str, given: str) -> list[str]:
        outcome.requests += 1
        proposals = ask(task, given)
        outcome.adapter = getattr(proposals, "adapter", "") or outcome.adapter
        return list(proposals)

    def _keep(task: str, slot: Any, found: tuple[str, ScoreReport], **detail: Any) -> None:
        nonlocal current
        trial, report = found
        outcome.kept.append(
            {
                "task": task,
                "group_index": slot,
                **detail,
                "before": outcome.boolean,
                "after": trial,
                "score_before": round(current.log_composite, 4),
                "score_after": round(report.log_composite, 4),
            }
        )
        outcome.boolean, current = trial, report

    groups = _groups(parse(boolean))
    eligible = [
        index
        for index, (terms, negated) in enumerate(groups)
        if not negated
        and _terms_are_contiguous(boolean, terms)
        and _single_concept(terms, lexicon)
    ]
    # Found on the string as authored: a whole-group edit leaves every other
    # group's terms as they were, so these runs are still there after it.
    runs = _concept_runs(boolean, groups, lexicon)
    outcome.whole_groups, outcome.concept_runs = len(eligible), len(runs)
    for index in eligible:
        source = outcome.boolean
        groups = _groups(parse(source))
        start, end = _extent(source, groups[index][0])
        proposals = _ask(REGISTER_TASK, source[start:end])
        trials = _register_trials(source, index, groups, proposals, lexicon)
        found = _best(trials, current, lexicon)
        if found is not None:
            _keep(REGISTER_TASK, index, found)

    for run in _chosen_runs(runs):
        source = outcome.boolean
        terms = _groups(parse(source))[run.group_index][0]
        members = _run_terms(terms, run.concept, lexicon)
        if not members:
            continue
        given = "(" + " OR ".join(_render_authored(source, term) for term in members) + ")"
        proposals = _ask(REGISTER_TASK, given)
        added = dict(_run_trials(source, run.group_index, run.concept, proposals, lexicon))
        found = _best(added, current, lexicon)
        if found is not None:
            _keep(
                REGISTER_RUN_TASK,
                run.group_index,
                found,
                concept=run.concept,
                added=list(added[found[0]]),
            )

    if not current.anchor_present and current.shape == LONE_SWEEP_SHAPE:
        source = outcome.boolean
        positive = [terms for terms, negated in _groups(parse(source)) if not negated]
        if len(positive) == 1:
            start, end = _extent(source, positive[0])
            proposals = _ask(ANCHOR_TASK, source[start:end])
            found = _best(_anchor_trials(source, proposals, lexicon), current, lexicon)
            if found is not None:
                _keep(ANCHOR_TASK, ANCHOR_SLOT, found)
    return outcome


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def _asker(breaker: _Breaker, url: str | None = None) -> Ask:
    """The configured client; ``url``, when given, overrides the configured
    address (the replay tool points it at a sidecar of the operator's choosing)."""
    url = url or str(getattr(_config, "LINKEDIN_ENRICHER_SIDECAR_URL", "") or DEFAULT_URL)
    timeout = float(
        getattr(_config, "LINKEDIN_ENRICHER_SIDECAR_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    )
    samples = int(getattr(_config, "LINKEDIN_ENRICHER_SIDECAR_SAMPLES", DEFAULT_SAMPLES))

    def ask(task: str, given: str) -> list[str]:
        if breaker.tripped:
            raise _SidecarDown
        found = request_candidates(task, given, url=url, timeout=timeout, n=samples)
        if getattr(found, "failed", False):
            breaker.tripped = True
            raise _SidecarDown
        return found

    return ask


def enrich_items_with_sidecar(
    items: Iterable[Any], field: str, *, breaker: _Breaker | None = None
) -> int:
    """Offer ``field`` on every mapping in ``items`` to the sidecar. Returns
    how many changed.

    A no-op when the flag is off or the lexicon will not load, and from the
    first unreachable request on. Pass one ``breaker`` to several calls to
    share that stop across them.
    """
    if not sidecar_enabled():
        return 0
    lexicon = load_role_lexicon()
    if lexicon is None:
        return 0
    breaker = breaker if breaker is not None else _Breaker()
    try:
        ask = _asker(breaker)
    except (TypeError, ValueError):
        return 0
    receipt_key = f"{field}_sidecar"

    rewritten = 0
    for item in items:
        if breaker.tripped:
            break
        if not isinstance(item, dict) or receipt_key in item:
            continue
        original = item.get(field)
        if not isinstance(original, str) or not original.strip():
            continue
        try:
            outcome = _improve(original, lexicon, ask)
            if not outcome.kept:
                continue
            summary = (
                _score_summary(outcome.boolean, lexicon) if field == "boolean" else None
            )
        except Exception:  # noqa: BLE001 — never fail plan formation; _SidecarDown too
            continue
        item.setdefault(f"{field}_original", original)
        item[field] = outcome.boolean
        item[receipt_key] = {
            "adapter": outcome.adapter,
            "requests": outcome.requests,
            "kept": outcome.kept,
        }
        if summary is not None:
            item["boolean_score"] = summary
        rewritten += 1
    return rewritten


def enrich_plan_with_sidecar(plan: Any) -> None:
    """Offer a finalized execution plan's authored strings, in place."""
    if not sidecar_enabled():
        return
    breaker = _Breaker()
    enrich_items_with_sidecar(
        getattr(plan, "generated_strings", None) or [], "boolean", breaker=breaker
    )
    enrich_items_with_sidecar(
        getattr(plan, "coverage_gaps", None) or [], "suggested_boolean", breaker=breaker
    )


def enrich_adaptation_with_sidecar(adaptation: Any) -> None:
    """Offer adaptation-authored strings, in place."""
    if not sidecar_enabled():
        return
    enrich_items_with_sidecar(getattr(adaptation, "new_strings", None) or [], "boolean")
