"""Exa identity: person keys, the do-not-save gate, and batch dedupe.

Three jobs, in the order a candidate meets them: mint a person key,
check the recruiter's do-not-save list, then drop anyone this campaign
has already seen.

The do-not-save gate is FAIL-CLOSED on a name match whenever the name is
all the gate can go on. That is deliberate and asymmetric: letting a
ruled-out person through costs the recruiter's credibility with a client,
so when either side lacks a public LinkedIn handle a namesake is blocked
too. When BOTH carry one and the handles differ, they are two people, and
the name match does not block: the entry names its person by URL, and a
namesake at a competitor is exactly who the search is for. The list is
written by a human in whatever form came to mind, so matching normalizes
across the forms one human uses for one name: diacritics, middle
initials, and "Last, First".

State comes in as parameters. Seen keys and blocked person keys live in
runtime state; a pure function that takes them is testable offline and
resumable for free.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

from exa.schemas import ExaCandidate
from shared.identity_resolution import (
    normalize_person_name,
    normalize_public_linkedin_handle,
)
from shared.person_identity import IdentityEvidence, merge_candidates


# Keys with this prefix are anchored to a public LinkedIn handle and
# identify one human. Everything else is a name+firm slug — good enough
# to fold a duplicate inside one batch, not good enough to assert two
# people are the same across batches.
STRONG_KEY_PREFIX = "linkedin.com/in/"

# What a do-not-save entry reports when the brief gave no reason.
DEFAULT_BLOCK_REASON = "on the brief's do-not-save list"

# What a URL from the brief's prior_search skip list reports.
PRIOR_SEARCH_BLOCK_REASON = "on the brief's prior_search skip list (already known)"

_PUNCT = re.compile(r"[^a-z0-9\s]")


def _fold_ascii(value: str) -> str:
    """Strip diacritics so "Ávery" and "Avery" compare equal.

    Duplicated from ``exa/extraction.py`` on purpose: importing that
    module here would drag ``shared.llm_clients`` into the dedupe path,
    which S5 runs with no model in play.
    """

    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _name_keys(name: str) -> set[str]:
    """Every form one human's name might have been written in.

    Three keys, and a match on any one blocks:

    * the full token string, so "Avery Lin Holt" matches itself;
    * first and last tokens only, so "Avery L. Holt" on the slate
      matches "Avery Holt" in the brief;
    * the reordered form when the input carried a comma, so a list
      pasted out of a spreadsheet as "Holt, Avery" matches a candidate
      named "Avery Holt".

    Diacritics fold to ASCII first, and
    :func:`shared.identity_resolution.normalize_person_name` does the
    lowercasing, punctuation stripping, and credential-suffix removal —
    the same normalizer every other module matches names with.
    """

    raw = _fold_ascii(name)
    tokens = _PUNCT.sub(" ", normalize_person_name(raw)).split()
    if not tokens:
        return set()

    keys = {" ".join(tokens)}
    if len(tokens) >= 2:
        keys.add(f"{tokens[0]} {tokens[-1]}")
        if "," in raw:
            reordered = tokens[1:] + tokens[:1]
            keys.add(" ".join(reordered))
            keys.add(f"{reordered[0]} {reordered[-1]}")
    return keys


def exa_person_key(candidate: ExaCandidate) -> str:
    """Mint the cross-hub person key for an Exa-discovered candidate.

    Goes through :func:`shared.person_identity.merge_candidates` rather
    than formatting a string locally, so an Exa key is minted by the
    same code that mints every other hub's and stays correct if those
    rules change.

    The HUB is chosen by what the evidence actually is. A candidate with
    a public LinkedIn handle is evidenced by a LinkedIn profile, so it
    keys as ``linkedin:<handle>`` — and a later backfill that mints keys
    for the LinkedIn campaign's own rows will mint the identical string
    for the same human. Keying it ``exa:`` would guarantee the two never meet,
    since ``merge_candidates`` merges on hub-plus-handle and reads
    ``linkedin_url`` in no rule at all
    (the merge evidence rules in ``shared.person_identity._merge_evidence``).

    A candidate with no handle has no LinkedIn evidence to claim, so it
    keys as ``exa:<name-firm-slug>``.
    """

    handle = normalize_public_linkedin_handle(
        candidate.linkedin_url or ""
    ) or normalize_public_linkedin_handle(candidate.identity_key or "")

    if handle:
        evidence = IdentityEvidence(
            hub="linkedin",
            handle=handle,
            display_name=candidate.name,
            linkedin_url=candidate.linkedin_url,
        )
    else:
        evidence = IdentityEvidence(
            hub="exa",
            handle=candidate.identity_key,
            display_name=candidate.name,
            linkedin_url=candidate.linkedin_url,
        )
    return merge_candidates([evidence])[0].key


# ---------------------------------------------------------------------------
# Do-not-save gate
# ---------------------------------------------------------------------------

class DoNotSaveGate:
    """The recruiter's veto list, applied at source.

    Built from the brief's source-agnostic ``do_not_save`` field: a list
    of ``{name, linkedin_url?, reason?}``. An empty list is a no-op
    gate, so a brief that names nobody costs nothing.
    """

    def __init__(self, entries: Iterable[dict] | None = None):
        self._by_handle: dict[str, str] = {}
        # Every entry under each name key, as ``(handle, message)`` in list
        # order. The handle is "" for an entry the brief named without a
        # public URL; that entry can block a namesake, a URL-bearing one
        # cannot block a candidate whose own handle differs.
        self._by_name_key: dict[str, list[tuple[str, str]]] = {}
        self._entries: list[dict] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            handle = normalize_public_linkedin_handle(
                str(entry.get("linkedin_url") or "")
            )
            if not name and not handle:
                continue
            reason = str(entry.get("reason") or "").strip() or DEFAULT_BLOCK_REASON
            message = f"{name or handle}: {reason}"
            self._entries.append(dict(entry))
            if handle:
                self._by_handle.setdefault(handle, message)
            for key in _name_keys(name):
                self._by_name_key.setdefault(key, []).append((handle, message))

    @classmethod
    def from_brief(cls, brief: Any) -> DoNotSaveGate:
        """Read the gate off either Brief dataclass, tolerantly.

        The brief's ``prior_search.ruled_out_urls`` (inline, plus any
        ``ruled_out_urls_file`` the loader read) join the gate as URL-only
        entries: a public profile on that list is someone already known, and
        an open-web result carries the public URL. Recruiter talent URLs carry
        no public handle and add nothing here.
        """

        entries = getattr(brief, "do_not_save", [])
        if not isinstance(entries, list):
            entries = []
        entries = list(entries)
        prior_search = getattr(brief, "prior_search", None)
        ruled_out = getattr(prior_search, "ruled_out_urls", None) or []
        if isinstance(ruled_out, list):
            for url in ruled_out:
                if isinstance(url, str) and normalize_public_linkedin_handle(url):
                    entries.append(
                        {"name": "", "linkedin_url": url, "reason": PRIOR_SEARCH_BLOCK_REASON}
                    )
        return cls(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        # An empty gate is falsy so a caller can skip the check without
        # having to know whether `len` or `is None` is the question.
        return bool(self._by_handle or self._by_name_key)

    def blocks(self, candidate: ExaCandidate) -> str | None:
        """The block reason for this candidate, or None.

        The handle is checked first because it is the certain match. The
        name is checked second, on any of the forms :func:`_name_keys`
        produces, and blocks on its own unless both sides carry a public
        handle: a candidate whose handle differs from a URL-bearing
        entry's is a namesake and passes. An entry with no URL, or a
        candidate with no public handle, still blocks on the name alone —
        see the module docstring.

        The candidate's handle is read off ``linkedin_url`` and, failing
        that, off a handle-anchored ``identity_key``, the same two places
        :func:`exa_person_key` reads it from.
        """

        handle = normalize_public_linkedin_handle(
            candidate.linkedin_url or ""
        ) or normalize_public_linkedin_handle(candidate.identity_key or "")
        if handle and handle in self._by_handle:
            return self._by_handle[handle]
        for key in _name_keys(candidate.name or ""):
            for entry_handle, message in self._by_name_key.get(key, ()):
                if handle and entry_handle:
                    # Two public handles, and they differ: two people.
                    continue
                return message
        return None


# ---------------------------------------------------------------------------
# Batch dedupe
# ---------------------------------------------------------------------------

@dataclass
class DedupeResult:
    """What survived a batch, why the rest did not, and what to watch.

    ``flagged`` is not a third outcome — everything in it is also in
    ``kept``. It carries the folds that were NOT made because the
    evidence argued against them, so the caller can log a weak-key collision
    instead of silently shipping two records that look like one person.
    """

    kept: list[ExaCandidate] = field(default_factory=list)
    skipped: list[tuple[ExaCandidate, str]] = field(default_factory=list)
    flagged: list[tuple[ExaCandidate, str]] = field(default_factory=list)


def dedupe_candidates(
    candidates: Iterable[ExaCandidate],
    *,
    seen_identity_keys: set[str],
    blocked_person_keys: set[str],
    gate: DoNotSaveGate,
) -> DedupeResult:
    """Filter a batch down to the people worth judging.

    Checks run in authority order, and the first hit wins so the skip
    reason names the reason that actually mattered:

    1. ``do_not_save:<reason>`` — the recruiter's own ruling;
    2. ``already_seen`` — this identity key came out of an earlier query
       or an earlier session;
    3. ``blocked_person_key`` — the cross-hub person key is on the
       run's block list;
    4. ``duplicate_in_batch`` — the same person twice in this batch.

    A batch duplicate on a WEAK key (a name+firm slug, not a public
    LinkedIn handle) folds only when the two reads agree: same title, or
    one of them blank. Two people who share a name and a firm and hold
    DIFFERENT titles are more likely two people than one, so both are
    kept and the second is reported in ``flagged`` as
    ``weak_key_collision:<other name / other title>``. Folding them
    would delete a real candidate on the strength of a slug; keeping
    them costs one duplicate a human can see. Input order is preserved
    throughout.
    """

    result = DedupeResult()
    batch_records: dict[str, ExaCandidate] = {}

    for candidate in candidates:
        reason = gate.blocks(candidate)
        if reason:
            result.skipped.append((candidate, f"do_not_save:{reason}"))
            continue

        identity_key = candidate.identity_key
        if identity_key in seen_identity_keys:
            result.skipped.append((candidate, "already_seen"))
            continue

        if exa_person_key(candidate) in blocked_person_keys:
            result.skipped.append((candidate, "blocked_person_key"))
            continue

        held = batch_records.get(identity_key)
        if held is not None:
            weak = not identity_key.startswith(STRONG_KEY_PREFIX)
            titles_disagree = bool(
                held.current_title
                and candidate.current_title
                and held.current_title.strip().lower()
                != candidate.current_title.strip().lower()
            )
            if weak and titles_disagree:
                result.kept.append(candidate)
                result.flagged.append(
                    (
                        candidate,
                        "weak_key_collision:"
                        f"{held.name} / {held.current_title}",
                    )
                )
                continue
            result.skipped.append(
                (
                    candidate,
                    "duplicate_in_batch:weak_key" if weak else "duplicate_in_batch",
                )
            )
            continue

        batch_records[identity_key] = candidate
        result.kept.append(candidate)

    return result


__all__ = [
    "DEFAULT_BLOCK_REASON",
    "STRONG_KEY_PREFIX",
    "DedupeResult",
    "DoNotSaveGate",
    "dedupe_candidates",
    "exa_person_key",
]
