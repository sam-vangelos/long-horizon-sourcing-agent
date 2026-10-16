"""Employer-blacklist gate for Exa candidates, applied before judgment.

The brief's ``employer_blacklist`` reaches the judge as prompt text, and a
model can honor that rule and still miss it. This gate reads the candidate's
current employer off the stored record and rejects a match with no model call;
the slate applies the same check when it renders, so a save the judge let
through leaves the short list on the next render. Matching is
:func:`shared.employer_blacklist.match_current_employer`, the rule the LinkedIn
facial gate applies, so both sources exclude the same people.

A gated candidate is a ``REJECT`` with ``path`` set to :data:`GATE_PATH` and
``reject_reason`` ``HARD_GATE``: no new decision vocabulary, and a marker the
store, ``judgments.jsonl``, and the slate can all read back.

The brief's ``alumni_blacklist`` gets the same treatment over the WHOLE work
history: anyone who has ever worked at a listed employer is never a lead. Its
matcher is
:func:`shared.employer_blacklist.match_any_employer`, the rule the LinkedIn
gates apply, and its verdict carries ``path`` :data:`ALUMNI_GATE_PATH`.
"""

from __future__ import annotations

from typing import Iterable

from shared.employer_blacklist import match_any_employer, match_current_employer
from shared.schemas import OpusDecision

from exa.schemas import ExaCandidate


GATE_PATH = "employer_blacklist"
ALUMNI_GATE_PATH = "alumni_blacklist"
GATE_PATHS = frozenset({GATE_PATH, ALUMNI_GATE_PATH})
GATE_REJECT_REASON = "HARD_GATE"


def _flatten(value: object) -> str:
    return " ".join(str(value or "").split())


def current_employer_signals(candidate: ExaCandidate) -> tuple[list[str], list[str]]:
    """``(companies, headlines)``: where this record says the person works now.

    Companies are matched on a word boundary. They are the record's own firm
    (the seat the judge and the console already treat as current), the company
    of every position the LinkedIn render marks open, and the most recent
    position when none is open.

    Headlines match only in "at X" / "@ X" context. They are the render's
    headline and the record's title, because a title read off a page title can
    carry the employer ("TPM at Centific") when the firm field is blank.
    """

    companies: list[str] = []
    headlines: list[str] = []

    def add(bucket: list[str], value: object) -> None:
        text = _flatten(value)
        if text and text not in bucket:
            bucket.append(text)

    add(companies, candidate.current_firm)
    render = candidate.render()
    if render is not None:
        open_seats = [seat for seat in render.experiences if seat.is_current]
        latest = render.most_recent_experience()
        for seat in open_seats or ([latest] if latest is not None else []):
            add(companies, seat.company)
        add(headlines, render.headline)
    add(headlines, candidate.current_title)
    return companies, headlines


def current_title_signals(candidate: ExaCandidate) -> list[str]:
    """The titles this record says the person holds now.

    The record's own title, then the title of every position the LinkedIn
    render marks open, or of the most recent position when none is: the
    same seats :func:`current_employer_signals` reads the companies from.
    """

    titles: list[str] = []

    def add(value: object) -> None:
        text = _flatten(value)
        if text and text not in titles:
            titles.append(text)

    add(candidate.current_title)
    render = candidate.render()
    if render is not None:
        open_seats = [seat for seat in render.experiences if seat.is_current]
        latest = render.most_recent_experience()
        for seat in open_seats or ([latest] if latest is not None else []):
            add(seat.title)
    return titles


def employer_blacklist_match(
    candidate: ExaCandidate, employer_blacklist: Iterable[str] | None
) -> tuple[str, str, str] | None:
    """``(blocked_name, field_name, field_value)`` for a current employer, or None."""

    blacklist = [str(name) for name in (employer_blacklist or []) if str(name or "").strip()]
    if not blacklist:
        return None
    companies, headlines = current_employer_signals(candidate)
    for company in companies:
        hit = match_current_employer(company, "", blacklist)
        if hit is not None:
            return hit
    for headline in headlines:
        hit = match_current_employer("", headline, blacklist)
        if hit is not None:
            return hit
    return None


def employment_history_signals(
    candidate: ExaCandidate,
) -> tuple[list[str], list[str], str]:
    """``(companies, headlines, about)``: everywhere this record says the person
    has worked, current or past.

    Companies are the record's own firm and the company of EVERY position the
    LinkedIn render lists, open or closed. Headlines are the render's headline
    and the record's title. ``about`` is the render's About prose.
    """

    companies: list[str] = []
    headlines: list[str] = []

    def add(bucket: list[str], value: object) -> None:
        text = _flatten(value)
        if text and text not in bucket:
            bucket.append(text)

    add(companies, candidate.current_firm)
    about = ""
    render = candidate.render()
    if render is not None:
        for seat in render.experiences:
            add(companies, seat.company)
        add(headlines, render.headline)
        about = _flatten(render.about)
    add(headlines, candidate.current_title)
    return companies, headlines, about


def alumni_blacklist_match(
    candidate: ExaCandidate, alumni_blacklist: Iterable[str] | None
) -> tuple[str, str, str] | None:
    """``(blocked_name, field_name, field_value)`` for any employer, or None."""

    names = [str(name) for name in (alumni_blacklist or []) if str(name or "").strip()]
    if not names:
        return None
    companies, headlines, about = employment_history_signals(candidate)
    current_firm = _flatten(candidate.current_firm)
    hit = match_any_employer(
        names,
        current_company=current_firm,
        experience_companies=[company for company in companies if company != current_firm],
    )
    if hit is not None:
        return hit
    for headline in headlines:
        hit = match_any_employer(names, headline=headline)
        if hit is not None:
            return hit
    return match_any_employer(names, about=about)


def gate_rationale(blocked: str, path: str = GATE_PATH) -> str:
    if path == ALUMNI_GATE_PATH:
        return f"has worked at an employer on the alumni blacklist ({blocked})"
    return f"current employer on the employer blacklist ({blocked})"


def gate_decision(
    candidate: ExaCandidate, blocked: str, path: str = GATE_PATH
) -> OpusDecision:
    """The deterministic verdict a gated candidate is stored under."""

    return OpusDecision(
        stage="full",
        decision="REJECT",
        path=path,
        confidence=1.0,
        rationale=gate_rationale(blocked, path),
        candidate_name=candidate.name,
        profile_url=candidate.linkedin_url or (candidate.evidence_urls() or [""])[0],
        reject_reason=GATE_REJECT_REASON,
    )


def is_gate_decision(decision: object) -> bool:
    return getattr(decision, "path", "") in GATE_PATHS


__all__ = [
    "ALUMNI_GATE_PATH",
    "GATE_PATH",
    "GATE_PATHS",
    "GATE_REJECT_REASON",
    "alumni_blacklist_match",
    "current_employer_signals",
    "current_title_signals",
    "employer_blacklist_match",
    "employment_history_signals",
    "gate_decision",
    "gate_rationale",
    "is_gate_decision",
]
