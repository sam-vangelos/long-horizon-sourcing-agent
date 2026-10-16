"""Render the Exa run slate and its structured report.

The renderer states campaign metrics, actionable short-list and review rows,
coverage notes, and a structural ``verify_before_outreach`` section. Names,
titles, firms, links, and verdict explanations come from candidate records or
judge rationales; missing rationale is rendered as ``(no rationale recorded)``.

Every Exa record carries the verify flag because web data may be stale. Keep
that flag as a section listing every affected entry rather than relying on a
closing paragraph.

Within the short list and worth-a-look, reachable people come first and long
shots (``shared/attainability.py``) last, each marked inline with the reason
and grouped again in their own section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from shared.attainability import LONG_SHOT, REACHABLE


# What a slate line says when the judge returned no rationale text. The
# renderer never fills this gap with its own prose.
NO_RATIONALE = "(no rationale recorded)"

# The section headings, in the order they must appear. Exported so the
# test that pins section order reads the same list the renderer writes.
# ``EXCLUDED_HEADING`` is not in it: that section renders only when someone
# was excluded, and it sits between "Worth a look" and "Already known".
SECTION_ORDER = (
    "## The short list",
    "## Worth a look",
    "## Already known",
    "## Coverage notes",
    "## Verify before outreach",
)

EXCLUDED_HEADING = "## Excluded: current employer on the blacklist"

# The same section once the alumni blacklist excluded anyone: the heading
# then has to cover past employers too.
EXCLUDED_ANY_TENSE_HEADING = "## Excluded: blacklisted employer, current or past"

# Renders only when someone on the short list or worth-a-look is a long
# shot, between "Worth a look" and the excluded section.
LONG_SHOTS_HEADING = "## Long shots"

# Renders only when a re-judge (exa/rejudge.py) moved someone off the short
# list, between "Long shots" and the excluded sections.
LEFT_SHORT_LIST_HEADING = "## Left the short list on re-judge"

LONG_SHOTS_EXPLANATION = (
    "Their current employer or title makes a move unlikely: frontier-lab "
    "staff and people who already run their own company rarely leave for a "
    "founding seat. Each stays in its section above, sorted last and marked; "
    "this list groups them."
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SlateEntry:
    """One person as the slate renders them."""

    name: str
    title: str = ""
    firm: str = ""
    location: str = ""
    linkedin_url: str = ""
    evidence_urls: list[str] = field(default_factory=list)
    rationale: str = ""
    decision: str = ""
    confidence: float | None = None
    review_reason_code: str = ""
    case_against: str = ""
    known_label: str = ""
    # Carried from the record, not inferred from the URL. Every Exa
    # find is web-derived, so the flag is true for all of them; a
    # missing public LinkedIn URL is a SECOND, worse problem (identity
    # itself is unconfirmed) and the verify section marks it as such
    # rather than treating a present URL as verification.
    verify_before_outreach: bool = True
    # Why the slate pulled this person off the short list or worth-a-look
    # at render time; empty for everyone who stays.
    excluded_reason: str = ""
    # ``reachable`` or ``long_shot``, with the matched value as the reason.
    attainability: str = REACHABLE
    attainability_reason: str = ""
    # The current-employer scope check's reason ("scope: current employer
    # X") when the record names an employer outside the brief's scope, and
    # whether that check is what moved a judge's save off the short list.
    scope_note: str = ""
    scope_moved: bool = False
    # The store's identity key, so a section built after the buckets can
    # find the person again.
    identity_key: str = ""
    # A re-judge whose two samples split: what the reject sample said. It is
    # the one thing keeping a split person off the short list.
    rejudge_note: str = ""
    # The same note followed by the reject sample's rationale, for the
    # left-the-short-list section.
    rejudge_reason: str = ""
    # For "Left the short list on re-judge": the verdict the person held
    # before, and where they are now.
    previous_decision: str = ""
    previous_confidence: float | None = None
    now: str = ""

    @property
    def is_long_shot(self) -> bool:
        return self.attainability == LONG_SHOT

    def long_shot_marker(self) -> str:
        if not self.is_long_shot:
            return ""
        reason = self.attainability_reason or "no reason recorded"
        return f" · long shot ({reason})"

    @property
    def headline(self) -> str:
        parts = [part for part in (self.title, self.firm) if part]
        line = ", ".join(parts)
        if self.location:
            line = f"{line} ({self.location})" if line else f"({self.location})"
        return line

    @property
    def has_public_linkedin(self) -> bool:
        return bool(str(self.linkedin_url or "").strip())

    def rationale_text(self) -> str:
        text = " ".join(str(self.rationale or "").split())
        return text or NO_RATIONALE

    def off_short_list_reason(self) -> str:
        """The ONE thing keeping this person off the short list.

        Preference order is authority order: the judge's structured
        review reason code, then its case-against sentence, then the
        bare decision. A blank at every level is stated as an absence.

        The scope check outranks all three: when it moved a save here, it is
        the one thing, and the judge's own verdict follows it; when the judge
        already put the person here, the scope note leads the judge's reason.
        """

        if self.rejudge_note:
            return self.rejudge_note
        if self.scope_note:
            if self.scope_moved:
                return f"{self.scope_note} (judge: {self.decision or 'SAVE'})"
            return f"{self.scope_note}; {self._judge_reason()}"
        return self._judge_reason()

    def _judge_reason(self) -> str:
        if self.review_reason_code:
            code = self.review_reason_code
            return f"{self.decision or 'REVIEW'} · {code}" if self.decision else code
        case_against = " ".join(str(self.case_against or "").split())
        if case_against:
            return case_against
        return self.decision or NO_RATIONALE


@dataclass
class FirmCoverage:
    """What one firm's queries actually returned."""

    firm: str
    queries: int = 0
    results: int = 0
    candidates: int = 0
    # Queries counted in ``queries`` that never ran, because a brief
    # edit retired them. They are NOT a gap: "no candidates" here means
    # nobody looked, not that the web holds nobody, and the two read
    # identically on the counts alone.
    skipped: int = 0
    drops: dict[str, int] = field(default_factory=dict)


@dataclass
class ArchetypeCoverage:
    """What ONE archetype query returned, and what came of it.

    Firm coverage answers "what did Benchmark return"; three lanes
    aimed at one firm are three angles on one question. An archetype
    query is not an angle on anything — it is a standalone hypothesis
    about a career shape, so it gets its own row. Carrying the
    short-list and worth-a-look counts is the point of the row: results
    and candidates say the query retrieved people, and only the buckets
    say whether they were the right ones. A query returning 25 results
    and no short list is a hypothesis to rewrite, and it is invisible
    on the counts alone.
    """

    archetype_id: str
    queries: int = 0
    results: int = 0
    candidates: int = 0
    skipped: int = 0
    short_list: int = 0
    worth_a_look: int = 0
    drops: dict[str, int] = field(default_factory=dict)


@dataclass
class RunView:
    """Everything the slate renders, already reduced to report currency."""

    brief_name: str
    run_id: int
    short_list: list[SlateEntry] = field(default_factory=list)
    worth_a_look: list[SlateEntry] = field(default_factory=list)
    already_known: list[SlateEntry] = field(default_factory=list)
    # Stored saves whose current employer is on the brief's blacklist,
    # pulled off the slate at render time. Not outreach targets, so they
    # are outside ``all_entries`` and the verify list.
    excluded_employer: list[SlateEntry] = field(default_factory=list)
    # People the pre-judgment gate rejected: counted here, listed nowhere,
    # and bucketed like any other REJECT.
    gated_employer: int = 0
    # The alumni blacklist's two counterparts: stored saves of anyone who
    # has EVER worked at an alumni-blacklisted employer, pulled off the
    # slate at render time, and people its pre-judgment gate rejected.
    excluded_alumni: list[SlateEntry] = field(default_factory=list)
    gated_alumni: int = 0
    # The location scope's two counterparts (``exa_location_scope``):
    # stored saves whose card location is outside the brief's countries,
    # pulled off the slate at render time, and people its pre-judgment gate
    # rejected. ``location_scope_label`` names the scope ("the United
    # States") and is empty when the brief does not carry one.
    excluded_location: list[SlateEntry] = field(default_factory=list)
    gated_location: int = 0
    location_scope_label: str = ""
    # People a re-judge moved off the short list, campaign-wide: each one's
    # earlier verdict, current reason and current place. Read from the
    # re-judge events and the current verdicts, so a person a later re-judge
    # puts back on the short list leaves this list.
    left_short_list: list[SlateEntry] = field(default_factory=list)
    # Saves the current-employer scope check moved to worth-a-look.
    scope_moved: int = 0
    coverage: list[FirmCoverage] = field(default_factory=list)
    archetype_coverage: list[ArchetypeCoverage] = field(default_factory=list)
    queries_run: int = 0
    # Queries a brief edit retired, keyed by the note that says which
    # edit did it. Separate from ``skipped`` above, which counts PEOPLE
    # dropped before judgment — a query nobody ran and a person nobody
    # judged are different absences and the slate must not add them up.
    skipped_queries: dict[str, int] = field(default_factory=dict)
    results_total: int = 0
    candidates_total: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    rejected: int = 0
    failed: int = 0
    # Two spend planes, never merged into one number. The campaign
    # figures come from the append-only receipts (every session's rows);
    # the session figures come from this run's meter and this run's
    # token-log rows. A slate that showed only the session's spend would
    # under-report a resumed campaign by everything it already spent.
    campaign_exa_cost_usd: float = 0.0
    campaign_llm_cost_usd: float = 0.0
    session_exa_cost_usd: float = 0.0
    session_llm_cost_usd: float = 0.0
    session_judged: int = 0
    paused: bool = False
    generated_at: str = field(default_factory=_now_utc_iso)

    @property
    def dropped_total(self) -> int:
        return sum(self.dropped.values()) + sum(self.skipped.values())

    def top_drop_reasons(self, limit: int = 2) -> list[tuple[str, int]]:
        merged: dict[str, int] = {}
        for bucket in (self.dropped, self.skipped):
            for reason, count in bucket.items():
                if count:
                    merged[reason] = merged.get(reason, 0) + int(count)
        ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
        return ordered[:limit]

    def all_entries(self) -> list[SlateEntry]:
        return [*self.short_list, *self.worth_a_look, *self.already_known]

    def long_shots(self) -> list[tuple[SlateEntry, str]]:
        """``(entry, section)`` for every long shot, in slate order."""

        return [
            (entry, section)
            for entries, section in (
                (self.short_list, "on the short list"),
                (self.worth_a_look, "worth a look"),
            )
            for entry in entries
            if entry.is_long_shot
        ]

    def verify_entries(self) -> list[SlateEntry]:
        """EVERY entry carrying the verify flag, in every bucket.

        Not just the ones missing a URL, and not just the short list. The
        flag says "this seat is web-derived and may be stale", which is
        true of a person with a perfect LinkedIn URL and true of one
        another campaign already holds. Listing only the URL-less ones
        would let a present URL read as verification, which is exactly
        the inference the flag exists to block.
        """

        return [entry for entry in self.all_entries() if entry.verify_before_outreach]

    def unidentified_entries(self) -> list[SlateEntry]:
        """The subset whose identity, not just whose title, is unconfirmed."""

        return [entry for entry in self.verify_entries() if not entry.has_public_linkedin]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _lede(run: RunView) -> str:
    """Campaign first, session second — in that order, always.

    The slate describes a CAMPAIGN: every person its state dir has ever
    judged under this brief. A resumed session that judged four people
    does not get to present itself as the whole search, so the session's
    own numbers are a trailing clause, clearly labelled.
    """

    drops = run.top_drop_reasons(2)
    drop_clause = ""
    if run.dropped_total:
        if drops:
            named = ", ".join(f"{reason} {count}" for reason, count in drops)
            drop_clause = f", and dropped {run.dropped_total} ({named})"
        else:
            drop_clause = f", and dropped {run.dropped_total}"

    campaign = (
        f"Campaign to date: {_plural(run.queries_run, 'query', 'queries')} returned "
        f"{_plural(run.results_total, 'result')} and produced "
        f"{_plural(run.candidates_total, 'candidate')}{drop_clause} — "
        f"a short list of {len(run.short_list)}, "
        f"{len(run.worth_a_look)} worth a look, and "
        f"{len(run.already_known)} already known to another campaign; "
        f"spend ${run.campaign_exa_cost_usd:.2f} Exa + "
        f"${run.campaign_llm_cost_usd:.2f} LLM."
    )
    session = (
        f"This session (run {run.run_id}): "
        f"{_plural(run.session_judged, 'judged', 'judged')}, "
        f"${run.session_exa_cost_usd:.2f} Exa + "
        f"${run.session_llm_cost_usd:.2f} LLM."
    )
    sentences = [campaign, session]
    if run.paused:
        sentences.append(
            "This session STOPPED SHORT — the query list is not finished; "
            "resume with the same command."
        )
    return " ".join(sentences)


def _reachable_first(entries: list[SlateEntry]) -> list[SlateEntry]:
    """Long shots last; the order within each group is kept."""

    return sorted(entries, key=lambda entry: entry.is_long_shot)


def _head(prefix: str, entry: SlateEntry) -> str:
    """``prefix`` + name, then headline and long-shot marker, closed by a period."""

    tail = f" — {entry.headline}" if entry.headline else ""
    tail += entry.long_shot_marker()
    head = f"{prefix}**{entry.name}**"
    return f"{head}{tail}." if tail else head


def _short_list_lines(run: RunView) -> list[str]:
    lines = [f"## The short list — {len(run.short_list)}", ""]
    if not run.short_list:
        lines.extend(["Nobody cleared the short-list bar on this run.", ""])
        return lines
    for index, entry in enumerate(_reachable_first(run.short_list), start=1):
        lines.append(_head(f"{index}. ", entry))
        lines.append(f"   {entry.rationale_text()}")
        lines.append(f"   Links: {_links(entry)}")
        lines.append("")
    return lines


def _worth_a_look_lines(run: RunView) -> list[str]:
    lines = [f"## Worth a look — {len(run.worth_a_look)}", ""]
    if not run.worth_a_look:
        lines.extend(["Nothing landed in the review bucket.", ""])
        return lines
    lines.append("Each line states the one thing keeping the person off the short list.")
    lines.append("")
    for entry in _reachable_first(run.worth_a_look):
        head = _head("- ", entry)
        lines.append(f"{head} {entry.off_short_list_reason()} {_links(entry)}".rstrip())
    lines.append("")
    return lines


def _long_shot_lines(run: RunView) -> list[str]:
    """The long shots from the two sections above, one line each."""

    long_shots = run.long_shots()
    if not long_shots:
        return []
    lines = [f"{LONG_SHOTS_HEADING} — {len(long_shots)}", "", LONG_SHOTS_EXPLANATION, ""]
    for entry, section in long_shots:
        head = f"- **{entry.name}**"
        if entry.headline:
            head = f"{head} — {entry.headline}"
        reason = entry.attainability_reason or "no reason recorded"
        reason = reason[:1].upper() + reason[1:]
        lines.append(f"{head}. {reason}; {section}.")
    lines.append("")
    return lines


def _left_short_list_lines(run: RunView) -> list[str]:
    """Everyone a re-judge took off the short list, one line each.

    The earlier verdict first, then where the person is now and the reason
    the current verdict gives, so a recruiter can audit each move without
    opening the store. The judge's moves come first; people a deterministic
    gate (the location scope, a blacklist) removed when re-judged follow in
    their own list, because no model weighed them. Each one's
    ``saves.jsonl`` line is followed by a line withdrawing it, so the two
    deliverables agree.
    """

    if not run.left_short_list:
        return []
    judged = [e for e in run.left_short_list if not e.now.startswith("excluded")]
    gated = [e for e in run.left_short_list if e.now.startswith("excluded")]
    lines = [
        f"{LEFT_SHORT_LIST_HEADING} — {len(run.left_short_list)}",
        "",
        "On the short list under earlier judging criteria and judged again under "
        f"the current brief: {len(judged)} on the judge's verdict (two samples; "
        "leaving the slate takes two rejects, and a split goes to worth a look), "
        f"{len(gated)} excluded by a gate with no model call, listed last. Each "
        "one's saves.jsonl line is followed by a line marking it superseded.",
        "",
    ]
    lines.extend(_left_line(entry) for entry in judged)
    if gated:
        if judged:
            lines.append("")
        lines.append(f"Excluded by a gate on re-judge — {len(gated)}:")
        lines.append("")
        lines.extend(_left_line(entry) for entry in gated)
    lines.append("")
    return lines


def _left_line(entry: SlateEntry) -> str:
    head = f"- **{entry.name}**"
    if entry.headline:
        head = f"{head} — {entry.headline}"
    was = entry.previous_decision or "SAVE"
    if isinstance(entry.previous_confidence, (int, float)) and not isinstance(
        entry.previous_confidence, bool
    ):
        was = f"{was} ({entry.previous_confidence:.2f})"
    now = entry.now or "rejected"
    if entry.rejudge_reason or entry.rejudge_note:
        reason = entry.rejudge_reason or entry.rejudge_note
    elif entry.scope_moved:
        reason = f"{entry.scope_note} (judge: {entry.decision or 'SAVE'})"
    else:
        reason = entry.excluded_reason or entry.rationale_text()
    reason = " ".join(reason.split())
    if not reason.endswith((".", "!", "?")):
        reason += "."
    return f"{head}. Was {was}; now {now}: {reason} {_links(entry)}"


def _excluded_lines(run: RunView) -> list[str]:
    """The people the employer blacklist kept off this slate.

    Listed by name when the judge saved them and the render-time check
    pulled them back out, so the operator sees exactly who left the short
    list and why. People the gate rejected before judgment are one count
    line: nobody judged them, so there is nothing to review.
    """

    if not (
        run.excluded_employer
        or run.gated_employer
        or run.excluded_alumni
        or run.gated_alumni
    ):
        return []
    total = (
        len(run.excluded_employer)
        + run.gated_employer
        + len(run.excluded_alumni)
        + run.gated_alumni
    )
    heading = (
        EXCLUDED_ANY_TENSE_HEADING
        if run.excluded_alumni or run.gated_alumni
        else EXCLUDED_HEADING
    )
    lines = [f"{heading} — {total}", ""]
    if run.excluded_employer:
        lines.append(
            "Saved by the judge, then excluded because the record names a "
            "blacklisted employer as their current one. Alumni are not "
            "excluded."
        )
        lines.append("")
        for entry in run.excluded_employer:
            headline = entry.headline
            head = f"- **{entry.name}**"
            if headline:
                head = f"{head} — {headline}."
            reason = entry.excluded_reason or "current employer on the blacklist"
            reason = reason[:1].upper() + reason[1:]
            lines.append(f"{head} {reason}. {_links(entry)}")
        lines.append("")
    if run.gated_employer:
        lines.append(
            f"{_plural(run.gated_employer, 'person', 'people')} rejected before "
            "judgment, with no model call; their rows in judgments.jsonl carry "
            "gate employer_blacklist."
        )
        lines.append("")
    if run.excluded_alumni:
        lines.append(
            "Saved by the judge, then excluded because the record shows they "
            "have worked at an employer on the alumni blacklist, current or past."
        )
        lines.append("")
        for entry in run.excluded_alumni:
            headline = entry.headline
            head = f"- **{entry.name}**"
            if headline:
                head = f"{head} — {headline}."
            reason = entry.excluded_reason or "has worked at an employer on the alumni blacklist"
            reason = reason[:1].upper() + reason[1:]
            lines.append(f"{head} {reason}. {_links(entry)}")
        lines.append("")
    if run.gated_alumni:
        lines.append(
            f"{_plural(run.gated_alumni, 'person', 'people')} rejected before "
            "judgment for having worked at an alumni-blacklisted employer, with "
            "no model call; their rows in judgments.jsonl carry gate "
            "alumni_blacklist."
        )
        lines.append("")
    return lines


def _excluded_location_lines(run: RunView) -> list[str]:
    """The people the brief's location scope kept off this slate.

    Its own section rather than a paragraph under the blacklist heading:
    the reason is where the person lives, not who employs them, and the
    heading names the scope so the count reads on its own.
    """

    if not (run.excluded_location or run.gated_location):
        return []
    label = run.location_scope_label or "the brief's location scope"
    total = len(run.excluded_location) + run.gated_location
    lines = [f"## Excluded: outside {label} — {total}", ""]
    if run.excluded_location:
        lines.append(
            "Saved by the judge, then excluded because the card location on "
            f"the record is outside {label}."
        )
        lines.append("")
        for entry in run.excluded_location:
            headline = entry.headline
            head = f"- **{entry.name}**"
            if headline:
                head = f"{head} — {headline}."
            reason = entry.excluded_reason or f"outside {label}"
            reason = reason[:1].upper() + reason[1:]
            lines.append(f"{head} {reason}. {_links(entry)}")
        lines.append("")
    if run.gated_location:
        lines.append(
            f"{_plural(run.gated_location, 'person', 'people')} rejected before "
            f"judgment for a card location outside {label}, with no model call; "
            f"their rows in judgments.jsonl carry gate location_scope."
        )
        lines.append("")
    return lines


def _already_known_lines(run: RunView) -> list[str]:
    lines = [f"## Already known — {len(run.already_known)}", ""]
    if not run.already_known:
        lines.extend(
            ["No one on this run is already held by another campaign.", ""]
        )
        return lines
    lines.append(
        "Found by Exa and matched, BY NAME, to someone another campaign has "
        "already saved. The match is advisory: this run still judged each of "
        "them on their own evidence and any save still landed. Check the "
        "named campaign before reaching out twice."
    )
    lines.append("")
    for entry in run.already_known:
        label = entry.known_label or "another campaign"
        lines.append(f"- **{entry.name}** — {label}")
    lines.append("")
    return lines


def _coverage_lines(run: RunView) -> list[str]:
    lines = ["## Coverage notes", ""]
    if not run.coverage and not run.archetype_coverage:
        lines.extend(["No per-firm coverage was recorded for this run.", ""])
        return lines
    if not run.coverage:
        lines.extend(["No per-firm coverage was recorded for this run.", ""])
    for firm in run.coverage:
        name = firm.firm or "(no firm)"
        drops = ", ".join(
            f"{reason} {count}"
            for reason, count in sorted(firm.drops.items())
            if count
        )
        detail = (
            f"- **{name}** — {_plural(firm.queries, 'query', 'queries')}, "
            f"{_plural(firm.results, 'result')}, "
            f"{_plural(firm.candidates, 'candidate')}"
        )
        if firm.skipped:
            detail = f"{detail}, {firm.skipped} not run"
        detail = f"{detail}; drops: {drops}." if drops else f"{detail}; no drops."
        lines.append(detail)
        if firm.drops.get("namesake"):
            lines.append(
                f"  Namesake noise: {firm.drops['namesake']} result(s) dropped as "
                f"a different firm sharing the name."
            )
        if firm.candidates == 0 and firm.skipped < firm.queries:
            lines.append(
                "  Zero candidates — the web did not surface anyone in band "
                "at this firm."
            )
    lines.append("")
    lines.extend(_skipped_query_lines(run))
    lines.extend(_archetype_coverage_lines(run))
    return lines


def _skipped_query_lines(run: RunView) -> list[str]:
    """One line per class of query a brief edit retired.

    Under Coverage rather than in its own section, and stated as
    "skipped" rather than left to be inferred from a firm with no
    candidates: the recruiter's question about a firm is always "what
    does the web hold here", and a query nobody ran answers it with
    silence that reads exactly like an empty bench.
    """

    if not run.skipped_queries:
        return []
    lines = []
    for note, count in sorted(run.skipped_queries.items()):
        reason = note or "retired from the brief"
        lines.append(
            f"Skipped — {_plural(count, 'query', 'queries')} ({reason})."
        )
    lines.append("")
    return lines


def _archetype_coverage_lines(run: RunView) -> list[str]:
    """One row per archetype query — the hypothesis, and what it caught.

    Separate from the firm rows because it answers a different
    question. A firm row asks what the web holds at Benchmark; an
    archetype row asks whether the SENTENCE someone wrote describes
    people worth reaching. That is a question about the query, and the
    recruiter's next move on a bad answer is to rewrite the sentence —
    which they can only do if they know which sentence it was.

    The short-list and worth-a-look counts lead the row's second half
    for the same reason. Twenty-five results and zero short list is a
    query that retrieved plenty and found nobody, and on results and
    candidates alone that reads as a healthy row.
    """

    if not run.archetype_coverage:
        return []

    lines = ["### Archetype queries", ""]
    lines.append(
        "Each of these searched a career shape rather than a firm list, so "
        "each row is one hypothesis about who to look for. A query with "
        "results but no short list is a sentence to rewrite, not a firm "
        "with no bench."
    )
    lines.append("")
    for entry in run.archetype_coverage:
        drops = ", ".join(
            f"{reason} {count}"
            for reason, count in sorted(entry.drops.items())
            if count
        )
        detail = (
            f"- **{entry.archetype_id}** — {_plural(entry.results, 'result')}, "
            f"{_plural(entry.candidates, 'candidate')}, "
            f"{entry.short_list} short list, {entry.worth_a_look} worth a look"
        )
        if entry.skipped:
            detail = f"{detail}, {entry.skipped} not run"
        detail = f"{detail}; drops: {drops}." if drops else f"{detail}; no drops."
        lines.append(detail)
        if entry.skipped >= entry.queries:
            lines.append(
                "  Not run — the brief retired this query before it searched."
            )
        if entry.queries > 1:
            lines.append(
                f"  Ran {entry.queries} times across this campaign's sessions."
            )
        if entry.skipped >= entry.queries:
            pass
        elif entry.candidates == 0:
            lines.append(
                "  Zero candidates — nothing the query retrieved was a public "
                "profile in band. Rewrite the description before running it "
                "again."
            )
        elif entry.short_list == 0 and entry.worth_a_look == 0:
            lines.append(
                "  Found people, placed none — the shape as written is "
                "retrieving the wrong population."
            )
    lines.append("")
    return lines


def _verify_lines(run: RunView) -> list[str]:
    entries = run.verify_entries()
    unidentified = run.unidentified_entries()
    lines = [f"## Verify before outreach — {len(entries)}", ""]
    lines.append(
        "Every line above comes from the open web, which lags LinkedIn: "
        "titles and firms go stale and firms inflate titles. Confirm each "
        "person's current seat before any outreach. The "
        f"{_plural(len(unidentified), 'entry', 'entries')} marked "
        "*no LinkedIn URL found* need more than a title check — nothing "
        "here confirms the person is who the page says."
    )
    lines.append("")
    if not entries:
        lines.append("- (none — no entry on this slate carries the flag)")
    else:
        for entry in entries:
            firm = f" — {entry.firm}" if entry.firm else ""
            marker = "" if entry.has_public_linkedin else " — no LinkedIn URL found"
            lines.append(f"- **{entry.name}**{firm}{marker}. {_links(entry)}")
    lines.append("")
    return lines


def _links(entry: SlateEntry) -> str:
    urls: list[str] = []
    if entry.linkedin_url:
        urls.append(entry.linkedin_url)
    for url in entry.evidence_urls:
        if url and url not in urls:
            urls.append(url)
    return " ; ".join(urls) if urls else "(no links recorded)"


def render_slate(run: RunView) -> str:
    """Render the full slate markdown for one run."""

    lines = [
        f"# {run.brief_name or 'Exa sourcing'} — Exa web-sourcing slate "
        f"(run {run.run_id})",
        "",
        _lede(run),
        "",
    ]
    lines.extend(_short_list_lines(run))
    lines.extend(_worth_a_look_lines(run))
    lines.extend(_long_shot_lines(run))
    lines.extend(_left_short_list_lines(run))
    lines.extend(_excluded_lines(run))
    lines.extend(_excluded_location_lines(run))
    lines.extend(_already_known_lines(run))
    lines.extend(_coverage_lines(run))
    lines.extend(_verify_lines(run))
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Structured report
# ---------------------------------------------------------------------------


def structured_report_payload(run: RunView, *, role_title: str = "") -> dict:
    """Build the ``run-report.json`` payload for one Exa run.

    Keep analytical debrief sections empty because this report performs no
    model-authored debrief. ``run_metadata``, ``metrics_summary``, and
    ``string_performance`` come from the run's query rows. Include ``role_title``
    and ``run_id`` so snapshot handling can tie the report to its brief and run.
    """

    return {
        "schema_version": 1,
        "run_metadata": {
            "source": "exa",
            "role_title": role_title or run.brief_name,
            "brief_name": run.brief_name,
            "run_id": run.run_id,
            "generated_at": run.generated_at,
            "paused": run.paused,
        },
        "metrics_summary": {
            "queries_run": run.queries_run,
            "results": run.results_total,
            "candidates_evaluated": run.candidates_total,
            "saved": len(run.short_list),
            "review": len(run.worth_a_look),
            "rejected": run.rejected,
            "failed": run.failed,
            "already_known": len(run.already_known),
            "long_shots": len(run.long_shots()),
            "excluded_employer_blacklist": len(run.excluded_employer),
            "gated_employer_blacklist": run.gated_employer,
            "excluded_alumni_blacklist": len(run.excluded_alumni),
            "gated_alumni_blacklist": run.gated_alumni,
            "excluded_location_scope": len(run.excluded_location),
            "gated_location_scope": run.gated_location,
            "moved_to_review_by_scope": run.scope_moved,
            "left_short_list_on_rejudge": len(run.left_short_list),
            "left_short_list_on_rejudge_by_gate": sum(
                1 for entry in run.left_short_list if entry.now.startswith("excluded")
            ),
            "dropped": run.dropped_total,
            "drops_by_reason": dict(run.dropped),
            "skips_by_reason": dict(run.skipped),
            "verify_before_outreach": len(run.verify_entries()),
            "no_public_linkedin_url": len(run.unidentified_entries()),
            "session_judged": run.session_judged,
            "session_exa_cost_usd": round(run.session_exa_cost_usd, 6),
            "session_llm_cost_usd": round(run.session_llm_cost_usd, 6),
            "campaign_exa_cost_usd": round(run.campaign_exa_cost_usd, 6),
            "campaign_llm_cost_usd": round(run.campaign_llm_cost_usd, 6),
        },
        # Keep archetype rows inside string_performance because the report schema
        # has a fixed top-level shape. Namespace the row with ``firm`` and retain
        # ``archetype_id`` plus bucket counts for consumers that understand it.
        "string_performance": [
            {
                "firm": firm.firm,
                "queries": firm.queries,
                "results": firm.results,
                "candidates": firm.candidates,
                "skipped": firm.skipped,
                "drops": dict(firm.drops),
            }
            for firm in run.coverage
        ]
        + [
            {
                "firm": f"archetype:{entry.archetype_id}",
                "archetype_id": entry.archetype_id,
                "queries": entry.queries,
                "results": entry.results,
                "candidates": entry.candidates,
                "skipped": entry.skipped,
                "short_list": entry.short_list,
                "worth_a_look": entry.worth_a_look,
                "drops": dict(entry.drops),
            }
            for entry in run.archetype_coverage
        ],
        "winning_lanes": [],
        "underperforming_lanes": [],
        # A row nobody ran is not a gap in the WEB, which is what a
        # coverage gap means downstream. Retired queries are reported as
        # skipped and excluded here, or brief iteration would be told to
        # widen a firm whose queries the brief itself just switched off.
        "coverage_gaps": [
            {"firm": firm.firm, "reason": "no candidates surfaced"}
            for firm in run.coverage
            if firm.candidates == 0 and firm.skipped < firm.queries
        ]
        + [
            {
                "firm": f"archetype:{entry.archetype_id}",
                "reason": "no candidates surfaced",
            }
            for entry in run.archetype_coverage
            if entry.candidates == 0 and entry.skipped < entry.queries
        ],
        "noise_patterns": [],
        "saved_candidate_patterns": {},
        "adaptation_assessment": {},
        "recommendations": {},
        "brief_iteration_hints": {},
    }


__all__ = [
    "EXCLUDED_ANY_TENSE_HEADING",
    "EXCLUDED_HEADING",
    "LEFT_SHORT_LIST_HEADING",
    "LONG_SHOTS_HEADING",
    "NO_RATIONALE",
    "ArchetypeCoverage",
    "SECTION_ORDER",
    "FirmCoverage",
    "RunView",
    "SlateEntry",
    "render_slate",
    "structured_report_payload",
]
