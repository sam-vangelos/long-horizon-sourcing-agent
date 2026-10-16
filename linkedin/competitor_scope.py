"""Keep every LinkedIn string of an allowlisted campaign inside a competitor facet.

A brief's ``employer_allowlist`` (shared/employer_allowlist.py) names the
current employers a campaign is bounded to and the company facet groups that
carry them. The strategist is told to facet every string; this module makes
that deterministic, so the restriction no longer depends on a prompt:

- :func:`stamp_competitor_facet` runs on every string the initial plan, an
  adaptation, or any other path queues. It maps aliases to their listed
  parent, drops unlisted companies, caps the facet at
  :data:`~shared.employer_allowlist.FACET_COMPANY_VALUE_CAP`, and gives a
  string with no listed company one facet group: the group its Boolean
  already names, else the group the queue carries least.
- :func:`competitor_facet_skip_reason` reads the opening facet receipt. A
  string none of whose listed companies landed on the rail would run
  keyword-only across the country, so it is skipped with a logged reason.
- :func:`stamp_competitor_facet_for_brief` is the stamp under a brief, with
  its one console line and run-log event; the driver and the page allocator
  call it wherever a string's experiment state is first bootstrapped.
- :func:`company_facet_scope` is the scope every company chip carries under
  an allowlist: CURRENT, since the facet's default also matches alumni.
- :func:`competitor_scope_prompt_block` is the short scope rule the
  adaptation and variant prompts carry, which never saw the seed's
  instructions.

Title-restricted employers (the allowlist's ``title_restricted`` block, such
as an AI lab counted only on its human-data side) ride only their own facet group, and
only with the block's family titles on the title facet. The stamp never hands
that group to an unfaceted string, drops those employers from a facet that
mixes them with others, and puts the family titles on every string they
bound; the skip refuses such a string whose title facet is empty or landed a
title outside the families.

Every function is inert when the brief has no well-formed allowlist.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from shared.employer_allowlist import (
    FACET_COMPANY_VALUE_CAP,
    EmployerAllowlist,
    employer_allowlist_from_brief,
)
from shared.storage import log_event

__all__ = [
    "company_facet_scope",
    "competitor_facet_skip_reason",
    "competitor_scope_prompt_block",
    "scoped_facet_companies",
    "stamp_competitor_facet",
    "stamp_competitor_facet_for_brief",
    "stamp_competitor_facets",
]


def scoped_facet_companies(
    companies: Iterable[object], allowlist: EmployerAllowlist
) -> list[str]:
    """The listed employers among ``companies``, aliases mapped, deduped, capped."""

    scoped: list[str] = []
    for company in companies or []:
        canonical = allowlist.canonical_for(company)
        if canonical and canonical not in scoped:
            scoped.append(canonical)
    return scoped[:FACET_COMPANY_VALUE_CAP]


def _group_usage(
    queue: Iterable[Any], allowlist: EmployerAllowlist
) -> Counter[int]:
    usage: Counter[int] = Counter()
    groups = [set(group) for group in allowlist.facet_groups]
    for item in queue or []:
        filters = getattr(item, "structured_filters", None) or {}
        companies = filters.get("companies") if isinstance(filters, dict) else None
        names = set(scoped_facet_companies(companies or [], allowlist))
        for index, group in enumerate(groups):
            if names & group:
                usage[index] += 1
    return usage


def _choose_group(
    boolean: str, allowlist: EmployerAllowlist, usage: Counter[int]
) -> tuple[str, ...]:
    mentioned = allowlist.mentioned_in(boolean)
    # A title-restricted group is never assigned by default: it runs only on a
    # string that named it and carries its title families.
    candidates = [
        index
        for index, group in enumerate(allowlist.facet_groups)
        if not any(allowlist.is_restricted(name) for name in group)
    ] or list(range(len(allowlist.facet_groups)))
    best = min(
        candidates,
        key=lambda index: (
            -len(mentioned & set(allowlist.facet_groups[index])),
            usage[index],
            index,
        ),
    )
    return allowlist.facet_groups[best]


def stamp_competitor_facet(
    search_string: Any,
    allowlist: EmployerAllowlist | None,
    *,
    queue: Iterable[Any] = (),
) -> dict[str, Any] | None:
    """Put the string's company facet inside the allowlist; a record when it changed.

    ``action`` is ``"stamped"`` (no listed company, a group was assigned) or
    ``"scoped"`` (unlisted values dropped, aliases mapped, or the facet
    capped). A string already inside the scope is returned untouched as None.
    """

    if allowlist is None:
        return None
    filters = dict(getattr(search_string, "structured_filters", None) or {})
    before = [str(value) for value in (filters.get("companies") or [])]
    after = scoped_facet_companies(before, allowlist)
    if any(allowlist.is_restricted(name) for name in after) and not all(
        allowlist.is_restricted(name) for name in after
    ):
        # A facet mixing clients with competitors keeps the competitors.
        after = [name for name in after if not allowlist.is_restricted(name)]
    action = "scoped"
    if not after:
        after = list(_choose_group(search_string.boolean, allowlist, _group_usage(queue, allowlist)))
        after = after[:FACET_COMPANY_VALUE_CAP]
        action = "stamped"
    titles_before = [str(value) for value in (filters.get("titles") or [])]
    titles_after = titles_before
    if after and all(allowlist.is_restricted(name) for name in after):
        titles_after = [title for title in titles_before if allowlist.title_in_family(title)]
        titles_after = (titles_after or list(allowlist.facet_titles))[:FACET_COMPANY_VALUE_CAP]
    if after == before and titles_after == titles_before:
        return None
    filters["companies"] = after
    if titles_after != titles_before:
        filters["titles"] = titles_after
    search_string.structured_filters = filters
    if search_string.acquisition_mode in ("", "linkedin_boolean"):
        search_string.acquisition_mode = "linkedin_hybrid"
    if search_string.surface == "boolean":
        search_string.surface = "hybrid"
    note = f" Competitor facet {action}: {', '.join(after)}."
    if titles_after != titles_before:
        note += f" Client title facet: {', '.join(titles_after)}."
    search_string.notes = ((search_string.notes or "") + note).strip()
    record = {
        "string_id": search_string.id,
        "action": action,
        "before": before,
        "after": list(after),
    }
    if titles_after != titles_before:
        record["titles_before"] = titles_before
        record["titles_after"] = list(titles_after)
    return record


def stamp_competitor_facet_for_brief(
    search_string: Any,
    brief: Any,
    *,
    queue: Iterable[Any] = (),
    log_path: Any = None,
) -> dict[str, Any] | None:
    """Stamp under ``brief``'s allowlist; one console line and event when it changed."""

    allowlist = employer_allowlist_from_brief(brief)
    if allowlist is None:
        return None
    record = stamp_competitor_facet(search_string, allowlist, queue=list(queue or []))
    if record is None:
        return None
    titles = record.get("titles_after")
    print(
        f"    [scope] String #{record['string_id']}: competitor facet "
        f"{record['action']} -> {', '.join(record['after'])}"
        + (f"; client titles -> {', '.join(titles)}" if titles else "")
    )
    if log_path:
        log_event(log_path, "competitor_facet_stamped", **record)
    return record


def stamp_competitor_facets(
    strings: Iterable[Any],
    allowlist: EmployerAllowlist | None,
    *,
    queue: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Stamp each string in turn; later strings see the groups earlier ones took."""

    if allowlist is None:
        return []
    pool = list(queue) if queue is not None else list(strings)
    records: list[dict[str, Any]] = []
    for search_string in strings:
        record = stamp_competitor_facet(search_string, allowlist, queue=pool)
        if record is not None:
            records.append(record)
    return records


def competitor_facet_skip_reason(
    requested: Iterable[object],
    receipt: dict[str, Any] | None,
    allowlist: EmployerAllowlist | None,
    *,
    requested_titles: Iterable[object] | None = None,
) -> str | None:
    """Why a string must not run, or None.

    ``requested`` is the active variant's company facet and ``receipt`` the
    opening reconcile receipt (None when no reconcile ran, which this cannot
    judge). A string with no listed company requested, or none landed on the
    rail (every value unapplied, or taken as a near miss onto an unlisted
    entity), would search the whole Location filter on keywords alone.

    A string bounded by title-restricted employers must also request
    (``requested_titles``, when given) and land only family titles on its
    title facet: without them it would page through the client's engineers.
    """

    if allowlist is None:
        return None
    wanted = scoped_facet_companies(requested, allowlist)
    if not wanted:
        return "the string carries no listed competitor on its company facet"
    client = [name for name in wanted if allowlist.is_restricted(name)]
    if client and requested_titles is not None:
        asked = [str(title).strip() for title in requested_titles if str(title).strip()]
        if not asked or not all(allowlist.title_in_family(title) for title in asked):
            return (
                "a client-side string must carry only human-data, post-training or "
                f"evaluation titles on its title facet (requested: {', '.join(asked) or 'none'})"
            )
    if not isinstance(receipt, dict):
        return None
    after = (receipt.get("after") or {}).get("companies") or []
    landed = [value for value in after if allowlist.listed_entity(value)]
    if landed:
        if any(allowlist.is_restricted(allowlist.listed_entity(value)) for value in landed):
            titles = [str(t) for t in ((receipt.get("after") or {}).get("titles") or [])]
            if not titles:
                return (
                    "the client-side title facet did not land "
                    f"(companies: {', '.join(map(str, landed))})"
                )
            off = [title for title in titles if not allowlist.title_in_family(title)]
            if off:
                return (
                    "a title outside the client-side families landed on the title "
                    f"facet: {', '.join(off)}"
                )
        return None
    unapplied = (receipt.get("unapplied") or {}).get("companies") or []
    substituted = (receipt.get("substituted") or {}).get("companies") or {}
    detail = []
    if unapplied:
        detail.append(f"unapplied: {', '.join(map(str, unapplied))}")
    if substituted:
        detail.append(
            "taken as: "
            + ", ".join(f"{key} -> {value}" for key, value in substituted.items())
        )
    suffix = f" ({'; '.join(detail)})" if detail else ""
    return (
        "no listed competitor landed on the company facet "
        f"(requested: {', '.join(wanted)}){suffix}"
    )


def company_facet_scope(brief: Any) -> str | None:
    """``"CURRENT"`` when ``brief`` carries a well-formed allowlist, else None."""

    return "CURRENT" if employer_allowlist_from_brief(brief) is not None else None


def competitor_scope_prompt_block(brief: Any) -> str:
    """The scope rule for prompts that never see the seed's instructions, or ""."""

    allowlist = employer_allowlist_from_brief(brief)
    if allowlist is None:
        return ""
    groups = "; ".join(f"[{', '.join(group)}]" for group in allowlist.facet_groups)
    client_rule = ""
    if allowlist.restricted:
        clients = [name for name in allowlist.names if allowlist.is_restricted(name)]
        client_rule = (
            f" The client companies ({', '.join(clients)}) ride only their own facet "
            "group and only with human-data, post-training or evaluation titles on the "
            f"title facet ({', '.join(allowlist.facet_titles)}), never with engineering "
            "or research titles; a client employee in any other current seat is "
            "skipped before judgment."
        )
    return (
        "\n\nCOMPETITOR SCOPE (deterministic; the run enforces it): this campaign "
        "sources only people CURRENTLY employed at the listed companies. Every "
        "string carries `structured_filters.companies` with one of these facet "
        f"groups: {groups}. A string or variant without one is stamped with a "
        "group, an unlisted company is dropped from the facet, and a string "
        "whose facet does not land is skipped. Never demote or drop the companies "
        "dimension and never promote an unlisted company into it; vary the "
        "keyword side instead. Cards whose current employer is not listed are "
        "skipped before judgment."
        + client_rule
    )
