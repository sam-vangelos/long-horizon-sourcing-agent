"""Designer module dataclasses — candidate + snippet shapes.

Designer Slice 2. The schema layer separates source-shape (Behance v2
dicts) from the unified evaluation surface (:class:`DesignerCandidate`,
:class:`DesignerSnippet`). Slice 3 extends with Google CSE shapes
(thumbnail-only, no project structure); Slice 5 adds the multimodal
asset references the vision-evaluation pipeline grounds itself in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal


SourceProvenance = Literal["behance", "google_cse", "dribbble"]


@dataclass(frozen=True)
class DesignerProjectSummary:
    """Light-weight project record returned by Behance's project list
    endpoint. No images yet — that comes from
    :func:`designer.sources.behance.BehanceClient.get_project`.
    """

    project_id: int
    title: str
    cover_image_url: str
    appreciation_count: int = 0
    view_count: int = 0
    published_at: str = ""
    fields: tuple[str, ...] = ()  # Behance creative-fields tags
    description: str = ""


@dataclass(frozen=True)
class DesignerSnippet:
    """The minimum payload a facial-stage judge needs to call
    ``designer_facial_judge_*``.

    Mirrors :class:`shared.schemas.CandidateSnippet` (LinkedIn) and
    :class:`github.schemas.GitHubSnippet` shape: a few text fields the
    judge can ground on without further API calls. The vision pipeline
    in Slice 5 receives a separate enriched payload (:class:`DesignerCandidate`
    + image acquisition output).
    """

    source: SourceProvenance
    identity_key: str  # `behance:<username>` or `cse:<portfolio_url>`
    display_name: str
    profile_url: str
    location: str = ""
    headline: str = ""
    fields: tuple[str, ...] = ()  # Behance creative-fields or CSE-derived tags
    tools: tuple[str, ...] = ()  # tool stack signals (Figma, After Effects, …)
    top_project_titles: tuple[str, ...] = ()
    appreciation_count_total: int = 0
    social_links: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DesignerCandidate:
    """The full evaluation payload the orchestrator hands to the
    full-stage judge and (in Slice 5) the vision-evaluation pipeline.

    The text-side fields (``snippet``, ``project_summaries``) carry the
    contextualization the LLM grounds on. The image-side fields
    (``project_image_urls``, ``cached_asset_paths``) arrive in Slice 5
    when image acquisition lands.
    """

    snippet: DesignerSnippet
    project_summaries: tuple[DesignerProjectSummary, ...] = ()
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignerSearchQuery:
    """One discovery query against a single source.

    The orchestrator turns each :class:`DesignerSearchQuery` into a
    work_unit (kind ``DESIGNER_BEHANCE_QUERY_KIND`` for Behance or
    ``DESIGNER_CSE_QUERY_KIND`` for Google CSE — Slice 3). The query
    payload is what gets serialized into ``work_unit.payload_json``.
    """

    source: SourceProvenance
    query_text: str  # free-text search; Behance ``q`` param or CSE ``q``
    sort: str = "appreciations"  # Behance: appreciations | views | published_date
    capability_area_name: str = ""  # which brief.capability_area drove this query
    discipline: str = ""  # rubric discipline tag (product/brand/motion/...)
    extra_filters: dict[str, Any] = field(default_factory=dict)


def behance_user_to_snippet(
    user: dict[str, Any],
    *,
    project_titles: tuple[str, ...] = (),
    tools: tuple[str, ...] = (),
) -> DesignerSnippet:
    """Map a Behance ``/v2/users`` response item to :class:`DesignerSnippet`.

    Defensive against missing fields — Behance occasionally returns
    null values for ``city`` / ``state`` / ``country`` / ``occupation``
    on profiles where the user hasn't filled them in.
    """

    username = str(user.get("username") or "")
    display_name = str(user.get("display_name") or username)
    location_parts = [
        part
        for part in (user.get("city"), user.get("state"), user.get("country"))
        if isinstance(part, str) and part
    ]
    location = ", ".join(location_parts)

    fields_value = user.get("fields") or []
    if isinstance(fields_value, list):
        fields = tuple(str(f) for f in fields_value if isinstance(f, str))
    else:
        fields = ()

    profile_url = (
        user.get("url")
        or (f"https://www.behance.net/{username}" if username else "")
    )

    raw_links = user.get("social_links") or []
    social_links: list[tuple[str, str]] = []
    if isinstance(raw_links, list):
        for link in raw_links:
            if isinstance(link, dict):
                service = str(link.get("social_media_service") or link.get("service") or "")
                url = str(link.get("url") or "")
                if service and url:
                    social_links.append((service, url))

    return DesignerSnippet(
        source="behance",
        identity_key=f"behance:{username}" if username else "behance:_unknown_",
        display_name=display_name,
        profile_url=str(profile_url),
        location=location,
        headline=str(user.get("occupation") or ""),
        fields=fields,
        tools=tools,
        top_project_titles=project_titles,
        appreciation_count_total=int(
            user.get("stats", {}).get("appreciations", 0)
            if isinstance(user.get("stats"), dict)
            else 0
        ),
        social_links=tuple(social_links),
    )


def cse_item_to_snippet(item: dict[str, Any]) -> DesignerSnippet | None:
    """Map a Google CSE result item to :class:`DesignerSnippet`, or
    return None if the item lacks the minimum identity fields.

    CSE results don't carry the structured taxonomy Behance does, so
    most :class:`DesignerSnippet` fields end up empty. The vision-
    evaluation pipeline (Slice 5) is the layer that recovers signal
    from CSE-discovered candidates — the thumbnail + the host domain
    are sufficient input to anchor the visual pass.
    """

    from designer.sources.google_cse import (
        cse_result_thumbnail_url,
        cse_result_to_display_name,
        cse_result_to_identity_key,
    )

    link = item.get("link")
    if not isinstance(link, str) or not link:
        return None

    identity_key = cse_result_to_identity_key(link)
    display_name = cse_result_to_display_name(item)

    return DesignerSnippet(
        source="google_cse",
        identity_key=identity_key,
        display_name=display_name or identity_key.split(":", 1)[-1],
        profile_url=link,
        location="",
        headline="",
        fields=(),
        tools=(),
        top_project_titles=(),
        appreciation_count_total=0,
    )


def _strip_html(text: str) -> str:
    """Strip HTML tags from a string, collapsing whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def behance_project_to_summary(project: dict[str, Any]) -> DesignerProjectSummary:
    """Map a Behance ``/v2/users/{u}/projects`` response item to a summary."""

    covers = project.get("covers") or {}
    cover_url = ""
    if isinstance(covers, dict):
        # Behance returns covers keyed by size: "115", "202", "404", "808",
        # "max_808", "original". Prefer the largest the API offers.
        for size_key in ("original", "max_808", "808", "404", "202", "115"):
            value = covers.get(size_key)
            if isinstance(value, str) and value:
                cover_url = value
                break

    fields_value = project.get("fields") or []
    if isinstance(fields_value, list):
        fields = tuple(str(f) for f in fields_value if isinstance(f, str))
    else:
        fields = ()

    stats = project.get("stats") or {}
    if not isinstance(stats, dict):
        stats = {}

    return DesignerProjectSummary(
        project_id=int(project.get("id") or 0),
        title=str(project.get("name") or ""),
        cover_image_url=cover_url,
        appreciation_count=int(stats.get("appreciations", 0) or 0),
        view_count=int(stats.get("views", 0) or 0),
        published_at=str(project.get("published_on") or ""),
        fields=fields,
        description=_strip_html(str(project.get("description") or ""))[:2000],
    )
