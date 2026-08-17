"""Pre-launch investigation for executive search briefs.

Slice 9 of the executive-search module. Net-new module per the
spec's 2026-05-03 amendment B: ``market_intelligence/engine.py`` is
post-run only (the candidate citations at L759-811 and L1621-1626
both run after a finalized run; no ``pre_launch`` mode exists).
This module is the brief-only execution surface.

Design contract:

- :func:`run_pre_launch_investigation` accepts a brief path + a
  prior-search context, returns an :class:`InvestigationPacket`.
- Reuses the existing pluggable backends (``PlannerBackend``,
  ``ExternalResearchBackend``, ``MarketIntelSynthesisBackend``)
  from ``market_intelligence/engine.py`` — this slice does NOT
  fork those interfaces. ``CriticBackend`` is skipped because it
  depends on run-state evidence the pre-launch pass doesn't have.
- Persists the packet at
  ``output/state/exec_search/<state_key>/investigation_packet.json``
  per the old spec at ``docs/exec-search-workflow-spec.md:126``.
- Honors ``brief.prior_search.ruled_out_urls`` so research excludes
  candidates already approached or formally ruled out.
- NEVER raises out to callers. Failures are typed
  :class:`InvestigationFailure` with a recruiter-readable detail.

Slice 9 ships the brief-only execution path + tests; Slice 9b (a
follow-up) wires the Cloris UI investigation review screen.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from shared.brief_loader import Brief as CompatBrief, load_brief
from shared.output_paths import resolve_exec_search_state_dir


_INVESTIGATION_PACKET_FILENAME = "investigation_packet.json"


@dataclass(frozen=True)
class InvestigationFinding:
    """One research observation for the recruiter to review pre-launch."""

    topic: str
    finding: str
    citations: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class InvestigationPacket:
    """Brief-only research packet emitted by the pre-launch flow.

    Returned by :func:`run_pre_launch_investigation` and persisted to
    disk. The recruiter reviews this in the Cloris UI before
    committing to a launch. Edits to the brief at this point feed
    into the strategy-formation phase per the old spec's Stage 0.
    """

    brief_id: str
    brief_path: str
    role_title: str
    geography: str
    confidentiality_class: str
    generated_at_iso: str
    findings: tuple[InvestigationFinding, ...] = ()
    market_context: str = ""
    sourcing_recommendations: tuple[str, ...] = ()
    excluded_urls: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "brief_id": self.brief_id,
            "brief_path": self.brief_path,
            "role_title": self.role_title,
            "geography": self.geography,
            "confidentiality_class": self.confidentiality_class,
            "generated_at_iso": self.generated_at_iso,
            "findings": [asdict(f) for f in self.findings],
            "market_context": self.market_context,
            "sourcing_recommendations": list(self.sourcing_recommendations),
            "excluded_urls": list(self.excluded_urls),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class InvestigationFailure:
    """Typed failure result from the pre-launch flow.

    Mirrors the per-source failure shape used by the off-LinkedIn
    signal adapters. Reasons:
    ``"brief_not_found"``, ``"brief_load_error"``,
    ``"backend_failure"``, ``"persistence_error"``.
    """

    reason: str
    detail: str = ""


def run_pre_launch_investigation(
    *,
    brief_path: str | Path,
    prior_search_context: Mapping[str, Any] | None = None,
    research_backend: Any | None = None,
    persist: bool = True,
) -> InvestigationPacket | InvestigationFailure:
    """Execute the brief-only pre-launch investigation.

    Returns an :class:`InvestigationPacket` on success. Returns an
    :class:`InvestigationFailure` for any error class. Never raises.

    ``research_backend`` is an optional pluggable backend (mirrors
    the ``ExternalResearchBackend`` shape from
    ``market_intelligence/engine.py``). When ``None``, the function
    runs in heuristic mode: emits a packet from the brief alone with
    a "research not run" note. This matches the
    "no LLM available" cascade-fallback posture used elsewhere in
    the codebase, so the pre-launch flow degrades gracefully.

    ``persist=True`` (default) writes the packet to disk at the
    canonical exec-search investigation path. ``persist=False`` is
    used by the API handler when the recruiter wants to preview the
    packet without committing it.
    """

    brief_path = Path(brief_path)
    if not brief_path.exists():
        return InvestigationFailure(
            reason="brief_not_found",
            detail=f"Brief file not found at {brief_path}",
        )
    try:
        brief = load_brief(str(brief_path))
    except Exception as exc:
        return InvestigationFailure(
            reason="brief_load_error",
            detail=f"{exc.__class__.__name__}: {exc}",
        )

    excluded_urls = _excluded_urls_from(brief, prior_search_context)

    findings, market_context, sourcing_recommendations, notes = _run_research(
        brief=brief,
        excluded_urls=excluded_urls,
        research_backend=research_backend,
    )

    packet = InvestigationPacket(
        brief_id=str(getattr(brief, "id", "") or brief.role_title),
        brief_path=str(brief_path),
        role_title=brief.role_title or "",
        geography=brief.permanent_filters.get("Location", "")
        if isinstance(brief.permanent_filters, dict)
        else "",
        confidentiality_class=str(
            getattr(brief, "confidentiality_class", "open") or "open"
        ),
        generated_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        findings=tuple(findings),
        market_context=market_context,
        sourcing_recommendations=tuple(sourcing_recommendations),
        excluded_urls=tuple(excluded_urls),
        notes=notes,
    )

    if persist:
        try:
            _persist_packet(brief_path=brief_path, brief=brief, packet=packet)
        except Exception as exc:
            return InvestigationFailure(
                reason="persistence_error",
                detail=f"{exc.__class__.__name__}: {exc}",
            )

    return packet


def _run_research(
    *,
    brief: CompatBrief,
    excluded_urls: list[str],
    research_backend: Any | None,
) -> tuple[list[InvestigationFinding], str, list[str], str]:
    """Run the pluggable research backend, or fall back to heuristic.

    Returns a tuple of ``(findings, market_context,
    sourcing_recommendations, notes)``.

    The backend interface (mirrored from
    ``ExternalResearchBackend`` in ``market_intelligence/engine.py``)
    is duck-typed: any object with a ``research_brief(brief,
    excluded_urls) -> dict`` method works. Tests inject a stub.
    """

    if research_backend is None:
        return (
            [],
            "",
            _heuristic_sourcing_recommendations(brief),
            (
                "Pre-launch investigation ran in heuristic mode (no "
                "research backend configured). The packet reflects the "
                "brief alone; recruiter should review the brief criteria "
                "before launch and consider configuring "
                "PERPLEXITY_API_KEY for richer pre-launch context."
            ),
        )
    try:
        result = research_backend.research_brief(
            brief=brief, excluded_urls=excluded_urls
        )
    except Exception as exc:
        return (
            [],
            "",
            _heuristic_sourcing_recommendations(brief),
            (
                "Research backend raised — degrading to heuristic mode. "
                f"Detail: {exc.__class__.__name__}: {exc}"
            ),
        )
    findings_raw = result.get("findings") if isinstance(result, Mapping) else None
    findings: list[InvestigationFinding] = []
    if isinstance(findings_raw, list):
        for raw in findings_raw:
            if not isinstance(raw, Mapping):
                continue
            findings.append(
                InvestigationFinding(
                    topic=str(raw.get("topic") or ""),
                    finding=str(raw.get("finding") or ""),
                    citations=tuple(
                        c for c in (raw.get("citations") or []) if isinstance(c, str)
                    ),
                    confidence=float(raw.get("confidence") or 0.0),
                )
            )
    market_context = ""
    if isinstance(result, Mapping):
        market_context = str(result.get("market_context") or "")
    recs_raw = result.get("sourcing_recommendations") if isinstance(result, Mapping) else None
    sourcing_recommendations: list[str] = []
    if isinstance(recs_raw, list):
        sourcing_recommendations = [r for r in recs_raw if isinstance(r, str)]
    if not sourcing_recommendations:
        sourcing_recommendations = _heuristic_sourcing_recommendations(brief)
    return findings, market_context, sourcing_recommendations, ""


def _heuristic_sourcing_recommendations(brief: CompatBrief) -> list[str]:
    """Generate recruiter-readable sourcing recommendations from the brief alone.

    Used when no research backend is configured. Pulls from the V2
    brief's capability areas and depth_distinction so the packet
    surfaces something useful even in offline mode.
    """

    recs: list[str] = []
    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is None:
        return recs
    if getattr(new_brief, "dossier_mode", False):
        recs.append(
            "Bias the LinkedIn strategy toward title_first architecture "
            "(executive recruiters know the title; capability layers add "
            "noise more than recall at this level)."
        )
    capability_areas = getattr(new_brief, "capability_areas", None) or []
    if capability_areas:
        recs.append(
            "Anchor the executive-register full-eval prompt on the "
            f"{len(capability_areas)} capability area(s) the recruiter "
            "encoded; the depth_distinction does the load-bearing work."
        )
    if getattr(new_brief, "executive_calibration", None) is not None:
        recs.append(
            "executive_calibration is populated — feed sector / stage / "
            "P&L vocabulary into the dossier prompt's user message so "
            "the rationale uses operator-grade phrasing."
        )
    if (
        getattr(new_brief, "prior_search", None)
        and len(getattr(new_brief.prior_search, "ruled_out_urls", []) or []) > 0
    ):
        recs.append(
            "Prior-search exclusion is non-empty — Slice 10's "
            "_load_candidate_history will merge ruled_out_urls into "
            "the LinkedIn orchestrator's _seen_urls at session init."
        )
    return recs


def _excluded_urls_from(
    brief: CompatBrief, prior_search_context: Mapping[str, Any] | None
) -> list[str]:
    """Build the merged exclusion list.

    Sources, in priority order:

      1. Caller-supplied ``prior_search_context.ruled_out_urls`` (the
         API handler can pass a session-scoped exclusion list).
      2. ``brief._new_brief.prior_search.ruled_out_urls`` (recruiter-
         authored).
      3. ``brief.prior_search.ruled_out_urls`` (compat-Brief mirror).

    De-duplicates while preserving caller-priority order.
    """

    seen: set[str] = set()
    out: list[str] = []

    def _add(url: Any) -> None:
        if not isinstance(url, str) or not url:
            return
        if url in seen:
            return
        seen.add(url)
        out.append(url)

    if isinstance(prior_search_context, Mapping):
        for url in prior_search_context.get("ruled_out_urls") or []:
            _add(url)

    new_brief = getattr(brief, "_new_brief", None)
    if new_brief is not None:
        for url in getattr(getattr(new_brief, "prior_search", None), "ruled_out_urls", []) or []:
            _add(url)

    compat_prior_search = getattr(brief, "prior_search", None)
    if compat_prior_search is not None:
        for url in getattr(compat_prior_search, "ruled_out_urls", []) or []:
            _add(url)

    return out


def _persist_packet(
    *, brief_path: Path, brief: CompatBrief, packet: InvestigationPacket
) -> None:
    """Write the packet to the canonical exec-search investigation path."""

    state_dir = resolve_exec_search_state_dir(
        brief_path=brief_path, brief=brief,
    )
    investigation_path = state_dir / _INVESTIGATION_PACKET_FILENAME
    investigation_path.write_text(json.dumps(packet.to_dict(), indent=2))


__all__ = (
    "InvestigationFailure",
    "InvestigationFinding",
    "InvestigationPacket",
    "run_pre_launch_investigation",
)
