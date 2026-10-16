"""Brief polish — Cloris-voice reshape over recruiter intake captures.

Transforms intake chapter captures into a schema-correct V2 brief draft at
intake time. The sibling :mod:`market_intelligence.briefing_polish` module
performs post-run editorial briefing work.

:class:`BriefPolishBackend` tries the LLM rewrite and falls back to the
deterministic :class:`HeuristicBriefPolishBackend` on validation or runtime
failure. The heuristic mirrors the frontend ``seedV2DraftFromChapters`` for
structured fields and validates the result with :func:`validate_v2_brief`.

Confidence is programmatic: the heuristic uses ``populated_chapter_fields / 7``
and successful LLM output uses ``1.0``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from shared.brief_v2_schema import (
    BriefSchemaError,
    validate_v2_brief,
)
# Use the cached LLM client because the system prompt is shared across polish
# calls in the same usage window.
from shared.llm_clients import opus_llm_cached as opus_llm
from shared.observability import observe

from market_intelligence.briefing_polish import (
    BANNED_BRIEFING_TOKENS,
    SNAKE_CASE_IDENTIFIER_RE,
    _has_llm_access,
)


# Minimum average lexical overlap between capability descriptions and the
# recruiter's good_looks.prose. Below it, treat the LLM output as ungrounded
# and use the heuristic seed. Keeping the threshold at module scope makes the
# policy explicit and easy to adjust.
HALLUCINATION_OVERLAP_THRESHOLD: float = 0.30

# Minimum prose length to count toward heuristic confidence's "populated"
# signal. Mirrors :data:`market_intelligence.briefing_polish.MIN_PARAGRAPH_CHARS`
# so the substantive-prose threshold is consistent across both polish
# surfaces.
MIN_SUBSTANTIVE_CHARS: int = 30

# Denominator for the heuristic confidence formula. The seven counted
# fields (see :class:`HeuristicBriefPolishBackend` docstring) yield
# 0.0 to 1.0 in 1/7 increments — enough granularity for the recruiter
# to distinguish "minimal capture" from "full capture" in the Reference
# Slip.
HEURISTIC_CONFIDENCE_DENOMINATOR: int = 7

# Polish prompt token budget. Modest because the recruiter's intake
# captures are bounded prose (a few paragraphs each), not an arbitrarily
# large run dump like the briefing polish's deterministic_summary.
POLISH_MAX_TOKENS: int = 4000

# Overlap tokens are lowercased alpha runs of at least three characters, which
# filters tiny stop-word-like tokens before lexical overlap is calculated.
_OVERLAP_TOKEN_RE = re.compile(r"[a-z]{3,}")


# ---------------------------------------------------------------------------
# Telemetry — log every polish call with [intake] prefix
# ---------------------------------------------------------------------------


_FALLBACK_REASON_RE = re.compile(r"\bfallback reason=([A-Za-z_][A-Za-z0-9_]*)\b")


def _emit_stage(message: str) -> None:
    """Print a single-line stderr log with the ``[intake]`` prefix.

    Keep the log shape aligned with the post-run briefing logger while using
    ``[intake]`` for this intake-time stream. A ``fallback reason=...`` token
    is also recorded as the Langfuse ``cascade.fallback_reason`` attribute.
    """

    import sys

    print(f"[intake] {message}", file=sys.stderr, flush=True)

    match = _FALLBACK_REASON_RE.search(message)
    if match is not None:
        try:
            from shared.observability import update_current_observation

            update_current_observation(
                metadata={"cascade.fallback_reason": match.group(1)}
            )
        except Exception:  # noqa: BLE001 — Langfuse path is fail-soft
            pass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BriefPolishResult:
    """Polished V2 brief draft + provenance metadata.

    ``v2_draft`` is a schema-correct V2 brief dict (passes
    :func:`validate_v2_brief`). ``source`` is one of ``"llm"`` (LLM
    success path), ``"deterministic"`` (heuristic fallback path or no
    LLM access), or ``"empty"`` (no usable captures). ``confidence`` is
    programmatic — never LLM self-rating.
    """

    v2_draft: dict
    source: str = "deterministic"
    confidence: float = 0.0
    polished_at: str = field(default="")
    # Recruiter priors are advisory context alongside the polished draft, never
    # merged into ``v2_draft``. Merging would violate the brief-authored invariant
    # and the ``RECOGNIZED_V2_KEYS`` whitelist. ``None`` means no overlay was
    # hydrated; when ``recruiter_id`` is provided, the caller surfaces the active
    # signals separately. Exclude the overlay from ``to_meta_dict`` so it cannot
    # enter the persisted brief blob.
    recruiter_priors_overlay: dict | None = None

    def __post_init__(self) -> None:
        if not self.polished_at:
            self.polished_at = datetime.now(timezone.utc).isoformat()

    def to_meta_dict(self) -> dict:
        """Return metadata for ``state_json["v2_draft_polish_meta"]``.

        Keep ``v2_draft`` at ``state_json["v2_draft"]``; the review chapter
        consumes this metadata shape separately.
        """

        return {
            "source": self.source,
            "confidence": round(float(self.confidence), 2),
            "polished_at": self.polished_at,
        }


# ---------------------------------------------------------------------------
# Heuristic backend — Python port of frontend seedV2DraftFromChapters
# ---------------------------------------------------------------------------


class HeuristicBriefPolishBackend:
    """Deterministic V2 draft builder. Always passes ``validate_v2_brief``.

    Mirrors ``cloris/frontend/src/components/OnboardingFlow.svelte``'s
    :func:`seedV2DraftFromChapters` so the heuristic fallback produces
    the same structural shape the frontend already produces on chapter
    advance into review. The Path 3 promotion (linkedin_project_id
    into source_config.linkedin) matches the frontend behavior exactly,
    so a recruiter who never clicks Polish lands on the same v2_draft
    whether the seed came from the frontend or this backend's fallback.

    Confidence is the population fraction across seven scoring fields:

    1. ``role.title`` — any non-empty.
    2. ``role.framing`` — ``≥MIN_SUBSTANTIVE_CHARS`` (30) chars.
    3. ``good_looks.prose`` — ``≥MIN_SUBSTANTIVE_CHARS`` chars
       (load-bearing — capability areas come from here).
    4. ``lookalikes.exemplars_prose`` — ``≥MIN_SUBSTANTIVE_CHARS`` chars.
    5. ``lookalikes.non_fit_prose`` — ``≥MIN_SUBSTANTIVE_CHARS`` chars.
    6. ``where_to_look.target_modules`` — non-empty list of strings.
    7. ``where_to_look.linkedin_project_id`` — any non-empty (Path 3).

    Source is ``"deterministic"`` whenever ``role.title`` OR
    ``good_looks.prose`` carries content; ``"empty"`` when neither does
    (the Reference Slip surfaces this so the recruiter knows polish had
    nothing to work with).
    """

    def polish(
        self,
        *,
        chapter_captures: dict[str, Any],
        role_title: str | None = None,
    ) -> BriefPolishResult:
        role = _as_dict(chapter_captures.get("role"))
        good_looks = _as_dict(chapter_captures.get("good_looks"))
        lookalikes = _as_dict(chapter_captures.get("lookalikes"))
        where_to_look = _as_dict(chapter_captures.get("where_to_look"))

        # Mirror the frontend seeder's title fallback: prefer the
        # chapter capture, fall back to the session-level role_title
        # hint (set at session creation), default to empty string.
        title = _as_str(role.get("title")) or _as_str(role_title)

        # Description fallback matches the frontend literal so a
        # recruiter who never typed in good_looks lands on the same
        # placeholder text whether they were rendered server-side or
        # client-side.
        prose_value = good_looks.get("prose")
        description = (
            prose_value
            if isinstance(prose_value, str)
            else "What this person needs to be able to do."
        )

        # Sort target_modules so the brief disk-shape stays stable
        # across writes. Mirrors the frontend's toggleTargetModule
        # canonicalization at OnboardingFlow.svelte:516.
        target_modules = _as_str_list(where_to_look.get("target_modules"))
        if not target_modules:
            target_modules = ["linkedin"]
        target_modules = sorted(set(target_modules))

        v2_draft: dict[str, Any] = {
            "role_title": title,
            "capability_areas": [
                {
                    "name": "Capability area 1",
                    "description": description,
                }
            ],
            "depth_distinction": {
                "builder_definition": "",
                "user_definition": "",
                "edge_case_guidance": "",
            },
            "non_fit_patterns": [],
            "target_modules": target_modules,
        }

        # Path 3 — promote linkedin_project_id only when present so the
        # source_config block doesn't survive as an empty {} (which is
        # validate_v2_brief-clean but pointless).
        li_project_id = _as_str(where_to_look.get("linkedin_project_id"))
        li_project_name = _as_str(where_to_look.get("linkedin_project_name"))
        if li_project_id:
            linkedin: dict[str, Any] = {"project_id": li_project_id}
            if li_project_name:
                linkedin["project_name"] = li_project_name
            v2_draft["source_config"] = {"linkedin": linkedin}

        # Seed source_config.researcher from where_to_look captures. Explicit
        # floor overrides resolve at evaluation time; carry recruiter-authored
        # free-text inputs through unchanged.
        researcher_topics = _as_str_list(where_to_look.get("research_topics"))
        researcher_venues = _as_str_list(where_to_look.get("conference_allowlist"))
        researcher_discipline = _as_str(where_to_look.get("discipline"))
        if researcher_topics or researcher_venues or researcher_discipline:
            researcher_block: dict[str, Any] = {}
            if researcher_topics:
                researcher_block["research_topics"] = researcher_topics
            if researcher_venues:
                researcher_block["conference_allowlist"] = researcher_venues
            if researcher_discipline:
                researcher_block["discipline"] = researcher_discipline
            v2_draft.setdefault("source_config", {})
            v2_draft["source_config"]["researcher"] = researcher_block

        # Pass the recruiter-authored design_rubric through so heuristic output
        # preserves it. Empty or missing capture omits the key, which the drift
        # route treats as no preservation contract.
        design_rubric_capture = _as_dict(chapter_captures.get("design_rubric"))
        if design_rubric_capture:
            v2_draft["design_rubric"] = design_rubric_capture

        confidence = _heuristic_confidence(
            role=role,
            good_looks=good_looks,
            lookalikes=lookalikes,
            where_to_look=where_to_look,
            li_project_id=li_project_id,
        )

        # source=empty when the recruiter has nothing to polish from.
        # Without title or prose, the seed is pure placeholder; the
        # Reference Slip should say so honestly.
        if not title and not _as_str(good_looks.get("prose")):
            return BriefPolishResult(
                v2_draft=v2_draft,
                source="empty",
                confidence=0.0,
            )

        return BriefPolishResult(
            v2_draft=v2_draft,
            source="deterministic",
            confidence=confidence,
        )


def _heuristic_confidence(
    *,
    role: dict,
    good_looks: dict,
    lookalikes: dict,
    where_to_look: dict,
    li_project_id: str,
) -> float:
    """``populated_chapter_fields / 7``. See backend docstring for the seven."""

    populated = 0
    if _as_str(role.get("title")):
        populated += 1
    if len(_as_str(role.get("framing"))) >= MIN_SUBSTANTIVE_CHARS:
        populated += 1
    if len(_as_str(good_looks.get("prose"))) >= MIN_SUBSTANTIVE_CHARS:
        populated += 1
    if len(_as_str(lookalikes.get("exemplars_prose"))) >= MIN_SUBSTANTIVE_CHARS:
        populated += 1
    if len(_as_str(lookalikes.get("non_fit_prose"))) >= MIN_SUBSTANTIVE_CHARS:
        populated += 1
    if _as_str_list(where_to_look.get("target_modules")):
        populated += 1
    if li_project_id:
        populated += 1
    return round(populated / HEURISTIC_CONFIDENCE_DENOMINATOR, 2)


# ---------------------------------------------------------------------------
# LLM backend — Opus, with seven-route failure cascade to heuristic
# ---------------------------------------------------------------------------


class BriefPolishBackend:
    """Run LLM brief polish with deterministic fallback.

    :func:`polish` falls back to :class:`HeuristicBriefPolishBackend` when the
    LLM raises, returns an invalid schema, emits banned or snake_case text,
    drops or changes ``source_config.linkedin.project_id`` or ``role_title``,
    fails the lexical-overlap grounding check, or drifts a source-specific
    preservation contract. Banned and snake_case scans cover capability
    names/descriptions, ``depth_distinction.*``, and
    ``non_fit_patterns[*].label|why_not``.

    Each route emits ``_emit_stage`` with a ``reason=`` token and route detail;
    output is accepted only after schema, vocabulary, grounding, identity, and
    source-specific preservation checks pass.
    """

    def __init__(
        self, fallback: HeuristicBriefPolishBackend | None = None
    ) -> None:
        self.fallback = fallback or HeuristicBriefPolishBackend()

    @observe(name="market_intel.brief_polish")
    def polish(
        self,
        *,
        chapter_captures: dict[str, Any],
        role_title: str | None = None,
        session_id: int | None = None,
        recruiter_id: int | None = None,
    ) -> BriefPolishResult:
        """Polish a v2 draft and optionally attach recruiter priors separately.

        The draft is generated from recruiter-authored captures; priors never
        feed into the polish call. When ``recruiter_id`` is supplied, the
        result carries active domain signals as advisory context for the caller.
        The overlay is never merged into ``v2_draft`` or persisted in the brief.
        """

        result = self._polish_core(
            chapter_captures=chapter_captures,
            role_title=role_title,
            session_id=session_id,
        )
        if recruiter_id is not None:
            result.recruiter_priors_overlay = _hydrate_recruiter_priors_overlay(
                recruiter_id=recruiter_id,
                v2_draft=result.v2_draft,
            )
        return result

    def _polish_core(
        self,
        *,
        chapter_captures: dict[str, Any],
        role_title: str | None = None,
        session_id: int | None = None,
    ) -> BriefPolishResult:
        good_looks_dict = _as_dict(chapter_captures.get("good_looks"))
        lookalikes_dict = _as_dict(chapter_captures.get("lookalikes"))
        role_dict = _as_dict(chapter_captures.get("role"))
        wtl_dict = _as_dict(chapter_captures.get("where_to_look"))

        good_looks_chars = len(_as_str(good_looks_dict.get("prose")))
        exemplars_chars = len(_as_str(lookalikes_dict.get("exemplars_prose")))
        non_fit_chars = len(_as_str(lookalikes_dict.get("non_fit_prose")))
        has_role_title = bool(
            _as_str(role_dict.get("title")) or _as_str(role_title)
        )
        has_linkedin_project = bool(_as_str(wtl_dict.get("linkedin_project_id")))

        # Emit the start line so cascade rates can be correlated with input
        # richness in downstream analysis.
        _emit_stage(
            f"brief.polish:start session_id={session_id} "
            f"good_looks_chars={good_looks_chars} "
            f"exemplars_chars={exemplars_chars} "
            f"non_fit_chars={non_fit_chars} "
            f"has_role_title={str(has_role_title).lower()} "
            f"has_linkedin_project={str(has_linkedin_project).lower()}"
        )

        t0 = time.monotonic()

        # The heuristic seed serves two purposes: (1) it's what we
        # fall back to on cascade, and (2) it gives the LLM a structural
        # starting point in the user prompt — including the preservation
        # contracts (role_title + source_config) the LLM is told to
        # respect verbatim.
        seeded = self.fallback.polish(
            chapter_captures=chapter_captures, role_title=role_title
        )

        # Empty captures: no point invoking the LLM with nothing to
        # polish from. Heuristic already marked source=empty.
        if seeded.source == "empty":
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:done source=empty confidence=0.00 "
                f"elapsed_ms={elapsed_ms}"
            )
            return seeded

        if not _has_llm_access():
            elapsed_ms_pre = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=no_llm_access "
                f"elapsed_ms={elapsed_ms_pre}"
            )
            return self._cascade_done(seeded, t0)

        # Route 1: opus_llm raise.
        try:
            raw = opus_llm(
                build_brief_polish_system_prompt(),
                build_brief_polish_user_prompt(
                    chapter_captures=chapter_captures,
                    seeded_v2_draft=seeded.v2_draft,
                    role_title=role_title,
                ),
                expect_json=True,
                max_tokens=POLISH_MAX_TOKENS,
                usage_context={
                    "stage": "intake_brief_polish",
                    "session_id": session_id,
                },
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=llm_raise "
                f"exc={exc.__class__.__name__} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Route 2: schema validity. We split the "not a dict" case from
        # the validate_v2_brief case so logs surface which kind of
        # malformation fired.
        if not isinstance(raw, dict):
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=schema_invalid "
                f"detail=not_dict elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)
        try:
            validate_v2_brief(raw)
        except BriefSchemaError as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            detail_keys = list(exc.missing_keys) + list(exc.invalid_keys)
            detail = ",".join(detail_keys) if detail_keys else "unknown"
            _emit_stage(
                f"brief.polish:fallback reason=schema_invalid "
                f"detail={detail} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Route 3: banned tokens (jargon recruiters shouldn't see).
        banned_hit = _banned_token_in_v2_draft(raw)
        if banned_hit is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            path_label, token = banned_hit
            _emit_stage(
                f"brief.polish:fallback reason=banned_token "
                f"token={token!r} path={path_label} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Route 4: snake_case identifiers (engineer-vocab leak).
        snake_hit = _snake_case_in_v2_draft(raw)
        if snake_hit is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            path_label, token = snake_hit
            _emit_stage(
                f"brief.polish:fallback reason=snake_case_token "
                f"token={token!r} path={path_label} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Route 5: Path 3 preservation. Hard contract — if the seed
        # carried a linkedin project_id, the LLM is not authorized to
        # drop or modify it.
        path3_drift = _path3_drift(seeded.v2_draft, raw)
        if path3_drift is not None:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=path3_drift "
                f"detail={path3_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Route 6: log overlap_avg and per-area values on every call for
        # calibration. Enforce the threshold only when good_looks.prose has
        # content to ground against.
        good_looks_prose = _as_str(good_looks_dict.get("prose"))
        overlap_avg, per_area = _capability_area_overlap(
            v2_draft=raw, good_looks_prose=good_looks_prose
        )
        per_area_str = "[" + ",".join(f"{x:.2f}" for x in per_area) + "]"
        _emit_stage(
            f"brief.polish:hallucination_check "
            f"overlap_avg={overlap_avg:.2f} "
            f"overlap_per_area={per_area_str} "
            f"threshold={HALLUCINATION_OVERLAP_THRESHOLD:.2f} "
            f"n_areas={len(per_area)}"
        )
        if good_looks_prose and overlap_avg < HALLUCINATION_OVERLAP_THRESHOLD:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=hallucination "
                f"overlap_avg={overlap_avg:.2f} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Route 7: role_title preservation. Same posture as Path 3 —
        # the seed's role_title is the recruiter's word, not the LLM's
        # to rewrite.
        role_drift = _role_title_drift(
            seeded_role_title=seeded.v2_draft.get("role_title", ""),
            polished_role_title=raw.get("role_title", ""),
        )
        if role_drift:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=role_title_drift "
                f"detail={role_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Design-rubric preservation. This named route keeps the recruiter-authored
        # rubric byte-identical; it is never an LLM editorial choice.
        rubric_drift = _design_rubric_drift(seeded=seeded.v2_draft, polished=raw)
        if rubric_drift:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=design_rubric_drift "
                f"detail={rubric_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Target-project preservation. Compare the recruiter-named project set
        # through the named route; sibling preservation routes keep independent
        # telemetry names.
        target_projects_drift = _target_projects_drift(
            seeded=seeded.v2_draft, polished=raw
        )
        if target_projects_drift:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=target_projects_drift "
                f"detail={target_projects_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Target-stack preservation uses set equality; its named route remains
        # independent of target-project preservation.
        target_stacks_drift = _target_stacks_drift(
            seeded=seeded.v2_draft, polished=raw
        )
        if target_stacks_drift:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=target_stacks_drift "
                f"detail={target_stacks_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Maintainership-level preservation: after a recruiter lowers the floor,
        # polish must not raise it. Compare normalized values.
        maintainership_level_drift = _maintainership_level_drift(
            seeded=seeded.v2_draft, polished=raw
        )
        if maintainership_level_drift:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=maintainership_level_drift "
                f"detail={maintainership_level_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Confidentiality is recruiter policy. A seeded declaration must not be
        # silently downgraded; the drift route forces explicit re-authoring.
        confidentiality_class_drift = _confidentiality_class_drift(
            seeded=seeded.v2_draft, polished=raw
        )
        if confidentiality_class_drift:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=confidentiality_class_drift "
                f"detail={confidentiality_class_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Preserve recruiter-authored source_config.researcher inputs. List fields
        # use set equality; discipline and floor fields use scalar equality. The
        # LLM may not rewrite research topics, conference allowlist, discipline,
        # h_index_floor, papers_in_window_floor, or papers_in_window_months.
        research_topics_drift = _research_topics_drift(
            seeded=seeded.v2_draft, polished=raw
        )
        if research_topics_drift:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=research_topics_drift "
                f"detail={research_topics_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Preserve recruiter-authored Exa fields as whole entries. The successful
        # path returns raw output wholesale, so a dropped do_not_save entry would
        # remove an exclusion and permit outreach to a named person.
        exa_fields_drift = _exa_fields_drift(seeded=seeded.v2_draft, polished=raw)
        if exa_fields_drift:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=exa_fields_drift "
                f"detail={exa_fields_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # Success: raw output passed schema, vocabulary, grounding, identity, and
        # source-specific preservation checks. LLM confidence is fixed at 1.0.
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        result = BriefPolishResult(
            v2_draft=raw,
            source="llm",
            confidence=1.0,
        )
        _emit_stage(
            f"brief.polish:done source={result.source} "
            f"confidence={result.confidence:.2f} elapsed_ms={elapsed_ms}"
        )
        return result

    def _cascade_done(
        self, seeded: BriefPolishResult, t0: float
    ) -> BriefPolishResult:
        """Emit the trailing ``done`` log for a cascade-fallback path."""

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _emit_stage(
            f"brief.polish:done source={seeded.source} "
            f"confidence={seeded.confidence:.2f} elapsed_ms={elapsed_ms}"
        )
        return seeded


# ---------------------------------------------------------------------------
# Cascade route helpers
# ---------------------------------------------------------------------------


def _scannable_text_fields(v2_draft: dict) -> list[tuple[str, str]]:
    """Yield ``(path_label, value)`` for every recruiter-facing string field.

    Scans capability areas (name + description), depth_distinction
    (three fields), non_fit_patterns (label + why_not). Excludes
    ``source_config`` and ``role_title`` because those have dedicated
    cascade routes (5 and 7); double-scanning them would surface the
    wrong route in logs.
    """

    out: list[tuple[str, str]] = []
    for idx, ca in enumerate(v2_draft.get("capability_areas") or []):
        if not isinstance(ca, dict):
            continue
        for key in ("name", "description"):
            value = ca.get(key)
            if isinstance(value, str) and value:
                out.append((f"capability_areas[{idx}].{key}", value))
    dd = v2_draft.get("depth_distinction")
    if isinstance(dd, dict):
        for key in ("builder_definition", "user_definition", "edge_case_guidance"):
            value = dd.get(key)
            if isinstance(value, str) and value:
                out.append((f"depth_distinction.{key}", value))
    for idx, nfp in enumerate(v2_draft.get("non_fit_patterns") or []):
        if not isinstance(nfp, dict):
            continue
        for key in ("label", "why_not"):
            value = nfp.get(key)
            if isinstance(value, str) and value:
                out.append((f"non_fit_patterns[{idx}].{key}", value))
    return out


def _banned_token_in_v2_draft(v2_draft: dict) -> tuple[str, str] | None:
    """First ``(path_label, token)`` hit for a banned token, else ``None``."""

    for path_label, value in _scannable_text_fields(v2_draft):
        lowered = value.lower()
        for token in BANNED_BRIEFING_TOKENS:
            if token in lowered:
                return path_label, token
    return None


def _snake_case_in_v2_draft(v2_draft: dict) -> tuple[str, str] | None:
    """First ``(path_label, token)`` hit for a snake_case identifier, else ``None``."""

    for path_label, value in _scannable_text_fields(v2_draft):
        match = SNAKE_CASE_IDENTIFIER_RE.search(value)
        if match is not None:
            return path_label, match.group(0)
    return None


def _path3_drift(seeded: dict, polished: dict) -> str | None:
    """Return drift descriptor when ``source_config.linkedin.project_id`` drifts.

    Contract: if the seeded draft (which mirrors the recruiter's intake
    state) carried a linkedin project_id, the polished output MUST carry
    the same project_id. Anything else (dropped, changed) cascades to
    Route 5. Returns ``None`` for no drift; a short string for diagnostic.
    """

    seed_li = _as_dict(_as_dict(seeded.get("source_config")).get("linkedin"))
    seed_pid = _as_str(seed_li.get("project_id"))
    if not seed_pid:
        return None
    polished_li = _as_dict(_as_dict(polished.get("source_config")).get("linkedin"))
    polished_pid = _as_str(polished_li.get("project_id"))
    if not polished_pid:
        return f"dropped seed_pid={seed_pid!r}"
    if polished_pid != seed_pid:
        return f"changed seed_pid={seed_pid!r} polished_pid={polished_pid!r}"
    return None


def _role_title_drift(
    *, seeded_role_title: Any, polished_role_title: Any
) -> str | None:
    """Return drift descriptor when seeded ``role_title`` was non-empty but polished differs."""

    seed = _as_str(seeded_role_title)
    polish = _as_str(polished_role_title)
    if not seed:
        return None
    if not polish:
        return "dropped"
    if polish != seed:
        return f"changed seed={seed!r} polished={polish!r}"
    return None


def _design_rubric_drift(seeded: dict, polished: dict) -> str | None:
    """Return a drift descriptor when ``design_rubric`` changes.

    A non-empty recruiter-authored rubric must be carried byte-for-byte into
    polished output. The LLM cannot alter principles, anchors, weights,
    exemplars, or hard-reject patterns.

    Return ``None`` when no rubric is seeded, the seeded rubric is empty, or
    the polished rubric is equal. Otherwise return a short descriptor so the
    cascade falls back to the heuristic.
    """

    seed = seeded.get("design_rubric")
    if not isinstance(seed, dict) or not seed:
        return None
    polish = polished.get("design_rubric")
    if not isinstance(polish, dict) or not polish:
        return "dropped"
    if polish != seed:
        return _describe_rubric_drift(seed=seed, polish=polish)
    return None


def _target_projects_drift(seeded: dict, polished: dict) -> str | None:
    """Return a drift descriptor when ``target_projects`` changes.

    A non-empty recruiter-named project list must return the same set; order and
    duplicates are normalized away. Empty or absent seed values have no
    preservation contract. ``target_stacks`` and ``maintainership_level`` are
    checked independently by their own preservation routes.
    """

    seed_projects = _normalize_target_projects(seeded.get("target_projects"))
    if not seed_projects:
        return None
    polish_projects = _normalize_target_projects(polished.get("target_projects"))
    if not polish_projects:
        return f"dropped seed={sorted(seed_projects)!r}"
    if seed_projects != polish_projects:
        added = polish_projects - seed_projects
        dropped = seed_projects - polish_projects
        parts: list[str] = []
        if dropped:
            parts.append(f"dropped={sorted(dropped)!r}")
        if added:
            parts.append(f"added={sorted(added)!r}")
        return ",".join(parts) if parts else "set_inequality"
    return None


def _normalize_target_projects(value: Any) -> set[str]:
    """Normalize a raw ``target_projects`` list to a set.

    Drop non-list, non-string, and blank entries; trim whitespace and lowercase
    owner/repo because GitHub resolves them case-insensitively.
    """

    if not isinstance(value, list):
        return set()
    out: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        if cleaned:
            out.add(cleaned)
    return out


def _confidentiality_class_drift(
    *, seeded: dict, polished: dict
) -> str | None:
    """Return a drift descriptor when ``confidentiality_class`` changes.

    A non-empty, non-``"open"`` seeded declaration is recruiter policy and must
    be preserved case-insensitively; polish cannot silently downgrade it. An
    absent, empty, or ``"open"`` value follows the downstream default and is
    not guarded. Return ``None`` for no drift, otherwise a short telemetry
    descriptor.
    """

    seed_value = seeded.get("confidentiality_class")
    if not isinstance(seed_value, str):
        return None
    seed_normalized = seed_value.strip().lower()
    # Empty and ``"open"`` use default behavior and are not guarded; a
    # non-empty alternative remains a recruiter declaration and is checked.
    if seed_normalized in ("", "open"):
        return None

    polish_value = polished.get("confidentiality_class")
    if not isinstance(polish_value, str):
        return f"dropped seed={seed_normalized!r}"
    polish_normalized = polish_value.strip().lower()
    if polish_normalized != seed_normalized:
        return f"changed seed={seed_normalized!r} polish={polish_normalized!r}"
    return None


def _research_topics_drift(*, seeded: dict, polished: dict) -> str | None:
    """Return a drift descriptor when ``source_config.researcher`` changes.

    Preserve recruiter-authored ``research_topics`` and ``conference_allowlist``
    by set equality, and preserve ``discipline``, ``h_index_floor``,
    ``papers_in_window_floor``, and ``papers_in_window_months`` by scalar
    equality. Return ``None`` when no researcher block is seeded or all fields
    match.
    """

    seed_researcher = _as_dict(_as_dict(seeded.get("source_config")).get("researcher"))
    if not seed_researcher:
        return None

    polish_researcher = _as_dict(
        _as_dict(polished.get("source_config")).get("researcher")
    )

    drift_parts: list[str] = []

    # Set-equality preservation for list fields.
    for list_key in ("research_topics", "conference_allowlist"):
        seed_value = seed_researcher.get(list_key)
        if not isinstance(seed_value, list) or not seed_value:
            continue
        seed_set = {str(v).strip() for v in seed_value if isinstance(v, str)}
        polish_value = polish_researcher.get(list_key)
        if not isinstance(polish_value, list):
            drift_parts.append(f"{list_key}=dropped")
            continue
        polish_set = {str(v).strip() for v in polish_value if isinstance(v, str)}
        if seed_set != polish_set:
            dropped = sorted(seed_set - polish_set)
            added = sorted(polish_set - seed_set)
            detail: list[str] = []
            if dropped:
                detail.append(f"-{dropped!r}")
            if added:
                detail.append(f"+{added!r}")
            drift_parts.append(f"{list_key}={'/'.join(detail)}")

    # Scalar equality preservation for discipline + floor overrides.
    for scalar_key in (
        "discipline",
        "h_index_floor",
        "papers_in_window_floor",
        "papers_in_window_months",
    ):
        if scalar_key not in seed_researcher:
            continue
        seed_value = seed_researcher[scalar_key]
        if scalar_key not in polish_researcher:
            drift_parts.append(f"{scalar_key}=dropped")
            continue
        polish_value = polish_researcher[scalar_key]
        if seed_value != polish_value:
            drift_parts.append(
                f"{scalar_key} seed={seed_value!r} polish={polish_value!r}"
            )

    if drift_parts:
        return "; ".join(drift_parts)
    return None


def _maintainership_level_drift(*, seeded: dict, polished: dict) -> str | None:
    """Return a drift descriptor when ``maintainership_level`` changes.

    A recognized recruiter-authored maintainership floor is policy. Polish must
    preserve it case-insensitively so a lower approved floor is not raised.
    Unseeded or unrecognized values have no preservation contract.
    """

    seed_value = seeded.get("maintainership_level")
    if not isinstance(seed_value, str):
        return None
    seed_normalized = seed_value.strip().lower()
    if seed_normalized not in {"contributor", "maintainer", "project_lead"}:
        return None

    polish_value = polished.get("maintainership_level")
    if not isinstance(polish_value, str):
        return f"dropped seed={seed_normalized!r}"
    polish_normalized = polish_value.strip().lower()
    if polish_normalized != seed_normalized:
        return f"changed seed={seed_normalized!r} polish={polish_normalized!r}"
    return None


def _target_stacks_drift(*, seeded: dict, polished: dict) -> str | None:
    """Return a drift descriptor when ``target_stacks`` changes.

    A non-empty recruiter-authored stack list is compared as a normalized set;
    order, duplicates, whitespace, and case do not carry meaning. Empty or
    absent seed values have no preservation contract.
    """

    seed_set = _normalize_target_stacks(seeded.get("target_stacks"))
    if not seed_set:
        return None
    polish_set = _normalize_target_stacks(polished.get("target_stacks"))
    if not polish_set:
        return f"dropped seed={sorted(seed_set)!r}"
    if seed_set != polish_set:
        added = polish_set - seed_set
        dropped = seed_set - polish_set
        parts: list[str] = []
        if dropped:
            parts.append(f"dropped={sorted(dropped)!r}")
        if added:
            parts.append(f"added={sorted(added)!r}")
        return ",".join(parts) if parts else "set_inequality"
    return None


def _normalize_target_stacks(value: Any) -> set[str]:
    """Coerce a raw ``target_stacks`` value to a normalized set.

    Empty / non-list / non-string entries drop out. Trims whitespace
    and lowercases — stack tags like ``"Rust"`` and ``"rust"`` are
    not contract-distinct (the strategy seeder lowercases them too).
    """

    if not isinstance(value, list):
        return set()
    out: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        if cleaned:
            out.add(cleaned)
    return out


def _exa_fields_drift(*, seeded: dict, polished: dict) -> str | None:
    """Return a drift descriptor when recruiter-authored Exa fields change.

    Preserve ``exa_target_firms``, ``exa_query_lanes``, ``exa_title_terms``,
    ``exa_archetype_queries``, and ``do_not_save`` by whole-entry set equality.
    Firm domains and exclude_terms affect search behavior, so comparing names
    alone is insufficient. ``do_not_save.reason`` remains editable prose.

    Presence is part of the contract: a seeded key, including an explicit empty
    list, must remain equal, while an absent key must not be invented.
    """

    parts: list[str] = []
    for field_name, canonicalize in (
        ("exa_target_firms", _canonical_exa_firms),
        ("exa_query_lanes", _canonical_exa_terms),
        ("exa_title_terms", _canonical_exa_terms),
        ("exa_archetype_queries", _canonical_exa_archetypes),
        ("do_not_save", _canonical_exa_people),
    ):
        seed_present = field_name in seeded and seeded.get(field_name) is not None
        seed_set = canonicalize(seeded.get(field_name))
        polish_set = canonicalize(polished.get(field_name))

        if not seed_present:
            if polish_set:
                parts.append(f"{field_name}:invented={_exa_labels(polish_set)!r}")
            continue

        if seed_set == polish_set:
            continue
        dropped = seed_set - polish_set
        added = polish_set - seed_set
        detail: list[str] = []
        if dropped:
            detail.append(f"dropped={_exa_labels(dropped)!r}")
        if added:
            detail.append(f"added={_exa_labels(added)!r}")
        parts.append(f"{field_name}:" + ",".join(detail))
    return " | ".join(parts) if parts else None


def _exa_labels(entries: set) -> list[str]:
    """Render canonical Exa entries as sorted, log-safe strings.

    Tuple entries render as ``name(field=value,...)`` so a changed
    domain reads as a change rather than as the same name appearing on
    both sides of the diff.
    """

    out: list[str] = []
    for entry in entries:
        if not isinstance(entry, tuple):
            out.append(str(entry))
            continue
        head, *rest = entry
        detail_parts: list[str] = []
        for part in rest:
            if isinstance(part, tuple):
                if part:
                    detail_parts.append("+".join(part))
            elif part:
                detail_parts.append(str(part))
        detail = ",".join(detail_parts)
        out.append(f"{head}({detail})" if detail else str(head))
    return sorted(out)


def _canonical_exa_terms(value: Any) -> set[str]:
    """Coerce a raw Exa string list to a canonical set.

    Non-list / non-string / blank entries drop out; entries are
    trimmed and lowercased so ``"Senior Associate"`` and ``"senior
    associate"`` are not contract-distinct.
    """

    if not isinstance(value, list):
        return set()
    out: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        if cleaned:
            out.add(cleaned)
    return out


def _canonical_str_tuple(value: Any) -> tuple[str, ...]:
    """Sorted, lowercased tuple of the strings in a raw list value."""

    if not isinstance(value, list):
        return ()
    return tuple(
        sorted(
            {
                item.strip().lower()
                for item in value
                if isinstance(item, str) and item.strip()
            }
        )
    )


def _canonical_exa_firms(value: Any) -> set[tuple]:
    """Canonicalize ``exa_target_firms`` entries whole.

    ``(name, domain, aliases, exclude_terms)`` — every field that
    changes what the compiled queries do. Only the name is required;
    the rest normalize to empty.
    """

    if not isinstance(value, list):
        return set()
    out: set[tuple] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        domain = item.get("domain")
        out.add(
            (
                name.strip().lower(),
                domain.strip().lower() if isinstance(domain, str) else "",
                _canonical_str_tuple(item.get("aliases")),
                _canonical_str_tuple(item.get("exclude_terms")),
            )
        )
    return out


def _canonical_exa_archetypes(value: Any) -> set[tuple]:
    """Canonicalize ``exa_archetype_queries`` entries whole.

    ``(id, text, axis, num_results, include_domains)`` — everything a
    recruiter authored about the query. The ``text`` is IN the key, not
    merely the id: an archetype query is its prose. A model that keeps
    ``a01-mbb-to-vc`` and rewrites its sentence into something more
    agreeable has changed the search while leaving the id-set identical,
    and id-only comparison would call that no drift — the same failure
    ``exclude_terms`` taught on the firm entries.

    PRESENCE is part of the key, not just value. An omitted
    ``num_results`` canonicalizes to ``None`` rather than to the
    compiler's default, so a polish step that writes ``25`` where the
    recruiter wrote nothing IS drift. The two are not the same
    statement: an omitted field follows the module's default wherever
    that default goes next, and a written one pins the query to 25
    forever. Same for ``include_domains``, where the difference is
    larger — an authored ``["linkedin.com"]`` is a scope the recruiter
    chose, and a default one is a scope they never saw.

    ``axis`` is in the key too. It groups the slate rather than steering
    the search, but it is still recruiter-authored text, and the polish
    step is not the author of it: a model free to relabel ``consulting``
    as ``Consulting → VC`` is a model quietly regrouping the report.
    """

    if not isinstance(value, list):
        return set()
    out: set[tuple] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        entry_id = item.get("id")
        text = item.get("text")
        if not isinstance(entry_id, str) or not entry_id.strip():
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        num = item.get("num_results")
        if isinstance(num, bool) or not isinstance(num, int):
            num = None
        domains = item.get("include_domains")
        domains = _canonical_str_tuple(domains) if domains is not None else None
        axis = item.get("axis")
        axis = " ".join(axis.split()).strip().lower() if isinstance(axis, str) else ""
        out.add(
            (
                entry_id.strip().lower(),
                " ".join(text.split()).strip().lower(),
                axis,
                num,
                domains,
            )
        )
    return out


def _canonical_exa_people(value: Any) -> set[tuple]:
    """Canonicalize ``do_not_save`` entries as ``(name, handle)``.

    The LinkedIn URL normalizes through
    :func:`shared.identity_resolution.normalize_public_linkedin_handle`
    — the same handle the identity service matches on, so a rewritten
    but equivalent URL is not drift. ``reason`` is deliberately
    excluded: it is prose, and the polish step's job is prose.
    """

    from shared.identity_resolution import normalize_public_linkedin_handle

    if not isinstance(value, list):
        return set()
    out: set[tuple] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        url = item.get("linkedin_url")
        handle = (
            normalize_public_linkedin_handle(url) if isinstance(url, str) else ""
        )
        out.add((name.strip().lower(), handle))
    return out


def _describe_rubric_drift(*, seed: dict, polish: dict) -> str:
    """Produce a short, recruiter-readable drift descriptor.

    The descriptor is emitted in fallback telemetry so downstream analysis can
    distinguish principle rewrites, weight changes, and exemplar additions.
    Trim it to keep the log line bounded.
    """

    reasons: list[str] = []
    seed_principles = seed.get("principles") or []
    polish_principles = polish.get("principles") or []
    if isinstance(seed_principles, list) and isinstance(polish_principles, list):
        if len(seed_principles) != len(polish_principles):
            reasons.append(
                f"principle_count_changed seed={len(seed_principles)} "
                f"polished={len(polish_principles)}"
            )
        else:
            for idx, (seed_p, polish_p) in enumerate(
                zip(seed_principles, polish_principles)
            ):
                if seed_p != polish_p:
                    reasons.append(f"principle[{idx}]_mutated")
                    break  # one is enough for the log line
    if seed.get("discipline_weight_overrides") != polish.get(
        "discipline_weight_overrides"
    ):
        reasons.append("discipline_weight_overrides_mutated")
    if seed.get("calibration_exemplars") != polish.get("calibration_exemplars"):
        reasons.append("calibration_exemplars_mutated")
    if seed.get("hard_reject_patterns") != polish.get("hard_reject_patterns"):
        reasons.append("hard_reject_patterns_mutated")
    if not reasons:
        reasons.append("rubric_dict_inequality")
    return ",".join(reasons)


def _capability_area_overlap(
    *, v2_draft: dict, good_looks_prose: str
) -> tuple[float, list[float]]:
    """Compute lexical overlap of each capability description with prose.

    Return ``(avg_overlap, per_area_overlap_list)`` using Jaccard overlap over
    lowercased alpha tokens of at least three characters. Empty inputs use 1.0
    for that area because no grounding comparison is possible. The per-area
    values are logged for downstream calibration.
    """

    prose_tokens = set(_OVERLAP_TOKEN_RE.findall(good_looks_prose.lower()))
    capability_areas = v2_draft.get("capability_areas") or []

    if not prose_tokens:
        # No prose to ground against. Caller already gates this case
        # (Route 6 only enforces when good_looks_prose is non-empty);
        # this branch keeps the function total for the always-on
        # telemetry log line.
        per = [1.0 for _ in capability_areas]
        avg = 1.0 if not per else sum(per) / len(per)
        return avg, per

    per_area: list[float] = []
    for ca in capability_areas:
        if not isinstance(ca, dict):
            per_area.append(0.0)
            continue
        desc = _as_str(ca.get("description"))
        desc_tokens = set(_OVERLAP_TOKEN_RE.findall(desc.lower()))
        if not desc_tokens:
            per_area.append(1.0)
            continue
        intersection = desc_tokens & prose_tokens
        union = desc_tokens | prose_tokens
        per_area.append(len(intersection) / len(union) if union else 0.0)
    avg = sum(per_area) / len(per_area) if per_area else 1.0
    return avg, per_area


# Recruiter priors overlay — separate from the polished brief.


def _recruiter_signal_domain_from_v2_draft(v2_draft: dict) -> str | None:
    """Resolve the taste-signal domain (subagent) from a v2 draft, or None.

    A taste signal's ``domain`` is the subagent it calibrates
    (``"designer"`` / ``"linkedin"`` / ...). A draft targeting exactly one
    module is unambiguously attributable; multiple/zero modules → no
    single domain, so priors can't be filtered cleanly and we return
    ``None`` (no overlay) rather than mixing domains.
    """

    modules = _as_str_list(v2_draft.get("target_modules")) if isinstance(v2_draft, dict) else []
    if len(modules) == 1:
        return modules[0]
    return None


def _hydrate_recruiter_priors_overlay(
    *, recruiter_id: int, v2_draft: dict
) -> dict | None:
    """Build the advisory recruiter-priors overlay for a polish result.

    Reads the recruiter's ACTIVE (non-superseded) taste signals for the
    brief's resolved domain and returns them in a structured envelope —
    the caller surfaces/audits this alongside the brief; it is NEVER
    merged into ``v2_draft``. Returns ``None`` when there's no single
    domain or no active signals (nothing advisory to attach). Fail-soft:
    any store error yields ``None`` rather than failing the polish.
    """

    domain = _recruiter_signal_domain_from_v2_draft(v2_draft)
    if domain is None:
        return None
    try:
        from shared.output_paths import resolve_recruiter_db_path
        from shared.runtime_state.recruiter_store import RecruiterStore

        store = RecruiterStore(resolve_recruiter_db_path())
        signals = store.active_taste_signals(recruiter_id, domain=domain)
    except Exception:  # noqa: BLE001 — advisory overlay must never break polish
        return None
    if not signals:
        return None
    return {
        "recruiter_id": recruiter_id,
        "domain": domain,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Type-coercion helpers — defensive against state_json drift
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    # Defensive on numeric/bool: stringify so e.g. a project_id stored
    # as int still resolves to a non-empty string. Mirrors the
    # backward-compat coercion in
    # :func:`shared.brief_v2_schema.linkedin_project_id_from_brief`.
    return str(value)


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and v]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_brief_polish_system_prompt() -> str:
    """Build the system prompt for the brief polish step.

    The source of truth for voice and copy rules is
    ``docs/cloris-ia-doctrine-cursor.md``. The prompt encodes the recruiter-
    readable register, V2 schema, role/source preservation, grounding check,
    and honesty about gaps.
    """

    return """You are Cloris's editorial voice. You read raw intake captures from a recruiter and reshape them into a polished, schema-correct V2 brief draft.

VOICE RULES (from current IA doctrine):
- Recruiter-readable register. No engineer vocabulary, no snake_case identifiers, no internal jargon. The recruiter has never read the engine source code; they should never see its words.
- Calm, plain, editorial. Not shouty, not hypey, not "AI-native." This is a brief, not a dashboard.
- Operational copy and voice copy never overlap. Brief content is operational — clean, declarative, recruiter-readable. Don't write character voice into capability descriptions.

SCHEMA — return JSON ONLY with this exact shape:
{
  "role_title": "<string from input role.title — preserve exactly>",
  "capability_areas": [
    {"name": "<short editorial name, no snake_case>", "description": "<1-3 sentences grounded in the recruiter's good_looks.prose>"}
  ],
  "depth_distinction": {
    "builder_definition": "<what 'building it' looks like — 1-3 sentences, or empty string if the captures don't support a confident answer>",
    "user_definition": "<what 'using it' looks like — 1-3 sentences, or empty string>",
    "edge_case_guidance": "<edge cases and borderline calls — 1-3 sentences, or empty string>"
  },
  "non_fit_patterns": [
    {"label": "<short editorial label>", "why_not": "<1-2 sentences grounded in lookalikes.non_fit_prose>"}
  ],
  "target_modules": ["linkedin", ...],
  "source_config": {"linkedin": {"project_id": "<from input — preserve exactly>", "project_name": "<from input — preserve exactly>"}, "researcher": {"research_topics": ["<from input — preserve set>"], "conference_allowlist": ["<from input — preserve set>"], "discipline": "<from input — preserve verbatim if present>"}},
  "design_rubric": "<from input — preserve byte-for-byte if present; omit the key entirely if not present>",
  "target_projects": ["<owner/repo from input — preserve set; order doesn't matter>", ...],
  "target_stacks": ["<from input — preserve set>", ...],
  "maintainership_level": "<from input — preserve verbatim if present>",
  "exa_target_firms": [{"name": "<firm name>", "domain": "<firm website domain>", "aliases": ["<other name for the firm>"], "exclude_terms": ["<namesake to exclude>"]}] — from input, preserve every entry field-for-field; omit the key entirely if not present,
  "exa_query_lanes": ["<from input — preserve set; an explicit empty list means every lane and must stay empty>", ...],
  "exa_title_terms": ["<from input — preserve set>", ...],
  "exa_archetype_queries": [{"id": "<slug>", "axis": "<grouping label>", "text": "<the natural-language description of the person being searched for>", "num_results": 25, "include_domains": ["linkedin.com"]}] — from input, preserve every entry field-for-field including the full `text`; omit the key entirely if not present,
  "do_not_save": [{"name": "<person's name>", "linkedin_url": "<their profile URL>", "reason": "<why they're ruled out>"}] — from input, preserve every entry; only `reason` may be reworded
}

PRESERVATION RULES (HARD CONTRACTS — output is rejected and the recruiter's scaffolded draft is shown instead if you violate these):
- If the input contains `seeded_v2_draft.role_title` (non-empty), the output `role_title` MUST equal it character-for-character. Do not "improve" the title.
- If the input contains `seeded_v2_draft.source_config.linkedin.project_id`, the output `source_config.linkedin.project_id` MUST equal it. Do not invent, drop, or modify it. The same applies to `project_name` if present.
- If the input contains `seeded_v2_draft.design_rubric` (non-empty), the output `design_rubric` MUST equal it byte-for-byte. Do not modify principles, anchors, weights, calibration_exemplars, hard_reject_patterns, or discipline_weight_overrides. The rubric encodes the recruiter's taste; the model is not authorized to "improve" it.
- If the input contains `seeded_v2_draft.target_projects` (non-empty), the output `target_projects` MUST contain the same set of "owner/repo" entries (case-insensitive; ordering and duplicates don't matter, but no add or drop). The recruiter named those projects; "kubernetes maintainer" is not the same query as "container orchestration maintainer".
- If the input contains `seeded_v2_draft.source_config.researcher` (non-empty), the output MUST preserve `research_topics` (set-equality), `conference_allowlist` (set-equality), `discipline` (verbatim — one of `nlp | ml_general | vision | rl | systems | theory | biomedical | other`), and any explicit floor overrides (`h_index_floor`, `papers_in_window_floor`, `papers_in_window_months`) byte-for-byte. The discipline is the load-bearing field for layered floor resolution; the model is not authorized to "improve" recruiter-authoritative inputs.
- If the input contains `seeded_v2_draft.exa_archetype_queries`, the output MUST carry every entry with its `id`, its `text` WORD FOR WORD, its `num_results`, and its `include_domains`. Those sentences are the search: an archetype query is prose that Exa matches semantically, so a "tightened" sentence is a different query returning different people. Do not reword, shorten, merge, or add one. If the input does NOT carry the key, do not invent it.
- If the input contains `seeded_v2_draft.exa_target_firms`, `exa_query_lanes`, or `exa_title_terms`, the output MUST carry the same entries — for a firm that means its `name`, `domain`, `aliases`, AND `exclude_terms`, not the name alone. The exclude terms are what keep a search for "Accel" from returning Accel-KKR; dropping them silently widens the search. An empty `exa_query_lanes` means "every lane" and must stay empty. If the input does NOT carry these keys, do not invent them.
- If the input contains `seeded_v2_draft.do_not_save`, the output MUST carry every entry with its `name` and `linkedin_url` intact; only `reason` may be reworded. That list is who the recruiter has already ruled out; dropping a name is how someone gets contacted who should not have been. If the input does NOT carry the key, do not invent it.
- The output `target_modules` SHOULD equal `seeded_v2_draft.target_modules` unless the recruiter explicitly mentioned other surfaces in `chapter_captures.where_to_look.anything_else`.

HALLUCINATION GUARD:
- Reformat / reorganize / tighten the recruiter's words into clean capability areas and depth definitions. Do NOT invent capability areas the recruiter didn't write about.
- Each `capability_areas[*].description` MUST share substantial vocabulary with `chapter_captures.good_looks.prose`. If you can't ground a description in the recruiter's prose, do not include that capability area.
- Better to ship 2 capability areas grounded in the recruiter's prose than 5 areas you partially invented.

HONESTY ABOUT GAPS:
- If the recruiter wrote a thin capture, the polished output is honest about gaps. Don't paper over with fabricated content.
- Empty depth_distinction fields (`""`) are FINE if the captures don't support a confident answer. Do NOT make up depth definitions to fill the schema.
- An empty `non_fit_patterns: []` is FINE if `chapter_captures.lookalikes.non_fit_prose` is empty.

BANNED TOKENS (engineer jargon — never let these appear in any output field):
- hypothesis, tracking, lane_key, planner, critic, artifact
- Any snake_case identifier (matches `[a-z]+(?:_[a-z]+)+`). Translate to readable form: write "forward-deployed engineering" not "forward_deployed_engineering"; write "ML platform" not "ml_platform".

Return JSON ONLY. No prose preamble, no closing remark, no markdown code fences."""


def build_brief_polish_user_prompt(
    *,
    chapter_captures: dict[str, Any],
    seeded_v2_draft: dict[str, Any],
    role_title: str | None = None,
) -> str:
    """User prompt: structured input the polish call grounds itself in.

    Pass-through of the recruiter's chapter captures plus the heuristic
    seed. The seed gives the LLM a structural starting point and the
    preservation contracts (role_title, source_config). The captures
    are the ground truth for the hallucination guard.
    """

    role = _as_dict(chapter_captures.get("role"))
    good_looks = _as_dict(chapter_captures.get("good_looks"))
    lookalikes = _as_dict(chapter_captures.get("lookalikes"))
    where_to_look = _as_dict(chapter_captures.get("where_to_look"))
    design_rubric_capture = _as_dict(chapter_captures.get("design_rubric"))

    captures_payload: dict[str, Any] = {
        "role": {
            "title": _as_str(role.get("title")) or _as_str(role_title),
            "framing": _as_str(role.get("framing")),
        },
        "good_looks": {
            "prose": _as_str(good_looks.get("prose")),
        },
        "lookalikes": {
            "exemplars_prose": _as_str(lookalikes.get("exemplars_prose")),
            "non_fit_prose": _as_str(lookalikes.get("non_fit_prose")),
        },
        "where_to_look": {
            "target_modules": _as_str_list(where_to_look.get("target_modules")),
            "linkedin_project_id": _as_str(
                where_to_look.get("linkedin_project_id")
            ),
            "linkedin_project_name": _as_str(
                where_to_look.get("linkedin_project_name")
            ),
            "anything_else": _as_str(where_to_look.get("anything_else")),
        },
    }
    if design_rubric_capture:
        # Include the recruiter-authored design_rubric so the LLM sees the exact
        # payload it must preserve under the system prompt.
        captures_payload["design_rubric"] = design_rubric_capture

    payload = {
        "chapter_captures": captures_payload,
        "seeded_v2_draft": seeded_v2_draft,
    }

    return (
        "Reshape the recruiter's intake captures into a polished V2 brief draft.\n\n"
        "INPUT (structured — preserve identity fields exactly; ground capability "
        "areas in good_looks.prose):\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Return JSON only matching the schema in the system prompt."
    )


# Append the executive-register addendum after the base schema, voice,
# preservation, and grounding rules. It adds executive vocabulary and preserves
# confidentiality_class, prior_search, and executive_calibration.
_EXEC_REGISTER_ADDENDUM = """

EXECUTIVE-REGISTER ADDENDUM (executive-search briefs only — recruiter targets exec hires; brief carries `target_modules` containing `"exec_search"`).

Voice tightening:
- Replace recruiter-readable but consumer-soft phrasing ("we're looking for someone who's owned scaling") with operator-grade phrasing ("VP-or-above scope; org of 50+; P&L responsibility"). Don't dilute the candidate-fit signal in the depth_distinction with adjective stacks.
- Capability area descriptions must lean on scope verbs (owned, led, ran, exited, acquired, restructured) rather than activity verbs (built, used, leveraged). Recruiters at this level care about the verbs that imply ownership and outcome, not the ones that imply contribution.
- The depth_distinction is the load-bearing field for an exec brief. Spend the editorial budget there: the builder vs. user definitions should clearly distinguish the operator level the role demands.

Schema preservation rules (HARD CONTRACTS — output is rejected and the recruiter's scaffolded draft is shown instead if you violate these):
- If the input contains `seeded_v2_draft.confidentiality_class` and its value is `"referenceable"` or `"blind"`, the output `confidentiality_class` MUST equal it character-for-character. Do NOT downgrade a confidential brief to `"open"` — confidentiality is the recruiter's policy decision, not yours to "improve."
- If the input contains `seeded_v2_draft.prior_search.ruled_out_urls` (non-empty), the output `prior_search` MUST carry the same list (order not contract-bearing; set-equality is the test).
- If the input contains `seeded_v2_draft.executive_calibration` (sector / stage / pnl_scale_usd / register_notes), the output `executive_calibration` MUST carry the same fields verbatim. The recruiter encoded these intentionally; they're not editorial.

Vocabulary:
- Use sector / stage / P&L scale vocabulary in capability area descriptions when the brief carries it. "Series-D-to-acquisition operator" reads better than "growth-stage leader."
- "Operator" and "builder" are valid words at this register. "Manager" is recruiter-readable but doesn't carry exec weight; prefer "leader" or the role-specific noun (CFO, VP Engineering, COO).

Honesty:
- If the recruiter's `executive_calibration` is partially populated (only sector, not stage), preserve what's there; don't invent the missing fields.
- An empty `prior_search` is FINE for fresh searches — preserve as empty rather than fabricating "no candidates ruled out yet."
"""


def build_brief_polish_exec_system_prompt() -> str:
    """Build the executive-register brief-polish system prompt.

    Return the base prompt plus the executive-register addendum. The addendum
    tightens voice, adds executive preservation contracts, and shifts capability
    vocabulary toward operator-grade verbs. Use it when ``target_modules``
    contains ``"exec_search"``; other briefs use the base prompt unchanged.
    """

    return build_brief_polish_system_prompt() + _EXEC_REGISTER_ADDENDUM


def _is_exec_search_brief(seeded_v2_draft: dict) -> bool:
    """Detect whether the seeded V2 draft targets the exec_search module."""

    target_modules = seeded_v2_draft.get("target_modules") if isinstance(seeded_v2_draft, dict) else None
    if not isinstance(target_modules, list):
        return False
    return "exec_search" in target_modules


# Re-export `_normalize_text` so tests of this module can use it without
# reaching into `briefing_polish` directly. Kept as a private alias.
__all__ = (
    "BANNED_BRIEFING_TOKENS",
    "BriefPolishBackend",
    "BriefPolishResult",
    "HALLUCINATION_OVERLAP_THRESHOLD",
    "HEURISTIC_CONFIDENCE_DENOMINATOR",
    "HeuristicBriefPolishBackend",
    "MIN_SUBSTANTIVE_CHARS",
    "POLISH_MAX_TOKENS",
    "SNAKE_CASE_IDENTIFIER_RE",
    "build_brief_polish_exec_system_prompt",
    "build_brief_polish_system_prompt",
    "build_brief_polish_user_prompt",
)
