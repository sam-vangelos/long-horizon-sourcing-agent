"""Brief polish — Cloris-voice reshape over recruiter intake captures.

Owns the transformation from intake chapter captures (free-form prose)
to a polished, schema-correct V2 brief draft. Runs at intake time, not
at post-run analysis time — sibling of :mod:`market_intelligence.briefing_polish`
in pattern, distinct in domain.

Two backends:

- :class:`BriefPolishBackend`: Opus LLM rewrite. Falls through to
  heuristic on any of seven conditions (see :func:`BriefPolishBackend.polish`
  docstring).
- :class:`HeuristicBriefPolishBackend`: deterministic builder from
  chapter captures. Mirrors the frontend ``seedV2DraftFromChapters``
  scaffolder so server-side and client-side scaffolds are byte-identical
  on the structured fields. Always passes :func:`validate_v2_brief`.

Confidence is computed PROGRAMMATICALLY (not LLM self-rating):

- Heuristic: signal-density. ``populated_chapter_fields / 7``.
- LLM: flat ``1.0`` on success (no detected failure mode across the
  seven cascade routes). Acknowledged placeholder; post-trial calibration
  target uses the logged ``overlap_avg`` distribution.

Domain-leak note: this module sits under ``market_intelligence/`` for
clean sibling import of the helpers in :mod:`briefing_polish`
(``BANNED_BRIEFING_TOKENS``, ``SNAKE_CASE_IDENTIFIER_RE``,
``_has_llm_access``, ``_normalize_text``). Architectural debt against
``AGENTS.md``'s "post-run intelligence" scoping — tracked for week-2
cleanup (extract shared helpers into ``shared/llm_polish_common.py``
and move this module to its proper domain).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from shared.brief_v2_schema import BriefSchemaError, validate_v2_brief
# A.2 cache-gap remediation: brief polish system prompt is large
# (~6KB voice/preservation contract); caching it cuts repeat-call
# cost by ~90% within the 5-minute TTL. Same signature as opus_llm
# (called once per intake session, but the cache extends across the
# polish + reflection + chief-of-staff calls in the same window).
from shared.llm_clients import opus_llm_cached as opus_llm
from shared.observability import observe

from market_intelligence.briefing_polish import (
    BANNED_BRIEFING_TOKENS,
    SNAKE_CASE_IDENTIFIER_RE,
    _has_llm_access,
)


# Hallucination guard threshold: minimum average lexical overlap between
# capability_areas[*].description and the recruiter's good_looks.prose.
# Below this, we treat the LLM as having invented capability areas and
# cascade to the heuristic seed. Starts at 0.30; tunable based on logged
# overlap_avg distribution post-trial. Threshold lives at module scope
# so it's a one-line tweak when telemetry tells us the cluster shifted.
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

# Overlap-token regex: lowercased, alpha-only, ≥3 chars. The 3-char
# floor filters stop-word-ish tiny tokens ("a", "is", "of") that would
# inflate spurious overlap. Empirical cutoff; tunable post-trial along
# with HALLUCINATION_OVERLAP_THRESHOLD.
_OVERLAP_TOKEN_RE = re.compile(r"[a-z]{3,}")


# ---------------------------------------------------------------------------
# Telemetry — log every polish call with [intake] prefix
# ---------------------------------------------------------------------------


_FALLBACK_REASON_RE = re.compile(r"\bfallback reason=([A-Za-z_][A-Za-z0-9_]*)\b")


def _emit_stage(message: str) -> None:
    """Print a single-line stderr log with the ``[intake]`` prefix.

    Mirrors :func:`market_intelligence.briefing_polish._emit_stage` in
    shape (single line, stderr, flushed) but uses ``[intake]`` instead
    of ``[market-intel]`` because brief polish runs at intake time, not
    at post-run analysis. Operators grep ``brief.polish:`` (this module)
    or ``reflection.polish:`` (Thread A) to see each polish stream
    interleaved with the engine's other stage logs.

    Phase 1 of Langfuse adoption: when the message carries a
    ``fallback reason=<reason>`` token, the helper ALSO emits the
    reason as a Langfuse span attribute under
    ``cascade.fallback_reason``. Same single-bridging-point pattern
    as :func:`cloris.chief_of_staff.agent._emit_stage`.
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
    # Reopen Stage 2 (Decision 3 / adversarial-ledger flaw "contract-break"):
    # recruiter priors ride ALONGSIDE the polished draft as advisory
    # context, NEVER merged into ``v2_draft``. Merging would break the
    # "brief = exactly what the recruiter authored" invariant and the
    # ``RECOGNIZED_V2_KEYS`` whitelist (``shared/brief_v2_schema.py``).
    # ``None`` means "no recruiter context hydrated for this polish call"
    # (the default — every existing caller is unaffected). When a
    # ``recruiter_id`` is passed to :meth:`BriefPolishBackend.polish`, this
    # carries the recruiter's active taste signals for the brief's domain,
    # for the caller to surface/log/audit separately — it is deliberately
    # excluded from :meth:`to_meta_dict` so it never leaks into the
    # persisted brief blob.
    recruiter_priors_overlay: dict | None = None

    def __post_init__(self) -> None:
        if not self.polished_at:
            self.polished_at = datetime.now(timezone.utc).isoformat()

    def to_meta_dict(self) -> dict:
        """Return metadata for ``state_json["v2_draft_polish_meta"]``.

        Excludes ``v2_draft`` (which lives at ``state_json["v2_draft"]``,
        not nested under polish_meta — see plan's state_json schema
        bullets). The Reference Slip in the review chapter reads from
        this shape verbatim.
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

        # Researcher Slice 7: seed `source_config.researcher` from the
        # where_to_look chapter captures (mirrors the Path 3 LinkedIn
        # promotion above). Heuristic does NOT seed explicit floor
        # overrides — those resolve at evaluation time via
        # `researcher.discipline_defaults.resolve_floors`. We only
        # propagate the recruiter's free-text inputs.
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

        # Designer Slice 4 — hydrate `design_rubric` from chapter
        # captures pass-through. The recruiter authors the rubric in
        # the intake wizard's `design_rubric` chapter; the heuristic
        # backend forwards it onto the v2_draft so a brief that
        # never invokes the LLM polish still carries the rubric
        # the recruiter wrote. Empty / missing → no key (the
        # `_design_rubric_drift` cascade entry treats absence as
        # "nothing to preserve," not "drift").
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
    """Opus-driven brief polish. Falls through to heuristic on failure.

    Single entry point: :func:`polish`. Seven failure modes converge
    on :class:`HeuristicBriefPolishBackend`:

    1. ``opus_llm`` raises (network, rate-limit, timeout, parse error).
    2. JSON valid but :func:`validate_v2_brief` raises (schema invalid).
    3. Banned-token check fails: any of :data:`BANNED_BRIEFING_TOKENS`
       in ``capability_areas[*].name|description``,
       ``depth_distinction.*``, or ``non_fit_patterns[*].label|why_not``.
    4. Snake_case identifier check fails in the same set of fields.
    5. Path 3 drift: input had ``source_config.linkedin.project_id``
       but output dropped or changed it.
    6. Hallucination: average lexical overlap between
       ``capability_areas[*].description`` and ``good_looks.prose`` is
       below :data:`HALLUCINATION_OVERLAP_THRESHOLD`.
    7. Role-title drift: input had non-empty ``role.title`` but output
       dropped or changed it.

    Each cascade emits ``_emit_stage`` with ``reason=`` and route-specific
    detail so the cascade is traceable in logs. Telemetry log lines
    (start, hallucination_check, fallback, done) are documented in the
    Telemetry section of the slice plan. Routes 5 and 7 are the hard
    preservation contracts new for brief polish; the others transplant
    Thread A's pattern with brief-shaped target fields.
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
        """Polish a v2 draft; optionally attach recruiter priors SEPARATELY.

        The polish itself (LLM cascade → heuristic fallback) is unchanged
        and runs on the recruiter-authored seed — recruiter priors NEVER
        feed into it. When ``recruiter_id`` is supplied (reopen Stage 2),
        the result also carries that recruiter's active taste signals for
        the brief's domain on
        :attr:`BriefPolishResult.recruiter_priors_overlay`, as advisory
        context for the caller to surface/audit. The overlay is never
        merged into ``v2_draft`` and never persisted into the brief.
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

        # Always emit the start line so cascade rates can be correlated
        # with input-richness during post-trial analysis.
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

        # Route 6: hallucination check. Always log overlap_avg + per-area
        # regardless of pass/fail — that's the post-trial calibration
        # data. Threshold is enforced only when good_looks.prose carries
        # content (otherwise there's nothing to ground against; the
        # empty-captures path catches the no-input case earlier).
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

        # Designer Slice 4: design_rubric preservation. Cascade entry
        # is named (`_design_rubric_drift`), not numbered, so parallel
        # sibling-module preservation helpers (researcher, oss
        # maintainers, exec search) can append at the next available
        # position without renumbering.
        rubric_drift = _design_rubric_drift(seeded=seeded.v2_draft, polished=raw)
        if rubric_drift:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _emit_stage(
                f"brief.polish:fallback reason=design_rubric_drift "
                f"detail={rubric_drift} elapsed_ms={elapsed_ms}"
            )
            return self._cascade_done(seeded, t0)

        # OSS Maintainers Slice 2: target_projects preservation.
        # Cascade entry is named (`_target_projects_drift`), not
        # numbered. Per the OSS Maintainers Module Spec §10 and the
        # source spec's named-not-numbered discipline, sibling
        # preservation helpers (researcher: `_research_topics_drift`,
        # exec search: `_confidentiality_class_drift`) append at the
        # next available position without renumbering each other.
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

        # OSS Maintainers + audit Move #24: target_stacks preservation.
        # Sibling of `_target_projects_drift`. Set-equality contract
        # (order is not contract-bearing for stack tags). Named-route
        # naming preserves cross-thread sibling discipline.
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

        # OSS Maintainers + audit Move #24: maintainership_level
        # preservation. Once a recruiter has approved a Move #9
        # `lower_maintainership_threshold` hunk, the brief carries the
        # lowered floor; the polish step must not silently raise it
        # back. Equality (case-insensitive after normalize).
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

        # Executive Search Slice 8: confidentiality_class preservation.
        # Cascade entry is named (`_confidentiality_class_drift`), not
        # numbered, so parallel-thread sibling routes append at the next
        # available position without renumbering. Per the spec's
        # confidentiality contract, the polish step MUST NOT silently
        # downgrade a brief's posture (e.g., `blind` → `open`); the
        # recruiter has to re-author confidentiality explicitly.
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

        # Researcher Slice 7: source_config.researcher preservation.
        # Cascade entry is named (`_research_topics_drift`), not
        # numbered, per the cross-thread "name by what it does"
        # discipline — the route preserves research_topics,
        # conference_allowlist, discipline, and any explicit floor
        # overrides character-for-character. The discipline default
        # resolves at evaluation time via
        # `researcher.discipline_defaults.resolve_floors`; the LLM is
        # not authorized to "improve" recruiter-authoritative inputs.
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

        # Success: LLM produced a polished, schema-valid, in-voice,
        # path-3-preserving, non-hallucinated, role-title-preserving,
        # design-rubric-preserving, target-projects-preserving,
        # confidentiality-class-preserving, researcher-source-config-
        # preserving v2_draft. Confidence is flat 1.0 for trial — see plan's
        # "Confidence formulas" subsection for the post-trial
        # calibration target.
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
    """Return drift descriptor when ``design_rubric`` drifts.

    Designer Slice 4 — preservation contract for the
    :class:`shared.brief_v2_schema.BriefDesignRubric` payload. If the
    seeded draft carried a non-empty ``design_rubric`` dict (i.e., the
    recruiter authored one in the intake wizard's ``design_rubric``
    chapter), the polished output MUST carry the same rubric byte-for-
    byte. The LLM is not authorized to "improve" principles, anchors,
    weights, or exemplars — taste is the recruiter's, not the model's.

    Cascade-entry naming follows the named-route discipline (mirrors
    :func:`_path3_drift` and :func:`_role_title_drift`); siblings like
    ``_research_topics_drift`` (Researcher), ``_target_projects_drift``
    (OSS Maintainers), and ``_confidentiality_class_drift`` (Executive
    Search) append at the next available cascade position without
    renumbering each other.

    Returns ``None`` when:
    - The seeded draft had no rubric (nothing to preserve).
    - The seeded rubric is empty dict (recruiter cleared it).
    - Seeded and polished rubrics are deep-equal.

    Returns a short diagnostic string otherwise. Whole-rubric byte
    equality is sufficient — any mutation across principles, anchors,
    weights, exemplars, or hard-reject patterns is a contract
    violation and the cascade falls through to the heuristic.
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
    """Return drift descriptor when ``target_projects`` drifts.

    OSS Maintainers Slice 2 — preservation contract for the
    recruiter-named GitHub repos that anchor maintainership-level
    classification. If the seeded draft carried a non-empty
    ``target_projects`` list, the polished output MUST carry the
    same set of projects. Order is not contract-bearing (recruiters
    name projects, not project order) so set-equality is the test;
    duplicates are normalized away.

    Cascade-entry naming follows the named-route discipline (mirrors
    :func:`_path3_drift`, :func:`_role_title_drift`,
    :func:`_design_rubric_drift`); siblings like
    ``_research_topics_drift`` (Researcher) and
    ``_confidentiality_class_drift`` (Executive Search) append at
    the next available cascade position without renumbering.

    Returns ``None`` when:
    - The seeded draft had no ``target_projects`` (nothing to
      preserve — this is the classic-github case per spec §11).
    - The seeded list is empty (recruiter cleared it).
    - Seeded and polished sets are equal modulo order + duplicates.

    Returns a short diagnostic string otherwise. ``target_stacks``
    and ``maintainership_level`` get a softer treatment (passthrough
    without drift gating) per spec §8: they are evaluation hints,
    not recruiter-authoritative anchors. Only ``target_projects``
    rises to a hard preservation contract.
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
    """Coerce a raw ``target_projects`` value to a normalized set.

    Empty / non-list / non-string entries drop out. Trims whitespace
    so ``"kubernetes/kubernetes"`` and ``"kubernetes/kubernetes "``
    are not treated as different. Lowercases ``owner/repo`` because
    GitHub treats those case-insensitively for resolution; this
    keeps the drift contract aligned with how the strategy seeder
    will dedup queries in Slice 7.
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
    """Return drift descriptor when ``confidentiality_class`` drifts.

    Executive Search Slice 8 — preservation contract for the
    recruiter's confidentiality posture. The brief polish step MUST
    NOT silently downgrade a brief's posture (e.g., ``"blind"`` →
    ``"open"``); the recruiter has to re-author confidentiality
    explicitly. Polish is editorial, not policy.

    Returns ``None`` (no drift) when:
    - The seeded draft had no ``confidentiality_class`` (no
      preservation contract — defaults to ``"open"`` downstream).
    - The seeded value is an empty string or ``"open"`` (the
      recruiter didn't make a confidentiality declaration; polish is
      free to leave it absent).
    - Seeded and polished values are equal (case-insensitive).

    Returns a short drift diagnostic otherwise. The descriptor lands
    in telemetry as ``brief.polish:fallback
    reason=confidentiality_class_drift detail=<descriptor>``.
    """

    seed_value = seeded.get("confidentiality_class")
    if not isinstance(seed_value, str):
        return None
    seed_normalized = seed_value.strip().lower()
    # Only `"referenceable"` and `"blind"` are recruiter-load-bearing;
    # `"open"` is the default and not subject to a hard preservation
    # contract (an LLM that drops it just falls back to default
    # behavior, which is fine).
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
    """Return drift descriptor when ``source_config.researcher`` drifts.

    Researcher Slice 7 — preservation contract for the recruiter-
    authored evaluation inputs at ``source_config.researcher``: the
    free-text ``research_topics``, the ``conference_allowlist``, the
    ``discipline`` (load-bearing for layered floor resolution per
    Spec Opinion 7), and any explicit floor overrides
    (``h_index_floor``, ``papers_in_window_floor``,
    ``papers_in_window_months``). The polish step MUST preserve these
    character-for-character — discipline + explicit floors are
    recruiter-authoritative; the LLM is not allowed to "improve" them.

    Cascade-entry naming follows the named-route discipline (mirrors
    :func:`_path3_drift`, :func:`_role_title_drift`,
    :func:`_design_rubric_drift`, :func:`_target_projects_drift`,
    :func:`_confidentiality_class_drift`); future sibling routes
    append at the next available position without renumbering.

    Returns ``None`` when:
    - The seeded draft had no ``source_config.researcher`` (nothing
      to preserve).
    - The seeded researcher sub-dict is empty (recruiter cleared it).
    - Each preserved key matches between seed and polish.

    Returns a short diagnostic string otherwise. ``research_topics``
    and ``conference_allowlist`` use set-equality (order is not
    contract-bearing for recruiter-named inputs); ``discipline`` and
    the floor fields use equality.
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
    """Return drift descriptor when ``maintainership_level`` drifts.

    OSS Maintainers + audit Move #24 — preservation contract for the
    recruiter's maintainership floor. The Move #9 reflection composer
    proposes ``lower_maintainership_threshold`` hunks; once a recruiter
    approves a hunk, the brief carries the lowered floor and the
    polish step MUST NOT silently raise it back. Polish is editorial,
    not policy.

    Returns ``None`` (no drift) when:
    - The seeded draft had no ``maintainership_level`` (no preservation
      contract — the github strategy seeder defaults are evaluation-time).
    - The seeded value is not a recognized level string.
    - Seeded and polished values are equal (case-insensitive).

    Returns a short drift diagnostic otherwise. The descriptor lands
    in telemetry as
    ``brief.polish:fallback reason=maintainership_level_drift detail=<descriptor>``.
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
    """Return drift descriptor when ``target_stacks`` drifts.

    OSS Maintainers + audit Move #24 — preservation contract for the
    recruiter-named language / framework / domain tags. Sibling of
    :func:`_target_projects_drift`. Set-equality contract (order is
    not contract-bearing); empty seed ⇒ no preservation.

    Returns ``None`` (no drift) when:
    - The seeded draft had no ``target_stacks`` (nothing to preserve).
    - The seeded list is empty.
    - Seeded and polished sets are equal modulo order + duplicates +
      whitespace + case.

    Returns a short drift diagnostic otherwise.
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


def _describe_rubric_drift(*, seed: dict, polish: dict) -> str:
    """Produce a short, recruiter-readable drift descriptor.

    The descriptor lands in telemetry (``brief.polish:fallback
    reason=design_rubric_drift detail=<descriptor>``) so post-trial
    analysis can tell whether drift came from principle-level
    rewrites, weight tweaks, or exemplar additions. Trimmed to keep
    the log line bounded.
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
    """Compute lexical overlap of each capability_area description with prose.

    Returns ``(avg_overlap, per_area_overlap_list)``. Overlap is Jaccard:
    size of intersection of ≥3-char lowercased alpha tokens divided by
    union. Empty description or empty good_looks.prose => overlap=1.0
    for that area (we can't measure, so don't penalize).

    The per-area list is logged on every LLM call (see Telemetry) so
    post-trial analysis can decide between "average ≥ threshold,"
    "min ≥ threshold," or a length-weighted variant.
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


# ---------------------------------------------------------------------------
# Reopen Stage 2: recruiter priors overlay (SEPARATE — never merged)
# ---------------------------------------------------------------------------


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
    """System prompt for the brief polish step.

    SOURCE OF TRUTH: ``docs/cloris-ia-doctrine-cursor.md`` voice/copy rules
    (mirror of :func:`market_intelligence.research_prompts.build_briefing_polish_system_prompt`,
    which last verified on 2026-05-25). When updating this prompt,
    re-read the doctrine and update the date.

    Encodes voice rules (recruiter-readable, no engineer vocab, no
    snake_case), schema output spec (V2 brief shape), preservation
    contracts (role_title + source_config), the hallucination guard
    (overlap with good_looks.prose), and honesty about gaps.
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
  "maintainership_level": "<from input — preserve verbatim if present>"
}

PRESERVATION RULES (HARD CONTRACTS — output is rejected and the recruiter's scaffolded draft is shown instead if you violate these):
- If the input contains `seeded_v2_draft.role_title` (non-empty), the output `role_title` MUST equal it character-for-character. Do not "improve" the title.
- If the input contains `seeded_v2_draft.source_config.linkedin.project_id`, the output `source_config.linkedin.project_id` MUST equal it. Do not invent, drop, or modify it. The same applies to `project_name` if present.
- If the input contains `seeded_v2_draft.design_rubric` (non-empty), the output `design_rubric` MUST equal it byte-for-byte. Do not modify principles, anchors, weights, calibration_exemplars, hard_reject_patterns, or discipline_weight_overrides. The rubric encodes the recruiter's taste; the model is not authorized to "improve" it.
- If the input contains `seeded_v2_draft.target_projects` (non-empty), the output `target_projects` MUST contain the same set of "owner/repo" entries (case-insensitive; ordering and duplicates don't matter, but no add or drop). The recruiter named those projects; "kubernetes maintainer" is not the same query as "container orchestration maintainer".
- If the input contains `seeded_v2_draft.source_config.researcher` (non-empty), the output MUST preserve `research_topics` (set-equality), `conference_allowlist` (set-equality), `discipline` (verbatim — one of `nlp | ml_general | vision | rl | systems | theory | biomedical | other`), and any explicit floor overrides (`h_index_floor`, `papers_in_window_floor`, `papers_in_window_months`) byte-for-byte. The discipline is the load-bearing field for layered floor resolution; the model is not authorized to "improve" recruiter-authoritative inputs.
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
        # Designer Slice 4: pass the rubric through to the LLM so it
        # has the byte-equality target in front of it. The LLM is
        # told (per system prompt) to preserve verbatim; this mirrors
        # the seeded_v2_draft pass-through.
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


# Executive Search Slice 8: exec-register addendum to the brief polish
# system prompt. Appended (not replacing) so the schema, voice rules,
# preservation contracts, and hallucination guards from the base prompt
# are kept intact. The addendum tightens the editorial register for
# executive-search briefs and adds preservation contracts for
# `confidentiality_class`, `prior_search`, and the executive-calibration
# vocabulary (sector, stage, P&L scale).
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
    """Slice 8: exec-register variant of the brief polish system prompt.

    Returns the base prompt + the executive-register addendum. The
    addendum tightens voice, adds confidentiality / prior-search /
    executive-calibration preservation contracts, and shifts the
    capability-description vocabulary toward operator-grade verbs.

    Called by :class:`BriefPolishBackend.polish` when the seeded V2
    draft's ``target_modules`` contains ``"exec_search"``. Non-exec
    briefs continue to use :func:`build_brief_polish_system_prompt`
    unchanged (characterization regression).
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
