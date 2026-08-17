"""Deterministic lint for machine-generated briefs (preflight v2 output).

P4 of plans/sourcing-rigor-hardening.md: generated prompts are code. A
preflight-generated brief is a live judgment document — it can carry doctrine
injections (disposition language inside pattern strings), target/blacklist
confusion (the hiring company filed under a positive employer tier), and
template artifacts (the example yes-rate band riding along unexamined). This
module is the injection scanner that runs between parse and go-live in
``linkedin/orchestrator._run_preflight_v2``: error findings abort the
generation attempt (the caller's retry→abort path treats them like a parse
failure), warnings print and proceed.

Pure functions over the parsed preflight dict — no I/O, no LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

# The historical template/loader constants. A generated band that equals them
# EXACTLY, with no stated rationale, is indistinguishable from the template
# riding along — which is precisely the failure this check exists to catch.
TEMPLATE_DEFAULT_BAND = (0.25, 0.55)

# Disposition language that must never appear inside a trajectory pattern
# string. Patterns DESCRIBE what a snippet shows; the evaluation template —
# not the brief — decides how each category is treated. ("default YES" inside
# a live judgment document was a confirmed doctrine violation; see
# plans/sourcing-rigor-audit.md region 3.)
HEDGE_PHRASES = (
    "default yes",
    "default to yes",
    "defaults to yes",
    "must default",
    "when in doubt",
    "lean toward yes",
    "lean towards yes",
    "favor yes",
    "favors yes",
    "err on the side of yes",
    "err on the side of inclusion",
)

# Save-directional hedges scanned in PROSE fields that render into the live
# judge (depth_distinction, minimum_bar_description, non_fit descriptions).
# Deliberately narrower than HEDGE_PHRASES: prose like edge_case_guidance
# legitimately discusses dispositions ("when in doubt, reject" is
# doctrine-ALIGNED), so bare "when in doubt" is excluded here and only
# save-favoring language blocks. A fixed-phrase list cannot catch every
# paraphrase — this is a tripwire for the known injection shapes, not a
# semantic guarantee (correctness lens, Wave 1); a disposition-verb heuristic
# is a Wave 2+ upgrade.
PROSE_SAVE_HEDGES = (
    "default yes",
    "default to yes",
    "defaults to yes",
    "lean toward yes",
    "lean towards yes",
    "favor yes",
    "favors yes",
    "err on the side of yes",
    "err on the side of inclusion",
    "err on the side of saving",
    "prefer inclusion",
    "when in doubt, save",
    "when in doubt, pass",
    "when uncertain, pass",
    "when uncertain, save",
    "resolve borderline toward save",
    "resolve toward save",
    "toward save when uncertain",
)

EXPERIENCE_MEASURE_QUALIFIERS = (
    ("post-graduation", re.compile(r"\bpost[-\s]grad(?:uation)?\b", re.IGNORECASE)),
    ("full-time", re.compile(r"\bfull[-\s]time\b", re.IGNORECASE)),
    ("consecutive", re.compile(r"\bconsecutive\b", re.IGNORECASE)),
    ("uninterrupted", re.compile(r"\buninterrupted\b", re.IGNORECASE)),
)
_QUALIFIER_SENTENCE_SPLIT = re.compile(r"[.!?\n;]+")
# Generic linguistic negation markers (not vertical vocabulary): a sentence
# mentioning a qualifier AUTHORIZES it only when none of these govern it.
_QUALIFIER_NEGATION_RE = re.compile(
    r"do not|don't|never|no such|not restrict|rather than|instead of",
    re.IGNORECASE,
)
_HARD_CEILING_NEGATION_RE = re.compile(
    r"do not|don't|never|not (?:a )?hard|isn't (?:a )?hard|"
    r"is not (?:a )?hard|advisory|soft (?:ceiling|cap|maximum)|"
    r"not (?:an )?automatic reject",
    re.IGNORECASE,
)
_HARD_CEILING_RE = re.compile(
    r"\b(?:hard|strict|absolute|non[- ]negotiable)\b",
    re.IGNORECASE,
)
_CEILING_RE = re.compile(
    r"\b(?:experience\s+)?(?:ceiling|cap|maximum|upper\s+bound)\b",
    re.IGNORECASE,
)

_REQUIRED_KEYS = (
    "role_title",
    "role_summary",
    "capability_areas",
    "depth_distinction",
    "non_fit_patterns",
    "facial_calibration",
)

_PATTERN_ARRAY_FIELDS = (
    "fast_exit_patterns",
    "trajectory_yes_patterns",
    "trajectory_ambiguous_patterns",
    "trajectory_no_patterns",
)


class GeneratedBriefLintError(RuntimeError):
    """Raised by callers when a generated brief carries blocking lint findings."""


@dataclass(frozen=True)
class BriefLintFinding:
    code: str
    severity: str  # "error" | "warning"
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _normalize_boolean(value: Any) -> str:
    """Collapse internal whitespace and casefold — the example-copy-check key."""
    return " ".join(str(value or "").split()).casefold()


def _quoted_terms_normalized(value: Any) -> set[str]:
    """Whitespace-collapsed, casefolded quoted spans inside a Boolean string."""
    return {
        " ".join(match.split()).casefold()
        for match in re.findall(r'"([^"]*)"', str(value or ""))
        if match.strip()
    }


def _operator_instruction_text(instructions: Iterable[str]) -> str:
    if isinstance(instructions, str):
        return instructions
    return "\n".join(str(item or "") for item in instructions or ())


def _has_non_negated_qualifier_authorization(
    source_text: str,
    pattern: re.Pattern[str],
) -> bool:
    """Heuristic authorization: qualifier mention must be in a non-negated sentence."""
    for sentence in _QUALIFIER_SENTENCE_SPLIT.split(source_text or ""):
        if not pattern.search(sentence):
            continue
        if _QUALIFIER_NEGATION_RE.search(sentence):
            continue
        return True
    return False


def _operator_authorizes_hard_experience_ceiling(
    operator_instructions: Iterable[str],
) -> bool:
    """Recognize only an explicit, non-negated operator-authored hard gate."""

    text = _operator_instruction_text(operator_instructions)
    for sentence in _QUALIFIER_SENTENCE_SPLIT.split(text):
        if _HARD_CEILING_NEGATION_RE.search(sentence):
            continue
        if _HARD_CEILING_RE.search(sentence) and _CEILING_RE.search(sentence):
            return True
    return False


_JD_HEADING_VERBS = re.compile(
    r"\b(?:am|are|be|been|being|build|builds|built|can|collaborate|"
    r"collaborates|created?|delivered?|design(?:ed|s)?|develop(?:ed|s)?|"
    r"did|do|does|drive|drives|drove|ensure[sd]?|execute[sd]?|had|has|"
    r"have|implement(?:ed|s)?|is|lead|leads|led|maintain(?:ed|s)?|"
    r"manage[sd]?|must|operate[sd]?|oversee[sn]?|own(?:ed|s)?|ran|run|"
    r"runs|ship(?:ped|s)?|should|support(?:ed|s)?|was|were|will|work(?:ed|s)?|"
    r"wrote|write[sd]?|you)\b",
    re.IGNORECASE,
)


def _register_norm(value: Any) -> str:
    return " ".join(str(value or "").strip(" \t#*-:").split()).casefold()


def _jd_heading_lines(jd_text: str) -> set[str]:
    """Return short JD lines that look like headings.

    Simple heuristic by design: after stripping common markdown/list markers
    and trailing colons, a heading is a short line with no obvious verb.
    """
    headings: set[str] = set()
    for line in str(jd_text or "").splitlines():
        text = line.strip()
        if not text:
            continue
        text = re.sub(r"^\s*(?:#{1,6}|\*|-|\d+[.)])\s*", "", text).strip()
        normalized = _register_norm(text)
        if not normalized:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z0-9'+/-]*", text)
        if 1 <= len(words) <= 8 and len(text.strip(" :")) <= 80:
            if not _JD_HEADING_VERBS.search(text):
                headings.add(normalized)
    return headings


def _trajectory_or_archetype_labels(data: dict[str, Any]) -> set[str]:
    labels: set[str] = set()

    fc = data.get("facial_calibration")
    fc = fc if isinstance(fc, dict) else {}
    for field_name in _PATTERN_ARRAY_FIELDS:
        labels.update(_register_norm(item) for item in _string_items(fc.get(field_name)))

    for key in ("archetypes", "noise_archetypes", "non_fit_patterns"):
        for item in data.get(key) or []:
            if isinstance(item, dict):
                for label_key in ("label", "name"):
                    normalized = _register_norm(item.get(label_key))
                    if normalized:
                        labels.add(normalized)
            else:
                normalized = _register_norm(item)
                if normalized:
                    labels.add(normalized)
    return labels


def _example_vocabulary(data: dict[str, Any]) -> set[str]:
    """The brief's own quoted vocabulary a model-authored example should touch:
    the union of capability_areas[*].key_terms and any canonical_*_patterns."""
    terms: set[str] = set()
    for area in data.get("capability_areas") or []:
        if isinstance(area, dict):
            for term in _string_items(area.get("key_terms")):
                normalized = " ".join(term.split()).casefold()
                if normalized:
                    terms.add(normalized)
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("canonical_") and key.endswith("_patterns"):
            for term in _string_items(value):
                normalized = " ".join(term.split()).casefold()
                if normalized:
                    terms.add(normalized)
    return terms


# RC3 (2026-07-04): the calibration-mirror fields the deterministic opening
# sort consumes (linkedin/strategy.py _opening_priority). Absent-en-bloc is a
# warning (blind-but-safe openings); malformed entries are errors.
_OPENING_MIRROR_FIELDS = (
    "canonical_title_patterns",
    "canonical_company_patterns",
    "canonical_framework_patterns",
    "canonical_broad_patterns",
    "edge_case_patterns",
    "edge_case_company_patterns",
)


def lint_generated_brief(
    data: dict[str, Any],
    *,
    seed_blacklist: Iterable[str] = (),
    jd_text: str = "",
    operator_instructions: Iterable[str] = (),
) -> list[BriefLintFinding]:
    """Lint a parsed preflight brief dict. Returns findings, worst first.

    ``seed_blacklist`` carries the operator's pre-existing employer blacklist
    (from the seed brief) so a hiring company the OPERATOR named is checked
    against the generated tiers even when the model omitted it.

    ``jd_text`` and ``operator_instructions`` are the only sources allowed to
    authorize restrictive experience-measure qualifiers.
    """
    findings: list[BriefLintFinding] = []

    if not isinstance(data, dict):
        return [
            BriefLintFinding(
                code="schema_not_object",
                severity="error",
                message="Preflight response is not a JSON object.",
            )
        ]

    # --- Schema presence -------------------------------------------------
    for key in _REQUIRED_KEYS:
        if key not in data or data.get(key) in (None, "", [], {}):
            findings.append(
                BriefLintFinding(
                    code="schema_missing_key",
                    severity="error",
                    message=f"Required field {key!r} is missing or empty.",
                )
            )

    engagement_context = data.get("engagement_context")
    if engagement_context in (None, "", {}):
        findings.append(
            BriefLintFinding(
                code="engagement_context_missing",
                severity="error",
                message=(
                    "engagement_context is required for every generated brief."
                ),
            )
        )
    elif not isinstance(engagement_context, dict):
        findings.append(
            BriefLintFinding(
                code="engagement_context_invalid",
                severity="error",
                message="engagement_context must be an object.",
            )
        )
    else:
        posture = engagement_context.get("selectivity_posture")
        engagement = engagement_context
        invalid_optional_fields = [
            key
            for key in (
                "hiring_company",
                "engagement_description",
                "talent_bar_statement",
            )
            if key in engagement
            and not isinstance(engagement[key], str)
        ]
        if posture not in {"selective", "coverage"} or invalid_optional_fields:
            details = (
                " Optional text fields must be strings: "
                f"{', '.join(invalid_optional_fields)}."
                if invalid_optional_fields
                else ""
            )
            findings.append(
                BriefLintFinding(
                    code="engagement_context_invalid",
                    severity="error",
                    message=(
                        "engagement_context requires selectivity_posture "
                        f"selective|coverage.{details}"
                    ),
                )
            )

    capability_areas = data.get("capability_areas")
    if isinstance(capability_areas, list) and len(capability_areas) == 0:
        # Covered by schema_missing_key above (empty list), but keep the
        # dedicated code for a non-empty-but-hollow list of non-dicts.
        pass
    elif isinstance(capability_areas, list) and not any(
        isinstance(area, dict) and str(area.get("name", "") or "").strip()
        for area in capability_areas
    ):
        findings.append(
            BriefLintFinding(
                code="empty_capability_areas",
                severity="error",
                message="capability_areas contains no named areas.",
            )
        )
    if isinstance(capability_areas, list):
        pattern_labels = _trajectory_or_archetype_labels(data)
        jd_headings = _jd_heading_lines(jd_text)
        for area in capability_areas:
            if not isinstance(area, dict):
                continue
            area_name = str(area.get("name", "") or "").strip()
            area_name_norm = _register_norm(area_name)
            for term in _string_items(area.get("candidate_register_terms")):
                term_norm = _register_norm(term)
                if not term_norm:
                    continue
                if (
                    term_norm == area_name_norm
                    or term_norm in pattern_labels
                    or term_norm in jd_headings
                ):
                    findings.append(
                        BriefLintFinding(
                            code="jd_register_in_candidate_terms",
                            severity="warning",
                            message=(
                                f"candidate_register_terms for capability area "
                                f"{area_name!r} includes JD/register label {term!r}; "
                                "candidate-register terms must be vocabulary a "
                                "qualified person would plausibly write on their "
                                "own profile."
                            ),
                        )
                    )

    # --- Band ceiling requires a measure -----------------------------------
    # A ceiling without a definition of WHICH years the band counts is
    # unenforceable: the first live run with a bare "4-10" band rejected one
    # candidate on months-of-domain-tenure and saved a 20-year career against
    # the same band. The measure is brief content; the requirement is
    # mechanism.
    ceiling = data.get("maximum_years_experience")
    has_ceiling = isinstance(ceiling, int) and not isinstance(ceiling, bool) and ceiling > 0
    if has_ceiling and not _norm(data.get("experience_measure")):
        findings.append(
            BriefLintFinding(
                code="band_ceiling_without_measure",
                severity="error",
                message=(
                    "maximum_years_experience is set but experience_measure is "
                    "empty — define which years the band counts or drop the "
                    "ceiling."
                ),
            )
        )
    ceiling_hardness = data.get("maximum_years_experience_is_hard", False)
    if not isinstance(ceiling_hardness, bool):
        findings.append(
            BriefLintFinding(
                code="experience_ceiling_hardness_invalid",
                severity="error",
                message=(
                    "maximum_years_experience_is_hard must be a JSON boolean; "
                    "omit it or use false for an advisory ceiling."
                ),
            )
        )
    elif ceiling_hardness:
        if not has_ceiling:
            findings.append(
                BriefLintFinding(
                    code="hard_experience_ceiling_without_ceiling",
                    severity="error",
                    message=(
                        "maximum_years_experience_is_hard is true but no valid "
                        "maximum_years_experience is set."
                    ),
                )
            )
        if not _operator_authorizes_hard_experience_ceiling(
            operator_instructions
        ):
            findings.append(
                BriefLintFinding(
                    code="hard_experience_ceiling_without_operator_authorization",
                    severity="error",
                    message=(
                        "A hard experience ceiling requires explicit operator "
                        "authorization naming the ceiling as a hard gate. A JD "
                        "range alone authorizes advisory leveling context only."
                    ),
                )
            )
    experience_measure = str(data.get("experience_measure", "") or "")
    qualifier_source_text = "\n".join(
        (str(jd_text or ""), _operator_instruction_text(operator_instructions))
    )
    for qualifier, pattern in EXPERIENCE_MEASURE_QUALIFIERS:
        if pattern.search(
            experience_measure
        ) and not _has_non_negated_qualifier_authorization(
            qualifier_source_text,
            pattern,
        ):
            findings.append(
                BriefLintFinding(
                    code="experience_measure_unstated_qualifier",
                    severity="warning",
                    message=(
                        f"experience_measure uses {qualifier!r}, but that qualifier "
                        "does not appear in the JD or operator instructions; absent "
                        "a stated restriction, count total professional experience."
                    ),
                )
            )

    # --- Domain absence must not ride the depth axis -----------------------
    # The depth slot (builder/user) carries ownership semantics; encoding
    # "has no <domain>" into user_definition turns domain absence into a
    # terminal USER=REJECT verdict and bypasses the transferability path
    # built for exactly those candidates. Detector: an absence marker
    # immediately preceding one of the brief's own capability key_terms.
    absence_marker = re.compile(
        r"\b(?:no|without|lacks|lacking|lack of|zero|absence of|none of|missing)\b[^.;]{0,40}$",
        re.IGNORECASE,
    )

    def _domain_absence_hits(text: str) -> list[str]:
        hits: list[str] = []
        lowered = str(text or "")
        for term in sorted(_example_vocabulary(data)):
            for match in re.finditer(re.escape(term), lowered, re.IGNORECASE):
                if absence_marker.search(lowered[: match.start()]):
                    hits.append(term)
                    break
        return hits

    depth = data.get("depth_distinction")
    depth = depth if isinstance(depth, dict) else {}
    user_hits = _domain_absence_hits(str(depth.get("user_definition", "") or ""))
    if user_hits:
        findings.append(
            BriefLintFinding(
                code="domain_absence_in_depth_slot",
                severity="error",
                message=(
                    "user_definition rejects on domain ABSENCE "
                    f"({', '.join(user_hits[:4])}) — the depth slot carries "
                    "ownership only; a missing domain is a transferability "
                    "question (or a non-fit with an explicit JD/operator "
                    "mandate), never a USER-depth verdict."
                ),
            )
        )
    for nf in data.get("non_fit_patterns") or []:
        if not isinstance(nf, dict):
            continue
        nf_hits = _domain_absence_hits(str(nf.get("why_not", "") or ""))
        if nf_hits:
            findings.append(
                BriefLintFinding(
                    code="domain_absence_non_fit",
                    severity="warning",
                    message=(
                        f"non_fit_pattern {str(nf.get('label', '?'))!r} rejects on "
                        f"domain absence ({', '.join(nf_hits[:4])}) — allowed only "
                        "when the JD or an operator instruction explicitly states "
                        "domain experience as a requirement; verify that mandate "
                        "exists."
                    ),
                )
            )

    # --- Facial calibration band -----------------------------------------
    fc = data.get("facial_calibration")
    fc = fc if isinstance(fc, dict) else {}
    low = fc.get("expected_yes_rate_low")
    high = fc.get("expected_yes_rate_high")
    band_ok = (
        isinstance(low, (int, float))
        and isinstance(high, (int, float))
        and not isinstance(low, bool)
        and not isinstance(high, bool)
        and 0.0 < float(low) < float(high) < 1.0
    )
    if not band_ok:
        findings.append(
            BriefLintFinding(
                code="band_invalid",
                severity="error",
                message=(
                    "facial_calibration expected_yes_rate_low/high must be decimals "
                    f"with 0 < low < high < 1; got low={low!r} high={high!r}."
                ),
            )
        )
    else:
        rationale = _norm(fc.get("yes_rate_rationale"))
        if (float(low), float(high)) == TEMPLATE_DEFAULT_BAND and not rationale:
            findings.append(
                BriefLintFinding(
                    code="band_template_default",
                    severity="error",
                    message=(
                        "Yes-rate band equals the historical template default "
                        f"{TEMPLATE_DEFAULT_BAND} with no yes_rate_rationale — "
                        "indistinguishable from the template riding along."
                    ),
                )
            )

    # --- Hedge language inside pattern strings -----------------------------
    for field_name in _PATTERN_ARRAY_FIELDS:
        for pattern in _string_items(fc.get(field_name)):
            lowered = pattern.lower()
            hits = [phrase for phrase in HEDGE_PHRASES if phrase in lowered]
            if hits:
                findings.append(
                    BriefLintFinding(
                        code="hedge_language",
                        severity="error",
                        message=(
                            f"Disposition language {hits[0]!r} inside "
                            f"facial_calibration.{field_name}: {pattern[:90]!r}. "
                            "Patterns describe; the evaluation template decides."
                        ),
                    )
                )

    # --- Save-directional hedges in prose fields that render into the judge --
    depth = data.get("depth_distinction")
    depth = depth if isinstance(depth, dict) else {}
    prose_fields: list[tuple[str, str]] = [
        (f"depth_distinction.{key}", str(depth.get(key, "") or ""))
        for key in ("builder_definition", "user_definition", "edge_case_guidance")
    ]
    prose_fields.append(
        ("minimum_bar_description", str(data.get("minimum_bar_description", "") or ""))
    )
    for idx, nf in enumerate(data.get("non_fit_patterns") or []):
        if isinstance(nf, dict):
            prose_fields.append(
                (f"non_fit_patterns[{idx}].description", str(nf.get("description", "") or ""))
            )
            prose_fields.append(
                (f"non_fit_patterns[{idx}].why_not", str(nf.get("why_not", "") or ""))
            )
    for area_idx, area in enumerate(data.get("capability_areas") or []):
        if isinstance(area, dict):
            prose_fields.append(
                (
                    f"capability_areas[{area_idx}].description",
                    str(area.get("description", "") or ""),
                )
            )
    for label, text in prose_fields:
        lowered = text.lower()
        hits = [phrase for phrase in PROSE_SAVE_HEDGES if phrase in lowered]
        if hits:
            findings.append(
                BriefLintFinding(
                    code="hedge_language",
                    severity="error",
                    message=(
                        f"Save-directional disposition {hits[0]!r} inside {label}: "
                        f"{text[:90]!r}. Uncertainty never favors save (high-bar "
                        "doctrine); the evaluation template owns dispositions."
                    ),
                )
            )

    # --- Hiring company vs employer tiers ---------------------------------
    hiring_company = _norm(data.get("hiring_company"))
    generated_blacklist = [_norm(b) for b in _string_items(data.get("employer_blacklist"))]
    protected = {name for name in [hiring_company, *generated_blacklist] if name}
    protected |= {_norm(b) for b in seed_blacklist if _norm(b)}

    if hiring_company and hiring_company not in generated_blacklist:
        findings.append(
            BriefLintFinding(
                code="hiring_company_not_blacklisted",
                severity="error",
                message=(
                    f"hiring_company {hiring_company!r} is not in employer_blacklist — "
                    "the one employer this search never sources from."
                ),
            )
        )

    def _token_overlap(name: str, pattern_norm: str) -> bool:
        """Word-boundary containment in either direction — a short employer
        name ("Ramp") must not spuriously match an unrelated tier pattern
        ("Rampart AI") the way raw substring containment did (correctness
        lens, Wave 1)."""
        if not name or not pattern_norm:
            return False
        return bool(
            re.search(rf"\b{re.escape(name)}\b", pattern_norm)
            or re.search(rf"\b{re.escape(pattern_norm)}\b", name)
        )

    for rule in data.get("employer_signal_rules") or []:
        if not isinstance(rule, dict):
            continue
        tier = str(rule.get("tier", "") or "")
        for pattern in _string_items(rule.get("employer_patterns")):
            pattern_norm = _norm(pattern)
            for name in protected:
                if _token_overlap(name, pattern_norm):
                    findings.append(
                        BriefLintFinding(
                            code="employer_conflict",
                            severity="error",
                            message=(
                                f"Blacklisted/hiring employer {name!r} appears in the "
                                f"{tier!r} tier as pattern {pattern!r} — target/blacklist "
                                "confusion."
                            ),
                        )
                    )

    # --- Geography shape ------------------------------------------------------
    # Legitimate shapes: absent, a plain string, or the structured object
    # {"facet_candidates": [str, ...], "rationale": str} (empty candidates OK —
    # it means "JD states no geography"). Anything else would either stringify
    # into a garbage Location facet or silently drop the operator's intent, so
    # it blocks (Codex review, Wave 1).
    geography = data.get("geography")
    if geography is not None and not isinstance(geography, str):
        candidates = geography.get("facet_candidates") if isinstance(geography, dict) else None
        shape_ok = isinstance(geography, dict) and (
            candidates is None
            or (
                isinstance(candidates, list)
                and all(isinstance(v, str) for v in candidates)
            )
        )
        if not shape_ok:
            findings.append(
                BriefLintFinding(
                    code="geography_invalid",
                    severity="error",
                    message=(
                        "geography must be a string or an object with "
                        f"facet_candidates as a list of strings; got {geography!r:.90}."
                    ),
                )
            )

    # --- Lane hints ---------------------------------------------------------
    lane_hints = data.get("domain_lane_hints")
    named_lanes = []
    for hint in lane_hints or []:
        if not isinstance(hint, dict) or not str(hint.get("lane", "") or "").strip():
            continue
        patterns = hint.get("patterns")
        # A bare-string patterns value would explode into single characters
        # downstream and make this lane match everything — malformed hints
        # BLOCK; missing hints only warn (Codex review, Wave 1).
        if patterns is not None and (
            not isinstance(patterns, list)
            or any(not isinstance(p, str) or not p.strip() for p in patterns)
        ):
            findings.append(
                BriefLintFinding(
                    code="lane_hint_patterns_invalid",
                    severity="error",
                    message=(
                        f"domain_lane_hints[{hint.get('lane')!r}].patterns must be a "
                        f"list of non-empty strings; got {patterns!r:.60}."
                    ),
                )
            )
            continue
        named_lanes.append(hint)
    if not named_lanes:
        findings.append(
            BriefLintFinding(
                code="missing_domain_lane_hints",
                severity="warning",
                message=(
                    "No domain_lane_hints — lane-level learning will collapse to "
                    "'general' for this brief."
                ),
            )
        )

    # --- Experience band sanity (RC4, 2026-07-04) -------------------------
    maximum_years = data.get("maximum_years_experience")
    if maximum_years is not None:
        minimum_years = data.get("minimum_years_experience")
        bad_type = not isinstance(maximum_years, int) or isinstance(maximum_years, bool)
        if bad_type or maximum_years <= 0 or (
            isinstance(minimum_years, int)
            and not isinstance(minimum_years, bool)
            and minimum_years > maximum_years
        ):
            findings.append(
                BriefLintFinding(
                    code="experience_band_invalid",
                    severity="error",
                    message=(
                        f"maximum_years_experience must be a positive int >= "
                        f"minimum_years_experience or null; got max={maximum_years!r}, "
                        f"min={minimum_years!r}."
                    ),
                )
            )

    # --- Opening mirrors (RC3, 2026-07-04) --------------------------------
    # The deterministic opening sort (_opening_priority) ranks strings from
    # these fields; with none present it runs neutral — blind — which is how
    # JD-only briefs lost their opening discipline. Malformed = error;
    # absent-en-bloc = warning (mirrors the missing_domain_lane_hints
    # posture: degraded, not dangerous).
    any_mirror = False
    for field_name in _OPENING_MIRROR_FIELDS:
        value = data.get(field_name)
        if value in (None, []):
            continue
        if not isinstance(value, list) or any(
            not isinstance(p, str) or not p.strip() for p in value
        ):
            findings.append(
                BriefLintFinding(
                    code="opening_mirror_invalid",
                    severity="error",
                    message=(
                        f"{field_name} must be a list of non-empty strings; "
                        f"got {value!r:.60}."
                    ),
                )
            )
            continue
        any_mirror = True
    if not any_mirror:
        findings.append(
            BriefLintFinding(
                code="missing_opening_mirrors",
                severity="warning",
                message=(
                    "No canonical/edge-case opening mirrors — the deterministic "
                    "opening sort runs neutral (blind) for this brief."
                ),
            )
        )
    hiring_company = str(data.get("hiring_company") or "").strip().lower()
    if hiring_company:
        for company in data.get("canonical_company_patterns") or []:
            if (
                isinstance(company, str)
                and company.strip()
                and hiring_company in company.strip().lower()
            ):
                findings.append(
                    BriefLintFinding(
                        code="hiring_company_in_canonical_companies",
                        severity="error",
                        message=(
                            f"Hiring company {data.get('hiring_company')!r} appears in "
                            f"canonical_company_patterns ({company!r}) — the one employer "
                            "this search never sources from cannot anchor the canonical pool."
                        ),
                    )
                )

    # --- Example compounds (the worked search levers) ---------------------
    # linkedin/strategy.py renders brief.example_compounds verbatim into the
    # formation system prompt ("Example rendered compounds:"); a JD-only brief
    # that carries none leaves the strategy model composing generically. The
    # loader (shared/brief_loader.py:400) hydrates each entry via ec.get(...)
    # with no isinstance guard, so a list of bare strings crashes _load_v2_brief
    # — malformed entries BLOCK. A boolean that fails lint_boolean or copies the
    # illustrative placeholder shape verbatim BLOCKS; a boolean that shares no
    # quoted term with the brief's own vocabulary is a legitimate paraphrase →
    # warning; absent-en-bloc mirrors missing_domain_lane_hints (degraded, not
    # dangerous) → warning.
    example_compounds = data.get("example_compounds")
    if not (isinstance(example_compounds, list) and example_compounds):
        findings.append(
            BriefLintFinding(
                code="missing_example_compounds",
                severity="warning",
                message=(
                    "No example_compounds — the formation prompt renders no worked "
                    "search levers and the strategy model composes generically."
                ),
            )
        )
    else:
        from linkedin.boolean_compiler import lint_boolean
        from shared.preflight_v2 import ILLUSTRATIVE_EXAMPLE_COMPOUNDS

        illustrative = {
            _normalize_boolean(ec.get("boolean"))
            for ec in ILLUSTRATIVE_EXAMPLE_COMPOUNDS
            if isinstance(ec, dict)
        }
        brief_terms = _example_vocabulary(data)
        for idx, entry in enumerate(example_compounds):
            if not isinstance(entry, dict):
                findings.append(
                    BriefLintFinding(
                        code="example_compound_invalid",
                        severity="error",
                        message=(
                            f"example_compounds[{idx}] is not an object ({entry!r:.60}); "
                            "the loader hydrates each entry via ec.get(...) and a bare "
                            "string crashes _load_v2_brief."
                        ),
                    )
                )
                continue
            boolean = str(entry.get("boolean", "") or "").strip()
            if not boolean:
                findings.append(
                    BriefLintFinding(
                        code="example_compound_invalid",
                        severity="error",
                        message=f"example_compounds[{idx}] has a missing or empty boolean.",
                    )
                )
                continue
            report = lint_boolean(boolean)
            if report.has_error:
                error_codes = ", ".join(
                    sorted({f.code for f in report.findings if f.severity == "error"})
                )
                findings.append(
                    BriefLintFinding(
                        code="example_compound_boolean_error",
                        severity="error",
                        message=(
                            f"example_compounds[{idx}] boolean fails lint_boolean "
                            f"({error_codes}): {boolean[:90]!r}."
                        ),
                    )
                )
                continue
            if _normalize_boolean(boolean) in illustrative:
                findings.append(
                    BriefLintFinding(
                        code="example_compound_copied",
                        severity="error",
                        message=(
                            f"example_compounds[{idx}] copies an illustrative placeholder "
                            f"shape verbatim instead of authoring from this brief: "
                            f"{boolean[:90]!r}."
                        ),
                    )
                )
                continue
            if brief_terms and not (_quoted_terms_normalized(boolean) & brief_terms):
                findings.append(
                    BriefLintFinding(
                        code="example_compound_off_vocabulary",
                        severity="warning",
                        message=(
                            f"example_compounds[{idx}] shares no quoted term with the "
                            "brief's capability key_terms or canonical patterns: "
                            f"{boolean[:90]!r}."
                        ),
                    )
                )

    severity_rank = {"error": 0, "warning": 1}
    findings.sort(key=lambda f: (severity_rank.get(f.severity, 2), f.code))
    return findings


def blocking_findings(findings: Iterable[BriefLintFinding]) -> list[BriefLintFinding]:
    return [f for f in findings if f.severity == "error"]


def format_findings(findings: Iterable[BriefLintFinding]) -> str:
    return "\n".join(
        f"  [preflight-lint] {f.severity.upper()}: {f.code} — {f.message}"
        for f in findings
    )
