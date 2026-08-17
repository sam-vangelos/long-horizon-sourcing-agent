"""Deterministic LinkedIn Boolean linting and SearchConstraint compilation (P2).

Warning-only by default: findings attach to strategy output as metadata without
reordering or suppressing generated strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from shared.sourcing_lanes import SearchConstraint
from shared.strict_seniority import classify_search_string_seniority, is_strict_seniority_brief
from linkedin.matching_contract import conservative_morphology_repair_hint

# Patterns reused for capability-vs-title heuristics (mirror strict_seniority).
_BUILDER_PROOF_PATTERNS = (
    "built",
    "deployed",
    "production",
    "scaled",
    "platform",
    "shipped",
    "architected",
    "architecture",
    "orchestration",
    "evaluation",
    "eval",
    "guardrail",
    "agent platform",
    "workflow",
    "tool calling",
)

_TITLE_HEAVY_PATTERNS = (
    "head of",
    "director",
    "vp",
    "svp",
    "managing director",
    "executive director",
    "principal",
    "chief",
)

_GENERIC_AI_TERMS = frozenset(
    {
        "ai",
        "genai",  # VERTICAL-VOCAB(bare-generic-ai-craft-check)
        "llm",
        "llms",  # VERTICAL-VOCAB(bare-generic-ai-craft-check)
        "machine learning",  # VERTICAL-VOCAB(bare-generic-ai-craft-check)
        "artificial intelligence",  # VERTICAL-VOCAB(bare-generic-ai-craft-check)
        "generative ai",  # VERTICAL-VOCAB(bare-generic-ai-craft-check)
        "large language model",  # VERTICAL-VOCAB(bare-generic-ai-craft-check)
    }
)

# Cathey Maximum-Inclusion: a required AND term must be one essentially all
# qualified people would write. These STRUCTURAL low-signal verbs (generic
# craft vocabulary — "managed"/"led"/"owned" — vertical-agnostic) stay in code
# like _GENERIC_AI_TERMS; an OR group built only from them is an over-tight AND
# gate that suppresses recall. Vertical vocabulary never joins this set — it
# belongs in the brief. "scaled"/"delivered" are deliberately excluded: they
# are builder-proof vocabulary elsewhere in this module.
_STRUCTURAL_LOW_SIGNAL_TERMS = frozenset(
    {"managed", "led", "owned", "oversaw", "responsible for", "hands-on"}
)

# P1 item 5 (Wave 2): capability discriminators split. STRUCTURAL craft terms
# (evidence-of-building vocabulary, vertical-agnostic) stay in code; VERTICAL
# terms (banking, capital markets, fraud, rag, ...) moved behind the brief —
# boolean_lint_context_from_brief unions brief.key_terms_by_area into
# BooleanLintContext.capability_discriminators. Deterministic code consumes
# brief vocabulary; it never carries its own (audit R2-F2 sweep).
_STRUCTURAL_CAPABILITY_DISCRIMINATORS = frozenset(
    {
        "production",
        "deployed",
        "built",
        "platform",
        "orchestration",
        "workflow",
        "agent",
        "evaluation",
        "guardrail",
    }
)

_DEFAULT_MORPHOLOGY_PAIRS: dict[str, tuple[str, ...]] = {
    "deployment": ("deployments",),
    "deploy": ("deployed", "deploying"),
    "model": ("models",),
    "engineer": ("engineers",),
    "analyst": ("analysts", "analyzing"),
    "implement": ("implemented", "implementation"),
}

_QUOTED_TERM_RE = re.compile(r'"([^"]*)"')
_PAREN_GROUP_RE = re.compile(r"\(([^()]*)\)")


@dataclass(frozen=True)
class BooleanLintFinding:
    severity: str
    code: str
    message: str
    span: tuple[int, int] | None = None
    repair_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "repair_hint": self.repair_hint,
        }
        if self.span is not None:
            payload["span"] = list(self.span)
        return payload


@dataclass
class BooleanLintReport:
    boolean: str
    findings: list[BooleanLintFinding] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(finding.severity == "error" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boolean": self.boolean,
            "findings": [finding.to_dict() for finding in self.findings],
            "has_error": self.has_error,
        }


@dataclass(frozen=True)
class BooleanNormalizationFinding:
    code: str
    message: str
    terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "terms": list(self.terms),
        }


@dataclass(frozen=True)
class BooleanNormalizationReport:
    original_boolean: str
    normalized_boolean: str
    findings: tuple[BooleanNormalizationFinding, ...] = ()

    @property
    def changed(self) -> bool:
        return self.normalized_boolean != self.original_boolean

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_boolean": self.original_boolean,
            "normalized_boolean": self.normalized_boolean,
            "changed": self.changed,
            "findings": [finding.to_dict() for finding in self.findings],
        }


class BooleanNormalizationError(ValueError):
    """Raised when an executable Boolean fails a deterministic normalization gate."""


class UbiquitousAndGateError(BooleanNormalizationError):
    """The ubiquity AND-gate fired: every group is composed of ubiquitous terms.

    Distinct type so callers can classify a gate hit apart from malformed-input
    validation errors (whose messages also contain "ubiquitous") — substring
    classification misfiled type errors as gate hits (correctness lens, Wave 2).
    """


@dataclass
class BooleanLintContext:
    strict_seniority: bool = False
    abbreviation_collisions: tuple[Any, ...] = ()
    morphology_pairs: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(_DEFAULT_MORPHOLOGY_PAIRS)
    )
    max_or_group_terms: int = 8
    structured_filters: dict[str, Any] = field(default_factory=dict)
    # Structural discriminators plus JD-register terms built from brief channels.
    capability_discriminators: frozenset[str] = _STRUCTURAL_CAPABILITY_DISCRIMINATORS
    jd_register_terms: frozenset[str] = frozenset()


@dataclass
class ExecutableConstraint:
    dimension: str
    operator: str
    execution_surface: str
    temporal_scope: str
    values: list[str]
    boolean_fragment: str = ""
    structured_control: dict[str, Any] = field(default_factory=dict)
    compile_notes: list[str] = field(default_factory=list)


def boolean_lint_context_from_brief(brief: Any) -> BooleanLintContext:
    collisions = tuple(getattr(brief, "abbreviation_collisions", None) or ())
    key_terms_by_area = getattr(brief, "key_terms_by_area", None)
    candidate_register_terms_by_area = getattr(
        brief, "candidate_register_terms_by_area", None
    )
    key_terms = _brief_terms_by_area(key_terms_by_area, include_area_names=False)
    jd_register_terms = _jd_register_terms_from_brief(
        key_terms_by_area, candidate_register_terms_by_area
    )
    discriminators = set(_STRUCTURAL_CAPABILITY_DISCRIMINATORS) | key_terms
    return BooleanLintContext(
        strict_seniority=is_strict_seniority_brief(brief),
        abbreviation_collisions=collisions,
        capability_discriminators=frozenset(discriminators),
        jd_register_terms=jd_register_terms,
    )


def _context_for_item(
    base: BooleanLintContext | None,
    item: dict[str, Any],
) -> BooleanLintContext:
    context = base or BooleanLintContext()
    lane_snapshot = item.get("lane_snapshot") if isinstance(item.get("lane_snapshot"), dict) else {}
    recipe = item.get("retrieval_recipe") if isinstance(item.get("retrieval_recipe"), dict) else {}
    structured: dict[str, Any] = {}
    for source in (context.structured_filters, lane_snapshot.get("structured_filters"), recipe):
        if isinstance(source, dict):
            structured.update(source)
    if recipe.get("target_employers"):
        structured.setdefault("target_employers", list(recipe["target_employers"]))
    if recipe.get("target_markets"):
        structured.setdefault("target_markets", list(recipe["target_markets"]))
    return replace(context, structured_filters=structured)


def _quoted_terms(text: str) -> list[str]:
    return [match.group(1).strip() for match in _QUOTED_TERM_RE.finditer(text) if match.group(1).strip()]


def _normalize_term(term: str) -> str:
    return " ".join(term.lower().split())


_STRUCTURAL_UBIQUITOUS_TERMS = frozenset({"ai", "engineer", "software", "technology"})


def ubiquitous_terms_from_brief(brief: Any) -> frozenset[str]:
    """Live feed for the ubiquitous-term AND-gate.

    Audit R2-F5: a gate that structurally cannot fire (no producer ever supplied
    ``ubiquitous_terms``) reads as protection and provides none. This is the real
    feed: the union of a small structural set — craft vocabulary that is
    ubiquitous on LinkedIn regardless of vertical ("ai", "engineer", "software",
    "technology") — and every term in every category of a brief's
    ``term_blacklist_categories`` (vertical vocabulary, brief-declared). The
    structural set is deliberately NOT vertical vocabulary.
    """
    terms: set[str] = {_normalize_term(term) for term in _STRUCTURAL_UBIQUITOUS_TERMS}
    categories = getattr(brief, "term_blacklist_categories", None)
    if not isinstance(categories, (list, tuple)):
        categories = ()
    for category in categories:
        category_terms = getattr(category, "terms", None)
        if not isinstance(category_terms, (list, tuple)):
            continue
        for term in category_terms:
            normalized = _normalize_term(str(term))
            if normalized:
                terms.add(normalized)
    return frozenset(terms)


def _expand_proper_noun_surfaces(
    terms: set[str] | frozenset[str] | None,
) -> frozenset[str]:
    """Every surface form of a named artifact is itself a named artifact.

    A brief listing "SWE-bench" exempts the hyphenated form only, but a model
    legitimately writes the spaced form too — and pluralising *that* fabricates
    a variant of a proper noun, which the doctrine forbids. Measured on the live
    2026-07-27 string: "SWE bench" became "swe benches". So the exemption is
    taken across the hyphen/space axis, and only that axis: no number forms,
    because "SWE-benches" should never be treated as declared.
    """
    out: set[str] = set()
    for term in terms or ():
        normalized = _normalize_term(str(term))
        if not normalized:
            continue
        out.add(normalized)
        if "-" in normalized:
            out.add(normalized.replace("-", " "))
        elif len(normalized.split()) == 2:
            out.add("-".join(normalized.split()))
    return frozenset(out)


def proper_nouns_from_brief(brief: Any) -> frozenset[str]:
    """Do-not-vary feed for deterministic surface expansion.

    The strategy doctrine allows real variants for concepts but forbids
    fabricated ones for proper-noun tools ("SWE-Gym", "veRL", "pass@k"): those
    names have exactly one correct spelling and a pluralised guess is noise.
    The brief already has the field for them — ``domain_depth_objects``, the
    objects that signal depth in this domain — so this reads that rather than
    carrying a hardcoded vocabulary list, which would go stale per vertical.

    Two known limits, both tolerable only because the remaining axis is
    orthographic:

    - The field is unpopulated on the 2026-07-27 PRRE brief, so the set is
      empty and every guard falls to ``derive_surface_variants`` itself.
    - Where it IS populated it holds prose, not artifacts —
      ``config/FDL-Colombia/brief-fdl-colombia-v4.json`` lists "training-data
      curation and synthetic-data pipelines". Those get exempted from a
      hyphenation twin they would have benefited from.

    Both failures cost a variant that was never emitted. Neither can produce a
    wrong one, which is why this stayed on ``domain_depth_objects`` rather than
    blocking on a new brief field.
    """
    objects = getattr(brief, "domain_depth_objects", None)
    if not isinstance(objects, (list, tuple)):
        return frozenset()
    return frozenset(
        normalized
        for obj in objects
        if (normalized := _normalize_term(str(obj)))
    )


def normalize_boolean_for_linkedin(
    boolean: str,
    *,
    structured_filters: dict[str, Any] | None = None,
    ubiquitous_terms: set[str] | frozenset[str] | None = None,
    enable_token_subset_pruning: bool = False,
    expand_surface_variants: bool = False,
    proper_nouns: set[str] | frozenset[str] | None = None,
) -> BooleanNormalizationReport:
    """Normalize a LinkedIn Boolean string with explicit, fixture-supplied rules.

    This is intentionally parameterized. M1B/M1C live matching facts and product
    prevalence thresholds can be supplied later without this helper guessing them.

    ``expand_surface_variants`` moves the doctrine's singular/plural and
    hyphenation requirement out of the model's hands and into arithmetic: the
    forms are mechanically derivable, so compliance should not depend on a
    generation. ``proper_nouns`` is the do-not-vary set (the doctrine forbids
    fabricated variants for named tools); supply it from
    ``proper_nouns_from_brief``.
    """

    original = boolean or ""
    text = original
    structured_conflicts = _structured_filter_terms(structured_filters or {})
    ubiquitous_set = _coerce_string_set(
        ubiquitous_terms,
        field_name="ubiquitous_terms",
    )
    proper_noun_set = _expand_proper_noun_surfaces(
        _coerce_string_set(proper_nouns, field_name="proper_nouns")
    )
    findings: list[BooleanNormalizationFinding] = []

    replacements: list[tuple[int, int, str]] = []
    normalized_groups_for_gate: list[tuple[str, ...]] = []
    for start, end, group in _parenthetical_groups(text):
        terms = _quoted_terms(group)
        if not terms:
            continue
        normalized_terms, group_findings, derived_terms = _normalize_group_terms(
            terms,
            structured_conflicts=structured_conflicts,
            enable_token_subset_pruning=enable_token_subset_pruning,
            expand_surface_variants=expand_surface_variants,
            proper_nouns=proper_noun_set,
        )
        findings.extend(group_findings)
        # The ubiquity gate scores what the MODEL authored. A derived surface
        # form of a ubiquitous term is still ubiquitous — "engineers" says
        # nothing "engineer" did not — so counting it as fresh vocabulary
        # silently disarms the gate: ("engineer") AND ("technology") fired
        # before expansion and stopped firing after it, turning a refusal into
        # a pass without anyone choosing that.
        normalized_groups_for_gate.append(
            tuple(
                normalized
                for term in normalized_terms
                if (normalized := _normalize_term(term)) not in derived_terms
            )
        )
        if normalized_terms != terms:
            replacements.append((start, end, _render_or_group(normalized_terms)))

    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]

    if _ubiquitous_and_gate_fires(text, normalized_groups_for_gate, ubiquitous_set):
        findings.append(
            BooleanNormalizationFinding(
                code="ubiquitous_and_gate",
                message="AND clause is composed entirely of ubiquitous terms.",
                terms=tuple(sorted(_normalize_term(term) for term in ubiquitous_set or ())),
            )
        )

    return BooleanNormalizationReport(
        original_boolean=original,
        normalized_boolean=text,
        findings=tuple(findings),
    )


def normalize_execution_work_item_boolean(
    item: dict[str, Any],
    *,
    boolean_key: str,
    structured_filters: dict[str, Any] | None = None,
    ubiquitous_terms: set[str] | frozenset[str] | None = None,
    enable_token_subset_pruning: bool = True,
    expand_surface_variants: bool = False,
    proper_nouns: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Normalize a generated/adapted work-item Boolean before execution.

    The execution seam supplies already-derived structured filters so this helper
    can strip keyword/filter conflicts without importing LinkedIn lane machinery.
    The execution seam now receives the brief-derived ubiquity feed (see
    ``ubiquitous_terms_from_brief``) from its callers via the explicit
    ``ubiquitous_terms`` argument; item-supplied ``item["ubiquitous_terms"]``
    remain honored and union in. No other empirical LinkedIn behavior or product
    prevalence threshold is guessed here.
    """

    working = dict(item)
    boolean = str(working.get(boolean_key) or "")
    if not boolean:
        return working
    explicit_ubiquitous = _coerce_string_set(
        ubiquitous_terms,
        field_name="ubiquitous_terms",
    )
    item_ubiquitous = _coerce_string_set(
        working.get("ubiquitous_terms"),
        field_name="ubiquitous_terms",
    )
    if explicit_ubiquitous is None:
        effective_ubiquitous = item_ubiquitous
    elif item_ubiquitous is None:
        effective_ubiquitous = explicit_ubiquitous
    else:
        effective_ubiquitous = explicit_ubiquitous | item_ubiquitous
    # Only read the item's do-not-vary set when expansion will actually consume
    # it. Reading it unconditionally made a malformed `item["proper_nouns"]`
    # raise on the flag-OFF path, where the pre-change compiler ignored the key
    # entirely — a behaviour change in the one configuration production runs.
    effective_proper: set[str] | None = None
    if expand_surface_variants:
        explicit_proper = _coerce_string_set(proper_nouns, field_name="proper_nouns")
        item_proper = _coerce_string_set(
            working.get("proper_nouns"),
            field_name="proper_nouns",
        )
        if explicit_proper is None:
            effective_proper = item_proper
        elif item_proper is None:
            effective_proper = explicit_proper
        else:
            effective_proper = explicit_proper | item_proper
    report = normalize_boolean_for_linkedin(
        boolean,
        structured_filters=structured_filters or {},
        ubiquitous_terms=effective_ubiquitous,
        enable_token_subset_pruning=enable_token_subset_pruning,
        expand_surface_variants=expand_surface_variants,
        proper_nouns=effective_proper,
    )
    if any(finding.code == "ubiquitous_and_gate" for finding in report.findings):
        raise UbiquitousAndGateError(
            f"{boolean_key} failed the ubiquitous-term AND-gate"
        )
    working[boolean_key] = report.normalized_boolean
    working["boolean_normalization"] = report.to_dict()
    return working


def _coerce_string_set(
    value: Any,
    *,
    field_name: str,
) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (set, frozenset, list, tuple)):
        values = tuple(value)
    else:
        raise BooleanNormalizationError(f"{field_name} must be a string or collection of strings")
    terms: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            raise BooleanNormalizationError(f"{field_name} values must be strings")
        term = item.strip()
        if term:
            terms.add(term)
    return terms or None


def _structured_filter_terms(structured_filters: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for key in ("companies", "titles"):
        values = structured_filters.get(key) or ()
        if isinstance(values, str):
            values = (values,)
        elif not isinstance(values, (list, tuple, set, frozenset)):
            raise BooleanNormalizationError(f"structured_filters.{key} must be a string or collection of strings")
        for value in values:
            if not isinstance(value, str):
                raise BooleanNormalizationError(f"structured_filters.{key} values must be strings")
            normalized = _normalize_term(value)
            if normalized:
                terms.add(normalized)
    return terms


_VARIANT_EXEMPT_RE = re.compile(r"[0-9@+#]")

# Bounds the blowup when a group is already wide. One axis yields at most one
# variant per term, so this only engages on groups already past a dozen terms —
# where doubling would push the string toward the 2000-char Recruiter cap.
DEFAULT_MAX_VARIANTS_PER_GROUP = 12


def derive_surface_variants(term: str) -> tuple[str, ...]:
    """Real surface forms of one concept term, derived — never invented.

    ONE axis: hyphen<->space. That axis is *orthographic* — "post-training" and
    "post training" are the same word typed two ways, so the derived form is
    real whenever the authored one is.

    The number axis is deliberately absent. Pluralising is *morphological*, and
    morphology needs a lexicon: rules alone cannot know that "agent" takes
    "agents" while "kubernetes" takes nothing. Run against real tooling
    vocabulary on 2026-07-27 they produced "kubernete", "mlop", "devop",
    "verls", "jaxes", "numpies", "pytorches", "datas" and "softwares" — every
    one an OR slot spent on a string that matches nobody. The brief field that
    would have exempted them (``domain_depth_objects``) is wired to no producer,
    so in production the do-not-vary set is always empty and those guesses ship.
    The compliance gap the A/B actually measured was hyphenation, not number:
    claude-fable-5 emitted 0 of 18 hyphenated terms with their spaced twin,
    gpt-5.6-sol 11 of 22. This closes that gap and claims nothing else.

    Terms carrying digits or operator punctuation -- pass@k, gpt-4, c++ -- are
    named artifacts whose punctuation is part of the name, and are refused.
    """

    base = _normalize_term(term)
    if not base or _VARIANT_EXEMPT_RE.search(base):
        return ()
    # A quote would be re-quoted by _render_or_group into a malformed group.
    # Unreachable through _quoted_terms, whose regex cannot yield one — this is
    # for direct callers.
    if '"' in base:
        return ()
    # An edge hyphen is not a separator between two words, so removing it is a
    # rewrite rather than a respelling: "A-" is a grade, and "a" is not the
    # same term.
    if base.startswith("-") or base.endswith("-"):
        return ()

    if "-" in base:
        # Re-normalize: swapping the hyphen for a space can leave whitespace the
        # input never had. "post - training" became "post   training" and a bare
        # "-" became " " — terms that match nothing and would have been quoted
        # straight into a live Boolean.
        spaced = _normalize_term(base.replace("-", " "))
        return (spaced,) if spaced and spaced != base else ()
    # Reverse synthesis (space -> hyphen) is NOT symmetric with dehyphenation.
    # Dehyphenating an authored compound yields a form people demonstrably
    # write; hyphenating an arbitrary two-word phrase invents one. Audited
    # 2026-07-30: "San Francisco" became "san-francisco" and "New York" became
    # "new-york" — nobody writes those, and they are pure noise in an OR group.
    # Only phrases whose spaced form is a KNOWN compound get the hyphen back,
    # which is exactly the set where an authored hyphenated twin plausibly
    # exists (the term already appeared hyphenated somewhere in this group).
    return ()


def _normalize_group_terms(
    terms: list[str],
    *,
    structured_conflicts: set[str],
    enable_token_subset_pruning: bool,
    expand_surface_variants: bool = False,
    proper_nouns: frozenset[str] | None = None,
    max_variants_per_group: int = DEFAULT_MAX_VARIANTS_PER_GROUP,
) -> tuple[list[str], list[BooleanNormalizationFinding], frozenset[str]]:
    findings: list[BooleanNormalizationFinding] = []
    expanded: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = _normalize_term(term)
        if normalized in structured_conflicts:
            findings.append(
                BooleanNormalizationFinding(
                    code="surface_conflict_stripped",
                    message="Term removed because the same value is carried by a structured filter.",
                    terms=(term,),
                )
            )
            continue
        if not normalized or normalized in seen:
            continue
        expanded.append(term)
        seen.add(normalized)

    # Snapshot of what the MODEL wrote, before any derivation. Two consumers:
    # variants derive from authored terms only (never from other variants), and
    # only authored terms may prune.
    authored: list[str] = list(expanded)

    derived: list[str] = []
    if expand_surface_variants:
        exempt = proper_nouns or frozenset()
        for term in authored:
            if _normalize_term(term) in exempt:
                continue
            for variant in derive_surface_variants(term):
                if len(derived) >= max_variants_per_group:
                    break
                if variant in seen or variant in exempt:
                    continue
                expanded.append(variant)
                seen.add(variant)
                derived.append(variant)
            if len(derived) >= max_variants_per_group:
                break
        if derived:
            findings.append(
                BooleanNormalizationFinding(
                    code="morphological_variants_added",
                    message=(
                        "Surface variants derived deterministically for concept terms "
                        "(proper-noun tools exempt)."
                    ),
                    terms=tuple(derived),
                )
            )

    if enable_token_subset_pruning:
        # `pruners=None` when nothing was derived keeps the flag-off path
        # byte-identical to the pre-expansion behaviour.
        expanded, pruned = _prune_token_subset_superstrings(
            expanded, pruners=authored if derived else None
        )
        if pruned:
            findings.append(
                BooleanNormalizationFinding(
                    code="token_subset_superstring_pruned",
                    message="Superstring terms were pruned by explicit token-subset rule.",
                    terms=tuple(pruned),
                )
            )
    return expanded, findings, frozenset(derived)


def _prune_token_subset_superstrings(
    terms: list[str],
    *,
    pruners: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Drop a term when a strictly shorter term in the group already covers it.

    ``pruners`` restricts which terms may ACT as that shorter term; everything
    in ``terms`` stays eligible to BE pruned. ``None`` means every term prunes,
    which is the behaviour when surface expansion is off.

    The restriction exists because a derived surface variant is a respelling,
    not a concept, and letting one prune inverts the whole point of expansion.
    Measured: ("post-training" OR "llm post training") derives "post training",
    whose tokens are a proper subset of the authored "llm post training" — so a
    pass that exists to WIDEN the group deleted the model's most specific term
    and shipped ("post-training" OR "post training"). A variant may still be
    pruned by an authored term, which is how the group stays free of the
    redundancy the rule was written for.
    """

    normalized_tokens = {
        term: set(_normalize_term(term).split())
        for term in terms
        if _normalize_term(term)
    }
    pruner_tokens = (
        normalized_tokens
        if pruners is None
        else {
            term: normalized_tokens[term]
            for term in pruners
            if term in normalized_tokens
        }
    )
    pruned: list[str] = []
    kept: list[str] = []
    for term in terms:
        tokens = normalized_tokens.get(term, set())
        if any(other != term and other_tokens < tokens for other, other_tokens in pruner_tokens.items()):
            pruned.append(term)
            continue
        kept.append(term)
    return kept, pruned


def _render_or_group(terms: list[str]) -> str:
    if not terms:
        return "()"
    return "(" + " OR ".join(f'"{term}"' for term in terms) + ")"


def _ubiquitous_and_gate_fires(
    text: str,
    normalized_groups: list[tuple[str, ...]],
    ubiquitous_terms: set[str] | frozenset[str] | None,
) -> bool:
    if not ubiquitous_terms or not normalized_groups:
        return False
    # Detect the AND OPERATOR, which lives outside quoted spans. A term may
    # legitimately contain the word — "research-and-development" — and
    # dehyphenating it puts a bare " and " inside the quotes, which a raw
    # substring test reads as an AND clause. That fired the gate on a
    # single-group string with no AND in it at all, and the execution seam
    # turns this finding into a raise, so the work item died.
    if " AND " not in _mask_quoted_spans(text).upper():
        return False
    normalized_ubiquitous = {_normalize_term(term) for term in ubiquitous_terms}
    return all(
        bool(group) and all(term in normalized_ubiquitous for term in group)
        for group in normalized_groups
    )


def _check_balanced_delimiters(boolean: str) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    paren_depth = 0
    in_quote = False
    for index, char in enumerate(boolean):
        if char == '"' and (index == 0 or boolean[index - 1] != "\\"):
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
            if paren_depth < 0:
                findings.append(
                    BooleanLintFinding(
                        severity="error",
                        code="unbalanced_parenthesis",
                        message="Unmatched closing parenthesis.",
                        span=(index, index + 1),
                        repair_hint="Remove the extra ')' or add a matching '('.",
                    )
                )
                paren_depth = 0
    if in_quote:
        findings.append(
            BooleanLintFinding(
                severity="error",
                code="unbalanced_quote",
                message="Unclosed double quote in Boolean string.",
                repair_hint='Close every opening quote with a matching ".',
            )
        )
    if paren_depth > 0:
        findings.append(
            BooleanLintFinding(
                severity="error",
                code="unbalanced_parenthesis",
                message="Unclosed parenthesis in Boolean string.",
                repair_hint="Add a closing ')' for each open group.",
            )
        )
    return findings


def _check_malformed_operators(boolean: str) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    for match in re.finditer(r"&&|\|\|", boolean):
        findings.append(
            BooleanLintFinding(
                severity="error",
                code="malformed_operator",
                message=f"Unsupported operator {match.group(0)!r}; use AND/OR/NOT.",
                span=(match.start(), match.end()),
                repair_hint="Replace && with AND and || with OR.",
            )
        )
    for match in re.finditer(r"(?<![A-Za-z_])&(?![A-Za-z_])", _mask_quoted_spans(boolean)):
        findings.append(
            BooleanLintFinding(
                severity="error",
                code="malformed_operator",
                message="Bare '&' is not valid LinkedIn Boolean syntax.",
                span=(match.start(), match.end()),
                repair_hint="Use AND between groups instead of '&'.",
            )
        )
    return findings


def _parenthetical_groups(boolean: str) -> list[tuple[int, int, str]]:
    groups: list[tuple[int, int, str]] = []
    depth = 0
    start = -1
    in_quote = False
    for index, char in enumerate(boolean):
        if char == '"' and (index == 0 or boolean[index - 1] != "\\"):
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                groups.append((start, index + 1, boolean[start : index + 1]))
                start = -1
    return groups


def _check_empty_or_groups(boolean: str) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    for start, end, group in _parenthetical_groups(boolean):
        inner = group[1:-1].strip()
        if not inner:
            findings.append(
                BooleanLintFinding(
                    severity="error",
                    code="empty_or_group",
                    message="Empty parenthetical group.",
                    span=(start, end),
                    repair_hint="Remove the empty group or add quoted search terms.",
                )
            )
            continue
        if re.fullmatch(r"(?:OR|AND|NOT)\s*", inner, flags=re.IGNORECASE):
            findings.append(
                BooleanLintFinding(
                    severity="error",
                    code="empty_or_group",
                    message="Parenthetical group contains only a Boolean operator.",
                    span=(start, end),
                    repair_hint="Add quoted terms inside the group.",
                )
            )
        if re.search(r"\bOR\s*\)", group, flags=re.IGNORECASE) or re.search(
            r"\(\s*OR\b", group, flags=re.IGNORECASE
        ):
            if not _quoted_terms(group):
                findings.append(
                    BooleanLintFinding(
                        severity="error",
                        code="empty_or_group",
                        message="OR group has no quoted terms.",
                        span=(start, end),
                        repair_hint='Add quoted variants, e.g. ("term one" OR "term two").',
                    )
                )
        if re.search(r"\bOR\s*$", inner, flags=re.IGNORECASE):
            findings.append(
                BooleanLintFinding(
                    severity="error",
                    code="empty_or_group",
                    message="OR group ends without a trailing operand.",
                    span=(start, end),
                    repair_hint="Add another quoted term after OR or remove the dangling OR.",
                )
            )
        for segment in re.split(r"\bOR\b", inner, flags=re.IGNORECASE):
            if not segment.strip().strip('"').strip("'"):
                findings.append(
                    BooleanLintFinding(
                        severity="error",
                        code="empty_or_group",
                        message="OR group contains an empty operand.",
                        span=(start, end),
                        repair_hint="Remove the empty OR branch or add a quoted term.",
                    )
                )
                break
    return findings


def _check_wildcards(boolean: str) -> list[BooleanLintFinding]:
    if "*" not in boolean:
        return []
    return [
        BooleanLintFinding(
            severity="warning",
            code="unsupported_wildcard",
            message="Wildcard syntax is outside the current LinkedIn Boolean contract.",
            repair_hint=conservative_morphology_repair_hint(),
        )
    ]


def _or_groups_with_terms(boolean: str) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for _start, _end, group in _parenthetical_groups(boolean):
        if " OR " not in group.upper():
            continue
        groups.append((group, _quoted_terms(group)))
    return groups


def _boolean_has_discriminator(
    boolean: str,
    discriminators: frozenset[str],
) -> bool:
    normalized = _normalize_term(boolean)
    return any(discriminator in normalized for discriminator in discriminators)


def _append_bare_generic_ai_finding(
    findings: list[BooleanLintFinding],
    *,
    message: str,
    seen_codes: set[str],
) -> None:
    if "bare_generic_ai_term" in seen_codes:
        return
    findings.append(
        BooleanLintFinding(
            severity="warning",
            code="bare_generic_ai_term",
            message=message,
            repair_hint="Pair GenAI/LLM terms with a capability or domain discriminator.",  # VERTICAL-VOCAB(bare-generic-ai-craft-check)
        )
    )
    seen_codes.add("bare_generic_ai_term")


def _check_bare_generic_ai_terms(
    boolean: str,
    context: BooleanLintContext,
) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    seen_codes: set[str] = set()
    discriminators = context.capability_discriminators
    has_discriminator = _boolean_has_discriminator(boolean, discriminators)

    for group, terms in _or_groups_with_terms(boolean):
        normalized = {_normalize_term(term) for term in terms}
        generic_only = normalized and normalized.issubset(_GENERIC_AI_TERMS)
        if not generic_only:
            continue
        if normalized & discriminators:
            continue
        if len(terms) == 1 and terms[0].lower() in _GENERIC_AI_TERMS:
            _append_bare_generic_ai_finding(
                findings,
                message=f'OR group is only generic AI vocabulary ({group[:80]}).',
                seen_codes=seen_codes,
            )
        elif generic_only:
            _append_bare_generic_ai_finding(
                findings,
                message="OR group contains only generic AI terms without a discriminator.",
                seen_codes=seen_codes,
            )

    if has_discriminator:
        return findings

    for term in _quoted_terms(boolean):
        if _normalize_term(term) in _GENERIC_AI_TERMS:
            _append_bare_generic_ai_finding(
                findings,
                message=f'Bare generic AI quoted term "{term}" without a discriminator.',
                seen_codes=seen_codes,
            )
            break

    for _start, _end, group in _parenthetical_groups(boolean):
        if " OR " in group.upper():
            continue
        terms = _quoted_terms(group)
        if len(terms) == 1 and _normalize_term(terms[0]) in _GENERIC_AI_TERMS:
            _append_bare_generic_ai_finding(
                findings,
                message=f'Single-term group ({group[:80]}) is bare generic AI vocabulary.',
                seen_codes=seen_codes,
            )
            break

    for generic in ("ai", "llm", "genai"):  # VERTICAL-VOCAB(bare-generic-ai-craft-check)
        if re.search(rf"\b{generic}\b", boolean, flags=re.IGNORECASE):
            _append_bare_generic_ai_finding(
                findings,
                message=f'Unquoted generic AI token "{generic}" without a discriminator.',
                seen_codes=seen_codes,
            )
            break

    return findings


_LOW_SIGNAL_AND_CLAUSE_REPAIR = (
    "apply the maximum-inclusion test — would essentially all qualified people "
    "write one of these? OR-expand with concrete variants, move to a softer "
    "signal, or drop the gate."
)


def _not_group_starts(boolean: str) -> set[int]:
    """Start offsets of parenthetical groups a NOT operator applies to.

    Mirrors the mask-quoted-spans + uppercase-NOT-token technique in
    ``_check_not_group_contains_and``: a group sitting inside a NOT is an
    exclusion, not an AND gate, so callers reasoning about AND-gating skip it.
    """
    starts: set[int] = set()
    masked = _mask_quoted_spans(boolean)
    groups = _parenthetical_groups(boolean)
    for match in _UPPER_NOT_TOKEN_RE.finditer(masked):
        after = boolean[match.end() :]
        gap = len(after) - len(after.lstrip())
        group_start = match.end() + gap
        if group_start >= len(boolean) or boolean[group_start] != "(":
            continue
        group_match = next((g for g in groups if g[0] == group_start), None)
        if group_match is None:
            continue
        starts.add(group_match[0])
    return starts


def _check_low_signal_and_clause(
    boolean: str,
    context: BooleanLintContext,
) -> list[BooleanLintFinding]:
    """Cathey Maximum-Inclusion: flag AND-required OR groups made entirely of
    generic low-signal verbs (managed/led/owned/...) that over-constrain recall.

    Mirrors ``_check_bare_generic_ai_terms`` (per-OR-group all-terms subset
    predicate, single-term-group handling, discriminator escape) but excludes
    any group inside a NOT — an exclusion is not an AND gate.
    """
    findings: list[BooleanLintFinding] = []
    discriminators = context.capability_discriminators
    not_group_starts = _not_group_starts(boolean)
    not_group_texts = {
        group
        for start, _end, group in _parenthetical_groups(boolean)
        if start in not_group_starts
    }

    for group, terms in _or_groups_with_terms(boolean):
        if group in not_group_texts:
            continue
        normalized = {_normalize_term(term) for term in terms}
        low_signal_only = normalized and normalized.issubset(_STRUCTURAL_LOW_SIGNAL_TERMS)
        if not low_signal_only:
            continue
        if normalized & discriminators:
            continue
        findings.append(
            BooleanLintFinding(
                severity="warning",
                code="low_signal_and_clause",
                message=f"AND-gated OR group is only low-signal generic verbs ({group[:80]}).",
                repair_hint=_LOW_SIGNAL_AND_CLAUSE_REPAIR,
            )
        )

    for start, _end, group in _parenthetical_groups(boolean):
        if start in not_group_starts:
            continue
        if " OR " in group.upper():
            continue
        terms = _quoted_terms(group)
        if len(terms) != 1:
            continue
        normalized = {_normalize_term(terms[0])}
        if not normalized.issubset(_STRUCTURAL_LOW_SIGNAL_TERMS):
            continue
        if normalized & discriminators:
            continue
        findings.append(
            BooleanLintFinding(
                severity="warning",
                code="low_signal_and_clause",
                message=f"Single-term AND group ({group[:80]}) is a low-signal generic verb.",
                repair_hint=_LOW_SIGNAL_AND_CLAUSE_REPAIR,
            )
        )

    return findings


def _check_abbreviation_collisions(
    boolean: str,
    context: BooleanLintContext,
) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    for collision in context.abbreviation_collisions:
        abbrev = str(getattr(collision, "abbreviation", "") or "").strip()
        expansion = str(getattr(collision, "expansion", "") or "").strip()
        if not abbrev:
            continue
        abbrev_lower = abbrev.lower()
        expansion_lower = expansion.lower()
        for group, terms in _or_groups_with_terms(boolean):
            normalized = [_normalize_term(term) for term in terms]
            if abbrev_lower not in normalized:
                continue
            if expansion_lower and any(expansion_lower in term for term in normalized):
                continue
            findings.append(
                BooleanLintFinding(
                    severity="warning",
                    code="abbreviation_collision",
                    message=(
                        f'Bare abbreviation "{abbrev}" appears without required expansion '
                        f'"{expansion}" in the same OR group.'
                    ),
                    repair_hint=f'Include "{expansion}" alongside "{abbrev}" or use a qualified compound.',
                )
            )
    return findings


def _check_morphology(boolean: str, context: BooleanLintContext) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    pairs = context.morphology_pairs or _DEFAULT_MORPHOLOGY_PAIRS
    for _group, terms in _or_groups_with_terms(boolean):
        normalized_terms = {_normalize_term(term) for term in terms}
        for term in terms:
            base = _normalize_term(term)
            variants = pairs.get(base, ())
            if not variants:
                continue
            if not any(variant in normalized_terms for variant in variants):
                findings.append(
                    BooleanLintFinding(
                        severity="warning",
                        code="morphology_variant_missing",
                        message=(
                            f'"{term}" may need morphological variants '
                            f'({", ".join(variants)}) in the same OR group.'
                        ),
                        repair_hint=conservative_morphology_repair_hint(),
                    )
                )
    return findings


def _check_strict_seniority_title_bucket(
    boolean: str,
    context: BooleanLintContext,
    *,
    rationale: str = "",
    domain_lane: str = "",
    item: dict[str, Any] | None = None,
) -> list[BooleanLintFinding]:
    if not context.strict_seniority:
        return []
    title_risk = ""
    if item:
        title_risk = str(item.get("title_bucket_risk", "") or "").lower()
    if not title_risk:
        risk = classify_search_string_seniority(boolean, rationale, domain_lane=domain_lane)
        title_risk = str(risk.get("title_bucket_risk", "")).lower()
    if title_risk != "high":
        return []
    return [
        BooleanLintFinding(
            severity="warning",
            code="strict_seniority_broad_title_bucket",
            message="Broad title-bucket OR group on a strict-seniority brief.",
            repair_hint="Prefer ED-scoped or narrow title families with builder proof.",
        )
    ]


def _check_company_or_title_only(boolean: str) -> list[BooleanLintFinding]:
    text = boolean.lower()
    has_builder = any(pattern in text for pattern in _BUILDER_PROOF_PATTERNS)
    if has_builder:
        return []
    terms = _quoted_terms(boolean)
    if not terms:
        return []
    title_hits = sum(1 for term in terms if any(p in term.lower() for p in _TITLE_HEAVY_PATTERNS))
    if title_hits >= 2 and title_hits >= len(terms) // 2:
        return [
            BooleanLintFinding(
                severity="warning",
                code="title_only_capability_shape",
                message="String is mostly title buckets without builder/capability proof.",
                repair_hint="Add production, orchestration, or domain capability anchors.",
            )
        ]
    return []


def _check_overlong_or_groups(boolean: str, context: BooleanLintContext) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    limit = max(1, context.max_or_group_terms)
    for group, terms in _or_groups_with_terms(boolean):
        if len(terms) > limit:
            findings.append(
                BooleanLintFinding(
                    severity="warning",
                    code="overlong_or_group",
                    message=f"OR group has {len(terms)} quoted terms (limit {limit}).",
                    repair_hint="Split into tighter focus variants or demote weak synonyms.",
                )
            )
    return findings


def _check_boolean_filter_conflicts(
    boolean: str,
    context: BooleanLintContext,
) -> list[BooleanLintFinding]:
    if not context.structured_filters:
        return []
    findings: list[BooleanLintFinding] = []
    boolean_lower = boolean.lower()
    employers = context.structured_filters.get("target_employers") or []
    for employer in employers:
        name = str(employer or "").strip()
        if name and name.lower() in boolean_lower:
            findings.append(
                BooleanLintFinding(
                    severity="warning",
                    code="boolean_filter_dimension_conflict",
                    message=f'Boolean also mentions employer "{name}" already targeted by structured filters.',
                    repair_hint="Keep employer targeting in one execution surface (Boolean or company filter).",
                )
            )
    titles = context.structured_filters.get("job_titles") or context.structured_filters.get("titles") or []
    for title in titles:
        name = str(title or "").strip()
        if name and name.lower() in boolean_lower:
            findings.append(
                BooleanLintFinding(
                    severity="warning",
                    code="boolean_filter_dimension_conflict",
                    message=f'Boolean duplicates structured title filter "{name}".',
                    repair_hint="Use Advanced Search title filter or Boolean, not both for the same title.",
                )
            )
    return findings


# --- Noop-token lint checks (R4-F2 / P5.2): these catch quoted terms the
# LinkedIn tokenizer will not match as written, and NOT-group grammar the
# lint was previously structurally blind to. ---

_NOOP_SPECIAL_CHARACTERS = frozenset("$%&+")
_COMMA_NUMERAL_RE = re.compile(r"\d,\d")
_MID_WORD_STEM_SUFFIXES = ("nc", "iz", "yz")
# Complete words that legitimately end in a stem-suspicious suffix. "inc" is
# the load-bearing entry — "Acme Inc" style company terms are routine in
# booleans and must never read as truncated stems (correctness lens, Wave 2).
_MID_WORD_STEM_EXCEPTIONS = frozenset(
    {"zinc", "quiz", "showbiz", "biz", "inc", "sync", "async", "func"}
)
_OPERATOR_TOKEN_RE = re.compile(r"\b(and|or|not)\b", re.IGNORECASE)
_UPPER_NOT_TOKEN_RE = re.compile(r"\bNOT\b")

# P5.2 / plan Verify-at-implementation: "LinkedIn treats lowercase and/or/not
# as literals" is LinkedIn-help knowledge, not observed live this session.
# Ship lowercase_operator as a WARNING; promote to error only after a live
# Recruiter check (one search, two casings, compare result counts).
_BOOLEAN_LENGTH_WARNING_CAP = 2000  # LinkedIn Recruiter's exact keyword-field
# truncation point is unverified live; this is a conservative craft ceiling —
# generated strings run well under it, so exceeding it signals runaway
# generation.


_BARE_AMPERSAND_RE = re.compile(r"(?<![A-Za-z_])&(?![A-Za-z_])")


def _quoted_term_spans(text: str) -> list[tuple[int, int, str]]:
    return [
        (match.start(), match.end(), match.group(1).strip())
        for match in _QUOTED_TERM_RE.finditer(text)
        if match.group(1).strip()
    ]


def _quoted_term_contexts(boolean: str) -> list[tuple[str, list[str]]]:
    contexts: list[tuple[str, list[str]]] = []
    quoted_spans = _quoted_term_spans(boolean)
    if not quoted_spans:
        return contexts

    group_spans = _parenthetical_groups(boolean)
    for start, end, group in group_spans:
        terms = [
            term
            for term_start, term_end, term in quoted_spans
            if start <= term_start and term_end <= end
        ]
        if terms:
            contexts.append((group, terms))

    outside_terms = [
        term
        for term_start, term_end, term in quoted_spans
        if not any(
            start <= term_start and term_end <= end
            for start, end, _group in group_spans
        )
    ]
    if outside_terms:
        contexts.append((boolean, outside_terms))
    return contexts


def _contains_or_operator(text: str) -> bool:
    return bool(re.search(r"\bOR\b", _mask_quoted_spans(text), flags=re.IGNORECASE))


def _and_twin_for_ampersand_phrase(term: str) -> str:
    if not _BARE_AMPERSAND_RE.search(term):
        return ""
    return _normalize_term(_BARE_AMPERSAND_RE.sub(" and ", term))


def _has_ampersand_and_twin(group: str, term: str, terms: list[str]) -> bool:
    twin = _and_twin_for_ampersand_phrase(term)
    if not twin or not _contains_or_operator(group):
        return False
    return twin in {_normalize_term(candidate) for candidate in terms}


def _mask_quoted_spans(text: str) -> str:
    """Return `text` with the interior of quoted spans blanked to spaces.

    Preserves length and every non-quoted character (including its offset) so
    regex matches against the masked text map back to the same indices in
    `text`, while guaranteeing quoted content can never match an
    operator/keyword pattern.
    """
    chars = list(text)
    in_quote = False
    for index, char in enumerate(text):
        if char == '"' and (index == 0 or text[index - 1] != "\\"):
            in_quote = not in_quote
            continue
        if in_quote:
            chars[index] = " "
    return "".join(chars)


def _check_noop_special_character(boolean: str) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    for group, terms in _quoted_term_contexts(boolean):
        for term in terms:
            special_chars = {char for char in term if char in _NOOP_SPECIAL_CHARACTERS}
            if not special_chars:
                continue
            if "&" in special_chars and _has_ampersand_and_twin(group, term, terms):
                special_chars.remove("&")
            if not special_chars:
                continue
            findings.append(
                BooleanLintFinding(
                    severity="warning",
                    code="noop_special_character",
                    message=(
                        f'Quoted term "{term}" contains a character ($ % & +) the '
                        "LinkedIn tokenizer strips/ignores, so the term will not "
                        "match as written."
                    ),
                    repair_hint='Spell the value out in words (e.g. "$M" -> "million").',
                )
            )
    return findings


def _check_ampersand_missing_and_twin(boolean: str) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    for group, terms in _quoted_term_contexts(boolean):
        for term in terms:
            twin = _and_twin_for_ampersand_phrase(term)
            if not twin:
                continue
            if _has_ampersand_and_twin(group, term, terms):
                continue
            findings.append(
                BooleanLintFinding(
                    severity="warning",
                    code="ampersand_missing_and_twin",
                    message=(
                        f'Quoted ampersand phrase "{term}" appears without its '
                        '"and" twin in the same OR group.'
                    ),
                    repair_hint=(
                        f'Include "{twin}" alongside "{term}" in the same OR group.'
                    ),
                )
            )
    return findings


def _check_noop_comma_numeral(boolean: str) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    for term in _quoted_terms(boolean):
        if _COMMA_NUMERAL_RE.search(term):
            findings.append(
                BooleanLintFinding(
                    severity="warning",
                    code="noop_comma_numeral",
                    message=(
                        f'Quoted term "{term}" contains a comma-separated numeral; '
                        "LinkedIn tokenizes on the comma."
                    ),
                    repair_hint="Write the number without separators or reword.",
                )
            )
    return findings


def _check_mid_word_stem(boolean: str) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    for term in _quoted_terms(boolean):
        tokens = term.split()
        if not tokens:
            continue
        last_token = tokens[-1]
        last_lower = last_token.lower()
        is_exception = last_lower in _MID_WORD_STEM_EXCEPTIONS
        ends_hyphen = last_token.endswith("-")
        ends_suffix = not is_exception and any(
            last_lower.endswith(suffix) for suffix in _MID_WORD_STEM_SUFFIXES
        )
        if ends_hyphen or ends_suffix:
            findings.append(
                BooleanLintFinding(
                    severity="warning",
                    code="mid_word_stem",
                    message=(
                        f'Quoted term "{term}" appears to end mid-word '
                        f'("{last_token}") — a stem fragment will not match as '
                        "a whole term."
                    ),
                    # Matching-behavior facts live in matching_contract.py
                    # (M1B pin) — defer to its conservative guidance rather
                    # than asserting stemming behavior here.
                    repair_hint=(
                        f"Write the full word. {conservative_morphology_repair_hint()}"
                    ),
                )
            )
    return findings


def _check_not_group_contains_and(boolean: str) -> list[BooleanLintFinding]:
    """NOT must take a single term or a pure OR-group, never an AND-bearing group.

    Only an UPPERCASE `NOT` token (outside quotes) is matched; lowercase `not`
    is deliberately left to the (warning-severity) lowercase_operator check.
    """
    findings: list[BooleanLintFinding] = []
    masked = _mask_quoted_spans(boolean)
    groups = _parenthetical_groups(boolean)
    for match in _UPPER_NOT_TOKEN_RE.finditer(masked):
        after = boolean[match.end() :]
        gap = len(after) - len(after.lstrip())
        group_start = match.end() + gap
        if group_start >= len(boolean) or boolean[group_start] != "(":
            continue
        group_match = next((g for g in groups if g[0] == group_start), None)
        if group_match is None:
            continue
        _group_start, group_end, group_text = group_match
        inner_masked = _mask_quoted_spans(group_text)
        if re.search(r"\bAND\b", inner_masked, flags=re.IGNORECASE):
            findings.append(
                BooleanLintFinding(
                    severity="error",
                    code="not_group_contains_and",
                    message=(
                        "NOT applies to a group containing AND; NOT must take a "
                        "single term or a pure OR-group."
                    ),
                    span=(match.start(), group_end),
                    repair_hint='Split into NOT "term" clauses or convert the group to OR.',
                )
            )
    return findings


def _check_lowercase_operator(boolean: str) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    masked = _mask_quoted_spans(boolean)
    for match in _OPERATOR_TOKEN_RE.finditer(masked):
        token = boolean[match.start() : match.end()]
        if token.isupper():
            continue
        findings.append(
            BooleanLintFinding(
                severity="warning",
                code="lowercase_operator",
                message=(
                    f'Boolean operator "{token}" is not fully uppercase. LinkedIn '
                    "help documentation states lowercase and/or/not are treated as "
                    "literal keyword text rather than operators; this has not been "
                    "verified against a live Recruiter search, so this check ships "
                    "as a warning until confirmed."
                ),
                span=(match.start(), match.end()),
                repair_hint=f'Rewrite as "{token.upper()}".',
            )
        )
    return findings


def _check_boolean_length_cap(boolean: str) -> list[BooleanLintFinding]:
    if len(boolean) <= _BOOLEAN_LENGTH_WARNING_CAP:
        return []
    return [
        BooleanLintFinding(
            severity="warning",
            code="boolean_length_cap",
            message=(
                f"Boolean string is {len(boolean)} characters, exceeding the "
                f"{_BOOLEAN_LENGTH_WARNING_CAP}-character craft ceiling."
            ),
            repair_hint="Split into multiple narrower strings or trim redundant OR variants.",
        )
    ]


def _brief_terms_by_area(terms_by_area: Any, *, include_area_names: bool) -> set[str]:
    if not isinstance(terms_by_area, dict):
        return set()
    terms: set[str] = set()
    area_names: set[str] = set()
    for area, raw_terms in terms_by_area.items():
        area_name = _normalize_term(str(area or ""))
        if area_name:
            area_names.add(area_name)
        if not isinstance(raw_terms, (list, tuple)):
            continue
        for term in raw_terms:
            normalized = _normalize_term(str(term))
            if normalized:
                terms.add(normalized)
    if include_area_names and terms:
        terms.update(area_names)
    return terms


def _jd_register_terms_from_brief(
    key_terms_by_area: Any,
    candidate_register_terms_by_area: Any,
) -> frozenset[str]:
    key_terms = _brief_terms_by_area(key_terms_by_area, include_area_names=False)
    if not key_terms:
        return frozenset()
    jd_terms = _brief_terms_by_area(key_terms_by_area, include_area_names=True)
    candidate_terms = _brief_terms_by_area(
        candidate_register_terms_by_area, include_area_names=False
    )
    return frozenset(jd_terms - candidate_terms)


def _check_jd_register_overuse(
    boolean: str,
    context: BooleanLintContext,
) -> list[BooleanLintFinding]:
    quoted_terms = [_normalize_term(term) for term in _quoted_terms(boolean)]
    quoted_terms = [term for term in quoted_terms if term]
    # Guard: tiny strings produce false-positive ratios at any threshold.
    if len(quoted_terms) < 4 or not context.jd_register_terms:
        return []
    jd_hits = [term for term in quoted_terms if term in context.jd_register_terms]
    rate = len(jd_hits) / len(quoted_terms)
    if rate <= 0.25:
        return []
    offending_terms = ", ".join(sorted(set(jd_hits)))
    return [
        BooleanLintFinding(
            severity="warning",
            code="jd_register_overuse",
            message=(
                f"{len(jd_hits)}/{len(quoted_terms)} quoted terms ({rate:.0%}) "
                f"match JD-register vocabulary: {offending_terms}."
            ),
            repair_hint=(
                "Prefer candidate self-description vocabulary from "
                "candidate_register_terms for executable Boolean strings."
            ),
        )
    ]


def lint_boolean(
    boolean: str,
    *,
    context: BooleanLintContext | None = None,
    rationale: str = "",
    domain_lane: str = "",
    item: dict[str, Any] | None = None,
) -> BooleanLintReport:
    context = context or BooleanLintContext()
    text = (boolean or "").strip()
    if not text:
        return BooleanLintReport(
            boolean=text,
            findings=[
                BooleanLintFinding(
                    severity="error",
                    code="empty_boolean",
                    message="Boolean string is empty.",
                    repair_hint="Provide a non-empty LinkedIn Boolean query.",
                )
            ],
        )

    findings: list[BooleanLintFinding] = []
    findings.extend(_check_balanced_delimiters(text))
    findings.extend(_check_malformed_operators(text))
    findings.extend(_check_empty_or_groups(text))
    findings.extend(_check_not_group_contains_and(text))
    if not any(f.severity == "error" for f in findings):
        findings.extend(_check_wildcards(text))
        findings.extend(_check_bare_generic_ai_terms(text, context))
        findings.extend(_check_abbreviation_collisions(text, context))
        findings.extend(_check_morphology(text, context))
        findings.extend(
            _check_strict_seniority_title_bucket(
                text,
                context,
                rationale=rationale,
                domain_lane=domain_lane,
                item=item,
            )
        )
        findings.extend(_check_company_or_title_only(text))
        findings.extend(_check_overlong_or_groups(text, context))
        findings.extend(_check_low_signal_and_clause(text, context))
        findings.extend(_check_jd_register_overuse(text, context))
        findings.extend(_check_boolean_filter_conflicts(text, context))
        findings.extend(_check_ampersand_missing_and_twin(text))
        findings.extend(_check_noop_special_character(text))
        findings.extend(_check_noop_comma_numeral(text))
        findings.extend(_check_mid_word_stem(text))
        findings.extend(_check_lowercase_operator(text))
        findings.extend(_check_boolean_length_cap(text))

    return BooleanLintReport(boolean=text, findings=findings)


def lint_generated_string(
    item: dict[str, Any],
    *,
    context: BooleanLintContext | None = None,
    boolean_key: str = "boolean",
) -> BooleanLintReport:
    item_context = _context_for_item(context, item)
    boolean = str(item.get(boolean_key, "") or "")
    rationale = str(item.get("rationale", "") or item.get("gap", "") or "")
    domain_lane = str(item.get("domain_lane", "") or "")
    return lint_boolean(
        boolean,
        context=item_context,
        rationale=rationale,
        domain_lane=domain_lane,
        item=item,
    )


def summarize_kit_lint(kit_strings: Sequence[Any]) -> str:
    """Advisory-only lint summary over kit vocabulary strings.

    Kit strings are vocabulary — per strategy.py's header doctrine, "Kit
    strings are vocabulary — they NEVER appear in the execution queue"
    (mirrored at orchestrator.py's ``_build_ordered_search_strings``). This
    never blocks or reorders kit extraction; it only prints a defect summary
    so the model composing from the kit (and the human reviewing extraction)
    can see craft health. Returns "" when no kit string has a finding.
    """
    flagged = 0
    code_counts: dict[str, int] = {}
    for kit_string in kit_strings:
        boolean = str(getattr(kit_string, "boolean", "") or "")
        if not boolean.strip():
            continue
        report = lint_boolean(boolean)
        if not report.findings:
            continue
        flagged += 1
        for finding in report.findings:
            code_counts[finding.code] = code_counts.get(finding.code, 0) + 1

    if not flagged:
        return ""

    total = len(kit_strings)
    top_codes = sorted(code_counts.items(), key=lambda pair: pair[1], reverse=True)[:3]
    codes_summary = ", ".join(f"{code} x{count}" for code, count in top_codes)
    lines = [f"Kit lint: {flagged}/{total} kit strings have at least one lint finding."]
    if codes_summary:
        lines.append(f"Top codes: {codes_summary}.")
    lines.append(
        "Advisory only — kit strings are vocabulary and never enter the execution queue."
    )
    return "\n".join(lines)


def _render_or_fragment(values: list[str]) -> str:
    quoted = [f'"{value}"' for value in values if str(value).strip()]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return quoted[0]
    return f"({' OR '.join(quoted)})"


def compile_constraint(
    constraint: SearchConstraint,
    *,
    source: str = "linkedin",
) -> ExecutableConstraint:
    values = [str(value).strip() for value in constraint.values if str(value).strip()]
    surface = constraint.execution_surface or "soft_hint"
    notes: list[str] = []
    boolean_fragment = ""
    structured_control: dict[str, Any] = {}

    if surface in {"boolean_keyword", "boolean_title"}:
        fragment = _render_or_fragment(values)
        if constraint.operator == "exclude" and fragment:
            boolean_fragment = f"NOT {fragment}"
        else:
            boolean_fragment = fragment
        if not fragment:
            notes.append("No values to compile into Boolean fragment.")
    elif surface in {
        "linkedin_title_filter",
        "linkedin_company_filter",
        "linkedin_location_filter",
    }:
        dimension = surface.replace("linkedin_", "").replace("_filter", "")
        structured_control = {
            "source": source,
            "dimension": dimension,
            "values": values,
            "operator": constraint.operator,
            "temporal_scope": constraint.temporal_scope,
        }
        notes.append(f"Compiled to structured {dimension} control for {source}.")
    else:
        notes.append("soft_hint constraint is planner metadata only in P2.")

    return ExecutableConstraint(
        dimension=constraint.dimension,
        operator=constraint.operator,
        execution_surface=surface,
        temporal_scope=constraint.temporal_scope,
        values=values,
        boolean_fragment=boolean_fragment,
        structured_control=structured_control,
        compile_notes=notes,
    )


def lint_constraint_compile(
    constraint: SearchConstraint,
    compiled: ExecutableConstraint,
) -> list[BooleanLintFinding]:
    findings: list[BooleanLintFinding] = []
    if constraint.operator in {"require", "exclude"} and constraint.execution_surface == "soft_hint":
        findings.append(
            BooleanLintFinding(
                severity="warning",
                code="execution_surface_ambiguous",
                message=(
                    f'Constraint {constraint.dimension!r} uses {constraint.operator} '
                    "but execution_surface is soft_hint."
                ),
                repair_hint="Set execution_surface to a Boolean or LinkedIn filter control.",
            )
        )
    if (
        constraint.temporal_scope == "current"
        and constraint.execution_surface == "boolean_keyword"
        and constraint.operator in {"require", "prefer"}
    ):
        findings.append(
            BooleanLintFinding(
                severity="warning",
                code="temporal_scope_mismatch",
                message=(
                    "current temporal_scope with boolean_keyword may not enforce "
                    "present-role filtering without a title filter."
                ),
                repair_hint="Use linkedin_title_filter for current-title constraints.",
            )
        )
    if not compiled.boolean_fragment and not compiled.structured_control:
        if constraint.execution_surface not in {"soft_hint"}:
            findings.append(
                BooleanLintFinding(
                    severity="warning",
                    code="non_executable_constraint",
                    message=f"Constraint {constraint.dimension!r} did not compile to an executable surface.",
                    repair_hint="Provide values or choose a supported execution_surface.",
                )
            )
    return findings


def attach_boolean_lint_to_plan(brief: Any, plan: Any) -> None:
    """Attach boolean_lint metadata to plan strings without reordering."""
    context = boolean_lint_context_from_brief(brief)
    for item in getattr(plan, "generated_strings", []) or []:
        if isinstance(item, dict) and item.get("boolean"):
            item["boolean_lint"] = lint_generated_string(item, context=context).to_dict()
    for gap in getattr(plan, "coverage_gaps", []) or []:
        if isinstance(gap, dict) and gap.get("suggested_boolean"):
            gap["boolean_lint"] = lint_generated_string(
                gap,
                context=context,
                boolean_key="suggested_boolean",
            ).to_dict()
    attach_constraint_lint_to_plan(plan)


def attach_constraint_lint_to_plan(plan: Any) -> None:
    """Attach constraint compile lint to sourcing lane payloads on the plan."""
    from shared.sourcing_lanes import SearchConstraint, SourcingLane

    for lane_dict in getattr(plan, "sourcing_lanes", []) or []:
        if not isinstance(lane_dict, dict):
            continue
        lane = SourcingLane.from_dict(lane_dict)
        findings: list[dict[str, Any]] = []
        for constraint in lane.slice.constraints:
            compiled = compile_constraint(constraint)
            for finding in lint_constraint_compile(constraint, compiled):
                findings.append(finding.to_dict())
        if findings:
            lane_dict["constraint_lint"] = findings


_TITLE_LIKE_DIMENSIONS = frozenset({"title", "seniority", "role", "job_title"})


def repair_constraint_surfaces(plan: Any) -> list[dict[str, Any]]:
    """Flip an obvious keyword/structured surface mismatch on the plan's lanes, in place.

    Slice A part 3. Narrow by DESIGN — repairs ONLY the single case where a constraint
    plainly wanted a structured title filter but was left a Boolean keyword: a title-like
    dimension (in ``_TITLE_LIKE_DIMENSIONS``), already on ``boolean_keyword``, whose
    compile lint raises ``temporal_scope_mismatch`` (a current-role title that a keyword
    can't actually bound to present tense — boolean_compiler.py:766-781). Every OTHER lint
    finding is left advisory; non-title dimensions and non-keyword surfaces are untouched.

    Mutates ``lane_dict["slice"]["constraints"][i]["execution_surface"]`` IN PLACE so the
    flipped surface compiles into ``structured_filters`` when the lane compiler runs next.
    (``attach_constraint_lint_to_plan`` builds a ``SourcingLane.from_dict`` COPY, so a copy
    mutation would never reach the compiler — the repair must run on the serialized dict.)
    Must run BEFORE ``apply_linkedin_lane_compiler_to_plan``. Returns every repair record.
    """
    from shared.sourcing_lanes import SearchConstraint

    repairs: list[dict[str, Any]] = []
    for lane_dict in getattr(plan, "sourcing_lanes", []) or []:
        if not isinstance(lane_dict, dict):
            continue
        slice_dict = lane_dict.get("slice")
        if not isinstance(slice_dict, dict):
            continue
        constraint_dicts = slice_dict.get("constraints")
        if not isinstance(constraint_dicts, list):
            continue
        for constraint_dict in constraint_dicts:
            if not isinstance(constraint_dict, dict):
                continue
            constraint = SearchConstraint.from_dict(constraint_dict)
            compiled = compile_constraint(constraint)
            codes = {
                finding.code
                for finding in lint_constraint_compile(constraint, compiled)
            }
            if "temporal_scope_mismatch" not in codes:
                continue
            if constraint.dimension not in _TITLE_LIKE_DIMENSIONS:
                continue
            if constraint_dict.get("execution_surface") != "boolean_keyword":
                continue
            constraint_dict["execution_surface"] = "linkedin_title_filter"
            record = {
                "dimension": constraint.dimension,
                "from_surface": "boolean_keyword",
                "to_surface": "linkedin_title_filter",
                "finding": "temporal_scope_mismatch",
                "values": list(constraint.values),
            }
            lane_dict.setdefault("constraint_repairs", []).append(record)
            repairs.append(record)
    return repairs
