"""LinkedIn Recruiter matching-model contract.

This module is deliberately conservative: until Recruiter seat-test evidence is
supplied, callers cannot obtain a verified matching contract. M1B can then wire
strategy generation and linting to this surface without independently
hardcoding external-system behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping


class EmpiricalStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"


class MatchingFact(str, Enum):
    KEYWORD_STEMMING = "keyword_stemming"
    HYPHEN_SPACE_TOKENIZATION = "hyphen_space_tokenization"
    KEYWORDS_SKILLS_FACET_EXPANSION = "keywords_skills_facet_expansion"


class KeywordStemmingModel(str, Enum):
    NO_STEMMING = "no_stemming"
    STEMS_OR_COLLAPSES = "stems_or_collapses"


class HyphenSpaceTokenizationModel(str, Enum):
    DISTINCT = "distinct"
    COLLAPSED = "collapsed"


class KeywordsSkillsFacetExpansionModel(str, Enum):
    NO_KEYWORDS_EXPANSION = "no_keywords_expansion"
    EXPANDS_OR_COLLAPSES = "expands_or_collapses"


class UnverifiedMatchingModelError(RuntimeError):
    """Raised when code tries to consume unverified external matching facts."""


class SeatTestEvidenceError(ValueError):
    """Raised when supplied Recruiter seat-test evidence is missing or incoherent."""


def render_strategy_matching_guidance(
    contract: LinkedInMatchingContract | None = None,
) -> str:
    """Render the single source of LinkedIn matching doctrine for strategy prompts."""

    contract = contract or load_persisted_matching_contract() or UNVERIFIED_LINKEDIN_MATCHING_CONTRACT
    return "\n".join(
        (
            "### LinkedIn Search Behavior (MANDATORY - contract-owned, pending live verification)",
            "",
            _render_matching_contract_status(contract),
            "",
            "LinkedIn Recruiter search doctrine for every OR group:",
            "",
            "1. **Case-insensitive authoring.** Do not include case-only variants; "
            "they waste OR slots and add no new authored signal.",
            "",
            "2. **Conservative morphology expansion.** Until Recruiter seat-test "
            "counts verify otherwise, treat character-level variants as distinct "
            "and include meaningful singular/plural, tense, noun/gerund, acronym, "
            "expansion, truncation, spacing, and hyphenation variants explicitly.",
            "",
            "3. **Substring behavior is not a verified contract.** Do not claim "
            "that superstrings add zero coverage. Keep OR groups compact, avoid "
            "obvious filler, and let the verified M1C normalizer handle any "
            "redundancy once the matching contract has live evidence.",
            "",
            "### Signal Test (every OR group must pass)",
            "",
            '- **Recall groups:** "Does this group anchor me to the right general '
            'population for this role?" It should return people plausibly in the '
            "right space, even if not all are perfect fits.",
            '  - PASS: ("PMP" OR "project management professional") -> returns '
            "project management practitioners",
            '  - FAIL: ("management") -> too broad, returns everyone with any '
            "leadership role",
            '  - FAIL: ("Python") -> returns all of software engineering',
            "",
            '- **Precision groups:** "Does this group confirm specific expertise '
            'that distinguishes specialists from generalists?"',
            '  - PASS: ("ISO 27001") -> specific certification, only '
            "security/compliance specialists hold it",
            '  - FAIL: ("code" OR "coding") -> everyone codes',
            '  - FAIL: ("trajectory") -> matches career-trajectory language on '
            "every profile",
            "",
            "### Disambiguation — No Bare Generic Terms",
            "",
            "A bare single-word term with a dominant non-target meaning on LinkedIn "
            "must not appear in any OR group. Use only qualified compound forms.",
            "",
            "### Tool/Library Names Are Proper Nouns",
            "",
            "Do not fabricate compound expansions for tool names. A single-term group "
            "is valid when the tool has no real variants.",
            "",
            "### Mandatory Self-Review Before Output",
            "",
            "1. Case dedup: remove case-only variants.",
            "2. Disambiguation: replace bare generic terms with compound forms.",
            "3. Abbreviation check: pair ambiguous abbreviations with expansions.",
            "4. Conservative expansion: include meaningful morphology and "
            "spacing/hyphenation variants while the live matching contract remains "
            "pending.",
        )
    )


def render_adaptation_matching_guidance(
    contract: LinkedInMatchingContract | None = None,
) -> str:
    """Render the contract-owned matching guidance for adaptation prompts."""

    contract = contract or load_persisted_matching_contract() or UNVERIFIED_LINKEDIN_MATCHING_CONTRACT
    return "\n".join(
        (
            "## LinkedIn Boolean Rules (MANDATORY - contract-owned, pending live verification)",
            _render_matching_contract_status(contract),
            "- Use conservative explicit morphology and spacing/hyphenation variants "
            "until the Recruiter seat-test counts verify the contract.",
            "- Do not claim substring/superstring coverage as a fact; keep groups "
            "compact and let the verified normalizer own redundancy pruning.",
            "- Never add case-only variants.",
            '- Bare ambiguous terms must be qualified: "agent" -> "AI agent".',
            "- Abbreviations with non-domain meanings must include the spelled-out form.",
            "- Tool/library names are proper nouns; do not fabricate compound expansions.",
        )
    )


def conservative_morphology_repair_hint() -> str:
    return (
        "Pending Recruiter seat tests, add explicit singular/plural and tense "
        "variants; do not rely on stemming until the matching contract is verified."
    )


def _render_matching_contract_status(contract: "LinkedInMatchingContract") -> str:
    status = "verified" if _contract_is_verified(contract) else "unverified"
    verified_at = contract.last_empirically_verified or "pending Recruiter seat tests"
    return (
        f"Contract status: {status}; lastEmpiricallyVerified: {verified_at}. "
        "Unverified rows use the conservative fallback and must not be described "
        "as live LinkedIn facts."
    )


def _contract_is_verified(contract: "LinkedInMatchingContract") -> bool:
    return all(value.status == EmpiricalStatus.VERIFIED for value in _empirical_values(contract))


@dataclass(frozen=True)
class SeatTestSpec:
    fact: MatchingFact
    query_key: str
    query: str
    evidence_required: str = "Recruiter result count"

    def to_dict(self) -> dict[str, str]:
        return {
            "fact": self.fact.value,
            "query_key": self.query_key,
            "query": self.query,
            "evidence_required": self.evidence_required,
        }


M1B_REQUIRED_SEAT_TESTS: tuple[SeatTestSpec, ...] = (
    SeatTestSpec(
        fact=MatchingFact.KEYWORD_STEMMING,
        query_key="benchmark",
        query="benchmark",
    ),
    SeatTestSpec(
        fact=MatchingFact.KEYWORD_STEMMING,
        query_key="benchmarks",
        query="benchmarks",
    ),
    SeatTestSpec(
        fact=MatchingFact.KEYWORD_STEMMING,
        query_key="benchmark_or_benchmarks",
        query="benchmark OR benchmarks",
    ),
    SeatTestSpec(
        fact=MatchingFact.HYPHEN_SPACE_TOKENIZATION,
        query_key="fine_tuning_hyphenated",
        query="fine-tuning",
    ),
    SeatTestSpec(
        fact=MatchingFact.HYPHEN_SPACE_TOKENIZATION,
        query_key="fine_tuning_spaced",
        query='"fine tuning"',
    ),
    SeatTestSpec(
        fact=MatchingFact.HYPHEN_SPACE_TOKENIZATION,
        query_key="finetuning_closed",
        query="finetuning",
    ),
    SeatTestSpec(
        fact=MatchingFact.HYPHEN_SPACE_TOKENIZATION,
        query_key="fine_tuning_union",
        query='fine-tuning OR "fine tuning" OR finetuning',
    ),
)

M1C_REQUIRED_SEAT_TESTS: tuple[SeatTestSpec, ...] = (
    SeatTestSpec(
        fact=MatchingFact.KEYWORDS_SKILLS_FACET_EXPANSION,
        query_key="saas_keyword",
        query="SaaS",
    ),
    SeatTestSpec(
        fact=MatchingFact.KEYWORDS_SKILLS_FACET_EXPANSION,
        query_key="saas_or_software_as_a_service",
        query='SaaS OR "Software as a Service"',
    ),
)

LINKEDIN_MATCHING_REQUIRED_SEAT_TESTS = (
    M1B_REQUIRED_SEAT_TESTS + M1C_REQUIRED_SEAT_TESTS
)


@dataclass(frozen=True)
class EmpiricalValue:
    fact: MatchingFact
    status: EmpiricalStatus
    value: Any | None = None
    evidence: Mapping[str, Any] | None = None
    verified_at: str | None = None

    @classmethod
    def unverified(cls, fact: MatchingFact) -> "EmpiricalValue":
        _require_matching_fact(fact)
        return cls(fact=fact, status=EmpiricalStatus.UNVERIFIED)

    @classmethod
    def verified(
        cls,
        *,
        fact: MatchingFact,
        value: Any,
        evidence: Mapping[str, Any],
        verified_at: str,
    ) -> "EmpiricalValue":
        _require_matching_fact(fact)
        _require_value_for_fact(fact, value)
        if not verified_at:
            raise ValueError("verified empirical values require verified_at")
        verified_at = _normalize_iso_timestamp(verified_at)
        if not evidence:
            raise ValueError("verified empirical values require evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("verified empirical values require evidence object")
        return cls(
            fact=fact,
            status=EmpiricalStatus.VERIFIED,
            value=value,
            evidence=dict(evidence),
            verified_at=verified_at,
        )

    def to_dict(self) -> dict[str, Any]:
        value = self.value
        if isinstance(value, Enum):
            value = value.value
        elif isinstance(value, tuple):
            value = [item.value if isinstance(item, Enum) else item for item in value]
        return {
            "fact": self.fact.value,
            "status": self.status.value,
            "value": value,
            "evidence": dict(self.evidence or {}),
            "verified_at": self.verified_at,
        }


@dataclass(frozen=True)
class LinkedInMatchingContract:
    keyword_stemming: EmpiricalValue
    hyphen_space_tokenization: EmpiricalValue
    keywords_skills_facet_expansion: EmpiricalValue
    boolean_operators: tuple[str, ...]
    schema_version: str = "linkedin.matching_contract.v1"

    @property
    def last_empirically_verified(self) -> str | None:
        verified_values = [
            value.verified_at
            for value in _empirical_values(self)
            if value.status == EmpiricalStatus.VERIFIED and value.verified_at
        ]
        return max(verified_values) if verified_values else None

    def require_verified(self) -> "LinkedInMatchingContract":
        missing = [
            value.fact.value
            for value in _empirical_values(self)
            if value.status != EmpiricalStatus.VERIFIED
        ]
        if missing:
            raise UnverifiedMatchingModelError(
                "LinkedIn matching contract is unverified for "
                f"{', '.join(missing)}. Supply Recruiter seat-test evidence: "
                f"{required_linkedin_matching_seat_test_summary()}."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "last_empirically_verified": self.last_empirically_verified,
            "keyword_stemming": self.keyword_stemming.to_dict(),
            "hyphen_space_tokenization": self.hyphen_space_tokenization.to_dict(),
            "keywords_skills_facet_expansion": (
                self.keywords_skills_facet_expansion.to_dict()
            ),
            "boolean_operators": list(self.boolean_operators),
        }


def _empirical_values(contract: LinkedInMatchingContract) -> tuple[EmpiricalValue, ...]:
    return (
        contract.keyword_stemming,
        contract.hyphen_space_tokenization,
        contract.keywords_skills_facet_expansion,
    )


def required_m1b_seat_tests() -> tuple[SeatTestSpec, ...]:
    return M1B_REQUIRED_SEAT_TESTS


def required_m1c_seat_tests() -> tuple[SeatTestSpec, ...]:
    return M1C_REQUIRED_SEAT_TESTS


def required_linkedin_matching_seat_tests() -> tuple[SeatTestSpec, ...]:
    return LINKEDIN_MATCHING_REQUIRED_SEAT_TESTS


def required_m1b_seat_test_summary() -> str:
    return "; ".join(spec.query for spec in M1B_REQUIRED_SEAT_TESTS)


def required_linkedin_matching_seat_test_summary() -> str:
    return "; ".join(spec.query for spec in LINKEDIN_MATCHING_REQUIRED_SEAT_TESTS)


def build_verified_contract_from_seat_test_counts(
    counts: Mapping[str, int],
    *,
    verified_at: str,
) -> LinkedInMatchingContract:
    """Build a matching contract only from explicit Recruiter seat-test counts."""

    verified_at = _normalize_iso_timestamp(verified_at)
    normalized_counts = _normalize_required_counts(counts)
    keyword_model = _derive_keyword_stemming_model(normalized_counts)
    hyphen_model = _derive_hyphen_space_tokenization_model(normalized_counts)
    skills_expansion_model = _derive_keywords_skills_facet_expansion_model(
        normalized_counts
    )

    return LinkedInMatchingContract(
        keyword_stemming=EmpiricalValue.verified(
            fact=MatchingFact.KEYWORD_STEMMING,
            value=keyword_model,
            evidence={
                "counts": {
                    key: normalized_counts[key]
                    for key in (
                        "benchmark",
                        "benchmarks",
                        "benchmark_or_benchmarks",
                    )
                },
                "rule": "benchmark_or_benchmarks > max(benchmark, benchmarks) => no_stemming",
            },
            verified_at=verified_at,
        ),
        hyphen_space_tokenization=EmpiricalValue.verified(
            fact=MatchingFact.HYPHEN_SPACE_TOKENIZATION,
            value=hyphen_model,
            evidence={
                "counts": {
                    key: normalized_counts[key]
                    for key in (
                        "fine_tuning_hyphenated",
                        "fine_tuning_spaced",
                        "finetuning_closed",
                        "fine_tuning_union",
                    )
                },
                "rule": "fine_tuning_union > max(single forms) => distinct",
            },
            verified_at=verified_at,
        ),
        keywords_skills_facet_expansion=EmpiricalValue.verified(
            fact=MatchingFact.KEYWORDS_SKILLS_FACET_EXPANSION,
            value=skills_expansion_model,
            evidence={
                "counts": {
                    key: normalized_counts[key]
                    for key in (
                        "saas_keyword",
                        "saas_or_software_as_a_service",
                    )
                },
                "rule": (
                    'saas_or_software_as_a_service > saas_keyword => '
                    "no_keywords_expansion"
                ),
            },
            verified_at=verified_at,
        ),
        boolean_operators=("AND", "OR", "NOT"),
    )


def _normalize_required_counts(counts: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(counts, Mapping):
        raise SeatTestEvidenceError("Recruiter seat-test counts must be an object")
    required_keys = {spec.query_key for spec in LINKEDIN_MATCHING_REQUIRED_SEAT_TESTS}
    for key in counts:
        if not isinstance(key, str):
            raise SeatTestEvidenceError(f"Seat-test count key {key!r} must be a string")
    missing = sorted(required_keys.difference(counts))
    if missing:
        raise SeatTestEvidenceError(
            "Missing Recruiter seat-test counts: " + ", ".join(missing)
        )
    unexpected = sorted(set(counts).difference(required_keys))
    if unexpected:
        raise SeatTestEvidenceError(
            "Unexpected Recruiter seat-test counts: " + ", ".join(unexpected)
        )

    normalized: dict[str, int] = {}
    for key in sorted(required_keys):
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SeatTestEvidenceError(f"Seat-test count {key!r} must be an integer")
        if value < 0:
            raise SeatTestEvidenceError(f"Seat-test count {key!r} must be non-negative")
        normalized[key] = value
    return normalized


def _derive_keyword_stemming_model(
    counts: Mapping[str, int],
) -> KeywordStemmingModel:
    benchmark = counts["benchmark"]
    benchmarks = counts["benchmarks"]
    union = counts["benchmark_or_benchmarks"]
    max_single = max(benchmark, benchmarks)
    if union < max_single:
        raise SeatTestEvidenceError(
            "benchmark OR benchmarks count cannot be lower than both single-form counts"
        )
    if union > max_single:
        return KeywordStemmingModel.NO_STEMMING
    return KeywordStemmingModel.STEMS_OR_COLLAPSES


def _derive_hyphen_space_tokenization_model(
    counts: Mapping[str, int],
) -> HyphenSpaceTokenizationModel:
    singles = (
        counts["fine_tuning_hyphenated"],
        counts["fine_tuning_spaced"],
        counts["finetuning_closed"],
    )
    union = counts["fine_tuning_union"]
    max_single = max(singles)
    if union < max_single:
        raise SeatTestEvidenceError(
            "fine-tuning union count cannot be lower than all single-form counts"
        )
    if union > max_single:
        return HyphenSpaceTokenizationModel.DISTINCT
    return HyphenSpaceTokenizationModel.COLLAPSED


def _derive_keywords_skills_facet_expansion_model(
    counts: Mapping[str, int],
) -> KeywordsSkillsFacetExpansionModel:
    saas = counts["saas_keyword"]
    expanded = counts["saas_or_software_as_a_service"]
    if expanded < saas:
        raise SeatTestEvidenceError(
            'SaaS OR "Software as a Service" count cannot be lower than SaaS count'
        )
    if expanded > saas:
        return KeywordsSkillsFacetExpansionModel.NO_KEYWORDS_EXPANSION
    return KeywordsSkillsFacetExpansionModel.EXPANDS_OR_COLLAPSES


def _normalize_iso_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("verified_at must be a string ISO timestamp")
    text = value.strip()
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("verified_at must be an ISO timestamp") from exc
    return text


def _require_matching_fact(fact: MatchingFact) -> None:
    if not isinstance(fact, MatchingFact):
        raise ValueError("empirical fact must be a MatchingFact")


def _require_value_for_fact(fact: MatchingFact, value: Any) -> None:
    expected_type = {
        MatchingFact.KEYWORD_STEMMING: KeywordStemmingModel,
        MatchingFact.HYPHEN_SPACE_TOKENIZATION: HyphenSpaceTokenizationModel,
        MatchingFact.KEYWORDS_SKILLS_FACET_EXPANSION: (
            KeywordsSkillsFacetExpansionModel
        ),
    }[fact]
    if not isinstance(value, expected_type):
        raise ValueError(f"{fact.value} value must be {expected_type.__name__}")


# Known artifact path where `tools/validate_linkedin_final_live_bucket.py` persists
# a verified contract when Sam runs it live with real Recruiter seat-test evidence
# (`--persist-contract`). Never written by tests or by importing this module.
MATCHING_CONTRACT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "linkedin-matching-contract.json"
)


def load_persisted_matching_contract(
    path: str | Path | None = None,
) -> "LinkedInMatchingContract | None":
    """Load a previously persisted verified matching contract, if present and valid.

    Returns ``None`` when the artifact is absent, unreadable, malformed, or fails
    validation — callers (the render_* functions) fall back to
    ``UNVERIFIED_LINKEDIN_MATCHING_CONTRACT`` in that case. This function never
    raises for a missing/bad artifact; it is a best-effort load, not a gate.
    """

    target = Path(path) if path is not None else MATCHING_CONTRACT_ARTIFACT_PATH
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, Mapping):
        return None
    try:
        return _matching_contract_from_dict(data)
    except (KeyError, ValueError, TypeError):
        return None


def _matching_contract_from_dict(data: Mapping[str, Any]) -> LinkedInMatchingContract:
    def _empirical_value(
        fact: MatchingFact, model_enum: type[Enum], key: str
    ) -> EmpiricalValue:
        raw_value = data[key]
        if not isinstance(raw_value, Mapping):
            raise ValueError(f"{key} must be an object")
        status = EmpiricalStatus(raw_value["status"])
        if status != EmpiricalStatus.VERIFIED:
            return EmpiricalValue.unverified(fact)
        return EmpiricalValue.verified(
            fact=fact,
            value=model_enum(raw_value["value"]),
            evidence=raw_value.get("evidence") or {},
            verified_at=raw_value["verified_at"],
        )

    return LinkedInMatchingContract(
        keyword_stemming=_empirical_value(
            MatchingFact.KEYWORD_STEMMING, KeywordStemmingModel, "keyword_stemming"
        ),
        hyphen_space_tokenization=_empirical_value(
            MatchingFact.HYPHEN_SPACE_TOKENIZATION,
            HyphenSpaceTokenizationModel,
            "hyphen_space_tokenization",
        ),
        keywords_skills_facet_expansion=_empirical_value(
            MatchingFact.KEYWORDS_SKILLS_FACET_EXPANSION,
            KeywordsSkillsFacetExpansionModel,
            "keywords_skills_facet_expansion",
        ),
        boolean_operators=tuple(data.get("boolean_operators") or ("AND", "OR", "NOT")),
        schema_version=data.get("schema_version", "linkedin.matching_contract.v1"),
    )


UNVERIFIED_LINKEDIN_MATCHING_CONTRACT = LinkedInMatchingContract(
    keyword_stemming=EmpiricalValue.unverified(MatchingFact.KEYWORD_STEMMING),
    hyphen_space_tokenization=EmpiricalValue.unverified(
        MatchingFact.HYPHEN_SPACE_TOKENIZATION
    ),
    keywords_skills_facet_expansion=EmpiricalValue.unverified(
        MatchingFact.KEYWORDS_SKILLS_FACET_EXPANSION
    ),
    boolean_operators=("AND", "OR", "NOT"),
)
