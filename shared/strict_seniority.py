"""Shared guardrails for strict-seniority sourcing briefs.

These helpers intentionally focus on conservative, low-ambiguity heuristics.
They are used to keep high-band searches from drifting into broad title buckets
or above-band executive populations.
"""

from __future__ import annotations

from typing import Any
import re


_STRICT_SENIORITY_TRIGGERS = (
    "executive director",
    "ed-analogous",
    "ed analogous",
    "one layer below broad enterprise executives",
    "executive-builder",
    "executive builder",
    "lab-head scope",
    "lab head scope",
)

_GENERIC_TITLE_BUCKETS = {
    "head of",
    "director",
    "vp",
    "svp",
    "senior vice president",
    "managing director",
    "cto",
    "principal",
    "chief architect",
}

_ABOVE_BAND_PROFILE_PATTERNS = (
    "managing director",
    "senior managing director",
    "global head",
    "chief data officer",
    "cdao",
    "chief information officer",
    "cio",
    "executive vice president",
    "evp",
    "president",
)

_NARROW_TITLE_PATTERNS = (
    "executive director",
    "head of ai",
    "head of applied ai",
    "head of ai platform",
    "head of ai platforms",
    "head of ai lab",
    "head of ai labs",
    "head of ai core",
    "head of ai research",
    "head of ai engineering",
    "chief ai architect",
    "principal architect",
    "technology fellow",
    "distinguished engineer",
    "applied science leader",
    "applied science manager",
    "research leader",
    "research manager",
)

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
    "shared ai platform",
    "common ai infrastructure",
)

_BUY_SIDE_PATTERNS = (
    "blackrock",
    "bridgewater",
    "citadel",
    "citadel securities",
    "two sigma",
    "d.e. shaw",
    "point72",
    "aqr",
    "man group",
    "vanguard",
    "tiaa",
    "wellington",
    "fidelity",
    "asset manager",
    "hedge fund",  # VERTICAL-VOCAB(strict-seniority-gate)
    "quant fund",
    "buy side",  # VERTICAL-VOCAB(strict-seniority-gate)
    "buy-side",  # VERTICAL-VOCAB(strict-seniority-gate)
)

_FINTECH_PATTERNS = (
    "stripe",
    "plaid",
    "revolut",
    "sofi",
    "affirm",
    "adyen",
    "klarna",
    "wise",
    "marqeta",
    "visa",
    "mastercard",
    "payments",
    "payment",
    "transaction banking",  # VERTICAL-VOCAB(strict-seniority-gate)
    "fintech",  # VERTICAL-VOCAB(strict-seniority-gate)
)

_PREFERRED_OPENING_LANES = {
    "capital_markets",
    "market_infra",
    "market_data",
    "risk_compliance",
    "bfsi_vendors",
    "general",
}


def _payload_lookup(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _brief_text(payload: Any) -> str:
    pieces: list[str] = []
    for field in (
        "role_description",
        "role_summary",
        "minimum_bar",
        "minimum_bar_description",
        "intake_notes",
        "notes",
    ):
        value = _payload_lookup(payload, field, "")
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())
    instructions = _payload_lookup(payload, "instructions", []) or []
    if isinstance(instructions, list):
        pieces.extend(str(item).strip() for item in instructions if str(item).strip())
    return " ".join(pieces).lower()


def minimum_years_experience(payload: Any) -> int:
    raw = _payload_lookup(payload, "raw", None)
    candidates = [
        _payload_lookup(payload, "minimum_years_experience", None),
        _payload_lookup(_payload_lookup(payload, "_new_brief", None), "minimum_years_experience", None),
        _payload_lookup(raw, "minimum_years_experience", None) if raw is not None else None,
    ]
    for value in candidates:
        try:
            if value is None:
                continue
            return int(value)
        except Exception:
            continue
    return 0


def is_strict_seniority_brief(payload: Any) -> bool:
    text = _brief_text(payload)
    years = minimum_years_experience(payload)
    if years < 12:
        return False
    if not any(trigger in text for trigger in _STRICT_SENIORITY_TRIGGERS):
        return False
    return "bfsi" in text or "financial" in text or "bank" in text  # VERTICAL-VOCAB(strict-seniority-gate)


def recommended_yoe_window(payload: Any) -> tuple[int, int]:
    minimum = max(minimum_years_experience(payload), 15)
    return minimum, 27


def classify_search_string_seniority(
    boolean: str,
    rationale: str = "",
    *,
    domain_lane: str = "",
) -> dict[str, Any]:
    text = f"{boolean} {rationale}".lower()
    quoted_terms = [term.strip().lower() for term in re.findall(r'"([^"]+)"', text)]

    generic_title_terms = sorted({term for term in quoted_terms if term in _GENERIC_TITLE_BUCKETS})
    narrow_title_hits = sorted({pattern for pattern in _NARROW_TITLE_PATTERNS if pattern in text})
    above_band_hits = sorted({pattern for pattern in _ABOVE_BAND_PROFILE_PATTERNS if pattern in text})
    builder_hits = sorted({pattern for pattern in _BUILDER_PROOF_PATTERNS if pattern in text})
    buy_side_hits = sorted({pattern for pattern in _BUY_SIDE_PATTERNS if pattern in text})
    fintech_hits = sorted({pattern for pattern in _FINTECH_PATTERNS if pattern in text})
    explicit_ed_scope = "executive director" in text

    title_bucket_risk = "low"
    if len(generic_title_terms) >= 2:
        title_bucket_risk = "high"
    elif len(generic_title_terms) == 1:
        title_bucket_risk = "medium"

    seniority_risk = "low"
    if above_band_hits or "managing director" in generic_title_terms:
        seniority_risk = "high"
    elif title_bucket_risk == "high":
        seniority_risk = "high"
    elif title_bucket_risk == "medium":
        seniority_risk = "medium"

    opening_eligible = True
    if seniority_risk == "high" or title_bucket_risk == "high":
        opening_eligible = False
    if (buy_side_hits or fintech_hits) and not (builder_hits or narrow_title_hits or explicit_ed_scope):
        opening_eligible = False
        if seniority_risk == "low":
            seniority_risk = "medium"

    preferred_opening_lane = (domain_lane or "").strip().lower() in _PREFERRED_OPENING_LANES

    return {
        "seniority_risk": seniority_risk,
        "title_bucket_risk": title_bucket_risk,
        "opening_eligible": opening_eligible,
        "generic_title_terms": generic_title_terms,
        "narrow_title_hits": narrow_title_hits,
        "above_band_hits": above_band_hits,
        "builder_proof_hits": builder_hits,
        "preferred_opening_lane": preferred_opening_lane or explicit_ed_scope,
    }


def profile_reads_above_band(profile: dict[str, Any] | None) -> bool:
    if not isinstance(profile, dict):
        return False
    text = " ".join(
        str(profile.get(field, "")).strip()
        for field in ("title", "headline")
        if str(profile.get(field, "")).strip()
    ).lower()
    if not text:
        return False
    return any(pattern in text for pattern in _ABOVE_BAND_PROFILE_PATTERNS)


def looks_like_company_inventory(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    if any(
        marker in normalized
        for marker in (
            "company-first anchors",
            "expanded bank set",
            "expanded market infrastructure",
            "title variants",
        )
    ):
        return True
    if normalized.count(",") >= 4:
        return True
    company_hits = sum(
        1
        for pattern in _BUY_SIDE_PATTERNS + _FINTECH_PATTERNS
        if pattern in normalized
    )
    return company_hits >= 3


def looks_like_title_inventory(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    if "title variants" in normalized:
        return True
    inventory_hits = sum(
        normalized.count(pattern)
        for pattern in (
            "head of ai",
            "head of ai labs",
            "head of ai core",
            "head of ai research",
            "head of ai engineering",
            "vp engineering",
            "managing director",
            "senior vice president",
        )
    )
    return inventory_hits >= 2


def mentions_risky_title_translation(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return any(
        pattern in normalized
        for pattern in (
            "buy-side md",  # VERTICAL-VOCAB(strict-seniority-gate)
            "buy side md",  # VERTICAL-VOCAB(strict-seniority-gate)
            "fintech vp",  # VERTICAL-VOCAB(strict-seniority-gate)
            "managing director",
            "vp title",
            "vp titles",
            "equivalent to executive director",
        )
    )
