"""Helpers for brief-scoped search family memory and lightweight metadata inference."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.brief_loader import Brief


_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")

_STOPWORDS = {
    "and",
    "or",
    "not",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "into",
    "workflow",
    "workflows",
    "system",
    "systems",
    "production",
    "enterprise",
    "built",
    "build",
    "deployed",
    "deploy",
    "shipped",
    "ship",
    "implemented",
    "implement",
    "engineering",
    "engineer",
}


def _slugify(value: str) -> str:
    text = _NON_ALNUM.sub("_", (value or "").lower()).strip("_")
    return re.sub(r"_+", "_", text) or "unlabeled_family"


def _canonicalize_lane_id(value: str) -> str:
    """Normalize a domain lane label into the canonical lane-id shape.

    Both the explicit-value path and the brief-hint fallback path in
    ``infer_domain_lane`` must return the same shape, so this helper is the
    single normalizer for lane labels (e.g. ``"Capital Markets"`` →
    ``"capital_markets"``).
    """
    return _slugify(value)


def _stringify_list(value: list[object]) -> list[str]:
    out: list[str] = []
    for item in value or []:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _domain_lane_hint_map(brief: "Brief | None") -> dict[str, tuple[str, ...]]:
    """Build a {lane: (pattern, ...)} map from a brief's domain_lane_hints, if any.

    Lane keys are canonicalized to match the explicit-value path in
    ``infer_domain_lane``; pattern values stay as raw lowercase strings so
    substring matching against the boolean/rationale text still works.
    """
    if brief is None:
        return {}
    hints = getattr(brief, "domain_lane_hints", None) or []
    out: dict[str, tuple[str, ...]] = {}
    for hint in hints:
        lane = str(getattr(hint, "lane", "") or "").strip()
        patterns = getattr(hint, "patterns", None) or []
        normalized = tuple(
            str(pattern).strip().lower()
            for pattern in patterns
            if str(pattern).strip()
        )
        if lane and normalized:
            out[_canonicalize_lane_id(lane)] = normalized
    return out


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial proportion.

    P6 (Wave 2): the volume-aware replacement for the flat hypothesis
    confidence constants — a rate backed by 400 candidates scores its
    evidence; a rate backed by 6 scores its uncertainty (audit R6-F4).
    Returns 0.0 when there is no volume or no successes.
    """
    if n <= 0 or successes <= 0:
        return 0.0
    successes = min(successes, n)
    p_hat = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) ** 0.5
    return max(0.0, (center - margin) / denominator)


def normalize_family_key(
    family_key: str | None,
    boolean: str = "",
    rationale: str = "",
) -> str:
    """Return a stable-ish family identifier for a generated string."""
    if family_key:
        return _slugify(family_key)

    anchors = extract_dominant_anchors(boolean or rationale, limit=4)
    if anchors:
        return _slugify("_".join(anchors))
    return "unlabeled_family"


def normalize_novelty_bucket(
    novelty_bucket: str | None,
    boolean: str = "",
    rationale: str = "",
    *,
    brief: "Brief | None" = None,
) -> str:
    """Normalize an explicit novelty value to ``edge_case``/``canonical``.

    When no explicit value is supplied, fall back to ``"canonical"`` — the
    vertical-agnostic default. ``boolean`` and ``rationale`` are accepted for
    signature stability with prior callers and for future brief-aware
    classification, but are not consulted here.
    """
    if novelty_bucket:
        value = _slugify(novelty_bucket)
        if value in {"edge_case", "canonical"}:
            return value
    return "canonical"


def infer_domain_lane(
    domain_lane: str | None,
    boolean: str = "",
    rationale: str = "",
    *,
    brief: "Brief | None" = None,
) -> str:
    """Infer the primary domain lane for a string.

    When ``brief`` exposes non-empty ``domain_lane_hints``, those patterns are
    consulted as a fallback after the explicit ``domain_lane`` value. Otherwise
    we default to ``"general"`` — the vertical-agnostic baseline.
    """
    if domain_lane:
        return _canonicalize_lane_id(domain_lane)

    lane_hints = _domain_lane_hint_map(brief)
    if lane_hints:
        text = f"{boolean} {rationale}".lower()
        for lane, patterns in lane_hints.items():
            if any(pattern in text for pattern in patterns):
                return lane

    return "general"


def extract_dominant_anchors(text: str, limit: int = 5) -> list[str]:
    """Extract a compact list of likely anchor tokens from a Boolean string.

    Uses generic English tokenization and stopword filtering only; no
    vertical-coupled phrase pre-pass.
    """
    lowered = (text or "").lower()
    tokens = [
        token for token in _WHITESPACE.split(_NON_ALNUM.sub(" ", lowered))
        if token and token not in _STOPWORDS and len(token) > 2
    ]
    counts = Counter(tokens)
    anchors: list[str] = []
    seen: set[str] = set()
    for token, _count in counts.most_common():
        if token in seen:
            continue
        anchors.append(token)
        seen.add(token)
        if len(anchors) >= limit:
            break
    return anchors


def build_search_memory_summary(memory: dict | None, limit: int = 8) -> dict:
    """Return a compact, LLM-facing summary of stored search family history."""
    memory = memory or {}
    overall = memory.get("overall", {})
    families = memory.get("families", {})

    summary_families = []
    for family in families.values():
        candidates = family.get("candidates_seen", 0)
        duplicates = family.get("duplicates", 0)
        saves = family.get("saves", 0)
        total_seen = candidates + duplicates
        summary_families.append(
            {
                "family_key": family.get("family_key", ""),
                "novelty_bucket": family.get("novelty_bucket", "canonical"),
                "domain_lane": family.get("domain_lane", "general"),
                "status": family.get("status", "active"),
                "status_reason": family.get("status_reason", ""),
                "save_rate": round(saves / max(candidates, 1), 4),
                "duplicate_rate": round(duplicates / max(total_seen, 1), 4),
                "saves": saves,
                "strings_seen": family.get("strings_seen", 0),
                "pages_reviewed": family.get("pages_reviewed", 0),
                "dominant_anchors": family.get("dominant_anchors", [])[:5],
                "save_exemplars": list(family.get("save_exemplars", []) or [])[:3],
                "winning_boolean": str(family.get("winning_boolean", "") or ""),
            }
        )

    summary_families.sort(
        key=lambda family: (
            0 if family["status"] == "exhausted" else 1,
            0 if family["novelty_bucket"] == "edge_case" else 1,
            family["save_rate"],
        )
    )
    layer_items = memory.get("layer_items", {})
    layer_item_summary = sorted(
        (
            {
                "layer_item_id": item.get("layer_item_id", ""),
                "layer_name": item.get("layer_name", ""),
                "label": item.get("label", ""),
                "family_keys": item.get("family_keys", [])[:3],
                "saves": item.get("saves", 0),
                "strings_seen": item.get("strings_seen", 0),
                "noise_rate": round(item.get("noise_rate", 0.0), 4),
                "duplicate_rate": round(item.get("duplicate_rate", 0.0), 4),
            }
            for item in layer_items.values()
            if isinstance(item, dict)
        ),
        key=lambda item: (item["saves"], item["strings_seen"]),
        reverse=True,
    )
    hypotheses = memory.get("hypotheses", {})
    hypothesis_summary = sorted(
        (
            {
                "hypothesis_id": item.get("hypothesis_id", ""),
                "status": item.get("status", "hypothesis"),
                "source": item.get("source", ""),
                "saves": item.get("saves", 0),
                "strings_seen": item.get("strings_seen", 0),
                "confidence": round(item.get("confidence", 0.0), 4),
            }
            for item in hypotheses.values()
            if isinstance(item, dict)
        ),
        key=lambda item: (item["saves"], item["strings_seen"]),
        reverse=True,
    )

    total_candidates = overall.get("candidates_seen", 0)
    total_duplicates = overall.get("duplicates", 0)
    total_seen = total_candidates + total_duplicates

    return {
        "project_id": memory.get("project_id", ""),
        "overall": {
            "families_tracked": len(summary_families),
            "strings_seen": overall.get("strings_seen", 0),
            "save_rate": round(overall.get("saves", 0) / max(total_candidates, 1), 4),
            "duplicate_rate": round(total_duplicates / max(total_seen, 1), 4),
            "novelty_mix": {
                "edge_case_saves": overall.get("edge_case_saves", 0),
                "canonical_saves": overall.get("canonical_saves", 0),
            },
        },
        "families": summary_families[:limit],
        "layer_items": layer_item_summary[:limit],
        "hypotheses": hypothesis_summary[:limit],
    }


def get_search_memory_families(memory: dict | None) -> list[dict]:
    """Return family records from either raw memory artifacts or summarized memory."""
    if not memory:
        return []

    families = memory.get("families", [])
    if isinstance(families, dict):
        iterable = families.values()
    elif isinstance(families, list):
        iterable = families
    else:
        return []

    return [family for family in iterable if isinstance(family, dict)]


def format_search_memory_summary(memory: dict | None, limit: int = 8) -> str:
    """Human-readable summary for prompt injection."""
    summary = memory or {}
    if not summary or not isinstance(summary.get("families"), list):
        summary = build_search_memory_summary(memory, limit=limit)
    overall = summary["overall"]
    lines = [
        "Prior search-family memory:",
        (
            f"- {overall['families_tracked']} families tracked; "
            f"save_rate={overall['save_rate']:.1%}; "
            f"duplicate_rate={overall['duplicate_rate']:.1%}; "
            f"novelty_mix=edge_case:{overall['novelty_mix']['edge_case_saves']} / "
            f"canonical:{overall['novelty_mix']['canonical_saves']}"
        ),
    ]

    if not summary["families"]:
        lines.append("- No family history available yet.")
        if summary.get("layer_items"):
            lines.append("- Layer items observed:")
            for item in summary["layer_items"][:limit]:
                lines.append(
                    "  "
                    f"* {item['layer_name']}::{item['label']} saves={item['saves']} "
                    f"strings={item['strings_seen']} noise_rate={item['noise_rate']:.1%}"
                )
        return "\n".join(lines)

    lines.append("- Family status:")
    for family in summary["families"]:
        anchors = ", ".join(family["dominant_anchors"]) or "n/a"
        lines.append(
            "  "
            f"* {family['family_key']} [{family['novelty_bucket']} / {family['domain_lane']}] "
            f"status={family['status']} save_rate={family['save_rate']:.1%} "
            f"duplicate_rate={family['duplicate_rate']:.1%} anchors={anchors}"
        )
        if family["status_reason"]:
            lines.append(f"    reason: {family['status_reason']}")
        # RC2: a family with saves renders as a located pocket, never a
        # bare counter a model can compress into "this pool is worked".
        if family.get("saves"):
            lines.append(
                f"    PROVEN VEIN: {family['saves']} saves over "
                f"{family.get('pages_reviewed', 0)} pages reviewed"
            )
            exemplars = family.get("save_exemplars") or []
            if exemplars:
                rendered = "; ".join(
                    f"{e.get('title', '?')} @ {e.get('company', '?')}"
                    for e in exemplars
                    if isinstance(e, dict)
                )
                if rendered:
                    lines.append(f"    save exemplars: {rendered}")
            if family.get("winning_boolean"):
                lines.append(
                    f"    winning boolean: {family['winning_boolean'][:180]}"
                )
    if summary.get("layer_items"):
        lines.append("- High-signal retrieval layer items:")
        for item in summary["layer_items"][:5]:
            lines.append(
                "  "
                f"* {item['layer_name']}::{item['label']} saves={item['saves']} "
                f"strings={item['strings_seen']} duplicate_rate={item['duplicate_rate']:.1%}"
            )
    if summary.get("hypotheses"):
        lines.append("- Edge-case hypothesis status:")
        for item in summary["hypotheses"][:5]:
            # P6 Wave-2 residual: `status` is a volume-independent 2/2 gate
            # while `confidence` is the Wilson lower bound on saves/candidates
            # — a hypothesis can be status=validated at confidence~0.10.
            # Render both so the LLM-facing text can't read "validated" as
            # stronger evidence than the volume actually supports.
            lines.append(
                "  "
                f"* {item['hypothesis_id']} status={item['status']} "
                f"confidence={item['confidence']:.2f} (Wilson floor on saves/candidates) "
                f"saves={item['saves']} strings={item['strings_seen']}"
            )

    return "\n".join(lines)


def _compute_key_stability(
    families_by_run: dict[int, set[str]],
    epoch_by_run: dict[int, str],
) -> dict | None:
    """P7 Stage C: Jaccard overlap of the latest run's family_key set vs the
    NEAREST prior run with the same brief_epoch (a revised brief is expected
    to churn its keys — the metric only compares like with like). Returns
    None when there is no comparable prior run; absence IS "no prior", the
    same no-affirmative-negative posture as ``_mark_if_truncated``."""
    runs = sorted(families_by_run)
    if len(runs) < 2:
        return None
    current = runs[-1]
    current_epoch = epoch_by_run.get(current, "")
    prior = next(
        (r for r in reversed(runs[:-1]) if epoch_by_run.get(r, "") == current_epoch),
        None,
    )
    if prior is None:
        return None
    current_keys = families_by_run[current]
    prior_keys = families_by_run[prior]
    union = current_keys | prior_keys
    jaccard = len(current_keys & prior_keys) / len(union) if union else 0.0
    return {
        "current_run_id": current,
        "prior_run_id": prior,
        "current_family_count": len(current_keys),
        "prior_family_count": len(prior_keys),
        "brief_epoch": current_epoch,
        "jaccard": round(jaccard, 4),
        "warning": bool(prior_keys) and jaccard < 0.3,
    }


def update_search_memory(
    memory: dict | None,
    project_id: str,
    strings: list,
) -> dict:
    """Merge completed search strings into the memory artifact."""
    memory = dict(memory or {})
    families = dict(memory.get("families", {}))
    overall = dict(memory.get("overall", {}))
    # P7 Stage C: family_key sets grouped per run (the projection attaches
    # run_id the same way it attaches brief_epoch); feeds the key-stability
    # metric after the merge loop. Persisted under "_families_by_run" and
    # MERGED with prior calls — the full-rebuild projection and the
    # incremental persisted-memory calling pattern (see the epoch tests)
    # must both keep the metric alive (correctness lens, slice 12).
    families_by_run: dict[int, set[str]] = {}
    epoch_by_run: dict[int, str] = {}
    for stored_run, stored_entry in (memory.get("_families_by_run") or {}).items():
        try:
            stored_run_id = int(stored_run)
        except (TypeError, ValueError):
            continue
        if not isinstance(stored_entry, dict):
            continue
        families_by_run[stored_run_id] = {
            str(f) for f in (stored_entry.get("families") or []) if str(f or "")
        }
        epoch_by_run[stored_run_id] = str(stored_entry.get("epoch") or "")

    overall.setdefault("strings_seen", 0)
    overall.setdefault("pages_reviewed", 0)
    overall.setdefault("candidates_seen", 0)
    overall.setdefault("duplicates", 0)
    overall.setdefault("saves", 0)
    overall.setdefault("edge_case_saves", 0)
    overall.setdefault("canonical_saves", 0)
    layer_items = dict(memory.get("layer_items", {}))
    layer_combinations = dict(memory.get("layer_combinations", {}))
    hypotheses = dict(memory.get("hypotheses", {}))

    for string in strings:
        family_key = normalize_family_key(
            getattr(string, "family_key", ""),
            getattr(string, "boolean", ""),
            getattr(string, "name", ""),
        )
        novelty_bucket = normalize_novelty_bucket(
            getattr(string, "novelty_bucket", ""),
            getattr(string, "boolean", ""),
            getattr(string, "name", ""),
        )
        domain_lane = infer_domain_lane(
            getattr(string, "domain_lane", ""),
            getattr(string, "boolean", ""),
            getattr(string, "name", ""),
        )

        entry = dict(families.get(family_key, {}))
        entry.setdefault("family_key", family_key)
        entry.setdefault("novelty_bucket", novelty_bucket)
        entry.setdefault("domain_lane", domain_lane)
        entry.setdefault("strings_seen", 0)
        entry.setdefault("pages_reviewed", 0)
        entry.setdefault("candidates_seen", 0)
        entry.setdefault("duplicates", 0)
        entry.setdefault("suppressed_prior_session", 0)
        entry.setdefault("saves", 0)
        entry.setdefault("facial_yes", 0)
        entry.setdefault("facial_no", 0)
        entry.setdefault("anchor_counts", {})
        entry.setdefault("example_booleans", [])
        # RC2 (2026-07-04): the discovered pocket, not just the label —
        # who the saves were and the boolean AS EXECUTED when they landed
        # (refinement mutates in place under the same family key).
        entry.setdefault("save_exemplars", [])
        entry.setdefault("winning_boolean", "")
        entry.setdefault("status", "active")
        entry.setdefault("status_reason", "")
        entry.setdefault("brief_epoch", "")
        entry.setdefault("archived_epochs", [])

        # P3.2: exhaustion epochs. A family's counters accumulate under one
        # brief content hash; when the brief revises, the old counters are
        # archived (kept for the prompt as history) and a fresh epoch starts —
        # a family exhausted under v1 must not be deterministically steered
        # away from under v2. Unchanged briefs never trigger this branch.
        string_epoch = str(getattr(string, "brief_epoch", "") or "")
        entry_epoch = str(entry.get("brief_epoch", "") or "")
        run_marker = getattr(string, "run_id", None)
        if isinstance(run_marker, int) and not isinstance(run_marker, bool):
            families_by_run.setdefault(run_marker, set()).add(family_key)
            epoch_by_run[run_marker] = string_epoch or epoch_by_run.get(run_marker, "")
        if string_epoch and entry_epoch and string_epoch != entry_epoch:
            archived = list(entry.get("archived_epochs", []))
            archived.append(
                {
                    "brief_epoch": entry_epoch,
                    "strings_seen": entry["strings_seen"],
                    "pages_reviewed": entry["pages_reviewed"],
                    "candidates_seen": entry["candidates_seen"],
                    "duplicates": entry["duplicates"],
                    "suppressed_prior_session": entry["suppressed_prior_session"],
                    "saves": entry["saves"],
                    "facial_yes": entry["facial_yes"],
                    "facial_no": entry["facial_no"],
                    "status": entry["status"],
                    "status_reason": entry["status_reason"],
                }
            )
            entry["archived_epochs"] = archived[-5:]
            for counter in (
                "strings_seen",
                "pages_reviewed",
                "candidates_seen",
                "duplicates",
                "suppressed_prior_session",
                "saves",
                "facial_yes",
                "facial_no",
            ):
                entry[counter] = 0
            entry["anchor_counts"] = {}
            # Exemplars and the winning boolean describe the OLD epoch's
            # pocket; a revised brief starts its discovery fresh.
            entry["save_exemplars"] = []
            entry["winning_boolean"] = ""
            entry["status"] = "active"
            entry["status_reason"] = ""
        if string_epoch:
            entry["brief_epoch"] = string_epoch

        candidates_seen = int(getattr(string, "candidates_count", 0) or 0)
        duplicates = int(getattr(string, "duplicates_count", 0) or 0)
        suppressed_prior = int(
            getattr(string, "suppressed_prior_session_count", 0) or 0
        )
        saves = len(getattr(string, "saves", []) or [])
        facial_yes = int(getattr(string, "facial_yes_count", 0) or 0)
        facial_no = int(getattr(string, "facial_no_count", 0) or 0)

        entry["strings_seen"] += 1
        entry["pages_reviewed"] += int(getattr(string, "pages_reviewed", 0) or 0)
        entry["candidates_seen"] += candidates_seen
        entry["duplicates"] += duplicates
        entry["suppressed_prior_session"] += min(suppressed_prior, duplicates)
        entry["saves"] += saves
        entry["facial_yes"] += facial_yes
        entry["facial_no"] += facial_no
        if saves:
            # The boolean as executed when the saves landed — the refined
            # form, not formation's original — plus who the saves were.
            entry["winning_boolean"] = str(getattr(string, "boolean", "") or "")
            merged = list(entry.get("save_exemplars", []))
            for exemplar in list(getattr(string, "save_exemplars", []) or []):
                if not isinstance(exemplar, dict):
                    continue
                item = {
                    "title": str(exemplar.get("title", "") or ""),
                    "company": str(exemplar.get("company", "") or ""),
                }
                if (item["title"] or item["company"]) and item not in merged:
                    merged.append(item)
            entry["save_exemplars"] = merged[-5:]
        entry["last_seen_at"] = datetime.now(timezone.utc).isoformat()

        anchors = extract_dominant_anchors(getattr(string, "boolean", ""))
        anchor_counts = Counter(entry.get("anchor_counts", {}))
        anchor_counts.update(anchors)
        entry["anchor_counts"] = dict(anchor_counts)
        entry["dominant_anchors"] = [
            anchor for anchor, _count in anchor_counts.most_common(5)
        ]

        example_booleans = list(entry.get("example_booleans", []))
        boolean = getattr(string, "boolean", "")
        if boolean and boolean not in example_booleans:
            example_booleans.append(boolean)
        entry["example_booleans"] = example_booleans[-3:]

        # P3.2: only same-epoch overlap feeds the exhaustion rule. Duplicates
        # that were prior-session suppressions are territory the CURRENT brief
        # never actually covered — counting them as overlap deterministically
        # steers post-revision runs away from re-coverable ground.
        same_epoch_duplicates = max(
            entry["duplicates"] - entry.get("suppressed_prior_session", 0), 0
        )
        total_seen = entry["candidates_seen"] + same_epoch_duplicates
        duplicate_rate = same_epoch_duplicates / max(total_seen, 1)
        save_rate = entry["saves"] / max(entry["candidates_seen"], 1)
        low_novelty = entry["novelty_bucket"] == "canonical"
        repeated = entry["strings_seen"] >= 2

        if repeated and duplicate_rate >= 0.40:
            entry["status"] = "exhausted"
            entry["status_reason"] = "Repeated family with high duplicate overlap."
        elif repeated and low_novelty and save_rate <= 0.03:
            entry["status"] = "exhausted"
            entry["status_reason"] = "Repeated canonical family with low save yield."
        else:
            entry["status"] = "active"
            entry["status_reason"] = ""

        families[family_key] = entry

        overall["strings_seen"] += 1
        overall["pages_reviewed"] += int(getattr(string, "pages_reviewed", 0) or 0)
        overall["candidates_seen"] += candidates_seen
        overall["duplicates"] += duplicates
        overall["saves"] += saves
        if novelty_bucket == "edge_case":
            overall["edge_case_saves"] += saves
        else:
            overall["canonical_saves"] += saves

        retrieval_recipe = getattr(string, "retrieval_recipe", {}) or {}
        used_layer_items = retrieval_recipe.get("used_layer_item_ids", {})
        if isinstance(used_layer_items, dict):
            for layer_name, item_ids in used_layer_items.items():
                for item_id in _stringify_list(item_ids):
                    layer_entry = dict(layer_items.get(item_id, {}))
                    layer_entry.setdefault("layer_item_id", item_id)
                    layer_entry.setdefault("layer_name", layer_name)
                    layer_entry.setdefault("label", item_id.replace("_", " "))
                    layer_entry.setdefault("family_keys", [])
                    layer_entry.setdefault("strings_seen", 0)
                    layer_entry.setdefault("saves", 0)
                    layer_entry.setdefault("duplicates", 0)
                    layer_entry.setdefault("candidates_seen", 0)
                    layer_entry["strings_seen"] += 1
                    layer_entry["saves"] += saves
                    layer_entry["duplicates"] += duplicates
                    layer_entry["candidates_seen"] += candidates_seen
                    if family_key not in layer_entry["family_keys"]:
                        layer_entry["family_keys"] = list(layer_entry["family_keys"]) + [family_key]
                    total_seen = layer_entry["candidates_seen"] + layer_entry["duplicates"]
                    layer_entry["duplicate_rate"] = round(
                        layer_entry["duplicates"] / max(total_seen, 1), 4
                    )
                    layer_entry["noise_rate"] = round(
                        max(layer_entry["candidates_seen"] - layer_entry["saves"], 0)
                        / max(layer_entry["candidates_seen"], 1),
                        4,
                    )
                    layer_items[item_id] = layer_entry

            combo_key = "|".join(
                sorted(
                    item_id
                    for values in used_layer_items.values()
                    for item_id in _stringify_list(values)
                )
            )
            if combo_key:
                combo_entry = dict(layer_combinations.get(combo_key, {}))
                combo_entry.setdefault("combo_key", combo_key)
                combo_entry.setdefault("strings_seen", 0)
                combo_entry.setdefault("saves", 0)
                combo_entry.setdefault("duplicates", 0)
                combo_entry.setdefault("candidate_count", 0)
                combo_entry["strings_seen"] += 1
                combo_entry["saves"] += saves
                combo_entry["duplicates"] += duplicates
                combo_entry["candidate_count"] += candidates_seen
                layer_combinations[combo_key] = combo_entry

        for hypothesis_id in _stringify_list(
            getattr(string, "retrieval_hypothesis_ids", [])
            or retrieval_recipe.get("applied_hypothesis_ids", [])
        ):
            hypothesis_entry = dict(hypotheses.get(hypothesis_id, {}))
            hypothesis_entry.setdefault("hypothesis_id", hypothesis_id)
            hypothesis_entry.setdefault("status", "hypothesis")
            hypothesis_entry.setdefault("source", "run")
            hypothesis_entry.setdefault("strings_seen", 0)
            hypothesis_entry.setdefault("saves", 0)
            hypothesis_entry.setdefault("duplicates", 0)
            hypothesis_entry.setdefault("candidate_count", 0)
            hypothesis_entry["strings_seen"] += 1
            hypothesis_entry["saves"] += saves
            hypothesis_entry["duplicates"] += duplicates
            hypothesis_entry["candidate_count"] += candidates_seen
            if hypothesis_entry["strings_seen"] >= 2 and hypothesis_entry["saves"] >= 2:
                hypothesis_entry["status"] = "validated"
            # P6 (Wave 2): volume-aware confidence — the Wilson lower bound
            # on saves/candidates replaces the flat 0.7/0.35/0.15 template
            # constants ("2 saves of 6" and "2 saves of 400" no longer
            # validate identically; audit R6-F4).
            hypothesis_entry["confidence"] = wilson_lower_bound(
                hypothesis_entry["saves"], hypothesis_entry["candidate_count"]
            )
            hypotheses[hypothesis_id] = hypothesis_entry

    memory["version"] = 1
    memory["project_id"] = project_id
    memory["updated_at"] = datetime.now(timezone.utc).isoformat()
    memory["overall"] = overall
    memory["families"] = families
    memory["layer_items"] = layer_items
    memory["layer_combinations"] = layer_combinations
    memory["hypotheses"] = hypotheses
    if families_by_run:
        # Prune to the most recent runs (the metric needs current + nearest
        # same-epoch prior; 8 keeps interleaved-epoch histories comparable
        # without unbounded artifact growth).
        kept_runs = sorted(families_by_run)[-8:]
        memory["_families_by_run"] = {
            str(run_id): {
                "epoch": epoch_by_run.get(run_id, ""),
                "families": sorted(families_by_run[run_id]),
            }
            for run_id in kept_runs
        }
        families_by_run = {run_id: families_by_run[run_id] for run_id in kept_runs}
    key_stability = _compute_key_stability(families_by_run, epoch_by_run)
    if key_stability:
        memory["key_stability"] = key_stability
    else:
        memory.pop("key_stability", None)
    return memory
