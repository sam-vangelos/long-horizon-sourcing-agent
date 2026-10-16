"""Apply the lexicon enricher to a plan's strings, behind its flag.

The enricher itself (``linkedin/boolean_score/enrich.py``) knows nothing
about plans and imports nothing from this repo. This is the adapter: it
reads the flag, loads the role lexicon once, and rewrites the string fields
of the dicts a plan carries, leaving a receipt beside each one.

WHERE IT RUNS. At strategy formation, on ``generated_strings[].boolean`` and
``coverage_gaps[].suggested_boolean``, because the plan is finalized before
the run starts and a string enriched there is enriched everywhere it is
later read. And on adaptation's ``new_strings[].boolean``, because those are
authored mid-run and never pass through formation. The queue-time normalizer
is untouched and stays where it is: it is the safety net for anything that
reaches the queue by another road.

WHAT IT LEAVES BEHIND. With the flag on, each rewritten item carries
``<field>_original`` (what the model wrote), ``<field>_enrichment`` (the
report: what was added, what was dropped, whether the budget bit) and, for
the primary boolean, ``boolean_score`` (the composite and its components).
With the flag off it writes nothing at all, so a plan serializes byte for
byte as it did before this module existed — which is the one property the
default has to have.

FAILURE IS SILENT AND LOCAL. A missing or invalid lexicon disables
enrichment for the process; a string the enricher cannot handle is left as
authored. Neither prints: the run console is a fixed surface and this is not
on it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Mapping

import shared.config as _config
from linkedin.boolean_score.enrich import enrich
from linkedin.boolean_score.lexicon import (
    LEXICON_DIR,
    Lexicon,
    LexiconError,
    load_lexicon,
)
from linkedin.boolean_score.score import score

DEFAULT_LEXICON_PATH = LEXICON_DIR / "research-engineer-post-training.json"


def enrichment_enabled() -> bool:
    return bool(getattr(_config, "LINKEDIN_LEXICON_ENRICHMENT_ENABLED", False))


@lru_cache(maxsize=4)
def _load(path: str) -> Lexicon | None:
    try:
        return load_lexicon(path)
    except (LexiconError, OSError):
        return None


@lru_cache(maxsize=1)
def load_role_lexicon() -> Lexicon | None:
    """The configured role lexicon, or None when it cannot be read."""
    configured = str(getattr(_config, "LINKEDIN_BOOLEAN_LEXICON_PATH", "") or "").strip()
    return _load(configured or str(DEFAULT_LEXICON_PATH))


def enrich_items(items: Iterable[Any], field: str) -> int:
    """Enrich ``field`` on every mapping in ``items``. Returns how many changed.

    A no-op when the flag is off, when the lexicon will not load, or when an
    item carries nothing at ``field``.
    """
    if not enrichment_enabled():
        return 0
    lexicon = load_role_lexicon()
    if lexicon is None:
        return 0

    rewritten = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        original = item.get(field)
        if not isinstance(original, str) or not original.strip():
            continue
        try:
            report = enrich(original, lexicon)
        except Exception:  # noqa: BLE001 — never fail plan formation
            continue
        item[f"{field}_enrichment"] = report.to_dict()
        if not report.legal:
            continue
        item[f"{field}_original"] = original
        if not report.unchanged:
            item[field] = report.enriched
            rewritten += 1
        if field == "boolean":
            try:
                item["boolean_score"] = _score_summary(report.enriched, lexicon)
            except Exception:  # noqa: BLE001
                pass
    return rewritten


def _score_summary(boolean: str, lexicon: Lexicon) -> dict[str, Any]:
    report = score(boolean, lexicon)
    return {
        "composite": round(report.composite, 4),
        "log_composite": round(report.log_composite, 4),
        "shape": report.shape,
        "anchor_present": report.anchor_present,
        "components": {
            key: (round(value, 4) if value is not None else None)
            for key, value in report.components.items()
        },
    }


def enrich_plan(plan: Any) -> None:
    """Enrich a finalized execution plan's authored strings, in place."""
    if not enrichment_enabled():
        return
    enrich_items(getattr(plan, "generated_strings", None) or [], "boolean")
    enrich_items(getattr(plan, "coverage_gaps", None) or [], "suggested_boolean")


def enrich_adaptation(adaptation: Any) -> None:
    """Enrich adaptation-authored strings, in place."""
    if not enrichment_enabled():
        return
    enrich_items(getattr(adaptation, "new_strings", None) or [], "boolean")
