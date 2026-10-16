"""Offline frontier preflight plus the legacy launcher readiness probe.

Frontier preflight checks role policy, limits, state ownership and configured
provider prices without creating state or making provider requests."""

from __future__ import annotations

from shared import config
from shared.health_types import ReadinessBlocker, ReadinessReport


DEFAULT_LIMITS = {"max_queries": 12, "max_candidates": 50, "max_minutes": 30}


def validate_frontier_config(value: dict) -> None:
    from researcher.schemas import ROLE_MODES
    from researcher.sources.public_web import public_reference
    from shared.brief_v2_schema import SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE

    if value.get("role_mode") not in ROLE_MODES:
        raise ValueError("Set source_config.researcher.role_mode to research_scientist or research_engineer; the title does not select policy")
    unsupported = set(value) - SOURCE_CONFIG_RECOGNIZED_KEYS_BY_SOURCE["researcher"]
    if unsupported:
        raise ValueError(f"Unsupported researcher controls: {sorted(unsupported)}")
    for key in ("research_topics", "conference_allowlist", "seed_urls"):
        items = value.get(key, [])
        if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError(f"researcher.{key} must be a list of nonempty strings")
    for key in ("h_index_floor", "papers_in_window_floor", "papers_in_window_months"):
        if key in value and (type(value[key]) is not int or value[key] < (1 if key.endswith("months") else 0)):
            raise ValueError(f"researcher.{key} must be a valid nonnegative integer (window months must be positive)")
    for url in value.get("seed_urls", []):
        public_reference(url)


def validate_limits(overrides: dict, *, inherited: dict | None = None) -> dict:
    import math

    limits = {**DEFAULT_LIMITS, **(inherited or {})}
    limits.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("max_queries", "max_candidates", "max_minutes", "max_cost_usd"):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} requires an explicit finite positive limit; resume may inherit it")
        if key in {"max_queries", "max_candidates"} and type(value) is not int:
            raise ValueError(f"{key} must be an integer")
    return limits


def frontier_preflight(*, brief_path, state_dir, limits: dict, resume=False, replan=False, revisit=None, check_providers=True) -> dict:
    """Read-only configuration and ownership check; never creates state or calls a provider."""
    import json
    import os
    import sqlite3
    from pathlib import Path
    from shared.brief_loader import load_brief
    from shared.brief_identity import compute_brief_identity
    from researcher.strategy import _brief_raw, frontier_plan
    from shared.brief_v2_schema import source_config_for

    if (replan or revisit) and not resume:
        raise ValueError("--replan and --revisit require --resume")
    if not Path(brief_path).is_file():
        raise ValueError(f"brief not found at {brief_path}")
    brief = load_brief(str(brief_path))
    source_config = source_config_for(_brief_raw(brief), "researcher")
    validate_frontier_config(source_config)
    identity = compute_brief_identity(brief_path)
    if identity is None:
        raise ValueError("Cannot establish brief identity")
    state_dir = Path(state_dir).resolve()
    if (state_dir / "run-manifest.json").exists():
        raise ValueError("Finalized run snapshots are immutable; use the campaign state directory")
    ancestor = state_dir
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise ValueError("State directory must have a writable directory ancestor")
    db = state_dir / "runtime_state.sqlite3"
    previous = None
    inherited = {}
    if db.exists():
        with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
            if any(r["source"] != "researcher" for r in rows):
                raise ValueError("State directory belongs to another source or brief")
            previous = dict(rows[0]) if rows else None
            if previous:
                prior_brief = json.loads(previous.get("brief_snapshot_json") or "{}")
                same_id = bool(prior_brief.get("id")) and prior_brief.get("id") == _brief_raw(brief).get("id")
                if previous["brief_id"] != brief.id and not same_id:
                    raise ValueError("State directory belongs to another brief")
                # The campaign key survives a title revision; the raw brief hash still changes.
                brief.id = previous["brief_id"]
                inherited = json.loads(previous.get("resume_state_json") or "{}").get("frontier", {}).get("limits", {})
            if revisit:
                row = conn.execute("SELECT terminal_decision, user_status FROM candidates WHERE identity_key=? AND source='researcher' AND brief_id=?", (revisit, brief.id)).fetchone()
                from shared.contracts import LEGACY_REVIEW_DECISIONS
                if not row or row[0] not in LEGACY_REVIEW_DECISIONS or row[1]:
                    raise ValueError("--revisit requires an existing non-save review identity")
    if resume and previous is None:
        raise ValueError("--resume requires an existing researcher campaign")
    if previous and not resume:
        raise ValueError("Existing campaign requires --resume")
    if previous and previous.get("brief_content_hash") != identity["brief_content_hash"] and not replan:
        raise ValueError("Brief hash changed or is unknown; use --resume --replan")
    effective = validate_limits(limits, inherited=inherited)
    plan = frontier_plan(brief)
    if check_providers:
        from shared.llm_clients import _is_fireworks_model, _is_claude_cli_model
        from shared.llm_usage import estimate_usage_cost_usd
        for model in (config.FACIAL_MODEL_NAME, config.OPUS_MODEL_NAME):
            if _is_claude_cli_model(model):
                raise ValueError("Configured CLI model has no bounded API cost estimate for this campaign")
            key = "FIREWORKS_API_KEY" if _is_fireworks_model(model) else "ANTHROPIC_API_KEY"
            if not getattr(config, key, ""):
                raise ValueError(f"{key} is required for the configured researcher model")
            if estimate_usage_cost_usd(model=model, output_tokens=4096)[0] is None:
                raise ValueError("Configured researcher model has no known price; supply a priced model configuration")
        if any(q["provider"] == "exa" for q in plan.generated_strings) and not os.environ.get("EXA_API_KEY"):
            raise ValueError("EXA_API_KEY is required for public-web discovery")
    return {"brief": brief, "identity": identity, "limits": effective, "previous": previous, "plan": plan}


def probe_researcher_readiness(
    *,
    anthropic_api_key: str | None = None,
) -> ReadinessReport:
    """Launch-readiness probe for Researcher.

    Args:
        anthropic_api_key: Override the ANTHROPIC_API_KEY env var.
            Defaults to ``shared.config.ANTHROPIC_API_KEY``.

    Returns:
        :class:`ReadinessReport` with ``ready`` true iff every check
        passed.
    """

    blockers: list[ReadinessBlocker] = []

    effective_anthropic = (
        anthropic_api_key
        if anthropic_api_key is not None
        else config.ANTHROPIC_API_KEY
    )
    if not effective_anthropic:
        blockers.append(
            ReadinessBlocker(
                kind="config",
                message="No Anthropic API key configured.",
                remediation=(
                    "Add ANTHROPIC_API_KEY to your .env file. "
                    "Researcher's identity, facial, and full judges "
                    "all call Claude — without a key the run fails at "
                    "the first judgment step."
                ),
            )
        )

    return ReadinessReport(ready=not blockers, blockers=tuple(blockers))
