#!/usr/bin/env python3
"""Measure a strategy prompt variant on a model, and score what it produces.

One experiment, two questions: is the register-and-shape prompt worth its
place, and can Opus 5 take the strategy seat from Fable. Both are answered by
running the real formation path against a real preflight brief and scoring
every string it returns with ``linkedin.boolean_score``.

This tool SPENDS MONEY. ``--max-calls`` is a hard pre-flight guard: the run
aborts before any call that would exceed it, and ``--dry-run`` renders the
prompt and prints its size without calling anything.

THE ONE DEPARTURE FROM THE PRODUCTION CALL. ``form_strategy`` takes no effort
argument, and adding one would touch the live strategy path for an
experiment's sake. So the tool wraps ``linkedin.strategy.opus_llm`` — the
module-level binding of ``opus_llm_cached`` — with a partial carrying
``effort`` for the duration of the run, and restores it after. Production
renders the same request it always did; only this process passes an effort.

Effort is NOT the thinking parameter. It is ``output_config={"effort": ...}``,
current and GA, and Opus-family models still never receive a ``thinking``
argument.

THE CLI SEAT. A ``claude-cli:`` model prefix routes the strategy tier through
``shared/claude_cli_transport.py`` (``shared/llm_clients.py:857``) and spends
subscription usage instead of API credits. That path takes no ``effort`` and
reads ``timeout_seconds``, which ``--timeout`` supplies for this run only — a
formation call on the seat runs for many minutes, and the client's own default
is left alone.

    python tools/strategy_prompt_experiment.py --dry-run --variant register
    python tools/strategy_prompt_experiment.py \
        --model claude-opus-5 --effort max --variant live --samples 3
    python tools/strategy_prompt_experiment.py \
        --model claude-cli:claude-opus-5 --variant live --samples 3 --timeout 1500
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BRIEF = Path(
    "output/state/linkedin/research-engineer-frontier/preflight_v2_brief.json"
)
DEFAULT_OUT = Path("output/prompt-experiment")
BASELINE_PREFIX = "research_engineer_frontier-"

# Cost is not priced here. ``record_llm_usage`` already prices every call
# against the repo's one rate table (shared/llm_usage.py) and hands the result
# back on the receipt, so the experiment reads that number rather than keeping
# a second price list that can drift from the one production reconciles.


def _load_brief(path: Path):
    from shared.brief_loader import load_brief

    if not path.is_file():
        raise SystemExit(f"no preflight brief at {path}")
    return load_brief(str(path))


def _render_system(brief, variant: str) -> str:
    import shared.config as config
    from linkedin import strategy

    config.STRATEGY_PROMPT_VARIANT = variant
    strategy._config.STRATEGY_PROMPT_VARIANT = variant
    return strategy._build_strategy_system(brief, has_kit=False)


def _score_strings(strings: Sequence[str]) -> list[dict[str, Any]]:
    from linkedin.boolean_score import load_lexicon
    from linkedin.boolean_score.lexicon import LEXICON_DIR
    from linkedin.boolean_score.score import score

    lexicon = load_lexicon(LEXICON_DIR / "research-engineer-post-training.json")
    rows = []
    for boolean in strings:
        report = score(boolean, lexicon)
        rows.append(
            {
                "boolean": boolean,
                "composite": report.composite,
                "log_composite": report.log_composite,
                "shape": report.shape,
                "anchor_present": report.anchor_present,
                "components": {
                    key: value for key, value in report.components.items()
                },
            }
        )
    return rows


def _cell_metrics(rows: Sequence[dict[str, Any]], portfolios: int) -> dict[str, Any]:
    if not rows:
        return {"strings": 0}
    composites = [row["composite"] for row in rows]

    def _mean(name: str) -> float | None:
        values = [
            row["components"][name]
            for row in rows
            if row["components"].get(name) is not None
        ]
        return statistics.fmean(values) if values else None

    shapes: dict[str, int] = {}
    for row in rows:
        shapes[row["shape"]] = shapes.get(row["shape"], 0) + 1
    return {
        "portfolios": portfolios,
        "strings": len(rows),
        "strings_per_portfolio": round(len(rows) / portfolios, 1) if portfolios else None,
        "composite_mean": round(statistics.fmean(composites), 4),
        "composite_median": round(statistics.median(composites), 4),
        "register_mean": _mean("register_coverage"),
        "variant_mean": _mean("variant_coverage"),
        "cohesion_mean": _mean("cohesion"),
        "relevance_mean": _mean("relevance"),
        "parsimony_mean": _mean("parsimony"),
        "class_anchor_rate": round(
            sum(1 for row in rows if row["anchor_present"]) / len(rows), 3
        ),
        "shapes": dict(sorted(shapes.items())),
    }


def _baseline_rows() -> list[dict[str, Any]]:
    fixture = ROOT / "tests" / "fixtures" / "boolean_corpus" / "ml_runs_76.json"
    entries = json.loads(fixture.read_text())["strings"]
    return _score_strings(
        [row["boolean"] for row in entries if row["id"].startswith(BASELINE_PREFIX)]
    )


def _extremes(rows: Sequence[dict[str, Any]], count: int = 5) -> dict[str, list]:
    ordered = sorted(rows, key=lambda row: row["composite"])
    shape = lambda row: (round(row["composite"], 4), row["boolean"][:110])
    return {
        "lowest": [shape(row) for row in ordered[:count]],
        "highest": [shape(row) for row in ordered[-count:][::-1]],
    }


def _usage_row(context: dict[str, Any]) -> dict[str, Any]:
    """Read one call's measured tokens and priced cost out of its context.

    The client writes no token counts onto the ``usage_context`` dict it is
    handed. ``record_llm_usage`` writes exactly one key — ``_llm_receipt`` —
    whose ``actual_detail`` carries the token counts read off the response
    and the cost priced from them. That is the only place the numbers exist,
    so this is where the cost column comes from.

    A missing receipt means the call never reached the provider and came back
    through ``form_strategy``'s salvage path, which swallows the error. Raise
    rather than record a blank row: a silent zero would let the run keep
    spending against a cap on calls that are not producing anything.

    A receipt that is present but not ``ok`` is the same failure wearing a
    receipt. ``_record_llm_error`` writes one on the exception path with
    zeroed tokens, so presence alone proves nothing: without this check a
    timed-out sample falls straight through to the next one.
    """
    receipt = context.get("_llm_receipt") or {}
    detail = receipt.get("actual_detail") or {}
    if not detail:
        raise RuntimeError(
            "no LLM receipt on the usage context: the call never reached the "
            "provider (form_strategy salvages its own failures). Stop here "
            "rather than spending the rest of the cap."
        )
    status = receipt.get("actual_status")
    if status != "ok":
        error = detail.get("error") or {}
        raise RuntimeError(
            f"the call did not succeed (receipt status {status!r}): "
            f"{error.get('type')}: {error.get('message')}. Stop here rather "
            "than spending the rest of the cap."
        )
    return {
        "model": detail.get("model"),
        "input_tokens": detail.get("input_tokens"),
        "output_tokens": detail.get("output_tokens"),
        "cache_read_input_tokens": detail.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": detail.get("cache_creation_input_tokens"),
        "cost_usd": detail.get("estimated_cost_usd"),
        "rate_source": detail.get("rate_source"),
        "usage_status": detail.get("usage_status"),
        "stop_reason": receipt.get("actual_status"),
    }


def _usage_totals(usage: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum the measured columns across a cell's calls."""
    if not usage:
        return {"calls": 0}

    def _total(key: str) -> float:
        return sum(row.get(key) or 0 for row in usage)

    cost = _total("cost_usd")
    return {
        "calls": len(usage),
        "input_tokens": _total("input_tokens"),
        "output_tokens": _total("output_tokens"),
        "cache_read_input_tokens": _total("cache_read_input_tokens"),
        "cache_creation_input_tokens": _total("cache_creation_input_tokens"),
        "cost_usd_total": round(cost, 4),
        "cost_usd_per_call": round(cost / len(usage), 4),
        "seconds_mean": round(
            statistics.fmean([row.get("seconds") or 0 for row in usage]), 1
        ),
    }


def _model_slug(model: str) -> str:
    """A filename-safe form of a model id (``claude-cli:`` carries a colon)."""
    return model.replace(":", "-").replace("/", "-")


def run_samples(
    brief,
    *,
    model: str,
    variant: str,
    samples: int,
    effort: str | None,
    out_dir: Path,
    timeout_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Call the real formation path ``samples`` times. Returns (plans, usage)."""
    import shared.config as config
    from linkedin import strategy

    config.STRATEGY_PROMPT_VARIANT = variant
    strategy._config.STRATEGY_PROMPT_VARIANT = variant

    original = strategy.opus_llm
    plans: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    try:
        if effort is not None:
            strategy.opus_llm = functools.partial(original, effort=effort)
        for index in range(samples):
            context: dict[str, Any] = {"stage": "prompt_experiment"}
            started = time.time()
            call = functools.partial(original, effort=effort) if effort else original
            # One attempt, not five. The client's default is _MAX_RETRIES=5,
            # and a 300s Anthropic timeout retried five times is 25 silent
            # minutes and five billed generations before the run learns
            # anything. Under a hard call cap, a failed call should stop the
            # run rather than quietly buy four more.
            injected: dict[str, Any] = {
                "usage_context": context,
                "max_attempts": 1,
            }
            if timeout_seconds is not None:
                # A formation call on the CLI seat runs for many minutes, and
                # both the Anthropic path and _call_claude_cli read this same
                # argument. Per-run, never a change to the client's default.
                injected["timeout_seconds"] = timeout_seconds
            strategy.opus_llm = lambda *a, **k: call(*a, **{**k, **injected})  # noqa: E731
            plan = strategy.form_strategy(brief, [], None)
            elapsed = time.time() - started
            payload = plan.to_dict()
            plans.append(payload)
            # Write the plan before reading the receipt: a call that failed
            # still leaves whatever came back on disk when _usage_row raises.
            destination = out_dir / f"{_model_slug(model)}-{variant}-{index}.json"
            destination.write_text(json.dumps(payload, indent=2) + "\n")
            row = {"sample": index, "seconds": round(elapsed, 1), **_usage_row(context)}
            usage.append(row)
            produced = payload.get("generated_strings") or []
            print(f"    sample {index + 1}/{samples}: "
                  f"{len(produced)} strings, "
                  f"{elapsed:.0f}s, {row['input_tokens']}+{row['output_tokens']} tok, "
                  f"${row['cost_usd']:.4f}")
            if not produced:
                # A response that ran into max_tokens is priced and receipted
                # as a success — record_llm_usage runs before the stop_reason
                # check — so the string count is the only thing that shows it.
                # One such sample cost $5.02 and returned nothing.
                raise RuntimeError(
                    f"sample {index} produced no strings after "
                    f"{row['output_tokens']} output tokens and "
                    f"${row['cost_usd']:.4f}: the response was almost "
                    "certainly truncated at max_tokens. Stop here rather "
                    "than spending the rest of the cap."
                )
    finally:
        strategy.opus_llm = original
    return plans, usage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v2-brief", default=str(DEFAULT_BRIEF))
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--variant", default="live", choices=("live", "register"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--effort", default=None, choices=("low", "medium", "high", "xhigh", "max")
    )
    parser.add_argument("--max-calls", type=int, default=10)
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="per-call timeout in seconds, passed to the client for this run only",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args(argv)

    brief = _load_brief(Path(args.v2_brief))
    system = _render_system(brief, args.variant)

    if args.baseline_only:
        rows = _baseline_rows()
        print(json.dumps({"fable_live_baseline": _cell_metrics(rows, 1),
                          "extremes": _extremes(rows)}, indent=2))
        return 0

    print(f"model={args.model} variant={args.variant} samples={args.samples} "
          f"effort={args.effort or 'unset'}")
    print(f"system prompt: {len(system)} chars, {len(system.splitlines())} lines")

    if args.dry_run:
        print("dry run: no call made")
        return 0

    if args.samples > args.max_calls:
        raise SystemExit(
            f"refusing to start: {args.samples} calls would exceed the cap of "
            f"{args.max_calls}. Raise --max-calls deliberately or lower --samples."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import shared.config as config

    config.STRATEGY_MODEL_NAME = args.model

    plans, usage = run_samples(
        brief,
        model=args.model,
        variant=args.variant,
        samples=args.samples,
        effort=args.effort,
        out_dir=out_dir,
        timeout_seconds=args.timeout,
    )
    strings = [
        item.get("boolean", "")
        for plan in plans
        for item in plan.get("generated_strings", [])
        if item.get("boolean")
    ]
    rows = _score_strings(strings)
    summary = {
        "model": args.model,
        "variant": args.variant,
        "effort": args.effort,
        "timeout_seconds": args.timeout,
        "cell": _cell_metrics(rows, len(plans)),
        "cost": _usage_totals(usage),
        "extremes": _extremes(rows),
        "usage": usage,
    }
    (out_dir / f"{_model_slug(args.model)}-{args.variant}-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({"cell": summary["cell"], "cost": summary["cost"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
