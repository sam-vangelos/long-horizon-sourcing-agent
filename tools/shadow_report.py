#!/usr/bin/env python3
"""Live feed + digest of the shadow experiments for one project state dir.

The feed MIRRORS the live runtime log (Sam, 2026-07-05): each comparison
renders as the same one-line format the run console prints — decisions by
model name, outcome in words, the RUNNING per-tier agreement tally — plus
one `[DECISION] summary` line per judgment in the exact grammar the live
console uses for the primary's verdicts. On top of that mirror it adds
what the run console doesn't carry: Fable's thinking and its compound
boolean outputs (plans for formation, example compounds for preflight),
side by side with Opus's. No JSON, no metrics tables, no chain-of-thought
on the default surface — raw depth stays in the artifacts and
shadow_judgments.jsonl, and `--reasoning` opts the full reasoning/verdict
prose back in for deep dives.

One-shot digest (renders everything captured so far, then exits):

  .venv/bin/python tools/shadow_report.py output/state/linkedin/<project-id>
  .venv/bin/python tools/shadow_report.py <state-dir> --disagreements

LIVE FEED — run beside a live sourcing session; streams every shadow
result the moment it lands. Ctrl-C to stop.

  .venv/bin/python tools/shadow_report.py <state-dir> --follow
  .venv/bin/python tools/shadow_report.py <state-dir> --follow --reasoning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Standalone-invocation safety: `python tools/shadow_report.py …` puts
# tools/ (not the repo root) on sys.path, which silently broke the
# shared.llm_clients import inside _extract_strings — fenced shadow plans
# rendered as unparsed text instead of strings. Self-insert the root so
# the viewer works identically standalone and when loaded via importlib.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RULE = "─" * 72

_MODEL_DISPLAY_WORDS = {
    "opus": "Opus",
    "glm": "GLM",
    "fable": "Fable",
    "mythos": "Mythos",
    "sonnet": "Sonnet",
    "haiku": "Haiku",
    "gpt": "GPT",
}


def _display_model(model_name: object, fallback: str) -> str:
    """Attribution-by-name (presentation doctrine): every section names its
    model — 'GLM', 'Opus', 'Fable' — never bare 'shadow'/'primary'."""
    text = str(model_name or "").rsplit("/", 1)[-1].lower()
    for word, display in _MODEL_DISPLAY_WORDS.items():
        if word in text:
            return display
    return text[:12] or fallback


def _model_word(model_name: object, fallback: str) -> str:
    """Lowercase model word for the comparison line (`opus=SAVE`), matching
    the run console's `_short_model_name` rendering exactly."""
    return _display_model(model_name, fallback).lower()


def _fit(text: object, width: int) -> str:
    """One line, padded or truncated-with-ellipsis to exactly ``width`` —
    same column discipline as the run console's comparison lines."""
    flat = " ".join(str(text).split())
    if len(flat) > width:
        return flat[: width - 1] + "…"
    return flat.ljust(width)


def _ts(epoch: float | None) -> str:
    if not epoch:
        return "?"
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime(
        "%H:%M:%SZ"
    )


def _latency_word(latency_ms: object) -> str:
    """'121s' — humans scanning a feed read seconds, not raw ms floats."""
    try:
        return f"{float(latency_ms) / 1000:.0f}s"
    except (TypeError, ValueError):
        return "?"


def _candidate_name(user_prompt: str | None) -> str:
    """Best-effort candidate identity from the captured eval prompt.
    Prefers an explicit Name:/Candidate:/Profile: line ANYWHERE in the
    prompt — the first line is sometimes a triage banner, not the
    candidate — falling back to the first non-empty line."""
    first_line = ""
    for line in (user_prompt or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not first_line:
            first_line = stripped[:80]
        lowered = stripped.lower()
        for prefix in ("name:", "candidate:", "profile:"):
            if lowered.startswith(prefix):
                value = stripped[len(prefix):].strip()
                if value:
                    return value
    return first_line or "(unknown candidate)"


def _finish_reason_note(rec: dict) -> str:
    """', finish_reason=length' — names truncation as truncation. Both
    2026-07-05 full-eval parse failures were max_tokens exhaustion, not
    format misses; the label should say so without opening the artifact."""
    reason = rec.get("finish_reason")
    return f", finish_reason={reason}" if reason else ""


def _parse_json(raw: object) -> dict | None:
    """Repair-tolerant JSON parse (the same helper every production path
    uses — strips code fences and surrounding prose)."""
    if isinstance(raw, dict):
        return raw
    if not (isinstance(raw, str) and raw.strip()):
        return None
    try:
        from shared.llm_clients import _parse_json_response

        parsed = _parse_json_response(raw)
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def _print_block(label: str, text: str) -> None:
    """A labeled prose section, indented under the current entry."""
    print(f"    {label}:")
    print("      " + text.strip().replace("\n", "\n      "))


def _print_excerpt(label: str, text: str, max_lines: int = 40) -> None:
    """A labeled, BOUNDED excerpt — the fallback for content that is
    genuinely unparseable. Never the whole thing."""
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        shown = "\n".join(lines[:max_lines])
        note = f"first {max_lines} of {len(lines)} lines; the rest is in the artifact"
    else:
        shown = "\n".join(lines)
        note = f"{len(lines)} lines"
    print(f"    {label} ({note}):")
    print("      " + shown.replace("\n", "\n      "))


# ---------------------------------------------------------------------------
# Running tally — recomputed from the record stream, so the feed's numbers
# are the true running totals from the first capture, identical semantics
# to the run console's: comparable outcomes drive the percentage;
# unparsed / not-comparable / error counted beside it in words.
# ---------------------------------------------------------------------------

_TALLY_KEYS = ("agree", "disagree", "unparsed", "not_comparable", "error")


def _new_tally() -> dict:
    return {tier: {key: 0 for key in _TALLY_KEYS} for tier in ("facial", "full")}


def _outcome_class(agrees, shadow_decision, parse_failed) -> str:
    if agrees is True:
        return "agree"
    if agrees is False:
        return "disagree"
    if shadow_decision is None or parse_failed:
        return "unparsed"
    return "not_comparable"


_OUTCOME_WORDS = {
    "agree": "AGREE",
    "disagree": "DISAGREE",
    "unparsed": "unparsed",
    "not_comparable": "not comparable",
}


def _render_tally(tier: str, tally: dict) -> str:
    counts = tally[tier]
    comparable = counts["agree"] + counts["disagree"]
    text = f"{tier}: {counts['agree']}/{comparable} agree"
    if comparable:
        text += f" ({counts['agree'] / comparable * 100:.1f}%)"
    if counts["unparsed"]:
        text += f", {counts['unparsed']} unparsed"
    if counts["not_comparable"]:
        text += f", {counts['not_comparable']} not comparable"
    if counts["error"]:
        n = counts["error"]
        text += f", {n} error" + ("s" if n != 1 else "")
    return text


def _comparison_line(tier_tag: str, subject: str, middle: str, tally_text: str) -> str:
    """The same one-line format the run console prints — the feed mirrors
    it exactly so the two surfaces read as one system."""
    return f"[shadow] {_fit(tier_tag, 9)} {_fit(subject, 28)} {_fit(middle, 62)} | {tally_text}"


# ---------------------------------------------------------------------------
# Judgment summaries — the live runtime log's grammar: `[DECISION] summary`.
# ---------------------------------------------------------------------------


def _clip_prose(text: str, limit: int = 400) -> str:
    """One collapsed line, hard-capped — a truncated response's
    CASE_AGAINST can swallow thousands of words of leaked deliberation
    (live-caught), and the summary line must stay a summary."""
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        return flat[: limit - 1].rstrip() + "…"
    return flat


def _full_summary_line(rec: dict, shadow_word: str) -> str:
    """GLM's full-eval verdict in the live log's judgment grammar. The
    bracket ALWAYS carries the RECORDED decision (the same one the
    comparison line shows — a truncated response can parse to a valid
    DECISION with no SUMMARY line, and the two lines must not
    contradict); the prose degrades: summary → case-against/for with a
    truncation note → a worded pointer at the raw capture."""
    decision = rec.get("shadow_decision") or "NO VERDICT"
    raw = rec.get("raw")
    if not raw:
        return f"    [{decision}] no response text captured"
    prose = None
    try:
        from linkedin.judgment_templates import parse_full_evaluation_response

        result = parse_full_evaluation_response(str(raw))
        prose = _clip_prose(result.summary) if (result.summary or "").strip() else None
        if prose is None and result.decision != "PARSE_FAILURE":
            fallback = (result.case_against or result.case_for or "").strip()
            if fallback:
                prose = (
                    f"{_clip_prose(fallback, 300)} (SUMMARY line missing"
                    f"{_finish_reason_note(rec)})"
                )
    except Exception:  # noqa: BLE001
        pass
    if prose:
        return f"    [{decision}] {prose}"
    return (
        f"    [{decision}{_finish_reason_note(rec)}] {shadow_word}'s raw text "
        "is in shadow_judgments.jsonl and the artifact"
    )


def _facial_summary_line(rec: dict, shadow_word: str) -> str:
    """GLM's facial verdict in the live log's judgment grammar. Same
    recorded-decision + degrading-prose contract as the full-eval line."""
    decision = rec.get("shadow_decision") or "NO VERDICT"
    raw = rec.get("raw")
    if not raw:
        return f"    [{decision}] no response text captured"
    prose = None
    try:
        from linkedin.judgment_templates import parse_facial_response

        result = parse_facial_response(str(raw))
        prose = _clip_prose(result.reason) if (result.reason or "").strip() else None
    except Exception:  # noqa: BLE001
        pass
    if prose:
        return f"    [{decision}] {prose}"
    return (
        f"    [{decision}{_finish_reason_note(rec)}] {shadow_word}'s raw text "
        "is in shadow_judgments.jsonl and the artifact"
    )


def _render_judgment_record(
    rec: dict,
    tally: dict,
    *,
    disagreements_only: bool = False,
    show_reasoning: bool = False,
) -> bool:
    """Render ONE shadow-judge capture: the mirrored comparison line, then
    the shadow's verdict in live-log grammar. Returns True when it printed
    (the tally is updated regardless, so --disagreements never skews it)."""
    stage = rec.get("stage", "?")
    shadow_word = _display_model(rec.get("shadow_model"), "the shadow")
    primary_word = _model_word(rec.get("primary_model"), "opus")
    glm_word = _model_word(rec.get("shadow_model"), "shadow")

    if stage == "facial_batch":
        count = int(rec.get("candidate_count") or 0)
        agrees = rec.get("agrees") or []
        shadows = rec.get("shadow_decisions") or []
        primaries = rec.get("primary_decisions") or []
        flags = rec.get("shadow_parse_failed") or []
        counts = {key: 0 for key in _TALLY_KEYS}
        if rec.get("shadow_error"):
            counts["error"] = 1
            middle = f"SHADOW ERROR: {rec['shadow_error']}"
        else:
            for i in range(count):
                counts[
                    _outcome_class(
                        agrees[i] if i < len(agrees) else None,
                        shadows[i] if i < len(shadows) else None,
                        flags[i] if i < len(flags) else True,
                    )
                ] += 1
            parts = [f"{counts['agree']} agree"]
            for key in ("disagree", "unparsed", "not_comparable"):
                if counts[key]:
                    parts.append(f"{counts[key]} {_OUTCOME_WORDS[key].lower()}")
            middle = ", ".join(parts)
        for key, n in counts.items():
            tally["facial"][key] += n
        if disagreements_only and not counts["disagree"] and not counts["error"]:
            return False
        print(
            _comparison_line(
                f"facial×{count}", "batch", middle, _render_tally("facial", tally)
            )
        )
        disagreed = [
            f"[{i + 1}] {primaries[i]}→{shadows[i] if i < len(shadows) else 'NO VERDICT'}"
            for i in range(min(count, len(primaries)))
            if i < len(agrees) and agrees[i] is False
        ]
        if disagreed:
            print(f"    disagreements ({primary_word}→{glm_word}): " + "  ".join(disagreed))
        if show_reasoning and rec.get("reasoning_content"):
            _print_block(f"{shadow_word}'s reasoning", str(rec["reasoning_content"]))
        return True

    tier = "full" if stage == "full" else "facial"
    if rec.get("shadow_error"):
        outcome = "error"
        middle = f"SHADOW ERROR: {rec['shadow_error']}"
    else:
        outcome = _outcome_class(
            rec.get("agrees"),
            rec.get("shadow_decision"),
            bool(rec.get("shadow_parse_failed")),
        )
        middle = (
            f"{primary_word}={rec.get('primary_decision') or 'NO VERDICT'}  "
            f"{glm_word}={rec.get('shadow_decision') or 'NO VERDICT'}  "
            f"{_OUTCOME_WORDS[outcome]}"
        )
    tally[tier][outcome] += 1
    if disagreements_only and outcome not in ("disagree", "error"):
        return False
    print(
        _comparison_line(
            tier,
            _candidate_name(rec.get("user_prompt")),
            middle,
            _render_tally(tier, tally),
        )
    )
    if not rec.get("shadow_error"):
        summary = (
            _full_summary_line(rec, shadow_word)
            if tier == "full"
            else _facial_summary_line(rec, shadow_word)
        )
        print(summary)
    if show_reasoning:
        if rec.get("reasoning_content"):
            _print_block(f"{shadow_word}'s reasoning", str(rec["reasoning_content"]))
        if rec.get("raw"):
            _print_block(f"{shadow_word}'s full verdict text", str(rec["raw"]))
    return True


# ---------------------------------------------------------------------------
# Shadow strategist artifacts — Fable's thinking + its compound booleans
# (a formation plan, or a preflight brief's example compounds), side by
# side with Opus's.
# ---------------------------------------------------------------------------


def _extract_strings(raw: object) -> list[dict]:
    """Pull generated_strings out of a raw plan response (dict or JSON text)."""
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict):
        return []
    strings = parsed.get("generated_strings")
    if not isinstance(strings, list):
        return []
    return [s for s in strings if isinstance(s, dict)]


def _print_plan(label: str, strings: list[dict]) -> None:
    print(f"    {label} ({len(strings)} strings):")
    for i, s in enumerate(strings):
        print(f"      [{i + 1}] {s.get('boolean', '')}")
        if s.get("rationale"):
            print(f"          why: {' '.join(str(s['rationale']).split())}")


def _metrics_words(metrics: dict | None) -> str | None:
    """'19 strings/5 skeletons/novelty 0.61' — None when not plan-shaped."""
    if not isinstance(metrics, dict) or metrics.get("parse_failed"):
        return None
    if not metrics.get("n_strings"):
        return None
    text = f"{metrics['n_strings']} strings/{metrics.get('distinct_skeletons')} skeletons"
    novelty = metrics.get("vocab_novelty")
    if novelty is not None:
        text += f"/novelty {novelty:.2f}"
    return text


def _metrics_compact(metrics: dict | None) -> str | None:
    if not isinstance(metrics, dict) or metrics.get("parse_failed"):
        return None
    if not metrics.get("n_strings"):
        return None
    text = f"{metrics['n_strings']}/{metrics.get('distinct_skeletons')}"
    novelty = metrics.get("vocab_novelty")
    if novelty is not None:
        text += f"/{novelty:.2f}"
    return text


def _brief_words(parsed: dict) -> str:
    """One compact worded line describing a preflight brief response."""
    parts = []
    if parsed.get("role_title"):
        parts.append(str(parsed["role_title"]))
    areas = parsed.get("capability_areas")
    if isinstance(areas, list):
        parts.append(f"{len(areas)} capability areas")
    lanes = parsed.get("domain_lane_hints")
    if isinstance(lanes, list) and lanes:
        parts.append(f"{len(lanes)} lanes")
    cal = parsed.get("facial_calibration")
    if isinstance(cal, dict):
        low, high = cal.get("expected_yes_rate_low"), cal.get("expected_yes_rate_high")
        if low is not None and high is not None:
            parts.append(f"yes-rate {low}-{high}")
    density = parsed.get("market_density")
    if density:
        parts.append(f"{density} market")
    return " — ".join(parts) if parts else "(brief fields unavailable)"


def _print_compounds(label: str, parsed: dict) -> None:
    compounds = parsed.get("example_compounds")
    if not (isinstance(compounds, list) and compounds):
        return
    print(f"    {label} ({len(compounds)}):")
    for i, comp in enumerate(compounds):
        if not isinstance(comp, dict):
            continue
        print(f"      [{i + 1}] {comp.get('boolean', '')}")
        if comp.get("purpose"):
            print(f"          why: {' '.join(str(comp['purpose']).split())}")


def _render_response_side(word: str, raw: object) -> None:
    """Render one side (Fable's or Opus's) of a strategist comparison:
    a plan's strings, a preflight brief's summary + example compounds, or
    — only when genuinely unparseable — a bounded excerpt."""
    strings = _extract_strings(raw)
    if strings:
        _print_plan(f"{word}'s plan", strings)
        return
    parsed = _parse_json(raw)
    if isinstance(parsed, dict):
        print(f"    {word}'s brief: {_brief_words(parsed)}")
        _print_compounds(f"{word}'s example compounds", parsed)
        return
    if isinstance(raw, str) and raw.strip():
        _print_excerpt(f"{word}'s response (unparseable)", raw)


def _render_strategy_artifact(path: Path) -> None:
    """Render ONE shadow-strategist artifact (one-shot report and, per
    artifact as it lands, --follow)."""
    try:
        art = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"[shadow] strategist  {path.name}: unreadable ({exc})")
        return
    primary_meta = art.get("primary_meta") or {}
    shadow_display = _display_model(art.get("shadow_model"), "the shadow")
    primary_display = _display_model(primary_meta.get("primary_model"), "Opus")
    shadow_low = shadow_display.lower()
    primary_low = primary_display.lower()
    stage = art.get("stage", "?")
    latency = _latency_word(art.get("latency_ms"))

    if art.get("shadow_error"):
        error = " ".join(str(art["shadow_error"]).split())[:160]
        print(f"[shadow] strategist({shadow_low}) {stage} SHADOW ERROR: {error}")
        if art.get("thinking_summary"):
            _print_block(
                f"{shadow_display}'s thinking (summarized)", art["thinking_summary"]
            )
        print(f"    artifact: {path.name}")
        return

    shadow_summary = _metrics_words(art.get("metrics"))
    if shadow_summary:
        primary_compact = _metrics_compact(primary_meta.get("metrics"))
        versus = (
            f"vs {primary_low} {primary_compact}"
            if primary_compact
            else f"vs {primary_low} (no plan metrics)"
        )
        print(
            f"[shadow] strategist({shadow_low}) {stage} done {latency} — "
            f"{shadow_summary} {versus}"
        )
    else:
        print(
            f"[shadow] strategist({shadow_low}) {stage} done {latency} — "
            f"{shadow_display} vs {primary_display} below"
        )
    if art.get("thinking_summary"):
        _print_block(
            f"{shadow_display}'s thinking (summarized)", art["thinking_summary"]
        )
    _render_response_side(shadow_display, art.get("raw_response"))
    _render_response_side(primary_display, primary_meta.get("raw_response"))
    print(f"    artifact: {path.name}")


def _strategy_artifacts(state_dir: Path) -> list[Path]:
    shadow_dir = state_dir / "shadow_strategy"
    return sorted(shadow_dir.glob("shadow-*.json")) if shadow_dir.exists() else []


def render_strategy_shadows(state_dir: Path) -> None:
    artifacts = _strategy_artifacts(state_dir)
    if not artifacts:
        print(
            "(no strategist shadows yet — they land at preflight and "
            "strategy formation)"
        )
        return
    for path in artifacts:
        print()
        _render_strategy_artifact(path)


def render_judgments(
    state_dir: Path,
    disagreements_only: bool,
    tally: dict | None = None,
    show_reasoning: bool = False,
) -> dict:
    """Render all captured judgments; returns the tally so --follow keeps
    counting from the true running totals."""
    tally = tally if tally is not None else _new_tally()
    path = state_dir / "shadow_judgments.jsonl"
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    if not records:
        print("(no shadow judgments yet — captures land per evaluated candidate)")
        return tally
    print()
    for rec in records:
        _render_judgment_record(
            rec,
            tally,
            disagreements_only=disagreements_only,
            show_reasoning=show_reasoning,
        )
    return tally


# ---------------------------------------------------------------------------
# --follow: live feed beside a running sourcing session
# ---------------------------------------------------------------------------


def _read_new_judgments(path: Path, offset: int) -> tuple[list[dict], int]:
    """Tail complete JSONL lines past ``offset``; a partially-written last
    line is left for the next poll (offset only advances past a newline)."""
    if not path.exists():
        return [], offset
    size = path.stat().st_size
    if size <= offset:
        return [], offset
    with path.open("rb") as fh:
        fh.seek(offset)
        chunk = fh.read(size - offset)
    last_newline = chunk.rfind(b"\n")
    if last_newline < 0:
        return [], offset
    complete = chunk[: last_newline + 1]
    records = []
    for line in complete.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return records, offset + len(complete)


def follow(
    state_dir: Path,
    disagreements_only: bool,
    show_reasoning: bool,
    poll_seconds: float = 1.0,
) -> int:
    """Render everything captured so far, then stream each new shadow
    result (GLM judgment or Fable strategy artifact) as it lands."""
    render_strategy_shadows(state_dir)
    tally = render_judgments(
        state_dir, disagreements_only, show_reasoning=show_reasoning
    )

    judgments_path = state_dir / "shadow_judgments.jsonl"
    offset = judgments_path.stat().st_size if judgments_path.exists() else 0
    seen_artifacts = set(_strategy_artifacts(state_dir))

    print(f"\n{RULE}")
    print(f"following {state_dir} — streaming new shadow results, Ctrl-C to stop")
    print(RULE, flush=True)
    try:
        while True:
            for path in _strategy_artifacts(state_dir):
                if path not in seen_artifacts:
                    seen_artifacts.add(path)
                    print()
                    _render_strategy_artifact(path)
                    sys.stdout.flush()
            new_records, offset = _read_new_judgments(judgments_path, offset)
            for rec in new_records:
                if _render_judgment_record(
                    rec,
                    tally,
                    disagreements_only=disagreements_only,
                    show_reasoning=show_reasoning,
                ):
                    sys.stdout.flush()
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\n(stopped)")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_dir", type=Path, help="project state dir")
    parser.add_argument(
        "--disagreements",
        action="store_true",
        help="show only judgments where the shadow disagreed (or errored); "
        "the running tally still counts everything",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="include the shadow model's full chain-of-thought and verdict "
        "prose under each comparison (default: live-log-style summaries "
        "only; the raw depth is always in shadow_judgments.jsonl)",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="after rendering current state, stream new shadow results live "
        "(run beside a live sourcing session; Ctrl-C to stop)",
    )
    args = parser.parse_args()
    if not args.state_dir.exists():
        print(f"state dir not found: {args.state_dir}", file=sys.stderr)
        return 1
    if args.follow:
        return follow(args.state_dir, args.disagreements, args.reasoning)
    render_strategy_shadows(args.state_dir)
    render_judgments(args.state_dir, args.disagreements, show_reasoning=args.reasoning)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
