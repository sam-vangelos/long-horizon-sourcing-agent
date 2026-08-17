"""Advisory-only live sidecar for LinkedIn checkpoint guidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_intelligence.schema import MarketIntelAdvisory, MarketIntelAgentState
from shared.output_paths import derive_market_key_from_brief, output_root_for_path
from shared.search_memory import extract_dominant_anchors
from shared.storage import append_jsonl, read_json, read_jsonl, write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _string_list(values: list[Any]) -> list[str]:
    return [_normalize_text(value) for value in values if _normalize_text(value)]


@dataclass
class LiveCheckpointPacket:
    checkpoint_key: str
    checkpoint_scope: str
    checkpoint_index: int
    created_at: str
    brief_id: str
    string_family_key: str = ""
    block_name: str = ""
    recent_runtime_events: list[dict] = field(default_factory=list)
    recent_saved_examples: list[dict] = field(default_factory=list)
    recent_rejected_examples: list[dict] = field(default_factory=list)
    active_hypotheses: list[dict] = field(default_factory=list)
    prior_advisories: list[dict] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "checkpoint_key": self.checkpoint_key,
            "checkpoint_scope": self.checkpoint_scope,
            "checkpoint_index": self.checkpoint_index,
            "created_at": self.created_at,
            "brief_id": self.brief_id,
            "string_family_key": self.string_family_key,
            "block_name": self.block_name,
            "recent_runtime_events": self.recent_runtime_events,
            "recent_saved_examples": self.recent_saved_examples,
            "recent_rejected_examples": self.recent_rejected_examples,
            "active_hypotheses": self.active_hypotheses,
            "prior_advisories": self.prior_advisories,
            "payload": self.payload,
        }


def live_market_intel_dir(state_dir: str | Path) -> Path:
    return Path(state_dir).resolve() / "market_intel"


def _paths(state_dir: str | Path) -> dict[str, Path]:
    root = live_market_intel_dir(state_dir)
    return {
        "root": root,
        "observations": root / "live-observations.jsonl",
        "advisories": root / "live-advisories.jsonl",
        "summary": root / "live-summary.json",
    }


def _load_summary(state_dir: str | Path) -> dict:
    path = _paths(state_dir)["summary"]
    if not path.exists():
        return {"checkpoint_index": 0, "consumed_by": {}, "last_context": ""}
    try:
        data = read_json(path)
    except Exception:
        return {"checkpoint_index": 0, "consumed_by": {}, "last_context": ""}
    return data if isinstance(data, dict) else {"checkpoint_index": 0, "consumed_by": {}, "last_context": ""}


def _write_summary(state_dir: str | Path, summary: dict) -> None:
    write_json(_paths(state_dir)["summary"], summary)


def load_market_intel_agent_state(
    *,
    brief_path: str | Path,
    state_dir: str | Path,
) -> MarketIntelAgentState | None:
    output_root = output_root_for_path(state_dir)
    market_key = derive_market_key_from_brief(brief_path=brief_path)
    path = output_root / "market_intelligence" / market_key / "agent-state.json"
    if not path.exists():
        return None
    try:
        return MarketIntelAgentState.from_dict(read_json(path))
    except Exception:
        return None


def _load_recent_events(state_dir: str | Path, limit: int = 12) -> list[dict]:
    path = Path(state_dir).resolve() / "run_log.jsonl"
    return [item for item in read_jsonl(path) if isinstance(item, dict)][-limit:]


def _load_recent_candidates(state_dir: str | Path) -> tuple[list[dict], list[dict]]:
    final_path = Path(state_dir).resolve() / "final_judgments.jsonl"
    saved: list[dict] = []
    rejected: list[dict] = []
    for item in read_jsonl(final_path):
        if not isinstance(item, dict):
            continue
        decision = _normalize_text(item.get("decision")).upper()
        example = {
            "candidate_name": _normalize_text(item.get("candidate_name")),
            "path": _normalize_text(item.get("path")),
            "rationale": _normalize_text(item.get("rationale") or item.get("reason")),
            "decision": decision,
        }
        if decision in {"SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"}:
            saved.append(example)
        elif decision:
            rejected.append(example)
    return saved[-5:], rejected[-5:]


def _next_checkpoint_key(summary: dict, scope: str, family_key: str = "", block_name: str = "") -> tuple[int, str]:
    checkpoint_index = int(summary.get("checkpoint_index", 0) or 0) + 1
    anchor = family_key or block_name or scope
    return checkpoint_index, f"{scope}:{anchor}:{checkpoint_index}"


def _advisory_from_dict(record: dict) -> MarketIntelAdvisory | None:
    try:
        return MarketIntelAdvisory.from_dict(record)
    except Exception:
        return None


def _load_active_advisories(
    *,
    state_dir: str | Path,
    checkpoint_index: int,
    checkpoint_scope: str,
) -> list[MarketIntelAdvisory]:
    summary = _load_summary(state_dir)
    consumed_by = summary.get("consumed_by", {}) if isinstance(summary.get("consumed_by"), dict) else {}
    advisories: list[MarketIntelAdvisory] = []
    for item in read_jsonl(_paths(state_dir)["advisories"]):
        if not isinstance(item, dict):
            continue
        advisory = _advisory_from_dict(item)
        if advisory is None:
            continue
        if advisory.expires_at_checkpoint < checkpoint_index:
            continue
        if advisory.scope not in {checkpoint_scope, "run", "architecture"}:
            continue
        if any(
            str(marker).startswith(f"{checkpoint_scope}:")
            for marker in consumed_by.get(advisory.advisory_id, [])
        ):
            continue
        advisories.append(advisory)
    return advisories[:3]


def _record_consumed(
    *,
    state_dir: str | Path,
    advisories: list[MarketIntelAdvisory],
    checkpoint_scope: str,
    checkpoint_key: str,
) -> None:
    summary = _load_summary(state_dir)
    consumed_by = summary.setdefault("consumed_by", {})
    for advisory in advisories:
        scopes = consumed_by.setdefault(advisory.advisory_id, [])
        marker = f"{checkpoint_scope}:{checkpoint_key}"
        if marker not in scopes:
            scopes.append(marker)
    _write_summary(state_dir, summary)


def _heuristic_new_advisories(
    packet: LiveCheckpointPacket,
) -> list[MarketIntelAdvisory]:
    advisories: list[MarketIntelAdvisory] = []
    supporting_refs = []
    for hypothesis in packet.active_hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        supporting_refs.extend(_string_list(hypothesis.get("supporting_run_refs", [])))
    if not supporting_refs:
        supporting_refs = [f"live_state:{packet.brief_id}"]

    payload = packet.payload
    if packet.checkpoint_scope == "page":
        pages = int(payload.get("pages", 0) or 0)
        saves = int(payload.get("saves", 0) or 0)
        facial_yes = int(payload.get("facial_yes", 0) or 0)
        facial_no = int(payload.get("facial_no", 0) or 0)
        family_key = _normalize_text(packet.string_family_key)
        if family_key and any(
            family_key in _normalize_text(item.get("statement")).lower().replace(" ", "_")
            or family_key == _normalize_text(item.get("lane_key"))
            for item in packet.active_hypotheses
            if isinstance(item, dict)
        ):
            advisories.append(
                MarketIntelAdvisory(
                    advisory_id=f"adv-{family_key}-{packet.checkpoint_index}",
                    scope="page",
                    kind="exploit",
                    rationale="Prior market memory suggests this lane family is worth validating before narrowing aggressively.",
                    created_at=packet.created_at,
                    expires_at_checkpoint=packet.checkpoint_index + 2,
                    checkpoint_key=packet.checkpoint_key,
                    supporting_run_refs=supporting_refs,
                )
            )
        if pages >= 2 and saves == 0 and facial_yes == 0 and facial_no >= 8:
            advisories.append(
                MarketIntelAdvisory(
                    advisory_id=f"adv-noise-{packet.checkpoint_index}",
                    scope="page",
                    kind="deprioritize",
                    rationale="Recent page-level evidence is trending toward a dead/noisy lane. Prefer narrowing or stopping instead of deep pagination.",
                    created_at=packet.created_at,
                    expires_at_checkpoint=packet.checkpoint_index + 1,
                    checkpoint_key=packet.checkpoint_key,
                    supporting_run_refs=supporting_refs,
                )
            )

    if packet.checkpoint_scope == "block":
        top_performers = payload.get("top_performers", [])
        zero_save_count = int(payload.get("zero_save_count", 0) or 0)
        if top_performers:
            top = top_performers[0]
            family_key = _normalize_text(top.get("family_key") or top.get("name"))
            advisories.append(
                MarketIntelAdvisory(
                    advisory_id=f"adv-block-exploit-{family_key or packet.checkpoint_index}",
                    scope="block",
                    kind="exploit",
                    rationale=f"Lean into {family_key or 'the top-performing lane'} next; it is the clearest live winner in this checkpoint.",
                    created_at=packet.created_at,
                    expires_at_checkpoint=packet.checkpoint_index + 2,
                    checkpoint_key=packet.checkpoint_key,
                    supporting_run_refs=supporting_refs,
                )
            )
        elif zero_save_count >= 2:
            advisories.append(
                MarketIntelAdvisory(
                    advisory_id=f"adv-block-caution-{packet.checkpoint_index}",
                    scope="block",
                    kind="caution",
                    rationale="The block is coherently weak. Prefer a targeted pivot or explicit narrowing rather than incremental variants.",
                    created_at=packet.created_at,
                    expires_at_checkpoint=packet.checkpoint_index + 1,
                    checkpoint_key=packet.checkpoint_key,
                    supporting_run_refs=supporting_refs,
                )
            )

    deduped: list[MarketIntelAdvisory] = []
    seen: set[str] = set()
    for advisory in advisories:
        if advisory.kind in seen:
            continue
        seen.add(advisory.kind)
        deduped.append(advisory)
    return deduped


def _render_context(advisories: list[MarketIntelAdvisory]) -> str:
    if not advisories:
        return ""
    lines = ["## Market Intel Advisory Context"]
    for advisory in advisories[:3]:
        lines.append(f"- [{advisory.kind}] {advisory.rationale}")
    return "\n".join(lines)


def record_page_checkpoint_and_get_context(
    *,
    brief_path: str | Path,
    state_dir: str | Path,
    brief_id: str,
    search_string_family_key: str,
    string_stats: dict,
    recent_candidates: list[dict],
    glance_summary: str | None,
    architecture: str,
) -> str:
    live_dir_paths = _paths(state_dir)
    live_dir_paths["root"].mkdir(parents=True, exist_ok=True)
    summary = _load_summary(state_dir)
    checkpoint_index, checkpoint_key = _next_checkpoint_key(
        summary,
        "page",
        family_key=search_string_family_key,
    )
    saved_examples, rejected_examples = _load_recent_candidates(state_dir)
    agent_state = load_market_intel_agent_state(brief_path=brief_path, state_dir=state_dir)
    packet = LiveCheckpointPacket(
        checkpoint_key=checkpoint_key,
        checkpoint_scope="page",
        checkpoint_index=checkpoint_index,
        created_at=_utc_now(),
        brief_id=brief_id,
        string_family_key=search_string_family_key,
        recent_runtime_events=_load_recent_events(state_dir),
        recent_saved_examples=saved_examples,
        recent_rejected_examples=rejected_examples,
        active_hypotheses=[
            item.to_dict() for item in (agent_state.active_hypotheses if agent_state else [])
        ][:5],
        prior_advisories=[
            item.to_dict() for item in (agent_state.prior_advisories if agent_state else [])
        ][:5],
        payload={
            **(string_stats or {}),
            "glance_summary": _normalize_text(glance_summary),
            "architecture": _normalize_text(architecture),
            "recent_candidates": recent_candidates[:5],
            "dominant_anchors": extract_dominant_anchors(
                " ".join(_string_list([candidate.get("rationale", "") for candidate in recent_candidates])),
                limit=5,
            ),
        },
    )
    append_jsonl(live_dir_paths["observations"], packet.to_dict())
    new_advisories = _heuristic_new_advisories(packet)
    for advisory in new_advisories:
        append_jsonl(live_dir_paths["advisories"], advisory.to_dict())
    active = _load_active_advisories(
        state_dir=state_dir,
        checkpoint_index=checkpoint_index,
        checkpoint_scope="page",
    )
    summary["checkpoint_index"] = checkpoint_index
    summary["last_context"] = _render_context(active)
    _write_summary(state_dir, summary)
    _record_consumed(
        state_dir=state_dir,
        advisories=active,
        checkpoint_scope="page",
        checkpoint_key=checkpoint_key,
    )
    return summary["last_context"]


def record_block_checkpoint_and_get_context(
    *,
    brief_path: str | Path,
    state_dir: str | Path,
    brief_id: str,
    block_name: str,
    block_report: Any,
    search_memory_summary: dict | None,
) -> str:
    live_dir_paths = _paths(state_dir)
    live_dir_paths["root"].mkdir(parents=True, exist_ok=True)
    summary = _load_summary(state_dir)
    checkpoint_index, checkpoint_key = _next_checkpoint_key(
        summary,
        "block",
        block_name=block_name,
    )
    saved_examples, rejected_examples = _load_recent_candidates(state_dir)
    agent_state = load_market_intel_agent_state(brief_path=brief_path, state_dir=state_dir)
    packet = LiveCheckpointPacket(
        checkpoint_key=checkpoint_key,
        checkpoint_scope="block",
        checkpoint_index=checkpoint_index,
        created_at=_utc_now(),
        brief_id=brief_id,
        block_name=block_name,
        recent_runtime_events=_load_recent_events(state_dir),
        recent_saved_examples=saved_examples,
        recent_rejected_examples=rejected_examples,
        active_hypotheses=[
            item.to_dict() for item in (agent_state.active_hypotheses if agent_state else [])
        ][:5],
        prior_advisories=[
            item.to_dict() for item in (agent_state.prior_advisories if agent_state else [])
        ][:5],
        payload={
            "strings_run": int(getattr(block_report, "strings_run", 0) or 0),
            "strings_with_saves": int(getattr(block_report, "strings_with_saves", 0) or 0),
            "total_saves": int(getattr(block_report, "total_saves", 0) or 0),
            "top_performers": getattr(block_report, "top_performers", [])[:3],
            "zero_save_count": len(getattr(block_report, "zero_save_string_ids", []) or []),
            "search_intelligence_summary": getattr(block_report, "search_intelligence_summary", {}) or {},
            "search_memory_summary": search_memory_summary or {},
        },
    )
    append_jsonl(live_dir_paths["observations"], packet.to_dict())
    new_advisories = _heuristic_new_advisories(packet)
    for advisory in new_advisories:
        append_jsonl(live_dir_paths["advisories"], advisory.to_dict())
    active = _load_active_advisories(
        state_dir=state_dir,
        checkpoint_index=checkpoint_index,
        checkpoint_scope="block",
    )
    summary["checkpoint_index"] = checkpoint_index
    summary["last_context"] = _render_context(active)
    _write_summary(state_dir, summary)
    _record_consumed(
        state_dir=state_dir,
        advisories=active,
        checkpoint_scope="block",
        checkpoint_key=checkpoint_key,
    )
    return summary["last_context"]
