"""Compressed real-time console output.

Replaces ~30 orchestrator print statements with structured, scannable output.
Only prints: query headers, saves, query summaries, adaptations, graph expansions, errors.
"""

from __future__ import annotations

import time


def mask_email(email: str | None) -> str | None:
    if not email or "@" not in email:
        return email
    local, domain = email.split("@", 1)
    return f"{local[0]}***@{domain}" if local and domain else email


class ConsoleOutput:
    def __init__(self):
        self._session_start = time.time()

    def _ts(self) -> str:
        return time.strftime("%H:%M")

    def emit_session_start(self, brief_id: str, query_count: int):
        print(f"[{self._ts()}] GitHub sourcing — {brief_id} | {query_count} queries")

    def emit_query_start(self, query, query_num: int, total: int, saves: int, save_rate: str, api_remaining: int):
        print(f"\n[{self._ts()}] Q {query_num}/{total} | {query.name} [{query.channel}]")
        print(f"        Saves: {saves} | Rate: {save_rate} | API: {api_remaining} rest")

    def emit_save(self, username: str, name: str, confidence: float, decision_type: str, capability_area: str, contact_str: str):
        display_name = name or username
        print(f"[{self._ts()}] + SAVE {display_name} ({confidence:.2f} {decision_type}:{capability_area}) {contact_str}")

    def emit_query_end(self, query, stats: dict):
        found = stats.get("found", 0)
        geo = stats.get("geo_filtered", 0)
        insuff = stats.get("insufficient", 0)
        fno = stats.get("facial_no", 0)
        fyes = stats.get("facial_yes", 0)
        saved = stats.get("saved", 0)
        rejected = stats.get("rejected", 0)
        print(f"[{self._ts()}] Q {query.id} done: {found} found → {geo} geo → {insuff} insuff → {fno} facial_no → {fyes} yes → {saved} save, {rejected} reject")

    def emit_adaptation(self, new_count: int, skipped_count: int, rationale: str):
        short_rationale = rationale[:80] if rationale else ""
        print(f'[{self._ts()}] ◆ ADAPT +{new_count} queries, {skipped_count} skipped | "{short_rationale}"')

    def emit_graph_expansion(self, seed_count: int, new_query_count: int):
        print(f"[{self._ts()}] ◆ GRAPH {seed_count} expansions, +{new_query_count} queries")

    def emit_query_stopped(self, query, reason: str, processed: int, total: int):
        print(f"[{self._ts()}] Q {query.id} stopped at {processed}/{total}: {reason}")

    def emit_enrichment_only(self):
        print(f"[{self._ts()}] ⚠ Low API budget — entering enrichment-only mode")

    def emit_stop_recommendation(self, recommendation: str):
        if recommendation.startswith("STOP"):
            print(f"[{self._ts()}] ⚠ {recommendation}")

    def emit_info(self, msg: str):
        print(f"[{self._ts()}] {msg}")

    def emit_warn(self, msg: str):
        print(f"[{self._ts()}] ⚠ {msg}")

    def emit_error(self, context: str, error: str):
        print(f"[{self._ts()}] ✗ [{context}] {error}")

    def emit_session_end(self, stats: dict, duration_seconds: float):
        m = int(duration_seconds // 60)
        s = int(duration_seconds % 60)
        geo_total = stats.get('geo_filtered', 0)
        print(f"\n{'─' * 50}")
        print(f"[{self._ts()}] Session complete ({m}m{s}s)")
        print(f"  Discovered: {stats.get('candidates_discovered', 0)} | "
              f"Enriched: {stats.get('candidates_enriched', 0)} | "
              f"Geo-filtered: {geo_total}")
        print(f"  Facial YES: {stats.get('facial_yes', 0)} | "
              f"Facial NO: {stats.get('facial_no', 0)} | "
              f"Insufficient: {stats.get('insufficient', 0)}")
        saves = stats.get('saved', 0)
        fy = stats.get('facial_yes', 0)
        rate = f"{saves / fy * 100:.1f}%" if fy > 0 else "n/a"
        print(f"  SAVED: {saves} | REJECTED: {stats.get('rejected', 0)} | "
              f"Save rate: {rate}")
