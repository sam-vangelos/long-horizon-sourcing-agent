"""Block-adaptation helpers for LinkedIn sourcing runs.

Owns lint gating, search-string metadata hydration, saved-profile snapshot
projection, and exploitation-bias overlay for block adaptation. ``Pipeline``
delegates to ``BlockAdaptationService``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from linkedin.boolean_compiler import lint_generated_string
from shared import config
from shared.schemas import AdaptationResponse, Progress, SearchString
from shared.search_memory import infer_domain_lane, normalize_family_key, normalize_novelty_bucket
from shared.storage import log_event
from shared.strict_seniority import classify_search_string_seniority, is_strict_seniority_brief


@dataclass(frozen=True)
class BlockAdaptationDeps:
    log_path: Path
    get_brief_obj: Callable[[], Any]
    get_lint_blocked_strings: Callable[[], list[dict]]
    ensure_services: Callable[[], None]
    get_work_unit_service: Callable[[], Any]
    set_search_memory: Callable[[dict], None]
    normalize_candidate_name_key: Callable[[str], str]


class BlockAdaptationService:
    """Owns block-adaptation lint, metadata, and exploitation overlay helpers."""

    def __init__(self, deps: BlockAdaptationDeps):
        self.deps = deps

    def _record_lint_blocked(
        self,
        item: dict[str, Any],
        *,
        boolean_key: str,
        source: str,
        codes: list[str],
        messages: list[str],
        repair_hints: list[str],
    ) -> None:
        """Record a string refused at queue build (P5: errors block, warnings inform).

        Prints the repair hints (the operator-actionable half of the finding),
        appends to the in-memory list the run report renders, and logs the
        durable search_string_lint_blocked event.
        """
        record = {
            "source": source,
            "name": str(item.get("rationale") or item.get("gap") or "")[:80],
            "family_key": str(item.get("family_key") or ""),
            "boolean": str(item.get(boolean_key) or ""),
            "codes": codes,
            "messages": messages,
            "repair_hints": [hint for hint in repair_hints if hint],
        }
        self.deps.get_lint_blocked_strings().append(record)
        label = record["family_key"] or record["name"] or "unnamed"
        print(
            f"  [lint-blocked] {source} string ({label}) refused at queue build: "
            f"{', '.join(codes)}"
        )
        for hint in record["repair_hints"]:
            print(f"    repair: {hint}")
        log_event(self.deps.log_path, "search_string_lint_blocked", **record)

    def _queue_lint_gate(
        self,
        work_item: dict[str, Any],
        *,
        boolean_key: str,
        source: str,
        lint_context,
    ) -> dict | None:
        """Lint one queue candidate. Returns the lint report dict to attach to
        the SearchString, or None when the string must NOT queue (error-severity
        findings). Empty booleans are legitimate (structured_only lanes) and
        pass untouched."""
        if not str(work_item.get(boolean_key) or "").strip():
            return {}
        report = lint_generated_string(
            work_item, context=lint_context, boolean_key=boolean_key
        )
        if report.has_error:
            errors = [f for f in report.findings if f.severity == "error"]
            self._record_lint_blocked(
                work_item,
                boolean_key=boolean_key,
                source=source,
                codes=[f.code for f in errors],
                messages=[f.message for f in errors],
                repair_hints=[f.repair_hint for f in errors],
            )
            return None
        return report.to_dict() if report.findings else {}

    def _update_search_memory_from_block(self, block_strings: list[SearchString]) -> None:
        """Update family memory with the completed block's observed performance."""
        self.deps.ensure_services()
        self.deps.set_search_memory(
            self.deps.get_work_unit_service().update_search_memory_from_block(block_strings)
        )

    def _hydrate_search_string_metadata(self, search_string: SearchString) -> None:
        """Backfill family metadata for older progress files or unlabeled strings."""
        retrieval_recipe = (
            search_string.retrieval_recipe
            if isinstance(search_string.retrieval_recipe, dict)
            else {}
        )
        if not search_string.family_key:
            search_string.family_key = normalize_family_key(
                retrieval_recipe.get("family_id"),
                search_string.boolean,
                search_string.name,
            )
        if not search_string.novelty_bucket:
            search_string.novelty_bucket = normalize_novelty_bucket(
                "edge_case" if search_string.retrieval_hypothesis_ids or retrieval_recipe.get("applied_hypothesis_ids") else None,
                search_string.boolean,
                search_string.name,
                brief=self.deps.get_brief_obj(),
            )
        if not search_string.domain_lane:
            search_string.domain_lane = infer_domain_lane(
                (retrieval_recipe.get("target_markets") or [None])[0],
                search_string.boolean,
                search_string.name,
                brief=self.deps.get_brief_obj(),
            )
        if not search_string.retrieval_hypothesis_ids and retrieval_recipe:
            search_string.retrieval_hypothesis_ids = [
                str(hypothesis_id).strip()
                for hypothesis_id in retrieval_recipe.get("applied_hypothesis_ids", [])
                if str(hypothesis_id).strip()
            ]
        if is_strict_seniority_brief(self.deps.get_brief_obj()):
            # BFSI-era risk stamps are scoped to strict-seniority briefs; on
            # every other brief the fields stay neutral so the exploitation
            # overlay's opening_eligible/seniority_risk demotions cannot fire
            # exec-search heuristics against it.
            risk = classify_search_string_seniority(
                search_string.boolean,
                search_string.name,
                domain_lane=search_string.domain_lane,
            )
            if not search_string.seniority_risk:
                search_string.seniority_risk = risk["seniority_risk"]
            if not search_string.title_bucket_risk:
                search_string.title_bucket_risk = risk["title_bucket_risk"]
            if search_string.opening_eligible is None:
                search_string.opening_eligible = bool(risk["opening_eligible"])

    def _saved_profile_snapshots(
        self,
        saved_names: list[str],
        profile_index: dict[str, dict],
    ) -> list[dict]:
        """Small LinkedIn-facing snapshots so adaptation can judge novelty, not just counts."""
        snapshots: list[dict] = []
        seen: set[str] = set()

        for name in saved_names:
            key = self.deps.normalize_candidate_name_key(name)
            if not key or key in seen:
                continue
            entry = profile_index.get(key)
            if not entry:
                continue
            seen.add(key)
            current = (entry.get("experiences") or [{}])[0] or {}
            snapshots.append(
                {
                    "name": entry.get("name", name),
                    "title": current.get("title", ""),
                    "company": current.get("company", ""),
                    "headline": entry.get("headline", ""),
                }
            )

        return snapshots

    @staticmethod
    def _reprioritize_new_strings_for_exploitation(
        new_strings: list[dict[str, Any]],
        *,
        proven_family_keys: set[str],
        proven_domain_lanes: set[str],
    ) -> list[dict[str, Any]]:
        annotated: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
        for index, item in enumerate(new_strings):
            family_match = item.get("family_key", "") in proven_family_keys
            lane_match = item.get("domain_lane", "") in proven_domain_lanes
            novelty_bucket = item.get("novelty_bucket", "")
            priority = (
                0 if family_match else 1 if lane_match else 2,
                0 if novelty_bucket == "edge_case" else 1,
                index,
            )
            annotated.append((priority, item))
        annotated.sort(key=lambda pair: pair[0])
        return [item for _, item in annotated]

    @staticmethod
    def _merge_reorder_actions(
        base_actions: list[dict[str, Any]],
        overlay_actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        sequence = 0

        for item in list(base_actions) + list(overlay_actions):
            try:
                string_id = int(item.get("string_id"))
            except Exception:
                continue
            candidate = dict(item)
            candidate["_priority"] = int(candidate.get("_priority", 0) or 0)
            candidate["_sequence"] = sequence
            sequence += 1
            existing = merged.get(string_id)
            if existing is None:
                merged[string_id] = candidate
                continue

            existing_priority = int(existing.get("_priority", 0) or 0)
            candidate_priority = int(candidate.get("_priority", 0) or 0)
            if candidate_priority > existing_priority:
                merged[string_id] = candidate
                continue
            if (
                candidate_priority == existing_priority
                and existing.get("move_to") != "next"
                and candidate.get("move_to") == "next"
            ):
                merged[string_id] = candidate

        next_actions = sorted(
            (
                item
                for item in merged.values()
                if item.get("move_to") == "next"
            ),
            key=lambda item: (-int(item.get("_priority", 0) or 0), int(item.get("_sequence", 0) or 0)),
        )
        last_actions = sorted(
            (
                item
                for item in merged.values()
                if item.get("move_to") != "next"
            ),
            key=lambda item: (-int(item.get("_priority", 0) or 0), int(item.get("_sequence", 0) or 0)),
        )

        ordered = next_actions + last_actions
        for item in ordered:
            item.pop("_priority", None)
            item.pop("_sequence", None)
        return ordered

    def _apply_exploitation_bias_to_adaptation(
        self,
        *,
        adaptation: AdaptationResponse,
        remaining: list[SearchString],
        block_summary: dict[str, Any],
        checkpoint_mode: str,
    ) -> dict[str, Any]:
        proven_family_keys = set(block_summary.get("proven_family_keys", []))
        proven_domain_lanes = set(block_summary.get("proven_domain_lanes", []))
        dead_family_keys = set(block_summary.get("dead_family_keys", [])) - proven_family_keys
        contaminated_family_keys = set(block_summary.get("contaminated_family_keys", []))
        contaminated_domain_lanes = set(block_summary.get("contaminated_domain_lanes", []))

        def _lane_spellings(ss: SearchString) -> set[str]:
            # P7 Stage B remaps model-emitted lanes to declared spellings at
            # annotation time; strings queued before the remap (a resumed
            # pre-remap plan) keep the raw spelling. Membership here must
            # match EITHER spelling or a proven lane's history silently
            # splits across the resume boundary (contract lens, slice 12).
            return {
                lane
                for lane in (
                    getattr(ss, "domain_lane", "") or "",
                    getattr(ss, "domain_lane_raw", "") or "",
                )
                if lane
            }
        if (
            not proven_family_keys
            and not proven_domain_lanes
            and not dead_family_keys
            and not contaminated_family_keys
            and not contaminated_domain_lanes
        ):
            return {
                "promoted_string_ids": [],
                "demoted_string_ids": [],
                "reprioritized_new_strings": False,
            }

        if adaptation.new_strings:
            adaptation.new_strings = self._reprioritize_new_strings_for_exploitation(
                adaptation.new_strings,
                proven_family_keys=proven_family_keys,
                proven_domain_lanes=proven_domain_lanes,
            )

        overlay_actions: list[dict[str, Any]] = []
        skip_ids = {
            int(item.get("string_id"))
            for item in adaptation.skip_remaining
            if item.get("string_id") is not None
        }
        promoted_string_ids: list[int] = []
        demoted_string_ids: list[int] = []

        promotion_limit = max(1, config.SEARCH_INTELLIGENCE_EXPLOIT_PROMOTION_LIMIT)
        demotion_limit = max(1, config.SEARCH_INTELLIGENCE_EXPLOIT_DEMOTION_LIMIT)

        for search_string in remaining:
            if search_string.id in skip_ids:
                continue
            self._hydrate_search_string_metadata(search_string)

            if (
                len(promoted_string_ids) < promotion_limit
                and search_string.family_key
                and search_string.family_key in proven_family_keys
                and search_string.family_key not in contaminated_family_keys
                and search_string.seniority_risk != "high"
            ):
                overlay_actions.append(
                    {
                        "string_id": search_string.id,
                        "move_to": "next",
                        "reason": "Promoted because this exact family is already producing saves in the live run.",
                        "_priority": 300,
                    }
                )
                promoted_string_ids.append(search_string.id)
                continue

            if (
                len(promoted_string_ids) < promotion_limit
                and search_string.domain_lane
                and _lane_spellings(search_string) & proven_domain_lanes
                and search_string.family_key not in dead_family_keys
                and not (_lane_spellings(search_string) & contaminated_domain_lanes)
                and search_string.seniority_risk != "high"
            ):
                overlay_actions.append(
                    {
                        "string_id": search_string.id,
                        "move_to": "next",
                        "reason": (
                            "Promoted because this lane is already working and should get more runway before colder hypotheses."
                        ),
                        "_priority": 200 if search_string.novelty_bucket == "edge_case" else 150,
                    }
                )
                promoted_string_ids.append(search_string.id)
                continue

            if (
                len(demoted_string_ids) < demotion_limit
                and search_string.family_key
                and search_string.family_key in dead_family_keys
            ):
                overlay_actions.append(
                    {
                        "string_id": search_string.id,
                        "move_to": "last",
                        "reason": (
                            "Demoted because this family already ran cold while other families are producing real signal."
                        ),
                        "_priority": 260,
                    }
                )
                demoted_string_ids.append(search_string.id)
                continue

            if (
                len(demoted_string_ids) < demotion_limit
                and (
                    search_string.family_key in contaminated_family_keys
                    or _lane_spellings(search_string) & contaminated_domain_lanes
                    or search_string.seniority_risk == "high"
                    or search_string.opening_eligible is False
                )
            ):
                overlay_actions.append(
                    {
                        "string_id": search_string.id,
                        "move_to": "last",
                        "reason": (
                            "Demoted because early signal from this family/lane looks contaminated by above-band seniority drift or broad title buckets."
                        ),
                        "_priority": 320,
                    }
                )
                demoted_string_ids.append(search_string.id)
                continue

        if checkpoint_mode == "opening_checkpoint" and proven_domain_lanes:
            for search_string in remaining:
                if (
                    len(demoted_string_ids) >= demotion_limit
                    or search_string.id in skip_ids
                    or search_string.id in demoted_string_ids
                    or search_string.id in promoted_string_ids
                ):
                    continue
                self._hydrate_search_string_metadata(search_string)
                if (
                    search_string.novelty_bucket == "canonical"
                    and search_string.domain_lane
                    and not (_lane_spellings(search_string) & proven_domain_lanes)
                ):
                    overlay_actions.append(
                        {
                            "string_id": search_string.id,
                            "move_to": "last",
                            "reason": (
                                "Demoted because the opening checkpoint already found a productive lane elsewhere and this looks like lower-leverage cleanup."
                            ),
                            "_priority": 120,
                        }
                    )
                    demoted_string_ids.append(search_string.id)
                    if len(demoted_string_ids) >= demotion_limit:
                        break
                elif search_string.opening_eligible is False:
                    overlay_actions.append(
                        {
                            "string_id": search_string.id,
                            "move_to": "last",
                            "reason": (
                                "Demoted because this string is not opening-safe for a strict-seniority brief."
                            ),
                            "_priority": 180,
                        }
                    )
                    demoted_string_ids.append(search_string.id)
                    if len(demoted_string_ids) >= demotion_limit:
                        break

        adaptation.reorder = self._merge_reorder_actions(adaptation.reorder, overlay_actions)

        return {
            "promoted_string_ids": promoted_string_ids,
            "demoted_string_ids": demoted_string_ids,
            "reprioritized_new_strings": bool(adaptation.new_strings),
        }

    @staticmethod
    def _apply_reorder_actions(progress: Progress, reorder_actions: list[dict[str, Any]]) -> None:
        next_actions = [action for action in reorder_actions if action.get("move_to") == "next"]
        last_actions = [action for action in reorder_actions if action.get("move_to") != "next"]

        for action in reversed(next_actions):
            string_id = action.get("string_id")
            for index, search_string in enumerate(progress.strings):
                if search_string.id == string_id and search_string.status == "queued":
                    progress.strings.pop(index)
                    for insert_index, queued in enumerate(progress.strings):
                        if queued.status == "queued":
                            progress.strings.insert(insert_index, search_string)
                            break
                    else:
                        progress.strings.append(search_string)
                    break

        for action in last_actions:
            string_id = action.get("string_id")
            for index, search_string in enumerate(progress.strings):
                if search_string.id == string_id and search_string.status == "queued":
                    progress.strings.pop(index)
                    progress.strings.append(search_string)
                    break

    def _clear_pending_block_adaptation(self, progress: Progress | None) -> None:
        """Clear any persisted block-adaptation checkpoint."""
        self.deps.ensure_services()
        self.deps.get_work_unit_service().clear_pending_block_adaptation(progress)
