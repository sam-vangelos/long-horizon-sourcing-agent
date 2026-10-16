"""LinkedIn work-unit coordination service."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable

from shared.execution import WorkUnitCheckpoint
from shared.schemas import Progress, SearchString


logger = logging.getLogger(__name__)


@dataclass
class LinkedInWorkUnitDeps:
    ensure_runtime_state: Callable[[], None]
    get_runtime_bridge: Callable[[], Any]
    get_runtime_run_id: Callable[[], int | None]
    set_runtime_run_id: Callable[[int | None], None]
    get_progress: Callable[[], Progress | None]
    set_progress: Callable[[Progress | None], None]
    stats: dict[str, int]
    get_experiment_states: Callable[[], dict[int, Any]]
    set_experiment_states: Callable[[dict[int, Any]], None]
    get_search_memory: Callable[[], dict]
    set_search_memory: Callable[[dict], None]
    reset_restart_history: Callable[[], None]


class LinkedInWorkUnitService:
    """Owns LinkedIn progress, search-memory, and restart semantics."""

    def __init__(self, deps: LinkedInWorkUnitDeps):
        self.deps = deps

    def load_search_memory(self) -> dict:
        runtime_bridge = self.deps.get_runtime_bridge()
        if not runtime_bridge or not runtime_bridge.has_runtime_state():
            return {}
        memory = runtime_bridge.load_search_memory()
        families = len(memory.get("families", {}))
        print(f"  [memory] Loaded runtime search memory ({families} families)")
        return memory

    def save_search_memory(self) -> dict:
        runtime_bridge = self.deps.get_runtime_bridge()
        if not runtime_bridge:
            return self.deps.get_search_memory()
        run_id = self._ensure_runtime_run(progress=self.deps.get_progress())
        runtime_bridge.rebuild_artifacts(run_id)
        return runtime_bridge.load_search_memory()

    def update_search_memory_from_block(self, block_strings: list["SearchString"]) -> dict:
        runtime_bridge = self.deps.get_runtime_bridge()
        progress = self.deps.get_progress()
        if not runtime_bridge:
            return self.deps.get_search_memory()
        if self.deps.get_runtime_run_id() is None and progress is None:
            return self.deps.get_search_memory()
        run_id = self._ensure_runtime_run(progress=progress)
        if progress:
            runtime_bridge.sync_progress(
                run_id,
                progress,
                experiment_states=self.deps.get_experiment_states(),
            )
        return runtime_bridge.load_search_memory()

    def checkpoint_progress(
        self,
        progress: "Progress" | None,
        *,
        search_string: "SearchString" | None = None,
        page_num: int | None = None,
        timings: dict[str, float] | None = None,
    ) -> WorkUnitCheckpoint | None:
        if not progress:
            return None

        if search_string is not None:
            progress.current_string_id = search_string.id
            if page_num is not None:
                progress.current_page = page_num
                search_string.pages_reviewed = max(search_string.pages_reviewed, page_num)

        progress.candidates_saved = self.deps.stats.get("saved", progress.candidates_saved)
        progress.candidates_rejected = self.deps.stats.get("rejected", progress.candidates_rejected)
        run_id = self._ensure_runtime_run(progress=progress)
        runtime_bridge = self._require_runtime_bridge()
        sync_kwargs = {
            "experiment_states": self.deps.get_experiment_states(),
        }
        if timings is not None:
            sync_kwargs["timings"] = timings
        runtime_bridge.sync_progress(run_id, progress, **sync_kwargs)
        memory_started = time.monotonic()
        try:
            self.deps.set_search_memory(runtime_bridge.load_search_memory())
        except Exception as exc:
            # SQLite sync above is the durability boundary. Search memory is
            # a derived read model and must not make a committed page appear
            # uncommitted to the orchestrator.
            logger.warning(
                "LinkedIn search-memory refresh failed after sync (%s)",
                type(exc).__name__,
            )
        finally:
            if timings is not None:
                timings["search_memory_reload_ms"] = round(
                    (time.monotonic() - memory_started) * 1000.0,
                    3,
                )
        return WorkUnitCheckpoint(
            status="synced",
            cursor={
                "current_string_id": progress.current_string_id,
                "current_page": progress.current_page,
            },
            metrics={
                "candidates_saved": progress.candidates_saved,
                "candidates_rejected": progress.candidates_rejected,
            },
        )

    def set_pending_block_adaptation(
        self,
        progress: "Progress" | None,
        block_name: str,
        block_strings: list["SearchString"],
        *,
        ready: bool,
    ) -> None:
        if not progress:
            return
        progress.pending_block_name = block_name
        progress.pending_block_string_ids = [search_string.id for search_string in block_strings]
        progress.pending_block_ready = ready

    def clear_pending_block_adaptation(self, progress: "Progress" | None) -> None:
        if not progress:
            return
        progress.pending_block_name = ""
        progress.pending_block_string_ids = []
        progress.pending_block_ready = False

    def load_or_create_progress(self):
        self.deps.ensure_runtime_state()
        runtime_bridge = self._require_runtime_bridge()
        existing_states = dict(self.deps.get_experiment_states())
        if (
            runtime_bridge.has_runtime_state()
            or runtime_bridge.has_legacy_state()
        ):
            runtime_run_id, progress = runtime_bridge.start_or_resume_run(resume=True)
            self.deps.set_runtime_run_id(runtime_run_id)
            loaded_states = runtime_bridge.load_experiment_states(
                runtime_run_id,
                progress=progress,
            )
            # SQLite checkpoint state is canonical on resume. Keep any
            # in-memory-only IDs, but never let a stale object replace the
            # state loaded for the same persisted work unit.
            self.deps.set_experiment_states({**existing_states, **loaded_states})
            return progress

        raise RuntimeError(
            "LinkedIn resume requires canonical or legacy runtime state"
        )

    def restart_string(self, progress: "Progress", string_id: int) -> None:
        runtime_bridge = self.deps.get_runtime_bridge()
        if not runtime_bridge:
            raise RuntimeError("LinkedIn runtime bridge is required for restart semantics")
        run_id = self._ensure_runtime_run(progress=progress, prefer_resume=True)
        runtime_bridge.sync_progress(
            run_id,
            progress,
            experiment_states=self.deps.get_experiment_states(),
        )
        runtime_bridge.restart_string(
            run_id=run_id,
            progress=progress,
            string_id=string_id,
        )
        self.deps.set_experiment_states(runtime_bridge.load_experiment_states(
            run_id,
            progress=progress,
        ))
        self.deps.reset_restart_history()
        self.deps.set_search_memory(self.load_search_memory())
        print(f"  String #{string_id} reset to page 1. Ready for re-evaluation.")

    def restart_strings(self, progress: "Progress", string_ids: list[int]) -> None:
        seen_ids: set[int] = set()
        ordered_ids: list[int] = []
        for string_id in string_ids:
            if string_id in seen_ids:
                continue
            seen_ids.add(string_id)
            ordered_ids.append(string_id)
        for string_id in ordered_ids:
            self.restart_string(progress, string_id)

    def _ensure_runtime_run(
        self,
        *,
        progress: "Progress" | None,
        prefer_resume: bool = False,
    ) -> int:
        runtime_run_id = self.deps.get_runtime_run_id()
        if runtime_run_id:
            return int(runtime_run_id)
        runtime_bridge = self.deps.get_runtime_bridge()
        if not runtime_bridge:
            raise RuntimeError("runtime_state is required for LinkedIn work-unit operations")
        existing_states = dict(self.deps.get_experiment_states())
        if prefer_resume and (runtime_bridge.has_runtime_state() or runtime_bridge.has_legacy_state()):
            runtime_run_id, loaded_progress = runtime_bridge.start_or_resume_run(resume=True)
            self.deps.set_runtime_run_id(runtime_run_id)
            loaded_states = runtime_bridge.load_experiment_states(
                runtime_run_id,
                progress=loaded_progress,
            )
            self.deps.set_experiment_states({**existing_states, **loaded_states})
            if self.deps.get_progress() is None:
                self.deps.set_progress(loaded_progress)
            return int(runtime_run_id)
        if progress is None:
            raise RuntimeError("runtime_state run has not been initialized")
        runtime_run_id, loaded_progress = runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
            experiment_states=self.deps.get_experiment_states(),
        )
        self.deps.set_runtime_run_id(runtime_run_id)
        loaded_states = runtime_bridge.load_experiment_states(
            runtime_run_id,
            progress=loaded_progress,
        )
        # This is a fresh run, not a resume: the persisted states were just
        # created from the in-memory objects passed to start_or_resume_run.
        self.deps.set_experiment_states({**loaded_states, **existing_states})
        if self.deps.get_progress() is None:
            self.deps.set_progress(loaded_progress)
        return int(runtime_run_id)

    def _require_runtime_bridge(self) -> Any:
        runtime_bridge = self.deps.get_runtime_bridge()
        if not runtime_bridge:
            raise RuntimeError("runtime_state is required for LinkedIn work-unit operations")
        return runtime_bridge
