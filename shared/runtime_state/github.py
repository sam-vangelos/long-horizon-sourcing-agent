"""GitHub-specific runtime-state bridge."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from github.schemas import GitHubProgress

from shared.person_identity import IdentityEvidence, merge_candidates
from shared.runtime_state.admin import rebuild_compat_projections
from shared.runtime_state.store import RuntimeStateStore


def github_person_key(username: str) -> str:
    """Derive the canonical cross-hub person key for a GitHub login."""

    if not username:
        return ""
    if username.startswith("gh:"):
        return username.lower()
    keys = merge_candidates([IdentityEvidence(hub="github", handle=username)])
    return keys[0].key


def normalize_to_person_key(value: str) -> str:
    """Map a legacy username or person-key string to a canonical person key."""

    if not value:
        return ""
    if value.startswith("gh:"):
        return value.lower()
    return github_person_key(value)


class PersonKeySet:
    """In-memory dedup set keyed by canonical person keys."""

    def __init__(self, values: Iterable[str] = ()) -> None:
        self._keys: set[str] = set()
        self._identity_by_key: dict[str, str] = {}
        for value in values:
            self.add(value)

    def add(self, value: str) -> None:
        if not value:
            return
        key = normalize_to_person_key(value)
        if not key:
            return
        self._keys.add(key)
        if key not in self._identity_by_key:
            if value.startswith("gh:"):
                self._identity_by_key[key] = key[3:]
            else:
                self._identity_by_key[key] = value

    def identity_keys(self) -> list[str]:
        """Return GitHub identity_key strings preserving first-seen casing."""

        return sorted(self._identity_by_key[key] for key in self._keys)

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str) or not item:
            return False
        return normalize_to_person_key(item) in self._keys

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __bool__(self) -> bool:
        return bool(self._keys)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PersonKeySet):
            return self._keys == other._keys
        if isinstance(other, set):
            return self._keys == {normalize_to_person_key(item) for item in other}
        return NotImplemented


def person_key_seen(username: str, seen: PersonKeySet | set[str]) -> bool:
    """Return whether ``username`` is already represented in ``seen``."""

    person_key = github_person_key(username)
    if isinstance(seen, PersonKeySet):
        return person_key in seen
    return any(normalize_to_person_key(item) == person_key for item in seen)


def github_identity_keys_from_seen(seen: PersonKeySet | set[str]) -> list[str]:
    """Project an in-memory dedup set to GitHub ``identity_key`` strings."""

    if isinstance(seen, PersonKeySet):
        return seen.identity_keys()
    return sorted(seen)


def _resolve_recruiter_id() -> int | None:
    """Resolve the acting recruiter for a run-start, fail-soft to None.

    reopen Stage 2 (R5a-3): stamps ``runs.recruiter_id`` so the read-only
    taste aggregator (R5a-4) can attribute adaptation decisions. The
    resolver is the single auth seam (``shared.recruiter_context``); we
    catch broadly because a run launch must never die on recruiter
    resolution — a None recruiter_id is a clean "unknown" (the aggregator
    skips it), whereas a raised exception here would abort the run.
    """

    try:
        from shared.recruiter_context import get_current_recruiter_id

        return get_current_recruiter_id()
    except Exception:  # noqa: BLE001 — resolution must never break a run launch
        return None


class GitHubRuntimeStateBridge:
    """Keeps GitHub progress/resume semantics DB-authoritative."""

    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        output_dir: str | Path,
        brief_id: str,
        brief_name: str,
        brief_path: str | None = None,
    ):
        self.store = store
        self.output_dir = Path(output_dir)
        self.brief_id = brief_id
        self.brief_name = brief_name
        self.brief_path = brief_path
        self.progress_path = self.output_dir / "progress.json"

    def has_runtime_state(self) -> bool:
        latest_run = self.store.get_latest_run(source="github", brief_id=self.brief_id)
        return bool(latest_run or self.store.has_candidates(source="github", brief_id=self.brief_id))

    def start_or_resume_run(
        self,
        *,
        resume: bool,
        initial_progress: GitHubProgress | None = None,
    ) -> tuple[int, GitHubProgress]:
        self.store.reconcile_open_attempts(source="github", brief_id=self.brief_id)
        self.store.reconcile_pending_side_effects(source="github", brief_id=self.brief_id)
        latest_run = self.store.get_latest_run(source="github", brief_id=self.brief_id)

        # Phase 3: pin the brief identity on the new run row so Run
        # Review and brief-drift detection have stable references.
        from shared.brief_identity import compute_brief_identity

        identity = (
            compute_brief_identity(self.brief_path) if self.brief_path else None
        )
        identity_kwargs: dict = {}
        if identity is not None:
            identity_kwargs = {
                "brief_path_at_launch": identity["brief_path_at_launch"],
                "brief_content_hash": identity["brief_content_hash"],
                "brief_snapshot_json": identity["brief_snapshot_json"],
            }

        recruiter_id = _resolve_recruiter_id()

        if resume and latest_run and self.store.has_work_units(int(latest_run["id"])):
            run_id = self.store.start_run(
                source="github",
                brief_id=self.brief_id,
                output_dir=str(self.output_dir),
                mode="resume",
                resume_state=self.store.get_run_resume_state(int(latest_run["id"])),
                resumed_from_run_id=int(latest_run["id"]),
                clone_work_units_from_run_id=int(latest_run["id"]),
                recruiter_id=recruiter_id,
                **identity_kwargs,
            )
            self.rebuild_artifacts(run_id)
            return run_id, self.store.load_github_progress(run_id)

        run_id = self.store.start_run(
            source="github",
            brief_id=self.brief_id,
            output_dir=str(self.output_dir),
            mode="resume" if resume else "fresh",
            resume_state={"brief_name": self.brief_name},
            resumed_from_run_id=int(latest_run["id"]) if resume and latest_run else None,
            recruiter_id=recruiter_id,
            **identity_kwargs,
        )

        if initial_progress is not None:
            self.sync_progress(run_id, initial_progress)
            return run_id, self.store.load_github_progress(run_id)

        progress = GitHubProgress(brief_name=self.brief_name)
        self.sync_progress(run_id, progress)
        return run_id, progress

    def sync_progress(self, run_id: int, progress: GitHubProgress) -> None:
        self.store.sync_github_progress(run_id, progress)
        self.rebuild_artifacts(run_id)

    def load_progress(self, run_id: int) -> GitHubProgress:
        return self.store.load_github_progress(run_id)

    def load_blocked_usernames(self, usernames: list[str]) -> set[str]:
        return self.store.get_github_blocked_usernames(self.brief_id, usernames)

    def load_blocked_person_keys(self, person_keys: list[str]) -> set[str]:
        return self.store.get_blocked_person_keys(self.brief_id, person_keys)

    def rebuild_artifacts(self, run_id: int) -> None:
        rebuild_compat_projections(
            self.store,
            run_id=run_id,
            output_dir=self.output_dir,
        )

    def record_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any] | None = None,
        run_id: int | None = None,
        work_unit_id: int | None = None,
    ) -> None:
        self.store.record_event(
            run_id=run_id,
            work_unit_id=work_unit_id,
            event_type=event_type,
            payload=payload,
        )
