"""Advisory lock for single-writer runtime state access."""

from __future__ import annotations

import fcntl
from pathlib import Path


class RuntimeStateLock:
    """Best-effort per-output-directory advisory lock."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        filename: str = "runtime_state.lock",
        resource_name: str = "runtime state",
    ):
        self.path = Path(output_dir) / filename
        self.resource_name = resource_name
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                f"{self.resource_name} is already locked by another process"
            ) from exc
        self._handle = handle

    def release(self) -> None:
        if not self._handle:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "RuntimeStateLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
