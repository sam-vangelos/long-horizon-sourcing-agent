"""Mirror stdout/stderr to a live console log file."""

from __future__ import annotations

import atexit
import sys
import threading
from pathlib import Path
from typing import TextIO

MAX_CONSOLE_LOG_BYTES = 50 * 1024 * 1024

_installed = False
_cleanup = None


class _TeeStream:
    """Write stream data to both the original stream and a log file."""

    def __init__(self, original: TextIO, mirror: TextIO):
        self._original = original
        self._mirror = mirror
        self._lock = threading.Lock()
        self.encoding = getattr(original, "encoding", "utf-8")
        self.errors = getattr(original, "errors", "strict")

    def write(self, data: str) -> int:
        with self._lock:
            written = self._original.write(data)
            self._mirror.write(data)
            return written

    def flush(self) -> None:
        with self._lock:
            self._original.flush()
            self._mirror.flush()

    def isatty(self) -> bool:
        return self._original.isatty()

    def fileno(self) -> int:
        return self._original.fileno()

    def writable(self) -> bool:
        return True


def enable_console_tee(output_dir: str | Path, filename: str = "live-console.log") -> Path:
    """Mirror stdout/stderr into a file in the output directory.

    This is process-global. Repeated calls are ignored after the first install.
    """
    global _installed, _cleanup
    if _installed:
        return Path(output_dir) / filename

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.exists() and output_path.stat().st_size > MAX_CONSOLE_LOG_BYTES:
            output_path.replace(output_path.with_name(f"{output_path.name}.1"))
    except OSError:
        pass
    mirror = output_path.open("w", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, mirror)
    sys.stderr = _TeeStream(original_stderr, mirror)

    def cleanup() -> None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        mirror.flush()
        mirror.close()

    _cleanup = cleanup
    atexit.register(cleanup)
    _installed = True
    print(f"[console] Mirroring stdout/stderr to {output_path}", flush=True)
    return output_path
