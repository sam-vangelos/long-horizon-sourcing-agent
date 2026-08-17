"""Centralized logging configuration for the Cloris app process.

Call configure_logging() once from create_app() before anything else.
Idempotent: a second call is a no-op so test runners that pre-configure
the root logger are unaffected.
"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
from pathlib import Path

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()  # type: ignore[attr-defined]
        return True


def configure_logging(log_dir: Path) -> None:
    """Attach rotating file + stream handlers to the root logger.

    Sets root level to INFO so all module loggers (e.g. cloris.api,
    cloris.app) emit without per-logger configuration. Rotating file
    lives at log_dir/cloris.log (10 MB × 5 = 50 MB max on disk).
    """
    root = logging.getLogger()
    if root.handlers:
        return  # idempotent — test runners may have pre-configured handlers

    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(
        log_dir / "cloris.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    request_id_filter = _RequestIdFilter()
    fh.addFilter(request_id_filter)
    sh.addFilter(request_id_filter)

    root.setLevel(logging.INFO)
    root.addFilter(_RequestIdFilter())
    root.addHandler(fh)
    root.addHandler(sh)
