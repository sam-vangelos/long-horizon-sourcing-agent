"""Cloris app process: FastAPI assembly + lifecycle.

This module owns three things and only three things in Slice 1:

1. :func:`create_app` — build the FastAPI app and include the API router.
2. The :class:`WindowLauncher` protocol plus two concrete launchers
   (:class:`PyWebviewLauncher` for production, :class:`NullWindowLauncher` for
   tests).
3. :func:`run_app` — own one clean lifecycle: start uvicorn in a background
   thread, wait for readiness, hand off to the launcher, then trigger a
   bounded shutdown.

There is no static-files mount and no broader frontend tree exposed. The HTTP
surface is exactly what ``cloris.api`` declares.
"""

from __future__ import annotations

import logging
import os
import socket as _socket
import threading
import time
from typing import Any, Callable, Optional, Protocol, runtime_checkable
from urllib.error import URLError
from urllib.request import urlopen

if False:  # pragma: no cover - import-time-only typing aid
    from fastapi import FastAPI


log = logging.getLogger(__name__)


def create_app() -> "FastAPI":
    """Construct the Cloris FastAPI app.

    ``fastapi`` is imported lazily so that importing :mod:`cloris.app` (e.g.
    by tests that only need the launcher protocol or the CLI parser) does not
    require FastAPI to be installed.

    Slice 5 mounts the built Svelte UI's ``/assets/`` directory via
    ``StaticFiles``.
    """

    from pathlib import Path

    from fastapi import FastAPI

    from cloris import __version__
    from cloris.api import mount_static, router
    from cloris.api.auth import BearerAuthMiddleware
    from cloris.api.logging_setup import configure_logging
    from cloris.api.request_id import RequestIdMiddleware
    from shared.config import MissingRequiredKeyError, validate_startup_keys
    from shared.user_data_dir import cloris_user_data_dir, should_use_user_data_dir

    log_dir = (
        cloris_user_data_dir() / "logs"
        if should_use_user_data_dir()
        else Path(__file__).parent.parent / "logs"
    )
    configure_logging(log_dir)

    validate_startup_keys()

    app = FastAPI(
        title="Cloris",
        version=__version__,
        description="Cloris desktop shell — v0 / slice 1.",
    )
    # Middleware executes in reverse-add order. RequestIdMiddleware is added
    # first so it runs outermost: request IDs are stamped before auth runs,
    # meaning rejection log lines carry a correlation ID.
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(BearerAuthMiddleware)
    app.include_router(router)
    mount_static(app)

    import os

    # Empty string counts as unset for boot noise — ``export PYTEST_CURRENT_TEST=``
    # yields getenv(...) == "" which must not suppress startup diagnostics.
    if not os.getenv("PYTEST_CURRENT_TEST"):
        log.info(
            "cloris_http_boot version=%s abs_log_dir=%s",
            __version__,
            log_dir.resolve(),
        )
    return app


@runtime_checkable
class WindowLauncher(Protocol):
    """Open a native window pointed at the local Cloris server.

    ``open(url)`` MUST block until the window is closed. The blocking contract
    is what makes the lifecycle in :func:`run_app` simple: when ``open``
    returns, the user is done with the window and the server should shut down.
    """

    def open(self, url: str) -> None:  # pragma: no cover - protocol stub
        ...


class PyWebviewLauncher:
    """Launcher that opens a real native window via ``pywebview``.

    ``webview`` is imported inside :meth:`open` so that importing the launcher
    (or the rest of :mod:`cloris.app`) does not pull pywebview into the
    process. CI hosts and test machines without GTK/Qt stay unaffected until
    the user actually runs ``cloris start``.
    """

    def open(self, url: str) -> None:
        try:
            import webview
        except ModuleNotFoundError:
            import time
            import webbrowser

            webbrowser.open(url)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                return
            return

        webview.create_window("Cloris", url)
        webview.start()


class NullWindowLauncher:
    """No-op launcher used by tests.

    Records every ``open`` call and returns immediately, so the lifecycle in
    :func:`run_app` proceeds straight to shutdown without opening anything.
    """

    def __init__(self) -> None:
        self.opened: list[str] = []

    def open(self, url: str) -> None:
        self.opened.append(url)


class CertificationWindowLauncher:
    """Headless launcher used by packaged-app certification.

    Prints one machine-readable ready line and blocks until the process is
    terminated by the certification harness. The FastAPI lifecycle remains the
    production lifecycle: uvicorn is started, readiness is checked, then the
    launcher owns the blocking window phase.
    """

    def open(self, url: str) -> None:
        print(f"cloris_certify_ready url={url}", flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            return


ServerFactory = Callable[[Any, str, int], Any]


def require_loopback_host(host: str) -> None:
    """Allow only Cloris's supported IPv4 loopback address and hostname."""

    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "Cloris has no network-grade auth; non-loopback binds are refused"
        )


def _resolve_port(host: str, port: int) -> int:
    """Return ``port`` unchanged if nonzero; otherwise bind a transient socket
    to ``(host, 0)`` to obtain a concrete free local port, close it, and
    return that port number.

    This deliberately uses stdlib socket binding rather than reaching into
    ``uvicorn.Server.servers[...].sockets[...]`` because the test seams in
    Slice 1 are a swappable ``server_factory`` and a swappable
    ``_wait_until_ready``; coupling to uvicorn internals would force every
    stub to mimic uvicorn's nested startup state. The small bind/close race
    window is acceptable: the worst case is a clean uvicorn bind error, not
    a silent ``:0`` URL.
    """

    if port != 0:
        return port
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _default_server_factory(app: Any, host: str, port: int) -> Any:
    """Build a real ``uvicorn.Server`` configured for the app process."""

    import uvicorn

    require_loopback_host(host)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    return uvicorn.Server(config)


def _wait_until_ready(host: str, port: int, *, timeout: float = 5.0) -> None:
    """Poll ``GET /healthz`` until it returns 200 or ``timeout`` elapses.

    Exposed at module scope so tests can monkeypatch it without having to
    bind a real socket. Raises :class:`RuntimeError` on timeout.
    """

    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/healthz"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=0.5) as resp:  # noqa: S310 - localhost only
                if resp.status == 200:
                    return
        except (URLError, OSError) as exc:
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(
        f"Cloris server did not become ready at {url} within {timeout:.1f}s"
        f" (last_error={last_error!r})"
    )


def _frontend_build_cache_buster() -> str:
    """Return a stable cache key for the frontend build currently on disk."""

    try:
        from cloris.api import _paths

        stat = (_paths._DIST_DIR / "index.html").stat()
    except Exception as exc:  # pragma: no cover - defensive fallback
        log.debug("cloris frontend cache-buster fallback: %r", exc)
        return str(time.time_ns())
    return f"{stat.st_mtime_ns:x}-{stat.st_size:x}"


def _window_url(host: str, port: int) -> str:
    """Build the native-window URL with a frontend-build cache buster."""

    return f"http://{host}:{port}/?cloris_build={_frontend_build_cache_buster()}"


def _ensure_chrome_running_best_effort() -> None:
    """Pre-boot Chrome on Cloris's dedicated CDP profile, best-effort.

    Phase 0 ``chrome-launcher`` slice: the trial-day .app must not
    require the recipient to open a Terminal and run
    ``./launch-chrome.sh``. Auto-launching here means that by the
    time the welcome surface polls ``/api/chrome-status``, Chrome is
    either already up or currently spawning.

    Best-effort because:

    - On dev (running from the repo on a non-frozen Python),
      ``shared/user_data_dir.should_use_user_data_dir()`` is false
      and the launcher's profile path is the historical
      ``~/.chrome-cdp``. Devs who already have a Chrome on that
      profile pay no cost; the auto-launch no-ops via the
      ``is_healthy`` short-circuit.
    - On the recipient's machine, if Chrome.app isn't installed
      ("they uninstalled it" / "they're on Chromium / Edge only"),
      :func:`cloris.chrome_launcher.status` returns
      ``missing_chrome`` and the welcome surface explains the
      install path. The Cloris UI itself still loads — we never
      block on Chrome here.

    Logged but not raised. Errors don't gate the FastAPI boot.
    """

    try:
        from cloris.chrome_launcher import ensure_running

        result = ensure_running(force=False)
        log.info(
            "cloris.chrome_launcher: pre-boot status=%s message=%r",
            result.state,
            result.message,
        )
    except Exception as exc:
        log.warning(
            "cloris.chrome_launcher: pre-boot failed (best-effort): %r",
            exc,
        )


def run_app(
    app: Any,
    *,
    host: str,
    port: int,
    launcher: Optional[WindowLauncher] = None,
    server_factory: Optional[ServerFactory] = None,
    readiness_timeout: float = 5.0,
    shutdown_timeout: float = 5.0,
    ensure_chrome: Optional[Callable[[], None]] = None,
) -> None:
    """Run the Cloris app process through one full lifecycle.

    Steps:

    1. Best-effort pre-boot Chrome on Cloris's dedicated CDP profile
       (:func:`_ensure_chrome_running_best_effort`) so by the time
       the recipient sees the welcome surface, Chrome is already up
       or spawning. Pass a no-op via ``ensure_chrome`` to skip in
       tests.
    2. Build the server via ``server_factory`` (default uses uvicorn).
    3. Start ``server.run()`` in a background daemon thread.
    4. Poll ``/healthz`` until the server is reachable, or fail fast.
    5. Hand off to ``launcher.open(url)`` (blocks until the window closes).
    6. Set ``server.should_exit = True`` to ask uvicorn to wind down.
    7. Join the server thread with a bounded timeout. On overrun, log one
       warning and return — daemon thread will be reaped at process exit.

    The ``launcher``, ``server_factory``, and ``ensure_chrome`` kwargs
    are the test seams. All default to production implementations when
    not provided.
    """

    require_loopback_host(host)

    if launcher is None:
        launcher = PyWebviewLauncher()
    if server_factory is None:
        server_factory = _default_server_factory
    if ensure_chrome is None:
        ensure_chrome = _ensure_chrome_running_best_effort

    if os.getenv("CLORIS_CERTIFY_SKIP_CHROME_PREBOOT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        ensure_chrome = lambda: None

    ensure_chrome()

    resolved_port = _resolve_port(host, port)
    server = server_factory(app, host, resolved_port)

    thread = threading.Thread(
        target=server.run,
        name="cloris-uvicorn",
        daemon=True,
    )
    thread.start()

    try:
        _wait_until_ready(host, resolved_port, timeout=readiness_timeout)
    except Exception:
        server.should_exit = True
        thread.join(timeout=shutdown_timeout)
        raise

    url = _window_url(host, resolved_port)
    try:
        launcher.open(url)
    finally:
        server.should_exit = True
        thread.join(timeout=shutdown_timeout)
        if thread.is_alive():
            log.warning(
                "cloris: uvicorn server did not stop within %.1fs of "
                "should_exit=True; leaving daemon thread to be reaped at "
                "process exit.",
                shutdown_timeout,
            )
