"""Client TLS context for API-native sources behind an intercepting proxy.

Every aiohttp client in this repo pins certifi's bundle
(``ssl.create_default_context(cafile=certifi.where())``). On a machine whose
egress runs through a TLS-inspecting gateway — a managed laptop whose
Cloudflare Gateway / WARP re-signs some hosts with its own gateway CA —
certifi cannot verify those hosts, and Python's ``ssl`` never consults the
macOS keychain where the gateway CA lives. Such gateways commonly bypass
Anthropic and GitHub but intercept Exa.

``client_ssl_context()`` is certifi PLUS one operator-supplied CA file named
by ``CLORIS_EXTRA_CA_FILE``. Absent the variable, the context is exactly the
certifi one every other client builds. A configured-but-missing file raises
at client construction — a misconfigured trust path must fail loudly before
the first request, not as an opaque handshake error mid-run.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path

import certifi

EXTRA_CA_ENV = "CLORIS_EXTRA_CA_FILE"


def extra_ca_file() -> Path | None:
    """The operator's extra CA file, or None when the variable is unset."""

    raw = os.environ.get(EXTRA_CA_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def client_ssl_context() -> ssl.SSLContext:
    """certifi's bundle plus ``CLORIS_EXTRA_CA_FILE`` when configured."""

    ctx = ssl.create_default_context(cafile=certifi.where())
    extra = extra_ca_file()
    if extra is None:
        return ctx
    if not extra.is_file():
        raise ValueError(
            f"{EXTRA_CA_ENV} points at {extra}, which does not exist. "
            "Fix the path in .env or unset the variable."
        )
    ctx.load_verify_locations(cafile=str(extra))
    return ctx


__all__ = ["EXTRA_CA_ENV", "client_ssl_context", "extra_ca_file"]
