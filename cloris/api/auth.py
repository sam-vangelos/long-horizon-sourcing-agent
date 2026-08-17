"""Process-lifetime bearer token authentication for the Cloris API.

SESSION_TOKEN is a 256-bit random hex string generated once when this
module is first imported (which happens inside create_app()). It persists
for the full process lifetime and is delivered to the Svelte frontend via
GET /api/bootstrap (an exempt, localhost-only endpoint).

All API routes are gated by BearerAuthMiddleware except:
  - /healthz, /, /api/bootstrap,
    /manifest.webmanifest        (exact-match exempt)
  - /assets/*, /brand/*          (static file prefix exempt)
  - /api/conversation/{brief_id}/stream
                                  (SSE: EventSource cannot set headers;
                                  token accepted via ?token= query param)
  - /api/intake/sessions/{session_id}/converse/stream
                                  (SSE: same EventSource constraint;
                                  conversational intake streaming added
                                  in Phase C5 of plans/conversational-intake.md)

``/brand/*`` and ``/manifest.webmanifest`` are exempt because browsers
fetch them from ``<link rel="icon">`` / ``<link rel="manifest">`` /
``<meta property="og:image">`` without an Authorization header. The
files contain only public brand assets — no PII, no credentials — so
the security posture matches ``/assets/`` (the Vite-built JS/CSS
bundle).

Audit finding F-5: SSE exemptions are matched against an explicit
regex allowlist (one pattern per documented route), not a structural
``startswith(prefix) and endswith(suffix)`` test. A new SSE route
**must** add its pattern to ``_SSE_QUERY_TOKEN_PATTERNS`` AND the
docstring above; otherwise the bearer header check still applies.
This forecloses the failure mode where a future
``/api/conversation/{brief_id}/admin/stream`` would auto-exempt simply
because it shares the prefix and suffix.

Test environments set CLORIS_SKIP_AUTH_FOR_TESTING=1 (via tests/conftest.py)
to bypass auth without touching every existing TestClient call. This env var
is never set in packaged production builds.
"""

from __future__ import annotations

import hmac
import os
import re
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

SESSION_TOKEN: str = secrets.token_hex(32)

_EXEMPT_EXACT: frozenset[str] = frozenset(
    {"/healthz", "/", "/api/bootstrap", "/manifest.webmanifest"}
)
_EXEMPT_PREFIXES: tuple[str, ...] = ("/assets/", "/brand/")

# Closed enumeration of SSE routes that accept ``?token=`` instead of a
# bearer header. Each path-segment placeholder is ``[^/]+`` so a new
# nested route (e.g. ``/admin/stream``) does NOT match by accident.
# Patterns are anchored with ``^`` and ``$``; ``re.fullmatch`` adds
# implicit anchoring on top. See module docstring for the
# documentation contract this list pairs with.
_SSE_QUERY_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^/api/conversation/[^/]+/stream$"),
    re.compile(r"^/api/intake/sessions/[^/]+/converse/stream$"),
)


def _skip_auth_for_testing() -> bool:
    return os.getenv("CLORIS_SKIP_AUTH_FOR_TESTING", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _is_sse_query_token_route(path: str) -> bool:
    """True iff ``path`` is one of the explicitly enumerated SSE routes
    that accept ``?token=`` for auth. Used by the middleware AND
    ``tests/test_bearer_auth.py`` so the enumeration stays the single
    source of truth."""

    return any(p.fullmatch(path) is not None for p in _SSE_QUERY_TOKEN_PATTERNS)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Validate Authorization: Bearer <token> on all non-exempt routes."""

    async def dispatch(self, request: Request, call_next):
        if _skip_auth_for_testing():
            return await call_next(request)

        path = request.url.path

        if path in _EXEMPT_EXACT:
            return await call_next(request)
        for prefix in _EXEMPT_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)

        # SSE: native EventSource (and POST + Accept: text/event-stream)
        # cannot set Authorization headers. Accept token as query param
        # for the explicit allowlist above. Adding a new SSE endpoint?
        # Append a regex to ``_SSE_QUERY_TOKEN_PATTERNS`` AND document
        # it in the module docstring.
        if _is_sse_query_token_route(path):
            provided = request.query_params.get("token", "")
            if hmac.compare_digest(provided.encode(), SESSION_TOKEN.encode()):
                return await call_next(request)
            return JSONResponse(status_code=401, content={"detail": "invalid_token"})

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "missing_bearer_token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        provided = auth[len("Bearer "):]
        if not hmac.compare_digest(provided.encode(), SESSION_TOKEN.encode()):
            return JSONResponse(
                status_code=401,
                content={"detail": "invalid_bearer_token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)
