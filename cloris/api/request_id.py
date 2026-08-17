"""Request ID middleware: stamps X-Request-Id on every request and response.

Runs outermost (added to the app first) so the correlation ID is present
in log lines emitted by BearerAuthMiddleware and all route handlers.
"""

from __future__ import annotations

import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from cloris.api.logging_setup import _request_id_var


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())[:8]
        token = _request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            _request_id_var.reset(token)
        response.headers["X-Request-Id"] = rid
        return response
