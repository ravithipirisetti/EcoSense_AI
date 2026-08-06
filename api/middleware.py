"""
EcoSense AI Server - Middleware Module.

Attaches UUID request_id, ISO timestamp, and tracks processing duration.
"""

import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from utils.response import get_current_timestamp


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware injecting unique UUID request_id and timestamp into request state and response headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        start_time = time.perf_counter()

        request.state.request_id = request_id
        request.state.timestamp = timestamp
        request.state.start_time = start_time

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["x-request-id"] = request_id
        response.headers["x-processing-time-ms"] = f"{duration_ms:.2f}"
        return response
