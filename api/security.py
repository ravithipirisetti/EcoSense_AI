"""
EcoSense AI Server - Security Authentication Module.

Validates x-api-key header against environment variable API_KEY.
"""

import uuid
from typing import Optional

from fastapi import Header, HTTPException, Request, status

from utils.config import API_KEY
from utils.logger import log_server_error
from utils.response import format_error_response, get_current_timestamp


async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
) -> None:
    """Validate x-api-key header against API_KEY environment variable."""
    target_key = API_KEY.strip()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    timestamp = getattr(request.state, "timestamp", get_current_timestamp())

    if target_key:
        if not x_api_key or x_api_key.strip() != target_key:
            client_ip = request.client.host if request.client else "unknown"
            log_server_error(f"RequestID: {request_id} | Unauthorized attempt from IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=format_error_response(
                    message="Invalid or missing API key in x-api-key header.",
                    request_id=request_id,
                    timestamp=timestamp,
                ),
            )
