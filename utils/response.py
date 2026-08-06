"""
EcoSense AI Server - Response Formatting Utility.

Defines standard response envelopes for success and error payloads.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from utils.config import MODEL_VERSION, SERVER_VERSION


def get_current_timestamp() -> str:
    """Return ISO-8601 formatted UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_success_response(
    request_id: str,
    common_name: str,
    scientific_name: str,
    confidence: float,
    top_predictions: List[Dict[str, Any]],
    processing_time_ms: float,
    timestamp: str = None,
    api_version: str = SERVER_VERSION,
    model_version: str = MODEL_VERSION,
) -> Dict[str, Any]:
    """Format standardized prediction success payload."""
    ts = timestamp or get_current_timestamp()
    return {
        "status": "success",
        "request_id": request_id,
        "timestamp": ts,
        "processing_time_ms": round(processing_time_ms, 2),
        "api_version": api_version,
        "model_version": model_version,
        "prediction": {
            "common_name": common_name,
            "scientific_name": scientific_name,
            "confidence": round(confidence, 2),
        },
        "top_predictions": top_predictions,
    }


def format_error_response(
    message: str,
    request_id: str,
    timestamp: str = None,
) -> Dict[str, Any]:
    """Format standardized error payload."""
    ts = timestamp or get_current_timestamp()
    return {
        "status": "error",
        "request_id": request_id,
        "timestamp": ts,
        "message": message,
    }
