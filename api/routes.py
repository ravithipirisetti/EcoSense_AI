"""
EcoSense AI Server - API Routes Definition.

Defines GET /, GET /health, GET /health/details, GET /info, GET /version,
POST /predict, POST /identify, POST /api/v1/predict.
"""

import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from ai.audio_processor import save_temp_audio
from ai.predictor import get_predictor
from api.security import verify_api_key
from utils.config import (
    FRAMEWORK_NAME,
    MODEL_NAME,
    MODEL_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
)
from utils.logger import log_prediction_event
from utils.response import format_error_response, get_current_timestamp

router = APIRouter()
SERVER_START_TIME: float = time.time()


@router.get("/", summary="Root Status Endpoint")
async def root_status() -> Dict[str, Any]:
    """Root status endpoint."""
    return {
        "name": SERVER_NAME,
        "status": "running",
    }


@router.get("/health", summary="Health Check Endpoint")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    predictor = get_predictor()
    model_ready = getattr(predictor, "_initialized", False)
    encoder_ready = model_ready and hasattr(predictor, "classes") and len(predictor.classes) > 0
    class_count = len(predictor.classes) if encoder_ready else 0

    return {
        "status": "healthy" if model_ready else "initializing",
        "model_loaded": model_ready,
        "encoder_loaded": encoder_ready,
        "classes": class_count,
        "version": SERVER_VERSION,
    }


@router.get("/health/details", summary="Detailed Health & Component Probes")
async def health_details() -> Dict[str, Any]:
    """Detailed health check endpoint reporting component statuses and server uptime."""
    predictor = get_predictor()
    model_ready = getattr(predictor, "_initialized", False)
    encoder_ready = model_ready and hasattr(predictor, "classes") and len(predictor.classes) > 0
    class_count = len(predictor.classes) if encoder_ready else 0
    uptime_seconds = round(time.time() - SERVER_START_TIME, 2)

    return {
        "status": "healthy" if model_ready else "initializing",
        "timestamp": get_current_timestamp(),
        "uptime_seconds": uptime_seconds,
        "components": {
            "model_predictor": {"status": "up" if model_ready else "down", "model_file": MODEL_NAME},
            "label_encoder": {"status": "up" if encoder_ready else "down", "classes_count": class_count},
            "species_mapping": {"status": "up"},
        },
        "version": SERVER_VERSION,
    }


@router.get("/info", summary="Server Capabilities & Metadata")
async def server_info() -> Dict[str, Any]:
    """Metadata info endpoint."""
    predictor = get_predictor()
    model_ready = getattr(predictor, "_initialized", False)
    class_count = len(predictor.classes) if model_ready else 0

    return {
        "server": "EcoSense AI",
        "framework": FRAMEWORK_NAME,
        "model": MODEL_NAME,
        "version": SERVER_VERSION,
        "classes": class_count,
    }


@router.get("/version", summary="API & Model Version Information")
async def server_version() -> Dict[str, Any]:
    """Version metadata endpoint."""
    return {
        "server": SERVER_NAME,
        "api_version": SERVER_VERSION,
        "model_version": MODEL_VERSION,
    }


async def _run_inference_pipeline(request: Request, audio: UploadFile) -> Dict[str, Any]:
    """Unified inference pipeline processing requests for all prediction routes."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    timestamp = getattr(request.state, "timestamp", get_current_timestamp())
    start_time = getattr(request.state, "start_time", time.perf_counter())

    if not audio or not audio.filename:
        log_prediction_event(
            request_id=request_id,
            timestamp=timestamp,
            filename="none",
            duration_ms=0.0,
            species="None",
            confidence=0.0,
            error="No audio file provided.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=format_error_response("No audio file provided.", request_id=request_id, timestamp=timestamp),
        )

    filename = audio.filename

    try:
        temp_path, _ = await save_temp_audio(audio)
        predictor = get_predictor()
        result = predictor.predict(
            audio_file_path=temp_path,
            request_id=request_id,
            timestamp=timestamp,
            start_time=start_time,
        )

        response_dict = result.to_dict()

        log_prediction_event(
            request_id=request_id,
            timestamp=timestamp,
            filename=filename,
            duration_ms=result.processing_time_ms,
            species=result.predicted_bird,
            confidence=result.confidence,
        )

        return response_dict

    except HTTPException:
        raise
    except Exception as err:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        log_prediction_event(
            request_id=request_id,
            timestamp=timestamp,
            filename=filename,
            duration_ms=elapsed_ms,
            species="Error",
            confidence=0.0,
            error=str(err),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=format_error_response(f"Inference execution failed: {str(err)}", request_id=request_id, timestamp=timestamp),
        )
    finally:
        if "temp_path" in locals() and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@router.post(
    "/predict",
    summary="Audio Species Classification Endpoint",
    dependencies=[Depends(verify_api_key)],
)
async def predict_endpoint(request: Request, audio: UploadFile = File(..., alias="audio")) -> Dict[str, Any]:
    """Primary audio species prediction endpoint."""
    return await _run_inference_pipeline(request, audio)


@router.post(
    "/identify",
    summary="Audio Species Identification Alias Endpoint",
    dependencies=[Depends(verify_api_key)],
)
async def identify_endpoint(request: Request, audio: UploadFile = File(..., alias="audio")) -> Dict[str, Any]:
    """Alias prediction endpoint."""
    return await _run_inference_pipeline(request, audio)


@router.post(
    "/api/v1/predict",
    summary="Versioned Audio Species Prediction Endpoint Alias",
    dependencies=[Depends(verify_api_key)],
)
async def predict_v1_endpoint(request: Request, audio: UploadFile = File(..., alias="audio")) -> Dict[str, Any]:
    """Versioned prediction endpoint alias."""
    return await _run_inference_pipeline(request, audio)
