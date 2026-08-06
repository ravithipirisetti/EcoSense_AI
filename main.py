"""
EcoSense AI Server - Official Production REST API.

Headless AI Inference Server for Bird Sound Classification.
Deployed on Render, serving REST API calls for Website, Raspberry Pi, Mobile & Desktop apps.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai.predictor import get_predictor
from api.middleware import RequestIDMiddleware
from api.routes import router
from utils.config import (
    ALLOWED_ORIGINS,
    MODEL_NAME,
    MODEL_VERSION,
    PORT,
    SERVER_NAME,
    SERVER_VERSION,
)
from utils.logger import log_server_error, log_server_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager executing one-time startup sequence for AI Model & Encoder."""
    try:
        predictor = get_predictor()
        tf_loaded = predictor._initialized
        encoder_loaded = hasattr(predictor, "classes") and len(predictor.classes) > 0
        mapping_loaded = encoder_loaded

        print("\n" + "=" * 48)
        print(f"{SERVER_NAME}")
        print(f"API Version: {SERVER_VERSION}")
        print(f"Model Name: {MODEL_NAME}")
        print(f"Model Version: {MODEL_VERSION}")
        print(f"TensorFlow Loaded: {tf_loaded}")
        print(f"Label Encoder Loaded: {encoder_loaded}")
        print(f"Species Mapping Loaded: {mapping_loaded}")
        print(f"Listening Port: {PORT}")
        print("=" * 48 + "\n")

        log_server_info(f"{SERVER_NAME} initialized on port {PORT} with {len(predictor.classes)} species.")
    except Exception as err:
        log_server_error(f"CRITICAL: Server startup failed: {err}")
        raise err

    yield
    log_server_info(f"Shutting down {SERVER_NAME}.")


app = FastAPI(
    title=SERVER_NAME,
    description="Official Production AI Inference Server for Bird Sound Classification.",
    version=SERVER_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Custom Request ID Middleware
app.add_middleware(RequestIDMiddleware)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
