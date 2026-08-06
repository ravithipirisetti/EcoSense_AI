"""
EcoSense AI Server - Configuration Management.

Reads settings and environment variables for server execution.
"""

import os
from pathlib import Path
from typing import List

# Set TensorFlow C++ Log Level early
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")

# Base directory paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Model and Encoder Paths
DEFAULT_MODEL_PATH: Path = BASE_DIR / "models" / "audio_model_yamnet.keras"
FALLBACK_MODEL_PATH: Path = BASE_DIR / "models" / "audio" / "audio_model_yamnet.keras"

DEFAULT_ENCODER_PATH: Path = BASE_DIR / "models" / "label_encoder.json"
FALLBACK_ENCODER_PATH: Path = BASE_DIR / "models" / "audio" / "label_encoder.json"

MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
if not MODEL_PATH.exists() and FALLBACK_MODEL_PATH.exists():
    MODEL_PATH = FALLBACK_MODEL_PATH

ENCODER_PATH: Path = Path(os.getenv("ENCODER_PATH", str(DEFAULT_ENCODER_PATH)))
if not ENCODER_PATH.exists() and FALLBACK_ENCODER_PATH.exists():
    ENCODER_PATH = FALLBACK_ENCODER_PATH

# Server Environment Settings
SERVER_NAME: str = "EcoSense AI Server"
SERVER_VERSION: str = "1.0.0"
MODEL_NAME: str = "audio_model_yamnet.keras"
MODEL_VERSION: str = "1.0.0"
FRAMEWORK_NAME: str = "FastAPI"

PORT: int = int(os.getenv("PORT", "8000"))
API_KEY: str = os.getenv("API_KEY", "").strip()

ALLOWED_ORIGINS_RAW: str = os.getenv(
    "ALLOWED_ORIGINS",
    "https://ecosense.onrender.com,https://ecosense-ai.web.app,http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000",
).strip()

ALLOWED_ORIGINS: List[str] = [
    origin.strip() for origin in ALLOWED_ORIGINS_RAW.split(",") if origin.strip()
]
if not ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

CONFIDENCE_THRESHOLD: float = 0.60  # 60%
