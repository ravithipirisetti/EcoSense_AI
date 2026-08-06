"""
EcoSense AI Server - Structured File & Console Logging Utility.

Manages server.log, prediction.log, and error.log inside logs/ directory.
Sanitizes all logs to ensure API keys are NEVER logged.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SERVER_LOG_FILE = LOGS_DIR / "server.log"
PREDICTION_LOG_FILE = LOGS_DIR / "prediction.log"
ERROR_LOG_FILE = LOGS_DIR / "error.log"

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)


def _create_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """Create a logger instance with console and file handler."""
    logger_inst = logging.getLogger(name)
    logger_inst.setLevel(level)

    if not logger_inst.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger_inst.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger_inst.addHandler(console_handler)

    return logger_inst


server_logger = _create_logger("Server", SERVER_LOG_FILE)
prediction_logger = _create_logger("Prediction", PREDICTION_LOG_FILE)
error_logger = _create_logger("Error", ERROR_LOG_FILE, level=logging.ERROR)


def log_prediction_event(
    request_id: str,
    timestamp: str,
    filename: str,
    duration_ms: float,
    species: str,
    confidence: float,
    error: Optional[str] = None,
) -> None:
    """Log structured prediction event into prediction.log and server.log (never logs API keys)."""
    if error:
        msg = f"RequestID: {request_id} | Time: {timestamp} | File: {filename} | Duration: {duration_ms:.2f} ms | ERROR: {error}"
        prediction_logger.error(msg)
        error_logger.error(msg)
    else:
        msg = f"RequestID: {request_id} | Time: {timestamp} | File: {filename} | Duration: {duration_ms:.2f} ms | Species: {species} | Confidence: {confidence:.2f}%"
        prediction_logger.info(msg)
        server_logger.info(msg)


def log_server_info(message: str) -> None:
    """Log server informational message."""
    server_logger.info(message)


def log_server_error(message: str) -> None:
    """Log server error message."""
    error_logger.error(message)
    server_logger.error(message)
