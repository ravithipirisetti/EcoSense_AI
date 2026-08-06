"""
EcoSense AI - Singleton Model Predictor Module.

Encapsulates TensorFlow/Keras YAMNet model loading and audio inference.
Loaded ONLY ONCE during server application startup.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from ai.species_mapping import parse_label
from ai.yamnet_preprocessor import extract_features
from utils.config import (
    CONFIDENCE_THRESHOLD,
    ENCODER_PATH,
    FALLBACK_ENCODER_PATH,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    SERVER_VERSION,
)
from utils.logger import log_server_info
from utils.response import format_success_response, get_current_timestamp

UNKNOWN_BIRD_LABEL: str = "Unknown Bird"
UNKNOWN_SCIENTIFIC_NAME: str = "Unknown"


@dataclass
class PredictionResult:
    """Dataclass holding inference results."""

    request_id: str
    predicted_bird: str
    scientific_name: str
    confidence: float
    top_predictions: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: str = ""
    api_version: str = SERVER_VERSION
    model_version: str = MODEL_VERSION
    is_unknown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standardized success response dictionary."""
        return format_success_response(
            request_id=self.request_id,
            common_name=self.predicted_bird,
            scientific_name=self.scientific_name,
            confidence=self.confidence,
            top_predictions=self.top_predictions,
            processing_time_ms=self.processing_time_ms,
            timestamp=self.timestamp,
            api_version=self.api_version,
            model_version=self.model_version,
        )


class AudioPredictor:
    """Singleton class managing TensorFlow model and species label encoder."""

    _instance: Optional["AudioPredictor"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "AudioPredictor":
        """Enforce Singleton design pattern."""
        if cls._instance is None:
            cls._instance = super(AudioPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        encoder_path: Union[str, Path] = ENCODER_PATH,
    ) -> None:
        """Initialize predictor ONCE on server startup."""
        if self._initialized:
            return

        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.classes: List[str] = self._load_encoder(self.encoder_path)

        try:
            self.model = tf.keras.models.load_model(str(self.model_path), safe_mode=False)
            num_model_classes = self.model.output_shape[-1]
            num_encoder_classes = len(self.classes)

            if num_model_classes != num_encoder_classes:
                raise ValueError(
                    f"Model output classes ({num_model_classes}) does not match "
                    f"LabelEncoder classes ({num_encoder_classes}). Model and Encoder out of sync!"
                )

            # Pre-warm YAMNet from TF Hub during startup
            try:
                from ai.yamnet_extractor import get_yamnet
                get_yamnet()
            except Exception as ex:
                pass

            self._initialized = True
            log_server_info(f"Audio Predictor initialized successfully ({num_encoder_classes} species classes).")
        except Exception as err:
            raise RuntimeError(f"Predictor initialization error: {err}") from err

    def _load_encoder(self, encoder_path: Path) -> List[str]:
        """Load class labels from JSON encoder file."""
        target_path = encoder_path
        if not target_path.is_file() and FALLBACK_ENCODER_PATH.is_file():
            target_path = FALLBACK_ENCODER_PATH

        if not target_path.is_file():
            raise FileNotFoundError(f"Label encoder missing at {encoder_path}")

        if target_path.suffix == ".json":
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            elif isinstance(data, dict):
                if "classes" in data:
                    return [str(item) for item in data["classes"]]
                sorted_keys = sorted(data.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
                return [str(data[k]) for k in sorted_keys]
            raise ValueError(f"Unsupported JSON structure in {target_path}")

        import joblib
        label_encoder = joblib.load(target_path)
        return [str(c) for c in label_encoder.classes_]

    def predict(
        self,
        audio_file_path: Union[str, Path],
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        request_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> PredictionResult:
        """Execute inference on a single audio clip."""
        req_id = request_id or str(uuid.uuid4())
        ts = timestamp or get_current_timestamp()
        start = start_time or time.perf_counter()
        file_path = Path(audio_file_path)

        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        input_dim = self.model.input_shape[-1]
        feature_vector = extract_features(file_path, target_dim=input_dim)

        if feature_vector.size == 0:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PredictionResult(
                request_id=req_id,
                predicted_bird=UNKNOWN_BIRD_LABEL,
                scientific_name=UNKNOWN_SCIENTIFIC_NAME,
                confidence=0.0,
                top_predictions=[],
                processing_time_ms=round(elapsed_ms, 2),
                timestamp=ts,
                is_unknown=True,
            )

        input_data = np.expand_dims(feature_vector, axis=0)
        predictions_array = self.model.predict(input_data, verbose=0)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        sorted_indices = np.argsort(predictions_array)[::-1]

        top_predictions: List[Dict[str, Any]] = []
        for idx in sorted_indices[:5]:
            raw_label = self.classes[idx]
            common_name, scientific_name = parse_label(raw_label)
            prob_percent = float(predictions_array[idx] * 100.0)
            top_predictions.append(
                {
                    "common_name": common_name,
                    "scientific_name": scientific_name,
                    "confidence": round(prob_percent, 2),
                }
            )

        top_idx = sorted_indices[0]
        top_confidence = float(predictions_array[top_idx])
        top_confidence_percent = round(top_confidence * 100.0, 2)

        if top_confidence < confidence_threshold:
            predicted_bird = UNKNOWN_BIRD_LABEL
            scientific_name = UNKNOWN_SCIENTIFIC_NAME
            top_confidence_percent = 0.0
            is_unknown = True
            top_predictions = []
        else:
            raw_top_label = self.classes[top_idx]
            predicted_bird, scientific_name = parse_label(raw_top_label)
            is_unknown = False

        return PredictionResult(
            request_id=req_id,
            predicted_bird=predicted_bird,
            scientific_name=scientific_name,
            confidence=top_confidence_percent,
            top_predictions=top_predictions,
            processing_time_ms=round(elapsed_ms, 2),
            timestamp=ts,
            is_unknown=is_unknown,
        )


def get_predictor() -> AudioPredictor:
    """Return Singleton AudioPredictor instance."""
    return AudioPredictor()
