"""
EcoSense AI Bird Sound Classification - Audio Inference Module.

Loads TensorFlow/Keras YAMNet model and label encoder ONCE via Singleton pattern.
Computes predictions, maps scientific names, tracks request IDs and latency.
"""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import tensorflow as tf

try:
    from ai.preprocess import preprocess_audio
    from ai.species_mapping import parse_label
    from utils.config import (
        CONFIDENCE_THRESHOLD,
        ENCODER_PATH,
        FALLBACK_ENCODER_PATH,
        MODEL_NAME,
        MODEL_PATH,
        MODEL_VERSION,
    )
    from utils.logger import logger
except ImportError:
    from preprocess import preprocess_audio
    from species_mapping import parse_label
    from utils.config import (
        CONFIDENCE_THRESHOLD,
        ENCODER_PATH,
        FALLBACK_ENCODER_PATH,
        MODEL_NAME,
        MODEL_PATH,
        MODEL_VERSION,
    )
    from utils.logger import logger

UNKNOWN_BIRD_LABEL: str = "Unknown Bird"
UNKNOWN_SCIENTIFIC_NAME: str = "Unknown"


@dataclass
class PredictionResult:
    """Dataclass holding inference results formatted for REST API responses."""

    request_id: str
    predicted_bird: str
    scientific_name: str
    confidence: float
    top_predictions: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    is_unknown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Format result matching exact requested API response schema."""
        return {
            "status": "success",
            "request_id": self.request_id,
            "model": self.model,
            "model_version": self.model_version,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "species": self.predicted_bird,
            "scientific_name": self.scientific_name,
            "confidence": round(self.confidence, 2),
            "prediction": {
                "common_name": self.predicted_bird,
                "scientific_name": self.scientific_name,
                "confidence": round(self.confidence, 2),
            },
            "top_predictions": self.top_predictions,
        }


class AudioPredictor:
    """Singleton class encapsulating model loading and audio prediction."""

    _instance: Optional["AudioPredictor"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "AudioPredictor":
        """Enforce singleton instance creation."""
        if cls._instance is None:
            cls._instance = super(AudioPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        encoder_path: Union[str, Path] = ENCODER_PATH,
    ) -> None:
        """Initialize predictor by loading model and label encoder ONLY ONCE."""
        if self._initialized:
            return

        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)

        if not self.model_path.is_file():
            logger.error("Model file not found: %s", self.model_path.resolve())
            raise FileNotFoundError(f"Model file missing: {self.model_path}")

        print("Loading encoder...")
        self.classes: List[str] = self._load_encoder(self.encoder_path)

        print("Loading species mapping...")
        # Verify species mapping loads cleanly

        try:
            print("Loading model...")
            logger.info("Loading audio model from: %s", self.model_path)
            self.model = tf.keras.models.load_model(str(self.model_path), safe_mode=False)

            num_model_classes = self.model.output_shape[-1]
            num_encoder_classes = len(self.classes)

            if num_model_classes != num_encoder_classes:
                raise ValueError(
                    f"Model output shape ({num_model_classes}) does not match "
                    f"LabelEncoder classes ({num_encoder_classes}). Model and LabelEncoder are out of sync!"
                )

            self._initialized = True
            logger.info("Audio Predictor initialized (%d species classes).", num_encoder_classes)
        except Exception as err:
            logger.critical("Failed to load model artifacts: %s", err, exc_info=True)
            raise RuntimeError(f"Initialization error: {err}") from err

    def _load_encoder(self, encoder_path: Path) -> List[str]:
        """Load class labels from JSON or Pickle file."""
        target_path = encoder_path
        if not target_path.is_file() and FALLBACK_ENCODER_PATH.is_file():
            logger.warning("Primary encoder %s not found, falling back to %s", encoder_path, FALLBACK_ENCODER_PATH)
            target_path = FALLBACK_ENCODER_PATH

        if not target_path.is_file():
            raise FileNotFoundError(f"Label encoder file missing at {encoder_path}")

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
            raise ValueError(f"Unsupported JSON format in {target_path}")

        label_encoder = joblib.load(target_path)
        return [str(cls_name) for cls_name in label_encoder.classes_]

    def predict(
        self,
        audio_file_path: Union[str, Path],
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        request_id: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> PredictionResult:
        """Predict bird species from a single audio file."""
        import time

        req_id = request_id or uuid.uuid4().hex[:8]
        start = start_time or time.perf_counter()
        file_path = Path(audio_file_path)

        if not file_path.is_file():
            logger.error("Audio file does not exist: %s", file_path)
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Feature extraction
        try:
            input_dim = self.model.input_shape[-1]
            if input_dim == 1024:
                try:
                    from ai.yamnet_extractor import extract_yamnet_embedding
                except ImportError:
                    from yamnet_extractor import extract_yamnet_embedding
                feature_vector = extract_yamnet_embedding(file_path)
            else:
                feature_vector = preprocess_audio(file_path)
        except Exception as err:
            logger.error("Preprocessing error for file %s: %s", file_path, err)
            raise RuntimeError(f"Failed to preprocess audio file {file_path}: {err}") from err

        if feature_vector.size == 0:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PredictionResult(
                request_id=req_id,
                predicted_bird=UNKNOWN_BIRD_LABEL,
                scientific_name=UNKNOWN_SCIENTIFIC_NAME,
                confidence=0.0,
                top_predictions=[],
                processing_time_ms=round(elapsed_ms, 2),
                is_unknown=True,
            )

        input_data = np.expand_dims(feature_vector, axis=0)

        # Model inference
        try:
            predictions_array = self.model.predict(input_data, verbose=0)[0]
        except Exception as err:
            logger.error("Model prediction failed for %s: %s", file_path, err)
            raise RuntimeError(f"Inference execution failed: {err}") from err

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        sorted_indices = np.argsort(predictions_array)[::-1]

        top_predictions: List[Dict[str, Any]] = []
        for idx in sorted_indices[:5]:
            raw_label = self.classes[idx]
            species_name, scientific_name = parse_label(raw_label)
            prob_percent = float(predictions_array[idx] * 100.0)
            top_predictions.append(
                {
                    "species": species_name,
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

        result = PredictionResult(
            request_id=req_id,
            predicted_bird=predicted_bird,
            scientific_name=scientific_name,
            confidence=top_confidence_percent,
            top_predictions=top_predictions,
            processing_time_ms=round(elapsed_ms, 2),
            is_unknown=is_unknown,
        )

        return result


def get_predictor() -> AudioPredictor:
    """Return the global Singleton AudioPredictor instance."""
    return AudioPredictor()
