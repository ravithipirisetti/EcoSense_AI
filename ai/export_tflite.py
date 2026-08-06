"""
EcoSense AI - Convert Keras Model to TFLite for Raspberry Pi 5 Deployment.
Generates an ultra-fast, lightweight .tflite model optimized for ARM Cortex-A76.
"""

import logging
from pathlib import Path
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/audio")
KERAS_MODEL_PATH = MODELS_DIR / "audio_model_yamnet.keras"
TFLITE_MODEL_PATH = MODELS_DIR / "audio_model_yamnet.tflite"


def convert_to_tflite():
    if not KERAS_MODEL_PATH.exists():
        logger.error("Keras model not found at %s", KERAS_MODEL_PATH)
        return

    logger.info("Loading Keras model from: %s", KERAS_MODEL_PATH)
    keras_model = tf.keras.models.load_model(str(KERAS_MODEL_PATH), safe_mode=False)

    logger.info("Converting model to TensorFlow Lite (.tflite)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Enable TFLite default optimizations for ARM CPUs
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    TFLITE_MODEL_PATH.write_bytes(tflite_model)
    
    keras_size_mb = KERAS_MODEL_PATH.stat().st_size / (1024 * 1024)
    tflite_size_mb = TFLITE_MODEL_PATH.stat().st_size / (1024 * 1024)

    logger.info("=" * 60)
    logger.info("SUCCESS: TFLite model exported for Raspberry Pi 5!")
    logger.info("  Keras Model Size  : %.2f MB", keras_size_mb)
    logger.info("  TFLite Model Size : %.2f MB", tflite_size_mb)
    logger.info("  Output Path       : %s", TFLITE_MODEL_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    convert_to_tflite()
