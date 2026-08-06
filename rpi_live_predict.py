"""
EcoSense AI - Raspberry Pi 5 Dedicated Live Bird Identifier.
Runs using TensorFlow Lite (.tflite) for ultra-fast, low-memory ARM Cortex-A76 inference.

Usage on Raspberry Pi 5:
    python3 rpi_live_predict.py
    python3 rpi_live_predict.py --duration 5
    python3 rpi_live_predict.py --continuous
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths for Raspberry Pi
MODEL_PATH = Path("models/audio/audio_model_yamnet.tflite")
ENCODER_PATH = Path("models/audio/label_encoder.pkl")
MIN_CONFIDENCE_THRESHOLD = 50.0  # Require >= 50% for clear bird detection
TOP_N = 5


def load_tflite_interpreter(tflite_path: Path):
    """Load TFLite interpreter using tflite_runtime or full tensorflow."""
    try:
        import tflite_runtime.interpreter as tflite
        logger.info("Using tflite_runtime engine for ARM64")
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite
        logger.info("Using TensorFlow Lite engine")

    interpreter = tflite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    return interpreter


def run_tflite_inference(interpreter, embedding: np.ndarray) -> np.ndarray:
    """Execute TFLite interpreter inference on (1, 1024) embedding input."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = embedding.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]


def record_rpi_audio(duration: float = 5.0) -> tuple[np.ndarray, int]:
    """Record audio from USB microphone or I2S mic module on Raspberry Pi."""
    try:
        import sounddevice as sd
    except ImportError:
        logger.error("Missing sounddevice module! Run: pip3 install sounddevice")
        sys.exit(1)

    print(f"\n[RPi 5 MIC] Recording for {duration} seconds...")
    print("            3...", end="", flush=True)
    time.sleep(1)
    print(" 2...", end="", flush=True)
    time.sleep(1)
    print(" 1...", end="", flush=True)
    time.sleep(1)
    print(" GO!", flush=True)

    input_device = sd.default.device[0]
    if input_device is None or input_device < 0:
        input_device = None

    try:
        dev_info = sd.query_devices(input_device if input_device is not None else sd.default.device[0])
        native_sr = int(dev_info.get("default_samplerate", 44100))
    except Exception:
        native_sr = 44100

    audio = sd.rec(
        int(duration * native_sr),
        samplerate=native_sr,
        channels=1,
        dtype="float32",
        device=input_device
    )
    sd.wait()
    print("[RPi 5 MIC] Recording complete!")
    return audio.flatten(), native_sr


def predict_rpi_audio(audio: np.ndarray, native_sr: int, interpreter, label_encoder) -> None:
    """Extract YAMNet embedding and run TFLite prediction on Raspberry Pi 5."""
    from ai.yamnet_extractor import extract_yamnet_embedding_from_signal

    rms = float(np.sqrt(np.mean(audio**2)))
    max_amp = float(np.max(np.abs(audio)))
    print(f"[RPi 5 MIC] Signal Level: Peak Amplitude={max_amp:.4f}, RMS={rms:.6f}")

    if max_amp < 1e-6:
        print("\n[!] Microphone signal is pure zero. Please check USB/I2S mic connection!\n")
        return

    # Auto Gain Boost for quiet microphone inputs
    if max_amp > 1e-6 and max_amp < 0.1:
        gain_factor = 0.7 / (max_amp + 1e-8)
        audio = audio * gain_factor
        new_amp = float(np.max(np.abs(audio)))
        print(f"[RPi 5 MIC] Amplified low-volume mic recording: {max_amp:.4f} -> {new_amp:.4f} ({gain_factor:.1f}x gain)")

    # Extract 1024-dim YAMNet embedding
    embedding = extract_yamnet_embedding_from_signal(audio, sr=native_sr)
    embedding = np.expand_dims(embedding, axis=0)  # (1, 1024)

    # Run TFLite Inference
    probs = run_tflite_inference(interpreter, embedding)
    top_indices = np.argsort(probs)[::-1][:TOP_N]
    top_confidence = probs[top_indices[0]] * 100

    if top_confidence < MIN_CONFIDENCE_THRESHOLD:
        print(f"\n{'='*50}")
        print("  NO CLEAR BIRD CALL DETECTED")
        print(f"{'='*50}")
        print(f"  Highest Match : {top_confidence:.1f}% (Low Confidence)")
        print("  Status        : Ambient room noise / ambiguous sound captured.")
        print("  Recommendation: Hold sound source closer & play a clear bird call.")
        print(f"{'='*50}\n")
        return

    print(f"\n{'='*50}")
    print(f"  BIRD DETECTED — TOP {TOP_N} PREDICTIONS (TFLite RPi 5)")
    print(f"{'='*50}")
    for rank, idx in enumerate(top_indices, start=1):
        if idx >= len(label_encoder.classes_):
            continue
        species = label_encoder.classes_[idx]
        confidence = probs[idx] * 100
        name = species.split("_", 1)[-1] if "_" in species else species
        bar_len = int(confidence / 5)
        bar = "|" * bar_len
        print(f"  #{rank}  {confidence:5.1f}%  {bar:<22}  {name}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="EcoSense AI - Raspberry Pi 5 Bird Identifier")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--continuous", action="store_true", help="Run continuous loop mode")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        logger.error("TFLite model missing! Run 'python -m ai.export_tflite' first.")
        sys.exit(1)

    if not ENCODER_PATH.exists():
        logger.error("Label encoder missing at %s", ENCODER_PATH)
        sys.exit(1)

    print("=" * 60)
    print("  EcoSense AI — Raspberry Pi 5 Live Bird Identifier")
    print("  Engine : TensorFlow Lite (.tflite)")
    print("=" * 60)

    interpreter = load_tflite_interpreter(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    logger.info("Loaded %d species classes from encoder", len(label_encoder.classes_))

    if args.continuous:
        print("\n[LOOP] Starting continuous listening mode (Press Ctrl+C to stop)...\n")
        while True:
            try:
                audio, native_sr = record_rpi_audio(duration=args.duration)
                predict_rpi_audio(audio, native_sr, interpreter, label_encoder)
                time.sleep(1)
            except KeyboardInterrupt:
                print("\n[STOP] Live detection stopped by user.")
                break
    else:
        audio, native_sr = record_rpi_audio(duration=args.duration)
        predict_rpi_audio(audio, native_sr, interpreter, label_encoder)


if __name__ == "__main__":
    main()
