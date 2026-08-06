"""
EcoSense AI - Live Microphone Bird Identifier.

Records audio from your microphone and predicts the bird species
using the trained YAMNet model.

Usage:
    python -m ai.live_predict
    python -m ai.live_predict --duration 5    (record 5 seconds)
    python -m ai.live_predict --continuous     (keep recording in a loop)
"""

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.WARNING)

MODEL_PATH   = Path("models/audio/audio_model_yamnet.keras")
ENCODER_PATH = Path("models/audio/label_encoder.pkl")
SAMPLE_RATE  = 16000   # YAMNet requires 16kHz
TOP_N        = 5


def record_audio(duration: float = 3.0) -> tuple[np.ndarray, int]:
    """Record audio from default microphone at hardware native sample rate."""
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError("Run: pip install sounddevice")

    print(f"\n[MIC] Recording for {duration} seconds... (stay quiet then play bird sound!)")
    print("      3...", end="", flush=True)
    time.sleep(1)
    print(" 2...", end="", flush=True)
    time.sleep(1)
    print(" 1...", end="", flush=True)
    time.sleep(1)
    print(" GO!", flush=True)

    input_device = sd.default.device[0]
    if input_device is None or input_device < 0:
        input_device = None

    # Get device native sample rate (e.g. 44100Hz / 48000Hz) to avoid Windows MME 16kHz silence bug
    try:
        dev_info = sd.query_devices(input_device if input_device is not None else sd.default.device[0])
        native_sr = int(dev_info.get("default_samplerate", 44100))
    except Exception:
        native_sr = 44100

    try:
        audio = sd.rec(
            int(duration * native_sr),
            samplerate=native_sr,
            channels=1,
            dtype="float32",
            device=input_device,
        )
        sd.wait()
        print("[MIC] Recording done!")
        return audio.flatten(), native_sr
    except Exception as err:
        print(f"\n[!] Microphone Recording Error: {err}")
        audio = sd.rec(
            int(duration * native_sr),
            samplerate=native_sr,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        print("[MIC] Fallback Recording done!")
        return audio.flatten(), native_sr


def predict_from_audio(audio: np.ndarray, native_sr: int, model: tf.keras.Model, label_encoder) -> None:
    """Run prediction on audio array and display top results."""
    from ai.yamnet_extractor import extract_yamnet_embedding_from_signal

    rms = float(np.sqrt(np.mean(audio**2)))
    max_amp = float(np.max(np.abs(audio)))
    print(f"[MIC] Recorded Signal Level: Peak Amplitude={max_amp:.4f}, RMS={rms:.6f}")

    if max_amp < 1e-6:
        print("\n[!] Microphone signal is pure zero. Please check microphone permissions in Windows!\n")
        return

    # Auto Gain Amplification for quiet/low-volume recordings
    if max_amp > 1e-6 and max_amp < 0.1:
        gain_factor = 0.7 / (max_amp + 1e-8)
        audio = audio * gain_factor
        new_amp = float(np.max(np.abs(audio)))
        print(f"[MIC] Amplified low-volume mic recording: {max_amp:.4f} -> {new_amp:.4f} ({gain_factor:.1f}x gain)")

    # Save last recording to WAV file for verification
    try:
        import soundfile as sf
        log_wav = Path("logs/last_mic_recording.wav")
        log_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(log_wav), audio, native_sr)
        print(f"[MIC] Saved clip to: {log_wav}")
    except Exception:
        pass

    # Extract YAMNet embedding (handles resampling from native_sr -> 16000Hz internally)
    embedding = extract_yamnet_embedding_from_signal(audio, sr=native_sr)
    embedding = np.expand_dims(embedding, axis=0)  # (1, 1024)

    # Predict
    probs = model.predict(embedding, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:TOP_N]
    top_confidence = probs[top_indices[0]] * 100

    if top_confidence < 50.0:
        print(f"\n{'='*50}")
        print("  NO CLEAR BIRD CALL DETECTED")
        print(f"{'='*50}")
        print(f"  Highest Match : {top_confidence:.1f}% (Low Confidence)")
        print("  Status        : Ambient room noise / ambiguous sound captured.")
        print("  Recommendation: Hold sound source closer & play a clear bird call.")
        print(f"{'='*50}\n")
        return

    print(f"\n{'='*50}")
    print(f"  BIRD DETECTED — TOP {TOP_N} PREDICTIONS")
    print(f"{'='*50}")
    for rank, idx in enumerate(top_indices, start=1):
        if idx >= len(label_encoder.classes_):
            continue
        species = label_encoder.classes_[idx]
        confidence = probs[idx] * 100
        # Strip leading code like "S10_"
        name = species.split("_", 1)[-1] if "_" in species else species
        bar = "|" * int(confidence / 3)
        print(f"  #{rank}  {confidence:5.1f}%  {bar:<30}  {name}")
    print(f"{'='*50}\n")


def run(duration: float = 3.0, continuous: bool = False) -> None:
    """Main entry point."""
    if not MODEL_PATH.exists():
        print(f"[!] Model not found: {MODEL_PATH}")
        print("    Run: python -m ai.train_birdnet")
        return
    if not ENCODER_PATH.exists():
        print(f"[!] Label encoder not found: {ENCODER_PATH}")
        return

    print("\n" + "="*50)
    print("  EcoSense AI - Live Bird Identifier")
    print("  Model: YAMNet Transfer Learning")
    print("="*50)
    print(f"  Record duration : {duration} seconds")
    print(f"  Mode            : {'Continuous' if continuous else 'Single'}")
    print("="*50)

    print("\n[LOAD] Loading model and label encoder...")
    model = tf.keras.models.load_model(str(MODEL_PATH), safe_mode=False)
    label_encoder = joblib.load(ENCODER_PATH)

    # Validate model output shape matches label encoder classes
    num_model_classes = model.output_shape[-1]
    num_encoder_classes = len(label_encoder.classes_)
    if num_model_classes != num_encoder_classes:
        print(f"\n[❌] CRITICAL ERROR: Model output size ({num_model_classes}) does not match LabelEncoder ({num_encoder_classes} classes).")
        print("    Your model and label encoder are OUT OF SYNC!")
        print("    Run: python -m ai.train_birdnet to retrain and resync.\n")
        return

    print(f"[LOAD] Model ready! ({num_encoder_classes} species configured)")

    try:
        while True:
            audio, native_sr = record_audio(duration)
            predict_from_audio(audio, native_sr, model, label_encoder)

            if not continuous:
                break

            again = input("Press ENTER to record again, or 'q' to quit: ").strip().lower()
            if again == "q":
                break

    except KeyboardInterrupt:
        print("\n\n[EXIT] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoSense AI Live Bird Identifier")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Recording duration in seconds (default: 3)")
    parser.add_argument("--continuous", action="store_true",
                        help="Keep recording in a loop")
    args = parser.parse_args()
    run(duration=args.duration, continuous=args.continuous)
