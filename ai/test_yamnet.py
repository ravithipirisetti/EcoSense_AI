"""
EcoSense AI - Quick Test Script for YAMNet Model.

Usage:
    python -m ai.test_yamnet <path_to_audio_file>

Example:
    python -m ai.test_yamnet datasets/audio/S14_Asian Koel/some_file.mp3
    python -m ai.test_yamnet audio/recorded_clip.wav
"""

import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from ai.yamnet_extractor import extract_yamnet_embedding

logging.basicConfig(level=logging.WARNING)

MODEL_PATH   = Path("models/audio/audio_model_yamnet.keras")
ENCODER_PATH = Path("models/audio/label_encoder.pkl")


def predict(audio_path: Path, top_n: int = 5):
    if not MODEL_PATH.exists():
        print(f"[!] Model not found: {MODEL_PATH}")
        return
    if not ENCODER_PATH.exists():
        print(f"[!] Label encoder not found: {ENCODER_PATH}")
        return
    if not audio_path.exists():
        print(f"[!] Audio file not found: {audio_path}")
        return

    print(f"\nAudio   : {audio_path}")
    print(">> Extracting YAMNet embedding...")

    embedding = extract_yamnet_embedding(audio_path)
    embedding = np.expand_dims(embedding, axis=0)  # (1, 1024)

    print(">> Running model prediction...")
    model = tf.keras.models.load_model(str(MODEL_PATH), safe_mode=False)
    label_encoder = joblib.load(ENCODER_PATH)

    if model.output_shape[-1] != len(label_encoder.classes_):
        print(f"[!] ERROR: Model output shape ({model.output_shape[-1]}) does not match LabelEncoder ({len(label_encoder.classes_)}). Out of sync!")
        return

    probs = model.predict(embedding, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_n]

    print(f"\n{'='*45}")
    print(f"  TOP {top_n} PREDICTIONS")
    print(f"{'='*45}")
    for rank, idx in enumerate(top_indices, start=1):
        species = label_encoder.classes_[idx]
        confidence = probs[idx] * 100
        bar = "|" * int(confidence / 5)
        print(f"  #{rank}  {confidence:5.1f}%  {bar:<20}  {species}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # If no argument, test on first available audio file in dataset
        dataset = Path("datasets/audio")
        sample = next(
            (f for d in dataset.iterdir() if d.is_dir()
             for f in d.iterdir() if f.suffix.lower() in {".wav", ".mp3", ".flac"}),
            None
        )
        if sample:
            print(f"No file specified. Using sample: {sample}")
            predict(sample)
        else:
            print("Usage: python -m ai.test_yamnet <audio_file>")
    else:
        predict(Path(" ".join(sys.argv[1:])))
