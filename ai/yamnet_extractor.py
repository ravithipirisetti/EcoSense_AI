"""
EcoSense AI - YAMNet Transfer Learning Feature Extractor.

Uses Google's YAMNet (trained on AudioSet, 521 audio classes) to extract
1024-dimensional embeddings from audio files. These embeddings replace
hand-crafted MFCC features and give dramatically better accuracy.

YAMNet is loaded from TensorFlow Hub (downloaded once, cached locally).
Each audio file produces one 1024-dim float32 embedding via mean pooling.

Requirements:
    pip install tensorflow-hub
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

logger = logging.getLogger(__name__)

# YAMNet requires 16 kHz mono audio
YAMNET_SR: int = 16_000
YAMNET_URL: str = "https://tfhub.dev/google/yamnet/1"
EMBEDDING_DIM: int = 1024

# Module-level singleton — loaded once
_yamnet_model = None


def get_yamnet() -> object:
    """Load YAMNet from TF Hub (cached after first call)."""
    global _yamnet_model
    if _yamnet_model is None:
        logger.info("Loading YAMNet from TensorFlow Hub (first time may take a minute)...")
        _yamnet_model = hub.load(YAMNET_URL)
        logger.info("YAMNet loaded successfully.")
    return _yamnet_model


def load_audio_16k(audio_path: Union[str, Path]) -> np.ndarray:
    """Load audio file and resample to 16 kHz mono float32."""
    audio_path = Path(audio_path)

    if _HAS_LIBROSA:
        sig, _ = librosa.load(str(audio_path), sr=YAMNET_SR, mono=True)
        return sig.astype(np.float32)

    if _HAS_SOUNDFILE:
        sig, sr = sf.read(str(audio_path))
        if sig.ndim > 1:
            sig = np.mean(sig, axis=1)
        sig = sig.astype(np.float32)
        if sr != YAMNET_SR:
            # Simple resampling via numpy (rough but works if librosa unavailable)
            length = int(len(sig) * YAMNET_SR / sr)
            indices = np.linspace(0, len(sig) - 1, length)
            sig = np.interp(indices, np.arange(len(sig)), sig)
        return sig.astype(np.float32)

    raise ImportError("Install librosa or soundfile: pip install librosa soundfile")


def extract_yamnet_embedding_from_signal(
    sig: np.ndarray, sr: int = YAMNET_SR, top_db: float = 25.0
) -> np.ndarray:
    """Extract a 1024-dim YAMNet embedding from a 1D audio signal array.

    Applies amplitude normalization, optional silence trimming, minimum length padding,
    and frame-level mean pooling.

    Args:
        sig (np.ndarray): 1D float32 numpy array of audio samples.
        sr (int): Sampling rate of input signal (must be 16000 Hz).
        top_db (float): Silence trimming threshold in dB below peak.

    Returns:
        np.ndarray: 1024-dim float32 feature embedding vector.
    """
    if sig is None or len(sig) == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    sig = sig.astype(np.float32)

    # 1. Resample if necessary
    if sr != YAMNET_SR:
        if _HAS_LIBROSA:
            sig = librosa.resample(sig, orig_sr=sr, target_sr=YAMNET_SR)
        else:
            length = int(len(sig) * YAMNET_SR / sr)
            indices = np.linspace(0, len(sig) - 1, length)
            sig = np.interp(indices, np.arange(len(sig)), sig).astype(np.float32)

    # 2. Amplitude normalization to range [-1.0, 1.0]
    max_val = np.max(np.abs(sig))
    if max_val > 1e-6:
        sig = sig / max_val

    # 3. Trim leading/trailing silence if signal is long (> 1.5s)
    if _HAS_LIBROSA and len(sig) > int(1.5 * YAMNET_SR):
        try:
            trimmed, _ = librosa.effects.trim(sig, top_db=top_db)
            if len(trimmed) >= int(0.48 * YAMNET_SR):
                sig = trimmed
        except Exception:
            pass

    # 4. YAMNet requires at least 0.96 seconds (15,360 samples); pad if shorter
    min_samples = int(0.96 * YAMNET_SR)
    if len(sig) < min_samples:
        sig = np.pad(sig, (0, min_samples - len(sig)))

    # 5. Extract YAMNet embeddings
    try:
        model = get_yamnet()
        sig_tensor = tf.constant(sig, dtype=tf.float32)
        _, embeddings, _ = model(sig_tensor)
        embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        return embedding.astype(np.float32)
    except Exception as err:
        logger.warning("YAMNet feature extraction failed: %s", err)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def extract_yamnet_embedding(audio_path: Union[str, Path]) -> np.ndarray:
    """Extract a single 1024-dim YAMNet embedding from an audio file.

    Args:
        audio_path: Path to audio file (wav, mp3, flac).

    Returns:
        np.ndarray of shape (1024,) dtype float32.
    """
    try:
        sig = load_audio_16k(audio_path)
        return extract_yamnet_embedding_from_signal(sig, sr=YAMNET_SR)
    except Exception as err:
        logger.warning("YAMNet embedding failed for %s: %s", audio_path, err)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m ai.yamnet_extractor <audio_file>")
        sys.exit(1)

    path = Path(sys.argv[1])
    emb = extract_yamnet_embedding(path)
    print(f"Embedding shape : {emb.shape}")
    print(f"Embedding range : [{emb.min():.4f}, {emb.max():.4f}]")
    print(f"Embedding mean  : {emb.mean():.4f}")
