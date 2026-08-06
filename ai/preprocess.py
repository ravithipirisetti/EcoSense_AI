"""
EcoSense AI Bird Sound Classification - Audio Preprocessing & Feature Extraction.

This module loads audio files (.wav, .mp3, .flac), converts them to mono, resamples
to 22050 Hz, removes silence, normalizes signal amplitude, and extracts spectral features
(40 MFCCs, Chroma, Mel Spectrogram, Spectral Contrast, Zero Crossing Rate, and RMS Energy)
concatenated into a single 1D NumPy feature vector.
"""

import logging
from pathlib import Path
from typing import Set, Union

import librosa
import numpy as np

# Configure logger for audio preprocessing module
logger = logging.getLogger(__name__)

# Supported audio file extensions
SUPPORTED_EXTENSIONS: Set[str] = {".wav", ".mp3", ".flac"}
DEFAULT_SAMPLING_RATE: int = 22050
DEFAULT_NUM_MFCC: int = 40


def load_and_preprocess_audio(
    file_path: Union[str, Path],
    target_sr: int = DEFAULT_SAMPLING_RATE,
    top_db: float = 20.0,
) -> np.ndarray:
    """Load, convert to mono, resample, trim silence, and normalize an audio file.

    Args:
        file_path (Union[str, Path]): Path to input audio file (.wav, .mp3, .flac).
        target_sr (int): Target sampling rate in Hz (default: 22050).
        top_db (float): Silence threshold in decibels below peak (default: 20.0).

    Returns:
        np.ndarray: Normalized 1D float NumPy array of trimmed audio signal.

    Raises:
        ValueError: If file extension is unsupported or file fails to load.
        FileNotFoundError: If file path does not exist.
    """
    path = Path(file_path)

    if not path.is_file():
        logger.error("Audio file does not exist: %s", path)
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported file extension '%s': %s", ext, path)
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    try:
        # Load audio, convert to mono, and resample to target sampling rate (22050 Hz)
        y, sr = librosa.load(path, sr=target_sr, mono=True)
    except Exception as err:
        logger.error("Failed to load audio file %s: %s", path, err)
        raise ValueError(f"Could not load audio file {path}: {err}") from err

    if y is None or len(y) == 0:
        logger.warning("Loaded empty audio array from %s", path)
        return np.array([], dtype=np.float32)

    # Remove silence (trim leading/trailing silence below top_db threshold)
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if len(y_trimmed) > 0:
            y = y_trimmed
        else:
            logger.warning("Trimming resulted in empty audio for %s, keeping untrimmed.", path)
    except Exception as err:
        logger.warning("Silence removal failed for %s (%s), keeping raw signal.", path, err)

    # Normalize audio signal amplitude to range [-1.0, 1.0]
    try:
        y_normalized = librosa.util.normalize(y)
        return y_normalized.astype(np.float32)
    except Exception as err:
        logger.warning("Normalization failed for %s (%s), returning unnormalized.", path, err)
        return y.astype(np.float32)


def extract_features(
    y: np.ndarray,
    sr: int = DEFAULT_SAMPLING_RATE,
    n_mfcc: int = DEFAULT_NUM_MFCC,
) -> np.ndarray:
    """Extract acoustic features from audio signal and concatenate into a 1D vector.

    Extracted features:
        - 40 MFCCs (Mean across time)
        - Chroma STFT (Mean across time)
        - Mel Spectrogram (Mean in dB scale across time)
        - Spectral Contrast (Mean across time)
        - Zero Crossing Rate (Mean across time)
        - RMS Energy (Mean across time)

    Args:
        y (np.ndarray): Audio signal 1D array.
        sr (int): Sampling rate of the audio (default: 22050).
        n_mfcc (int): Number of MFCC coefficients to extract (default: 40).

    Returns:
        np.ndarray: Combined 1D feature vector.
    """
    if len(y) == 0:
        logger.warning("Empty audio signal provided for feature extraction.")
        return np.array([], dtype=np.float32)

    features: list[np.ndarray] = []

    # 1. Extract 40 MFCCs
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        features.append(mfcc_mean)
    except Exception as err:
        logger.error("Error extracting MFCCs: %s", err)
        features.append(np.zeros(n_mfcc, dtype=np.float32))

    # 2. Extract Chroma STFT
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        features.append(chroma_mean)
    except Exception as err:
        logger.error("Error extracting Chroma: %s", err)
        features.append(np.zeros(12, dtype=np.float32))

    # 3. Extract Mel Spectrogram (converted to dB scale)
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db.T, axis=0)
        features.append(mel_mean)
    except Exception as err:
        logger.error("Error extracting Mel Spectrogram: %s", err)
        features.append(np.zeros(128, dtype=np.float32))

    # 4. Extract Spectral Contrast
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        features.append(contrast_mean)
    except Exception as err:
        logger.error("Error extracting Spectral Contrast: %s", err)
        features.append(np.zeros(7, dtype=np.float32))

    # 5. Extract Zero Crossing Rate
    try:
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr.T, axis=0)
        features.append(zcr_mean)
    except Exception as err:
        logger.error("Error extracting Zero Crossing Rate: %s", err)
        features.append(np.zeros(1, dtype=np.float32))

    # 6. Extract RMS Energy
    try:
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms.T, axis=0)
        features.append(rms_mean)
    except Exception as err:
        logger.error("Error extracting RMS Energy: %s", err)
        features.append(np.zeros(1, dtype=np.float32))

    # Concatenate all extracted feature vectors into one combined 1D array
    combined_feature_vector: np.ndarray = np.hstack(features).astype(np.float32)
    return combined_feature_vector


def preprocess_audio(
    file_path: Union[str, Path],
    target_sr: int = DEFAULT_SAMPLING_RATE,
    n_mfcc: int = DEFAULT_NUM_MFCC,
) -> np.ndarray:
    """Complete audio preprocessing pipeline: load, clean, and extract feature vector.

    Args:
        file_path (Union[str, Path]): Path to audio file.
        target_sr (int): Target sampling rate (default: 22050 Hz).
        n_mfcc (int): Number of MFCCs to extract (default: 40).

    Returns:
        np.ndarray: Combined 1D NumPy feature vector.
    """
    audio_signal = load_and_preprocess_audio(file_path=file_path, target_sr=target_sr)
    feature_vector = extract_features(y=audio_signal, sr=target_sr, n_mfcc=n_mfcc)
    return feature_vector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Audio preprocessing module initialized.")
