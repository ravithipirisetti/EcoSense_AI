"""
EcoSense AI - Audio Feature Extractor

Extracts:
- MFCC (40)
- Chroma STFT (12)
- Mel Spectrogram Mean (128)
- Spectral Contrast (7)
- Zero Crossing Rate (1)
- RMS Energy (1)

Total Features = 189
"""

import librosa
import numpy as np


def extract_features(y, sr=22050):
    """
    Extract 189-dimensional feature vector.
    """

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )
    mfcc = np.mean(mfcc.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )
    chroma = np.mean(chroma.T, axis=0)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr
    )
    mel = librosa.power_to_db(mel)
    mel = np.mean(mel.T, axis=0)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )
    contrast = np.mean(contrast.T, axis=0)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr = np.mean(zcr)

    # RMS
    rms = librosa.feature.rms(y=y)
    rms = np.mean(rms)

    feature_vector = np.concatenate([
        mfcc,
        chroma,
        mel,
        contrast,
        [zcr],
        [rms]
    ])

    return feature_vector.astype(np.float32)


if __name__ == "__main__":
    import soundfile as sf

    audio, sr = sf.read("sample.wav")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    features = extract_features(audio, sr)

    print("Feature Shape :", features.shape)