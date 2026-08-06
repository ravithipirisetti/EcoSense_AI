"""
EcoSense AI - YAMNet Preprocessor Module.

Provides preprocessor interface for audio feature extraction using YAMNet embeddings.
"""

from pathlib import Path
from typing import Union
import numpy as np

try:
    from ai.yamnet_extractor import extract_yamnet_embedding
    from ai.preprocess import preprocess_audio
except ImportError:
    from yamnet_extractor import extract_yamnet_embedding
    from preprocess import preprocess_audio


def extract_features(audio_file_path: Union[str, Path], target_dim: int = 1024) -> np.ndarray:
    """Extract feature vector from audio file matching target dimension (1024 for YAMNet, 189 for MFCC).

    Args:
        audio_file_path (Union[str, Path]): Path to target audio clip (.wav, .mp3, .flac).
        target_dim (int): Target feature dimension expected by the model.

    Returns:
        np.ndarray: Feature vector array.
    """
    file_path = Path(audio_file_path)
    if target_dim == 1024:
        return extract_yamnet_embedding(file_path)
    return preprocess_audio(file_path)
