"""
EcoSense AI - Production Dataset Loader & Splitter Module.

Pipeline Architecture:
    Scan datasets/audio/ -> Skip empty species -> Preprocess Audio
    -> 80/10/10 Split -> Online Augmentation on Train (1 pass)
    -> Extract 189 Features -> LabelEncoder -> Return splits
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ai.audio_augmenter import random_augmentation
from ai.feature_extractor import extract_features
from ai.preprocess import load_and_preprocess_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR: Path = Path("datasets/audio")
DEFAULT_ENCODER_PATH: Path = Path("models/audio/label_encoder.pkl")
SUPPORTED_EXTENSIONS: Set[str] = {".wav", ".mp3", ".flac"}


@dataclass
class AudioDatasetResult:
    audio_paths: List[Path]
    labels: np.ndarray
    label_encoder: LabelEncoder
    class_counts: Dict[str, int] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.audio_paths, self.labels, self.label_encoder))


def is_valid_audio(file_path: Path) -> bool:
    p = Path(file_path)
    if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    try:
        return p.stat().st_size > 0
    except OSError:
        return False


def load_audio_dataset(
    dataset_dir: Union[str, Path] = DEFAULT_DATASET_DIR,
) -> AudioDatasetResult:
    """Scan species subdirectories. Skip folders with zero audio files."""
    root_path = Path(dataset_dir)
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory missing: {root_path}")

    audio_paths: List[Path] = []
    raw_labels: List[str] = []
    class_counts: Dict[str, int] = {}
    skipped: List[str] = []

    species_dirs = sorted(
        [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    for species_dir in species_dirs:
        species_name = species_dir.name
        count = 0
        for file_path in species_dir.rglob("*"):
            if is_valid_audio(file_path):
                audio_paths.append(file_path)
                raw_labels.append(species_name)
                count += 1
        class_counts[species_name] = count
        if count == 0:
            skipped.append(species_name)

    if skipped:
        logger.warning(
            "Skipping %d species with 0 audio files: %s", len(skipped), skipped
        )

    label_encoder = LabelEncoder()
    if raw_labels:
        labels = label_encoder.fit_transform(raw_labels)
    else:
        labels = np.array([], dtype=int)
        label_encoder.fit([])

    logger.info(
        "Loaded %d audio files across %d species.",
        len(audio_paths),
        len(label_encoder.classes_),
    )

    return AudioDatasetResult(
        audio_paths=audio_paths,
        labels=labels,
        label_encoder=label_encoder,
        class_counts=class_counts,
    )


def load_dataset(
    dataset_dir: Union[str, Path] = DEFAULT_DATASET_DIR,
    encoder_path: Union[str, Path] = DEFAULT_ENCODER_PATH,
    random_state: int = 42,
    augment_training: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    LabelEncoder,
]:
    """Load dataset, split 80/10/10, apply single-pass online augmentation on train.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
    """
    dataset_result = load_audio_dataset(dataset_dir)
    audio_paths = dataset_result.audio_paths
    y_encoded = dataset_result.labels
    label_encoder = dataset_result.label_encoder

    if len(audio_paths) == 0:
        raise ValueError("No valid audio files found in dataset directory.")

    # Save LabelEncoder
    save_encoder_path = Path(encoder_path)
    save_encoder_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(label_encoder, save_encoder_path)
        logger.info("Saved LabelEncoder to %s", save_encoder_path.resolve())
    except Exception as err:
        logger.error("Failed saving LabelEncoder: %s", err)

    # Stratified 80/10/10 split
    counts = np.bincount(y_encoded)
    min_count = int(np.min(counts))
    use_stratify = len(np.unique(y_encoded)) > 1 and min_count >= 2

    train_paths, temp_paths, y_train, y_temp = train_test_split(
        audio_paths,
        y_encoded,
        test_size=0.20,
        random_state=random_state,
        stratify=y_encoded if use_stratify else None,
    )

    temp_counts = np.bincount(y_temp)
    min_temp = int(np.min(temp_counts))
    use_stratify_temp = len(np.unique(y_temp)) > 1 and min_temp >= 2

    val_paths, test_paths, y_val, y_test = train_test_split(
        temp_paths,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp if use_stratify_temp else None,
    )

    logger.info(
        "Split: %d train / %d val / %d test",
        len(train_paths), len(val_paths), len(test_paths),
    )

    def process_paths(paths: List[Path], is_training: bool) -> np.ndarray:
        features_list: List[np.ndarray] = []
        for p in paths:
            try:
                sig = load_and_preprocess_audio(p)
                if is_training and augment_training:
                    sig = random_augmentation(sig)
                feat = extract_features(sig)
                features_list.append(feat)
            except Exception as err:
                logger.warning("Feature extraction failed for %s: %s", p, err)
                features_list.append(np.zeros(189, dtype=np.float32))
        return np.array(features_list, dtype=np.float32)

    logger.info("Extracting features from %d audio files...", len(audio_paths))
    X_train = process_paths(train_paths, is_training=True)
    X_val   = process_paths(val_paths,   is_training=False)
    X_test  = process_paths(test_paths,  is_training=False)

    logger.info(
        "Feature extraction complete. X_train: %s  X_val: %s  X_test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


if __name__ == "__main__":
    load_dataset()
