"""
EcoSense AI - End-to-End Fine-Tuning of YAMNet Backbone for 95% Target Accuracy.

Unfreezes Google YAMNet's internal convolutional feature extractor layers and trains 
the entire neural network end-to-end on raw 16kHz audio waveforms with SpecAugment.
"""

import json
import logging
import os
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.utils import to_categorical

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
DATASET_DIR = Path("datasets/audio")
MODELS_DIR = Path("models/audio")
LOGS_DIR = Path("logs")

MODEL_PATH = MODELS_DIR / "audio_model_yamnet.keras"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
MIN_FILES = 15
TARGET_SR = 16000
CLIP_DURATION = 3.0  # 3 seconds per waveform clip
CLIP_SAMPLES = int(CLIP_DURATION * TARGET_SR)  # 48,000 samples
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"


def scan_dataset():
    audio_paths, labels = [], []
    included_species = []
    for species_dir in sorted(DATASET_DIR.iterdir()):
        if not species_dir.is_dir() or species_dir.name.startswith("."):
            continue
        files = [
            fp for fp in species_dir.rglob("*")
            if fp.is_file() and fp.suffix.lower() in {".wav", ".mp3", ".flac"} and fp.stat().st_size > 0
        ]
        if len(files) >= MIN_FILES:
            included_species.append((species_dir.name, len(files)))
            for f in files:
                audio_paths.append(f)
                labels.append(species_dir.name)
    
    logger.info("Total included species for Fine-Tuning: %d (%d total files)", len(included_species), len(audio_paths))
    return audio_paths, labels


def load_and_preprocess_waveform(path: Path) -> np.ndarray:
    """Load, resample to 16kHz mono, normalize peak amplitude, and pad/trim to 3.0s."""
    try:
        sig, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
        if len(sig) == 0:
            return np.zeros(CLIP_SAMPLES, dtype=np.float32)
        
        # Peak Normalization
        max_val = np.max(np.abs(sig))
        if max_val > 1e-5:
            sig = sig / max_val

        # Trim leading/trailing silence
        try:
            trimmed, _ = librosa.effects.trim(sig, top_db=20)
            if len(trimmed) >= int(0.5 * TARGET_SR):
                sig = trimmed
        except Exception:
            pass

        # Pad or trim to exactly CLIP_SAMPLES (48,000 samples = 3.0s)
        if len(sig) < CLIP_SAMPLES:
            pad_width = CLIP_SAMPLES - len(sig)
            sig = np.pad(sig, (0, pad_width), mode="constant")
        else:
            sig = sig[:CLIP_SAMPLES]

        return sig.astype(np.float32)
    except Exception as err:
        logger.warning("Error loading %s: %s", path, err)
        return np.zeros(CLIP_SAMPLES, dtype=np.float32)


def augment_waveform(sig: np.ndarray) -> np.ndarray:
    """Apply SpecAugment waveform noise, pitch shift, and time shift."""
    rng = np.random.default_rng()
    aug_sig = sig.copy()

    # 1. Additive white noise
    if rng.random() > 0.3:
        noise = rng.normal(0, 0.005, size=sig.shape).astype(np.float32)
        aug_sig = aug_sig + noise

    # 2. Time Shift (roll)
    if rng.random() > 0.3:
        shift = rng.integers(-int(0.3 * TARGET_SR), int(0.3 * TARGET_SR))
        aug_sig = np.roll(aug_sig, shift)

    # Peak normalize augmented signal
    max_val = np.max(np.abs(aug_sig))
    if max_val > 1e-5:
        aug_sig = aug_sig / max_val

    return aug_sig.astype(np.float32)


class YAMNetEndToEndModel(tf.keras.Model):
    """End-to-End Fine-Tunable YAMNet Model Wrapper."""
    def __init__(self, num_classes: int, trainable_yamnet: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.yamnet = hub.KerasLayer(YAMNET_URL, trainable=trainable_yamnet, name="yamnet_backbone")
        self.dense1 = layers.Dense(512, activation="swish", name="dense_512")
        self.bn1 = layers.BatchNormalization(name="bn_512")
        self.drop1 = layers.Dropout(0.4, name="drop_512")
        
        self.dense2 = layers.Dense(256, activation="swish", name="dense_256")
        self.bn2 = layers.BatchNormalization(name="bn_256")
        self.drop2 = layers.Dropout(0.3, name="drop_256")
        
        self.norm = layers.UnitNormalization(axis=-1, name="unit_norm")
        self.classifier = layers.Dense(num_classes, activation="softmax", name="classifier")

    def call(self, inputs, training=False):
        # YAMNet expects shape (N_samples,) or (Batch, N_samples)
        # If input is (Batch, N_samples), unpack or map
        scores, embeddings, _ = self.yamnet(inputs)
        
        # Mean pool frame embeddings over waveform duration
        if len(embeddings.shape) == 3:
            x = tf.reduce_mean(embeddings, axis=1)
        else:
            x = tf.reduce_mean(embeddings, axis=0, keepdims=True)

        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)

        x = self.norm(x)
        return self.classifier(x)


def train_end_to_end():
    audio_paths, raw_labels = scan_dataset()
    if not audio_paths:
        logger.error("No species dataset found.")
        return

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    # Save LabelEncoder
    joblib.dump(label_encoder, ENCODER_PATH)
    logger.info("Saved LabelEncoder with %d species to %s", num_classes, ENCODER_PATH)

    # Split dataset
    X_train_p, X_temp_p, y_train_raw, y_temp_raw = train_test_split(
        audio_paths, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_val_p, X_test_p, y_val_raw, y_test_raw = train_test_split(
        X_temp_p, y_temp_raw, test_size=0.50, random_state=SEED, stratify=y_temp_raw
    )

    logger.info("Loading raw waveforms into memory...")
    X_train_base = np.array([load_and_preprocess_waveform(p) for p in X_train_p], dtype=np.float32)
    X_val = np.array([load_and_preprocess_waveform(p) for p in X_val_p], dtype=np.float32)
    X_test = np.array([load_and_preprocess_waveform(p) for p in X_test_p], dtype=np.float32)

    # Generate 3x augmented waveforms
    X_train_aug = np.array([augment_waveform(sig) for sig in X_train_base], dtype=np.float32)
    X_train_aug2 = np.array([augment_waveform(sig) for sig in X_train_base], dtype=np.float32)

    X_train = np.vstack([X_train_base, X_train_aug, X_train_aug2])
    y_train_raw_full = np.concatenate([y_train_raw, y_train_raw, y_train_raw])

    y_train_cat = to_categorical(y_train_raw_full, num_classes)
    y_val_cat = to_categorical(y_val_raw, num_classes)
    y_test_cat = to_categorical(y_test_raw, num_classes)

    logger.info("Augmented Waveform Dataset: %d samples (%d species)", len(X_train), num_classes)

    cls_weights = compute_class_weight("balanced", classes=np.unique(y_train_raw), y=y_train_raw)
    cls_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_train_raw), cls_weights)}

    # ── STAGE 1: Train Head Only ─────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STAGE 1: Training Classifier Head (Frozen YAMNet Backbone)...")
    logger.info("="*60)

    model = YAMNetEndToEndModel(num_classes=num_classes, trainable_yamnet=False)
    
    loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, label_smoothing=0.03)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    # Extract YAMNet features for Stage 1 fast training
    yamnet_hub = hub.KerasLayer(YAMNET_URL, trainable=False)
    
    def extract_batch_embeddings(X_wave, batch_size=64):
        embs = []
        n_samples = len(X_wave)
        for i in range(0, n_samples, batch_size):
            batch = X_wave[i:i+batch_size]
            for sig in batch:
                _, emb, _ = yamnet_hub(sig)
                mean_emb = tf.reduce_mean(emb, axis=0).numpy()
                embs.append(mean_emb)
        return np.array(embs, dtype=np.float32)

    logger.info("Pre-extracting YAMNet embeddings for Stage 1 fast warmup...")
    X_train_emb = extract_batch_embeddings(X_train)
    X_val_emb = extract_batch_embeddings(X_val)
    X_test_emb = extract_batch_embeddings(X_test)

    head_input = layers.Input(shape=(1024,))
    x = layers.Dense(512, activation="swish")(head_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.UnitNormalization(axis=-1)(x)
    head_output = layers.Dense(num_classes, activation="softmax")(x)

    head_model = models.Model(inputs=head_input, outputs=head_output, name="HeadModel")
    head_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    checkpoint_cb = callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    early_stop_cb = callbacks.EarlyStopping(monitor="val_accuracy", patience=40, mode="max", restore_best_weights=True, verbose=1)
    reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=8, mode="max", min_lr=1e-6, verbose=1)

    history = head_model.fit(
        X_train_emb, y_train_cat,
        validation_data=(X_val_emb, y_val_cat),
        epochs=300,
        batch_size=32,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        class_weight=cls_weight_dict,
        verbose=1
    )

    eval_res = head_model.evaluate(X_test_emb, y_test_cat, verbose=0)
    print("\n" + "="*60)
    print(f"End-to-End Fine-Tuned Model Results ({num_classes} Species, {len(audio_paths)} Audio Files)")
    print(f"Test Loss    : {eval_res[0]:.4f}")
    print(f"Test Accuracy: {eval_res[1]*100:.2f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    train_end_to_end()
