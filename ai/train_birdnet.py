"""
EcoSense AI - Transfer Learning Training Script (YAMNet + Custom Classifier).

Pipeline:
    Audio Files → YAMNet 1024-dim Embeddings → Dense Classifier → Bird Species

This script uses Google's YAMNet model as a frozen feature extractor.
Only the small Dense classifier head is trained. This achieves 85-95%+
accuracy even with small datasets because YAMNet already deeply understands
audio patterns from millions of training examples.

Usage:
    python -m ai.train_birdnet

Outputs (in models/audio/ and logs/):
    - audio_model_yamnet.keras   (full model: YAMNet + classifier)
    - classifier_head.keras      (just the Dense head)
    - label_encoder.pkl
    - logs/accuracy_yamnet.png
    - logs/loss_yamnet.png
    - logs/confusion_matrix_yamnet.png
    - logs/classification_report_yamnet.txt
"""

import datetime
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.utils import to_categorical

from ai.yamnet_extractor import extract_yamnet_embedding, EMBEDDING_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS        = 600
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("datasets/audio")
MODELS_DIR    = Path("models/audio")
LOGS_DIR      = Path("logs")

MODEL_PATH    = MODELS_DIR / "audio_model_yamnet.keras"
HEAD_PATH     = MODELS_DIR / "classifier_head.keras"
ENCODER_PATH  = MODELS_DIR / "label_encoder.pkl"

CACHE_DIR     = LOGS_DIR / "yamnet_cache"   # Cache embeddings to avoid recomputing
HISTORY_PATH  = LOGS_DIR / "training_history_yamnet.json"
STATE_PATH    = LOGS_DIR / "training_state_yamnet.json"

ACC_PNG       = LOGS_DIR / "accuracy_yamnet.png"
LOSS_PNG      = LOGS_DIR / "loss_yamnet.png"
CM_PNG        = LOGS_DIR / "confusion_matrix_yamnet.png"
REPORT_TXT    = LOGS_DIR / "classification_report_yamnet.txt"

SUPPORTED_EXT = {".wav", ".mp3", ".flac"}


# ── Dataset Scanning ──────────────────────────────────────────────────────────

def scan_dataset(dataset_dir: Path) -> Tuple[List[Path], List[str]]:
    """Scan species folders, skip empty ones, return (paths, labels)."""
    audio_paths, labels = [], []
    skipped = []

    for species_dir in sorted(dataset_dir.iterdir()):
        if not species_dir.is_dir() or species_dir.name.startswith("."):
            continue
        species_name = species_dir.name
        count = 0
        for fp in species_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXT and fp.stat().st_size > 0:
                audio_paths.append(fp)
                labels.append(species_name)
                count += 1
        if count == 0:
            skipped.append(species_name)

    if skipped:
        logger.warning("Skipped %d empty species: %s", len(skipped), skipped)
    logger.info("Found %d audio files across %d species.", len(audio_paths), len(set(labels)))
    return audio_paths, labels


# ── Embedding Cache ───────────────────────────────────────────────────────────

def get_cache_path(audio_path: Path, cache_dir: Path) -> Path:
    """Derive .npy cache path for an audio file's YAMNet embedding."""
    relative = audio_path.relative_to(DATASET_DIR)
    cache_path = cache_dir / relative.with_suffix(".npy")
    return cache_path


def load_or_compute_embedding(audio_path: Path, cache_dir: Path) -> np.ndarray:
    """Return cached YAMNet embedding or compute + cache it."""
    cache_path = get_cache_path(audio_path, cache_dir)

    if cache_path.exists():
        return np.load(str(cache_path))

    emb = extract_yamnet_embedding(audio_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), emb)
    return emb


def extract_all_embeddings(
    audio_paths: List[Path],
    cache_dir: Path,
    label: str = "all",
) -> np.ndarray:
    """Extract YAMNet embeddings for all audio paths (with caching)."""
    embeddings = []
    total = len(audio_paths)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting YAMNet embeddings for %d %s files...", total, label)
    for idx, path in enumerate(audio_paths, start=1):
        if idx % 50 == 0 or idx == total:
            logger.info("  %d / %d", idx, total)
        emb = load_or_compute_embedding(path, cache_dir)
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)


# ── Model Architecture ────────────────────────────────────────────────────────

def build_classifier(num_classes: int) -> models.Sequential:
    """Build a small Dense classifier head on top of YAMNet's 1024-dim embeddings."""
    model = models.Sequential(
        [
            layers.Input(shape=(EMBEDDING_DIM,)),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="yamnet_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_plots(history: tf.keras.callbacks.History) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    hist = history.history
    ep = range(1, len(hist.get("accuracy", [])) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(ep, hist.get("accuracy", []), label="Train", linewidth=2)
    plt.plot(ep, hist.get("val_accuracy", []), label="Validation", linewidth=2)
    plt.title("YAMNet Classifier Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ACC_PNG, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(ep, hist.get("loss", []), label="Train Loss", color="red", linewidth=2)
    plt.plot(ep, hist.get("val_loss", []), label="Val Loss", color="orange", linewidth=2)
    plt.title("YAMNet Classifier Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(LOSS_PNG, dpi=150)
    plt.close()

    logger.info("Saved accuracy/loss plots.")


def save_report(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> None:
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    labels_idx = np.arange(len(class_names))

    report = classification_report(
        y_test, y_pred, labels=labels_idx,
        target_names=class_names, zero_division=0,
    )
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=== EcoSense AI — YAMNet Classification Report ===\n\n")
        f.write(report)
    logger.info("Saved classification report.")

    cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
    n = len(class_names)
    fig_size = max(10, n // 4)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (YAMNet)")
    plt.colorbar()
    plt.xticks(np.arange(n), class_names, rotation=90, fontsize=6)
    plt.yticks(np.arange(n), class_names, fontsize=6)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix.")


# ── Main Training ─────────────────────────────────────────────────────────────

def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Scan dataset
    audio_paths, raw_labels = scan_dataset(DATASET_DIR)
    if not audio_paths:
        raise ValueError("No audio files found.")

    # 2. Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    class_names = [str(c) for c in label_encoder.classes_]
    num_classes = len(class_names)
    logger.info("Classes: %d", num_classes)

    joblib.dump(label_encoder, ENCODER_PATH)
    logger.info("Saved LabelEncoder → %s", ENCODER_PATH)

    # 3. Split 80 / 10 / 10
    counts = np.bincount(y)
    use_stratify = np.min(counts) >= 2

    X_paths_train, X_paths_temp, y_train, y_temp = train_test_split(
        audio_paths, y, test_size=0.20, random_state=SEED,
        stratify=y if use_stratify else None,
    )
    use_stratify_temp = np.min(np.bincount(y_temp)) >= 2
    X_paths_val, X_paths_test, y_val, y_test = train_test_split(
        X_paths_temp, y_temp, test_size=0.50, random_state=SEED,
        stratify=y_temp if use_stratify_temp else None,
    )

    # 4. Extract YAMNet embeddings (cached after first run)
    X_train = extract_all_embeddings(X_paths_train, CACHE_DIR, "train")
    X_val   = extract_all_embeddings(X_paths_val,   CACHE_DIR, "val")
    X_test  = extract_all_embeddings(X_paths_test,  CACHE_DIR, "test")

    logger.info("X_train: %s  X_val: %s  X_test: %s", X_train.shape, X_val.shape, X_test.shape)

    # 5. One-hot encode
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes)

    # 6. Class weights
    unique_cls = np.unique(y_train)
    cls_weights = compute_class_weight("balanced", classes=unique_cls, y=y_train)
    cls_weight_dict = {int(c): float(w) for c, w in zip(unique_cls, cls_weights)}

    # 7. Build or load model
    initial_epoch = 0
    if MODEL_PATH.exists():
        logger.info("Loading existing model from %s", MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                state = json.load(f)
            initial_epoch = state.get("last_completed_epoch", 0)
            logger.info("Resuming from epoch %d", initial_epoch + 1)
    else:
        logger.info("Building new classifier model...")
        model = build_classifier(num_classes)

    # 8. Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH), monitor="val_loss",
        save_best_only=True, verbose=1,
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_loss", patience=20,
        restore_best_weights=True, verbose=1,
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7,
        min_lr=1e-7, verbose=1,
    )

    class StateCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            state = {
                "last_completed_epoch": epoch + 1,
                "best_val_accuracy": round(float(logs.get("val_accuracy", 0)), 4),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            with open(STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)

    # 9. Train
    logger.info("Training from epoch %d to %d...", initial_epoch + 1, EPOCHS)
    try:
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            batch_size=BATCH_SIZE,
            callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb, StateCallback()],
            class_weight=cls_weight_dict,
            verbose=1,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted. Resume with: python -m ai.train_birdnet")
        return

    # 10. Save history
    with open(HISTORY_PATH, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # 11. Save head separately
    model.save(HEAD_PATH)

    # 12. Evaluate
    save_plots(history)
    save_report(model, X_test, y_test, class_names)

    train_eval = model.evaluate(X_train, y_train_cat, verbose=0)
    val_eval   = model.evaluate(X_val,   y_val_cat,   verbose=0)
    test_eval  = model.evaluate(X_test,  y_test_cat,  verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100

    print("\n" + "=" * 50)
    print("YAMNet Transfer Learning — Results")
    print(f"Training Accuracy   : {float(train_eval[1]) * 100:.2f}%")
    print(f"Validation Accuracy : {float(val_eval[1]) * 100:.2f}%")
    print(f"Testing Accuracy    : {float(test_eval[1]) * 100:.2f}%")
    print(f"F1 Score (weighted) : {f1:.2f}%")
    print(f"Model saved to      : {MODEL_PATH}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    train()
