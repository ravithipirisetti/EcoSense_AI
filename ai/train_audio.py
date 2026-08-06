"""
EcoSense AI - Production Audio Classifier Training Script.

Pipeline Architecture:
    Load Audio -> Preprocess Audio -> Training Only: Random Augmentation -> Extract 189 Features -> Label Encoding -> Train/Val/Test Split -> Keras DNN Training

This module manages the complete, robust audio classification training workflow:
- Always uses strict `ai.*` package imports.
- Loads preprocessed dataset splits (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder) from ai.dataset_loader.
- Computes balanced class weights for imbalanced species datasets.
- Automatically resumes from existing models/audio/audio_model.keras and logs/training_state.json.
- Trains a Keras Sequential DNN classifier (Dense, BatchNormalization, Dropout, Softmax) using Adam and Categorical Crossentropy.
- Utilizes ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, and TensorBoard callbacks.
- Generates accuracy.png, loss.png, confusion_matrix.png, and classification_report.txt safely (even with zero-sample species).
"""

import datetime
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.utils import to_categorical

# Strict ai.* package import
from ai.dataset_loader import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reproducibility Seed Initialization
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
EPOCHS: int = 300
BATCH_SIZE: int = 16
LEARNING_RATE: float = 0.001

# System Paths and Directory Constants
DATASET_DIR: Path = Path("datasets/audio")
MODELS_DIR: Path = Path("models/audio")
LOGS_DIR: Path = Path("logs")

MODEL_PATH: Path = MODELS_DIR / "audio_model.keras"
ENCODER_PATH: Path = MODELS_DIR / "label_encoder.pkl"

STATE_JSON_PATH: Path = LOGS_DIR / "training_state.json"
HISTORY_JSON_PATH: Path = LOGS_DIR / "training_history.json"
TENSORBOARD_LOG_DIR: Path = LOGS_DIR / "tensorboard"

ACCURACY_PNG_PATH: Path = LOGS_DIR / "accuracy.png"
LOSS_PNG_PATH: Path = LOGS_DIR / "loss.png"
CONF_MATRIX_PNG_PATH: Path = LOGS_DIR / "confusion_matrix.png"
REPORT_TXT_PATH: Path = LOGS_DIR / "classification_report.txt"


class TrainingStateCallback(callbacks.Callback):
    """Custom Keras Callback to persist epoch state into logs/training_state.json."""

    def __init__(self, state_path: Path = STATE_JSON_PATH) -> None:
        super().__init__()
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_val_acc: float = 0.0
        self.best_val_loss: float = float("inf")

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        logs = logs or {}
        val_acc = float(logs.get("val_accuracy", 0.0))
        val_loss = float(logs.get("val_loss", float("inf")))

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        current_state = {
            "last_completed_epoch": epoch + 1,
            "best_validation_accuracy": round(self.best_val_acc, 4),
            "best_validation_loss": round(self.best_val_loss, 4),
            "training_timestamp": datetime.datetime.now().isoformat(),
        }

        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(current_state, f, indent=4)
        except Exception as err:
            logger.error("Failed updating training_state.json: %s", err)


def build_audio_model(input_dim: int, num_classes: int) -> models.Sequential:
    """Construct and compile the TensorFlow/Keras Sequential Deep Neural Network.

    Args:
        input_dim (int): Feature dimensionality (expected 189).
        num_classes (int): Number of target species classes.

    Returns:
        models.Sequential: Compiled Sequential Keras model.
    """
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            # Layer 1
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            # Layer 2
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # Layer 3
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            # Output Layer
            layers.Dense(num_classes),
            layers.Softmax(),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    logger.info("Compiled Keras Sequential model with Adam (lr=%.4f).", LEARNING_RATE)
    model.summary()
    return model


def save_training_plots(
    history: tf.keras.callbacks.History,
    accuracy_path: Path = ACCURACY_PNG_PATH,
    loss_path: Path = LOSS_PNG_PATH,
) -> None:
    """Generate and save accuracy.png and loss.png into logs/."""
    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    loss_path.parent.mkdir(parents=True, exist_ok=True)

    history_dict = history.history
    epochs_range = range(1, len(history_dict.get("accuracy", [])) + 1)

    # 1. Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history_dict.get("accuracy", []), label="Train Accuracy", linewidth=2)
    plt.plot(epochs_range, history_dict.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
    plt.title("EcoSense Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    # 2. Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history_dict.get("loss", []), label="Train Loss", linewidth=2, color="red")
    plt.plot(epochs_range, history_dict.get("val_loss", []), label="Validation Loss", linewidth=2, color="orange")
    plt.title("EcoSense Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()


def save_confusion_matrix_and_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    cm_path: Path = CONF_MATRIX_PNG_PATH,
    report_path: Path = REPORT_TXT_PATH,
) -> None:
    """Generate and save confusion_matrix.png and classification_report.txt safely."""
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    labels_indices = np.arange(len(class_names))

    # Safe Classification Report generation handling zero-audio species
    report_str = classification_report(
        y_true,
        y_pred,
        labels=labels_indices,
        target_names=class_names,
        zero_division=0,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== EcoSense AI Classification Report ===\n\n")
        f.write(report_str)

    # Safe Confusion Matrix calculation
    cm = confusion_matrix(y_true, y_pred, labels=labels_indices)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("EcoSense Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()


def plot_confusion_matrix_and_report(
    model: tf.keras.Model,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    label_encoder: Any,
) -> None:
    """Compatibility helper for evaluating and plotting metrics safely."""
    class_names = [str(cls) for cls in label_encoder.classes_]
    y_pred_probs = model.predict(x_eval, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    save_confusion_matrix_and_report(y_eval, y_pred, class_names)


def train() -> None:
    """Execute complete production training pipeline."""
    # Ensure system directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset splits (7 items returned from ai.dataset_loader)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder,
    ) = load_dataset()

    class_names = [str(cls) for cls in label_encoder.classes_]
    num_classes = len(class_names)
    input_dim = X_train.shape[1]

    # Compute balanced class weights safely
    unique_train_classes = np.unique(y_train)
    if len(unique_train_classes) > 0:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_train_classes,
            y=y_train,
        )
        class_weights_dict = {
            int(cls): float(weight)
            for cls, weight in zip(unique_train_classes, class_weights)
        }
    else:
        class_weights_dict = None

    # One-hot categorical label encoding
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    initial_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")

    # Resume state from logs/training_state.json if present
    if STATE_JSON_PATH.is_file():
        try:
            with open(STATE_JSON_PATH, "r", encoding="utf-8") as f:
                saved_state = json.load(f)
            initial_epoch = saved_state.get("last_completed_epoch", 0)
            best_val_acc = saved_state.get("best_validation_accuracy", 0.0)
            best_val_loss = saved_state.get("best_validation_loss", float("inf"))
            logger.info("Resuming training from epoch %d", initial_epoch + 1)
        except Exception as err:
            logger.warning("Could not load logs/training_state.json: %s", err)

    # Resume existing model or construct a new model
    if MODEL_PATH.is_file():
        print("Loading existing model...")
        logger.info("Loading existing model from %s...", MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("Creating new model...")
        logger.info("Creating new model...")
        model = build_audio_model(input_dim=input_dim, num_classes=num_classes)

    # Configure Callbacks
    state_cb = TrainingStateCallback(state_path=STATE_JSON_PATH)
    state_cb.best_val_acc = best_val_acc
    state_cb.best_val_loss = best_val_loss

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    tensorboard_cb = None
    try:
        import tensorboard  # noqa: F401 - optional dependency
        tensorboard_cb = callbacks.TensorBoard(
            log_dir=str(TENSORBOARD_LOG_DIR),
            histogram_freq=1,
        )
        logger.info("TensorBoard logging enabled at %s", TENSORBOARD_LOG_DIR)
    except ImportError:
        logger.warning("TensorBoard not installed — skipping TensorBoard callback. Install with: pip install tensorboard")

    # Train model with exception and interruption handling
    try:
        logger.info("Training from epoch %d to %d...", initial_epoch + 1, EPOCHS)
        history = model.fit(
            X_train,
            y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            batch_size=BATCH_SIZE,
            callbacks=[
                cb for cb in [
                    checkpoint_cb,
                    early_stopping_cb,
                    reduce_lr_cb,
                    tensorboard_cb,
                    state_cb,
                ] if cb is not None
            ],
            class_weight=class_weights_dict,
            verbose=1,
        )
    except (KeyboardInterrupt, Exception) as err:
        print("\nTraining interrupted.")
        logger.warning("Training interrupted: %s", err)
        print("Resume later using:")
        print("python -m ai.train_audio\n")
        return

    # Save logs/training_history.json
    history_data: Dict[str, List[float]] = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4)

    save_training_plots(history)

    # Evaluation on Test set
    plot_confusion_matrix_and_report(model, X_test, y_test, label_encoder)

    # Calculate metrics
    train_eval = model.evaluate(X_train, y_train_cat, verbose=0)
    val_eval = model.evaluate(X_val, y_val_cat, verbose=0)
    test_eval = model.evaluate(X_test, y_test_cat, verbose=0)

    train_acc = float(train_eval[1]) * 100.0
    val_acc = float(val_eval[1]) * 100.0
    test_acc = float(test_eval[1]) * 100.0

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    labels_indices = np.arange(num_classes)
    test_precision = float(precision_score(y_test, y_pred, labels=labels_indices, average="weighted", zero_division=0)) * 100.0
    test_recall = float(recall_score(y_test, y_pred, labels=labels_indices, average="weighted", zero_division=0)) * 100.0
    test_f1 = float(f1_score(y_test, y_pred, labels=labels_indices, average="weighted", zero_division=0)) * 100.0

    final_best_val_acc = state_cb.best_val_acc * 100.0
    final_best_val_loss = state_cb.best_val_loss

    # Print Final Training Summary
    print("\n" + "=" * 40)
    print("Training Completed")
    print(f"Training Accuracy       : {train_acc:.2f}%")
    print(f"Validation Accuracy     : {val_acc:.2f}%")
    print(f"Testing Accuracy        : {test_acc:.2f}%")
    print(f"Precision               : {test_precision:.2f}%")
    print(f"Recall                  : {test_recall:.2f}%")
    print(f"F1 Score                : {test_f1:.2f}%")
    print(f"Best Validation Accuracy: {final_best_val_acc:.2f}%")
    print(f"Best Validation Loss    : {final_best_val_loss:.4f}")
    print(f"Model Path              : {MODEL_PATH}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    train()
