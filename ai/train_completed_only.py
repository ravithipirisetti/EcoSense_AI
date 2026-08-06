"""
EcoSense AI - High Accuracy Deep Classifier Training (66 Species).
Uses Focal Loss + Cosine Similarity Density Head + Data Augmentation to push Validation Accuracy toward 90%+.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.utils import to_categorical

from ai.yamnet_extractor import extract_yamnet_embedding, EMBEDDING_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
DATASET_DIR = Path("datasets/audio")
MODELS_DIR = Path("models/audio")
LOGS_DIR = Path("logs")

MODEL_PATH = MODELS_DIR / "audio_model_yamnet.keras"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
CACHE_DIR = LOGS_DIR / "yamnet_cache"
MIN_FILES = 15   # Include all species with at least 15 clips
EPOCHS = 750     # Train for exactly 750 epochs


def scan_filtered_dataset():
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
    
    logger.info("Total included species: %d", len(included_species))
    for name, count in included_species:
        logger.info("  - %s: %d files", name, count)
    return audio_paths, labels


def get_embedding(path: Path) -> np.ndarray:
    relative = path.relative_to(DATASET_DIR)
    cache_path = CACHE_DIR / relative.with_suffix(".npy")
    if cache_path.exists():
        return np.load(str(cache_path))
    emb = extract_yamnet_embedding(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), emb)
    return emb


def apply_mixup_and_augmentation(X: np.ndarray, y: np.ndarray, alpha: float = 0.2, num_aug: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Generate Mixup + Jitter augmented feature representations for higher generalization."""
    X_list = [X]
    y_list = [y]
    n_samples = len(X)
    rng = np.random.default_rng(SEED)

    for _ in range(num_aug):
        # 1. Feature Noise & Scaling Jitter
        noise = rng.normal(0, 0.015, size=X.shape).astype(np.float32)
        scale = rng.uniform(0.96, 1.04, size=(n_samples, 1)).astype(np.float32)
        X_jit = (X * scale) + noise
        
        # 2. Sample-to-Sample Mixup
        indices = rng.permutation(n_samples)
        lam = rng.beta(alpha, alpha, size=(n_samples, 1)).astype(np.float32)
        X_mix = lam * X_jit + (1.0 - lam) * X_jit[indices]
        y_mix = lam * y + (1.0 - lam) * y[indices]

        X_list.append(X_mix)
        y_list.append(y_mix)

    return np.vstack(X_list), np.vstack(y_list)


def build_cosine_classifier(num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(EMBEDDING_DIM,))
    
    # Feature Projection with Layer Normalization
    x = layers.Dense(512, kernel_initializer="he_normal")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.3)(x)
    
    # Bottleneck Representation
    x = layers.Dense(256, kernel_initializer="he_normal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.2)(x)
    
    # L2 Normalized Embeddings
    norm_embeddings = layers.UnitNormalization(axis=-1)(x)
    
    # Softmax Classification Head
    outputs = layers.Dense(num_classes, activation="softmax")(norm_embeddings)
    
    return models.Model(inputs=inputs, outputs=outputs, name="YAMNet_CosineClassifier")


def train():
    audio_paths, raw_labels = scan_filtered_dataset()
    if not audio_paths:
        logger.error("No species met the minimum file requirement.")
        return

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    # Save LabelEncoder atomically
    joblib.dump(label_encoder, ENCODER_PATH)
    logger.info("Saved LabelEncoder with %d species to %s", num_classes, ENCODER_PATH)

    X_train_p, X_temp_p, y_train_raw, y_temp_raw = train_test_split(
        audio_paths, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_val_p, X_test_p, y_val_raw, y_test_raw = train_test_split(
        X_temp_p, y_temp_raw, test_size=0.50, random_state=SEED, stratify=y_temp_raw
    )

    X_train_orig = np.array([get_embedding(p) for p in X_train_p], dtype=np.float32)
    X_val = np.array([get_embedding(p) for p in X_val_p], dtype=np.float32)
    X_test = np.array([get_embedding(p) for p in X_test_p], dtype=np.float32)

    y_train_cat_orig = to_categorical(y_train_raw, num_classes)
    y_val_cat = to_categorical(y_val_raw, num_classes)
    y_test_cat = to_categorical(y_test_raw, num_classes)

    # Apply Mixup + Augmentation (4x training set)
    X_train, y_train_cat = apply_mixup_and_augmentation(X_train_orig, y_train_cat_orig, num_aug=3)
    logger.info("Augmented dataset: %d -> %d samples", len(X_train_orig), len(X_train))

    cls_weights = compute_class_weight("balanced", classes=np.unique(y_train_raw), y=y_train_raw)
    cls_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_train_raw), cls_weights)}

    model = build_cosine_classifier(num_classes)

    # Categorical Focal Crossentropy for hard sample learning
    loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, label_smoothing=0.03)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    checkpoint_cb = callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    early_stop_cb = callbacks.EarlyStopping(monitor="val_accuracy", patience=750, mode="max", restore_best_weights=True, verbose=1)
    reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=15, mode="max", min_lr=1e-6, verbose=1)

    logger.info("Starting High-Accuracy Training (%d species, %d samples, max epochs: %d)...", num_classes, len(X_train), EPOCHS)
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        class_weight=cls_weight_dict,
        verbose=1
    )

    eval_res = model.evaluate(X_test, y_test_cat, verbose=0)
    print("\n" + "="*60)
    print(f"High-Accuracy Model Results ({num_classes} Species, {len(audio_paths)} Files)")
    print(f"Test Loss    : {eval_res[0]:.4f}")
    print(f"Test Accuracy: {eval_res[1]*100:.2f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    train()
