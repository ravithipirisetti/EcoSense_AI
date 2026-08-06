"""
EcoSense AI Bird Image Classification - Preprocessing & Object Detection Module.

This module processes raw bird images from datasets/images/ by utilizing YOLOv8 object
detection to locate and crop bird bounding boxes (ignoring background). Cropped regions
(or original images if no bird is detected) are resized to 224x224, pixel-normalized,
and saved under datasets/processed_images/ preserving folder structure.
"""

import logging
from pathlib import Path
from typing import Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_INPUT_DIR: Path = Path("datasets/images")
DEFAULT_OUTPUT_DIR: Path = Path("datasets/processed_images")
TARGET_SIZE: Tuple[int, int] = (224, 224)
SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BIRD_CLASS_ID: int = 14  # COCO class index for 'bird'


def detect_and_crop_bird(
    image_bgr: np.ndarray, yolo_model: YOLO, conf_threshold: float = 0.25
) -> Tuple[np.ndarray, bool]:
    """Detect bird in BGR image using YOLOv8 and crop the bounding box.

    Args:
        image_bgr (np.ndarray): Original OpenCV image array in BGR format.
        yolo_model (YOLO): Loaded Ultralytics YOLO model instance.
        conf_threshold (float): Detection confidence threshold.

    Returns:
        Tuple[np.ndarray, bool]: Cropped BGR bird image (or original image if no bird found),
                                 and boolean flag indicating whether a bird was detected.
    """
    height, width = image_bgr.shape[:2]

    try:
        results = yolo_model(image_bgr, verbose=False, conf=conf_threshold)[0]
    except Exception as err:
        logger.warning("YOLO detection failed: %s. Returning original image.", err)
        return image_bgr, False

    best_box = None
    max_conf = -1.0

    # Search for detections matching bird class ID (14 in COCO schema)
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls_id == BIRD_CLASS_ID and conf > max_conf:
                max_conf = conf
                best_box = box.xyxy[0].cpu().numpy()

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        # Ensure bounding coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        if (x2 - x1) > 10 and (y2 - y1) > 10:
            cropped = image_bgr[y1:y2, x1:x2]
            return cropped, True

    return image_bgr, False


def preprocess_single_image(
    image_path: Path,
    output_path: Path,
    yolo_model: YOLO,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[bool, bool, Optional[np.ndarray]]:
    """Preprocess single image: load, detect bird, crop, resize 224x224, and normalize.

    Args:
        image_path (Path): Source image filepath.
        output_path (Path): Destination filepath to save processed image.
        yolo_model (YOLO): Loaded YOLOv8 object detection model.
        target_size (Tuple[int, int]): Dimensions for resizing (width, height).

    Returns:
        Tuple[bool, bool, Optional[np.ndarray]]:
            - processed_successfully (bool)
            - bird_detected (bool)
            - normalized_array (Optional[np.ndarray]): Normalized float32 numpy array [0.0, 1.0].
    """
    if not image_path.is_file():
        logger.warning("File does not exist: %s", image_path)
        return False, False, None

    try:
        if image_path.stat().st_size == 0:
            logger.warning("Skipping 0-byte file: %s", image_path)
            return False, False, None
    except OSError:
        return False, False, None

    # Load image via OpenCV
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None or image_bgr.size == 0:
        logger.warning("Failed to read image with OpenCV: %s", image_path)
        return False, False, None

    # Detect bird and crop bounding box
    cropped_bgr, bird_detected = detect_and_crop_bird(image_bgr, yolo_model)

    # Resize to target 224x224 dimensions
    resized_bgr = cv2.resize(cropped_bgr, target_size, interpolation=cv2.INTER_AREA)

    # Normalize pixels to range [0.0, 1.0] float32 array
    rgb_image = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    normalized_array = rgb_image.astype(np.float32) / 255.0

    # Save processed image into datasets/processed_images using Pillow/OpenCV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pil_img = Image.fromarray(rgb_image)
        pil_img.save(output_path, quality=95)
    except Exception as err:
        logger.warning("Pillow save failed for %s (%s), falling back to OpenCV.", output_path, err)
        cv2.imwrite(str(output_path), resized_bgr)

    return True, bird_detected, normalized_array


def preprocess_image_dataset(
    input_dir: Union[str, Path] = DEFAULT_INPUT_DIR,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    yolo_weights: str = "yolov8n.pt",
) -> None:
    """Preprocess entire image dataset from input_dir and save into output_dir.

    Maintains directory folder structure, crops detected birds, resizes to 224x224,
    normalizes pixels, and displays processing statistics.

    Args:
        input_dir (Union[str, Path]): Input directory (e.g. datasets/images/).
        output_dir (Union[str, Path]): Output directory (e.g. datasets/processed_images/).
        yolo_weights (str): Weights name or path for YOLOv8 model.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists() or not in_path.is_dir():
        logger.error("Input image directory does not exist: %s", in_path.resolve())
        print(f"Error: Directory {in_path} not found.")
        return

    logger.info("Loading YOLOv8 object detection model (%s)...", yolo_weights)
    try:
        yolo_model = YOLO(yolo_weights)
    except Exception as err:
        logger.critical("Failed to load YOLO model: %s", err, exc_info=True)
        return

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    bird_detected_count = 0

    logger.info("Processing images from %s -> %s...", in_path.resolve(), out_path.resolve())

    # Traverse input directory and maintain folder structure
    for file_path in in_path.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                skipped_count += 1
                logger.debug("Skipping unsupported non-image file: %s", file_path)
                continue

            # Compute destination path preserving folder hierarchy
            relative_subpath = file_path.relative_to(in_path)
            dest_file_path = out_path / relative_subpath

            success, bird_found, _ = preprocess_single_image(
                image_path=file_path,
                output_path=dest_file_path,
                yolo_model=yolo_model,
                target_size=TARGET_SIZE,
            )

            if success:
                processed_count += 1
                if bird_found:
                    bird_detected_count += 1
                logger.info("Processed: %s -> %s", relative_subpath, dest_file_path)
            else:
                failed_count += 1
                logger.error("Failed to process image: %s", file_path)

    # Display processing summary statistics
    print("\n" + "=" * 50)
    print("      IMAGE PREPROCESSING SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Processed Images : {processed_count}")
    print(f"  - Bird Cropped : {bird_detected_count}")
    print(f"  - Full Resized : {processed_count - bird_detected_count}")
    print(f"Skipped Images   : {skipped_count}")
    print(f"Failed Images    : {failed_count}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    preprocess_image_dataset()
