"""
EcoSense AI - Automated Dataset Organizer.

This module reads bird species metadata from datasets/metadata/birds.json,
automatically creates target audio and image dataset directory structures,
and organizes raw files from raw_audio/ and raw_images/ into species-specific folders
based on case-insensitive, punctuation-insensitive string matching.
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# File System Paths
METADATA_JSON_PATH: Path = Path("datasets/metadata/birds.json")
RAW_AUDIO_DIR: Path = Path("raw_audio")
RAW_IMAGES_DIR: Path = Path("raw_images")
TARGET_AUDIO_DIR: Path = Path("datasets/audio")
TARGET_IMAGES_DIR: Path = Path("datasets/images")
TARGET_UNKNOWN_DIR: Path = Path("datasets/unknown")


def normalize_string(text: str) -> str:
    """Normalize a string by removing whitespace, underscores, and hyphens, and lowercasing.

    Args:
        text (str): Input string.

    Returns:
        str: Normalized lowercase string.
    """
    return re.sub(r"[\s_\-]+", "", text).lower()


def load_bird_names(json_path: Path) -> List[str]:
    """Read bird names from metadata JSON file.

    Args:
        json_path (Path): Path to birds.json file.

    Returns:
        List[str]: List of bird names parsed from JSON.
    """
    if not json_path.is_file():
        logger.warning("Metadata JSON file missing at: %s", json_path.resolve())
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        bird_names: List[str] = []

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("birds", data.get("data", [data]))
        else:
            items = []

        for item in items:
            if isinstance(item, dict):
                name = item.get("Bird Name") or item.get("bird_name") or item.get("name")
                if name:
                    bird_names.append(str(name))
            elif isinstance(item, str):
                bird_names.append(item)

        logger.info("Loaded %d bird species from %s", len(bird_names), json_path)
        return bird_names
    except Exception as err:
        logger.error("Failed to parse metadata JSON from %s: %s", json_path, err)
        return []


def find_matching_bird(
    filename_stem: str, normalized_birds: List[Tuple[str, str]]
) -> Optional[str]:
    """Match filename stem against normalized bird names ignoring case, spaces, _, and -.

    Args:
        filename_stem (str): Filename stem without extension.
        normalized_birds (List[Tuple[str, str]]): List of (original_bird_name, normalized_name) tuples,
                                                 sorted by length descending.

    Returns:
        Optional[str]: Matched original bird name if found, else None.
    """
    norm_filename = normalize_string(filename_stem)

    for original_name, norm_name in normalized_birds:
        if norm_name in norm_filename:
            return original_name

    return None


def create_dataset_directories(bird_names: List[str]) -> None:
    """Create target dataset directories for audio, images, and unknown files.

    Args:
        bird_names (List[str]): List of bird names for subfolder creation.
    """
    TARGET_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)

    for bird in bird_names:
        (TARGET_AUDIO_DIR / bird).mkdir(parents=True, exist_ok=True)
        (TARGET_IMAGES_DIR / bird).mkdir(parents=True, exist_ok=True)


def organize_dataset() -> None:
    """Execute dataset organization by moving raw audio and image files to target folders."""
    # Step 1: Read bird names from datasets/metadata/birds.json
    bird_names = load_bird_names(METADATA_JSON_PATH)

    # Sort normalized bird names by length descending to prevent sub-word misclassification
    normalized_birds: List[Tuple[str, str]] = sorted(
        [(b, normalize_string(b)) for b in bird_names if b],
        key=lambda x: len(x[1]),
        reverse=True,
    )

    # Step 2: Automatically create datasets/audio/ and datasets/images/ subfolders
    create_dataset_directories(bird_names)

    total_audio_moved = 0
    total_images_moved = 0
    unknown_files_moved = 0

    # Step 3: Move .mp3 (and audio files) from raw_audio/ into datasets/audio/<Bird Name>/
    if RAW_AUDIO_DIR.exists() and RAW_AUDIO_DIR.is_dir():
        for file_path in RAW_AUDIO_DIR.iterdir():
            if file_path.is_file():
                matched_bird = find_matching_bird(file_path.stem, normalized_birds)
                if matched_bird:
                    dest_dir = TARGET_AUDIO_DIR / matched_bird
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    logger.info("Moved Audio: '%s' -> '%s'", file_path.name, dest_path)
                    total_audio_moved += 1
                else:
                    dest_path = TARGET_UNKNOWN_DIR / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    logger.warning("Unmatched Audio moved to Unknown: '%s'", file_path.name)
                    unknown_files_moved += 1
    else:
        logger.info("Source directory '%s' not found. Skipping audio organization.", RAW_AUDIO_DIR)

    # Step 4: Move .jpg (and image files) from raw_images/ into datasets/images/<Bird Name>/
    if RAW_IMAGES_DIR.exists() and RAW_IMAGES_DIR.is_dir():
        for file_path in RAW_IMAGES_DIR.iterdir():
            if file_path.is_file():
                matched_bird = find_matching_bird(file_path.stem, normalized_birds)
                if matched_bird:
                    dest_dir = TARGET_IMAGES_DIR / matched_bird
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    logger.info("Moved Image: '%s' -> '%s'", file_path.name, dest_path)
                    total_images_moved += 1
                else:
                    dest_path = TARGET_UNKNOWN_DIR / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    logger.warning("Unmatched Image moved to Unknown: '%s'", file_path.name)
                    unknown_files_moved += 1
    else:
        logger.info("Source directory '%s' not found. Skipping image organization.", RAW_IMAGES_DIR)

    # Step 5: Display Summary Statistics
    total_birds = len(bird_names)

    print("\n" + "=" * 50)
    print("        ORGANIZATION SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total Birds       : {total_birds}")
    print(f"Total Audio Files : {total_audio_moved}")
    print(f"Total Image Files : {total_images_moved}")
    print(f"Unknown Files     : {unknown_files_moved}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    organize_dataset()
