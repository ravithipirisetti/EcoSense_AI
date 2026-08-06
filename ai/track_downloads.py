"""
EcoSense AI - Live Terminal Dataset & Download Tracker (Target: 50 per species).

Run this command in your PowerShell terminal to track live progress:
    python -m ai.track_downloads
"""

import os
import sys
import time
from pathlib import Path

DATASET_DIR = Path("datasets/audio")
TARGET_PER_SPECIES = 50


def track():
    if not DATASET_DIR.exists():
        print("Dataset directory missing!")
        return

    species_folders = sorted(
        [d for d in DATASET_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    total_species = len(species_folders)
    completed_50 = 0
    in_progress = 0
    total_files = 0

    # Clear terminal output
    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 72)
    print(" 🦅 EcoSense AI — Live Dataset Download Tracker (Target: 50 per species)")
    print("=" * 72)
    print(f"{'#':<4} {'Species Folder':<38} {'Files':<10} {'Status':<18}")
    print("-" * 72)

    for idx, folder in enumerate(species_folders, start=1):
        files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in {".wav", ".mp3", ".flac"}
        ]
        count = len(files)
        total_files += count

        if count >= TARGET_PER_SPECIES:
            status = "✅ Ready (50)"
            completed_50 += 1
        elif count > 5:
            status = f"📥 Downloading ({count}/{TARGET_PER_SPECIES})"
            in_progress += 1
        else:
            status = f"⏳ Waiting ({count}/{TARGET_PER_SPECIES})"

        # Display species that are downloading or completed
        if count >= 15 or "Downloading" in status:
            print(f"{idx:<4} {folder.name:<38} {count:<10} {status:<18}")

    pct = (total_files / (total_species * TARGET_PER_SPECIES)) * 100
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    print("-" * 72)
    print(f"📊 SUMMARY DASHBOARD:")
    print(f"   • Progress Bar           : [{bar}] {pct:.1f}%")
    print(f"   • Total Species Folders  : {total_species}")
    print(f"   • Completed (50 Files)   : {completed_50} / {total_species}")
    print(f"   • Active / In-Progress   : {in_progress}")
    print(f"   • Total Audio Files Now  : {total_files}")
    print("=" * 72)
    print("Press Ctrl+C to exit tracker.\n")


if __name__ == "__main__":
    try:
        while True:
            track()
            time.sleep(3)  # Refresh every 3 seconds
    except KeyboardInterrupt:
        print("\nTracker closed.")
