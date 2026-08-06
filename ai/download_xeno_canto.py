"""
EcoSense AI - Fast Parallel Xeno-canto Downloader (Multi-threaded).

Downloads up to TARGET_PER_SPECIES using 4 parallel download threads.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR: Path = Path("datasets/audio")
TARGET_PER_SPECIES: int = 50          # Target total clips per species
ACCEPTED_QUALITY: set = {"A", "B", "C", "D"}  # Include D-quality for scarce species
MAX_WORKERS: int = 8                  # Download 8 files simultaneously
DOWNLOAD_TIMEOUT: int = 45            # Seconds per file download
API_TIMEOUT: int = 30                 # Seconds for API requests
XC_API_BASE: str = "https://xeno-canto.org/api/3/recordings"

# Alternate English name aliases for scarce species that return few results
ALT_NAMES: dict = {
    "Common Myna":                  ["Indian Myna", "Acridotheres tristis"],
    "Indian Pied Myna":             ["Pied Myna", "Asian Pied Starling", "Gracupica contra"],
    "Jungle Myna":                  ["Acridotheres fuscus"],
    "Common Kestrel":               ["Eurasian Kestrel", "Falco tinnunculus"],
    "Peregrine Falcon":             ["Falco peregrinus"],
    "White-eyed Buzzard":           ["Butastur teesa"],
    "Brown Fish Owl":               ["Ketupa zeylonensis"],
    "Eastern Barn Owl":             ["Common Barn Owl", "Tyto alba"],
    "Indian Eagle-Owl":             ["Rock Eagle-Owl", "Bubo bengalensis"],
    "Eurasian Hoopoe":              ["Common Hoopoe", "Upupa epops"],
    "Little Swift":                 ["Apus affinis"],
    "Rock Dove":                    ["Feral Pigeon", "Columba livia"],
    "Pied Kingfisher":              ["Ceryle rudis"],
    "Chestnut-headed Bee-eater":    ["Merops leschenaulti"],
    "Blue-tailed Bee-eater":        ["Merops philippinus"],
    "White-bellied Drongo":         ["Dicrurus caerulescens"],
    "Indian Jungle Crow":           ["Large-billed Crow", "Corvus macrorhynchos"],
    "White-naped Woodpecker":       ["Chrysocolaptes festivus"],
    "Streak-throated Woodpecker":   ["Picus xanthopygaeus"],
    "Indian Pitta":                 ["Pitta brachyura"],
    "Indian Nightjar":              ["Caprimulgus asiaticus"],
    "Brahminy Kite":                ["Haliastur indus"],
    "White-rumped Munia":           ["White-rumped Mannikin", "Lonchura striata"],
    "Scaly-breasted Munia":         ["Spotted Munia", "Lonchura punctulata"],
    "Indian Silverbill":            ["Euodice malabarica"],
    "Red Avadavat":                 ["Strawberry Finch", "Amandava amandava"],
    "Yellow-billed Babbler":        ["Turdoides affinis"],
    "Indian Roller":                ["Blue Jay", "Coracias benghalensis"],
    "Eurasian Collared Dove":       ["Collared Dove", "Streptopelia decaocto"],
    "Laughing Dove":                ["Spilopelia senegalensis"],
    "Spotted Dove":                 ["Spilopelia chinensis"],
    "Red Junglefowl":               ["Gallus gallus"],
    "Indian Peafowl":               ["Peacock", "Pavo cristatus"],
    "Common Iora":                  ["Aegithina tiphia"],
    "Indian Robin":                 ["Copsychus fulicatus"],
    "Oriental Magpie-Robin":        ["Copsychus saularis"],
    "White-rumped Shama":           ["Copsychus malabaricus"],
    "Loten's Sunbird":              ["Long-billed Sunbird", "Cinnyris lotenius"],
    "Common Tailorbird":            ["Orthotomus sutorius"],
    "Common Babbler":               ["Argya caudata"],
    "Indian Grey Hornbill":         ["Ocyceros birostris"],
    "Black-winged Kite":            ["Black-shouldered Kite", "Elanus caeruleus"],
    "Coppersmith Barbet":           ["Crimson-breasted Barbet", "Psilopogon haemacephalus"],
}


def _load_api_key() -> str:
    key = os.environ.get("XC_API_KEY", "").strip()
    if key:
        return key
    key_file = Path("xc_api_key.txt")
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    return ""

XC_API_KEY: str = _load_api_key()


def get_common_name(folder_name: str) -> str:
    parts = folder_name.split("_", 1)
    return parts[1].strip() if len(parts) > 1 else folder_name.strip()


def count_existing_audio(species_dir: Path) -> int:
    count = 0
    for ext in ("*.wav", "*.mp3", "*.flac"):
        count += len(list(species_dir.glob(ext)))
    return count


def build_en_query(species_name: str) -> str:
    words = species_name.strip().split()
    return " ".join(f"en:{w}" for w in words)


def search_xeno_canto(species_name: str, page: int = 1) -> Optional[dict]:
    if not XC_API_KEY:
        logger.error("No API key found in xc_api_key.txt!")
        return None

    # Build a list of all query variants to try
    queries_to_try = []
    en_tags = build_en_query(species_name)
    queries_to_try.append(f"{en_tags} cnt:india")  # Primary: India only
    queries_to_try.append(en_tags)                  # Fallback: worldwide

    # Alternate names / scientific names for scarce species
    for alt in ALT_NAMES.get(species_name, []):
        alt_tags = build_en_query(alt)
        queries_to_try.append(f"{alt_tags} cnt:india")
        queries_to_try.append(alt_tags)

    for query in queries_to_try:
        try:
            params = {"query": query, "page": page, "key": XC_API_KEY}
            response = requests.get(
                XC_API_BASE, params=params, timeout=API_TIMEOUT,
                headers={"User-Agent": "EcoSenseAI/1.0 (bird classification research)"},
            )
            response.raise_for_status()
            data = response.json()
            if int(data.get("numRecordings", 0)) > 0:
                logger.debug("Query '%s' returned %s recordings", query, data.get('numRecordings'))
                return data
        except Exception as err:
            logger.debug("Query '%s' failed: %s", query, err)
            time.sleep(0.5)

    logger.warning("No recordings found for '%s' (tried %d queries)", species_name, len(queries_to_try))
    return None


def download_single_file(item: dict, species_dir: Path, species_name: str) -> bool:
    rec_id = str(item.get("id", ""))
    quality = item.get("q", "E")
    file_url = item.get("file", "")

    if not file_url:
        return False
    if file_url.startswith("//"):
        file_url = "https:" + file_url

    raw_filename = item.get("file-name", f"xc_{rec_id}.mp3")
    ext = Path(raw_filename).suffix or ".mp3"
    save_path = species_dir / f"xc_{rec_id}{ext}"

    if save_path.exists():
        return False

    try:
        with requests.get(file_url, timeout=DOWNLOAD_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16384):
                    f.write(chunk)
        logger.info("  ⚡ Downloaded XC%s (quality=%s) for %s", rec_id, quality, species_name)
        return True
    except Exception as err:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        return False


def download_species(folder_name: str) -> int:
    species_dir = DATASET_DIR / folder_name
    species_dir.mkdir(parents=True, exist_ok=True)
    species_name = get_common_name(folder_name)

    already_have = count_existing_audio(species_dir)
    if already_have >= TARGET_PER_SPECIES:
        logger.info("[SKIP] %-35s already has %d files", folder_name, already_have)
        return 0

    need = TARGET_PER_SPECIES - already_have
    logger.info("[START] %-35s need %d more (have %d)", folder_name, need, already_have)

    page = 1
    num_pages = 1
    items_to_download = []
    seen_ids = set()

    while len(items_to_download) < need and page <= num_pages:
        data = search_xeno_canto(species_name, page)
        if not data:
            break

        num_pages = int(data.get("numPages", 1))
        recordings = data.get("recordings", [])

        if not recordings:
            break

        for rec in recordings:
            if len(items_to_download) >= need:
                break
            rec_id = str(rec.get("id", ""))
            quality = rec.get("q", "E")

            if rec_id in seen_ids or quality not in ACCEPTED_QUALITY:
                continue
            seen_ids.add(rec_id)
            items_to_download.append(rec)

        page += 1
        time.sleep(0.5)

    if not items_to_download:
        return 0

    downloaded_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_single_file, item, species_dir, species_name)
            for item in items_to_download
        ]
        for future in as_completed(futures):
            if future.result():
                downloaded_count += 1

    logger.info("[DONE] %-35s downloaded %d new files (total: %d)", folder_name, downloaded_count, already_have + downloaded_count)
    return downloaded_count


def run_downloader() -> None:
    if not DATASET_DIR.exists():
        logger.error("Dataset dir missing!")
        return

    species_folders = sorted(
        [d.name for d in DATASET_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    logger.info("=" * 60)
    logger.info("EcoSense AI — Fast Parallel Xeno-canto Downloader (4x Speed)")
    logger.info("Species found : %d", len(species_folders))
    logger.info("Target per sp : %d clips", TARGET_PER_SPECIES)
    logger.info("=" * 60)

    total_new = 0
    for idx, folder in enumerate(species_folders, start=1):
        logger.info("\n[%d/%d] Processing: %s", idx, len(species_folders), folder)
        n = download_species(folder)
        total_new += n

    logger.info("\n" + "=" * 60)
    logger.info("Parallel Download complete! Total new files: %d", total_new)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_downloader()
