"""
Dataset acquisition, validation, and preparation module for MVTec AD.

Implements:
1. Automated dataset retrieval from official/verified repositories.
2. Validation of raw directory structures and image integrity.
3. Binary adaptation of MVTec AD for supervised classification.
4. Stratified, reproducible train/validation/test splitting (70/15/15).
5. Comprehensive data leakage verification.
"""

import json
import logging
import os
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HF_API_TREE_URL = "https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/main"
HF_RESOLVE_BASE = "https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main"


def fetch_hf_tree(category: str, subpath: str = "") -> List[str]:
    """Recursively fetch list of files from the Hugging Face repository."""
    url = f"{HF_API_TREE_URL}/{category}"
    if subpath:
        url += f"/{subpath}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    file_paths = []
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read().decode())
            for item in items:
                if item.get("type") == "file" and item.get("path", "").endswith((".png", ".jpg", ".jpeg")):
                    file_paths.append(item["path"])
                elif item.get("type") == "directory" and "ground_truth" not in item.get("path", ""):
                    # Exclude ground_truth masks (as required: do not use defect masks as model input)
                    rel_sub = item["path"][len(category) + 1 :]
                    file_paths.extend(fetch_hf_tree(category, rel_sub))
    except Exception as e:
        logger.warning("Failed to fetch tree at %s: %s", url, e)
    return file_paths


def download_single_file(remote_rel_path: str, local_dest: Path) -> bool:
    """Download a single image file with retries."""
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    if local_dest.exists() and local_dest.stat().st_size > 0:
        return True

    file_url = f"{HF_RESOLVE_BASE}/{remote_rel_path}"
    req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
                if len(data) > 0:
                    with open(local_dest, "wb") as f:
                        f.write(data)
                    return True
        except Exception as e:
            if attempt == 2:
                logger.error("Failed downloading %s after 3 attempts: %s", remote_rel_path, e)
    return False


def acquire_dataset(category: str = config.DEFAULT_CATEGORY, force: bool = False) -> Path:
    """
    Acquires the MVTec AD category dataset.
    Checks existing local raw directory or downloads from official Hugging Face mirror.
    """
    raw_category_dir = config.RAW_DATA_DIR / category
    if raw_category_dir.exists() and not force:
        # Check if files are already downloaded
        train_good = list((raw_category_dir / "train" / "good").glob("*.png"))
        test_dir = raw_category_dir / "test"
        if len(train_good) > 0 and test_dir.exists():
            logger.info("Found existing raw dataset for '%s' with %d train/good images.", category, len(train_good))
            return raw_category_dir

    # Check if a tar.xz archive exists in raw directory
    archive_path = config.RAW_DATA_DIR / f"{category}.tar.xz"
    if archive_path.exists():
        logger.info("Extracting local archive %s...", archive_path)
        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(path=config.RAW_DATA_DIR)
        return raw_category_dir

    logger.info("Acquiring '%s' from official MVTec AD mirror on Hugging Face...", category)
    files = fetch_hf_tree(category)
    if not files:
        raise RuntimeError(
            f"Could not retrieve file listing for MVTec AD category '{category}'. "
            "Ensure internet access or place the raw dataset manually in "
            f"'{raw_category_dir}'."
        )

    logger.info("Discovered %d image files for category '%s'. Downloading concurrently...", len(files), category)
    download_tasks = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for remote_file in files:
            # relative path inside category, e.g. 'bottle/train/good/000.png' -> 'train/good/000.png'
            rel = Path(remote_file).relative_to(category)
            dest = raw_category_dir / rel
            download_tasks.append(executor.submit(download_single_file, remote_file, dest))

        success_count = sum(1 for task in as_completed(download_tasks) if task.result())

    logger.info("Successfully downloaded %d / %d images for '%s'.", success_count, len(files), category)
    if success_count < len(files) * 0.9:
        raise RuntimeError(f"Download incomplete: only {success_count}/{len(files)} downloaded successfully.")

    return raw_category_dir


def validate_image(path: Path) -> bool:
    """Checks that an image file is readable and non-corrupt."""
    try:
        with Image.open(path) as img:
            img.verify()
        # Verify can actually load pixels
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def collect_dataset_samples(category: str = config.DEFAULT_CATEGORY) -> Tuple[List[Dict], Dict]:
    """
    Scans the raw category directory and constructs records for binary classification.
    Excludes ground_truth mask folders.

    MVTec AD directory structure:
    - category/train/good/*.png -> Label: normal (0)
    - category/test/good/*.png -> Label: normal (0)
    - category/test/<defect_type>/*.png -> Label: defective (1)
    """
    raw_category_dir = config.RAW_DATA_DIR / category
    if not raw_category_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found at: {raw_category_dir}")

    samples = []
    defect_counts = {}
    corrupted_files = []

    # 1. Train Good images -> Normal
    train_good_dir = raw_category_dir / "train" / "good"
    if train_good_dir.exists():
        for img_path in sorted(train_good_dir.glob("*.png")):
            if validate_image(img_path):
                samples.append({
                    "path": str(img_path.resolve()),
                    "filename": img_path.name,
                    "class_name": "normal",
                    "label": config.CLASS_TO_IDX["normal"],
                    "defect_type": "none",
                    "original_split": "train"
                })
            else:
                corrupted_files.append(str(img_path))

    # 2. Test Good images -> Normal
    test_good_dir = raw_category_dir / "test" / "good"
    if test_good_dir.exists():
        for img_path in sorted(test_good_dir.glob("*.png")):
            if validate_image(img_path):
                samples.append({
                    "path": str(img_path.resolve()),
                    "filename": img_path.name,
                    "class_name": "normal",
                    "label": config.CLASS_TO_IDX["normal"],
                    "defect_type": "none",
                    "original_split": "test"
                })
            else:
                corrupted_files.append(str(img_path))

    # 3. Test Defect images -> Defective
    test_dir = raw_category_dir / "test"
    if test_dir.exists():
        for defect_folder in sorted(test_dir.iterdir()):
            if defect_folder.is_dir() and defect_folder.name != "good":
                defect_name = defect_folder.name
                defect_counts[defect_name] = 0
                for img_path in sorted(defect_folder.glob("*.png")):
                    if validate_image(img_path):
                        samples.append({
                            "path": str(img_path.resolve()),
                            "filename": img_path.name,
                            "class_name": "defective",
                            "label": config.CLASS_TO_IDX["defective"],
                            "defect_type": defect_name,
                            "original_split": "test"
                        })
                        defect_counts[defect_name] += 1
                    else:
                        corrupted_files.append(str(img_path))

    summary = {
        "category": category,
        "total_valid_samples": len(samples),
        "normal_samples": sum(1 for s in samples if s["label"] == 0),
        "defective_samples": sum(1 for s in samples if s["label"] == 1),
        "defect_type_breakdown": defect_counts,
        "corrupted_files_count": len(corrupted_files),
        "corrupted_files": corrupted_files
    }

    logger.info(
        "Dataset '%s' collected: %d total (%d normal, %d defective). Corrupt: %d.",
        category,
        summary["total_valid_samples"],
        summary["normal_samples"],
        summary["defective_samples"],
        summary["corrupted_files_count"]
    )
    return samples, summary


def prepare_splits(category: str = config.DEFAULT_CATEGORY, force: bool = False) -> Dict:
    """
    Splits samples into Train (70%), Validation (15%), and Test (15%) sets.
    Uses stratified splitting to maintain class proportions across splits.
    Copies images to `data/processed/<category>/<split>/<class_name>/`.
    Saves split metadata to `data/splits_metadata.json`.
    """
    processed_dir = config.PROCESSED_DATA_DIR / category
    metadata_file = config.SPLITS_METADATA_PATH

    if processed_dir.exists() and metadata_file.exists() and not force:
        with open(metadata_file, "r") as f:
            meta = json.load(f)
            if meta.get("category") == category:
                logger.info("Processed dataset splits already exist for category '%s'.", category)
                return meta

    # Acquire and collect samples
    acquire_dataset(category, force=force)
    samples, raw_summary = collect_dataset_samples(category)

    if raw_summary["normal_samples"] == 0 or raw_summary["defective_samples"] == 0:
        raise ValueError("Cannot perform binary classification: missing normal or defective samples.")

    # Stratified split: first split train (70%) vs temp (30%)
    labels = [s["label"] for s in samples]
    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=labels,
        random_state=config.RANDOM_SEED
    )

    # Split temp (30%) into val (15%) and test (15%) equally
    temp_labels = [s["label"] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.5,
        stratify=temp_labels,
        random_state=config.RANDOM_SEED
    )

    # Strict Data Leakage Check
    train_paths = set(s["path"] for s in train_samples)
    val_paths = set(s["path"] for s in val_samples)
    test_paths = set(s["path"] for s in test_samples)

    assert len(train_paths.intersection(val_paths)) == 0, "Data Leakage: Train and Validation sets overlap!"
    assert len(train_paths.intersection(test_paths)) == 0, "Data Leakage: Train and Test sets overlap!"
    assert len(val_paths.intersection(test_paths)) == 0, "Data Leakage: Validation and Test sets overlap!"

    # Populate processed directory
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    split_manifest = {"train": train_samples, "val": val_samples, "test": test_samples}
    stats = {}

    for split_name, items in split_manifest.items():
        for class_name in config.CLASS_NAMES:
            (processed_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

        n_norm = 0
        n_def = 0
        for i, item in enumerate(items):
            src_path = Path(item["path"])
            c_name = item["class_name"]
            dest_filename = f"{i:04d}_{src_path.name}"
            dest_path = processed_dir / split_name / c_name / dest_filename
            shutil.copy2(src_path, dest_path)
            item["processed_path"] = str(dest_path.resolve())
            if c_name == "normal":
                n_norm += 1
            else:
                n_def += 1

        stats[split_name] = {
            "total": len(items),
            "normal": n_norm,
            "defective": n_def,
            "defect_ratio": round(n_def / max(1, len(items)), 4)
        }

    splits_metadata = {
        "category": category,
        "seed": config.RANDOM_SEED,
        "raw_summary": raw_summary,
        "splits": stats,
        "manifest": {
            k: [{"processed_path": item["processed_path"], "label": item["label"], "defect_type": item["defect_type"]} for item in v]
            for k, v in split_manifest.items()
        }
    }

    with open(metadata_file, "w") as f:
        json.dump(splits_metadata, f, indent=2)

    logger.info("Dataset splits prepared successfully for '%s':", category)
    for sp, s in stats.items():
        logger.info("  %s: %d images (Normal: %d, Defective: %d)", sp.upper(), s["total"], s["normal"], s["defective"])

    return splits_metadata


def get_dataset_summary() -> Dict:
    """Returns dataset split summary from metadata or raises if not prepared."""
    if not config.SPLITS_METADATA_PATH.exists():
        return {}
    with open(config.SPLITS_METADATA_PATH, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    prepare_splits(config.DEFAULT_CATEGORY, force=True)
