#!/usr/bin/env python3
"""
Phase 1: Dataset Preparation (FINAL CLEAN VERSION)
"""

import sys
import shutil
import random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except ImportError:
    tf = None

# CONFIG
MAX_IMAGES_PER_CLASS = 1000
SEED = 42


# =========================
# SET SEED
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


# =========================
# VERIFY ENV
# =========================
def verify_environment():
    print("=== Environment Check ===")
    print(f"Python: {sys.version.split()[0]}")
    if tf is None:
        print("❌ TensorFlow not installed")
        return False

    print(f"TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print("GPU:", "Yes (Metal)" if gpus else "CPU only")
    print("Environment OK\n")
    return True


# =========================
# SETUP DIRECTORIES
# =========================
def setup_directories():
    RAW_DIR = Path("colon_images")
    DATASET_DIR = Path("dataset")

    # Create structure
    for split in ["train", "val", "test"]:
        for cls in ["cancer", "normal"]:
            path = DATASET_DIR / split / cls
            if path.exists():
                shutil.rmtree(path)   # 🔥 CLEAN OLD DATA
            path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    cancer_dir = RAW_DIR / "cancer"
    normal_dir = RAW_DIR / "normal"

    if not cancer_dir.exists() or not normal_dir.exists():
        raise ValueError("❌ Missing colon_images/cancer or normal")

    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.tif']

    cancer_paths = []
    normal_paths = []

    for ext in img_exts:
        cancer_paths.extend(cancer_dir.glob(ext))
        normal_paths.extend(normal_dir.glob(ext))

    random.shuffle(cancer_paths)
    random.shuffle(normal_paths)

    cancer_paths = cancer_paths[:MAX_IMAGES_PER_CLASS]
    normal_paths = normal_paths[:MAX_IMAGES_PER_CLASS]

    print(f"Using {len(cancer_paths)} cancer & {len(normal_paths)} normal images\n")

    return DATASET_DIR, cancer_paths, normal_paths


# =========================
# SPLIT DATA
# =========================
def split_dataset(dataset_dir, cancer_paths, normal_paths):
    print("=== Splitting Dataset ===")

    for cls, paths in [("cancer", cancer_paths), ("normal", normal_paths)]:
        random.shuffle(paths)

        n = len(paths)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        splits = {
            "train": paths[:train_end],
            "val": paths[train_end:val_end],
            "test": paths[val_end:]
        }

        for split, split_paths in splits.items():
            for p in split_paths:
                shutil.copy2(p, dataset_dir / split / cls / p.name)

        print(f"{cls}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    print()


# =========================
# VERIFY SPLIT
# =========================
def verify_split(dataset_dir):
    print("=== Verification ===")

    for split in ["train", "val", "test"]:
        c = len(list((dataset_dir / split / "cancer").glob("*")))
        n = len(list((dataset_dir / split / "normal").glob("*")))
        print(f"{split.upper()} → Cancer: {c}, Normal: {n}")

    print()


# =========================
# VISUALIZE
# =========================
def visualize(dataset_dir):
    print("=== Sample Images ===")

    c_img = next((dataset_dir / "train" / "cancer").glob("*"))
    n_img = next((dataset_dir / "train" / "normal").glob("*"))

    img_c = cv2.cvtColor(cv2.imread(str(c_img)), cv2.COLOR_BGR2RGB)
    img_n = cv2.cvtColor(cv2.imread(str(n_img)), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_c)
    plt.title("Cancer")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_n)
    plt.title("Normal")
    plt.axis("off")

    plt.show()


# =========================
# MAIN
# =========================
def main():
    set_seed()

    if not verify_environment():
        return

    dataset_dir, cancer, normal = setup_directories()
    split_dataset(dataset_dir, cancer, normal)
    verify_split(dataset_dir)
    visualize(dataset_dir)

    print("✅ Dataset ready for training!")


if __name__ == "__main__":
    main()