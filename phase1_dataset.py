#!/usr/bin/env python3
""" 
Phase 1: Environment Setup and Dataset Preparation for Colon Cancer Classification
LC25000 Dataset (Colon only)

INSTRUCTIONS:
1. Download LC25000 from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia or search LC25000
2. Use only colon folders: colon_aca -> colon_images/cancer/, colon_n -> colon_images/normal/
3. Install: pip install tensorflow-macos tensorflow-metal keras numpy pandas matplotlib scikit-learn opencv-python
4. python phase1_dataset.py
"""

import os
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

# === CONFIGURATION ===
MAX_IMAGES_PER_CLASS = 200
SEED = 42

def set_seed(seed=SEED):
    """Set global random seed for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)

def get_sample_image(folder):
    """Safely get first image from folder with multiple extensions"""
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        files = list(folder.glob(ext))
        if files:
            return files[0]
    raise ValueError(f"No images found in {folder}")


def verify_environment():
    print('=== Environment Verification ===')
    print(f'Python: {sys.version.split()[0]}')
    if tf is None:
        print('TensorFlow missing - pip install tensorflow-macos tensorflow-metal')
        return False
    print(f'TensorFlow: {tf.__version__}')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('GPU: Yes (Metal)')
        print(f'Devices: { [d.name for d in gpus] }')
    else:
        print('GPU: No (CPU)')
    print('Environment OK')
    print()
    return True

def setup_directories():
    print('=== Directory Setup ===')
    RAW_DIR = Path('colon_images')
    DATASET_DIR = Path('dataset')
    for split in ['train', 'val', 'test']:
        for cls in ['cancer', 'normal']:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    print(f'Dataset ready: {DATASET_DIR}')
    raw_cancer = RAW_DIR / 'cancer'
    raw_normal = RAW_DIR / 'normal'
    if not raw_cancer.exists() or not raw_normal.exists():
        raise ValueError('Create colon_images/cancer/ and normal/ with LC25000 images')
    img_exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.tiff', '*.TIFF', '*.tif', '*.TIF']
    cancer_paths = [p for ext in img_exts for p in raw_cancer.glob(ext)]
    normal_paths = [p for ext in img_exts for p in raw_normal.glob(ext)]
    
    # Shuffle before limiting for randomness
    random.shuffle(cancer_paths)
    random.shuffle(normal_paths)
    
    # Limit for experimentation
    cancer_paths = cancer_paths[:MAX_IMAGES_PER_CLASS]
    normal_paths = normal_paths[:MAX_IMAGES_PER_CLASS]
    print(f"Using {len(cancer_paths)} cancer and {len(normal_paths)} normal images (limited to {MAX_IMAGES_PER_CLASS} per class)")
    print()
    return RAW_DIR, DATASET_DIR, cancer_paths, normal_paths

def split_dataset(raw_dir, dataset_dir, cancer_paths, normal_paths, seed=42):
    print('=== Splitting 70/15/15 ===')
    random.seed(seed)
    for cls, paths in [('cancer', cancer_paths), ('normal', normal_paths)]:
        if not paths:
            continue
        random.shuffle(paths)
        n = len(paths)
        train_end = int(0.7 * n)
        val_end = train_end + int(0.15 * n)
        train_paths = paths[:train_end]
        val_paths = paths[train_end:val_end]
        test_paths = paths[val_end:]
        for split_name, split_paths in [('train', train_paths), ('val', val_paths), ('test', test_paths)]:
            target = dataset_dir / split_name / cls
            if len(list(target.glob('*'))) > 0:
                print(f"{target} already contains data, skipping...")
                continue
            for p in split_paths:
                shutil.copy2(p, target / p.name)
        print(f'{cls}: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}')
    print()

def verify_split(dataset_dir):
    print('=== Verification ===')
    img_exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.tiff', '*.TIFF', '*.tif', '*.TIF']
    counts = {}
    for split in ['train', 'val', 'test']:
        cancer_count = len([p for ext in img_exts for p in (dataset_dir / split / 'cancer').glob(ext)])
        normal_count = len([p for ext in img_exts for p in (dataset_dir / split / 'normal').glob(ext)])
        counts[split] = {'cancer': cancer_count, 'normal': normal_count}
        print(f'{split.upper()} Cancer: {cancer_count}, Normal: {normal_count}')
    # Check balance
    for split, c in counts.items():
        balance = abs(c['cancer'] - c['normal']) / ((c['cancer'] + c['normal']) / 2)
        status = '✅' if balance < 0.05 else '⚠️'
        print(f'{split} Balance: {balance:.1%} {status}')
    print()

def visualize_samples(dataset_dir):
    print('=== Visualization ===')
    train_cancer = get_sample_image(dataset_dir / 'train' / 'cancer')
    train_normal = get_sample_image(dataset_dir / 'train' / 'normal')
    img_c = cv2.imread(str(train_cancer))
    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
    img_n = cv2.imread(str(train_normal))
    img_n = cv2.cvtColor(img_n, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_c)
    axes[0].set_title('Cancer')
    axes[0].axis('off')
    axes[1].imshow(img_n)
    axes[1].set_title('Normal')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    print('Samples displayed.')

def main():
    set_seed(SEED)
    if not verify_environment():
        return
    try:
        raw_dir, dataset_dir, cancer_paths, normal_paths = setup_directories()
        split_dataset(raw_dir, dataset_dir, cancer_paths, normal_paths)
        verify_split(dataset_dir)
        visualize_samples(dataset_dir)
        print('Phase 1 COMPLETE!')
        print('Dataset ready in dataset/ for Phase 2 (EfficientNet training).')
    except Exception as e:
        print(f'Error: {e}')
        print('Ensure colon_images/ exists with images.')

if __name__ == '__main__':
    main()

