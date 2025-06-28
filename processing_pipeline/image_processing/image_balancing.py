# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 9: DATA BALANCING & AUGMENTATION
# =====================================================================
# This module balances the training dataset by augmenting minority classes.
# It ensures that for any given augmented subject, the *same* random
# transformations are applied across all slice planes and timepoints to
# maintain temporal and structural integrity.

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
from tqdm import tqdm

from configs import config
from utils import utils

class MRIAugmenter:
    """Class for MRI-specific data augmentation techniques."""
    def __init__(self, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

    def elastic_deformation(self, image, alpha, sigma):
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, order=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, order=0) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    def rotation_translation(self, image, angle, tx, ty):
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    def bias_field_simulation(self, image, scale):
        rows, cols = image.shape
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        X, Y = np.meshgrid(x, y)
        bias = 1 + scale * (np.random.randn() * X**2 + np.random.randn() * Y**2 + np.random.randn() * X * Y)
        return image * bias

    def rician_noise(self, image, sigma):
        img_float = image.astype(np.float32)
        noise_real = np.random.normal(0, sigma, image.shape)
        noise_imag = np.random.normal(0, sigma, image.shape)
        noisy = np.sqrt((img_float + noise_real)**2 + noise_imag**2)
        return noisy

def _generate_augmentation_params():
    """Generates one set of random augmentation parameters for a new subject."""
    return {
        'elastic': {'alpha': random.uniform(25, 35), 'sigma': random.uniform(4, 6)},
        'rotation': {'angle': random.uniform(-8, 8), 'tx': random.uniform(-8, 8), 'ty': random.uniform(-8, 8)},
        'bias_scale': random.uniform(0.2, 0.4),
        'rician_noise': {'sigma': random.uniform(8, 12)}
    }

def _apply_augmentations(image, params, augmenter):
    """Applies a consistent set of augmentations to a single image."""
    img_aug = augmenter.elastic_deformation(image, **params['elastic'])
    img_aug = augmenter.rotation_translation(img_aug, **params['rotation'])
    img_aug = augmenter.bias_field_simulation(img_aug, params['bias_scale'])
    img_aug = augmenter.rician_noise(img_aug, params['rician_noise']['sigma'])
    return np.clip(img_aug, 0, 255).astype(np.uint8)

def balance_dataset(logger):
    """
    Balances the training dataset by augmenting minority classes using a
    temporally consistent approach.
    """
    logger.info("\n🧠 STEP 9: DATASET BALANCING & AUGMENTATION (Temporally Consistent)")
    logger.info("=" * 80)
    
    input_root = config.STEP8_ENHANCED_DIR
    output_root = config.STEP9_BALANCED_DIR
    augmenter = MRIAugmenter(seed=config.RANDOM_SEED)
    augmentation_log = {}

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    logger.info(f"Cleaned and created output directory: {output_root}")
    
    logger.info("Copying validation and test sets...")
    for slice_type in config.SLICE_TYPES:
        for split in ["val", "test"]:
            src_dir = input_root / slice_type / split
            if src_dir.exists():
                shutil.copytree(src_dir, dst=(output_root / slice_type / split), dirs_exist_ok=True)

    logger.info("Analyzing training set and generating augmentation plan...")
    subject_counts = defaultdict(int)
    source_subjects = defaultdict(list)
    ref_train_dir = input_root / "coronal" / "train" # Use one plane as reference
    
    for class_dir in ref_train_dir.iterdir():
        if class_dir.is_dir() and class_dir.name in config.CLASSES_TO_INCLUDE:
            subjects = {utils.extract_subject_id_from_filename(f.name) for f in class_dir.glob("*.png")}
            subject_counts[class_dir.name] = len(subjects)
            source_subjects[class_dir.name] = list(subjects)

    target_count = max(config.AUGMENTATION_TARGETS.values())
    logger.info(f"Current subject counts in train set: {dict(subject_counts)}")
    logger.info(f"Target subjects per class after balancing: {target_count}")

    augmentation_plan = defaultdict(list)
    for class_name in config.CLASSES_TO_INCLUDE:
        num_to_generate = config.AUGMENTATION_TARGETS.get(class_name, target_count) - subject_counts[class_name]
        if num_to_generate > 0:
            source_ids = random.choices(source_subjects[class_name], k=num_to_generate)
            for i, source_id in enumerate(source_ids):
                aug_id = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}{i}"
                params = _generate_augmentation_params()
                augmentation_plan[class_name].append({
                    "aug_id": aug_id, "source_subject_id": source_id, "params": params
                })
    
    augmentation_log['plan'] = augmentation_plan

    logger.info("Executing augmentation and copying original training files...")
    for slice_type in config.SLICE_TYPES:
        logger.info(f"\n--- Processing Plane: {slice_type.upper()} ---")
        src_train_dir = input_root / slice_type / "train"
        dest_train_dir = output_root / slice_type / "train"
        
        for class_name in config.CLASSES_TO_INCLUDE:
            src_class_dir = src_train_dir / class_name
            dest_class_dir = dest_train_dir / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy originals
            if src_class_dir.exists():
                for f_path in src_class_dir.glob("*.png"):
                    shutil.copy2(f_path, dest_class_dir / f_path.name)
            
            # Augment based on the pre-generated plan
            if class_name in augmentation_plan:
                for plan_item in tqdm(augmentation_plan[class_name], desc=f"Augmenting {class_name} ({slice_type})"):
                    source_id = plan_item["source_subject_id"]
                    params = plan_item["params"]
                    
                    source_files = list(src_class_dir.glob(f"{source_id}_*.png"))
                    for src_path in source_files:
                        img = np.array(Image.open(src_path))
                        img_aug = _apply_augmentations(img, params, augmenter)
                        out_filename = f"AUG_{plan_item['aug_id']}_{src_path.name}"
                        Image.fromarray(img_aug).save(dest_class_dir / out_filename)

    log_path = output_root / 'augmentation_log.json'
    with open(log_path, 'w') as f:
        json.dump(augmentation_log, f, indent=4)
    logger.info(f"\nAugmentation log saved to {log_path}")
    logger.info("✅ Step 9: Data balancing complete!")