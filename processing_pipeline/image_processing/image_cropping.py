# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 7: IMAGE CROPPING
# =====================================================================
# This script performs center cropping, resizing, and rotation of the
# PNG images to prepare them for the enhancement and training steps.

from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

from configs import config
from utils import utils

def get_distribution_stats(root_dir, slice_types, splits, classes, logger=None, use_recursive=False):
    """
    Get distribution statistics of files across slice types, splits, and classes.
    
    Args:
        root_dir: Root directory to search
        slice_types: List of slice types to look for
        splits: List of splits to look for
        classes: List of classes to look for
        logger: Optional logger instance
        use_recursive: Whether to search recursively in subdirectories (True for input dir, False for output dir)
    """
    total_files = 0
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for slice_type in slice_types:
        slice_total = 0
        for split in splits:
            split_total = 0
            for cls in classes:
                class_dir = root_dir / slice_type / split / cls
                if class_dir.exists():
                    # Use rglob for input directory (with subject subdirs) and glob for output directory
                    class_pngs = list(class_dir.rglob("*.png") if use_recursive else class_dir.glob("*.png"))
                    count = len(class_pngs)
                    stats[slice_type][split][cls] = count
                    split_total += count
                    if logger:
                        logger.info(f"  {split}/{cls}: {count} files")
            
            if split_total > 0 and logger:
                logger.info(f"  {split} total: {split_total} files")
            slice_total += split_total
        
        if logger:
            logger.info(f"  {slice_type} total: {slice_total} files\n")
        total_files += slice_total
    
    return stats, total_files

def crop_images(logger=None):
    """
    Performs center cropping, resizing, and rotation on PNG images.
    Creates a standardized set of images for subsequent processing steps.
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)

    logger.info("\n🧠 STEP 7: IMAGE CROPPING")
    logger.info("=" * 60)

    # Ensure output directories exist (simplified structure)
    for slice_type in config.SLICE_TYPES:
        for split in config.SPLITS:
            for cls in config.CLASSES_TO_INCLUDE:
                (config.STEP7_CROPPED_DIR / slice_type / split / cls).mkdir(parents=True, exist_ok=True)

    # Gather all PNG files
    png_paths = list(config.STEP6_2D_CONVERTED_DIR.rglob("*.png"))
    logger.info(f"Found {len(png_paths)} PNG files to process")

    # Show input distribution - use recursive search for input
    logger.info("\nInput distribution:")
    input_stats, input_total = get_distribution_stats(
        config.STEP6_2D_CONVERTED_DIR,
        config.SLICE_TYPES,
        config.SPLITS,
        config.CLASSES_TO_INCLUDE,
        logger,
        use_recursive=True  # Use recursive search for input directory
    )

    # Process images
    processed_count = 0
    skipped_count = 0

    for in_path in tqdm(png_paths, desc="Cropping, resizing & rotating"):
        # Extract components from the input path
        rel_parts = in_path.relative_to(config.STEP6_2D_CONVERTED_DIR).parts
        slice_type, split, cls = rel_parts[0:3]
        
        # Get the filename without the subject directory
        filename = in_path.name
        
        # Create the output path directly in the class directory
        out_path = config.STEP7_CROPPED_DIR / slice_type / split / cls / filename

        # Skip if output already exists
        if out_path.exists():
            skipped_count += 1
            continue

        # Load as grayscale
        img = Image.open(in_path).convert("L")
        arr = np.array(img)

        # Find bright (brain) pixels
        ys, xs = np.where(arr > 0)
        if len(ys) == 0:
            # Nothing to crop: just resize, rotate, save
            img.resize(config.TARGET_2D_SIZE, Image.BILINEAR) \
               .rotate(config.CROP_ROTATION_ANGLE, expand=True) \
               .save(out_path)
            processed_count += 1
            continue

        # Bounding box of the brain
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        # Add padding
        y0 = max(0, y0 - config.CROP_PADDING)
        x0 = max(0, x0 - config.CROP_PADDING)
        y1 = min(arr.shape[0], y1 + config.CROP_PADDING)
        x1 = min(arr.shape[1], x1 + config.CROP_PADDING)

        # Crop to brain + padding, then resize and rotate
        cropped = img.crop((x0, y0, x1, y1))
        resized = cropped.resize(config.TARGET_2D_SIZE, Image.BILINEAR)
        rotated = resized.rotate(config.CROP_ROTATION_ANGLE, expand=True)

        # Save
        rotated.save(out_path)
        processed_count += 1

    # Show final report
    logger.info("\n" + "="*60)
    logger.info("CROPPING COMPLETE")
    logger.info("="*60)
    logger.info(f"Files processed: {processed_count}")
    logger.info(f"Files skipped (already exist): {skipped_count}")
    logger.info(f"Total files handled: {processed_count + skipped_count}")

    # Show output distribution - don't use recursive search for output
    logger.info("\nOutput distribution:")
    output_stats, output_total = get_distribution_stats(
        config.STEP7_CROPPED_DIR,
        config.SLICE_TYPES,
        config.SPLITS,
        config.CLASSES_TO_INCLUDE,
        logger,
        use_recursive=False  # Don't use recursive search for output directory
    )

    logger.info(f"\n✅ Step 7: Cropping complete! Output at {config.STEP7_CROPPED_DIR}")

if __name__ == '__main__':
    crop_images() 