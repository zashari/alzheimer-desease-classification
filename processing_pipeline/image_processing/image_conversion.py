# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 6: 2D IMAGE CONVERSION
# =====================================================================
# This script converts the 2D NIfTI slices from the previous step
# into a standard image format (PNG) for easier visualization and
# use in deep learning frameworks.

from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

from configs import config
from utils import utils

def convert_nifti_to_png(logger):
    """
    Converts organized 2D NIfTI slices into PNG format.

    This function iterates through all labeled NIfTI files, performs
    intensity normalization and resizing, and saves them as PNG images,
    maintaining the same directory structure.

    Args:
        logger: A configured logger instance for output.
    """
    logger.info("\n🧠 STEP 6: CONVERTING NIFTI SLICES TO PNG")
    logger.info("=" * 80)
    
    input_root = config.STEP5_LABELED_DIR
    output_root = config.STEP6_2D_CONVERTED_DIR
    
    if not input_root.exists():
        logger.error(f"❌ Input directory not found: {input_root}. Aborting.")
        return
        
    # --- Gather all input NIfTI files ---
    nifti_paths = list(input_root.rglob("*.nii.gz"))
    if not nifti_paths:
        logger.error(f"❌ No NIfTI files found in {input_root} to process.")
        return
        
    logger.info(f"🔍 Found {len(nifti_paths)} NIfTI files to convert.")
    
    # --- Process Images ---
    processed_count = 0
    skipped_count = 0
    error_count = 0
    original_sizes = set()

    for in_path in tqdm(nifti_paths, desc="Converting NIfTI to PNG"):
        try:
            # Mirror the directory structure for the output
            rel_path = in_path.relative_to(input_root)
            out_path = (output_root / rel_path).with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                skipped_count += 1
                continue

            # Load the NIfTI slice
            slice_nifti = nib.load(str(in_path))
            slice_data = slice_nifti.get_fdata()
            
            # Squeeze data to remove singleton dimensions -> 2D
            slice_2d = np.squeeze(slice_data)

            # Normalize slice intensity to 0-255 range
            slice_normalized = utils.normalize_to_uint8(
                slice_2d,
                config.INTENSITY_NORM_PERCENTILE
            )

            # Convert to PIL Image (transpose for correct orientation)
            img = Image.fromarray(slice_normalized.T, mode='L')
            
            original_sizes.add(f"{img.width}x{img.height}")
            
            # Resize to target resolution using high-quality interpolation
            img_resized = utils.resize_image(
                img,
                config.TARGET_2D_SIZE,
                config.PNG_INTERPOLATION
            )
            
            # Save the final PNG image
            img_resized.save(out_path)
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {in_path}: {e}")
            error_count += 1

    # --- Final Report ---
    logger.info("\n" + "="*80)
    logger.info("✅ 2D CONVERSION COMPLETE")
    logger.info("="*80)
    logger.info(f"  - Files converted successfully: {processed_count}")
    logger.info(f"  - Files skipped (already exist): {skipped_count}")
    logger.info(f"  - Files with errors: {error_count}")
    logger.info(f"  - Original image sizes found: {sorted(list(original_sizes))}")
    logger.info(f"  - All images resized to: {config.TARGET_2D_SIZE[0]}x{config.TARGET_2D_SIZE[1]}")
    logger.info(f"  - Output located at: {output_root}")

if __name__ == '__main__':
    from utils.logging_utils import setup_logging
    
    main_logger = setup_logging("image_conversion_standalone", config.LOG_DIR)
    convert_nifti_to_png(logger=main_logger)