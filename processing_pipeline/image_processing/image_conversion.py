# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 6: 2D IMAGE CONVERSION
# =====================================================================
# This script converts the 2D NIfTI slices into PNG format.
# It performs normalization and resizing to prepare the images for
# deep learning models.

from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

from configs import config
from utils import utils

def _convert_slice_to_png(task: tuple):
    """Worker function to convert a single NIfTI slice to a PNG file."""
    nii_path, out_path = task
    try:
        # Load the NIfTI file (already a single 2D slice)
        slice_img = nib.load(str(nii_path))
        slice_data = np.squeeze(slice_img.get_fdata())

        # Normalize slice intensity to 0-255 range
        slice_normalized = utils.normalize_to_uint8(slice_data, config.INTENSITY_NORM_PERCENTILE)

        # Convert to PIL Image (transpose for correct orientation)
        img = Image.fromarray(slice_normalized.T, mode='L')
        
        # Resize to target resolution
        img_resized = utils.resize_image(img, config.TARGET_2D_SIZE, config.PNG_INTERPOLATION)

        # Save the final PNG file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_resized.save(out_path)
        
        return "success"
    except Exception as e:
        return f"error: {e}"

def convert_nifti_to_png():
    """
    Finds all 2D NIfTI slices from the labeling step and converts them
    to PNG format in parallel.
    """
    print("\n🧠 STEP 6: 2D NIFTI SLICE TO PNG CONVERSION")
    print("=" * 60)

    # Define input directories for each slice type
    slice_type_dirs = {
        "axial": config.STEP4_SLICES_AXIAL_DIR,
        "coronal": config.STEP4_SLICES_CORONAL_DIR,
        "sagittal": config.STEP4_SLICES_SAGITTAL_DIR,
    }

    # Gather all NIfTI files to be processed
    tasks = []
    for slice_type, slice_dir in slice_type_dirs.items():
        if not slice_dir.exists():
            print(f"⚠️ Warning: Input directory for {slice_type} not found at {slice_dir}")
            continue
        
        for nii_path in slice_dir.rglob("*.nii.gz"):
            # Construct the output path mirroring the input structure but under STEP6_2D_CONVERTED_DIR
            relative_path = nii_path.relative_to(slice_dir)
            out_path = config.STEP6_2D_CONVERTED_DIR / slice_type / relative_path.with_suffix(".png")
            tasks.append((nii_path, out_path))

    if not tasks:
        print("❌ No NIfTI slice files found to convert. Aborting.")
        return

    print(f"🔍 Found {len(tasks)} NIfTI slices to convert to PNG.")
    print(f"🚀 Starting conversion with {config.MAX_THREADS} parallel threads...")
    
    stats = defaultdict(int)
    with ThreadPoolExecutor(max_workers=config.MAX_THREADS) as exe:
        futures = {exe.submit(_convert_slice_to_png, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Converting to PNG"):
            stats[fut.result()] += 1

    print("\n📋 2D Conversion Summary:")
    for k, v in stats.items():
        print(f"  - {k.capitalize()}: {v}")
    print(f"\n✅ Step 6: 2D conversion complete! Output at {config.STEP6_2D_CONVERTED_DIR}")

if __name__ == '__main__':
    convert_nifti_to_png()
