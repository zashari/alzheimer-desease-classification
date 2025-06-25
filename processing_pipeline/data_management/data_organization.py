# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 5: DATA ORGANIZATION & LABELING
# =====================================================================
# This script organizes the extracted 2D NIfTI slices into a structured
# directory based on their slice type, data split (train/val/test),
# and diagnostic class (e.g., AD, CN). This prepares the data for
# the subsequent conversion and training steps.

import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from configs import config
from utils import utils

def organize_slices_by_label():
    """
    Organizes the optimal 2D NIfTI slices into a labeled directory structure.
    Structure: .../slice_type/split/class/subject_id/slice_files
    """
    print("\n🧠 STEP 5: ORGANIZING AND LABELING 2D SLICES")
    print("=" * 60)

    # 1. Load metadata to map subjects to their group and split
    meta_path = config.STEP1_SPLIT_DIR / "metadata_split.csv"
    if not meta_path.exists():
        print(f"❌ Metadata file not found at {meta_path}. Cannot proceed.")
        return

    df_meta = pd.read_csv(meta_path)
    subject_to_group = dict(zip(df_meta.Subject, df_meta.Group))
    subject_to_split = dict(zip(df_meta.Subject, df_meta.Split))
    print(f"✅ Loaded metadata for {len(subject_to_group)} subjects.")

    # 2. Define input directories for each slice type
    slice_type_dirs = {
        "axial": config.STEP4_SLICES_AXIAL_DIR,
        "coronal": config.STEP4_SLICES_CORONAL_DIR,
        "sagittal": config.STEP4_SLICES_SAGITTAL_DIR,
    }

    # 3. Process each slice type
    stats = defaultdict(int)
    all_files_to_process = []
    
    print("🔍 Gathering all slice files to organize...")
    for slice_type, slice_dir in slice_type_dirs.items():
        if not slice_dir.exists():
            print(f"⚠️ Warning: Input directory for {slice_type} not found at {slice_dir}")
            continue
        all_files_to_process.extend(list(slice_dir.rglob("*.nii.gz")))
    
    if not all_files_to_process:
        print("❌ No slice files found to organize. Aborting.")
        return
        
    print(f"🚀 Found {len(all_files_to_process)} total slice files. Starting organization...")

    for src_path in tqdm(all_files_to_process, desc="Organizing slices"):
        filename = src_path.name
        subject_id = utils.extract_subject_id_from_filename(filename)

        if not subject_id:
            stats['unparsable_id'] += 1
            continue

        # Get group and split from metadata
        group = subject_to_group.get(subject_id)
        split = subject_to_split.get(subject_id)
        slice_type = utils.extract_slice_type_from_filename(filename)

        if not all([group, split, slice_type]):
            stats['missing_metadata'] += 1
            continue
            
        # Skip classes that are not in the inclusion list
        if group not in config.CLASSES_TO_INCLUDE:
            stats['skipped_class'] += 1
            continue

        # Define destination directory
        dest_dir = config.STEP5_LABELED_DIR / slice_type / split / group / subject_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        # Copy the file
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)
            stats['copied'] += 1
        else:
            stats['skipped_exist'] += 1

    print("\n📋 Organization Summary:")
    for k, v in stats.items():
        print(f"  - {k.replace('_', ' ').capitalize()}: {v}")
    
    print(f"\n✅ Step 5: Data organization complete! Output at {config.STEP5_LABELED_DIR}")

if __name__ == '__main__':
    organize_slices_by_label()
