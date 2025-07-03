# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 5: DATA ORGANIZATION & LABELING
# =====================================================================
# This script organizes extracted 2D NIfTI slices into a structured
# directory based on slice type, split, and class.
# It STRICTLY enforces that only subjects with a complete temporal
# sequence of slices are included in the final organized dataset.

import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from configs import config
from utils import utils

def organize_slices_by_label(logger):
    """
    Organizes the optimal 2D NIfTI slices into a labeled directory structure,
    ensuring temporal integrity by only including subjects with a complete
    set of visits for each slice type.

    Structure: .../slice_type/split/class/subject_id/slice_files...

    Args:
        logger: A configured logger instance for output.
    """
    logger.info("\n🧠 STEP 5: ORGANIZING AND LABELING 2D SLICES (TEMPORAL - STRICT)")
    logger.info("=" * 80)

    # 1. Load metadata to map subjects to their group and split
    meta_path = config.STEP1_SPLIT_DIR / "metadata_split.csv"
    if not meta_path.exists():
        logger.error(f"❌ Metadata file not found at {meta_path}. Cannot proceed.")
        return

    df_meta = pd.read_csv(meta_path)
    subject_to_group = dict(zip(df_meta.Subject, df_meta.Group))
    subject_to_split = dict(zip(df_meta.Subject, df_meta.Split))
    logger.info(f"✅ Loaded metadata for {len(subject_to_group)} subjects.")

    # 2. Define input directories for each slice type
    slice_type_dirs = {
        "axial": config.STEP4_SLICES_AXIAL_DIR,
        "coronal": config.STEP4_SLICES_CORONAL_DIR,
        "sagittal": config.STEP4_SLICES_SAGITTAL_DIR,
    }

    # 3. Pre-create output directories
    config.STEP5_LABELED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Ensured output directory exists: {config.STEP5_LABELED_DIR}")

    # 4. Process each slice type sequentially
    overall_stats = defaultdict(int)
    
    for slice_type, slice_dir in slice_type_dirs.items():
        logger.info("\n" + "=" * 80)
        logger.info(f"PROCESSING {slice_type.upper()} SLICES")
        logger.info("=" * 80)
        
        if not slice_dir.exists():
            logger.warning(f"⚠️ Input directory for {slice_type} not found, skipping: {slice_dir}")
            continue

        # --- Gather and group all files by subject and visit ---
        subject_files = defaultdict(lambda: defaultdict(list))
        all_slice_files = list(slice_dir.rglob("*.nii.gz"))
        
        for f_path in all_slice_files:
            subj_id = utils.extract_subject_id_from_filename(f_path.name)
            visit = utils.extract_visit_from_filename(f_path.name)
            if subj_id and visit:
                subject_files[subj_id][visit].append(f_path)

        logger.info(f"🔍 Found {len(all_slice_files)} files across {len(subject_files)} unique subjects.")

        # --- Process subjects, enforcing temporal completeness ---
        slice_stats = defaultdict(int)
        for subject_id, visit_files in tqdm(subject_files.items(), desc=f"Organizing {slice_type}"):
            # STRICT check: ensure all required visits are present
            if set(visit_files.keys()) != set(config.REQUIRED_VISITS):
                slice_stats['incomplete_sequences'] += 1
                continue

            # Get metadata for the subject
            group = subject_to_group.get(subject_id)
            split = subject_to_split.get(subject_id)

            if not group or not split:
                slice_stats['unmatched_metadata'] += 1
                continue

            if group not in config.CLASSES_TO_INCLUDE:
                slice_stats['excluded_class'] += 1
                continue

            # Create destination directory for the subject
            dest_dir = config.STEP5_LABELED_DIR / slice_type / split / group / subject_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy all files for the complete temporal sequence
            copied_count = 0
            for visit, files in visit_files.items():
                for src_path in files:
                    dest_path = dest_dir / src_path.name
                    if not dest_path.exists():
                        shutil.copy2(src_path, dest_path)
                        copied_count += 1
            
            if copied_count > 0:
                slice_stats['successful_subjects'] += 1
                slice_stats['files_copied'] += copied_count

        logger.info(f"\n📊 {slice_type.upper()} Organization Summary:")
        logger.info(f"  - Subjects with complete sequences processed: {slice_stats['successful_subjects']}")
        logger.info(f"  - Subjects with incomplete sequences skipped: {slice_stats['incomplete_sequences']}")
        logger.info(f"  - Subjects with missing metadata skipped: {slice_stats['unmatched_metadata']}")
        logger.info(f"  - Total files organized for {slice_type}: {slice_stats['files_copied']}")
        overall_stats['total_files_organized'] += slice_stats['files_copied']

    logger.info("\n" + "=" * 80)
    logger.info("✅ OVERALL ORGANIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  - Grand total of files organized across all types: {overall_stats['total_files_organized']}")
    logger.info(f"  - Final output is ready for 2D conversion at: {config.STEP5_LABELED_DIR}")

if __name__ == '__main__':
    # This allows the script to be run directly for testing.
    # A basic logger is created if one is not provided.
    import logging
    from utils.logging_utils import setup_logging
    
    main_logger = setup_logging("data_organization_standalone", config.LOG_DIR)
    organize_slices_by_label(logger=main_logger)
