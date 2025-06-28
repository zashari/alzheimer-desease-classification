# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 1: DATASET SPLITTER
# =====================================================================
# This script performs the initial subject-level stratified split of the
# dataset into training, validation, and test sets.
# It ensures that only subjects with a complete temporal sequence of
# visits are included and provides a comprehensive analysis of the process.

import os
import glob
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Import configurations
from configs import config
from utils import utils

def split_dataset(logger):
    """
    Main function to perform the sequential dataset split.
    - Loads and cleans metadata.
    - Analyzes data availability to find subjects with complete temporal visits.
    - Performs a robust, stratified 70/15/15 split on subjects, handling small groups.
    - Creates a new directory structure with simplified NIfTI filenames.
    - Saves metadata manifests and a comprehensive summary report.

    Args:
        logger: A configured logger instance for output.
    """
    logger.info("🧠 STEP 1: SEQUENTIAL ADNI DATA SPLITTER")
    logger.info("=" * 80)
    logger.info("📋 Requirements:")
    logger.info(f"  • Subjects must have all visits: {config.REQUIRED_VISITS}")
    logger.info(f"  • Split proportions: Train={config.SPLIT_RATIOS['train']}, Val={config.SPLIT_RATIOS['val']}, Test={config.SPLIT_RATIOS['test']}")
    logger.info(f"  • Output Directory: {config.STEP1_SPLIT_DIR}")
    logger.info("-" * 80)

    # 0. Setup
    config.STEP1_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and Clean Metadata
    try:
        df = pd.read_csv(config.METADATA_CSV)
    except FileNotFoundError:
        logger.error(f"❌ CRITICAL: Metadata file not found at {config.METADATA_CSV}")
        return

    logger.info(f"📊 Initial dataset: {len(df)} total rows, {df['Subject'].nunique()} unique subjects.")

    # Prioritize 'Scaled_2' descriptions over 'Scaled' for the same visit
    def keep_scaled2(group: pd.DataFrame) -> pd.DataFrame:
        if group["Description"].str.contains(r"Scaled_2").any():
            return group[group["Description"].str.contains(r"Scaled_2")]
        return group
    df = df.groupby(["Subject", "Visit"], group_keys=False).apply(keep_scaled2).reset_index(drop=True)
    logger.info(f"✓ After resolving duplicates: {len(df)} rows remain.")

    # 2. Data Availability Analysis
    logger.info("\n" + "=" * 80)
    logger.info("🔍 DATA AVAILABILITY ANALYSIS")
    logger.info("=" * 80)
    subject_visits = df.groupby('Subject')['Visit'].apply(set).to_dict()
    subject_groups = df.groupby('Subject')['Group'].first().to_dict()
    required_set = set(config.REQUIRED_VISITS)

    complete_subjects = []
    incomplete_subjects = []
    complete_by_group = defaultdict(list)

    for subject, visits in subject_visits.items():
        if required_set.issubset(visits):
            complete_subjects.append(subject)
            group = subject_groups.get(subject, 'Unknown')
            complete_by_group[group].append(subject)
        else:
            incomplete_subjects.append(subject)

    retention_rate = (len(complete_subjects) / len(subject_visits)) * 100 if subject_visits else 0
    logger.info("📊 VISIT COMPLETENESS ANALYSIS:")
    logger.info(f"  • Total subjects analyzed: {len(subject_visits)}")
    logger.info(f"  • Complete sequences (all {config.REQUIRED_VISITS}): {len(complete_subjects)}")
    logger.info(f"  • Incomplete sequences: {len(incomplete_subjects)}")
    logger.info(f"  • Data retention rate: {retention_rate:.1f}%")

    logger.info("\n📋 COMPLETE SUBJECTS BY DIAGNOSTIC GROUP:")
    for group in sorted(complete_by_group.keys()):
        count = len(complete_by_group[group])
        logger.info(f"  • {group}: {count} subjects")

    # 3. Stratified Subject-Level Splitting
    logger.info("\n" + "=" * 80)
    logger.info("🎲 STRATIFIED SUBJECT-LEVEL SPLITTING")
    logger.info("=" * 80)
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    subject_split_map = {}
    split_summary = {}

    for grp_label, subjects in complete_by_group.items():
        random.shuffle(subjects)
        n_total = len(subjects)
        
        n_train = int(n_total * config.SPLIT_RATIOS['train'])
        n_val = int(n_total * config.SPLIT_RATIOS['val'])
        n_test = n_total - n_train - n_val

        # Robust handling for small groups to ensure each split gets at least 1 subject
        if n_total >= 3 and 0 in [n_train, n_val, n_test]:
             n_train = max(1, n_train)
             n_val = max(1, n_val)
             n_test = max(1, n_test)
             # Reduce the largest split to maintain total count
             while (n_train + n_val + n_test) > n_total:
                 if n_train > n_val and n_train > n_test: n_train -= 1
                 elif n_val > n_test: n_val -= 1
                 else: n_test -= 1

        train_rids = subjects[:n_train]
        val_rids = subjects[n_train:n_train + n_val]
        test_rids = subjects[n_train + n_val:]

        for rid in train_rids: subject_split_map[rid] = "train"
        for rid in val_rids: subject_split_map[rid] = "val"
        for rid in test_rids: subject_split_map[rid] = "test"
        
        logger.info(f"📊 Processing {grp_label} group ({n_total} subjects):")
        logger.info(f"  • Train: {len(train_rids)} subjects")
        logger.info(f"  • Val:   {len(val_rids)} subjects")
        logger.info(f"  • Test:  {len(test_rids)} subjects")
        if n_total < 7:
             logger.warning(f"  ⚠️ WARNING: Small group size for {grp_label} may result in uneven ratios.")

    # 4. Finalize DataFrame and Save Manifests
    df_complete = df[df['Subject'].isin(complete_subjects)].copy()
    df_complete["Split"] = df_complete["Subject"].map(subject_split_map)

    selected_rows = []
    for subject in complete_subjects:
        subject_data = df_complete[df_complete['Subject'] == subject]
        for visit in config.REQUIRED_VISITS:
            visit_rows = subject_data[subject_data['Visit'] == visit]
            if not visit_rows.empty:
                selected_rows.append(visit_rows.iloc[0])

    df_final = pd.DataFrame(selected_rows)
    logger.info(f"\n✓ Final dataset for processing: {len(df_final)} rows ({len(complete_subjects)} subjects × {len(config.REQUIRED_VISITS)} visits)")

    logger.info("\n💾 SAVING METADATA...")
    for split in config.SPLIT_RATIOS.keys():
        split_df = df_final[df_final.Split == split]
        split_df.to_csv(config.STEP1_SPLIT_DIR / f"{split}.csv", index=False)
        logger.info(f"  ✓ Saved {split}.csv: {len(split_df)} rows")
    df_final.to_csv(config.STEP1_SPLIT_DIR / "metadata_split.csv", index=False)
    logger.info(f"  ✓ Saved metadata_split.csv: {len(df_final)} rows")

    # 5. Mirror NIfTI Files
    logger.info("\n" + "=" * 80)
    logger.info(f"📁 COPYING/LINKING NIFTI FILES (use_symlinks={config.USE_SYMLINKS})")
    logger.info("=" * 80)
    copy_stats = defaultdict(lambda: defaultdict(int))
    errors = []

    for _, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Mirroring NIfTI files"):
        split, subject, visit, image_id = row.Split, row.Subject, row.Visit, row["Image Data ID"]
        
        pattern = str(config.RAW_NIFTI_DIR / subject / "**" / f"*{image_id}*.nii*")
        matches = glob.glob(pattern, recursive=True)
        
        if len(matches) != 1:
            errors.append(f"[{split}] {subject}_{visit}: Found {len(matches)} files for ImageID {image_id}")
            copy_stats[split]['errors'] += 1
            continue
            
        src = Path(matches[0])
        dest_subj_dir = config.STEP1_SPLIT_DIR / split / subject
        dest_subj_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_subj_dir / f"{subject}_{visit}.nii"

        try:
            if dest.exists() or dest.is_symlink():
                copy_stats[split]['skipped'] += 1
                continue
            if config.USE_SYMLINKS:
                os.symlink(src.resolve(), dest)
            else:
                shutil.copy2(src, dest)
            copy_stats[split]['copied'] += 1
        except OSError as e:
            errors.append(f"Could not link/copy {src} -> {dest}: {e}")
            copy_stats[split]['errors'] += 1
    
    # 6. Comprehensive Summary Report
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPREHENSIVE SUMMARY REPORT")
    logger.info("=" * 80)

    logger.info("\n🔍 FINAL SPLIT DISTRIBUTION (Subject counts):")
    split_class_counts = df_final.groupby(["Split", "Group"])['Subject'].nunique().unstack(fill_value=0)
    logger.info("\n" + str(split_class_counts))

    logger.info("\n📋 FINAL SPLIT DISTRIBUTION (Image counts):")
    split_visit_counts = df_final.groupby(["Split", "Group"]).size().unstack(fill_value=0)
    logger.info("\n" + str(split_visit_counts))

    logger.info("\n📁 FILE COPY STATISTICS:")
    for split in config.SPLIT_RATIOS.keys():
        stats = copy_stats[split]
        logger.info(f"  • {split.capitalize():<5}: {stats['copied']} new files, {stats['skipped']} already exist, {stats['errors']} errors.")

    if errors:
        logger.warning(f"\n⚠️ ERRORS ENCOUNTERED ({len(errors)} total):")
        for error in errors[:10]:
            logger.warning(f"  • {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors.")
            
    logger.info("\n" + "="*80)
    logger.info("✅ STEP 1: SEQUENTIAL SPLITTING COMPLETE!")
    logger.info(f"📂 Output directory: {config.STEP1_SPLIT_DIR}")
    logger.info("="*80)


if __name__ == '__main__':
    # This allows the script to be run directly for testing
    # Note: Requires a configured logger to be passed in a real pipeline
    from utils.logging_utils import setup_logging
    
    # Create a basic logger for standalone execution
    main_logger = setup_logging("data_splitter_standalone", config.LOG_DIR)
    
    split_dataset(logger=main_logger)