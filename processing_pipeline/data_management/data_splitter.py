# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 1: DATASET SPLITTER
# =====================================================================
# This script performs the initial subject-level stratified split of the
# dataset into training, validation, and test sets.
# It ensures that only subjects with a complete temporal sequence of
# visits are included.

import os
import glob
import shutil
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Import configurations
from configs import config
from utils import utils # Assuming you save utils in a file named utils.py

def split_dataset():
    """
    Main function to perform the sequential dataset split.
    - Loads metadata.
    - Identifies subjects with complete temporal visits.
    - Performs a stratified 70/15/15 split on subjects.
    - Saves metadata manifests for each split.
    - Creates a new directory structure with simplified NIfTI filenames.
    """
    print("🧠 STEP 1: SEQUENTIAL ADNI DATA SPLITTER")
    print("=" * 60)
    print("📋 Requirements:")
    print(f"  • Subjects must have all visits: {config.REQUIRED_VISITS}")
    print(f"  • Split proportions: {config.SPLIT_RATIOS}")
    print(f"  • Output Directory: {config.STEP1_SPLIT_DIR}")
    print("-" * 60)

    # Ensure output directory exists
    config.STEP1_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare metadata
    df = pd.read_csv(config.METADATA_CSV)
    print(f"📊 Initial dataset: {len(df)} rows, {df['Subject'].nunique()} unique subjects.")

    # Prioritize 'Scaled_2' descriptions over 'Scaled' for the same visit
    def keep_scaled2(group: pd.DataFrame) -> pd.DataFrame:
        if group["Description"].str.contains(r"Scaled_2").any():
            return group[group["Description"].str.contains(r"Scaled_2")]
        return group

    df = df.groupby(["Subject", "Visit"], group_keys=False).apply(keep_scaled2).reset_index(drop=True)
    print(f"✓ After resolving duplicates: {len(df)} rows remain.")

    # 2. Identify subjects with complete visit sequences
    subject_visits = df.groupby('Subject')['Visit'].apply(set).to_dict()
    required_set = set(config.REQUIRED_VISITS)
    complete_subjects = [
        subj for subj, visits in subject_visits.items() if required_set.issubset(visits)
    ]
    
    retention_rate = (len(complete_subjects) / len(subject_visits)) * 100
    print(f"✅ Found {len(complete_subjects)} subjects with complete sequences. ({retention_rate:.1f}% retention)")

    # 3. Filter dataframe to complete subjects only
    df_complete = df[df['Subject'].isin(complete_subjects)].copy()
    
    # 4. Stratified subject-level splitting
    print("\n🎲 Performing stratified subject-level split...")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    rid_to_group = df_complete.groupby("Subject")["Group"].first()
    subject_split_map = {}

    for grp_label, rid_series in rid_to_group.groupby(rid_to_group):
        rid_list = list(rid_series.index)
        random.shuffle(rid_list)
        
        n_total = len(rid_list)
        n_train = int(n_total * config.SPLIT_RATIOS['train'])
        n_val = int(n_total * config.SPLIT_RATIOS['val'])
        
        train_rids = rid_list[:n_train]
        val_rids = rid_list[n_train : n_train + n_val]
        test_rids = rid_list[n_train + n_val:]

        for rid in train_rids: subject_split_map[rid] = "train"
        for rid in val_rids: subject_split_map[rid] = "val"
        for rid in test_rids: subject_split_map[rid] = "test"
        
        print(f"  - {grp_label}: {len(train_rids)} train, {len(val_rids)} val, {len(test_rids)} test")

    df_complete["Split"] = df_complete["Subject"].map(subject_split_map)

    # 5. Select one record per required visit for each subject
    selected_rows = []
    for subject in complete_subjects:
        subject_data = df_complete[df_complete['Subject'] == subject]
        for visit in config.REQUIRED_VISITS:
            visit_rows = subject_data[subject_data['Visit'] == visit]
            if not visit_rows.empty:
                selected_rows.append(visit_rows.iloc[0])

    df_final = pd.DataFrame(selected_rows)
    print(f"\n✓ Final dataset for processing: {len(df_final)} rows ({len(complete_subjects)} subjects × {len(config.REQUIRED_VISITS)} visits)")

    # 6. Save metadata manifests
    print("\n💾 Saving metadata manifests...")
    for split in config.SPLIT_RATIOS.keys():
        split_df = df_final[df_final.Split == split]
        split_df.to_csv(config.STEP1_SPLIT_DIR / f"{split}.csv", index=False)
        print(f"  ✓ Saved {split}.csv: {len(split_df)} rows")
    df_final.to_csv(config.STEP1_SPLIT_DIR / "metadata_split.csv", index=False)
    print(f"  ✓ Saved metadata_split.csv: {len(df_final)} rows")

    # 7. Mirror NIfTI files with simplified naming
    print("\n📁 Copying/linking NIfTI files with simplified names...")
    copy_stats = defaultdict(lambda: defaultdict(int))
    errors = []

    for _, row in df_final.iterrows():
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
            if dest.exists():
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

    print("\n📋 File Copy Summary:")
    for split in config.SPLIT_RATIOS.keys():
        stats = copy_stats[split]
        print(f"  - {split.capitalize()}: {stats['copied']} copied, {stats['skipped']} skipped, {stats['errors']} errors.")

    if errors:
        print("\n⚠️ Errors encountered:")
        for error in errors[:10]: print(f"  - {error}")

    print("\n✅ Step 1: Dataset splitting complete!")

if __name__ == '__main__':
    # This allows the script to be run directly
    import numpy as np # Needed for the random seed setting
    split_dataset()
