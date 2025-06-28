**Author: Zaky Ashari** (izzat.zaky@gmail.com)

---

# Overview


# References
* Hippocampal ROI Template: https://neurovault.org/images/1700/
* Image Enhancement: https://www.researchgate.net/publication/381917548_Brain_tumor_classification_in_VIT-B16_based_on_relative_position_encoding_and_residual_MLP


# Environment Setup


```python
# %pip install pandas tqdm Pillow numpy nibabel matplotlib ipython scikit-learn opencv-python scipy torch torchvision timm seaborn
# !pip install --upgrade pandas pyarrow scikit-learn
```

# Package Loader


```python
# =====================================================================
# PACKAGE IMPORTS - ADNI MRI PROCESSING PIPELINE
# =====================================================================

# Standard Library Imports
import datetime
import glob
import json
import multiprocessing
import os
import random
import re
from pathlib import Path
import shutil
import string
import subprocess
import threading
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Core Scientific Computing
import numpy as np
import pandas as pd

# Image Processing & Computer Vision
import cv2
from PIL import Image
from skimage import filters, measure, morphology, segmentation
from skimage.exposure import equalize_adapthist
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates

# Neuroimaging & Medical Data
import nibabel as nib

# Visualization & Display
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

# Machine Learning & Deep Learning
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

# Scikit-learn Components
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import label_binarize

# Progress Tracking
from tqdm import tqdm

# Augmentation
import os
import numpy as np
import cv2
from pathlib import Path
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RegularGridInterpolator
from skimage import exposure
import shutil
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration
warnings.filterwarnings('ignore')

# =====================================================================
# VERIFY CRITICAL DEPENDENCIES
# =====================================================================

print("📦 Package Import Summary:")
print(f"   • NumPy: {np.__version__}")
print(f"   • Pandas: {pd.__version__}")
print(f"   • OpenCV: {cv2.__version__}")
print(f"   • NiBabel: {nib.__version__}")
print(f"   • PyTorch: {torch.__version__}")
print(f"   • Matplotlib: {plt.matplotlib.__version__}")
print("✅ All packages loaded successfully!")
```

# Code

### Data Exploration

#### CSV Inspection


```python
dataframe = pd.read_csv("../datasets/ADNI_1_5_T/ADNI1_Complete_1Yr_1.5T_3_21_2025.csv")
```

#### Check Sample Data


```python
dataframe
```

#### Check Unique Values


```python
print("Unique Value of", dataframe['Visit'].value_counts())
print("\nUnique Value of", dataframe['Modality'].value_counts())
print("\nUnique Value of", dataframe['Description'].value_counts())
```

#### Visualize the Visits on Each Subjects


```python
# === USER CONFIGURATION ===
base_dir     = '../datasets/ADNI_1_5_T/.ADNI'  
metadata_csv = '../datasets/ADNI_1_5_T/ADNI1_Complete_1Yr_1.5T_3_21_2025.csv'

# === LOAD METADATA ===
meta = pd.read_csv(metadata_csv, parse_dates=['Acq Date'])
meta['DateStr'] = meta['Acq Date'].dt.strftime('%Y-%m-%d')

# === SELECT ONE RANDOM SUBJECT PER GROUP THAT HAS NIfTI FILES ===
groups = meta['Group'].unique()
selected_subjects = {}

for group in groups:
    # Unique subjects in this group
    subjects = meta.loc[meta['Group'] == group, 'Subject'].unique().tolist()
    random.shuffle(subjects)
    
    # Find the first subject with available NIfTI files
    for subj in subjects:
        pattern = os.path.join(base_dir, subj, '**', '*.nii')
        if glob.glob(pattern, recursive=True):
            selected_subjects[group] = subj
            break
    else:
        print(f"No NIfTI files found for any subject in group '{group}'")

# === VISUALIZE EACH SELECTED SUBJECT ===
for group_label, subject_id in selected_subjects.items():
    # Filter metadata for this subject
    sub_meta = meta.loc[meta['Subject'] == subject_id].copy()
    date_to_visit = dict(zip(sub_meta['DateStr'], sub_meta['Visit']))
    
    # Find NIfTI files
    pattern   = os.path.join(base_dir, subject_id, '**', '*.nii')
    nii_files = sorted(glob.glob(pattern, recursive=True))
    
    # Determine reference coronal slice index from 'sc' visit if available
    sc_dates = sub_meta.loc[sub_meta['Visit'] == 'sc', 'DateStr'].tolist()
    ref_file = None
    for f in nii_files:
        if any(date in f for date in sc_dates):
            ref_file = f
            break
    if not ref_file:
        ref_file = nii_files[0]
    ref_data  = nib.load(ref_file).get_fdata()
    ref_mid_y = ref_data.shape[1] // 2
    
    # Sort files by acquisition date
    def get_date(path):
        folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        return datetime.datetime.strptime(folder.split('_')[0], '%Y-%m-%d')
    sorted_files = sorted(nii_files, key=get_date)
    
    # Plotting
    n_visits = len(sorted_files)
    fig, axes = plt.subplots(1, n_visits, figsize=(5 * n_visits, 5))
    if n_visits == 1:
        axes = [axes]
    
    for ax, nii_path in zip(axes, sorted_files):
        data = nib.load(nii_path).get_fdata()
        y_idx = min(ref_mid_y, data.shape[1] - 1)
        coronal_slice = data[:, y_idx, :]
        
        session_folder = os.path.basename(os.path.dirname(os.path.dirname(nii_path)))
        session_date   = session_folder.split('_')[0]
        visit_code     = date_to_visit.get(session_date, session_date)
        
        ax.imshow(np.rot90(coronal_slice), cmap='gray')
        ax.set_title(f"{group_label} – {visit_code}", fontsize=12)
        ax.axis('off')
    
    plt.suptitle(f"{group_label} | Subject: {subject_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
```

#### Initial Splitting


```python
# =====================================================================
# SEQUENTIAL ADNI DATA SPLITTER - STRICT THREE-VISIT REQUIREMENT
# =====================================================================
# Modified to require all subjects have complete temporal sequences:
# sc (screening) → m06 (month 6) → m12 (month 12)
# Output: {subjectID}_{visit}.nii
# Split: 70% train / 15% val / 15% test

# ----------------- CONFIGURATION ---------------------------------------
RAW_DIR    = Path("../datasets/ADNI_1_5_T/.ADNI")       # original ADNI root
SPLIT_DIR  = Path("../datasets/ADNI_1_5_T/1_splitted_sequential")  # output folder
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

USE_SYMLINKS = True
RNG_SEED = 42

# Required visits for sequential analysis
REQUIRED_VISITS = ["sc", "m06", "m12"]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("🧠 SEQUENTIAL ADNI DATA SPLITTER")
print("=" * 60)
print("📋 Requirements:")
print(f"   • All subjects must have visits: {REQUIRED_VISITS}")
print(f"   • Output filename format: {{subjectID}}_{{visit}}.nii")
print(f"   • Split proportions: {TRAIN_RATIO*100:.0f}% train / {VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test")
print()

# -------------------------------------------------------------------
# (0) Load & basic clean-up
# -------------------------------------------------------------------
df = dataframe.copy()          # assumes `dataframe` already exists
df["RID"] = df["Subject"]      # alias for clarity

print(f"📊 Initial dataset: {len(df)} total rows")
print(f"📈 Unique subjects: {df['Subject'].nunique()}")
print(f"🏷️  Diagnostic groups: {df['Group'].value_counts().to_dict()}")
print(f"📅 Available visits: {sorted(df['Visit'].unique())}")

# -------------------------------------------------------------------
# (0.5) Keep *Scaled_2* over *Scaled* duplicates
# -------------------------------------------------------------------
def keep_scaled2(group: pd.DataFrame) -> pd.DataFrame:
    """Within (Subject, Visit) keep only *Scaled_2* rows if they exist."""
    if group["Description"].str.contains(r"Scaled_2").any():
        return group[group["Description"].str.contains(r"Scaled_2")]
    return group

df = (
    df.groupby(["Subject", "Visit"], group_keys=False)
      .apply(keep_scaled2)
      .reset_index(drop=True)
)

print(f"✓ After removing duplicates: {len(df)} rows")

# -------------------------------------------------------------------
# (1) DATA AVAILABILITY ANALYSIS - CHECK COMPLETE VISIT SEQUENCES
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("🔍 DATA AVAILABILITY ANALYSIS")
print("=" * 60)

# Group by subject and check visit availability
subject_visits = df.groupby('Subject')['Visit'].apply(set).to_dict()
subject_groups = df.groupby('Subject')['Group'].first().to_dict()

# Analyze visit patterns
visit_patterns = defaultdict(list)
complete_subjects = []
incomplete_subjects = []

for subject, visits in subject_visits.items():
    group = subject_groups[subject]
    required_set = set(REQUIRED_VISITS)
    
    if required_set.issubset(visits):
        complete_subjects.append(subject)
        visit_patterns['complete'].append((subject, group))
    else:
        incomplete_subjects.append(subject)
        missing = required_set - visits
        available = visits & required_set
        visit_patterns['incomplete'].append((subject, group, available, missing))

# Print analysis results
print(f"📊 VISIT COMPLETENESS ANALYSIS:")
print(f"   • Total subjects analyzed: {len(subject_visits)}")
print(f"   • Complete sequences (all {REQUIRED_VISITS}): {len(complete_subjects)}")
print(f"   • Incomplete sequences: {len(incomplete_subjects)}")
print(f"   • Data retention rate: {len(complete_subjects)/len(subject_visits)*100:.1f}%")

# Analyze by diagnostic group
complete_by_group = defaultdict(list)
incomplete_by_group = defaultdict(list)

for subject, group in visit_patterns['complete']:
    complete_by_group[group].append(subject)

for subject, group, available, missing in visit_patterns['incomplete']:
    incomplete_by_group[group].append((subject, available, missing))

print(f"\n📋 COMPLETE SUBJECTS BY DIAGNOSTIC GROUP:")
total_complete = 0
for group in sorted(complete_by_group.keys()):
    count = len(complete_by_group[group])
    total_complete += count
    print(f"   • {group}: {count} subjects")
print(f"   • Total: {total_complete} subjects")

print(f"\n⚠️  INCOMPLETE SUBJECTS BY DIAGNOSTIC GROUP:")
for group in sorted(incomplete_by_group.keys()):
    count = len(incomplete_by_group[group])
    print(f"   • {group}: {count} subjects")
    
    # Show sample of missing visits
    if count > 0:
        sample_size = min(3, count)
        for i, (subj, available, missing) in enumerate(incomplete_by_group[group][:sample_size]):
            print(f"     - {subj}: has {sorted(available)}, missing {sorted(missing)}")
        if count > sample_size:
            print(f"     ... and {count - sample_size} more")

# Check if we have enough subjects for splitting
print(f"\n🎯 SPLIT FEASIBILITY CHECK:")
min_subjects_needed_per_group = 7  # Need at least 7 subjects to get 1 in each split with 70/15/15
feasible_groups = []

for group in complete_by_group.keys():
    count = len(complete_by_group[group])
    needed = min_subjects_needed_per_group
    feasible = count >= needed
    feasible_groups.append((group, count, feasible))
    status = "✅ FEASIBLE" if feasible else "❌ INSUFFICIENT"
    print(f"   • {group}: {count} subjects (need ≥{needed}) - {status}")

all_groups_feasible = all(feasible for _, _, feasible in feasible_groups)

if not all_groups_feasible:
    print(f"\n❌ CRITICAL: Some groups have insufficient subjects for the required split!")
    print(f"   Consider using a different split strategy for groups with few subjects.")
    # We'll continue anyway to show what's possible
else:
    print(f"\n✅ SUCCESS: All groups have sufficient subjects for splitting!")

# -------------------------------------------------------------------
# (2) FILTER TO COMPLETE SUBJECTS ONLY
# -------------------------------------------------------------------
print(f"\n🔍 FILTERING TO COMPLETE SUBJECTS ONLY...")

# Filter dataframe to only include subjects with complete visit sequences
df_complete = df[df['Subject'].isin(complete_subjects)].copy()

print(f"✓ Filtered dataset: {len(df_complete)} rows from {len(complete_subjects)} subjects")

# Verify each subject has exactly the required visits
verification_failed = []
for subject in complete_subjects:
    subject_data = df_complete[df_complete['Subject'] == subject]
    subject_visits = set(subject_data['Visit'].unique())
    if subject_visits != set(REQUIRED_VISITS):
        verification_failed.append((subject, subject_visits))

if verification_failed:
    print(f"⚠️  WARNING: {len(verification_failed)} subjects failed verification!")
    for subj, visits in verification_failed[:3]:
        print(f"   - {subj}: {visits}")
else:
    print(f"✅ Verification passed: All subjects have exactly {REQUIRED_VISITS}")

# -------------------------------------------------------------------
# (3) STRATIFIED THREE-WAY SUBJECT-LEVEL ASSIGNMENT (70/15/15)
# -------------------------------------------------------------------
print(f"\n" + "=" * 60)
print("🎲 STRATIFIED SUBJECT-LEVEL SPLITTING (70/15/15)")
print("=" * 60)

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# Create subject to group mapping from complete subjects only
rid_to_group = df_complete.groupby("Subject")["Group"].first()
subject_split = {}

split_summary = {}

for grp_label, rid_series in rid_to_group.groupby(rid_to_group):
    rid_list = list(rid_series.index)
    random.shuffle(rid_list)
    
    print(f"\n📊 Processing {grp_label} group ({len(rid_list)} subjects):")
    
    n_total = len(rid_list)
    
    # Calculate split sizes using the specified ratios
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val  # Remaining subjects go to test
    
    # Ensure each split gets at least 1 subject if possible
    if n_total >= 3:
        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1
        if n_test == 0:
            n_test = 1
        
        # Redistribute if we over-allocated
        total_allocated = n_train + n_val + n_test
        if total_allocated > n_total:
            # Reduce largest allocation
            if n_train >= max(n_val, n_test):
                n_train = n_total - n_val - n_test
            elif n_val >= n_test:
                n_val = n_total - n_train - n_test
            else:
                n_test = n_total - n_train - n_val
    
    # Split subjects
    train_rids = rid_list[:n_train]
    val_rids = rid_list[n_train:n_train + n_val]
    test_rids = rid_list[n_train + n_val:]
    
    # Assign splits
    for rid in train_rids:
        subject_split[rid] = "train"
    for rid in val_rids:
        subject_split[rid] = "val"
    for rid in test_rids:
        subject_split[rid] = "test"
    
    # Store summary
    split_summary[grp_label] = {
        'total': len(rid_list),
        'train': len(train_rids),
        'val': len(val_rids),
        'test': len(test_rids)
    }
    
    # Calculate actual percentages
    train_pct = len(train_rids) / len(rid_list) * 100 if len(rid_list) > 0 else 0
    val_pct = len(val_rids) / len(rid_list) * 100 if len(rid_list) > 0 else 0
    test_pct = len(test_rids) / len(rid_list) * 100 if len(rid_list) > 0 else 0
    
    print(f"   • Total: {len(rid_list)} subjects")
    print(f"   • Train: {len(train_rids)} subjects ({train_pct:.1f}%)")
    print(f"   • Val: {len(val_rids)} subjects ({val_pct:.1f}%)")
    print(f"   • Test: {len(test_rids)} subjects ({test_pct:.1f}%)")
    
    if len(rid_list) < 7:
        print(f"   ⚠️  WARNING: Small group size may result in uneven split ratios")

# Add split assignment to dataframe
df_complete["Split"] = df_complete["Subject"].map(subject_split)

# -------------------------------------------------------------------
# (4) SELECT ALL REQUIRED VISITS FOR EACH SUBJECT
# -------------------------------------------------------------------
print(f"\n📅 SELECTING COMPLETE VISIT SEQUENCES...")

# For sequential analysis, we keep ALL required visits for each subject
selected_rows = []

for subject in complete_subjects:
    subject_data = df_complete[df_complete['Subject'] == subject]
    
    # Get one row per required visit
    for visit in REQUIRED_VISITS:
        visit_rows = subject_data[subject_data['Visit'] == visit]
        if not visit_rows.empty:
            selected_rows.append(visit_rows.iloc[0])
        else:
            print(f"⚠️  WARNING: Subject {subject} missing {visit} (unexpected!)")

df_final = pd.DataFrame(selected_rows)

print(f"✓ Final dataset: {len(df_final)} rows")
print(f"✓ Expected: {len(complete_subjects) * len(REQUIRED_VISITS)} rows")

# Verify we have correct number of visits per subject
visits_per_subject = df_final.groupby('Subject')['Visit'].count()
incorrect_counts = visits_per_subject[visits_per_subject != len(REQUIRED_VISITS)]

if len(incorrect_counts) > 0:
    print(f"⚠️  WARNING: {len(incorrect_counts)} subjects don't have exactly {len(REQUIRED_VISITS)} visits!")
else:
    print(f"✅ Verification: All subjects have exactly {len(REQUIRED_VISITS)} visits")

# -------------------------------------------------------------------
# (5) Save CSV manifests
# -------------------------------------------------------------------
print(f"\n💾 SAVING METADATA...")

for split in ("train", "val", "test"):
    split_df = df_final[df_final.Split == split]
    split_df.to_csv(SPLIT_DIR / f"{split}.csv", index=False)
    print(f"✓ Saved {split}.csv: {len(split_df)} rows")

df_final.to_csv(SPLIT_DIR / "metadata_split.csv", index=False)
print(f"✓ Saved metadata_split.csv: {len(df_final)} rows")

# -------------------------------------------------------------------
# (6) Mirror NIfTI files with simplified naming
# -------------------------------------------------------------------
print(f"\n📁 COPYING NIFTI FILES WITH SIMPLIFIED NAMING...")
print(f"   Format: {{subjectID}}_{{visit}}.nii")

copy_stats = defaultdict(lambda: defaultdict(int))
errors = []

for split in ("train", "val", "test"):
    subset_dir = SPLIT_DIR / split
    subset_dir.mkdir(exist_ok=True)
    
    split_data = df_final[df_final.Split == split]
    
    for _, row in split_data.iterrows():
        subject = row.Subject
        visit = row.Visit
        image_id = row["Image Data ID"]  # e.g. I45108
        
        # Locate the source NIfTI file
        pattern = str((RAW_DIR / subject).joinpath("**", f"*{image_id}*.nii*"))
        matches = glob.glob(pattern, recursive=True)
        
        if len(matches) != 1:
            error_msg = f"[{split}] {subject}_{visit}: expected 1 match, found {len(matches)} for {image_id}"
            errors.append(error_msg)
            copy_stats[split]['errors'] += 1
            continue
        
        src = Path(matches[0])
        
        # Create subject directory
        dest_subj_dir = subset_dir / subject
        dest_subj_dir.mkdir(exist_ok=True)
        
        # Simplified destination filename: {subjectID}_{visit}.nii
        dest_filename = f"{subject}_{visit}.nii"
        dest = dest_subj_dir / dest_filename
        
        try:
            if dest.exists():
                copy_stats[split]['skipped'] += 1
                continue
                
            if USE_SYMLINKS:
                os.symlink(src.resolve(), dest)
            else:
                shutil.copy2(src, dest)
            
            copy_stats[split]['copied'] += 1
            
        except OSError as e:
            error_msg = f"[{split}] Could not link/copy {src} → {dest}: {e}"
            errors.append(error_msg)
            copy_stats[split]['errors'] += 1

# -------------------------------------------------------------------
# (7) COMPREHENSIVE SUMMARY REPORT
# -------------------------------------------------------------------
print(f"\n" + "=" * 80)
print("📊 COMPREHENSIVE SUMMARY REPORT")
print("=" * 80)

print(f"\n🔍 DATA RETENTION ANALYSIS:")
original_subjects = len(subject_visits)
retained_subjects = len(complete_subjects)
retention_rate = (retained_subjects / original_subjects * 100)

print(f"   • Original subjects: {original_subjects}")
print(f"   • Retained subjects: {retained_subjects}")
print(f"   • Retention rate: {retention_rate:.1f}%")
print(f"   • Excluded subjects: {original_subjects - retained_subjects}")

print(f"\n📋 FINAL SPLIT DISTRIBUTION:")
print("Subject counts per split & class:")
split_class_counts = df_final.groupby(["Split", "Group"]).size().unstack(fill_value=0)
print(split_class_counts)

print(f"\nVisit distribution per split:")
split_visit_counts = df_final.groupby(["Split", "Visit"]).size().unstack(fill_value=0)
print(split_visit_counts)

print(f"\nTotal samples per split:")
total_subjects_all_splits = 0
for split in ["train", "val", "test"]:
    count = len(df_final[df_final.Split == split])
    subjects = len(df_final[df_final.Split == split]['Subject'].unique())
    total_subjects_all_splits += subjects
    percentage = (count / len(df_final)) * 100 if len(df_final) > 0 else 0
    print(f"  {split}: {count} samples from {subjects} subjects ({count//subjects} visits/subject) - {percentage:.1f}%")

print(f"\nActual split ratios:")
if total_subjects_all_splits > 0:
    for split in ["train", "val", "test"]:
        subjects = len(df_final[df_final.Split == split]['Subject'].unique())
        percentage = (subjects / total_subjects_all_splits) * 100
        print(f"  {split}: {percentage:.1f}% of subjects")

print(f"\n📁 FILE COPY STATISTICS:")
for split in ["train", "val", "test"]:
    stats = copy_stats[split]
    print(f"   {split}:")
    print(f"     • Copied: {stats['copied']}")
    print(f"     • Skipped: {stats['skipped']}")
    print(f"     • Errors: {stats['errors']}")

if errors:
    print(f"\n⚠️  ERRORS ENCOUNTERED ({len(errors)} total):")
    for error in errors[:5]:  # Show first 5 errors
        print(f"   • {error}")
    if len(errors) > 5:
        print(f"   ... and {len(errors) - 5} more errors")

print(f"\n✅ SEQUENTIAL SPLITTING COMPLETE!")
print(f"📂 Output directory: {SPLIT_DIR}")
print(f"🎯 Ready for CNN+LSTM temporal modeling!")

print(f"\n🔄 NEXT STEPS:")
print(f"   1. Verify file structure: {SPLIT_DIR}/{{split}}/{{subject}}/{{subject}}_{{visit}}.nii")
print(f"   2. Each subject now has complete temporal sequence: sc → m06 → m12")
print(f"   3. Proceed with skull stripping and ROI extraction on this sequential dataset")
print("=" * 80)
```

## NIFTI PROCESSING

#### [QC] Visualize Sample Data


```python
# -----------------------------------------------------------------
# SEQUENTIAL DATASET QC VISUALIZATION
# Modified for temporal sequence visualization with simplified naming
# -----------------------------------------------------------------

INPUT_ROOT = Path("../datasets/ADNI_1_5_T/1_splitted_sequential") 
SUBSETS = ["train", "val", "test"]
REQUIRED_VISITS = ["sc", "m06", "m12"]  # Expected temporal sequence
RNG_SEED = 2025
random.seed(RNG_SEED)

plt.rcParams["figure.facecolor"] = "white"

print("🧠 SEQUENTIAL DATASET QC VISUALIZATION")
print("=" * 60)
print(f"📂 Input Root: {INPUT_ROOT}")
print(f"📅 Expected visits per subject: {REQUIRED_VISITS}")
print()

for subset in SUBSETS:
    subset_dir = INPUT_ROOT / subset
    if not subset_dir.exists():
        display(Markdown(f"⚠️ Subset `{subset}` not found at `{subset_dir}`"))
        continue

    subjects = [d.name for d in subset_dir.iterdir() if d.is_dir()]
    if not subjects:
        display(Markdown(f"⚠️ No subjects in subset `{subset}`"))
        continue

    # Select a random subject for visualization
    subj = random.choice(subjects)
    subj_dir = subset_dir / subj
    
    # Look for files with simplified naming pattern: {subjectID}_{visit}.nii
    nii_files = []
    visit_files = {}
    
    for visit in REQUIRED_VISITS:
        expected_filename = f"{subj}_{visit}.nii"
        expected_path = subj_dir / expected_filename
        
        if expected_path.exists():
            nii_files.append(expected_path)
            visit_files[visit] = expected_path
        else:
            # Fallback: look for any .nii files containing the visit code
            pattern_files = list(subj_dir.glob(f"*{visit}*.nii*"))
            if pattern_files:
                nii_files.extend(pattern_files)
                visit_files[visit] = pattern_files[0]
    
    # Also collect any other .nii files as backup
    all_nii = sorted([p for p in subj_dir.rglob("*") if p.suffix in (".nii", ".gz")])
    if not nii_files and all_nii:
        nii_files = all_nii
    
    if not nii_files:
        display(Markdown(f"⚠️ No NIfTI files found for subject `{subj}` in `{subset}`"))
        continue

    # Display subject information
    display(Markdown(f"<br>**Subset: `{subset.upper()}` &nbsp;&nbsp;|&nbsp;&nbsp; Subject: _{subj}_**"))
    
    # Show available visits for this subject
    available_visits = list(visit_files.keys())
    if available_visits:
        display(Markdown(f"**Available visits:** {available_visits}"))
        if set(available_visits) == set(REQUIRED_VISITS):
            display(Markdown("✅ **Complete temporal sequence found!**"))
        else:
            missing = set(REQUIRED_VISITS) - set(available_visits)
            if missing:
                display(Markdown(f"⚠️ **Missing visits:** {list(missing)}"))
    
    # If we have temporal sequence, visualize all visits
    if len(visit_files) >= 2:
        # Visualize temporal sequence
        n_visits = len(visit_files)
        fig, axes = plt.subplots(3, n_visits, figsize=(5 * n_visits, 12))
        if n_visits == 1:
            axes = axes.reshape(-1, 1)  # Ensure 2D array
        
        fig.suptitle(f'Temporal Sequence Visualization\nSubject: {subj} | Subset: {subset.upper()}', 
                     fontsize=14, fontweight='bold')
        
        # Define the plane views
        plane_info = [
            ("Axial (transverse)", "Z"),
            ("Coronal (frontal)", "Y"), 
            ("Sagittal (lateral)", "X")
        ]
        
        # Process each visit in temporal order
        for visit_idx, visit in enumerate(REQUIRED_VISITS):
            if visit not in visit_files:
                # Fill with placeholder if visit missing
                for plane_idx in range(3):
                    ax = axes[plane_idx, visit_idx]
                    ax.text(0.5, 0.5, f'{visit.upper()}\nMISSING', 
                           ha='center', va='center', fontsize=12, 
                           transform=ax.transAxes, alpha=0.5)
                    ax.set_title(f"{visit.upper()}\n{plane_info[plane_idx][0]}")
                    ax.axis('off')
                continue
            
            # Load the NIfTI file for this visit
            nii_path = visit_files[visit]
            
            try:
                # Load and orient to canonical (RAS) space
                img = nib.load(str(nii_path))
                img_c = nib.as_closest_canonical(img)
                data = img_c.get_fdata()
                affine = img_c.affine

                # Compute centre voxel indices
                xc, yc, zc = (np.array(data.shape) // 2).astype(int)
                x_mm, y_mm, z_mm, _ = affine.dot([xc, yc, zc, 1])

                planes = [
                    ("Axial", "Z", zc, z_mm, data[:, :, zc]),
                    ("Coronal", "Y", yc, y_mm, data[:, yc, :]),
                    ("Sagittal", "X", xc, x_mm, data[xc, :, :])
                ]

                # Plot each plane for this visit
                for plane_idx, (plane_name, axis, idx, coord_mm, slice_data) in enumerate(planes):
                    ax = axes[plane_idx, visit_idx]
                    
                    # Display the slice
                    im = ax.imshow(np.flipud(slice_data.T), cmap="gray", interpolation="none")
                    
                    # Set title with visit and plane info
                    ax.set_title(f"{visit.upper()}\n{plane_name} ({axis}={idx})")
                    ax.axis('off')
                    
                    # Add colorbar for the first visit of each plane
                    if visit_idx == 0:
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            except Exception as e:
                # Handle any loading errors
                for plane_idx in range(3):
                    ax = axes[plane_idx, visit_idx]
                    ax.text(0.5, 0.5, f'{visit.upper()}\nERROR\n{str(e)[:20]}...', 
                           ha='center', va='center', fontsize=10, 
                           transform=ax.transAxes, alpha=0.5, color='red')
                    ax.set_title(f"{visit.upper()}\nError Loading")
                    ax.axis('off')

        plt.tight_layout()
        plt.show()
        
        # Display detailed file information
        display(Markdown("**📁 File Details:**"))
        for visit in REQUIRED_VISITS:
            if visit in visit_files:
                file_path = visit_files[visit]
                file_size = file_path.stat().st_size / (1024*1024)  # MB
                display(Markdown(f"- **{visit.upper()}**: `{file_path.name}` ({file_size:.1f} MB)"))
            else:
                display(Markdown(f"- **{visit.upper()}**: Missing"))
        
    else:
        # Fallback: single file visualization (original behavior)
        nii_path = random.choice(nii_files)
        
        display(Markdown(f"**Single file visualization:** `{nii_path.name}`"))
        
        # Load and orient to canonical (RAS) space
        img = nib.load(str(nii_path))
        img_c = nib.as_closest_canonical(img)
        data = img_c.get_fdata()
        affine = img_c.affine

        # Compute centre voxel indices
        xc, yc, zc = (np.array(data.shape) // 2).astype(int)
        x_mm, y_mm, z_mm, _ = affine.dot([xc, yc, zc, 1])

        planes = [
            ("Axial (transverse)",  "Z", zc, z_mm, data[:, :, zc]),
            ("Coronal (frontal)",   "Y", yc, y_mm, data[:, yc, :]),
            ("Sagittal (lateral)",  "X", xc, x_mm, data[xc, :, :])
        ]

        # Display slice metadata
        for title, axis, idx, coord_mm, _ in planes:
            display(Markdown(f"**{title}** | Axis: {axis} | Slice index: {idx} | {axis}-coord (mm): {coord_mm:.1f}"))

        # Plot three orthogonal views
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (title, axis, idx, coord_mm, slc) in zip(axes, planes):
            ax.imshow(np.flipud(slc.T), cmap="gray", interpolation="none")
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

print("\n✅ Sequential dataset QC visualization complete!")
```

#### Skull Stripping


```python
# — make sure FSL is on PATH in this kernel —
os.environ["FSLDIR"] = "/Users/AndiZakyAshari/fsl"
os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:" + os.environ["PATH"]

# verify `bet`
which = subprocess.run(["which", "bet"], capture_output=True, text=True)
if which.returncode != 0:
    raise RuntimeError("Cannot find BET on your PATH; adjust FSLDIR/PATH above.")
print("→ using BET at", which.stdout.strip())
```


```python
# =====================================================================
# SEQUENTIAL DATASET SKULL STRIPPING
# Modified for temporal sequence data with simplified naming
# =====================================================================

# ------------------- CONFIG -------------------------------------------------
BET_ARGS   = ["-f", "0.5", "-g", "0", "-B"]

# Updated paths for sequential dataset
INPUT_ROOT  = Path("../datasets/ADNI_1_5_T/1_splitted_sequential")  # train/val/test/<RID>/<visit_files>
OUTPUT_ROOT = Path("../datasets/ADNI_1_5_T/2_skull_stripping_sequential")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_THREADS = 12                  # adjust to taste
TIMEOUT_SEC = 600                 # 10-min timeout per BET call
REQUIRED_VISITS = ["sc", "m06", "m12"]  # Expected visits

print("🧠 SEQUENTIAL DATASET SKULL STRIPPING")
print("=" * 60)
print(f"📂 Input: {INPUT_ROOT}")
print(f"📁 Output: {OUTPUT_ROOT}")
print(f"📅 Expected visits: {REQUIRED_VISITS}")
print(f"⚙️  BET args: {BET_ARGS}")
print(f"🔧 Max threads: {MAX_THREADS}")
print()

# ---------------------------------------------------------------------------

dir_lock = threading.Lock()       # thread-safe mkdir

def process_brain_extraction(task):
    """
    BET one file; task = (split, subj, visit, nii_path)
    Output naming: {subj}_{visit}_brain.nii.gz
    """
    split, subj, visit, nii_path = task
    subj_out = OUTPUT_ROOT / split / subj

    # thread-safe mkdir
    with dir_lock:
        subj_out.mkdir(parents=True, exist_ok=True)

    # Create output filename: {subj}_{visit}_brain.nii.gz
    # This maintains temporal information and matches simplified input naming
    out_brain = subj_out / f"{subj}_{visit}_brain.nii.gz"

    if out_brain.exists():
        return "skip"

    cmd = ["bet", str(nii_path), str(out_brain)] + BET_ARGS
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       timeout=TIMEOUT_SEC, text=True)
        return "success"
    except subprocess.TimeoutExpired:
        return f"timeout ({TIMEOUT_SEC}s)"
    except subprocess.CalledProcessError as e:
        return f"err rc={e.returncode}"
    except Exception as e:
        return f"err {e}"

# --------------- BUILD TASK LIST -------------------------------------------
tasks = []
sequential_stats = {"total_subjects": 0, "complete_sequences": 0, "incomplete_sequences": 0}
missing_files = []

print("🔍 Scanning for sequential data...")

for split in ("train", "val", "test"):
    split_dir = INPUT_ROOT / split
    if not split_dir.exists():
        print(f"⚠️  Split directory not found: {split_dir}")
        continue
    
    print(f"\n📊 Processing {split.upper()} split:")
    split_subjects = 0
    split_complete = 0
    split_files = 0
    
    for subj_dir in split_dir.iterdir():
        if not subj_dir.is_dir():
            continue
        
        subj = subj_dir.name
        split_subjects += 1
        sequential_stats["total_subjects"] += 1
        
        # Check for simplified naming pattern: {subj}_{visit}.nii
        found_visits = []
        subject_files = []
        
        for visit in REQUIRED_VISITS:
            expected_file = subj_dir / f"{subj}_{visit}.nii"
            
            if expected_file.exists():
                found_visits.append(visit)
                subject_files.append((split, subj, visit, str(expected_file)))
                split_files += 1
            else:
                # Try to find any file with visit pattern (fallback)
                pattern_files = list(subj_dir.glob(f"*{visit}*.nii*"))
                if pattern_files:
                    # Use the first match
                    found_visits.append(visit)
                    subject_files.append((split, subj, visit, str(pattern_files[0])))
                    split_files += 1
                else:
                    missing_files.append(f"{split}/{subj}/{visit}")
        
        # Add all found files for this subject
        tasks.extend(subject_files)
        
        # Check if subject has complete sequence
        if set(found_visits) == set(REQUIRED_VISITS):
            split_complete += 1
            sequential_stats["complete_sequences"] += 1
        else:
            sequential_stats["incomplete_sequences"] += 1
            missing = set(REQUIRED_VISITS) - set(found_visits)
            print(f"   ⚠️  {subj}: missing {sorted(missing)}")
    
    print(f"   • Subjects: {split_subjects}")
    print(f"   • Complete sequences: {split_complete}")
    print(f"   • Files to process: {split_files}")

print(f"\n📈 OVERALL STATISTICS:")
print(f"   • Total subjects: {sequential_stats['total_subjects']}")
print(f"   • Complete sequences: {sequential_stats['complete_sequences']}")
print(f"   • Incomplete sequences: {sequential_stats['incomplete_sequences']}")
print(f"   • Total files to process: {len(tasks)}")

if missing_files:
    print(f"\n⚠️  MISSING FILES ({len(missing_files)} total):")
    for missing in missing_files[:10]:  # Show first 10
        print(f"   • {missing}")
    if len(missing_files) > 10:
        print(f"   ... and {len(missing_files) - 10} more")

if len(tasks) == 0:
    print("❌ No files found to process! Check input directory structure.")
    exit()

max_workers = min(MAX_THREADS, len(tasks))
print(f"\n🚀 Using {max_workers} parallel threads")

# --------------- PARALLEL EXECUTION ----------------------------------------
print(f"\n⚡ Starting skull stripping...")
stats = {"success": 0, "skip": 0, "timeout": 0, "err": 0}

with ThreadPoolExecutor(max_workers=max_workers) as exe:
    futures = {exe.submit(process_brain_extraction, t): t for t in tasks}
    for fut in tqdm(as_completed(futures), total=len(tasks), desc="Skull-stripping"):
        res = fut.result()
        if res == "success":
            stats["success"] += 1
        elif res == "skip":
            stats["skip"] += 1
        elif res.startswith("timeout"):
            stats["timeout"] += 1
        else:                      # any err/exception code
            stats["err"] += 1

# --------------- DETAILED SUMMARY -------------------------------------------
print(f"\n" + "=" * 60)
print("📊 SKULL STRIPPING SUMMARY")
print("=" * 60)

print(f"\n⚡ Processing Results:")
for k, v in stats.items():
    percentage = (v / len(tasks) * 100) if len(tasks) > 0 else 0
    print(f"  {k.capitalize():8s}: {v:4d} ({percentage:5.1f}%)")

print(f"\n🧠 Sequential Data Summary:")
print(f"  Total subjects processed: {sequential_stats['total_subjects']}")
print(f"  Complete sequences: {sequential_stats['complete_sequences']}")
print(f"  Incomplete sequences: {sequential_stats['incomplete_sequences']}")

# Check output structure
print(f"\n📁 Output Structure Verification:")
for split in ("train", "val", "test"):
    split_out = OUTPUT_ROOT / split
    if split_out.exists():
        subjects = [d for d in split_out.iterdir() if d.is_dir()]
        total_brain_files = 0
        complete_subjects = 0
        
        for subj_dir in subjects:
            brain_files = list(subj_dir.glob("*_brain.nii.gz"))
            total_brain_files += len(brain_files)
            
            # Check if subject has all three brain files
            subj = subj_dir.name
            expected_brains = [f"{subj}_{visit}_brain.nii.gz" for visit in REQUIRED_VISITS]
            if all((subj_dir / brain).exists() for brain in expected_brains):
                complete_subjects += 1
        
        print(f"  {split:5s}: {len(subjects):3d} subjects, {total_brain_files:3d} brain files, {complete_subjects:3d} complete")

print(f"\n✅ Skull stripping complete!")
print(f"📂 Output directory: {OUTPUT_ROOT}")

# Show sample output files
print(f"\n📄 Sample Output Files:")
sample_files = list(OUTPUT_ROOT.rglob("*_brain.nii.gz"))[:6]
for i, sample_file in enumerate(sample_files):
    rel_path = sample_file.relative_to(OUTPUT_ROOT)
    file_size = sample_file.stat().st_size / (1024*1024)  # MB
    print(f"  {i+1}. {rel_path} ({file_size:.1f} MB)")

print(f"\n🎯 Ready for next step: ROI Registration & Warping!")
print("=" * 60)
```

#### ROI Registration & Warping


```python
# =====================================================================
# SEQUENTIAL DATASET ROI REGISTRATION & WARPING
# Modified for temporal sequence data with simplified naming
# =====================================================================

# ----------------------- CONFIGURATION ----------------------------------
ROI_TEMPLATE   = "../hippocampal-ROI/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
MNI_BRAIN      = "/Users/AndiZakyAshari/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"

# Updated paths for sequential dataset
SKULL_ROOT     = Path("../datasets/ADNI_1_5_T/2_skull_stripping_sequential")
ROI_SUBJ_ROOT  = Path("../datasets/ADNI_1_5_T/3_roi_subj_space_sequential")
ROI_SUBJ_ROOT.mkdir(parents=True, exist_ok=True)

MAX_THREADS = 10   # Number of parallel threads
REQUIRED_VISITS = ["sc", "m06", "m12"]  # Expected visits

print("🧠 SEQUENTIAL DATASET ROI REGISTRATION & WARPING")
print("=" * 60)
print(f"📂 Input: {SKULL_ROOT}")
print(f"📁 Output: {ROI_SUBJ_ROOT}")
print(f"🎯 ROI Template: {ROI_TEMPLATE}")
print(f"🧭 MNI Reference: {MNI_BRAIN}")
print(f"📅 Expected visits: {REQUIRED_VISITS}")
print(f"🔧 Max threads: {MAX_THREADS}")
print()

# ----------------------- BUILD TASK LIST ---------------------------------
tasks = []
sequential_stats = {"total_subjects": 0, "complete_sequences": 0, "incomplete_sequences": 0, "excluded_subjects": 0}
missing_files = []
excluded_subjects = []

print("🔍 Scanning for skull-stripped sequential data...")

for split in ("train", "val", "test"):
    split_dir = SKULL_ROOT / split
    if not split_dir.exists():
        print(f"⚠️  Split directory not found: {split_dir}")
        continue

    print(f"\n📊 Processing {split.upper()} split:")
    split_subjects = 0
    split_complete = 0
    split_excluded = 0
    split_files = 0

    for subj_dir in split_dir.iterdir():
        if not subj_dir.is_dir():
            continue

        subj = subj_dir.name
        split_subjects += 1
        sequential_stats["total_subjects"] += 1
        
        # Check for skull-stripped files with temporal naming: {subj}_{visit}_brain.nii.gz
        found_visits = []
        subject_files = []
        
        for visit in REQUIRED_VISITS:
            expected_brain = subj_dir / f"{subj}_{visit}_brain.nii.gz"
            
            if expected_brain.exists():
                found_visits.append(visit)
                subject_files.append((split, subj, visit, expected_brain))
            else:
                # Try to find any brain file with visit pattern (fallback)
                pattern_files = list(subj_dir.glob(f"*{visit}*_brain.nii*"))
                if pattern_files:
                    found_visits.append(visit)
                    subject_files.append((split, subj, visit, pattern_files[0]))
                else:
                    missing_files.append(f"{split}/{subj}/{visit}_brain")
        
        # STRICT REQUIREMENT: Only process subjects with complete sequences
        if set(found_visits) == set(REQUIRED_VISITS):
            # Subject has complete sequence - add to processing tasks
            tasks.extend(subject_files)
            split_complete += 1
            split_files += len(subject_files)
            sequential_stats["complete_sequences"] += 1
        else:
            # Subject has incomplete sequence - EXCLUDE ENTIRELY
            sequential_stats["incomplete_sequences"] += 1
            sequential_stats["excluded_subjects"] += 1
            split_excluded += 1
            excluded_subjects.append(f"{split}/{subj}")
            missing = set(REQUIRED_VISITS) - set(found_visits)
            print(f"   ❌ EXCLUDING {subj}: missing {sorted(missing)}")
    
    print(f"   • Total subjects scanned: {split_subjects}")
    print(f"   • Complete sequences (included): {split_complete}")
    print(f"   • Incomplete sequences (excluded): {split_excluded}")
    print(f"   • Brain files to process: {split_files}")

print(f"\n📈 OVERALL STATISTICS:")
print(f"   • Total subjects scanned: {sequential_stats['total_subjects']}")
print(f"   • Complete sequences (included): {sequential_stats['complete_sequences']}")
print(f"   • Incomplete sequences (excluded): {sequential_stats['incomplete_sequences']}")
print(f"   • Subjects excluded due to missing files: {sequential_stats['excluded_subjects']}")
print(f"   • Data retention rate: {sequential_stats['complete_sequences']/sequential_stats['total_subjects']*100:.1f}%")
print(f"   • Total brain files to process: {len(tasks)}")

if excluded_subjects:
    print(f"\n❌ EXCLUDED SUBJECTS ({len(excluded_subjects)} total):")
    print("   These subjects were completely excluded due to missing skull-stripped files:")
    for excluded in excluded_subjects[:15]:  # Show first 15
        print(f"   • {excluded}")
    if len(excluded_subjects) > 15:
        print(f"   ... and {len(excluded_subjects) - 15} more")

if missing_files:
    print(f"\n📋 MISSING FILES DETAILS ({len(missing_files)} total):")
    print("   Specific files that caused subject exclusions:")
    for missing in missing_files[:15]:  # Show first 15
        print(f"   • {missing}")
    if len(missing_files) > 15:
        print(f"   ... and {len(missing_files) - 15} more")

if len(tasks) == 0:
    print("❌ No skull-stripped files found to process! Check previous step.")
    exit()

# ----------------------- WORKER FUNCTION ---------------------------------
def process_roi(task):
    """
    Process ROI registration for one brain file
    task = (split, subj, visit, brain_path)
    Output naming: {subj}_{visit}_hippo_mask.nii.gz
    """
    split, subj, visit, subj_brain = task
    out_dir = ROI_SUBJ_ROOT / split / subj
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create temporal-aware output filenames
    base = f"{subj}_{visit}"
    mat     = out_dir / f"{base}_subj2mni.mat"
    inv_mat = out_dir / f"{base}_mni2subj.mat"
    out_roi = out_dir / f"{base}_hippo_mask.nii.gz"

    if out_roi.exists():
        return "skipped"

    try:
        # Step 1: Register subject brain to MNI space
        subprocess.run(
            ["flirt", "-in", str(subj_brain), "-ref", MNI_BRAIN, "-omat", str(mat)],
            check=True, capture_output=True
        )
        
        # Step 2: Create inverse transformation matrix
        subprocess.run(
            ["convert_xfm", "-omat", str(inv_mat), "-inverse", str(mat)],
            check=True, capture_output=True
        )
        
        # Step 3: Warp ROI template to subject space
        subprocess.run(
            [
                "flirt",
                "-in", ROI_TEMPLATE,
                "-ref", str(subj_brain),
                "-applyxfm", "-init", str(inv_mat),
                "-interp", "nearestneighbour",
                "-out", str(out_roi)
            ],
            check=True, capture_output=True
        )
        return "success"
    except subprocess.CalledProcessError as e:
        return f"error_cmd_{e.returncode}"
    except Exception as e:
        return f"error_{str(e)[:20]}"

# ----------------------- PARALLEL EXECUTION -----------------------------
stats = defaultdict(int)
max_workers = min(MAX_THREADS, len(tasks))
print(f"\n🚀 Using {max_workers} parallel threads")
print(f"⚡ Starting ROI registration and warping...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_roi, t): t for t in tasks}
    for fut in tqdm(as_completed(futures), total=len(tasks), 
                    desc="Warping ROI to skull-stripped brains"):
        res = fut.result()
        stats[res] += 1

# ----------------------- DETAILED SUMMARY --------------------------------
print(f"\n" + "=" * 60)
print("📊 ROI REGISTRATION & WARPING SUMMARY")
print("=" * 60)

print(f"\n⚡ Processing Results:")
total_processed = sum(stats.values())
for key, val in sorted(stats.items()):
    percentage = (val / total_processed * 100) if total_processed > 0 else 0
    print(f"  {key.capitalize():12s}: {val:4d} ({percentage:5.1f}%)")

print(f"\n🧠 Sequential Data Summary:")
print(f"  Total subjects scanned: {sequential_stats['total_subjects']}")
print(f"  Complete sequences (processed): {sequential_stats['complete_sequences']}")
print(f"  Incomplete sequences (excluded): {sequential_stats['incomplete_sequences']}")
print(f"  Data retention rate: {sequential_stats['complete_sequences']/sequential_stats['total_subjects']*100:.1f}%")

# Show impact of exclusions
print(f"\n📊 Subject Exclusion Impact:")
print("  Subjects with missing skull-stripped files were completely removed")
print("  to ensure temporal sequence integrity for CNN+LSTM training.")

# Check output structure and verify temporal completeness
print(f"\n📁 Output Structure Verification:")
for split in ("train", "val", "test"):
    split_out = ROI_SUBJ_ROOT / split
    if split_out.exists():
        subjects = [d for d in split_out.iterdir() if d.is_dir()]
        total_roi_files = 0
        total_mat_files = 0
        complete_subjects = 0
        
        for subj_dir in subjects:
            roi_files = list(subj_dir.glob("*_hippo_mask.nii.gz"))
            mat_files = list(subj_dir.glob("*.mat"))
            total_roi_files += len(roi_files)
            total_mat_files += len(mat_files)
            
            # Check if subject has all three ROI files and matrices
            subj = subj_dir.name
            expected_rois = [f"{subj}_{visit}_hippo_mask.nii.gz" for visit in REQUIRED_VISITS]
            expected_mats = [f"{subj}_{visit}_{mat_type}.mat" 
                           for visit in REQUIRED_VISITS 
                           for mat_type in ["subj2mni", "mni2subj"]]
            
            if (all((subj_dir / roi).exists() for roi in expected_rois) and
                all((subj_dir / mat).exists() for mat in expected_mats)):
                complete_subjects += 1
        
        print(f"  {split:5s}: {len(subjects):3d} subjects, {total_roi_files:3d} ROI files, "
              f"{total_mat_files:3d} matrices, {complete_subjects:3d} complete")

# Show sample output files
print(f"\n📄 Sample Output Files:")
sample_rois = list(ROI_SUBJ_ROOT.rglob("*_hippo_mask.nii.gz"))[:6]
for i, sample_file in enumerate(sample_rois):
    rel_path = sample_file.relative_to(ROI_SUBJ_ROOT)
    if sample_file.exists():
        file_size = sample_file.stat().st_size / (1024*1024)  # MB
        print(f"  {i+1}. {rel_path} ({file_size:.2f} MB)")

print(f"\n📊 Temporal File Distribution:")
visit_counts = defaultdict(int)
for roi_file in ROI_SUBJ_ROOT.rglob("*_hippo_mask.nii.gz"):
    filename = roi_file.name
    for visit in REQUIRED_VISITS:
        if f"_{visit}_" in filename:
            visit_counts[visit] += 1
            break

for visit in REQUIRED_VISITS:
    print(f"  {visit.upper()}: {visit_counts[visit]} ROI files")

print(f"\n✅ ROI registration and warping complete!")
print(f"📂 Output directory: {ROI_SUBJ_ROOT}")
print(f"🎯 Ready for next step: Optimal Slice Extraction!")
print("=" * 60)
```

#### Optimal Slice Extraction


```python
# =====================================================================
# SEQUENTIAL DATASET OPTIMAL SLICE EXTRACTION
# Modified for temporal sequence data with visit-aware processing
# =====================================================================

# ------------ PATHS --------------------------------------------
ROI_SUBJ_ROOT = Path("../datasets/ADNI_1_5_T/3_roi_subj_space_sequential")
SKULL_ROOT    = Path("../datasets/ADNI_1_5_T/2_skull_stripping_sequential")

MAX_THREADS = 8
REQUIRED_VISITS = ["sc", "m06", "m12"]  # Expected temporal sequence
dir_lock = threading.Lock()  # Thread-safe directory creation

print("🧠 SEQUENTIAL DATASET OPTIMAL SLICE EXTRACTION")
print("=" * 70)
print(f"📂 ROI Input: {ROI_SUBJ_ROOT}")
print(f"📂 Brain Input: {SKULL_ROOT}")
print(f"📅 Expected visits: {REQUIRED_VISITS}")
print(f"🔧 Max threads: {MAX_THREADS}")

# ------------ SLICE TYPE CONFIGURATIONS -----------------------
SLICE_CONFIGS = {
    "axial": {
        "root": Path("../datasets/ADNI_1_5_T/4_optimal_axial_sequential"),
        "axis": 0,  # X-axis
        "coord": "x",
        "slice_func": lambda data, center: data[center, :, :]
    },
    "coronal": {
        "root": Path("../datasets/ADNI_1_5_T/4_optimal_coronal_sequential"),
        "axis": 1,  # Y-axis
        "coord": "y", 
        "slice_func": lambda data, center: data[:, center, :]
    },
    "sagittal": {
        "root": Path("../datasets/ADNI_1_5_T/4_optimal_sagittal_sequential"),
        "axis": 2,  # Z-axis
        "coord": "z",
        "slice_func": lambda data, center: data[:, :, center]
    }
}

# Create output directories
for slice_type, config in SLICE_CONFIGS.items():
    config["root"].mkdir(parents=True, exist_ok=True)
    print(f"📁 {slice_type.capitalize()} output: {config['root']}")

print()

# ------------ WORKER FUNCTION ---------------------------------
def process_slice(task, slice_type, config):
    """
    Process a single slice extraction task for temporal data
    task = (split, subj, visit, roi_path, brain_path)
    """
    split, subj, visit, roi_path, brain_path = task
    
    # Thread-safe directory creation
    out_dir = config["root"] / split / subj
    with dir_lock:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Check if this specific slice already exists
    coord = config["coord"]
    existing_files = list(out_dir.glob(f"{subj}_{visit}_optimal_{slice_type}_{coord}*.nii.gz"))
    if existing_files:
        return "skip"

    try:
        # Load ROI mask and brain data
        roi_data = nib.load(str(roi_path)).get_fdata()
        brain_img = nib.load(str(brain_path))
        brain_data = brain_img.get_fdata()

        # Find mean coordinate index of ROI voxels
        coord_inds = np.where(roi_data > 0)[config["axis"]]
        if coord_inds.size == 0:
            return "empty_roi"  # empty mask

        coord_center = int(round(coord_inds.mean()))

        # Extract slice using the appropriate function
        slice_2d = config["slice_func"](brain_data, coord_center)
        slice_img = nib.Nifti1Image(slice_2d[..., np.newaxis], brain_img.affine)

        # Save with temporal naming: {subj}_{visit}_optimal_{slice_type}_{coord}{position}.nii.gz
        out_file = out_dir / f"{subj}_{visit}_optimal_{slice_type}_{coord}{coord_center}.nii.gz"
        nib.save(slice_img, str(out_file))
        
        return "success"
        
    except Exception as e:
        return f"error: {str(e)}"

# ------------ GATHER TEMPORAL TASKS ---------------------------
def gather_temporal_tasks():
    """Gather all temporal slice extraction tasks"""
    tasks = []
    sequential_stats = {"total_subjects": 0, "complete_sequences": 0, "incomplete_sequences": 0}
    missing_pairs = []
    
    print("🔍 Scanning for temporal ROI and brain file pairs...")
    
    for split in ("train", "val", "test"):
        roi_split_dir = ROI_SUBJ_ROOT / split
        brain_split_dir = SKULL_ROOT / split
        
        if not roi_split_dir.exists() or not brain_split_dir.exists():
            print(f"⚠️  Split directory not found: {split}")
            continue
        
        print(f"\n📊 Processing {split.upper()} split:")
        split_subjects = 0
        split_complete = 0
        split_tasks = 0
        
        for subj_dir in roi_split_dir.iterdir():
            if not subj_dir.is_dir():
                continue
            
            subj = subj_dir.name
            split_subjects += 1
            sequential_stats["total_subjects"] += 1
            
            # Check for temporal ROI files: {subj}_{visit}_hippo_mask.nii.gz
            found_visits = []
            subject_tasks = []
            
            for visit in REQUIRED_VISITS:
                roi_file = subj_dir / f"{subj}_{visit}_hippo_mask.nii.gz"
                brain_file = brain_split_dir / subj / f"{subj}_{visit}_brain.nii.gz"
                
                if roi_file.exists() and brain_file.exists():
                    found_visits.append(visit)
                    subject_tasks.append((split, subj, visit, roi_file, brain_file))
                else:
                    if not roi_file.exists():
                        missing_pairs.append(f"{split}/{subj}/{visit} - ROI missing")
                    if not brain_file.exists():
                        missing_pairs.append(f"{split}/{subj}/{visit} - Brain missing")
            
            # STRICT REQUIREMENT: Only process subjects with complete temporal sequences
            if set(found_visits) == set(REQUIRED_VISITS):
                tasks.extend(subject_tasks)
                split_complete += 1
                split_tasks += len(subject_tasks)
                sequential_stats["complete_sequences"] += 1
            else:
                sequential_stats["incomplete_sequences"] += 1
                missing = set(REQUIRED_VISITS) - set(found_visits)
                print(f"   ❌ EXCLUDING {subj}: missing {sorted(missing)}")
        
        print(f"   • Total subjects scanned: {split_subjects}")
        print(f"   • Complete sequences (included): {split_complete}")
        print(f"   • Tasks to process: {split_tasks}")
    
    print(f"\n📈 OVERALL TEMPORAL STATISTICS:")
    print(f"   • Total subjects scanned: {sequential_stats['total_subjects']}")
    print(f"   • Complete sequences: {sequential_stats['complete_sequences']}")
    print(f"   • Incomplete sequences (excluded): {sequential_stats['incomplete_sequences']}")
    if sequential_stats['total_subjects'] > 0:
        retention_rate = sequential_stats['complete_sequences'] / sequential_stats['total_subjects'] * 100
        print(f"   • Data retention rate: {retention_rate:.1f}%")
    print(f"   • Total temporal tasks: {len(tasks)}")
    
    if missing_pairs:
        print(f"\n❌ MISSING FILE PAIRS ({len(missing_pairs)} total):")
        for missing in missing_pairs[:10]:
            print(f"   • {missing}")
        if len(missing_pairs) > 10:
            print(f"   ... and {len(missing_pairs) - 10} more")
    
    return tasks, sequential_stats

# ------------ PROCESS ALL SLICE TYPES -------------------------
print("=" * 70)
print("PARALLEL TEMPORAL SLICE EXTRACTION")
print("=" * 70)

# Gather temporal tasks once (shared for all slice types)
all_tasks, seq_stats = gather_temporal_tasks()

if len(all_tasks) == 0:
    print("❌ No complete temporal sequences found! Check previous steps.")
    exit()

overall_stats = {}

# Process each slice type sequentially but with parallel workers
for slice_type, config in SLICE_CONFIGS.items():
    print(f"\n{'='*50}")
    print(f"PROCESSING {slice_type.upper()} SLICES")
    print(f"{'='*50}")
    
    max_workers = min(MAX_THREADS, len(all_tasks))
    print(f"Using {max_workers} parallel threads for {slice_type}")
    
    # Process with parallel workers
    stats = {"success": 0, "skip": 0, "empty_roi": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_slice, task, slice_type, config): task 
            for task in all_tasks
        }
        
        for future in tqdm(
            as_completed(futures), 
            total=len(all_tasks), 
            desc=f"Extracting {slice_type} slices"
        ):
            result = future.result()
            
            if result == "success":
                stats["success"] += 1
            elif result == "skip":
                stats["skip"] += 1
            elif result == "empty_roi":
                stats["empty_roi"] += 1
            else:  # error
                stats["error"] += 1
    
    # Store results
    overall_stats[slice_type] = stats
    
    # Report for this slice type
    print(f"\n{slice_type.upper()} Results:")
    for status, count in stats.items():
        print(f"  {status.capitalize()}: {count}")

# ------------ TEMPORAL VERIFICATION & SUMMARY ----------------
print(f"\n{'='*70}")
print("TEMPORAL SLICE EXTRACTION SUMMARY")
print(f"{'='*70}")

# Overall statistics
for slice_type, stats in overall_stats.items():
    total_processed = sum(stats.values())
    success_rate = (stats["success"] / total_processed * 100) if total_processed > 0 else 0
    print(f"\n{slice_type.upper()}:")
    print(f"  Total processed: {total_processed}")
    print(f"  Successful: {stats['success']} ({success_rate:.1f}%)")
    print(f"  Skipped: {stats['skip']}")
    print(f"  Empty ROI: {stats['empty_roi']}")
    print(f"  Errors: {stats['error']}")

grand_total_success = sum(stats["success"] for stats in overall_stats.values())
print(f"\nGrand total successful extractions: {grand_total_success}")

# Verify temporal completeness in output
print(f"\n📊 TEMPORAL COMPLETENESS VERIFICATION:")
for slice_type, config in SLICE_CONFIGS.items():
    print(f"\n{slice_type.upper()} temporal verification:")
    
    for split in ("train", "val", "test"):
        split_dir = config["root"] / split
        if not split_dir.exists():
            continue
            
        subjects_with_complete_sequences = 0
        total_subjects = 0
        visit_counts = defaultdict(int)
        
        for subj_dir in split_dir.iterdir():
            if not subj_dir.is_dir():
                continue
                
            total_subjects += 1
            subj = subj_dir.name
            
            # Check for all three temporal slices
            temporal_files = []
            for visit in REQUIRED_VISITS:
                pattern = f"{subj}_{visit}_optimal_{slice_type}_*.nii.gz"
                files = list(subj_dir.glob(pattern))
                if files:
                    temporal_files.append(visit)
                    visit_counts[visit] += 1
            
            if set(temporal_files) == set(REQUIRED_VISITS):
                subjects_with_complete_sequences += 1
        
        if total_subjects > 0:
            completeness = subjects_with_complete_sequences / total_subjects * 100
            print(f"  {split}: {subjects_with_complete_sequences}/{total_subjects} subjects complete ({completeness:.1f}%)")
            for visit in REQUIRED_VISITS:
                print(f"    {visit}: {visit_counts[visit]} slices")

print(f"\n🎯 SEQUENTIAL DATA SUMMARY:")
print(f"  • Subjects with complete temporal sequences: {seq_stats['complete_sequences']}")
print(f"  • Subjects excluded due to missing data: {seq_stats['incomplete_sequences']}")
print(f"  • Total temporal slices extracted: {grand_total_success}")
print(f"  • Expected slices per complete subject: {len(SLICE_CONFIGS)} slice types")

print(f"\n✅ Sequential optimal slice extraction complete!")
print(f"📂 Output directories:")
for slice_type, config in SLICE_CONFIGS.items():
    print(f"   • {slice_type.capitalize()}: {config['root']}")

print(f"\n🎯 Ready for temporal sequence labeling and organization!")
print("=" * 70)
```

#### Labelling


```python
# =====================================================================
# SEQUENTIAL TEMPORAL DATASET LABELLING - VARIABLE REFERENCE
# Organizes temporal sequences by subject and diagnostic group
# For variable reference approach where each visit uses optimal ROI center
# =====================================================================

# ---------------- PATHS ----------------------------------------
LABEL_ROOT = Path("../datasets/ADNI_1_5_T/5_labelling_sequential_variable")
LABEL_ROOT.mkdir(parents=True, exist_ok=True)

# Updated paths for sequential data
meta_csv = Path("../datasets/ADNI_1_5_T/1_splitted_sequential/metadata_split.csv")
REQUIRED_VISITS = ["sc", "m06", "m12"]

print("🧠 SEQUENTIAL TEMPORAL DATASET LABELLING - VARIABLE REFERENCE")
print("=" * 70)
print(f"📂 Output: {LABEL_ROOT}")
print(f"📄 Metadata: {meta_csv}")
print(f"📅 Expected visits: {REQUIRED_VISITS}")
print(f"🔍 Approach: Variable reference (each visit uses optimal ROI center)")
print()

# ----------- SUBJECT → GROUP mapping --------------------------
if not meta_csv.exists():
    print(f"❌ Metadata file not found: {meta_csv}")
    print("Please run the sequential splitting step first!")
    exit()

filtered_dataframe = pd.read_csv(meta_csv)
subject_to_group = dict(zip(filtered_dataframe["Subject"], filtered_dataframe["Group"]))
subject_to_split = dict(zip(filtered_dataframe["Subject"], filtered_dataframe["Split"]))

print(f"✅ Loaded {len(subject_to_group)} subject→group mappings")
print(f"📊 Group distribution: {dict(pd.Series(list(subject_to_group.values())).value_counts())}")
print(f"📈 Split distribution: {dict(pd.Series(list(subject_to_split.values())).value_counts())}")

# ----------- Define slice types and their properties ----------
slice_types = {
    "axial": {
        "root": Path("../datasets/ADNI_1_5_T/4_optimal_axial_sequential"),
        "pattern": "*_optimal_axial*.nii*",
        "split_key": "_optimal_axial"
    },
    "coronal": {
        "root": Path("../datasets/ADNI_1_5_T/4_optimal_coronal_sequential"),
        "pattern": "*_optimal_coronal*.nii*", 
        "split_key": "_optimal_coronal"
    },
    "sagittal": {
        "root": Path("../datasets/ADNI_1_5_T/4_optimal_sagittal_sequential"),
        "pattern": "*_optimal_sagittal*.nii*",
        "split_key": "_optimal_sagittal"
    }
}

# Verify input directories exist
missing_dirs = []
for slice_type, config in slice_types.items():
    if not config["root"].exists():
        missing_dirs.append(config["root"])

if missing_dirs:
    print(f"⚠️  Missing input directories:")
    for missing in missing_dirs:
        print(f"   • {missing}")
    print("Please run the sequential slice extraction step first!")

# ----------- Prepare output dirs ------------------------------
splits = ["train", "val", "test"]
groups = ["AD", "CN"]

# Create directories organized by SUBJECT, not individual slices
# Structure: label_root/slice_type/split/group/subject/temporal_files
label_dirs = {}
for slice_type in slice_types.keys():
    for split in splits:
        for grp in groups:
            path = LABEL_ROOT / slice_type / split / grp
            path.mkdir(parents=True, exist_ok=True)
            label_dirs[(slice_type, split, grp)] = path

print(f"✅ Created output directory structure")

# ----------- TEMPORAL SUBJECT ID EXTRACTION -------------------
def extract_subject_id_temporal(filename):
    """
    Extract subject ID from temporal filename
    
    Examples:
    - 062_S_0690_sc_optimal_coronal_y123.nii.gz → 062_S_0690
    - 062_S_0690_m06_optimal_axial_x45.nii.gz → 062_S_0690
    """
    # Remove extensions
    clean_name = filename.replace('.nii.gz', '').replace('.nii', '')
    
    # Split by underscores
    parts = clean_name.split('_')
    
    # Look for pattern: XXX_S_YYYY_visit_...
    for i in range(len(parts) - 2):
        if (parts[i+1] == 'S' and 
            parts[i].isdigit() and 
            parts[i+2].isdigit() and
            i+3 < len(parts) and parts[i+3] in REQUIRED_VISITS):
            return f"{parts[i]}_S_{parts[i+2]}"
    
    # Fallback: look for any XXX_S_YYYY pattern
    for i in range(len(parts) - 2):
        if parts[i+1] == 'S' and parts[i].isdigit() and parts[i+2].isdigit():
            return f"{parts[i]}_S_{parts[i+2]}"
    
    return None

def extract_visit_from_filename(filename):
    """Extract visit code from temporal filename"""
    for visit in REQUIRED_VISITS:
        if f"_{visit}_" in filename:
            return visit
    return None

# ----------- Process each slice type sequentially -------------
total_stats = defaultdict(int)
all_unmatched_subjects = set()
all_unknown_groups = []
temporal_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for slice_type, config in slice_types.items():
    print(f"\n{'='*60}")
    print(f"PROCESSING {slice_type.upper()} TEMPORAL SLICES")
    print(f"{'='*60}")
    
    SLICE_ROOT = config["root"]
    
    if not SLICE_ROOT.exists():
        print(f"⚠️  Directory not found: {SLICE_ROOT}")
        print(f"Skipping {slice_type} slices...")
        continue
    
    # ----------- Gather temporal slice files by subject -----------
    print("🔍 Scanning for temporal slice files...")
    
    subject_files = defaultdict(lambda: defaultdict(list))  # subject -> visit -> [files]
    total_files = 0
    
    for split in splits:
        split_dir = SLICE_ROOT / split
        if not split_dir.exists():
            continue
            
        # Get all slice files in this split
        slice_files = glob.glob(str(split_dir / "**" / config["pattern"]), recursive=True)
        total_files += len(slice_files)
        
        for filepath in slice_files:
            filename = Path(filepath).name
            subject_id = extract_subject_id_temporal(filename)
            visit = extract_visit_from_filename(filename)
            
            if subject_id and visit:
                subject_files[subject_id][visit].append(filepath)

    print(f"📊 Found {total_files} total {slice_type} slice files")
    print(f"👥 Covering {len(subject_files)} unique subjects")

    # ----------- Process subjects with complete temporal sequences ---------
    stats = defaultdict(int)
    unmatched_subjects, unknown_groups = [], []
    processed_subjects = defaultdict(lambda: defaultdict(int))  # split -> group -> count

    for subject_id, visit_files in tqdm(subject_files.items(), desc=f"Processing {slice_type} subjects"):
        
        # Check if subject has complete temporal sequence
        available_visits = set(visit_files.keys())
        if available_visits != set(REQUIRED_VISITS):
            missing_visits = set(REQUIRED_VISITS) - available_visits
            stats["incomplete_sequence"] += 1
            continue
        
        # Extract subject metadata
        if subject_id not in subject_to_group:
            unmatched_subjects.append(subject_id)
            all_unmatched_subjects.add(subject_id)
            stats["unmatched"] += 1
            continue
            
        group = subject_to_group[subject_id]
        split = subject_to_split[subject_id]
        
        if group not in groups:
            unknown_groups.append((subject_id, group))
            stats["unknown_group"] += 1
            continue
            
        if split not in splits:
            stats["unknown_split"] += 1
            continue
        
        # Create subject directory
        subject_dir = label_dirs[(slice_type, split, group)] / subject_id
        subject_dir.mkdir(exist_ok=True)
        
        # Copy all temporal files for this subject
        subject_file_count = 0
        for visit in REQUIRED_VISITS:
            for filepath in visit_files[visit]:
                filename = Path(filepath).name
                dst = subject_dir / filename
                
                if not dst.exists():
                    shutil.copy2(filepath, dst)
                    subject_file_count += 1
                    temporal_stats[slice_type][split][group] += 1
        
        if subject_file_count > 0:
            processed_subjects[split][group] += 1
            stats[(split, group)] += subject_file_count
            total_stats[(slice_type, split, group)] += subject_file_count
            stats["successful_subjects"] += 1

    # ----------- Report for this slice type -------------------
    print(f"\n{slice_type.upper()} TEMPORAL LABELLING COMPLETE:")
    
    # Subject-level statistics
    print(f"\n📊 Subject-level Results:")
    print(f"  Successful subjects: {stats['successful_subjects']}")
    print(f"  Incomplete sequences: {stats['incomplete_sequence']}")
    print(f"  Unmatched subjects: {stats['unmatched']}")
    print(f"  Unknown groups: {stats['unknown_group']}")
    
    # File-level statistics by split and group
    print(f"\n📁 File Distribution by Split and Group:")
    for split in splits:
        if any(processed_subjects[split].values()):
            print(f"\n  {split.upper()} Split:")
            for grp in groups:
                file_count = temporal_stats[slice_type][split][grp]
                subj_count = processed_subjects[split][grp]
                if subj_count > 0:
                    avg_files = file_count / subj_count
                    print(f"    {grp}: {subj_count} subjects, {file_count} files (avg {avg_files:.1f} files/subject)")
                    print(f"         → {label_dirs[(slice_type, split, grp)]}")
                else:
                    print(f"    {grp}: 0 subjects")

    total_ok = sum(stats[(s, g)] for s in splits for g in groups)
    print(f"\n📈 Total {slice_type} files organized: {total_ok}")

    all_unknown_groups.extend(unknown_groups)

# ----------- Final Summary Report -----------------------------
print("\n" + "="*70)
print("OVERALL TEMPORAL LABELLING SUMMARY")
print("="*70)

print(f"\n📊 TEMPORAL ORGANIZATION RESULTS:")
for slice_type in slice_types.keys():
    if slice_type in temporal_stats:
        print(f"\n**{slice_type.upper()} SLICES:**")
        slice_total = 0
        for split in splits:
            split_files = sum(temporal_stats[slice_type][split].values())
            if split_files > 0:
                print(f"  {split}: {split_files} files")
                for grp in groups:
                    n = temporal_stats[slice_type][split][grp]
                    if n > 0:
                        print(f"    └─ {grp}: {n:4d} files")
                slice_total += split_files
        print(f"  Total: {slice_total} files")

# Overall statistics
grand_total = 0
for slice_type in slice_types.keys():
    for split in splits:
        for grp in groups:
            grand_total += temporal_stats[slice_type][split][grp]

total_unmatched = len(all_unmatched_subjects)
total_unknown = len(all_unknown_groups)

print(f"\n**GRAND TOTAL:**")
print(f"  Successfully organized: {grand_total} temporal files")
print(f"  Unmatched subjects: {total_unmatched} unique subjects")
print(f"  Unknown groups: {total_unknown} subjects")

# Show temporal completeness statistics
print(f"\n🔄 TEMPORAL SEQUENCE COMPLETENESS:")
complete_subjects_per_type = {}
for slice_type in slice_types.keys():
    if slice_type in temporal_stats:
        total_subjects = 0
        for split in splits:
            for grp in groups:
                files = temporal_stats[slice_type][split][grp]
                # Assuming 3 files per subject (sc, m06, m12)
                subjects = files // len(REQUIRED_VISITS) if files > 0 else 0
                total_subjects += subjects
        complete_subjects_per_type[slice_type] = total_subjects
        print(f"  {slice_type.capitalize()}: ~{total_subjects} subjects with complete temporal sequences")

if all_unmatched_subjects:
    print(f"\n❌ Unmatched subjects ({len(all_unmatched_subjects)}):")
    for s in sorted(list(all_unmatched_subjects))[:10]:
        print(f"  • {s}")
    if len(all_unmatched_subjects) > 10:
        print(f"  … and {len(all_unmatched_subjects) - 10} more")

if all_unknown_groups:
    unique_unknown = list(set(all_unknown_groups))
    print(f"\n⚠️  Subjects with unknown diagnostic labels ({len(unique_unknown)}):")
    for s, g in unique_unknown[:10]:
        print(f"  • {s}: {g}")

print(f"\n📁 OUTPUT STRUCTURE:")
print(f"   {LABEL_ROOT}/")
print(f"   ├── axial/coronal/sagittal/")
print(f"   │   ├── train/val/test/")
print(f"   │   │   ├── AD/CN/")
print(f"   │   │   │   ├── {list(subject_to_group.keys())[0] if subject_to_group else 'SUBJECT_ID'}/")
print(f"   │   │   │   │   ├── {list(subject_to_group.keys())[0] if subject_to_group else 'SUBJECT'}_sc_optimal_TYPE_*.nii.gz")
print(f"   │   │   │   │   ├── {list(subject_to_group.keys())[0] if subject_to_group else 'SUBJECT'}_m06_optimal_TYPE_*.nii.gz")
print(f"   │   │   │   │   └── {list(subject_to_group.keys())[0] if subject_to_group else 'SUBJECT'}_m12_optimal_TYPE_*.nii.gz")

print(f"\n✅ Sequential temporal labelling complete!")
print(f"🎯 Ready for CNN+LSTM temporal sequence modeling!")
print(f"⚠️  Note: Variable reference approach - each visit uses its optimal ROI center")
print("="*70)
```

#### [QC] Visualize Sample Data


```python
# =====================================================================
# SEQUENTIAL TEMPORAL DATASET VISUALIZATION
# Visualizes temporal sequences (sc, m06, m12) for sample subjects
# =====================================================================

# ----------------- CONFIGURATION ---------------------------------------
LABEL_ROOT   = "../datasets/ADNI_1_5_T/5_labelling_sequential_variable"
splits       = ['train', 'val', 'test']
# classes      = ['CN', 'MCI', 'AD']
classes      = ['CN', 'AD']
visits       = ['sc', 'm06', 'm12']  # Temporal sequence
class_colors = {'CN': 'green', 'MCI': 'orange', 'AD': 'red'}
visit_colors = {'sc': '#2E8B57', 'm06': '#4682B4', 'm12': '#8B0000'}  # Dark green, steel blue, dark red

# Define slice types and their coordinate patterns for temporal data
slice_types = {
    'axial': {
        'pattern': re.compile(r"_x(\d+)"),   # captures digits after "_x" (variable reference)
        'coord_name': 'X',
        'split_key': '_optimal_axial'
    },
    'coronal': {
        'pattern': re.compile(r"_y(\d+)"),   # captures digits after "_y" (variable reference)
        'coord_name': 'Y', 
        'split_key': '_optimal_coronal'
    },
    'sagittal': {
        'pattern': re.compile(r"_z(\d+)"),   # captures digits after "_z" (variable reference)
        'coord_name': 'Z',
        'split_key': '_optimal_sagittal'
    }
}

random.seed(2025)

print("🧠 SEQUENTIAL TEMPORAL DATASET VISUALIZATION")
print("=" * 80)
print(f"📂 Dataset Root: {LABEL_ROOT}")
print(f"📅 Temporal Sequence: {' → '.join(visits)}")
print(f"🎯 Objective: Show disease progression across time")
print()

def extract_coordinate(filename, pattern):
    """Return coordinate index (string) if pattern exists, else '?'."""
    m = pattern.search(filename)
    return m.group(1) if m else "?"

def extract_subject_id_from_path(subject_dir_name):
    """Extract subject ID from directory name (should be the subject ID itself)"""
    return subject_dir_name

def get_temporal_files_for_subject(subject_dir, slice_type_config):
    """Get temporal files for a subject, organized by visit"""
    visit_files = {}
    
    if not os.path.exists(subject_dir):
        return visit_files
    
    # Look for files matching the temporal pattern
    for filename in os.listdir(subject_dir):
        if filename.endswith('.nii.gz') and slice_type_config['split_key'] in filename:
            # Extract visit from filename
            for visit in visits:
                if f"_{visit}_" in filename:
                    visit_files[visit] = os.path.join(subject_dir, filename)
                    break
    
    return visit_files

def select_representative_subject(cls_dir, slice_type_config):
    """Select a subject with complete temporal sequence"""
    if not os.path.exists(cls_dir):
        return None, {}
    
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(cls_dir) 
                   if os.path.isdir(os.path.join(cls_dir, d))]
    
    if not subject_dirs:
        return None, {}
    
    # Shuffle to get random selection
    random.shuffle(subject_dirs)
    
    # Find a subject with complete temporal sequence
    for subject_id in subject_dirs:
        subject_path = os.path.join(cls_dir, subject_id)
        temporal_files = get_temporal_files_for_subject(subject_path, slice_type_config)
        
        # Check if subject has all required visits
        if set(temporal_files.keys()) == set(visits):
            return subject_id, temporal_files
    
    # If no complete sequence found, return the first subject with any files
    if subject_dirs:
        subject_id = subject_dirs[0]
        subject_path = os.path.join(cls_dir, subject_id)
        temporal_files = get_temporal_files_for_subject(subject_path, slice_type_config)
        return subject_id, temporal_files
    
    return None, {}

# Process each slice type
for slice_type, config in slice_types.items():
    pattern = config['pattern']
    coord_name = config['coord_name']
    split_key = config['split_key']
    
    print(f"\n{'='*80}")
    print(f"VISUALIZING {slice_type.upper()} TEMPORAL SEQUENCES")
    print(f"{'='*80}")
    
    # Create comprehensive temporal visualization for this slice type
    fig, axes = plt.subplots(len(classes), len(splits) * len(visits), 
                            figsize=(5 * len(splits) * len(visits), 4 * len(classes)))
    
    if len(classes) == 1:
        axes = axes.reshape(1, -1)
    if len(splits) * len(visits) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Temporal Disease Progression - {slice_type.title()} View\n'
                 f'Fixed Reference Slices: Same Anatomical Location Across Time',
                 fontsize=16, fontweight='bold', y=1)
    
    # Collect subjects and their temporal data
    all_subjects_data = {}
    reference_positions = []
    
    for row, cls in enumerate(classes):
        all_subjects_data[cls] = {}
        
        for split_idx, split in enumerate(splits):
            cls_dir = os.path.join(LABEL_ROOT, slice_type, split, cls)
            
            # Select representative subject with complete temporal sequence
            subject_id, temporal_files = select_representative_subject(cls_dir, config)
            all_subjects_data[cls][split] = (subject_id, temporal_files)
            
            if subject_id and temporal_files:
                # display(Markdown(f"**{cls} - {split.upper()}**: Subject `{subject_id}` "
                #                f"({len(temporal_files)}/{len(visits)} visits available)"))
                
                # Show available visits
                available_visits = sorted(temporal_files.keys())
                missing_visits = set(visits) - set(available_visits)
                if missing_visits:
                    display(Markdown(f"  ⚠️ Missing visits: {sorted(missing_visits)}"))
            else:
                display(Markdown(f"**{cls} - {split.upper()}**: No subjects found"))
    
    # Plot temporal sequences
    for row, cls in enumerate(classes):
        for split_idx, split in enumerate(splits):
            subject_id, temporal_files = all_subjects_data[cls][split]
            
            for visit_idx, visit in enumerate(visits):
                col = split_idx * len(visits) + visit_idx
                ax = axes[row, col]
                
                if subject_id and visit in temporal_files:
                    try:
                        # Load and display the image
                        img_path = temporal_files[visit]
                        data = np.squeeze(nib.load(img_path).get_fdata())
                        
                        im = ax.imshow(data.T, cmap='gray', origin='lower', aspect='equal')
                        
                        # Extract coordinate position
                        filename = os.path.basename(img_path)
                        coord_pos = extract_coordinate(filename, pattern)
                        reference_positions.append(coord_pos)
                        
                        # # Title with subject, visit, and coordinate
                        # ax.set_title(f'{subject_id}\n{visit.upper()} | {coord_name}={coord_pos}',
                        #            fontsize=10, pad=8,
                        #            color=visit_colors[visit], fontweight='bold')
                        
                        # # Add intensity range info
                        # ax.text(0.02, 0.98, f'[{data.min():.0f}, {data.max():.0f}]',
                        #        transform=ax.transAxes, fontsize=8,
                        #        verticalalignment='top', horizontalalignment='left',
                        #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                        
                        ax.axis('off')
                        
                        # # Add colorbar for first visit of each split
                        # if visit_idx == 0:
                        #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    except Exception as e:
                        ax.text(0.5, 0.5, f'{visit.upper()}\nERROR\n{str(e)[:20]}...',
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=10, color='red')
                        ax.set_title(f'{subject_id if subject_id else "No Subject"}\n{visit.upper()}',
                                   fontsize=10, color='red')
                        ax.axis('off')
                else:
                    # Missing data
                    ax.text(0.5, 0.5, f'{visit.upper()}\nMISSING',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, color='gray', alpha=0.7)
                    ax.set_title(f'{subject_id if subject_id else "No Subject"}\n{visit.upper()}',
                               fontsize=10, color='gray')
                    ax.axis('off')
        
        # Add class label on the left
        if row < len(axes) and len(axes[row]) > 0:
            axes[row, 0].text(-0.15, 0.5, f'{cls}\n{class_colors[cls]}',
                            transform=axes[row, 0].transAxes,
                            fontsize=14, fontweight='bold',
                            color=class_colors[cls],
                            rotation=90, va='center', ha='center')
    
    # Add split and visit labels at the top
    for split_idx, split in enumerate(splits):
        for visit_idx, visit in enumerate(visits):
            col = split_idx * len(visits) + visit_idx
            if col < axes.shape[1]:
                # Split label (spanning all visits for this split)
                if visit_idx == len(visits) // 2:  # Middle visit for split label
                    axes[0, col].text(0.5, 1.15, split.upper(),
                                     transform=axes[0, col].transAxes,
                                     fontsize=14, fontweight='bold',
                                     ha='center', va='center',
                                     bbox=dict(boxstyle='round,pad=0.5', 
                                             facecolor='lightblue', alpha=0.7))
                
                # Visit label
                axes[0, col].text(0.5, 1.05, visit.upper(),
                                transform=axes[0, col].transAxes,
                                fontsize=12, fontweight='bold',
                                color=visit_colors[visit],
                                ha='center', va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, left=0.08, hspace=0.3, wspace=0.1)
    plt.show()
    
    # Summary statistics for this slice type
    print(f"\n📊 {slice_type.upper()} TEMPORAL SUMMARY:")
    
    total_subjects = 0
    complete_sequences = 0
    
    for cls in classes:
        print(f"\n  {cls} Class:")
        for split in splits:
            subject_id, temporal_files = all_subjects_data[cls][split]
            if subject_id:
                total_subjects += 1
                available_visits = len(temporal_files)
                is_complete = available_visits == len(visits)
                if is_complete:
                    complete_sequences += 1
                
                print(f"    {split:5s}: {subject_id} ({available_visits}/{len(visits)} visits)")
                if not is_complete:
                    missing = set(visits) - set(temporal_files.keys())
                    print(f"           Missing: {sorted(missing)}")
            else:
                print(f"    {split:5s}: No subjects found")
    
    if total_subjects > 0:
        completion_rate = complete_sequences / total_subjects * 100
        print(f"\n  Overall: {complete_sequences}/{total_subjects} subjects with complete sequences ({completion_rate:.1f}%)")
    
    # Reference position variability analysis (important for variable reference)
    if reference_positions:
        # Convert to integers for analysis
        try:
            positions_int = [int(p) for p in reference_positions if p != "?"]
            if positions_int:
                pos_mean = np.mean(positions_int)
                pos_std = np.std(positions_int)
                pos_range = max(positions_int) - min(positions_int)
                print(f"  Reference {coord_name}-position statistics:")
                print(f"    Range: {min(positions_int)} - {max(positions_int)} (span: {pos_range})")
                print(f"    Mean: {pos_mean:.1f}, Std: {pos_std:.1f}")
                
                if pos_std > 2:
                    print(f"    ⚠️  High variability in slice positions across visits!")
                    print(f"    ⚠️  This may confound temporal learning - consider fixed reference")
                else:
                    print(f"    ✅ Relatively consistent slice positions")
        except:
            pass
    
    unique_positions = list(set(reference_positions))
    if len(unique_positions) <= 5:  # Show if few unique positions
        print(f"  Unique {coord_name}-positions used: {sorted(unique_positions)}")
    else:
        print(f"  Total unique {coord_name}-positions: {len(unique_positions)}")
    
    print(f"\n  🎯 Temporal Analysis Insights:")
    print(f"     • Variable slice positions: Each visit uses its optimal ROI center")
    print(f"     • Shows ROI center changes + tissue changes over time")
    print(f"     • May confound CNN+LSTM learning (position + disease changes)")
    print(f"     • Consider fixed reference approach for pure disease progression")

print(f"\n{'='*80}")
print("🎉 TEMPORAL VISUALIZATION COMPLETE!")
print(f"{'='*80}")
print(f"✅ Visualized temporal disease progression across {len(visits)} timepoints")
print(f"✅ Fixed reference approach ensures consistent anatomical locations")
print(f"✅ Data ready for CNN+LSTM temporal sequence modeling")
print(f"✅ Each subject contributes complete temporal story: {' → '.join(visits)}")
print("="*80)
```

#### 2D Conversion


```python
# =====================================================================
# SEQUENTIAL TEMPORAL 2D CONVERSION - DIRECT CONVERSION
# Convert temporal NIfTI slices directly to PNG without transformations
# Each visit contributes one slice per view (sc, m06, m12)
# =====================================================================

# ----------------- PARAMETERS ----------------------------------
INTENSITY_PERCENTILE = (1, 99)
TARGET_SIZE = (256, 256)  # Target resolution for all PNG images
INTERPOLATION_METHOD = Image.Resampling.LANCZOS  # High-quality interpolation

random.seed(2025)
np.random.seed(2025)

# ----------------- PATHS ---------------------------------------
ROOT = Path("../datasets/ADNI_1_5_T")
META_CSV = ROOT / "1_splitted_sequential" / "metadata_split.csv"
REQUIRED_VISITS = ["sc", "m06", "m12"]

# Input: Sequential labeled data (organized by subject)
SLICE_TYPES = {
    "axial": {
        "label_root": ROOT / "5_labelling_sequential_variable" / "axial",
        "pattern": "_optimal_axial_x",
        "coord_name": "x"
    },
    "coronal": {
        "label_root": ROOT / "5_labelling_sequential_variable" / "coronal",
        "pattern": "_optimal_coronal_y", 
        "coord_name": "y"
    },
    "sagittal": {
        "label_root": ROOT / "5_labelling_sequential_variable" / "sagittal",
        "pattern": "_optimal_sagittal_z",
        "coord_name": "z"
    }
}

# Create output directories for each slice type
OUT_ROOT = ROOT / "6_2Dconverted_sequential"
for slice_type in SLICE_TYPES.keys():
    for split in ("train", "val", "test"):
        # for cls in ("AD", "MCI", "CN"):
        for cls in ("AD", "CN"):
            (OUT_ROOT / slice_type / split / cls).mkdir(parents=True, exist_ok=True)

print("🧠 SEQUENTIAL TEMPORAL 2D CONVERSION - DIRECT CONVERSION")
print("=" * 70)
print(f"📂 Input: Labeled sequential data")
print(f"📁 Output: {OUT_ROOT}")
print(f"📅 Expected visits: {REQUIRED_VISITS}")
print(f"📐 Target resolution: {TARGET_SIZE[0]}×{TARGET_SIZE[1]} pixels")
print(f"🎨 Interpolation: {INTERPOLATION_METHOD}")
print(f"🔄 Approach: Direct conversion (no transformations)")
print()

# ----------------- HELPER FUNCTIONS ----------------------------
def normalize_image(data, p_range=(1, 99)):
    """Normalize image intensity for PNG conversion"""
    p_low, p_high = np.percentile(data, p_range)
    data = np.clip(data, p_low, p_high)
    data = (data - p_low) / (p_high - p_low + 1e-8)
    return (data * 255).astype(np.uint8)

def resize_to_target(img, target_size=(256, 256), method=Image.Resampling.LANCZOS):
    """Resize PIL Image to target size using high-quality interpolation"""
    return img.resize(target_size, method)

def extract_visit_from_filename(filename):
    """Extract visit code from filename"""
    for visit in REQUIRED_VISITS:
        if f"_{visit}_" in filename:
            return visit
    return None

def extract_coordinate_position(filename, pattern, coord_name):
    """Extract coordinate position from filename"""
    try:
        coord_part = filename.split(f'_{coord_name}')[1].split('.')[0]
        return int(coord_part)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not extract {coord_name.upper()} position from filename: {filename}")

def extract_subject_id_from_filename(filename):
    """Extract subject ID from temporal filename"""
    # Remove extensions
    clean_name = filename.replace('.nii.gz', '').replace('.nii', '')
    
    # Split by underscores
    parts = clean_name.split('_')
    
    # Look for pattern: XXX_S_YYYY_visit_...
    for i in range(len(parts) - 2):
        if (parts[i+1] == 'S' and 
            parts[i].isdigit() and 
            parts[i+2].isdigit() and
            i+3 < len(parts) and parts[i+3] in REQUIRED_VISITS):
            return f"{parts[i]}_S_{parts[i+2]}"
    
    # Fallback: look for any XXX_S_YYYY pattern
    for i in range(len(parts) - 2):
        if parts[i+1] == 'S' and parts[i].isdigit() and parts[i+2].isdigit():
            return f"{parts[i]}_S_{parts[i+2]}"
    
    return None

# ----------------- SUBJECT → GROUP / SPLIT ---------------------
meta = pd.read_csv(META_CSV)
subj_to_group = dict(zip(meta.Subject, meta.Group))
subj_to_split = dict(zip(meta.Subject, meta.Split))

print(f"✅ Loaded {len(subj_to_group)} subject mappings")
print(f"📊 Group distribution: {dict(pd.Series(list(subj_to_group.values())).value_counts())}")
print(f"📈 Split distribution: {dict(pd.Series(list(subj_to_split.values())).value_counts())}")

# Check if labeled slice files exist before processing
print(f"\n🔍 Checking sequential labeled data availability:")
for slice_type, config in SLICE_TYPES.items():
    label_root = config["label_root"]
    if label_root.exists():
        total_subjects = 0
        total_files = 0
        
        for split in ["train", "val", "test"]:
            # for cls in ["AD", "MCI", "CN"]:
            for cls in ["AD", "CN"]:
                cls_dir = label_root / split / cls
                if cls_dir.exists():
                    subjects = [d for d in cls_dir.iterdir() if d.is_dir()]
                    total_subjects += len(subjects)
                    
                    for subj_dir in subjects:
                        files = list(subj_dir.glob("*.nii*"))
                        total_files += len(files)
                        
        print(f"  {slice_type}: {total_subjects} subjects, {total_files} NIfTI files")
    else:
        print(f"  {slice_type}: directory {label_root} does not exist")

# ----------------- PROCESS EACH SLICE TYPE ---------------------
for slice_type, config in SLICE_TYPES.items():
    print(f"\n{'='*70}")
    print(f"PROCESSING {slice_type.upper()} TEMPORAL SLICES → PNG")
    print(f"{'='*70}")
    
    LABEL_ROOT = config["label_root"]
    pattern = config["pattern"]
    coord_name = config["coord_name"]
    
    if not LABEL_ROOT.exists():
        print(f"⚠️  Directory not found: {LABEL_ROOT}")
        print(f"Skipping {slice_type} slices...")
        continue
    
    # ----------------- GATHER TEMPORAL TASKS FROM SUBJECTS --------
    tasks = []
    subject_stats = defaultdict(lambda: defaultdict(int))
    
    for split in ("train", "val", "test"):
        split_dir = LABEL_ROOT / split
        if not split_dir.exists():
            continue

        # for cls in ("AD", "MCI", "CN"):
        for cls in ("AD", "CN"):
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            
            # Process each subject directory
            subject_dirs = [d for d in cls_dir.iterdir() if d.is_dir()]
            subject_stats[split][cls] = len(subject_dirs)
            
            for subj_dir in subject_dirs:
                subject_id = subj_dir.name
                
                # Find NIfTI files in this subject directory
                nii_files = list(subj_dir.glob(f"*{pattern}*.nii*"))
                
                for nii_file in nii_files:
                    # Extract visit from filename
                    visit = extract_visit_from_filename(nii_file.name)
                    if visit:
                        tasks.append((split, cls, subject_id, visit, nii_file))

    print(f"📊 Found {len(tasks)} temporal NIfTI files to convert")
    
    # Display subject distribution
    print(f"👥 Subject distribution:")
    for split in ("train", "val", "test"):
        if any(subject_stats[split].values()):
            print(f"  {split}: {sum(subject_stats[split].values())} subjects")
            # for cls in ("AD", "MCI", "CN"):
            for cls in ("AD", "CN"):
                count = subject_stats[split][cls]
                if count > 0:
                    print(f"    {cls}: {count} subjects")

    if len(tasks) == 0:
        print(f"No {slice_type} files found. Skipping...")
        continue

    # ----------------- DIRECT CONVERSION TO PNG --------------------
    stats = defaultdict(lambda: defaultdict(int))
    errors = []
    original_sizes = set()
    visit_counts = defaultdict(int)

    for split, cls, subject_id, visit, nii_file in tqdm(tasks, desc=f"Converting {slice_type} slices to PNG"):
        
        try:
            # Extract coordinate position from filename
            coord_pos = extract_coordinate_position(nii_file.name, pattern, coord_name)
            
            # Load the NIfTI file (already a single 2D slice)
            slice_img = nib.load(str(nii_file))
            slice_data = slice_img.get_fdata()
            # Remove singleton dimensions to get 2D slice
            slice_2d = np.squeeze(slice_data)
            
            # Normalize slice
            slice_normalized = normalize_image(slice_2d, INTENSITY_PERCENTILE)

            # Convert to PIL Image (transpose for correct orientation)
            img = Image.fromarray(slice_normalized.T, mode='L')
            
            # Track original size for reporting
            original_sizes.add(f"{img.size[0]}×{img.size[1]}")
            
            # Resize to target resolution using high-quality interpolation
            img_resized = resize_to_target(img, TARGET_SIZE, INTERPOLATION_METHOD)

            # Create output filename: {subject_id}_{visit}_{slice_type}_{coord}{position}.png
            out_name = f"{subject_id}_{visit}_{slice_type}_{coord_name}{coord_pos}.png"
            out_path = OUT_ROOT / slice_type / split / cls / out_name

            # Save PNG
            img_resized.save(out_path)
            stats[cls]["saved"] += 1
            visit_counts[visit] += 1
            
        except ValueError as e:
            stats[cls]["coord_parse_error"] += 1
            errors.append(f"Subject {subject_id}: {e}")
            continue
            
        except Exception as e:
            stats[cls]["load_error"] += 1
            errors.append(f"Subject {subject_id}: Error processing {nii_file.name}: {e}")
            continue

    # Count unique subjects processed
    for split in ("train", "val", "test"):
        # for cls in ("AD", "MCI", "CN"):
        for cls in ("AD", "CN"):
            output_dir = OUT_ROOT / slice_type / split / cls
            if output_dir.exists():
                png_files = list(output_dir.glob("*.png"))
                # Count unique subjects from PNG filenames
                subjects = set()
                for png_file in png_files:
                    subj_id = extract_subject_id_from_filename(png_file.name)
                    if subj_id:
                        subjects.add(subj_id)
                stats[cls]["subjects"] += len(subjects)

    # ----------------- REPORT FOR THIS SLICE TYPE ------------------
    print(f"\n{slice_type.upper()} CONVERSION COMPLETE")
    print("-" * 50)
    
    print(f"Original image sizes found: {sorted(original_sizes)}")
    print(f"All images resized to: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    
    print(f"\nVisit distribution for {slice_type}:")
    for visit in REQUIRED_VISITS:
        count = visit_counts[visit]
        print(f"  {visit}: {count} slices")

    print(f"\nResults by diagnostic group:")
    total_saved = 0
    total_subjects = 0

    # for grp in ("AD", "MCI", "CN"):
    for grp in ("AD", "CN"):
        s = stats[grp]
        subjects = s['subjects']
        saved = s['saved']
        total_saved += saved
        total_subjects += subjects

        print(f"\n{grp}:")
        print(f"  Subjects processed: {subjects}")
        print(f"  Slices converted: {saved}")
        if subjects > 0:
            print(f"  Average slices/subject: {saved/subjects:.1f}")

        # Report errors for this group
        if s['coord_parse_error'] > 0:
            print(f"  Coordinate parse errors: {s['coord_parse_error']}")
        if s['load_error'] > 0:
            print(f"  Load errors: {s['load_error']}")

    # Summary for this slice type
    print(f"\n{slice_type.title()} Summary:")
    print(f"  Total subjects processed: {total_subjects}")
    print(f"  Total slices converted: {total_saved}")
    if total_subjects > 0:
        print(f"  Average slices per subject: {total_saved/total_subjects:.1f}")

    # Show first few errors if any
    if errors:
        print(f"\nFirst few {slice_type} errors (total: {len(errors)}):")
        for err in errors[:3]:
            print(f"  - {err}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more errors")

# ----------------- FINAL REPORT --------------------------------
print(f"\n{'='*70}")
print("OVERALL SEQUENTIAL 2D CONVERSION SUMMARY")
print(f"{'='*70}")

print(f"\nConversion Parameters:")
print(f"  Target resolution: {TARGET_SIZE[0]}×{TARGET_SIZE[1]} pixels")
print(f"  Interpolation method: {INTERPOLATION_METHOD}")
print(f"  Intensity normalization: {INTENSITY_PERCENTILE[0]}-{INTENSITY_PERCENTILE[1]} percentile")
print(f"  Expected visits per subject: {REQUIRED_VISITS}")

# Output directory contents by slice type
print(f"\nOutput directory contents:")
grand_total = 0

for slice_type in SLICE_TYPES.keys():
    print(f"\n{slice_type.upper()}:")
    slice_total = 0
    
    for split in ("train", "val", "test"):
        split_total = 0
        print(f"  {split}:")
        for grp in ("AD", "CN"):
            png_dir = OUT_ROOT / slice_type / split / grp
            if png_dir.exists():
                n_png = len(list(png_dir.glob("*.png")))
                split_total += n_png
                print(f"    {grp}: {n_png} PNG files")
            else:
                print(f"    {grp}: directory not found")
        print(f"  Split total: {split_total} PNG files")
        slice_total += split_total
    
    print(f"  {slice_type.title()} total: {slice_total} PNG files")
    grand_total += slice_total

print(f"\n📊 GRAND TOTAL: {grand_total} PNG files")

# Show sample output files
print(f"\n📄 Sample Output Files:")
sample_files = list(OUT_ROOT.rglob("*.png"))[:9]
for i, sample_file in enumerate(sample_files):
    rel_path = sample_file.relative_to(OUT_ROOT)
    print(f"  {i+1}. {rel_path}")

print(f"\n🎯 SEQUENTIAL TEMPORAL DATA CHARACTERISTICS:")
print(f"   • Each PNG represents one temporal slice (sc, m06, or m12)")
print(f"   • Filename format: {{subject_id}}_{{visit}}_{{slice_type}}_{{coord}}{{position}}.png")
print(f"   • Variable reference: Each visit uses its optimal slice position")
print(f"   • Ready for CNN+LSTM temporal sequence modeling")
print(f"   • Each subject contributes 3 timepoints × 3 slice types = 9 images")

print(f"\n✅ Sequential temporal 2D conversion complete!")
print(f"🧠 Each image maintains temporal and anatomical information")
print(f"🎯 Ready for CNN+LSTM sequential learning!")
print("=" * 70)
```

## Image Processing

#### Center Crop


```python
# =====================================================================
# CENTER CROP
# =====================================================================

# ----------------- PATHS ---------------------------------------
INPUT_ROOT  = Path("../datasets/ADNI_1_5_T/6_2Dconverted_sequential") 
OUTPUT_ROOT = Path("../datasets/ADNI_1_5_T/7_cropped")     

# Define slice types and splits
slice_types = ["axial", "coronal", "sagittal"]
splits = ["train", "val", "test"]  
# classes = ["AD", "MCI", "CN"]
classes = ["AD", "CN"]

# Create output directories
for slice_type in slice_types:
    for split in splits:
        for cls in classes:
            (OUTPUT_ROOT / slice_type / split / cls).mkdir(parents=True, exist_ok=True)

# ----------------- PARAMETERS ----------------------------------
CROP_PADDING = 5
TARGET_SIZE = (256, 256)
ROTATION_ANGLE = -90

# ----------------- GATHER PNG FILES ----------------------------
png_paths = list(INPUT_ROOT.rglob("*.png"))
print(f"Found {len(png_paths)} PNG files to process")

# Show distribution by slice type, split and class
print("\nInput distribution:\n")
total_files = 0

for slice_type in slice_types:
    print(f"**{slice_type.upper()} SLICES:**")
    slice_total = 0
    
    for split in splits:
        split_total = 0
        for cls in classes:
            class_dir = INPUT_ROOT / slice_type / split / cls
            if class_dir.exists():
                class_pngs = list(class_dir.glob("*.png"))
                print(f"  {split}/{cls}: {len(class_pngs)} files")
                split_total += len(class_pngs)
            else:
                print(f"  {split}/{cls}: 0 files (directory not found)")
        
        if split_total > 0:
            print(f"  {split} total: {split_total} files")
        slice_total += split_total
    
    print(f"  {slice_type} total: {slice_total} files\n")
    total_files += slice_total

print(f"**GRAND TOTAL: {total_files} files**\n")

# ----------------- PROCESSING ----------------------------------
processed_count = 0
skipped_count = 0

for in_path in tqdm(png_paths, desc="Cropping, resizing & rotating"):
    # compute relative path, mirror directory tree
    rel = in_path.relative_to(INPUT_ROOT)
    out_path = OUTPUT_ROOT / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if output already exists
    if out_path.exists():
        skipped_count += 1
        continue

    # load as grayscale
    img = Image.open(in_path).convert("L")
    arr = np.array(img)

    # find bright (brain) pixels
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        # nothing to crop: just resize, rotate, save
        img.resize(TARGET_SIZE, Image.BILINEAR) \
           .rotate(ROTATION_ANGLE, expand=True) \
           .save(out_path)
        processed_count += 1
        continue

    # bounding box of the brain
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # add padding
    y0 = max(0, y0 - CROP_PADDING)
    x0 = max(0, x0 - CROP_PADDING)
    y1 = min(arr.shape[0], y1 + CROP_PADDING)
    x1 = min(arr.shape[1], x1 + CROP_PADDING)

    # crop to brain + padding, then resize and rotate
    cropped = img.crop((x0, y0, x1, y1))
    resized = cropped.resize(TARGET_SIZE, Image.BILINEAR)
    rotated = resized.rotate(ROTATION_ANGLE, expand=True)

    # save
    rotated.save(out_path)
    processed_count += 1

# ----------------- FINAL REPORT --------------------------------
print(f"\n{'='*60}")
print("CROPPING COMPLETE")
print(f"{'='*60}")
print(f"Files processed: {processed_count}")
print(f"Files skipped (already exist): {skipped_count}")
print(f"Total files handled: {processed_count + skipped_count}")

# Show output distribution
print(f"\nOutput distribution:")
output_total = 0

for slice_type in slice_types:
    print(f"\n**{slice_type.upper()} SLICES:**")
    slice_total = 0
    
    for split in splits:
        split_total = 0
        for cls in classes:
            class_dir = OUTPUT_ROOT / slice_type / split / cls
            if class_dir.exists():
                class_pngs = list(class_dir.glob("*.png"))
                print(f"  {split}/{cls}: {len(class_pngs)} files")
                split_total += len(class_pngs)
            else:
                print(f"  {split}/{cls}: 0 files")
        
        print(f"  {split} total: {split_total} files")
        slice_total += split_total
    
    print(f"  {slice_type} total: {slice_total} files")
    output_total += slice_total

print(f"\n**OUTPUT GRAND TOTAL: {output_total} files**")
print(f"✅ Cropping, resizing & rotation complete. Output under: {OUTPUT_ROOT}")
```

#### [QC] Visualize Sample Data


```python
# =====================================================================
# SEQUENTIAL TEMPORAL CROPPED DATA VISUALIZATION
# Shows temporal progression (sc, m06, m12) for selected subjects
# =====================================================================

# ----------------- CONFIG --------------------------------------
random.seed(42)
np.random.seed(42)

# CLASSES = ['CN', 'MCI', 'AD']
CLASSES = ['CN', 'AD']
SPLITS = ['train', 'val', 'test']
SLICE_TYPES = ['axial', 'coronal', 'sagittal']
VISITS = ['sc', 'm06', 'm12']  # Temporal sequence

CLASS_NAMES = {
    'CN': 'Cognitively Normal',
    # 'MCI': 'Mild Cognitive Impairment',
    'AD': "Alzheimer's Disease"
}

CLASS_COLORS = {
    'CN': '#2E8B57',    # Sea Green
    # 'MCI': '#FF8C00',   # Dark Orange
    'AD': '#DC143C'     # Crimson
}

VISIT_COLORS = {
    'sc': '#2E8B57',    # Dark green for baseline
    'm06': '#4682B4',   # Steel blue for 6 months
    'm12': '#8B0000'    # Dark red for 12 months
}

SLICE_TYPE_NAMES = {
    'axial': 'Axial (Top-Bottom)',
    'coronal': 'Coronal (Front-Back)', 
    'sagittal': 'Sagittal (Left-Right)'
}

# ----------------- PATHS ---------------------------------------
OUTPUT_ROOT = Path("../datasets/ADNI_1_5_T/7_cropped")

print("🧠 SEQUENTIAL TEMPORAL CROPPED DATA VISUALIZATION")
print("=" * 80)
print(f"📂 Dataset Root: {OUTPUT_ROOT}")
print(f"📅 Temporal Sequence: {' → '.join(VISITS)}")
print(f"🎯 Objective: Show temporal disease progression across visits")
print()

# ----------------- HELPER FUNCTIONS ----------------------------
def extract_subject_id_from_filename(filename):
    """Extract subject ID from temporal filename"""
    # Format: {subject_id}_{visit}_{slice_type}_{coord}{position}.png
    # Example: 002_S_0295_sc_axial_x123.png → 002_S_0295
    parts = filename.split('_')
    if len(parts) >= 3 and parts[1] == 'S':
        return f"{parts[0]}_S_{parts[2]}"
    return None

def extract_visit_from_filename(filename):
    """Extract visit from temporal filename"""
    for visit in VISITS:
        if f"_{visit}_" in filename:
            return visit
    return None

def get_subject_temporal_files(class_dir, slice_type):
    """Get temporal files for subjects with complete sequences"""
    if not class_dir.exists():
        return {}
    
    png_files = list(class_dir.glob("*.png"))
    if not png_files:
        return {}
    
    # Group files by subject
    subject_files = {}
    for png_path in png_files:
        filename = png_path.name
        subject_id = extract_subject_id_from_filename(filename)
        visit = extract_visit_from_filename(filename)
        
        if subject_id and visit and slice_type in filename:
            if subject_id not in subject_files:
                subject_files[subject_id] = {}
            subject_files[subject_id][visit] = png_path
    
    # Filter to subjects with complete temporal sequences
    complete_subjects = {}
    for subject_id, visit_files in subject_files.items():
        if set(visit_files.keys()) == set(VISITS):
            complete_subjects[subject_id] = visit_files
    
    return complete_subjects

def select_representative_subject(class_dir, slice_type):
    """Select one subject with complete temporal sequence"""
    complete_subjects = get_subject_temporal_files(class_dir, slice_type)
    
    if not complete_subjects:
        return None, {}
    
    # Randomly select one subject
    subject_id = random.choice(list(complete_subjects.keys()))
    return subject_id, complete_subjects[subject_id]

# ----------------- TEMPORAL PROGRESSION VISUALIZATION ----------
print("Creating temporal progression visualization...")

# Create comprehensive temporal visualization
fig, axes = plt.subplots(len(SLICE_TYPES), len(CLASSES) * len(VISITS), 
                        figsize=(4 * len(CLASSES) * len(VISITS), 4 * len(SLICE_TYPES)))

if len(SLICE_TYPES) == 1:
    axes = axes.reshape(1, -1)
if len(CLASSES) * len(VISITS) == 1:
    axes = axes.reshape(-1, 1)

fig.suptitle('Temporal Disease Progression Across Brain Views\n'
             'Each Row: Different Brain View | Each Column Group: Disease Class Progression',
             fontsize=16, fontweight='bold', y=1)

# Collect subjects and their temporal data
all_subjects_data = {}
temporal_stats = {}

for slice_idx, slice_type in enumerate(SLICE_TYPES):
    print(f"\n📊 Processing {slice_type.upper()} slices:")
    all_subjects_data[slice_type] = {}
    temporal_stats[slice_type] = {}
    
    for class_idx, class_name in enumerate(CLASSES):
        # Use train data for visualization
        class_dir = OUTPUT_ROOT / slice_type / 'train' / class_name
        
        # Select representative subject with complete temporal sequence
        subject_id, temporal_files = select_representative_subject(class_dir, slice_type)
        all_subjects_data[slice_type][class_name] = (subject_id, temporal_files)
        
        if subject_id and temporal_files:
            print(f"  {class_name}: Subject {subject_id} ({len(temporal_files)}/{len(VISITS)} visits)")
            temporal_stats[slice_type][class_name] = len(temporal_files)
        else:
            print(f"  {class_name}: No complete subjects found")
            temporal_stats[slice_type][class_name] = 0
        
        # Plot temporal sequence for this class
        for visit_idx, visit in enumerate(VISITS):
            col = class_idx * len(VISITS) + visit_idx
            ax = axes[slice_idx, col]
            
            if subject_id and visit in temporal_files:
                try:
                    # Load and display the image
                    img_path = temporal_files[visit]
                    img = np.array(Image.open(img_path))
                    
                    im = ax.imshow(img, cmap='gray', aspect='equal')
                    
                    # # Title with visit and time progression
                    # ax.set_title(f'{visit.upper()}\n{subject_id}',
                    #            fontsize=10, pad=8,
                    #            color=VISIT_COLORS[visit], fontweight='bold')
                    
                    # Add intensity range info
                    ax.text(0.02, 0.98, f'[{img.min():.0f}, {img.max():.0f}]',
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    
                    ax.axis('off')
                    
                    # Add colorbar for first visit of each class
                    # if visit_idx == 0:
                    #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                except Exception as e:
                    ax.text(0.5, 0.5, f'{visit.upper()}\nERROR\n{str(e)[:15]}...',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='red')
                    ax.set_title(f'{visit.upper()}\nError',
                               fontsize=10, color='red')
                    ax.axis('off')
            else:
                # Missing data
                ax.text(0.5, 0.5, f'{visit.upper()}\nMISSING',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray', alpha=0.7)
                ax.set_title(f'{visit.upper()}\nNo Data',
                           fontsize=10, color='gray')
                ax.axis('off')
    
    # Add slice type label on the left
    if slice_idx < len(axes):
        axes[slice_idx, 0].text(-0.15, 0.5, f'{SLICE_TYPE_NAMES[slice_type]}',
                              transform=axes[slice_idx, 0].transAxes,
                              fontsize=14, fontweight='bold',
                              rotation=90, va='center', ha='center')

# Add class and visit labels at the top
for class_idx, class_name in enumerate(CLASSES):
    for visit_idx, visit in enumerate(VISITS):
        col = class_idx * len(VISITS) + visit_idx
        if col < axes.shape[1]:
            # Class label (spanning all visits for this class)
            if visit_idx == len(VISITS) // 2:  # Middle visit for class label
                axes[0, col].text(0.5, 1.2, f'{CLASS_NAMES[class_name]}\n({class_name})',
                                 transform=axes[0, col].transAxes,
                                 fontsize=12, fontweight='bold',
                                 ha='center', va='center',
                                 color=CLASS_COLORS[class_name],
                                 bbox=dict(boxstyle='round,pad=0.5', 
                                         facecolor='lightgray', alpha=0.5))
            
            # Visit label
            axes[0, col].text(0.5, 1.05, visit.upper(),
                            transform=axes[0, col].transAxes,
                            fontsize=11, fontweight='bold',
                            color=VISIT_COLORS[visit],
                            ha='center', va='center')

plt.tight_layout()
plt.subplots_adjust(top=0.88, left=0.08, hspace=0.3, wspace=0.1)
plt.show()


# ----------------- TEMPORAL COMPLETENESS SUMMARY ---------------
print(f"\n🎯 TEMPORAL SEQUENCE COMPLETENESS SUMMARY:")
print(f"Expected visits per subject: {VISITS}")
print(f"Complete sequence = all {len(VISITS)} visits present")

total_complete_subjects = 0
for slice_type in SLICE_TYPES:
    slice_complete = 0
    for class_name in CLASSES:
        for split in SPLITS:
            class_dir = OUTPUT_ROOT / slice_type / split / class_name
            complete_subjects = get_subject_temporal_files(class_dir, slice_type)
            slice_complete += len(complete_subjects)
    
    total_complete_subjects += slice_complete
    print(f"  {slice_type.capitalize()}: {slice_complete} subjects with complete temporal sequences")

print(f"\nOverall: {total_complete_subjects} complete temporal sequences across all slice types")

print(f"\n✅ Temporal progression visualization complete!")
print(f"🧠 Shows disease progression across time for selected subjects")
print(f"🎯 Each row represents different brain view, each column group shows class progression")
print("=" * 80)
```

#### Image Enhancement


```python
# =====================================================================
# SEQUENTIAL TEMPORAL GWO BRAIN MRI ENHANCEMENT
# Modified for temporal sequence data with mask-aware processing
# =====================================================================

# ------------------------------------------------------------------
# 1.  SIMPLE BRAIN–BACKGROUND SEPARATION
# ------------------------------------------------------------------

def get_brain_mask(image: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Return boolean mask: True = brain, False = background (≈0).

    *image* must be a float array in [0,1].  Pixels ≤ *threshold* are
    considered background.  Keep *threshold* tiny so dark but non‑zero
    brain tissue is not lost.
    """
    return image > threshold

# ------------------------------------------------------------------
# 2.  GREY WOLF OPTIMIZER (unchanged)
# ------------------------------------------------------------------

class GreyWolfOptimizer:
    """Grey Wolf Optimization algorithm for parameter optimisation"""

    def __init__(self, objective_func, bounds, num_wolves=20, max_iterations=50,
                 convergence_threshold=1e-6):
        self.objective_func = objective_func
        self.bounds = np.array(bounds, dtype=np.float32)
        self.num_wolves = num_wolves
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.dim = len(bounds)

        # Initialise wolves uniformly in bounds
        self.wolves = np.zeros((self.num_wolves, self.dim), dtype=np.float32)
        for d in range(self.dim):
            self.wolves[:, d] = np.random.uniform(self.bounds[d, 0],
                                                 self.bounds[d, 1],
                                                 self.num_wolves)
        self.fitness = np.full(self.num_wolves, -np.inf, dtype=np.float32)

        # Leader positions/scores
        self.alpha_pos = np.zeros(self.dim, dtype=np.float32)
        self.beta_pos  = np.zeros(self.dim, dtype=np.float32)
        self.delta_pos = np.zeros(self.dim, dtype=np.float32)
        self.alpha_score = self.beta_score = self.delta_score = -np.inf

        self.convergence_curve = []

    # --------------------------------------------------------------
    def _clip(self):
        for d in range(self.dim):
            self.wolves[:, d] = np.clip(self.wolves[:, d],
                                        self.bounds[d, 0],
                                        self.bounds[d, 1])

    # --------------------------------------------------------------
    def optimize(self, verbose: bool = True):
        for it in range(self.max_iterations):
            # Evaluate fitness
            for i in range(self.num_wolves):
                self.fitness[i] = self.objective_func(self.wolves[i])

            # Update alpha, beta, delta
            for i in range(self.num_wolves):
                fit = self.fitness[i]
                if fit > self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score,  self.beta_pos  = self.alpha_score, self.alpha_pos.copy()
                    self.alpha_score, self.alpha_pos = fit, self.wolves[i].copy()
                elif fit > self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score,  self.beta_pos  = fit, self.wolves[i].copy()
                elif fit > self.delta_score:
                    self.delta_score, self.delta_pos = fit, self.wolves[i].copy()

            # Linearly decreasing a from 2→0
            a = 2.0 - (2.0 * it) / self.max_iterations

            # Update positions
            for i in range(self.num_wolves):
                for d in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[d] - self.wolves[i, d])
                    X1 = self.alpha_pos[d] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[d] - self.wolves[i, d])
                    X2 = self.beta_pos[d] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[d] - self.wolves[i, d])
                    X3 = self.delta_pos[d] - A3 * D_delta

                    self.wolves[i, d] = (X1 + X2 + X3) / 3.0

            self._clip()
            self.convergence_curve.append(self.alpha_score)

            # Early stop on plateau
            if it > 10:
                recent = self.convergence_curve[-10:]
                if max(recent) - min(recent) < self.convergence_threshold:
                    if verbose:
                        print(f"Converged at iter {it}")
                    break

            if verbose and it % 10 == 0:
                print(f"Iter {it:02d}  best fitness = {self.alpha_score:.6f}")

        return self.alpha_pos, self.alpha_score

# ------------------------------------------------------------------
# 3.  IMAGE ENHANCEMENT OPERATORS (unchanged from original)
# ------------------------------------------------------------------

class ImageEnhancer:
    """Collection of basic enhancement operators"""

    @staticmethod
    def clahe_enhancement(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        out = equalize_adapthist(img, kernel_size=tile_grid_size,
                                 clip_limit=clip_limit)
        return (out * 255).astype(np.uint8)

    @staticmethod
    def gabor_enhancement(img, wavelength=10, theta=0, sigma_x=5, sigma_y=5, gamma=0.5):
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        ksize = int(6 * max(sigma_x, sigma_y)) | 1  # ensure odd
        kernel = cv2.getGaborKernel((ksize, ksize), sigma_x, np.deg2rad(theta),
                                    2*np.pi/wavelength, gamma, 0, ktype=cv2.CV_32F)
        resp = cv2.filter2D(img, cv2.CV_32F, kernel)
        out = np.clip(img + 0.3 * resp, 0, 1)
        return (out * 255).astype(np.uint8)

    @staticmethod
    def unsharp_masking(img, sigma=1.0, strength=1.5):
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        blurred = ndimage.gaussian_filter(img, sigma=sigma)
        mask = img - blurred
        out = np.clip(img + strength * mask, 0, 1)
        return (out * 255).astype(np.uint8)

    @staticmethod
    def adaptive_enhancement(img, clahe_clip=2.0, gabor_strength=0.3,
                             unsharp_sigma=1.0, unsharp_strength=1.5):
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        # CLAHE
        out = equalize_adapthist(img, clip_limit=clahe_clip)
        # Gabor
        if gabor_strength > 0:
            kernel = cv2.getGaborKernel((15, 15), 3, 0, 10, 0.5, 0,
                                        ktype=cv2.CV_32F)
            resp = cv2.filter2D(out, cv2.CV_32F, kernel)
            out = out + gabor_strength * resp
        # Unsharp
        blur = ndimage.gaussian_filter(out, sigma=unsharp_sigma)
        out = np.clip(out + unsharp_strength * (out - blur), 0, 1)
        return (out * 255).astype(np.uint8)

# ------------------------------------------------------------------
# 4.  MASK‑AWARE QUALITY METRICS
# ------------------------------------------------------------------

class ImageQualityMetrics:
    @staticmethod
    def _prep(img):
        return img.astype(np.float32) / 255.0 if img.max() > 1.0 else img

    @staticmethod
    def calculate_entropy(img, mask=None):
        img = ImageQualityMetrics._prep(img)
        if mask is not None:
            img = img[mask]
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 1))
        if hist.sum() == 0:
            return 0.0
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    @staticmethod
    def calculate_edge_energy(img, mask=None):
        img = ImageQualityMetrics._prep(img)
        sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        return float(np.mean(mag[mask])) if mask is not None else float(np.mean(mag))

    @staticmethod
    def calculate_local_contrast(img, mask=None):
        img = ImageQualityMetrics._prep(img)
        k = np.ones((9, 9), dtype=np.float32) / 81.0
        mu = cv2.filter2D(img, -1, k)
        var = cv2.filter2D(img**2, -1, k) - mu**2
        std = np.sqrt(np.clip(var, 0, None))
        return float(np.mean(std[mask])) if mask is not None else float(np.mean(std))

# ------------------------------------------------------------------
# 5.  FITNESS FUNCTION (mask‑aware)
# ------------------------------------------------------------------

def create_fitness_function(orig_img, brain_mask, method='adaptive'):
    def fitness(params):
        try:
            # ------------------------------------------------------
            if method == 'adaptive':
                out = ImageEnhancer.adaptive_enhancement(orig_img, *params)
            elif method == 'clahe':
                clip, tile = params
                out = ImageEnhancer.clahe_enhancement(orig_img, clip, (int(tile), int(tile)))
            elif method == 'gabor':
                out = ImageEnhancer.gabor_enhancement(orig_img, *params)
            elif method == 'unsharp':
                out = ImageEnhancer.unsharp_masking(orig_img, *params)
            else:
                raise ValueError('Unknown method')

            out[~brain_mask] = 0  # background stays black

            ent = ImageQualityMetrics.calculate_entropy(out, brain_mask)
            edge = ImageQualityMetrics.calculate_edge_energy(out, brain_mask)
            ctr  = ImageQualityMetrics.calculate_local_contrast(out, brain_mask)
            return 0.4*ent + 0.4*edge*100 + 0.2*ctr*100
        except Exception:
            return -1000.0
    return fitness

# ------------------------------------------------------------------
# 6.  APPLY BEST ENHANCEMENT (mask‑aware)
# ------------------------------------------------------------------

def apply_best_enhancement(orig_img, brain_mask, method, params):
    if method == 'adaptive':
        out = ImageEnhancer.adaptive_enhancement(orig_img, *params)
    elif method == 'clahe':
        clip, tile = params
        out = ImageEnhancer.clahe_enhancement(orig_img, clip, (int(tile), int(tile)))
    elif method == 'gabor':
        out = ImageEnhancer.gabor_enhancement(orig_img, *params)
    elif method == 'unsharp':
        out = ImageEnhancer.unsharp_masking(orig_img, *params)
    else:
        raise ValueError('Unknown method')
    out[~brain_mask] = 0
    return out

# ------------------------------------------------------------------
# 7.  SINGLE‑IMAGE PIPELINE
# ------------------------------------------------------------------

def enhance_single_image(img_path, out_path, method='adaptive',
                         gwo_iters=30, num_wolves=15, verbose=False):
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f'Cannot load {img_path}')
        img = img.astype(np.float32) / 255.0
        mask = get_brain_mask(img)

        # Bounds
        if method == 'adaptive':
            bounds = [(1,4), (0,0.5), (0.5,2), (1,3)]
        elif method == 'clahe':
            bounds = [(1,4), (4,16)]
        elif method == 'gabor':
            bounds = [(5,20), (0,180), (2,8), (2,8)]
        elif method == 'unsharp':
            bounds = [(0.5,2), (1,3)]
        else:
            raise ValueError('Unknown method')

        fit_func = create_fitness_function(img, mask, method)
        gwo = GreyWolfOptimizer(fit_func, bounds, num_wolves, gwo_iters)
        best_params, best_fitness = gwo.optimize(verbose=verbose)

        enhanced = apply_best_enhancement(img, mask, method, best_params)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), enhanced)

        return dict(success=True, best_params=best_params, best_fitness=best_fitness,
                    input_path=str(img_path), output_path=str(out_path))
    except Exception as e:
        return dict(success=False, error=str(e), input_path=str(img_path),
                    output_path=str(out_path))

# ------------------------------------------------------------------
# 8.  SEQUENTIAL TEMPORAL DATASET PROCESSOR
# ------------------------------------------------------------------

def process_sequential_dataset(input_root, output_root, method='adaptive', max_workers=4,
                              gwo_iters=20, num_wolves=10, max_images_per_class=None):
    """
    Process sequential temporal dataset with temporal structure preservation
    """
    input_root, output_root = Path(input_root), Path(output_root)

    slice_types = ['axial', 'coronal', 'sagittal']
    splits = ['train', 'val', 'test']
    # classes = ['AD', 'MCI', 'CN']
    classes = ['AD', 'CN']
    visits = ['sc', 'm06', 'm12']

    print("🧠 SEQUENTIAL TEMPORAL GWO ENHANCEMENT")
    print("=" * 60)
    print(f"📂 Input: {input_root}")
    print(f"📁 Output: {output_root}")
    print(f"🔧 Method: {method}")
    print(f"📅 Expected visits: {visits}")
    print()

    # Gather tasks and statistics
    tasks = []
    temporal_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    subject_stats = defaultdict(lambda: defaultdict(int))
    
    for slice_t in slice_types:
        for split in splits:
            for cls in classes:
                in_dir = input_root / slice_t / split / cls
                out_dir = output_root / slice_t / split / cls
                if not in_dir.exists():
                    continue
                
                pngs = list(in_dir.glob('*.png'))
                
                # Count subjects and visits
                subjects = set()
                visit_counts = {visit: 0 for visit in visits}
                
                for png in pngs:
                    # Extract subject ID and visit from filename
                    # Format: {subject_id}_{visit}_{slice_type}_{coord}{position}.png
                    filename = png.name
                    parts = filename.split('_')
                    
                    if len(parts) >= 3 and parts[1] == 'S':
                        subject_id = f"{parts[0]}_S_{parts[2]}"
                        subjects.add(subject_id)
                        
                        # Extract visit
                        for visit in visits:
                            if f"_{visit}_" in filename:
                                visit_counts[visit] += 1
                                temporal_stats[slice_t][f'{split}_{cls}'][visit] += 1
                                break
                
                subject_stats[slice_t][f'{split}_{cls}'] = len(subjects)
                
                # Apply sampling if requested
                if max_images_per_class and len(pngs) > max_images_per_class:
                    pngs = random.sample(pngs, max_images_per_class)
                
                # Add to processing tasks
                for p in pngs:
                    tasks.append((p, out_dir / p.name))

    print(f"📊 TEMPORAL DATASET STATISTICS:")
    total_tasks = len(tasks)
    print(f"   • Total images to enhance: {total_tasks}")
    
    # Show subject and visit distribution
    for slice_t in slice_types:
        print(f"\n   {slice_t.upper()} slice type:")
        for split in splits:
            for cls in classes:
                key = f'{split}_{cls}'
                subjects = subject_stats[slice_t].get(key, 0)
                if subjects > 0:
                    visit_dist = temporal_stats[slice_t][key]
                    visit_str = ", ".join([f"{v}:{visit_dist[v]}" for v in visits])
                    print(f"     {split}/{cls}: {subjects} subjects, visits({visit_str})")

    if total_tasks == 0:
        print("❌ No images found to process!")
        return []

    # Worker function
    def worker(task):
        return enhance_single_image(*task, method, gwo_iters, num_wolves, False)

    # Parallel processing
    print(f"\n🚀 Starting enhancement with {max_workers} workers...")
    results, ok, fail = [], 0, 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in tqdm(as_completed({ex.submit(worker, t): t for t in tasks}),
                        total=len(tasks), desc=f'Enhancing {method}'):
            r = fut.result()
            results.append(r)
            ok += r['success']
            fail += (not r['success'])

    # Results summary
    print(f"\n📈 ENHANCEMENT RESULTS:")
    print(f"   ✅ Success: {ok}")
    print(f"   ❌ Failed: {fail}")
    print(f"   📊 Success rate: {(ok/total_tasks*100):.1f}%")

    # Verify temporal structure preservation
    print(f"\n🔍 TEMPORAL STRUCTURE VERIFICATION:")
    for slice_t in slice_types:
        output_slice_dir = output_root / slice_t
        if output_slice_dir.exists():
            total_enhanced = len(list(output_slice_dir.rglob("*.png")))
            print(f"   {slice_t}: {total_enhanced} enhanced images")
            
            # Check visit distribution in output
            visit_counts = {visit: 0 for visit in visits}
            for png in output_slice_dir.rglob("*.png"):
                for visit in visits:
                    if f"_{visit}_" in png.name:
                        visit_counts[visit] += 1
                        break
            
            visit_str = ", ".join([f"{v}:{visit_counts[v]}" for v in visits])
            print(f"      Visit distribution: {visit_str}")

    print(f"\n✅ Sequential temporal enhancement complete!")
    return results
```


```python
# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------

print('🧠 Sequential Temporal Mask‑aware GWO Brain MRI Enhancement')
input_root  = '../datasets/ADNI_1_5_T/7_cropped'
output_root = '../datasets/ADNI_1_5_T/8_enhanced'

results = process_sequential_dataset(
    input_root, 
    output_root, 
    method='adaptive',
    max_workers=12, 
    gwo_iters=50,
    num_wolves=20,
    max_images_per_class=None  # Process all images
)
```

#### [QC] Visualize Enhanced Data


```python
# ------------------------------------------------------------------
# VISUALIZATION HELPER
# ------------------------------------------------------------------

def visualize_enhancement_comparison(input_root, output_root, num_samples=6):
    """Visualize enhancement comparison for temporal data"""
    input_root, output_root = Path(input_root), Path(output_root)
    samples = []
    
    visits = ['sc', 'm06', 'm12']
    
    # for cls in ['AD','MCI','CN']:
    for cls in ['AD', 'CN']:
        inp_dir = input_root / 'coronal' / 'train' / cls
        out_dir = output_root / 'coronal' / 'train' / cls
        
        if not inp_dir.exists() or not out_dir.exists():
            continue
            
        # Find files for different visits
        for visit in visits:
            inp_files = [f for f in inp_dir.glob('*.png') if f"_{visit}_" in f.name]
            if inp_files:
                inp_file = random.choice(inp_files)
                out_file = out_dir / inp_file.name
                if out_file.exists():
                    samples.append((inp_file, out_file, f"{cls}_{visit}"))
                    break
    
    if not samples:
        print('No enhancement samples to show')
        return
    
    samples = random.sample(samples, min(num_samples, len(samples)))

    fig, ax = plt.subplots(2, len(samples), figsize=(4*len(samples), 8))
    fig.suptitle('Sequential Temporal Enhancement Comparison', fontsize=16, weight='bold')
    
    for i,(orig_p,en_p,label) in enumerate(samples):
        o = cv2.imread(str(orig_p), cv2.IMREAD_GRAYSCALE)
        e = cv2.imread(str(en_p), cv2.IMREAD_GRAYSCALE)
        ax[0,i].imshow(o, cmap='gray'); ax[0,i].axis('off'); ax[0,i].set_title(f'Original\n{label}')
        ax[1,i].imshow(e, cmap='gray'); ax[1,i].axis('off'); ax[1,i].set_title('Enhanced')
    plt.tight_layout(); plt.show()
```


```python
# Show enhancement comparison
print("\n📊 Showing enhancement comparison samples...")
visualize_enhancement_comparison(input_root, output_root, num_samples=3)
```

#### [QC] Print Data Distribution


```python
# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
random.seed(42)

# ----------------- CONFIGURATION ---------------------------------------
FINAL_ROOT = Path("../datasets/ADNI_1_5_T/8_enhanced")
SLICE_TYPES = ['axial', 'coronal', 'sagittal']
SPLITS = ['train', 'val', 'test']
# CLASSES = ['AD', 'MCI', 'CN']
CLASSES = ['AD', 'CN']
VISITS = ['sc', 'm06', 'm12']  # Temporal sequence

CLASS_INFO = {
    'AD': {'name': "Alzheimer's Disease", 'color': '#DC143C'},
    # 'MCI': {'name': 'Mild Cognitive Impairment', 'color': '#FF8C00'},
    'CN': {'name': 'Cognitively Normal', 'color': '#2E8B57'}
}

SLICE_INFO = {
    'axial': {'name': 'Axial (Top-Bottom)', 'description': 'Horizontal slices'},
    'coronal': {'name': 'Coronal (Front-Back)', 'description': 'Frontal slices'},
    'sagittal': {'name': 'Sagittal (Left-Right)', 'description': 'Side slices'}
}

VISIT_INFO = {
    'sc': {'name': 'Screening', 'color': '#2E8B57', 'description': 'Baseline visit'},
    'm06': {'name': '6 Months', 'color': '#4682B4', 'description': '6-month follow-up'},
    'm12': {'name': '12 Months', 'color': '#8B0000', 'description': '12-month follow-up'}
}

print("🧠 SEQUENTIAL TEMPORAL DATASET DISTRIBUTION ANALYSIS")
print("=" * 80)

# ----------------- HELPER FUNCTIONS ------------------------------------
def extract_subject_id_from_filename(filename):
    """Extract subject ID from temporal filename"""
    # Format: {subject_id}_{visit}_{slice_type}_{coord}{position}.png
    # Example: 002_S_0295_sc_axial_x123.png → 002_S_0295
    parts = filename.split('_')
    if len(parts) >= 3 and parts[1] == 'S':
        return f"{parts[0]}_S_{parts[2]}"
    return None

def extract_visit_from_filename(filename):
    """Extract visit from temporal filename"""
    for visit in VISITS:
        if f"_{visit}_" in filename:
            return visit
    return None

def extract_slice_type_from_filename(filename):
    """Extract slice type from filename"""
    for slice_type in SLICE_TYPES:
        if f"_{slice_type}_" in filename:
            return slice_type
    return None

# ----------------- 1. COLLECT TEMPORAL STATISTICS ----------------------
def collect_temporal_statistics():
    """Collect detailed temporal statistics about the final dataset"""
    temporal_data = []
    subject_data = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))  # slice_type -> split -> class -> subjects
    visit_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))  # slice_type -> split -> class -> visit -> count
    
    for slice_type in SLICE_TYPES:
        for split in SPLITS:
            for cls in CLASSES:
                class_dir = FINAL_ROOT / slice_type / split / cls
                if class_dir.exists():
                    png_files = list(class_dir.glob("*.png"))
                    
                    for png_file in png_files:
                        filename = png_file.name
                        subject_id = extract_subject_id_from_filename(filename)
                        visit = extract_visit_from_filename(filename)
                        
                        if subject_id and visit:
                            # Track subjects
                            subject_data[slice_type][split][cls].add(subject_id)
                            
                            # Track visits
                            visit_data[slice_type][split][cls][visit] += 1
                            
                            # Collect detailed info
                            temporal_data.append({
                                'slice_type': slice_type,
                                'split': split,
                                'class': cls,
                                'subject_id': subject_id,
                                'visit': visit,
                                'filename': filename,
                                'path': str(png_file)
                            })
    
    return temporal_data, subject_data, visit_data

print("🔍 Collecting temporal statistics...")
temporal_data, subject_data, visit_data = collect_temporal_statistics()

# ----------------- 2. ANALYZE TEMPORAL COMPLETENESS -------------------
def analyze_temporal_completeness():
    """Analyze which subjects have complete temporal sequences"""
    subject_completeness = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    
    # Group by subject and analyze their visit coverage
    for item in temporal_data:
        slice_type = item['slice_type']
        split = item['split']
        cls = item['class']
        subject_id = item['subject_id']
        visit = item['visit']
        
        subject_completeness[slice_type][split][cls][subject_id].add(visit)
    
    # Calculate completeness statistics
    completeness_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'complete': 0, 'incomplete': 0, 'total': 0})))
    
    for slice_type in SLICE_TYPES:
        for split in SPLITS:
            for cls in CLASSES:
                for subject_id, visits in subject_completeness[slice_type][split][cls].items():
                    is_complete = len(visits) == len(VISITS)
                    completeness_stats[slice_type][split][cls]['total'] += 1
                    if is_complete:
                        completeness_stats[slice_type][split][cls]['complete'] += 1
                    else:
                        completeness_stats[slice_type][split][cls]['incomplete'] += 1
    
    return subject_completeness, completeness_stats

subject_completeness, completeness_stats = analyze_temporal_completeness()

# ----------------- 3. CREATE COMPREHENSIVE SUMMARY ---------------------
def create_temporal_summary():
    """Create comprehensive temporal summary"""
    summary_data = []
    
    for slice_type in SLICE_TYPES:
        for split in SPLITS:
            for cls in CLASSES:
                # Basic counts
                total_images = len([item for item in temporal_data 
                                  if item['slice_type'] == slice_type and 
                                     item['split'] == split and 
                                     item['class'] == cls])
                
                unique_subjects = len(subject_data[slice_type][split][cls])
                
                # Visit distribution
                visit_counts = {}
                for visit in VISITS:
                    visit_counts[visit] = visit_data[slice_type][split][cls][visit]
                
                # Temporal completeness
                complete_subjects = completeness_stats[slice_type][split][cls]['complete']
                incomplete_subjects = completeness_stats[slice_type][split][cls]['incomplete']
                
                summary_data.append({
                    'slice_type': slice_type,
                    'split': split,
                    'class': cls,
                    'total_images': total_images,
                    'unique_subjects': unique_subjects,
                    'complete_subjects': complete_subjects,
                    'incomplete_subjects': incomplete_subjects,
                    'completeness_rate': (complete_subjects / unique_subjects * 100) if unique_subjects > 0 else 0,
                    'avg_images_per_subject': total_images / unique_subjects if unique_subjects > 0 else 0,
                    'sc_visits': visit_counts.get('sc', 0),
                    'm06_visits': visit_counts.get('m06', 0),
                    'm12_visits': visit_counts.get('m12', 0)
                })
    
    return pd.DataFrame(summary_data)

df_summary = create_temporal_summary()

# ----------------- 4. PRINT DETAILED STATISTICS ------------------------
print(f"\n📊 TEMPORAL DATASET OVERVIEW")
print("-" * 80)

# Overall totals
total_images = df_summary['total_images'].sum()
total_subjects = df_summary['unique_subjects'].sum()
total_complete_subjects = df_summary['complete_subjects'].sum()
total_train_images = df_summary[df_summary['split'] == 'train']['total_images'].sum()
total_val_images = df_summary[df_summary['split'] == 'val']['total_images'].sum()
total_test_images = df_summary[df_summary['split'] == 'test']['total_images'].sum()

print(f"📁 Root Directory: {FINAL_ROOT}")
print(f"🖼️  Total Images: {total_images:,}")
print(f"👥 Total Unique Subjects: {total_subjects:,}")
print(f"✅ Complete Temporal Sequences: {total_complete_subjects:,}")
print(f"📊 Overall Completeness Rate: {(total_complete_subjects/total_subjects*100):.1f}%")
print(f"🏋️  Training Images: {total_train_images:,} ({total_train_images/total_images*100:.1f}%)")
print(f"🔬 Validation Images: {total_val_images:,} ({total_val_images/total_images*100:.1f}%)")
print(f"🧪 Testing Images: {total_test_images:,} ({total_test_images/total_images*100:.1f}%)")
print(f"📅 Expected Visits per Subject: {len(VISITS)} ({', '.join(VISITS)})")

# Detailed breakdown by slice type
for slice_type in SLICE_TYPES:
    print(f"\n{'='*80}")
    print(f"🔍 {SLICE_INFO[slice_type]['name'].upper()} - {SLICE_INFO[slice_type]['description']}")
    print(f"{'='*80}")
    
    slice_df = df_summary[df_summary['slice_type'] == slice_type]
    slice_total_images = slice_df['total_images'].sum()
    slice_total_subjects = slice_df['unique_subjects'].sum()
    
    for split in SPLITS:
        split_df = slice_df[slice_df['split'] == split]
        split_total_images = split_df['total_images'].sum()
        split_total_subjects = split_df['unique_subjects'].sum()
        
        print(f"\n📂 {split.upper()} Split ({split_total_images:,} images, {split_total_subjects:,} subjects):")
        print(f"{'Class':<6} {'Images':<8} {'Subjects':<9} {'Complete':<9} {'SC':<4} {'M06':<4} {'M12':<4} {'Comp%':<6}")
        print("-" * 70)
        
        for cls in CLASSES:
            row = split_df[split_df['class'] == cls].iloc[0]
            images = row['total_images']
            subjects = row['unique_subjects']
            complete = row['complete_subjects']
            sc_count = row['sc_visits']
            m06_count = row['m06_visits']
            m12_count = row['m12_visits']
            comp_rate = row['completeness_rate']
            
            print(f"{cls:<6} {images:<8} {subjects:<9} {complete:<9} {sc_count:<4} {m06_count:<4} {m12_count:<4} {comp_rate:<6.1f}%")
    
    print(f"\n🎯 {slice_type.capitalize()} Total: {slice_total_images:,} images from {slice_total_subjects:,} subjects")

# ----------------- 5. TEMPORAL COMPLETENESS ANALYSIS -------------------
print(f"\n📅 TEMPORAL COMPLETENESS DETAILED ANALYSIS")
print("-" * 80)

for slice_type in SLICE_TYPES:
    print(f"\n**{slice_type.upper()} SLICE TYPE:**")
    
    for split in SPLITS:
        print(f"\n  {split.upper()} Split:")
        
        for cls in CLASSES:
            stats = completeness_stats[slice_type][split][cls]
            total = stats['total']
            complete = stats['complete']
            incomplete = stats['incomplete']
            
            if total > 0:
                comp_rate = (complete / total * 100)
                print(f"    {cls}: {complete}/{total} complete ({comp_rate:.1f}%), {incomplete} incomplete")
                
                # Show incomplete subjects and their missing visits
                if incomplete > 0:
                    incomplete_subjects = []
                    for subject_id, visits in subject_completeness[slice_type][split][cls].items():
                        if len(visits) < len(VISITS):
                            missing = set(VISITS) - visits
                            incomplete_subjects.append(f"{subject_id}[missing: {','.join(sorted(missing))}]")
                    
                    if incomplete_subjects:
                        sample_size = min(3, len(incomplete_subjects))
                        print(f"      Incomplete examples: {', '.join(incomplete_subjects[:sample_size])}")
                        if len(incomplete_subjects) > sample_size:
                            print(f"      ... and {len(incomplete_subjects) - sample_size} more")
            else:
                print(f"    {cls}: No subjects found")

# ----------------- 6. VISUALIZATION ------------------------------------
print(f"\n📈 GENERATING TEMPORAL VISUALIZATIONS...")

fig = plt.figure(figsize=(20, 16))

# 6.1 Overall distribution by slice type and split
ax1 = plt.subplot(3, 4, 1)
slice_split_data = df_summary.groupby(['slice_type', 'split'])['total_images'].sum().unstack()
slice_split_data.plot(kind='bar', ax=ax1)
ax1.set_title('Images by Slice Type & Split', fontsize=12, weight='bold')
ax1.set_xlabel('Slice Type')
ax1.set_ylabel('Number of Images')
ax1.legend(title='Split')
ax1.tick_params(axis='x', rotation=45)

# 6.2 Subject distribution by class
ax2 = plt.subplot(3, 4, 2)
class_subjects = df_summary.groupby('class')['unique_subjects'].sum()
colors = [CLASS_INFO[cls]['color'] for cls in class_subjects.index]
wedges, texts, autotexts = ax2.pie(class_subjects.values, labels=class_subjects.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Subject Distribution by Class', fontsize=12, weight='bold')

# 6.3 Temporal completeness by slice type
ax3 = plt.subplot(3, 4, 3)
completeness_data = df_summary.groupby('slice_type')[['complete_subjects', 'incomplete_subjects']].sum()
completeness_data.plot(kind='bar', stacked=True, ax=ax3, color=['#2E8B57', '#DC143C'])
ax3.set_title('Temporal Completeness by Slice Type', fontsize=12, weight='bold')
ax3.set_xlabel('Slice Type')
ax3.set_ylabel('Number of Subjects')
ax3.legend(['Complete', 'Incomplete'])
ax3.tick_params(axis='x', rotation=45)

# 6.4 Visit distribution across all slice types
ax4 = plt.subplot(3, 4, 4)
visit_totals = {
    'sc': df_summary['sc_visits'].sum(),
    'm06': df_summary['m06_visits'].sum(),
    'm12': df_summary['m12_visits'].sum()
}
visit_colors = [VISIT_INFO[visit]['color'] for visit in visit_totals.keys()]
ax4.pie(visit_totals.values(), labels=[f"{v.upper()}\n({visit_totals[v]})" for v in visit_totals.keys()], 
        autopct='%1.1f%%', colors=visit_colors, startangle=90)
ax4.set_title('Visit Distribution', fontsize=12, weight='bold')

# 6.5 Class distribution by split (subjects)
ax5 = plt.subplot(3, 4, 5)
class_split_subjects = df_summary.groupby(['class', 'split'])['unique_subjects'].sum().unstack()
class_split_subjects.plot(kind='bar', ax=ax5)
ax5.set_title('Subjects by Class & Split', fontsize=12, weight='bold')
ax5.set_xlabel('Class')
ax5.set_ylabel('Number of Subjects')
ax5.legend(title='Split')
ax5.tick_params(axis='x', rotation=45)

# 6.6 Temporal completeness rates
ax6 = plt.subplot(3, 4, 6)
comp_rates = df_summary.groupby(['slice_type', 'class'])['completeness_rate'].mean().unstack()
comp_rates.plot(kind='bar', ax=ax6)
ax6.set_title('Temporal Completeness Rates', fontsize=12, weight='bold')
ax6.set_xlabel('Slice Type')
ax6.set_ylabel('Completeness Rate (%)')
ax6.legend(title='Class')
ax6.tick_params(axis='x', rotation=45)

# 6.7 Average images per subject
ax7 = plt.subplot(3, 4, 7)
avg_images = df_summary.groupby(['slice_type', 'class'])['avg_images_per_subject'].mean().unstack()
avg_images.plot(kind='bar', ax=ax7)
ax7.set_title('Average Images per Subject', fontsize=12, weight='bold')
ax7.set_xlabel('Slice Type')
ax7.set_ylabel('Images per Subject')
ax7.legend(title='Class')
ax7.tick_params(axis='x', rotation=45)

# 6.8 Split distribution (images)
ax8 = plt.subplot(3, 4, 8)
split_totals = df_summary.groupby('split')['total_images'].sum()
colors_split = ['#2E8B57', '#FF8C00', '#DC143C']  # train, validation, test
ax8.pie(split_totals.values, labels=split_totals.index, autopct='%1.1f%%', 
        colors=colors_split, startangle=90)
ax8.set_title('Split Distribution (Images)', fontsize=12, weight='bold')

# 6.9 Heatmap of subjects by slice type and class
ax9 = plt.subplot(3, 4, 9)
heatmap_data = df_summary.groupby(['slice_type', 'class'])['unique_subjects'].sum().unstack()
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax9)
ax9.set_title('Subject Count Heatmap', fontsize=12, weight='bold')

# 6.10 Temporal progression visualization
ax10 = plt.subplot(3, 4, 10)
visits_by_slice = df_summary.groupby('slice_type')[['sc_visits', 'm06_visits', 'm12_visits']].sum()
visits_by_slice.plot(kind='bar', ax=ax10, color=[VISIT_INFO[v]['color'] for v in ['sc', 'm06', 'm12']])
ax10.set_title('Temporal Visits by Slice Type', fontsize=12, weight='bold')
ax10.set_xlabel('Slice Type')
ax10.set_ylabel('Number of Visits')
ax10.legend(['Screening', '6 Months', '12 Months'])
ax10.tick_params(axis='x', rotation=45)

# 6.11 Class balance across splits
ax11 = plt.subplot(3, 4, 11)
balance_data = df_summary.groupby(['split', 'class'])['unique_subjects'].sum().unstack()
balance_data.plot(kind='bar', ax=ax11, color=[CLASS_INFO[cls]['color'] for cls in CLASSES])
ax11.set_title('Class Balance Across Splits', fontsize=12, weight='bold')
ax11.set_xlabel('Split')
ax11.set_ylabel('Number of Subjects')
ax11.legend(title='Class')
ax11.tick_params(axis='x', rotation=45)

# 6.12 Completeness summary
ax12 = plt.subplot(3, 4, 12)
overall_completeness = [
    df_summary['complete_subjects'].sum(),
    df_summary['incomplete_subjects'].sum()
]
ax12.pie(overall_completeness, labels=['Complete\nSequences', 'Incomplete\nSequences'], 
         autopct='%1.1f%%', colors=['#2E8B57', '#DC143C'], startangle=90)
ax12.set_title('Overall Temporal Completeness', fontsize=12, weight='bold')

plt.tight_layout()
plt.suptitle('COMPREHENSIVE TEMPORAL DATASET DISTRIBUTION', fontsize=16, weight='bold', y=0.98)
plt.show()

# ----------------- 7. FINAL SUMMARY REPORT -----------------------------
print(f"\n📄 FINAL TEMPORAL DATASET SUMMARY REPORT")
print("=" * 80)

print(f"""
🎯 TEMPORAL DATASET COMPLETION STATUS: ✅ READY FOR CNN+LSTM TRAINING

📊 OVERALL TEMPORAL STATISTICS:
   • Total Images: {total_images:,}
   • Total Unique Subjects: {total_subjects:,}
   • Complete Temporal Sequences: {total_complete_subjects:,}
   • Overall Completeness Rate: {(total_complete_subjects/total_subjects*100):.1f}%
   • Expected Images per Complete Subject: {len(VISITS) * len(SLICE_TYPES)} ({len(VISITS)} visits × {len(SLICE_TYPES)} slice types)

🏷️  CLASS DISTRIBUTION (All Slice Types Combined):
""")
for cls in CLASSES:
    cls_subjects = df_summary[df_summary['class'] == cls]['unique_subjects'].sum()
    cls_complete = df_summary[df_summary['class'] == cls]['complete_subjects'].sum()
    cls_images = df_summary[df_summary['class'] == cls]['total_images'].sum()
    cls_completeness = (cls_complete / cls_subjects * 100) if cls_subjects > 0 else 0
    print(f"   • {CLASS_INFO[cls]['name']} ({cls}): {cls_subjects:,} subjects ({cls_complete} complete, {cls_completeness:.1f}%), {cls_images:,} images")

print(f"\n📂 SLICE TYPE DISTRIBUTION:")
for slice_type in SLICE_TYPES:
    slice_subjects = df_summary[df_summary['slice_type'] == slice_type]['unique_subjects'].sum()
    slice_images = df_summary[df_summary['slice_type'] == slice_type]['total_images'].sum()
    slice_percentage = (slice_images / total_images * 100)
    print(f"   • {SLICE_INFO[slice_type]['name']}: {slice_subjects:,} subjects, {slice_images:,} images ({slice_percentage:.1f}%)")

print(f"\n📅 TEMPORAL VISIT DISTRIBUTION:")
total_sc = df_summary['sc_visits'].sum()
total_m06 = df_summary['m06_visits'].sum()
total_m12 = df_summary['m12_visits'].sum()
print(f"   • Screening (sc): {total_sc:,} visits ({total_sc/total_images*100:.1f}%)")
print(f"   • 6 Months (m06): {total_m06:,} visits ({total_m06/total_images*100:.1f}%)")
print(f"   • 12 Months (m12): {total_m12:,} visits ({total_m12/total_images*100:.1f}%)")

print(f"\n📈 SPLIT DISTRIBUTION:")
for split in SPLITS:
    split_subjects = df_summary[df_summary['split'] == split]['unique_subjects'].sum()
    split_images = df_summary[df_summary['split'] == split]['total_images'].sum()
    split_percentage = (split_images / total_images * 100)
    print(f"   • {split.capitalize()}: {split_subjects:,} subjects, {split_images:,} images ({split_percentage:.1f}%)")

print("\n🔍 DATA QUALITY INDICATORS:")
print("   • Image Format: PNG (256×256 pixels)")
print("   • Enhancement: Grey Wolf Optimization (GWO) applied")
print("   • Background: Masked (brain tissue only)")
print("   • Preprocessing: Skull-stripped, ROI-registered, optimally sliced")
print("   • Temporal Structure: Complete longitudinal sequences (sc → m06 → m12)")
print("   • Variable Reference: Each visit uses optimal slice position")

print("\n💾 STORAGE LOCATION:")
print(f"   • Root: {FINAL_ROOT}")
print("   • Structure: [slice_type]/[split]/[class]/[temporal_image_files]")
print(f"   • Naming: {{subject_id}}_{{visit}}_{{slice_type}}_{{coord}}{{position}}.png")

print("\n✅ TEMPORAL DATASET IS READY FOR:")
print("   • CNN+LSTM Sequential Learning")
print("   • Disease Progression Modeling")
print("   • Temporal Biomarker Discovery")
print("   • Longitudinal Classification")
print("   • Multi-timepoint Analysis")
print("   • Clinical Progression Prediction")

print(f"\n🧠 TEMPORAL MODELING CHARACTERISTICS:")
print(f"   • Each subject contributes a temporal sequence: {' → '.join(VISITS)}")
print(f"   • Variable slice positions handle real-world scanning variability")
print(f"   • Complete sequences ensure robust temporal learning")
print(f"   • Multi-view approach (axial, coronal, sagittal) for comprehensive analysis")

print("=" * 80)
print("🎉 TEMPORAL DATASET ANALYSIS COMPLETE!")
print("=" * 80)
```

#### Data Balancing


```python
# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define Augmentation Functions
class MRIAugmenter:
    def __init__(self, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
    
    def elastic_deformation(self, image, alpha=30, sigma=5):
        """Apply elastic deformation to image"""
        shape = image.shape
        
        # Create random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply deformation
        deformed = map_coordinates(image, indices, order=1, mode='reflect')
        return deformed.reshape(shape)
    
    def rotation_translation(self, image, angle, tx, ty):
        """Apply rotation and translation"""
        rows, cols = image.shape
        center = (cols // 2, rows // 2)
        
        # Rotation matrix
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Add translation
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty
        
        # Apply transformation
        transformed = cv2.warpAffine(image, M_rot, (cols, rows), 
                                    borderMode=cv2.BORDER_REFLECT)
        return transformed
    
    def bias_field_simulation(self, image, scale=0.3):
        """Simulate bias field artifact"""
        rows, cols = image.shape
        
        # Create smooth bias field
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        X, Y = np.meshgrid(x, y)
        
        # Random polynomial bias field
        bias = 1 + scale * (
            np.random.randn() * X**2 + 
            np.random.randn() * Y**2 + 
            np.random.randn() * X * Y
        )
        
        # Apply bias field
        biased = image * bias
        return np.clip(biased, 0, 255).astype(np.uint8)
    
    def motion_blur(self, image, kernel_size=5):
        """Apply motion blur to simulate motion artifacts"""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply motion blur
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    def intensity_inhomogeneity(self, image, gamma_range=(0.8, 1.2)):
        """Apply intensity inhomogeneity"""
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        
        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(image, table)
    
    def rician_noise(self, image, sigma=10):
        """Add Rician noise (common in MRI)"""
        # Normalize image
        img_norm = image.astype(np.float32) / 255.0
        
        # Add Rician noise
        noise_real = np.random.normal(0, sigma/255.0, image.shape)
        noise_imag = np.random.normal(0, sigma/255.0, image.shape)
        
        noisy = np.sqrt((img_norm + noise_real)**2 + noise_imag**2)
        
        # Denormalize
        noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        return noisy
    
    def intensity_shift(self, image, shift_range=(-20, 20)):
        """Apply intensity shift to simulate scanner variability"""
        shift = np.random.uniform(shift_range[0], shift_range[1])
        shifted = image.astype(np.float32) + shift
        return np.clip(shifted, 0, 255).astype(np.uint8)
```


```python
# Create Augmentation Pipeline
def generate_alphabetical_id(index):
    """Generate 3-letter alphabetical ID (AAA, AAB, AAC, etc.)"""
    letters = []
    for i in range(3):
        letters.append(chr(65 + (index // (26 ** (2 - i))) % 26))
    return ''.join(letters)

def generate_augmentation_params():
    """Generate random augmentation parameters"""
    params = {
        'rotation_angle': float(np.random.uniform(-10, 10)),
        'translation_x': float(np.random.uniform(-10, 10)),
        'translation_y': float(np.random.uniform(-10, 10)),
        'elastic_alpha': float(np.random.uniform(20, 40)),
        'elastic_sigma': float(np.random.uniform(4, 6)),
        'bias_scale': float(np.random.uniform(0.2, 0.4)),
        'motion_kernel': int(np.random.choice([3, 5, 7])),
        'gamma': float(np.random.uniform(0.8, 1.2)),
        'noise_sigma': float(np.random.uniform(5, 15)),
        'intensity_shift': float(np.random.uniform(-15, 15)),
        'apply_motion': bool(np.random.random() > 0.5),  # 50% chance
        'apply_elastic': bool(np.random.random() > 0.7),  # 30% chance
    }
    return params

def augment_image(image, params, augmenter):
    """Apply augmentations to a single image using given parameters"""
    # Always apply rotation and translation
    augmented = augmenter.rotation_translation(
        image, 
        params['rotation_angle'],
        params['translation_x'],
        params['translation_y']
    )
    
    # Apply elastic deformation (30% chance)
    if params['apply_elastic']:
        augmented = augmenter.elastic_deformation(
            augmented,
            alpha=params['elastic_alpha'],
            sigma=params['elastic_sigma']
        )
    
    # Apply bias field
    augmented = augmenter.bias_field_simulation(
        augmented,
        scale=params['bias_scale']
    )
    
    # Apply motion blur (50% chance)
    if params['apply_motion']:
        augmented = augmenter.motion_blur(
            augmented,
            kernel_size=params['motion_kernel']
        )
    
    # Apply intensity inhomogeneity
    augmented = augmenter.intensity_inhomogeneity(
        augmented,
        gamma_range=(params['gamma'], params['gamma'])
    )
    
    # Apply Rician noise
    augmented = augmenter.rician_noise(
        augmented,
        sigma=params['noise_sigma']
    )
    
    # Apply intensity shift
    augmented = augmenter.intensity_shift(
        augmented,
        shift_range=(params['intensity_shift'], params['intensity_shift'])
    )
    
    return augmented
```


```python
# Data Processing Functions
def get_subject_id(filename):
    """Extract subject ID from filename"""
    # Handle augmented files: AUG_XXX_originalname.png
    if filename.startswith('AUG_'):
        # Extract original subject ID from augmented filename
        parts = filename.split('_')
        # Skip AUG and 3-letter ID, then reconstruct original subject ID
        # Assuming subject ID format is like 002_S_0413
        return '_'.join(parts[2:5])
    else:
        # Original format: extract subject ID (first 3 parts)
        parts = filename.split('_')
        return '_'.join(parts[:3])

def get_timepoint(filename):
    """Extract timepoint from filename"""
    # Handle both original and augmented filenames
    if filename.startswith('AUG_'):
        # AUG_XXX_002_S_0413_sc_coronal_y148.png
        parts = filename.split('_')
        return parts[5]  # timepoint is at position 5
    else:
        # 002_S_0413_sc_coronal_y148.png
        parts = filename.split('_')
        return parts[3]  # timepoint is at position 3

def organize_subjects_by_class(class_path):
    """Organize images by subject"""
    subjects = defaultdict(list)
    
    for img_file in os.listdir(class_path):
        if img_file.endswith('.png'):
            subject_id = get_subject_id(img_file)
            subjects[subject_id].append(img_file)
    
    # Sort images within each subject by timepoint
    for subject_id in subjects:
        subjects[subject_id].sort(key=lambda x: get_timepoint(x))
    
    return dict(subjects)
```


```python
# Main Augmentation Process
def augment_dataset(input_base, output_base, augmentation_targets):
    """
    Main function to augment the dataset
    
    augmentation_targets: dict with class -> (current_count, target_count)
    """
    planes = ['axial', 'coronal', 'sagittal']
    # classes = ['AD', 'MCI', 'CN']
    classes = ['AD', 'CN']
    
    augmenter = MRIAugmenter(seed=42)
    augmentation_log = {}
    
    for plane in planes:
        print(f"\n{'='*60}")
        print(f"Processing {plane.upper()} plane")
        print(f"{'='*60}")
        
        plane_log = {}
        
        for class_name in classes:
            print(f"\n--- Processing {class_name} ---")
            
            input_class_path = Path(input_base) / plane / 'train' / class_name
            output_class_path = Path(output_base) / plane / 'train' / class_name
            
            # Create output directory
            output_class_path.mkdir(parents=True, exist_ok=True)
            
            # Get current subjects
            subjects = organize_subjects_by_class(input_class_path)
            current_count = len(subjects)
            target_count = augmentation_targets[class_name]['target']
            
            print(f"Current subjects: {current_count}")
            print(f"Target subjects: {target_count}")
            
            # Copy all original images first
            print("Copying original images...")
            for subject_id, images in tqdm(subjects.items()):
                for img_file in images:
                    shutil.copy2(
                        input_class_path / img_file,
                        output_class_path / img_file
                    )
            
            # Calculate how many augmented subjects needed
            subjects_to_augment = target_count - current_count
            
            # Initialize augmentation_params
            augmentation_params = {}
            
            if subjects_to_augment > 0:
                print(f"Need to create {subjects_to_augment} augmented subjects")
                
                # Select subjects to augment (randomly sample with replacement)
                subject_ids = list(subjects.keys())
                selected_subjects = np.random.choice(
                    subject_ids, 
                    size=subjects_to_augment,
                    replace=True
                )
                
                # Generate augmentation parameters for each new subject
                # This ensures same augmentation across planes
                if plane == 'axial':  # Generate params only once
                    augmentation_params = {}
                    for i, source_subject in enumerate(selected_subjects):
                        aug_params = generate_augmentation_params()
                        alpha_id = generate_alphabetical_id(i)
                        augmentation_params[alpha_id] = {
                            'source': source_subject,
                            'params': aug_params
                        }
                    
                    # Save params for other planes to use
                    plane_log['augmentation_params'] = augmentation_params
                else:
                    # Load params from axial processing
                    augmentation_params = augmentation_log['axial'][class_name]['augmentation_params']
                
                # Apply augmentations
                print("Creating augmented subjects...")
                for alpha_id, aug_info in tqdm(augmentation_params.items()):
                    source_subject = aug_info['source']
                    params = aug_info['params']
                    
                    # Process each timepoint for this subject
                    for img_file in subjects[source_subject]:
                        # Load image
                        img_path = input_class_path / img_file
                        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        
                        # Apply augmentation
                        augmented = augment_image(image, params, augmenter)
                        
                        # Save with new name format: AUG_XXX_originalfilename
                        new_filename = f"AUG_{alpha_id}_{img_file}"
                        cv2.imwrite(
                            str(output_class_path / new_filename),
                            augmented
                        )
                
                print(f"Created {subjects_to_augment} augmented subjects")
            else:
                print("No augmentation needed for this class")
            
            # Store log info
            plane_log[class_name] = {
                'original_subjects': current_count,
                'augmented_subjects': subjects_to_augment if subjects_to_augment > 0 else 0,
                'total_subjects': target_count,
                'augmentation_params': augmentation_params
            }
        
        augmentation_log[plane] = plane_log
    
    return augmentation_log
```


```python
# Copy validation and test sets (no augmentation)
def copy_non_train_sets(input_base, output_base):
    """Copy validation and test sets without augmentation"""
    planes = ['axial', 'coronal', 'sagittal']
    splits = ['val', 'test']
    # classes = ['AD', 'MCI', 'CN']
    classes = ['AD', 'CN']
    
    print("\nCopying validation and test sets...")
    
    for plane in planes:
        for split in splits:
            for class_name in classes:
                input_path = Path(input_base) / plane / split / class_name
                output_path = Path(output_base) / plane / split / class_name
                
                if input_path.exists():
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all files
                    for file in input_path.glob('*.png'):
                        shutil.copy2(file, output_path / file.name)
    
    print("Validation and test sets copied successfully!")

# Execute Augmentation
# Define paths
input_base = '../datasets/ADNI_1_5_T/8_enhanced'
output_base = '../datasets/ADNI_1_5_T/9_balanced'

# Define augmentation targets
augmentation_targets = {
    'AD': {'current': 89, 'target': 180},
    # 'MCI': {'current': 209, 'target': 209},  # No augmentation
    'CN': {'current': 131, 'target': 180}
}

# Run augmentation
print("Starting augmentation process...")
augmentation_log = augment_dataset(input_base, output_base, augmentation_targets)

# Copy validation and test sets
copy_non_train_sets(input_base, output_base)

# Convert numpy types for JSON serialization
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Save augmentation log
log_path = Path(output_base) / 'augmentation_log.json'
with open(log_path, 'w') as f:
    json.dump(convert_numpy_types(augmentation_log), f, indent=2)

print(f"\nAugmentation complete! Log saved to {log_path}")
```

#### [QC] Verify Augmentation Result


```python
def verify_augmentation(output_base):
    """Verify the augmentation results"""
    planes = ['axial', 'coronal', 'sagittal']
    splits = ['train', 'val', 'test']
    # classes = ['AD', 'MCI', 'CN']
    classes = ['AD', 'CN']
    
    print("\n" + "="*60)
    print("AUGMENTATION VERIFICATION")
    print("="*60)
    
    for plane in planes:
        print(f"\n{plane.upper()} Plane:")
        print("-"*40)
        
        for split in splits:
            print(f"\n{split.upper()} Split:")
            total_images = 0
            
            for class_name in classes:
                class_path = Path(output_base) / plane / split / class_name
                if class_path.exists():
                    # Count images
                    images = list(class_path.glob('*.png'))
                    
                    # Separate original and augmented files
                    original_files = []
                    augmented_files = []
                    
                    for img in images:
                        if img.name.startswith('AUG_'):
                            augmented_files.append(img)
                        else:
                            original_files.append(img)
                    
                    # Count unique subjects from original files
                    original_subjects = set()
                    for img in original_files:
                        subject_id = get_subject_id(img.name)
                        original_subjects.add(subject_id)
                    
                    # Count augmented subjects (by their 3-letter ID)
                    augmented_subjects = set()
                    for img in augmented_files:
                        aug_id = img.name.split('_')[1]  # Get the 3-letter ID
                        augmented_subjects.add(aug_id)
                    
                    total_subjects = len(original_subjects) + len(augmented_subjects)
                    total_images += len(images)
                    
                    print(f"  {class_name}: {total_subjects} subjects "
                          f"({len(original_subjects)} original + "
                          f"{len(augmented_subjects)} augmented), {len(images)} images")
            
            print(f"  Total: {total_images} images")

# Run verification
verify_augmentation(output_base)
```

#### [QC] Visualize Augmentation Result


```python
# Visualize Original vs Augmented Images
def visualize_augmentation_comparison(input_base, output_base):
    """Visualize original vs augmented images for each class and plane"""
    
    planes = ['axial', 'coronal', 'sagittal']
    # classes = ['AD', 'MCI', 'CN']
    classes = ['AD', 'CN']
    timepoints = ['sc', 'm06', 'm12']
    
    # Create figure with subplots
    fig, axes = plt.subplots(9, 6, figsize=(18, 27))
    fig.suptitle('Original vs Augmented Images Comparison', fontsize=20, y=1)
    
    # Load augmentation log to get mappings
    log_path = Path(output_base) / 'augmentation_log.json'
    with open(log_path, 'r') as f:
        aug_log = json.load(f)
    
    row = 0
    for plane_idx, plane in enumerate(planes):
        for class_idx, class_name in enumerate(classes):
            # Paths
            orig_path = Path(input_base) / plane / 'train' / class_name
            aug_path = Path(output_base) / plane / 'train' / class_name
            
            # Get a sample subject
            orig_images = list(orig_path.glob('*.png'))
            subjects = {}
            for img in orig_images:
                subj_id = get_subject_id(img.name)
                if subj_id not in subjects:
                    subjects[subj_id] = []
                subjects[subj_id].append(img.name)
            
            # Pick first subject
            sample_subject = list(subjects.keys())[0]
            subject_images = sorted(subjects[sample_subject])
            
            # Plot original images
            for col, img_name in enumerate(subject_images[:3]):
                img_path = orig_path / img_name
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                ax = axes[row, col]
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
                # Add labels
                if row == 0:
                    ax.set_title(f'Original {timepoints[col]}', fontsize=12)
                if col == 0:
                    ax.text(-20, img.shape[0]//2, f'{plane.upper()}\n{class_name}', 
                           rotation=90, va='center', ha='center', fontsize=14, weight='bold')
            
            # Plot augmented images (if available)
            if class_name != 'MCI':  # MCI has no augmentation
                # Get first augmented subject
                aug_params = aug_log[plane][class_name]['augmentation_params']
                if aug_params:
                    first_aug_id = sorted(aug_params.keys())[0]
                    source_subject = aug_params[first_aug_id]['source']
                    
                    # Find augmented files
                    for col, timepoint in enumerate(timepoints):
                        # Find the augmented file
                        aug_pattern = f"AUG_{first_aug_id}_{source_subject}_{timepoint}_*.png"
                        aug_files = list(aug_path.glob(aug_pattern))
                        
                        if aug_files:
                            aug_img = cv2.imread(str(aug_files[0]), cv2.IMREAD_GRAYSCALE)
                            
                            ax = axes[row, col + 3]
                            ax.imshow(aug_img, cmap='gray')
                            ax.axis('off')
                            
                            # Add labels
                            if row == 0:
                                ax.set_title(f'Augmented {timepoint}', fontsize=12)
                            
                            # Add border to highlight augmented images
                            rect = patches.Rectangle((0, 0), aug_img.shape[1]-1, aug_img.shape[0]-1,
                                                   linewidth=3, edgecolor='green', facecolor='none')
                            ax.add_patch(rect)
            else:
                # MCI - no augmentation, show text
                for col in range(3, 6):
                    ax = axes[row, col]
                    ax.text(0.5, 0.5, 'No Augmentation\nfor MCI', 
                           ha='center', va='center', fontsize=14, 
                           transform=ax.transAxes, style='italic', color='gray')
                    ax.axis('off')
            
            row += 1
    
    # Add separation lines between planes
    for i in range(1, 3):
        line = plt.Line2D([0, 1], [1 - i/3, 1 - i/3], 
                         transform=fig.transFigure, 
                         color='black', linewidth=2)
        fig.add_artist(line)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.3, wspace=0.1)
    
    # Save figure
    save_path = Path(output_base) / 'augmentation_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    
    plt.show()

# Run visualization
visualize_augmentation_comparison(input_base, output_base)
```


```python
# Detailed Augmentation Effects Visualization
def visualize_augmentation_effects(input_base, output_base):
    """Show detailed effects of augmentation on a single subject"""
    
    # Select one subject from AD class, axial plane
    plane = 'coronal'
    class_name = 'AD'
    
    # Paths
    orig_path = Path(input_base) / plane / 'train' / class_name
    aug_path = Path(output_base) / plane / 'train' / class_name
    
    # Get augmentation log
    log_path = Path(output_base) / 'augmentation_log.json'
    with open(log_path, 'r') as f:
        aug_log = json.load(f)
    
    # Get first augmented subject
    aug_params = aug_log[plane][class_name]['augmentation_params']
    first_aug_id = sorted(aug_params.keys())[0]
    source_subject = aug_params[first_aug_id]['source']
    params = aug_params[first_aug_id]['params']
    
    # Find original and augmented screening images
    orig_files = list(orig_path.glob(f"{source_subject}_sc_*.png"))
    aug_files = list(aug_path.glob(f"AUG_{first_aug_id}_{source_subject}_sc_*.png"))
    
    if orig_files and aug_files:
        # Load images
        orig_img = cv2.imread(str(orig_files[0]), cv2.IMREAD_GRAYSCALE)
        aug_img = cv2.imread(str(aug_files[0]), cv2.IMREAD_GRAYSCALE)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Augmentation Effects on Subject {source_subject}', fontsize=16)
        
        # Original image
        axes[0].imshow(orig_img, cmap='gray')
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Augmented image
        axes[1].imshow(aug_img, cmap='gray')
        axes[1].set_title('Augmented Image', fontsize=14)
        axes[1].axis('off')
        
        # Difference map
        diff = cv2.absdiff(orig_img, aug_img)
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Difference Map', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Print augmentation parameters
        print("\nAugmentation Parameters Applied:")
        print("-" * 40)
        print(f"Rotation: {params['rotation_angle']:.2f}°")
        print(f"Translation: ({params['translation_x']:.2f}, {params['translation_y']:.2f}) pixels")
        print(f"Bias field scale: {params['bias_scale']:.3f}")
        print(f"Noise sigma: {params['noise_sigma']:.2f}")
        print(f"Gamma: {params['gamma']:.3f}")
        print(f"Intensity shift: {params['intensity_shift']:.2f}")
        print(f"Motion blur applied: {params['apply_motion']}")
        print(f"Elastic deformation applied: {params['apply_elastic']}")
        
        if params['apply_elastic']:
            print(f"  - Elastic alpha: {params['elastic_alpha']:.2f}")
            print(f"  - Elastic sigma: {params['elastic_sigma']:.2f}")
        
        if params['apply_motion']:
            print(f"  - Motion kernel size: {params['motion_kernel']}")
        
        plt.tight_layout()
        
        # Save figure
        save_path = Path(output_base) / 'augmentation_effects_detail.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nDetail visualization saved to: {save_path}")
        
        plt.show()

# Run detailed visualization
visualize_augmentation_effects(input_base, output_base)
```


```python
# Cell 14: Generate Augmentation Report
def generate_augmentation_report(output_base):
    """Generate a summary report of the augmentation process"""
    from datetime import datetime
    
    # Load augmentation log
    log_path = Path(output_base) / 'augmentation_log.json'
    with open(log_path, 'r') as f:
        aug_log = json.load(f)
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("BRAIN MRI AUGMENTATION REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("AUGMENTATION SUMMARY")
    report_lines.append("-"*40)
    
    total_aug_subjects = 0
    for plane in ['axial']:  # Use axial as reference
        for class_name in ['AD', 'CN']:
            if class_name in aug_log[plane]:
                n_aug = len(aug_log[plane][class_name].get('augmentation_params', {}))
                total_aug_subjects += n_aug
                report_lines.append(f"{class_name}: {n_aug} augmented subjects created")
    
    report_lines.append(f"\nTotal augmented subjects: {total_aug_subjects}")
    report_lines.append(f"Total augmented images: {total_aug_subjects * 3 * 3} (3 timepoints × 3 planes)")
    report_lines.append("")
    
    # Augmentation techniques used
    report_lines.append("AUGMENTATION TECHNIQUES APPLIED")
    report_lines.append("-"*40)
    report_lines.append("• Geometric transformations:")
    report_lines.append("  - Rotation: ±10 degrees")
    report_lines.append("  - Translation: ±10 pixels (X and Y)")
    report_lines.append("  - Elastic deformation: 30% probability")
    report_lines.append("• Intensity transformations:")
    report_lines.append("  - Bias field simulation")
    report_lines.append("  - Gamma correction")
    report_lines.append("  - Intensity shift")
    report_lines.append("• MRI-specific artifacts:")
    report_lines.append("  - Motion blur: 50% probability")
    report_lines.append("  - Rician noise")
    report_lines.append("")
    
    # Final dataset composition
    report_lines.append("FINAL DATASET COMPOSITION")
    report_lines.append("-"*40)
    report_lines.append("Training Set:")
    report_lines.append("  AD:  180 subjects (89 original + 91 augmented) = 540 images")
    # report_lines.append("  MCI: 209 subjects (209 original + 0 augmented) = 627 images")
    report_lines.append("  CN:  180 subjects (131 original + 49 augmented) = 540 images")
    report_lines.append("  Total: 569 subjects, 1,707 images")
    report_lines.append("")
    report_lines.append("Validation Set: 188 subjects, 564 images (no augmentation)")
    report_lines.append("Test Set: 9 subjects, 27 images (no augmentation)")
    report_lines.append("")
    
    # Naming convention
    report_lines.append("FILE NAMING CONVENTION")
    report_lines.append("-"*40)
    report_lines.append("Original: {SubjectID}_{timepoint}_{plane}_{slice}.png")
    report_lines.append("Augmented: AUG_{3-letter-ID}_{SubjectID}_{timepoint}_{plane}_{slice}.png")
    report_lines.append("")
    report_lines.append("Where:")
    report_lines.append("  - 3-letter-ID: Unique augmentation identifier (AAA, AAB, etc.)")
    report_lines.append("  - SubjectID: Original subject identifier")
    report_lines.append("  - timepoint: sc (screening), m06 (6 months), m12 (12 months)")
    report_lines.append("  - plane: axial, coronal, or sagittal")
    report_lines.append("")
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = Path(output_base) / 'augmentation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {report_path}")

# Generate report
generate_augmentation_report(output_base)
```