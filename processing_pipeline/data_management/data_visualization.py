# =====================================================================
# ADNI MRI PROCESSING PIPELINE - DATA VISUALIZATION & QC
# =====================================================================
# This module contains functions for visualizing the data at various
# stages of the pipeline to perform Quality Control (QC).

import random
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import nibabel as nib
import cv2
from PIL import Image

# Import configurations and utilities
from configs import config
from utils import utils

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def visualize_split_qc(num_subjects_per_split=1):
    """
    Visualizes a few random subjects from each data split (train, val, test)
    after the initial splitting step to ensure the temporal sequence is correct.
    """
    print("\n🔬 QC: Visualizing raw data from splits...")
    input_root = config.STEP1_SPLIT_DIR
    if not input_root.exists():
        print(f"❌ Cannot run QC. Directory not found: {input_root}")
        return

    for split in ["train", "val", "test"]:
        split_dir = input_root / split
        if not split_dir.is_dir(): continue
        
        subjects = [d for d in split_dir.iterdir() if d.is_dir()]
        if not subjects: continue

        sample_subjects = random.sample(subjects, min(num_subjects_per_split, len(subjects)))

        for subj_dir in sample_subjects:
            nii_files = sorted(list(subj_dir.glob("*.nii")))
            if not nii_files: continue

            fig, axes = plt.subplots(1, len(nii_files), figsize=(5 * len(nii_files), 5))
            if len(nii_files) == 1: axes = [axes]
            
            fig.suptitle(f'QC for Split: {split.upper()} | Subject: {subj_dir.name}', fontsize=16)
            
            for ax, nii_path in zip(axes, nii_files):
                try:
                    data = nib.load(str(nii_path)).get_fdata()
                    # Display a central coronal slice
                    coronal_slice = data[:, data.shape[1] // 2, :]
                    ax.imshow(np.rot90(coronal_slice), cmap='gray')
                    ax.set_title(nii_path.name.split('_')[-1].split('.')[0].upper())
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error loading', ha='center')
                    print(f"Error loading {nii_path}: {e}")
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

def visualize_enhancement_comparison(num_samples_per_class=2):
    """
    Compares original cropped images with their GWO-enhanced counterparts.
    """
    print("\n🔬 QC: Visualizing enhancement results...")
    cropped_root = config.STEP7_CROPPED_DIR
    enhanced_root = config.STEP8_ENHANCED_DIR

    if not cropped_root.exists() or not enhanced_root.exists():
        print("❌ Cannot run enhancement QC. Input or output directories missing.")
        return

    samples = []
    for cls in config.CLASSES_TO_INCLUDE:
        # Use coronal training images for visualization
        cropped_cls_dir = cropped_root / "coronal" / "train" / cls
        enhanced_cls_dir = enhanced_root / "coronal" / "train" / cls
        
        if cropped_cls_dir.exists():
            cropped_files = list(cropped_cls_dir.glob("*.png"))
            if not cropped_files: continue

            chosen_files = random.sample(cropped_files, min(num_samples_per_class, len(cropped_files)))
            for f in chosen_files:
                enhanced_f = enhanced_cls_dir / f.name
                if enhanced_f.exists():
                    samples.append((f, enhanced_f, f"{cls} | {utils.extract_visit_from_filename(f.name)}"))
    
    if not samples:
        print("No samples found for comparison.")
        return

    fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
    fig.suptitle('Enhancement Comparison (Original vs. Enhanced)', fontsize=16, weight='bold')

    for i, (orig_p, en_p, label) in enumerate(samples):
        orig_img = np.array(Image.open(orig_p))
        en_img = np.array(Image.open(en_p))
        
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title(f'Original\n{label}')
        axes[0, i].axis('off')

        axes[1, i].imshow(en_img, cmap='gray')
        axes[1, i].set_title('Enhanced')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_augmentation_results():
    """
    Visualizes a comparison between original images and their augmented versions
    to verify the data balancing step.
    """
    print("\n🔬 QC: Visualizing augmentation results...")
    enhanced_root = config.STEP8_ENHANCED_DIR
    balanced_root = config.STEP9_BALANCED_DIR

    # Load augmentation log to find a source and its augmented version
    log_path = balanced_root / 'augmentation_log.json'
    if not log_path.exists():
        print("❌ Augmentation log not found. Cannot visualize results.")
        return
    with open(log_path, 'r') as f:
        aug_log = json.load(f)

    # Find a sample to visualize
    plane = 'coronal'
    class_name = 'AD' # Typically, the minority class is augmented
    if not aug_log[plane][class_name]['augmentation_params']:
        print(f"No augmentation was performed for {class_name} in {plane} plane.")
        return

    # Get the first augmented subject from the log
    first_aug_id = list(aug_log[plane][class_name]['augmentation_params'].keys())[0]
    aug_info = aug_log[plane][class_name]['augmentation_params'][first_aug_id]
    source_subject = aug_info['source']
    
    # Get the files for one timepoint (e.g., 'sc')
    orig_file = next((enhanced_root / plane / 'train' / class_name).glob(f"{source_subject}_sc_*.png"), None)
    aug_file = next((balanced_root / plane / 'train' / class_name).glob(f"AUG_{first_aug_id}_{source_subject}_sc_*.png"), None)

    if not orig_file or not aug_file:
        print("Could not find a matching original/augmented pair to visualize.")
        return

    orig_img = np.array(Image.open(orig_file))
    aug_img = np.array(Image.open(aug_file))
    diff = cv2.absdiff(orig_img, aug_img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Augmentation Effect on Subject {source_subject}', fontsize=16)

    axes[0].imshow(orig_img, cmap='gray'); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(aug_img, cmap='gray'); axes[1].set_title('Augmented'); axes[1].axis('off')
    im = axes[2].imshow(diff, cmap='hot'); axes[2].set_title('Difference Map'); axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    plt.show()


def generate_final_distribution_report():
    """
    Generates a comprehensive report and set of visualizations for the
    final processed dataset, showing distributions by class, split, etc.
    """
    print("\n📊 Generating final dataset distribution report...")
    if not config.STEP9_BALANCED_DIR.exists():
        print("❌ Final dataset directory not found. Cannot generate report.")
        return
        
    # This function can be expanded to create the detailed multi-plot
    # visualization from the end of the notebook. For brevity, we'll
    # create a simplified text report here.

    all_data = []
    for f in config.STEP9_BALANCED_DIR.rglob("*.png"):
        parts = f.parts
        # Expected structure: .../9_balanced/slice_type/split/class/file.png
        if len(parts) >= 4:
            all_data.append({
                "slice_type": parts[-4],
                "split": parts[-3],
                "class": parts[-2],
                "subject": utils.extract_subject_id_from_filename(f.name)
            })
    
    if not all_data:
        print("No files found in the final directory.")
        return
        
    df = pd.DataFrame(all_data)

    print("\n--- Final Dataset Composition ---")
    print(df.groupby(['split', 'class', 'slice_type']).size().unstack(fill_value=0))
    
    print("\n--- Unique Subjects per Split/Class (Across all slice types) ---")
    print(df.groupby(['split', 'class'])['subject'].nunique().unstack(fill_value=0))
    
    print("\n✅ Report generation complete.")


if __name__ == '__main__':
    # You can call these functions to test them individually
    visualize_split_qc()
    visualize_enhancement_comparison()
    visualize_augmentation_results()
    generate_final_distribution_report()
