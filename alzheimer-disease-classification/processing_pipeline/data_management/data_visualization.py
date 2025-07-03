# =====================================================================
# ADNI MRI PROCESSING PIPELINE - DATA VISUALIZATION & QC
# =====================================================================
# This module contains functions for visualizing the data at various
# stages of the pipeline to perform Quality Control (QC).

import os
import random
from pathlib import Path
from collections import defaultdict
import logging

import matplotlib.pyplot as plt
import cv2

# Constants for visualization
VISIT_INFO = {
    'sc': {'name': 'Screening', 'color': '#2E8B57', 'description': 'Baseline visit'},
    'm06': {'name': '6 Months', 'color': '#4682B4', 'description': '6-month follow-up'},
    'm12': {'name': '12 Months', 'color': '#8B0000', 'description': '12-month follow-up'}
}

def ensure_vis_dir(subdir):
    """Ensure visualization directory exists"""
    vis_dir = Path("visualizations") / "processing_pipeline" / subdir
    vis_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir

def get_subject_id(filename):
    """Extract subject ID from filename"""
    # Handle augmented files: AUG_XXX_originalname.png
    if filename.startswith('AUG_'):
        # Extract original subject ID from augmented filename
        parts = filename.split('_')
        # Skip AUG and 3-letter ID, then reconstruct original subject ID
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

def print_final_data_distribution(logger):
    """Print the final data distribution statistics"""
    logger.info("\n📊 FINAL DATA DISTRIBUTION")
    logger.info("=" * 80)

    # Setup parameters
    slice_types = ['axial', 'coronal', 'sagittal']
    splits = ['train', 'val', 'test']
    classes = ['AD', 'CN']
    
    # Use the correct balanced data directory path
    balanced_dir = Path("../datasets/ADNI_1_5_T/9_balanced")
    logger.info(f"Reading data from: {balanced_dir}")

    for slice_type in slice_types:
        logger.info(f"\n{slice_type.upper()} Plane:")
        logger.info("-" * 40)
        
        for split in splits:
            logger.info(f"\n{split.upper()} Split:")
            total_images = 0
            
            for class_name in classes:
                class_path = balanced_dir / slice_type / split / class_name
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
                    
                    logger.info(f"  {class_name}: {total_subjects} subjects "
                              f"({len(original_subjects)} original + "
                              f"{len(augmented_subjects)} augmented), {len(images)} images")
            
            logger.info(f"  Total: {total_images} images")

def visualize_sample_data(logger):
    """Visualize sample data for each plane and class"""
    vis_dir = ensure_vis_dir("samples")
    balanced_dir = Path("../datasets/ADNI_1_5_T/9_balanced")
    cropped_dir = Path("../datasets/ADNI_1_5_T/7_cropped")  # Directory for pre-enhanced images
    
    # Setup parameters
    slice_types = ['axial', 'coronal', 'sagittal']
    classes = ['AD', 'CN']
    visits = ['sc', 'm06', 'm12']
    
    for slice_type in slice_types:
        for class_name in classes:
            # Create figure for this plane and class
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # Changed to 3 rows
            fig.suptitle(f'{class_name} - {slice_type.capitalize()} Plane', fontsize=16)
            
            # Get data directories
            balanced_class_dir = balanced_dir / slice_type / 'train' / class_name
            cropped_class_dir = cropped_dir / slice_type / 'train' / class_name
            
            if not (balanced_class_dir.exists() and cropped_class_dir.exists()):
                continue
            
            # Get all original files from balanced directory
            original_files = [f for f in balanced_class_dir.glob('*.png') if not f.name.startswith('AUG_')]
            
            # Group files by subject
            subjects = defaultdict(list)
            for f in original_files:
                subject_id = get_subject_id(f.name)
                subjects[subject_id].append(f)
            
            # Find subjects with all timepoints and augmented versions
            complete_subjects = []
            for sid, files in subjects.items():
                # Check if subject has all timepoints
                timepoints = set(get_timepoint(f.name) for f in files)
                if len(timepoints) != len(visits):
                    continue
                
                # Check if all timepoints have augmented versions
                has_all_augmented = True
                for f in files:
                    aug_pattern = f"AUG_*_{get_subject_id(f.name)}_{get_timepoint(f.name)}_*.png"
                    aug_files = list(balanced_class_dir.glob(aug_pattern))
                    if not aug_files:
                        has_all_augmented = False
                        break
                
                # Check if all timepoints exist in cropped directory
                has_all_cropped = True
                for f in files:
                    cropped_file = cropped_class_dir / f.name
                    if not cropped_file.exists():
                        has_all_cropped = False
                        break
                
                if has_all_augmented and has_all_cropped:
                    complete_subjects.append(sid)
            
            if not complete_subjects:
                logger.warning(f"No complete subjects with all versions found for {class_name} - {slice_type}")
                continue
            
            # Select a random complete subject
            subject_id = random.choice(complete_subjects)
            subject_files = subjects[subject_id]
            
            # Sort files by timepoint
            subject_files.sort(key=lambda x: get_timepoint(x.name))
            
            # Find corresponding cropped and augmented files
            cropped_files = []
            augmented_files = []
            for orig_file in subject_files:
                # Get cropped file (pre-enhanced version)
                cropped_file = cropped_class_dir / orig_file.name
                cropped_files.append(cropped_file)
                
                # Find augmented version
                aug_pattern = f"AUG_*_{get_subject_id(orig_file.name)}_{get_timepoint(orig_file.name)}_*.png"
                aug_files = list(balanced_class_dir.glob(aug_pattern))
                augmented_files.append(aug_files[0])  # We know it exists from our earlier check
            
            # Plot all versions of images
            for col, (cropped_file, enhanced_file, aug_file, visit) in enumerate(zip(cropped_files, subject_files, augmented_files, visits)):
                # Plot cropped (pre-enhanced)
                img = cv2.imread(str(cropped_file), cv2.IMREAD_GRAYSCALE)
                img = cv2.rotate(img, cv2.ROTATE_180)
                axes[0, col].imshow(img, cmap='gray')
                axes[0, col].set_title(f'Pre-enhanced - {visit}')
                axes[0, col].axis('off')
                
                # Plot enhanced
                img = cv2.imread(str(enhanced_file), cv2.IMREAD_GRAYSCALE)
                img = cv2.rotate(img, cv2.ROTATE_180)
                axes[1, col].imshow(img, cmap='gray')
                axes[1, col].set_title(f'Enhanced - {visit}')
                axes[1, col].axis('off')
                
                # Plot augmented
                img = cv2.imread(str(aug_file), cv2.IMREAD_GRAYSCALE)
                img = cv2.rotate(img, cv2.ROTATE_180)
                axes[2, col].imshow(img, cmap='gray')
                axes[2, col].set_title(f'Augmented - {visit}')
                axes[2, col].axis('off')
            
            # Save the figure
            plt.tight_layout()
            save_path = vis_dir / f"sample_{class_name}_{slice_type}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved sample visualization for {class_name} - {slice_type} to {save_path}")

def generate_qc_report(logger):
    """Generate quality control report"""
    logger.info("\n🔍 GENERATING QUALITY CONTROL REPORT")
    logger.info("=" * 80)

    try:
        # Print final data distribution
        print_final_data_distribution(logger)
        
        # Visualize sample data
        logger.info("\nGenerating sample visualizations...")
        visualize_sample_data(logger)

        logger.info("\n✅ Quality control report generation complete!")
        logger.info("📂 Visualizations saved in the 'visualizations/processing_pipeline/samples' directory")

    except Exception as e:
        logger.error(f"❌ Error generating QC report: {str(e)}")
        raise

if __name__ == '__main__':
    # Setup basic logger for standalone execution
    main_logger = logging.getLogger(__name__)
    if not main_logger.handlers:
        main_logger.addHandler(logging.StreamHandler())
        main_logger.setLevel(logging.INFO)
    
    # Generate QC report
    generate_qc_report(main_logger)
