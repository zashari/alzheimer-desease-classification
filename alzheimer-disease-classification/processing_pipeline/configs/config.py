# =====================================================================
# ADNI MRI PROCESSING PIPELINE - CONFIGURATION
# =====================================================================
# This file centralizes all paths, parameters, and settings for the
# ADNI data processing pipeline.

from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------
# 1. CORE PATHS
# ---------------------------------------------------------------------

# --- Input Paths ---
# Assumes a base project directory containing the 'datasets' folder
PROJECT_ROOT = Path(".")
BASE_DATA_DIR = Path("../datasets/ADNI_1_5_T")  # Keep this path relative to parent

# --- Visualization Output Path ---
VISUALIZATION_DIR = PROJECT_ROOT / "visualizations" / "processing_pipeline"

# Raw ADNI data directory (where original .nii files are)
RAW_NIFTI_DIR = BASE_DATA_DIR / ".ADNI"

# Metadata CSV file from ADNI
METADATA_CSV = BASE_DATA_DIR / "ADNI1_Complete_1Yr_1.5T_3_21_2025.csv"

# --- FSL (FMRIB Software Library) Paths ---
# Adjust these paths to match your FSL installation
FSL_DIR = Path("/Users/AndiZakyAshari/fsl")
FSL_BIN_DIR = FSL_DIR / "bin"
FSL_DATA_DIR = FSL_DIR / "data"

# Standard brain reference for registration
MNI_BRAIN_TEMPLATE = FSL_DATA_DIR / "standard" / "MNI152_T1_1mm_brain.nii.gz"

# Hippocampal ROI template path
ROI_TEMPLATE = PROJECT_ROOT / "hippocampal-ROI" / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"

# --- Data Organization ---
SLICE_TYPES = ["axial", "coronal", "sagittal"]
SPLITS = ["train", "val", "test"]

# --- Output Paths for Each Pipeline Step ---
# Each step's output serves as the input for the next.
OUTPUT_ROOT = BASE_DATA_DIR # All processed data will be stored here.
STEP1_SPLIT_DIR = OUTPUT_ROOT / "1_splitted_sequential"
STEP2_SKULLSTRIP_DIR = OUTPUT_ROOT / "2_skull_stripping_sequential"
STEP3_ROI_REG_DIR = OUTPUT_ROOT / "3_roi_subj_space_sequential"
STEP4_SLICES_AXIAL_DIR = OUTPUT_ROOT / "4_optimal_axial_sequential"
STEP4_SLICES_CORONAL_DIR = OUTPUT_ROOT / "4_optimal_coronal_sequential"
STEP4_SLICES_SAGITTAL_DIR = OUTPUT_ROOT / "4_optimal_sagittal_sequential"
STEP5_LABELED_DIR = OUTPUT_ROOT / "5_labelling_sequential_variable"
STEP6_2D_CONVERTED_DIR = OUTPUT_ROOT / "6_2Dconverted_sequential"
STEP7_CROPPED_DIR = OUTPUT_ROOT / "7_cropped"
STEP8_ENHANCED_DIR = OUTPUT_ROOT / "8_enhanced"
STEP9_BALANCED_DIR = OUTPUT_ROOT / "9_balanced"


# ---------------------------------------------------------------------
# 2. DATA SPLITTING & ORGANIZATION PARAMETERS
# ---------------------------------------------------------------------
# --- Dataset Splitting ---
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
USE_SYMLINKS = True # Use symlinks instead of copying files to save space

# --- Temporal Sequence ---
# Define the required visits for a subject to be included in the analysis
REQUIRED_VISITS = ["sc", "m06", "m12"]
# Define the classes to be included in the final dataset
# CLASSES_TO_INCLUDE = ["AD", "MCI", "CN"]
CLASSES_TO_INCLUDE = ["AD", "CN"]


# ---------------------------------------------------------------------
# 3. NIFTI PROCESSING PARAMETERS
# ---------------------------------------------------------------------
# --- Skull Stripping (FSL BET) ---
# Arguments for the Brain Extraction Tool (BET)
# -f: fractional intensity threshold (0->1); smaller values give larger brain outlines
# -g: vertical gradient in fractional intensity threshold (-1->1)
# -B: Bias field & neck cleanup
BET_ARGS = ["-f", "0.5", "-g", "0", "-B"]
BET_TIMEOUT = 600 # 10-minute timeout per BET call

# --- Parallel Processing ---
# Set the number of parallel threads for CPU-intensive tasks
MAX_THREADS = 12


# ---------------------------------------------------------------------
# 4. 2D CONVERSION & IMAGE PROCESSING PARAMETERS
# ---------------------------------------------------------------------
# --- 2D Conversion ---
TARGET_2D_SIZE = (256, 256) # Target resolution for all PNG images
PNG_INTERPOLATION = Image.Resampling.LANCZOS # High-quality interpolation method
INTENSITY_NORM_PERCENTILE = (1, 99) # Clip intensity to this percentile range

# --- Cropping ---
CROP_PADDING = 5
CROP_ROTATION_ANGLE = -90

# --- Image Enhancement (GWO) ---
GWO_ITERATIONS = 50
GWO_NUM_WOLVES = 20
GWO_METHOD = 'adaptive' # 'adaptive', 'clahe', 'gabor', 'unsharp'

# --- Data Augmentation & Balancing ---
# Define target number of subjects per class in the training set
AUGMENTATION_TARGETS = {
    'AD': 180,
    # 'MCI': 209,  # No augmentation if target is same as original
    'CN': 180
}

# Logging configuration
LOG_DIR = Path("logs")