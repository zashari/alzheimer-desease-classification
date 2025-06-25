# =====================================================================
# ADNI MRI PROCESSING PIPELINE - UTILITY FUNCTIONS
# =====================================================================
# This file contains common helper functions used across multiple
# modules of the pipeline.

import numpy as np
from PIL import Image

from configs import config

# ---------------------------------------------------------------------
# 1. FILENAME & PATH PARSING HELPERS
# ---------------------------------------------------------------------

def extract_subject_id_from_filename(filename: str) -> str | None:
    """
    Extracts a subject ID (e.g., '002_S_0295') from various filename formats.

    Handles original files, augmented files, and different processing stages.
    - Original: '002_S_0295_sc_axial_x123.png' -> '002_S_0295'
    - Augmented: 'AUG_AAA_002_S_0295_sc_axial_x123.png' -> '002_S_0295'
    """
    # Clean up common extensions
    clean_name = filename.replace('.nii.gz', '').replace('.nii', '').replace('.png', '')
    parts = clean_name.split('_')

    # Find the XXX_S_YYYY pattern
    for i in range(len(parts) - 2):
        if (parts[i].isdigit() and
            parts[i+1] == 'S' and
            parts[i+2].isdigit()):
            return f"{parts[i]}_S_{parts[i+2]}"
    return None


def extract_visit_from_filename(filename: str) -> str | None:
    """
    Extracts a visit code (e.g., 'sc', 'm06', 'm12') from a filename.
    """
    for visit in config.REQUIRED_VISITS:
        if f"_{visit}_" in filename:
            return visit
    return None


def extract_slice_type_from_filename(filename: str) -> str | None:
    """
    Extracts a slice type (e.g., 'axial') from a filename.
    """
    for slice_type in ['axial', 'coronal', 'sagittal']:
        if f"_{slice_type}_" in filename:
            return slice_type
    return None


def extract_coordinate_position(filename: str, coord_name: str) -> int:
    """
    Extracts a coordinate position (e.g., 123 from '..._x123.nii.gz').
    """
    try:
        coord_part = filename.split(f'_{coord_name}')[1].split('.')[0]
        return int(coord_part)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not extract {coord_name.upper()} position from filename: {filename}") from e

# ---------------------------------------------------------------------
# 2. IMAGE MANIPULATION HELPERS
# ---------------------------------------------------------------------

def normalize_to_uint8(data: np.ndarray, p_range: tuple = (1, 99)) -> np.ndarray:
    """
    Normalizes image data to 0-255 uint8 range based on intensity percentiles.
    This is ideal for converting float data from NIfTI to visual PNGs.

    Args:
        data (np.ndarray): The input image data.
        p_range (tuple): The lower and upper percentile to clip intensities.

    Returns:
        np.ndarray: The normalized image as uint8.
    """
    p_low, p_high = np.percentile(data, p_range)
    data_clipped = np.clip(data, p_low, p_high)
    
    # Avoid division by zero if p_high == p_low
    denominator = p_high - p_low
    if denominator < 1e-8:
        denominator = 1e-8
        
    data_normalized = (data_clipped - p_low) / denominator
    return (data_normalized * 255).astype(np.uint8)


def resize_image(image: Image.Image, target_size: tuple = (256, 256), method=Image.Resampling.LANCZOS) -> Image.Image:
    """
    Resizes a PIL Image to a target size using a specified interpolation method.
    """
    return image.resize(target_size, method)


def get_brain_mask(image: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """
    Creates a boolean mask to separate the brain from the background.
    Assumes background pixels are at or near zero.

    Args:
        image (np.ndarray): A float image array, typically in [0, 1] range.
        threshold (float): Pixels below this value are considered background.

    Returns:
        np.ndarray: A boolean mask where True indicates brain tissue.
    """
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    return image > threshold
