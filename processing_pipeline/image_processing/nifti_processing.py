# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 2, 3, 4: NIFTI PROCESSING
# =====================================================================
# This module handles the core 3D NIfTI processing steps:
# 1. Skull Stripping using FSL's BET.
# 2. ROI Registration using FSL's FLIRT.
# 3. Optimal 2D Slice Extraction based on the registered ROI.

import os
import subprocess
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from tqdm import tqdm

# Import configurations
from configs import config

# --- Setup FSL Environment ---
# Ensure FSL executables are on the system's PATH
os.environ["FSLDIR"] = str(config.FSL_DIR)
os.environ["PATH"] = f"{config.FSL_BIN_DIR}:{os.environ['PATH']}"

# --- Thread-safe lock for directory creation ---
dir_lock = threading.Lock()

# =====================================================================
# PART A: SKULL STRIPPING
# =====================================================================

def _run_bet_on_file(task: tuple):
    """Worker function to run FSL's BET on a single NIfTI file."""
    split, subj, visit, nii_path, out_dir = task
    
    with dir_lock:
        out_dir.mkdir(parents=True, exist_ok=True)

    out_brain = out_dir / f"{subj}_{visit}_brain.nii.gz"
    if out_brain.exists():
        return "skipped"

    cmd = ["bet", str(nii_path), str(out_brain)] + config.BET_ARGS
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=config.BET_TIMEOUT, text=True)
        return "success"
    except subprocess.TimeoutExpired:
        return f"timeout ({config.BET_TIMEOUT}s)"
    except subprocess.CalledProcessError as e:
        return f"error_rc={e.returncode}"
    except Exception as e:
        return f"error_exc={e}"

def run_skull_stripping():
    """
    Identifies all NIfTI files from the split dataset and runs skull stripping
    in parallel using FSL's BET.
    """
    print("\n🧠 STEP 2: SKULL STRIPPING")
    print("=" * 60)

    tasks = []
    for split_dir in config.STEP1_SPLIT_DIR.iterdir():
        if not split_dir.is_dir(): continue
        split = split_dir.name
        for subj_dir in split_dir.iterdir():
            if not subj_dir.is_dir(): continue
            subj = subj_dir.name
            for nii_path in subj_dir.glob("*.nii"):
                visit = nii_path.stem.split('_')[-1]
                out_dir = config.STEP2_SKULLSTRIP_DIR / split / subj
                tasks.append((split, subj, visit, nii_path, out_dir))

    if not tasks:
        print("❌ No NIfTI files found in input directory. Aborting.")
        return

    print(f"🔍 Found {len(tasks)} NIfTI files to process.")
    print(f"🚀 Starting skull stripping with {config.MAX_THREADS} parallel threads...")

    stats = defaultdict(int)
    with ThreadPoolExecutor(max_workers=config.MAX_THREADS) as exe:
        futures = {exe.submit(_run_bet_on_file, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Skull-stripping"):
            stats[fut.result()] += 1

    print("\n📋 Skull Stripping Summary:")
    for k, v in stats.items():
        print(f"  - {k.capitalize()}: {v}")
    print("\n✅ Step 2: Skull stripping complete!")


# =====================================================================
# PART B: ROI REGISTRATION & WARPING
# =====================================================================

def _run_flirt_on_file(task: tuple):
    """Worker function to register ROI to a single brain file using FSL's FLIRT."""
    split, subj, visit, brain_path, out_dir = task
    
    with dir_lock:
        out_dir.mkdir(parents=True, exist_ok=True)

    base = f"{subj}_{visit}"
    mat = out_dir / f"{base}_subj2mni.mat"
    inv_mat = out_dir / f"{base}_mni2subj.mat"
    out_roi = out_dir / f"{base}_hippo_mask.nii.gz"

    if out_roi.exists():
        return "skipped"

    try:
        # Step 1: Register subject brain to MNI space (get transformation matrix)
        subprocess.run(["flirt", "-in", str(brain_path), "-ref", str(config.MNI_BRAIN_TEMPLATE), "-omat", str(mat)], check=True, capture_output=True)
        # Step 2: Invert the transformation matrix
        subprocess.run(["convert_xfm", "-omat", str(inv_mat), "-inverse", str(mat)], check=True, capture_output=True)
        # Step 3: Apply the inverse transform to warp the ROI template to subject space
        subprocess.run(["flirt", "-in", str(config.ROI_TEMPLATE), "-ref", str(brain_path), "-applyxfm", "-init", str(inv_mat), "-interp", "nearestneighbour", "-out", str(out_roi)], check=True, capture_output=True)
        return "success"
    except subprocess.CalledProcessError as e:
        return f"error_rc={e.returncode}"
    except Exception as e:
        return f"error_exc={e}"

def run_roi_registration():
    """
    Identifies all skull-stripped brains and registers the standard ROI
    template to each one in parallel.
    """
    print("\n🧠 STEP 3: ROI REGISTRATION & WARPING")
    print("=" * 60)

    tasks = []
    for split_dir in config.STEP2_SKULLSTRIP_DIR.iterdir():
        if not split_dir.is_dir(): continue
        split = split_dir.name
        for subj_dir in split_dir.iterdir():
            if not subj_dir.is_dir(): continue
            subj = subj_dir.name
            for brain_path in subj_dir.glob("*_brain.nii.gz"):
                visit = brain_path.stem.split('_')[-2]
                out_dir = config.STEP3_ROI_REG_DIR / split / subj
                tasks.append((split, subj, visit, brain_path, out_dir))
    
    if not tasks:
        print("❌ No skull-stripped files found. Aborting.")
        return

    print(f"🔍 Found {len(tasks)} brain files to process.")
    print(f"🚀 Starting ROI registration with {config.MAX_THREADS} parallel threads...")
    
    stats = defaultdict(int)
    with ThreadPoolExecutor(max_workers=config.MAX_THREADS) as exe:
        futures = {exe.submit(_run_flirt_on_file, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Warping ROI"):
            stats[fut.result()] += 1
            
    print("\n📋 ROI Registration Summary:")
    for k, v in stats.items():
        print(f"  - {k.capitalize()}: {v}")
    print("\n✅ Step 3: ROI registration complete!")


# =====================================================================
# PART C: OPTIMAL SLICE EXTRACTION
# =====================================================================

def _extract_slice(task: tuple):
    """Worker function to extract the optimal 2D slice from a 3D brain volume."""
    slice_type, split, subj, visit, roi_path, brain_path = task
    
    slice_configs = {
        "axial": {"root": config.STEP4_SLICES_AXIAL_DIR, "axis": 0, "coord": "x"},
        "coronal": {"root": config.STEP4_SLICES_CORONAL_DIR, "axis": 1, "coord": "y"},
        "sagittal": {"root": config.STEP4_SLICES_SAGITTAL_DIR, "axis": 2, "coord": "z"},
    }
    cfg = slice_configs[slice_type]
    out_dir = cfg["root"] / split / subj
    
    with dir_lock:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Simplified check for existence
    if list(out_dir.glob(f"{subj}_{visit}_optimal_{slice_type}_*.nii.gz")):
        return "skipped"

    try:
        roi_data = nib.load(str(roi_path)).get_fdata()
        coord_inds = np.where(roi_data > 0)[cfg["axis"]]
        if coord_inds.size == 0:
            return "empty_roi"
        
        coord_center = int(round(coord_inds.mean()))
        
        brain_img = nib.load(str(brain_path))
        brain_data = brain_img.get_fdata()

        if cfg["axis"] == 0: slice_2d = brain_data[coord_center, :, :]
        elif cfg["axis"] == 1: slice_2d = brain_data[:, coord_center, :]
        else: slice_2d = brain_data[:, :, coord_center]

        slice_img = nib.Nifti1Image(slice_2d[..., np.newaxis], brain_img.affine)
        out_file = out_dir / f"{subj}_{visit}_optimal_{slice_type}_{cfg['coord']}{coord_center}.nii.gz"
        nib.save(slice_img, str(out_file))
        return "success"
    except Exception as e:
        return f"error: {e}"

def run_slice_extraction():
    """
    For each subject and visit, finds the center of the hippocampal ROI
    and extracts the corresponding 2D slice from the brain image for all
    three anatomical planes (axial, coronal, sagittal).
    """
    print("\n🧠 STEP 4: OPTIMAL SLICE EXTRACTION")
    print("=" * 60)

    # Create output directories first
    for d in [config.STEP4_SLICES_AXIAL_DIR, config.STEP4_SLICES_CORONAL_DIR, config.STEP4_SLICES_SAGITTAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    tasks = []
    for split_dir in config.STEP3_ROI_REG_DIR.iterdir():
        if not split_dir.is_dir(): continue
        split = split_dir.name
        for subj_dir in split_dir.iterdir():
            if not subj_dir.is_dir(): continue
            subj = subj_dir.name
            for roi_path in subj_dir.glob("*_hippo_mask.nii.gz"):
                visit = roi_path.stem.split('_')[-3]
                brain_path = config.STEP2_SKULLSTRIP_DIR / split / subj / f"{subj}_{visit}_brain.nii.gz"
                if brain_path.exists():
                    for slice_type in ["axial", "coronal", "sagittal"]:
                        tasks.append((slice_type, split, subj, visit, roi_path, brain_path))
    
    if not tasks:
        print("❌ No ROI masks found. Aborting.")
        return

    print(f"🔍 Found {len(tasks)} slice extraction tasks to perform.")
    print(f"🚀 Starting slice extraction with {config.MAX_THREADS} parallel threads...")

    stats = defaultdict(int)
    with ThreadPoolExecutor(max_workers=config.MAX_THREADS) as exe:
        futures = {exe.submit(_extract_slice, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Extracting slices"):
            stats[fut.result()] += 1

    print("\n📋 Slice Extraction Summary:")
    for k, v in stats.items():
        print(f"  - {k.capitalize()}: {v}")
    print("\n✅ Step 4: Optimal slice extraction complete!")


if __name__ == '__main__':
    # This allows running each step individually if needed, e.g. for debugging
    # Make sure the previous step's output exists before running.
    print("Running NIfTI processing steps...")
    run_skull_stripping()
    run_roi_registration()
    run_slice_extraction()
    print("All NIfTI processing steps finished.")
