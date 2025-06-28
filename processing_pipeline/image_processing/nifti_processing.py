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
# PART A: SKULL STRIPPING (LOGIC UPDATED FROM NOTEBOOK)
# =====================================================================

def _run_bet_on_file(task: tuple):
    """
    Worker function to run FSL's BET on a single NIfTI file.
    Output naming is preserved: {subj}_{visit}_brain.nii.gz
    """
    split, subj, visit, nii_path = task
    
    out_dir = config.STEP2_SKULLSTRIP_DIR / split / subj
    with dir_lock:
        out_dir.mkdir(parents=True, exist_ok=True)

    out_brain = out_dir / f"{subj}_{visit}_brain.nii.gz"
    if out_brain.exists():
        return "skipped"

    cmd = ["bet", str(nii_path), str(out_brain)] + config.BET_ARGS
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=config.BET_TIMEOUT,
            text=True
        )
        return "success"
    except subprocess.TimeoutExpired:
        return f"timeout"
    except subprocess.CalledProcessError as e:
        return f"error_rc={e.returncode}"
    except Exception as e:
        return f"error_exc={e}"

def run_skull_stripping(logger):
    """
    Identifies NIfTI files from the split dataset, validates their temporal
    sequence, and runs skull stripping in parallel using FSL's BET.
    
    Args:
        logger: A configured logger instance for output.
    """
    logger.info("\n🧠 STEP 2: SKULL STRIPPING")
    logger.info("=" * 80)

    # Verify FSL installation before starting
    try:
        which = subprocess.run(["which", "bet"], capture_output=True, text=True, check=True)
        logger.info(f"✅ Found FSL BET executable at: {which.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ CRITICAL: FSL 'bet' command not found on system PATH.")
        logger.error(f"   Please ensure FSL is installed and FSL_BIN_DIR is set correctly in config.py")
        return

    # --- Build Task List with validation ---
    logger.info("🔍 Scanning for sequential data and building task list...")
    tasks = []
    sequential_stats = defaultdict(int)
    missing_files = []

    for split in ("train", "val", "test"):
        split_dir = config.STEP1_SPLIT_DIR / split
        if not split_dir.exists():
            logger.warning(f"⚠️ Split directory not found, skipping: {split_dir}")
            continue

        split_subjects = 0
        split_complete = 0
        split_files_found = 0
        
        subject_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        logger.info(f"\n📊 Processing {split.upper()} split ({len(subject_dirs)} subjects):")
        
        for subj_dir in subject_dirs:
            subj = subj_dir.name
            split_subjects += 1
            
            found_visits = []
            subject_files_to_add = []
            
            for visit in config.REQUIRED_VISITS:
                # Prioritize simplified naming convention
                expected_file = subj_dir / f"{subj}_{visit}.nii"
                if expected_file.exists():
                    found_visits.append(visit)
                    subject_files_to_add.append((split, subj, visit, expected_file))
                else:
                    missing_files.append(f"{split}/{subj}/{visit}")

            tasks.extend(subject_files_to_add)
            split_files_found += len(subject_files_to_add)

            if set(found_visits) == set(config.REQUIRED_VISITS):
                split_complete += 1
            else:
                missing = sorted(list(set(config.REQUIRED_VISITS) - set(found_visits)))
                logger.warning(f"  - Incomplete sequence for {subj}: missing {missing}")
        
        sequential_stats['total_subjects'] += split_subjects
        sequential_stats['complete_sequences'] += split_complete
        logger.info(f"  • Found {split_complete} subjects with complete sequences out of {split_subjects}.")

    sequential_stats['incomplete_sequences'] = sequential_stats['total_subjects'] - sequential_stats['complete_sequences']
    
    logger.info("\n" + "-" * 40)
    logger.info("📈 OVERALL INPUT STATISTICS:")
    logger.info(f"  • Total subjects found: {sequential_stats['total_subjects']}")
    logger.info(f"  • Complete sequences:   {sequential_stats['complete_sequences']}")
    logger.info(f"  • Incomplete sequences: {sequential_stats['incomplete_sequences']}")
    logger.info(f"  • Total files to process: {len(tasks)}")
    logger.info("-" * 40)

    if not tasks:
        logger.error("❌ No NIfTI files found to process. Check input directory structure. Aborting.")
        return

    # --- Parallel Execution ---
    logger.info(f"🚀 Starting skull stripping with up to {config.MAX_THREADS} parallel threads...")
    stats = defaultdict(int)
    with ThreadPoolExecutor(max_workers=config.MAX_THREADS) as exe:
        futures = {exe.submit(_run_bet_on_file, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Skull-stripping"):
            result = fut.result()
            if result.startswith("error"):
                stats["error"] += 1
            elif result.startswith("timeout"):
                stats["timeout"] += 1
            else:
                stats[result] += 1
    
    # --- Detailed Summary ---
    logger.info("\n" + "=" * 80)
    logger.info("📊 SKULL STRIPPING SUMMARY")
    logger.info("=" * 80)

    logger.info("\n⚡ Processing Results:")
    total_processed = sum(stats.values())
    for k, v in stats.items():
        percentage = (v / total_processed * 100) if total_processed > 0 else 0
        logger.info(f"  - {k.capitalize():<10}: {v:>4d} ({percentage:5.1f}%)")

    logger.info("\n📁 Output Structure Verification:")
    for split in ("train", "val", "test"):
        split_out = config.STEP2_SKULLSTRIP_DIR / split
        if split_out.exists():
            subjects_in_out = [d for d in split_out.iterdir() if d.is_dir()]
            total_brain_files = len(list(split_out.rglob("*_brain.nii.gz")))
            logger.info(f"  - {split:<5}: {len(subjects_in_out):>3d} subjects, {total_brain_files:>4d} brain files created.")
    
    logger.info("\n✅ Step 2: Skull stripping complete!")
    logger.info(f"📂 Output directory: {config.STEP2_SKULLSTRIP_DIR}")
    logger.info("="*80)


# =====================================================================
# PART B: ROI REGISTRATION & WARPING (LOGIC UPDATED FROM NOTEBOOK)
# =====================================================================

def _run_roi_registration_on_file(task: tuple):
    """
    Worker function to register ROI to a single brain file using FSL's FLIRT.
    task = (split, subj, visit, brain_path)
    """
    split, subj, visit, brain_path = task
    
    out_dir = config.STEP3_ROI_REG_DIR / split / subj
    with dir_lock:
        out_dir.mkdir(parents=True, exist_ok=True)

    base = f"{subj}_{visit}"
    mat = out_dir / f"{base}_subj2mni.mat"
    inv_mat = out_dir / f"{base}_mni2subj.mat"
    out_roi = out_dir / f"{base}_hippo_mask.nii.gz"

    if out_roi.exists() and mat.exists() and inv_mat.exists():
        return "skipped"

    try:
        # Step 1: Register subject brain to MNI space (get transformation matrix)
        cmd_flirt1 = ["flirt", "-in", str(brain_path), "-ref", str(config.MNI_BRAIN_TEMPLATE), "-omat", str(mat)]
        subprocess.run(cmd_flirt1, check=True, capture_output=True, text=True)
        
        # Step 2: Invert the transformation matrix
        cmd_inv = ["convert_xfm", "-omat", str(inv_mat), "-inverse", str(mat)]
        subprocess.run(cmd_inv, check=True, capture_output=True, text=True)
        
        # Step 3: Apply the inverse transform to warp the ROI template to subject space
        cmd_flirt2 = ["flirt", "-in", str(config.ROI_TEMPLATE), "-ref", str(brain_path), "-applyxfm", "-init", str(inv_mat), "-interp", "nearestneighbour", "-out", str(out_roi)]
        subprocess.run(cmd_flirt2, check=True, capture_output=True, text=True)
        
        return "success"
    except subprocess.CalledProcessError as e:
        return f"error_rc={e.returncode}"
    except Exception as e:
        return f"error_exc={e}"

def run_roi_registration(logger):
    """
    Identifies skull-stripped brains, STRICTLY filters for subjects with
    complete temporal sequences, and registers the standard ROI template to each one.
    
    Args:
        logger: A configured logger instance for output.
    """
    logger.info("\n🧠 STEP 3: ROI REGISTRATION & WARPING")
    logger.info("=" * 80)
    logger.info("   STRICT MODE: Subjects without a complete temporal sequence will be EXCLUDED.")
    logger.info("-" * 80)

    # --- Build Task List with STRICT validation ---
    logger.info("🔍 Scanning for complete skull-stripped sequences...")
    tasks = []
    stats = defaultdict(int)
    excluded_subjects = []
    
    for split in ("train", "val", "test"):
        split_dir = config.STEP2_SKULLSTRIP_DIR / split
        if not split_dir.exists():
            logger.warning(f"⚠️ Skull-stripped directory for '{split}' not found, skipping.")
            continue
        
        subject_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        logger.info(f"\n📊 Processing {split.upper()} split ({len(subject_dirs)} subjects):")
        stats['total_subjects_scanned'] += len(subject_dirs)
        
        for subj_dir in subject_dirs:
            subj = subj_dir.name
            
            # Find all required brain files for this subject
            subject_files_to_add = []
            is_complete = True
            for visit in config.REQUIRED_VISITS:
                brain_file = subj_dir / f"{subj}_{visit}_brain.nii.gz"
                if brain_file.exists():
                    subject_files_to_add.append((split, subj, visit, brain_file))
                else:
                    is_complete = False
                    break # No need to check other visits if one is missing
            
            # Only add tasks if the ENTIRE sequence is present
            if is_complete:
                tasks.extend(subject_files_to_add)
                stats['complete_sequences_found'] += 1
            else:
                excluded_subjects.append(f"{split}/{subj}")
                stats['incomplete_sequences_excluded'] += 1

    retention_rate = (stats['complete_sequences_found'] / stats['total_subjects_scanned'] * 100) if stats['total_subjects_scanned'] > 0 else 0
    logger.info("\n" + "-" * 40)
    logger.info("📈 INPUT VALIDATION SUMMARY:")
    logger.info(f"  • Total subjects scanned:            {stats['total_subjects_scanned']}")
    logger.info(f"  • Subjects with complete sequences:  {stats['complete_sequences_found']}")
    logger.info(f"  • Subjects excluded (incomplete):    {stats['incomplete_sequences_excluded']}")
    logger.info(f"  • Data retention rate for this step: {retention_rate:.1f}%")
    logger.info(f"  • Total brain files to process:      {len(tasks)}")
    logger.info("-" * 40)

    if excluded_subjects:
        logger.warning(f"❌ Excluded {len(excluded_subjects)} subjects due to missing files:")
        for subj_path in excluded_subjects[:10]:
             logger.warning(f"  - {subj_path}")
        if len(excluded_subjects) > 10:
             logger.warning(f"  ... and {len(excluded_subjects) - 10} more.")

    if not tasks:
        logger.error("❌ No subjects with complete sequences found. Aborting ROI registration.")
        return

    # --- Parallel Execution ---
    logger.info(f"🚀 Starting ROI registration with up to {config.MAX_THREADS} parallel threads...")
    proc_stats = defaultdict(int)
    with ThreadPoolExecutor(max_workers=config.MAX_THREADS) as exe:
        futures = {exe.submit(_run_roi_registration_on_file, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Warping ROI"):
            result = fut.result()
            proc_stats[result.split('_')[0]] += 1
            
    # --- Detailed Summary ---
    logger.info("\n" + "=" * 80)
    logger.info("📊 ROI REGISTRATION SUMMARY")
    logger.info("=" * 80)
    
    logger.info("\n⚡ Processing Results:")
    total_processed = sum(proc_stats.values())
    for k, v in proc_stats.items():
        percentage = (v / total_processed * 100) if total_processed > 0 else 0
        logger.info(f"  - {k.capitalize():<10}: {v:>4d} ({percentage:5.1f}%)")
        
    logger.info("\n📁 Output Structure Verification:")
    for split in ("train", "val", "test"):
        split_out = config.STEP3_ROI_REG_DIR / split
        if split_out.exists():
            subjects_out = [d for d in split_out.iterdir() if d.is_dir()]
            total_rois = len(list(split_out.rglob("*_hippo_mask.nii.gz")))
            total_mats = len(list(split_out.rglob("*.mat")))
            logger.info(f"  - {split:<5}: {len(subjects_out):>3d} subjects, {total_rois:>4d} ROI masks, {total_mats:>4d} matrix files.")

    logger.info("\n✅ Step 3: ROI registration complete!")
    logger.info(f"📂 Output directory: {config.STEP3_ROI_REG_DIR}")
    logger.info("="*80)


# =====================================================================
# PART C: OPTIMAL SLICE EXTRACTION (LOGIC UPDATED FROM NOTEBOOK)
# =====================================================================

def _extract_slice_worker(task: tuple, slice_config: dict):
    """
    Worker function to extract the optimal 2D slice from a 3D brain volume.
    task = (split, subj, visit, roi_path, brain_path)
    """
    split, subj, visit, roi_path, brain_path = task
    
    out_dir = slice_config["root"] / split / subj
    with dir_lock:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing file
    coord = slice_config["coord"]
    slice_type = slice_config["name"]
    if list(out_dir.glob(f"{subj}_{visit}_optimal_{slice_type}_{coord}*.nii.gz")):
        return "skipped"

    try:
        roi_img = nib.load(str(roi_path))
        brain_img = nib.load(str(brain_path))
        
        # Ensure consistent orientation
        roi_img_ras = nib.as_closest_canonical(roi_img)
        brain_img_ras = nib.as_closest_canonical(brain_img)
        
        roi_data = roi_img_ras.get_fdata()
        brain_data = brain_img_ras.get_fdata()

        # Find center of mass of ROI
        axis = slice_config["axis"]
        coord_inds = np.where(roi_data > 0)[axis]
        if coord_inds.size == 0:
            return "empty_roi"
        
        coord_center = int(round(coord_inds.mean()))
        
        # Extract 2D slice using the numpy slice function
        slice_2d = slice_config["slice_func"](brain_data, coord_center)
        
        # Create a new NIfTI image for the 2D slice
        # Transpose for correct radiological orientation in many viewers (axial/coronal)
        slice_3d = slice_2d.T[..., np.newaxis] 
        slice_nifti = nib.Nifti1Image(slice_3d, np.eye(4)) # Use identity affine for 2D images
        
        out_file = out_dir / f"{subj}_{visit}_optimal_{slice_type}_{coord}{coord_center}.nii.gz"
        nib.save(slice_nifti, str(out_file))
        return "success"
    except Exception as e:
        return f"error: {e}"

def run_slice_extraction(logger):
    """
    Finds matching ROI/brain pairs, STRICTLY filters for complete temporal
    sequences, and extracts the optimal 2D slice for axial, coronal, and
    sagittal views in parallel.
    
    Args:
        logger: A configured logger instance for output.
    """
    logger.info("\n🧠 STEP 4: OPTIMAL SLICE EXTRACTION")
    logger.info("=" * 80)
    logger.info("   STRICT MODE: Subjects without a complete set of ROI/brain files will be EXCLUDED.")
    
    slice_configs = {
        "axial": {
            "name": "axial", "root": config.STEP4_SLICES_AXIAL_DIR, "axis": 2, "coord": "z",
            "slice_func": lambda data, center: data[:, :, center]
        },
        "coronal": {
            "name": "coronal", "root": config.STEP4_SLICES_CORONAL_DIR, "axis": 1, "coord": "y",
            "slice_func": lambda data, center: data[:, center, :]
        },
        "sagittal": {
            "name": "sagittal", "root": config.STEP4_SLICES_SAGITTAL_DIR, "axis": 0, "coord": "x",
            "slice_func": lambda data, center: data[center, :, :]
        }
    }
    # Create output directories
    for cfg in slice_configs.values():
        cfg["root"].mkdir(parents=True, exist_ok=True)
    
    # --- Gather Tasks with STRICT validation ---
    logger.info("\n🔍 Scanning for complete temporal ROI and brain file pairs...")
    tasks = []
    stats = defaultdict(int)
    missing_pairs = []

    for split in ("train", "val", "test"):
        roi_split_dir = config.STEP3_ROI_REG_DIR / split
        brain_split_dir = config.STEP2_SKULLSTRIP_DIR / split

        if not roi_split_dir.exists() or not brain_split_dir.exists():
            logger.warning(f"⚠️ Input directory for '{split}' not found, skipping.")
            continue
        
        subject_dirs = [d for d in roi_split_dir.iterdir() if d.is_dir()]
        stats['total_subjects_scanned'] += len(subject_dirs)
        
        for subj_dir in subject_dirs:
            subj = subj_dir.name
            
            subject_files_to_add = []
            is_complete = True
            for visit in config.REQUIRED_VISITS:
                roi_file = subj_dir / f"{subj}_{visit}_hippo_mask.nii.gz"
                brain_file = brain_split_dir / subj / f"{subj}_{visit}_brain.nii.gz"
                
                if roi_file.exists() and brain_file.exists():
                    subject_files_to_add.append((split, subj, visit, roi_file, brain_file))
                else:
                    is_complete = False
                    if not roi_file.exists(): missing_pairs.append(f"{split}/{subj}/{visit} (ROI)")
                    if not brain_file.exists(): missing_pairs.append(f"{split}/{subj}/{visit} (Brain)")
                    break
            
            if is_complete:
                tasks.extend(subject_files_to_add)
                stats['complete_sequences_found'] += 1
            else:
                stats['incomplete_sequences_excluded'] += 1

    retention_rate = (stats['complete_sequences_found'] / stats['total_subjects_scanned'] * 100) if stats['total_subjects_scanned'] > 0 else 0
    logger.info("\n" + "-" * 40)
    logger.info("📈 INPUT VALIDATION SUMMARY:")
    logger.info(f"  • Total subjects scanned:            {stats['total_subjects_scanned']}")
    logger.info(f"  • Subjects with complete sequences:  {stats['complete_sequences_found']}")
    logger.info(f"  • Subjects excluded (incomplete):    {stats['incomplete_sequences_excluded']}")
    logger.info(f"  • Data retention rate for this step: {retention_rate:.1f}%")
    logger.info(f"  • Total extraction tasks:            {len(tasks) * 3} ({len(tasks)} per plane)")
    logger.info("-" * 40)
    
    if missing_pairs:
        logger.warning(f"❌ Found {len(missing_pairs)} missing file pairs causing exclusions:")
        for pair in missing_pairs[:10]:
             logger.warning(f"  - {pair}")
        if len(missing_pairs) > 10:
             logger.warning(f"  ... and {len(missing_pairs) - 10} more.")

    if not tasks:
        logger.error("❌ No subjects with complete file sequences found. Aborting slice extraction.")
        return

    # --- Parallel Execution for each slice type ---
    overall_stats = {}
    for slice_type, cfg in slice_configs.items():
        logger.info(f"\n🚀 Processing {slice_type.upper()} slices with up to {config.MAX_THREADS} threads...")
        proc_stats = defaultdict(int)
        
        with ThreadPoolExecutor(max_workers=config.MAX_THREADS) as exe:
            # Create a future for each task for the current slice type
            futures = {exe.submit(_extract_slice_worker, task, cfg): task for task in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc=f"Extracting {slice_type}"):
                result = fut.result()
                proc_stats[result.split(':')[0]] += 1
        
        overall_stats[slice_type] = proc_stats

    # --- Detailed Summary and Verification ---
    logger.info("\n" + "=" * 80)
    logger.info("📊 OPTIMAL SLICE EXTRACTION SUMMARY")
    logger.info("=" * 80)
    for slice_type, stats_dict in overall_stats.items():
        total = sum(stats_dict.values())
        success_rate = (stats_dict.get("success", 0) / total * 100) if total > 0 else 0
        logger.info(f"\n⚡ {slice_type.upper()} Results ({total} tasks):")
        logger.info(f"  - Success:    {stats_dict.get('success', 0)} ({success_rate:.1f}%)")
        logger.info(f"  - Skipped:    {stats_dict.get('skipped', 0)}")
        logger.info(f"  - Empty ROI:  {stats_dict.get('empty_roi', 0)}")
        logger.info(f"  - Errors:     {stats_dict.get('error', 0)}")

    logger.info("\n✅ Step 4: Optimal slice extraction complete!")
    logger.info(f"📂 Output directories:")
    for cfg in slice_configs.values():
        logger.info(f"  - {cfg['name'].capitalize()}: {cfg['root']}")
    logger.info("="*80)

if __name__ == '__main__':
    from utils.logging_utils import setup_logging
    
    main_logger = setup_logging("nifti_processing_standalone", config.LOG_DIR)
    
    # You can comment out steps you don't want to run
    run_skull_stripping(logger=main_logger)
    run_roi_registration(logger=main_logger)
    run_slice_extraction(logger=main_logger)
    
    main_logger.info("All NIfTI processing steps finished.")

