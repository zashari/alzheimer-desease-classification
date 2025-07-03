# =====================================================================
# ADNI MRI PROCESSING PIPELINE - MAIN EXECUTABLE SCRIPT
# =====================================================================
# This script orchestrates the entire ADNI data processing pipeline,
# from initial data splitting to final augmentation.
#
# Usage:
#   python main_pipeline.py --step all
#   python main_pipeline.py --step split
#   python main_pipeline.py --step nifti
#   python main_pipeline.py --step qc
#   python main_pipeline.py --step all --force  # Force re-run all steps
#
# See --help for more options.
# =====================================================================

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

# Import pipeline steps from their respective modules
from data_management import dataset_splitter, data_organization, data_visualization
from image_processing import nifti_processing, image_conversion, image_enhancement, image_cropping, image_balancing
from configs import config
from utils.logging_utils import setup_logging

class PipelineStepTracker:
    def __init__(self, pipeline_root):
        self.pipeline_root = Path(pipeline_root)
        self.status_file = self.pipeline_root / '.pipeline_status.json'
        self._load_status()

    def _load_status(self):
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                self.status = json.load(f)
        else:
            self.status = {}

    def _save_status(self):
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def is_step_completed(self, step):
        return self.status.get(step, False)

    def mark_step_completed(self, step):
        self.status[step] = True
        self._save_status()

    def verify_step_completion(self, step):
        """Verify if a step is truly complete by checking its output."""
        if step == 'split':
            return (config.STEP1_SPLIT_DIR / "metadata_split.csv").exists()
        elif step == 'nifti':
            return all(d.exists() for d in [
                config.STEP2_SKULLSTRIP_DIR,
                config.STEP3_ROI_REG_DIR,
                config.STEP4_SLICES_AXIAL_DIR
            ])
        elif step == 'convert':
            return config.STEP6_2D_CONVERTED_DIR.exists()
        elif step == 'crop':
            return config.STEP7_CROPPED_DIR.exists()
        elif step == 'enhance':
            return config.STEP8_ENHANCED_DIR.exists()
        elif step == 'balance':
            return config.STEP9_BALANCED_DIR.exists()
        return False

def setup_step_logging(step_name):
    """Configure logging for a specific pipeline step."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"{step_name}_{timestamp}.log"
    return setup_logging(f"pipeline_{step_name}", log_file)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ADNI MRI Processing Pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "split", "nifti", "convert", "crop", "enhance", "balance", "qc"],
        default="all",
        help="Pipeline step to execute"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of steps even if marked as completed"
    )
    return parser.parse_args()

def main():
    """
    Main pipeline execution function.
    
    This orchestrates the entire processing pipeline, allowing execution
    of individual steps or the complete pipeline based on command line
    arguments.
    """
    args = parse_args()
    
    # Create necessary directories
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize step tracker
    tracker = PipelineStepTracker(config.PROJECT_ROOT)
    
    try:
        steps_to_run = ['split', 'nifti', 'convert', 'crop', 'enhance', 'balance', 'qc'] if args.step == 'all' else [args.step]
        
        for step in steps_to_run:
            # Skip if step is completed and not forced
            if tracker.is_step_completed(step) and tracker.verify_step_completion(step) and not args.force:
                print(f"Skipping completed step: {step}")
                continue
                
            logger = setup_step_logging(step)
            logger.info(f"Starting {step} step")
            
            if step == 'split':
                with tqdm(total=1, desc="Splitting dataset", mininterval=0.1) as pbar:
                    dataset_splitter.split_dataset(logger)
                    pbar.update(1)
                logger.info("Dataset splitting completed")
                tracker.mark_step_completed(step)
                
            elif step == 'nifti':
                with tqdm(total=3, desc="Processing NIfTI files", mininterval=0.1) as pbar:
                    nifti_processing.run_skull_stripping(logger)
                    pbar.update(1)
                    nifti_processing.run_roi_registration(logger)
                    pbar.update(1)
                    nifti_processing.run_slice_extraction(logger)
                    pbar.update(1)
                logger.info("NIfTI processing completed")
                tracker.mark_step_completed(step)
                
            elif step == 'convert':
                with tqdm(total=2, desc="Converting images", mininterval=0.1) as pbar:
                    # First run data organization
                    data_organization.organize_slices_by_label(logger)
                    pbar.update(1)
                    # Then run image conversion
                    image_conversion.convert_nifti_to_png(logger)
                    pbar.update(1)
                logger.info("Image conversion completed")
                tracker.mark_step_completed(step)
                
            elif step == 'crop':
                with tqdm(total=1, desc="Cropping images", mininterval=0.1) as pbar:
                    image_cropping.crop_images(logger)
                    pbar.update(1)
                logger.info("Image cropping completed")
                tracker.mark_step_completed(step)
                
            elif step == 'enhance':
                with tqdm(total=1, desc="Enhancing images", mininterval=0.1) as pbar:
                    image_enhancement.enhance_dataset(logger)
                    pbar.update(1)
                logger.info("Image enhancement completed")
                tracker.mark_step_completed(step)
                
            elif step == 'balance':
                with tqdm(total=1, desc="Balancing dataset", mininterval=0.1) as pbar:
                    image_balancing.balance_dataset(logger)
                    pbar.update(1)
                logger.info("Data balancing completed")
                tracker.mark_step_completed(step)
                
            elif step == 'qc':
                logger.info("Starting quality control visualization")
                data_visualization.generate_qc_report(logger)
                logger.info("Quality control visualization completed")
                tracker.mark_step_completed(step)
            
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
