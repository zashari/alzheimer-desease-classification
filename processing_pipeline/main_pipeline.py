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
#
# See --help for more options.
# =====================================================================

import argparse
import time

# Import pipeline steps from their respective modules
from data_management import dataset_splitter, data_organization, data_visualization
from image_processing import nifti_processing, image_conversion, image_enhancement

def main():
    """
    Main function to parse arguments and run the selected pipeline steps.
    """
    parser = argparse.ArgumentParser(
        description="Run the ADNI MRI processing pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=[
            "all", "split", "nifti", "organize", "convert",
            "crop", "enhance", "balance", "qc"
        ],
        help="""Which pipeline step to run:
        - all:      Run the entire pipeline from start to finish.
        - split:    Step 1: Perform initial data splitting.
        - nifti:    Steps 2-4: Run all NIfTI processing (skullstrip, ROI reg, slice extract).
        - organize: Step 5: Organize slices into labeled directories.
        - convert:  Step 6: Convert NIfTI slices to PNG.
        - crop:     Step 7: Crop PNG images.
        - enhance:  Step 8: Enhance images with GWO.
        - balance:  Step 9: Balance dataset with augmentation.
        - qc:       Run all quality control visualizations.
        """
    )

    args = parser.parse_args()
    step = args.step
    
    start_time = time.time()
    
    print(f"🚀 Starting ADNI Processing Pipeline. Selected step: '{step.upper()}'")
    print("=" * 70)

    if step in ["all", "split"]:
        dataset_splitter.split_dataset()

    if step in ["all", "nifti"]:
        nifti_processing.run_skull_stripping()
        nifti_processing.run_roi_registration()
        nifti_processing.run_slice_extraction()
        
    if step in ["all", "organize"]:
        data_organization.organize_slices_by_label()

    if step in ["all", "convert"]:
        image_conversion.convert_nifti_to_png()
        
    if step in ["all", "crop"]:
        image_enhancement.crop_images()
        
    if step in ["all", "enhance"]:
        image_enhancement.enhance_images_gwo()
        
    if step in ["all", "balance"]:
        # Note: The balance function in the module is a placeholder.
        # The full logic from the notebook needs to be implemented for it to work.
        image_enhancement.balance_dataset()

    if step == "qc":
        print("Running all Quality Control visualizations...")
        data_visualization.visualize_split_qc()
        data_visualization.visualize_enhancement_comparison()
        data_visualization.visualize_augmentation_results()
        data_visualization.generate_final_distribution_report()
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=" * 70)
    print(f"✅ Pipeline execution finished for step '{step.upper()}'.")
    print(f"Total time taken: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
