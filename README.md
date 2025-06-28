# ADNI MRI Processing Pipeline

A comprehensive Python pipeline for processing Alzheimer's Disease Neuroimaging Initiative (ADNI) MRI data, designed to support both temporal sequence analysis and single timepoint studies. This pipeline handles the complete workflow from raw DICOM data to enhanced, balanced datasets ready for deep learning applications.

## Overview

This pipeline processes longitudinal MRI data from ADNI subjects with maximum flexibility:
- **Dual-purpose design**: Supports both temporal (multi-timepoint) and non-temporal (single timepoint) model development
- **Temporal consistency**: Maintains subject identity across multiple timepoints (screening, 6-month, 12-month)
- **Multi-plane analysis**: Processes axial, coronal, and sagittal views comprehensively
- **Automated preprocessing**: Skull stripping, ROI registration, and optimal slice extraction
- **Advanced enhancement**: Grey Wolf Optimizer (GWO) based image enhancement
- **Smart augmentation**: Temporally consistent data augmentation for class balancing

## Key Feature: Temporal & Non-Temporal Support

This pipeline is uniquely designed to support both research approaches:

### For Temporal Models
- Use the complete output with all three timepoints (sc, m06, m12)
- Maintains temporal consistency across all processing steps
- Perfect for longitudinal studies and disease progression analysis

### For Non-Temporal Models
- Simply use only the screening (`sc`) visit data
- All processing steps maintain individual timepoint quality
- Ideal for cross-sectional studies or baseline analysis

The pipeline processes ALL data comprehensively, giving you the flexibility to choose your approach after processing is complete!

## Prerequisites

- **Python 3.8+**
- **FSL (FMRIB Software Library)** - Required for skull stripping and registration
  - Install from: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
  - Ensure `bet` and `flirt` commands are accessible
- **ADNI Dataset** - 1.5T MRI scans with metadata CSV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adni-mri-pipeline.git
cd adni-mri-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure FSL paths in `processing_pipeline/configs/config.py`:
```python
FSL_DIR = Path("/path/to/your/fsl")
FSL_BIN_DIR = FSL_DIR / "bin"
```

## Configuration

Edit `processing_pipeline/configs/config.py` to set:

- **Data paths**: Location of raw ADNI data and output directories
- **Processing parameters**: 
  - Required visits: `["sc", "m06", "m12"]`
  - Classes to include: `["AD", "CN"]` (or add "MCI")
  - Target image size: `(256, 256)`
  - Enhancement parameters
- **Computational resources**: `MAX_THREADS` for parallel processing

## Pipeline Steps

The pipeline consists of 9 sequential steps:

### 1. **Dataset Splitting** (`split`)
- Stratified train/val/test split (70/15/15)
- Ensures temporal completeness per subject
- Filters subjects with all required timepoints

### 2. **Skull Stripping** (`nifti` - part 1)
- Uses FSL's BET (Brain Extraction Tool)
- Parallel processing with configurable threads
- Robust error handling with timeouts

### 3. **ROI Registration** (`nifti` - part 2)
- Registers hippocampal ROI template to subject space
- Uses FSL's FLIRT for affine registration
- Maintains temporal consistency

### 4. **Slice Extraction** (`nifti` - part 3)
- Extracts optimal 2D slices based on ROI center
- Processes axial, coronal, and sagittal planes
- Preserves NIFTI format for compatibility

### 5. **Data Organization** (`convert` - part 1)
- Organizes slices by split/class/subject structure
- Enforces temporal sequence integrity
- Prepares for 2D conversion

### 6. **2D Conversion** (`convert` - part 2)
- Converts NIFTI slices to PNG format
- Normalizes intensities (1-99 percentile)
- Resizes to target dimensions

### 7. **Image Cropping** (`crop`)
- Center crops to brain region
- Adds configurable padding
- Applies rotation for standard orientation

### 8. **GWO Enhancement** (`enhance`)
- Grey Wolf Optimizer finds optimal enhancement parameters
- Combines CLAHE, Gabor filtering, and unsharp masking
- Mask-aware processing (brain region only)

### 9. **Dataset Balancing** (`balance`)
- Augments minority classes to balance training set
- Temporally consistent augmentation across timepoints
- Preserves validation/test set integrity

### 10. **Quality Control** (`qc`)
- Generates visualization reports
- Shows sample images across processing stages
- Provides distribution statistics

## Usage

### Run the complete pipeline:
```bash
python processing_pipeline/main_pipeline.py --step all
```

### Run individual steps:
```bash
# Initial data splitting
python processing_pipeline/main_pipeline.py --step split

# NIFTI processing (skull strip, registration, extraction)
python processing_pipeline/main_pipeline.py --step nifti

# 2D conversion and organization
python processing_pipeline/main_pipeline.py --step convert

# Cropping
python processing_pipeline/main_pipeline.py --step crop

# Enhancement
python processing_pipeline/main_pipeline.py --step enhance

# Balancing
python processing_pipeline/main_pipeline.py --step balance

# Quality control
python processing_pipeline/main_pipeline.py --step qc
```

### Force re-run a completed step:
```bash
python processing_pipeline/main_pipeline.py --step enhance --force
```

## Output Structure

The pipeline produces a complete, organized output with three separate directories for each anatomical plane:

```
datasets/ADNI_1_5_T/
├── 1_splitted_sequential/      # Raw NIFTI files organized by split
├── 2_skull_stripping_sequential/  # Brain-extracted NIFTI files
├── 3_roi_subj_space_sequential/   # ROI masks and transformation matrices
├── 4_optimal_*_sequential/        # Extracted 2D slices (3 directories)
├── 5_labelling_sequential_variable/  # Organized by class labels
├── 6_2Dconverted_sequential/      # PNG format images
├── 7_cropped/                     # Cropped and rotated images
├── 8_enhanced/                    # GWO-enhanced images
└── 9_balanced/                    # Final balanced dataset
    ├── axial/                     # Complete axial plane data
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── coronal/                   # Complete coronal plane data
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── sagittal/                  # Complete sagittal plane data
        ├── train/
        ├── val/
        └── test/
```

## Flexible Data Usage

After pipeline completion, you have maximum flexibility:

### For Temporal Models:
```python
# Use all timepoints from your chosen plane
data_path = "datasets/ADNI_1_5_T/9_balanced/axial/"
# Each subject will have: subjectID_sc.png, subjectID_m06.png, subjectID_m12.png
```

### For Non-Temporal Models:
```python
# Use only screening timepoint
data_path = "datasets/ADNI_1_5_T/9_balanced/coronal/"
# Filter for only *_sc.png files
```

### Multi-Plane Analysis:
```python
# Combine data from all three planes
axial_data = "datasets/ADNI_1_5_T/9_balanced/axial/"
coronal_data = "datasets/ADNI_1_5_T/9_balanced/coronal/"
sagittal_data = "datasets/ADNI_1_5_T/9_balanced/sagittal/"
```

## Pipeline Status

The pipeline tracks completion status in `.pipeline_status.json`. Steps are automatically skipped if already completed unless `--force` is used.

## Logging

Detailed logs are saved in `logs/` directory with timestamps:
- Each step creates its own log file
- Progress bars show real-time status
- Comprehensive error reporting

## Important Notes

- **Memory Requirements**: ~16GB RAM recommended for parallel processing
- **Storage**: Ensure sufficient disk space (~50GB for full pipeline)
- **Processing Time**: Complete pipeline takes 4-8 hours depending on hardware
- **Data Integrity**: Pipeline strictly enforces temporal consistency - subjects missing any timepoint are excluded
- **Flexibility**: Although the pipeline ensures temporal consistency, the output supports both temporal and non-temporal usage

## Training Pipeline

**Note: The training pipeline is currently under development!**

The `training_pipeline/` directory will contain:
- Multi-modal CNN architectures
- Temporal sequence models (LSTM, Transformer-based)
- Single timepoint models
- Cross-validation strategies
- Performance evaluation metrics
- Model comparison frameworks

Stay tuned for updates!

## Sample Results

After running the pipeline, you'll have:
- Balanced dataset with 180 subjects per class (AD/CN) in training
- 3 timepoints × 3 planes = 9 images per subject
- Complete flexibility to use:
  - All 9 images (full temporal, multi-plane)
  - 3 images per plane (temporal, single plane)
  - 1 image per plane (non-temporal, screening only)
  - Any combination that suits your research
- Consistent preprocessing across all images
- Quality control visualizations in `visualizations/`

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Citation

If you use this pipeline in your research, please cite:
```
[Your citation information here]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ADNI (Alzheimer's Disease Neuroimaging Initiative) for providing the dataset
- FSL developers for neuroimaging tools
- All contributors to the scientific Python ecosystem

---

For questions or issues, please open a GitHub issue or contact izzat.zaky@gmail.com