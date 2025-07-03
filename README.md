# Alzheimer's Disease Classification

This repository contains a comprehensive suite of tools for Alzheimer's disease classification, including a Python-based data processing pipeline and a React-based 3D data viewer.

## 🧠 Python Processing Pipeline

The `alzheimer-disease-classification` directory contains a robust Python pipeline for processing ADNI MRI data. It handles everything from raw DICOM/NIfTI files to a final, enhanced, and balanced dataset ready for deep learning models.

### Key Features

-   **End-to-End Processing**: From raw data to model-ready images.
-   **Temporal & Non-Temporal Support**: Flexible for both longitudinal and cross-sectional studies.
-   **Multi-Plane Analysis**: Processes axial, coronal, and sagittal views.
-   **Advanced Image Processing**: Includes skull stripping, ROI registration, and GWO-based enhancement.
-   **Data Augmentation**: Temporally consistent augmentation for robust training.

For detailed instructions on how to use the pipeline, please refer to the `alzheimer-disease-classification/README.md`.

## 🖼️ 3D Data Viewer

The `data-viewer` directory contains a React application built with Vite and Three.js for visualizing the processed MRI data in a 3D space.

### Key Features

-   **Interactive 3D Visualization**: Explore thousands of MRI slices in a spherical arrangement.
-   **Dynamic Filtering**: Filter images by plane, class, version, and dataset split.
-   **Performance Optimized**:
    -   **CloudFront CDN**: For fast global image delivery.
    -   **Level of Detail (LOD)**: Renders a maximum of 500 images at once.
    -   **Progressive Loading**: Shows placeholders and thumbnails for a smooth user experience.
-   **Image Comparison**: View original and enhanced images side-by-side.

### Getting Started

1.  **Navigate to the data viewer directory:**
    ```bash
    cd data-viewer
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    ```
3.  **Run the development server:**
    ```bash
    npm run dev
    ```

## ☁️ Cloud Infrastructure

The data viewer is configured to leverage AWS S3 and CloudFront for efficient and scalable image hosting. You will need to set up your own bucket and distribution.

-   **S3 Bucket**: Set this up to host your image data.
-   **CloudFront Domain**: Connect this to your S3 bucket for CDN delivery.

Scripts to help you manage the cloud infrastructure are available in the `data-viewer/scripts` directory. You will need to update the placeholder values in these scripts with your own resource names.

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

## 📧 Contact

For any questions or issues, please contact Izzat Zaky at [izzat.zaky@gmail.com](mailto:izzat.zaky@gmail.com).
