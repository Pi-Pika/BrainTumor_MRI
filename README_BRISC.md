# BRISC 2025 --- Brain Tumor MRI Dataset

## Overview

BRISC (BRain tumor Image Segmentation & Classification) is a curated,
expert-annotated T1-weighted MRI dataset designed for:

-   Multi-class brain tumor classification
-   Pixel-wise tumor segmentation
-   Multi-task deep learning research

The dataset contains 6,000 MRI slices: - 5,000 training samples - 1,000
test samples

Classes: - Glioma - Meningioma - Pituitary Tumor - No Tumor

Each image has structured metadata provided in `manifest.csv`.

------------------------------------------------------------------------

## Dataset Structure

    BRISC2025/
    ├── classification_task/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── pituitary/
    │   └── no_tumor/
    │
    ├── segmentation_task/
    │   ├── images/
    │   └── masks/
    │
    ├── manifest.csv
    ├── manifest.json
    └── README.md

------------------------------------------------------------------------

## manifest.csv Description

The `manifest.csv` file is the master metadata index for the dataset.\
Each row corresponds to one MRI slice.

### Columns Explained

### 1. filename

Full MRI image filename.\
Example:

    brisc2025_train_00010_gl_ax_t1.jpg

### 2. split

Dataset partition: - `train` - `test`

### 3. tumor

Full tumor class name: - glioma - meningioma - pituitary - no_tumor

### 4. tumor_code

Short encoded tumor label: - gl → Glioma - me → Meningioma - pi →
Pituitary - nt → No Tumor

### 5. view

MRI anatomical plane: - ax → Axial - co → Coronal - sa → Sagittal

### 6. sequence

MRI sequence type: - t1 (T1-weighted)

### 7. image_path

Relative path to the image file.

### 8. mask_path

Relative path to the corresponding segmentation mask (.png). Mask
filenames share the same basename as image files.

------------------------------------------------------------------------

## File Naming Convention

All filenames follow:

    brisc2025_<split>_<index>_<tumor>_<view>_<sequence>.<ext>

Example:

Image:

    brisc2025_test_00010_gl_ax_t1.jpg

Mask:

    brisc2025_test_00010_gl_ax_t1.png

------------------------------------------------------------------------

## Usage Recommendations

-   Always split data based on the `split` column to avoid data leakage.
-   Use `tumor` for classification labels.
-   Use `mask_path` for segmentation tasks.
-   Ensure patient-level grouping if performing custom splitting.

------------------------------------------------------------------------

## Citation

If you use this dataset, please cite:

Fateh et al., 2025 --- BRISC: Annotated dataset for brain tumor
segmentation and classification.
