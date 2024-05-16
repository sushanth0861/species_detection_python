# Project README

## Overview

This project is part of the iWildCam 2019 - FGVC6 competition, which aims to classify animal species in camera trap images from different regions.

## Competition Context

The iWildCam 2019 - FGVC6 competition challenges participants to classify species in new regions using training data from different areas. Training data is from the American Southwest, while test data is from the American Northwest.

## Folder Structure and Contents

### `src` Folder
Code for basic 32x32 image transformation and model implementation.

- **Notebook:** `Preprocess_model_raw.ipynb` - High-level code flow for 32x32 image transformation and model implementation.

### `src2` Folder
Code for 32x32 image transformation with CLAHE and model implementation.

- **Notebook:** `Preprocess_model_CLAHE.ipynb` - High-level code flow for 32x32 + CLAHE image transformation and model implementation.

### `src3` Folder
Code for 32x32 image transformation with CLAHE and Grayscale preprocessing, and model implementation.

- **Notebook:** `Preprocess_model_CLAHE_GS.ipynb` - High-level code flow for 32x32 + CLAHE + Grayscale image transformation and model implementation.

### Additional Notebooks
- `EDA.ipynb`: Exploratory Data Analysis.
- Refer to the relevant Preprocess_model notebook for high-level code flow and function calls.- Detailed scripts for specific preprocessing methods and model implementations are in the src, src2, and src3 folders.

## Setup

### Create a Virtual Environment and Install Packages

1. **Create a virtual environment:**
   python -m venv env
2. **Activate the virtual environment:**
    On Windows: bash
    .\env\Scripts\activate
    
    On macOS and Linux: bash
    source env/bin/activate
3. **Install required packages:**
    pip install -r requirements.txt

