# ROI-Based Feature Extraction and Machine Learning Pipeline

This repository provides an end-to-end pipeline for image-based analysis using
region-of-interest (ROI) masks. The workflow includes:

1. Feature extraction from RGB images using binary masks
2. Feature preprocessing and selection with leakage-safe cross-validation
3. Training and benchmarking of ~20 classical machine learning models

The pipeline is designed for scientific and engineering image analysis where
features must be extracted only from annotated regions.

---

## 1. Input Data Requirements

### RGB Images
- Format: `.png`
- Color images used for handcrafted and deep feature extraction

### Binary Label Masks
- Format: `.png`
- Same spatial dimensions as the corresponding RGB image
- Pixel encoding:
  - Background = 0
  - ROI = 255 (white)

All features are computed **only within the ROI** defined by the mask.

---

## 2. Image–Mask Pairing Convention

The feature extraction script automatically pairs images and masks based on
filename patterns:

- RGB images must contain: `cropped_label`
- Binary masks must contain: `label_mask`

Expected naming format:
<prefix>cropped_label<ID>.png
<prefix>label_mask<ID>.png


Example:
Modern_1_cropped_label_12.png
Modern_1_label_mask_12.png


Only correctly matched pairs are processed.

---

## 3. Pipeline Overview

### Step 1 — Feature Extraction
**Script:** `Feature_extraction_from_images.py`

- Extracts handcrafted features:
  - Intensity statistics
  - Texture (LBP, Haralick)
  - Shape and morphological descriptors
  - Frequency-domain features (FFT, DWT)
- Extracts deep features using pretrained CNNs:
  - VGG16 / VGG19
  - ResNet50
  - InceptionResNetV2
- Uses binary masks to restrict feature computation to ROI pixels
- Outputs CSV files containing extracted features

---

### Step 2 — Feature Preprocessing and Selection
**Script:** `feature_processing_all_files.py`

- Removes features with:
  - High missing values
  - Zero variance or zero IQR
- Applies:
  - Stratified 5-fold cross-validation
  - Robust scaling (trained on training folds only)
  - Mutual information–based filtering
  - Correlation-based feature pruning
  - Boruta feature selection
- Ensures no data leakage between training and test sets
- Outputs processed feature sets and selection summaries

---

### Step 3 — Machine Learning Training and Evaluation
**Script:** `ML_classification.py`

- Trains and benchmarks ~20 ML classifiers, including:
  - LDA / QDA
  - k-NN
  - SVM variants
  - Random Forest, Extra Trees
  - Gradient Boosting, AdaBoost
  - XGBoost, CatBoost
  - MLP, Naive Bayes, SGD, Ridge, Bagging
- Supports grid-search hyperparameter tuning
- Produces:
  - Accuracy, precision, F1-score
  - ROC and precision–recall curves
  - Model rankings and saved models
- Includes checkpointing to resume interrupted runs

---

## 4. Recommended Execution Order

```bash
python Feature_extraction_from_images.py
python feature_processing_all_files.py
python ML_classification.py
