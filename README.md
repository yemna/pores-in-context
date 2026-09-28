# Pores in Context — feature extraction and classification code

Code for the paper *Pores in Context: Leveraging Matrix-Informed CNN Embeddings for Transparent Carbonate Pore Classification* (Artificial Intelligence in Geosciences).

The pipeline has three steps: feature extraction from segmented pores, feature preprocessing and selection inside a thin-section-grouped cross-validation, and training/evaluation of 20 classifiers.

---

## Scripts and run order

```bash
python Feature_extraction_from_images.py        # pore-only features
python Feature_extraction_neighbourhood.py      # neighbourhood features
python feature_processing_all_files.py          # preprocessing, grouped 5-fold CV, Boruta
python ML_script_all.py                         # 20 classifiers, all folds
```

Folder paths are set at the top of each script (`main_folder`, `results_root`, `BASE_DIR`) and must be changed to your own locations. The extraction scripts are run once per class folder.

---

## 1. Input data

For each labelled pore:

- `<prefix>_cropped_label_<ID>.png` — RGB crop
- `<prefix>_label_mask_<ID>.png` — binary pore mask (pore = 255, elsewhere = 0), used in the pore-only run
- `<prefix>_label_mask_inverted_<ID>.png` — inverted mask (pore = 0), used in the neighbourhood run

The `<prefix>` is the thin-section image name (e.g. `Modern_1`). It is used later as the grouping variable for cross-validation.

For the pore-only run the RGB crop is the bounding box of the pore. For the neighbourhood run it is the bounding box enlarged by 300 pixels on each side.

---

## 2. Feature extraction

Masks are used only to select RGB pixels; the CNNs receive masked RGB images, not binary masks.

**Pore-only run** (`Feature_extraction_from_images.py`), input = RGB crop ⊙ pore mask:

- first-order statistics of the pore pixels
- size and shape descriptors, Fourier shape descriptors (from the mask)
- LBP (P/R = 24/8, 16/2, 8/1), Haralick texture, Zernike moments, FFT statistics (grayscale of the masked image)
- discrete wavelet transform statistics (13 wavelets × 4 sub-bands × 9 statistics); note that these are computed on the **unmasked** bounding-box crop
- CNN embeddings (ImageNet weights, no fine-tuning, global average pooling): VGG16, VGG19, ResNet50, InceptionResNetV2, DenseNet121, EfficientNetB4 — 7,424 dimensions in total

**Neighbourhood run** (`Feature_extraction_neighbourhood.py`), input = enlarged RGB crop ⊙ inverted mask (the target pore is set to zero):

- LBP, Haralick texture and the same six CNN embeddings; these columns carry the suffix `_NI`

---

## 3. Preprocessing and feature selection

`feature_processing_all_files.py`

- removes features with more than 5% missing values and features with zero variance or zero IQR
- builds **5 grouped cross-validation folds** with scikit-learn `StratifiedGroupKFold` (`shuffle=True`, `random_state=42`). The group is the thin-section image name taken from the pore label, so all pores of one image are either in the training set or in the test set of a fold. Class balance is approximated at the image level and reported after the split; the script checks that no image appears in both partitions.

Within each training fold only (nothing is fitted on test data):

- robust scaling
- mutual-information ranking (top 30%) and removal of correlated features (|r| > 0.7)
- Boruta (BorutaPy 0.4.3): 250-tree random forest, `max_iter=50`, `perc=70`, `two_step=True`, `alpha=0.05`; only confirmed features are kept

The same pipeline is run for six feature sets: traditional, deep-learning and combined features, each pore-only and with neighbourhood features.

---

## 4. Classification

`ML_script_all.py` trains 20 classifiers on every fold with grid-search hyperparameter tuning:

LDA, k-NN, decision tree, random forest, extra trees, AdaBoost, gradient boosting, histogram gradient boosting, XGBoost, CatBoost, bagging, SVM, linear SVM, NuSVC, Gaussian naive Bayes, MLP, SGD, ridge, passive-aggressive and perceptron.

Outputs per fold: saved models, accuracy/precision/F1, classification reports, ROC and precision–recall curves; an aggregate over folds is written at the end. The script checkpoints its progress and can resume.

---

## Environment

Python 3.12, scikit-learn 1.6.1, xgboost 2.1.4, catboost 1.2.7, boruta 0.4.3, TensorFlow/Keras (feature extraction), OpenCV, scikit-image, mahotas, PyWavelets.

---

## Note on an earlier version

An earlier version of this repository contained preprocessing code with ordinary (random) stratified 5-fold splitting. It has been replaced by the thin-section-grouped version above, which is the one used for the results in the paper.

## Data

The thin-section images are subject to confidentiality restrictions and are not included. Extracted feature tables are available from the corresponding author on reasonable request.
