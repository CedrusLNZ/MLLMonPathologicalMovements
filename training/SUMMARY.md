# ViT Training Pipeline Summary

## Overview

This training pipeline implements a Vision Transformer (ViT) classifier for the baseline MLLM comparison. The pipeline follows the requirements:

- **Data Split**: 90 patients → 60 train / 30 test (2:1 ratio)
- **Cross-Validation**: 3-fold CV on training set
- **Early Stopping**: Training stops when validation loss stops decreasing
- **Output**: Test results for 30 patients in CSV format

## Pipeline Components

### 1. Patient Stratification (`patient_stratification.py`)
- Extracts patient IDs from file names (e.g., `A0002@...` → `A0002`)
- Creates stratified train/test split maintaining label distribution
- Generates 3-fold cross-validation splits for training set
- Outputs split files to `training/splits/`

### 2. ViT Feature Extraction (`vit_feature_extractor.py`)
- Uses pre-trained Vision Transformer (default: `google/vit-base-patch16-224`)
- Extracts features from video frames
- Supports mean/max/last/all pooling strategies
- Caches extracted features for efficiency

### 3. Training (`train_vit_classifier.py`)
- Multi-label binary classifier (18 action features)
- 3-fold cross-validation with early stopping
- Saves best model from each fold
- Tracks validation loss and stops when not improving

### 4. Testing (`test_vit_classifier.py`)
- Loads best model from cross-validation
- Evaluates on held-out test set (30 patients)
- Outputs predictions to CSV with probabilities
- Calculates per-feature metrics

## Usage Example

```bash
# Run complete pipeline
python training/run_vit_training.py \
    --video_dir /path/to/videos \
    --annotation_csv evaluation/dataset/90_FeatureAnnotation.csv \
    --output_dir training/vit_results
```

## Output Structure

```
training/
├── splits/
│   ├── train_patients.csv          # 60 training patients
│   ├── test_patients.csv            # 30 test patients
│   ├── fold_0_train.csv             # CV fold splits
│   ├── fold_0_val.csv
│   └── ...
├── vit_results/
│   ├── train_features/
│   │   ├── features.npy            # Extracted ViT features
│   │   └── file_mapping.csv
│   ├── fold_0_model.pth            # Trained models
│   ├── fold_1_model.pth
│   ├── fold_2_model.pth
│   ├── cv_results.json             # CV metrics
│   └── test_results.csv             # Final test predictions
```

## Test Results Format

The output CSV (`test_results.csv`) contains:
- `file_name`: Video file name
- For each of 18 features:
  - `{feature_name}`: Binary prediction (0/1)
  - `{feature_name}_prob`: Prediction probability

## Key Features

1. **Stratified Splitting**: Maintains label distribution across splits
2. **Feature Caching**: Reuses extracted features to save computation
3. **Early Stopping**: Prevents overfitting with configurable patience
4. **Best Model Selection**: Uses fold with lowest validation loss for testing
5. **Robust File Matching**: Handles case sensitivity and file extensions (.mp4/.m2t)

## References

- Patient stratification: [Google Drive](https://drive.google.com/drive/u/1/folders/13AXmnOK5G7KcNAEh2dYWowyCGiLYHjNp)
- ViT implementation: [Colab Notebook](https://colab.research.google.com/drive/1X6-fjea6xT6jrxWaTWjYUJy1ORZntXNJ?authuser=1#scrollTo=g3RtGTUlwfPi)
