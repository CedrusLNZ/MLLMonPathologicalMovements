# ViT Training Pipeline

This directory contains scripts for training a Vision Transformer (ViT) classifier on the baseline MLLM setup.

## Overview

The pipeline implements:
1. **Patient Stratification**: Split 90 patients into 60 train / 30 test (2:1 ratio) with 3-fold cross-validation
2. **ViT Feature Extraction**: Extract features from video frames using a pre-trained Vision Transformer
3. **Training with Early Stopping**: Train a classifier with 3-fold cross-validation and early stopping
4. **Test Evaluation**: Evaluate on held-out test set and output results to CSV

## Files

- `patient_stratification.py`: Creates stratified train/test splits and CV folds
- `vit_feature_extractor.py`: ViT-based feature extraction from video frames
- `train_vit_classifier.py`: Training script with cross-validation and early stopping
- `test_vit_classifier.py`: Test evaluation script
- `run_vit_training.py`: Main script to run the complete pipeline

## Usage

### Quick Start

Run the complete pipeline:

```bash
python training/run_vit_training.py \
    --video_dir /path/to/video/directory \
    --annotation_csv evaluation/dataset/90_FeatureAnnotation.csv
```

### Step-by-Step

1. **Create patient stratification**:
```bash
python training/patient_stratification.py
```

2. **Train with cross-validation**:
```bash
python training/train_vit_classifier.py \
    --video_dir /path/to/video/directory \
    --annotation_csv evaluation/dataset/90_FeatureAnnotation.csv \
    --output_dir training/vit_results
```

3. **Test on held-out set**:
```bash
python training/test_vit_classifier.py \
    --video_dir /path/to/video/directory \
    --model_dir training/vit_results \
    --output_csv training/vit_test_results.csv
```

## Configuration

### Key Parameters

- `--vit_model`: ViT model to use (default: `google/vit-base-patch16-224`)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--patience`: Early stopping patience (default: 10)
- `--device`: Device to use (default: `cuda`)

### Data Structure

The script expects:
- Video files in the specified `--video_dir`
- Annotation CSV with 18 action features
- File names matching the pattern in the annotation CSV

## Output

The pipeline generates:
- `training/splits/`: Patient stratification files
- `training/vit_results/`: Model checkpoints and CV results
- `training/vit_test_results.csv`: Test set predictions for 30 patients

## Features

The model predicts 18 binary action features:
1. occur_during_sleep
2. blank_stare
3. close_eyes
4. eye_blinking
5. tonic
6. clonic
7. arm_flexion
8. arm_straightening
9. figure4
10. oral_automatisms
11. limb_automatisms
12. face_pulling
13. face_twitching
14. head_turning
15. asynchronous_movement
16. pelvic_thrusting
17. arms_move_simultaneously
18. full_body_shaking

## References

- Patient stratification approach: [Google Drive folder](https://drive.google.com/drive/u/1/folders/13AXmnOK5G7KcNAEh2dYWowyCGiLYHjNp)
- ViT sample code: [Colab notebook](https://colab.research.google.com/drive/1X6-fjea6xT6jrxWaTWjYUJy1ORZntXNJ?authuser=1#scrollTo=g3RtGTUlwfPi)
