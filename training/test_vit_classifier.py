"""
Test script for ViT classifier on held-out test set.
Outputs results to CSV file.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import argparse
import json
from torch.utils.data import DataLoader
from vit_feature_extractor import ViTFeatureExtractor
from train_vit_classifier import ViTClassifier, VideoFeatureDataset, prepare_data

def load_best_model(model_dir, input_dim, num_features, device):
    """Load the best model from cross-validation (use fold with best val loss)."""
    # Load CV results to find best fold
    cv_results_file = os.path.join(model_dir, 'cv_results.json')
    if os.path.exists(cv_results_file):
        with open(cv_results_file, 'r') as f:
            cv_results = json.load(f)
        
        # Find fold with lowest validation loss
        best_fold = min(cv_results, key=lambda x: x['val_loss'])['fold']
        model_path = os.path.join(model_dir, f'fold_{best_fold}_model.pth')
    else:
        # Default to fold 0 if no results file
        model_path = os.path.join(model_dir, 'fold_0_model.pth')
    
    # Initialize model
    model = ViTClassifier(input_dim=input_dim, num_features=num_features)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    return model

def evaluate_test_set(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    file_names = []
    
    with torch.no_grad():
        for features, labels, fnames in test_loader:
            features = features.to(device)
            
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            file_names.extend(fnames)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    return all_preds, all_labels, all_probs, file_names

def main():
    parser = argparse.ArgumentParser(description='Test ViT classifier on test set')
    parser.add_argument('--annotation_csv', type=str,
                       default='evaluation/dataset/90_FeatureAnnotation.csv',
                       help='Path to annotation CSV')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--splits_dir', type=str, default='training/splits',
                       help='Directory containing train/test splits')
    parser.add_argument('--model_dir', type=str, default='training/vit_results',
                       help='Directory containing trained models')
    parser.add_argument('--output_csv', type=str, default='training/vit_test_results.csv',
                       help='Output CSV file for test results')
    parser.add_argument('--vit_model', type=str, default='google/vit-base-patch16-224',
                       help='ViT model name')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 18 action features
    feature_names = [
        'occur_during_sleep', 'blank_stare', 'close_eyes', 'eye_blinking',
        'tonic', 'clonic', 'arm_flexion', 'arm_straightening', 'figure4',
        'oral_automatisms', 'limb_automatisms', 'face_pulling', 'face_twitching',
        'head_turning', 'asynchronous_movement', 'pelvic_thrusting',
        'arms_move_simultaneously', 'full_body_shaking'
    ]
    
    # Initialize ViT feature extractor
    print("Initializing ViT feature extractor...")
    # Set up cache directory for frames (similar to MLLM code)
    video_cache_dir = os.path.join(args.model_dir, 'video_cache')
    os.makedirs(video_cache_dir, exist_ok=True)
    
    feature_extractor = ViTFeatureExtractor(
        model_name=args.vit_model,
        device=args.device,
        cache_dir=video_cache_dir
    )
    
    # Extract features for test set
    print("\nExtracting features for test set...")
    test_file_list = os.path.join(args.splits_dir, 'test_files.csv')
    test_file_df = pd.read_csv(test_file_list)
    test_file_names = test_file_df['file_name'].tolist()
    
    # Build full paths
    test_video_paths = [os.path.join(args.video_dir, fname) for fname in test_file_names]
    
    # Extract features
    print(f"Extracting features for {len(test_video_paths)} test videos...")
    test_features = feature_extractor.extract_features_batch(
        test_video_paths, pooling='mean'
    )
    
    # Prepare test data
    test_features_processed, test_labels, test_file_names_processed = prepare_data(
        args.annotation_csv, test_features, test_file_names, feature_names
    )
    
    # Create test dataset
    test_dataset = VideoFeatureDataset(
        test_features_processed, test_labels, test_file_names_processed
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load best model
    print("\nLoading best model from cross-validation...")
    model = load_best_model(
        args.model_dir,
        input_dim=feature_extractor.feature_dim,
        num_features=len(feature_names),
        device=args.device
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_preds, test_labels, test_probs, test_files = evaluate_test_set(
        model, test_loader, args.device
    )
    
    # Create results DataFrame
    results_data = {'file_name': test_files}
    
    # Add predictions for each feature
    for feat_idx, feat_name in enumerate(feature_names):
        results_data[feat_name] = test_preds[:, feat_idx].astype(int)
        results_data[f'{feat_name}_prob'] = test_probs[:, feat_idx]
    
    results_df = pd.DataFrame(results_data)
    
    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nTest results saved to: {args.output_csv}")
    print(f"Test set size: {len(results_df)} videos")
    
    # Calculate and print metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("\nPer-feature metrics on test set:")
    print("-" * 80)
    print(f"{'Feature':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 80)
    
    for feat_idx, feat_name in enumerate(feature_names):
        y_true = test_labels[:, feat_idx]
        y_pred = test_preds[:, feat_idx]
        
        if y_true.sum() == 0:
            continue
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"{feat_name:<25} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}")

if __name__ == '__main__':
    main()
