"""
Training script for ViT-based classifier with 3-fold cross-validation and early stopping.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
from tqdm import tqdm
import argparse
from vit_feature_extractor import ViTFeatureExtractor
from patient_stratification import extract_patient_id

class VideoFeatureDataset(Dataset):
    """Dataset for video features and labels."""
    
    def __init__(self, features, labels, file_names):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.file_names = file_names
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.file_names[idx]

class ViTClassifier(nn.Module):
    """Classifier head on top of ViT features."""
    
    def __init__(self, input_dim, num_features=18, hidden_dims=[512, 256], dropout=0.3):
        """
        Args:
            input_dim: Dimension of ViT features
            num_features: Number of binary classification tasks (18 action features)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super(ViTClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer: 18 binary classifiers
        layers.append(nn.Linear(prev_dim, num_features))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for features, labels, _ in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        
        # Multi-label binary cross-entropy loss
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    file_names = []
    
    with torch.no_grad():
        for features, labels, fnames in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels.float())
            
            total_loss += loss.item()
            
            # Convert to binary predictions
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            file_names.extend(fnames)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, all_preds, all_labels, file_names

def train_with_early_stopping(
    model, train_loader, val_loader, criterion, optimizer, device,
    max_epochs=100, patience=10, min_delta=1e-4
):
    """
    Train with early stopping.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(max_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, _, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  -> New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def extract_features_for_split(video_dir, file_list_csv, feature_extractor, output_dir):
    """Extract ViT features for a split of files."""
    os.makedirs(output_dir, exist_ok=True)
    
    feature_file = os.path.join(output_dir, 'features.npy')
    mapping_file = os.path.join(output_dir, 'file_mapping.csv')
    
    # Check if features already exist
    if os.path.exists(feature_file) and os.path.exists(mapping_file):
        print(f"Loading cached features from {feature_file}...")
        features = np.load(feature_file)
        mapping_df = pd.read_csv(mapping_file)
        file_names = mapping_df['file_name'].tolist()
        return features, file_names
    
    # Read file list
    file_df = pd.read_csv(file_list_csv)
    file_names = file_df['file_name'].tolist()
    
    # Build full paths
    video_paths = [os.path.join(video_dir, fname) for fname in file_names]
    
    # Extract features
    print(f"Extracting features for {len(video_paths)} videos...")
    features = feature_extractor.extract_features_batch(video_paths, pooling='mean')
    
    # Save features
    np.save(feature_file, features)
    
    # Save file mapping
    mapping_df = pd.DataFrame({
        'file_name': file_names,
        'feature_idx': range(len(file_names))
    })
    mapping_df.to_csv(mapping_file, index=False)
    
    return features, file_names

def prepare_data(annotation_csv, features, file_names, feature_names):
    """Prepare features and labels for training."""
    # Read annotations
    df_gt = pd.read_csv(annotation_csv)
    df_gt['file_name'] = df_gt['file_name'].str.strip().str.lower()
    
    # Normalize file names (remove path, lowercase, handle .mp4/.m2t)
    def normalize_fname(fname):
        fname = os.path.basename(fname).lower()
        fname = fname.replace('.m2t', '.mp4')
        return fname.strip()
    
    # Create mapping
    file_to_idx = {normalize_fname(fname): idx for idx, fname in enumerate(file_names)}
    
    # Extract labels for each feature
    labels_list = []
    valid_indices = []
    
    for idx, fname in enumerate(file_names):
        fname_normalized = normalize_fname(fname)
        matching_rows = df_gt[df_gt['file_name'] == fname_normalized]
        
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            labels = []
            for feat in feature_names:
                val = str(row[feat]).strip().lower()
                labels.append(1 if val == 'yes' else 0)
            labels_list.append(labels)
            valid_indices.append(idx)
        else:
            print(f"Warning: {fname} (normalized: {fname_normalized}) not found in annotations")
    
    # Filter features and labels
    if len(valid_indices) == 0:
        raise ValueError("No matching files found between features and annotations!")
    
    valid_features = features[valid_indices]
    valid_labels = np.array(labels_list)
    valid_file_names = [file_names[i] for i in valid_indices]
    
    print(f"Matched {len(valid_indices)}/{len(file_names)} files with annotations")
    
    return valid_features, valid_labels, valid_file_names

def main():
    parser = argparse.ArgumentParser(description='Train ViT classifier with cross-validation')
    parser.add_argument('--annotation_csv', type=str, 
                       default='evaluation/dataset/90_FeatureAnnotation.csv',
                       help='Path to annotation CSV')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--splits_dir', type=str, default='training/splits',
                       help='Directory containing train/test splits')
    parser.add_argument('--output_dir', type=str, default='training/vit_results',
                       help='Output directory for results')
    parser.add_argument('--vit_model', type=str, default='google/vit-base-patch16-224',
                       help='ViT model name')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    feature_extractor = ViTFeatureExtractor(
        model_name=args.vit_model,
        device=args.device
    )
    
    # Extract features for training set
    print("\nExtracting features for training set...")
    train_features_dir = os.path.join(args.output_dir, 'train_features')
    
    train_file_list = os.path.join(args.splits_dir, 'train_files.csv')
    train_features, train_file_names = extract_features_for_split(
        args.video_dir, train_file_list, feature_extractor, train_features_dir
    )
    
    # Prepare training data
    train_features_processed, train_labels, train_file_names_processed = prepare_data(
        args.annotation_csv, train_features, train_file_names, feature_names
    )
    
    # Cross-validation
    print("\nStarting 3-fold cross-validation...")
    cv_results = []
    
    for fold in range(3):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/3")
        print(f"{'='*50}")
        
        # Load fold splits
        fold_train_csv = os.path.join(args.splits_dir, f'fold_{fold}_train.csv')
        fold_val_csv = os.path.join(args.splits_dir, f'fold_{fold}_val.csv')
        
        fold_train_patients = pd.read_csv(fold_train_csv)['patient_id'].tolist()
        fold_val_patients = pd.read_csv(fold_val_csv)['patient_id'].tolist()
        
        # Filter data by patient ID
        def get_patient_id(fname):
            return extract_patient_id(fname)
        
        train_mask = [get_patient_id(fname) in fold_train_patients 
                     for fname in train_file_names_processed]
        val_mask = [get_patient_id(fname) in fold_val_patients 
                   for fname in train_file_names_processed]
        
        fold_train_features = train_features_processed[train_mask]
        fold_train_labels = train_labels[train_mask]
        fold_train_files = [train_file_names_processed[i] for i in range(len(train_mask)) if train_mask[i]]
        
        fold_val_features = train_features_processed[val_mask]
        fold_val_labels = train_labels[val_mask]
        fold_val_files = [train_file_names_processed[i] for i in range(len(val_mask)) if val_mask[i]]
        
        # Create datasets
        train_dataset = VideoFeatureDataset(fold_train_features, fold_train_labels, fold_train_files)
        val_dataset = VideoFeatureDataset(fold_val_features, fold_val_labels, fold_val_files)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = ViTClassifier(
            input_dim=feature_extractor.feature_dim,
            num_features=len(feature_names)
        ).to(args.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Train with early stopping
        model, train_losses, val_losses = train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer, args.device,
            max_epochs=args.max_epochs, patience=args.patience
        )
        
        # Evaluate
        val_loss, val_preds, val_labels, val_files = evaluate(
            model, val_loader, criterion, args.device
        )
        
        # Calculate per-feature metrics
        fold_metrics = {}
        for feat_idx, feat_name in enumerate(feature_names):
            y_true = val_labels[:, feat_idx]
            y_pred = val_preds[:, feat_idx]
            
            # Skip if no positive samples
            if y_true.sum() == 0:
                continue
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            fold_metrics[feat_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
        
        cv_results.append({
            'fold': fold,
            'val_loss': val_loss,
            'metrics': fold_metrics
        })
        
        # Save fold model
        torch.save(model.state_dict(), 
                  os.path.join(args.output_dir, f'fold_{fold}_model.pth'))
    
    # Save CV results
    with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print("\nCross-validation complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
