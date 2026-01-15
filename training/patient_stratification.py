"""
Patient stratification script for train/test split (2:1 ratio) and cross-validation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import re

def extract_patient_id(file_name):
    """Extract patient ID from file name (e.g., 'A0002@5-13-2021@UA6693LK@sz_v1.mp4' -> 'A0002')"""
    match = re.match(r'^([A-Z]\d+)@', file_name)
    if match:
        return match.group(1)
    return None

def create_patient_stratification(annotation_csv, output_dir='training/splits'):
    """
    Create stratified train/test split (60 patients train, 30 patients test) and 3-fold CV splits.
    
    Args:
        annotation_csv: Path to the 90_FeatureAnnotation.csv file
        output_dir: Directory to save split files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read annotation file
    df = pd.read_csv(annotation_csv)
    
    # Extract patient IDs
    df['patient_id'] = df['file_name'].apply(extract_patient_id)
    
    # Get unique patients
    patient_df = df.groupby('patient_id').agg({
        'label': 'first',  # Use first label for each patient
        'gender': 'first',
        'file_name': list  # Keep all file names for this patient
    }).reset_index()
    
    print(f"Total unique patients: {len(patient_df)}")
    print(f"Patients with label 1 (seizure): {patient_df['label'].sum()}")
    print(f"Patients with label 0 (non-seizure): {(patient_df['label'] == 0).sum()}")
    
    # Stratified split: approximately 2:1 ratio (60 train / 30 test videos)
    # Split at patient level to avoid data leakage (no patient in both sets)
    from sklearn.model_selection import train_test_split
    
    X = patient_df[['patient_id']]
    y = patient_df['label']
    
    # Use 2:1 ratio (test_size = 1/3 ≈ 0.333)
    # This ensures no patient appears in both training and testing sets
    train_patients, test_patients = train_test_split(
        patient_df, 
        test_size=1/3,  # 2:1 ratio (approximately 60 train / 30 test videos)
        random_state=42,
        stratify=y
    )
    
    print(f"\nTrain patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Save train/test split
    train_patients[['patient_id']].to_csv(
        os.path.join(output_dir, 'train_patients.csv'), 
        index=False
    )
    test_patients[['patient_id']].to_csv(
        os.path.join(output_dir, 'test_patients.csv'), 
        index=False
    )
    
    # Create 3-fold cross-validation splits for training set
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    fold_splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_patients, train_patients['label'])):
        fold_train = train_patients.iloc[train_idx]
        fold_val = train_patients.iloc[val_idx]
        
        fold_train[['patient_id']].to_csv(
            os.path.join(output_dir, f'fold_{fold_idx}_train.csv'),
            index=False
        )
        fold_val[['patient_id']].to_csv(
            os.path.join(output_dir, f'fold_{fold_idx}_val.csv'),
            index=False
        )
        
        fold_splits.append({
            'fold': fold_idx,
            'train_patients': len(fold_train),
            'val_patients': len(fold_val)
        })
    
    # Save all file names for each split
    def save_file_lists(patient_list, split_name):
        all_files = []
        for _, row in patient_list.iterrows():
            all_files.extend(row['file_name'])
        
        file_df = pd.DataFrame({'file_name': all_files})
        file_df.to_csv(
            os.path.join(output_dir, f'{split_name}_files.csv'),
            index=False
        )
        return len(all_files)
    
    # Save file lists
    train_files_count = save_file_lists(train_patients, 'train')
    test_files_count = save_file_lists(test_patients, 'test')
    
    print(f"\nTrain files: {train_files_count}")
    print(f"Test files: {test_files_count}")
    
    # Print fold information
    print("\nCross-validation folds:")
    for fold_info in fold_splits:
        print(f"  Fold {fold_info['fold']}: {fold_info['train_patients']} train, {fold_info['val_patients']} val patients")
    
    # Save summary
    summary = {
        'total_patients': len(patient_df),
        'train_patients': len(train_patients),
        'test_patients': len(test_patients),
        'train_files': train_files_count,
        'test_files': test_files_count,
        'folds': 3
    }
    
    import json
    with open(os.path.join(output_dir, 'split_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSplits saved to: {output_dir}")
    return train_patients, test_patients, fold_splits

if __name__ == '__main__':
    annotation_csv = 'evaluation/dataset/90_FeatureAnnotation.csv'
    create_patient_stratification(annotation_csv)
