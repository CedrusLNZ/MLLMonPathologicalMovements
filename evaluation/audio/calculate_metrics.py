import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from typing import Optional, Dict, List, Tuple


def calculate_metrics_for_column(y_true: pd.Series, y_pred: pd.Series, positive_label: str) -> Tuple[float, float, float, float]:
    """
    Calculate accuracy, precision, recall, and F1 score for a single column.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        positive_label: Label considered as positive class
        
    Returns:
        Tuple of (accuracy, precision, recall, f1_score)
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    return accuracy, precision, recall, f1


def calculate_metrics_for_multiclass(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float, float]:
    """
    Calculate accuracy, precision, recall, and F1 score for multiclass classification.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Tuple of (accuracy, precision, recall, f1_score)
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return accuracy, precision, recall, f1


def calculate_feature_metrics(
    predictions_csv: str,
    ground_truth_csv: str = '/mnt/SSD1/prateik/icassp_vlm/FeatureAnnotation_V3.csv',
    experiment_name: str = 'experiment',
    output_csv: Optional[str] = None,
    file_name_column: str = 'file_name',
    skip_columns: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate feature metrics by comparing predictions with ground truth.
    
    This function:
    1. Loads ground truth and prediction CSV files
    2. Preprocesses file names for matching
    3. Merges data on file names
    4. Calculates metrics for each feature column
    5. Returns and optionally saves results as CSV
    
    Args:
        predictions_csv: Path to CSV file containing predictions
        ground_truth_csv: Path to CSV file containing ground truth (default: FeatureAnnotation_V3.csv)
        experiment_name: Name for the experiment (used in output filename if output_csv not specified)
        output_csv: Optional path to save results CSV. If None, uses experiment_name + '_metrics.csv'
        file_name_column: Name of the column containing file names (default: 'file_name')
        skip_columns: List of columns to skip during metrics calculation
        verbose: Whether to print progress information
        
    Returns:
        pd.DataFrame: DataFrame containing metrics for each feature
        
    Raises:
        FileNotFoundError: If input CSV files don't exist
        ValueError: If specified file_name_column doesn't exist in the CSV
    """
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(f"Predictions CSV file not found: {predictions_csv}")
    if not os.path.exists(ground_truth_csv):
        raise FileNotFoundError(f"Ground truth CSV file not found: {ground_truth_csv}")
    
    # Default columns to skip
    if skip_columns is None:
        skip_columns = ['full_body_jerking', 'start_time', 'end_time', 'label', ]
    
    # Read CSV files
    if verbose:
        print(f"Reading ground truth from: {ground_truth_csv}")
        print(f"Reading predictions from: {predictions_csv}")
    
    df_gt = pd.read_csv(ground_truth_csv)
    df_pred = pd.read_csv(predictions_csv)
    
    # Validate file_name_column exists
    if file_name_column not in df_gt.columns:
        raise ValueError(f"Column '{file_name_column}' not found in ground truth CSV. Available columns: {list(df_gt.columns)}")
    if file_name_column not in df_pred.columns:
        raise ValueError(f"Column '{file_name_column}' not found in predictions CSV. Available columns: {list(df_pred.columns)}")
    
    # Data preprocessing
    df_gt[file_name_column] = df_gt[file_name_column].str.replace('.m2t', '.mp4').str.strip().str.lower()
    df_pred[file_name_column] = df_pred[file_name_column].str.replace('.wav', '.mp4')
    df_pred[file_name_column] = df_pred[file_name_column].str.strip().str.lower()
    
    # Align data
    merged_df = pd.merge(df_gt, df_pred, on=file_name_column, suffixes=('_gt', '_pred'), how='inner')
    
    if verbose:
        print(f"Successfully merged {len(merged_df)} matching files")
    
    # Print unmatched file names
    unmatched_gt = df_gt[~df_gt[file_name_column].isin(merged_df[file_name_column])]
    unmatched_pred = df_pred[~df_pred[file_name_column].isin(merged_df[file_name_column])]
    
    if verbose:
        if len(unmatched_gt) > 0:
            print(f"Unmatched file names in ground truth ({len(unmatched_gt)}):")
            print(unmatched_gt[file_name_column].tolist()[:10])  # Show first 10
            if len(unmatched_gt) > 10:
                print(f"... and {len(unmatched_gt) - 10} more")
        
        if len(unmatched_pred) > 0:
            print(f"Unmatched file names in predictions ({len(unmatched_pred)}):")
            print(unmatched_pred[file_name_column].tolist()[:10])  # Show first 10
            if len(unmatched_pred) > 10:
                print(f"... and {len(unmatched_pred) - 10} more")
    
    # Initialize results dictionary
    results = {'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Positive Num']}
    
    # Process each feature column
    feature_columns = [col for col in df_pred.columns[1:] if col not in skip_columns]
    
    if verbose:
        print(f"\nCalculating metrics for {len(feature_columns)} features...")
    
    for col in feature_columns:
        if verbose:
            print(f"Processing: {col}")

        if 'justiification' in col:
            continue    
        
        # Check if the feature exists in ground truth
        if col not in df_gt.columns:
            print(f"ERROR: Feature '{col}' not found in ground truth CSV. Skipping this feature.")
            continue
        
        # Get ground truth and predictions for this column
        y_true = merged_df[col + '_gt'].astype(str).str.replace('.', '', regex=False).str.strip().str.lower()
        y_pred = merged_df[col + '_pred'].astype(str).str.replace('.', '', regex=False).str.strip().str.lower()
        
        positive_num = (y_true == 'yes').sum()

        valid_indices = ((y_true != 'nan') & (y_pred != 'fail'))
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]

        # Special handling for verbal_responsiveness (multiclass)
        if col == 'verbal_responsiveness':
            accuracy, precision, recall, f1 = calculate_metrics_for_multiclass(y_true, y_pred)
        else:
            # Special handling for close_eyes and eye_blinking: remove NA samples
            if col in ['close_eyes', 'eye_blinking']:
                valid_indices = y_true != 'nan'
                y_true = y_true[valid_indices]
                y_pred = y_pred[valid_indices]
            
            # Determine positive label
            positive_label = 'female' if col == 'gender' else 'yes'
            
            # Calculate metrics
            accuracy, precision, recall, f1 = calculate_metrics_for_column(y_true, y_pred, positive_label)
            #positive_num = (y_true == positive_label).sum()
        
        # Store results
        results[col] = [accuracy, precision, recall, f1, positive_num]
        
        if verbose:
            print(f"  Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Positive: {positive_num}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.round(3)
    
    # Save results
    if output_csv is None:
        output_csv = f"{experiment_name}_metrics.csv"
    
    if verbose:
        print(f"\nSaving results to: {output_csv}")
    
    results_df.to_csv(output_csv, index=False)
    
    if verbose:
        print("Metrics calculation completed!")
    
    return results_df


def compare_experiments(experiment_results: Dict[str, pd.DataFrame], 
                       output_csv: Optional[str] = None,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Compare metrics across multiple experiments.
    
    Args:
        experiment_results: Dictionary mapping experiment names to their metrics DataFrames
        output_csv: Optional path to save comparison results
        verbose: Whether to print progress information
        
    Returns:
        pd.DataFrame: Combined metrics from all experiments
    """
    if not experiment_results:
        raise ValueError("No experiment results provided")
    
    # Get all unique metrics and features
    all_metrics = set()
    all_features = set()
    
    for exp_name, df in experiment_results.items():
        all_metrics.update(df['Metric'].tolist())
        all_features.update([col for col in df.columns if col != 'Metric'])
    
    # Create comparison DataFrame
    comparison_data = []
    
    for metric in sorted(all_metrics):
        for feature in sorted(all_features):
            row = {'Metric': metric, 'Feature': feature}
            
            for exp_name, df in experiment_results.items():
                if feature in df.columns and metric in df['Metric'].values:
                    metric_idx = df[df['Metric'] == metric].index[0]
                    row[f'{exp_name}_value'] = df.loc[metric_idx, feature]
                else:
                    row[f'{exp_name}_value'] = None
            
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if output_csv and verbose:
        print(f"Saving experiment comparison to: {output_csv}")
        comparison_df.to_csv(output_csv, index=False)
    
    return comparison_df


# Example usage function
def example_usage():
    """
    Example of how to use the calculate_feature_metrics function.
    """
    # Example 1: Basic usage
    predictions_file = "AF3_90_audio_feature.csv" 
    #predictions_file = "AF3_93_audio_clear_results.csv"
    #predictions_file = "AF3_audio-text_results2.csv"

    try:
        results_df = calculate_feature_metrics(
            predictions_csv=predictions_file,
            experiment_name="audio_base",
            verbose=True
        )
        print("Metrics calculation completed successfully!")
        print(f"Results DataFrame shape: {results_df.shape}")
        print(results_df.head())
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 2: With custom output path
    try:
        results_df = calculate_feature_metrics(
            predictions_csv=predictions_file,
            output_csv="custom_metrics.csv",
            skip_columns=['custom_column'],
            verbose=False
        )
        print("Custom metrics calculation completed!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run example usage
    example_usage()
