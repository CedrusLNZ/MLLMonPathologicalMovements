import pandas as pd
import re
import os
from typing import Optional


def check_duplicates(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Check for duplicate entries in the first column of the dataframe.
    
    Args:
        df: DataFrame to check for duplicates
        verbose: Whether to print duplicate information
        
    Returns:
        bool: True if duplicates found, False otherwise
    """
    first_column = df.iloc[:, 0]
    duplicates = first_column.duplicated()
    has_duplicates = duplicates.any()
    
    if verbose and has_duplicates:
        print(f"Warning: {duplicates.sum()} duplicate files found in first column")
    
    return has_duplicates


def get_original_filename(fn: str) -> str:
    """
    Convert segmented file name to original file name by removing segment identifiers.
    
    Args:
        fn: Filename with potential segment identifiers
        
    Returns:
        str: Original filename without segment identifiers
    """
    # Use regex to remove the '_segment_' followed by digits
    return re.sub(r'_segment_\d+', '', fn)


def majority_vote(series: pd.Series) -> Optional[str]:
    """
    Return the value with the highest count in the series.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        str or None: Value with highest count, or None if series is empty
    """
    counts = series.value_counts()
    return counts.idxmax() if not counts.empty else None


def any_yes(series: pd.Series) -> str:
    """
    Return 'yes' if any value in the series is 'yes' (case insensitive), otherwise 'no'.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        str: 'yes' or 'no'
    """
    return 'yes' if series.str.lower().eq('yes').any() else 'no'


def merge_segments(input_csv: str, output_csv: Optional[str] = None, 
                  file_name_column: str = 'file_name', verbose: bool = True) -> pd.DataFrame:
    """
    Merge segmented video features by grouping segments from the same original video file.
    
    This function:
    1. Reads a CSV file containing segmented video features
    2. Groups segments by their original filename (removing segment identifiers)
    3. Merges features using different strategies for different columns:
       - 'occur_during_sleep': takes first value
       - 'gender': uses majority vote
       - 'limb_movements_pattern': takes minimum numeric value
       - Other columns: returns 'yes' if any segment has 'yes', otherwise 'no'
    4. Returns and optionally saves the merged DataFrame
    
    Args:
        input_csv: Path to input CSV file with segmented features
        output_csv: Optional path to save merged CSV file. If None, file won't be saved
        file_name_column: Name of the column containing file names (default: 'file_name')
        verbose: Whether to print progress information
        
    Returns:
        pd.DataFrame: Merged DataFrame with one row per original video file
        
    Raises:
        FileNotFoundError: If input CSV file doesn't exist
        ValueError: If specified file_name_column doesn't exist in the CSV
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")
    
    # Read the CSV file
    if verbose:
        print(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Validate file_name_column exists
    if file_name_column not in df.columns:
        raise ValueError(f"Column '{file_name_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Clean data: strip whitespace and convert to lowercase for columns after file_name
    file_name_col_idx = df.columns.get_loc(file_name_column)
    for col in df.columns[file_name_col_idx + 1:]:
        df[col] = df[col].astype(str).str.strip().str.split('.').str[0].str.lower()
    
    # Check for duplicates before merging
    check_duplicates(df, verbose=verbose)
    
    # Add original file name column
    df['orig_file_name'] = df[file_name_column].apply(get_original_filename)
    
    # Initialize the result list
    result_rows = []
    
    # Process groups based on the original file name
    if verbose:
        print(f"Merging {len(df)} segments into original video files...")
    
    grouped = df.groupby('orig_file_name')
    for orig_name, group in grouped:
        merged_row = {}
        # Use the original file name for the file_name column
        merged_row[file_name_column] = orig_name
        
        # Iterate over the other columns
        for col in group.columns:
            # Skip already processed columns
            if col in [file_name_column, 'orig_file_name']:
                continue
                
            # Apply different merging strategies based on column name
            if col == 'occur_during_sleep':
                # For "occur_during_sleep", directly take the first row
                merged_row[col] = group.iloc[0][col]
            elif col == 'gender':
                # For gender, use majority vote
                merged_row[col] = majority_vote(group[col].astype(str))
            elif col == 'limb_movements_pattern':
                # For limb movements pattern, take minimum numeric value
                numeric_values = pd.to_numeric(group[col], errors='coerce')
                merged_row[col] = numeric_values.min()
            else:
                # For other columns, return 'yes' if any segment has 'yes'
                merged_row[col] = any_yes(group[col].astype(str))
        
        result_rows.append(merged_row)
    
    # Create a new DataFrame from the result rows
    result_df = pd.DataFrame(result_rows)
    
    # Check for duplicates after merging
    check_duplicates(result_df, verbose=verbose)
    
    # Save the result if output path is provided
    if output_csv:
        if verbose:
            print(f"Saving merged results to: {output_csv}")
        result_df.to_csv(output_csv, index=False)
        if verbose:
            print("Merge complete!")
    
    if verbose:
        print(f"Successfully merged {len(df)} segments into {len(result_df)} original video files")
    
    return result_df


# Example usage function
def example_usage():
    """
    Example of how to use the merge_segments function.
    """
    # Example 1: Basic usage with file paths
    input_file = "path/to/your/input.csv"
    output_file = "path/to/your/output.csv"
    
    try:
        merged_df = merge_segments(input_file, output_file)
        print("Merging completed successfully!")
        print(f"Merged DataFrame shape: {merged_df.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 2: Just return DataFrame without saving
    try:
        merged_df = merge_segments(input_file, verbose=False)
        print("DataFrame merged in memory only")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run example usage
    example_usage()
