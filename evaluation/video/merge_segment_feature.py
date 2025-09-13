import pandas as pd
import re
import os
# Read the CSV file
input_csv = "/home/lina/git/SeizureVision/experiment_results/internvl3_78_segment_newaddpatient/visual_segment_feature2.csv"
df = pd.read_csv(input_csv)
for col in df.columns[1:]:
    df[col] = df[col].astype(str).str.strip().str.split('.').str[0].str.lower()

# df["file_name"] = df["file_name"].str.replace(
#     r"^/mnt/SSD1/richard/mp4_seizure_segments/", "", regex=True
# )
# df["tonic"] = (
#     df["tonic"]
#     .str.extract(r"^(yes|no)\b", flags=re.IGNORECASE)[0]  # 只在行首找 yes 或 no
#     .str.lower()
# )

# for col in df.columns[1:]:
#     df[col] = df[col].astype(str).str.strip().str.split('.').str[0].str.lower()

# output_csv = "/home/lina/git/SeizureVision/experiment_results/internvl3_78_segment_newaddpatient/visual_segment_feature2.csv"
# df.to_csv(output_csv, index=False)

def check_duplecated(df_tmp):
    first_column = df_tmp.iloc[:, 0]
    duplicates = first_column.duplicated()
    if duplicates.any():
        print("repeat files:")
        print(first_column[duplicates].unique())
    else:
        print("No repeat video file.")

check_duplecated(df)        

# Define function: convert segmented file name to original file name
def get_original_filename(fn):
    # Use regex to remove the '_segment_' followed by digits
    return re.sub(r'_segment_\d+', '', fn)

# Add a new column to store the original file name
df['orig_file_name'] = df['file_name'].apply(get_original_filename)

# Define the majority rule: decide yes/no (can be improved if needed)
# def majority_vote(series):
#     # Convert values to lowercase and count the number of "yes"
#     yes_count = series.str.lower().eq('yes').sum()
#     no_count = series.str.lower().eq('no').sum()
#     # If the count of yes is greater than or equal to no, return "yes", otherwise "no"
#     return 'yes' if yes_count >= no_count else 'no'

def majority_vote(series):
    # Count the occurrence of each unique value
    counts = series.value_counts()
    # Return the value with the highest count; if there's a tie, the first one is returned
    return counts.idxmax() if not counts.empty else None


def any_yes(series):
    return 'yes' if series.str.lower().eq('yes').any() else 'no'

# Initialize the result list
result_rows = []

# Process groups based on the original file name
grouped = df.groupby('orig_file_name')
for orig_name, group in grouped:
    merged_row = {}
    # Use the original file name for the file_name column
    merged_row['file_name'] = orig_name
    
    # Iterate over the other columns
    for col in group.columns:

        # Skip already processed columns
        if col in ['file_name', 'orig_file_name']:
            continue
        # For "occur_during_sleep", directly take the first row
        if col == 'occur_during_sleep':
            merged_row[col] = group.iloc[0][col]
        elif col == 'gender':
            merged_row[col] = majority_vote(group[col].astype(str))
        elif col == 'limb_movements_pattern':
            # Convert to numeric, ignoring errors (non-numeric become NaN)
            numeric_values = pd.to_numeric(group[col], errors='coerce')
            merged_row[col] = numeric_values.min()
        else:
            merged_row[col] = any_yes(group[col].astype(str))
    
    result_rows.append(merged_row)

# Create a new DataFrame from the result rows
result_df = pd.DataFrame(result_rows)

#result_df['file_name'] = result_df['file_name'].apply(os.path.basename)

check_duplecated(result_df)


# Save the result to a new CSV file
output_csv = "/home/lina/git/SeizureVision/experiment_results/internvl3_78_segment_newaddpatient/visual_segment_feature_merge.csv"
result_df.to_csv(output_csv, index=False)

print("Merge complete, result saved to", output_csv)
