import pandas as pd
import glob

# Get all CSV files matching the pattern
csv_files = sorted(glob.glob('api_data_aadhar_demographic_*.csv'))

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {f}")

# Read and concatenate all CSV files
dfs = []
for csv_file in csv_files:
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    dfs.append(df)
    print(f"  Rows: {len(df)}")

# Concatenate all dataframes
print("\nMerging all files...")
merged_df = pd.concat(dfs, ignore_index=True)

# Save to new file
output_file = 'api_data_aadhar_demographic_merged.csv'
print(f"\nSaving merged data to {output_file}...")
merged_df.to_csv(output_file, index=False)

print(f"\nDone! Total rows in merged file: {len(merged_df)}")
