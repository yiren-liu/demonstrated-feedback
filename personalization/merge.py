import os
import pandas as pd
import glob
import argparse


def merge_csv_files(folder_path, output_path="merged.csv"):
    # Get all CSV files in the folder
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    print(f"Found {len(csv_files)} CSV file(s) to merge:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add filename column
            df['filename'] = os.path.basename(csv_file)
            dfs.append(df)
            print(f"Loaded {csv_file}: {len(df)} rows")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not dfs:
        print("No CSV files could be loaded successfully")
        return

    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Add or reset index column
    if 'index' in merged_df.columns:
        merged_df.drop('index', axis=1, inplace=True)
    merged_df.insert(0, 'index', range(len(merged_df)))

    # Save to output file
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merged_df.to_csv(output_path, index=False)

    print(f"\nMerged {len(dfs)} file(s) into {output_path}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Columns: {list(merged_df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge CSV files from a folder into a single CSV file"
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing CSV files (e.g., single_vector/outputs/results/test)"
    )
    parser.add_argument(
        "-o", "--output",
        default="merged.csv",
        help="Output file path (default: merged.csv in current directory)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return

    merge_csv_files(args.folder, args.output)


if __name__ == "__main__":
    main()
