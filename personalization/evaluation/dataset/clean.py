import pandas as pd
import os
from pathlib import Path


def clean_rows(folder_path):
    folder = Path(folder_path)

    # Find the CSV files
    multi_vectors_file = None
    single_vector_file = None
    without_vector_file = None

    for file in folder.glob("*.csv"):
        if "multi_vectors" in file.name:
            multi_vectors_file = file
        elif "single_vector" in file.name:
            single_vector_file = file
        elif "without_vector" in file.name:
            without_vector_file = file

    if not multi_vectors_file:
        print(f"No multi_vectors CSV file found in {folder_path}")
        return

    print(f"Found multi_vectors file: {multi_vectors_file}")
    print(f"Found single_vector file: {single_vector_file}")
    print(f"Found without_vector file: {without_vector_file}")

    # Read the multi_vectors file
    df_multi = pd.read_csv(multi_vectors_file)

    # Find rows where rouge_l_fmeasure is 0.0
    zero_rouge_mask = df_multi['rouge_l_fmeasure'] < 0.1
    rows_to_remove = df_multi[zero_rouge_mask].index.tolist()

    print(f"\nTotal rows in multi_vectors: {len(df_multi)}")
    print(f"Rows with rouge_l_fmeasure < 0.1: {len(rows_to_remove)}")
    print(f"Indexes to remove: {rows_to_remove}")

    if len(rows_to_remove) == 0:
        print("\nNo rows to remove. Exiting.")
        return

    # Remove rows from multi_vectors
    df_multi_cleaned = df_multi.drop(rows_to_remove)
    print(
        f"\nRows remaining in multi_vectors after cleaning: {len(df_multi_cleaned)}")

    # Save cleaned multi_vectors file
    df_multi_cleaned.to_csv(multi_vectors_file, index=False)
    print(f"Saved cleaned multi_vectors file: {multi_vectors_file}")

    # Remove the same rows from single_vector file
    if single_vector_file and single_vector_file.exists():
        df_single = pd.read_csv(single_vector_file)
        print(f"\nTotal rows in single_vector: {len(df_single)}")
        df_single_cleaned = df_single.drop(rows_to_remove)
        df_single_cleaned.to_csv(single_vector_file, index=False)
        print(f"Saved cleaned single_vector file: {single_vector_file}")
        print(f"Rows remaining: {len(df_single_cleaned)}")

    # Remove the same rows from without_vector file
    if without_vector_file and without_vector_file.exists():
        df_without = pd.read_csv(without_vector_file)
        print(f"\nTotal rows in without_vector: {len(df_without)}")
        df_without_cleaned = df_without.drop(rows_to_remove)
        df_without_cleaned.to_csv(without_vector_file, index=False)
        print(f"Saved cleaned without_vector file: {without_vector_file}")
        print(f"Rows remaining: {len(df_without_cleaned)}")

    print(f"\n✓ Successfully cleaned all files!")
    print(f"  Removed {len(rows_to_remove)} rows from each file")


if __name__ == "__main__":
    folder_path = "evaluation/dataset/test/email"
    clean_rows(folder_path)
