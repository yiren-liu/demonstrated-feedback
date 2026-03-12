import json
import random
import csv
import sys
from pathlib import Path


def split_dataset(
    input_file,
    train_file,
    test_file,
    test_ratio=0.2,
    random_seed=42
):
    """Split dataset into train and test sets. Supports both JSON and CSV input files."""
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Read the dataset
    print(f"Reading dataset from {input_file}...")
    input_path = Path(input_file)

    if input_path.suffix.lower() == '.csv':
        # Read CSV file
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    else:
        # Read JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Rename fields
    def rename_fields(item):
        """Rename fields to match expected naming convention."""
        renamed = {}
        if "scenario" in item:
            renamed["prompt"] = item["scenario"]
        # Keep any other fields as-is
        for key, value in item.items():
            if key != "scenario":
                renamed[key] = value
        return renamed

    # Rename fields in all data
    shuffled_data = [rename_fields(item) for item in shuffled_data]

    # Calculate split point
    test_size = int(len(shuffled_data) * test_ratio)
    train_size = len(shuffled_data) - test_size

    # Split the data
    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Save train set
    print(f"Saving training set to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    # Save test set
    print(f"Saving test set to {test_file}...")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print("Split complete!")
    print(f"  Train: {train_size} samples ({(1-test_ratio)*100:.1f}%)")
    print(f"  Test: {test_size} samples ({test_ratio*100:.1f}%)")


if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

        # Determine output file paths
        input_path = Path(input_file)
        output_dir = input_path.parent

        # Default output files in the same directory as input
        train_file = output_dir / "train.json"
        test_file = output_dir / "test.json"

        # Allow custom output files as arguments
        if len(sys.argv) > 2:
            train_file = sys.argv[2]
        if len(sys.argv) > 3:
            test_file = sys.argv[3]

        split_dataset(
            input_file=input_file,
            train_file=str(train_file),
            test_file=str(test_file),
            test_ratio=0.2,
            random_seed=42
        )
    else:
        # Default behavior
        split_dataset(
            input_file="dataset/v5-gpt-5-mini-update-v2/dataset.json",
            train_file="dataset/v5-gpt-5-mini-update-v2/train.json",
            test_file="dataset/v5-gpt-5-mini-update-v2/test.json",
            test_ratio=0.2,  # 20% for test, 80% for train
            random_seed=42   # For reproducibility
        )
