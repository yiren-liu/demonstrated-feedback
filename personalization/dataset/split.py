import json
import random
import csv
import sys
from pathlib import Path


def split_dataset(
    input_file,
    train_file,
    val_file,
    test_file,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """Split dataset into train, validation, and test sets. Supports both JSON and CSV input files."""
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

    # Calculate split points
    total_size = len(shuffled_data)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - test_size - val_size

    # Split the data
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Save train set
    print(f"Saving training set to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    # Save validation set
    print(f"Saving validation set to {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)

    # Save test set
    print(f"Saving test set to {test_file}...")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print("Split complete!")
    train_pct = (train_size / total_size) * 100
    val_pct = (val_size / total_size) * 100
    test_pct = (test_size / total_size) * 100
    print(f"  Train: {train_size} samples ({train_pct:.1f}%)")
    print(f"  Validation: {val_size} samples ({val_pct:.1f}%)")
    print(f"  Test: {test_size} samples ({test_pct:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split.py <input_dataset_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    input_path = Path(input_file)
    output_dir = input_path.parent

    # Output files in the same directory as input
    train_file = output_dir / "train.json"
    val_file = output_dir / "val.json"
    test_file = output_dir / "test.json"

    split_dataset(
        input_file=input_file,
        train_file=str(train_file),
        val_file=str(val_file),
        test_file=str(test_file),
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
