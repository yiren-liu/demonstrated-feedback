import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a CSV file into train and test subsets. The subsets are written "
            "to <output-dir>/<employee>/train.csv and test.csv where <employee> is "
            "derived from the input file name (e.g. allen-p.prompts.csv -> allen-p)."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the CSV file to split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset") / "dataset",
        help="Root directory to place the split dataset (default: dataset/dataset).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of rows to place in the training set (0 < ratio < 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used when shuffling rows prior to splitting.",
    )
    return parser.parse_args()


def validate_inputs(input_csv: Path, train_ratio: float) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")
    if not input_csv.is_file():
        raise ValueError(f"Input path is not a file: {input_csv}")
    if not (0 < train_ratio < 1):
        raise ValueError("--train-ratio must be between 0 and 1 (exclusive).")


def derive_employee_name(input_csv: Path) -> str:
    stem = input_csv.stem  # e.g. allen-p.prompts
    suffix = ".prompts"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def read_rows(input_csv: Path) -> Tuple[List[str], List[List[str]]]:
    with input_csv.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"No header row found in {input_csv}")
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No data rows found in {input_csv}")
    return header, rows


def split_rows(
    rows: Sequence[Sequence[str]],
    train_ratio: float,
    rng: random.Random,
) -> Tuple[List[List[str]], List[List[str]]]:
    mutable_rows = list(rows)
    rng.shuffle(mutable_rows)

    if len(mutable_rows) == 1:
        # Only one row: put it entirely in the train split.
        return [list(mutable_rows[0])], []

    train_count = math.floor(len(mutable_rows) * train_ratio)
    if train_count <= 0:
        train_count = 1
    elif train_count >= len(mutable_rows):
        train_count = len(mutable_rows) - 1

    train_rows = [list(row) for row in mutable_rows[:train_count]]
    test_rows = [list(row) for row in mutable_rows[train_count:]]
    return train_rows, test_rows


def write_split(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        validate_inputs(args.input_csv, args.train_ratio)
        header, rows = read_rows(args.input_csv)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    employee = derive_employee_name(args.input_csv)
    output_dir = args.output_dir / employee
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    train_rows, test_rows = split_rows(rows, args.train_ratio, rng)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    write_split(train_path, header, train_rows)
    write_split(test_path, header, test_rows)

    print(
        f"Wrote {len(train_rows)} train rows to {train_path} and "
        f"{len(test_rows)} test rows to {test_path}"
    )


if __name__ == "__main__":
    main()
