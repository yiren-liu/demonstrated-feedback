import argparse
import subprocess
import sys
from pathlib import Path


def run_with_min_cluster_size(min_size: int, dataset: str = None, extra_args: list = None):
    """Run the pipeline with a specific min_cluster_size."""
    print(f"\n{'='*70}")
    print(f"Running with min_cluster_size = {min_size}")
    print(f"{'='*70}\n")

    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "-m",
        "multi_vectors.run",
        "--skip-infer-test",
        "--min-cluster-size",
        str(min_size),
    ]

    # Add dataset if specified
    if dataset:
        cmd.extend(["--dataset", dataset])

    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)

    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print(
            f"\n✓ Successfully completed run with min_cluster_size = {min_size}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed run with min_cluster_size = {min_size}")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sweep min_cluster_size parameter for clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=6,
        help="Starting value for min_cluster_size (default: 6)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=15,
        help="Ending value for min_cluster_size (default: 15)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to use (default: uses run.py default)"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue sweep even if a run fails"
    )

    args, extra_args = parser.parse_known_args()

    print(f"Starting min_cluster_size sweep from {args.start} to {args.end}")
    print(f"{'='*70}")

    if args.dataset:
        print(f"Dataset: {args.dataset}")
    if extra_args:
        print(f"Extra arguments: {' '.join(extra_args)}")

    # Track successes and failures
    successful = []
    failed = []

    # Loop through min_cluster_size values
    for min_size in range(args.start, args.end + 1):
        success = run_with_min_cluster_size(
            min_size=min_size,
            dataset=args.dataset,
            extra_args=extra_args
        )

        if success:
            successful.append(min_size)
        else:
            failed.append(min_size)
            if not args.continue_on_error:
                print("\nStopping sweep due to error.")
                break

    # Print summary
    print(f"\n{'='*70}")
    print("Sweep Summary")
    print(f"{'='*70}")
    print(f"Total runs: {len(successful) + len(failed)}")
    print(f"Successful: {len(successful)}")
    if successful:
        print(f"  Values: {successful}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"  Values: {failed}")
    print(f"{'='*70}")

    # Return exit code based on failures
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
