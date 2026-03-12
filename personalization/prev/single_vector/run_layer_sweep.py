#!/usr/bin/env python3
"""
Layer Sweep Script

This script runs extract.py and infer.py with different single layers (from 15 to 31)
and evaluates style similarity for each layer to find the optimal layer.

Usage:
    python run_layer_sweep.py --dataset v5-gpt-5-mini [--start_layer 15] [--end_layer 31]
"""

import argparse
import subprocess
import sys
from pathlib import Path
import csv


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(
            f"ERROR: {description} failed with return code {result.returncode}")
        return False

    return True


def extract_style_similarity(csv_path):
    """Extract style similarity score from CSV file"""
    try:
        # Run evaluate.py and capture output
        cmd = [
            sys.executable, "-m", "evaluate",
            "--csv_path", str(csv_path),
            "--metrics", "style"
        ]

        result = subprocess.run(cmd, capture_output=True,
                                text=True, cwd=Path(__file__).parent.parent)

        # Parse the output to find style similarity score
        output = result.stdout

        # Look for "Average cosine similarity" in the output
        for line in output.split('\n'):
            if 'Average cosine similarity' in line or 'average' in line.lower():
                # Try to extract the number
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        score = float(parts[-1].strip())
                        return score
                    except ValueError:
                        continue

        # If we can't parse, return None
        print(f"Warning: Could not parse style similarity from output")
        return None

    except Exception as e:
        print(f"Error extracting style similarity: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run layer sweep for steering vector optimization"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset identifier (e.g., v5-gpt-5-mini)"
    )
    parser.add_argument(
        "--start_layer", type=int, default=15,
        help="Starting layer (default: 15)"
    )
    parser.add_argument(
        "--end_layer", type=int, default=31,
        help="Ending layer (default: 31, last layer for Llama-3.1-8B)"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model identifier"
    )
    parser.add_argument(
        "--multiplier", type=float, default=3.0,
        help="Steering multiplier (default: 3.0)"
    )

    args = parser.parse_args()

    # Store results
    results = []

    print(f"\n{'#'*60}")
    print(f"# Layer Sweep: {args.dataset}")
    print(f"# Layers: {args.start_layer} to {args.end_layer}")
    print(f"{'#'*60}\n")

    base_dir = Path(__file__).parent

    # Iterate through each layer
    for layer in range(args.start_layer, args.end_layer + 1):
        print(f"\n{'*'*60}")
        print(f"* Processing Layer {layer}")
        print(f"{'*'*60}")

        # 1. Run extract.py
        extract_cmd = [
            sys.executable, str(base_dir / "extract.py"),
            "--dataset", args.dataset,
            "--layers", str(layer),
            "--model", args.model,
        ]

        if not run_command(extract_cmd, f"Extracting steering vector for layer {layer}"):
            print(f"Skipping layer {layer} due to extraction error")
            continue

        # 2. Run infer.py
        infer_cmd = [
            sys.executable, str(base_dir / "infer.py"),
            "--dataset", args.dataset,
            "--layers", str(layer),
            "--model", args.model,
            "--multiplier", str(args.multiplier),
        ]

        if not run_command(infer_cmd, f"Running inference for layer {layer}"):
            print(f"Skipping layer {layer} due to inference error")
            continue

        # 3. Evaluate style similarity
        csv_path = base_dir / "outputs" / \
            f"{args.dataset}_test_results_with_steering_layer_{layer}.csv"

        if not csv_path.exists():
            print(f"Warning: Output file not found at {csv_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating style similarity for layer {layer}")
        print(f"{'='*60}")

        style_similarity = extract_style_similarity(csv_path)

        if style_similarity is not None:
            results.append({
                'layer': layer,
                'style_similarity': style_similarity,
                'csv_path': str(csv_path)
            })
            print(
                f"\n✓ Layer {layer}: Style Similarity = {style_similarity:.4f}")
        else:
            print(f"\n✗ Layer {layer}: Could not compute style similarity")

    # Print summary
    print(f"\n\n{'#'*60}")
    print(f"# SUMMARY - Layer Sweep Results")
    print(f"{'#'*60}\n")

    if not results:
        print("No results to display. All layers failed.")
        return

    print(f"{'Layer':<10} {'Style Similarity':<20}")
    print(f"{'-'*30}")

    for result in results:
        print(f"{result['layer']:<10} {result['style_similarity']:<20.4f}")

    # Find best layer
    best_result = max(results, key=lambda x: x['style_similarity'])
    print(f"\n{'='*60}")
    print(f"BEST LAYER: {best_result['layer']}")
    print(f"Style Similarity: {best_result['style_similarity']:.4f}")
    print(f"{'='*60}\n")

    # Save results to CSV
    results_csv = base_dir / "outputs" / \
        f"{args.dataset}_layer_sweep_results.csv"
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['layer', 'style_similarity', 'csv_path'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {results_csv}")


if __name__ == "__main__":
    main()
