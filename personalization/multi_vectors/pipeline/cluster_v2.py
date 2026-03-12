import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from multi_vectors.utils import load_train_rows


def cluster_by_scenario(dataset: str, output_dir: str = "multi_vectors/outputs") -> str:
    """
    Cluster data points by scenario_description.
    Data points with the same scenario_description are placed in the same cluster.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_path = base_dir / "dataset" / dataset / "train.json"

    # Build output directory path maintaining dataset structure
    clusters_base_dir = base_dir / output_dir / "clusters"
    dataset_parts = Path(dataset).parts
    if len(dataset_parts) > 1:
        # For nested datasets like "email/persona_1"
        output_dir = clusters_base_dir / dataset_parts[0]
        filename_prefix = dataset_parts[1]
    else:
        # For flat datasets
        output_dir = clusters_base_dir
        filename_prefix = dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if cluster metadata already exists (with timestamp)
    existing_files = sorted(output_dir.glob(
        f"{filename_prefix}_scenario_clusters_*.json"))
    if existing_files:
        clusters_filepath = existing_files[-1]
        print(f"Cluster metadata already exists at {clusters_filepath}")
        print("Skipping clustering step.")
        return str(clusters_filepath)

    # Load data
    print(f"Loading training data from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} data points")

    # Group by scenario_description
    scenario_to_indices = defaultdict(list)
    for idx, item in enumerate(data):
        scenario = item.get("contextual_writing_style", {}).get(
            "scenario_description", "unknown")
        scenario_to_indices[scenario].append(idx)

    print(f"\nFound {len(scenario_to_indices)} unique scenarios:")
    for scenario, indices in sorted(scenario_to_indices.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  - '{scenario}': {len(indices)} samples")

    # Prepare output metadata in the same format as cluster.py
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / \
        f"{filename_prefix}_scenario_clusters_{timestamp}.json"

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build cluster metadata in the expected format
    cluster_metadata = []
    for cluster_id, (scenario, indices) in enumerate(scenario_to_indices.items()):
        examples = [
            {
                "index": idx,
                "prompt": data[idx].get("prompts", ""),
                "personalized": data[idx].get("personalized", ""),
                "style_agnostic": data[idx].get("style_agnostic", ""),
            }
            for idx in indices
        ]
        cluster_metadata.append({
            "cluster_id": cluster_id,
            "prompt_count": len(indices),
            "scenario_description": scenario,
            "examples": examples,
        })

    # Create the full metadata payload
    metadata_payload = {
        "embedding_model": "N/A (grouped by scenario_description)",
        "clustering_algorithm": "scenario_grouping",
        "clustering_params": {
            "method": "Group by scenario_description field",
            "num_clusters": len(scenario_to_indices)
        },
        "clusters": cluster_metadata,
        "noise_examples": []  # No noise when clustering by ground truth
    }

    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(metadata_payload, f, indent=2)

    print(f"\nCluster metadata written to {output_path}")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster data points by scenario_description"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="email/persona_1",
        help="Dataset name (default: email/persona_1)",
    )

    args = parser.parse_args()
    cluster_by_scenario(dataset=args.dataset)


if __name__ == "__main__":
    main()
