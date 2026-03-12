import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils import load_train_rows
from multi_vectors.cluster_utils import (
    build_embeddings,
    cluster_embeddings,
    save_cluster_metadata,
    set_seed,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Cluster embeddings from prompts")
    parser.add_argument(
        "--dataset",
        type=str,
        default="v5-gpt-5-mini-update-v2",
        help="Dataset name (default: v5-gpt-5-mini-update-v2)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="hdbscan",
        choices=["hdbscan", "dbscan"],
        help="Clustering algorithm to use (default: hdbscan)",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        default=True,
        help="Apply PCA dimensionality reduction (default: True)",
    )

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "dataset" / args.dataset / "train.json"
    output_dir = base_dir / "multi_vectors" / "style_clusters"

    set_seed()

    # 1. Load and clean data
    rows = load_train_rows(dataset_path)
    prompts = rows["personalized"]

    # 2. Build embeddings
    embeddings = build_embeddings(prompts)

    # 3. Cluster embeddings
    cluster_info = cluster_embeddings(
        embeddings,
        apply_dimensionality_reduction=args.pca,
        algorithm=args.algorithm,
    )
    print(f"Identified {len(cluster_info['label_to_indices'])} clusters "
          f"with {len(cluster_info['noise_indices'])} noise points.")

    # 4. Save cluster metadata
    filename_parts = [args.dataset, "style_clusters", args.algorithm]
    if args.pca:
        filename_parts.append("pca")
    filename = "_".join(filename_parts) + "_metadata.jsonl"
    metadata_path = output_dir / filename
    rows_with_cleaned_prompts = {
        "prompts": prompts,
        "personalized": rows["personalized"],
        "style_agnostic": rows["style_agnostic"],
    }
    save_cluster_metadata(
        cluster_info, rows_with_cleaned_prompts, metadata_path)
    print(f"Cluster metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
