import argparse
from pathlib import Path
import logging
from typing import Dict, List, Optional

import numpy as np

from multi_vectors.utils import load_train_rows
from multi_vectors.utils.cluster_utils import (
    build_embeddings,
    cluster_embeddings,
    save_cluster_metadata,
    set_seed,
)


def run_clustering(
    dataset: str,
    algorithm: str = "hdbscan",
    pca: bool = True,
    embedding_model: str = "AnnaWegmann/Style-Embedding",
    min_cluster_size: int = 15,
    output_dir: str = "multi_vectors/outputs",
    logger: Optional[logging.Logger] = None,
) -> str:
    if logger is None:
        logger = logging.getLogger(__name__)
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_path = base_dir / "dataset" / dataset / "train.json"
    output_dir = base_dir / output_dir / "clusters"

    # Build the expected metadata filename (without timestamp)
    filename_parts = [dataset, "style_clusters", algorithm]
    if pca:
        filename_parts.append("pca")
    filename_parts.append(f"mcs{min_cluster_size}")
    filename = "_".join(filename_parts) + "_metadata.jsonl"
    metadata_path = output_dir / filename

    # Ensure parent directory exists for nested dataset paths
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if cluster metadata already exists (with timestamp)
    existing_files = sorted(metadata_path.parent.glob(
        f"{metadata_path.stem}_*.jsonl"))
    if existing_files:
        clusters_filepath = existing_files[-1]
        logger.info(f"Cluster metadata already exists at {clusters_filepath}")
        logger.info("Skipping clustering step.")
        return str(clusters_filepath)

    set_seed()

    # 1. Load and clean data
    logger.info("Loading training data...")
    rows = load_train_rows(dataset_path)
    prompts = rows["prompts"]
    logger.info(f"Loaded {len(prompts)} prompts")

    # 2. Build embeddings
    logger.info(f"Building embeddings using {embedding_model}...")
    embeddings = build_embeddings(prompts, model_name=embedding_model)

    # 3. Cluster embeddings
    logger.info(f"Clustering with {algorithm}...")
    cluster_info = cluster_embeddings(
        embeddings,
        apply_dimensionality_reduction=pca,
        algorithm=algorithm,
        min_cluster_size=min_cluster_size,
    )
    logger.info(
        f"Identified {len(cluster_info['label_to_indices'])} clusters "
        f"with {len(cluster_info['noise_indices'])} noise points"
    )

    # 4. Save cluster metadata
    rows_with_cleaned_prompts = {
        "prompts": prompts,
        "personalized": rows["personalized"],
        "style_agnostic": rows["style_agnostic"],
    }
    clusters_filepath = save_cluster_metadata(
        cluster_info, rows_with_cleaned_prompts, metadata_path, embedding_model=embedding_model)

    logger.info(f"Cluster metadata written to {clusters_filepath}")
    return str(clusters_filepath)


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
    run_clustering(
        dataset=args.dataset,
        algorithm=args.algorithm,
        pca=args.pca,
    )


if __name__ == "__main__":
    main()
