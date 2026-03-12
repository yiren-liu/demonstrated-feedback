import argparse
import json
import logging
import torch
from pathlib import Path
from typing import List, Optional

# Steering-vectors API
from steering_vectors import SteeringVector, train_steering_vector

from multi_vectors.utils import get_model_and_tokenizer, make_train_dataset_from_data, fp32_pca_aggregator


def run_extraction(
    clusters_file: str,
    model,
    tokenizer,
    dataset: str,
    layers: Optional[List[int]] = None,
    output_dir: str = "multi_vectors/outputs",
    logger: Optional[logging.Logger] = None,
) -> str:
    if logger is None:
        logger = logging.getLogger(__name__)

    clusters_path = Path(clusters_file)

    if not clusters_path.exists():
        raise FileNotFoundError(f"Clusters file not found: {clusters_path}")

    # Load clusters metadata
    logger.info(f"Loading clusters from {clusters_path}")
    with clusters_path.open('r') as f:
        clusters_data = json.load(f)

    clusters = clusters_data.get('clusters', [])
    logger.info(f"Found {len(clusters)} clusters")

    # Process each cluster
    base_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = base_dir / output_dir / "vectors" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_name = (
        "layer_all"
        if layers is None
        else f"layer_{'_'.join(map(str, layers))}"
    )

    already_done = 0
    for cluster in clusters:
        # Check if all steering vectors already exist
        cluster_id = cluster['cluster_id']
        out_path = out_dir / \
            f"{clusters_path.stem}_cluster_{cluster_id}_sv_{layer_name}.pt"
        if out_path.exists():
            already_done += 1
            continue

    if already_done == len(clusters):
        logger.info("All steering vectors already exist. Skipping extraction.")
        return str(out_dir)

    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        examples = cluster['examples']
        prompt_count = cluster['prompt_count']

        logger.info("\n" + "=" * 60)
        logger.info(
            f"Processing Cluster {cluster_id} ({prompt_count} examples)")
        logger.info("=" * 60)

        # Generate output filename
        out_path = out_dir / \
            f"{clusters_path.stem}_cluster_{cluster_id}_sv_{layer_name}.pt"

        # Prepare training samples for this cluster
        training_samples = make_train_dataset_from_data(tokenizer, examples)
        logger.info(
            f"Created {len(training_samples)} training samples for cluster {cluster_id}")

        if len(training_samples) == 0:
            logger.warning(
                f"No valid training samples for cluster {cluster_id}, skipping")
            continue

        # Train steering vector for this cluster
        logger.info(f"Training steering vector for cluster {cluster_id}...")
        steering_vector: SteeringVector = train_steering_vector(
            model,
            tokenizer,
            training_samples,
            layers=layers,
            show_progress=True,
        )

        # Save the steering vector
        torch.save(steering_vector, out_path)
        logger.info(f"Saved steering vector to {out_path}")

    logger.info("\n" + "=" * 60)
    logger.info("All individual clusters processed!")
    logger.info("=" * 60)

    # Train vector on all data (all clusters + noise)
    logger.info("\n" + "=" * 60)
    logger.info("Training vector on ALL data (clusters + noise)")
    logger.info("=" * 60)

    all_data_out_path = out_dir / \
        f"{clusters_path.stem}_all_data_sv_{layer_name}.pt"

    if all_data_out_path.exists():
        logger.info(
            f"Steering vector for all data already exists at {all_data_out_path}")
        logger.info("Skipping all data extraction.")
    else:
        # Collect all examples from all clusters
        all_examples = []
        for cluster in clusters:
            all_examples.extend(cluster['examples'])

        # Add noise examples
        noise_examples = clusters_data.get('noise_examples', [])
        all_examples.extend(noise_examples)

        logger.info(
            f"Total examples (all clusters + noise): {len(all_examples)}")

        # Prepare training samples for all data
        all_training_samples = make_train_dataset_from_data(
            tokenizer, all_examples)
        logger.info(
            f"Created {len(all_training_samples)} training samples from all data")

        if len(all_training_samples) > 0:
            # Train steering vector on all data
            logger.info("Training steering vector on all data...")
            all_data_steering_vector: SteeringVector = train_steering_vector(
                model,
                tokenizer,
                all_training_samples,
                layers=layers,
                show_progress=True,
            )

            # Save the steering vector
            torch.save(all_data_steering_vector, all_data_out_path)
            logger.info(
                f"Saved all-data steering vector to {all_data_out_path}")
        else:
            logger.warning("No valid training samples from all data, skipping")

    logger.info("\n" + "=" * 60)
    logger.info("All processing complete!")
    logger.info("=" * 60)

    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract steering vectors for each cluster")
    parser.add_argument("--clusters", type=str, required=True,
                        help="Clusters metadata file name (e.g., v3-gpt-4o-mini-1-500_clusters_hdbscan_pca_metadata.jsonl)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset identifier (e.g., 'email', 'email/persona_1')")
    parser.add_argument("--layers", type=int, nargs="+", default=[],
                        help="Layers to train the steering vector on")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name for extraction (default: meta-llama/Llama-3.1-8B-Instruct)")
    args = parser.parse_args()
    layers = args.layers if args.layers else None

    # Load model and tokenizer for standalone usage
    print(f"Loading model and tokenizer: {args.model}...")
    model, tokenizer = get_model_and_tokenizer(args.model)

    run_extraction(
        clusters_file=args.clusters,
        model=model,
        tokenizer=tokenizer,
        dataset=args.dataset,
        layers=layers,
    )


if __name__ == "__main__":
    main()
