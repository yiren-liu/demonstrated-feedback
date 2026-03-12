import argparse
import json
import torch
from pathlib import Path

# Steering-vectors API
from steering_vectors import SteeringVector, train_steering_vector

from utils import get_model_and_tokenizer, make_train_dataset_from_data, fp32_pca_aggregator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract steering vectors for each cluster")
    parser.add_argument("--clusters", type=str, required=True,
                        help="Clusters metadata file name (e.g., v3-gpt-4o-mini-1-500_clusters_hdbscan_pca_metadata.jsonl)")
    parser.add_argument("--layers", type=int, nargs="+", default=[],
                        help="Layers to train the steering vector on")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    clusters = args.clusters
    clusters_path = base_dir / "multi_vectors" / "style_clusters" / clusters
    layers = args.layers if args.layers else None

    if not clusters_path.exists():
        raise FileNotFoundError(f"Clusters file not found: {clusters_path}")

    # Load clusters metadata
    print(f"Loading clusters from {clusters_path}")
    with clusters_path.open('r') as f:
        clusters_data = json.load(f)

    clusters = clusters_data.get('clusters', [])
    print(f"Found {len(clusters)} clusters")

    # Load model & tokenizer (load once for all clusters)
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(
        "meta-llama/Llama-3.1-8B-Instruct")

    # Process each cluster
    out_dir = Path(__file__).resolve().parent / "style_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        examples = cluster['examples']
        prompt_count = cluster['prompt_count']

        print(f"\n{'='*60}")
        print(f"Processing Cluster {cluster_id} ({prompt_count} examples)")
        print(f"{'='*60}")

        # Generate output filename
        layer_name = (
            "layer_all"
            if layers is None
            else f"layer_{'_'.join(map(str, layers))}"
        )
        out_path = out_dir / \
            f"{clusters_path.stem}_cluster_{cluster_id}_sv_{layer_name}.pt"

        # Check if output already exists
        if out_path.exists():
            print(f"Steering vector already exists at {out_path}")
            print("Skipping this cluster.")
            continue

        # Prepare training samples for this cluster
        training_samples = make_train_dataset_from_data(tokenizer, examples)
        print(
            f"Created {len(training_samples)} training samples for cluster {cluster_id}")

        if len(training_samples) == 0:
            print(
                f"Warning: No valid training samples for cluster {cluster_id}, skipping")
            continue

        # Train steering vector for this cluster
        print(f"Training steering vector for cluster {cluster_id}...")
        steering_vector: SteeringVector = train_steering_vector(
            model,
            tokenizer,
            training_samples,
            layers=layers,
            show_progress=True,
            # aggregator=fp32_pca_aggregator()
        )

        # Save the steering vector
        torch.save(steering_vector, out_path)
        print(f"Saved steering vector to {out_path}")

    print(f"\n{'='*60}")
    print("All individual clusters processed!")
    print(f"{'='*60}")

    # Train vector on all data (all clusters + noise)
    print(f"\n{'='*60}")
    print("Training vector on ALL data (clusters + noise)")
    print(f"{'='*60}")

    all_data_out_path = out_dir / \
        f"{clusters_path.stem}_all_data_sv_{layer_name}.pt"

    if all_data_out_path.exists():
        print(
            f"Steering vector for all data already exists at {all_data_out_path}")
        print("Skipping all data extraction.")
    else:
        # Collect all examples from all clusters
        all_examples = []
        for cluster in clusters:
            all_examples.extend(cluster['examples'])

        # Add noise examples
        noise_examples = clusters_data.get('noise_examples', [])
        all_examples.extend(noise_examples)

        print(f"Total examples (all clusters + noise): {len(all_examples)}")

        # Prepare training samples for all data
        all_training_samples = make_train_dataset_from_data(
            tokenizer, all_examples)
        print(
            f"Created {len(all_training_samples)} training samples from all data")

        if len(all_training_samples) > 0:
            # Train steering vector on all data
            print("Training steering vector on all data...")
            all_data_steering_vector: SteeringVector = train_steering_vector(
                model,
                tokenizer,
                all_training_samples,
                layers=layers,
                show_progress=True,
                # aggregator=fp32_pca_aggregator()
            )

            # Save the steering vector
            torch.save(all_data_steering_vector, all_data_out_path)
            print(f"Saved all-data steering vector to {all_data_out_path}")
        else:
            print("Warning: No valid training samples from all data, skipping")

    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
