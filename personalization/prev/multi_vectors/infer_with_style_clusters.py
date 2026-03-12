import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    get_model_and_tokenizer,
    generate_batch,
)
from utils.data_utils import make_test_dataset, save_responses
from .infer_utils import (
    load_clusters_metadata,
    load_cluster_prompts_with_embeddings,
    load_steering_vectors,
    generate_with_vectors,
    get_output_paths,
)


def select_steering_vectors(
    prompts: List[str],
    cluster_prompts: List[str],
    cluster_ids: List[int],
    cluster_embeddings: np.ndarray,
    cluster_vectors: Dict[int, object],
    all_data_vector: object,
) -> Tuple[List[object], List[str]]:
    """Select appropriate steering vectors for prompts using KNN."""
    # Compute embeddings for test prompts
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print(f"Computing embeddings for {len(prompts)} test prompts...")
    prompt_embeddings = embedder.encode(
        prompts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Find the nearest neighbors using cosine similarity
    similarities = cosine_similarity(prompt_embeddings, cluster_embeddings)
    nearest_indices = np.argmax(similarities, axis=1)

    selected_vectors = []
    selection_reasons = []

    for i, nearest_idx in enumerate(nearest_indices):
        nearest_cluster_id = cluster_ids[nearest_idx]
        nearest_similarity = similarities[i, nearest_idx]

        # Select steering vector based on the nearest neighbor's cluster
        if nearest_cluster_id == -1:
            # Nearest neighbor is noise, use all_data vector
            selected_vectors.append(all_data_vector)
            selection_reasons.append(
                f"all_data (nearest is noise, sim: {nearest_similarity:.3f})")
        else:
            # Use the steering vector of the nearest neighbor's cluster
            if nearest_cluster_id in cluster_vectors:
                selected_vectors.append(cluster_vectors[nearest_cluster_id])
                selection_reasons.append(
                    f"cluster_{nearest_cluster_id} (knn, sim: {nearest_similarity:.3f})")
            else:
                # Cluster doesn't have a vector, fall back to all_data
                selected_vectors.append(all_data_vector)
                selection_reasons.append(
                    f"all_data (cluster_{nearest_cluster_id} vector not available)")

    return selected_vectors, selection_reasons


def generate_with_cluster_selection(
    model,
    tokenizer,
    chats: List[str],
    prompts: List[str],
    description: str,
    cluster_prompts: List[str],
    cluster_ids: List[int],
    cluster_embeddings: np.ndarray,
    cluster_vectors: Dict[int, object],
    all_data_vector: object,
    multiplier: float = 1.0,
) -> Tuple[List[str], List[str]]:
    """Generate responses with KNN-based cluster vector selection."""
    # Pre-compute all steering vector selections
    selected_vectors, selection_reasons = select_steering_vectors(
        prompts=prompts,
        cluster_prompts=cluster_prompts,
        cluster_ids=cluster_ids,
        cluster_embeddings=cluster_embeddings,
        cluster_vectors=cluster_vectors,
        all_data_vector=all_data_vector,
    )

    # Generate responses using the common utility function
    responses = generate_with_vectors(
        model=model,
        tokenizer=tokenizer,
        chats=chats,
        selected_vectors=selected_vectors,
        description=description,
        multiplier=multiplier,
    )

    return responses, selection_reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference with cluster-based steering vector selection")
    parser.add_argument("--clusters", type=str, required=True,
                        help="Path to clusters metadata file (e.g., v3-gpt-4o-mini-1-500_clusters_hdbscan_pca_metadata.jsonl)")
    parser.add_argument("--dataset", type=str, default="v5-gpt-5-mini",
                        help="Dataset identifier (e.g., v3-gpt-4o-mini-1-500)")
    parser.add_argument("--layers", type=int, nargs="+", default=[],
                        help="Layers used in the steering vector (e.g., 19)")
    parser.add_argument("--multiplier", type=float, default=0.15,
                        help="Steering vector multiplier (default: 0.15)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: generate response for one test datapoint and print result")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    dataset = args.dataset
    layers = args.layers if args.layers else None
    clusters = args.clusters
    dataset_path = base_dir / "dataset" / dataset / "test.json"
    clusters_path = base_dir / "multi_vectors" / "style_clusters" / clusters

    if not clusters_path.exists():
        raise FileNotFoundError(f"Clusters file not found: {clusters_path}")

    # 1. Load clusters metadata
    print(f"Loading clusters metadata from {clusters_path}")
    clusters_data = load_clusters_metadata(clusters_path)
    print(f"Found {len(clusters_data.get('clusters', []))} clusters")

    # 2. Load all cluster prompts with embeddings for KNN
    print("Loading cluster prompts and computing embeddings...")
    cluster_prompts, cluster_ids, cluster_embeddings = load_cluster_prompts_with_embeddings(
        clusters_data)
    print(
        f"Loaded {len(cluster_prompts)} prompts from clusters (including noise)")

    # 3. Load all steering vectors (cluster-specific and all-data)
    cluster_vectors, all_data_vector = load_steering_vectors(
        clusters_data=clusters_data,
        clusters_path=clusters_path,
        layers=layers,
        vectors_subdir="style_vectors",
    )

    # 4. Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model, tokenizer = get_model_and_tokenizer(model_name)

    # 5. Load test dataset
    chats, references, prompts = make_test_dataset(
        tokenizer, dataset_path)

    # Handle test mode
    if args.test:
        print("\n" + "="*80)
        print("TEST MODE: Testing on one datapoint")
        print("="*80)

        # Use first datapoint
        test_chat = [chats[0]]
        test_prompt = [prompts[0]]
        test_reference = references[0]

        print(f"\nPrompt: {test_prompt[0]}")
        print(f"\nReference: {test_reference}")

        # Generate without steering
        print("\n--- Generating WITHOUT steering ---")
        response_without = generate_batch(
            model=model,
            tokenizer=tokenizer,
            chats=test_chat,
            description="Testing without steering",
            use_steering=False,
        )[0]
        print(f"\nResponse (no steering): {response_without}")

        # Generate with steering
        print("\n--- Generating WITH KNN-based cluster steering ---")
        responses_with, selection_reasons = generate_with_cluster_selection(
            model=model,
            tokenizer=tokenizer,
            chats=test_chat,
            prompts=test_prompt,
            description="Testing with KNN-based steering",
            cluster_prompts=cluster_prompts,
            cluster_ids=cluster_ids,
            cluster_embeddings=cluster_embeddings,
            cluster_vectors=cluster_vectors,
            all_data_vector=all_data_vector,
            multiplier=args.multiplier,
        )
        response_with = responses_with[0]
        selection_reason = selection_reasons[0]

        print(f"\nSelected vector: {selection_reason}")
        print(f"\nResponse (with steering): {response_with}")

        print("\n" + "="*80)
        print("Test complete!")
        print("="*80)
        return

    # 6. Define output paths
    without_path, with_path = get_output_paths(
        script_path=Path(__file__),
        dataset=dataset,
        layers=layers,
        method_suffix=f"style_clusters_multiplier{args.multiplier}",
    )

    # Check if results already exist
    if without_path.exists() and with_path.exists():
        print(f"\nSkipping generation - results already exist:")
        print(f"  - {without_path}")
        print(f"  - {with_path}")
        return

    # 7. Generate responses without steering
    if without_path.exists():
        print(
            f"\nSkipping generation without steering - results already exist at {without_path}")
    else:
        responses_without_steering = generate_batch(
            model=model,
            tokenizer=tokenizer,
            chats=chats,
            description="Generating without steering",
            use_steering=False,
        )

        # Save responses without steering
        save_responses(
            output_path=without_path,
            prompts=prompts,
            references=references,
            predictions=responses_without_steering,
        )

    # 8. Generate responses with KNN-based cluster steering
    if with_path.exists():
        print(
            f"\nSkipping generation with KNN steering - results already exist at {with_path}")
    else:
        print(
            f"\nGenerating with KNN-based cluster steering...")
        responses_with_steering, selection_reasons = generate_with_cluster_selection(
            model=model,
            tokenizer=tokenizer,
            chats=chats,
            prompts=prompts,
            description="Generating with KNN-based steering on layer(s): " + (
                (",".join(map(str, layers))) if layers else "all"),
            cluster_prompts=cluster_prompts,
            cluster_ids=cluster_ids,
            cluster_embeddings=cluster_embeddings,
            cluster_vectors=cluster_vectors,
            all_data_vector=all_data_vector,
            multiplier=args.multiplier,
        )

        # Save responses with steering
        save_responses(
            output_path=with_path,
            prompts=prompts,
            references=references,
            predictions=responses_with_steering,
            selection_reasons=selection_reasons,
        )

        # Print cluster selection statistics
        print("\nCluster Selection Statistics:")
        cluster_counts = {}
        for reason in selection_reasons:
            cluster_counts[reason.split(' ')[0]] = cluster_counts.get(
                reason.split(' ')[0], 0) + 1

        for cluster_name, count in sorted(cluster_counts.items()):
            percentage = (count / len(selection_reasons)) * 100
            print(f"  {cluster_name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
