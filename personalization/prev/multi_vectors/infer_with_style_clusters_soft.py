import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    get_model_and_tokenizer,
    generate_batch,
)
from utils.data_utils import make_test_dataset, save_responses
from multi_vectors.infer_utils import (
    load_clusters_metadata,
    load_cluster_prompts_with_embeddings,
    load_steering_vectors,
    generate_with_vectors,
    get_output_paths,
)


_EMBEDDER = None


def _get_embedder():
    """Lazily load and cache the sentence embedder to avoid repeated loads."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2")
    return _EMBEDDER


def _vector_l2_norm(vector) -> float:
    """Compute an aggregate L2 norm across all layer activations."""
    return sum(torch.linalg.norm(act).item() for act in vector.layer_activations.values())


def select_steering_vectors(
    prompts: List[str],
    cluster_prompts: List[str],
    cluster_ids: List[int],
    cluster_embeddings: np.ndarray,
    cluster_vectors: Dict[int, object],
    all_data_vector: object,
    k: int = 5,
    temperature: float = 1.0,
    min_similarity: float = 0.35,
) -> Tuple[List[object], List[str]]:
    """Select appropriate steering vectors for prompts using top-k KNN with softmax weighting."""
    # Compute embeddings for test prompts
    embedder = _get_embedder()
    print(f"Computing embeddings for {len(prompts)} test prompts...")
    prompt_embeddings = embedder.encode(
        prompts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Find the top-k nearest neighbors using cosine similarity
    similarities = cosine_similarity(prompt_embeddings, cluster_embeddings)

    # Get top-k indices and similarities for each prompt
    top_k_indices = np.argsort(similarities, axis=1)[
        :, -k:][:, ::-1]  # Shape: (n_prompts, k)

    selected_vectors = []
    selection_reasons = []

    for i in range(len(prompts)):
        # Get top-k neighbors and their similarities
        neighbor_indices = top_k_indices[i]
        neighbor_similarities = similarities[i, neighbor_indices]
        neighbor_cluster_ids = [cluster_ids[idx] for idx in neighbor_indices]

        top_similarity = float(neighbor_similarities[0]) if len(
            neighbor_similarities) else -1.0
        if top_similarity < min_similarity:
            # Low confidence match; prefer the stable all-data vector
            selected_vectors.append(all_data_vector)
            selection_reasons.append(
                f"all_data (top_sim<{min_similarity:.2f}, actual:{top_similarity:.3f})")
            continue

        # Compute softmax weights based on similarities (higher similarity = higher weight)
        # Apply temperature scaling
        scaled_sims = neighbor_similarities / temperature
        # Subtract max for numerical stability
        exp_sims = np.exp(scaled_sims - np.max(scaled_sims))
        weights = exp_sims / np.sum(exp_sims)

        # Collect vectors and their weights for each cluster
        cluster_weight_map = {}  # cluster_id -> total_weight

        for j, (cluster_id, weight) in enumerate(zip(neighbor_cluster_ids, weights)):
            # Skip noise points or clusters without vectors
            if cluster_id == -1 or cluster_id not in cluster_vectors:
                continue

            # Accumulate weights for the same cluster
            if cluster_id not in cluster_weight_map:
                cluster_weight_map[cluster_id] = 0.0
            cluster_weight_map[cluster_id] += weight

        # If no valid clusters found, use all_data vector
        if not cluster_weight_map:
            selected_vectors.append(all_data_vector)
            selection_reasons.append(
                f"all_data (no valid clusters in top-{k}, max_sim: {neighbor_similarities[0]:.3f})")
            continue

        # Normalize weights if we filtered out some neighbors
        total_valid_weight = sum(cluster_weight_map.values())
        if total_valid_weight > 0:
            cluster_weight_map = {
                cid: w / total_valid_weight for cid, w in cluster_weight_map.items()}

        # Create combined steering vector by weighted average
        combined_vector = None
        expected_layers = None
        reason_parts = []
        target_norm = 0.0

        for cluster_id, weight in sorted(cluster_weight_map.items(), key=lambda x: -x[1]):
            vec = cluster_vectors[cluster_id]
            layers = set(vec.layer_activations.keys())

            if expected_layers is None:
                expected_layers = layers
            elif layers != expected_layers:
                print(
                    f"Warning: Cluster {cluster_id} vector layers {layers} mismatch expected {expected_layers}; skipping this vector.")
                continue

            vec_norm = _vector_l2_norm(vec)
            target_norm += weight * vec_norm

            if combined_vector is None:
                # Initialize with first vector (weighted)
                weighted_activations = {
                    layer: activation * weight
                    for layer, activation in vec.layer_activations.items()
                }
                combined_vector = type(vec)(
                    weighted_activations, vec.layer_type)
            else:
                # Add weighted contribution
                combined_vector.layer_activations = {
                    layer: combined_vector.layer_activations[layer] +
                    vec.layer_activations[layer] * weight
                    for layer in combined_vector.layer_activations.keys()
                }

            # Add to reason string (only show top 3 clusters for brevity)
            if len(reason_parts) < 3:
                reason_parts.append(f"c{cluster_id}({weight:.2f})")

        if combined_vector is None:
            selected_vectors.append(all_data_vector)
            selection_reasons.append(
                f"all_data (no compatible vectors after filtering, top_sim:{top_similarity:.3f})")
            continue

        # Normalize combined vector to expected magnitude to avoid over/under-steering
        combined_norm = _vector_l2_norm(combined_vector)
        if combined_norm > 0 and target_norm > 0:
            scale = target_norm / combined_norm
            for layer in combined_vector.layer_activations:
                combined_vector.layer_activations[layer] = (
                    combined_vector.layer_activations[layer] * scale
                )

        selected_vectors.append(combined_vector)
        selection_reasons.append(
            f"weighted_avg: {'+'.join(reason_parts)} (top-{k}, top_sim:{top_similarity:.3f})")

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
    k: int = 5,
    temperature: float = 1.0,
    min_similarity: float = 0.35,
) -> Tuple[List[str], List[str]]:
    """Generate responses with top-k KNN-based cluster vector selection."""
    # Pre-compute all steering vector selections
    selected_vectors, selection_reasons = select_steering_vectors(
        prompts=prompts,
        cluster_prompts=cluster_prompts,
        cluster_ids=cluster_ids,
        cluster_embeddings=cluster_embeddings,
        cluster_vectors=cluster_vectors,
        all_data_vector=all_data_vector,
        k=k,
        temperature=temperature,
        min_similarity=min_similarity,
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
    parser.add_argument("--dataset", type=str, default="v5-gpt-5-mini-update-v2",
                        help="Dataset identifier (e.g., v5-gpt-5-mini-update-v2)")
    parser.add_argument("--layers", type=int, nargs="+", default=[],
                        help="Layers used in the steering vector (e.g., 19)")
    parser.add_argument("--multiplier", type=float, default=0.15,
                        help="Steering vector multiplier (default: 0.15)")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of nearest neighbors for KNN (default: 3)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for softmax weighting (default: 1.0)")
    parser.add_argument("--min-similarity", type=float, default=0.35,
                        help="Minimum top-1 similarity to apply cluster steering; otherwise fallback to all-data vector (default: 0.35)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: generate response for one test datapoint and print result")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    dataset = args.dataset
    layers = args.layers if args.layers else None
    clusters = args.clusters
    min_similarity = args.min_similarity
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
        print(
            f"\n--- Generating WITH top-{args.k} KNN-based cluster steering (temp={args.temperature}) ---")
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
            k=args.k,
            temperature=args.temperature,
            min_similarity=min_similarity,
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
        method_suffix=f"style_clusters_soft_k{args.k}_temp{args.temperature}_mins{min_similarity}_multiplier{args.multiplier}",
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
            f"\nGenerating with top-{args.k} KNN-based cluster steering (temp={args.temperature})...")
        responses_with_steering, selection_reasons = generate_with_cluster_selection(
            model=model,
            tokenizer=tokenizer,
            chats=chats,
            prompts=prompts,
            description=f"Generating with top-{args.k} KNN-based steering on layer(s): " + (
                (",".join(map(str, layers))) if layers else "all"),
            cluster_prompts=cluster_prompts,
            cluster_ids=cluster_ids,
            cluster_embeddings=cluster_embeddings,
            cluster_vectors=cluster_vectors,
            all_data_vector=all_data_vector,
            multiplier=args.multiplier,
            k=args.k,
            temperature=args.temperature,
            min_similarity=min_similarity,
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
