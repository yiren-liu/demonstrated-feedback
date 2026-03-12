import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from multi_vectors.utils import (
    get_model_and_tokenizer,
    generate_batch,
    make_test_dataset,
    save_responses,
)
from multi_vectors.utils.infer_utils import (
    load_clusters_metadata,
    load_cluster_prompts_with_embeddings,
    load_steering_vectors,
    generate_with_vectors,
    get_output_paths,
)


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
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[object], List[str]]:
    """Select appropriate steering vectors for prompts using top-k KNN with softmax weighting."""
    if logger is None:
        logger = logging.getLogger(__name__)

    # Compute embeddings for test prompts
    embedder = SentenceTransformer(embedding_model)
    logger.info(f"Computing embeddings for {len(prompts)} test prompts...")
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
        noise_count = 0

        for j, (cluster_id, weight) in enumerate(zip(neighbor_cluster_ids, weights)):
            # Skip noise points
            if cluster_id == -1:
                noise_count += 1
                continue

            # All clusters should have vectors; raise error if not found
            if cluster_id not in cluster_vectors:
                raise ValueError(
                    f"Cluster {cluster_id} found in top-{k} neighbors but vector not loaded. "
                    f"This should not happen - check vector loading logic.")

            # Accumulate weights for the same cluster
            if cluster_id not in cluster_weight_map:
                cluster_weight_map[cluster_id] = 0.0
            cluster_weight_map[cluster_id] += weight

        # If no valid clusters found, use all_data vector
        if not cluster_weight_map:
            selected_vectors.append(all_data_vector)
            selection_reasons.append(
                f"all_data (all top-{k} are noises, max_sim: {neighbor_similarities[0]:.3f})")
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
                raise ValueError(
                    f"Cluster {cluster_id} vector layers {layers} mismatch expected {expected_layers}. "
                    f"All cluster vectors must have the same layers. Check vector generation process.")

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
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    logger: Optional[logging.Logger] = None,
    batch_size: int = 2,
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
        embedding_model=embedding_model,
        logger=logger,
    )

    # Generate responses using the common utility function
    responses = generate_with_vectors(
        model=model,
        tokenizer=tokenizer,
        chats=chats,
        selected_vectors=selected_vectors,
        description=description,
        multiplier=multiplier,
        batch_size=batch_size,
    )

    return responses, selection_reasons


def run_inference(
    clusters_file: str,
    vectors_path: str,
    dataset: str,
    model,
    tokenizer,
    data_file: str = "val.json",
    layers: Optional[List[int]] = None,
    k: int = 3,
    temperature: float = 1.0,
    multiplier: float = 0.15,
    min_similarity: float = 0.35,
    test_mode: bool = False,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    output_dir: str = "multi_vectors/outputs",
    logger: Optional[logging.Logger] = None,
    batch_size: int = 2,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_path = base_dir / "dataset" / dataset / data_file

    clusters_path = Path(clusters_file)

    if not clusters_path.exists():
        raise FileNotFoundError(f"Clusters file not found: {clusters_path}")

    # 0. Define output paths
    # Determine the split name (val or test) from data_file
    split_name = Path(data_file).stem  # e.g., "val" or "test"

    # Extract min_cluster_size from clusters filename if present (e.g., "mcs15")
    cluster_stem = clusters_path.stem
    mcs_suffix = ""
    if "_mcs" in cluster_stem:
        # Extract the mcs value from the filename
        mcs_part = cluster_stem.split("_mcs")[1].split("_")[0]
        mcs_suffix = f"_mcs{mcs_part}"

    without_path, with_path = get_output_paths(
        script_path=Path(__file__),
        dataset=dataset,
        layers=layers,
        output_dir=output_dir,
        method_suffix=f"style_clusters_soft_k{k}_temp{temperature}_mins{min_similarity}_multiplier{multiplier}{mcs_suffix}",
        split=split_name,  # Add split parameter
    )
    existing_with_paths = sorted(
        with_path.parent.glob(f"{with_path.stem[:-16]}_*.csv")
    )

    # Check if results already exist
    if without_path.exists() and (with_path.exists() or existing_with_paths):
        logger.info("Skipping generation - results already exist:")
        logger.info(f"  - {without_path}")
        if with_path.exists():
            logger.info(f"  - {with_path}")
        else:
            for path in existing_with_paths:
                logger.info(f"  - {path}")
        return

    # 1. Load clusters metadata
    logger.info(f"Loading clusters metadata from {clusters_path}")
    clusters_data = load_clusters_metadata(clusters_path)
    logger.info(f"Found {len(clusters_data.get('clusters', []))} clusters")

    # 2. Load all cluster prompts with embeddings for KNN
    logger.info(
        f"Loading cluster prompts and computing embeddings using {embedding_model}...")
    cluster_prompts, cluster_ids, cluster_embeddings = load_cluster_prompts_with_embeddings(
        clusters_data, embedding_model=embedding_model)
    logger.info(
        f"Loaded {len(cluster_prompts)} prompts from clusters (including noise)")

    # 3. Load all steering vectors (cluster-specific and all-data)
    cluster_vectors, all_data_vector = load_steering_vectors(
        clusters_data=clusters_data,
        clusters_path=clusters_path,
        layers=layers,
        vectors_path=vectors_path,
    )

    # 4. Load test dataset
    chats, references, prompts = make_test_dataset(tokenizer, dataset_path)

    # Handle test mode
    if test_mode:
        logger.info("\n" + "=" * 80)
        logger.info("TEST MODE: Testing on one datapoint")
        logger.info("=" * 80)

        # Use first datapoint
        test_chat = [chats[0]]
        test_prompt = [prompts[0]]
        test_reference = references[0]

        logger.info(f"\nPrompt: {test_prompt[0]}")
        logger.info(f"\nReference: {test_reference}")

        # Generate without steering
        logger.info("\n--- Generating WITHOUT steering ---")
        response_without = generate_batch(
            model=model,
            tokenizer=tokenizer,
            chats=test_chat,
            description="Testing without steering",
            use_steering=False,
        )[0]
        logger.info(f"\nResponse (no steering): {response_without}")

        # Generate with steering
        logger.info(
            f"\n--- Generating WITH top-{k} KNN-based cluster steering (temp={temperature}) ---")
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
            multiplier=multiplier,
            k=k,
            temperature=temperature,
            min_similarity=min_similarity,
            embedding_model=embedding_model,
            logger=logger,
            batch_size=batch_size,
        )
        response_with = responses_with[0]
        selection_reason = selection_reasons[0]

        logger.info(f"\nSelected vector: {selection_reason}")
        logger.info(f"\nResponse (with steering): {response_with}")

        logger.info("\n" + "=" * 80)
        logger.info("Test complete!")
        logger.info("=" * 80)
        return

    # # 5. Generate responses without steering
    # if without_path.exists():
    #     logger.info(
    #         f"\nSkipping generation without steering - results already exist at {without_path}")
    # else:
    #     responses_without_steering = generate_batch(
    #         model=model,
    #         tokenizer=tokenizer,
    #         chats=chats,
    #         description="Generating without steering",
    #         use_steering=False,
    #     )

    #     # Save responses without steering
    #     save_responses(
    #         output_path=without_path,
    #         prompts=prompts,
    #         references=references,
    #         predictions=responses_without_steering,
    #     )

    # 6. Generate responses with KNN-based cluster steering
    if with_path.exists() or existing_with_paths:
        logger.info(
            "\nSkipping generation with KNN steering - results already exist at:"
        )
        if with_path.exists():
            logger.info(f"  - {with_path}")
        for path in existing_with_paths:
            logger.info(f"  - {path}")
    else:
        logger.info(
            f"\nGenerating with top-{k} KNN-based cluster steering (temp={temperature})...")
        responses_with_steering, selection_reasons = generate_with_cluster_selection(
            model=model,
            tokenizer=tokenizer,
            chats=chats,
            prompts=prompts,
            description=f"Generating with top-{k} KNN-based steering on layer(s): " + (
                (",".join(map(str, layers))) if layers else "all"),
            cluster_prompts=cluster_prompts,
            cluster_ids=cluster_ids,
            cluster_embeddings=cluster_embeddings,
            cluster_vectors=cluster_vectors,
            all_data_vector=all_data_vector,
            multiplier=multiplier,
            k=k,
            temperature=temperature,
            min_similarity=min_similarity,
            embedding_model=embedding_model,
            logger=logger,
            batch_size=batch_size,
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
        logger.info("\nCluster Selection Statistics:")
        cluster_counts = {}
        for reason in selection_reasons:
            cluster_counts[reason.split(' ')[0]] = cluster_counts.get(
                reason.split(' ')[0], 0) + 1

        for cluster_name, count in sorted(cluster_counts.items()):
            percentage = (count / len(selection_reasons)) * 100
            logger.info(f"  {cluster_name}: {count} ({percentage:.1f}%)")


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
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name for inference (default: meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--vectors-path", type=str, default=None,
                        help="Path to vectors directory (optional)")
    args = parser.parse_args()
    layers = args.layers if args.layers else None

    # Load model and tokenizer for standalone usage
    print(f"Loading model and tokenizer: {args.model}...")
    model, tokenizer = get_model_and_tokenizer(args.model)

    run_inference(
        clusters_file=args.clusters,
        vectors_path=args.vectors_path,
        dataset=args.dataset,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        k=args.k,
        temperature=args.temperature,
        multiplier=args.multiplier,
        min_similarity=args.min_similarity,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()
