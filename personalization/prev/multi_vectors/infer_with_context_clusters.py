import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import hdbscan
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from utils import (
    get_model_and_tokenizer,
    generate_batch,
)
from utils.data_utils import make_test_dataset, save_responses
from .infer_utils import (
    load_clusters_metadata,
    load_pca_reducer,
    load_steering_vectors,
    generate_with_vectors,
    get_output_paths,
)


def select_steering_vector_for_prompt(
    prompt: str,
    clusters_data: Dict,
    pca_reducer: Optional[PCA],
    cluster_vectors: Dict[int, object],
    all_data_vector: object,
    clusterer_model,
    threshold: float = 0.1,
) -> Tuple[object, str]:
    """Select appropriate steering vector based on HDBSCAN cluster assignment.
    Uses approximate_predict to determine cluster membership."""
    # Build embedding for the test prompt
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    prompt_embedding = embedder.encode(
        [prompt],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]

    # Apply PCA dimensionality reduction if it was used during clustering
    if pca_reducer is not None:
        reduced = pca_reducer.transform([prompt_embedding])
        # Normalize after PCA
        norm = np.linalg.norm(reduced, axis=1, keepdims=True)
        if norm[0, 0] > 0:
            reduced = reduced / norm
        prompt_embedding_reduced = reduced[0]
    else:
        prompt_embedding_reduced = prompt_embedding

    # Use HDBSCAN approximate_predict for cluster assignment
    labels, strengths = hdbscan.approximate_predict(
        clusterer_model, [prompt_embedding_reduced])
    predicted_cluster = labels[0]
    strength = strengths[0]

    # Use predicted cluster if it's not noise and has sufficient strength
    if predicted_cluster != -1 and strength >= threshold:
        if predicted_cluster in cluster_vectors:
            return (
                cluster_vectors[predicted_cluster],
                f"cluster_{predicted_cluster} (strength: {strength:.3f})"
            )
        else:
            # Predicted cluster doesn't have a vector, fall back to all_data
            return all_data_vector, f"all_data (cluster_{predicted_cluster} vector not available)"
    else:
        # Point is noise or weak prediction
        return all_data_vector, f"all_data (noise or weak strength: {strength:.3f})"


def generate_with_cluster_selection(
    model,
    tokenizer,
    chats: List[str],
    prompts: List[str],
    description: str,
    clusters_data: Dict,
    pca_reducer: Optional[PCA],
    cluster_vectors: Dict[int, object],
    all_data_vector: object,
    clusterer_model,
    threshold: float = 0.1,
    multiplier: float = 1.0,
) -> Tuple[List[str], List[str]]:
    """Generate responses with cluster-based vector selection."""
    # Pre-compute all steering vector selections
    print(f"Selecting steering vectors for {len(prompts)} prompts...")
    selected_vectors = []
    selection_reasons = []

    for prompt in prompts:
        selected_vector, reason = select_steering_vector_for_prompt(
            prompt=prompt,
            clusters_data=clusters_data,
            pca_reducer=pca_reducer,
            cluster_vectors=cluster_vectors,
            all_data_vector=all_data_vector,
            clusterer_model=clusterer_model,
            threshold=threshold,
        )
        selected_vectors.append(selected_vector)
        selection_reasons.append(reason)

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
                        help="Layers used in the steering vector")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Minimum strength threshold for approximate_predict cluster assignment (default: 0.1)")
    parser.add_argument("--multiplier", type=float, default=3.0,
                        help="Steering vector multiplier (default: 3.0)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    dataset = args.dataset
    layers = args.layers if args.layers else None
    clusters = args.clusters
    dataset_path = base_dir / "dataset" / dataset / "test.json"
    clusters_path = base_dir / "multi_vectors" / "context_clusters" / clusters
    threshold = args.threshold

    if not clusters_path.exists():
        raise FileNotFoundError(f"Clusters file not found: {clusters_path}")

    # 1. Load clusters metadata
    print(f"Loading clusters metadata from {clusters_path}")
    clusters_data = load_clusters_metadata(clusters_path)
    print(f"Found {len(clusters_data.get('clusters', []))} clusters")

    # 2. Load PCA reducer if needed
    print("Loading PCA reducer for dimensionality reduction...")
    pca_reducer = load_pca_reducer(clusters_path)
    if pca_reducer:
        print(f"PCA reducer loaded with {pca_reducer.n_components} components")
    else:
        print("No PCA dimensionality reduction needed")

    # 2b. Load HDBSCAN clusterer model for hard cluster assignment
    print("Loading HDBSCAN clusterer model...")
    clusterer_path = clusters_path.with_suffix('.hdbscan.pkl')
    if not clusterer_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN clusterer model not found at {clusterer_path}. "
            f"Please re-run clustering to generate the model file."
        )

    import pickle
    with clusterer_path.open('rb') as f:
        clusterer_model = pickle.load(f)
    print(f"HDBSCAN clusterer model loaded from {clusterer_path}")
    print("  Will use approximate_predict for hard cluster assignment")

    # 3. Load all steering vectors (cluster-specific and all-data)
    cluster_vectors, all_data_vector = load_steering_vectors(
        clusters_data=clusters_data,
        clusters_path=clusters_path,
        layers=layers,
        vectors_subdir="context_vectors",
    )

    # 4. Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model, tokenizer = get_model_and_tokenizer(model_name)

    # Ensure tokenizer pad token and left padding for decoder-only models
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    # 5. Load test dataset
    chats, references, prompts = make_test_dataset(
        tokenizer, dataset_path)

    # 6. Define output paths
    threshold_str = f"{threshold:.2f}".replace('.', '_')
    without_path, with_path = get_output_paths(
        script_path=Path(__file__),
        dataset=dataset,
        layers=layers,
        method_suffix=f"context_clusters_threshold{threshold_str}_multiplier{args.multiplier}",
    )

    multiplier = args.multiplier

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

    # 8. Generate responses with cluster-based steering
    if with_path.exists():
        print(
            f"\nSkipping generation with cluster steering - results already exist at {with_path}")
    else:
        print(
            f"\nGenerating with cluster-based steering (threshold={threshold})...")
        responses_with_steering, selection_reasons = generate_with_cluster_selection(
            model=model,
            tokenizer=tokenizer,
            chats=chats,
            prompts=prompts,
            description="Generating with cluster-based steering on layer(s) " + ",".join(
                map(str, layers)),
            clusters_data=clusters_data,
            pca_reducer=pca_reducer,
            cluster_vectors=cluster_vectors,
            all_data_vector=all_data_vector,
            clusterer_model=clusterer_model,
            threshold=threshold,
            multiplier=multiplier,
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
