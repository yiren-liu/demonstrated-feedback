"""Clustering and embedding utilities"""

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import hdbscan

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def build_embeddings(prompts: List[str]) -> np.ndarray:
    """Build embeddings for a list of prompts."""
    # embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # embedder = SentenceTransformer("StyleDistance/styledistance")
    embedder = SentenceTransformer("AnnaWegmann/Style-Embedding")
    return embedder.encode(
        prompts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )


def cluster_embeddings(
    embeddings: np.ndarray,
    apply_dimensionality_reduction: bool = False,
    n_components: int = 128,
    algorithm: str = "hdbscan",
) -> Dict[str, object]:
    """Cluster embeddings using HDBSCAN or DBSCAN."""
    def reduce_dimensionality(
        vectors: np.ndarray, n_components: int = 128
    ) -> Dict[str, object]:
        if vectors.shape[1] <= n_components:
            return {
                "reduced": vectors,
                "metadata": None,
                "reducer": None,
            }
        reducer = PCA(n_components=n_components, random_state=SEED)
        reduced = reducer.fit_transform(vectors)
        # Normalize after PCA to preserve angular relationships
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero for degenerate rows
        normalized = reduced / norms
        return {
            "reduced": normalized,
            "metadata": {
                "method": "pca",
                "n_components": n_components,
                "random_state": SEED,
                "explained_variance_ratio_sum": float(
                    np.sum(reducer.explained_variance_ratio_)
                ),
            },
            "reducer": reducer,
        }

    # 1. Dimensionality reduction (optional)
    if apply_dimensionality_reduction:
        reduction = reduce_dimensionality(embeddings, n_components)
    else:
        reduction = {
            "reduced": embeddings,
            "metadata": None,
            "reducer": None,
        }

    # 2. Clustering with chosen algorithm
    if algorithm.lower() == "hdbscan":
        params = {
            # "min_cluster_size": 2,
            # "min_samples": 1,
            "min_cluster_size": 15,
            "min_samples": 1,
            "metric": "euclidean",
            "cluster_selection_method": "leaf",
            "cluster_selection_epsilon": 0.0,
            # "cluster_selection_epsilon_max": 0.5,
            # "prediction_data": True,
        }

        clusterer = hdbscan.HDBSCAN(**params)
        clusterer.fit(reduction["reduced"])
        labels = clusterer.labels_

        noise_ratio = float(np.mean(labels == -1))
        params["noise_ratio"] = noise_ratio
    elif algorithm.lower() == "dbscan":
        params = {
            "eps": 0.58,
            "min_samples": 15,
            "metric": "euclidean",
            "n_jobs": -1,
        }
        clusterer = DBSCAN(**params)
        labels = clusterer.fit_predict(reduction["reduced"])
        noise_ratio = float(np.mean(labels == -1))
        params["noise_ratio"] = noise_ratio
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Choose 'hdbscan' or 'dbscan'."
        )

    if reduction["metadata"]:
        params["dimensionality_reduction"] = reduction["metadata"]

    # 3. Organize clustering results
    labels = labels.astype(int)
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    noise_indices: List[int] = []
    for idx, label in enumerate(labels):
        if label == -1:
            noise_indices.append(idx)
            continue
        label_to_indices[label].append(idx)

    # Fallback: treat all points as one cluster if the algorithm produced no clusters.
    if not label_to_indices:
        label_to_indices[0] = list(range(len(embeddings)))

    # Compute centroids in the same space as clustering (reduced space if PCA was used)
    clustering_space = reduction["reduced"]
    centroids = {
        cluster_id: clustering_space[indices].mean(axis=0)
        for cluster_id, indices in label_to_indices.items()
    }

    result = {
        "labels": labels,
        "label_to_indices": label_to_indices,
        "centroids": centroids,
        "noise_indices": noise_indices,
        "algorithm": algorithm,
        "params": params,
        "pca_reducer": reduction["reducer"],
        "clusterer": clusterer if algorithm.lower() == "hdbscan" else None,
    }

    return result


def save_cluster_metadata(
    cluster_info: Dict[str, object],
    rows: Dict[str, List[str]],
    metadata_path: Path,
    add_timestamp: bool = True,
) -> None:
    """Save cluster metadata to JSON file.

    Args:
        cluster_info: Dictionary containing clustering results.
        rows: Dictionary containing prompts, personalized, and style_agnostic lists.
        metadata_path: Path to save the metadata JSON file.
        add_timestamp: If True, adds a timestamp suffix to all saved files.
    """
    # Generate timestamp suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""

    def add_suffix_to_path(path: Path, suffix: str, new_ext: str = None) -> Path:
        """Add a suffix to the filename before the extension."""
        ext = new_ext if new_ext else path.suffix
        stem = path.stem
        # Remove existing extension from stem if we're changing extension
        if new_ext and stem.endswith(path.suffix.replace(".", "")):
            stem = stem[:-len(path.suffix) + 1]
        new_name = f"{stem}_{suffix}{ext}" if suffix else f"{stem}{ext}"
        return path.parent / new_name

    prompts = rows["prompts"]
    personalized = rows["personalized"]
    style_agnostic = rows["style_agnostic"]
    cluster_metadata: List[Dict[str, object]] = []

    for cluster_id, indices in sorted(cluster_info["label_to_indices"].items()):
        examples = [
            {
                "index": idx,
                "prompt": prompts[idx],
                "personalized": personalized[idx],
                "style_agnostic": style_agnostic[idx],
            }
            for idx in indices
        ]
        cluster_metadata.append(
            {
                "cluster_id": int(cluster_id),
                "prompt_count": len(indices),
                "centroid": cluster_info["centroids"][cluster_id].tolist(),
                "examples": examples,
            }
        )

        # Print 3 examples for this cluster
        print(f"\nCluster {cluster_id} ({len(indices)} examples):")
        for i, example in enumerate(examples[:3]):
            print(f"  Example {i+1}:")
            print(f"    Prompt: {example['prompt']}")

    noise_examples = [
        {
            "index": idx,
            "prompt": prompts[idx],
            "personalized": personalized[idx],
            "style_agnostic": style_agnostic[idx],
        }
        for idx in cluster_info["noise_indices"]
    ]

    metadata_payload = {
        "embedding_model": "AnnaWegmann/Style-Embedding",
        "clustering_algorithm": cluster_info["algorithm"],
        "clustering_params": cluster_info["params"],
        "clusters": cluster_metadata,
        "noise_examples": noise_examples,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply timestamp suffix to metadata path
    final_metadata_path = add_suffix_to_path(
        metadata_path, timestamp) if timestamp else metadata_path
    final_metadata_path.write_text(json.dumps(metadata_payload, indent=2))
    print(f"\nSaved cluster metadata to {final_metadata_path}")

    # Save PCA reducer if it exists
    if cluster_info.get("pca_reducer") is not None:
        import pickle

        pca_path = add_suffix_to_path(
            metadata_path, timestamp, ".pca.pkl") if timestamp else metadata_path.with_suffix(".pca.pkl")
        with pca_path.open("wb") as f:
            pickle.dump(cluster_info["pca_reducer"], f)
        print(f"Saved PCA reducer to {pca_path}")

    # Save HDBSCAN clusterer model if it exists (for soft clustering)
    if cluster_info.get("clusterer") is not None:
        import pickle

        clusterer_path = add_suffix_to_path(
            metadata_path, timestamp, ".hdbscan.pkl") if timestamp else metadata_path.with_suffix(".hdbscan.pkl")
        with clusterer_path.open("wb") as f:
            pickle.dump(cluster_info["clusterer"], f)
        print(f"Saved HDBSCAN clusterer model to {clusterer_path}")
