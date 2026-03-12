import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm

from multi_vectors.utils import generate_text, generate_batch


def load_clusters_metadata(clusters_path: Path) -> Dict:
    """Load clusters metadata from JSONL file."""
    with clusters_path.open('r') as f:
        return json.load(f)


def load_pca_reducer(clusters_path: Path) -> Optional[PCA]:
    """Load PCA reducer from saved pickle file if it exists."""
    pca_path = clusters_path.with_suffix('.pca.pkl')

    if not pca_path.exists():
        print(f"Warning: PCA file not found at {pca_path}")
        return None

    with pca_path.open('rb') as f:
        reducer = pickle.load(f)

    return reducer


def load_cluster_prompts_with_embeddings(
    clusters_data: Dict,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
) -> Tuple[List[str], List[int], np.ndarray]:
    """Load all prompts from all clusters with their embeddings and cluster IDs."""
    embedder = SentenceTransformer(embedding_model)

    all_prompts = []
    all_cluster_ids = []

    # Collect prompts from all clusters
    for cluster in clusters_data.get('clusters', []):
        cluster_id = cluster['cluster_id']
        examples = cluster.get('examples', [])
        for example in examples:
            all_prompts.append(example['prompt'])
            all_cluster_ids.append(cluster_id)

    # Collect noise examples
    noise_examples = clusters_data.get('noise_examples', [])
    for example in noise_examples:
        all_prompts.append(example['prompt'])
        all_cluster_ids.append(-1)  # -1 indicates noise

    # Compute embeddings for all prompts
    print(f"Computing embeddings for {len(all_prompts)} cluster prompts...")
    embeddings = embedder.encode(
        all_prompts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return all_prompts, all_cluster_ids, embeddings


def load_steering_vectors(
    clusters_data: Dict,
    clusters_path: Path,
    layers: List[int] | None,
    vectors_path: str,
) -> Tuple[Dict[int, object], object]:
    """Load cluster-specific steering vectors and the all-data fallback vector."""
    vectors_dir = Path(vectors_path)
    dataset_name = clusters_path.stem
    layer_str = (
        "layer_all"
        if layers is None
        else f"layer_{'_'.join(map(str, layers))}"
    )

    print(f"\nLoading steering vectors from {vectors_dir}...")
    cluster_vectors = {}

    # Load cluster-specific vectors
    for cluster in clusters_data.get('clusters', []):
        cluster_id = cluster['cluster_id']
        cluster_vector_path = vectors_dir / \
            f"{dataset_name}_cluster_{cluster_id}_sv_{layer_str}.pt"

        if not cluster_vector_path.exists():
            raise FileNotFoundError(
                f"Cluster {cluster_id} vector not found at {cluster_vector_path}")

        cluster_vectors[cluster_id] = torch.load(
            str(cluster_vector_path), weights_only=False)
        print(f"  Loaded cluster {cluster_id} vector")

    # Load all-data fallback vector
    all_data_vector_path = vectors_dir / \
        f"{dataset_name}_all_data_sv_{layer_str}.pt"

    if not all_data_vector_path.exists():
        raise FileNotFoundError(
            f"All-data vector not found at {all_data_vector_path}")

    all_data_vector = torch.load(str(all_data_vector_path), weights_only=False)
    print(f"  Loaded all-data vector")

    return cluster_vectors, all_data_vector


def generate_with_vectors(
    model,
    tokenizer,
    chats: List[str],
    selected_vectors: List[object],
    description: str,
    multiplier: float = 1.0,
    batch_size: int = 2,
) -> List[str]:
    """Generate responses with per-sample steering vectors, processing one sample at a time."""
    responses = []

    if batch_size <= 1:
        for chat, steering_vector in tqdm(zip(chats, selected_vectors), total=len(chats), desc=description):
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                chat=chat,
                use_steering=True,
                steering_vector=steering_vector,
                multiplier=multiplier,
            )
            responses.append(response)
    else:
        responses = generate_batch(
            model=model,
            tokenizer=tokenizer,
            chats=chats,
            description=description,
            use_steering=True,
            steering_vector=selected_vectors,
            multiplier=multiplier,
            batch_size=batch_size,
        )

    return responses


def get_output_paths(
    script_path: Path,
    dataset: str,
    layers: List[int] | None,
    output_dir: str = "multi_vectors/outputs",
    method_suffix: Optional[str] = None,
    split: str = "val",
) -> Tuple[Path, Path]:
    """Generate output file paths for inference results."""
    # Get base directory (go up from pipeline -> multi_vectors -> project root)
    base_dir = script_path.parent.parent.parent
    # Create split-specific subdirectory (e.g., outputs/results/val or outputs/results/test)
    results_dir = base_dir / output_dir / "results" / split
    results_dir.mkdir(parents=True, exist_ok=True)

    without_path = results_dir / f"{dataset}_without_steering.csv"

    # Add timestamp suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct with_steering filename
    layer_str = "layer_all" if layers is None else 'layer_' + \
        '_'.join(map(str, layers))
    if method_suffix:
        with_path = results_dir / \
            f"{dataset}_with_{method_suffix}_{layer_str}_{timestamp}.csv"
    else:
        with_path = results_dir / \
            f"{dataset}_with_steering_{layer_str}_{timestamp}.csv"

    return without_path, with_path
