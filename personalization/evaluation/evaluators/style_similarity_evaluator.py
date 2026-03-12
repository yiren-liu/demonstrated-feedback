"""Style similarity evaluation for measuring similarity between reference and prediction."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from evaluation.evaluators.style_embedder import StyleEmbedder


def clean_text(text: str) -> str:
    cleaned = str(text).replace('\n', '')
    return cleaned.strip('"')


class StyleSimilarityEvaluator:
    def __init__(self, style_model: str = "AnnaWegmann/Style-Embedding"):
        self.embedder = StyleEmbedder(model_name=style_model)

    def compute_style_similarities(self, df: pd.DataFrame) -> np.ndarray:
        """Compute style similarity between reference and prediction for each row."""
        print("Extracting style embeddings for references...")
        ref_texts = [clean_text(text) for text in df['reference'].tolist()]
        # ref_texts = df['reference'].tolist()
        ref_embeddings = self.embedder.encode_batch(
            ref_texts, show_progress=True)

        print("Extracting style embeddings for predictions...")
        pred_texts = [clean_text(text) for text in df['prediction'].tolist()]
        # pred_texts = df['prediction'].tolist()
        pred_embeddings = self.embedder.encode_batch(
            pred_texts, show_progress=True)

        print("Computing pairwise similarities...")
        # Compute element-wise cosine similarity
        similarities = np.array([
            float(np.dot(ref_embeddings[i], pred_embeddings[i]) /
                  (np.linalg.norm(ref_embeddings[i]) * np.linalg.norm(pred_embeddings[i])))
            for i in range(len(df))
        ])

        return similarities

    def compute_statistics(self, similarities: np.ndarray) -> Dict:
        """Compute statistical summary of similarities."""
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'median': float(np.median(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'q25': float(np.percentile(similarities, 25)),
            'q75': float(np.percentile(similarities, 75)),
            'count': len(similarities)
        }

    def save_similarities_to_csv(
        self,
        filepath: Path,
        similarities: np.ndarray,
        column_name: str = 'style_similarity'
    ) -> None:
        """Save similarity scores back to the CSV file."""
        df = pd.read_csv(filepath)

        # Filter out the average row if it exists
        avg_row = None
        if 'avg' in df['index'].values:
            avg_row = df[df['index'] == 'avg'].copy()
            df = df[df['index'] != 'avg'].reset_index(drop=True)

        # Add similarity scores
        df[column_name] = [f"{sim:.6f}" for sim in similarities]

        # Reappend average row with similarity average
        if avg_row is not None:
            avg_row[column_name] = f"{np.mean(similarities):.6f}"
            df = pd.concat([df, avg_row], ignore_index=True)

        # Save back to file
        df.to_csv(filepath, index=False)
        print(f"Saved {column_name} scores to {filepath}")

    def evaluate(self, csv_path: Path, column_name: str = 'style_similarity') -> Dict:
        """Main evaluation method for computing style similarities."""
        # Load data
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        # Filter out average row if present
        df = df[df['index'] != 'avg'].reset_index(drop=True)
        print(f"Loaded {len(df)} samples")

        # Compute similarities
        print("\nComputing style similarities...")
        similarities = self.compute_style_similarities(df)

        # Save similarities to CSV file
        print("\nSaving similarity scores...")
        self.save_similarities_to_csv(csv_path, similarities, column_name)

        # Compute statistics
        stats = self.compute_statistics(similarities)

        return stats


def compute_style_similarity_metrics(
    csv_path: Path,
    reference_column: str = "reference",
    prediction_column: str = "prediction",
    style_model: str = "AnnaWegmann/Style-Embedding",
    column_name: str = "style_similarity"
) -> Dict[str, float]:
    """Compute style similarity metrics for a CSV file."""
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return {}

    # Initialize evaluator
    evaluator = StyleSimilarityEvaluator(style_model=style_model)

    # Evaluate
    results = evaluator.evaluate(csv_path, column_name=column_name)

    return results


def print_style_similarity_metrics(metrics: Dict[str, float], description: str) -> None:
    """Print style similarity metrics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"STYLE SIMILARITY RESULTS: {description}")
    print(f"{'='*70}")
    print(f"Total samples:         {metrics['count']}")
    print(f"\nStyle Similarity Statistics (higher = more similar):")
    print(
        f"  Mean:                {metrics['mean']:.4f} (+/- {metrics['std']:.4f})")
    print(f"  Median:              {metrics['median']:.4f}")
    print(
        f"  Range:               [{metrics['min']:.4f}, {metrics['max']:.4f}]")
    print(
        f"  IQR (Q25-Q75):       [{metrics['q25']:.4f}, {metrics['q75']:.4f}]")
    print(f"{'='*70}")
