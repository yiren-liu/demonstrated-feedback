"""AI content detection evaluator for assessing text authenticity."""

import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import pipeline
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def clean_text(text: str) -> str:
    cleaned = str(text).replace('\n', ' ')
    return cleaned.strip('"')


class AIContentDetector:
    def __init__(self, model_name: str = "fakespot-ai/roberta-base-ai-text-detection-v1"):
        """Initialize the AI content detector using transformers pipeline."""
        print(f"Loading AI content detector: {model_name}...")
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            max_length=512
        )
        print(f"Successfully loaded {model_name}")

    def predict_single(self, text: str) -> float:
        """Predict AI score for a single text."""
        result = self.classifier(clean_text(text))

        # result format: [{'label': str, 'score': float}]
        # Check if the label indicates AI-generated content
        if result[0]['label'] == 'AI':
            return result[0]['score']
        else:
            # If label is 'Human', return 1 - score for AI probability
            return 1.0 - result[0]['score']

    def predict_batch(self, texts: List[str], show_progress: bool = True) -> List[float]:
        """Predict AI scores for a batch of texts."""
        # Clean all texts
        cleaned_texts = [clean_text(t) for t in texts]

        # Use batch prediction for efficiency
        results = self.classifier(cleaned_texts)

        # Extract AI scores from results
        scores = []
        for result in results:
            if result['label'] == 'AI':
                scores.append(result['score'])
            else:
                # If label is 'Human', return 1 - score for AI probability
                scores.append(1.0 - result['score'])

        return scores


def compute_ai_detection_metrics(csv_path: Path, column: str = "prediction") -> Dict[str, float]:
    """Compute AI detection scores for a CSV file."""
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return {}

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Separate out average row if present
    avg_row = None
    if 'avg' in df['index'].values:
        avg_row = df[df['index'] == 'avg'].copy()
        df = df[df['index'] != 'avg'].reset_index(drop=True)

    if column not in df.columns:
        print(f"Error: Column '{column}' not found in {csv_path}")
        return {}

    # Initialize detector
    detector = AIContentDetector()

    # Get AI scores
    print(f"\nAnalyzing {column} column from {csv_path}...")
    texts = [str(text).replace('\n', ' ') for text in df[column].tolist()]
    ai_scores = detector.predict_batch(texts, show_progress=True)

    # Add to dataframe
    df[f'{column}_ai_score'] = [f"{score:.6f}" for score in ai_scores]

    # Reappend average row with AI score average
    if avg_row is not None:
        avg_row[f'{column}_ai_score'] = f"{np.mean(ai_scores):.6f}"
        df = pd.concat([df, avg_row], ignore_index=True)

    # Save updated CSV
    df.to_csv(csv_path, index=False)

    # Compute statistics
    metrics = {
        'mean_ai_score': float(np.mean(ai_scores)),
        'std_ai_score': float(np.std(ai_scores)),
        'median_ai_score': float(np.median(ai_scores)),
        'min_ai_score': float(np.min(ai_scores)),
        'max_ai_score': float(np.max(ai_scores)),
        'q25_ai_score': float(np.percentile(ai_scores, 25)),
        'q75_ai_score': float(np.percentile(ai_scores, 75)),
        'total_samples': len(ai_scores)
    }

    return metrics


def print_ai_detection_metrics(metrics: Dict[str, float], description: str) -> None:
    """Print AI detection metrics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"AI CONTENT DETECTION RESULTS: {description}")
    print(f"{'='*70}")
    print(f"Total samples:          {metrics['total_samples']}")
    print(f"\nAI Score Statistics (higher = more AI-like):")
    print(
        f"  Mean:                 {metrics['mean_ai_score']:.4f} (+/- {metrics['std_ai_score']:.4f})")
    print(f"  Median:               {metrics['median_ai_score']:.4f}")
    print(
        f"  Range:                [{metrics['min_ai_score']:.4f}, {metrics['max_ai_score']:.4f}]")
    print(
        f"  IQR (Q25-Q75):        [{metrics['q25_ai_score']:.4f}, {metrics['q75_ai_score']:.4f}]")
    print(f"{'='*70}")


def compare_ai_detection(
    with_steering_metrics: Dict[str, float],
    without_steering_metrics: Dict[str, float]
) -> None:
    """Compare AI detection scores between steered and unsteered outputs."""
    print(f"\n{'='*70}")
    print("AI SCORE COMPARISON")
    print(f"{'='*70}")

    mean_diff = with_steering_metrics['mean_ai_score'] - \
        without_steering_metrics['mean_ai_score']
    median_diff = with_steering_metrics['median_ai_score'] - \
        without_steering_metrics['median_ai_score']

    print(f"\nMean AI Score:")
    print(
        f"  With steering:      {with_steering_metrics['mean_ai_score']:.4f}")
    print(
        f"  Without steering:   {without_steering_metrics['mean_ai_score']:.4f}")
    print(f"  Difference:         {mean_diff:+.4f}")

    print(f"\nMedian AI Score:")
    print(
        f"  With steering:      {with_steering_metrics['median_ai_score']:.4f}")
    print(
        f"  Without steering:   {without_steering_metrics['median_ai_score']:.4f}")
    print(f"  Difference:         {median_diff:+.4f}")

    # Calculate percentage change
    if without_steering_metrics['mean_ai_score'] != 0:
        percent_change = (
            mean_diff / without_steering_metrics['mean_ai_score']) * 100
        print(f"\nRelative change:      {percent_change:+.2f}%")

    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    if mean_diff > 0.05:
        print("Steering vectors SIGNIFICANTLY INCREASE AI-likeness of outputs.")
    elif mean_diff > 0:
        print("Steering vectors slightly increase AI-likeness of outputs.")
    elif mean_diff < -0.05:
        print("Steering vectors SIGNIFICANTLY DECREASE AI-likeness (more human-like).")
    elif mean_diff < 0:
        print("Steering vectors slightly decrease AI-likeness (more human-like).")
    else:
        print("No significant difference in AI-likeness between steered and unsteered outputs.")
    print(f"{'='*70}\n")
