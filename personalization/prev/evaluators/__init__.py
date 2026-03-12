"""Evaluation modules for ROUGE, style similarity, AI detection, authorship verification, and persona attribution metrics."""

from .rouge_score_evaluator import (
    compute_rouge_metrics,
    load_and_print_rouge_metrics,
    save_rouge_results_to_csv,
    score_with_rouge_l,
)
from .style_similarity_evaluator import (
    StyleSimilarityEvaluator,
    compute_style_similarity_metrics,
    print_style_similarity_metrics,
)
from .ai_detector_evaluator import (
    AIContentDetector,
    compute_ai_detection_metrics,
    print_ai_detection_metrics,
    compare_ai_detection,
)


__all__ = [
    # ROUGE evaluation
    'compute_rouge_metrics',
    'load_and_print_rouge_metrics',
    'save_rouge_results_to_csv',
    'score_with_rouge_l',
    # Style similarity evaluation
    'StyleSimilarityEvaluator',
    'compute_style_similarity_metrics',
    'print_style_similarity_metrics',
    # AI content detection
    'AIContentDetector',
    'compute_ai_detection_metrics',
    'print_ai_detection_metrics',
    'compare_ai_detection',
]
