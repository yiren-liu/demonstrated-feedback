"""
Evaluation Module

This module computes ROUGE-L metrics, style similarity, and AI content detection
for model outputs.
"""

import argparse
import json
import io
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from evaluators.rouge_score_evaluator import compute_rouge_metrics
from evaluators.style_similarity_evaluator import (
    compute_style_similarity_metrics,
    print_style_similarity_metrics,
)
from evaluators.ai_detector_evaluator import (
    compute_ai_detection_metrics,
    print_ai_detection_metrics,
)


def _ensure_results_dir() -> Path:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    """Main entry point for evaluation script"""
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for model outputs"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file with results",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        choices=["rouge", "style", "ai", "all"],
        default="all",
        help="Which metrics to compute: rouge, style, ai, or all (default: all)",
    )
    # Optional: allow overriding output dir, but defaults to results/
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation reports (default: results)",
    )

    args = parser.parse_args()

    results = {}
    csv_path = Path(args.csv_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    report_txt_path = output_dir / \
        f"evaluation_report_{csv_path.stem}_{ts}.txt"
    report_json_path = output_dir / \
        f"evaluation_metrics_{csv_path.stem}_{ts}.json"

    report_sections = []
    report_sections.append(f"Evaluation Report\n")
    report_sections.append(f"CSV: {csv_path}\n")
    report_sections.append(f"Generated: {datetime.now().isoformat()}\n")
    report_sections.append("=" * 60 + "\n")

    # 1. Compute ROUGE-L metrics
    if args.metrics in ["rouge", "all"]:
        print("=" * 60)
        print("Computing ROUGE-L metrics...")
        print("=" * 60)

        rouge_buf = io.StringIO()
        with redirect_stdout(rouge_buf):
            compute_rouge_metrics(csv_path)

        rouge_text = rouge_buf.getvalue()
        # Keep console output visible too (since we captured it)
        print(rouge_text, end="")

        report_sections.append("ROUGE-L metrics\n")
        report_sections.append("-" * 60 + "\n")
        report_sections.append(rouge_text.strip() + "\n\n")
        results["rouge_stdout"] = rouge_text

    # 2. Compute style similarity
    if args.metrics in ["style", "all"]:
        print("\n" + "=" * 60)
        print("Computing Style Similarity...")
        print("=" * 60)

        style_metrics = compute_style_similarity_metrics(
            csv_path,
            reference_column="reference",
            prediction_column="prediction",
            column_name="style_similarity",
        )

        style_buf = io.StringIO()
        with redirect_stdout(style_buf):
            print_style_similarity_metrics(style_metrics, "Results")
        style_text = style_buf.getvalue()
        print(style_text, end="")

        report_sections.append("Style Similarity\n")
        report_sections.append("-" * 60 + "\n")
        report_sections.append(style_text.strip() + "\n\n")
        results["style_similarity"] = style_metrics

        # 3. Compute style distance
        print("\n" + "=" * 60)
        print("Computing Style Distance...")
        print("=" * 60)

        style_dist_metrics = compute_style_similarity_metrics(
            csv_path,
            reference_column="reference",
            prediction_column="prediction",
            style_model="StyleDistance/styledistance",
            column_name="style_distance",
        )

        style_dist_buf = io.StringIO()
        with redirect_stdout(style_dist_buf):
            print_style_similarity_metrics(style_dist_metrics, "Results")
        style_dist_text = style_dist_buf.getvalue()
        print(style_dist_text, end="")

        report_sections.append("Style Distance\n")
        report_sections.append("-" * 60 + "\n")
        report_sections.append(style_dist_text.strip() + "\n\n")
        results["style_distance"] = style_dist_metrics

    # 4. Compute AI detection
    if args.metrics in ["ai", "all"]:
        print("\n" + "=" * 60)
        print("Computing AI Content Detection...")
        print("=" * 60)

        ai_metrics = compute_ai_detection_metrics(
            csv_path, column="prediction")

        ai_buf = io.StringIO()
        with redirect_stdout(ai_buf):
            print_ai_detection_metrics(ai_metrics, "Results")
        ai_text = ai_buf.getvalue()
        print(ai_text, end="")

        report_sections.append("AI Content Detection\n")
        report_sections.append("-" * 60 + "\n")
        report_sections.append(ai_text.strip() + "\n\n")
        results["ai_detection"] = ai_metrics

    # Save report(s)
    report_txt_path.write_text("".join(report_sections), encoding="utf-8")
    report_json_path.write_text(json.dumps(
        results, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Saved report: {report_txt_path}")
    print(f"Saved metrics JSON: {report_json_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
