"""ROUGE-L evaluation utilities for comparing predictions against references."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rouge import Rouge


def score_with_rouge_l(
    references: List[str],
    predictions: List[str],
    rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    """Score predictions against references using ROUGE-L."""
    progress_columns = (
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )

    if len(references) != len(predictions) or len(references) != len(rows):
        raise ValueError(
            "References, predictions, and rows must be the same length."
        )

    metrics: List[Dict[str, float]] = []
    results_rows: List[Dict[str, str]] = []
    rouge = Rouge(metrics=["rouge-l"])

    with Progress(*progress_columns) as progress:
        scoring_task = progress.add_task(
            "Scoring outputs", total=len(predictions))
        for idx, (reference, prediction, row) in enumerate(
            zip(references, predictions, rows), start=1
        ):
            # Skip empty predictions or references
            if not prediction or not prediction.strip() or not reference or not reference.strip():
                print(
                    f"\nWarning: Skipping row {idx} due to empty prediction or reference")
                progress.advance(scoring_task)
                continue

            scores = rouge.get_scores(prediction, reference, avg=False)
            rouge_l_scores = scores[0]["rouge-l"]

            metrics.append(
                {
                    "precision": rouge_l_scores["p"],
                    "recall": rouge_l_scores["r"],
                    "fmeasure": rouge_l_scores["f"],
                }
            )

            identifier = row.get("file") or f"row_{idx}"
            progress.advance(scoring_task)

            results_rows.append(
                {
                    "index": str(idx),
                    "file": identifier,
                    "prompt": row.get("prompt", ""),
                    "reference": reference,
                    "prediction": prediction,
                    "rouge_l_precision": f"{metrics[-1]['precision']:.6f}",
                    "rouge_l_recall": f"{metrics[-1]['recall']:.6f}",
                    "rouge_l_fmeasure": f"{metrics[-1]['fmeasure']:.6f}",
                }
            )

    average_precision = sum(m["precision"] for m in metrics) / len(metrics)
    average_recall = sum(m["recall"] for m in metrics) / len(metrics)
    average_f1 = sum(m["fmeasure"] for m in metrics) / len(metrics)

    summary = {
        "precision": average_precision,
        "recall": average_recall,
        "fmeasure": average_f1,
    }

    return results_rows, summary


def save_rouge_results_to_csv(
    results_rows: List[Dict[str, str]],
    output_path: Path,
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Save ROUGE results to CSV file."""
    if not results_rows:
        raise ValueError("results_rows must not be empty.")

    if fieldnames is None:
        # Preserve insertion order by using the first row keys
        fieldnames = list(results_rows[0].keys())

    average_precision = sum(
        float(row["rouge_l_precision"]) for row in results_rows
    ) / len(results_rows)
    average_recall = sum(
        float(row["rouge_l_recall"]) for row in results_rows
    ) / len(results_rows)
    average_f1 = sum(
        float(row["rouge_l_fmeasure"]) for row in results_rows
    ) / len(results_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)
        writer.writerow(
            {
                "index": "avg",
                "rouge_l_precision": f"{average_precision:.6f}",
                "rouge_l_recall": f"{average_recall:.6f}",
                "rouge_l_fmeasure": f"{average_f1:.6f}",
            }
        )

    print("\nOverall ROUGE-L")
    print(
        f"Precision: {average_precision:.4f}, "
        f"Recall: {average_recall:.4f}, "
        f"F1: {average_f1:.4f}"
    )

    print(f"\nSaved detailed results to {output_path}")


def load_and_print_rouge_metrics(csv_path: Path, description: str) -> None:
    """Load ROUGE-L metrics from CSV file and print them."""
    if not csv_path.exists():
        print(f"Warning: {csv_path} does not exist.")
        return

    with csv_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    if not rows:
        print(f"Warning: {csv_path} is empty.")
        return

    # The last row contains the averages with index="avg"
    avg_row = None
    for row in rows:
        if row.get("index") == "avg":
            avg_row = row
            break

    if avg_row:
        precision = float(avg_row["rouge_l_precision"])
        recall = float(avg_row["rouge_l_recall"])
        f1 = float(avg_row["rouge_l_fmeasure"])

        print(f"\n{description}")
        print(
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    else:
        print(f"Warning: No average row found in {csv_path}")


def compute_rouge_metrics(csv_path: Path) -> None:
    """Compute ROUGE-L metrics for a CSV file with predictions and references."""
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    # Read CSV file
    with csv_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    if not rows:
        print(f"Error: {csv_path} is empty.")
        return

    # Extract references and predictions
    references = [row["reference"] for row in rows]
    predictions = [row["prediction"] for row in rows]

    # Score with ROUGE-L
    scored_rows, metrics = score_with_rouge_l(
        references=references,
        predictions=predictions,
        rows=rows,
    )

    # Save results (this also prints the metrics)
    print(f"Saving ROUGE-L metrics to {csv_path}...")
    save_rouge_results_to_csv(
        results_rows=scored_rows,
        output_path=csv_path,
        fieldnames=list(scored_rows[0].keys()),
    )
