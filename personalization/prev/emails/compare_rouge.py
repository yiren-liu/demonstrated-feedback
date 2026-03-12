import csv
import sys
from pathlib import Path
from typing import List, Dict

from rouge import Rouge


def load_emails_from_csv(csv_path: Path) -> tuple[List[str], List[str], List[str]]:
    """
    Load email and llm_email columns from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (identifiers, reference_emails, llm_emails)
    """
    identifiers: List[str] = []
    reference_emails: List[str] = []
    llm_emails: List[str] = []

    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        # Verify required columns exist
        if reader.fieldnames:
            if "email" not in reader.fieldnames:
                raise ValueError("CSV file must have 'email' column")
            if "llm_email" not in reader.fieldnames:
                raise ValueError("CSV file must have 'llm_email' column")

        for idx, row in enumerate(reader, start=1):
            email = row.get("email", "").strip()
            llm_email = row.get("llm_email", "").strip()

            # Use file column as identifier if available, otherwise use row number
            identifier = row.get("file", f"row_{idx}")

            if not email or not llm_email:
                print(
                    f"Warning: Skipping row {idx} ({identifier}) - missing email or llm_email")
                continue

            identifiers.append(identifier)
            reference_emails.append(email)
            llm_emails.append(llm_email)

    if not reference_emails:
        raise ValueError(f"No valid rows found in {csv_path}")

    return identifiers, reference_emails, llm_emails


def calculate_rouge_l(
    identifiers: List[str],
    references: List[str],
    predictions: List[str]
) -> tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Calculate ROUGE-L scores for each pair of emails.

    Args:
        identifiers: List of row identifiers
        references: List of reference emails (email column)
        predictions: List of predicted emails (llm_email column)

    Returns:
        Tuple of (individual_scores, average_scores)
    """
    rouge = Rouge(metrics=["rouge-l"])
    individual_scores: List[Dict[str, float]] = []

    print("\nROUGE-L Scores (comparing llm_email to email):")
    print("=" * 80)
    print(f"{'ID':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)

    for identifier, reference, prediction in zip(identifiers, references, predictions):
        # Calculate ROUGE-L score
        scores = rouge.get_scores(prediction, reference, avg=False)
        rouge_l = scores[0]["rouge-l"]

        score_dict = {
            "precision": rouge_l["p"],
            "recall": rouge_l["r"],
            "fmeasure": rouge_l["f"],
        }
        individual_scores.append(score_dict)

        # Print individual score
        print(
            f"{identifier:<20} "
            f"{score_dict['precision']:>10.4f} "
            f"{score_dict['recall']:>10.4f} "
            f"{score_dict['fmeasure']:>10.4f}"
        )

    # Calculate averages
    avg_precision = sum(s["precision"]
                        for s in individual_scores) / len(individual_scores)
    avg_recall = sum(s["recall"]
                     for s in individual_scores) / len(individual_scores)
    avg_fmeasure = sum(s["fmeasure"]
                       for s in individual_scores) / len(individual_scores)

    average_scores = {
        "precision": avg_precision,
        "recall": avg_recall,
        "fmeasure": avg_fmeasure,
    }

    print("-" * 80)
    print(
        f"{'AVERAGE':<20} "
        f"{avg_precision:>10.4f} "
        f"{avg_recall:>10.4f} "
        f"{avg_fmeasure:>10.4f}"
    )
    print("=" * 80)

    return individual_scores, average_scores


def save_results_to_csv(
    output_path: Path,
    identifiers: List[str],
    references: List[str],
    predictions: List[str],
    scores: List[Dict[str, float]],
    average_scores: Dict[str, float]
) -> None:
    """
    Save ROUGE-L comparison results to a CSV file.

    Args:
        output_path: Path to save the results
        identifiers: List of row identifiers
        references: List of reference emails
        predictions: List of predicted emails
        scores: List of individual ROUGE-L scores
        average_scores: Average ROUGE-L scores
    """
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "identifier",
            "email",
            "llm_email",
            "rouge_l_precision",
            "rouge_l_recall",
            "rouge_l_fmeasure",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write individual results
        for identifier, reference, prediction, score in zip(
            identifiers, references, predictions, scores
        ):
            writer.writerow({
                "identifier": identifier,
                "email": reference,
                "llm_email": prediction,
                "rouge_l_precision": f"{score['precision']:.6f}",
                "rouge_l_recall": f"{score['recall']:.6f}",
                "rouge_l_fmeasure": f"{score['fmeasure']:.6f}",
            })

        # Write average row
        writer.writerow({
            "identifier": "AVERAGE",
            "email": "",
            "llm_email": "",
            "rouge_l_precision": f"{average_scores['precision']:.6f}",
            "rouge_l_recall": f"{average_scores['recall']:.6f}",
            "rouge_l_fmeasure": f"{average_scores['fmeasure']:.6f}",
        })

    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run ROUGE-L comparison."""
    if len(sys.argv) < 2:
        print(
            "Usage: python compare_rouge.py <path_to_text.csv> [output_path.csv]")
        print("\nExample:")
        print("  python compare_rouge.py text.csv")
        print("  python compare_rouge.py text.csv results_rouge.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.parent / \
            f"{input_path.stem}_rouge_results.csv"

    print(f"Loading emails from: {input_path}")

    try:
        identifiers, references, predictions = load_emails_from_csv(input_path)
        print(f"Loaded {len(references)} email pairs for comparison")

        scores, average_scores = calculate_rouge_l(
            identifiers, references, predictions)

        save_results_to_csv(
            output_path, identifiers, references, predictions, scores, average_scores
        )

        print(f"\nSummary:")
        print(f"  Total comparisons: {len(scores)}")
        print(f"  Average ROUGE-L F1: {average_scores['fmeasure']:.4f}")
        print(
            f"  Average ROUGE-L Precision: {average_scores['precision']:.4f}")
        print(f"  Average ROUGE-L Recall: {average_scores['recall']:.4f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
