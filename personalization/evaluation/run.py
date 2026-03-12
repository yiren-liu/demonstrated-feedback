import argparse
import json
import io
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
import pandas as pd

from evaluation.evaluators.rouge_score_evaluator import compute_rouge_metrics
from evaluation.evaluators.style_similarity_evaluator import (
    compute_style_similarity_metrics,
    print_style_similarity_metrics,
)
from evaluation.evaluators.ai_detector_evaluator_v2 import (
    compute_ai_detection_metrics,
    print_ai_detection_metrics,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _compute_grouped_statistics(
    df: pd.DataFrame,
    metric_columns: list[str],
    group_column: str | None,
) -> dict:
    results = {
        "overall": {},
        "by_group": {}
    }

    # Overall statistics
    for col in metric_columns:
        if col in df.columns:
            results["overall"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "count": int(df[col].count())
            }

    # Per-group statistics
    if group_column and group_column in df.columns:
        for group_value, group in df.groupby(group_column):
            results["by_group"][group_value] = {}
            for col in metric_columns:
                if col in group.columns:
                    results["by_group"][group_value][col] = {
                        "mean": float(group[col].mean()),
                        "std": float(group[col].std()),
                        "min": float(group[col].min()),
                        "max": float(group[col].max()),
                        "count": int(group[col].count())
                    }

    return results


def _format_grouped_stats(stats: dict, metric_name: str, group_label: str) -> str:
    """Format grouped statistics for text output."""
    lines = []
    lines.append(f"\n{metric_name} - Grouped Statistics")
    lines.append("=" * 60)

    # Overall stats
    lines.append("\nOverall Statistics:")
    lines.append("-" * 60)
    for metric_col, values in stats["overall"].items():
        lines.append(f"\n{metric_col}:")
        lines.append(f"  Mean: {values['mean']:.4f}")
        lines.append(f"  Std:  {values['std']:.4f}")
        lines.append(f"  Min:  {values['min']:.4f}")
        lines.append(f"  Max:  {values['max']:.4f}")
        lines.append(f"  Count: {values['count']}")

    # Per-group stats
    if stats["by_group"]:
        lines.append(f"\n\nPer-{group_label} Statistics:")
        lines.append("-" * 60)
        for group_value, metrics in stats["by_group"].items():
            lines.append(f"\n{group_label}: {group_value}")
            for metric_col, values in metrics.items():
                lines.append(f"  {metric_col}:")
                lines.append(f"    Mean: {values['mean']:.4f}")
                lines.append(f"    Std:  {values['std']:.4f}")
                lines.append(f"    Min:  {values['min']:.4f}")
                lines.append(f"    Max:  {values['max']:.4f}")
                lines.append(f"    Count: {values['count']}")

    return "\n".join(lines) + "\n"


def _format_per_group_results(
    df: pd.DataFrame,
    metric_cols: list[str],
    metric_name: str,
    group_label: str,
    group_column: str,
) -> str:
    """Format detailed per-group results showing all data points."""
    lines = []
    lines.append(f"\n{metric_name} - Results by {group_label}")
    lines.append("=" * 60)

    if group_column not in df.columns:
        return ""

    for group_value, group in df.groupby(group_column):
        lines.append(f"\n--- {group_value} ---")
        lines.append(f"Count: {len(group)}")

        # Show statistics for this group
        for col in metric_cols:
            if col in group.columns:
                lines.append(f"\n{col}:")
                lines.append(f"  Mean: {group[col].mean():.4f}")
                lines.append(f"  Std:  {group[col].std():.4f}")
                lines.append(f"  Min:  {group[col].min():.4f}")
                lines.append(f"  Max:  {group[col].max():.4f}")

    return "\n".join(lines) + "\n"


def _get_group_column(df: pd.DataFrame) -> str | None:
    return "filename" if "filename" in df.columns else None


def _group_label(group_column: str | None) -> str:
    if group_column == "filename":
        return "Filename"
    return "Group"


def _build_group_map(df: pd.DataFrame, group_column: str | None) -> dict:
    if not group_column:
        return {}
    if "index" in df.columns:
        return df.set_index(df["index"].astype(str))[group_column].to_dict()
    return {
        str(i + 1): value
        for i, value in enumerate(df[group_column].tolist())
    }


def _load_metric_df(
    csv_path: Path,
    metric_cols: list[str],
    group_column: str | None,
    group_map: dict,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "index" in df.columns:
        df = df[df["index"] != "avg"].reset_index(drop=True)

    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if group_column and group_column not in df.columns and group_map and "index" in df.columns:
        df[group_column] = df["index"].astype(str).map(group_map)

    return df


def _append_mean_summary(
    summary_rows: dict,
    df: pd.DataFrame,
    metric_cols: list[str],
    group_column: str | None,
    group_values: list,
) -> None:
    for col in metric_cols:
        if col not in df.columns:
            continue
        if col not in summary_rows:
            summary_rows[col] = {"overall": float(df[col].mean())}
        else:
            summary_rows[col]["overall"] = float(df[col].mean())

        if group_column and group_column in df.columns:
            grouped_means = df.groupby(group_column)[col].mean()
            for group_value in group_values:
                if group_value in grouped_means.index:
                    summary_rows[col][group_value] = float(
                        grouped_means.loc[group_value]
                    )


def _write_mean_summary_csv(
    summary_rows: dict,
    output_dir: Path,
    csv_stem: str,
    ts: str,
    group_values: list,
) -> Path | None:
    if not summary_rows:
        return None

    columns = ["metric", "overall"] + group_values
    rows = []
    for metric_name, values in summary_rows.items():
        row = {"metric": metric_name, "overall": values.get("overall")}
        for group_value in group_values:
            row[group_value] = values.get(group_value)
        rows.append(row)

    summary_df = pd.DataFrame(rows, columns=columns)
    for col in summary_df.columns:
        if col == "metric":
            continue
        summary_df[col] = summary_df[col].map(
            lambda v: "" if pd.isna(v) else f"{v:.2f}"
        )

    summary_csv_path = output_dir / \
        f"evaluation_mean_metrics_{csv_stem}_{ts}.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    return summary_csv_path


def _capture_stdout(func, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        func(*args, **kwargs)
    return buf.getvalue()


def _print_section_header(title: str, leading_newline: bool = False) -> None:
    if leading_newline:
        print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _emit_grouped_sections(
    report_sections: list[str],
    df: pd.DataFrame,
    metric_cols: list[str],
    metric_name: str,
    group_column: str | None,
) -> dict | None:
    if not group_column or group_column not in df.columns:
        return None

    group_label = _group_label(group_column)
    per_group_text = _format_per_group_results(
        df, metric_cols, metric_name, group_label, group_column)
    if per_group_text:
        print(per_group_text)
        report_sections.append(per_group_text)

    grouped = _compute_grouped_statistics(df, metric_cols, group_column)
    grouped_text = _format_grouped_stats(
        grouped, metric_name, group_label)
    print(grouped_text)
    report_sections.append(grouped_text)
    return grouped


def _emit_metric_report(
    report_sections: list[str],
    title: str,
    body: str,
) -> None:
    report_sections.append(f"{title}\n")
    report_sections.append("-" * 60 + "\n")
    report_sections.append(body.strip() + "\n\n")


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
        default="evaluation/results",
        help="Directory to save evaluation reports (default: evaluation/results)",
    )

    args = parser.parse_args()

    results = {}
    csv_path = Path(args.csv_path)

    # Load the CSV to check for grouping column
    df = pd.read_csv(csv_path)
    group_column = _get_group_column(df)
    group_map = _build_group_map(df, group_column)
    group_values = []
    if group_column and group_column in df.columns:
        group_values = df[group_column].dropna().unique().tolist()

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
    if group_column:
        report_sections.append(f"Grouping by column: {group_column}\n")
        report_sections.append(
            f"Unique groups: {df[group_column].nunique()}\n")
    report_sections.append("=" * 60 + "\n")

    summary_rows = {}

    # 1. Compute ROUGE-L metrics
    if args.metrics in ["rouge", "all"]:
        _print_section_header("Computing ROUGE-L metrics...")
        rouge_text = _capture_stdout(compute_rouge_metrics, csv_path)
        # Keep console output visible too (since we captured it)
        print(rouge_text, end="")

        _emit_metric_report(report_sections, "ROUGE-L metrics", rouge_text)
        results["rouge_stdout"] = rouge_text

        rouge_cols = [
            'rouge_l_precision',
            'rouge_l_recall',
            'rouge_l_fmeasure',
        ]
        rouge_df = _load_metric_df(
            csv_path, rouge_cols, group_column, group_map)
        _append_mean_summary(
            summary_rows, rouge_df, rouge_cols, group_column, group_values)
        rouge_grouped = _emit_grouped_sections(
            report_sections, rouge_df, rouge_cols, "ROUGE-L", group_column)
        if rouge_grouped:
            results["rouge_grouped"] = rouge_grouped

    # 2. Compute style similarity
    if args.metrics in ["style", "all"]:
        style_jobs = [
            {
                "title": "Style Similarity",
                "model": "AnnaWegmann/Style-Embedding",
                "column": "style_similarity",
                "result_key": "style_similarity",
            },
            {
                "title": "Style Distance",
                "model": "StyleDistance/styledistance",
                "column": "style_distance",
                "result_key": "style_distance",
            },
        ]

        for job in style_jobs:
            _print_section_header(
                f"Computing {job['title']}...", leading_newline=True)
            style_metrics = compute_style_similarity_metrics(
                csv_path,
                reference_column="reference",
                prediction_column="prediction",
                style_model=job["model"],
                column_name=job["column"],
            )

            style_text = _capture_stdout(
                print_style_similarity_metrics, style_metrics, "Results")
            print(style_text, end="")

            _emit_metric_report(
                report_sections, job["title"], style_text)
            results[job["result_key"]] = style_metrics

            style_cols = [job["column"]]
            style_df = _load_metric_df(
                csv_path, style_cols, group_column, group_map)
            _append_mean_summary(
                summary_rows, style_df, style_cols, group_column, group_values)
            style_grouped = _emit_grouped_sections(
                report_sections,
                style_df,
                style_cols,
                job["title"],
                group_column,
            )
            if style_grouped:
                results[f"{job['result_key']}_grouped"] = style_grouped

    # 4. Compute AI detection
    if args.metrics in ["ai", "all"]:
        _print_section_header(
            "Computing AI Content Detection...", leading_newline=True)

        ai_metrics = compute_ai_detection_metrics(
            csv_path, column="prediction")

        ai_text = _capture_stdout(
            print_ai_detection_metrics, ai_metrics, "Results")
        print(ai_text, end="")

        _emit_metric_report(
            report_sections, "AI Content Detection", ai_text)
        results["ai_detection"] = ai_metrics

        ai_cols = ["prediction_ai_score"]
        ai_df = _load_metric_df(csv_path, ai_cols, group_column, group_map)
        _append_mean_summary(
            summary_rows, ai_df, ai_cols, group_column, group_values)
        ai_grouped = _emit_grouped_sections(
            report_sections, ai_df, ai_cols, "AI Content Detection", group_column)
        if ai_grouped:
            results["ai_detection_grouped"] = ai_grouped

    summary_csv_path = _write_mean_summary_csv(
        summary_rows, output_dir, csv_path.stem, ts, group_values)

    # Save report(s)
    report_txt_path.write_text("".join(report_sections), encoding="utf-8")
    report_json_path.write_text(json.dumps(
        results, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Saved report: {report_txt_path}")
    print(f"Saved metrics JSON: {report_json_path}")
    if summary_csv_path:
        print(f"Saved mean summary CSV: {summary_csv_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
