import json
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from synthesis.utils.validator import (
    StyleSimilarityValidator,
    load_entries_from_list
)

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")


def evaluate_per_persona(data, validator, context_threshold=0.8):
    """Evaluate entries grouped by persona."""
    from collections import defaultdict

    # Group data by persona
    data_by_persona = defaultdict(list)
    for item in data:
        persona_name = item['persona']['name']
        data_by_persona[persona_name].append(item)

    print("\n" + "="*80)
    print(f"EVALUATING {len(data_by_persona)} PERSONAS SEPARATELY")
    print("="*80)

    persona_results = {}

    for persona_name, persona_data in data_by_persona.items():
        print(f"\n{'='*80}")
        print(f"Persona: {persona_name}")
        print(f"Entries: {len(persona_data)}")
        print(f"{'='*80}")

        # Convert persona data to sample format
        sample_data = [
            {
                "context": item["prompt"],
                "output": item["personalized"],
            }
            for item in persona_data
        ]

        # Load entries
        entries = load_entries_from_list(sample_data)

        # Run validation
        results = validator.validate_hypothesis(
            entries, context_threshold=context_threshold)

        persona_results[persona_name] = results

    return persona_results


def aggregate_persona_results(persona_results):
    """Aggregate statistics across all personas."""
    print("\n" + "="*80)
    print("AGGREGATED STATISTICS ACROSS ALL PERSONAS")
    print("="*80)

    # Collect metrics from each persona
    similar_means = []
    dissimilar_means = []
    pearson_rs = []
    spearman_rs = []
    cohens_ds = []
    mann_whitney_pvals = []
    hypothesis_supported_count = 0

    for persona_name, results in persona_results.items():
        stats = results['statistics']
        corr = results['correlation']

        similar_means.append(stats['similar_contexts']['mean'])
        dissimilar_means.append(stats['dissimilar_contexts']['mean'])

        if 'pearson' in corr:
            pearson_rs.append(corr['pearson']['r'])
        if 'spearman' in corr:
            spearman_rs.append(corr['spearman']['r'])
        if 'effect_size' in stats:
            cohens_ds.append(stats['effect_size']['cohens_d'])
        if 'mann_whitney_u' in stats:
            mann_whitney_pvals.append(stats['mann_whitney_u']['p_value'])

        if results['hypothesis_supported']:
            hypothesis_supported_count += 1

    # Compute aggregates
    print(f"Number of personas analyzed: {len(persona_results)}")
    print("\nOutput Similarity for Similar Contexts:")
    print(
        f"  Mean across personas: {np.mean(similar_means):.4f} (±{np.std(similar_means):.4f})")
    print(
        f"  Range: [{np.min(similar_means):.4f}, {np.max(similar_means):.4f}]")

    print("\nOutput Similarity for Dissimilar Contexts:")
    print(
        f"  Mean across personas: {np.mean(dissimilar_means):.4f} (±{np.std(dissimilar_means):.4f})")
    print(
        f"  Range: [{np.min(dissimilar_means):.4f}, {np.max(dissimilar_means):.4f}]")

    print("\nMean Difference (Similar - Dissimilar):")
    differences = np.array(similar_means) - np.array(dissimilar_means)
    print(f"  Mean: {np.mean(differences):.4f} (±{np.std(differences):.4f})")
    print(f"  Range: [{np.min(differences):.4f}, {np.max(differences):.4f}]")

    if pearson_rs:
        print("\nPearson Correlation (r):")
        print(
            f"  Mean across personas: {np.mean(pearson_rs):.4f} (±{np.std(pearson_rs):.4f})")
        print(f"  Range: [{np.min(pearson_rs):.4f}, {np.max(pearson_rs):.4f}]")

    if spearman_rs:
        print("\nSpearman Correlation (ρ):")
        print(
            f"  Mean across personas: {np.mean(spearman_rs):.4f} (±{np.std(spearman_rs):.4f})")
        print(
            f"  Range: [{np.min(spearman_rs):.4f}, {np.max(spearman_rs):.4f}]")

    if cohens_ds:
        print("\nEffect Size (Cohen's d):")
        print(
            f"  Mean across personas: {np.mean(cohens_ds):.4f} (±{np.std(cohens_ds):.4f})")
        print(f"  Range: [{np.min(cohens_ds):.4f}, {np.max(cohens_ds):.4f}]")

    if mann_whitney_pvals:
        print("\nMann-Whitney U Test p-values:")
        print(f"  Mean: {np.mean(mann_whitney_pvals):.4f}")
        significant_count = sum(1 for p in mann_whitney_pvals if p < 0.05)
        print(
            f"  Significant (p < 0.05): {significant_count}/{len(mann_whitney_pvals)}")

    print(
        f"\nHypothesis Supported: {hypothesis_supported_count}/{len(persona_results)} personas")
    print("="*80)

    return {
        'similar_means': similar_means,
        'dissimilar_means': dissimilar_means,
        'differences': differences.tolist(),
        'pearson_rs': pearson_rs,
        'spearman_rs': spearman_rs,
        'cohens_ds': cohens_ds,
        'mann_whitney_pvals': mann_whitney_pvals,
        'hypothesis_supported_count': hypothesis_supported_count
    }


def main():
    # Load data
    with open('./synthesis/output/email/dataset.json', 'r') as f:
        data = json.load(f)

    save_dir = os.path.join(os.path.dirname(__file__), "save")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"evaluate_{timestamp}.log")

    with open(log_path, "w", encoding="utf-8") as log_file:
        class TeeStdout:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                return len(data)

            def flush(self):
                for stream in self.streams:
                    stream.flush()

        tee = TeeStdout(sys.stdout, log_file)
        original_stdout = sys.stdout
        sys.stdout = tee
        try:
            print(f"Loaded {len(data)} entries\n")

            # Create validator instance (reuse for all analyses)
            validator = StyleSimilarityValidator(
                context_embedding="all-MiniLM-L6-v2", style_embedding="AnnaWegmann/Style-Embedding")

            context_threshold = 0.8

            # Evaluate per persona
            persona_results = evaluate_per_persona(
                data, validator, context_threshold)

            # Aggregate results across personas
            aggregate_stats = aggregate_persona_results(persona_results)

            print("\n" + "="*80)
            print("Evaluation complete")
            print("="*80)
            print(f"Saved evaluation log to: {log_path}")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
