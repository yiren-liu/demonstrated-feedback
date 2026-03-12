import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from synthesis.utils.validator import StyleSimilarityValidator, load_entries_from_csv

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")


def clean_prompts(prompts: List[str]) -> List[str]:
    """
    Remove common prefix and suffix from prompts.

    Args:
        prompts: List of prompt strings

    Returns:
        List of cleaned prompt strings
    """
    cleaned = []
    for prompt in prompts:
        cleaned_prompt = prompt
        if cleaned_prompt.startswith("Write an email to "):
            cleaned_prompt = cleaned_prompt[len("Write an email to "):]
        if cleaned_prompt.endswith(" DO NOT include the subject line."):
            cleaned_prompt = cleaned_prompt[: -
                                            len(" DO NOT include the subject line.")]
        cleaned.append(cleaned_prompt)
    return cleaned


def main():
    # Load data from CSV
    # csv_path = 'emails/emails_by_employee.prompts/allen-p.prompts.csv'
    csv_path = 'emails/emails_by_employee.prompts/smith-m.prompts.csv'

    # Load entries directly from CSV with correct column mappings
    # "prompt" column -> context, "email" column -> output
    entries = load_entries_from_csv(
        csv_path, context_col="prompt", output_col="email")

    print(f"Loaded {len(entries)} entries from {csv_path}")

    # Clean the prompts (contexts) before validation
    contexts = [entry.context for entry in entries]
    # cleaned_contexts = clean_prompts(contexts)
    cleaned_contexts = contexts
    for i, entry in enumerate(entries):
        entry.context = cleaned_contexts[i]

    print("Cleaned prompts to remove common prefix/suffix")

    # Create validator instance
    validator = StyleSimilarityValidator(
        context_embedding="all-MiniLM-L6-v2", style_embedding="AnnaWegmann/Style-Embedding")

    print(f"\n{'='*80}")
    print("Evaluating smith-m emails")
    print(f"{'='*80}")
    print(f"Number of entries: {len(entries)}")

    # Run validation with context similarity threshold of 0.3
    # (lower threshold means we're more liberal in considering contexts as "similar")
    results = validator.validate_hypothesis(entries, context_threshold=0.3, plot_correlation=True, save_plot_path="correlation_plot.png")

    # Extract similarity matrices from results
    context_sim_matrix = results['context_similarity_matrix']
    output_sim_matrix = results['output_similarity_matrix']

    # Find outlier pairs: high context similarity but low output similarity
    print(f"\n{'='*80}")
    print("FINDING OUTLIERS (High Context Sim, Low Output Sim)")
    print(f"{'='*80}")
    
    outliers = validator.find_outlier_pairs(
        entries,
        context_sim_matrix,
        output_sim_matrix,
        context_sim_threshold=0.4,  # Context similarity >= 0.7
        output_sim_threshold=0,   # Output similarity <= 0.5
        top_k=1000  # Get top 20 outliers by gap
    )
    
    print(f"\nFound {len(outliers)} outlier pairs")
    
    if outliers:
        # Print top 5 outliers
        print("\nTop 5 outliers:")
        for idx, outlier in enumerate(outliers[:5]):
            print(f"\n--- Outlier #{idx+1} ---")
            print(f"Context similarity: {outlier['context_similarity']:.4f}")
            print(f"Output similarity: {outlier['output_similarity']:.4f}")
            print(f"Gap: {outlier['gap']:.4f}")
            print(f"\nContext 1 (ID: {outlier['context_id_1']}): {outlier['context_1'][:100]}...")
            print(f"Context 2 (ID: {outlier['context_id_2']}): {outlier['context_2'][:100]}...")
            print(f"\nOutput 1: {outlier['output_1'][:150]}...")
            print(f"Output 2: {outlier['output_2'][:150]}...")
        
        # Save outliers to CSV for detailed examination
        validator.save_outliers_to_file(outliers, 'outliers.csv', format='csv')
        validator.save_outliers_to_file(outliers, 'outliers.json', format='json')
        
        # Create a scatter plot with outliers highlighted
        print("\nGenerating scatter plot with highlighted outliers...")
        validator.plot_correlation_scatter(
            context_sim_matrix,
            output_sim_matrix,
            save_path='correlation_plot_with_outliers.png',
            show_plot=False,
            title='Context vs Output Similarity (Outliers Highlighted)',
            highlight_outliers=outliers
        )
        print("Saved plot with outliers to: correlation_plot_with_outliers.png")

    print(f"\n{'='*80}")
    print("Evaluation complete for all personas")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
