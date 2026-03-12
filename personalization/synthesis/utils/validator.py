"""
Style Similarity Validator

This module validates the hypothesis that outputs from similar contexts
have similar writing styles, while outputs from different contexts have
different writing styles.

Usage:
    from style_similarity_validator import StyleSimilarityValidator

    validator = StyleSimilarityValidator()
    results = validator.validate_hypothesis(entries)
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


@dataclass
class Entry:
    """Represents a single context-output pair"""
    context: str
    output: str
    context_id: str = None  # Optional: to group similar contexts


class StyleFeatureExtractor:
    """Extract style embeddings from text using pre-trained Style-Embedding model"""

    def __init__(self, context_embedding_model="all-MiniLM-L6-v2", style_embedding_model="AnnaWegmann/Style-Embedding"):
        """
        Initialize the style feature extractor.
        """
        self.embedding_model = SentenceTransformer(context_embedding_model)
        self.style_embedding_model = SentenceTransformer(style_embedding_model)
        print("Model loaded successfully!")

    def extract_styles(self, text: str) -> np.ndarray:
        """
        Extract style embedding from text using Style-Embedding model.

        Args:
            text: Input text string

        Returns:
            numpy array of style embedding features
        """
        if not text or len(text.strip()) == 0:
            # Return zero vector with same dimension as model output
            return np.zeros(self.style_embedding_model.get_sentence_embedding_dimension())

        # Extract style embedding
        embedding = self.style_embedding_model.encode(text, convert_to_numpy=True)
        return embedding

    def extract_embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract general embeddings for multiple texts efficiently.

        Args:
            texts: List of input text strings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        # Handle empty texts
        processed_texts = [text if text and len(text.strip()) > 0 else " " for text in texts]

        # Batch encode for efficiency
        embeddings = self.embedding_model.encode(processed_texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

    def extract_styles_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract style embeddings for multiple texts efficiently.

        Args:
            texts: List of input text strings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        # Handle empty texts
        processed_texts = [text if text and len(text.strip()) > 0 else " " for text in texts]

        # Batch encode for efficiency
        embeddings = self.style_embedding_model.encode(processed_texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings


class StyleSimilarityValidator:
    """Main validator for the style similarity hypothesis"""

    def __init__(self, context_embedding, style_embedding):
        self.feature_extractor = StyleFeatureExtractor(context_embedding, style_embedding)

    def extract_all_features(self, entries: List[Entry]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from all entries using batch processing for efficiency.

        Args:
            entries: List of Entry objects

        Returns:
            Tuple of (context_features, output_features) as numpy arrays
        """
        # Extract all contexts and outputs
        contexts = [entry.context for entry in entries]
        outputs = [entry.output for entry in entries]

        # Batch process for efficiency
        print("Extracting context embeddings...")
        context_features = self.feature_extractor.extract_embed_batch(contexts)

        print("Extracting output style embeddings...")
        output_features = self.feature_extractor.extract_styles_batch(outputs)

        return context_features, output_features

    def compute_pairwise_similarities(
        self,
        entries: List[Entry]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise similarities for contexts and outputs using embeddings.

        Args:
            entries: List of Entry objects

        Returns:
            Tuple of (context_similarity_matrix, output_similarity_matrix)
        """
        n = len(entries)
        context_sim_matrix = np.zeros((n, n))

        # Extract embeddings for both context and output
        context_features, output_features = self.extract_all_features(entries)

        context_sim_matrix = cosine_similarity(context_features)
        output_sim_matrix = cosine_similarity(output_features)

        return context_sim_matrix, output_sim_matrix

    def categorize_pairs(
        self,
        context_sim_matrix: np.ndarray,
        output_sim_matrix: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, List[float]]:
        """
        Categorize pairs into similar/dissimilar contexts.

        Args:
            context_sim_matrix: Matrix of context similarities
            output_sim_matrix: Matrix of output style similarities
            threshold: Threshold for considering contexts as similar

        Returns:
            Dictionary with 'similar_contexts' and 'dissimilar_contexts'
            containing output similarities
        """
        n = len(context_sim_matrix)
        similar_context_pairs = []
        dissimilar_context_pairs = []

        # Extract upper triangle (excluding diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                ctx_sim = context_sim_matrix[i, j]
                out_sim = output_sim_matrix[i, j]

                if ctx_sim >= threshold:
                    similar_context_pairs.append(out_sim)
                else:
                    dissimilar_context_pairs.append(out_sim)

        return {
            'similar_contexts': similar_context_pairs,
            'dissimilar_contexts': dissimilar_context_pairs
        }

    def compute_statistics(
        self,
        similar_output_sims: List[float],
        dissimilar_output_sims: List[float]
    ) -> Dict[str, Any]:
        """
        Compute statistical measures for hypothesis testing.

        Args:
            similar_output_sims: Output similarities for similar contexts
            dissimilar_output_sims: Output similarities for dissimilar contexts

        Returns:
            Dictionary containing statistical test results
        """
        results = {}

        # Basic statistics
        results['similar_contexts'] = {
            'mean': np.mean(similar_output_sims) if similar_output_sims else 0,
            'std': np.std(similar_output_sims) if similar_output_sims else 0,
            'median': np.median(similar_output_sims) if similar_output_sims else 0,
            'count': len(similar_output_sims)
        }

        results['dissimilar_contexts'] = {
            'mean': np.mean(dissimilar_output_sims) if dissimilar_output_sims else 0,
            'std': np.std(dissimilar_output_sims) if dissimilar_output_sims else 0,
            'median': np.median(dissimilar_output_sims) if dissimilar_output_sims else 0,
            'count': len(dissimilar_output_sims)
        }

        # Mann-Whitney U test (non-parametric test for difference in distributions)
        if similar_output_sims and dissimilar_output_sims:
            statistic, pvalue = mannwhitneyu(
                similar_output_sims,
                dissimilar_output_sims,
                alternative='greater'  # Test if similar contexts have higher output similarity
            )
            results['mann_whitney_u'] = {
                'statistic': statistic,
                'p_value': pvalue,
                'significant': pvalue < 0.05
            }

        # Effect size (Cohen's d)
        if similar_output_sims and dissimilar_output_sims:
            mean_diff = results['similar_contexts']['mean'] - results['dissimilar_contexts']['mean']
            pooled_std = np.sqrt(
                (results['similar_contexts']['std']**2 + results['dissimilar_contexts']['std']**2) / 2
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            results['effect_size'] = {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d)
            }

        return results

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient"""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "weak"
        elif abs_r < 0.5:
            return "moderate"
        elif abs_r < 0.7:
            return "strong"
        else:
            return "very strong"

    def compute_correlation(
        self,
        context_sim_matrix: np.ndarray,
        output_sim_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute correlation between context similarity and output similarity.

        This provides a direct measure of how context similarity relates to
        output similarity across all pairs.

        Args:
            context_sim_matrix: Matrix of context similarities
            output_sim_matrix: Matrix of output style similarities

        Returns:
            Dictionary containing correlation statistics
        """
        n = len(context_sim_matrix)

        # Extract upper triangle (excluding diagonal) to get unique pairs
        context_similarities = []
        output_similarities = []

        for i in range(n):
            for j in range(i + 1, n):
                context_similarities.append(context_sim_matrix[i, j])
                output_similarities.append(output_sim_matrix[i, j])

        context_similarities = np.array(context_similarities)
        output_similarities = np.array(output_similarities)

        results = {
            'n_pairs': len(context_similarities),
            'context_sim_range': {
                'min': float(np.min(context_similarities)),
                'max': float(np.max(context_similarities)),
                'mean': float(np.mean(context_similarities)),
                'std': float(np.std(context_similarities))
            },
            'output_sim_range': {
                'min': float(np.min(output_similarities)),
                'max': float(np.max(output_similarities)),
                'mean': float(np.mean(output_similarities)),
                'std': float(np.std(output_similarities))
            }
        }

        # Pearson correlation (linear relationship)
        if len(context_similarities) > 1:
            pearson_r, pearson_p = pearsonr(context_similarities, output_similarities)
            results['pearson'] = {
                'r': float(pearson_r),
                'p_value': float(pearson_p),
                'r_squared': float(pearson_r ** 2),
                'significant': pearson_p < 0.05,
                'interpretation': self._interpret_correlation(pearson_r)
            }

        # Spearman correlation (monotonic relationship, more robust to outliers)
        if len(context_similarities) > 1:
            spearman_r, spearman_p = spearmanr(context_similarities, output_similarities)
            results['spearman'] = {
                'r': float(spearman_r),
                'p_value': float(spearman_p),
                'r_squared': float(spearman_r ** 2),
                'significant': spearman_p < 0.05,
                'interpretation': self._interpret_correlation(spearman_r)
            }

        return results

    def find_outlier_pairs(
        self,
        entries: List[Entry],
        context_sim_matrix: np.ndarray,
        output_sim_matrix: np.ndarray,
        context_sim_threshold: float = 0.7,
        output_sim_threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find outlier pairs with high context similarity but low output similarity.
        
        These are interesting cases where similar contexts produce different styles,
        potentially indicating context features that don't transfer to style.
        
        Args:
            entries: List of Entry objects
            context_sim_matrix: Matrix of context similarities
            output_sim_matrix: Matrix of output style similarities
            context_sim_threshold: Minimum context similarity to consider (default: 0.7)
            output_sim_threshold: Maximum output similarity to consider (default: 0.5)
            top_k: If specified, return only top k outliers ranked by context-output gap
            
        Returns:
            List of dictionaries containing outlier pair information
        """
        n = len(entries)
        outliers = []
        
        # Find pairs meeting outlier criteria
        for i in range(n):
            for j in range(i + 1, n):
                ctx_sim = context_sim_matrix[i, j]
                out_sim = output_sim_matrix[i, j]
                
                # High context similarity but low output similarity
                if ctx_sim >= context_sim_threshold and out_sim <= output_sim_threshold:
                    gap = ctx_sim - out_sim  # How much the output differs despite similar context
                    
                    outliers.append({
                        'index_1': i,
                        'index_2': j,
                        'entry_1': entries[i],
                        'entry_2': entries[j],
                        'context_similarity': float(ctx_sim),
                        'output_similarity': float(out_sim),
                        'gap': float(gap),
                        'context_1': entries[i].context,
                        'context_2': entries[j].context,
                        'output_1': entries[i].output,
                        'output_2': entries[j].output,
                        'context_id_1': entries[i].context_id,
                        'context_id_2': entries[j].context_id
                    })
        
        # Sort by gap (largest first)
        outliers.sort(key=lambda x: x['gap'], reverse=True)
        
        # Return top k if specified
        if top_k is not None:
            outliers = outliers[:top_k]
        
        return outliers

    def save_outliers_to_file(
        self,
        outliers: List[Dict[str, Any]],
        filepath: str,
        format: str = 'csv'
    ) -> None:
        """
        Save outlier pairs to a file for human examination.
        
        Args:
            outliers: List of outlier dictionaries from find_outlier_pairs
            filepath: Path to save the file
            format: Output format ('csv' or 'json')
        """
        if not outliers:
            print("No outliers to save.")
            return
        
        if format == 'json':
            import json
            # Convert Entry objects to dicts for JSON serialization
            serializable_outliers = []
            for o in outliers:
                o_copy = o.copy()
                o_copy.pop('entry_1', None)
                o_copy.pop('entry_2', None)
                serializable_outliers.append(o_copy)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_outliers, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(outliers)} outliers to {filepath}")
            
        elif format == 'csv':
            import pandas as pd
            
            # Prepare data for CSV
            csv_data = []
            for o in outliers:
                csv_data.append({
                    'index_1': o['index_1'],
                    'index_2': o['index_2'],
                    'context_id_1': o.get('context_id_1', ''),
                    'context_id_2': o.get('context_id_2', ''),
                    'context_similarity': o['context_similarity'],
                    'output_similarity': o['output_similarity'],
                    'gap': o['gap'],
                    'context_1': o['context_1'],
                    'context_2': o['context_2'],
                    'output_1': o['output_1'],
                    'output_2': o['output_2']
                })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"Saved {len(outliers)} outliers to {filepath}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")

    def plot_correlation_scatter(
        self,
        context_sim_matrix: np.ndarray,
        output_sim_matrix: np.ndarray,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        title: str = "Context Similarity vs Output Similarity",
        highlight_outliers: Optional[List[Dict[str, Any]]] = None
    ) -> plt.Figure:
        """
        Create a scatter plot showing the correlation between context similarity 
        and output similarity.

        Args:
            context_sim_matrix: Matrix of context similarities
            output_sim_matrix: Matrix of output style similarities
            save_path: Optional path to save the figure
            show_plot: Whether to display the plot
            title: Title for the plot
            highlight_outliers: Optional list of outlier dicts to highlight in red

        Returns:
            matplotlib Figure object
        """
        n = len(context_sim_matrix)
        
        # Extract upper triangle (excluding diagonal) to get unique pairs
        context_similarities = []
        output_similarities = []
        
        for i in range(n):
            for j in range(i + 1, n):
                context_similarities.append(context_sim_matrix[i, j])
                output_similarities.append(output_sim_matrix[i, j])
        
        context_similarities = np.array(context_similarities)
        output_similarities = np.array(output_similarities)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        ax.scatter(context_similarities, output_similarities, 
                   alpha=0.5, s=30, c='steelblue', edgecolors='navy', linewidth=0.5,
                   label='All pairs')
        
        # Highlight outliers if provided
        if highlight_outliers:
            outlier_ctx_sims = [o['context_similarity'] for o in highlight_outliers]
            outlier_out_sims = [o['output_similarity'] for o in highlight_outliers]
            ax.scatter(outlier_ctx_sims, outlier_out_sims,
                      alpha=0.8, s=100, c='red', edgecolors='darkred', 
                      linewidth=2, marker='o', label=f'Outliers (n={len(highlight_outliers)})')
        
        # Add trend line
        z = np.polyfit(context_similarities, output_similarities, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(context_similarities.min(), context_similarities.max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend line (y={z[0]:.3f}x+{z[1]:.3f})')
        
        # Compute correlations
        pearson_r, pearson_p = pearsonr(context_similarities, output_similarities)
        spearman_r, spearman_p = spearmanr(context_similarities, output_similarities)
        
        # Add correlation info to plot
        textstr = f'Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})\n'
        textstr += f'Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.4f})\n'
        textstr += f'n = {len(context_similarities)} pairs'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        # Labels and title
        ax.set_xlabel('Context Similarity (Cosine)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Output Style Similarity (Cosine)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set axis limits with some padding
        ax.set_xlim([context_similarities.min() - 0.05, context_similarities.max() + 0.05])
        ax.set_ylim([output_similarities.min() - 0.05, output_similarities.max() + 0.05])
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to: {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig

    def validate_hypothesis(
        self,
        entries: List[Entry],
        context_threshold: float = 0.5,
        plot_correlation: bool = False,
        save_plot_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to validate the style similarity hypothesis.

        Hypothesis:
        - H1: Outputs from similar contexts have similar writing styles
        - H2: Outputs from different contexts have different writing styles

        Args:
            entries: List of Entry objects with context and output
            context_threshold: Threshold for considering contexts as similar
            plot_correlation: Whether to generate correlation scatter plot
            save_plot_path: Optional path to save the correlation plot

        Returns:
            Dictionary containing all analysis results
        """
        print(f"Analyzing {len(entries)} entries...")

        # Step 1: Compute pairwise similarities
        print("Computing pairwise similarities...")
        context_sim_matrix, output_sim_matrix = self.compute_pairwise_similarities(entries)

        # Step 2: Categorize pairs
        print("Categorizing pairs...")
        categorized = self.categorize_pairs(
            context_sim_matrix,
            output_sim_matrix,
            context_threshold
        )

        # Step 3: Compute statistics
        print("Computing statistics...")
        statistics = self.compute_statistics(
            categorized['similar_contexts'],
            categorized['dissimilar_contexts']
        )

        # Step 3.5: Compute correlation
        print("Computing correlation...")
        correlation_results = self.compute_correlation(
            context_sim_matrix,
            output_sim_matrix
        )

        # Step 3.6: Plot correlation if requested
        if plot_correlation:
            print("Generating correlation scatter plot...")
            fig = self.plot_correlation_scatter(
                context_sim_matrix,
                output_sim_matrix,
                save_path=save_plot_path,
                show_plot=False  # Don't auto-show in batch analysis
            )
            correlation_results['plot'] = fig

        # Step 4: Compile results
        results = {
            'context_similarity_matrix': context_sim_matrix,
            'output_similarity_matrix': output_sim_matrix,
            'categorized_pairs': categorized,
            'statistics': statistics,
            'correlation': correlation_results,
            'hypothesis_supported': (
                statistics.get('mann_whitney_u', {}).get('significant', False) and
                statistics.get('similar_contexts', {}).get('mean', 0) >
                statistics.get('dissimilar_contexts', {}).get('mean', 0)
            )
        }

        print("="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)

        if 'pearson' in correlation_results:
            pearson = correlation_results['pearson']
            print("Pearson correlation:")
            print(f"  r = {pearson['r']:.4f} (p = {pearson['p_value']:.4f})")
            print(f"  R² = {pearson['r_squared']:.4f}")
            print(f"  Interpretation: {pearson['interpretation']}")
            print(f"  Significant: {pearson['significant']}")

        if 'spearman' in correlation_results:
            spearman = correlation_results['spearman']
            print("\nSpearman correlation:")
            print(f"  ρ = {spearman['r']:.4f} (p = {spearman['p_value']:.4f})")
            print(f"  ρ² = {spearman['r_squared']:.4f}")
            print(f"  Interpretation: {spearman['interpretation']}")
            print(f"  Significant: {spearman['significant']}")

        print("="*60)
        print("HYPOTHESIS VALIDATION RESULTS")
        print("="*60)

        print(f"Number of similar pairs: {len(categorized['similar_contexts'])}")
        print(f"Number of dissimilar pairs: {len(categorized['dissimilar_contexts'])}")

        print(f"\nSimilar contexts - Mean output similarity: {statistics['similar_contexts']['mean']:.4f}")
        print(f"Dissimilar contexts - Mean output similarity: {statistics['dissimilar_contexts']['mean']:.4f}")

        if 'mann_whitney_u' in statistics:
            print(f"\nMann-Whitney U test p-value: {statistics['mann_whitney_u']['p_value']:.4f}")
            print(f"Significant difference: {statistics['mann_whitney_u']['significant']}")

        if 'effect_size' in statistics:
            print(f"\nEffect size (Cohen's d): {statistics['effect_size']['cohens_d']:.4f}")
            print(f"Interpretation: {statistics['effect_size']['interpretation']}")

        print(f"Number of pairs analyzed: {correlation_results['n_pairs']}")
        print(f"\nContext similarity range: [{correlation_results['context_sim_range']['min']:.4f}, {correlation_results['context_sim_range']['max']:.4f}]")
        print(f"Context similarity mean: {correlation_results['context_sim_range']['mean']:.4f} (±{correlation_results['context_sim_range']['std']:.4f})")
        print(f"\nOutput similarity range: [{correlation_results['output_sim_range']['min']:.4f}, {correlation_results['output_sim_range']['max']:.4f}]")
        print(f"Output similarity mean: {correlation_results['output_sim_range']['mean']:.4f} (±{correlation_results['output_sim_range']['std']:.4f})")

        print(f"\nHypothesis supported: {results['hypothesis_supported']}")
        print("="*60)

        return results


def load_entries_from_list(data: List[Dict[str, str]]) -> List[Entry]:
    """
    Helper function to load entries from a list of dictionaries.

    Args:
        data: List of dicts with 'context' and 'output' keys

    Returns:
        List of Entry objects
    """
    entries = []
    for i, item in enumerate(data):
        entry = Entry(
            context=item['context'],
            output=item['output'],
            context_id=item.get('context_id', f"ctx_{i}")
        )
        entries.append(entry)
    return entries


def load_entries_from_csv(filepath: str, context_col: str, output_col: str) -> List[Entry]:
    """
    Helper function to load entries from a CSV file.

    Args:
        filepath: Path to CSV file
        context_col: Name of the context column
        output_col: Name of the output column

    Returns:
        List of Entry objects
    """
    import pandas as pd

    df = pd.read_csv(filepath)
    entries = []

    for idx, row in df.iterrows():
        entry = Entry(
            context=str(row[context_col]),
            output=str(row[output_col]),
            context_id=f"ctx_{idx}"
        )
        entries.append(entry)

    return entries
