import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class StyleEmbedder:
    """Extract and compare style embeddings from text"""

    def __init__(self, model_name: str = "AnnaWegmann/Style-Embedding"):
        """Initialize the style embedder."""
        self.model = SentenceTransformer(model_name)
        print(f"Loaded style embedding model: {model_name}")

    def encode(self, text: str) -> np.ndarray:
        """Extract style embedding from a single text."""
        if not text or len(text.strip()) == 0:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Extract style embeddings for multiple texts efficiently."""
        processed_texts = [text if text and len(
            text.strip()) > 0 else " " for text in texts]
        return self.model.encode(processed_texts, convert_to_numpy=True, show_progress_bar=show_progress)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between style embeddings of two texts."""
        emb1 = self.encode(text1).reshape(1, -1)
        emb2 = self.encode(text2).reshape(1, -1)
        return float(cosine_similarity(emb1, emb2)[0, 0])

    def compute_similarity_batch(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarities between two sets of embeddings."""
        return cosine_similarity(embeddings1, embeddings2)
