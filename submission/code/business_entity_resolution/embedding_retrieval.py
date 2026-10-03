"""Multilingual Dense Embedding Retrieval Layer for Business Entity Resolution.

Provides semantic candidate retrieval using local pre-trained sentence transformer:
- Model: paraphrase-multilingual-MiniLM-L12-v2
- Dimensions: 384
- Similarity: Cosine Inner Product via FAISS IndexFlatIP
- Purpose: Recovers cross-script Indic transliterations and severe optical/OCR distortions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class MultilingualEmbeddingRetriever:
    """Manages local dense embedding encoding and FAISS ANN retrieval."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.target_ids: List[str] = []

    def encode_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encodes texts into unit-normalized 384-dimensional dense vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    def build_index(self, target_df: pd.DataFrame, text_col: str = "business_name") -> None:
        """Encodes target pool and constructs FAISS IndexFlatIP."""
        self.target_ids = target_df["entity_id"].tolist()
        texts = target_df[text_col].fillna("").astype(str).tolist()
        vectors = self.encode_texts(texts, normalize=True)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)

    def retrieve(self, query_texts: List[str], top_k: int = 50) -> List[List[Tuple[str, float]]]:
        """Performs cosine inner product search for query texts."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        query_vectors = self.encode_texts(query_texts, normalize=True)
        scores, indices = self.index.search(query_vectors, top_k)

        results = []
        for q_scores, q_indices in zip(scores, indices):
            res = []
            for score, idx in zip(q_scores, q_indices):
                if idx >= 0 and idx < len(self.target_ids):
                    res.append((self.target_ids[idx], float(score)))
            results.append(res)
        return results
