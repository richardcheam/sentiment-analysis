"""Embedding backends for clustering and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(slots=True)
class EmbeddingResult:
    """Dense review embeddings with metadata useful for theme extraction."""

    vectors: np.ndarray
    vectorizer: TfidfVectorizer
    tfidf_matrix: Any


def build_embeddings(
    texts: list[str],
    backend: str,
    seed: int,
    max_features: int,
    dimensions: int,
) -> EmbeddingResult:
    """Create dense embeddings for review texts."""
    if backend != "tfidf_svd":
        raise ValueError(
            f"Unsupported embedding backend: {backend}. "
            "Only 'tfidf_svd' is currently implemented in the local workflow."
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2 if len(texts) >= 20 else 1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    if tfidf_matrix.shape[1] <= 1:
        vectors = tfidf_matrix.toarray()
        return EmbeddingResult(vectors=vectors, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)

    n_components = min(dimensions, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    if n_components < 2:
        vectors = tfidf_matrix.toarray()
        return EmbeddingResult(vectors=vectors, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)

    reducer = TruncatedSVD(n_components=n_components, random_state=seed)
    vectors = reducer.fit_transform(tfidf_matrix)
    return EmbeddingResult(vectors=vectors, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)
