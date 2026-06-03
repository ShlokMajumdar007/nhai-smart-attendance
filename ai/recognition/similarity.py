"""
similarity.py
Cosine similarity computation and threshold-based decisions.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized embeddings.
    For L2-normalized vectors: cosine_sim = dot product.
    """
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between embeddings."""
    return float(np.linalg.norm(a - b))


def find_best_match(
    query_embedding: np.ndarray,
    enrolled_embeddings: List[Tuple[str, np.ndarray]],
    threshold: float = 0.65,
) -> Tuple[Optional[str], float]:
    """
    Find the best matching identity from enrolled embeddings.

    Args:
        query_embedding: (128,) embedding of the face to recognize
        enrolled_embeddings: list of (person_id, embedding) tuples
        threshold: minimum cosine similarity to accept a match

    Returns:
        (person_id, score) if matched, (None, best_score) otherwise
    """
    if not enrolled_embeddings:
        return None, 0.0

    best_id = None
    best_score = -1.0

    for person_id, enrolled_emb in enrolled_embeddings:
        score = cosine_similarity(query_embedding, enrolled_emb)
        if score > best_score:
            best_score = score
            best_id = person_id

    if best_score >= threshold:
        logger.info(f"Match found: {best_id} with score {best_score:.4f}")
        return best_id, best_score

    logger.info(f"No match. Best score: {best_score:.4f} (threshold: {threshold})")
    return None, best_score


def rank_matches(
    query_embedding: np.ndarray,
    enrolled_embeddings: List[Tuple[str, np.ndarray]],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Return top-k matches sorted by descending similarity.
    Useful for debugging and analytics.
    """
    scores = [
        (pid, cosine_similarity(query_embedding, emb))
        for pid, emb in enrolled_embeddings
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]