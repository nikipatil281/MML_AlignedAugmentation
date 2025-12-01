# src/utils/metrics.py

import numpy as np
from typing import Dict


def compute_retrieval_metrics(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute R@1, R@5, R@10 for a square similarity matrix.

    Assumes:
        - similarity_matrix[i, j] is the similarity between item i and j
        - ground-truth match for i is at index i (diagonal)

    Args:
        similarity_matrix: [N, N] numpy array

    Returns:
        dict with keys: 'R@1', 'R@5', 'R@10'
    """
    assert similarity_matrix.ndim == 2
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1]
    N = similarity_matrix.shape[0]

    ranks = []

    for i in range(N):
        sims = similarity_matrix[i]  # similarities to all candidates
        # Sort indices by descending similarity
        sorted_indices = np.argsort(-sims)

        # Rank of the ground-truth (i)
        rank = np.where(sorted_indices == i)[0][0]  # position of i in sorted list
        ranks.append(rank)

    ranks = np.array(ranks)

    r1 = 100.0 * np.mean(ranks < 1)
    r5 = 100.0 * np.mean(ranks < 5)
    r10 = 100.0 * np.mean(ranks < 10)

    return {"R@1": r1, "R@5": r5, "R@10": r10}

