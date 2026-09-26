"""Decision Logic and Threshold Calibration Module.

Because the evaluation metric is Macro-F0.5 (precision-weighted, penalizing false
positives 2x more than false negatives) and includes singletons (which score 0.0 upon
any false positive), threshold calibration is critical.
"""

from typing import Dict, List, Set, Tuple
import pandas as pd


def calibrate_threshold(
    probabilities: List[float],
    labels: List[int],
    beta: float = 0.5,
) -> float:
    """Finds optimal decision probability threshold maximizing F_beta.

    Args:
        probabilities: Predicted pair match probabilities.
        labels: Ground truth binary labels.
        beta: Metric weight (0.5 for precision-heavy).

    Returns:
        Optimal decision threshold float.
    """
    raise NotImplementedError("Decision threshold calibration will be implemented in Phase 5.")


def filter_predictions(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    threshold: float = 0.5,
) -> Dict[str, List[str]]:
    """Applies decision threshold to candidate pairs.

    Args:
        candidate_scores: Mapping from source1_id to list of (candidate_id, score).
        threshold: Minimum score required to accept a match.

    Returns:
        Mapping from source1_id to list of matched candidate IDs.
    """
    results: Dict[str, List[str]] = {}
    for s1_id, pairs in candidate_scores.items():
        matched = [c_id for c_id, score in pairs if score >= threshold]
        results[s1_id] = matched
    return results
