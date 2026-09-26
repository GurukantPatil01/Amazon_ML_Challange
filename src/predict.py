"""Inference and Output Generation Module.

Generates the two official competition output files:
1. matching_results.tsv - final resolved entity links (scored on leaderboard)
2. candidate_pairs.tsv - blocking candidate set (audited for blocking efficiency)
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd

from src.config import CANDIDATE_PAIRS_PATH, MATCHING_RESULTS_PATH
from src.utils import setup_logger

logger = setup_logger("predict")


def format_id_list(entity_ids: List[str]) -> str:
    """Formats a list of entity IDs into a clean comma-separated string without duplicates.

    Args:
        entity_ids: List of entity ID strings.

    Returns:
        Comma-separated string of unique IDs preserving order.
    """
    seen: Set[str] = set()
    unique_ids: List[str] = []
    for eid in entity_ids:
        cleaned = eid.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_ids.append(cleaned)
    return ",".join(unique_ids)


def save_submission_tsv(
    predictions: Dict[str, List[str]],
    output_path: Path,
    value_column_name: str = "matched_entity_ids",
) -> None:
    """Saves predictions to a strict TSV file meeting competition specifications.

    Requirements:
    - Tab-separated
    - Header: source1_entity_id, <value_column_name>
    - No quoting
    - Exactly one row per source1 entity

    Args:
        predictions: Mapping from source1_entity_id to list of target entity IDs.
        output_path: Destination path for the TSV.
        value_column_name: Either 'matched_entity_ids' or 'candidate_entity_ids'.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for s1_id, targets in predictions.items():
        records.append({
            "source1_entity_id": s1_id,
            value_column_name: format_id_list(targets),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved {len(df):,} rows to {output_path}")
