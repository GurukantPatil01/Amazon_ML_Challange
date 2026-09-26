"""Data loading and profiling module for Amazon ML Challenge 2026 Entity Resolution.

Provides typed, validated, and logged loaders for training and test sources,
as well as comprehensive profiling utilities for data health inspection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import (
    EXPECTED_GROUND_TRUTH_COLUMNS,
    EXPECTED_SOURCE_COLUMNS,
    LOGS_DIR,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)
from src.utils import setup_logger, timer

logger = setup_logger("load_data", log_file=LOGS_DIR / "data_loading.log")


def infer_source_from_id(entity_id: str) -> str:
    """Infers the source name from an entity ID prefix without hard-coding records.

    Args:
        entity_id: Identifier string (e.g., 'S1-00042', 'S2-00109', 'S3-00001').

    Returns:
        Source name ('source1', 'source2', 'source3', or 'unknown').
    """
    if not isinstance(entity_id, str):
        return "unknown"
    prefix = entity_id.split("-")[0] if "-" in entity_id else ""
    mapping = {
        "S1": "source1",
        "S2": "source2",
        "S3": "source3",
    }
    return mapping.get(prefix, "unknown")


def load_tsv(filepath: Path, expected_columns: List[str]) -> pd.DataFrame:
    """Safely loads a TSV file with strict validation and logging.

    Preserves original data types as strings and avoids silent column corruption.

    Args:
        filepath: Path to the TSV file.
        expected_columns: Expected column names in the header.

    Returns:
        Loaded DataFrame with raw string values preserved.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If expected columns are missing.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Required dataset file not found at: {filepath.resolve()}")

    with timer(f"loading TSV from {filepath.name}", logger):
        df = pd.read_csv(
            filepath,
            sep="\t",
            dtype=str,
            keep_default_na=False,  # Treat missing fields explicitly as empty strings
            na_values=[],           # Prevent auto-converting words like 'NA' or 'None' to NaN
        )

    # Validate schema
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Schema validation failed for {filepath.name}! "
            f"Missing required columns: {missing_cols}. Found columns: {list(df.columns)}"
        )

    logger.info(
        f"Loaded '{filepath.name}': {len(df):,} records, {len(df.columns)} columns: {list(df.columns)}"
    )
    return df


def load_train_source1(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads Source 1 training records."""
    target_path = path or TRAIN_SOURCE1_PATH
    return load_tsv(target_path, EXPECTED_SOURCE_COLUMNS)


def load_train_source2(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads Source 2 training records."""
    target_path = path or TRAIN_SOURCE2_PATH
    return load_tsv(target_path, EXPECTED_SOURCE_COLUMNS)


def load_train_source3(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads Source 3 training records."""
    target_path = path or TRAIN_SOURCE3_PATH
    return load_tsv(target_path, EXPECTED_SOURCE_COLUMNS)


def load_ground_truth(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads ground truth matching labels for the training set."""
    target_path = path or TRAIN_GROUND_TRUTH_PATH
    return load_tsv(target_path, EXPECTED_GROUND_TRUTH_COLUMNS)


def load_test_source1(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads Source 1 test records."""
    target_path = path or TEST_SOURCE1_PATH
    return load_tsv(target_path, EXPECTED_SOURCE_COLUMNS)


def load_test_source2(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads Source 2 test records."""
    target_path = path or TEST_SOURCE2_PATH
    return load_tsv(target_path, EXPECTED_SOURCE_COLUMNS)


def load_test_source3(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads Source 3 test records."""
    target_path = path or TEST_SOURCE3_PATH
    return load_tsv(target_path, EXPECTED_SOURCE_COLUMNS)


def profile_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    is_ground_truth: bool = False,
) -> Dict[str, Any]:
    """Generates comprehensive profiling statistics for a dataset table.

    Computes:
    - Row count
    - Unique entity IDs
    - Missing / empty values per column
    - Duplicate IDs
    - Duplicate names & addresses (for source tables)
    - Country distribution (for source tables)
    - Text length statistics (min, max, mean, median)

    Args:
        df: Dataset DataFrame.
        dataset_name: Identifier name for logging/reporting.
        is_ground_truth: True if profiling the ground truth label table.

    Returns:
        Dictionary of profiling metrics.
    """
    total_rows = len(df)
    profile: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "columns": list(df.columns),
    }

    if is_ground_truth:
        s1_ids = df["source1_entity_id"]
        unique_s1 = s1_ids.nunique()
        duplicate_s1 = total_rows - unique_s1
        empty_matches = (df["matched_entity_ids"].str.strip() == "").sum()
        with_matches = total_rows - empty_matches

        # Count total matched pairs and distribution of matches per entity
        match_counts = df["matched_entity_ids"].apply(
            lambda x: len([i for i in x.split(",") if i.strip()]) if x.strip() else 0
        )

        profile.update({
            "unique_source1_ids": int(unique_s1),
            "duplicate_source1_ids": int(duplicate_s1),
            "singleton_count (0 matches)": int(empty_matches),
            "entities_with_matches": int(with_matches),
            "singleton_percentage": round((empty_matches / total_rows * 100), 2) if total_rows > 0 else 0.0,
            "total_positive_pairs": int(match_counts.sum()),
            "max_matches_per_entity": int(match_counts.max()) if total_rows > 0 else 0,
            "avg_matches_per_entity": round(float(match_counts.mean()), 3) if total_rows > 0 else 0.0,
        })
        return profile

    # For source datasets
    entity_col = "entity_id"
    if entity_col in df.columns:
        unique_ids = df[entity_col].nunique()
        duplicate_ids = total_rows - unique_ids
        profile["unique_entity_ids"] = int(unique_ids)
        profile["duplicate_entity_ids"] = int(duplicate_ids)

        # Source inference check
        inferred_sources = df[entity_col].apply(infer_source_from_id).value_counts().to_dict()
        profile["inferred_source_distribution"] = inferred_sources

    # Missing / empty values
    missing_stats = {}
    for col in df.columns:
        empty_count = ((df[col].isna()) | (df[col].str.strip() == "")).sum()
        missing_stats[col] = {
            "empty_count": int(empty_count),
            "empty_pct": round((empty_count / total_rows * 100), 2) if total_rows > 0 else 0.0,
        }
    profile["missing_values"] = missing_stats

    # Duplicate names and addresses
    if "business_name" in df.columns:
        profile["unique_names"] = int(df["business_name"].nunique())
        profile["duplicate_names"] = int(total_rows - df["business_name"].nunique())
        name_lens = df["business_name"].str.len()
        profile["name_length_stats"] = {
            "min": int(name_lens.min()) if total_rows > 0 else 0,
            "max": int(name_lens.max()) if total_rows > 0 else 0,
            "mean": round(float(name_lens.mean()), 2) if total_rows > 0 else 0.0,
            "median": float(name_lens.median()) if total_rows > 0 else 0.0,
        }

    if "business_address" in df.columns:
        profile["unique_addresses"] = int(df["business_address"].nunique())
        profile["duplicate_addresses"] = int(total_rows - df["business_address"].nunique())
        addr_lens = df["business_address"].str.len()
        profile["address_length_stats"] = {
            "min": int(addr_lens.min()) if total_rows > 0 else 0,
            "max": int(addr_lens.max()) if total_rows > 0 else 0,
            "mean": round(float(addr_lens.mean()), 2) if total_rows > 0 else 0.0,
            "median": float(addr_lens.median()) if total_rows > 0 else 0.0,
        }

    # Country distribution
    if "country" in df.columns:
        profile["country_distribution"] = df["country"].value_counts().to_dict()

    return profile


def run_phase0_profiling() -> Dict[str, Any]:
    """Executes Phase 0 data loading and profiling for all available dataset files.

    Returns:
        Structured dictionary of profile reports for each dataset file.
    """
    logger.info("=" * 60)
    logger.info("STARTING PHASE 0: DATASET LOADING AND PROFILING")
    logger.info("=" * 60)

    datasets_to_check = [
        ("train_source1", load_train_source1, False),
        ("train_source2", load_train_source2, False),
        ("train_source3", load_train_source3, False),
        ("train_ground_truth", load_ground_truth, True),
        ("test_source1", load_test_source1, False),
        ("test_source2", load_test_source2, False),
        ("test_source3", load_test_source3, False),
    ]

    all_profiles: Dict[str, Any] = {}
    missing_files: List[str] = []

    for name, loader_fn, is_gt in datasets_to_check:
        try:
            logger.info(f"--- Profiling {name} ---")
            df = loader_fn()
            profile = profile_dataframe(df, name, is_ground_truth=is_gt)
            all_profiles[name] = profile
            del df
            import gc
            gc.collect()

            # Print brief summary to log
            logger.info(f"Rows: {profile['total_rows']:,}")
            if is_gt:
                logger.info(
                    f"Singletons (no match): {profile['singleton_count (0 matches)']:,} "
                    f"({profile['singleton_percentage']}%), "
                    f"Total positive links: {profile['total_positive_pairs']:,}"
                )
            else:
                logger.info(
                    f"Unique IDs: {profile.get('unique_entity_ids', 'N/A'):,}, "
                    f"Duplicates: {profile.get('duplicate_entity_ids', 'N/A')}"
                )
                if "country_distribution" in profile:
                    logger.info(f"Countries: {profile['country_distribution']}")

        except FileNotFoundError as err:
            logger.warning(f"File not found for {name}: {err}")
            missing_files.append(name)
        except Exception as err:
            logger.error(f"Error profiling {name}: {err}", exc_info=True)
            raise

    logger.info("=" * 60)
    logger.info(
        f"PHASE 0 PROFILING COMPLETE. Successfully profiled {len(all_profiles)} files. "
        f"Missing files: {len(missing_files)}"
    )
    logger.info("=" * 60)

    return {
        "profiles": all_profiles,
        "missing_files": missing_files,
        "status": "PASS" if len(missing_files) == 0 else "PARTIAL",
    }


if __name__ == "__main__":
    report = run_phase0_profiling()
    import json
    print("\n--- PHASE 0 SUMMARY REPORT JSON ---")
    print(json.dumps(report, indent=2))
