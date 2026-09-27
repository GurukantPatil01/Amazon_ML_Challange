"""Centralized configuration for Amazon ML Challenge 2026 Business Entity Resolution.

Defines all dataset paths, random seeds, validation parameters, blocking settings,
model hyperparameters, threshold configurations, and output paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Base directories
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
TRAIN_DIR: Path = DATASET_DIR / "train"
TEST_DIR: Path = DATASET_DIR / "test"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Ensure runtime directories exist
for directory in (MODELS_DIR, OUTPUT_DIR, EXPERIMENTS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Dataset file paths
TRAIN_SOURCE1_PATH: Path = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH: Path = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH: Path = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH: Path = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1_PATH: Path = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH: Path = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH: Path = TEST_DIR / "test_source3.tsv"

# Submission & Evaluation output file paths
MATCHING_RESULTS_PATH: Path = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH: Path = OUTPUT_DIR / "candidate_pairs.tsv"

# Schema definitions
EXPECTED_SOURCE_COLUMNS: List[str] = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

EXPECTED_GROUND_TRUTH_COLUMNS: List[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

EXPECTED_SUBMISSION_COLUMNS: List[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

EXPECTED_CANDIDATE_COLUMNS: List[str] = [
    "source1_entity_id",
    "candidate_entity_ids",
]


@dataclass(frozen=True)
class PipelineConfig:
    """Configurable pipeline parameters for reproducibility and experimentation."""

    # Reproducibility
    random_seed: int = 42

    # Validation
    val_split_ratio: float = 0.2
    stratify_by_match_status: bool = True

    # Evaluation Metric
    f_beta: float = 0.5  # Macro-F0.5 precision-weighted metric

    # Blocking & Candidate Generation
    max_candidates_per_entity: int = 50
    blocking_ngram_min: int = 2
    blocking_ngram_max: int = 3
    token_min_length: int = 2

    # Classification / Matching Model
    model_type: str = "lightgbm"
    learning_rate: float = 0.05
    n_estimators: int = 300
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Decision Threshold
    decision_threshold: float = 0.5

    # Allowed entity ID prefixes
    source1_prefix: str = "S1-"
    source2_prefix: str = "S2-"
    source3_prefix: str = "S3-"


# Configurable business legal suffixes (sorted by length descending for regex matching)
DEFAULT_LEGAL_SUFFIXES: List[str] = [
    "private limited",
    "pvt limited",
    "pvt ltd",
    "pte ltd",
    "incorporated",
    "corporation",
    "limited liability company",
    "limited liability partnership",
    "sociedad anonima",
    "societe anonyme",
    "societe a responsabilite limitee",
    "enterprise unipersonnelle a responsabilite limitee",
    "limited",
    "company",
    "private",
    "corp",
    "inc",
    "ltd",
    "llc",
    "llp",
    "plc",
    "pvt",
    "pte",
    "sarl",
    "sasu",
    "sas",
    "eurl",
    "sci",
    "snc",
    "gie",
    "gmbh",
    "co",
    "sa",
]

# Standard address abbreviations mapping (word -> standard token)
DEFAULT_ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    # Thoroughfares
    "street": "st",
    "streets": "st",
    "road": "rd",
    "roads": "rd",
    "avenue": "ave",
    "avenues": "ave",
    "boulevard": "blvd",
    "bd": "blvd",
    "bvd": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "square": "sq",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "expressway": "expy",
    "terrace": "ter",
    "way": "way",
    "alley": "aly",
    "route": "rte",
    "chemin": "chem",
    "impasse": "imp",
    "allee": "all",
    # Units / Sub-premises
    "suite": "ste",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "room": "rm",
    "department": "dept",
    "unit": "unit",
    "number": "no",
    # Directions
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
}

# Default active configuration instance
DEFAULT_CONFIG: PipelineConfig = PipelineConfig()

