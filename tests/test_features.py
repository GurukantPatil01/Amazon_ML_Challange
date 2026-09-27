"""Unit Tests for Pairwise Feature Extraction Engine.

Tests:
1. Feature extraction correctness on exact and partial matches
2. Safe missing address handling (never produces similarity=1.0, sets flag)
3. Interaction features and blocking provenance extraction
4. Vectorized batch feature matrix shape and type
"""

import numpy as np
import pytest

from src.pair_features import (
    FEATURE_NAMES,
    extract_batch_features,
    extract_pair_features,
    token_jaccard,
    token_overlap,
)


def test_token_similarity_metrics():
    toks1 = ["starbucks", "coffee", "co"]
    toks2 = ["starbucks", "coffee"]
    
    # Jaccard: 2 / 3 = 0.6667
    jaccard = token_jaccard(toks1, toks2)
    assert abs(jaccard - 2/3) < 1e-4

    # Overlap: 2 / min(3, 2) = 2 / 2 = 1.0
    overlap = token_overlap(toks1, toks2)
    assert abs(overlap - 1.0) < 1e-4


def test_feature_correctness_exact_match():
    s1 = {
        "name_norm": "target pharmacy",
        "name_core": "target pharmacy",
        "name_tokens": "pharmacy target",
        "name_sorted_tokens": "pharmacy target",
        "name_char_3gram": "tar arg rge get pha har arm rma mac acy",
        "address_norm": "100 main st seattle wa",
        "postal_code": "98101",
        "house_number": "100",
    }
    cand = dict(s1)
    cand["candidate_entity_id"] = "S2-12345"

    meta = {
        "blocking_sources": {"exact_name", "name_core", "house_name"},
        "rank": 1,
        "target_source": "source2",
    }

    feats = extract_pair_features(s1, cand, blocking_meta=meta)

    assert feats["feat_exact_name_match"] == 1.0
    assert feats["feat_exact_name_core_match"] == 1.0
    assert feats["feat_exact_address_match"] == 1.0
    assert feats["feat_postal_exact_match"] == 1.0
    assert feats["feat_house_number_exact_match"] == 1.0
    assert feats["feat_name_levenshtein_sim"] == 1.0
    assert feats["feat_address_levenshtein_sim"] == 1.0
    assert feats["feat_both_exact"] == 1.0
    assert feats["feat_address_missing_flag"] == 0.0
    assert feats["feat_block_count"] == 3.0
    assert feats["feat_is_source2"] == 1.0
    assert feats["feat_is_source3"] == 0.0


def test_missing_address_handling():
    """Missing addresses must never be converted to similarity=1.0 or misleading values."""
    s1 = {
        "name_norm": "apple store",
        "name_core": "apple store",
        "name_tokens": "apple store",
        "name_sorted_tokens": "apple store",
        "name_char_3gram": "app ppl ple sto tor ore",
        "address_norm": "",  # Missing
        "postal_code": "",
        "house_number": "",
    }
    cand = {
        "name_norm": "apple store",
        "name_core": "apple store",
        "name_tokens": "apple store",
        "name_sorted_tokens": "apple store",
        "name_char_3gram": "app ppl ple sto tor ore",
        "address_norm": "1 infinite loop cupertino ca",
        "postal_code": "95014",
        "house_number": "1",
    }

    feats = extract_pair_features(s1, cand)

    assert feats["feat_address_missing_flag"] == 1.0
    assert feats["feat_exact_address_match"] == 0.0
    assert feats["feat_address_levenshtein_sim"] == 0.0
    assert feats["feat_address_token_jaccard"] == 0.0
    assert feats["feat_address_token_overlap"] == 0.0
    assert feats["feat_both_exact"] == 0.0


def test_batch_feature_matrix_vectorization():
    s1_lookup = {
        "S1-1": {
            "name_norm": "nike retail",
            "name_core": "nike",
            "name_tokens": "nike retail",
            "name_sorted_tokens": "nike retail",
            "name_char_3gram": "nik ike ret eta tai ail",
            "address_norm": "100 broadway new york ny",
            "postal_code": "10001",
            "house_number": "100",
        }
    }
    target_lookup = {
        "S2-1": {
            "name_norm": "nike factory store",
            "name_core": "nike",
            "name_tokens": "factory nike store",
            "name_sorted_tokens": "factory nike store",
            "name_char_3gram": "nik ike fac act cto tor ory sto tor ore",
            "address_norm": "100 broadway new york ny",
            "postal_code": "10001",
            "house_number": "100",
        }
    }
    pairs = [
        {
            "source1_entity_id": "S1-1",
            "candidate_entity_id": "S2-1",
            "blocking_sources": "name_core,house_name",
            "rank": 1,
            "target_source": "source2",
        }
    ]

    mat = extract_batch_features(pairs, s1_lookup, target_lookup)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (1, len(FEATURE_NAMES))
    assert mat.dtype == np.float32

    # Check key indices
    idx_core = FEATURE_NAMES.index("feat_exact_name_core_match")
    idx_block_count = FEATURE_NAMES.index("feat_block_count")
    assert mat[0, idx_core] == 1.0
    assert mat[0, idx_block_count] == 2.0
