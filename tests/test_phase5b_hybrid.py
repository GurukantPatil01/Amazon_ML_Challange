import pytest
import numpy as np
from src.phase5b_hybrid_matcher import (
    extract_classical_targeted_features,
    CLASSICAL_FEATURE_NAMES,
    EMBEDDING_FEATURE_NAMES,
    HYBRID_FEATURE_NAMES,
)


def test_feature_name_definitions():
    assert len(CLASSICAL_FEATURE_NAMES) == 46
    assert len(EMBEDDING_FEATURE_NAMES) == 8
    assert len(HYBRID_FEATURE_NAMES) == 54
    assert "feat_emb_name_cosine" in HYBRID_FEATURE_NAMES
    assert "feat_emb_name_address_cosine" in HYBRID_FEATURE_NAMES
    assert "feat_has_house_token" in HYBRID_FEATURE_NAMES


def test_extract_classical_targeted_features():
    s1_row = {"business_name": "Acme Tools", "business_address": "100 Industrial Pkwy"}
    target_row = {"business_name": "Acme Tools Inc", "business_address": "100 Industrial Parkway"}
    cand_row = {"blocking_sources": "exact_name,house_token,name_web_norm"}

    feats = extract_classical_targeted_features(s1_row, target_row, cand_row)
    assert len(feats) == 3
    # has_house_token
    assert feats[0] == 1.0
    # has_web_norm
    assert feats[1] == 1.0
    # indep_count (name, house_postal, web = 3)
    assert feats[2] == 3.0
