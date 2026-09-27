"""High-Performance Pairwise Feature Extraction Engine for Entity Resolution.

Extracts comprehensive, memory-efficient feature vectors comparing Source 1 records
with candidate Source 2 / Source 3 records:
1. Name Similarity Features (Levenshtein, Jaro-Winkler, Jaccard, Overlap, Char 3-gram, Length/Token diffs)
2. Address Similarity Features (Levenshtein, Jaccard, Overlap, Postal, House number, Missing-safe)
3. Combined Interaction Features (Multiplications, min/max, joint exact flags)
4. Blocking Provenance Features (Multi-block support count, sum strength, individual block flags, rank, source)
"""

from collections import Counter
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein


# Predefined feature names in deterministic order
FEATURE_NAMES: List[str] = [
    # Name Features
    "feat_exact_name_match",
    "feat_exact_name_core_match",
    "feat_name_jaro_winkler",
    "feat_name_levenshtein_sim",
    "feat_name_token_jaccard",
    "feat_name_token_overlap",
    "feat_name_sorted_token_sim",
    "feat_name_char_3gram_jaccard",
    "feat_name_len_diff",
    "feat_name_len_ratio",
    "feat_name_token_count_diff",
    "feat_name_first_token_match",
    "feat_name_last_token_match",
    
    # Address Features
    "feat_exact_address_match",
    "feat_address_levenshtein_sim",
    "feat_address_token_jaccard",
    "feat_address_token_overlap",
    "feat_postal_exact_match",
    "feat_house_number_exact_match",
    "feat_house_number_overlap",
    "feat_numeric_token_overlap",
    "feat_address_len_diff",
    "feat_address_token_count_diff",
    "feat_address_missing_flag",
    
    # Combined / Interaction Features
    "feat_name_sim_x_addr_sim",
    "feat_name_sim_plus_addr_sim",
    "feat_max_name_addr_sim",
    "feat_min_name_addr_sim",
    "feat_both_exact",
    "feat_name_exact_and_addr_similar",
    "feat_addr_exact_and_name_similar",
    
    # Blocking Metadata Features
    "feat_block_count",
    "feat_block_strength_sum",
    "feat_exact_name_hit",
    "feat_name_core_hit",
    "feat_house_name_hit",
    "feat_combined_postal_name_hit",
    "feat_rare_token_hit",
    "feat_char_3gram_hit",
    "feat_address_token_hit",
    "feat_candidate_rank",
    "feat_is_source2",
    "feat_is_source3",
]


def token_jaccard(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Calculates Jaccard similarity between two token sequences."""
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 or not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0.0


def token_overlap(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Calculates Szymkiewicz-Simpson overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 or not s2:
        return 0.0
    min_len = min(len(s1), len(s2))
    if min_len == 0:
        return 0.0
    return len(s1.intersection(s2)) / min_len


def extract_pair_features(
    s1_row: Dict[str, Any],
    cand_row: Dict[str, Any],
    blocking_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Extracts a complete dictionary of pairwise features between S1 and candidate.

    Handles missing values safely: missing address yields similarity = 0.0 and sets
    feat_address_missing_flag = 1.0.

    Args:
        s1_row: Normalized Source 1 record dictionary.
        cand_row: Normalized candidate record dictionary.
        blocking_meta: Optional dictionary containing blocking provenance metadata:
                       {'blocking_sources': set/list, 'rank': int, 'target_source': str}

    Returns:
        Dictionary mapping feature names to float values.
    """
    feats: Dict[str, float] = {}
    
    # -------------------------------------------------------------
    # 1. NAME FEATURES
    # -------------------------------------------------------------
    name1 = s1_row.get("name_norm", "")
    name2 = cand_row.get("name_norm", "")
    core1 = s1_row.get("name_core", "")
    core2 = cand_row.get("name_core", "")
    
    feats["feat_exact_name_match"] = 1.0 if (name1 and name1 == name2) else 0.0
    feats["feat_exact_name_core_match"] = 1.0 if (core1 and core1 == core2) else 0.0
    
    # Rapidfuzz normalized string distances (0.0 to 1.0)
    if name1 and name2:
        feats["feat_name_jaro_winkler"] = float(JaroWinkler.similarity(name1, name2))
        feats["feat_name_levenshtein_sim"] = float(Levenshtein.normalized_similarity(name1, name2))
    else:
        feats["feat_name_jaro_winkler"] = 0.0
        feats["feat_name_levenshtein_sim"] = 0.0
        
    toks1 = s1_row.get("name_tokens", "").split()
    toks2 = cand_row.get("name_tokens", "").split()
    
    feats["feat_name_token_jaccard"] = token_jaccard(toks1, toks2)
    feats["feat_name_token_overlap"] = token_overlap(toks1, toks2)
    
    sorted1 = s1_row.get("name_sorted_tokens", "")
    sorted2 = cand_row.get("name_sorted_tokens", "")
    if sorted1 and sorted2:
        feats["feat_name_sorted_token_sim"] = float(Levenshtein.normalized_similarity(sorted1, sorted2))
    else:
        feats["feat_name_sorted_token_sim"] = 0.0
        
    grams1 = s1_row.get("name_char_3gram", "").split()
    grams2 = cand_row.get("name_char_3gram", "").split()
    feats["feat_name_char_3gram_jaccard"] = token_jaccard(grams1, grams2)
    
    feats["feat_name_len_diff"] = float(abs(len(name1) - len(name2)))
    max_len = max(len(name1), len(name2))
    feats["feat_name_len_ratio"] = (min(len(name1), len(name2)) / max_len) if max_len > 0 else 1.0
    feats["feat_name_token_count_diff"] = float(abs(len(toks1) - len(toks2)))
    
    feats["feat_name_first_token_match"] = 1.0 if (toks1 and toks2 and toks1[0] == toks2[0]) else 0.0
    feats["feat_name_last_token_match"] = 1.0 if (toks1 and toks2 and toks1[-1] == toks2[-1]) else 0.0
    
    # -------------------------------------------------------------
    # 2. ADDRESS FEATURES
    # -------------------------------------------------------------
    addr1 = s1_row.get("address_norm", "")
    addr2 = cand_row.get("address_norm", "")
    addr_missing = (not addr1) or (not addr2)
    feats["feat_address_missing_flag"] = 1.0 if addr_missing else 0.0
    
    if addr_missing:
        feats["feat_exact_address_match"] = 0.0
        feats["feat_address_levenshtein_sim"] = 0.0
        feats["feat_address_token_jaccard"] = 0.0
        feats["feat_address_token_overlap"] = 0.0
        feats["feat_address_len_diff"] = float(abs(len(addr1) - len(addr2)))
        feats["feat_address_token_count_diff"] = 0.0
    else:
        feats["feat_exact_address_match"] = 1.0 if addr1 == addr2 else 0.0
        feats["feat_address_levenshtein_sim"] = float(Levenshtein.normalized_similarity(addr1, addr2))
        addr_toks1 = addr1.split()
        addr_toks2 = addr2.split()
        feats["feat_address_token_jaccard"] = token_jaccard(addr_toks1, addr_toks2)
        feats["feat_address_token_overlap"] = token_overlap(addr_toks1, addr_toks2)
        feats["feat_address_len_diff"] = float(abs(len(addr1) - len(addr2)))
        feats["feat_address_token_count_diff"] = float(abs(len(addr_toks1) - len(addr_toks2)))
        
    post1 = s1_row.get("postal_code", "")
    post2 = cand_row.get("postal_code", "")
    feats["feat_postal_exact_match"] = 1.0 if (post1 and post2 and post1 == post2) else 0.0
    
    house1 = s1_row.get("house_number", "")
    house2 = cand_row.get("house_number", "")
    feats["feat_house_number_exact_match"] = 1.0 if (house1 and house2 and house1 == house2) else 0.0
    
    # House number overlap: check if house1 appears anywhere in addr2 or vice versa
    if house1 and addr2 and (house1 in addr2.split()):
        feats["feat_house_number_overlap"] = 1.0
    elif house2 and addr1 and (house2 in addr1.split()):
        feats["feat_house_number_overlap"] = 1.0
    else:
        feats["feat_house_number_overlap"] = 0.0
        
    nums1 = set(re.findall(r"\b\d+\b", addr1))
    nums2 = set(re.findall(r"\b\d+\b", addr2))
    feats["feat_numeric_token_overlap"] = (
        len(nums1.intersection(nums2)) / min(len(nums1), len(nums2))
        if (nums1 and nums2)
        else 0.0
    )
    
    # -------------------------------------------------------------
    # 3. COMBINED / INTERACTION FEATURES
    # -------------------------------------------------------------
    name_sim = feats["feat_name_levenshtein_sim"]
    addr_sim = feats["feat_address_levenshtein_sim"]
    
    feats["feat_name_sim_x_addr_sim"] = round(name_sim * addr_sim, 6)
    feats["feat_name_sim_plus_addr_sim"] = round(name_sim + addr_sim, 6)
    feats["feat_max_name_addr_sim"] = max(name_sim, addr_sim)
    feats["feat_min_name_addr_sim"] = min(name_sim, addr_sim)
    
    feats["feat_both_exact"] = 1.0 if (feats["feat_exact_name_match"] == 1.0 and feats["feat_exact_address_match"] == 1.0) else 0.0
    feats["feat_name_exact_and_addr_similar"] = 1.0 if (feats["feat_exact_name_match"] == 1.0 and addr_sim >= 0.70) else 0.0
    feats["feat_addr_exact_and_name_similar"] = 1.0 if (feats["feat_exact_address_match"] == 1.0 and name_sim >= 0.70) else 0.0
    
    # -------------------------------------------------------------
    # 4. BLOCKING PROVENANCE METADATA FEATURES
    # -------------------------------------------------------------
    b_meta = blocking_meta or {}
    sources: Set[str] = set()
    raw_sources = b_meta.get("blocking_sources", set())
    if isinstance(raw_sources, str):
        sources = set(raw_sources.split(","))
    elif isinstance(raw_sources, (set, list)):
        sources = set(raw_sources)
        
    feats["feat_block_count"] = float(len(sources))
    
    # Priority mapping for strength sum
    strength_map = {
        "exact_name": 7.0,
        "name_core": 6.0,
        "house_name": 5.0,
        "combined_postal_name": 4.0,
        "rare_token": 3.0,
        "char_3gram_k10": 2.0,
        "address_token": 1.0,
    }
    feats["feat_block_strength_sum"] = sum(strength_map.get(s, 1.0) for s in sources)
    
    feats["feat_exact_name_hit"] = 1.0 if "exact_name" in sources else 0.0
    feats["feat_name_core_hit"] = 1.0 if "name_core" in sources else 0.0
    feats["feat_house_name_hit"] = 1.0 if "house_name" in sources else 0.0
    feats["feat_combined_postal_name_hit"] = 1.0 if "combined_postal_name" in sources else 0.0
    feats["feat_rare_token_hit"] = 1.0 if "rare_token" in sources else 0.0
    feats["feat_char_3gram_hit"] = 1.0 if any("char_3gram" in s for s in sources) else 0.0
    feats["feat_address_token_hit"] = 1.0 if "address_token" in sources else 0.0
    
    feats["feat_candidate_rank"] = float(b_meta.get("rank", 1.0))
    target_src = b_meta.get("target_source", "")
    if not target_src and "candidate_entity_id" in cand_row:
        cid = cand_row["candidate_entity_id"]
        target_src = "source2" if "S2" in cid else "source3"
    feats["feat_is_source2"] = 1.0 if target_src == "source2" else 0.0
    feats["feat_is_source3"] = 1.0 if target_src == "source3" else 0.0
    
    return feats


def extract_batch_features(
    pairs: Sequence[Dict[str, Any]],
    s1_lookup: Dict[str, Dict[str, Any]],
    target_lookup: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """Vectorized batch feature extractor transforming candidate pairs into a 2D float32 array.

    Args:
        pairs: List of candidate pair records, each containing 'source1_entity_id',
               'candidate_entity_id', and optional blocking metadata.
        s1_lookup: Dict mapping source1_entity_id -> normalized record dict.
        target_lookup: Dict mapping candidate_entity_id -> normalized record dict.

    Returns:
        2D numpy array of shape (len(pairs), len(FEATURE_NAMES)) in float32.
    """
    n_pairs = len(pairs)
    n_features = len(FEATURE_NAMES)
    matrix = np.zeros((n_pairs, n_features), dtype=np.float32)
    
    for i, pair in enumerate(pairs):
        s1_id = pair["source1_entity_id"]
        cid = pair["candidate_entity_id"]
        
        s1_rec = s1_lookup.get(s1_id, {})
        cand_rec = target_lookup.get(cid, {})
        
        feat_dict = extract_pair_features(s1_rec, cand_rec, blocking_meta=pair)
        for j, feat_name in enumerate(FEATURE_NAMES):
            matrix[i, j] = feat_dict.get(feat_name, 0.0)
            
    return matrix
