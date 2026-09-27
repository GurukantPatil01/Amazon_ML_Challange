"""Phase 4 Error-Mining and Diagnostic Engine for Entity Resolution.

Generates comprehensive diagnostic datasets:
1. candidate_misses.tsv: True pairs completely missed by Phase 3 candidate generation (Union E K=200).
2. model_misses.tsv: True candidate pairs accepted by blocking but rejected by LightGBM at tau=0.45.
3. false_positives.tsv: Non-matching candidate pairs accepted by LightGBM at tau=0.45.
4. candidate_miss_summary.json & candidate_miss_summary.md: Diagnostic summaries classifying errors.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.build_training_pairs import (
    STRATEGY_PRIORITY_WEIGHTS,
    UNION_E_STRATEGIES,
    build_labeled_pairs,
    generate_candidates,
    split_s1_groups,
)
from src.config import (
    EXPERIMENTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
)
from src.decision import predict_matches
from src.normalize import normalize_dataframe
from src.pair_features import (
    FEATURE_NAMES,
    extract_batch_features,
    extract_pair_features,
    token_jaccard,
    token_overlap,
)
from src.utils import setup_logger, timer

logger = setup_logger("phase4_error_mining", log_file=LOGS_DIR / "phase4_error_mining.log")

PHASE4_DIR = EXPERIMENTS_DIR / "phase4"


def run_error_mining(
    sample_queries: int = 15_000,
    target_pool_size: int = 250_000,
    candidate_budget: int = 200,
    threshold: float = 0.45,
) -> Dict[str, Any]:
    """Extracts, categorizes, and saves all Phase 3 errors for Phase 4 analysis."""
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("PHASE 4: ERROR MINING AND CANDIDATE MISS DIAGNOSTICS")
    logger.info("=" * 70)

    # 1. Load Ground Truth
    with timer("Loading Ground Truth", logger):
        gt_df = pd.read_csv(TRAIN_GROUND_TRUTH_PATH, sep="\t", keep_default_na=False)
        gt_dict: Dict[str, Set[str]] = {}
        for s1_id, m_str in zip(gt_df["source1_entity_id"].tolist(), gt_df["matched_entity_ids"].tolist()):
            gt_dict[s1_id] = set(m_str.split(",")) if m_str.strip() else set()

    # 2. Load and Normalize Benchmark Subsets
    with timer("Loading and Normalizing Benchmark Records", logger):
        s1_raw = pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t", nrows=sample_queries, dtype=str, keep_default_na=False)
        s1_norm = normalize_dataframe(s1_raw)
        s1_lookup = {r["entity_id"]: r for r in s1_norm.to_dict(orient="records")}
        s1_ids = s1_norm["entity_id"].tolist()

        s2_raw = pd.read_csv(TRAIN_SOURCE2_PATH, sep="\t", nrows=target_pool_size, dtype=str, keep_default_na=False)
        s2_norm = normalize_dataframe(s2_raw)
        s2_lookup = {r["entity_id"]: r for r in s2_norm.to_dict(orient="records")}
        s2_ids_set = set(s2_lookup.keys())

        s3_raw = pd.read_csv(TRAIN_SOURCE3_PATH, sep="\t", nrows=target_pool_size, dtype=str, keep_default_na=False)
        s3_norm = normalize_dataframe(s3_raw)
        s3_lookup = {r["entity_id"]: r for r in s3_norm.to_dict(orient="records")}
        s3_ids_set = set(s3_lookup.keys())

        target_lookup = {**s2_lookup, **s3_lookup}

    # Filter GT to indexed targets
    scoped_gt: Dict[str, Set[str]] = {}
    for s1_id in s1_ids:
        all_true = gt_dict.get(s1_id, set())
        scoped_gt[s1_id] = {m for m in all_true if (m in s2_ids_set or m in s3_ids_set)}

    # Group Split (Must match Phase 3 exactly: seed=42, ratio=0.80)
    train_s1_set, val_s1_set = split_s1_groups(s1_ids, train_ratio=0.80, random_seed=42)
    val_gt = {s1: scoped_gt[s1] for s1 in val_s1_set}

    # 3. Generate Candidates (Union E K=200)
    with timer("Generating Candidates (Union E K=200)", logger):
        cand_df = generate_candidates(s1_norm, s2_norm, s3_norm, budget=candidate_budget)

    val_cand_df = cand_df[cand_df["source1_entity_id"].isin(val_s1_set)].copy()

    val_cand_map: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in val_cand_df.to_dict(orient="records"):
        val_cand_map[row["source1_entity_id"]][row["candidate_entity_id"]] = row

    val_cand_dict: Dict[str, Set[str]] = {
        s1: set(c_dict.keys()) for s1, c_dict in val_cand_map.items()
    }

    # 4. Load Saved LightGBM Model & Predict on Validation Set
    model_path = MODELS_DIR / "lightgbm_baseline.txt"
    logger.info(f"Loading LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=str(model_path))

    val_pairs_df = build_labeled_pairs(
        val_cand_df,
        s1_lookup,
        target_lookup,
        scoped_gt,
        max_negatives_per_s1=24,
        random_seed=123,
    )
    val_pairs_list = val_pairs_df.to_dict(orient="records")
    X_val = extract_batch_features(val_pairs_list, s1_lookup, target_lookup)
    y_val_probs = model.predict(X_val, num_iteration=model.best_iteration)

    val_scored_dict: Dict[Tuple[str, str], float] = {}
    val_cand_scored: List[Dict[str, Any]] = []
    for i, row in enumerate(val_pairs_list):
        score = float(y_val_probs[i])
        val_scored_dict[(row["source1_entity_id"], row["candidate_entity_id"])] = score
        val_cand_scored.append({
            "source1_entity_id": row["source1_entity_id"],
            "candidate_entity_id": row["candidate_entity_id"],
            "score": score,
        })

    # Predict at threshold 0.45
    preds_045 = predict_matches(val_cand_scored, threshold=threshold)

    # =======================================================================
    # DATASET A: CANDIDATE MISSES (GT positive pairs missed by blocking)
    # =======================================================================
    logger.info("Extracting and classifying candidate misses (GT true pairs absent from Union E)...")
    candidate_misses: List[Dict[str, Any]] = []

    for s1_id in val_s1_set:
        true_set = val_gt.get(s1_id, set())
        if not true_set:
            continue
        cands = val_cand_dict.get(s1_id, set())

        for cid in true_set:
            if cid not in cands:
                s1_rec = s1_lookup[s1_id]
                cand_rec = target_lookup[cid]
                target_src = "source2" if "S2" in cid else "source3"

                name1 = s1_rec.get("name_norm", "")
                name2 = cand_rec.get("name_norm", "")
                core1 = s1_rec.get("name_core", "")
                core2 = cand_rec.get("name_core", "")
                toks1 = s1_rec.get("name_tokens", "").split()
                toks2 = cand_rec.get("name_tokens", "").split()
                addr1 = s1_rec.get("address_norm", "")
                addr2 = cand_rec.get("address_norm", "")
                post1 = s1_rec.get("postal_code", "")
                post2 = cand_rec.get("postal_code", "")
                house1 = s1_rec.get("house_number", "")
                house2 = cand_rec.get("house_number", "")

                # Detailed Name Signals
                exact_name = 1 if (name1 and name1 == name2) else 0
                exact_core = 1 if (core1 and core1 == core2) else 0
                jaccard_name = token_jaccard(toks1, toks2)
                overlap_name = token_overlap(toks1, toks2)
                lev_name = float(Levenshtein.normalized_similarity(name1, name2)) if (name1 and name2) else 0.0
                jw_name = float(JaroWinkler.similarity(name1, name2)) if (name1 and name2) else 0.0
                sorted_sim = float(Levenshtein.normalized_similarity(
                    s1_rec.get("name_sorted_tokens", ""), cand_rec.get("name_sorted_tokens", "")
                )) if (s1_rec.get("name_sorted_tokens", "") and cand_rec.get("name_sorted_tokens", "")) else 0.0
                gram_sim = token_jaccard(s1_rec.get("name_char_3gram", "").split(), cand_rec.get("name_char_3gram", "").split())

                first_tok_match = 1 if (toks1 and toks2 and toks1[0] == toks2[0]) else 0
                last_tok_match = 1 if (toks1 and toks2 and toks1[-1] == toks2[-1]) else 0
                initials1 = "".join([t[0] for t in toks1 if t])
                initials2 = "".join([t[0] for t in toks2 if t])
                same_initials = 1 if (initials1 and initials1 == initials2) else 0
                same_tok_count = 1 if len(toks1) == len(toks2) else 0

                # Detailed Address Signals
                exact_addr = 1 if (addr1 and addr1 == addr2) else 0
                addr_missing = 1 if (not addr1 or not addr2) else 0
                addr_toks1 = addr1.split()
                addr_toks2 = addr2.split()
                jaccard_addr = token_jaccard(addr_toks1, addr_toks2) if not addr_missing else 0.0
                overlap_addr = token_overlap(addr_toks1, addr_toks2) if not addr_missing else 0.0
                lev_addr = float(Levenshtein.normalized_similarity(addr1, addr2)) if not addr_missing else 0.0
                postal_match = 1 if (post1 and post2 and post1 == post2) else 0
                house_match = 1 if (house1 and house2 and house1 == house2) else 0

                nums1 = set(re.findall(r"\b\d+\b", addr1))
                nums2 = set(re.findall(r"\b\d+\b", addr2))
                num_overlap = (len(nums1.intersection(nums2)) / min(len(nums1), len(nums2))) if (nums1 and nums2) else 0.0

                # Diagnose Reason
                reasons = []
                if addr_missing:
                    reasons.append("Missing Address in Target or Query")
                if lev_name < 0.40:
                    reasons.append("Extreme Name Dissimilarity / Alias / DBA")
                elif lev_name >= 0.70:
                    reasons.append("High Name Similarity but token variation / legal truncation")
                if house_match and not first_tok_match:
                    reasons.append("Same Building, Divergent Name Prefix")
                if same_initials and lev_name < 0.70:
                    reasons.append("Abbreviated Acronym / Initialism")
                if not reasons:
                    reasons.append("General Tail / Multi-Token Divergence")

                candidate_misses.append({
                    "s1_id": s1_id,
                    "target_source": target_src,
                    "s2_or_s3_id": cid,
                    "country": s1_rec.get("country_norm", ""),
                    "business_name_s1": s1_rec.get("business_name", ""),
                    "business_name_target": cand_rec.get("business_name", ""),
                    "address_s1": s1_rec.get("business_address", ""),
                    "address_target": cand_rec.get("business_address", ""),
                    "exact_name_match": exact_name,
                    "exact_name_core_match": exact_core,
                    "name_levenshtein_sim": round(lev_name, 4),
                    "name_jaro_winkler": round(jw_name, 4),
                    "name_token_jaccard": round(jaccard_name, 4),
                    "name_token_overlap": round(overlap_name, 4),
                    "name_sorted_token_sim": round(sorted_sim, 4),
                    "name_char_3gram_sim": round(gram_sim, 4),
                    "first_token_match": first_tok_match,
                    "last_token_match": last_tok_match,
                    "same_initials": same_initials,
                    "same_token_count": same_tok_count,
                    "exact_address_match": exact_addr,
                    "address_missing": addr_missing,
                    "address_levenshtein_sim": round(lev_addr, 4),
                    "address_token_jaccard": round(jaccard_addr, 4),
                    "address_token_overlap": round(overlap_addr, 4),
                    "postal_code_match": postal_match,
                    "house_number_match": house_match,
                    "numeric_token_overlap": round(num_overlap, 4),
                    "miss_reason_candidates": " | ".join(reasons),
                })

    df_cand_misses = pd.DataFrame(candidate_misses)
    cand_miss_path = PHASE4_DIR / "candidate_misses.tsv"
    df_cand_misses.to_csv(cand_miss_path, sep="\t", index=False)
    logger.info(f"Saved {len(df_cand_misses):,} candidate misses to {cand_miss_path}")

    # =======================================================================
    # DATASET B: MODEL MISSES (Candidate hits rejected by LightGBM at tau=0.45)
    # =======================================================================
    logger.info("Extracting model misses (True candidate hits scored below threshold 0.45)...")
    model_misses: List[Dict[str, Any]] = []

    for s1_id in val_s1_set:
        true_set = val_gt.get(s1_id, set())
        if not true_set:
            continue
        cands = val_cand_dict.get(s1_id, set())
        preds = preds_045.get(s1_id, set())

        for cid in true_set:
            if cid in cands and cid not in preds:
                s1_rec = s1_lookup[s1_id]
                cand_rec = target_lookup[cid]
                b_row = val_cand_map[s1_id][cid]
                target_src = b_row["candidate_source"]
                score = val_scored_dict.get((s1_id, cid), 0.0)

                feat_dict = extract_pair_features(s1_rec, cand_rec, blocking_meta=b_row)

                entry = {
                    "s1_id": s1_id,
                    "target_source": target_src,
                    "target_id": cid,
                    "model_score": round(score, 6),
                    "candidate_rank": b_row.get("rank", 999),
                    "block_count": b_row.get("block_count", 1),
                    "blocking_sources": b_row.get("blocking_sources", ""),
                    "country": s1_rec.get("country_norm", ""),
                    "business_name_s1": s1_rec.get("business_name", ""),
                    "business_name_target": cand_rec.get("business_name", ""),
                    "address_s1": s1_rec.get("business_address", ""),
                    "address_target": cand_rec.get("business_address", ""),
                    "true_match": 1,
                }
                # Attach all features
                for fn in FEATURE_NAMES:
                    entry[fn] = feat_dict.get(fn, 0.0)

                model_misses.append(entry)

    df_model_misses = pd.DataFrame(model_misses)
    if not df_model_misses.empty:
        df_model_misses = df_model_misses.sort_values("model_score", ascending=False).reset_index(drop=True)
    model_miss_path = PHASE4_DIR / "model_misses.tsv"
    df_model_misses.to_csv(model_miss_path, sep="\t", index=False)
    logger.info(f"Saved {len(df_model_misses):,} model misses to {model_miss_path}")

    # =======================================================================
    # DATASET C: FALSE POSITIVES (Non-matching pairs accepted by LightGBM at tau=0.45)
    # =======================================================================
    logger.info("Extracting false positives (Candidate pairs scored >= 0.45 but not true matches)...")
    false_positives: List[Dict[str, Any]] = []

    for s1_id, pred_set in preds_045.items():
        true_set = val_gt.get(s1_id, set())
        is_singleton = len(true_set) == 0

        for cid in pred_set:
            if cid not in true_set:
                s1_rec = s1_lookup.get(s1_id, {})
                cand_rec = target_lookup.get(cid, {})
                b_row = val_cand_map.get(s1_id, {}).get(cid, {})
                target_src = "source2" if "S2" in cid else "source3"
                score = val_scored_dict.get((s1_id, cid), 0.0)

                feat_dict = extract_pair_features(s1_rec, cand_rec, blocking_meta=b_row)

                # Classify reason
                name1 = s1_rec.get("name_norm", "")
                name2 = cand_rec.get("name_norm", "")
                core1 = s1_rec.get("name_core", "")
                core2 = cand_rec.get("name_core", "")
                addr1 = s1_rec.get("address_norm", "")
                addr2 = cand_rec.get("address_norm", "")
                house1 = s1_rec.get("house_number", "")
                house2 = cand_rec.get("house_number", "")

                fp_type = "Singleton False Positive" if is_singleton else "Non-Singleton False Positive"
                reason = "General Multi-Feature Coincidence"
                if name1 and name1 == name2:
                    reason = "Identical Business Name, Different Entity"
                elif core1 and core1 == core2:
                    reason = "Identical Core Name, Different Entity"
                elif addr1 and addr1 == addr2:
                    reason = "Identical Address (Co-located / Shared Complex), Different Entity"
                elif house1 and house1 == house2 and Levenshtein.normalized_similarity(name1, name2) >= 0.70:
                    reason = "Same Building Number + High Name Similarity"

                entry = {
                    "s1_id": s1_id,
                    "target_id": cid,
                    "target_source": target_src,
                    "model_score": round(score, 6),
                    "candidate_rank": b_row.get("rank", 999),
                    "block_count": b_row.get("block_count", 1),
                    "blocking_sources": b_row.get("blocking_sources", ""),
                    "country": s1_rec.get("country_norm", ""),
                    "fp_category": fp_type,
                    "classifiable_reason": reason,
                    "business_name_s1": s1_rec.get("business_name", ""),
                    "business_name_target": cand_rec.get("business_name", ""),
                    "address_s1": s1_rec.get("business_address", ""),
                    "address_target": cand_rec.get("business_address", ""),
                }
                for fn in FEATURE_NAMES:
                    entry[fn] = feat_dict.get(fn, 0.0)

                false_positives.append(entry)

    df_fp = pd.DataFrame(false_positives)
    if not df_fp.empty:
        df_fp = df_fp.sort_values("model_score", ascending=False).reset_index(drop=True)
    fp_path = PHASE4_DIR / "false_positives.tsv"
    df_fp.to_csv(fp_path, sep="\t", index=False)
    logger.info(f"Saved {len(df_fp):,} false positives to {fp_path}")

    # =======================================================================
    # DATASET D: CANDIDATE MISS SUMMARY (JSON & Markdown)
    # =======================================================================
    logger.info("Computing candidate miss statistical profiles and summaries...")
    total_misses = len(df_cand_misses)

    summary_payload = {
        "total_ground_truth_pairs": sum(len(v) for v in val_gt.values()),
        "total_candidate_misses": total_misses,
        "candidate_recall_lost_pct": round(total_misses / sum(len(v) for v in val_gt.values()) * 100, 2),
        "source_breakdown": {
            "source2": int((df_cand_misses["target_source"] == "source2").sum()),
            "source3": int((df_cand_misses["target_source"] == "source3").sum()),
        },
        "country_breakdown": df_cand_misses["country"].value_counts().to_dict(),
        "name_similarity_distribution": {
            "exact_normalized_name": int((df_cand_misses["exact_name_match"] == 1).sum()),
            "exact_core_name": int((df_cand_misses["exact_name_core_match"] == 1).sum()),
            "high_name_similarity_ge_070": int((df_cand_misses["name_levenshtein_sim"] >= 0.70).sum()),
            "moderate_name_similarity_040_to_070": int(((df_cand_misses["name_levenshtein_sim"] >= 0.40) & (df_cand_misses["name_levenshtein_sim"] < 0.70)).sum()),
            "low_name_similarity_lt_040": int((df_cand_misses["name_levenshtein_sim"] < 0.40).sum()),
            "first_token_match": int((df_cand_misses["first_token_match"] == 1).sum()),
            "same_initials": int((df_cand_misses["same_initials"] == 1).sum()),
        },
        "address_similarity_distribution": {
            "exact_address_match": int((df_cand_misses["exact_address_match"] == 1).sum()),
            "missing_address": int((df_cand_misses["address_missing"] == 1).sum()),
            "house_number_match": int((df_cand_misses["house_number_match"] == 1).sum()),
            "postal_code_match": int((df_cand_misses["postal_code_match"] == 1).sum()),
            "high_address_overlap_ge_050": int((df_cand_misses["address_token_overlap"] >= 0.50).sum()),
        },
        "model_misses_count": len(df_model_misses),
        "false_positives_count": len(df_fp),
    }

    sum_json_path = PHASE4_DIR / "candidate_miss_summary.json"
    with open(sum_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Format Markdown
    md = []
    md.append("# Phase 4 Candidate Miss Diagnostic Summary")
    md.append("")
    md.append(f"- **Total Ground-Truth Positive Pairs in Validation Split:** {summary_payload['total_ground_truth_pairs']:,}")
    md.append(f"- **Total Pairs Missed at Candidate Generation:** **{total_misses} pairs** ({summary_payload['candidate_recall_lost_pct']}% of ground truth)")
    md.append(f"- **Target Source Breakdown:** Source 2 = {summary_payload['source_breakdown']['source2']} | Source 3 = {summary_payload['source_breakdown']['source3']}")
    md.append(f"- **Geographic Distribution:** {summary_payload['country_breakdown']}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Primary Name-Signal Characteristics of Missed Pairs")
    md.append("")
    ns = summary_payload["name_similarity_distribution"]
    md.append("| Name Characteristic | Missed Pair Count | Percentage of Misses (%) | Opportunity for Recovery |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| High Name Similarity (Levenshtein >= 0.70) | **{ns['high_name_similarity_ge_070']}** | {ns['high_name_similarity_ge_070']/total_misses*100:.1f}% | Recoverable via higher char-ngram depth or adaptive char retrieval |")
    md.append(f"| Moderate Similarity (0.40 <= Levenshtein < 0.70) | **{ns['moderate_name_similarity_040_to_070']}** | {ns['moderate_name_similarity_040_to_070']/total_misses*100:.1f}% | Word-prefix, initials, or token overlap combinations |")
    md.append(f"| Low Similarity (Levenshtein < 0.40) | **{ns['low_name_similarity_lt_040']}** | {ns['low_name_similarity_lt_040']/total_misses*100:.1f}% | Extreme brand renames; requires address-assisted or house-name variants |")
    md.append(f"| Same First Token | **{ns['first_token_match']}** | {ns['first_token_match']/total_misses*100:.1f}% | Word prefix / first token signature |")
    md.append(f"| Same Initials / Acronym | **{ns['same_initials']}** | {ns['same_initials']/total_misses*100:.1f}% | Company acronym / initial signature |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Address-Signal Characteristics of Missed Pairs")
    md.append("")
    addr_s = summary_payload["address_similarity_distribution"]
    md.append("| Address Characteristic | Missed Pair Count | Percentage of Misses (%) | Opportunity for Recovery |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| Same House/Street Number | **{addr_s['house_number_match']}** | {addr_s['house_number_match']/total_misses*100:.1f}% | House number + single word / initial block |")
    md.append(f"| High Address Token Overlap ($\\ge 0.50$) | **{addr_s['high_address_overlap_ge_050']}** | {addr_s['high_address_overlap_ge_050']/total_misses*100:.1f}% | Co-located street / address-assisted tokens |")
    md.append(f"| Missing Address in Query/Target | **{addr_s['missing_address']}** | {addr_s['missing_address']/total_misses*100:.1f}% | Pure name matching required (cannot use address blocks) |")
    md.append(f"| Postal Code Match | **{addr_s['postal_code_match']}** | {addr_s['postal_code_match']/total_misses*100:.1f}% | Postal prefix combinations |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Representative Candidate Miss Examples")
    md.append("")
    md.append("| S1 Query Name | Target Match Name | S1 Address | Target Address | Levenshtein Sim | Primary Failure Reason |")
    md.append("| :--- | :--- | :--- | :--- | :---: | :--- |")
    for row in candidate_misses[:10]:
        md.append(f"| `{row['business_name_s1'][:30]}` | `{row['business_name_target'][:30]}` | `{row['address_s1'][:30]}` | `{row['address_target'][:30]}` | {row['name_levenshtein_sim']:.2f} | {row['miss_reason_candidates']} |")
    md.append("")

    sum_md_path = PHASE4_DIR / "candidate_miss_summary.md"
    with open(sum_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    logger.info(f"Saved diagnostic summary to {sum_json_path} and {sum_md_path}")
    return summary_payload


if __name__ == "__main__":
    run_error_mining()
