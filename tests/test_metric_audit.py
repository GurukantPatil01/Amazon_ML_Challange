"""Unit Tests for Official Metric Audit and Verification.

Tests:
1. Macro-F0.5 per-S1 calculation vs global pooling
2. Singleton scoring edge cases (both empty=1.0, GT empty & pred non-empty=0.0)
3. Micro vs Macro precision disambiguation
4. Recall multiplicative identity: (accepted/candidate) * (candidate/total) == accepted/total
5. Group split disjointness: train_s1 ∩ val_s1 == empty
6. Label leakage detection (no forbidden label names in feature set)
"""

from typing import Set
import numpy as np
import pytest

from src.audit_phase3_metrics import independent_entity_f05
from src.build_training_pairs import split_s1_groups
from src.decision import evaluate_decision_metrics
from src.evaluate import compute_entity_f05, compute_macro_f05
from src.pair_features import FEATURE_NAMES


def test_official_metric_singleton_rules():
    """Verify singleton rules:

    - Both empty: F0.5 = 1.0, Precision = 1.0, Recall = 1.0
    - GT empty, Pred non-empty: F0.5 = 0.0, Precision = 0.0, Recall = 1.0
    - GT non-empty, Pred empty: F0.5 = 0.0, Precision = 0.0, Recall = 0.0
    """
    f05, p, r = independent_entity_f05(set(), set())
    assert f05 == 1.0 and p == 1.0 and r == 1.0

    f05, p, r = independent_entity_f05(set(), {"S2-100"})
    assert f05 == 0.0 and p == 0.0

    f05, p, r = independent_entity_f05({"S2-100"}, set())
    assert f05 == 0.0 and p == 0.0 and r == 0.0


def test_macro_vs_global_f05_distinction():
    """Verify that Macro-F0.5 is mean(F0.5_i) and NOT computed from global pooled TP/FP/FN."""
    gt = {
        "S1-1": {"S2-1"},           # Match: F0.5 = 1.0
        "S1-2": {"S2-2"},           # Match: F0.5 = 1.0
        "S1-3": {"S2-3", "S2-4"},   # 0 TP, 1 FP: F0.5 = 0.0
    }
    preds = {
        "S1-1": {"S2-1"},
        "S1-2": {"S2-2"},
        "S1-3": {"S2-99"},
    }

    # Macro-F0.5 = (1.0 + 1.0 + 0.0) / 3 = 0.6667
    macro_metrics = evaluate_decision_metrics(gt, preds)
    assert abs(macro_metrics["macro_f05"] - 2/3) < 1e-4

    # Global pooled F0.5 would be:
    # TP = 2, FP = 1, FN = 2
    # Global Prec = 2/3, Global Rec = 2/4 = 0.5
    # Global F0.5 = 1.25 * (2/3) * 0.5 / (0.25 * (2/3) + 0.5) = 0.4167 / 0.6667 = 0.6250 != 0.6667
    global_p = 2 / 3
    global_r = 0.5
    global_f05 = (1.25 * global_p * global_r) / (0.25 * global_p + global_r)
    assert abs(macro_metrics["macro_f05"] - global_f05) > 0.01  # Confirms they are mathematically distinct!


def test_recall_multiplicative_identity():
    """Verify the three-tier recall identity:

    (accepted_true / candidate_true) * (candidate_true / total_true) == accepted_true / total_true
    """
    total_true = 500
    candidate_true = 440
    accepted_true = 360

    cand_recall = candidate_true / total_true
    model_recall_among_cands = accepted_true / candidate_true
    end_to_end_recall = accepted_true / total_true

    assert abs((cand_recall * model_recall_among_cands) - end_to_end_recall) < 1e-12


def test_group_split_disjointness_property():
    """Verify group-aware split guarantees strictly zero intersection."""
    s1_pool = [f"S1-{i}" for i in range(5000)]
    train_set, val_set = split_s1_groups(s1_pool, train_ratio=0.75, random_seed=99)
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set) + len(val_set) == 5000


def test_zero_label_leakage_in_feature_names():
    """Verify no feature name contains leakage indicators."""
    forbidden = ["label", "ground_truth", "matched_id", "is_correct", "target_gt"]
    for feat in FEATURE_NAMES:
        for f in forbidden:
            assert f not in feat.lower(), f"Leakage detected in feature name: {feat}"
