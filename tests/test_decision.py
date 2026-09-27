"""Unit Tests for Decision Engine, Threshold Calibration, and Macro-F0.5 Logic.

Tests:
1. Exact F0.5 numerical calculation per entity and macro-averaging
2. Strict singleton behavior (score=1.0 only if empty, 0.0 upon any false positive)
3. Multi-match entity support (capable of capturing 2+ matches)
4. Score-gap multi-match filtering logic
5. Group-aware train/val split guarantees (zero S1 overlap)
"""

import pytest

from src.build_training_pairs import split_s1_groups
from src.decision import evaluate_decision_metrics, predict_matches
from src.evaluate import compute_entity_f05, compute_macro_f05


def test_f05_numerical_calculation():
    # Exact match: precision=1.0, recall=1.0 -> F0.5 = 1.0
    assert compute_entity_f05({"S2-1"}, {"S2-1"}) == 1.0

    # Partial match with precision penalty:
    # true: {A}, pred: {A, B} -> precision = 0.5, recall = 1.0
    # beta = 0.5 -> beta^2 = 0.25
    # num = 1.25 * 0.5 * 1.0 = 0.625
    # den = (0.25 * 0.5) + 1.0 = 0.125 + 1.0 = 1.125
    # F0.5 = 0.625 / 1.125 = 5 / 9 = 0.5555...
    score = compute_entity_f05({"S2-1"}, {"S2-1", "S2-2"})
    assert abs(score - 5/9) < 1e-4

    # Recall penalty:
    # true: {A, B}, pred: {A} -> precision = 1.0, recall = 0.5
    # num = 1.25 * 1.0 * 0.5 = 0.625
    # den = (0.25 * 1.0) + 0.5 = 0.25 + 0.5 = 0.75
    # F0.5 = 0.625 / 0.75 = 5 / 6 = 0.8333...
    # Notice F0.5 penalizes false positives (score 0.556) heavier than false negatives (score 0.833)!
    score_recall = compute_entity_f05({"S2-1", "S2-2"}, {"S2-1"})
    assert abs(score_recall - 5/6) < 1e-4
    assert score_recall > score  # Confirms precision-weighting!


def test_singleton_strict_scoring():
    # True singleton predicted empty -> 1.0
    assert compute_entity_f05(set(), set()) == 1.0

    # True singleton predicted with 1 false positive -> 0.0
    assert compute_entity_f05(set(), {"S2-1"}) == 0.0

    # Non-singleton predicted empty -> 0.0
    assert compute_entity_f05({"S2-1"}, set()) == 0.0


def test_macro_f05_aggregation():
    ground_truth = {
        "S1-1": {"S2-1"},          # Correct match (1.0)
        "S1-2": set(),             # Singleton correct (1.0)
        "S1-3": set(),             # Singleton false match (0.0)
        "S1-4": {"S3-1", "S3-2"},  # Multi-match 1/2 correct (0.8333)
    }
    predictions = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
        "S1-3": {"S2-99"},
        "S1-4": {"S3-1"},
    }

    metrics = evaluate_decision_metrics(ground_truth, predictions)
    expected_mean = (1.0 + 1.0 + 0.0 + 5/6) / 4  # 2.8333 / 4 = 0.7083
    assert abs(metrics["macro_f05"] - expected_mean) < 1e-3
    assert metrics["singleton_count"] == 2
    assert metrics["singleton_exact_empty_acc"] == 0.50
    assert metrics["singleton_false_match_rate"] == 0.50
    assert metrics["false_merges_count"] == 1


def test_predict_matches_threshold_and_gap():
    candidates = [
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "score": 0.85},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2", "score": 0.82},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-3", "score": 0.40},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S3-1", "score": 0.30},
    ]

    # Threshold 0.50 without score gap
    preds = predict_matches(candidates, threshold=0.50)
    assert preds["S1-1"] == {"S2-1", "S2-2"}
    assert preds["S1-2"] == set()

    # Threshold 0.50 with strict score gap 0.02
    # S2-1 has 0.85, S2-2 has 0.82 -> gap is 0.03 > 0.02 -> S2-2 excluded!
    preds_gap = predict_matches(candidates, threshold=0.50, score_gap_threshold=0.02)
    assert preds_gap["S1-1"] == {"S2-1"}

    # Threshold 0.50 with relaxed score gap 0.05
    # S2-1 has 0.85, S2-2 has 0.82 -> gap is 0.03 <= 0.05 -> S2-2 included!
    preds_gap2 = predict_matches(candidates, threshold=0.50, score_gap_threshold=0.05)
    assert preds_gap2["S1-1"] == {"S2-1", "S2-2"}


def test_split_s1_groups_leakage_free():
    s1_ids = [f"S1-{i}" for i in range(1000)]
    train_ids, val_ids = split_s1_groups(s1_ids, train_ratio=0.80, random_seed=42)

    assert len(train_ids) == 800
    assert len(val_ids) == 200
    assert len(train_ids.intersection(val_ids)) == 0
