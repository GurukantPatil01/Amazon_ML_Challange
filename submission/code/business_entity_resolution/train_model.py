"""LightGBM Pairwise Ranking Model Training Module.

Trains the binary classification GBDT on labeled candidate pairs
using 43 calibrated features and early stopping.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from .pair_features import FEATURE_NAMES, extract_batch_features


def train_lightgbm_matcher(
    train_pairs: List[Dict],
    val_pairs: List[Dict],
    s1_lookup: Dict[str, Dict],
    target_lookup: Dict[str, Dict],
    y_train: np.ndarray,
    y_val: np.ndarray,
    model_save_path: Optional[Path] = None,
    params: Optional[Dict] = None,
) -> lgb.Booster:
    """Trains LightGBM classifier with early stopping and AUC monitoring."""
    if params is None:
        params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "boosting_type": "gbdt",
            "n_estimators": 350,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 25,
            "colsample_bytree": 0.80,
            "subsample": 0.80,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    X_train = extract_batch_features(train_pairs, s1_lookup, target_lookup)
    X_val = extract_batch_features(val_pairs, s1_lookup, target_lookup)

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    if model_save_path:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_save_path))

    return model
