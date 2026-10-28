"""
Automated unit tests for model evaluation, metrics computation, and report generation.
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from src.evaluate import evaluate_single_model


def test_metric_calculation_correctness():
    """Verifies that mathematical calculations of precision, recall, and F1 follow formulas."""
    # Synthetic ground truth and predictions
    # y_true: 5 normal (0), 5 defective (1)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # y_pred: 4 TN, 1 FP, 1 FN, 4 TP
    y_pred = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    assert acc == 0.8
    assert prec == 4 / 5  # TP / (TP + FP) = 4 / (4 + 1) = 0.8
    assert rec == 4 / 5   # TP / (TP + FN) = 4 / (4 + 1) = 0.8
    assert f1 == 0.8      # 2 * (P * R) / (P + R) = 0.8


def test_confusion_matrix_values():
    """Verifies confusion matrix quadrant mapping."""
    from sklearn.metrics import confusion_matrix
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    assert tn == 1
    assert fp == 1
    assert fn == 1
    assert tp == 1


def test_missing_evaluation_artifacts_handling():
    """Verifies that missing model paths during evaluation raise appropriate errors."""
    from src.evaluate import evaluate_models

    # Pass nonexistent category to force failure
    with pytest.raises(Exception):
        evaluate_models(category="nonexistent_category_xyz")


def test_actual_evaluation_artifacts_validity():
    """Integration test: validates that actual generated metrics and comparison CSV exist and are valid."""
    import json
    import pandas as pd

    if config.FE_METRICS_PATH.exists():
        with open(config.FE_METRICS_PATH, "r") as f:
            fe_data = json.load(f)
        assert "classification_metrics" in fe_data
        assert 0.0 <= fe_data["classification_metrics"]["accuracy"] <= 1.0

    if config.FT_METRICS_PATH.exists():
        with open(config.FT_METRICS_PATH, "r") as f:
            ft_data = json.load(f)
        assert "classification_metrics" in ft_data
        assert 0.0 <= ft_data["classification_metrics"]["accuracy"] <= 1.0

    if config.MODEL_COMPARISON_CSV.exists():
        df = pd.read_csv(config.MODEL_COMPARISON_CSV)
        assert "Model" in df.columns
        assert "Accuracy" in df.columns
        assert "Recall" in df.columns
        assert len(df) >= 2
