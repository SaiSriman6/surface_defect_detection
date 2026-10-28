"""
Evaluation module for Industrial Surface Defect Detection.

Evaluates trained models (Feature Extraction and Fine-Tuning) strictly on
the held-out test set. Computes Accuracy, Precision, Recall, F1-score, ROC-AUC,
Confusion Matrix, and generates comparison CSV and visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.preprocessing import create_split_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
    class_names: list = ["Normal", "Defective"],
):
    """Generates and saves a styled Confusion Matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Ground Truth", fontsize=11, fontweight="bold")

    # Add industrial context annotations
    tn, fp, fn, tp = cm.ravel()
    fig.text(0.15, -0.05, f"TN (Normal Correct): {tn} | FP (False Alarm): {fp}\nFN (Defect Missed): {fn} | TP (Defect Caught): {tp}",
             fontsize=9, ha="left", style="italic", bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix heatmap saved to: %s", output_path)


def evaluate_single_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    model_name: str,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> Dict:
    """
    Evaluates a single model on the test dataset.
    Returns metrics dictionary with actual calculations from predictions.
    """
    all_y_true = []
    all_y_pred_prob = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_y_pred_prob.extend(preds.flatten())
        all_y_true.extend(labels.numpy().flatten())

    y_true = np.array(all_y_true, dtype=int)
    y_prob = np.array(all_y_pred_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Defective"],
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_name": model_name,
        "sample_counts": {
            "total_test_samples": len(y_true),
            "normal_samples": int(np.sum(y_true == 0)),
            "defective_samples": int(np.sum(y_true == 1)),
        },
        "classification_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "classification_report": report,
        "threshold": threshold,
    }

    return metrics, cm


def evaluate_models(category: str = config.DEFAULT_CATEGORY) -> pd.DataFrame:
    """
    Evaluates both feature-extraction and fine-tuned models on held-out test data.
    Generates comparison tables, json metrics, and confusion matrix charts.
    """
    logger.info("--- Starting Stage: Held-out Test Set Model Evaluation ---")

    _, _, test_ds, counts = create_split_datasets(category=category, batch_size=config.BATCH_SIZE)
    logger.info("Test set contains %d samples (%d normal, %d defective).",
                counts["test"]["total"], counts["test"]["normal"], counts["test"]["defective"])

    results = []

    # 1. Feature Extraction Evaluation
    if config.FEATURE_EXTRACTION_MODEL_PATH.exists():
        logger.info("Evaluating Feature Extraction model from '%s'...", config.FEATURE_EXTRACTION_MODEL_PATH)
        fe_model = tf.keras.models.load_model(config.FEATURE_EXTRACTION_MODEL_PATH)
        fe_metrics, fe_cm = evaluate_single_model(fe_model, test_ds, "Feature Extraction")

        with open(config.FE_METRICS_PATH, "w") as f:
            json.dump(fe_metrics, f, indent=2)

        plot_confusion_matrix(
            fe_cm,
            config.FE_CONF_MATRIX_PATH,
            title="Confusion Matrix: Feature Extraction (Frozen Backbone)",
        )

        results.append({
            "Model": "Feature Extraction",
            "Accuracy": fe_metrics["classification_metrics"]["accuracy"],
            "Precision": fe_metrics["classification_metrics"]["precision"],
            "Recall": fe_metrics["classification_metrics"]["recall"],
            "F1-Score": fe_metrics["classification_metrics"]["f1_score"],
            "ROC-AUC": fe_metrics["classification_metrics"]["roc_auc"],
            "True Positives": fe_metrics["confusion_matrix"]["tp"],
            "False Positives": fe_metrics["confusion_matrix"]["fp"],
            "True Negatives": fe_metrics["confusion_matrix"]["tn"],
            "False Negatives": fe_metrics["confusion_matrix"]["fn"],
        })
    else:
        logger.warning("Feature extraction model not found at '%s'. Skipping evaluation.", config.FEATURE_EXTRACTION_MODEL_PATH)

    # 2. Fine-Tuning Evaluation
    if config.FINETUNED_MODEL_PATH.exists():
        logger.info("Evaluating Fine-Tuned model from '%s'...", config.FINETUNED_MODEL_PATH)
        ft_model = tf.keras.models.load_model(config.FINETUNED_MODEL_PATH)
        ft_metrics, ft_cm = evaluate_single_model(ft_model, test_ds, "Fine-Tuning")

        with open(config.FT_METRICS_PATH, "w") as f:
            json.dump(ft_metrics, f, indent=2)

        plot_confusion_matrix(
            ft_cm,
            config.FT_CONF_MATRIX_PATH,
            title="Confusion Matrix: Fine-Tuning (Upper Layers Unfrozen)",
        )

        results.append({
            "Model": "Fine-Tuning",
            "Accuracy": ft_metrics["classification_metrics"]["accuracy"],
            "Precision": ft_metrics["classification_metrics"]["precision"],
            "Recall": ft_metrics["classification_metrics"]["recall"],
            "F1-Score": ft_metrics["classification_metrics"]["f1_score"],
            "ROC-AUC": ft_metrics["classification_metrics"]["roc_auc"],
            "True Positives": ft_metrics["confusion_matrix"]["tp"],
            "False Positives": ft_metrics["confusion_matrix"]["fp"],
            "True Negatives": ft_metrics["confusion_matrix"]["tn"],
            "False Negatives": ft_metrics["confusion_matrix"]["fn"],
        })
    else:
        logger.warning("Fine-tuned model not found at '%s'. Skipping evaluation.", config.FINETUNED_MODEL_PATH)

    if not results:
        raise FileNotFoundError("Neither model checkpoint exists to evaluate.")

    df_comparison = pd.DataFrame(results)
    df_comparison.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    logger.info("Model comparison results saved to: %s\n\n%s", config.MODEL_COMPARISON_CSV, df_comparison.to_string(index=False))

    return df_comparison


if __name__ == "__main__":
    evaluate_models()
