"""Evaluation metrics for IDS.

Convention: 0 = BENIGN, 1 = ATTACK. We care about detecting ATTACK (1).

Reported metrics:
    accuracy             - overall correctness (informative but misleading
                           when classes are imbalanced; reported for completeness)
    f1                   - harmonic mean of precision/recall, balanced
    precision_attack     - of all flagged attacks, what fraction are real?
                           = TP / (TP + FP). Low precision = many false alarms.
    recall_attack        - of all real attacks, what fraction did we catch?
                           = TP / (TP + FN). Low recall = missed attacks (dangerous!)
    fnr                  - false negative rate = 1 - recall_attack
                           Fraction of attacks we missed. The "miss rate".
    fpr                  - false positive rate = FP / (FP + TN)
                           Fraction of benign flagged as attack. The "false alarm rate".
    roc_auc              - threshold-independent ranking quality
    confusion_matrix     - [[TN, FP], [FN, TP]]

Plus gap metrics (train - test) for overfitting analysis.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute all classification metrics for binary IDS task."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_attack": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_attack": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_benign": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
    }
    metrics["fnr"] = 1.0 - metrics["recall_attack"]
    metrics["fpr"] = 1.0 - metrics["recall_benign"]

    # ROC-AUC requires probabilities and both classes present
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def compute_train_test_metrics(
    y_train: np.ndarray, y_train_pred: np.ndarray, y_train_prob: np.ndarray | None,
    y_test: np.ndarray, y_test_pred: np.ndarray, y_test_prob: np.ndarray | None,
) -> dict:
    """Compute train and test metrics, plus train-test gap.

    Gap = train_metric - test_metric. Positive gap = overfitting.
    """
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

    out = {}
    for k, v in train_metrics.items():
        out["train_confusion" if k == "confusion_matrix" else f"train_{k}"] = v
    for k, v in test_metrics.items():
        out["test_confusion" if k == "confusion_matrix" else f"test_{k}"] = v

    # Generalization gaps - higher = more overfitting
    out["gap_f1"] = out["train_f1"] - out["test_f1"]
    out["gap_accuracy"] = out["train_accuracy"] - out["test_accuracy"]
    out["gap_recall_attack"] = out["train_recall_attack"] - out["test_recall_attack"]

    return out
