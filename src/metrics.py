"""Evaluation metrics shared by training (TensorFlow) and serving (TF-free) code."""
import numpy as np

from src.config import CLASSES


def compute_metrics(y_true, y_prob):
    """Accuracy, macro precision/recall/F1, OvR ROC-AUC, cross-entropy loss and confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # exact normalisation for log_loss
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)),
        "loss": float(log_loss(y_true, y_prob, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0
        ),
        "n_test": int(len(y_true)),
    }
