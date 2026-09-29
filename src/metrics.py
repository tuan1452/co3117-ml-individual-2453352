"""Evaluation metrics module."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate multiclass classification metrics with Macro-F1."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "confusion_matrix": cm,
    }


def print_evaluation_report(metrics: Dict[str, Any], title: str = "Model Evaluation") -> None:
    """Print evaluation report."""
    print(f"\n{'=' * 15} {title} {'=' * 15}")
    print(f"Macro-F1 (Primary Metric): {metrics['macro_f1']:.4f}")
    print(f"Accuracy:                 {metrics['accuracy']:.4f}")
    print(f"Macro-Precision:          {metrics['macro_precision']:.4f}")
    print(f"Macro-Recall:             {metrics['macro_recall']:.4f}")
    print("-" * 45)
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("=" * 45)


if __name__ == "__main__":
    # Test run
    y_test_mock = np.array([1, 2, 3, 4, 5, 6, 1, 2])
    y_pred_mock = np.array([1, 2, 3, 4, 5, 6, 2, 2])
    res = evaluate_multiclass(y_test_mock, y_pred_mock)
    print_evaluation_report(res, title="Unit Test Demo")