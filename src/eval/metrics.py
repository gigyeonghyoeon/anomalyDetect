"""AUROC, Accuracy, Precision, Recall, F1, Confusion Matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auroc"] = float("nan")
    return metrics


def save_confusion_matrix(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float, out_path: Path, title: str = ""
) -> None:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Pneumonia"])
    ax.set_yticklabels(["Normal", "Pneumonia"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_score_histogram(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float, out_path: Path, title: str = ""
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    normal = y_score[y_true == 0]
    anomaly = y_score[y_true == 1]
    if len(normal):
        ax.hist(normal, bins=30, alpha=0.6, label="Normal")
    if len(anomaly):
        ax.hist(anomaly, bins=30, alpha=0.6, label="Pneumonia")
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title(title or "Score distribution")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return auroc
