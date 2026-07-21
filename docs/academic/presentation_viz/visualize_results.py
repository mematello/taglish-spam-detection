"""
Visualization utilities for TAGLISH-SPAM_DETECTION.

This script reads the unified evaluation outputs (metrics.json) and generates
high-quality charts and confusion matrices suitable for presentations.

Usage (from project root):
    python -m presentation_viz.visualize_results

Outputs (saved in presentation_viz/):
    - metrics_overview.png
    - metrics_detailed.png
    - confusion_matrices_grid.png
"""

import json
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(ROOT_DIR, "metrics.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "presentation_viz")


def load_metrics() -> List[Dict[str, Any]]:
    """Load model_results list from metrics.json."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"metrics.json not found at {METRICS_PATH}. "
            "Run evaluate_models.py first to generate it."
        )

    with open(METRICS_PATH, "r") as f:
        data = json.load(f)

    model_results = data.get("model_results", [])
    if not model_results:
        raise ValueError("No model_results found in metrics.json")

    return model_results


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_metrics_overview(results: List[Dict[str, Any]]) -> None:
    """
    Create a simple overview chart of Accuracy, Precision, Recall, and F1
    for each model.
    """
    models = [m["model_name"] for m in results]
    accuracy = [m["accuracy"] for m in results]
    precision = [m["precision"] for m in results]
    recall = [m["recall"] for m in results]
    f1 = [m["f1_score"] for m in results]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-darkgrid")

    plt.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
    plt.bar(x - 0.5 * width, precision, width, label="Precision")
    plt.bar(x + 0.5 * width, recall, width, label="Recall")
    plt.bar(x + 1.5 * width, f1, width, label="F1-Score")

    plt.xticks(x, models, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Overview (Accuracy / Precision / Recall / F1)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "metrics_overview.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_detailed_metrics(results: List[Dict[str, Any]]) -> None:
    """
    Create more detailed charts derived from the confusion matrices:
      - Spam vs Ham recall per model
      - False Positive Rate vs False Negative Rate per model
    This avoids relying on optional fields that may not exist in metrics.json.
    """
    models = [m["model_name"] for m in results]

    # Derived from confusion matrix:
    # cm = [[TN, FP],
    #       [FN, TP]]
    spam_recall = []  # TP / (TP + FN)
    ham_recall = []   # TN / (TN + FP)
    fpr = []          # FP rate: FP / (TN + FP)
    fnr = []          # FN rate: FN / (FN + TP)

    for m in results:
        cm = np.array(m["confusion_matrix"], dtype=float)
        tn, fp, fn, tp = cm.ravel()

        spam_recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        ham_recall.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        fpr.append(fp / (tn + fp) if (tn + fp) > 0 else 0.0)
        fnr.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 8))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Top subplot: recalls
    plt.subplot(2, 1, 1)
    plt.bar(x - width / 2, ham_recall, width, label="Ham Recall")
    plt.bar(x + width / 2, spam_recall, width, label="Spam Recall")
    plt.xticks(x, models, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Recall")
    plt.title("Per-Class Recall by Model")
    plt.legend()

    # Bottom subplot: error rates
    plt.subplot(2, 1, 2)
    plt.bar(x - width / 2, fpr, width, label="False Positive Rate (Ham → Spam)")
    plt.bar(x + width / 2, fnr, width, label="False Negative Rate (Spam → Ham)")
    plt.xticks(x, models, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Error Rate")
    plt.title("Error Rates by Model")
    plt.legend()

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "metrics_detailed.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrices(results: List[Dict[str, Any]]) -> None:
    """
    Create a grid of confusion matrices (one per model).
    """
    n_models = len(results)
    if n_models == 0:
        return

    plt.figure(figsize=(5 * n_models, 4))
    plt.style.use("seaborn-v0_8-darkgrid")

    for idx, metrics in enumerate(results, start=1):
        cm = np.array(metrics["confusion_matrix"])
        model_name = metrics["model_name"]
        acc = metrics["accuracy"]

        plt.subplot(1, n_models, idx)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
        )
        plt.title(f"{model_name}\nAccuracy: {acc:.3f}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "confusion_matrices_grid.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    ensure_output_dir()
    results = load_metrics()

    print(f"Loaded metrics for {len(results)} models from {METRICS_PATH}")
    print(f"Saving figures to {OUTPUT_DIR}")

    plot_metrics_overview(results)
    plot_detailed_metrics(results)
    plot_confusion_matrices(results)

    print("Visualization generation completed successfully.")


if __name__ == "__main__":
    main()


