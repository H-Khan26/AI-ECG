# ai_ecg/eval.py

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report
)


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute common binary classification metrics.
    """
    # binary predictions at given threshold
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_scores),
        "pr_auc":  average_precision_score(y_true, y_scores),
        "accuracy":     accuracy_score(y_true, y_pred),
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "f1":           f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
    }
    return metrics


def plot_roc(y_true: np.ndarray, y_scores: np.ndarray, ax=None) -> plt.Axes:
    """
    Plot ROC curve and return the Axes.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return ax


def plot_pr(y_true: np.ndarray, y_scores: np.ndarray, ax=None) -> plt.Axes:
    """
    Plot Precision-Recall curve and return the Axes.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    return ax


def main(pred_csv: str, output_dir: str, threshold: float = 0.5):
    """
    Load a CSV with columns ['y_true','y_score'] (and optionally 'y_pred'),
    compute metrics, save a JSON report, and write ROC/PR plots as PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(pred_csv)

    # infer columns
    if "y_true" in df.columns and "y_score" in df.columns:
        y_true  = df["y_true"].values
        y_score = df["y_score"].values
    elif {"Group","y_score"}.issubset(df.columns):
        # legacy: Group== 'CTRCD' → 1
        y_true  = (df["Group"]=="CTRCD").astype(int).values
        y_score = df["y_score"].values
    else:
        raise ValueError("CSV must contain 'y_true' and 'y_score' columns")

    # compute & save metrics
    metrics = compute_metrics(y_true, y_score, threshold=threshold)
    rpt_path = os.path.join(output_dir, "metrics.json")
    pd.Series({k:v for k,v in metrics.items() if not isinstance(v,list)}).to_json(rpt_path, indent=2)
    print(f"Saved metrics to {rpt_path}")

    # ROC plot
    ax1 = plot_roc(y_true, y_score)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    ax1.figure.savefig(roc_path, dpi=300, bbox_inches="tight")
    print(f"Saved ROC plot to {roc_path}")

    # PR plot
    ax2 = plot_pr(y_true, y_score)
    pr_path = os.path.join(output_dir, "pr_curve.png")
    ax2.figure.savefig(pr_path, dpi=300, bbox_inches="tight")
    print(f"Saved PR plot to {pr_path}")

    # classification report & confusion matrix
    y_pred = (y_score >= threshold).astype(int)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    rep_df = pd.DataFrame(report).T
    rep_df.to_csv(os.path.join(output_dir, "classification_report.csv"))
    pd.DataFrame(cm, index=["Control","CTRCD"], columns=["Pred=0","Pred=1"])\
      .to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
    print(f"Saved classification_report.csv and confusion_matrix.csv in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions: compute metrics + plots"
    )
    parser.add_argument("pred_csv", help="CSV with columns y_true and y_score")
    parser.add_argument(
        "-o", "--output_dir", default="eval_outputs",
        help="Where to write metrics.json, plots, and tables"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5,
        help="Decision threshold for binary predictions"
    )
    args = parser.parse_args()

    main(args.pred_csv, args.output_dir, threshold=args.threshold)
