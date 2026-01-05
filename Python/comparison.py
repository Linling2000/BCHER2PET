#!/usr/bin/env python3
"""
Friedman + Nemenyi on test-set model predictions.
Inputs: table with true labels and predicted labels from multiple models.
Outputs: Excel summary and PDF with Critical Difference diagrams.
"""

import os
import argparse
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute macro metrics and multiclass specificity (mean over classes)."""
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Sensitivity"] = recall_score(y_true, y_pred, average="macro")
    metrics["F1"] = f1_score(y_true, y_pred, average="macro")
    metrics["Precision"] = precision_score(y_true, y_pred, average="macro")

    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity_per_class = []
    for i in range(cm.shape[0]):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    metrics["Specificity"] = float(np.mean(specificity_per_class))
    return metrics

def bootstrap_metrics(
    df: pd.DataFrame,
    model_cols: List[str],
    true_col: str,
    n_bootstrap: int = 50,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Return dict of arrays with shape (n_bootstrap, n_models) for each metric."""
    rng = np.random.RandomState(random_state)
    n_samples = len(df)
    metrics_list = ["Accuracy", "Sensitivity", "F1", "Specificity", "Precision"]
    result = {m: [] for m in metrics_list}

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot = df.iloc[idx].reset_index(drop=True)
        y_true = boot[true_col].values
        scores = {m: [] for m in metrics_list}
        for model in model_cols:
            y_pred = boot[model].values
            mets = calculate_metrics(y_true, y_pred)
            for k, v in mets.items():
                scores[k].append(v)
        for k in metrics_list:
            result[k].append(scores[k])

    for k in result:
        result[k] = np.array(result[k])
    return result

def friedman_nemenyi_test(data_matrix: np.ndarray, model_names: List[str]) -> Dict:
    """Perform Friedman test and compute Nemenyi critical difference."""
    n_samples, n_models = data_matrix.shape
    # ranks per row (higher value -> better rank = 1)
    ranks = np.zeros_like(data_matrix)
    for i in range(n_samples):
        ranks[i] = stats.rankdata(-data_matrix[i])
    avg_ranks = np.mean(ranks, axis=0)
    stat, p_value = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(n_models)])
    q_alpha_dict = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    q_alpha = q_alpha_dict.get(n_models, 2.850)
    cd = float(q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_samples)))
    return {
        "avg_ranks": avg_ranks,
        "friedman_stat": float(stat),
        "p_value": float(p_value),
        "CD": cd,
        "model_names": list(model_names),
    }

def plot_all_metrics_cd(results: Dict[str, Dict], figsize=(14, 18)):
    """Plot CD diagrams for all metrics into a single figure."""
    metrics = list(results.keys())
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        res = results[metric]
        avg_ranks = np.array(res["avg_ranks"])
        cd = res["CD"]
        names = res["model_names"]
        k = len(names)
        sorted_idx = np.argsort(avg_ranks)
        sr = avg_ranks[sorted_idx]
        sn = [names[i] for i in sorted_idx]

        ax.hlines(0.5, 0.5, k + 0.5, color="black", linewidth=2)
        for i in range(1, k + 1):
            ax.vlines(i, 0.45, 0.55, color="black", linewidth=1.5)
            ax.text(i, 0.35, str(i), ha="center", va="top", fontsize=10, fontweight="bold")

        ax.hlines(0.85, 1, 1 + cd, colors="red", linewidth=3)
        ax.text(1 + cd / 2, 0.92, f"CD={cd:.2f}", ha="center", va="bottom", fontsize=10, color="red", fontweight="bold")

        n_top = (k + 1) // 2
        top = [(sr[i], sn[i]) for i in range(n_top)]
        bottom = [(sr[i], sn[i]) for i in range(n_top, k)]

        for i, (rank, name) in enumerate(top):
            y_off = 0.06 * i
            ax.vlines(rank, 0.55, 0.62 + y_off, color="blue", linewidth=1)
            ax.hlines(0.62 + y_off, 0.2, rank, color="blue", linewidth=1)
            ax.text(0.15, 0.62 + y_off, f"{name} ({rank:.2f})", ha="right", va="center", fontsize=10, color="blue")

        for i, (rank, name) in enumerate(bottom):
            y_off = 0.06 * i
            ax.vlines(rank, 0.45, 0.38 - y_off, color="green", linewidth=1)
            ax.hlines(0.38 - y_off, rank, k + 0.8, color="green", linewidth=1)
            ax.text(k + 0.85, 0.38 - y_off, f"({rank:.2f}) {name}", ha="left", va="center", fontsize=10, color="green")

        y_bar = 0.56
        for i in range(len(sr)):
            for j in range(i + 1, len(sr)):
                if sr[j] - sr[i] < cd:
                    ax.hlines(y_bar, sr[i], sr[j], color="black", linewidth=4, alpha=0.6)
                    y_bar += 0.025

        p_value = res.get("p_value", np.nan)
        sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns")
        ax.set_title(f"{metric} (p={p_value:.4f}) {sig}", fontsize=12, loc="left", fontweight="bold")
        ax.set_xlim(-0.5, k + 1.5)
        ax.set_ylim(0, 1.0)
        ax.axis("off")

    fig.suptitle("Test Set - Critical Difference Diagrams (* p<0.05, ** p<0.01, ns: not significant)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    else:
        return pd.read_csv(path)

def main():
    parser = argparse.ArgumentParser(description="Friedman-Nemenyi analysis on test-set predictions")
    parser.add_argument("--input", required=True, help="Input table (csv or xlsx)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model-cols", default="", help="Comma-separated model prediction columns (optional)")
    parser.add_argument("--true-col", default="HER23", help="True label column name")
    parser.add_argument("--split-col", default="tt", help="Split column name (test indicated by test-value)")
    parser.add_argument("--test-value", default="0", help="Value indicating test rows in split column")
    parser.add_argument("--n-bootstrap", type=int, default=50, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = read_table(args.input)
    test_mask = df[args.split_col].astype(str) == str(args.test_value)
    test_df = df[test_mask].copy()
    if test_df.empty:
        raise SystemExit("No test rows found. Check split column and test value.")

    if args.model_cols:
        model_cols = [c.strip() for c in args.model_cols.split(",") if c.strip()]
    else:
        skip = {args.true_col, args.split_col}
        model_cols = [c for c in test_df.columns if c not in skip]

    missing = [c for c in model_cols if c not in test_df.columns]
    if missing:
        raise SystemExit(f"Missing model columns in input: {missing}")

    model_names = [f"Model {i+1}" for i in range(len(model_cols))]

    bootstrap = bootstrap_metrics(test_df, model_cols, true_col=args.true_col, n_bootstrap=args.n_bootstrap, random_state=args.seed)

    metrics_to_analyze = ["Accuracy", "Sensitivity", "F1", "Specificity", "Precision"]
    all_results = {}
    for metric in metrics_to_analyze:
        data_mat = bootstrap[metric]
        res = friedman_nemenyi_test(data_mat, model_names)
        all_results[metric] = res

    summary_rows = []
    for metric in metrics_to_analyze:
        res = all_results[metric]
        best_idx = int(np.argmin(res["avg_ranks"]))
        summary_rows.append({
            "metric": metric,
            "best_model": model_names[best_idx],
            "avg_rank": float(res["avg_ranks"][best_idx]),
            "p_value": res["p_value"],
            "significant": res["p_value"] < 0.05,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "test_set_performance_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    pdf_path = os.path.join(args.output_dir, "cd_diagrams.pdf")
    with PdfPages(pdf_path) as pdf:
        fig = plot_all_metrics_cd(all_results, figsize=(14, 18))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print("Done.")
    print("Summary saved to:", summary_path)
    print("PDF saved to:", pdf_path)

if __name__ == "__main__":
    main()