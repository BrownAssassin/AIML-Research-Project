#!/usr/bin/env python3
"""
Visualization and Analysis of Traversability Model Metrics

This module generates comparison visualizations between traversability prediction
models (Grounded-SAM and Follow-the-Footprints), including:
  - Metrics bar charts (Accuracy, Precision, Recall, F1, IoU)
  - Confusion matrices with normalized distributions
  - Qualitative 2x2 comparison grids (Input, GT, GSAM, FTF)

All visualizations are saved as PNG files to the OUTPUT_DIR directory.
Target metrics are for RELLIS-3D multi-modal traversability dataset.

Example usage:
  python plot_metrics.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# Path Configuration
# ==============================================================================
# Determines paths relative to project structure
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # /home/brownassassin/research-project
RELLIS = ROOT / "RELLIS-3D" / "Rellis-3D"
GSAM_OUT = ROOT / "rellis-output"
FTF_OUT = (
    ROOT / "outputs" / "prediction" / "RELLIS_3D-24-11-25-18:46:11-model_best-RELLIS_3D"
)

OUTPUT_DIR = Path("../results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Model Performance Metrics (Final Models on RELLIS-3D Test Set)
# ==============================================================================
# Metrics from evaluation on RELLIS-3D test split for final model versions:
#   Grounded-SAM: Final iteration using GroundingDINO + SAM
#   Follow-the-Footprints: End of training at Epoch 21
METRICS = {
    "Grounded-SAM": {
        "Accuracy": 0.6791,
        "Precision": 0.7664,
        "Recall": 0.1733,
        "F1": 0.2827,
        "IoU": 0.1646,
    },
    "Follow-the-Footprints": {
        "Accuracy": 0.703,
        "Precision": 0.164,
        "Recall": 0.959,
        "F1": 0.280,
        "IoU": 0.163,
    },
}

# ==============================================================================
# Confusion Matrix Data (normalized counts)
# ==============================================================================
# Per-frame pixel-level confusion matrix for Grounded-SAM on RELLIS-3D
# Format: [True Negatives, False Positives]
#         [False Negatives, True Positives]
# Conversion: pixel labels converted to traversable/non-traversable binary
CONFUSION_MATRICES = {
    "Grounded-SAM": np.array(
        [
            [8846467461, 276713438],  # TN, FP
            [4331880827, 908074274],  # FN, TP
        ]
    ),
}

# ==============================================================================
# Matplotlib Configuration
# ==============================================================================
# Set DPI for high-quality output when saving figures

plt.rcParams["savefig.dpi"] = 200


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_metrics_bar_chart(metrics):
    """
    Generate a grouped bar chart comparing metrics between models.

    Args:
        metrics (dict): Dictionary mapping model names to metric dictionaries.
                       Example: {"Model A": {"Accuracy": 0.8, "Precision": 0.75, ...}}

    Output: Saves PNG file to OUTPUT_DIR/metrics_bar_chart.png
    """
    model_names = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for i, model in enumerate(model_names):
        values = [metrics[model][m] for m in metric_names]
        offset = (i - 0.5) * width

        ax.bar(x + offset, values, width, label=model)

        for j, v in enumerate(values):
            ax.text(
                x[j] + offset,
                v + 0.02,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Traversability Metrics on RELLIS-3D")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.legend()

    fig.tight_layout()

    out = OUTPUT_DIR / "metrics_bar_chart.png"

    fig.savefig(out)
    plt.close(fig)

    print(f"[Saved] {out}")


# Confusion Matrix Heatmap
def plot_confusion_matrix(cm, title, out_name):
    """
    Generate a normalized confusion matrix heatmap visualization.

    Args:
        cm (np.ndarray): 2x2 confusion matrix with counts.
                        Format: [[TN, FP], [FN, TP]]
        title (str): Title for the plot
        out_name (str): Output filename (relative to OUTPUT_DIR)

    Output: Saves normalized heatmap PNG to OUTPUT_DIR/{out_name}
    """
    cm_sum = cm.sum()
    cm_norm = cm / cm_sum if cm_sum > 0 else cm.astype(float)

    fig, ax = plt.subplots(figsize=(5.5, 4))

    im = ax.imshow(cm_norm, cmap="viridis")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-traversable", "Traversable"])
    ax.set_yticklabels(["Non-traversable", "Traversable"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i,j]:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out = OUTPUT_DIR / out_name

    fig.savefig(out)
    plt.close()

    print(f"[Saved] {out}")


# Qualitative 2×2 Grid
def qualitative_example_grid(
    original_path: Path,
    gt_path: Path,
    gsam_path: Path,
    ftf_path: Path,
    out_name: str = "qualitative_example.png",
    title: str = "Qualitative Comparison on RELLIS-3D",
):
    """
    Create a 2x2 grid showing qualitative comparison between models.

    Grid layout:
        [RGB input image]                 [Ground truth traversability]
        [Grounded-SAM prediction]         [FTF-Epoch21 prediction]

    Args:
        original_path (Path): RGB input image file (.jpg)
        gt_path (Path): Ground truth binary/ID annotation (.png)
        gsam_path (Path): Grounded-SAM prediction output (.png)
        ftf_path (Path): Follow-the-Footprints prediction (.png)
        out_name (str): Output filename (default: "qualitative_example.png")
        title (str): Figure title (default: "Qualitative Comparison on RELLIS-3D")

    Output: Saves 2x2 comparison grid PNG to OUTPUT_DIR/{out_name}
    
    Note: Missing images are shown as placeholder text instead of causing errors.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # taller figure

    items = [
        (original_path, "Input RGB"),
        (gt_path, "Ground Truth Traversability"),
        (gsam_path, "Grounded-SAM Prediction"),
        (ftf_path, "FTFoot Prediction"),
    ]

    for ax, (path, subtitle) in zip(axes.flatten(), items):
        img = cv2.imread(str(path))

        if img is None:
            ax.text(0.5, 0.5, f"Missing: {path.name}", ha="center", va="center")
            ax.axis("off")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=12)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = OUTPUT_DIR / out_name

    fig.savefig(out)
    plt.close(fig)

    print(f"[Saved] {out}")


def main():
    """
    Main entry point: Generate all metric visualizations.
    
    Outputs:
    1. metrics_bar_chart.png - Grouped bar chart of model metrics
    2. cm_grounded_sam.png - Confusion matrix heatmap for GSAM
    3. qualitative.png - 2x2 comparison grid from sample frame (00004)
    """
    print("Generating metrics bar chart...")
    plot_metrics_bar_chart(METRICS)

    print("Generating confusion matrix for GSAM...")
    plot_confusion_matrix(
        CONFUSION_MATRICES["Grounded-SAM"],
        "Confusion Matrix (Normalized) — Grounded-SAM",
        "cm_grounded_sam.png",
    )

    print("Generating qualitative example grid...")
    qualitative_example_grid(
        original_path=RELLIS / "00004/pylon_camera_node/frame000000-1581791678_408.jpg",
        gt_path=RELLIS
        / "00004/pylon_camera_node_label_id/frame000000-1581791678_408.png",
        gsam_path=GSAM_OUT / "00004/frame000000-1581791678_408/pred_traversable.png",
        ftf_path=FTF_OUT / "00004_000100.png",
        out_name="qualitative.png",
    )

    print("Done.")


if __name__ == "__main__":
    main()
