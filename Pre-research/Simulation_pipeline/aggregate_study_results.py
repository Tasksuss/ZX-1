import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/Users/apple/Downloads/Simulation/outputs/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/Users/apple/Downloads/Simulation/outputs/.cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from pipeline_utils import ensure_dir, load_config, read_video_gray, save_csv, save_json


def aggregate(config_path: str) -> None:
    config = load_config(config_path)
    exp_root = Path(config["experiment_root"])
    analysis_root = ensure_dir(Path(config["analysis_root"]))

    summaries = []
    for summary_path in sorted(exp_root.glob("size_*/repeat_*/summary.json")):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["run_dir"] = str(summary_path.parent)
        summaries.append(data)

    grouped = defaultdict(list)
    for row in summaries:
        grouped[row["train_size_per_class_subject"]].append(row)

    aggregate_rows = []
    for size in sorted(grouped):
        rows = grouped[size]
        aggregate_rows.append({
            "train_size": size,
            "synthetic_test_accuracy_mean": float(np.mean([r["synthetic_test_accuracy"] for r in rows])),
            "synthetic_test_accuracy_std": float(np.std([r["synthetic_test_accuracy"] for r in rows])),
            "real_test_accuracy_mean": float(np.mean([r["real_test_accuracy"] for r in rows])),
            "real_test_accuracy_std": float(np.std([r["real_test_accuracy"] for r in rows])),
            "synthetic_test_loss_mean": float(np.mean([r["synthetic_test_loss"] for r in rows])),
            "synthetic_test_loss_std": float(np.std([r["synthetic_test_loss"] for r in rows])),
            "real_test_loss_mean": float(np.mean([r["real_test_loss"] for r in rows])),
            "real_test_loss_std": float(np.std([r["real_test_loss"] for r in rows])),
            "macro_f1_mean": float(np.mean([r["real_macro_f1"] for r in rows])),
            "macro_f1_std": float(np.std([r["real_macro_f1"] for r in rows])),
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in rows])),
        })

    save_csv(aggregate_rows, analysis_root / "aggregate_metrics.csv")
    save_json({"aggregate_metrics": aggregate_rows}, analysis_root / "aggregate_metrics.json")

    sizes = [row["train_size"] for row in aggregate_rows]
    real_acc = [row["real_test_accuracy_mean"] for row in aggregate_rows]
    real_acc_std = [row["real_test_accuracy_std"] for row in aggregate_rows]
    synth_acc = [row["synthetic_test_accuracy_mean"] for row in aggregate_rows]
    synth_acc_std = [row["synthetic_test_accuracy_std"] for row in aggregate_rows]

    plt.figure(figsize=(7, 5))
    plt.errorbar(sizes, real_acc, yerr=real_acc_std, marker="o", label="Real Test")
    plt.xlabel("Train Size Per Class/Subject")
    plt.ylabel("Accuracy (%)")
    plt.title("Learning Curve on Real Test Set")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(analysis_root / "learning_curve_real_accuracy.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.errorbar(sizes, synth_acc, yerr=synth_acc_std, marker="o", label="Synthetic Test", color="#f58518")
    plt.xlabel("Train Size Per Class/Subject")
    plt.ylabel("Accuracy (%)")
    plt.title("Learning Curve on Synthetic Test Set")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(analysis_root / "learning_curve_synthetic_accuracy.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    gap = np.array(synth_acc) - np.array(real_acc)
    plt.plot(sizes, gap, marker="o")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Train Size Per Class/Subject")
    plt.ylabel("Synthetic - Real Accuracy Gap")
    plt.title("Real vs Synthetic Performance Gap")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(analysis_root / "real_vs_synthetic_gap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        sizes,
        [row["macro_f1_mean"] for row in aggregate_rows],
        yerr=[row["macro_f1_std"] for row in aggregate_rows],
        marker="o",
        color="#54a24b",
    )
    plt.xlabel("Train Size Per Class/Subject")
    plt.ylabel("Macro F1")
    plt.title("Macro F1 vs Train Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(analysis_root / "macro_f1_vs_size.png", dpi=220)
    plt.close()

    per_class_scores = defaultdict(list)
    for size in sorted(grouped):
        class_values = defaultdict(list)
        for row in grouped[size]:
            report_path = Path(row["run_dir"]) / "classification_report_real.json"
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            for label in config["classes"]:
                class_values[label].append(report[label]["f1-score"])
        for label in config["classes"]:
            per_class_scores[label].append(float(np.mean(class_values[label])))

    plt.figure(figsize=(9, 5))
    for label in config["classes"]:
        plt.plot(sizes, per_class_scores[label], marker="o", label=label)
    plt.xlabel("Train Size Per Class/Subject")
    plt.ylabel("F1 Score")
    plt.title("Per-class F1 vs Train Size (Real Test)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_root / "per_class_f1_vs_size.png", dpi=220)
    plt.close()

    selected_sizes = []
    if sizes:
        selected_sizes = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]

    for split_name, output_name in [("real", "confusion_matrix_real_small_medium_large.png"), ("synthetic", "confusion_matrix_synth_small_medium_large.png")]:
        fig, axes = plt.subplots(1, len(selected_sizes), figsize=(5 * len(selected_sizes), 4))
        if len(selected_sizes) == 1:
            axes = [axes]
        for ax, size in zip(axes, selected_sizes):
            row = sorted(grouped[size], key=lambda x: x["real_test_accuracy"], reverse=True)[0]
            pred_path = Path(row["run_dir"]) / f"predictions_{split_name}.csv"
            trues, preds = [], []
            with open(pred_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for item in reader:
                    trues.append(item["label_true"])
                    preds.append(item["label_pred"])
            cm = confusion_matrix(trues, preds, labels=config["classes"])
            disp = ConfusionMatrixDisplay(cm, display_labels=config["classes"])
            disp.plot(ax=ax, colorbar=False, xticks_rotation=30)
            ax.set_title(f"{split_name} size={size}")
        fig.tight_layout()
        fig.savefig(analysis_root / output_name, dpi=220)
        plt.close(fig)

    sim_root = Path(config["sim_output_root"])
    preview_paths = sorted(sim_root.glob("*/*/*.mp4"))[:12]
    if preview_paths:
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        axes = axes.ravel()
        for ax, preview_path in zip(axes, preview_paths):
            frame = read_video_gray(str(preview_path), max_frames=1)[0]
            ax.imshow(frame, cmap="gray")
            ax.set_title(preview_path.parent.name, fontsize=8)
            ax.axis("off")
        for ax in axes[len(preview_paths):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(analysis_root / "simulated_dataset_gallery.png", dpi=220)
        plt.close(fig)

    ref_dir = Path(config["real_profile_dir"]) / "reference_frames"
    ref_images = sorted(ref_dir.glob("*.png"))[:6]
    sim_images = sorted(sim_root.glob("*/*/*.mp4"))[:6]
    if ref_images and sim_images:
        from PIL import Image
        fig, axes = plt.subplots(2, 6, figsize=(15, 5))
        for i, image_path in enumerate(ref_images):
            axes[0, i].imshow(Image.open(image_path))
            axes[0, i].axis("off")
            axes[0, i].set_title("Real", fontsize=8)
        for i, video_path in enumerate(sim_images):
            frame = read_video_gray(str(video_path), max_frames=1)[0]
            axes[1, i].imshow(frame, cmap="gray")
            axes[1, i].axis("off")
            axes[1, i].set_title("Synthetic", fontsize=8)
        fig.tight_layout()
        fig.savefig(analysis_root / "real_vs_simulated_larynx_frames.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/Users/apple/Downloads/Simulation/config.json")
    args = parser.parse_args()
    aggregate(args.config)


if __name__ == "__main__":
    main()
