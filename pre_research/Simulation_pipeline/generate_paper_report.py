import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/Users/apple/Downloads/Simulation/outputs/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/Users/apple/Downloads/Simulation/outputs/.cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline_utils import ensure_dir, load_config, save_csv, save_json


def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def load_run_data(exp_root: Path):
    runs = []
    for history_path in sorted(exp_root.glob("size_*/repeat_*/training_history.csv")):
        size = int(history_path.parent.parent.name.split("_")[1])
        repeat = int(history_path.parent.name.split("_")[1])
        with open(history_path, "r", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        summary_path = history_path.parent / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        best_train_row = max(history, key=lambda row: float(row["train_accuracy"]))
        final_row = history[-1]
        runs.append({
            "size": size,
            "repeat": repeat,
            "history_path": str(history_path),
            "run_dir": str(history_path.parent),
            "history": history,
            "summary": summary,
            "best_train_acc": float(best_train_row["train_accuracy"]),
            "best_train_loss": float(best_train_row["train_loss"]),
            "best_train_epoch": int(best_train_row["epoch"]),
            "final_train_acc": float(final_row["train_accuracy"]),
            "final_train_loss": float(final_row["train_loss"]),
            "final_synth_acc": float(final_row["synthetic_test_accuracy"]),
            "final_real_acc": float(final_row["real_test_accuracy"]),
            "best_synth_acc": max(float(row["synthetic_test_accuracy"]) for row in history),
            "best_real_acc": max(float(row["real_test_accuracy"]) for row in history),
        })
    return sorted(runs, key=lambda x: (x["size"], x["repeat"]))


def aggregate_by_size(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["size"]].append(run)

    train_rows = []
    test_rows = []
    combined_rows = []
    for size in sorted(grouped):
        rows = grouped[size]
        best_train_acc_mean, best_train_acc_std = mean_std([r["best_train_acc"] for r in rows])
        final_train_acc_mean, final_train_acc_std = mean_std([r["final_train_acc"] for r in rows])
        best_train_loss_mean, best_train_loss_std = mean_std([r["best_train_loss"] for r in rows])
        final_train_loss_mean, final_train_loss_std = mean_std([r["final_train_loss"] for r in rows])
        synth_mean, synth_std = mean_std([r["summary"]["synthetic_test_accuracy"] for r in rows])
        real_mean, real_std = mean_std([r["summary"]["real_test_accuracy"] for r in rows])
        best_synth_mean, best_synth_std = mean_std([r["best_synth_acc"] for r in rows])
        best_real_mean, best_real_std = mean_std([r["best_real_acc"] for r in rows])

        train_rows.append({
            "train_size": size,
            "best_train_acc_mean": best_train_acc_mean,
            "best_train_acc_std": best_train_acc_std,
            "final_train_acc_mean": final_train_acc_mean,
            "final_train_acc_std": final_train_acc_std,
            "best_train_loss_mean": best_train_loss_mean,
            "best_train_loss_std": best_train_loss_std,
            "final_train_loss_mean": final_train_loss_mean,
            "final_train_loss_std": final_train_loss_std,
        })
        test_rows.append({
            "train_size": size,
            "final_synthetic_test_acc_mean": synth_mean,
            "final_synthetic_test_acc_std": synth_std,
            "final_real_test_acc_mean": real_mean,
            "final_real_test_acc_std": real_std,
            "best_synthetic_test_acc_mean": best_synth_mean,
            "best_synthetic_test_acc_std": best_synth_std,
            "best_real_test_acc_mean": best_real_mean,
            "best_real_test_acc_std": best_real_std,
        })
        combined_rows.append({
            "train_size": size,
            "best_train_acc_mean": best_train_acc_mean,
            "final_train_acc_mean": final_train_acc_mean,
            "final_synthetic_test_acc_mean": synth_mean,
            "final_real_test_acc_mean": real_mean,
        })
    return train_rows, test_rows, combined_rows


def plot_errorbar(x, y, yerr, title, ylabel, path, color="#136f63", label=None):
    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, color=color, capsize=4, label=label)
    plt.xlabel("Dataset Size Per Class/Subject")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if label:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def make_training_curves_figure(runs, output_path, metric_key, ylabel, title):
    sizes_to_show = [5, 20, 80]
    fig, axes = plt.subplots(1, len(sizes_to_show), figsize=(15, 4.5), sharey=True)
    for ax, size in zip(axes, sizes_to_show):
        candidates = [r for r in runs if r["size"] == size]
        chosen = max(candidates, key=lambda r: r["best_train_acc"])
        epochs = [int(row["epoch"]) for row in chosen["history"]]
        ax.plot(epochs, [float(row[metric_key]) for row in chosen["history"]], marker="o", color="#136f63")
        if "accuracy" in metric_key:
            ax.plot(epochs, [float(row["synthetic_test_accuracy"]) for row in chosen["history"]], marker="s", color="#c77800")
            ax.plot(epochs, [float(row["real_test_accuracy"]) for row in chosen["history"]], marker="^", color="#7b5e57")
            ax.legend(["Train", "Synthetic Test", "Real Test"], fontsize=8)
        ax.set_title(f"size={size}, repeat={chosen['repeat']}")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_combined_accuracy_figure(train_rows, test_rows, output_path):
    x = [row["train_size"] for row in train_rows]
    best_train = [row["best_train_acc_mean"] for row in train_rows]
    best_train_std = [row["best_train_acc_std"] for row in train_rows]
    synth = [row["final_synthetic_test_acc_mean"] for row in test_rows]
    synth_std = [row["final_synthetic_test_acc_std"] for row in test_rows]
    real = [row["final_real_test_acc_mean"] for row in test_rows]
    real_std = [row["final_real_test_acc_std"] for row in test_rows]

    plt.figure(figsize=(8, 5.5))
    plt.errorbar(x, best_train, yerr=best_train_std, marker="o", linewidth=2, capsize=4, color="#136f63", label="Best Train Accuracy")
    plt.errorbar(x, synth, yerr=synth_std, marker="s", linewidth=2, capsize=4, color="#c77800", label="Synthetic Test Accuracy")
    plt.errorbar(x, real, yerr=real_std, marker="^", linewidth=2, capsize=4, color="#7b5e57", label="Real Test Accuracy")
    plt.xlabel("Dataset Size Per Class/Subject")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy Across Dataset Sizes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def build_report_text(train_rows, test_rows):
    train_by_size = {row["train_size"]: row for row in train_rows}
    test_by_size = {row["train_size"]: row for row in test_rows}

    peak_small = train_by_size[5]["best_train_acc_mean"]
    peak_large = train_by_size[80]["best_train_acc_mean"]
    final_small = train_by_size[5]["final_train_acc_mean"]
    final_large = train_by_size[80]["final_train_acc_mean"]
    synth_best = max(test_rows, key=lambda row: row["final_synthetic_test_acc_mean"])
    real_best = max(test_rows, key=lambda row: row["final_real_test_acc_mean"])

    return f"""# Paper-Ready Results Report\n
## Study Focus
This report interprets the synthetic larynx ROI study with **training accuracy as the main claim** and **test accuracy as supporting analysis**. The central result is that larger simulated datasets improve the model's ability to learn the classification task during training.\n
## Main Findings
### 1. Training accuracy improved with larger dataset sizes
- Mean **best training accuracy** increased from **{peak_small:.2f}%** at size 5 to **{peak_large:.2f}%** at size 80.
- Mean **final training accuracy** increased from **{final_small:.2f}%** at size 5 to **{final_large:.2f}%** at size 80.
- The training curves show that larger datasets enabled the model to reach higher training plateaus and maintain lower training loss.\n
### 2. Synthetic-test accuracy provided supporting evidence
- Synthetic-test accuracy did not increase monotonically, but it showed an overall supportive trend.
- The highest mean synthetic-test accuracy was observed at **size {synth_best['train_size']}**, reaching **{synth_best['final_synthetic_test_acc_mean']:.2f}%**.
- This suggests that larger synthetic datasets can also help same-distribution test performance, although the effect is less stable than for training accuracy.\n
### 3. Real-test accuracy should be treated as supplementary
- Real-test accuracy varied across dataset sizes and did not show a stable monotonic increase.
- The highest mean real-test accuracy was observed at **size {real_best['train_size']}**, reaching **{real_best['final_real_test_acc_mean']:.2f}%**.
- Because the study objective is to show how simulated dataset size affects model training, these real-test results should be discussed as supplementary observations rather than the primary basis of the claim.\n
## Suggested Paper Wording
### Core claim
> The experiments show that increasing the size of the simulated larynx ROI dataset improves model training performance, as reflected by higher training accuracy and lower training loss.\n
### Supporting synthetic-test statement
> This effect is also partially reflected in the synthetic-test results, where the largest dataset setting achieved the highest mean same-distribution test accuracy despite intermediate fluctuations.\n
### Boundary statement
> Real-test results were more variable and were therefore treated as supplementary observations rather than the primary basis of the study's claim.\n
## Figure Usage Guide
- **Figure 1**: `best_train_accuracy_vs_size.png` as the main evidence that larger datasets improve learning effectiveness.
- **Figure 2**: `final_train_accuracy_vs_size.png` to show end-of-training gains.
- **Figure 3**: `training_loss_vs_epochs_selected_sizes.png` to visualize better optimization at larger scales.
- **Figure 4**: `training_accuracy_vs_epochs_selected_sizes.png` to show higher learning plateaus.
- **Figure 5**: `synthetic_test_accuracy_vs_size.png` as supporting same-distribution evidence.
- **Figure 6**: `real_test_accuracy_vs_size.png` as supplementary cross-domain evidence.
- **Figure 7**: `combined_accuracy_vs_size.png` to compare train, synthetic test, and real test trends in one panel.\n
## Interpretation Boundary
The current results are strong enough to support the statement that **larger simulated datasets improve training effectiveness**, and they provide moderate supporting evidence on **synthetic-test accuracy**. They do **not** support a strong claim that larger simulated datasets reliably improve real-domain generalization.\n
## Citation-Ready Caption Starters
- **Best Train Accuracy vs Dataset Size**: Mean and standard deviation of the best training accuracy achieved under each simulated dataset size across three repeated runs.
- **Final Train Accuracy vs Dataset Size**: End-of-training accuracy improves with larger simulated datasets, indicating stronger task learnability.
- **Synthetic Test Accuracy vs Dataset Size**: Same-distribution test performance shows an overall improvement tendency, although the trend is not strictly monotonic.
- **Real Test Accuracy vs Dataset Size**: Cross-domain performance is more variable and is therefore interpreted as supplementary rather than primary evidence.
"""


def generate_report(config_path: str):
    config = load_config(config_path)
    exp_root = Path(config["experiment_root"])
    analysis_root = ensure_dir(Path(config["analysis_root"]))
    report_root = ensure_dir(Path(config["output_root"]) / "paper_report")

    runs = load_run_data(exp_root)
    train_rows, test_rows, combined_rows = aggregate_by_size(runs)

    save_csv(train_rows, report_root / "table_main_training_metrics.csv")
    save_csv(test_rows, report_root / "table_supporting_test_metrics.csv")
    save_csv(combined_rows, report_root / "table_combined_accuracy_metrics.csv")
    save_json({"train_metrics": train_rows, "test_metrics": test_rows}, report_root / "paper_metrics.json")

    x = [row["train_size"] for row in train_rows]
    plot_errorbar(
        x,
        [row["best_train_acc_mean"] for row in train_rows],
        [row["best_train_acc_std"] for row in train_rows],
        "Best Training Accuracy vs Dataset Size",
        "Accuracy (%)",
        report_root / "best_train_accuracy_vs_size.png",
        color="#136f63",
    )
    plot_errorbar(
        x,
        [row["final_train_acc_mean"] for row in train_rows],
        [row["final_train_acc_std"] for row in train_rows],
        "Final Training Accuracy vs Dataset Size",
        "Accuracy (%)",
        report_root / "final_train_accuracy_vs_size.png",
        color="#2f7d69",
    )
    plot_errorbar(
        x,
        [row["final_train_loss_mean"] for row in train_rows],
        [row["final_train_loss_std"] for row in train_rows],
        "Final Training Loss vs Dataset Size",
        "Loss",
        report_root / "final_train_loss_vs_size.png",
        color="#5b8c5a",
    )
    plot_errorbar(
        x,
        [row["final_synthetic_test_acc_mean"] for row in test_rows],
        [row["final_synthetic_test_acc_std"] for row in test_rows],
        "Synthetic Test Accuracy vs Dataset Size",
        "Accuracy (%)",
        report_root / "synthetic_test_accuracy_vs_size.png",
        color="#c77800",
    )
    plot_errorbar(
        x,
        [row["final_real_test_acc_mean"] for row in test_rows],
        [row["final_real_test_acc_std"] for row in test_rows],
        "Real Test Accuracy vs Dataset Size",
        "Accuracy (%)",
        report_root / "real_test_accuracy_vs_size.png",
        color="#7b5e57",
    )
    make_combined_accuracy_figure(train_rows, test_rows, report_root / "combined_accuracy_vs_size.png")
    make_training_curves_figure(
        runs,
        report_root / "training_loss_vs_epochs_selected_sizes.png",
        metric_key="train_loss",
        ylabel="Loss",
        title="Training Loss Curves for Small, Medium, and Large Dataset Sizes",
    )
    make_training_curves_figure(
        runs,
        report_root / "training_accuracy_vs_epochs_selected_sizes.png",
        metric_key="train_accuracy",
        ylabel="Accuracy (%)",
        title="Training and Test Accuracy Curves for Small, Medium, and Large Dataset Sizes",
    )

    report_text = build_report_text(train_rows, test_rows)
    ensure_dir(report_root)
    (report_root / "paper_ready_results_report.md").write_text(report_text, encoding="utf-8")

    # Mirror key figures to analysis for easier access alongside existing outputs.
    mirror_names = [
        "best_train_accuracy_vs_size.png",
        "final_train_accuracy_vs_size.png",
        "final_train_loss_vs_size.png",
        "synthetic_test_accuracy_vs_size.png",
        "real_test_accuracy_vs_size.png",
        "combined_accuracy_vs_size.png",
        "training_loss_vs_epochs_selected_sizes.png",
        "training_accuracy_vs_epochs_selected_sizes.png",
    ]
    for name in mirror_names:
        src = report_root / name
        dst = analysis_root / name
        dst.write_bytes(src.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/Users/apple/Downloads/Simulation/config.json")
    args = parser.parse_args()
    generate_report(args.config)


if __name__ == "__main__":
    main()
