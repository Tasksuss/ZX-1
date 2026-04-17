import argparse
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/Users/apple/Downloads/Simulation/outputs/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/Users/apple/Downloads/Simulation/outputs/.cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline_utils import (
    compute_basic_frame_stats,
    ensure_dir,
    ffprobe_metadata,
    iter_dataset_samples,
    load_config,
    pick_evenly_spaced,
    read_video_gray,
    save_csv,
    save_json,
    set_global_seed,
)


def analyze_dataset(config_path: str) -> None:
    config = load_config(config_path)
    set_global_seed(config["seed"])
    data_root = Path(config["real_data_root"])
    output_dir = ensure_dir(Path(config["real_profile_dir"]))
    ref_dir = ensure_dir(output_dir / "reference_frames")

    sample_rows = []
    class_counts = Counter()
    split_counts = Counter()
    subject_counts = Counter()
    class_subject_counts = defaultdict(int)
    stats_by_label = defaultdict(list)
    duration_values = []
    frame_count_values = []
    fps_values = []
    brightness_values = []
    motion_values = []

    samples = list(iter_dataset_samples(data_root, config["classes"]))
    for sample in samples:
        meta = ffprobe_metadata(sample["path"])
        frames = read_video_gray(sample["path"], max_frames=24)
        basic = compute_basic_frame_stats(frames)

        row = {
            **sample,
            **meta,
            **basic,
        }
        sample_rows.append(row)
        class_counts[sample["label"]] += 1
        split_counts[sample["split"]] += 1
        subject_counts[sample["subject"]] += 1
        class_subject_counts[(sample["split"], sample["subject"], sample["label"])] += 1
        stats_by_label[sample["label"]].append(row)
        duration_values.append(row["duration"])
        frame_count_values.append(row["nb_frames"])
        fps_values.append(row["fps"])
        brightness_values.append(row["brightness_mean"])
        motion_values.append(row["motion_energy_mean"])

    count_rows = []
    for (split, subject, label), count in sorted(class_subject_counts.items()):
        count_rows.append({
            "split": split,
            "subject": subject,
            "label": label,
            "count": count,
        })
    save_csv(count_rows, output_dir / "class_subject_counts.csv")
    save_csv(sample_rows, output_dir / "sample_metrics.csv")

    profile = {
        "real_data_root": str(data_root),
        "classes": config["classes"],
        "image_size_mode": {
            "width": int(np.median([row["width"] for row in sample_rows])),
            "height": int(np.median([row["height"] for row in sample_rows])),
        },
        "video_stats": {
            "duration_mean": float(np.mean(duration_values)),
            "duration_std": float(np.std(duration_values)),
            "frame_count_mean": float(np.mean(frame_count_values)),
            "frame_count_std": float(np.std(frame_count_values)),
            "fps_mean": float(np.mean(fps_values)),
            "fps_std": float(np.std(fps_values)),
        },
        "global_visual_stats": {
            "brightness_mean": float(np.mean(brightness_values)),
            "brightness_std": float(np.std(brightness_values)),
            "motion_energy_mean": float(np.mean(motion_values)),
            "motion_energy_std": float(np.std(motion_values)),
        },
        "split_counts": dict(split_counts),
        "subject_counts": dict(subject_counts),
        "class_counts": dict(class_counts),
        "class_profiles": {},
    }

    for label, rows in stats_by_label.items():
        profile["class_profiles"][label] = {
            "count": len(rows),
            "duration_mean": float(np.mean([r["duration"] for r in rows])),
            "duration_std": float(np.std([r["duration"] for r in rows])),
            "frame_count_mean": float(np.mean([r["nb_frames"] for r in rows])),
            "frame_count_std": float(np.std([r["nb_frames"] for r in rows])),
            "brightness_mean": float(np.mean([r["brightness_mean"] for r in rows])),
            "brightness_std": float(np.std([r["brightness_mean"] for r in rows])),
            "motion_energy_mean": float(np.mean([r["motion_energy_mean"] for r in rows])),
            "motion_energy_std": float(np.std([r["motion_energy_mean"] for r in rows])),
            "activity_center_x_mean": float(np.mean([r["activity_center_x"] for r in rows])),
            "activity_center_y_mean": float(np.mean([r["activity_center_y"] for r in rows])),
        }

    save_json(profile, output_dir / "dataset_profile.json")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    labels = list(class_counts.keys())
    counts = [class_counts[label] for label in labels]
    plt.bar(labels, counts)
    plt.xticks(rotation=30, ha="right")
    plt.title("Class Counts")

    plt.subplot(2, 2, 2)
    plt.hist(duration_values, bins=10, color="#4c78a8")
    plt.title("Duration Distribution")
    plt.xlabel("Seconds")

    plt.subplot(2, 2, 3)
    plt.hist(brightness_values, bins=10, color="#f58518")
    plt.title("Brightness Mean Distribution")
    plt.xlabel("Mean Intensity")

    plt.subplot(2, 2, 4)
    plt.hist(motion_values, bins=10, color="#54a24b")
    plt.title("Motion Energy Distribution")
    plt.xlabel("Mean Frame Difference")

    plt.tight_layout()
    plt.savefig(output_dir / "profile_plots.png", dpi=220, bbox_inches="tight")
    plt.close()

    for label in config["classes"]:
        label_samples = [row for row in sample_rows if row["label"] == label]
        chosen = pick_evenly_spaced(label_samples, limit=2)
        for sample in chosen:
            frames = read_video_gray(sample["path"], max_frames=24)
            mid_idx = len(frames) // 2
            mean_frame = frames.mean(axis=0)
            diff_frame = np.abs(np.diff(frames, axis=0)).mean(axis=0) if len(frames) > 1 else np.zeros_like(frames[0])

            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].imshow(frames[mid_idx], cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Mid Frame")
            axes[1].imshow(mean_frame, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Time Mean")
            axes[2].imshow(diff_frame, cmap="magma")
            axes[2].set_title("Mean Diff")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{sample['folder']} | {Path(sample['path']).name}", fontsize=9)
            fig.tight_layout()
            fig.savefig(ref_dir / f"{sample['folder']}_{Path(sample['path']).stem}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/Users/apple/Downloads/Simulation/config.json")
    args = parser.parse_args()
    analyze_dataset(args.config)


if __name__ == "__main__":
    main()
