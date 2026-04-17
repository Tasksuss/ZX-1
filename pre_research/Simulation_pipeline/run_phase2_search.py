import argparse
import csv
import json
import os
import random
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/Users/apple/Downloads/Simulation/outputs/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/Users/apple/Downloads/Simulation/outputs/.cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_utils import ensure_dir, load_config, read_video_gray, save_csv, save_json, set_global_seed, temporal_sample


VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".npy")


def import_torch():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import classification_report
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from torch.utils.data import DataLoader, Dataset
    return torch, nn, optim, classification_report, pack_padded_sequence, pad_packed_sequence, DataLoader, Dataset


def build_entries(root_dir: Path, classes, split_name):
    entries = []
    allowed = set(classes)
    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir() or "-" not in folder.name:
            continue
        subject, label = folder.name.split("-", 1)
        label = label.lower()
        if label not in allowed:
            continue
        for file_path in sorted(folder.iterdir()):
            if file_path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            entries.append({
                "split": split_name,
                "subject": subject,
                "label": label,
                "path": str(file_path),
                "folder": folder.name,
            })
    return entries


def select_balanced_subset(entries, train_size_per_class_subject, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry["subject"], entry["label"])].append(entry)
    subset = []
    for key, items in sorted(grouped.items()):
        rng.shuffle(items)
        subset.extend(items[: min(train_size_per_class_subject, len(items))])
    rng.shuffle(subset)
    return subset


def save_curves(history_rows, output_dir):
    epochs = [row["epoch"] for row in history_rows]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train_loss"] for row in history_rows], marker="o", label="Train Loss")
    plt.plot(epochs, [row["synthetic_test_loss"] for row in history_rows], marker="s", label="Synthetic Test Loss")
    plt.plot(epochs, [row["real_test_loss"] for row in history_rows], marker="^", label="Real Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_vs_epochs.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train_accuracy"] for row in history_rows], marker="o", label="Train Accuracy")
    plt.plot(epochs, [row["synthetic_test_accuracy"] for row in history_rows], marker="s", label="Synthetic Test Accuracy")
    plt.plot(epochs, [row["real_test_accuracy"] for row in history_rows], marker="^", label="Real Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_epochs.png", dpi=220)
    plt.close()


def load_existing_registry(registry_path: Path):
    if not registry_path.exists():
        return []
    with open(registry_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def now_local_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_iso_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def derive_run_timing(run_dir: Path):
    history_path = run_dir / "training_history.csv"
    summary_path = run_dir / "summary.json"
    candidate_paths = [
        history_path,
        run_dir / "best_model.pth",
        run_dir / "loss_vs_epochs.png",
        run_dir / "accuracy_vs_epochs.png",
        summary_path,
    ]
    existing = [path for path in candidate_paths if path.exists()]
    if not existing:
        return None, None, None
    mtimes = sorted(path.stat().st_mtime for path in existing)
    started_at = datetime.fromtimestamp(mtimes[0]).astimezone().isoformat(timespec="seconds")
    finished_at = None
    elapsed_seconds = None
    if summary_path.exists():
        finished_at = datetime.fromtimestamp(summary_path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
        elapsed_seconds = max(0.0, parse_iso_datetime(finished_at).timestamp() - parse_iso_datetime(started_at).timestamp())
    return started_at, finished_at, elapsed_seconds


def write_live_progress(run_dir: Path, payload):
    save_json(payload, run_dir / "live_progress.json")


def estimate_remaining_runtime(runs):
    completed = [run for run in runs if run["status"] == "completed" and run.get("elapsed_seconds") is not None]
    if not completed:
        return {
            "mean_elapsed_seconds_completed": None,
            "eta_remaining_seconds": None,
            "eta_finish_time_local": None,
            "per_size_runtime_estimate": {},
            "eta_state": "warming_up",
        }

    completed_by_size = defaultdict(list)
    seconds_per_sample = []
    for run in completed:
        completed_by_size[int(run["size"])].append(float(run["elapsed_seconds"]))
        train_samples = max(int(run["train_samples"]), 1)
        seconds_per_sample.append(float(run["elapsed_seconds"]) / train_samples)

    per_size_runtime_estimate = {
        str(size): float(sum(values) / len(values))
        for size, values in sorted(completed_by_size.items())
    }
    global_seconds_per_sample = float(sum(seconds_per_sample) / len(seconds_per_sample))

    def estimate_for_run(run):
        size_key = str(run["size"])
        if size_key in per_size_runtime_estimate:
            return per_size_runtime_estimate[size_key]
        return global_seconds_per_sample * max(int(run["train_samples"]), 1)

    remaining = 0.0
    for run in runs:
        if run["status"] == "completed":
            continue
        estimate = estimate_for_run(run)
        if run["status"] == "running" and run.get("started_at"):
            started = parse_iso_datetime(run["started_at"])
            if started is not None:
                spent = max(0.0, datetime.now().astimezone().timestamp() - started.timestamp())
                estimate = max(0.0, estimate - spent)
        remaining += estimate

    eta_finish = None
    if remaining > 0:
        eta_finish = datetime.fromtimestamp(datetime.now().astimezone().timestamp() + remaining).astimezone().isoformat(timespec="minutes")

    return {
        "mean_elapsed_seconds_completed": float(sum(run["elapsed_seconds"] for run in completed) / len(completed)),
        "eta_remaining_seconds": float(remaining),
        "eta_finish_time_local": eta_finish,
        "per_size_runtime_estimate": per_size_runtime_estimate,
        "eta_state": "warming_up" if len(completed) < 2 else "active",
    }


def write_registry(registry_path: Path, rows):
    ensure_dir(registry_path.parent)
    if not rows:
        return
    fieldnames = [
        "stage",
        "experiment_group",
        "dataset_size",
        "repeat_idx",
        "seed",
        "epochs",
        "lr",
        "weight_decay",
        "early_stopping",
        "patience",
        "selection_metric",
        "baseline",
        "changed_from_baseline",
        "change_note",
        "rationale",
        "setting_label",
        "source_setting_group",
        "run_dir",
        "status",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "best_epoch",
        "best_synthetic_test_accuracy",
        "final_synthetic_test_accuracy",
        "best_train_accuracy",
        "final_train_accuracy"
    ]
    with open(registry_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dashboard_status(config, registry_rows):
    phase2 = config["phase2"]
    status_path = Path(phase2["dashboard_status_path"])
    analysis_root = Path(config["analysis_root"])
    runs = []
    completed = 0
    for row in sorted(registry_rows, key=lambda r: (r["stage"], r["experiment_group"], int(r["dataset_size"]), int(r["repeat_idx"]))):
        run_dir = Path(row["run_dir"])
        history_rows = read_csv_rows(run_dir / "training_history.csv")
        latest = history_rows[-1] if history_rows else None
        started_at = row.get("started_at") or ""
        finished_at = row.get("finished_at") or ""
        elapsed_seconds = row.get("elapsed_seconds") or ""
        if not started_at or (row["status"] == "completed" and not finished_at):
            derived_started, derived_finished, derived_elapsed = derive_run_timing(run_dir)
            started_at = started_at or (derived_started or "")
            finished_at = finished_at or (derived_finished or "")
            if not elapsed_seconds and derived_elapsed is not None:
                elapsed_seconds = f"{derived_elapsed:.3f}"
        if row["status"] == "completed":
            completed += 1
        runs.append({
            "group": row["experiment_group"],
            "size": row["dataset_size"],
            "repeat": row["repeat_idx"],
            "status": row["status"],
            "train_samples": int(row["dataset_size"]) * len(config["classes"]) * len(config["synthetic_subjects_train"]),
            "latest_epoch": int(latest["epoch"]) if latest and latest.get("epoch") else None,
            "latest_train_loss": float(latest["train_loss"]) if latest and latest.get("train_loss") else None,
            "latest_real_acc": float(latest["real_test_accuracy"]) if latest and latest.get("real_test_accuracy") else None,
            "latest_synth_acc": float(latest["synthetic_test_accuracy"]) if latest and latest.get("synthetic_test_accuracy") else None,
            "history_rows": len(history_rows),
            "run_dir": row["run_dir"],
            "setting_label": row["setting_label"],
            "change_note": row["change_note"],
            "started_at": started_at or None,
            "finished_at": finished_at or None,
            "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds not in ("", None) else None,
            "loss_plot": f"/file?path={run_dir / 'loss_vs_epochs.png'}" if (run_dir / "loss_vs_epochs.png").exists() else None,
            "acc_plot": f"/file?path={run_dir / 'accuracy_vs_epochs.png'}" if (run_dir / "accuracy_vs_epochs.png").exists() else None,
        })
    eta_payload = estimate_remaining_runtime(runs)
    payload = {
        "expected_runs": len(runs),
        "completed_runs": completed,
        "progress_percent": (completed / len(runs) * 100.0) if runs else 0.0,
        **eta_payload,
        "runs": runs,
        "analysis_files": sorted([p.name for p in analysis_root.glob("*.png")]),
        "config_path": config.get("config_path", ""),
    }
    save_json(payload, status_path)


def run_phase2(config_path: str, stage: str, groups=None, sizes=None, repeats_override=None, smoke=False):
    config = load_config(config_path)
    phase2 = config["phase2"]
    registry_path = Path(phase2["registry_path"])
    analysis_root = ensure_dir(Path(config["analysis_root"]))
    report_root = ensure_dir(Path(config["report_root"]))
    exp_root = ensure_dir(Path(config["experiment_root"]))
    save_json(config, Path(phase2["config_snapshot_path"]))

    settings_groups = phase2["settings_groups"]
    if groups is None:
        groups = ["group_A1"] if stage == "A1" else ["group_B1", "group_B2", "group_B3"]
    if sizes is None:
        if stage == "A1":
            sizes = phase2["stage_a1_sizes"]
        elif stage == "A2":
            sizes = phase2["stage_a2_sizes"]
        elif stage == "C":
            sizes = phase2["stage_c_default_sizes"]
        else:
            sizes = phase2["stage_a1_sizes"]
    repeats = 1 if smoke else (repeats_override or config["num_repeats"])

    set_global_seed(config["seed"])
    torch, nn, optim, classification_report, pack_padded_sequence, pad_packed_sequence, DataLoader, Dataset = import_torch()
    label_to_idx = {label: idx for idx, label in enumerate(config["classes"])}

    class VideoDataset(Dataset):
        def __init__(self, entries, target_frames, cache_in_memory=False):
            self.entries = entries
            self.target_frames = target_frames
            self.cache_in_memory = cache_in_memory
            self.cache = {}
            if self.cache_in_memory:
                for idx, entry in enumerate(self.entries):
                    frames = read_video_gray(entry["path"])
                    frames = temporal_sample(frames, self.target_frames)
                    self.cache[idx] = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, idx):
            entry = self.entries[idx]
            if self.cache_in_memory:
                frames = self.cache[idx].astype(np.float32) / 255.0
            else:
                frames = read_video_gray(entry["path"])
                frames = temporal_sample(frames, self.target_frames)
                frames = np.clip(frames, 0.0, 1.0).astype(np.float32)
            frames_t = torch.from_numpy(frames).unsqueeze(1)
            label = torch.tensor(label_to_idx[entry["label"]], dtype=torch.long)
            length = torch.tensor(self.target_frames, dtype=torch.long)
            return frames_t, label, length, entry["path"]

    class FrameEncoder(nn.Module):
        def __init__(self, img_height, img_width, latent_dim=64):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.GroupNorm(4, 8),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.GroupNorm(4, 16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, img_height, img_width)
                self.flat = self.conv(dummy).view(1, -1).shape[1]
            self.fc = nn.Linear(self.flat, latent_dim)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class VideoModel(nn.Module):
        def __init__(self, img_height, img_width, num_classes):
            super().__init__()
            self.encoder = FrameEncoder(img_height, img_width)
            self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, num_classes),
            )

        def forward(self, x, lengths):
            bsz, seq_len, channels, height, width = x.shape
            encoded = self.encoder(x.view(bsz * seq_len, channels, height, width)).view(bsz, seq_len, -1)
            packed = pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
            mask = (torch.arange(seq_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)
            pooled = (out * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).float()
            return self.fc(self.dropout(pooled))

    def train_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for sequence, labels, lengths, _ in loader:
            sequence, labels, lengths = sequence.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            logits = model(sequence, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)

    def evaluate(model, loader, criterion, device, split_name, output_dir):
        model.eval()
        rows = []
        total_loss = 0.0
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for sequence, labels, lengths, paths in loader:
                sequence, labels, lengths = sequence.to(device), labels.to(device), lengths.to(device)
                logits = model(sequence, lengths)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                preds_all.extend(preds.tolist())
                labels_all.extend(labels_np.tolist())
                for path, truth, pred in zip(paths, labels_np, preds):
                    rows.append({
                        "path": path,
                        "label_true": config["classes"][truth],
                        "label_pred": config["classes"][pred],
                    })
        accuracy = 100.0 * np.mean(np.array(preds_all) == np.array(labels_all))
        report = classification_report(labels_all, preds_all, target_names=config["classes"], output_dict=True, zero_division=0)
        with open(output_dir / f"predictions_{split_name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label_true", "label_pred"])
            writer.writeheader()
            writer.writerows(rows)
        save_json(report, output_dir / f"classification_report_{split_name}.json")
        return total_loss / max(len(loader), 1), accuracy, report

    sim_root = Path(config["sim_output_root"])
    real_root = Path(config["real_data_root"])
    train_entries = build_entries(sim_root / "trainset", config["classes"], "trainset")
    synth_test_entries = build_entries(sim_root / "testset", config["classes"], "testset")
    real_test_entries = build_entries(real_root / "testset", config["classes"], "testset")

    width, height = config["image_size"]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    registry_rows = load_existing_registry(registry_path)
    registry_index = {(row["stage"], row["experiment_group"], row["dataset_size"], row["repeat_idx"]) for row in registry_rows}
    setting_summary = []

    for group_name in groups:
        if group_name not in settings_groups:
            raise ValueError(f"Unknown phase2 group: {group_name}")
        setting = settings_groups[group_name]
        setting_summary.append({
            "stage": stage,
            "experiment_group": group_name,
            "setting_label": setting["setting_label"],
            "epochs": setting["epochs"],
            "lr": setting["lr"],
            "weight_decay": setting["weight_decay"],
            "early_stopping": setting["early_stopping"],
            "patience": setting["patience"],
            "selection_metric": setting["selection_metric"],
            "changed_from_baseline": setting["changed_from_baseline"],
            "change_note": setting["change_note"],
            "rationale": setting["rationale"],
        })

        for size in sizes:
            for repeat_idx in range(repeats):
                seed = config["seed"] + size * 1000 + repeat_idx * 10 + sum(ord(ch) for ch in group_name)
                key = (stage, group_name, str(size), str(repeat_idx))
                run_dir = ensure_dir(exp_root / group_name / f"size_{size}" / f"repeat_{repeat_idx}")

                config_snapshot = {
                    "stage": stage,
                    "experiment_group": group_name,
                    "setting_label": setting["setting_label"],
                    "dataset_size": size,
                    "repeat_idx": repeat_idx,
                    "seed": seed,
                    "epochs": setting["epochs"],
                    "lr": setting["lr"],
                    "weight_decay": setting["weight_decay"],
                    "early_stopping": setting["early_stopping"],
                    "patience": setting["patience"],
                    "selection_metric": setting["selection_metric"],
                    "cache_train_in_memory": config.get("cache_train_in_memory", True),
                    "baseline": config["phase1_baseline"],
                }
                change_log = {
                    "baseline": config["phase1_baseline"],
                    "change_applied": setting["changed_from_baseline"],
                    "rationale": setting["rationale"],
                    "change_note": setting["change_note"],
                    "runtime_optimization_note": "train_dataset_cache: off -> on" if config.get("cache_train_in_memory", True) else "train_dataset_cache: off",
                    "source_setting_group": group_name,
                    "size_search_stage": stage,
                }
                save_json(config_snapshot, run_dir / "config_snapshot.json")
                save_json(change_log, run_dir / "change_log.json")

                if key not in registry_index:
                    registry_rows.append({
                        "stage": stage,
                        "experiment_group": group_name,
                        "dataset_size": str(size),
                        "repeat_idx": str(repeat_idx),
                        "seed": str(seed),
                        "epochs": str(setting["epochs"]),
                        "lr": str(setting["lr"]),
                        "weight_decay": str(setting["weight_decay"]),
                        "early_stopping": str(setting["early_stopping"]),
                        "patience": str(setting["patience"]),
                        "selection_metric": setting["selection_metric"],
                        "baseline": json.dumps(config["phase1_baseline"]),
                        "changed_from_baseline": setting["changed_from_baseline"],
                        "change_note": setting["change_note"],
                        "rationale": setting["rationale"],
                        "setting_label": setting["setting_label"],
                        "source_setting_group": group_name,
                        "run_dir": str(run_dir),
                        "status": "pending",
                        "started_at": "",
                        "finished_at": "",
                        "elapsed_seconds": "",
                        "best_epoch": "",
                        "best_synthetic_test_accuracy": "",
                        "final_synthetic_test_accuracy": "",
                        "best_train_accuracy": "",
                        "final_train_accuracy": "",
                    })
                    registry_index.add(key)

    sorted_registry = sorted(registry_rows, key=lambda r: (r["stage"], r["experiment_group"], int(r["dataset_size"]), int(r["repeat_idx"])))
    write_registry(registry_path, sorted_registry)
    write_dashboard_status({**config, "config_path": config_path}, sorted_registry)

    for group_name in groups:
        setting = settings_groups[group_name]
        for size in sizes:
            for repeat_idx in range(repeats):
                key = (stage, group_name, str(size), str(repeat_idx))
                run_dir = ensure_dir(exp_root / group_name / f"size_{size}" / f"repeat_{repeat_idx}")
                if (run_dir / "summary.json").exists():
                    continue

                seed = config["seed"] + size * 1000 + repeat_idx * 10 + sum(ord(ch) for ch in group_name)
                set_global_seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                start_iso = now_local_iso()
                existing_row = next((row for row in registry_rows if (row["stage"], row["experiment_group"], row["dataset_size"], row["repeat_idx"]) == key), None)
                registry_rows = [row for row in registry_rows if (row["stage"], row["experiment_group"], row["dataset_size"], row["repeat_idx"]) != key]
                registry_rows.append({
                    "stage": stage,
                    "experiment_group": group_name,
                    "dataset_size": str(size),
                    "repeat_idx": str(repeat_idx),
                    "seed": str(seed),
                    "epochs": str(setting["epochs"]),
                    "lr": str(setting["lr"]),
                    "weight_decay": str(setting["weight_decay"]),
                    "early_stopping": str(setting["early_stopping"]),
                    "patience": str(setting["patience"]),
                    "selection_metric": setting["selection_metric"],
                    "baseline": json.dumps(config["phase1_baseline"]),
                    "changed_from_baseline": setting["changed_from_baseline"],
                    "change_note": setting["change_note"],
                    "rationale": setting["rationale"],
                    "setting_label": setting["setting_label"],
                    "source_setting_group": group_name,
                    "run_dir": str(run_dir),
                    "status": "running",
                    "started_at": existing_row.get("started_at") if existing_row and existing_row.get("started_at") else start_iso,
                    "finished_at": "",
                    "elapsed_seconds": "",
                    "best_epoch": existing_row.get("best_epoch", "") if existing_row else "",
                    "best_synthetic_test_accuracy": existing_row.get("best_synthetic_test_accuracy", "") if existing_row else "",
                    "final_synthetic_test_accuracy": existing_row.get("final_synthetic_test_accuracy", "") if existing_row else "",
                    "best_train_accuracy": existing_row.get("best_train_accuracy", "") if existing_row else "",
                    "final_train_accuracy": existing_row.get("final_train_accuracy", "") if existing_row else "",
                })
                sorted_registry = sorted(registry_rows, key=lambda r: (r["stage"], r["experiment_group"], int(r["dataset_size"]), int(r["repeat_idx"])))
                write_registry(registry_path, sorted_registry)
                write_dashboard_status({**config, "config_path": config_path}, sorted_registry)

                selected_train = select_balanced_subset(train_entries, size, seed)
                train_cache_enabled = config.get("cache_train_in_memory", True)
                train_ds = VideoDataset(selected_train, config["target_frame_count"], cache_in_memory=train_cache_enabled)
                synth_test_ds = VideoDataset(synth_test_entries, config["target_frame_count"], cache_in_memory=True)
                real_test_ds = VideoDataset(real_test_entries, config["target_frame_count"], cache_in_memory=True)

                loader_kwargs = {
                    "num_workers": config["num_workers"],
                    "pin_memory": device.type != "cpu",
                }
                train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, **loader_kwargs)
                synth_loader = DataLoader(synth_test_ds, batch_size=config["batch_size"], shuffle=False, **loader_kwargs)
                real_loader = DataLoader(real_test_ds, batch_size=config["batch_size"], shuffle=False, **loader_kwargs)

                model = VideoModel(height, width, len(config["classes"])).to(device)
                optimizer = optim.Adam(model.parameters(), lr=setting["lr"], weight_decay=setting["weight_decay"])
                criterion = nn.CrossEntropyLoss()

                best_metric = float("inf")
                best_epoch = -1
                wait_count = 0
                history_rows = []
                epochs = 2 if smoke else setting["epochs"]
                metric_mode = setting["selection_metric"]
                write_live_progress(run_dir, {
                    "status": "running",
                    "current_epoch": 0,
                    "total_epochs": epochs,
                    "best_epoch": None,
                    "best_metric": None,
                    "updated_at": now_local_iso(),
                })
                for epoch in range(epochs):
                    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                    synth_loss, synth_acc, _ = evaluate(model, synth_loader, criterion, device, "synthetic", run_dir)
                    real_loss, real_acc, _ = evaluate(model, real_loader, criterion, device, "real", run_dir)
                    history_rows.append({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "synthetic_test_loss": synth_loss,
                        "synthetic_test_accuracy": synth_acc,
                        "real_test_loss": real_loss,
                        "real_test_accuracy": real_acc,
                    })
                    current_metric = synth_loss if metric_mode == "synthetic_test_loss" else real_loss
                    improved = current_metric < best_metric
                    if improved:
                        best_metric = current_metric
                        best_epoch = epoch + 1
                        wait_count = 0
                        torch.save(model.state_dict(), run_dir / "best_model.pth")
                    else:
                        wait_count += 1
                    write_live_progress(run_dir, {
                        "status": "running",
                        "current_epoch": epoch + 1,
                        "total_epochs": epochs,
                        "best_epoch": best_epoch if best_epoch > 0 else None,
                        "best_metric": best_metric if best_metric != float("inf") else None,
                        "latest_train_loss": train_loss,
                        "latest_train_accuracy": train_acc,
                        "latest_synthetic_test_loss": synth_loss,
                        "latest_synthetic_test_accuracy": synth_acc,
                        "latest_real_test_loss": real_loss,
                        "latest_real_test_accuracy": real_acc,
                        "updated_at": now_local_iso(),
                    })
                    if setting["early_stopping"] and wait_count >= setting["patience"]:
                        break

                with open(run_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(history_rows)
                save_curves(history_rows, run_dir)

                model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
                synth_loss, synth_acc, synth_report = evaluate(model, synth_loader, criterion, device, "synthetic", run_dir)
                real_loss, real_acc, real_report = evaluate(model, real_loader, criterion, device, "real", run_dir)

                summary = {
                    "stage": stage,
                    "experiment_group": group_name,
                    "setting_label": setting["setting_label"],
                    "train_size_per_class_subject": size,
                    "repeat_idx": repeat_idx,
                    "seed": seed,
                    "train_samples": len(selected_train),
                    "synthetic_test_loss": synth_loss,
                    "synthetic_test_accuracy": synth_acc,
                    "real_test_loss": real_loss,
                    "real_test_accuracy": real_acc,
                    "synthetic_macro_f1": synth_report["macro avg"]["f1-score"],
                    "real_macro_f1": real_report["macro avg"]["f1-score"],
                    "best_epoch": best_epoch,
                    "best_train_accuracy": max(row["train_accuracy"] for row in history_rows),
                    "final_train_accuracy": history_rows[-1]["train_accuracy"],
                    "changed_from_baseline": setting["changed_from_baseline"],
                    "change_note": setting["change_note"],
                    "selection_metric": setting["selection_metric"],
                }
                save_json(summary, run_dir / "summary.json")
                write_live_progress(run_dir, {
                    "status": "completed",
                    "current_epoch": len(history_rows),
                    "total_epochs": epochs,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric if best_metric != float("inf") else None,
                    "latest_train_loss": history_rows[-1]["train_loss"],
                    "latest_train_accuracy": history_rows[-1]["train_accuracy"],
                    "latest_synthetic_test_loss": history_rows[-1]["synthetic_test_loss"],
                    "latest_synthetic_test_accuracy": history_rows[-1]["synthetic_test_accuracy"],
                    "latest_real_test_loss": history_rows[-1]["real_test_loss"],
                    "latest_real_test_accuracy": history_rows[-1]["real_test_accuracy"],
                    "updated_at": now_local_iso(),
                })

                finish_iso = now_local_iso()
                started_dt = parse_iso_datetime(next((row["started_at"] for row in registry_rows if (row["stage"], row["experiment_group"], row["dataset_size"], row["repeat_idx"]) == key), ""))
                elapsed_seconds = None
                if started_dt is not None:
                    elapsed_seconds = max(0.0, parse_iso_datetime(finish_iso).timestamp() - started_dt.timestamp())

                registry_rows = [row for row in registry_rows if (row["stage"], row["experiment_group"], row["dataset_size"], row["repeat_idx"]) != key]
                registry_rows.append({
                    "stage": stage,
                    "experiment_group": group_name,
                    "dataset_size": str(size),
                    "repeat_idx": str(repeat_idx),
                    "seed": str(seed),
                    "epochs": str(setting["epochs"]),
                    "lr": str(setting["lr"]),
                    "weight_decay": str(setting["weight_decay"]),
                    "early_stopping": str(setting["early_stopping"]),
                    "patience": str(setting["patience"]),
                    "selection_metric": setting["selection_metric"],
                    "baseline": json.dumps(config["phase1_baseline"]),
                    "changed_from_baseline": setting["changed_from_baseline"],
                    "change_note": setting["change_note"],
                    "rationale": setting["rationale"],
                    "setting_label": setting["setting_label"],
                    "source_setting_group": group_name,
                    "run_dir": str(run_dir),
                    "status": "completed",
                    "started_at": started_dt.isoformat(timespec="seconds") if started_dt is not None else "",
                    "finished_at": finish_iso,
                    "elapsed_seconds": f"{elapsed_seconds:.3f}" if elapsed_seconds is not None else "",
                    "best_epoch": str(best_epoch),
                    "best_synthetic_test_accuracy": str(max(row["synthetic_test_accuracy"] for row in history_rows)),
                    "final_synthetic_test_accuracy": str(summary["synthetic_test_accuracy"]),
                    "best_train_accuracy": str(summary["best_train_accuracy"]),
                    "final_train_accuracy": str(summary["final_train_accuracy"]),
                })
                registry_index.add(key)
                sorted_registry = sorted(registry_rows, key=lambda r: (r["stage"], r["experiment_group"], int(r["dataset_size"]), int(r["repeat_idx"])))
                write_registry(registry_path, sorted_registry)
                write_dashboard_status({**config, "config_path": config_path}, sorted_registry)

    save_csv(setting_summary, Path(phase2["setting_comparison_path"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/Users/apple/Downloads/Simulation/config_phase2.json")
    parser.add_argument("--stage", default="A1", choices=["A1", "A2", "B", "C"])
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--sizes", nargs="*", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_phase2(args.config, stage=args.stage, groups=args.groups, sizes=args.sizes, repeats_override=args.repeats, smoke=args.smoke)


if __name__ == "__main__":
    main()
