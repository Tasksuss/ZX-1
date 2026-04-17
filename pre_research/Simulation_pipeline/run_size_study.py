import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/Users/apple/Downloads/Simulation/outputs/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/Users/apple/Downloads/Simulation/outputs/.cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_utils import (
    ensure_dir,
    iter_dataset_samples,
    load_config,
    read_video_gray,
    save_json,
    set_global_seed,
    temporal_sample,
)


def build_entries(root_dir: Path, classes, split_name=None):
    if split_name is None:
        return list(iter_dataset_samples(root_dir, classes))
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
            if file_path.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv", ".npy"):
                continue
            entries.append({
                "split": split_name,
                "subject": subject,
                "label": label,
                "path": str(file_path),
                "folder": folder.name,
            })
    return entries


def select_balanced_subset(entries, labels, train_size_per_class_subject, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry["subject"], entry["label"])].append(entry)
    subset = []
    for key, items in grouped.items():
        rng.shuffle(items)
        subset.extend(items[: min(train_size_per_class_subject, len(items))])
    rng.shuffle(subset)
    return subset


def import_torch():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import classification_report
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from torch.utils.data import DataLoader, Dataset
    return torch, nn, optim, classification_report, pack_padded_sequence, pad_packed_sequence, DataLoader, Dataset


def run_experiments(config_path: str, sizes=None, repeats=None, smoke=False) -> None:
    config = load_config(config_path)
    sizes = sizes or config["dataset_sizes"]
    repeats = repeats or config["num_repeats"]
    set_global_seed(config["seed"])
    torch, nn, optim, classification_report, pack_padded_sequence, pad_packed_sequence, DataLoader, Dataset = import_torch()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(config["output_root"]) / ".mplconfig"))

    label_to_idx = {label: idx for idx, label in enumerate(config["classes"])}

    class VideoDataset(Dataset):
        def __init__(self, entries, target_frames, image_size, cache_in_memory=False):
            self.entries = entries
            self.target_frames = target_frames
            self.image_size = image_size
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

    sim_root = Path(config["sim_output_root"])
    real_root = Path(config["real_data_root"])
    exp_root = ensure_dir(Path(config["experiment_root"]))

    train_entries = build_entries(sim_root / "trainset", config["classes"], split_name="trainset")
    synth_test_entries = build_entries(sim_root / "testset", config["classes"], split_name="testset")
    real_test_entries = build_entries(real_root / "testset", config["classes"], split_name="testset")
    width, height = config["image_size"]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    all_summaries = []
    for size in sizes:
        repeat_range = [0] if smoke else list(range(repeats))
        for repeat_idx in repeat_range:
            seed = config["seed"] + size * 100 + repeat_idx
            set_global_seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            run_dir = ensure_dir(exp_root / f"size_{size}" / f"repeat_{repeat_idx}")

            selected_train = select_balanced_subset(train_entries, config["classes"], size, seed)
            train_ds = VideoDataset(selected_train, config["target_frame_count"], config["image_size"])
            synth_test_ds = VideoDataset(synth_test_entries, config["target_frame_count"], config["image_size"], cache_in_memory=True)
            real_test_ds = VideoDataset(real_test_entries, config["target_frame_count"], config["image_size"], cache_in_memory=True)

            loader_kwargs = {
                "num_workers": config["num_workers"],
                "pin_memory": device.type != "cpu",
            }
            train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, **loader_kwargs)
            synth_loader = DataLoader(synth_test_ds, batch_size=config["batch_size"], shuffle=False, **loader_kwargs)
            real_loader = DataLoader(real_test_ds, batch_size=config["batch_size"], shuffle=False, **loader_kwargs)

            model = VideoModel(height, width, len(config["classes"])).to(device)
            optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            criterion = nn.CrossEntropyLoss()

            best_real_loss = float("inf")
            history_rows = []
            epochs = 1 if smoke else config["epochs"]
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
                if real_loss < best_real_loss:
                    best_real_loss = real_loss
                    torch.save(model.state_dict(), run_dir / "best_model.pth")

            with open(run_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
                writer.writeheader()
                writer.writerows(history_rows)
            save_curves(history_rows, run_dir)

            model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
            synth_loss, synth_acc, synth_report = evaluate(model, synth_loader, criterion, device, "synthetic", run_dir)
            real_loss, real_acc, real_report = evaluate(model, real_loader, criterion, device, "real", run_dir)

            summary = {
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
                "best_epoch": min(range(len(history_rows)), key=lambda i: history_rows[i]["real_test_loss"]) + 1,
            }
            save_json(summary, run_dir / "summary.json")
            all_summaries.append(summary)

    save_json({"runs": all_summaries}, exp_root / "all_runs_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/Users/apple/Downloads/Simulation/config.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sizes", nargs="*", type=int)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    run_experiments(args.config, sizes=args.sizes, repeats=args.repeats, smoke=args.smoke)


if __name__ == "__main__":
    main()
