import csv
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".npy")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_folder_name(folder_name: str) -> Tuple[str, str]:
    subject, label = folder_name.split("-", 1)
    return subject, label.lower()


def iter_dataset_samples(root_dir: Path, classes: Optional[Sequence[str]] = None) -> Iterable[Dict]:
    allowed = set(classes) if classes else None
    for split_dir in sorted(root_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir() or "-" not in folder.name:
                continue
            subject, label = parse_folder_name(folder.name)
            if allowed and label not in allowed:
                continue
            for file_path in sorted(folder.iterdir()):
                if file_path.suffix.lower() in VIDEO_SUFFIXES:
                    yield {
                        "split": split,
                        "subject": subject,
                        "label": label,
                        "path": str(file_path),
                        "folder": folder.name,
                    }


def ffprobe_metadata(path: str) -> Dict[str, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    def parse_rate(value: str) -> float:
        if "/" in value:
            a, b = value.split("/", 1)
            return float(a) / float(b) if float(b) else 0.0
        return float(value)

    nb_frames = stream.get("nb_frames")
    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": parse_rate(stream.get("avg_frame_rate", "0/1")),
        "r_fps": parse_rate(stream.get("r_frame_rate", "0/1")),
        "duration": float(stream.get("duration", 0.0) or 0.0),
        "nb_frames": int(nb_frames) if nb_frames not in (None, "N/A") else 0,
    }


def read_video_gray(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 4:
            arr = arr[..., 0]
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr[:max_frames] if max_frames else arr

    meta = ffprobe_metadata(path)
    width = meta["width"]
    height = meta["height"]
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
    ]
    if max_frames is not None:
        cmd.extend(["-vframes", str(max_frames)])
    cmd.append("-")
    result = subprocess.run(cmd, check=True, capture_output=True)
    frame_bytes = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_size = width * height
    if frame_size == 0 or frame_bytes.size == 0:
        raise ValueError(f"No readable frames in {path}")
    frame_count = frame_bytes.size // frame_size
    frames = frame_bytes[: frame_count * frame_size].reshape(frame_count, height, width)
    return (frames.astype(np.float32) / 255.0)


def temporal_sample(frames: np.ndarray, target_frames: int) -> np.ndarray:
    total = len(frames)
    if total == target_frames:
        return frames.astype(np.float32)
    indices = np.linspace(0, total - 1, num=target_frames)
    indices = np.clip(np.round(indices).astype(int), 0, total - 1)
    return frames[indices].astype(np.float32)


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(rows: List[Dict], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: Iterable[Dict], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def stable_hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % (2 ** 31 - 1)
    return value


def label_motion_template(label: str) -> Dict[str, float]:
    templates = {
        "banana": {"peak_count": 3.0, "amplitude": 0.11, "tempo": 1.05, "phase": 0.15},
        "cocoro": {"peak_count": 2.2, "amplitude": 0.13, "tempo": 0.95, "phase": 0.28},
        "nana": {"peak_count": 4.2, "amplitude": 0.09, "tempo": 1.3, "phase": 0.08},
        "tomorrow": {"peak_count": 2.6, "amplitude": 0.15, "tempo": 0.85, "phase": 0.36},
        "tuesday": {"peak_count": 3.5, "amplitude": 0.1, "tempo": 1.15, "phase": 0.22},
        "yesterday": {"peak_count": 2.0, "amplitude": 0.12, "tempo": 0.78, "phase": 0.4},
    }
    return templates[label]


def subject_style(subject: str) -> Dict[str, float]:
    styles = {
        "SynSubj1": {"brightness": 0.63, "contrast": 0.12, "blur": 1.7, "center_x": 0.48, "center_y": 0.52, "motion_scale": 0.95},
        "SynSubj3": {"brightness": 0.57, "contrast": 0.11, "blur": 1.5, "center_x": 0.47, "center_y": 0.49, "motion_scale": 1.00},
        "SynSubj4": {"brightness": 0.68, "contrast": 0.10, "blur": 2.1, "center_x": 0.50, "center_y": 0.54, "motion_scale": 0.85},
        "SynSubj5": {"brightness": 0.53, "contrast": 0.13, "blur": 1.4, "center_x": 0.45, "center_y": 0.51, "motion_scale": 1.08},
    }
    return styles.get(subject, styles["SynSubj3"])


def compute_basic_frame_stats(frames: np.ndarray) -> Dict[str, float]:
    diffs = np.diff(frames, axis=0) if len(frames) > 1 else np.zeros_like(frames)
    motion = np.abs(diffs).mean(axis=(1, 2)) if len(diffs) else np.array([0.0], dtype=np.float32)
    activity_map = np.abs(diffs).mean(axis=0) if len(diffs) else np.zeros_like(frames[0])
    yy, xx = np.mgrid[0:activity_map.shape[0], 0:activity_map.shape[1]]
    weight = activity_map + 1e-6
    center_y = float((yy * weight).sum() / weight.sum()) / activity_map.shape[0]
    center_x = float((xx * weight).sum() / weight.sum()) / activity_map.shape[1]
    return {
        "brightness_mean": float(frames.mean()),
        "brightness_std": float(frames.std()),
        "background_p10": float(np.percentile(frames, 10)),
        "background_p90": float(np.percentile(frames, 90)),
        "motion_energy_mean": float(motion.mean()),
        "motion_energy_std": float(motion.std()),
        "activity_center_x": center_x,
        "activity_center_y": center_y,
    }


def pick_evenly_spaced(items: Sequence, limit: int) -> List:
    if len(items) <= limit:
        return list(items)
    idx = np.linspace(0, len(items) - 1, num=limit)
    return [items[int(round(i))] for i in idx]
