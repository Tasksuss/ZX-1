import argparse
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from pipeline_utils import (
    ensure_dir,
    label_motion_template,
    load_config,
    save_csv,
    save_json,
    set_global_seed,
    stable_hash,
    subject_style,
)


def make_background(height: int, width: int, rng: np.random.Generator, style: dict) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    yy = yy / max(height - 1, 1)
    xx = xx / max(width - 1, 1)

    gradient = style["brightness"] + 0.04 * (yy - 0.5) - 0.03 * (xx - 0.5)
    noise = gaussian_filter(rng.normal(0.0, style["contrast"], size=(height, width)), sigma=6.0)
    ridge = 0.05 * np.exp(-((yy - style["center_y"]) ** 2) / 0.025) * np.cos(2 * math.pi * (xx * 1.4 + 0.1))
    return np.clip(gradient + noise + ridge, 0.0, 1.0).astype(np.float32)


def make_roi_mask(height: int, width: int, style: dict, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    yy = yy / max(height - 1, 1)
    xx = xx / max(width - 1, 1)
    cx = style["center_x"] + rng.uniform(-0.015, 0.015)
    cy = style["center_y"] + rng.uniform(-0.02, 0.02)
    rx = 0.26 + rng.uniform(-0.03, 0.03)
    ry = 0.18 + rng.uniform(-0.025, 0.025)
    mask = np.exp(-(((xx - cx) ** 2) / (2 * rx ** 2) + ((yy - cy) ** 2) / (2 * ry ** 2)))
    return mask.astype(np.float32)


def make_texture(height: int, width: int, rng: np.random.Generator, label_template: dict, style: dict) -> np.ndarray:
    base = gaussian_filter(rng.normal(0, 1.0, size=(height, width)), sigma=style["blur"] * 2.6)
    fibrous = gaussian_filter(rng.normal(0, 1.0, size=(height, width)), sigma=(style["blur"] * 1.1))
    yy, xx = np.mgrid[0:height, 0:width]
    yy = yy / max(height - 1, 1)
    xx = xx / max(width - 1, 1)
    band = np.sin((yy * (6.0 + label_template["peak_count"])) + label_template["phase"] * math.pi)
    texture = 0.6 * base + 0.25 * fibrous + 0.15 * band
    texture = texture / (np.std(texture) + 1e-6)
    return texture.astype(np.float32)


def motion_curve(frame_count: int, label_template: dict, style: dict, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num=frame_count, dtype=np.float32)
    peak_count = label_template["peak_count"] + rng.uniform(-0.2, 0.2)
    envelope = np.exp(-((t - 0.55) ** 2) / (0.08 + rng.uniform(-0.01, 0.015)))
    periodic = np.sin(2 * math.pi * (peak_count * t + label_template["phase"]))
    secondary = 0.5 * np.sin(2 * math.pi * ((peak_count * 0.5) * t + rng.uniform(0, 1)))
    curve = (0.55 + 0.45 * periodic + 0.2 * secondary) * envelope
    curve = np.maximum(curve, 0.0)
    curve = curve / (curve.max() + 1e-6)
    return (curve * label_template["amplitude"] * style["motion_scale"]).astype(np.float32)


def warp_texture(texture: np.ndarray, shift_x: np.ndarray, shift_y: np.ndarray) -> np.ndarray:
    height, width = texture.shape
    yy, xx = np.mgrid[0:height, 0:width]
    coords = np.array([yy + shift_y, xx + shift_x])
    warped = map_coordinates(texture, coords, order=1, mode="reflect")
    return warped.astype(np.float32)


def synthesize_video(height: int, width: int, frame_count: int, label: str, subject: str, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    style = subject_style(subject)
    label_template = label_motion_template(label)

    bg = make_background(height, width, rng, style)
    mask = make_roi_mask(height, width, style, rng)
    texture = make_texture(height, width, rng, label_template, style)
    motion = motion_curve(frame_count, label_template, style, rng)

    yy, xx = np.mgrid[0:height, 0:width]
    yy_n = yy / max(height - 1, 1)
    xx_n = xx / max(width - 1, 1)
    stripe = np.sin((yy_n * (10 + label_template["peak_count"])) + rng.uniform(0, 2 * math.pi))
    frames = []
    for idx, amp in enumerate(motion):
        temporal = idx / max(frame_count - 1, 1)
        shift_x = (amp * width * 0.7) * np.sin(2 * math.pi * (yy_n * 1.2 + temporal * label_template["tempo"]))
        shift_y = (amp * height * 0.9) * np.cos(2 * math.pi * (xx_n * 0.8 + temporal * (label_template["tempo"] + 0.12)))
        warped = warp_texture(texture, shift_x, shift_y)

        local_shadow = -0.08 * amp * np.exp(-((yy_n - style["center_y"]) ** 2) / 0.01)
        local_bright = 0.05 * amp * np.exp(-((xx_n - style["center_x"]) ** 2 + (yy_n - style["center_y"]) ** 2) / 0.018)
        tissue = bg + mask * (0.08 * warped + local_shadow + local_bright + 0.03 * stripe)
        tissue += gaussian_filter(rng.normal(0, 0.012, size=(height, width)), sigma=0.8 + style["blur"] * 0.2)
        tissue = gaussian_filter(tissue, sigma=style["blur"])
        frame = np.clip(tissue, 0.0, 1.0)
        frames.append(frame.astype(np.float32))

    return np.stack(frames), {
        "appearance_params": {
            "brightness": style["brightness"],
            "contrast": style["contrast"],
            "blur": style["blur"],
            "center_x": style["center_x"],
            "center_y": style["center_y"],
        },
        "motion_params": {
            "peak_count": label_template["peak_count"],
            "amplitude": label_template["amplitude"],
            "tempo": label_template["tempo"],
            "phase": label_template["phase"],
        },
    }


def write_preview_mp4(video: np.ndarray, path: Path, fps: float) -> None:
    ensure_dir(path.parent)
    height, width = video.shape[1], video.shape[2]
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "mpeg4",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in video:
            proc.stdin.write((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8).tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.read().decode("utf-8", errors="ignore"))
    finally:
        if proc.poll() is None:
            proc.kill()


def generate_dataset(config_path: str) -> None:
    config = load_config(config_path)
    set_global_seed(config["seed"])
    with open(config["generator_profile_path"], "r", encoding="utf-8") as f:
        profile = __import__("json").load(f)

    sim_root = ensure_dir(Path(config["sim_output_root"]))
    rows = []
    width, height = config["image_size"]
    video_stats = profile["video_stats"]
    fps = max(video_stats["fps_mean"], 1.0)
    train_count = config["synthetic_videos_per_class_subject_train"]
    test_count = config["synthetic_videos_per_class_subject_test"]

    preview_budget = config.get("synthetic_export_mp4_samples", 0)
    preview_done = {}

    for split, subjects, count_per_group in [
        ("trainset", config["synthetic_subjects_train"], train_count),
        ("testset", config["synthetic_subjects_test"], test_count),
    ]:
        for subject in subjects:
            for label in config["classes"]:
                target_dir = ensure_dir(sim_root / split / f"{subject}-{label}")
                preview_done.setdefault((split, subject, label), 0)
                for idx in range(count_per_group):
                    seed = stable_hash(f"{split}:{subject}:{label}:{idx}:{config['seed']}")
                    if "generation_frame_count" in config:
                        frame_count = int(config["generation_frame_count"])
                    else:
                        frame_count = int(round(np.random.normal(video_stats["frame_count_mean"], max(video_stats["frame_count_std"], 1.0))))
                        frame_count = max(frame_count, config["target_frame_count"])
                    video, params = synthesize_video(height, width, frame_count, label, subject, seed)
                    filename = f"{subject}_{label}_{idx:03d}.npy"
                    np.save(target_dir / filename, (np.clip(video, 0.0, 1.0) * 255).astype(np.uint8))
                    if preview_done[(split, subject, label)] < preview_budget:
                        write_preview_mp4(video, target_dir / f"{subject}_{label}_{idx:03d}.mp4", fps=fps)
                        preview_done[(split, subject, label)] += 1
                    rows.append({
                        "split": split,
                        "subject": subject,
                        "label": label,
                        "path": str(target_dir / filename),
                        "seed": seed,
                        "duration": frame_count / fps,
                        "frame_count": frame_count,
                        "appearance_params": __import__("json").dumps(params["appearance_params"]),
                        "motion_params": __import__("json").dumps(params["motion_params"]),
                    })

    save_csv(rows, sim_root / "metadata.csv")
    save_json({
        "sim_output_root": str(sim_root),
        "total_samples": len(rows),
        "train_samples": sum(1 for row in rows if row["split"] == "trainset"),
        "test_samples": sum(1 for row in rows if row["split"] == "testset"),
        "classes": config["classes"],
        "synthetic_subjects_train": config["synthetic_subjects_train"],
        "synthetic_subjects_test": config["synthetic_subjects_test"],
    }, sim_root / "generation_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/Users/apple/Downloads/Simulation/config.json")
    args = parser.parse_args()
    generate_dataset(args.config)


if __name__ == "__main__":
    main()
