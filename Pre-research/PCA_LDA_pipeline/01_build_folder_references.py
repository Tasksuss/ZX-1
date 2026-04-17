import os
import cv2
import json
import glob
import argparse
import numpy as np


# =========================================================
# Utils
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_gray(frame):
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.copy()


def odd_ksize_from_fraction(size_val, frac):
    k = int(size_val * frac)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return k


def is_video_file(filename):
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))


def find_video_files(folder):
    exts = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


# =========================================================
# Sampling / reference
# =========================================================
def sample_video_frames(video_path, max_samples=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open video: {video_path}")
        return []

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        print(f"[WARN] invalid frame count: {video_path}")
        cap.release()
        return []

    idxs = np.linspace(0, n - 1, num=min(max_samples, n), dtype=int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        # fallback: if random seek fails, try reading forward a few frames
        if not ret or frame is None:
            for _ in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break

        if ret and frame is not None:
            gray = to_gray(frame)
            if gray.size > 0 and np.mean(gray) > 1:
                frames.append(gray)
            else:
                print(f"[WARN] empty/dark frame ignored at {idx} in {video_path}")
        else:
            print(f"[WARN] failed reading frame {idx} from {video_path}")

    cap.release()
    return frames


def build_folder_reference(folder_path, max_videos=5, max_frames_per_video=8):
    videos = find_video_files(folder_path)
    videos = videos[:max_videos]

    all_frames = []
    for vp in videos:
        frames = sample_video_frames(vp, max_samples=max_frames_per_video)
        print(f"[INFO] {os.path.basename(vp)} -> sampled {len(frames)} frame(s)")
        all_frames.extend(frames)

    if len(all_frames) == 0:
        return None

    stack = np.stack([f.astype(np.float32) for f in all_frames], axis=0)
    ref = np.median(stack, axis=0)

    print(
        f"[INFO] reference stats for {folder_path}: "
        f"min={ref.min():.2f}, max={ref.max():.2f}, mean={ref.mean():.2f}"
    )

    ref = np.clip(ref, 0, 255).astype(np.uint8)
    return ref


# =========================================================
# Bias field + stats
# =========================================================
def estimate_foreground_mask(ref_gray, thresh_percentile=35):
    sm = cv2.GaussianBlur(ref_gray, (9, 9), 0)
    thr = np.percentile(sm, thresh_percentile)

    mask = (sm > thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def estimate_bias_field_from_reference(ref_gray, blur_frac=0.35, eps=1.0):
    h, w = ref_gray.shape[:2]
    k = odd_ksize_from_fraction(max(h, w), blur_frac)

    fg_mask = estimate_foreground_mask(ref_gray, thresh_percentile=35)

    ref_f = ref_gray.astype(np.float32)
    mask_f = (fg_mask > 0).astype(np.float32)

    masked = ref_f * mask_f

    blur_img = cv2.GaussianBlur(masked, (k, k), 0)
    blur_msk = cv2.GaussianBlur(mask_f, (k, k), 0)

    bias_field = blur_img / (blur_msk + eps)

    raw_blur = cv2.GaussianBlur(ref_f, (k, k), 0)
    bias_field = np.where(blur_msk > 0.05, bias_field, raw_blur)

    bias_field = np.maximum(bias_field, eps)
    return bias_field, fg_mask


def apply_reference_bias_correction(gray, bias_field):
    gray_f = gray.astype(np.float32)
    target_mean = float(np.mean(bias_field))
    corrected = gray_f / bias_field * target_mean
    return corrected


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--references_dir", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--dataset_target_path", required=True)
    parser.add_argument("--blur_frac", type=float, default=0.35)
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--max_frames_per_video", type=int, default=8)
    args = parser.parse_args()

    ensure_dir(args.references_dir)

    metadata = {}
    ref_means = []
    ref_p95s = []

    # Find all leaf folders that actually contain videos
    subfolders = []
    for root, dirs, files in os.walk(args.dataset_root):
        if any(is_video_file(f) for f in files):
            subfolders.append(root)

    subfolders = sorted(subfolders)

    if len(subfolders) == 0:
        print(f"[ERROR] no video folders found under {args.dataset_root}")
        return

    print(f"[INFO] found {len(subfolders)} video folder(s)")

    for folder in subfolders:
        # use relative path as unique key
        folder_name = os.path.relpath(folder, args.dataset_root)

        ref = build_folder_reference(
            folder,
            max_videos=args.max_videos,
            max_frames_per_video=args.max_frames_per_video
        )

        if ref is None:
            print(f"[WARN] no usable frames in {folder_name}")
            continue

        safe_name = folder_name.replace("/", "__").replace("\\", "__")

        # save raw reference
        ref_path = os.path.join(args.references_dir, f"{safe_name}_reference.png")
        cv2.imwrite(ref_path, ref)

        # save preview reference for manual clicking
        ref_preview = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX)
        ref_preview_path = os.path.join(args.references_dir, f"{safe_name}_reference_preview.png")
        cv2.imwrite(ref_preview_path, ref_preview)

        # compute brightness stats on corrected reference
        bias_field, fg_mask = estimate_bias_field_from_reference(ref, blur_frac=args.blur_frac)
        corrected = apply_reference_bias_correction(ref, bias_field)

        fg = corrected[fg_mask > 0]
        if fg.size == 0:
            ref_mean = float(np.mean(corrected))
            ref_p95 = float(np.percentile(corrected, 95))
        else:
            ref_mean = float(np.mean(fg))
            ref_p95 = float(np.percentile(fg, 95))

        ref_means.append(ref_mean)
        ref_p95s.append(ref_p95)

        metadata[folder_name] = {
            "reference_path": ref_path,
            "reference_preview_path": ref_preview_path,
            "center_x": None,
            "center_y": None,
            "ref_mean": ref_mean,
            "ref_p95": ref_p95
        }

        print(f"[OK] {folder_name}")

    with open(args.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    dataset_target = {
        "target_mean": float(np.median(ref_means)) if ref_means else 128.0,
        "target_p95": float(np.median(ref_p95s)) if ref_p95s else 200.0
    }

    with open(args.dataset_target_path, "w", encoding="utf-8") as f:
        json.dump(dataset_target, f, indent=2)

    print(f"[DONE] metadata: {args.metadata_path}")
    print(f"[DONE] dataset target: {args.dataset_target_path}")


if __name__ == "__main__":
    main()

"""
python build_folder_references.py \
  --dataset_root /Users/apple/Downloads/Troat/Preprocessing \
  --references_dir /Users/apple/Downloads/Troat/workspace/references \
  --metadata_path /Users/apple/Downloads/Troat/workspace/folder_metadata.json \
  --dataset_target_path /Users/apple/Downloads/Troat/workspace/dataset_target.json
"""