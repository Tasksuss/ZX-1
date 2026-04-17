import os
import cv2
import json
import glob
import argparse
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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


def collect_videos(folder):
    exts = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


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


def brightness_match(corrected, scale):
    out = corrected * scale
    return out


def map_to_uint8(corrected_float, low_pct=3, high_pct=97, gamma=0.95):
    lo = np.percentile(corrected_float, low_pct)
    hi = np.percentile(corrected_float, high_pct)
    if hi <= lo:
        hi = lo + 1.0

    x = np.clip(corrected_float, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    x = np.power(x, gamma)
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


def make_roi_from_center(frame_w, frame_h, cx, cy, roi_w_ratio=0.48, roi_h_ratio=0.56):
    roi_w = int(round(frame_w * roi_w_ratio))
    roi_h = int(round(frame_h * roi_h_ratio))

    x1 = int(round(cx - roi_w / 2))
    y1 = int(round(cy - roi_h / 2))
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_w:
        diff = x2 - frame_w
        x1 -= diff
        x2 = frame_w
    if y2 > frame_h:
        diff = y2 - frame_h
        y1 -= diff
        y2 = frame_h

    x1 = clamp(x1, 0, frame_w - 1)
    y1 = clamp(y1, 0, frame_h - 1)
    x2 = clamp(x2, x1 + 1, frame_w)
    y2 = clamp(y2, y1 + 1, frame_h)
    return x1, y1, x2, y2


def process_video(video_path, folder_name, folder_info, dataset_target, output_root,
                  roi_w_ratio=0.48, roi_h_ratio=0.56, out_w=256, out_h=192, blur_frac=0.35):
    ref = cv2.imread(folder_info["reference_path"], cv2.IMREAD_GRAYSCALE)
    if ref is None:
        print(f"[WARN] missing reference for {folder_name}")
        return

    center_x = folder_info.get("center_x")
    center_y = folder_info.get("center_y")
    if center_x is None or center_y is None:
        print(f"[WARN] missing center for {folder_name}")
        return

    bias_field, fg_mask = estimate_bias_field_from_reference(ref, blur_frac=blur_frac)
    folder_ref_mean = folder_info["ref_mean"]
    target_mean = dataset_target["target_mean"]

    scale = target_mean / max(folder_ref_mean, 1e-6)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_box = make_roi_from_center(frame_w, frame_h, center_x, center_y, roi_w_ratio, roi_h_ratio)

    out_dir = os.path.join(output_root, folder_name)
    ensure_dir(out_dir)

    base = os.path.splitext(os.path.basename(video_path))[0]
    full_path = os.path.join(out_dir, f"{base}_full.mp4")
    crop_path = os.path.join(out_dir, f"{base}_crop.mp4")

    if (
            os.path.exists(full_path) and os.path.getsize(full_path) > 0 and
            os.path.exists(crop_path) and os.path.getsize(crop_path) > 0
    ):
        print(f"[SKIP] already processed: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_full = cv2.VideoWriter(full_path, fourcc, fps, (frame_w, frame_h), isColor=False)
    writer_crop = cv2.VideoWriter(crop_path, fourcc, fps, (out_w, out_h), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = to_gray(frame)
        corrected = apply_reference_bias_correction(gray, bias_field)
        corrected = brightness_match(corrected, scale)
        corrected_u8 = map_to_uint8(corrected, low_pct=3, high_pct=97, gamma=0.95)

        writer_full.write(corrected_u8)

        x1, y1, x2, y2 = roi_box
        crop = corrected_u8[y1:y2, x1:x2]
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer_crop.write(crop)

    cap.release()
    writer_full.release()
    writer_crop.release()
    print(f"[OK] {video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--dataset_target_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--roi_w_ratio", type=float, default=0.48)
    parser.add_argument("--roi_h_ratio", type=float, default=0.56)
    parser.add_argument("--out_w", type=int, default=256)
    parser.add_argument("--out_h", type=int, default=192)
    parser.add_argument("--blur_frac", type=float, default=0.35)
    args = parser.parse_args()

    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(args.dataset_target_path, "r", encoding="utf-8") as f:
        dataset_target = json.load(f)

    subfolders = []
    for root, dirs, files in os.walk(args.dataset_root):
        if any(f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")) for f in files):
            subfolders.append(root)

    subfolders = sorted(subfolders)

    if len(subfolders) == 0:
        print(f"[ERROR] no video folders found under {args.dataset_root}")
        return

    print(f"[INFO] found {len(subfolders)} video folder(s)")

    for folder in subfolders:
        folder_name = os.path.relpath(folder, args.dataset_root)

        if folder_name not in metadata:
            print(f"[WARN] no metadata for {folder_name}")
            continue

        videos = collect_videos(folder)
        print(f"[INFO] {folder_name}: {len(videos)} video(s)")

        for vp in videos:
            process_video(
                video_path=vp,
                folder_name=folder_name,
                folder_info=metadata[folder_name],
                dataset_target=dataset_target,
                output_root=args.output_root,
                roi_w_ratio=args.roi_w_ratio,
                roi_h_ratio=args.roi_h_ratio,
                out_w=args.out_w,
                out_h=args.out_h,
                blur_frac=args.blur_frac
            )

    print("[DONE] preprocessing finished")


if __name__ == "__main__":
    main()


"""
python preprocess_dataset.py \
  --dataset_root /Users/apple/Downloads/Troat/Preprocessing \
  --metadata_path /Users/apple/Downloads/Troat/workspace/folder_metadata.json \
  --dataset_target_path /Users/apple/Downloads/Troat/workspace/dataset_target.json \
  --output_root /Users/apple/Downloads/Troat/workspace/processed
  --roi_w_ratio 0.36 \
  --roi_h_ratio 0.42
  
  
650nm+task6+task6.1:  roi_w_ratio 0.48 roi_h_ratio 0.56
larnyx_trimmed: roi_w_ratio 0.36 roi_h_ratio 0.42
"""