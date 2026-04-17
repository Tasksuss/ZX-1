import os
import re
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.svm import SVC

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}


# =========================================================
# Utils
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_std(x, eps=1e-8):
    s = np.std(x)
    return s if s > eps else eps


def parse_subject_word_from_path(video_path: Path):
    """从路径解析 subject 和 word（保持原逻辑）"""
    parts = video_path.parts
    subject = "unknown_subject"
    word = "unknown_word"
    split = "unknown_split"

    for p in parts:
        if p.lower() in ["train", "val", "valid", "validation", "test"]:
            split = p.lower()
            break

    for p in parts[::-1]:
        m = re.match(r"^(Subj[^-/\\]+)[-_](.+)$", p, re.IGNORECASE)
        if m:
            subject = m.group(1)
            word = m.group(2)
            return subject, word, split

    for p in parts:
        if re.match(r"^Subj", p, re.IGNORECASE):
            subject = p
            break

    if split != "unknown_split":
        try:
            idx = [i for i, p in enumerate(parts) if p.lower() == split][0]
            if idx + 1 < len(parts):
                candidate = parts[idx + 1]
                if not re.match(r"^Subj", candidate, re.IGNORECASE):
                    word = candidate
                elif idx + 2 < len(parts):
                    word = parts[idx + 2]
        except Exception:
            pass

    if word == "unknown_word":
        parent = video_path.parent.name
        if not re.match(r"^Subj", parent, re.IGNORECASE):
            word = parent

    return subject, word, split


def list_videos(root):
    videos = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            videos.append(p)
    return sorted(videos)


def read_video_gray(video_path, target_size=(128, 128), target_frames=24):
    """读取视频并返回灰度帧序列 (T, H, W)，归一化到 [0,1]"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 0.0

    raw_frames = []
    orig_h, orig_w = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if orig_h is None:
            orig_h, orig_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        raw_frames.append(gray)

    cap.release()
    if len(raw_frames) == 0:
        raise RuntimeError(f"No frames read: {video_path}")

    raw_frames = np.stack(raw_frames, axis=0).astype(np.float32) / 255.0
    orig_n_frames = raw_frames.shape[0]

    # 线性重采样到固定帧数
    if orig_n_frames == target_frames:
        frames = raw_frames
    else:
        idx = np.linspace(0, orig_n_frames - 1, target_frames)
        idx0 = np.floor(idx).astype(int)
        idx1 = np.ceil(idx).astype(int)
        alpha = idx - idx0
        frames = []
        for a, i0, i1 in zip(alpha, idx0, idx1):
            if i0 == i1:
                f = raw_frames[i0]
            else:
                f = (1.0 - a) * raw_frames[i0] + a * raw_frames[i1]
            frames.append(f)
        frames = np.stack(frames, axis=0).astype(np.float32)

    return frames, float(fps), int(orig_n_frames), int(orig_h), int(orig_w)


def normalize_video(frames, mode="none"):
    if mode == "none":
        return frames
    elif mode == "video_zscore":
        mu = frames.mean()
        sd = safe_std(frames)
        return (frames - mu) / sd
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")


def compute_differences(frames, lag=1):
    if lag <= 0 or lag >= len(frames):
        raise ValueError("lag must be between 1 and T-1")
    return frames[lag:] - frames[:-lag]


def motion_energy(diff_frames):
    return np.mean(np.abs(diff_frames), axis=(1, 2))


def summarize_motion(diff_frames):
    """仅用于 metadata 中的运动统计，不用于后续 PCA/LDA 特征"""
    e = motion_energy(diff_frames)
    feat = {
        "motion_mean": float(np.mean(e)),
        "motion_std": float(np.std(e)),
        "motion_max": float(np.max(e)),
        "motion_min": float(np.min(e)),
        "motion_range": float(np.max(e) - np.min(e)),
        "motion_argmax_norm": float(np.argmax(e) / max(1, len(e) - 1)),
        "motion_skew_proxy": float(np.mean((e - np.mean(e)) ** 3) / (safe_std(e) ** 3)),
        "motion_kurt_proxy": float(np.mean((e - np.mean(e)) ** 4) / (safe_std(e) ** 4)),
        "motion_rms": float(np.sqrt(np.mean(e ** 2))),
    }
    absd = np.abs(diff_frames)
    feat.update({
        "absdiff_mean": float(np.mean(absd)),
        "absdiff_std": float(np.std(absd)),
        "absdiff_max": float(np.max(absd)),
    })
    return feat


def save_image(arr, out_path, vmin=None, vmax=None, cmap="gray"):
    if vmin is not None and vmax is not None and vmax <= vmin:
        vmax = vmin + 1e-6
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_motion_curve(e, out_path, title="motion energy"):
    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(len(e)), e)
    plt.xlabel("frame index")
    plt.ylabel("mean abs diff")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def fit_pca_projection(X, n_components=2):
    """Fit PCA on standardized features and return reusable artifacts."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    max_components = min(n_components, Xs.shape[0], Xs.shape[1])
    if max_components < 1:
        raise ValueError("PCA requires at least one valid component")
    pca = PCA(n_components=max_components)
    Z = pca.fit_transform(Xs)
    return {
        "scaler": scaler,
        "Xs": Xs,
        "pca": pca,
        "Z": Z,
    }


def pca_plot(X, meta_df, color_by="word", style_by="subject",
             out_path="pca.png", title="PCA", return_model=False):
    """原始 PCA 散点图（自适应颜色和形状）"""
    if len(X) < 2:
        print(f"[WARN] Not enough samples for PCA: {title}")
        return

    pca_result = fit_pca_projection(X, n_components=2)
    pca = pca_result["pca"]
    Z = pca_result["Z"]
    if Z.shape[1] == 1:
        Z = np.hstack([Z, np.zeros((len(Z), 1))])

    df = meta_df.copy()
    df["PC1"] = Z[:, 0]
    df["PC2"] = Z[:, 1]

    # 获取唯一颜色类别（词）和形状类别（被试）
    color_categories = sorted(df[color_by].unique())
    style_categories = sorted(df[style_by].unique())

    # 动态分配颜色（使用 colormap）
    # 如果类别数超过 colormap 长度，则循环使用
    cmap = plt.cm.get_cmap('tab10')  # 可选 'tab20', 'Set3' 等
    color_list = [cmap(i % cmap.N) for i in range(len(color_categories))]
    color_map = {cat: color_list[i] for i, cat in enumerate(color_categories)}

    # 动态分配形状（预定义标记列表）
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x', 'd']
    if len(style_categories) > len(markers):
        print(f"[WARN] {len(style_categories)} subjects, but only {len(markers)} markers available. Will cycle markers.")
    style_to_marker = {cat: markers[i % len(markers)] for i, cat in enumerate(style_categories)}

    plt.figure(figsize=(8, 6))
    for cat in color_categories:
        for subj in style_categories:
            sub = df[(df[color_by] == cat) & (df[style_by] == subj)]
            if len(sub) == 0:
                continue
            plt.scatter(
                sub["PC1"],
                sub["PC2"],
                label=f"{cat} | {subj}" if len(sub) > 0 else "",
                marker=style_to_marker[subj],
                color=color_map[cat],
                alpha=0.75
            )

    evr = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    plt.title(title)
    # 避免图例过长，可以只显示颜色对应的词，形状对应的被试可省略（但为了清晰保留）
    # 如果图例条目太多，可以设置 ncol 或调整位置
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    if return_model:
        pca_result["Z"] = Z
        return pca_result


def save_pca_diagnostics(X, img_shape, feature_name, save_dir,
                         n_mode_components=5, n_scree_components=10, mode_scale=2.0):
    """Save scree plots and PCA mode visualizations in image space."""
    if len(X) < 2:
        print(f"[WARN] Not enough samples for PCA diagnostics: {feature_name}")
        return

    diag_dir = Path(save_dir) / feature_name
    ensure_dir(diag_dir)

    n_samples, n_features = X.shape
    max_components = min(n_samples, n_features)
    pca_result = fit_pca_projection(X, n_components=max_components)
    pca = pca_result["pca"]
    scaler = pca_result["scaler"]

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    scree_n = min(n_scree_components, len(explained))

    plt.figure(figsize=(6, 4))
    xs = np.arange(1, scree_n + 1)
    plt.plot(xs, explained[:scree_n], marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title(f"{feature_name} - PCA scree plot")
    plt.tight_layout()
    plt.savefig(diag_dir / "scree_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(xs, cumulative[:scree_n], marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative explained variance")
    plt.ylim(0, 1.02)
    plt.title(f"{feature_name} - PCA cumulative variance")
    plt.tight_layout()
    plt.savefig(diag_dir / "cumulative_variance.png", dpi=180)
    plt.close()

    mean_std_img = pca.mean_.reshape(img_shape)
    mean_std_vmax = float(np.max(np.abs(mean_std_img)))
    save_image(mean_std_img, diag_dir / "mean_standardized_image.png",
               vmin=-mean_std_vmax, vmax=mean_std_vmax, cmap="coolwarm")

    mean_raw_img = scaler.mean_.reshape(img_shape)
    save_image(mean_raw_img, diag_dir / "mean_raw_image.png", cmap="gray")

    mode_n = min(n_mode_components, len(explained))
    raw_scale_img = scaler.scale_.reshape(img_shape)
    for idx in range(mode_n):
        component_std = pca.components_[idx].reshape(img_shape)
        component_std_vmax = float(np.max(np.abs(component_std)))
        save_image(component_std, diag_dir / f"pc{idx + 1:02d}_component_standardized.png",
                   vmin=-component_std_vmax, vmax=component_std_vmax, cmap="coolwarm")

        delta_std = mode_scale * np.sqrt(max(pca.explained_variance_[idx], 0.0)) * component_std
        minus_std_img = mean_std_img - delta_std
        plus_std_img = mean_std_img + delta_std
        mode_std_vmax = float(max(np.max(np.abs(minus_std_img)), np.max(np.abs(plus_std_img))))

        save_image(minus_std_img, diag_dir / f"pc{idx + 1:02d}_mode_minus_standardized.png",
                   vmin=-mode_std_vmax, vmax=mode_std_vmax, cmap="coolwarm")
        save_image(plus_std_img, diag_dir / f"pc{idx + 1:02d}_mode_plus_standardized.png",
                   vmin=-mode_std_vmax, vmax=mode_std_vmax, cmap="coolwarm")

        component_raw = component_std * raw_scale_img
        component_raw_vmax = float(np.max(np.abs(component_raw)))
        save_image(component_raw, diag_dir / f"pc{idx + 1:02d}_component_rawspace.png",
                   vmin=-component_raw_vmax, vmax=component_raw_vmax, cmap="coolwarm")

        delta_raw = delta_std * raw_scale_img
        minus_raw_img = mean_raw_img - delta_raw
        plus_raw_img = mean_raw_img + delta_raw
        raw_min = float(min(np.min(minus_raw_img), np.min(plus_raw_img), np.min(mean_raw_img)))
        raw_max = float(max(np.max(minus_raw_img), np.max(plus_raw_img), np.max(mean_raw_img)))

        save_image(minus_raw_img, diag_dir / f"pc{idx + 1:02d}_mode_minus_rawspace.png",
                   vmin=raw_min, vmax=raw_max, cmap="gray")
        save_image(plus_raw_img, diag_dir / f"pc{idx + 1:02d}_mode_plus_rawspace.png",
                   vmin=raw_min, vmax=raw_max, cmap="gray")

    summary_path = diag_dir / "pca_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"feature_name: {feature_name}\n")
        f.write(f"n_samples: {n_samples}\n")
        f.write(f"n_features: {n_features}\n")
        f.write(f"mode_scale: {mode_scale}\n")
        f.write("mode_outputs: standardized and rawspace\n")
        for idx, (evr, cum) in enumerate(zip(explained[:scree_n], cumulative[:scree_n]), start=1):
            f.write(f"PC{idx}: explained_variance_ratio={evr:.6f}, cumulative={cum:.6f}\n")

    print(f"[PCA] Saved diagnostics for {feature_name} to {diag_dir}")


def save_variance_map(X, img_shape, feature_name, save_dir, cmap="hot"):
    """Save raw-space variance map across all samples for one feature matrix."""
    if len(X) < 2:
        print(f"[WARN] Not enough samples for variance map: {feature_name}")
        return

    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    var_map = np.var(X, axis=0).reshape(img_shape)
    out_img = save_dir / f"{feature_name}_variance.png"
    save_image(var_map, out_img, vmin=0.0, vmax=float(np.max(var_map)), cmap=cmap)

    summary_path = save_dir / f"{feature_name}_variance_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"feature_name: {feature_name}\n")
        f.write(f"n_samples: {X.shape[0]}\n")
        f.write(f"n_features: {X.shape[1]}\n")
        f.write(f"variance_min: {float(np.min(var_map)):.8f}\n")
        f.write(f"variance_max: {float(np.max(var_map)):.8f}\n")
        f.write(f"variance_mean: {float(np.mean(var_map)):.8f}\n")

    print(f"[VAR] Saved variance map for {feature_name} to {out_img}")


def lda_plot(X, meta_df, color_by="word", style_by="subject",
             out_path="lda.png", title="LDA"):
    """原始 LDA 散点图（自适应颜色和形状）"""
    if len(X) < 2:
        print(f"[WARN] Not enough samples for LDA: {title}")
        return None, None

    labels = meta_df[color_by].values
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        print(f"[WARN] Need at least 2 classes for LDA, found {len(unique_labels)}")
        return None, None
    min_samples_per_class = min(counts)
    if min_samples_per_class < 2:
        print(f"[WARN] LDA requires at least 2 samples per class. Using PCA instead.")
        pca_plot(X, meta_df, color_by, style_by, out_path, title + " (PCA fallback)")
        return None, None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_components = min(2, len(unique_labels) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    Z = lda.fit_transform(Xs, labels)
    if Z.shape[1] == 1:
        Z = np.hstack([Z, np.zeros((len(Z), 1))])

    df = meta_df.copy()
    df["LD1"] = Z[:, 0]
    df["LD2"] = Z[:, 1]

    # 动态分配颜色和形状（与 pca_plot 逻辑相同）
    color_categories = sorted(df[color_by].unique())
    style_categories = sorted(df[style_by].unique())

    cmap = plt.cm.get_cmap('tab10')
    color_list = [cmap(i % cmap.N) for i in range(len(color_categories))]
    color_map = {cat: color_list[i] for i, cat in enumerate(color_categories)}

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x', 'd']
    if len(style_categories) > len(markers):
        print(f"[WARN] {len(style_categories)} subjects, but only {len(markers)} markers available. Will cycle markers.")
    style_to_marker = {cat: markers[i % len(markers)] for i, cat in enumerate(style_categories)}

    plt.figure(figsize=(8, 6))
    for cat in color_categories:
        for subj in style_categories:
            sub = df[(df[color_by] == cat) & (df[style_by] == subj)]
            if len(sub) == 0:
                continue
            plt.scatter(
                sub["LD1"],
                sub["LD2"],
                label=f"{cat} | {subj}",
                marker=style_to_marker[subj],
                color=color_map[cat],
                alpha=0.75
            )

    explained_ratio = lda.explained_variance_ratio_ if hasattr(lda, 'explained_variance_ratio_') else [1.0, 0.0]
    plt.xlabel(f"LD1 ({explained_ratio[0] * 100:.1f}%)")
    plt.ylabel(f"LD2 ({explained_ratio[1] * 100:.1f}%)")
    plt.title(title)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    # 留一法分类准确率（保持不变）
    loo = LeaveOneOut()
    svm = SVC(kernel='linear')
    try:
        scores = cross_val_score(svm, Xs, labels, cv=loo, scoring='accuracy')
        print(f"[LDA] {title}: LOO accuracy = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    except Exception as e:
        print(f"[LDA] {title}: Could not compute LOO accuracy: {e}")

    return lda, scaler


# =========================================================
# One-vs-Rest LDA weight map visualization
# =========================================================
def visualize_one_vs_rest_weights(features, labels, img_shape, feature_name, save_dir, pca_components=50):
    """
    对每个类别，训练一个“该类 vs 其余”的二分类 LDA，
    并将组合权重映射回像素空间，分别保存红蓝权重图和重要性热图。

    Parameters:
    -----------
    features : np.ndarray, shape (n_samples, n_pixels)
    labels : array-like, shape (n_samples,)
    img_shape : tuple (H, W)
    feature_name : str
    save_dir : str
    pca_components : int
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import LabelEncoder

    # PCA 降维
    n_samples, n_features = features.shape
    max_components = min(pca_components, n_features, n_samples - 1)
    pca = PCA(n_components=max_components)
    features_pca = pca.fit_transform(features)
    print(
        f"[{feature_name}] PCA explained variance: {pca.explained_variance_ratio_.sum():.3f} (using {max_components} components)")

    le = LabelEncoder()
    le.fit(labels)
    class_names = le.classes_

    # 对每个类别训练二分类 LDA
    for target_class in class_names:
        # 创建二分类标签：1 表示 target_class，0 表示其他
        binary_labels = (labels == target_class).astype(int)

        # 标准化 + LDA（输出 1 维）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_pca)
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X_scaled, binary_labels)

        # 组合权重：原始像素 → PCA → LDA
        combined_weight = pca.components_.T @ lda.scalings_  # (n_features, 1)
        w_img = combined_weight.reshape(img_shape)

        # 可视化
        vmax = np.abs(w_img).max()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im1 = axes[0].imshow(w_img, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[0].set_title(f'{target_class} vs rest\nweights')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(np.abs(w_img), cmap='hot')
        axes[1].set_title(f'{target_class} vs rest\nimportance')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        plt.suptitle(f'{feature_name} - One-vs-Rest LDA weight map')
        out_path = os.path.join(save_dir, f'{feature_name}_OvR_{target_class}_weights.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


# =========================================================
# Main feature extraction and analysis
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True,
                        help="processed_split root")
    parser.add_argument("--output_root", type=str, default="analysis_outputs",
                        help="output directory")
    parser.add_argument("--size", type=int, nargs=2, default=[128, 128],
                        help="target size H W")
    parser.add_argument("--frames", type=int, default=24,
                        help="resampled frame count")
    parser.add_argument("--norm", type=str, default="none", choices=["none", "video_zscore"],
                        help="normalization mode")
    parser.add_argument("--diff_lag", type=int, default=1,
                        help="difference lag for motion features")
    parser.add_argument("--save_npy", action="store_true",
                        help="save per-video npy arrays")
    parser.add_argument("--max_videos", type=int, default=-1,
                        help="debug limit, -1 means all")
    parser.add_argument("--save_pca_modes", action="store_true",
                        help="save PCA scree plots and PCA mode visualizations")
    parser.add_argument("--pca_mode_components", type=int, default=5,
                        help="number of PCA modes to visualize")
    parser.add_argument("--pca_scree_components", type=int, default=10,
                        help="number of components to show in scree plots")
    parser.add_argument("--pca_mode_scale", type=float, default=2.0,
                        help="scale factor for PCA mode visualization")
    args = parser.parse_args()

    target_h, target_w = args.size
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    features_dir = output_root / "features"
    previews_dir = output_root / "previews"
    plots_dir = output_root / "plots"
    per_video_dir = output_root / "per_video_npy"
    weight_maps_dir = plots_dir / "lda_weight_maps"
    pca_diag_dir = plots_dir / "pca_diagnostics"
    variance_maps_dir = plots_dir / "variance_maps"

    ensure_dir(features_dir)
    ensure_dir(previews_dir)
    ensure_dir(plots_dir)
    ensure_dir(weight_maps_dir)
    ensure_dir(variance_maps_dir)
    if args.save_pca_modes:
        ensure_dir(pca_diag_dir)
    ensure_dir(previews_dir / "raw_midframes")
    ensure_dir(previews_dir / "raw_meanframes")
    ensure_dir(previews_dir / "mean_absdiff_maps")
    ensure_dir(previews_dir / "motion_curves")
    if args.save_npy:
        ensure_dir(per_video_dir)

    videos = list_videos(input_root)
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    print(f"[INFO] found {len(videos)} videos")

    meta_rows = []
    feat_raw_mid = []
    feat_raw_mean = []
    feat_mean_absdiff = []

    for i, vp in enumerate(videos, 1):
        try:
            frames, fps, orig_n, orig_h, orig_w = read_video_gray(
                vp, target_size=(target_w, target_h), target_frames=args.frames
            )
            frames = normalize_video(frames, mode=args.norm)

            diff = compute_differences(frames, lag=args.diff_lag)
            e = motion_energy(diff)

            mid_idx = len(frames) // 2
            raw_mid = frames[mid_idx]
            raw_mean = np.mean(frames, axis=0)
            mean_absdiff = np.mean(np.abs(diff), axis=0)

            summary = summarize_motion(diff)
            subject, word, split = parse_subject_word_from_path(vp)
            clip_id = vp.stem

            meta_rows.append({
                "clip_id": clip_id,
                "video_path": str(vp),
                "split": split,
                "subject": subject,
                "word": word,
                "fps": fps,
                "orig_n_frames": orig_n,
                "orig_h": orig_h,
                "orig_w": orig_w,
                "resampled_frames": args.frames,
                "target_h": target_h,
                "target_w": target_w,
                "diff_lag": args.diff_lag,
                **summary
            })

            feat_raw_mid.append(raw_mid.flatten())
            feat_raw_mean.append(raw_mean.flatten())
            feat_mean_absdiff.append(mean_absdiff.flatten())

            # 预览图
            name = f"{i:04d}_{subject}_{word}_{clip_id}"
            save_image(raw_mid, previews_dir / "raw_midframes" / f"{name}.png")
            save_image(raw_mean, previews_dir / "raw_meanframes" / f"{name}.png")
            save_image(mean_absdiff, previews_dir / "mean_absdiff_maps" / f"{name}.png")
            plot_motion_curve(e, previews_dir / "motion_curves" / f"{name}.png",
                              title=f"{subject} | {word} | {clip_id}")

            if args.save_npy:
                out_subdir = per_video_dir / name
                ensure_dir(out_subdir)
                np.save(out_subdir / "raw_frames.npy", frames)
                np.save(out_subdir / "diff_frames.npy", diff)
                np.save(out_subdir / "motion_energy.npy", e)

            print(f"[{i}/{len(videos)}] OK: {vp}")

        except Exception as ex:
            print(f"[WARN] Failed on {vp}: {ex}")

    # =====================================================
    # Save metadata and features
    # =====================================================
    meta_df = pd.DataFrame(meta_rows)
    if len(meta_df) == 0:
        print("[ERROR] no valid videos processed")
        return

    meta_csv = output_root / "metadata.csv"
    meta_df.to_csv(meta_csv, index=False)
    print(f"[SAVE] {meta_csv}")

    feat_raw_mid = np.stack(feat_raw_mid, axis=0)
    feat_raw_mean = np.stack(feat_raw_mean, axis=0)
    feat_mean_absdiff = np.stack(feat_mean_absdiff, axis=0)

    np.save(features_dir / "raw_midframe.npy", feat_raw_mid)
    np.save(features_dir / "raw_meanframe.npy", feat_raw_mean)
    np.save(features_dir / "mean_absdiff_map.npy", feat_mean_absdiff)
    print("[SAVE] feature arrays done")

    # =====================================================
    # Raw-space variance maps
    # =====================================================
    print("\n[INFO] Generating variance maps...")
    save_variance_map(feat_raw_mid, (target_h, target_w), "Raw_Mid_Frame", variance_maps_dir)
    save_variance_map(feat_raw_mean, (target_h, target_w), "Raw_Mean_Frame", variance_maps_dir)
    save_variance_map(feat_mean_absdiff, (target_h, target_w), "Mean_Abs_Diff_Map", variance_maps_dir)

    # =====================================================
    # PCA plots
    # =====================================================
    print("\n[INFO] Generating PCA plots...")
    pca_plot(feat_raw_mid, meta_df, out_path=plots_dir / "pca_raw_midframe.png",
             title="PCA - Raw Mid Frame")
    pca_plot(feat_raw_mean, meta_df, out_path=plots_dir / "pca_raw_meanframe.png",
             title="PCA - Raw Mean Frame")
    pca_plot(feat_mean_absdiff, meta_df, out_path=plots_dir / "pca_mean_absdiff_map.png",
             title="PCA - Mean Abs Diff Map")

    if args.save_pca_modes:
        print("\n[INFO] Generating PCA diagnostics...")
        save_pca_diagnostics(
            feat_raw_mid, (target_h, target_w), "Raw_Mid_Frame", pca_diag_dir,
            n_mode_components=args.pca_mode_components,
            n_scree_components=args.pca_scree_components,
            mode_scale=args.pca_mode_scale
        )
        save_pca_diagnostics(
            feat_raw_mean, (target_h, target_w), "Raw_Mean_Frame", pca_diag_dir,
            n_mode_components=args.pca_mode_components,
            n_scree_components=args.pca_scree_components,
            mode_scale=args.pca_mode_scale
        )
        save_pca_diagnostics(
            feat_mean_absdiff, (target_h, target_w), "Mean_Abs_Diff_Map", pca_diag_dir,
            n_mode_components=args.pca_mode_components,
            n_scree_components=args.pca_scree_components,
            mode_scale=args.pca_mode_scale
        )

    # =====================================================
    # LDA scatter plots (original multi-class)
    # =====================================================
    print("\n[INFO] Generating LDA scatter plots...")
    lda_mean, scaler_mean = lda_plot(feat_raw_mean, meta_df,
                                     out_path=plots_dir / "lda_raw_meanframe.png",
                                     title="LDA - Raw Mean Frame")
    lda_mid, scaler_mid = lda_plot(feat_raw_mid, meta_df,
                                   out_path=plots_dir / "lda_raw_midframe.png",
                                   title="LDA - Raw Mid Frame")
    lda_absdiff, scaler_absdiff = lda_plot(feat_mean_absdiff, meta_df,
                                           out_path=plots_dir / "lda_mean_absdiff_map.png",
                                           title="LDA - Mean Abs Diff Map")

    # =====================================================
    # One-vs-Rest LDA weight maps (for each word)
    # =====================================================
    print("\n[INFO] Generating One-vs-Rest LDA weight maps...")
    # Raw Mean Frame
    visualize_one_vs_rest_weights(feat_raw_mean, meta_df['word'].values,
                                  (target_h, target_w), "Raw_Mean_Frame",
                                  weight_maps_dir, pca_components=50)
    # Raw Mid Frame
    visualize_one_vs_rest_weights(feat_raw_mid, meta_df['word'].values,
                                  (target_h, target_w), "Raw_Mid_Frame",
                                  weight_maps_dir, pca_components=50)
    # Mean Abs Diff Map
    visualize_one_vs_rest_weights(feat_mean_absdiff, meta_df['word'].values,
                                  (target_h, target_w), "Mean_Abs_Diff_Map",
                                  weight_maps_dir, pca_components=50)

    print(f"\n[DONE] all outputs saved to: {output_root}")
    print("\n[SUMMARY] Generated plots:")
    print(f"  - PCA: {plots_dir}/pca_*.png")
    print(f"  - Variance maps: {variance_maps_dir}/")
    if args.save_pca_modes:
        print(f"  - PCA diagnostics: {pca_diag_dir}/")
    print(f"  - LDA scatter: {plots_dir}/lda_*.png")
    print(f"  - One-vs-Rest weight maps: {weight_maps_dir}/")


if __name__ == "__main__":
    main()



"""
python analyze_processed_split2.py --input_root /Users/apple/Downloads/Troat/workspace/processed_split --output_root /Users/apple/Downloads/Troat/workspace/analysis_outputs --size 128 128 --frames 24      

"""
