"""Fold-local feature evaluation. No video decoding or plotting dependencies.

The input must contain one row per independent clip. A Pipeline protects only
the transforms fitted here; it cannot repair leakage in precomputed videos.
"""

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted


MODELS = ("linear_svm", "pca_svm", "lda_svm", "pca_lda_svm")
CV_MODES = ("loso", "group_kfold", "sample_loo")
SPLIT_NAMES = {"train", "trainset", "val", "valid", "validation", "valset", "test", "testset"}


def parse_subject_word_from_path(video_path):
    """Read directory metadata only; a filename is never a class label.

    Supports Subj3-word, Subj3_word, Subj3/word and word/Subj3.
    Ambiguous layouts fail instead of silently assigning an unknown class.
    """
    parts = Path(video_path).parent.parts
    splits = [part.lower() for part in parts if part.lower() in SPLIT_NAMES]
    if len(set(splits)) > 1:
        raise ValueError(f"Ambiguous split directories: {video_path}")
    split = splits[-1] if splits else "unknown_split"
    combined = []
    for part in parts:
        match = re.fullmatch(r"(Subj[^_\-/\\]+)[-_](.+)", part, re.IGNORECASE)
        if match:
            combined.append((match[1].lower(), match[2].lower()))
    if combined:
        if len(set(combined)) != 1:
            raise ValueError(f"Ambiguous subject/word directories: {video_path}")
        return *combined[0], split
    subjects = [(i, p.lower()) for i, p in enumerate(parts)
                if re.fullmatch(r"Subj[^_\-/\\]+", p, re.IGNORECASE)]
    if len(subjects) == 1:
        idx, subject = subjects[0]
        # Only accept adjacent word folders, not an arbitrary ancestor name.
        if idx == len(parts) - 2:
            word = parts[-1].lower()
        elif idx == len(parts) - 1 and idx > 0 and parts[idx - 1].lower() not in SPLIT_NAMES:
            word = parts[idx - 1].lower()
        else:
            raise ValueError(f"Missing word directory: {video_path}")
        if word not in SPLIT_NAMES and word not in {"full", "crop"}:
            return subject, word, split
    raise ValueError(f"Cannot parse subject/word directories: {video_path}")


def source_clip_id(subject, word, clip_id):
    """Conservative identity for known full/crop derivatives (not a content hash)."""
    stem = re.sub(r"_(full|crop)$", "", str(clip_id), flags=re.IGNORECASE)
    return f"{subject}/{word}/{stem}"


class FoldPCA(TransformerMixin, BaseEstimator):
    """Cap a requested component count using only the current training fold."""

    def __init__(self, n_components=50):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = check_array(X)
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise ValueError("pca_components must be a positive integer")
        self.n_features_in_ = X.shape[1]
        self.n_components_ = min(self.n_components, X.shape[0] - 1, X.shape[1])
        if self.n_components_ < 1:
            raise ValueError("PCA needs at least two training samples")
        self.pca_ = PCA(n_components=self.n_components_, svd_solver="full").fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "pca_")
        return self.pca_.transform(X)


def make_pipeline(model, pca_components=50, svm_c=1.0):
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if not np.isfinite(svm_c) or svm_c <= 0:
        raise ValueError("svm_c must be finite and positive")
    steps = [("scale", StandardScaler())]
    if "pca" in model:
        steps.append(("pca", FoldPCA(pca_components)))
    if "lda" in model:
        # All available discriminant dimensions, learned from training labels.
        # The exploratory figure's two-dimensional projection is not reused.
        steps.append(("lda", LinearDiscriminantAnalysis(solver="svd")))
    steps.append(("svm", SVC(kernel="linear", C=svm_c)))
    return Pipeline(steps)


def validate_inputs(X, metadata):
    X = check_array(X, dtype="numeric", ensure_min_samples=2)
    if len(metadata) != len(X):
        raise ValueError("Feature rows and metadata rows must match in order and length")
    meta = metadata.copy().reset_index(drop=True)
    for column in ("word", "subject", "clip_id", "split"):
        if column not in meta:
            raise ValueError(f"Missing metadata column: {column}")
        if meta[column].isna().any():
            raise ValueError(f"Missing values in {column}")
        values = meta[column].astype(str)
        if (values.str.strip() != values).any() or (values == "").any():
            raise ValueError(f"Empty or whitespace-padded values in {column}")
        if column in ("subject", "word"):
            invalid = values.str.lower().isin(["unknown", "unknown_subject", "unknown_word", "nan", "none"])
            if invalid.any():
                raise ValueError(f"Unknown {column}; supply verified metadata")
            # Avoid silently treating case aliases as independent groups/classes.
            if values.nunique() != values.str.lower().nunique():
                raise ValueError(f"Case aliases in {column}; resolve metadata explicitly")
        meta[column] = values
    for column in ("video_path", "source_clip_id"):
        if column in meta and (meta[column].isna().any() or meta[column].astype(str).str.strip().eq("").any()):
            raise ValueError(f"Missing values in {column}")
    if "video_path" in meta and meta.video_path.duplicated().any():
        raise ValueError("Duplicate video_path rows")
    inferred = [source_clip_id(s, w, c) for s, w, c in zip(meta.subject, meta.word, meta.clip_id)]
    if pd.Series(inferred).duplicated().any():
        raise ValueError("Duplicate source clips or full/crop variants; use one representation per clip")
    if "source_clip_id" not in meta:
        meta["source_clip_id"] = inferred
    if meta.source_clip_id.duplicated().any():
        raise ValueError("Duplicate source_clip_id; use one independent clip per row")
    if meta.word.nunique() < 2:
        raise ValueError("Classification requires at least two word classes")
    return X, meta


def select_splits(X, metadata, splits):
    """Require an explicit cohort; never silently pool a reserved test directory."""
    if not splits or "split" not in metadata:
        raise ValueError("Explicit evaluation splits and a split column are required")
    if len(X) != len(metadata):
        raise ValueError("Feature rows and metadata rows must match")
    available = set(metadata["split"].astype(str))
    if splits == ["all"]:
        mask = np.ones(len(metadata), dtype=bool)
    else:
        if not set(splits) <= available:
            raise ValueError(f"Requested splits absent; available: {sorted(available)}")
        mask = metadata["split"].isin(splits).to_numpy()
    return np.asarray(X)[mask], metadata.loc[mask].reset_index(drop=True)


def _metrics(y, predicted, classes):
    present = np.unique(y)
    return {
        "n_samples": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(recall_score(y, predicted, labels=present, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=classes).tolist(),
    }


def evaluate_features(X, metadata, *, model="lda_svm", cv="loso", n_splits=5,
                      pca_components=50, svm_c=1.0, data_kind="unspecified",
                      preprocessing_note="Upstream preprocessing has not been audited."):
    """Return auditable out-of-fold predictions; fail on any invalid fold.

    Hyperparameters are fixed inputs. Selection among these results requires
    nested, subject-grouped validation or a separate untouched cohort.
    """
    X, meta = validate_inputs(X, metadata)
    if data_kind not in {"real", "synthetic", "unspecified"}:
        raise ValueError("data_kind must be real, synthetic or unspecified")
    y = meta.word.to_numpy()
    groups = meta.subject.to_numpy()
    classes = np.unique(y)
    if cv == "sample_loo":
        splitter = LeaveOneOut()
    elif cv in ("loso", "group_kfold"):
        if len(np.unique(groups)) < 2:
            raise ValueError("Subject-held-out evaluation needs at least two subjects")
        splitter = LeaveOneGroupOut() if cv == "loso" else GroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"Unknown CV mode: {cv}")
    # Preflight every fold before fitting. A failed fold is never dropped from
    # the denominator or replaced by a silently different model.
    splits = list(splitter.split(X, y, groups if cv != "sample_loo" else None))
    for fold, (train, test) in enumerate(splits):
        train_classes, counts = np.unique(y[train], return_counts=True)
        missing = sorted(set(classes) - set(train_classes))
        if missing:
            raise ValueError(f"Fold {fold}: training is missing word classes {missing}; closed-set evaluation invalid")
        if "lda" in model and np.min(counts) < 2:
            raise ValueError(f"Fold {fold}: LDA requires at least two training samples per word in this protocol")
        if cv != "sample_loo" and set(groups[train]) & set(groups[test]):
            raise ValueError(f"Fold {fold}: subject overlap")

    estimator = make_pipeline(model, pca_components, svm_c)
    predictions = np.empty(len(y), dtype=object)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_rows = []
    for fold, (train, test) in enumerate(splits):
        fitted = clone(estimator)
        try:
            fitted.fit(X[train], y[train])
            predicted = fitted.predict(X[test])
        except (ValueError, np.linalg.LinAlgError, IndexError) as error:
            raise ValueError(f"Fold {fold} ({model}) failed: {error}") from error
        if (fold_ids[test] != -1).any():
            raise ValueError("A sample was tested more than once")
        predictions[test] = predicted
        fold_ids[test] = fold
        fold_rows.append({
            "fold": fold, "train_indices": train.tolist(), "test_indices": test.tolist(),
            "train_subjects": sorted(set(groups[train])), "test_subjects": sorted(set(groups[test])),
            "train_class_counts": meta.iloc[train].word.value_counts().sort_index().to_dict(),
            "test_class_counts": meta.iloc[test].word.value_counts().sort_index().to_dict(),
            "pca_components": fitted.named_steps["pca"].n_components_ if "pca" in fitted.named_steps else None,
            "lda_components": fitted.named_steps["lda"].transform(fitted[:-2].transform(X[train][:1])).shape[1]
            if "lda" in fitted.named_steps else None,
            **_metrics(y[test], predicted, classes),
        })
    if (fold_ids < 0).any():
        raise ValueError("Missing out-of-fold predictions")
    per_subject = {subject: _metrics(y[groups == subject], predictions[groups == subject], classes)
                   for subject in sorted(set(groups))}
    limitations = [
        preprocessing_note,
        "Pipeline isolation starts at the supplied feature matrix; upstream references, ROI decisions and acquisition confounds remain to be verified.",
        "Clip IDs detect known derivatives, not renamed duplicates or overlapping recording windows; verify the recording/session manifest.",
        "Model, feature and parameter selection on these scores requires nested validation or a separate test cohort.",
        "Fold/subject scores are descriptive; their standard deviation is not a confidence interval.",
    ]
    if cv == "sample_loo":
        limitations.append("Sample LOO may share subjects/sessions with training; it does not establish cross-subject generalisation.")
    if data_kind != "real":
        limitations.append("Synthetic or unspecified data cannot establish real-subject generalisation.")
    if meta["split"].nunique() > 1 or meta["split"].str.lower().isin(["test", "testset"]).any():
        limitations.append("The selected cohort includes multiple splits or a test split. CV reuses these rows for training; this is not an untouched external test.")
    coverage = pd.crosstab(meta.subject, meta.word).reindex(columns=classes, fill_value=0)
    if (coverage == 0).any().any():
        limitations.append("Not every subject has every word. Per-subject balanced accuracy uses observed words only; macro F1 uses the global class list.")
    report = {
        "schema_version": 1, "model": model, "cv": cv, "data_kind": data_kind,
        "evaluation_unit": "clip" if cv == "sample_loo" else "held_out_subject",
        "pipeline": [name for name, _ in estimator.steps],
        "parameters": {"svm_c": svm_c, "pca_components_requested": pca_components, "group_kfold_splits": n_splits if cv == "group_kfold" else None},
        "n_subjects": len(per_subject), "n_folds": len(splits), "classes": classes.tolist(),
        "selected_split_counts": meta["split"].value_counts().sort_index().to_dict(),
        "subject_word_counts": coverage.to_dict(orient="index"),
        "pooled_oof": _metrics(y, predictions, classes), "per_subject": per_subject,
        "subject_macro_accuracy": float(np.mean([row["accuracy"] for row in per_subject.values()])),
        "metric_definitions": {"accuracy": "Fraction correct; all metric values are in [0,1].",
                               "balanced_accuracy": "Macro recall over words observed in the evaluated subset.",
                               "macro_f1": "Macro F1 over the report's full class list; zero for absent/undefined classes.",
                               "subject_macro_accuracy": "Unweighted mean of per-subject OOF accuracy; not a confidence interval."},
        "folds": fold_rows, "limitations": limitations,
        "input_sha256": hashlib.sha256(np.ascontiguousarray(X).tobytes() + meta.to_csv(index=False).encode()).hexdigest(),
        "feature_shape": list(X.shape), "feature_dtype": str(X.dtype),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    out = meta.copy()
    out["row_index"] = np.arange(len(meta))
    out["fold"] = fold_ids
    out["true_word"] = y
    out["predicted_word"] = predictions
    return report, out


def save_evaluation(report, predictions, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    predictions.to_csv(output / "predictions.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Unscaled per-clip .npy matrix")
    parser.add_argument("--metadata", required=True, help="CSV in exactly the same row order")
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", required=True, help="Explicit split names, or all (reuses any test rows)")
    parser.add_argument("--cv", choices=CV_MODES, default="loso")
    parser.add_argument("--model", choices=MODELS, default="lda_svm")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--data_kind", choices=["real", "synthetic", "unspecified"], required=True)
    parser.add_argument("--preprocessing_note", default="Upstream preprocessing has not been audited.")
    args = parser.parse_args()
    X, meta = select_splits(np.load(args.features, allow_pickle=False), pd.read_csv(args.metadata, dtype=str), args.splits)
    report, predictions = evaluate_features(X, meta, model=args.model, cv=args.cv, n_splits=args.n_splits,
                                           pca_components=args.pca_components, svm_c=args.svm_c,
                                           data_kind=args.data_kind, preprocessing_note=args.preprocessing_note)
    save_evaluation(report, predictions, args.output)
    print(f"[{args.data_kind}] {args.cv} / {args.model}: results saved to {args.output}")
    for limitation in report["limitations"]:
        print(f"[LIMITATION] {limitation}")


if __name__ == "__main__":
    main()
