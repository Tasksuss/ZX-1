"""Protocol regression tests using generated arrays/videos only."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pre_research" / "PCA_LDA_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
import evaluation as ev

spec = importlib.util.spec_from_file_location("analysis", PIPELINE_DIR / "05_analyze_processed_split2.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(repeats=3, subjects=3, words=3, features=7):
    rng = np.random.default_rng(218)
    rows = [{"subject": f"s{s}", "word": f"w{w}", "clip_id": f"c{r}", "split": "trainset"}
            for s in range(subjects) for w in range(words) for r in range(repeats)]
    meta = pd.DataFrame(rows)
    X = rng.normal(size=(len(meta), features))
    X[:, 0] += meta.word.str[1:].astype(int).to_numpy()
    X[meta.subject == "s2"] += 500  # Held-out distribution must not affect its own training fold.
    return X, meta


class EvaluationTests(unittest.TestCase):
    def test_every_transform_sees_training_rows_and_labels_only(self):
        X, meta = fixture()
        seen = {"scale": [], "pca": [], "lda": []}

        class AuditScale(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                seen["scale"].append(np.array(X))
                return super().fit(X, y, sample_weight)

        class AuditPCA(PCA):
            def fit(self, X, y=None):
                seen["pca"].append(np.array(X))
                return super().fit(X, y)

        class AuditLDA(LinearDiscriminantAnalysis):
            def fit(self, X, y):
                seen["lda"].append((np.array(X), np.array(y)))
                return super().fit(X, y)

        with patch.object(ev, "StandardScaler", AuditScale), patch.object(ev, "PCA", AuditPCA), patch.object(ev, "LinearDiscriminantAnalysis", AuditLDA):
            report, _ = ev.evaluate_features(X, meta, model="pca_lda_svm", pca_components=4)
        self.assertEqual(len(seen["scale"]), 3)
        self.assertEqual(len(seen["pca"]), 3)
        self.assertEqual(len(seen["lda"]), 3)
        for i, fold in enumerate(report["folds"]):
            train = fold["train_indices"]
            expected_scaled = StandardScaler().fit_transform(X[train])
            expected_pca = PCA(n_components=4, svd_solver="full").fit_transform(expected_scaled)
            np.testing.assert_array_equal(seen["scale"][i], X[train])
            np.testing.assert_allclose(seen["pca"][i], expected_scaled)
            np.testing.assert_allclose(seen["lda"][i][0], expected_pca, atol=1e-10)
            np.testing.assert_array_equal(seen["lda"][i][1], meta.iloc[train].word)

    def test_all_models_and_splitters_have_complete_oof_predictions(self):
        X, meta = fixture()
        for model in ev.MODELS:
            for cv in ev.CV_MODES:
                with self.subTest(model=model, cv=cv):
                    report, pred = ev.evaluate_features(X, meta, model=model, cv=cv, n_splits=2, pca_components=4)
                    self.assertEqual(len(pred), len(X))
                    self.assertEqual(sorted(sum([f["test_indices"] for f in report["folds"]], [])), list(range(len(X))))
                    for fold in report["folds"]:
                        self.assertFalse(set(fold["train_indices"]) & set(fold["test_indices"]))
                        if cv != "sample_loo":
                            self.assertFalse(set(fold["train_subjects"]) & set(fold["test_subjects"]))
                    self.assertEqual(report["pooled_oof"]["accuracy"], accuracy_score(meta.word, pred.predicted_word))
                    np.testing.assert_array_equal(report["pooled_oof"]["confusion_matrix"], confusion_matrix(meta.word, pred.predicted_word, labels=report["classes"]))

    def test_held_out_labels_cannot_change_that_subjects_predictions(self):
        X, meta = fixture()
        _, first = ev.evaluate_features(X, meta, model="pca_lda_svm", pca_components=4)
        edited = meta.copy()
        mask = edited.subject == "s2"
        edited.loc[mask, "word"] = edited.loc[mask, "word"].map({"w0": "w1", "w1": "w2", "w2": "w0"})
        _, second = ev.evaluate_features(X, edited, model="pca_lda_svm", pca_components=4)
        np.testing.assert_array_equal(first.loc[mask, "predicted_word"], second.loc[mask, "predicted_word"])

    def test_pca_dimension_uses_training_size(self):
        X, meta = fixture(repeats=2, subjects=3, features=100)
        report, _ = ev.evaluate_features(X, meta, model="pca_svm", pca_components=50)
        for fold in report["folds"]:
            self.assertEqual(fold["pca_components"], len(fold["train_indices"]) - 1)

    def test_binary_and_high_dimensional_lda(self):
        X, meta = fixture(words=2, features=100)
        for model in ("lda_svm", "pca_lda_svm"):
            report, _ = ev.evaluate_features(X, meta, model=model, pca_components=4)
            self.assertTrue(all(fold["lda_components"] == 1 for fold in report["folds"]))

    def test_invalid_metadata_and_nonfinite_features_fail(self):
        X, meta = fixture()
        invalid = [meta.drop(columns="subject"), meta.iloc[:-1], meta.assign(word="only")]
        for column in ("subject", "word", "clip_id", "split"):
            bad = meta.copy()
            bad.loc[0, column] = None
            invalid.append(bad)
        for value in ("unknown_subject", "", "s0 "):
            bad = meta.copy()
            bad.loc[0, "subject"] = value
            invalid.append(bad)
        bad = meta.copy()
        bad.loc[0, "subject"] = "S0"
        invalid.append(bad)
        for bad in invalid:
            with self.subTest(columns=list(bad.columns)), self.assertRaises(ValueError):
                ev.evaluate_features(X, bad)
        for value in (np.nan, np.inf):
            bad_X = X.copy()
            bad_X[0, 0] = value
            with self.assertRaises(ValueError):
                ev.evaluate_features(bad_X, meta)

    def test_duplicates_and_full_crop_pairs_fail(self):
        X, meta = fixture()
        for first, second in [("c0", "c0"), ("c0_full", "c0_crop")]:
            bad = meta.copy()
            bad.loc[0, "clip_id"], bad.loc[1, "clip_id"] = first, second
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ev.evaluate_features(X, bad)
        with self.assertRaisesRegex(ValueError, "Duplicate source_clip_id"):
            ev.evaluate_features(X, meta.assign(source_clip_id="same"))

    def test_infeasible_group_folds_fail_without_partial_scores(self):
        X, meta = fixture()
        with self.assertRaisesRegex(ValueError, "two subjects"):
            ev.evaluate_features(X[:9], meta.iloc[:9])
        with self.assertRaises(ValueError):
            ev.evaluate_features(X, meta, cv="group_kfold", n_splits=4)
        bad = meta.copy()
        bad.loc[bad.subject == "s2", "word"] = "exclusive"
        # Make identities distinct even after changing the class labels.
        bad["clip_id"] = [str(i) for i in range(len(bad))]
        with self.assertRaisesRegex(ValueError, "missing word classes"):
            ev.evaluate_features(X, bad)
        small_X, small_meta = fixture(repeats=1, subjects=2)
        with self.assertRaisesRegex(ValueError, "two training samples"):
            ev.evaluate_features(small_X, small_meta)
        with self.assertRaisesRegex(ValueError, "Fold .* failed"):
            ev.evaluate_features(np.ones_like(X), meta)

    def test_split_selection_is_explicit_and_preserves_alignment(self):
        X, meta = fixture()
        meta.loc[meta.subject == "s2", "split"] = "testset"
        selected, rows = ev.select_splits(X, meta, ["trainset"])
        np.testing.assert_array_equal(selected, X[:18])
        self.assertEqual(len(rows), 18)
        with self.assertRaises(ValueError):
            ev.select_splits(X, meta, [])
        with self.assertRaises(ValueError):
            ev.select_splits(X, meta, ["nonexistent"])
        report, _ = ev.evaluate_features(X, meta)
        self.assertTrue(any("not an untouched external test" in s for s in report["limitations"]))

    def test_subject_macro_metric_is_not_sample_weighted(self):
        X, meta = fixture(repeats=5)
        keep = ~((meta.subject == "s2") & (meta.clip_id != "c0"))
        report, predictions = ev.evaluate_features(X[keep], meta.loc[keep], model="linear_svm")
        expected = predictions.groupby("subject").apply(lambda df: (df.true_word == df.predicted_word).mean(), include_groups=False).mean()
        self.assertAlmostEqual(report["subject_macro_accuracy"], expected)

    def test_incomplete_subject_word_coverage_is_reported(self):
        X, meta = fixture()
        keep = ~((meta.subject == "s2") & (meta.word == "w0"))
        report, _ = ev.evaluate_features(X[keep], meta.loc[keep])
        self.assertEqual(report["subject_word_counts"]["s2"]["w0"], 0)
        self.assertTrue(any("Not every subject" in s for s in report["limitations"]))

    def test_data_origin_and_sample_loo_limitations(self):
        X, meta = fixture()
        report, _ = ev.evaluate_features(X, meta, cv="sample_loo", data_kind="synthetic")
        self.assertEqual(report["data_kind"], "synthetic")
        self.assertEqual(report["evaluation_unit"], "clip")
        self.assertTrue(any("does not establish cross-subject" in s for s in report["limitations"]))
        self.assertTrue(any("cannot establish real-subject" in s for s in report["limitations"]))

    def test_cli_writes_reproducible_predictions_and_finite_json(self):
        X, meta = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "X.npy", X)
            meta.to_csv(root / "metadata.csv", index=False)
            command = [sys.executable, str(PIPELINE_DIR / "evaluation.py"), "--features", str(root / "X.npy"),
                       "--metadata", str(root / "metadata.csv"), "--output", str(root / "out"),
                       "--splits", "trainset", "--data_kind", "synthetic", "--model", "pca_lda_svm"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            before = (root / "out" / "metrics.json").read_text()
            report = json.loads(before)
            self.assertEqual(report["n_folds"], 3)
            self.assertEqual(len(pd.read_csv(root / "out" / "predictions.csv")), len(meta))
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(before, (root / "out" / "metrics.json").read_text())


class MetadataAndPlotTests(unittest.TestCase):
    def test_generated_videos_run_extraction_plots_and_grouped_evaluation(self):
        import cv2

        rng = np.random.default_rng(310)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject in range(3):
                for word in range(2):
                    folder = root / "input" / "crop" / "trainset" / f"Subj{subject}-w{word}"
                    folder.mkdir(parents=True)
                    for clip in range(2):
                        writer = cv2.VideoWriter(str(folder / f"c{clip}_crop.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
                        self.assertTrue(writer.isOpened())
                        for frame in range(5):
                            image = rng.integers(20, 180, (16, 16, 3), dtype=np.uint8)
                            image[:, 4 * word:4 * word + 4] += 35
                            writer.write(image)
                        writer.release()
            result = subprocess.run(
                [sys.executable, str(PIPELINE_DIR / "05_analyze_processed_split2.py"),
                 "--input_root", str(root / "input" / "crop"), "--output_root", str(root / "out"),
                 "--size", "4", "4", "--frames", "4", "--evaluate", "--evaluation_splits", "trainset",
                 "--data_kind", "synthetic", "--cv", "loso", "sample_loo"],
                check=True, capture_output=True, text=True,
            )
            self.assertNotIn("LOO accuracy", result.stdout)
            self.assertNotIn("[WARN] Failed", result.stdout)
            self.assertEqual(len(pd.read_csv(root / "out" / "metadata.csv")), 12)
            self.assertEqual(len(list((root / "out" / "evaluation").rglob("metrics.json"))), 12)
            self.assertEqual(len(list((root / "out" / "plots").glob("lda_*.png"))), 3)
            self.assertEqual(json.loads((root / "out" / "extraction_run.json").read_text())["failures"], [])

    def test_path_parser_ignores_filenames_and_supports_split_aliases(self):
        paths = ["crop/trainset/Subj3-Banana/Subj99_bad_crop.mp4", "crop/trainset/Subj3_banana/a.mp4",
                 "crop/trainset/Subj3/banana/a.mp4", "crop/trainset/banana/Subj3/a.mp4"]
        for path in paths:
            self.assertEqual(ev.parse_subject_word_from_path(path), ("subj3", "banana", "trainset"))
        self.assertEqual(ev.parse_subject_word_from_path("testset/Subj1-word/a.mp4")[2], "testset")
        for path in ("crop/trainset/Subj3/a.mp4", "crop/trainset/banana/a.mp4", "train/Subj1-a/test/Subj2-b/x.mp4"):
            with self.assertRaises(ValueError):
                ev.parse_subject_word_from_path(path)

    def test_binary_lda_and_single_feature_pca_plots_survive(self):
        X, meta = fixture(words=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis.lda_plot(X, meta, out_path=root / "lda.png")
            analysis.pca_plot(X[:, :1], meta, out_path=root / "pca.png")
            self.assertGreater((root / "lda.png").stat().st_size, 0)
            self.assertGreater((root / "pca.png").stat().st_size, 0)

    def test_backprojected_weights_match_pipeline_score_differences(self):
        X, meta = fixture(words=2)
        pca = PCA(n_components=4).fit(X)
        scores = pca.transform(X)
        scaler = StandardScaler().fit(scores)
        lda = LinearDiscriminantAnalysis().fit(scaler.transform(scores), meta.word)
        weights = analysis.backproject_lda_direction(pca, scaler, lda)
        projected = lda.transform(scaler.transform(pca.transform(X)))[:, :1]
        # Remove the intercept; raw linear weights must give the same differences.
        np.testing.assert_allclose((X - X[:1]) @ weights, projected - projected[:1], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
