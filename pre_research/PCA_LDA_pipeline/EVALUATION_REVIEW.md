# PCA/LDA/SVM evaluation review — 2026-11-04

Reviewed baseline: [`ade1af8ff9166d428b5cf63a8b4407e2057ecbe7`](https://github.com/Tasksuss/ZX-1/tree/ade1af8ff9166d428b5cf63a8b4407e2057ecbe7).
Scope: the five PCA/LDA scripts and method notes, plus the relevant simulation
loaders, configuration, profiling, generation and model-selection paths. This is
not a certification of the repository's notebooks, all experimental results or
any conference/journal manuscript. The repository does not establish which code
revision generated a particular published result.

No original throat-video dataset, matched feature/metadata export or acquisition
manifest was available for this review. No real-data accuracy was calculated.

## Findings and disposition

| Finding | Evidence at the reviewed baseline | Disposition |
| --- | --- | --- |
| Global feature scaling leaks held-out rows | `05_analyze_processed_split2.py:lda_plot` fits StandardScaler on all rows before `cross_val_score` | Fixed: fresh Pipeline fitted separately in every fold. |
| “LDA LOO” evaluates the wrong model and unit | The scored estimator is `SVC(kernel='linear')` on `Xs`; LDA's `Z` is not used; `LeaveOneOut` receives no subjects | Fixed: explicit `linear_svm`, `pca_svm`, `lda_svm`, `pca_lda_svm`; separate `sample_loo`, `loso`, `group_kfold`. |
| Supervised plots are overinterpreted | `LDA_method_notes.md:Interpretation` asserts reproducible patterns across speakers | Corrected: exploratory full-cohort projections, no held-out claim. |
| Filename can be misread as a class or subject | Path parser scans the filename before directories; greedy subject pattern accepts underscores; `trainset`/`testset` are not recognised | Fixed: directory-only parsing; explicit split aliases; ambiguous or unknown labels fail. Unusual layouts require a verified metadata export. |
| Full/crop copies can become two samples | `04_split_full_crop.py` creates two representations; `05` recursively reads both if given the common parent | Fixed for known derivatives: duplicate source identity is rejected. Choose one branch. Renamed duplicates and overlapping windows still need a manifest/content audit. |
| Binary LDA and one-feature PCA plots can index a nonexistent axis | Both plots pad coordinates to two dimensions but index variance ratio `[1]` unconditionally | Fixed: absent second axis is labelled as zero padding. |
| OvR pixel weights use the wrong scale | PCA scores are standardised, but backprojection omits division by `scaler.scale_` | Fixed and numerically tested; output renamed as projection weights rather than physiological importance. |
| Upstream references are fitted outside evaluation | `01_build_folder_references.py` samples several videos per leaf folder and pools folder statistics; `02` annotates a center per reference; `03` applies the matching folder reference/center to its clips | **Unresolved at the raw-video boundary.** Shared clips can contaminate sample LOO, and word-indexed references/ROIs can use target information even under LOSO. Already processed pixels cannot be repaired by the downstream Pipeline. |
| Simulation selects on a nominal real test set | `Simulation_pipeline/run_size_study.py:run_experiments` selects `best_model.pth` using minimum `real_test_loss`, then reports performance on that same set | **Legacy development result.** Requires a separate validation cohort and an untouched final test. Training code has not been redesigned in this focused patch. |
| Phase 2 selects on the nominal synthetic test set | `config_phase2.json` uses `synthetic_test_loss`; `run_phase2_search.py` uses it for checkpoint selection/early stopping and compares settings/thresholds | Synthetic scores on that reused set are development scores. A new synthetic test set is required; real performance also needs audited generator provenance and no tuning on real test results. |
| Subject lists do not enforce the experiment split | Simulation `build_entries` enumerates `trainset`/`testset`; runners do not filter or assert identities against `subjects_train`/`subjects_test` | Inspect actual manifests and assert subject disjointness/class coverage before rerunning. A config alone is not evidence of real cross-subject validation. |
| Synthetic preview videos can be counted twice | Generator saves `.npy` plus same-stem `.mp4` previews; both runners accept both extensions | Effective counts can include duplicate representations within a split. Deduplicate by source ID/prefer `.npy`; this is a counting/dependence problem, not proof of cross-split leakage by itself. |
| Generator profile can include the real test cohort | `analyze_real_dataset.py` profiles all splits; generator consumes profile FPS and (unless overridden) frame-count distribution | Rebuild profiles from training/calibration data only. The code does not establish which profile was used historically or quantify this effect. |

The global brightness target in `01` is cohort-derived. In `03`, per-frame
percentile rescaling largely cancels a common positive brightness multiplier
(apart from numerical terms), so code inspection alone does not quantify its
effect. Folder reference fields and ROI choices remain separate, material
dependencies. Per-clip temporal resampling, frame differencing and optional
per-video z-scoring do not pool other clips, but use the complete clip and do not
establish an online/causal prediction protocol.

## Evaluation contract

One row represents one independent clip. Features must be **unscaled** vectors
from that clip; do not supply globally fitted PCA/LDA scores. CSV rows must be in
exactly the same order as feature rows. Required columns are `subject`, `word`,
`clip_id`, `split`; optional `source_clip_id`, `video_path`, `session_id` and
`recording_id` improve auditability. Session/recording columns are retained but
are not automatic splitters. Use globally unambiguous subject IDs and
session-qualified clip IDs where filenames repeat. Source IDs must identify the
original clip independently of full/crop, codecs or augmentation.

The script validates finite values, row counts, labels, known duplicates,
subject counts and training-class coverage. It cannot prove that rows were
correctly paired or that different names identify different people/clips.
All global word classes must appear in every training fold; otherwise evaluation
fails instead of silently dropping that fold/class. The LDA protocol also
requires at least two training samples per class. Missing words in a held-out
subject are reported, not filled with invented samples.

Each fold clones and fits a Pipeline:

| Model | Steps |
| --- | --- |
| `linear_svm` | StandardScaler → linear SVC |
| `pca_svm` | StandardScaler → PCA → linear SVC |
| `lda_svm` | StandardScaler → supervised LDA projection → linear SVC |
| `pca_lda_svm` | StandardScaler → PCA → supervised LDA projection → linear SVC |

SVM `C=1` by default. PCA uses deterministic full SVD and caps the requested
dimension at `min(requested, n_train - 1, n_features)` inside each training fold.
LDA uses SVD and all available discriminant dimensions (at most `K_train - 1`),
not just the two dimensions shown in a plot. No tuning is performed. Component
counts, features, C and preprocessing choices must be fixed before evaluation;
choosing the best score among runs requires **nested subject-grouped validation**
or a separate untouched cohort.

- `sample_loo`: leaves one clip out. Subjects/sessions can overlap training;
  this is not a test of unseen people and may still be dependent within recordings.
- `loso` (default): leaves one whole subject out using LeaveOneGroupOut.
- `group_kfold`: leaves whole subject groups out using GroupKFold. The requested
  number of folds must be between 2 and the number of subjects. It is not
  class-stratified; class coverage is checked before fitting.

These splits act on the selected **feature cohort**. Their correctness does not
certify upstream preprocessing. If held-out subjects require calibration,
specify a label-blind calibration procedure with separate calibration clips;
never choose a reference or ROI using the test clip's true word. For a fully
inductive claim, regenerate learned preprocessing within each outer fold from
training data only. Subject-level generalisation is also conditional on the
sampled population, hardware, vocabulary and acquisition protocol.

## Running the corrected workflow

From the repository root, install the recorded environment (tested with Python
3.12):

```bash
python -m pip install -r pre_research/PCA_LDA_pipeline/requirements-evaluation.txt
python -m unittest discover -s tests -p 'test_feature_evaluation.py' -v
```

Generate exploratory figures as before, pointing to **one** crop/full branch.
Evaluation is opt-in because the legacy root may include a reserved test set.
This example uses only `trainset` as a development cohort:

```bash
python pre_research/PCA_LDA_pipeline/05_analyze_processed_split2.py \
  --input_root /path/to/processed_split/crop \
  --output_root /path/to/new_analysis_run \
  --evaluate --evaluation_splits trainset --data_kind real \
  --cv loso sample_loo \
  --models linear_svm lda_svm pca_svm pca_lda_svm
```

Choose the actual split names in the dataset (`train`, `trainset`, etc.). Use
`--evaluation_splits all` only when deliberately evaluating the whole cohort:
CV will reuse nominal test rows for training in other folds, consuming their
status as a reserved test set. The report records this limitation. The
exploratory figures still use all clips supplied through `--input_root`; point
that root to development-only data if a separate test set must remain unseen.

To evaluate already exported raw per-clip features without decoding videos:

```bash
python pre_research/PCA_LDA_pipeline/evaluation.py \
  --features /path/to/new_analysis_run/features/mean_absdiff_map.npy \
  --metadata /path/to/new_analysis_run/metadata.csv \
  --output /path/to/evaluation_run \
  --splits trainset --data_kind real \
  --cv group_kfold --n_splits 3 --model pca_lda_svm --pca_components 20
```

Use `--data_kind synthetic` for generated examples. Merely setting `real` is a
declaration, not proof of provenance. `--preprocessing_note` records the actual
upstream procedure; it is not an approval switch or a substitute for evidence.
Keep the original metadata and extraction settings with every feature export.

Each evaluation writes `metrics.json` and `predictions.csv`, including:

- model, data origin, explicit split counts, software versions and an input hash;
- fold membership and indices, train/test subjects and per-word counts;
- out-of-fold predictions in input order, class-ordered confusion matrices;
- pooled clip accuracy, balanced accuracy, macro F1 and per-subject results;
- unweighted mean subject accuracy, distinct from pooled clip accuracy;
- effective PCA/LDA dimensions and unresolved interpretation limitations.

Metrics are fractions in `[0, 1]`. Balanced accuracy averages recall over words
observed in the evaluated subset. Macro F1 uses the full class list and assigns
zero to undefined classes; see coverage before comparing subjects. No confidence
interval is inferred from single-sample LOO scores or overlapping CV training
sets. Subject-aware uncertainty estimates and a baseline/permutation protocol
should be specified for a real-data study.

## Validation completed

17 automated tests use generated arrays and temporary generated videos only.
They inspect actual fit inputs to scaler/PCA/LDA, check held-out label isolation,
all four models and all three splitters, complete OOF coverage, fixed confusion
matrix ordering, unequal subject weighting, fold-local PCA limits, binary and
high-dimensional LDA, invalid/unknown labels, infeasible folds, duplicate
full/crop clips, finite inputs, explicit split selection, report reproducibility,
binary plotting, raw-space weight backprojection and full extraction-to-report
execution. They do not estimate physiological performance or validate the
legacy simulation neural-network training workflow.

## What is still required for real-data conclusions

1. Raw videos or a matched feature export **plus** a verifiable preprocessing
   history; independent calibration data/references where needed. Already
   contaminated processed images may require regeneration.
2. Verified word/subject/session/recording/clip manifest, file hashes, full/crop
   and augmentation lineage, exclusion/QC log, and information about overlapping
   windows. Resolve label spellings from the actual experimental codebook; do
   not infer the class count from a different experiment's config.
3. Subject-by-word counts, repeat counts, acquisition conditions and the number
   of genuinely independent subjects. Passing software checks is not evidence
   that the cohort is large or balanced enough for a reliable estimate.
4. A declared evaluation target and cohort: new clips from known people, new
   sessions, unseen people, or synthetic-to-real transfer; reserved test subjects;
   fixed feature/model choices or an inner grouped tuning plan; appropriate
   baselines and subject-level uncertainty.
5. For simulation: deduplicated entries, asserted subject splits, training-only
   real profiles, a separate validation set for checkpoints/settings, and a
   fresh final test. Previously inspected test sets cannot be made untouched by
   renaming them.

Supported now: explicit exploratory projections and tested machinery for
fold-local sample/subject evaluation. Unsupported now: any new real accuracy,
reproducible cross-speaker word physiology, superiority of motion features,
or validity/invalidity of a manuscript result without its actual experiment
provenance. No numerical correction to historical accuracy can be inferred.

Method references: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
and [cross-validation with grouped data](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
