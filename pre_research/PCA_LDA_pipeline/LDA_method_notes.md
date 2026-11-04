# LDA Method Notes

> Revised 2026-11-04: these figures are exploratory, supervised fits on the
> displayed cohort. They do not demonstrate cross-speaker generalisation.
> See [evaluation protocol and technical review](EVALUATION_REVIEW.md) for
> fold-local classification, subject-held-out evaluation and upstream limits.

## Input Data and Preprocessing

The input to the LDA analysis was not the original full-frame video, but a throat ROI video sequence that had already been cropped during the earlier preprocessing stage. For the downstream analysis implemented in `05_analyze_processed_split2.py`, each processed clip was read frame by frame, converted to grayscale, resized to a fixed spatial resolution of 128 x 128 pixels, and temporally resampled to 24 frames per video. This temporal normalization was performed over the full clip duration rather than on a per-second basis, so that every sample contributed the same number of frames to the analysis regardless of its original length.

After temporal normalization, frame differences were computed with a lag of 1 frame in order to characterize motion. Three image-derived representations were then constructed for each clip. The first was the **Raw Mid Frame**, defined as the middle frame of the resampled sequence. The second was the **Raw Mean Frame**, defined as the pixel-wise average across all resampled frames. The third was the **Mean Absolute Difference Map**, defined as the pixel-wise average of the absolute frame-to-frame differences and therefore intended to summarize the dominant motion pattern within the clip. In parallel, summary motion statistics such as motion mean, motion range, and absolute-difference descriptors were also computed and written to metadata, but these scalar descriptors were not used as input to PCA or LDA.

Each of the three image representations was flattened into a one-dimensional pixel vector, so that each clip was represented as a feature vector in a high-dimensional image space. Before LDA, the feature matrix was standardized using `StandardScaler`, which centered each pixel feature to zero mean and scaled it to unit variance across samples.

## Feature Construction

Let each processed clip be represented by a feature vector \(x_i \in \mathbb{R}^d\), where \(d = 128 \times 128\) for a single flattened image representation. The dataset therefore consisted of a collection of standardized vectors

\[
\tilde{x}_i = \frac{x_i - \mu_f}{\sigma_f},
\]

where \(\mu_f\) and \(\sigma_f\) denote the per-feature mean and standard deviation estimated across all displayed clips **for the exploratory plots only**. In classification evaluation, they are estimated separately from each training fold. This standardization step ensured that no individual pixel dominated the analysis solely because of larger raw intensity magnitude.

For visualization, the points in the LDA plots were colored according to the spoken word and marked according to subject identity. However, subject identity was not used as the target variable in the LDA optimization itself; it appeared only as a visualization aid to inspect subject structure within the fitted cohort. All displayed word labels influenced that projection, including those belonging to any subject that might later be used for evaluation; this figure is not a held-out test.

## LDA Formulation

LDA was applied as a supervised dimensionality reduction method in which the class label corresponded to **word identity**. Suppose the dataset contains \(K\) word classes. For each class \(k\), let \(\mu_k\) be the mean feature vector of that class, let \(\mu\) be the global mean across all samples, and let \(n_k\) be the number of samples in class \(k\).

The within-class scatter matrix was defined as

\[
S_W = \sum_{k=1}^{K} \sum_{x_i \in C_k} (x_i - \mu_k)(x_i - \mu_k)^T,
\]

and the between-class scatter matrix was defined as

\[
S_B = \sum_{k=1}^{K} n_k (\mu_k - \mu)(\mu_k - \mu)^T.
\]

LDA seeks a projection matrix \(W\) that maximizes the ratio of between-class scatter to within-class scatter:

\[
W^* = \arg \max_W \frac{|W^T S_B W|}{|W^T S_W W|}.
\]

In practical terms, this means that LDA looks for projection directions along which samples from different word classes are as far apart as possible, while samples belonging to the same word class remain as compact as possible. In the present implementation, the number of LDA output dimensions was set to

\[
\min(2, K - 1, d),
\]

subject to the fitted discriminant rank. A single available dimension is plotted with a zero-padded second coordinate, explicitly labelled as unavailable.

After projection into the LDA space, the script generated two types of outputs. First, it saved a two-dimensional LDA scatter plot for each feature representation. Second, it generated one-vs-rest LDA weight maps for each word class. In these maps, a binary LDA model was trained for one target word against all remaining words, and the first discriminant direction was mapped back into image space. The mapping now accounts for standardisation of the PCA scores: `W_pixel = PCA.components_.T @ (LDA.scalings_[:, :1] / scaler.scale_[:, None])`. These are projection weights, with arbitrary sign and correlated predictors; they are not validated physiological importance or causal maps. Centering terms affect the intercept and do not change these weights.

## Why LDA Strengthens Word Separation

The key difference between PCA and LDA is that PCA is unsupervised, whereas LDA is supervised. PCA follows the directions of largest total variance in the dataset. In throat imaging data, these dominant variance directions can easily be influenced by subject-specific appearance, residual illumination structure, ROI geometry, and other low-frequency acquisition-related effects. As a result, PCA may separate subjects well even when the main objective is to distinguish words.

LDA behaves differently because it uses **word labels** during optimization. Variability that is not consistently aligned with word identity, including part of the subject-driven variation, contributes to the within-class scatter term \(S_W\) and is therefore penalized. Differences in observed class means contribute to the between-class scatter term \(S_B\) and are emphasized. Such differences can arise from word-related signal, acquisition confounds, subject/class imbalance or finite-sample noise. LDA has no constraint that removes subject identity or guarantees transfer to an unseen speaker.

This distinction is especially important for the three feature types used here. The Raw Mid Frame and Raw Mean Frame preserve relatively strong static appearance information and may therefore contain stable subject-specific structure. The Mean Absolute Difference Map is more motion-focused and may capture articulation as well as nuisance motion and recording effects. Whether it improves held-out classification must be tested. When LDA is applied with word identity as the supervision signal, all three feature spaces are forced to favor directions that separate word classes, and any advantage of dynamic over static features remains an empirical question.

## Interpretation

Clearer grouping in an LDA plot than in PCA is an exploratory observation of
supervised separation within the fitted dataset. LDA explicitly optimises word
separation, particularly in this high-dimensional, potentially small-sample
setting. The plot alone cannot establish reproducible word-related patterns
across speakers, classifier accuracy or independence from subject/session cues.

The former output labelled “LDA LOO accuracy” was actually a linear SVM on the
original pixel features after globally fitted standardisation. It neither
cross-validated an LDA projection nor held out entire subjects. That output has
been removed; no corrected accuracy is available from this code review.

`evaluation.py` now distinguishes:

- **Sample LOO:** held-out clips, possibly sharing subjects and recording sessions
  with training. This is a sample-level diagnostic, not a cross-subject claim.
- **LOSO / GroupKFold:** subject-disjoint folds with StandardScaler, optional PCA,
  optional supervised LDA and linear SVM fitted on training rows only. Any
  generalisation claim remains conditional on valid upstream preprocessing,
  independent recording units, a suitable cohort and no selection on test scores.
- **Synthetic evaluation:** tests of a specified generator/domain, labelled as
  synthetic, with no automatic implication for real participants.

Exploratory plots remain separate from these evaluations. Choosing features,
ROI settings or model parameters after seeing these scores makes them development
results; use grouped nested validation or a fresh external cohort for the final
estimate. Folder-level reference images and manual ROI selection in the legacy
preprocessing require a separate, label-blind protocol before a whole-pipeline
cross-speaker claim is justified.
