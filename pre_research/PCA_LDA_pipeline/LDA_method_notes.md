# LDA Method Notes

## Input Data and Preprocessing

The input to the LDA analysis was not the original full-frame video, but a throat ROI video sequence that had already been cropped during the earlier preprocessing stage. For the downstream analysis implemented in `05_analyze_processed_split2.py`, each processed clip was read frame by frame, converted to grayscale, resized to a fixed spatial resolution of 128 x 128 pixels, and temporally resampled to 24 frames per video. This temporal normalization was performed over the full clip duration rather than on a per-second basis, so that every sample contributed the same number of frames to the analysis regardless of its original length.

After temporal normalization, frame differences were computed with a lag of 1 frame in order to characterize motion. Three image-derived representations were then constructed for each clip. The first was the **Raw Mid Frame**, defined as the middle frame of the resampled sequence. The second was the **Raw Mean Frame**, defined as the pixel-wise average across all resampled frames. The third was the **Mean Absolute Difference Map**, defined as the pixel-wise average of the absolute frame-to-frame differences and therefore intended to summarize the dominant motion pattern within the clip. In parallel, summary motion statistics such as motion mean, motion range, and absolute-difference descriptors were also computed and written to metadata, but these scalar descriptors were not used as input to PCA or LDA.

Each of the three image representations was flattened into a one-dimensional pixel vector, so that each clip was represented as a feature vector in a high-dimensional image space. Before LDA, the feature matrix was standardized using `StandardScaler`, which centered each pixel feature to zero mean and scaled it to unit variance across samples.

## Feature Construction

Let each processed clip be represented by a feature vector \(x_i \in \mathbb{R}^d\), where \(d = 128 \times 128\) for a single flattened image representation. The dataset therefore consisted of a collection of standardized vectors

\[
\tilde{x}_i = \frac{x_i - \mu_f}{\sigma_f},
\]

where \(\mu_f\) and \(\sigma_f\) denote the per-feature mean and standard deviation estimated across all clips. This standardization step ensured that no individual pixel dominated the analysis solely because of larger raw intensity magnitude.

For visualization, the points in the LDA plots were colored according to the spoken word and marked according to subject identity. However, subject identity was not used as the target variable in the LDA optimization itself; it appeared only as a visualization aid to inspect whether the learned discriminant space generalized across speakers.

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
\min(2, K - 1),
\]

which allowed the data to be visualized in a two-dimensional discriminant space whenever the number of word classes was at least three.

After projection into the LDA space, the script generated two types of outputs. First, it saved a two-dimensional LDA scatter plot for each feature representation. Second, it generated one-vs-rest LDA weight maps for each word class. In these maps, a binary LDA model was trained for one target word against all remaining words, and the learned discriminant weights were projected back into image space to highlight the throat regions contributing most strongly to the discrimination.

## Why LDA Strengthens Word Separation

The key difference between PCA and LDA is that PCA is unsupervised, whereas LDA is supervised. PCA follows the directions of largest total variance in the dataset. In throat imaging data, these dominant variance directions can easily be influenced by subject-specific appearance, residual illumination structure, ROI geometry, and other low-frequency acquisition-related effects. As a result, PCA may separate subjects well even when the main objective is to distinguish words.

LDA behaves differently because it uses **word labels** during optimization. Variability that is not consistently aligned with word identity, including part of the subject-driven variation, contributes to the within-class scatter term \(S_W\) and is therefore penalized. By contrast, patterns that recur across different speakers for the same word contribute to the between-class scatter term \(S_B\) and are therefore emphasized. In this sense, LDA does not eliminate subject information entirely, but it reweights the feature space so that word-discriminative structure becomes more prominent than subject-related variability.

This distinction is especially important for the three feature types used here. The Raw Mid Frame and Raw Mean Frame preserve relatively strong static appearance information and may therefore contain stable subject-specific structure. The Mean Absolute Difference Map is more motion-focused and is therefore more likely to reflect articulatory differences associated with the spoken word. When LDA is applied with word identity as the supervision signal, all three feature spaces are forced to favor directions that separate word classes, but dynamic features are often more naturally aligned with this objective.

## Interpretation

For this reason, clearer grouping in LDA than in PCA should be interpreted as evidence that the processed throat-image representations contain reproducible word-related patterns across speakers. PCA may still reveal clustering, but that clustering can be dominated by total variance sources that are not specific to the linguistic task. LDA is therefore better suited for demonstrating whether the processed dataset supports word-level discrimination, because its optimization directly favors between-word separation while comparatively suppressing variation that is unrelated to the class labels.
