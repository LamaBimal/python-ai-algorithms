# Clustering and PCA

[Learning path](../learning-path.md) | [Project README](../../README.md)

## Learning objectives and intuition

Clustering describes similarity without target labels. PCA finds directions
of greatest variance and projects features onto fewer axes. Neither clusters
nor principal components are automatically meaningful real-world categories.

## Mathematics and assumptions

After training-only centering/scaling, covariance describes how features
vary together. PCA axes are orthogonal directions maximizing retained variance;
equivalently, truncated SVD provides a low-rank approximation. PCA scores are
projections onto those axes. Reconstruction maps those scores back to feature
space; discarded components create reconstruction error.

## Worked example

For points lying close to y=x, the first principal axis points diagonally,
retaining variation along that line. The perpendicular axis mostly captures
small deviations. Reconstructing from one component projects points onto the
diagonal; the perpendicular residual is lost information.

## Implementation

Run `MachineLearning.clustering`, `MachineLearning.dimensionality_reduction`,
and the PCA projection in `Learning/VisualFoundations.ipynb`. Fit scaling and
PCA on training observations, then transform held-out observations without
refitting. The synthetic PCA picture is a descriptive geometric demonstration.

## Evaluation

Use silhouette and inertia to inspect clustering, with known synthetic
labels used only for external adjusted Rand evaluation. Measure held-out PCA
reconstruction error and report the feature units or scaling used. Explained
variance is a training-data summary, not predictive accuracy.

## Limitations and failure cases

PCA maximizes variance, which may be driven by noise or irrelevant units;
it does not maximize class separation. Cluster counts require judgment.
K-means struggles with elongated, unequal-density, or nonconvex groups.

## Exercises and checkpoint

1. Scale one feature by 100 and compare PCA with and without standardization.
2. Try two, three, and four clusters and inspect the actual scatter plots.
3. Explain why swapping cluster numbers cannot change clustering quality.

Checkpoint: distinguish feature compression, grouping, and supervised prediction.
