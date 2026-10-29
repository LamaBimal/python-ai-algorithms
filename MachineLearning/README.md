# Additional machine-learning algorithms

These educational examples use scikit-learn implementations and locally
generated or bundled datasets. No network access is needed after installation.

From the repository root:

```sh
python -m pip install numpy scikit-learn
python -m MachineLearning.classification
python -m MachineLearning.clustering
python -m MachineLearning.dimensionality_reduction
```

For plots and a walkthrough, open `AdditionalAlgorithms.ipynb` in Jupyter.
Install `matplotlib notebook` as well for that version.

| Algorithm | What it demonstrates | Dataset and evaluation |
| --- | --- | --- |
| Decision tree | Learn feature thresholds; cap depth to limit overfitting | Iris; held-out accuracy, per-class report, confusion matrix |
| Random forest | Combine many trees trained with randomness | Same Iris split and metrics |
| Support vector machine (SVM) | Separate classes with an RBF kernel | Same Iris split; scaler fitted only on training rows |
| Gaussian Naive Bayes | Model each feature per class with a Gaussian distribution and assume conditional independence | Same Iris split and metrics |
| K-means | Alternate assignment to centers and center updates | Synthetic blobs; silhouette, inertia, adjusted Rand score |
| Principal component analysis (PCA) | Project standardized features onto directions with greatest variance | Wine; training variance retained and held-out reconstruction error |

The four classifiers share a fixed, stratified training/test split. Their
settings are illustrative defaults; this small test set is not sufficient to
declare one algorithm best. Use a separate validation set or cross-validation
inside the training partition before tuning parameters.

K-means is a descriptive experiment fitted to all generated points. True labels
are used only to calculate the adjusted Rand score. Cluster numbers are arbitrary
identifiers, so ordinary classification accuracy would be misleading. Inertia
and silhouette are calculated in the standardized space used for clustering.

PCA learns scaling and components on training rows only. Test rows are projected
without refitting. Its reconstruction error is in standardized feature units,
and its variance ratio describes the training data. Neither operation uses
class labels to fit the model.

All demos use a fixed random seed and return fitted estimators and arrays, so
you can inspect results or call `model.predict` / `model.transform` on new data.

## Learn the mechanics

Start with the [learning path](../docs/learning-path.md). The
[from-scratch notebook](../Learning/FromScratch.ipynb) implements regression,
KNN, and K-means in NumPy and compares them with scikit-learn. The
[visual walkthrough](../Learning/VisualFoundations.ipynb) illustrates decisions
and updates. The [evaluation lesson](../docs/lessons/evaluation.md) and
[notebook](../Learning/Evaluation.ipynb) demonstrate training-only tuning,
baselines, class imbalance, leakage, learning curves, and error analysis.
