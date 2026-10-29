"""K-means on synthetic clusters, without using labels during fitting.

Run from the repository root: python -m MachineLearning.clustering
"""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_demo(random_state=42):
    """Return a descriptive clustering demo, rather than a prediction benchmark.

    Ground-truth labels are used only for adjusted Rand evaluation. Cluster
    numbers have no class meaning and can be permuted between runs.
    """
    X, true_labels = make_blobs(
        n_samples=300, centers=[(-6, -4), (0, 6), (6, -3)],
        cluster_std=0.8, random_state=random_state,
    )
    model = make_pipeline(
        StandardScaler(), KMeans(n_clusters=3, n_init=10, random_state=random_state)
    )
    cluster_labels = model.fit_predict(X)
    scaled_X = model.named_steps["standardscaler"].transform(X)
    return {
        "model": model, "X": X, "true_labels": true_labels,
        "cluster_labels": cluster_labels,
        "silhouette": silhouette_score(scaled_X, cluster_labels),
        "adjusted_rand": adjusted_rand_score(true_labels, cluster_labels),
        "inertia": model.named_steps["kmeans"].inertia_,
    }


def main():
    result = run_demo()
    print("K-means: 300 synthetic points, 3 clusters")
    print(f"Silhouette score: {result['silhouette']:.3f}")
    print(f"Adjusted Rand score: {result['adjusted_rand']:.3f}")
    print(f"Inertia in standardized feature space: {result['inertia']:.3f}")


if __name__ == "__main__":
    main()
