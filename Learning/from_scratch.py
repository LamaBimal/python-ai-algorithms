"""NumPy teaching models; see docs/lessons/from-scratch.md for derivations.

These deliberately small estimators support finite, dense numeric features.
They are teaching tools, rather than replacements for production estimators.
"""

import numpy as np


def _features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite 2D numeric array")
    return X


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _predict_features(model, X):
    if not hasattr(model, "n_features_in_"):
        raise ValueError("Fit the model before prediction")
    X = _features(X)
    if X.shape[1] != model.n_features_in_:
        raise ValueError("Prediction features must match training features")
    return X


class LinearRegressionGD:
    """Full-batch gradient descent on training-standardized features.

    Objective: mean squared error. The intercept is not regularized.
    coef_ and intercept_ use original feature units; parameter_history_ uses
    standardized units with the intercept first. Zero-variance features use
    a scale of one. A fixed iteration budget makes experiments reproducible.
    """

    def __init__(self, learning_rate=0.05, n_iter=2000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        X = _features(X)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("y must contain one finite target per row")
        _positive_integer(self.n_iter, "n_iter")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        design = np.column_stack([np.ones(len(X)), (X - self.mean_) / self.scale_])
        weights = np.zeros(design.shape[1])
        self.loss_history_ = []
        self.parameter_history_ = [weights.copy()]
        for _ in range(self.n_iter):
            residual = design @ weights - y
            self.loss_history_.append(float(np.mean(residual ** 2)))
            weights -= self.learning_rate * (2 / len(X)) * design.T @ residual
            if not np.isfinite(weights).all():
                raise ValueError("Gradient descent diverged; reduce the learning rate")
            self.parameter_history_.append(weights.copy())
        self.loss_history_.append(float(np.mean((design @ weights - y) ** 2)))
        self.coef_ = weights[1:] / self.scale_
        self.intercept_ = float(weights[0] - self.mean_ @ self.coef_)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return _predict_features(self, X) @ self.coef_ + self.intercept_


class KNNClassifier:
    """Uniform Euclidean voting; scale features before fitting if needed.

    Distance ties use training-row order. Vote ties choose the first class
    in np.unique's sorted ordering. Labels must be consistently sortable.
    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X = _features(X)
        y = np.asarray(y)
        _positive_integer(self.n_neighbors, "n_neighbors")
        if y.ndim != 1 or len(y) != len(X) or self.n_neighbors > len(X):
            raise ValueError("Provide one label per row and k <= training rows")
        self.classes_, self._encoded = np.unique(y, return_inverse=True)
        self._X = X.copy()
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = _predict_features(self, X)
        predictions = []
        for row in X:
            squared = np.sum((self._X - row) ** 2, axis=1)
            nearest = np.argsort(squared, kind="stable")[:self.n_neighbors]
            votes = np.bincount(self._encoded[nearest], minlength=len(self.classes_))
            predictions.append(self.classes_[votes.argmax()])
        return np.asarray(predictions)


class KMeans:
    """Lloyd's algorithm with seeded random starts and best-inertia selection.

    Empty clusters retain their previous center. Duplicate input points can
    therefore yield fewer occupied clusters than requested. Cluster numbers
    are arbitrary. center_history_ records the selected run for visualization.
    """

    def __init__(self, n_clusters=3, n_init=5, max_iter=100, tol=1e-6, random_state=42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _assign(X, centers):
        squared = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = squared.argmin(axis=1)
        return labels, float(squared[np.arange(len(X)), labels].sum())

    def fit(self, X):
        X = _features(X)
        for name in ("n_clusters", "n_init", "max_iter"):
            _positive_integer(getattr(self, name), name)
        if self.n_clusters > len(X) or not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("Require n_clusters <= rows and finite nonnegative tol")
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            centers = X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
            history = [centers.copy()]
            for iteration in range(self.max_iter):
                labels, _ = self._assign(X, centers)
                updated = centers.copy()
                for cluster in range(self.n_clusters):
                    members = X[labels == cluster]
                    if len(members):
                        updated[cluster] = members.mean(axis=0)
                shift = np.linalg.norm(updated - centers)
                centers = updated
                history.append(centers.copy())
                if shift <= self.tol:
                    break
            labels, inertia = self._assign(X, centers)
            if best is None or inertia < best[0]:
                best = (inertia, centers.copy(), labels.copy(), history, iteration + 1)
        (self.inertia_, self.cluster_centers_, self.labels_,
         self.center_history_, self.n_iter_) = best
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self._assign(_predict_features(self, X), self.cluster_centers_)[0]
