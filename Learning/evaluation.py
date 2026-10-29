"""Train-only selection and held-out evaluation on imbalanced synthetic data."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,
                                     learning_curve, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, predictions),
            "balanced_accuracy": balanced_accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, zero_division=0),
            "recall": recall_score(y, predictions, zero_division=0),
            "f1": f1_score(y, predictions, zero_division=0),
            "average_precision": average_precision_score(y, probabilities),
            "roc_auc": roc_auc_score(y, probabilities),
            "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1])}


def run_demo():
    X, y = make_classification(n_samples=600, n_features=12, n_informative=6,
                               weights=[0.9, 0.1], class_sep=1.2, random_state=42)
    train_indices, test_indices = train_test_split(
        np.arange(len(X)), test_size=0.25, stratify=y, random_state=42)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                          {"logisticregression__C": [0.01, 0.1, 1, 10]},
                          scoring="average_precision", cv=cv, n_jobs=1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    # Learning curves and depth comparisons use ONLY the training partition.
    sizes, train_scores, validation_scores = learning_curve(
        model, X_train, y_train, cv=cv, train_sizes=[0.3, 0.6, 1.0],
        scoring="average_precision", shuffle=True, random_state=42, n_jobs=1)
    depth_results = {}
    for depth in [1, 4, None]:
        depth_results[str(depth)] = cross_validate(
            DecisionTreeClassifier(max_depth=depth, random_state=42), X_train, y_train,
            cv=cv, scoring="balanced_accuracy", return_train_score=True)
    # This is a final report after model selection, not input to tuning.
    return {"model": model, "search": search, "baseline": baseline,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "train_indices": train_indices, "test_indices": test_indices,
            "model_metrics": metrics(model, X_test, y_test),
            "baseline_metrics": metrics(baseline, X_test, y_test),
            "learning_sizes": sizes, "train_scores": train_scores,
            "validation_scores": validation_scores, "depth_results": depth_results,
            "test_predictions": model.predict(X_test),
            "test_probabilities": model.predict_proba(X_test)[:, 1]}


if __name__ == "__main__":
    result = run_demo()
    print("Chosen C from training cross-validation:", result["search"].best_params_)
    for name in ("baseline_metrics", "model_metrics"):
        print(name, result[name])
