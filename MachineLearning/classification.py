"""Compare four classifiers on held-out Iris flowers.

Run from the repository root: python -m MachineLearning.classification
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_models(random_state=42):
    """Create fresh estimators; the SVM learns scaling only during fit."""
    return {
        "Decision tree": DecisionTreeClassifier(max_depth=3, random_state=random_state),
        "Random forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def run_demo(random_state=42):
    """Return fitted classifiers and evaluation on one shared, stratified split.

    The test partition is reserved for evaluation; it is never used to fit
    preprocessing, tune parameters, or select a winning model.
    """
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25,
        random_state=random_state, stratify=dataset.target,
    )
    results = {}
    for name, model in build_models(random_state).items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "model": model,
            "predictions": predictions,
            "accuracy": accuracy_score(y_test, predictions),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1, 2]),
            "report": classification_report(
                y_test, predictions, labels=[0, 1, 2],
                target_names=dataset.target_names, zero_division=0,
            ),
        }
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "target_names": dataset.target_names, "results": results,
    }


def main():
    experiment = run_demo()
    print("Iris classification: 112 training samples, 38 held-out samples")
    for name, result in experiment["results"].items():
        print(f"\n{name}: accuracy={result['accuracy']:.3f}")
        print(result["report"])
        print("Confusion matrix (rows=actual, columns=predicted):")
        print(result["confusion_matrix"])


if __name__ == "__main__":
    main()
