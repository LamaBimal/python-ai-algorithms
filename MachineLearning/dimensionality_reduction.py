"""Reduce Wine's 13 numeric features to two principal components.

Run from the repository root: python -m MachineLearning.dimensionality_reduction
"""

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_demo(random_state=42):
    """Fit scaling and PCA on training rows and transform unseen rows.

    Reconstruction MSE is reported in standardized feature space, where each
    original feature has equal scale. PCA uses no class labels during fitting.
    """
    dataset = load_wine()
    X_train, X_test = train_test_split(
        dataset.data, test_size=0.25, random_state=random_state,
        stratify=dataset.target,
    )
    model = make_pipeline(StandardScaler(), PCA(n_components=2, svd_solver="full"))
    train_components = model.fit_transform(X_train)
    test_components = model.transform(X_test)
    scaler = model.named_steps["standardscaler"]
    pca = model.named_steps["pca"]
    scaled_test = scaler.transform(X_test)
    reconstructed_test = pca.inverse_transform(test_components)
    return {
        "model": model, "X_train": X_train, "X_test": X_test,
        "train_components": train_components, "test_components": test_components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "reconstruction_mse": mean_squared_error(scaled_test, reconstructed_test),
    }


def main():
    result = run_demo()
    print("Wine PCA: 13 features reduced to 2 components")
    print(f"Training variance retained: {result['explained_variance_ratio'].sum():.1%}")
    print(f"Held-out reconstruction MSE (standardized units): {result['reconstruction_mse']:.3f}")


if __name__ == "__main__":
    main()
