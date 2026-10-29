"""Behavior checks and complete code-cell runs for the foundation lessons."""

import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from Learning.search_walkthrough import trace_search

ROOT = Path(__file__).resolve().parents[1]


class SearchWalkthroughTests(unittest.TestCase):
    def test_edge_count_cost_and_heuristic_choose_different_routes(self):
        graph = {"S": {"A": 1, "B": 4}, "A": {"C": 1}, "C": {"G": 1}, "B": {"G": 1}}
        h = {"S": 3, "A": 2, "B": 0, "C": 1, "G": 0}
        for strategy, expected in [("bfs", 5), ("dfs", 3), ("ucs", 3), ("greedy", 5), ("astar", 3)]:
            with self.subTest(strategy=strategy):
                result = trace_search(graph, "S", "G", strategy, lambda n, g: h[n])
                self.assertEqual(result["cost"], expected)
                path = result["path"]
                self.assertEqual(sum(graph[a][b] for a, b in zip(path, path[1:])), expected)
                self.assertEqual(result["snapshots"][-1]["current"], "G")

    def test_cycles_missing_nodes_start_and_unreachable(self):
        for strategy in ["bfs", "dfs", "ucs", "greedy", "astar"]:
            self.assertEqual(trace_search({}, "S", "S", strategy)["path"], ["S"])
            self.assertIsNone(trace_search({"S": {"A": 0}, "A": {"S": 0, "B": 1}},
                                           "S", "G", strategy)["path"])

    def test_astar_reopens_improved_states(self):
        graph = {"S": {"A": 3, "B": 1}, "B": {"A": 1}, "A": {"G": 3}}
        h = {"S": 0, "A": 0, "B": 4, "G": 0}
        result = trace_search(graph, "S", "G", "astar", lambda n, g: h[n])
        self.assertEqual(result["path"], ["S", "B", "A", "G"])
        self.assertEqual([s["current"] for s in result["snapshots"]].count("A"), 2)

    def test_dfs_follows_depth_even_when_neighbor_is_already_queued(self):
        graph = {"S": {"A": 1, "B": 1}, "A": {"B": 1}, "B": {"G": 1}}
        self.assertEqual(trace_search(graph, "S", "G", "dfs")["path"], ["S", "A", "B", "G"])

    def test_rejects_invalid_costs_and_strategy(self):
        for cost in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                trace_search({"S": {"G": cost}}, "S", "G")
        with self.assertRaises(ValueError):
            trace_search({}, "S", "G", "unknown")


try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans as ReferenceKMeans
    from sklearn.datasets import make_blobs
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import adjusted_rand_score
    from sklearn.neighbors import KNeighborsClassifier
    from Learning.from_scratch import LinearRegressionGD, KNNClassifier, KMeans
    from Learning.evaluation import run_demo
except ImportError:
    if os.environ.get("REQUIRE_ML_TESTS") == "1":
        raise
    HAS_ML = False
else:
    HAS_ML = True


@unittest.skipUnless(HAS_ML, "Install requirements-dev.txt for learning checks")
class FoundationModelTests(unittest.TestCase):
    def test_regression_matches_least_squares_on_unseen_multifeature_rows(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 3)) * [1, 10, 100]
        y = 4 + X @ [2, -3, 0.5]
        model = LinearRegressionGD().fit(X, y)
        reference = LinearRegression().fit(X, y)
        unseen = rng.normal(size=(10, 3)) * [1, 10, 100]
        np.testing.assert_allclose(model.predict(unseen), reference.predict(unseen), atol=1e-6)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0] * 1e-8)

    def test_regression_constant_features_and_no_prediction_refitting(self):
        model = LinearRegressionGD().fit([[1, 2], [1, 4], [1, 6]], [5, 9, 13])
        before = model.mean_.copy()
        np.testing.assert_allclose(model.predict([[1, 8]]), [17], atol=1e-6)
        np.testing.assert_array_equal(model.mean_, before)

    def test_knn_matches_reference_with_string_labels(self):
        X = [[0], [1], [3], [8], [9], [11]]
        y = ["a", "a", "a", "b", "b", "b"]
        queries = [[0.2], [4.2], [9.1]]
        model = KNNClassifier(3).fit(X, y)
        reference = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        np.testing.assert_array_equal(model.predict(queries), reference.predict(queries))

    def test_knn_defines_vote_and_distance_ties_and_copies_training_data(self):
        X = np.array([[0.0], [2.0]])
        model = KNNClassifier(2).fit(X, ["b", "a"])
        self.assertEqual(model.predict([[1]])[0], "a")
        nearest = KNNClassifier(1).fit(X, ["b", "a"])
        X[0] = 100
        self.assertEqual(nearest.predict([[1]])[0], "b")

    def test_kmeans_matches_clear_partition_independent_of_label_numbers(self):
        X, known = make_blobs(n_samples=120, centers=[[-5, -4], [0, 5], [5, -3]],
                              cluster_std=0.3, random_state=42)
        model = KMeans(3, n_init=8).fit(X)
        reference = ReferenceKMeans(3, n_init=8, random_state=42).fit(X)
        self.assertEqual(adjusted_rand_score(model.labels_, known), 1)
        self.assertEqual(adjusted_rand_score(model.labels_, reference.labels_), 1)
        self.assertAlmostEqual(model.inertia_, reference.inertia_)
        np.testing.assert_array_equal(model.predict(X), model.labels_)
        repeated = KMeans(3, n_init=8).fit(X)
        np.testing.assert_allclose(model.cluster_centers_, repeated.cluster_centers_)

    def test_kmeans_empty_clusters_remain_finite_and_inertia_does_not_increase(self):
        model = KMeans(3, n_init=1).fit(np.ones((6, 2)))
        self.assertEqual(model.inertia_, 0)
        self.assertTrue(np.isfinite(model.cluster_centers_).all())
        X, _ = make_blobs(n_samples=60, random_state=7)
        model = KMeans(3, n_init=1).fit(X)
        inertia = [np.min(((X[:, None] - c) ** 2).sum(axis=2), axis=1).sum()
                   for c in model.center_history_]
        self.assertTrue(np.all(np.diff(inertia) <= 1e-8))

    def test_models_reject_invalid_inputs_and_unfitted_prediction(self):
        for model in [LinearRegressionGD(), KNNClassifier(), KMeans()]:
            with self.assertRaises(ValueError):
                model.predict([[1]])
        for model, args in [(LinearRegressionGD(), ()), (KNNClassifier(1), ())]:
            with self.assertRaises(ValueError):
                model.fit([[float("nan")]], [1])
            model.fit([[1], [2]], [1, 2])
            with self.assertRaises(ValueError):
                model.predict([[1, 2]])
        with self.assertRaises(ValueError):
            KNNClassifier(3).fit([[1], [2]], [0, 1])
        with self.assertRaises(ValueError):
            KMeans(3).fit([[1], [2]])
        with self.assertRaises(ValueError):
            LinearRegressionGD(n_iter=0).fit([[1]], [1])

    def test_evaluation_reserves_test_rows_and_refits_scaler_only_on_training(self):
        demo = run_demo()
        self.assertFalse(set(demo["train_indices"]) & set(demo["test_indices"]))
        self.assertEqual(len(demo["X_train"]) + len(demo["X_test"]), 600)
        scaler = demo["model"].named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, demo["X_train"].mean(axis=0))
        self.assertEqual(scaler.n_samples_seen_, len(demo["X_train"]))
        self.assertEqual(demo["baseline_metrics"]["recall"], 0)
        self.assertEqual(demo["baseline_metrics"]["balanced_accuracy"], 0.5)
        self.assertGreater(demo["model_metrics"]["average_precision"],
                           demo["baseline_metrics"]["average_precision"])
        self.assertEqual(demo["model_metrics"]["confusion_matrix"].sum(), len(demo["y_test"]))
        self.assertTrue(np.isfinite(demo["validation_scores"]).all())

    def test_all_learning_notebooks_execute_in_order(self):
        previous = Path.cwd()
        try:
            os.chdir(ROOT / "Learning")
            for name in ["FromScratch.ipynb", "VisualFoundations.ipynb", "Evaluation.ipynb"]:
                with self.subTest(notebook=name):
                    namespace = {"__name__": "__main__"}
                    notebook = json.loads(Path(name).read_text(encoding="utf-8"))
                    with contextlib.redirect_stdout(io.StringIO()), patch("matplotlib.pyplot.show"):
                        for index, cell in enumerate(notebook["cells"]):
                            if cell["cell_type"] == "code":
                                source = "".join(cell["source"])
                                self.assertTrue(source.strip())
                                exec(compile(source, f"{name}:{index}", "exec"), namespace)
                    if name == "FromScratch.ipynb":
                        self.assertLess(namespace["regression_comparison"]["prediction_gap"], 1e-5)
                    elif name == "VisualFoundations.ipynb":
                        self.assertEqual(namespace["search_results"]["ucs"]["cost"], 3)
                    else:
                        self.assertEqual(len(namespace["experiment"]["test_predictions"]), 150)
                    plt.close("all")
        finally:
            os.chdir(previous)
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
