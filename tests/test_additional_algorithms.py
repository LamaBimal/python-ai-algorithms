"""Behavioral tests for the new search and machine-learning demonstrations."""

import os
import unittest

from SearchingTechnique.greedy_best_first_search import greedy_best_first_search
from SearchingTechnique.uniform_cost_search import uniform_cost_search


class UniformCostTests(unittest.TestCase):
    def test_cheaper_route_can_have_more_edges(self):
        graph = {"A": {"G": 10, "B": 1}, "B": {"C": 1}, "C": {"G": 1}}
        self.assertEqual(uniform_cost_search(graph, "A", "G"), (["A", "B", "C", "G"], 3))

    def test_zero_cost_cycles_and_unreachable_nodes(self):
        graph = {"A": {"B": 0}, "B": {"A": 0, "G": 2}}
        self.assertEqual(uniform_cost_search(graph, "A", "G"), (["A", "B", "G"], 2))
        self.assertIsNone(uniform_cost_search(graph, "A", "Z"))

    def test_equal_priorities_support_different_node_types(self):
        graph = {"A": {1: 1, "B": 1}, 1: {"G": 1}, "B": {"G": 2}}
        self.assertEqual(uniform_cost_search(graph, "A", "G"), (["A", 1, "G"], 2))

    def test_start_equals_goal(self):
        self.assertEqual(uniform_cost_search({}, "A", "A"), (["A"], 0))

    def test_reject_invalid_costs_even_on_edges_after_the_goal(self):
        for invalid in [-1, float("inf"), float("nan")]:
            with self.subTest(cost=invalid), self.assertRaises(ValueError):
                uniform_cost_search({"A": {"G": 1}, "G": {"B": invalid}}, "A", "G")


class GreedySearchTests(unittest.TestCase):
    def test_heuristic_can_choose_a_longer_path(self):
        graph = {"A": ["B", "C"], "B": ["D"], "D": ["G"], "C": ["G"]}
        estimates = {"A": 3, "B": 0, "C": 2, "D": 0, "G": 0}
        path = greedy_best_first_search(graph, "A", "G", lambda node, goal: estimates[node])
        self.assertEqual(path, ["A", "B", "D", "G"])

    def test_cycles_and_unreachable_goal(self):
        graph = {"A": ["B"], "B": ["A"]}
        self.assertIsNone(greedy_best_first_search(graph, "A", "G", lambda node, goal: 0))

    def test_tied_priorities_and_implicit_leaf(self):
        graph = {"A": [1, "B"], 1: ["G"]}
        self.assertEqual(
            greedy_best_first_search(graph, "A", "G", lambda node, goal: 0), ["A", 1, "G"]
        )

    def test_start_equals_goal(self):
        self.assertEqual(greedy_best_first_search({}, "A", "A", lambda node, goal: 0), ["A"])


try:
    import numpy as np
    from MachineLearning.classification import run_demo as classify
    from MachineLearning.clustering import run_demo as cluster
    from MachineLearning.dimensionality_reduction import run_demo as reduce_dimensions
except ImportError:
    if os.environ.get("REQUIRE_ML_TESTS") == "1":
        raise
    HAS_ML = False
else:
    HAS_ML = True


@unittest.skipUnless(HAS_ML, "Install numpy and scikit-learn for ML checks")
class MachineLearningTests(unittest.TestCase):
    def test_all_classifiers_predict_on_the_same_held_out_samples(self):
        demo = classify()
        self.assertEqual(len(demo["results"]), 4)
        self.assertEqual(len(demo["X_train"]) + len(demo["X_test"]), 150)
        self.assertEqual(set(demo["y_train"]), {0, 1, 2})
        self.assertEqual(set(demo["y_test"]), {0, 1, 2})
        for name, result in demo["results"].items():
            with self.subTest(model=name):
                self.assertEqual(len(result["predictions"]), len(demo["y_test"]))
                self.assertEqual(result["confusion_matrix"].shape, (3, 3))
                self.assertEqual(result["confusion_matrix"].sum(), len(demo["y_test"]))
                # A loose floor catches broken feature/label handling on simple Iris data.
                self.assertGreater(result["accuracy"], 0.75)

    def test_svm_scaling_is_learned_only_from_training_rows(self):
        demo = classify()
        scaler = demo["results"]["SVM"]["model"].named_steps["standardscaler"]
        self.assertEqual(scaler.n_samples_seen_, len(demo["X_train"]))
        np.testing.assert_allclose(scaler.mean_, demo["X_train"].mean(axis=0))

    def test_clustering_recovers_known_groups_without_label_order_assumptions(self):
        demo = cluster()
        self.assertEqual(len(set(demo["cluster_labels"])), 3)
        self.assertGreater(demo["adjusted_rand"], 0.95)
        self.assertGreater(demo["silhouette"], 0.5)
        np.testing.assert_array_equal(
            demo["model"].predict(demo["X"]), demo["cluster_labels"]
        )

    def test_pca_reduces_unseen_data_and_beats_mean_only_reconstruction(self):
        demo = reduce_dimensions()
        self.assertEqual(demo["train_components"].shape, (len(demo["X_train"]), 2))
        self.assertEqual(demo["test_components"].shape, (len(demo["X_test"]), 2))
        scaler = demo["model"].named_steps["standardscaler"]
        self.assertEqual(scaler.n_samples_seen_, len(demo["X_train"]))
        np.testing.assert_allclose(scaler.mean_, demo["X_train"].mean(axis=0))
        mean_only_mse = np.mean(scaler.transform(demo["X_test"]) ** 2)
        self.assertGreaterEqual(demo["reconstruction_mse"], 0)
        self.assertLess(demo["reconstruction_mse"], mean_only_mse)
        self.assertTrue(0 < demo["explained_variance_ratio"].sum() <= 1)

    def test_fixed_seeds_reproduce_results(self):
        first, second = classify(), classify()
        for name in first["results"]:
            np.testing.assert_array_equal(
                first["results"][name]["predictions"], second["results"][name]["predictions"]
            )
        np.testing.assert_array_equal(cluster()["cluster_labels"], cluster()["cluster_labels"])
        np.testing.assert_allclose(
            reduce_dimensions()["test_components"], reduce_dimensions()["test_components"]
        )


if __name__ == "__main__":
    unittest.main()
