"""Regression checks that execute the notebooks' actual code cells.

Run: python -m unittest discover -s tests -v
Model checks need numpy, pandas, and scikit-learn; search checks use stdlib only.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SEARCH = "SearchingTechnique/Case_Study(BFS_&_A_).ipynb"
DIABETES = (
    ("KNN/KNN_Classifier.ipynb", 1, 14),
    ("LogisticRegression/dibetics_prediction_classification.ipynb", 2, 15),
)
SALARY = "LinearRegression/salary_prediction.ipynb"


def run_cell(notebook, index, namespace):
    cells = json.loads((ROOT / notebook).read_text(encoding="utf-8"))["cells"]
    source = "".join(cells[index]["source"])
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, f"{notebook}:cell {index}", "exec"), namespace)


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {}
        run_cell(SEARCH, 5, self.namespace)
        self.search = self.namespace["a_star_shortest_path"]

    def test_direct_edge(self):
        self.assertEqual(self.search({"A": ["B"], "B": []}, "A", "B"), ["A", "B"])

    def test_start_is_goal(self):
        self.assertEqual(self.search({}, "A", "A"), ["A"])

    def test_cycle_and_unreachable_goal(self):
        self.assertIsNone(self.search({"A": ["B"], "B": ["A"]}, "A", "C"))

    def test_weighted_shortest_path(self):
        graph = {"A": {"B": 10, "C": 1}, "C": {"B": 1}, "B": {"D": 1}}
        self.assertEqual(self.search(graph, "A", "D"), ["A", "C", "B", "D"])

    def test_heuristic_does_not_accumulate_as_path_cost(self):
        graph = {"S": {"A": 1, "B": 5}, "A": {"C": 1}, "C": {"G": 1}, "B": {"G": 0}}
        estimates = {"S": 3, "A": 2, "C": 1, "B": 0, "G": 0}
        result = self.search(graph, "S", "G", lambda node, goal: estimates[node])
        self.assertEqual(result, ["S", "A", "C", "G"])

    def test_reopen_node_with_admissible_inconsistent_heuristic(self):
        graph = {"S": {"A": 3, "B": 1}, "B": {"A": 1}, "A": {"G": 3}}
        estimates = {"S": 0, "A": 0, "B": 4, "G": 0}
        result = self.search(graph, "S", "G", lambda node, goal: estimates[node])
        self.assertEqual(result, ["S", "B", "A", "G"])

    def test_tied_priorities_do_not_compare_node_labels(self):
        graph = {"S": {1: 1, "B": 1}, 1: {"G": 1}, "B": {"G": 1}}
        self.assertEqual(self.search(graph, "S", "G"), ["S", 1, "G"])

    def test_coordinate_example_uses_real_connections(self):
        ns = self.namespace
        path = ns["shortest_path"]
        self.assertEqual(path, ["Hattisar", "Location2", "TribhuvanAirport"])
        self.assertEqual(path, self.search(ns["road_graph"], ns["start_node"], ns["goal_node"]))
        for start, end in zip(path, path[1:]):
            self.assertIn(end, ns["connections"][start])


try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
except ImportError:
    if os.environ.get("REQUIRE_ML_TESTS") == "1":
        raise
    HAS_ML = False
else:
    HAS_ML = True


@unittest.skipUnless(HAS_ML, "Install numpy, pandas, and scikit-learn for model checks")
class DataTests(unittest.TestCase):
    def namespace(self):
        return {"np": np, "pd": pd, "train_test_split": train_test_split,
                "select_col": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]}

    def test_dataset_paths_from_root_and_notebook_directory(self):
        for name, index in [(name, index) for name, index, _ in DIABETES] + [(SALARY, 2)]:
            for directory in [ROOT, (ROOT / name).parent]:
                with self.subTest(notebook=name, cwd=directory), working_directory(directory):
                    ns = self.namespace()
                    run_cell(name, index, ns)
                    frame = ns["df"] if name == SALARY else ns["data"]
                    self.assertGreater(len(frame), 0)

    def test_test_rows_cannot_influence_training_medians(self):
        for name, _, split_index in DIABETES:
            with self.subTest(notebook=name):
                ns = self.namespace()
                frame = pd.read_csv(ROOT / "LogisticRegression/diabetes.csv").iloc[:50].copy()
                frame[ns["select_col"]] = 10.0
                _, held_out = train_test_split(frame.index, test_size=0.20, random_state=1)
                frame.loc[held_out, ns["select_col"]] = 10000.0
                frame.loc[held_out[0], ns["select_col"]] = 0.0
                frame["Pregnancies"] = 0
                ns["data"] = frame
                run_cell(name, split_index, ns)
                self.assertTrue((ns["training_medians"] == 10).all())
                self.assertTrue((ns["X_test"].loc[held_out[0], ns["select_col"]] == 10).all())
                self.assertTrue((ns["X_train"]["Pregnancies"] == 0).all())
                self.assertTrue((frame.loc[held_out[0], ns["select_col"]] == 0).all())

    def test_diabetes_models_fit_and_predict(self):
        from sklearn.metrics import accuracy_score
        from sklearn.neighbors import KNeighborsClassifier

        for name, read_index, split_index in DIABETES:
            with self.subTest(notebook=name), working_directory(ROOT):
                ns = self.namespace()
                ns.update(KNeighborsClassifier=KNeighborsClassifier, accuracy_score=accuracy_score)
                run_cell(name, read_index, ns)
                run_cell(name, split_index, ns)
                self.assertFalse(ns["X_train"].isna().any().any())
                self.assertFalse(ns["X_test"].isna().any().any())
                self.assertTrue((ns["X_train"][ns["select_col"]] != 0).all().all())
                run_cell(name, 16 if name.startswith("KNN") else 19, ns)
                if name.startswith("KNN"):
                    run_cell(name, 19, ns)
                    run_cell(name, 20, ns)
                    self.assertAlmostEqual(float(ns["Y_score"].sum()), 1.0)
                    predictions = ns["Y_pred"]
                else:
                    predictions = ns["prediction"]
                self.assertEqual(len(predictions), len(ns["Y_test"]))

    def test_salary_fit_and_predict(self):
        with working_directory(ROOT / "LinearRegression"):
            ns = self.namespace()
            for index in [2, 6, 15, 21, 22, 23, 24, 26, 30, 31, 35, 37, 38]:
                run_cell(SALARY, index, ns)
            self.assertEqual(ns["predicted_salary"].shape, (1,))
            self.assertTrue(np.isfinite(ns["predicted_salary"]).all())


if __name__ == "__main__":
    unittest.main()
