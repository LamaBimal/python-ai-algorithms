"""Execute completed walkthroughs and check search and text-model behavior."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from SearchingTechnique.breadth_first_search import bfs_shortest_path
from SearchingTechnique.deapth_first_search import dfs

ROOT = Path(__file__).resolve().parents[1]


class BasicSearchTests(unittest.TestCase):
    def test_cycles_unreachable_nodes_and_implicit_leaves(self):
        for search in (bfs_shortest_path, dfs):
            with self.subTest(search=search.__name__):
                self.assertEqual(search({}, "A", "A"), ["A"])
                self.assertIsNone(search({}, "A", "G"))
                self.assertIsNone(search({"A": ["B"], "B": ["A", "C"]}, "A", "G"))
                self.assertEqual(search({"A": ["B"]}, "A", "B"), ["A", "B"])

    def test_bfs_prefers_fewer_edges_while_dfs_follows_depth(self):
        graph = {"S": ["A", "B"], "A": ["C"], "C": ["G"], "B": ["G"]}
        self.assertEqual(bfs_shortest_path(graph, "S", "G"), ["S", "B", "G"])
        self.assertEqual(dfs(graph, "S", "G"), ["S", "A", "C", "G"])


try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from NLP.news_classification import clean_text, load_news, split_news, build_model
except ImportError:
    if os.environ.get("REQUIRE_ML_TESTS") == "1":
        raise
    HAS_ML = False
else:
    HAS_ML = True


def run_notebook(name, directory):
    notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    previous = Path.cwd()
    try:
        os.chdir(directory)
        with contextlib.redirect_stdout(io.StringIO()), patch("matplotlib.pyplot.show"):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    if not source.strip():
                        raise AssertionError(f"Unfinished cell {index} in {name}")
                    exec(compile(source, f"{name}:cell {index}", "exec"), namespace)
    finally:
        os.chdir(previous)
        plt.close("all")
    return namespace


@unittest.skipUnless(HAS_ML, "Install requirements-dev.txt for text-model checks")
class TextModelTests(unittest.TestCase):
    def test_cleaning_preserves_nepali_and_uses_longest_suffix(self):
        self.assertEqual(clean_text("नेपाल र खेल १२३ @ hello"), "नेपाल खेल")
        self.assertEqual(clean_text("नेपालमा", ("मा",)), "नेपाल")
        self.assertEqual(clean_text("नेपाल"), "नेपाल")

    def test_news_split_has_no_duplicate_article_leakage(self):
        frame = load_news(max_per_class=12)
        X_train, X_test, y_train, y_test = split_news(frame)
        self.assertEqual(len(frame["Category"].unique()), 4)
        self.assertTrue(set(X_train).isdisjoint(X_test))
        self.assertEqual(set(y_train), set(y_test))
        self.assertTrue(frame["Language"].eq("ne").all())

    def test_test_only_words_cannot_change_fitted_vocabulary(self):
        for neural in (False, True):
            model = build_model(neural=neural)
            model.fit(["खेल जित", "खेल टोली", "अर्थ बजार", "अर्थ व्यापार"],
                      ["sports", "sports", "business", "business"])
            vocabulary = dict(model.named_steps["tfidfvectorizer"].vocabulary_)
            probabilities = model.predict_proba(["अनदेखाशब्द खेल"])
            self.assertNotIn("अनदेखाशब्द", vocabulary)
            self.assertEqual(vocabulary, model.named_steps["tfidfvectorizer"].vocabulary_)
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    def test_complete_news_notebooks_from_root_and_notebook_directory(self):
        for name, directory in [
            ("NLP/NLPWorkshop.ipynb", ROOT),
            ("NLP/NepaliNewsClassification.ipynb", ROOT / "NLP"),
        ]:
            with self.subTest(notebook=name):
                ns = run_notebook(name, directory)
                for result in (ns["logistic_results"], ns["neural_results"]):
                    self.assertEqual(len(result["predictions"]), len(ns["y_test"]))
                    self.assertEqual(result["confusion_matrix"].sum(), len(ns["y_test"]))
                self.assertIn(ns["predicted_topic"], ns["model"].classes_)

    def test_spam_notebook_with_included_data(self):
        for directory in (ROOT, ROOT / "LogisticRegression"):
            with self.subTest(cwd=directory), patch.dict(os.environ):
                os.environ.pop("MAIL_DATA_PATH", None)
                ns = run_notebook("LogisticRegression/SpamEmailDetection.ipynb", directory)
                self.assertEqual(len(ns["data"]), 40)
                self.assertTrue(set(ns["X_train"]).isdisjoint(ns["X_test"]))
                self.assertEqual(len(ns["predictions"]), len(ns["y_test"]))
                self.assertIn(ns["prediction"][0], ("ham", "spam"))

    def test_spam_rejects_conflicting_labels_in_custom_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "mail.csv"
            dataset.write_text("Category,Message\nham,Same message\nspam,Same message\n", encoding="utf-8")
            with patch.dict(os.environ, MAIL_DATA_PATH=str(dataset)):
                with self.assertRaisesRegex(ValueError, "conflicting labels"):
                    run_notebook("LogisticRegression/SpamEmailDetection.ipynb", ROOT)


if __name__ == "__main__":
    unittest.main()
