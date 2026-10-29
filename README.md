# Python AI Algorithms

Educational Python examples for graph search, regression, classification,
clustering, dimensionality reduction, and Nepali natural language processing.
The repository includes executable notebooks, reusable Python demos, datasets,
and regression tests.

## Start learning

Follow the [learning path](docs/learning-path.md) for prerequisites, a recommended
sequence, exercises, and checkpoints. The foundation lessons use a consistent
format: intuition, mathematics, worked example, implementation, evaluation,
limitations, and exercises.

- [From scratch](Learning/FromScratch.ipynb): NumPy linear regression with
  gradient descent, KNN, and K-means compared with scikit-learn.
- [Visual foundations](Learning/VisualFoundations.ipynb): five search strategies,
  gradient updates, KNN boundaries, K-means center movement, and PCA projections.
- [Evaluation](Learning/Evaluation.ipynb): training-only cross-validation,
  leakage, class imbalance, learning curves, overfitting, and held-out errors.

See the [learning examples guide](Learning/README.md) for entry points and
expected outputs. All three notebooks use local or generated data.

## Examples

| Area | Algorithms and examples | Start here |
| --- | --- | --- |
| Graph search | BFS, DFS, uniform-cost search, greedy best-first search, and A* | [Search guide](SearchingTechnique/README.md) |
| Linear regression | Salary prediction from experience | [Salary notebook](LinearRegression/salary_prediction.ipynb) |
| Logistic regression | Diabetes classification and spam email detection | [Diabetes notebook](LogisticRegression/dibetics_prediction_classification.ipynb), [spam notebook](LogisticRegression/SpamEmailDetection.ipynb) |
| K-nearest neighbors | Diabetes classification | [KNN notebook](KNN/KNN_Classifier.ipynb) |
| Classification | Decision tree, random forest, SVM, and Gaussian Naive Bayes | [Machine-learning guide](MachineLearning/README.md) |
| Unsupervised learning | K-means clustering and PCA | [Additional algorithms notebook](MachineLearning/AdditionalAlgorithms.ipynb) |
| Nepali NLP | Text cleaning, suffix stemming, TF-IDF, logistic regression, and a feed-forward neural network | [NLP guide](NLP/ReadMe.md), [completed workshop](NLP/NLPWorkshop.ipynb) |

## Setup

Use Python 3.11 or newer and Git LFS. After installing Git LFS, clone the
repository and retrieve its tracked files:

```sh
git lfs install
git clone https://github.com/LamaBimal/python-ai-algorithms.git
cd python-ai-algorithms
git lfs pull
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install dependencies and launch Jupyter:

```sh
python -m pip install -r requirements.txt
python -m notebook
```

Select the virtual environment's Python kernel and run notebook cells in order.
Start Jupyter from the repository root or a directory inside it. Included
CSV files are found locally; Colab uploads and `/content` paths are unnecessary.
If a data file contains a short Git LFS pointer instead of records, run
`git lfs pull` before opening the notebook.

## Run Python demos

From the repository root with the environment activated:

```sh
python -m SearchingTechnique.breadth_first_search
python -m SearchingTechnique.deapth_first_search
python -m SearchingTechnique.uniform_cost_search
python -m SearchingTechnique.greedy_best_first_search
python -m MachineLearning.classification
python -m MachineLearning.clustering
python -m MachineLearning.dimensionality_reduction
python -m Learning.evaluation
```

The DFS filename retains its original spelling, `deapth_first_search.py`.
The [search case study](SearchingTechnique/Case_Study(BFS_&_A_).ipynb) demonstrates
BFS and A*. A* accepts unit-cost adjacency lists or dictionaries of neighbor
costs. Its coordinate example uses illustrative great-circle distances in
kilometers, rather than measured road distances. See the search guide for
input formats and shortest-path guarantees.

## Data and evaluation

- Salary and diabetes examples use the included CSV files. Diabetes imputation
  medians are learned after splitting, from training rows only.
- The spam notebook includes 40 synthetic teaching messages. Its accuracy is
  a pipeline demonstration, not a measure of real-world spam detection.
  To supply another CSV, set `MAIL_DATA_PATH` before launching Jupyter. It must
  contain `Category` (`ham` or `spam`) and `Message` columns.
- Both NLP notebooks use the included UTF-8 news and suffix files. They sample
  up to 200 articles per topic across sports, business, entertainment, and
  domestic news. TF-IDF replaces the earlier FastText download; no NLTK corpus,
  PyTorch installation, pretrained model, or GPU is needed. Reported scores
  describe the balanced teaching sample.
- Additional ML demos use scikit-learn's bundled datasets or generated data.
  They need no data downloads after dependency installation.

Text-model vocabularies and IDF weights are fitted only on training data.
The NLP models share a fixed stratified split. Use separate validation data
for tuning and reserve independent test data for final evaluation.

## Tests and CI

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The suite covers search edge cases, dataset paths, training-only preprocessing,
model predictions, from-scratch comparisons, evaluation behavior, and every
code cell in the three learning notebooks, spam notebook, and both NLP notebooks. Search checks use the standard library;
ML checks are skipped when their dependencies are missing.

To require all ML checks in PowerShell:

```powershell
$env:REQUIRE_ML_TESTS = '1'
python -m unittest discover -s tests -v
```

On macOS/Linux:

```sh
REQUIRE_ML_TESTS=1 python -m unittest discover -s tests -v
```

[GitHub Actions](.github/workflows/tests.yml) installs the development
requirements and runs the suite with ML checks required on Python 3.11 and
3.14 for pushes and pull requests.

## Roadmap

See [Future enhancements](docs/FUTURE_ENHANCEMENTS.md) for the proposed release plan:
learning improvements and search visualizations, stronger evaluation, reusable
experiment tools, and broader algorithm coverage. The recommended next release
focuses on the learning experience. Version labels describe priorities, not
scheduled release dates.

## Background reading

- [Foundations of AI](docs/foundations-of-ai.md): intelligence, AI perspectives,
  foundational disciplines, and applications.
- [History of AI](docs/history-of-ai.md): an introductory timeline.
