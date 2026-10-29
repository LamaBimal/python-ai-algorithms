# Foundation learning examples

Follow the [learning path](../docs/learning-path.md) and install the root
`requirements.txt`. Start Jupyter from the repository root or a subdirectory,
select the project environment's kernel, and run cells in order. All examples
use local, bundled, or generated data.

| Notebook | Implementation | What to inspect |
| --- | --- | --- |
| [From scratch](FromScratch.ipynb) | [NumPy models](from_scratch.py) | Regression vs least squares, KNN voting, K-means vs library partitions |
| [Visual foundations](VisualFoundations.ipynb) | [Search trace](search_walkthrough.py) and NumPy models | Frontier snapshots, loss and gradient trajectory, boundaries, centers, PCA |
| [Evaluation](Evaluation.ipynb) | [Evaluation experiment](evaluation.py) | Train-only CV and tuning, majority baseline, imbalance, learning curves, final errors |

For a text report without Jupyter, run from the repository root:

```sh
python -m Learning.evaluation
```

## Expected behavior

- Gradient descent approaches the least-squares solution on the synthetic
  regression example. A decreasing training objective measures optimization;
  inspect held-out predictions separately.
- KNN agrees with scikit-learn on the example's shared, scaled split. Equal
  distances or tied votes can lead to implementation-dependent choices.
- K-means recovers the clearly separated synthetic groups. Cluster IDs can
  differ; compare adjusted Rand score and inertia. More difficult data and
  different initializations need not yield the same local solution.
- BFS and greedy search take the cost-5 example route, while UCS and admissible
  A* take the cost-3 route. DFS follows the given neighbor order.
- The majority baseline has high accuracy but zero positive recall. The final
  test figures are reports after model selection, not inputs to model tuning.

The NumPy estimators accept finite dense numeric feature matrices. They omit
many production estimator features; read their documented tie rules, scaling
requirements, stopping budgets, and empty-cluster behavior before extending them.

## Verification

After installing `requirements-dev.txt`, run:

```sh
python -m unittest discover -s tests -v
```

Tests compare model behavior with known solutions and reference estimators,
exercise search edge cases and A* reopening, check training-only preprocessing,
and execute every code cell of the learning notebooks. Runtime timings and
floating-point values depend on the local environment.
