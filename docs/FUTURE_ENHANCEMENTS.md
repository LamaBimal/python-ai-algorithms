# Future enhancements

This roadmap proposes the next stages for Python AI Algorithms. Version labels
express an intended sequence, not release dates or committed delivery promises.
The foundation learning priorities now have an initial implementation:
[learning path](learning-path.md), seven structured lessons, three
[executable walkthroughs](../Learning/README.md), NumPy regression/KNN/K-means,
and training-only evaluation experiments. These provide a starting point;
extending the same lesson structure to every legacy notebook remains future work.
Unchecked items below describe further work rather than existing features.

## v1.1: Learning experience

Recommended next release: help learners understand and navigate the examples
before expanding the algorithm collection.

- [x] Add a learning path with prerequisites and suggested notebook order.
- [ ] Give each notebook consistent learning objectives, explanations,
  exercises, and expected outputs.
- [x] Visualize BFS, DFS, uniform-cost search, greedy best-first search, and A*
  step by step, showing the frontier, visited nodes, path, and accumulated cost.
- [x] Add troubleshooting for virtual environments, Jupyter kernels, missing
  dependencies, and Git LFS pointer files.

Completion target: a new learner can set up the repository, follow a documented
path, and explain how the search algorithms choose their next node.

## v1.2: Better experiments

Strengthen evaluation before adding more models.

- [x] Add cross-validation and parameter tuning within the training partition,
  keeping the final test set untouched.
- [ ] Compare algorithms on shared datasets and splits, reporting predictive
  metrics, fit/prediction time, and relevant complexity trade-offs.
- [x] Add learning curves, error analysis, and metrics for imbalanced classes,
  including per-class precision, recall, F1, and appropriate PR/ROC curves.
- [ ] Offer a documented real spam dataset alongside the synthetic sample,
  with source, license, retrieval instructions, and duplicate handling.

Completion target: experiment reports distinguish training, validation, and
final test results and explain the limitations of small or synthetic datasets.
Timing results should record the environment used rather than imply universal
performance rankings.

## v1.3: Reusable tools

Make experiments repeatable outside notebooks.

- [ ] Add a CLI with configurable datasets, algorithms, seeds, and parameters.
- [ ] Save complete fitted preprocessing/model pipelines and load them for
  prediction, documenting dependency versions and trusted-artifact requirements.
- [ ] Add an interactive search and classification demo with explanatory inputs
  and outputs.
- [ ] Export experiment settings and results in machine-readable formats.

Completion target: users can run, record, and reproduce an experiment through
the CLI and apply a saved pipeline to new data without refitting preprocessing.

## v2.0: Broader algorithm coverage

Expand only after the learning and evaluation foundations are established.

- [ ] Add gradient boosting, DBSCAN, and hierarchical clustering demonstrations.
- [ ] Introduce optimization methods such as hill climbing, simulated annealing,
  and genetic algorithms, with explicit objectives and stopping criteria.
- [ ] Offer optional Nepali FastText or transformer embeddings and compare them
  with the TF-IDF baseline using the same data partitions and evaluation metrics.
- [ ] Record runtime, memory, download size, and hardware requirements for the
  optional NLP approaches; retain a lightweight local baseline.
- [ ] Add experiment tracking and versioned dataset metadata, including seeds,
  dependency versions, data identifiers, and preprocessing settings.

Completion target: new algorithms have explanations, runnable examples,
behavioral checks, and comparable evaluations. Optional heavy dependencies
should not be required to run the core examples.

## Improvements across releases

- [ ] Document dataset sources, licenses, provenance, and known limitations.
- [ ] Keep reusable logic in Python modules and use notebooks as guided lessons.
- [ ] Extend CI with meaningful behavioral checks for each new algorithm and
  notebook workflow.
- [ ] Document tested dependency versions and how to reproduce reported results.

When selecting work, prioritize a clear learning benefit, reproducible results,
reasonable local resource requirements, and maintainable test coverage. Review
this roadmap after each release and adjust the scope based on learner feedback.
