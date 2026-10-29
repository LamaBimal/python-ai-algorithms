# Learning path

[Project README](../README.md)

Start with basic Python functions, loops, lists, dictionaries, and NumPy arrays.
Install the project requirements and select the virtual environment's kernel.
Work through the lessons in order; use the checkpoints to decide when to move on.
The order expresses dependencies, not a fixed study schedule.

| Stage | Lesson | Practice | Ready to move on when you can... |
| --- | --- | --- | --- |
| 0 | [Math prerequisites](lessons/prerequisites.md) | Dot products, scaling, shapes, gradients | Explain data, parameters, loss, and held-out rows |
| 1 | [Search](lessons/search.md) | Trace five frontiers in [Visual foundations](../Learning/VisualFoundations.ipynb) | Explain edge count versus path cost and A* assumptions |
| 2 | [Models from scratch](lessons/from-scratch.md) | Run [From scratch](../Learning/FromScratch.ipynb), then existing salary/KNN examples | Derive the regression update and explain neighbor voting |
| 3 | [Classification](lessons/classification.md) and [evaluation](lessons/evaluation.md) | Run [Evaluation](../Learning/Evaluation.ipynb), then the classification demos | Separate training, validation, and final test decisions |
| 4 | [Unsupervised learning](lessons/unsupervised.md) | K-means updates and PCA projections | Explain arbitrary cluster IDs and lost PCA information |
| 5 | [NLP](lessons/nlp.md) | Run the Nepali workshop and inspect errors | Explain the complete text pipeline and its limitations |

Each lesson uses the same structure: objectives and intuition, mathematics,
a worked example, implementation, evaluation, limitations, and exercises.
Read the equations alongside the source; run each notebook from top to bottom.
Notebook outputs are left clear so readers generate results in their own environment.

## How to study actively

1. Predict the next step or output before running a cell.
2. Work through the numerical example by hand.
3. Change one parameter, write a prediction, and compare the result.
4. Complete the exercises and explain the checkpoint in your own words.
5. Keep a small experiment log with data, seed, parameter choices, and findings.

## Expected outputs

The from-scratch notebook prints regression errors, KNN agreement, and a
label-permutation-invariant clustering comparison. Visual foundations produces
search frontier snapshots, a gradient-descent loss/trajectory figure, KNN
decision regions, K-means center movement, and PCA reconstruction geometry.
Evaluation reports the selected C, a majority baseline, per-class metrics,
training/validation curves, and held-out error counts. Exact floating-point
values can vary with library versions; inspect the behavior rather than
memorizing one score.

## Setup troubleshooting

- If an import fails, install `requirements.txt` using the same Python that
  runs your notebook kernel. Restart the kernel after installing packages.
- If data contains `version https://git-lfs.github.com/spec/v1`, run
  `git lfs install` and `git lfs pull` from the repository.
- Start Jupyter from the repository root or one of its subdirectories.
- If a noninteractive backend prevents plots from displaying, restart Jupyter
  without `MPLBACKEND=Agg`; that setting is useful for automated tests only.
- Source code, Markdown, and requirements are ordinary Git files. Data and
  notebooks remain tracked by Git LFS.
