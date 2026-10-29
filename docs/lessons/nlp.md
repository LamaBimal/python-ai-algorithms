# NLP as an integrated application

[Learning path](../learning-path.md) | [Project README](../../README.md)

## Learning objectives and intuition

Treat text classification as a complete data pipeline: establish labels,
inspect sources, clean consistently, represent text numerically, split safely,
and evaluate against a baseline. Use the earlier lessons to reason about each
choice rather than treating vectorization as a magic step.

## Mathematics and assumptions

TF-IDF combines within-document term frequency with inverse document
frequency learned from the training corpus. Logistic regression learns topic
scores from these features; a feed-forward network can learn nonlinear
combinations. A vocabulary fitted before splitting leaks corpus information.

## Worked example

If a word appears only in held-out articles, a training-fitted vocabulary
does not acquire a new column for it during prediction. This is expected:
prediction transforms inputs using a fixed learned representation.

## Implementation

Run `NLP/NLPWorkshop.ipynb` and then
`NLP/NepaliNewsClassification.ipynb`. The notebooks share tested helpers for
Devanagari cleaning, suffix rules, balanced sampling, TF-IDF, and classification.
Use `LogisticRegression/SpamEmailDetection.ipynb` as a smaller text pipeline.

## Evaluation

Read per-class reports and inspect errors. Deduplicate before splitting;
when articles share sources, authors, events, or dates, consider grouped or
time-based evaluation. Report the sampling procedure and class distribution.
The included spam sample is synthetic and supports a smoke test only.

## Limitations and failure cases

The stopwords and suffix rules are illustrative; stemming can remove useful
information. TF-IDF does not supply pretrained semantic understanding. Models
always select a known class even for unrelated input. A neural network is not
guaranteed to outperform the linear baseline.

## Exercises and checkpoint

1. Compare cleaned and original examples and identify lost information.
2. Explain why unseen words must not change the fitted vocabulary.
3. Design a validation-only comparison of stemming versus no stemming.

Checkpoint: identify the split, fitted transformations, evaluation population,
and assumptions behind one reported score.
