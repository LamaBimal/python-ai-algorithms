# Mathematical prerequisites

[Learning path](../learning-path.md) | [Project README](../../README.md)

## Learning objectives and intuition

Read a table as a feature matrix X with n rows (observations) and d columns
(features), and targets y as a vector of length n. Distinguish training data
from unseen observations. Practice array shapes before fitting any model.

## Mathematics and assumptions

A dot product x^T w = sum_j x_j w_j is a weighted sum. A matrix product
Xw applies that sum to every observation. The mean is sum(x)/n; variance
measures squared deviations from the mean. Standardization uses
z = (x - training_mean) / training_std. Never recompute those statistics on
test data. Conditional probability P(y | x) describes uncertainty about a
label given features; it is not a guarantee that a fitted model is calibrated.

A derivative measures a local rate of change. A gradient collects partial
derivatives for all parameters. Gradient descent updates parameters in the
direction opposite that gradient.

## Worked example

For x = [2, 3] and w = [4, -1], x^T w = 5. If training values are
[1, 3], their mean is 2 and population standard deviation is 1, so a new
value 5 transforms to 3. Including 5 while learning that mean would change
the training transformation using future information.

## Implementation

Use NumPy to reproduce the arithmetic, then inspect the training-only
mean and scale in `Learning/from_scratch.py`. For a first experiment, use a
single feature and print X.shape, y.shape, and each parameter shape.

## Evaluation

Check hand calculations against array calculations. Compare a constant
predictor with a fitted predictor. A lower training loss alone does not show
that the fitted predictor generalizes.

## Limitations and failure cases

Highly correlated features can make coefficients unstable. A zero-variance
feature contains no variation to learn from. Scaling does not remove outliers
or repair biased data collection.

## Exercises and checkpoint

1. Compute a dot product and a matrix-vector product by hand.
2. Explain why the gradient has one entry per parameter.
3. Standardize [2, 4, 6] using statistics from [2, 4] only.

Checkpoint: explain features, targets, parameters, loss, and held-out data
without using the words interchangeably.
