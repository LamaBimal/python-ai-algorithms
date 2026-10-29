# Regression, KNN, and K-means from scratch

[Learning path](../learning-path.md) | [Project README](../../README.md)

## Learning objectives and intuition

Linear regression learns numeric predictions by minimizing squared errors.
KNN classifies by nearby training examples. K-means groups observations around
centers without using labels. Work through their update rules before comparing
with library implementations.

## Mathematics and assumptions

For regression, prediction is b + x^T w and
L = (1/n) sum_i (prediction_i - y_i)^2. With an intercept column in design
matrix D, the gradient is (2/n) D^T(D theta - y). Update
 theta <- theta - learning_rate * gradient.

For KNN, squared Euclidean distance is sum_j (x_j - training_x_j)^2. Pick
the k smallest distances and take a majority vote. For K-means, assign each
point to its nearest center, then replace each center with its members' mean.
The objective (inertia) is the sum of squared distances to assigned centers.

One full regression gradient step costs O(nd). A simple KNN query computes
O(nd) distances and this implementation sorts n distances in O(n log n).
A K-means assignment costs O(nkd) per iteration and start.

## Worked example

Regression: points (0,1), (1,3), (2,5) fit y = 1 + 2x.
KNN: query 2.2 among values [0, 1, 3] with labels [A, A, B] gives B
for k=1 but A for k=3. K-means: assigning [0, 2, 8, 10] to centers
[0, 10] gives new centers [1, 9].

## Implementation

Read `Learning/from_scratch.py`: `LinearRegressionGD`, `KNNClassifier`,
and `KMeans`. Run `Learning/FromScratch.ipynb` for complete comparisons with
scikit-learn. Regression learns its scaling from training rows; KNN and
K-means expect you to choose feature scaling explicitly.

## Evaluation

Compare regression predictions with least squares on held-out points.
Compare KNN with the same scaling, k, and metric, avoiding ambiguous distance
ties when expecting exact equality. Compare K-means using adjusted Rand score
or inertia, not cluster IDs, which can be permuted. Different initialization
algorithms need not converge to the same local solution.

## Limitations and failure cases

Gradient descent may converge slowly or diverge with a poor step size.
KNN is sensitive to feature units, noisy dimensions, and k. Sorted label order
breaks vote ties; training order breaks distance ties in this implementation.
K-means favors compact, roughly spherical groups and can settle in local minima.
Empty clusters retain their old center; duplicate points may leave clusters
unoccupied. These dense-array teaching models are not production replacements.

## Exercises and checkpoint

1. Vary the regression learning rate and compare loss curves.
2. Repeat KNN with and without training-only scaling.
3. Change K-means seeds and compare inertia across starts.

Checkpoint: derive the regression gradient and perform one K-means update by hand.
