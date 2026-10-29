# Classification and model assumptions

[Learning path](../learning-path.md) | [Project README](../../README.md)

## Learning objectives and intuition

Classification predicts a discrete label from features. Learn how different
models draw boundaries: logistic regression combines features linearly, trees
split regions, forests combine trees, SVMs seek separating margins, and Gaussian
Naive Bayes compares class-conditional feature distributions. Model complexity
is a choice to validate, not proof of better predictions.

## Mathematics and assumptions

For binary logistic regression, z = b + x^T w and P(y=1|x) = 1/(1+exp(-z)).
Binary cross-entropy is -mean[y log(p) + (1-y) log(1-p)]. Regularization
penalizes large coefficients; scikit-learn's C controls inverse regularization
strength. Multiclass logistic regression uses class scores and softmax.

A classification tree selects splits reducing impurity. Gini impurity is
1 - sum_k p_k^2; deep trees can create nearly pure leaves that memorize noise.
Random forests average tree class probabilities after fitting trees with
randomness in observations and candidate features. This often reduces variance,
but does not remove the need for validation or eliminate bias.

A linear SVM balances a large margin with penalties for violations. Its hinge
loss is max(0, 1-y*f(x)) for labels -1 and +1. RBF kernels represent nonlinear
similarity: exp(-gamma * squared_distance(x, z)). Scaling matters for distances;
C and gamma jointly affect flexibility. An SVM decision score is not itself a
calibrated class probability.

Gaussian Naive Bayes estimates class priors and feature means/variances, using
P(y|x) proportional to P(y) times the product of P(x_j|y). It assumes conditional
independence and Gaussian feature likelihoods, assumptions often violated in
real data. Work in log probabilities to avoid underflow.

## Worked example

A node with two positives and two negatives has Gini impurity
1 - 0.5^2 - 0.5^2 = 0.5. A split into two pure, equally sized children gives
weighted child impurity 0, an improvement of 0.5. A split that isolates a noisy
training label can also reduce impurity while harming future predictions.

For logistic regression, z=0 gives p=0.5. With y=1 the loss is -log(0.5), about
0.693. Increasing the correct-class probability decreases this loss, but a
confident incorrect prediction receives a large penalty.

## Implementation

Run `python -m MachineLearning.classification` and open
`MachineLearning/AdditionalAlgorithms.ipynb` for the shared Iris experiment.
Inspect the capped-depth decision tree, random forest, scaled RBF SVM, and
Gaussian Naive Bayes on exactly the same held-out rows. Then study the logistic
regression pipeline and training-only tuning in `Learning/Evaluation.ipynb`.

## Evaluation

Compare per-class precision, recall, confusion matrices, and a baseline.
Use training-only cross-validation to select depth, C, gamma, or other settings.
Keep the final test observations untouched during selection. Scaling belongs
inside each CV pipeline. Report fold variability and small-test-set limitations;
a single Iris split cannot establish a universally best classifier.

## Limitations and failure cases

Linear boundaries can underfit nonlinear structure. Deep trees can overfit;
forests can be harder to interpret. SVM distances can be dominated by irrelevant
or unscaled features. Naive Bayes may double-count correlated evidence and
produce poorly calibrated probabilities. Class imbalance, mislabeled data,
and dataset shift affect all these models. A prediction is an association
learned from data, not a causal explanation.

## Exercises and checkpoint

1. Calculate Gini impurity for three positives and one negative.
2. Sketch a boundary for a linear model and a depth-two tree.
3. Explain why scaling should be learned within every SVM CV fold.
4. Propose an experiment to compare a shallow tree and forest without touching
   the final test set until after model selection.

Checkpoint: explain one model's representation, training objective, assumptions,
complexity controls, and a plausible failure case before looking at its score.
