# Evaluation, leakage, imbalance, and overfitting

[Learning path](../learning-path.md) | [Project README](../../README.md)

## Learning objectives and intuition

Reserve test observations before learning anything from data. Use training
cross-validation to select settings. A final test report estimates performance
after selection; repeatedly using it to choose models makes it validation data.

## Mathematics and assumptions

Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 is their harmonic mean.
Balanced accuracy averages class recalls. Accuracy can be high while minority
recall is zero. Average precision summarizes precision across recall levels;
it is distinct from trapezoidal PR area. ROC-AUC measures score ranking across
positive/negative pairs and can look favorable despite many false positives
when positives are rare. Neither ranking metric specifies an operating threshold.

For each CV fold: fit preprocessing on that fold's training rows, transform its
validation rows, and score. A pipeline passed into GridSearchCV ensures those
steps are repeated independently. Learning curves compare training and
validation scores as training size grows.

## Worked example

With 90 negative and 10 positive examples, always predicting negative gives
90% accuracy, zero positive recall, and 50% balanced accuracy. This is a useful
baseline, not a useful detector. If a flexible tree memorizes training rows,
its training score can increase while cross-validation performance stagnates.

## Implementation

Run `Learning/Evaluation.ipynb` or `python -m Learning.evaluation`. The
experiment uses a reproducible imbalanced synthetic dataset, reserves 25% for
testing, tunes logistic-regression C by average precision in four-fold training
CV, and compares with a prior-probability dummy model. The notebook shows
learning curves, depth/overfitting comparisons, PR/ROC curves, and held-out errors.

The leakage illustration changes only a held-out outlier: the correct
training-only scaler stays fixed, while a scaler fitted globally changes.

## Evaluation

Inspect minority recall, precision, average precision, balanced accuracy,
and confusion counts alongside accuracy. Examine false-positive and
false-negative examples without tuning to the test set. If you subsequently
adjust features, thresholds, or parameters based on these errors, obtain a
fresh test set. Choose thresholds using validation data and application costs.

## Limitations and failure cases

Stratified random splits assume sufficiently independent observations.
Repeated people, duplicate texts, time series, and grouped observations require
group-aware or chronological splits. Small samples and fold-to-fold variation
limit conclusions; CV mean/std is not a confidence interval. Synthetic-data
success does not establish real-world usefulness or probability calibration.

## Exercises and checkpoint

1. Compare a majority baseline with a model using accuracy and recall.
2. Draw a data-flow diagram showing where scaling occurs in each fold.
3. Predict how a very deep tree changes the training/validation gap.
4. Propose a chronological split for next-month forecasting.

Checkpoint: explain which data may influence scaling, tuning, threshold choice,
and final evaluation at each stage.
