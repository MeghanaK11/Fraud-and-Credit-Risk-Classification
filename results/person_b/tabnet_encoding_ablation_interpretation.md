# TabNet German Credit: categorical embeddings vs one-hot

## Design
Paired repeated stratified 5-fold CV × 5 repeats (50 model fits).
Both representations used the same folds, inner validation split, TabNet architecture,
training hyperparameters, seed per fold, and metrics. Preprocessing was fitted only
on the inner training partition.

- Embeddings: 7 standardized numeric columns and 13 integer-encoded categorical columns, learned 2-dimensional categorical embeddings.
- One-hot: standardized numeric columns plus one-hot encoded categorical columns.

## Results (means across folds)
- PR-AUC: embeddings 0.4670; one-hot 0.4892. Higher is better; one-hot scored higher.
- ROC-AUC: embeddings 0.6638; one-hot 0.6998.
- Brier: embeddings 0.1972; one-hot 0.1921. Lower is better; one-hot scored lower.
- Mean cost at threshold 1/6: embeddings 0.6964; one-hot 0.6552.

## Interpretation limits
This is an encoding ablation for this particular dataset, model configuration and
training budget, not proof that one encoding is always superior.
Repeated CV folds overlap across repeats; avoid treating the 25 differences as
25 independent datasets. Cost threshold 1/6 is theoretically optimal only for
well-calibrated probabilities under the specified 5:1 error costs.
