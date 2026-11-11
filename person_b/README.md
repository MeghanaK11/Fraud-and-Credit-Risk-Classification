# Person B: Keras, TabNet, calibration and explainability

This folder has the Keras and TabNet models for both datasets, plus three experiments: how well calibrated each model's probabilities are, whether TabNet's built-in attention agrees with SHAP, and whether TabNet does better with learned category embeddings or one-hot encoding on German Credit.

All notebooks use the shared `common/` loaders and splits and Person A's `base_features`, so Keras and TabNet are trained and tested on exactly the same rows as the other frameworks.

## Notebooks

| Notebook | What it does | Design |
|---|---|---|
| `01_keras_models.ipynb` | Keras on both datasets | German: 5-fold CV repeated 5 times; fraud: time-based split, 10 seeds |
| `02_tabnet_models.ipynb` | TabNet on both datasets | Same design as Keras |
| `03_calibration_analysis.ipynb` | Calibration of the saved test probabilities | Brier score, log loss, 10-bin ECE, reliability diagrams |
| `04_tabnet_attention_vs_shap.ipynb` | TabNet attention vs SHAP on fraud | Seed-0 model, 20 test transactions, 30 background rows |
| `05_tabnet_encoding_ablation.ipynb` | Embeddings vs one-hot for TabNet | Paired, same 25 German Credit folds, 50 fits |

Run them in order. 01 and 02 train the models and save predictions to `results/person_b/`; 03 and 04 read those files; 05 is independent. Each notebook finds the repo folder by itself, so it runs locally or in Colab after cloning. The saved notebooks include their outputs.

## Results

Means over 25 German Credit folds or 10 fraud seeds.

| Dataset | Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|
| German Credit | Keras | 0.610 | 0.781 | 0.167 |
| German Credit | TabNet | 0.450 | 0.658 | 0.201 |
| Fraud | Keras | 0.863 | 0.959 | 0.00094 |
| Fraud | TabNet | 0.782 | 0.943 | 0.00107 |

Keras beat TabNet on every measure here, for both datasets. That holds for these settings (small networks, fixed training budgets, no tuning) and doesn't mean Keras is better in general. TabNet also varied far more between fraud seeds: PR-AUC std 0.042 against 0.012 for Keras.

German Credit cost per applicant (missed bad borrower costs 5, wrongly rejected good one costs 1):

| Model | Cut-off 0.5 | Cut-off 1/6 |
|---|---|---|
| Keras | 0.927 | 0.549 |
| TabNet | 1.334 | 0.703 |

Switching Keras from 0.5 to the cost-matrix cut-off of 1/6 lowers its cost by about 41%. At 0.549, Keras is the cheapest of all five frameworks on German Credit. Both cut-offs are fixed in advance, not tuned on the test folds.

## Calibration

`03_calibration_analysis.ipynb` scores the existing test probabilities. It doesn't fit any recalibration (Platt, isotonic or temperature scaling). ECE uses 10 quantile bins and changes with the binning.

| Dataset | Model | Brier | ECE | Log loss |
|---|---|---|---|---|
| German Credit | Keras | 0.1668 | 0.0661 | 0.505 |
| German Credit | TabNet | 0.2009 | 0.0907 | 0.596 |
| Fraud | Keras | 0.00094 | 0.00185 | 0.00578 |
| Fraud | TabNet | 0.00107 | 0.00118 | 0.00720 |

Keras has the lower Brier score and log loss on both datasets, but TabNet has the lower ECE on fraud, so the verdict depends on which calibration measure you pick. Fraud reliability bins are noisy because there are only 48 frauds; the bin counts are in `calibration_bin_details.csv`.

## TabNet attention vs SHAP

For the seed-0 fraud model, TabNet's built-in feature importance was compared with permutation SHAP on 20 test transactions (10 frauds, 10 normal) against 30 background rows from the training period.

The two rankings agree moderately: Spearman correlation 0.586. Both put V17 first, and V14 and V11 in their top three. They disagree further down: V3 is 10th by attention but 26th by SHAP, and V6 is 6th against 16th. Attention says which features the model looked at, while SHAP says how much each feature moved the prediction, so they answer different questions. This is one seed and a deliberately balanced sample, so the rankings are exploratory.

## Embeddings vs one-hot on German Credit

The same TabNet setup was trained twice on each of the 25 folds, once with learned 2-dimensional embeddings for the 13 categorical columns and once with one-hot encoding.

| Metric | Embeddings | One-hot |
|---|---|---|
| PR-AUC | 0.450 | 0.503 |
| ROC-AUC | 0.658 | 0.712 |
| Brier | 0.201 | 0.190 |
| Cost at 1/6 | 0.703 | 0.651 |

One-hot did better on every measure. With 1,000 rows and categories of 2 to 10 levels, there's probably too little data for embeddings to learn anything one-hot can't already represent.

## Output files (in `results/person_b/`)

| Files | Contents |
|---|---|
| `keras_german_25fold_*.csv`, `keras_fraud_runs.csv`, `keras_fraud_predictions.npz` | Keras metrics and test predictions |
| `tabnet_german_25fold_*.csv`, `tabnet_fraud_runs.csv`, `tabnet_fraud_predictions.npz` | TabNet metrics and test predictions |
| `keras_fraud_model.keras`, `tabnet_fraud_model.zip`, `tabnet_fraud_scaler.npz` | Saved seed-0 fraud models and TabNet's input scaler |
| `keras_vs_person_a.csv`, `keras_fraud_ci.csv` | Keras compared with XGBoost and fastai, and its bootstrap CI |
| `calibration_*.csv`, `calibration_interpretation.md` | Calibration results |
| `tabnet_attention_vs_shap_*`, `tabnet_explained_sample.csv` | Attention vs SHAP rankings |
| `tabnet_encoding_ablation_*` | Embeddings vs one-hot results |
| `figures/` | PR curves, confusion matrices, reliability diagrams, explanation plots |

## Limitations

- German Credit has 1,000 rows, and the 25 folds overlap across repeats, so they aren't 25 independent tests.
- Both fraud models use class weighting, which shifts their probabilities and affects calibration.
- All fraud seeds share the same 48 test frauds.
- The attention vs SHAP comparison uses one seed and a small balanced sample.
- No hyperparameter search was done for either model.
