
# Person B — Keras, TabNet, Calibration and Explainability

**Project:** Fraud and Credit Risk Classification  
**Contribution:** Person B — Keras neural networks, PyTorch TabNet, probability calibration evaluation, TabNet attention vs SHAP, and categorical-encoding ablation.

## Overview

This folder documents Person B's experiments on two binary classification datasets:

- **German Credit:** 1,000 applicants; 700 good-credit (`bad=0`) and 300 bad-credit (`bad=1`).
- **Credit Card Fraud:** highly imbalanced transaction data; `Class=1` denotes fraud. The experiments use the shared repository's dataset loader and temporal splitting utility.

The central questions were whether Keras or TabNet performed better under the tested configurations, how trustworthy their predicted probabilities were, whether TabNet's built-in attention agreed with SHAP, and whether categorical embeddings improved TabNet over one-hot encoding.

## Reproducible experiments

| Notebook | Experiment | Design |
|---|---|---|
| [`01_keras_models.ipynb`](01_keras_models.ipynb) | Keras on German Credit and fraud | German: repeated stratified 5-fold CV × 5; fraud: temporal split, 5 seeds |
| [`02_tabnet_models.ipynb`](02_tabnet_models.ipynb) | TabNet on German Credit and fraud | German: repeated stratified 5-fold CV × 5; fraud: temporal split, 5 seeds |
| [`03_calibration_analysis.ipynb`](03_calibration_analysis.ipynb) | Calibration evaluation | Brier score, log loss, 10-bin ECE and reliability diagrams from saved held-out predictions |
| [`04_tabnet_attention_vs_shap.ipynb`](04_tabnet_attention_vs_shap.ipynb) | Attention vs SHAP | Seed-0 fraud TabNet; 20 selected test transactions and 30 background samples |
| [`05_tabnet_encoding_ablation.ipynb`](05_tabnet_encoding_ablation.ipynb) | Categorical embeddings vs one-hot | Paired 25-fold German Credit evaluation; 50 TabNet fits |

The notebooks rely on the team's shared [`common`](../common) utilities and existing dataset files. The fraud experiments use the same temporal splitting function and base-feature function as Person A. Preprocessing is fitted only to training partitions, with validation sets used for early stopping.

## Main model results

**Means across 25 test folds (German Credit) or 5 training seeds (fraud).** Higher PR-AUC/ROC-AUC is better; lower Brier is better.

| Dataset | Model | PR-AUC ↑ | ROC-AUC ↑ | Brier ↓ |
|---|---|---:|---:|---:|
| German Credit | Keras MLP | **0.6096** | **0.7813** | **0.1668** |
| German Credit | TabNet (embeddings) | 0.4622 | 0.6649 | 0.1986 |
| Credit Card Fraud | Keras MLP | **0.8624** | **0.9566** | **0.000882** |
| Credit Card Fraud | TabNet | 0.8025 | 0.9544 | 0.001074 |

Keras performed better on these measures for both datasets **under the configurations tested**. These results do not establish that Keras always outperforms TabNet. The fraud test set contained 48 positive transactions; rare-event metrics should be interpreted with care.

### German Credit cost-sensitive decisions

False negatives (bad applicants predicted good) were assigned cost 5; false positives cost 1. The shared evaluation reports **average cost per applicant**, not raw total cost.

| Model | Threshold 0.50 | Threshold 1/6 ≈ 0.1667 |
|---|---:|---:|
| Keras | 0.9272 | **0.5494** |
| TabNet | 1.2854 | 0.6878 |

The 1/6 threshold follows the assumed 5:1 misclassification costs **when probabilities are well calibrated**. It reduced Keras's mean evaluated cost by approximately 40.7% compared with threshold 0.50. These are held-out evaluation results at prespecified thresholds, not a threshold optimized on the test set.

## Calibration evaluation

The calibration notebook evaluates **existing held-out predictions**; it does **not** train a Platt, isotonic or temperature calibrator. ECE values below use **10 quantile bins** and depend on the binning scheme.

| Dataset | Model | Brier ↓ | ECE ↓ | Log loss ↓ |
|---|---|---:|---:|---:|
| German Credit | Keras | **0.166835** | **0.066058** | **0.505342** |
| German Credit | TabNet | 0.198622 | 0.090191 | 0.589675 |
| Fraud | Keras | **0.000882** | 0.001701 | **0.005724** |
| Fraud | TabNet | 0.001074 | **0.001226** | 0.006621 |

Keras had lower Brier and log loss on both datasets. TabNet had slightly lower 10-bin ECE on fraud, demonstrating that calibration conclusions can depend on the metric. Fraud reliability bins are particularly noisy because positive cases are rare.

## TabNet attention vs SHAP

For the saved **seed-0 fraud TabNet model**, built-in mask-based feature importance was compared with permutation SHAP on 20 deliberately class-balanced held-out transactions, using 30 background transactions.

- **Spearman rank correlation:** **0.4867**, indicating moderate positive ranking agreement.
- **Agreement:** `V14` ranked first and `V17` second for both methods; `V4` ranked fourth in both.
- **Disagreement:** `V9` ranked third by TabNet attention but twentieth by SHAP.

Attention masks and SHAP answer related but different questions. Attention is a nonnegative measure of feature selection/importance; SHAP attributes changes in model predictions. Neither establishes causality. Because the sample was intentionally balanced and only one seed was examined, these rankings are **exploratory**, not population-wide or cross-seed conclusions.

## TabNet categorical embeddings vs one-hot encoding

A separate **paired** ablation trained TabNet with either learned categorical embeddings or one-hot encoded features on the **same 25 German Credit outer folds**, with matched inner validation splits and seeds. Both approaches used the same TabNet training configuration; preprocessing was fitted on inner training data only.

| Metric | Embeddings | One-hot | Better |
|---|---:|---:|---|
| PR-AUC ↑ | 0.4670 | **0.4892** | One-hot |
| ROC-AUC ↑ | 0.6638 | **0.6998** | One-hot |
| Brier ↓ | 0.1972 | **0.1921** | One-hot |
| Mean cost at 1/6 ↓ | 0.6964 | **0.6552** | One-hot |

One-hot performed better in this ablation. The difference may reflect the small dataset, low-cardinality categories, or model settings; it should not be generalized to larger tabular datasets. The ablation's embeddings results are a **separate paired experiment**, so they need not exactly equal the earlier TabNet baseline results.

## Where the outputs are stored

The team's shared output directory is [`../results/person_b/`](../results/person_b/), with images under [`../results/person_b/figures/`](../results/person_b/figures/).

| Output pattern | Purpose |
|---|---|
| `keras_german_25fold_*.csv`, `keras_fraud_runs.csv`, `keras_fraud_predictions.npz` | Keras evaluation metrics and held-out predictions |
| `tabnet_german_25fold_*.csv`, `tabnet_fraud_runs.csv`, `tabnet_fraud_predictions.npz` | TabNet evaluation metrics and held-out predictions |
| `keras_fraud_model.keras`, `tabnet_fraud_model.zip`, `tabnet_fraud_scaler.npz` | Saved fraud models and TabNet input scaler |
| `calibration_*.csv`, `calibration_interpretation.md` | Calibration summaries, per-run results and reliability-bin details |
| `tabnet_attention_vs_shap_*.csv`, `tabnet_explained_sample.csv` | Explanation rankings and sample information |
| `tabnet_encoding_ablation_*.csv`, `tabnet_encoding_ablation_interpretation.md` | Paired encoding comparison and interpretation |
| `figures/*.png` | PR curves, confusion matrices, calibration and explainability plots |

## How to run

1. Open the repository in **Google Colab** or an environment with the dependencies installed from `requirements.txt` plus `pytorch-tabnet`, `tensorflow`, and `shap` as required by individual notebooks.
2. Keep the repository's `common/`, `data/`, and `person_a/` directories accessible; the notebooks import the shared data, metrics and baseline-feature utilities.
3. Run the notebooks in numerical order. The first two notebooks train models and save predictions. The calibration notebook uses their saved results ZIPs; the explanation notebook uses the saved TabNet model; the encoding notebook runs an independent paired experiment.
4. Download generated results from Colab before the runtime resets. Upload the notebooks to `person_b/`, the individual result files to `results/person_b/`, and figures to `results/person_b/figures/`.

**Note:** Some notebooks prompt for ZIP uploads when run in a fresh Colab session. Follow the notebook's instructions. Do not upload a ZIP containing all outputs in place of the individual result files.

## Limitations and interpretation

- German Credit contains only 1,000 rows; model comparisons can vary with initialization and validation splits.
- The repeated cross-validation folds overlap across repeats and should not be treated as 25 independent datasets.
- Fraud models were trained with class weighting, which can affect probability calibration. Fraud seeds share the same held-out test observations.
- The calibration notebook **evaluates** probability quality; it does not perform post-hoc probability recalibration.
- The attention-vs-SHAP comparison uses a small, deliberately class-balanced sample from one model seed.
- The embedding ablation uses a fixed training budget rather than an extensive hyperparameter search.
- Results compare these **specific implementations and settings**, not intrinsic superiority of a framework.

## Key takeaways

1. **Keras MLPs were stronger than the tested TabNet baselines** on PR-AUC and Brier score for both datasets.
2. **Cost-aware classification mattered** on German Credit: using the prespecified 1/6 threshold reduced average Keras decision cost relative to 0.50.
3. **Probability quality and ranking quality are different:** Brier/log loss and ECE gave slightly different conclusions on fraud.
4. **TabNet attention only partly agreed with SHAP**, so attention should not be treated as a complete substitute for explanation methods.
5. **One-hot encoding outperformed learned categorical embeddings** for this small German Credit ablation.

---

**Person B contribution:** implementation, evaluation and interpretation of Keras and TabNet experiments, calibration analysis, attention-vs-SHAP comparison and categorical-encoding ablation.
