# Person B: Calibration evaluation

Existing held-out test probabilities were analyzed without fitting a calibrator on test labels.
German Credit: 25 repeated stratified folds per model. Fraud: 10 training seeds on the same temporal test set.
Lower Brier, log loss, and ECE indicate better calibration-related scores; ECE depends on binning.
PR-AUC and ROC-AUC measure ranking/discrimination, not calibration.

## Mean results
- Fraud, keras: Brier=0.00094, log loss=0.00578, ECE(quantile)=0.00185, PR-AUC=0.86283.
- Fraud, tabnet: Brier=0.00107, log loss=0.00720, ECE(quantile)=0.00118, PR-AUC=0.78232.
- German Credit, keras: Brier=0.16684, log loss=0.50534, ECE(quantile)=0.06606, PR-AUC=0.60965.
- German Credit, tabnet: Brier=0.20092, log loss=0.59641, ECE(quantile)=0.09072, PR-AUC=0.45012.

## Limitations
- No post-hoc calibrator was fitted; do not describe this as calibrated model performance.
- German repeated folds overlap across repeats; fraud seeds share the same test observations.
- Rare fraud events make reliability bins noisy. Inspect bin counts in calibration_bin_details.csv.
- Model training and class weighting may differ, so interpret framework differences cautiously.