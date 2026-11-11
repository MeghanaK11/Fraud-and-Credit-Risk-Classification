# Fraud and credit risk classification

MLBD Project 4. We compare five frameworks (XGBoost, fastai, Keras, TabNet and PyTorch Lightning) on two problems: spotting fraudulent card transactions and flagging risky loan applicants.

Our central claim: which framework "wins" depends on what you measure. Ranking frauds well and losing the least money are different goals, and they don't pick the same model. With only 48 frauds in the test period, the split and the random seed also move the results about as much as the choice of framework does.

## Data

| | Credit card sample | German Credit |
|---|---|---|
| Rows | 99,999 (99,618 after dropping 381 exact duplicates) | 1,000 |
| Positive class | 223 frauds (0.22%) | 300 bad borrowers (30%) |
| Features | Time, Amount, V1 to V28 (anonymized PCA components) | 20 readable attributes, 13 of them categorical |
| Evaluation | Time-based split: earliest 70% trains, latest 30% tests (48 frauds); the last 15% of the training period is validation | 5-fold stratified CV repeated 5 times (25 folds), same folds for every framework |
| Costs | We define them from Amount (see Person A) | Given: a missed bad borrower costs 5, a wrongly rejected good one costs 1 |

Both files are in `data/`. The German Credit cost matrix is described in `data/german.doc`.

## Results

Means over 10 seeds (fraud) or 25 folds (German Credit). All numbers come from `results/summary_all_frameworks.csv`, which `summarize_results.py` builds from each person's result files.

| Framework | Fraud PR-AUC | Fraud ROC-AUC | Fraud recall at 0.1% FPR | German ROC-AUC | German Brier | German cost at 0.5 | German cost at 1/6 |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.875 ± 0.006 | 0.975 | 0.871 | 0.780 | 0.169 | 0.962 | 0.559 |
| fastai | 0.855 ± 0.017 | 0.962 | 0.850 | 0.757 | 0.181 | 0.859 | 0.608 |
| Keras | 0.863 ± 0.012 | 0.959 | 0.865 | 0.781 | 0.167 | 0.927 | 0.549 |
| TabNet | 0.782 ± 0.042 | 0.943 | 0.808 | 0.658 | 0.201 | 1.334 | 0.703 |
| Lightning | 0.786 ± 0.014 | 0.897 | 0.783 | 0.764 | 0.192 | 0.839 | 0.670 |

German Credit cost is the average cost per applicant. Rejecting every applicant costs 0.70, so a model has to come in below that to be worth using.

What we take from it:

- XGBoost ranks frauds best and is the most stable across seeds. Keras is close behind on fraud and is the cheapest model on German Credit (0.549 per applicant at the 1/6 cut-off).
- The cut-off matters more than the framework on German Credit. At the default 0.5, every framework does worse than rejecting everyone. At 1/(1+5) = 0.167, which follows from the cost matrix, all of them except TabNet beat that line.
- On the fraud data, which model is cheaper depends on the cost of reviewing an alert. With cheap reviews XGBoost costs about half as much as fastai; at a fee of 10 per alert fastai is cheaper (Person A, cost analysis).
- With 48 frauds in the test period, the 95% bootstrap intervals for PR-AUC run from about 0.78 to 0.96 for XGBoost, fastai and Keras alike, so these three can't be cleanly separated on fraud.
- On simplicity, XGBoost needs 3 packages and imports in under a second; fastai pulls in 88 packages. Full table in `results/person_c/simplicity.csv`.

## Things in the brief we handled on purpose

- The brief asks for RMSE, R² and MAE, which are regression metrics. We compute them on predicted probabilities against the 0/1 label, so RMSE is the square root of the Brier score and measures how well calibrated the probabilities are.
- Lightning Flash was archived in October 2023 and is no longer maintained, so Person C used plain PyTorch Lightning.
- German Credit is a loan default dataset, not a fraud dataset. We use it as the second problem because its features are readable and it comes with a cost matrix.

## Repository layout

| Folder | What's in it |
|---|---|
| `common/` | Shared data loading, splits, metrics and cost functions. Every framework uses these, so all models see the same rows. |
| `person_a/` | XGBoost and fastai, cost-based thresholds, card-testing cost model, velocity features, time vs random split, EDA figures. See `person_a/README.md`. |
| `person_b/` | Keras and TabNet notebooks, calibration, TabNet attention vs SHAP, embeddings vs one-hot. See `person_b/README.md`. |
| `person_c/` | PyTorch Lightning on both datasets, fairness on German Credit, simplicity measurements. See `person_c/README.md`. |
| `results/` | All CSVs and figures, one folder per person, plus the combined summary and team figures in `results/figures/`. |

## How to run

```bash
pip install -r requirements.txt

bash person_a/run_all.sh                 # XGBoost, fastai, cost, split and EDA (about 10 min on 2 CPU cores)
jupyter nbconvert --to notebook --execute --inplace person_b/0*.ipynb   # Keras and TabNet, in order (about 20 min)
python3 person_c/fraud_lightning.py
python3 person_c/german_lightning.py
python3 person_c/simplicity.py           # needs the other results first
python3 summarize_results.py
python3 make_summary_figures.py    # fig5 (all frameworks) and fig6 (fairness by age)
```

Person B's notebooks also run in Colab: clone the repo, open a notebook from inside it, and run all cells.
