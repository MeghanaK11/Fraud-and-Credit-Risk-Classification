# Person C: PyTorch Lightning, fairness and simplicity

This folder has the PyTorch Lightning models for both datasets, a fairness check on German Credit, and the simplicity measurements for all five frameworks.

The brief names Lightning Flash, but Flash was archived in October 2023 and is no longer maintained, so these models use plain PyTorch Lightning.

## Scripts

| Script | What it does | Output (in `results/person_c/`) |
|---|---|---|
| `fraud_lightning.py` | Small MLP (64-32-1), time-based split from `common/`, 10 seeds, 5 epochs | `lightning_fraud_runs.csv` |
| `german_lightning.py` | MLP with dropout and weight decay, one-hot categories, the shared 25 folds, 35 epochs | `lightning_german_runs.csv`, `lightning_german_repeat_summary.csv`, `lightning_fairness.csv` |
| `simplicity.py` | Install footprint, import time and training time for all five frameworks | `simplicity.csv` |

Run the two model scripts first. `simplicity.py` reads everyone's result files, so it goes last.

## Lightning results

| Dataset | PR-AUC | ROC-AUC | Brier | Cost at 0.5 | Cost at 1/6 |
|---|---|---|---|---|---|
| Fraud (10 seeds) | 0.786 ± 0.014 | 0.897 | 0.00095 | | |
| German Credit (25 folds) | 0.596 | 0.764 | 0.192 | 0.839 | 0.670 |

On fraud, Lightning trails XGBoost, fastai and Keras. It trains for a fixed 5 epochs with no class weighting and no early stopping, so this measures a plain default setup more than the framework's ceiling. On German Credit, the 1/6 cut-off brings its cost under the reject-everyone line of 0.70 in 16 of 25 folds. At 0.5 it is above that line in 23 of 25.

## Fairness on German Credit

Approval rates at the 1/6 cut-off, averaged over the 25 test folds:

| Group | Applicants per fold | Approved | Bad borrowers approved |
|---|---|---|---|
| Age 18 to 25 | 38 | 42% | 22% |
| Age 26 to 35 | 80 | 56% | 31% |
| Age 36 to 50 | 60 | 66% | 35% |
| Age 51+ | 23 | 68% | 44% |
| A92: female, divorced/separated/married | 62 | 49% | 22% |
| A93: male, single | 110 | 63% | 34% |

Applicants aged 25 or under are approved far less often than those over 50 (42% against 68%). Part of that matches the data: young applicants really are riskier in German Credit. But the model is also more lenient with older applicants, letting through 44% of their bad borrowers against 22% for the youngest group. Women in A92 are approved less often than single men (49% against 63%). The dataset has no separate category for single women, and A91 (divorced or separated men) has only about 10 applicants per fold, so we leave it out of the table.

## Simplicity of the five frameworks

| Framework | Packages installed | Install size (MB) | Import time (s) | Fraud train time (s) | German train time (s) |
|---|---|---|---|---|---|
| XGBoost | 3 | 256 | 0.73 | 1.1 | 0.74 |
| fastai | 88 | 1,749 | 2.28 | 2.5 | 0.65 |
| Keras (TensorFlow) | 32 | 1,460 | 2.42 | 23.6 | 1.98 |
| TabNet | 20 | 1,390 | 1.74 | 47.4 | 1.21 |
| Lightning | 19 | 1,257 | 2.80 | 1.7 | 1.06 |

How these were measured:

- Packages and install size are the framework's full dependency tree as installed, read from package metadata. The three PyTorch-based frameworks all include PyTorch. GPU-only packages (CUDA, NVIDIA, Triton) are left out so every framework counts as a CPU install, though the PyTorch build here was the CUDA one, which makes its own files somewhat larger than a CPU-only install.
- Import time is the median of 5 cold imports in a fresh Python process.
- Train times are the mean per model, taken from each framework's result CSV. Everything ran on the same 2-core CPU machine. Keras and TabNet use early stopping with up to 60 epochs on fraud, so their times include many more passes over the data than Lightning's fixed 5 epochs.

We don't report lines of code. Each of us wrote training code in a different style (scripts, notebooks, different amounts of logging), so a line count would measure the author more than the framework.

XGBoost is the lightest by a wide margin: 3 packages, under a second to import, and the fastest fraud training. fastai pulls in the most packages (88) because it brings its own data, vision and text tooling along with PyTorch.
