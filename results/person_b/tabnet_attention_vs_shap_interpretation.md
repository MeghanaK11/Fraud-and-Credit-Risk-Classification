# TabNet attention vs SHAP: Fraud

- Compared built-in TabNet mask-based attribution and permutation SHAP for the saved **seed-0** TabNet fraud model.
- Used 20 held-out transactions (oversampled fraud for interpretation) and 30 training-background transactions.
- Spearman correlation between global feature rankings: **0.586**.
- Attention scores are nonnegative mask-based importance, while SHAP scores are signed contributions to predicted fraud probability. These are not interchangeable or causal explanations.
- Global importances are normalized independently to sum to one before plotting.
- Because the sample is deliberately class-balanced, global importance is **not representative of the natural fraud distribution**.
- This is an exploratory explanation comparison for one seed, not a cross-seed stability study.
