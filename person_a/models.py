"""XGBoost and fastai wrappers with one interface.

fit_predict(...) trains on (X_tr, y_tr), uses (X_va, y_va) for early
stopping (XGBoost) or nothing (fastai trains a fixed number of epochs),
and returns predicted probabilities of the positive class for X_va and
X_te, plus the training time in seconds.
"""
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)


XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    eval_metric="aucpr",
    early_stopping_rounds=100,
    tree_method="hist",
    n_jobs=2,
)


def xgb_fit_predict(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols=None):
    """cat_cols: columns to one-hot encode (German Credit). The encoding is
    fitted on the training rows only."""
    if cat_cols:
        X_tr, X_va, X_te = _one_hot(X_tr, X_va, X_te, cat_cols)
    model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    train_s = time.perf_counter() - t0
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    return p_va, p_te, train_s, model


def _one_hot(X_tr, X_va, X_te, cat_cols):
    allx = pd.concat([X_tr, X_va, X_te], keys=["tr", "va", "te"])
    cats = {c: sorted(X_tr[c].unique()) for c in cat_cols}
    for c in cat_cols:
        allx[c] = pd.Categorical(allx[c], categories=cats[c])
    allx = pd.get_dummies(allx, columns=cat_cols, dtype="float32")
    return allx.loc["tr"], allx.loc["va"], allx.loc["te"]


FASTAI_LAYERS = [200, 100]
FASTAI_EPOCHS = 5
FASTAI_LR = 3e-3


def fastai_fit_predict(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols=None,
                       epochs=FASTAI_EPOCHS, bs=1024):
    """fastai tabular_learner. Categorical columns get learned embeddings,
    numeric columns are normalised on the training rows."""
    from fastai.tabular.all import (
        Categorify, CategoryBlock, FillMissing, IndexSplitter, Normalize,
        TabularPandas, accuracy, set_seed, tabular_learner,
    )
    import torch

    torch.set_num_threads(2)
    set_seed(seed, reproducible=True)
    cat_cols = list(cat_cols or [])
    cont_cols = [c for c in X_tr.columns if c not in cat_cols]

    df = pd.concat([X_tr, X_va], ignore_index=True)
    df["target"] = np.concatenate([y_tr, y_va]).astype(int)
    valid_idx = list(range(len(X_tr), len(df)))
    to = TabularPandas(
        df, procs=[Categorify, FillMissing, Normalize],
        cat_names=cat_cols, cont_names=cont_cols, y_names="target",
        y_block=CategoryBlock(), splits=IndexSplitter(valid_idx)(range(len(df))),
    )
    dls = to.dataloaders(bs=min(bs, len(X_tr)))
    learn = tabular_learner(dls, layers=FASTAI_LAYERS, metrics=accuracy)
    t0 = time.perf_counter()
    with learn.no_bar(), learn.no_logging():
        learn.fit_one_cycle(epochs, FASTAI_LR)
    train_s = time.perf_counter() - t0

    def predict(X):
        dl = learn.dls.test_dl(X.copy())
        with learn.no_bar():
            probs, _ = learn.get_preds(dl=dl)
        return probs[:, 1].numpy()

    return predict(X_va), predict(X_te), train_s, learn


MODELS = {"xgboost": xgb_fit_predict, "fastai": fastai_fit_predict}
