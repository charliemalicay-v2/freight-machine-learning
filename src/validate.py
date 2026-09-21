"""Expanding-window time-based validation on the labelled data (Jan-Oct 2025)."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import data, features

PARAMS = dict(n_estimators=800, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=0, verbose=-1)

# (train through, test month): each fold trains on everything before the test month.
FOLDS = ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10"]


def metrics(y, p):
    ape = np.abs(p - y) / y
    return {"MAE": float(np.mean(np.abs(p - y))), "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
            "MAPE": float(100 * ape.mean()), "MedAPE": float(100 * np.median(ape))}


def baseline_fit(tr):
    """Median rate/mile by equipment x distance bucket, times distance."""
    edges = np.array([0, 200, 400, 800, 1500, 2500, 1e9])
    b = pd.cut(tr.distance, edges, labels=False)
    tab = (tr.posted_rate / tr.distance).groupby([tr.equipment, b]).median()
    def predict(df):
        bb = pd.cut(df.distance, edges, labels=False)
        rpm = [tab.get((e, k), np.nan) for e, k in zip(df.equipment, bb)]
        return df.distance.to_numpy() * np.array(rpm)
    return predict


def ridge_fit(tr, cols):
    def X(df):
        return pd.concat([pd.DataFrame({"ld": df.log_distance, "ld2": df.log_distance ** 2,
                                        "w": df.weight / 1e4}),
                          pd.get_dummies(df.equipment, dtype=float)], axis=1)
    m = Ridge(alpha=1.0).fit(X(tr), np.log(tr.posted_rate))
    return lambda df: np.exp(m.predict(X(df)))


def lgb_fit(tr, cols, **over):
    m = lgb.LGBMRegressor(**{**PARAMS, **over}).fit(tr[cols], np.log(tr.posted_rate))
    return lambda df: np.exp(m.predict(df[cols]))


def run(save=True):
    train, valid = data.load_raw()
    med = data.weight_medians(train)
    train = data.clean_features(train, med)
    train["bad"] = data.flag_label_outliers(train)
    cats = features.category_levels(train, valid)
    train = features.build(train, cats)
    sets = features.feature_sets()
    month = train.date.dt.strftime("%Y-%m")

    models = {
        "baseline_rpm": lambda tr: baseline_fit(tr),
        "ridge": lambda tr: ridge_fit(tr, None),
        "lgb_no_market": lambda tr: lgb_fit(tr, sets["no_market"]),
        "lgb_full": lambda tr: lgb_fit(tr, sets["full"]),
    }
    rows = []
    for fold in FOLDS:
        te = train[month == fold]
        tr = train[month < fold]
        tr_clean = tr[~tr.bad]
        te_clean = te[~te.bad]
        for name, fit in models.items():
            pred = fit(tr_clean)
            for scope, d in (("clean", te_clean), ("raw", te)):
                rows.append({"fold": fold, "model": name, "scope": scope, **metrics(d.posted_rate.to_numpy(), pred(d))})
    res = pd.DataFrame(rows)
    summary = res.groupby(["scope", "model"])[["MAE", "RMSE", "MAPE", "MedAPE"]].mean().round(3)
    print(f"label outliers flagged in train: {int(train.bad.sum())} ({train.bad.mean():.2%})")
    print(summary.to_string())
    if save:
        res.to_csv(data.ROOT / "outputs" / "cv_results.csv", index=False)
        summary.to_csv(data.ROOT / "outputs" / "cv_summary.csv")
    return res, summary


if __name__ == "__main__":
    run()
