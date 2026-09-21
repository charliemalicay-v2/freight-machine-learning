"""Loading and cleaning. Files live flat in the repo root (hyphenated names)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train-test.csv"
VALID = ROOT / "validation.csv"
TEMPLATE = ROOT / "validation-predictions-template.csv"
DECEMBER = ROOT / "december-chart-inputs.csv"


def load_raw():
    train = pd.read_csv(TRAIN, parse_dates=["date"])
    valid = pd.read_csv(VALID, parse_dates=["date"])
    return train, valid


def clean_features(df, weight_medians):
    """Fix feature-side quality issues (applied identically to train, validation, December).

    - negative weights are sign errors (magnitudes match the positive distribution) -> abs()
    - missing weight -> equipment median from train, plus an indicator
    - missing market_index is left as NaN (LightGBM handles it), plus an indicator
    """
    df = df.copy()
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["weight"] = df["weight"].abs()
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_medians))
    if "market_index" in df:
        df["market_missing"] = df["market_index"].isna().astype(int)
    return df


def weight_medians(train):
    return train.assign(weight=train["weight"].abs()).groupby("equipment")["weight"].median().to_dict()


def flag_label_outliers(train, threshold=0.5):
    """Flag corrupted posted_rate values (rate/mile ~12x too high or ~3x too low).

    A deliberately low-capacity model (distance + equipment only) cannot memorise noise,
    so its log-residual isolates the corrupted labels. Returns a boolean mask (True = bad).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = pd.DataFrame({"d": np.log(train["distance"]), "e": train["equipment"].astype("category").cat.codes})
    y = np.log(train["posted_rate"])
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.1,
                                      categorical_features=[1], random_state=0).fit(X, y)
    return (np.abs(y - m.predict(X)) > threshold).to_numpy()
