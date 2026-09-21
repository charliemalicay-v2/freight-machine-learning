"""Feature engineering shared by validation and December prediction."""
import numpy as np
import pandas as pd

MARKET = ["market_index", "market_missing", "quote_signal"]
BASE = ["distance", "log_distance", "weight", "weight_missing", "circuity", "haversine",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "equipment", "pickup", "delivery"]
DATE = ["dow", "is_weekend"]
CATS = ["equipment", "pickup", "delivery"]


def haversine(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))


def build(df, categories):
    out = df.copy()
    out["haversine"] = haversine(out.pickup_lat, out.pickup_lon, out.delivery_lat, out.delivery_lon)
    out["log_distance"] = np.log(out["distance"])
    out["circuity"] = out["distance"] / out["haversine"]
    out["dow"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    for c in CATS:
        out[c] = pd.Categorical(out[c], categories=categories[c])
    return out


def category_levels(*frames):
    return {c: sorted(set().union(*[set(f[c].unique()) for f in frames])) for c in CATS}


def feature_sets():
    return {
        "full": BASE + DATE + MARKET,
        "no_market": BASE + DATE,
    }
