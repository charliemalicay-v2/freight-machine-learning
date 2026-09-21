"""Fit the final model on all labelled data; write validation + December predictions."""
import numpy as np
import pandas as pd

from . import data, features
from .validate import lgb_fit


def main():
    train, valid = data.load_raw()
    med = data.weight_medians(train)
    train = data.clean_features(train, med)
    train = train[~data.flag_label_outliers(train)].copy()
    valid = data.clean_features(valid, med)
    cats = features.category_levels(train, valid)
    train, valid = features.build(train, cats), features.build(valid, cats)

    sets = features.feature_sets()
    # Validation: market features add nothing under time-based CV (see outputs/cv_summary.csv),
    # so both prediction sets use the same date-free-of-market model, which also works for December.
    predict = lgb_fit(train, sets["no_market"])

    out = pd.read_csv(data.TEMPLATE)
    valid_pred = pd.Series(predict(valid), index=valid["load_id"])
    out["predicted_rate"] = out["load_id"].map(valid_pred).round(2)
    assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
    out.to_csv(data.ROOT / "validation_predictions.csv", index=False)

    dec = pd.read_csv(data.DECEMBER, parse_dates=["date"])
    coords = pd.concat([train[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates("pickup").set_index("pickup"),
                        ], axis=1)
    dcoords = train[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates("delivery").set_index("delivery")
    d = dec.copy()
    d[["pickup_lat", "pickup_lon"]] = coords.loc[d.pickup, ["pickup_lat", "pickup_lon"]].to_numpy()
    d[["delivery_lat", "delivery_lon"]] = dcoords.loc[d.delivery, ["delivery_lat", "delivery_lon"]].to_numpy()
    d["weight_missing"] = 0
    d = features.build(d, cats)
    dec["predicted_rate"] = predict(d).round(2)
    dec.to_csv(data.ROOT / "outputs" / "december_predictions.csv", index=False)
    print(out.predicted_rate.describe().round(1).to_string())
    print(dec[["date", "predicted_rate"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
