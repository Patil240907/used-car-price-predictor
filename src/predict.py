import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features import FEATURES, add_features

ROOT = Path(__file__).resolve().parent.parent


def load_artifacts():
    return {
        "model": joblib.load(ROOT / "models" / "best_model.joblib"),
        "calibration": json.load(open(ROOT / "models" / "calibration.json")),
        "support": json.load(open(ROOT / "models" / "support.json")),
        "ref_year": json.load(open(ROOT / "models" / "meta.json"))["ref_year"],
        "listings": pd.read_csv(ROOT / "data" / "processed" / "listings.csv"),
    }


def build_input(brand, model, year, km_driven, fuel, transmission, seller_type, owner,
                engine_cc, max_power_bhp, mileage_kmpl=None, torque_nm=None, seats=5, ref_year=None):
    nan = lambda v: np.nan if v is None else v
    row = pd.DataFrame([{"brand": brand, "model": model, "year": year, "km_driven": km_driven,
                         "fuel": fuel, "transmission": transmission, "seller_type": seller_type,
                         "owner": owner, "engine_cc": engine_cc, "max_power_bhp": max_power_bhp,
                         "mileage_kmpl": nan(mileage_kmpl), "torque_nm": nan(torque_nm), "seats": seats}])
    return add_features(row, ref_year)


def predict_price(model, row):
    return float(model.predict(row[FEATURES])[0])


def price_range(pred, cal):
    """Range from out-of-fold errors in the band that the predicted price falls in."""
    idx = int(np.clip(np.searchsorted(cal["edges"], pred, side="right") - 1, 0, len(cal["labels"]) - 1))
    label = cal["labels"][idx]
    b = cal["bands"][label]
    return pred * np.exp(b["lo"]), pred * np.exp(b["hi"]), label


def verdict(asking, pred, low, high):
    diff_pct = (asking - pred) / pred * 100
    if asking > high:
        status = "Above the expected range"
        msg = ("The asking price is higher than the typical range for this kind of car. Compare similar "
               "listings, negotiate, and check condition and service history before deciding.")
        target = (pred, high)
    elif asking < low:
        status = "Below the expected range"
        msg = ("The asking price is lower than expected. It may be a good deal, but it can also signal "
               "hidden problems, so inspect the car carefully.")
        target = None
    else:
        status = "Within the expected range"
        msg = "The asking price is in line with the typical range for this kind of car."
        target = None
    return {"status": status, "diff_pct": diff_pct, "message": msg, "negotiation_range": target}


def confidence_warnings(row, pred, support, cal, label):
    r, out = row.iloc[0], []
    n_brand = support["brand_counts"].get(r["brand"], 0)
    n_model = support["model_counts"].get(f"{r['brand']}|{r['model']}", 0)
    if n_brand == 0:
        out.append(("warning", "Brand not seen in the training data, so the estimate is unreliable."))
    elif n_model < 10:
        out.append(("warning", f"Rare model: only {n_model} similar listings in the data. Limited comparable data."))
    else:
        out.append(("info", f"Common model: {n_model} listings of this model in the data."))
    for col, text in [("km_driven", "Unusual kilometres driven"), ("car_age", "Unusual car age"),
                      ("engine_cc", "Unusual engine size"), ("max_power_bhp", "Unusual power")]:
        lo, hi = support["ranges"][col]
        if r[col] < lo or r[col] > hi:
            out.append(("warning", f"{text}: outside the range of most training data, so the estimate is less reliable."))
    if cal["bands"][label]["n"] < 150:
        out.append(("warning", f"Few listings at the {label} price level, so the range is less reliable."))
    return out


def find_similar(listings, brand, model, year, km_driven, fuel, transmission, n=8):
    pool = listings[(listings["brand"] == brand) & (listings["model"] == model)].copy()
    same_model = len(pool) >= 5
    if not same_model:
        pool = listings[listings["brand"] == brand].copy()
    if pool.empty:
        return pool, {"quality": "None", "n_close": 0, "scope": "no listings for this brand"}

    pool["distance"] = (abs(pool["year"] - year) / 2
                        + abs(pool["km_driven"] - km_driven) / 30000
                        + (pool["fuel"] != fuel) * 1.0
                        + (pool["transmission"] != transmission) * 1.0)
    close = pool[(abs(pool["year"] - year) <= 2) & (abs(pool["km_driven"] - km_driven) <= 40000)
                 & (pool["fuel"] == fuel) & (pool["transmission"] == transmission)]
    n_close = len(close) if same_model else 0
    quality = "Weak" if not same_model else ("Strong" if n_close >= 8 else "Moderate" if n_close >= 3 else "Weak")

    cols = ["name", "year", "km_driven", "fuel", "transmission", "owner", "selling_price", "distance"]
    top = pool.sort_values("distance").head(n)[cols].reset_index(drop=True)
    info = {"quality": quality, "n_close": int(n_close),
            "scope": "same brand and model" if same_model else "same brand (few listings of this model)",
            "median_price": float(top["selling_price"].median())}
    return top, info
