import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OWNER_RANK = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3, "Fourth & Above Owner": 4}

NUM_FEATURES = ["car_age", "km_driven", "km_per_year", "engine_cc", "max_power_bhp",
                "power_per_litre", "mileage_kmpl", "torque_nm", "seats", "owner_rank"]
CAT_FEATURES = ["brand", "model", "fuel", "transmission", "seller_type"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def brand_and_model(name):
    name = str(name).strip()
    brand = "Land Rover" if name.startswith("Land Rover") else name.split()[0]
    rest = name[len(brand):].strip()
    return brand, (rest.split()[0] if rest else "Unknown")


def add_features(df, ref_year):
    out = df.copy()
    out["car_age"] = (ref_year - out["year"]).clip(lower=0)
    out["km_per_year"] = out["km_driven"] / out["car_age"].clip(lower=1)
    out["power_per_litre"] = (out["max_power_bhp"] / out["engine_cc"] * 1000).replace([np.inf, -np.inf], np.nan)
    out["owner_rank"] = out["owner"].map(OWNER_RANK)
    return out


def build_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=15, sparse_output=False)
    return ColumnTransformer([("num", numeric, NUM_FEATURES), ("cat", categorical, CAT_FEATURES)])


def make_model(regressor):
    """Preprocessing + model in one pipeline; trains on log(price), predicts in rupees."""
    pipe = Pipeline([("prep", build_preprocessor()), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
