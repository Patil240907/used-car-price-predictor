import numpy as np
import pandas as pd
import shap
from src.features import CAT_FEATURES


def original_feature(raw_name):
    """'cat__fuel_Diesel' -> 'fuel', 'num__car_age' -> 'car_age'"""
    kind, rest = raw_name.split("__", 1)
    if kind == "num":
        return rest
    for f in sorted(CAT_FEATURES, key=len, reverse=True):
        if rest.startswith(f + "_"):
            return f
    return rest


class Explainer:
    def __init__(self, fitted_model):
        self.pipe = fitted_model.regressor_              # fitted pipeline inside TransformedTargetRegressor
        self.prep = self.pipe.named_steps["prep"]
        self.booster = self.pipe.named_steps["model"]
        self.raw_names = list(self.prep.get_feature_names_out())
        self.groups = [original_feature(n) for n in self.raw_names]
        self.explainer = shap.TreeExplainer(self.booster)
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])   # in log-price

    def shap_values(self, X):
        Xt = self.prep.transform(X)
        return self.explainer.shap_values(Xt, check_additivity=False), Xt

    def grouped(self, sv):
        """Add one-hot columns back together so each original feature has one SHAP value."""
        return pd.DataFrame(sv, columns=self.groups).T.groupby(level=0).sum().T

    def explain_one(self, X_row):
        sv, _ = self.shap_values(X_row)
        g = self.grouped(sv).iloc[0]
        out = pd.DataFrame({"feature": g.index,
                            "value": [X_row.iloc[0][f] for f in g.index],
                            "log_effect": g.values,
                            "approx_pct_effect": (np.exp(g.values) - 1) * 100})
        return out.reindex(out["log_effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)
