
import importlib.metadata as md

pkgs = [
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "shap",
    "joblib",
    "plotly",
    "matplotlib"
]

for p in pkgs:
    print(f"{p}=={md.version(p)}")
