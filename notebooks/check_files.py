
from pathlib import Path

required = [
    "models/best_model.joblib",
    "models/calibration.json",
    "models/support.json",
    "models/meta.json",
    "models/model_summary.json",
    "data/processed/listings.csv",
    "reports/model_results/baseline_comparison.csv",
    "reports/model_results/tuning_comparison.csv",
    "reports/model_results/oof_predictions.csv",
    "src/__init__.py",
    "src/features.py",
    "src/explain.py",
    "src/predict.py",
]

optional = [
    "reports/figures/shap_importance.png",
    "reports/figures/shap_beeswarm.png",
    "reports/figures/error_analysis.png",
    "reports/figures/partial_dependence.png",
]

for p in required:
    print("OK      " if Path(p).exists() else "MISSING ", p)

for p in optional:
    print("ok      " if Path(p).exists() else "(optional, missing)", p)
