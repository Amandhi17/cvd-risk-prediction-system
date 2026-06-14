"""
Model Explainability for the Cardiovascular Disease dataset.

Loads the best trained model and explains its predictions with SHAP.

Why SHAP for a medical use case:
- A doctor will not act on a black-box probability score.
- SHAP attributes each prediction to specific risk factors
  (e.g. "+0.18 from high systolic BP, +0.12 from age, -0.04 from
  being physically active").

Outputs (in reports/):
- shap_summary.png   — top features and their direction
- shap_bar.png       — global importance ranking
- shap_high_risk.png — waterfall for a likely-CVD patient
- shap_low_risk.png  — waterfall for a likely-healthy patient
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

X_TEST_PATH = "data/X_test.csv"
BEST_MODEL_PATH = "models/best_model.pkl"
REPORTS_DIR = "reports"

# SHAP on tree models with the full 70K test set is slow; sample for speed.
SAMPLE_SIZE = 2000


def pick_explainer(model, X):
    if isinstance(model, LogisticRegression):
        print("Model is linear -> using LinearExplainer")
        return shap.LinearExplainer(model, X)
    print(f"Model is tree-based ({type(model).__name__}) -> using TreeExplainer")
    return shap.TreeExplainer(model)


def get_positive_class_values(shap_values):
    if hasattr(shap_values, "values"):
        vals = shap_values.values
        if vals.ndim == 3:
            shap_values.values = vals[:, :, 1]
            shap_values.base_values = shap_values.base_values[:, 1]
        return shap_values
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values


def main() -> None:
    print("Loading test data and best model...")
    X_test = pd.read_csv(X_TEST_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    print(f"Best model: {type(model).__name__}")
    print(f"X_test shape: {X_test.shape}")

    if len(X_test) > SAMPLE_SIZE:
        X_test = X_test.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"Sampled {SAMPLE_SIZE} rows for faster SHAP computation\n")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Computing SHAP values...")
    explainer = pick_explainer(model, X_test)
    shap_values = explainer(X_test)
    shap_values = get_positive_class_values(shap_values)
    print("Done.\n")

    # 1. Summary (beeswarm)
    print("Generating summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=14)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {REPORTS_DIR}/shap_summary.png")

    # 2. Bar
    print("Generating bar plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      show=False, max_display=14)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {REPORTS_DIR}/shap_bar.png")

    # 3 & 4. Individual waterfalls
    print("Generating individual waterfall plots...")
    proba = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(proba))
    low_risk_idx = int(np.argmin(proba))
    print(f"  Highest-risk patient (idx {high_risk_idx}): "
          f"P(CVD) = {proba[high_risk_idx]:.2f}")
    print(f"  Lowest-risk patient  (idx {low_risk_idx}): "
          f"P(CVD) = {proba[low_risk_idx]:.2f}")

    plt.figure()
    shap.plots.waterfall(shap_values[high_risk_idx], max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_high_risk.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {REPORTS_DIR}/shap_high_risk.png")

    plt.figure()
    shap.plots.waterfall(shap_values[low_risk_idx], max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_low_risk.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {REPORTS_DIR}/shap_low_risk.png")

    print("\nTop 5 risk factors by mean |SHAP value|:")
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_features = (
        pd.Series(mean_abs, index=X_test.columns)
        .sort_values(ascending=False)
        .head(5)
    )
    print(top_features.round(4))


if __name__ == "__main__":
    main()
