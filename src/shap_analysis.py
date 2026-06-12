"""
Phase 8 — Model Explainability with SHAP.

Loads the best trained model and explains its predictions:
- Which features matter most globally (summary + bar plots)
- WHY individual employees are flagged (waterfall plots for 2 examples:
  one high-risk, one low-risk)

Why SHAP:
- HR teams won't act on a black-box probability score.
- SHAP attributes each prediction to specific features (e.g.
  "+0.18 from OverTime=Yes, -0.05 from JobSatisfaction=4").
- This turns the model into a decision-support tool, not a black box.

Outputs (in reports/):
- shap_summary.png   — beeswarm: top features and their direction
- shap_bar.png       — global feature importance ranking
- shap_high_risk.png — waterfall for a likely-to-leave employee
- shap_low_risk.png  — waterfall for a likely-to-stay employee
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

X_TEST_PATH = "data/X_test.csv"
Y_TEST_PATH = "data/y_test.csv"
BEST_MODEL_PATH = "models/best_model.pkl"
REPORTS_DIR = "reports"


def pick_explainer(model, X):
    """Pick the right SHAP explainer for the model type."""
    if isinstance(model, LogisticRegression):
        print("Model is linear -> using LinearExplainer")
        return shap.LinearExplainer(model, X)
    # Random Forest / Gradient Boosting / XGBoost all use TreeExplainer
    print(f"Model is tree-based ({type(model).__name__}) -> using TreeExplainer")
    return shap.TreeExplainer(model)


def get_positive_class_values(shap_values):
    """
    SHAP sometimes returns a list/array with one set of values per class.
    For binary classification we want the values for class 1 (Attrition=Yes).
    """
    # newer SHAP returns an Explanation object
    if hasattr(shap_values, "values"):
        vals = shap_values.values
        if vals.ndim == 3:  # (samples, features, classes)
            shap_values.values = vals[:, :, 1]
            shap_values.base_values = shap_values.base_values[:, 1]
        return shap_values
    # older SHAP returns a list [class_0_values, class_1_values]
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values


def main() -> None:
    print("Loading test data and best model...")
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    model = joblib.load(BEST_MODEL_PATH)
    print(f"Best model: {type(model).__name__}")
    print(f"X_test shape: {X_test.shape}\n")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Computing SHAP values (this may take 10-30 seconds)...")
    explainer = pick_explainer(model, X_test)
    shap_values = explainer(X_test)
    shap_values = get_positive_class_values(shap_values)
    print("Done.\n")

    # 1. Global summary (beeswarm) — top features + direction
    print("Generating summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {REPORTS_DIR}/shap_summary.png")

    # 2. Bar plot — mean absolute SHAP value per feature
    print("Generating bar plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {REPORTS_DIR}/shap_bar.png")

    # 3. Individual explanations: pick one likely-leaver, one likely-stayer
    print("Generating individual waterfall plots...")
    proba = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(proba))
    low_risk_idx = int(np.argmin(proba))
    print(f"  Highest-risk employee (idx {high_risk_idx}): "
          f"P(leave) = {proba[high_risk_idx]:.2f}")
    print(f"  Lowest-risk employee  (idx {low_risk_idx}): "
          f"P(leave) = {proba[low_risk_idx]:.2f}")

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

    # 4. Print top-5 features by importance
    print("\nTop 5 features by mean |SHAP value|:")
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_features = (
        pd.Series(mean_abs, index=X_test.columns)
        .sort_values(ascending=False)
        .head(5)
    )
    print(top_features.round(4))


if __name__ == "__main__":
    main()
