"""
Model Evaluation for the Cardiovascular Disease dataset.

Loads all 4 trained models and evaluates on the held-out test set.

Metrics:
- Accuracy, Precision, Recall, F1, ROC-AUC

Target is balanced, so accuracy and ROC-AUC are both meaningful.
We pick the winner by ROC-AUC (threshold-independent quality measure).

Outputs (in reports/):
- test_metrics.csv
- confusion_matrices.png
- roc_curves.png
And best model copied to models/best_model.pkl
"""

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

X_TEST_PATH = "data/X_test.csv"
Y_TEST_PATH = "data/y_test.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

MODEL_NAMES = ["LogisticRegression", "RandomForest",
               "GradientBoosting", "XGBoost"]


def load_models() -> dict:
    return {
        name: joblib.load(f"{MODELS_DIR}/{name}.joblib")
        for name in MODEL_NAMES
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }


def plot_confusion_matrices(results, y_test, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (name, res) in zip(axes.flatten(), results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No CVD", "CVD"],
                    yticklabels=["No CVD", "CVD"])
        ax.set_title(f"{name}\nRecall = {res['recall']:.2f} | "
                     f"F1 = {res['f1']:.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  saved confusion matrices -> {out_path}")


def plot_roc_curves(results, y_test, out_path):
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves — test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  saved ROC curves -> {out_path}")


def main() -> None:
    print("Loading test data and trained models...")
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    print(f"X_test: {X_test.shape} | y_test: {y_test.shape}\n")

    models = load_models()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    results = {name: evaluate(m, X_test, y_test) for name, m in models.items()}

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    table = pd.DataFrame(
        {name: {m: res[m] for m in metric_cols} for name, res in results.items()}
    ).T.round(4)
    print("Test-set metrics:")
    print(table.to_string())
    table.to_csv(f"{REPORTS_DIR}/test_metrics.csv")
    print(f"\nSaved table -> {REPORTS_DIR}/test_metrics.csv\n")

    plot_confusion_matrices(results, y_test,
                            f"{REPORTS_DIR}/confusion_matrices.png")
    plot_roc_curves(results, y_test, f"{REPORTS_DIR}/roc_curves.png")

    best_name = table["roc_auc"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_name} "
          f"(ROC-AUC = {table.loc[best_name, 'roc_auc']:.4f})")
    joblib.dump(models[best_name], f"{MODELS_DIR}/best_model.pkl")
    print(f"Saved -> {MODELS_DIR}/best_model.pkl")


if __name__ == "__main__":
    main()
