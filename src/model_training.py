"""
Phase 6 — Model Training.

Trains 4 classifiers on the preprocessed training data:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Uses 5-fold stratified cross-validation on the training set.
Saves all 4 fitted models to models/ for evaluation in Phase 7.

Class imbalance (84% No / 16% Yes) is handled with:
- class_weight='balanced' for LogReg and Random Forest
- scale_pos_weight for XGBoost
- sample_weight for Gradient Boosting (applied at final fit)
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

X_TRAIN_PATH = "data/X_train.csv"
Y_TRAIN_PATH = "data/y_train.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42


def build_models(y_train: pd.Series) -> dict:
    """Build the 4 classifiers with imbalance handling."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight for XGBoost = {scale_pos_weight:.2f}\n")

    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def cross_validate_model(model, X, y) -> dict:
    """5-fold stratified CV. Returns mean F1 and ROC-AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f1 = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    return {"f1": f1, "roc_auc": auc}


def main() -> None:
    print("Loading training data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"Class balance: {y_train.value_counts().to_dict()}\n")

    models = build_models(y_train)

    print("=" * 60)
    print(f"{'Model':<22} | {'CV F1':>7} | {'CV ROC-AUC':>10}")
    print("=" * 60)

    results = {}
    for name, model in models.items():
        scores = cross_validate_model(model, X_train, y_train)
        results[name] = scores
        print(f"{name:<22} | {scores['f1']:>7.4f} | {scores['roc_auc']:>10.4f}")

    print("=" * 60)

    # Fit each model on the full training set and save
    print("\nFitting on full training set and saving models...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in models.items():
        if name == "GradientBoosting":
            sw = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, sample_weight=sw)
        else:
            model.fit(X_train, y_train)
        path = f"{MODELS_DIR}/{name}.joblib"
        joblib.dump(model, path)
        print(f"  saved -> {path}")

    pd.DataFrame(results).T.to_csv(f"{MODELS_DIR}/cv_results.csv")
    print(f"\nCV scores saved to {MODELS_DIR}/cv_results.csv")


if __name__ == "__main__":
    main()
