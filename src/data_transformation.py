"""
Data transformation pipeline for the Cardiovascular Disease dataset.

Reads:  data/featured.csv  (produced by feature_engineering.py)
Writes:
    data/X_train.csv, data/X_test.csv,
    data/y_train.csv, data/y_test.csv,
    models/preprocessor.joblib  (fitted ColumnTransformer)

All cardio features are numeric:
- Continuous:  age_years, height, weight, ap_hi, ap_lo, bmi, pulse_pressure,
               map_pressure  -> StandardScaler
- Ordinal/binary: gender, cholesterol, gluc, smoke, alco, active
               -> passthrough (already numeric, scaling them adds noise)

Target `cardio` is already balanced (~50/50), so we just do a stratified split.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURED_PATH = "data/featured.csv"
DATA_DIR = "data"
MODELS_DIR = "models"

TARGET = "cardio"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CONTINUOUS_COLS = [
    "age_years", "height", "weight", "ap_hi", "ap_lo",
    "bmi", "pulse_pressure", "map_pressure",
]
PASSTHROUGH_COLS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]


def split_X_y(df: pd.DataFrame):
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    print(f"Target distribution: {y.value_counts().to_dict()} "
          f"({y.mean():.1%} positive)")
    return X, y


def build_preprocessor() -> ColumnTransformer:
    print(f"Continuous cols ({len(CONTINUOUS_COLS)}): {CONTINUOUS_COLS}")
    print(f"Passthrough cols ({len(PASSTHROUGH_COLS)}): {PASSTHROUGH_COLS}")
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_COLS),
            ("pass", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_to_df(arr: np.ndarray, preprocessor: ColumnTransformer):
    cols = preprocessor.get_feature_names_out()
    return pd.DataFrame(arr, columns=cols)


def main() -> None:
    print(f"Loading featured data from {FEATURED_PATH}")
    df = pd.read_csv(FEATURED_PATH)

    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    preprocessor = build_preprocessor()
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    X_train_df = transformed_to_df(X_train_arr, preprocessor)
    X_test_df = transformed_to_df(X_test_arr, preprocessor)
    print(f"Transformed train shape: {X_train_df.shape}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train_df.to_csv(f"{DATA_DIR}/X_train.csv", index=False)
    X_test_df.to_csv(f"{DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{DATA_DIR}/y_test.csv", index=False)
    joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.joblib")

    print("Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print(f"Saved fitted preprocessor to {MODELS_DIR}/preprocessor.joblib")


if __name__ == "__main__":
    main()
