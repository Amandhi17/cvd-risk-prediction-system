"""
Data transformation pipeline for the IBM HR Attrition dataset.

Reads:  data/featured.csv  (produced by feature_engineering.py)
Writes:
    data/X_train.csv, data/X_test.csv,
    data/y_train.csv, data/y_test.csv,
    models/preprocessor.joblib  (fitted ColumnTransformer)

Steps (Phase 4 of the project plan):
1. Label encode target: Attrition Yes -> 1, No -> 0
2. One-hot encode categorical columns
3. StandardScaler on numeric columns
4. Stratified 80/20 train/test split (target is imbalanced ~84/16)

The fitted preprocessor is saved so the FastAPI service can reuse it
at inference time (Phase 10).
"""

from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURED_PATH = PROJECT_ROOT / "data" / "featured.csv"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET = "Attrition"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Binary flag columns are already 0/1 — don't scale or encode them.
PASSTHROUGH_COLS = ["OvertimeRiskFlag"]


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])
    logger.info(
        f"Target distribution: {y.value_counts().to_dict()} "
        f"({y.mean():.1%} positive)"
    )
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [
        c for c in X.select_dtypes(include=np.number).columns
        if c not in PASSTHROUGH_COLS
    ]
    passthrough = [c for c in PASSTHROUGH_COLS if c in X.columns]

    logger.info(f"Categorical cols ({len(cat_cols)}): {cat_cols}")
    logger.info(f"Numeric cols ({len(num_cols)}): {num_cols}")
    logger.info(f"Passthrough cols ({len(passthrough)}): {passthrough}")

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols),
            ("pass", "passthrough", passthrough),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_to_df(
    arr: np.ndarray, preprocessor: ColumnTransformer
) -> pd.DataFrame:
    cols = preprocessor.get_feature_names_out()
    return pd.DataFrame(arr, columns=cols)


def main() -> None:
    logger.info(f"Loading featured data from {FEATURED_PATH}")
    df = pd.read_csv(FEATURED_PATH)

    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    X_train_df = transformed_to_df(X_train_arr, preprocessor)
    X_test_df = transformed_to_df(X_test_arr, preprocessor)
    logger.info(f"Transformed train shape: {X_train_df.shape}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_df.to_csv(DATA_DIR / "X_train.csv", index=False)
    X_test_df.to_csv(DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_DIR / "y_test.csv", index=False)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    logger.info("Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    logger.info(f"Saved fitted preprocessor to {MODELS_DIR / 'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
