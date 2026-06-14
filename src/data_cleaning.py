"""
Data cleaning for the Cardiovascular Disease dataset.

Raw schema (semicolon-separated):
  id, age (days), gender, height (cm), weight (kg),
  ap_hi (systolic), ap_lo (diastolic),
  cholesterol (1/2/3), gluc (1/2/3),
  smoke (0/1), alco (0/1), active (0/1),
  cardio (target 0/1)

Cleaning steps:
1. Drop the id column (no predictive value)
2. Drop duplicate rows
3. Remove physiologically impossible records:
   - ap_hi must be 70-250
   - ap_lo must be 40-200
   - ap_hi must be >= ap_lo
   - height 100-220 cm
   - weight 30-250 kg
   (This dataset is well known to contain data-entry errors in BP, hence
   the explicit range filtering — not just an assertion.)
"""

import pandas as pd

RAW_PATH = "data/cardio_data.csv"
CLEAN_PATH = "data/cleaned.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Original shape:", df.shape)

    # 1. drop id
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        print("Dropped id column")

    # 2. drop duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows")
    else:
        print("No duplicates found")

    # 3. range filtering — drop rows with implausible values
    before = len(df)
    df = df[
        (df["ap_hi"].between(70, 250)) &
        (df["ap_lo"].between(40, 200)) &
        (df["ap_hi"] >= df["ap_lo"]) &
        (df["height"].between(100, 220)) &
        (df["weight"].between(30, 250))
    ].reset_index(drop=True)
    removed = before - len(df)
    print(f"Removed {removed} rows with implausible measurements")

    # 4. target sanity check
    assert set(df["cardio"].unique()) == {0, 1}, \
        "cardio must only contain 0 / 1"

    print(f"Target distribution: {df['cardio'].value_counts().to_dict()}")
    print("Final cleaned shape:", df.shape)
    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH, sep=";")
    cleaned = clean_data(df)
    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH}")
