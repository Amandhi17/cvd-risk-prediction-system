"""
Feature engineering for the Cardiovascular Disease dataset.

Adds 4 derived features:
- age_years     = age / 365.25         (age was in days, convert to years)
- bmi           = weight / (height/100)^2
- pulse_pressure = ap_hi - ap_lo
- map_pressure  = (ap_hi + 2 * ap_lo) / 3   (mean arterial pressure)

The original `age` column (in days) is dropped after conversion.

Reads:  data/cleaned.csv
Writes: data/featured.csv
"""

import pandas as pd

CLEAN_PATH = "data/cleaned.csv"
FEATURED_PATH = "data/featured.csv"


def add_age_years(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_years"] = (df["age"] / 365.25).round(1)
    df = df.drop(columns=["age"])
    return df


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(2)
    return df


def add_pulse_pressure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    return df


def add_map_pressure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["map_pressure"] = ((df["ap_hi"] + 2 * df["ap_lo"]) / 3).round(2)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Input shape:", df.shape)
    df = add_age_years(df)
    df = add_bmi(df)
    df = add_pulse_pressure(df)
    df = add_map_pressure(df)
    print("Output shape:", df.shape)
    print("New features: age_years, bmi, pulse_pressure, map_pressure")
    return df


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    df = engineer_features(df)
    df.to_csv(FEATURED_PATH, index=False)
    print(f"Saved featured data to {FEATURED_PATH}")
