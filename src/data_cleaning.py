"""
Data cleaning for the IBM HR Attrition dataset.

Steps:
1. Drop 3 constant columns (EmployeeCount, Over18, StandardHours)
2. Drop the ID column (EmployeeNumber)
3. Drop any duplicate rows
4. Validate ranges and logical consistency
"""

import pandas as pd

RAW_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
CLEAN_PATH = "data/cleaned.csv"

DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Original shape:", df.shape)

    # 1. drop useless columns
    df = df.drop(columns=DROP_COLS)
    print("After dropping useless columns:", df.shape)

    # 2. drop duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows")
    else:
        print("No duplicates found")

    # 3. range validation
    assert (df["Age"] >= 18).all(),              "Age must be >= 18"
    assert (df["YearsAtCompany"] >= 0).all(),    "YearsAtCompany must be >= 0"
    assert (df["MonthlyIncome"] > 0).all(),      "MonthlyIncome must be > 0"

    # 4. logical consistency
    assert (df["YearsAtCompany"] <= df["TotalWorkingYears"]).all(), \
        "YearsAtCompany cannot exceed TotalWorkingYears"
    assert (df["YearsAtCompany"] <= (df["Age"] - 18)).all(), \
        "YearsAtCompany cannot exceed Age - 18"
    assert (df["YearsInCurrentRole"] <= df["YearsAtCompany"]).all(), \
        "YearsInCurrentRole cannot exceed YearsAtCompany"
    assert (df["YearsWithCurrManager"] <= df["YearsAtCompany"]).all(), \
        "YearsWithCurrManager cannot exceed YearsAtCompany"

    # 5. target check
    assert set(df["Attrition"].unique()) == {"Yes", "No"}, \
        "Attrition must only contain Yes/No"

    print("All validation checks passed")
    print("Final cleaned shape:", df.shape)
    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    cleaned = clean_data(df)
    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH}")
