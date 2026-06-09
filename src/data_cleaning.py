"""
Data cleaning for the IBM HR Attrition dataset.

Decisions confirmed from notebooks/eda.ipynb:
- No missing values, no duplicates in the raw file.
- Drop 3 constant columns: EmployeeCount, Over18, StandardHours.
- Drop EmployeeNumber (ID, no signal).
- Validate ranges and tenure-vs-age/career consistency; fail loudly on bad data.
"""

from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned.csv"

DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    return pd.read_csv(path)


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in DROP_COLS if c in df.columns]
    logger.info(f"Dropping columns: {present}")
    return df.drop(columns=present)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.warning(f"Removed {removed} duplicate rows")
    return df


def validate_ranges(df: pd.DataFrame) -> None:
    """Hard checks — raise if data violates expected ranges."""
    checks = {
        "Age < 18": (df["Age"] < 18).sum(),
        "YearsAtCompany < 0": (df["YearsAtCompany"] < 0).sum(),
        "MonthlyIncome <= 0": (df["MonthlyIncome"] <= 0).sum(),
        "YearsAtCompany > TotalWorkingYears":
            (df["YearsAtCompany"] > df["TotalWorkingYears"]).sum(),
        "YearsAtCompany > (Age - 18)":
            (df["YearsAtCompany"] > (df["Age"] - 18)).sum(),
        "YearsInCurrentRole > YearsAtCompany":
            (df["YearsInCurrentRole"] > df["YearsAtCompany"]).sum(),
        "YearsWithCurrManager > YearsAtCompany":
            (df["YearsWithCurrManager"] > df["YearsAtCompany"]).sum(),
    }
    failed = {rule: int(n) for rule, n in checks.items() if n > 0}
    if failed:
        raise ValueError(f"Data validation failed: {failed}")
    logger.info("All range and consistency checks passed")


def validate_target(df: pd.DataFrame) -> None:
    allowed = {"Yes", "No"}
    actual = set(df["Attrition"].unique())
    if not actual.issubset(allowed):
        raise ValueError(f"Unexpected Attrition values: {actual - allowed}")
    logger.info(
        f"Target distribution: {df['Attrition'].value_counts().to_dict()}"
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline. Returns a new DataFrame."""
    df = drop_useless_columns(df)
    df = drop_duplicates(df)
    validate_ranges(df)
    validate_target(df)
    logger.info(f"Cleaned shape: {df.shape}")
    return df


def main() -> None:
    raw = load_raw()
    cleaned = clean_data(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_PATH, index=False)
    logger.info(f"Wrote cleaned data to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
