"""
Feature engineering for the IBM HR Attrition dataset.

Adds 3 derived features (per project plan, Phase 5):
- IncomePerYear     = MonthlyIncome * 12
- OvertimeRiskFlag  = 1 if OverTime == 'Yes' else 0  (replaces OverTime column)
- ExperienceRatio   = YearsAtCompany / Age

Reads:  data/cleaned.csv
Writes: data/featured.csv
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
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned.csv"
FEATURED_PATH = PROJECT_ROOT / "data" / "featured.csv"


def add_income_per_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IncomePerYear"] = df["MonthlyIncome"] * 12
    return df


def add_overtime_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["OvertimeRiskFlag"] = (df["OverTime"] == "Yes").astype(int)
    df = df.drop(columns=["OverTime"])
    return df


def add_experience_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ExperienceRatio"] = df["YearsAtCompany"] / df["Age"]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps."""
    logger.info(f"Input shape: {df.shape}")
    df = add_income_per_year(df)
    df = add_overtime_flag(df)
    df = add_experience_ratio(df)
    logger.info(f"Output shape: {df.shape}")
    logger.info(
        "New features: IncomePerYear, OvertimeRiskFlag, ExperienceRatio"
    )
    return df


def main() -> None:
    logger.info(f"Loading cleaned data from {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    df = engineer_features(df)
    FEATURED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURED_PATH, index=False)
    logger.info(f"Wrote featured data to {FEATURED_PATH}")


if __name__ == "__main__":
    main()
