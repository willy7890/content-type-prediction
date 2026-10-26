"""Reusable dataset loading and feature preparation helpers."""
from pathlib import Path
import pandas as pd

TARGET_COLUMN = "type"
DROP_COLUMNS = ["show_id", "title"]  # identifiers/free-text title are not used as predictors

def load_dataset(path):
    """Load a CSV dataset and validate the expected target column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place Dataset.csv in data/raw/."
        )
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}'. Available columns: {list(df.columns)}"
        )
    return df

def clean_dataset(df):
    """Standardize column names and target labels, and remove unusable target rows."""
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype("string").str.strip()
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned = cleaned[cleaned[TARGET_COLUMN].str.lower().isin(["movie", "tv show"])]
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].str.title()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No usable rows remain after cleaning the target column.")
    return cleaned

def make_features_and_target(df):
    """Separate target from predictors and remove identifier/title columns when present."""
    df = clean_dataset(df)
    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=[TARGET_COLUMN] + drop_cols, errors="ignore")
    y = df[TARGET_COLUMN].astype(str)
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after excluding the target and identifiers.")
    return X, y
