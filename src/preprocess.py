"""
preprocess.py
-------------
Loads, cleans, splits, and scales any dataset listed in config.py.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATASETS

# Some manually-downloaded CSVs (like certain Pima Diabetes mirrors) don't
# include a header row. If we detect that, we apply these column names
# instead. This only matters for the diabetes dataset.
DIABETES_COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "See README.md for how to download/generate this file."
        )

    df = pd.read_csv(csv_path)

    # Detect a headerless diabetes CSV: if it has exactly 9 columns and
    # the expected "Outcome" column isn't present, assume the first row
    # was actually data (not a header) and re-read with explicit names.
    if "Outcome" not in df.columns and df.shape[1] == 9:
        df = pd.read_csv(csv_path, header=None, names=DIABETES_COLUMN_NAMES)

    return df


def clean_data(df: pd.DataFrame, target_column: str, binarize: bool) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()

    if binarize and target_column in df.columns:
        df[target_column] = df[target_column].apply(lambda x: 1 if x > 0 else 0)

    return df


def split_and_scale(df: pd.DataFrame, target_column: str,
                     test_size: float = 0.2, random_state: int = 42):
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def get_processed_data(dataset_key: str):
    """Load + clean + split + scale a dataset by its key in config.DATASETS."""
    if dataset_key not in DATASETS:
        raise KeyError(f"Unknown dataset key '{dataset_key}'. "
                        f"Available: {list(DATASETS.keys())}")

    cfg = DATASETS[dataset_key]
    df = load_data(cfg["csv_path"])
    df = clean_data(df, cfg["target_column"], cfg["binarize"])
    return split_and_scale(df, cfg["target_column"])