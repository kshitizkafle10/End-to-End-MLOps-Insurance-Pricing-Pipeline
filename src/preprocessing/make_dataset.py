from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------
# Paths
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "uk_insurance_synthetic.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = "personalised_price"


# -----------------------
# Cleaning functions
# -----------------------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns

    # Numeric → median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical → "Unknown"
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    return df


# -----------------------
# Main preprocessing
# -----------------------
def main(test_size: float = 0.2, random_state: int = 42) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}. "
            "Place uk_insurance_synthetic.csv inside data/raw/"
        )

    # Load
    df = pd.read_csv(RAW_PATH)

    # Clean
    df = basic_clean(df)
    df = handle_missing_values(df)

    # Split features / target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Recombine for saving
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    # Save
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Metadata for traceability
    metadata = {
        "target": TARGET_COL,
        "n_rows_raw": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "test_size": test_size,
        "random_state": random_state,
        "columns": list(df.columns),
    }

    with open(PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Preprocessing complete")
    print(f"Target column: {TARGET_COL}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"Saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
