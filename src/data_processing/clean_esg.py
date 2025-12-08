"""
Clean and validate ESG data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def clean_esg_data(
    input_file: str = "data/raw/sp500_esg_data.csv",
    output_file: str = "data/processed/esg_cleaned.csv",
) -> pd.DataFrame:
    """
    Clean and validate ESG scores data.

    Steps:
    1. Load CSV and validate columns
    2. Check for missing values in ESG scores
    3. Validate score ranges (0-100 typically)
    4. Remove duplicates by Ticker
    5. Standardize ticker symbols

    Args:
        input_file: Path to raw ESG data
        output_file: Path to save cleaned data

    Returns:
        Cleaned DataFrame
    """
    print("=" * 60)
    print("Cleaning ESG Data")
    print("=" * 60)

    # Load data
    print(f"\n[LOADING] Loading data from: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"[OK] Loaded {len(df)} records")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_file}")
        print("Please run data download script first.")
        return None

    print(f"\nColumns: {df.columns.tolist()}")

    # Identify column names (handle different naming conventions)
    # Try to find ticker column
    ticker_candidates = ["Ticker", "Symbol", "ticker", "symbol", "TICKER", "SYMBOL"]
    ticker_col = None
    for col in ticker_candidates:
        if col in df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        print(f"\n[ERROR] Could not find ticker column in: {df.columns.tolist()}")
        return None

    print(f"\n[OK] Using '{ticker_col}' as ticker column")

    # Standardize ticker symbols (uppercase, strip whitespace)
    print("\n[PROCESSING] Standardizing ticker symbols...")
    original_count = len(df)
    df[ticker_col] = df[ticker_col].str.upper().str.strip()

    # Remove duplicates by ticker
    print("[PROCESSING] Removing duplicate tickers...")
    duplicates = df[ticker_col].duplicated()
    if duplicates.sum() > 0:
        print(f"\tFound {duplicates.sum()} duplicate tickers")
        print("\tKeeping first occurrence")
        df = df[~duplicates].copy()

    # Identify ESG score columns
    esg_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if (
            "esg" in col_lower
            or "environment" in col_lower
            or "social" in col_lower
            or "governance" in col_lower
        ):
            if (
                "score" in col_lower
                or "rating" in col_lower
                or col_lower in ["e", "s", "g", "esg"]
            ):
                esg_columns.append(col)

    print(f"\n[INFO] Identified ESG columns: {esg_columns}")

    # Check for missing values
    print("\n[CHECKING] Checking for missing values...")
    missing_counts = df[esg_columns].isnull().sum()
    total_rows = len(df)

    for col, missing in missing_counts.items():
        pct = (missing / total_rows) * 100
        print(f"\t{col}: {missing} ({pct:.1f}%)")

    # Handle missing values based on percentage
    total_missing = df[esg_columns].isnull().any(axis=1).sum()
    missing_pct = (total_missing / total_rows) * 100

    print(
        f"\n[STATS] Rows with any missing ESG scores: {total_missing} ({missing_pct:.1f}%)"
    )

    if missing_pct < 5:
        print("\tStrategy: Dropping rows with missing values (< 5% threshold)")
        df = df.dropna(subset=esg_columns).copy()
    else:
        print("\tStrategy: Imputing with median by sector (>= 5% threshold)")
        # Find sector column
        sector_col = None
        for col in ["Sector", "sector", "SECTOR", "Industry", "industry"]:
            if col in df.columns:
                sector_col = col
                break

        if sector_col:
            print(f"\tUsing '{sector_col}' for sector-based imputation")
            for esg_col in esg_columns:
                df[esg_col] = df.groupby(sector_col)[esg_col].transform(
                    lambda x: x.fillna(x.median())
                )
        else:
            print("\tNo sector column found, using global median")
            for esg_col in esg_columns:
                df[esg_col] = df[esg_col].fillna(df[esg_col].median())

    # Validate ESG score ranges
    print("\n[CHECKING] Validating ESG score ranges...")
    for col in esg_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"\t{col}: {min_val:.2f} - {max_val:.2f}")

        # Flag suspicious values (typical range is 0-100)
        if min_val < 0 or max_val > 100:
            print(f"\t[WARNING] Warning: {col} has values outside typical 0-100 range")

    # Check for infinite values
    inf_check = np.isinf(df[esg_columns]).sum()
    if inf_check.sum() > 0:
        print("\n[WARNING] Found infinite values:")
        print(inf_check[inf_check > 0])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=esg_columns)

    # Final statistics
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Original records: {original_count}")
    print(f"Final records: {len(df)}")
    print(
        f"Records removed: {original_count - len(df)} ({((original_count - len(df)) / original_count * 100):.1f}%)"
    )

    # Save cleaned data
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n[OK] Cleaned data saved to: {output_file}")

    # Display sample
    print("\nData preview:")
    display_cols = [ticker_col] + esg_columns[:5]  # Show ticker + first 5 ESG columns
    print(df[display_cols].head())

    return df


if __name__ == "__main__":
    result = clean_esg_data()
    exit(0 if result is not None else 1)
