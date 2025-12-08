"""
Create control variables: log market cap, sector dummies.
"""

import numpy as np
import pandas as pd


def create_control_variables(
    df: pd.DataFrame,
    ticker_col: str = "Ticker",
    market_cap_col: str = "Market_Cap",
    sector_col: str = "Sector",
) -> pd.DataFrame:
    """
    Create control variables for regression analysis.

    Variables created:
    - Log market cap
    - Sector dummy variables (one-hot encoded, drop first to avoid multicollinearity)

    Args:
        df: DataFrame with company information
        ticker_col: Name of ticker column
        market_cap_col: Name of market cap column
        sector_col: Name of sector column

    Returns:
        DataFrame with control variables
    """
    print("=" * 60)
    print("Creating Control Variables")
    print("=" * 60)

    # Ensure one row per ticker
    if df[ticker_col].duplicated().any():
        print("\n[PROCESSING] Multiple rows per ticker detected, keeping first...")
        df = df.drop_duplicates(subset=[ticker_col], keep="first").copy()

    print("\n[INFO] Input data:")
    print(f"\tCompanies: {len(df)}")

    results = df[[ticker_col]].copy()

    # 1. Log Market Cap
    print("\n[PROCESSING] Creating log market cap...")

    if market_cap_col in df.columns:
        # Check for missing or zero values
        missing_mcap = df[market_cap_col].isnull().sum()
        zero_mcap = (df[market_cap_col] == 0).sum()

        if missing_mcap > 0:
            print(f"\t[WARNING]  Missing market cap for {missing_mcap} companies")

        if zero_mcap > 0:
            print(f"\t[WARNING]  Zero market cap for {zero_mcap} companies")

        # Calculate log market cap (in billions for interpretability)
        results["Market_Cap_Billions"] = df[market_cap_col] / 1e9
        results["Log_Market_Cap"] = np.log(results["Market_Cap_Billions"])

        # Replace inf/-inf with NaN
        results["Log_Market_Cap"] = results["Log_Market_Cap"].replace(
            [np.inf, -np.inf], np.nan
        )

        # Summary
        valid_log_mcap = results["Log_Market_Cap"].notna().sum()
        print(f"\t[OK] Created log market cap for {valid_log_mcap} companies")

        print("\n   Market Cap Distribution:")
        print(f"\t   Mean: ${results['Market_Cap_Billions'].mean():.2f}B")
        print(f"\t   Median: ${results['Market_Cap_Billions'].median():.2f}B")
        print(f"\t   Min: ${results['Market_Cap_Billions'].min():.2f}B")
        print(f"\t   Max: ${results['Market_Cap_Billions'].max():.2f}B")

        print("\n   Log Market Cap Distribution:")
        print(f"\t   Mean: {results['Log_Market_Cap'].mean():.4f}")
        print(f"\t   Std: {results['Log_Market_Cap'].std():.4f}")
    else:
        print(f"\t[WARNING]  Market cap column '{market_cap_col}' not found")
        results["Market_Cap_Billions"] = np.nan
        results["Log_Market_Cap"] = np.nan

    # 2. Sector Dummies
    print("\n[PROCESSING] Creating sector dummy variables...")

    if sector_col in df.columns:
        # Check sector distribution
        sector_counts = df[sector_col].value_counts()
        print(f"\tSectors found: {len(sector_counts)}")
        print("\n   Sector distribution:")
        for sector, count in sector_counts.items():
            print(f"\t   {sector}: {count}")

        # Create dummy variables (drop_first=True to avoid multicollinearity)
        sector_dummies = pd.get_dummies(
            df[sector_col], prefix="Sector", drop_first=True, dtype=int
        )

        print(
            f"\n   [OK] Created {len(sector_dummies.columns)} sector dummies (dropped first as baseline)"
        )

        # Merge with results
        results = pd.concat([results, sector_dummies], axis=1)

    else:
        print(f"\t[WARNING]  Sector column '{sector_col}' not found")

    # Summary
    print("\n" + "=" * 60)
    print("CONTROL VARIABLES SUMMARY")
    print("=" * 60)
    print(f"Total companies: {len(results)}")
    print(f"Total variables: {len(results.columns) - 1}")  # -1 for ticker
    print(f"\nColumns created: {results.columns.tolist()}")

    # Display sample
    print("\n[INFO] Sample results:")
    print(results.head(10))

    return results


if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Run: python scripts/run_feature_engineering.py")
