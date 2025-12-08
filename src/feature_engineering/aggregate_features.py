"""
Aggregate all features into a single firm-level dataset for regression analysis.
"""

from pathlib import Path

import pandas as pd


def aggregate_all_features(
    master_file: str = "data/final/master_dataset.csv",
    performance_df: pd.DataFrame = None,
    risk_df: pd.DataFrame = None,
    controls_df: pd.DataFrame = None,
    output_file: str = "data/final/analysis_dataset.csv",
) -> pd.DataFrame:
    """
    Aggregate all features into a firm-level dataset.

    This combines:
    - ESG scores (from master dataset)
    - Performance metrics (calculated)
    - Risk metrics (calculated)
    - Control variables (calculated)

    Args:
        master_file: Path to master dataset (for ESG scores)
        performance_df: DataFrame with performance metrics
        risk_df: DataFrame with risk metrics
        controls_df: DataFrame with control variables
        output_file: Path to save final analysis dataset

    Returns:
        DataFrame with one row per company and all features
    """
    print("=" * 60)
    print("Aggregating All Features")
    print("=" * 60)

    # Load master dataset to get ESG scores
    print(f"\n[LOADING] Loading master dataset from: {master_file}")
    master_df = pd.read_csv(master_file)

    # Get one row per ticker with ESG scores
    print("\n[PROCESSING] Extracting ESG scores (one row per ticker)...")

    # Find ESG columns
    esg_columns = [
        col
        for col in master_df.columns
        if any(
            keyword in col.lower()
            for keyword in [
                "esg",
                "environment",
                "social",
                "governance",
                "score",
                "rating",
            ]
        )
    ]

    # Also include Ticker and Company name if available
    base_columns = ["Ticker"]
    if "Company_Name" in master_df.columns:
        base_columns.append("Company_Name")
    elif "Company" in master_df.columns:
        base_columns.append("Company")

    esg_df = (
        master_df[base_columns + esg_columns].drop_duplicates(subset=["Ticker"]).copy()
    )

    print(f"\t[OK] ESG data: {len(esg_df)} companies")
    print(f"\tESG columns: {esg_columns}")

    # Start with ESG data
    analysis_df = esg_df.copy()

    # Merge performance metrics
    if performance_df is not None:
        print("\n[PROCESSING] Merging performance metrics...")
        print(f"\tPerformance data: {len(performance_df)} companies")

        analysis_df = analysis_df.merge(performance_df, on="Ticker", how="inner")
        print(f"\tAfter merge: {len(analysis_df)} companies")
    else:
        print("\n[WARNING] No performance metrics provided")

    # Merge risk metrics
    if risk_df is not None:
        print("\n[PROCESSING] Merging risk metrics...")
        print(f"\tRisk data: {len(risk_df)} companies")

        analysis_df = analysis_df.merge(risk_df, on="Ticker", how="inner")
        print(f"\tAfter merge: {len(analysis_df)} companies")
    else:
        print("\n[WARNING] No risk metrics provided")

    # Merge control variables
    if controls_df is not None:
        print("\n[PROCESSING] Merging control variables...")
        print(f"\tControls data: {len(controls_df)} companies")

        analysis_df = analysis_df.merge(controls_df, on="Ticker", how="inner")
        print(f"\tAfter merge: {len(analysis_df)} companies")
    else:
        print("\n[WARNING] No control variables provided")

    # Final validation
    print("\n[CHECKING] Final validation...")

    # Check for duplicates
    duplicates = analysis_df["Ticker"].duplicated().sum()
    if duplicates > 0:
        print(f"\t[WARNING] Found {duplicates} duplicate tickers, removing...")
        analysis_df = analysis_df.drop_duplicates(subset=["Ticker"], keep="first")

    # Check for missing values
    print("\n[INFO] Missing value summary:")
    missing_counts = analysis_df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if len(missing_counts) > 0:
        for col, count in missing_counts.head(10).items():
            pct = (count / len(analysis_df)) * 100
            print(f"\t{col}: {count} ({pct:.1f}%)")
    else:
        print("\t[OK] No missing values!")

    # Summary
    print("\n" + "=" * 60)
    print("AGGREGATION SUMMARY")
    print("=" * 60)
    print("Final dataset:")
    print(f"\tCompanies: {len(analysis_df)}")
    print(f"\tVariables: {len(analysis_df.columns)}")
    print(
        f"\tCompleteness: {((analysis_df.notna().sum().sum()) / (len(analysis_df) * len(analysis_df.columns)) * 100):.2f}%"
    )

    print("\nColumn categories:")

    # Count column types
    esg_cols = [
        col
        for col in analysis_df.columns
        if any(
            k in col.lower() for k in ["esg", "environmental", "social", "governance"]
        )
    ]
    perf_cols = [
        col
        for col in analysis_df.columns
        if any(k in col.lower() for k in ["return", "sharpe"])
    ]
    risk_cols = [
        col
        for col in analysis_df.columns
        if any(
            k in col.lower()
            for k in ["volatility", "beta", "deviation", "drawdown", "var"]
        )
    ]
    control_cols = [
        col
        for col in analysis_df.columns
        if any(k in col.lower() for k in ["market_cap", "sector"])
    ]

    print(f"\tESG variables: {len(esg_cols)}")
    print(f"\tPerformance variables: {len(perf_cols)}")
    print(f"\tRisk variables: {len(risk_cols)}")
    print(f"\tControl variables: {len(control_cols)}")

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(output_file, index=False)
    print(f"\n[OK] Analysis dataset saved to: {output_file}")

    # Display sample
    print("\nData preview:")
    print(analysis_df.head())

    print("\nAll columns:")
    for i, col in enumerate(analysis_df.columns, 1):
        print(f"\t{i}. {col}")

    return analysis_df


if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Run: python scripts/run_feature_engineering.py")
